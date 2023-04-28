// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "arrow/python/python_to_arrow.h"
#include "arrow/python/numpy_interop.h"

#include <datetime.h>

#include <algorithm>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "arrow/array.h"
#include "arrow/array/builder_binary.h"
#include "arrow/array/builder_decimal.h"
#include "arrow/array/builder_dict.h"
#include "arrow/array/builder_nested.h"
#include "arrow/array/builder_primitive.h"
#include "arrow/array/builder_time.h"
#include "arrow/chunked_array.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type.h"
#include "arrow/type_traits.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/converter.h"
#include "arrow/util/decimal.h"
#include "arrow/util/int_util_overflow.h"
#include "arrow/util/logging.h"

#include "arrow/visit_type_inline.h"
#include "arrow/python/datetime.h"
#include "arrow/python/decimal.h"
#include "arrow/python/helpers.h"
#include "arrow/python/inference.h"
#include "arrow/python/iterators.h"
#include "arrow/python/numpy_convert.h"
#include "arrow/python/type_traits.h"

namespace arrow {

using internal::checked_cast;
using internal::checked_pointer_cast;

using internal::Converter;
using internal::DictionaryConverter;
using internal::ListConverter;
using internal::PrimitiveConverter;
using internal::StructConverter;

using internal::MakeChunker;
using internal::MakeConverter;

namespace py {

namespace {
enum class MonthDayNanoField { kMonths, kWeeksAndDays, kDaysOnly, kNanoseconds };

template <MonthDayNanoField field>
struct MonthDayNanoTraits;

struct MonthDayNanoAttrData {
  const char* name;
  const int64_t multiplier;
};

template <>
struct MonthDayNanoTraits<MonthDayNanoField::kMonths> {
  using c_type = int32_t;
  static const MonthDayNanoAttrData attrs[];
};

const MonthDayNanoAttrData MonthDayNanoTraits<MonthDayNanoField::kMonths>::attrs[] = {
    {"years", 1}, {"months", /*months_in_year=*/12}, {nullptr, 0}};

template <>
struct MonthDayNanoTraits<MonthDayNanoField::kWeeksAndDays> {
  using c_type = int32_t;
  static const MonthDayNanoAttrData attrs[];
};

const MonthDayNanoAttrData MonthDayNanoTraits<MonthDayNanoField::kWeeksAndDays>::attrs[] =
    {{"weeks", 1}, {"days", /*days_in_week=*/7}, {nullptr, 0}};

template <>
struct MonthDayNanoTraits<MonthDayNanoField::kDaysOnly> {
  using c_type = int32_t;
  static const MonthDayNanoAttrData attrs[];
};

const MonthDayNanoAttrData MonthDayNanoTraits<MonthDayNanoField::kDaysOnly>::attrs[] = {
    {"days", 1}, {nullptr, 0}};

template <>
struct MonthDayNanoTraits<MonthDayNanoField::kNanoseconds> {
  using c_type = int64_t;
  static const MonthDayNanoAttrData attrs[];
};

const MonthDayNanoAttrData MonthDayNanoTraits<MonthDayNanoField::kNanoseconds>::attrs[] =
    {{"hours", 1},
     {"minutes", /*minutes_in_hours=*/60},
     {"seconds", /*seconds_in_minute=*/60},
     {"milliseconds", /*milliseconds_in_seconds*/ 1000},
     {"microseconds", /*microseconds_in_millseconds=*/1000},
     {"nanoseconds", /*nanoseconds_in_microseconds=*/1000},
     {nullptr, 0}};

template <MonthDayNanoField field>
struct PopulateMonthDayNano {
  using Traits = MonthDayNanoTraits<field>;
  using field_c_type = typename Traits::c_type;

  static Status Field(PyObject* obj, field_c_type* out, bool* found_attrs) {
    *out = 0;
    for (const MonthDayNanoAttrData* attr = &Traits::attrs[0]; attr->multiplier != 0;
         ++attr) {
      if (attr->multiplier != 1 &&
          ::arrow::internal::MultiplyWithOverflow(
              static_cast<field_c_type>(attr->multiplier), *out, out)) {
        return Status::Invalid("Overflow on: ", (attr - 1)->name,
                               " for: ", internal::PyObject_StdStringRepr(obj));
      }

      OwnedRef field_value(PyObject_GetAttrString(obj, attr->name));
      if (field_value.obj() == nullptr) {
        // No attribute present, skip  to the next one.
        PyErr_Clear();
        continue;
      }
      RETURN_IF_PYERROR();
      *found_attrs = true;
      field_c_type value;
      RETURN_NOT_OK(internal::CIntFromPython(field_value.obj(), &value, attr->name));
      if (::arrow::internal::AddWithOverflow(*out, value, out)) {
        return Status::Invalid("Overflow on: ", attr->name,
                               " for: ", internal::PyObject_StdStringRepr(obj));
      }
    }

    return Status::OK();
  }
};

// Utility for converting single python objects to their intermediate C representations
// which can be fed to the typed builders
class PyValue {
 public:
  // Type aliases for shorter signature definitions
  using I = PyObject*;
  using O = PyConversionOptions;

  // Used for null checking before actually converting the values
  static bool IsNull(const O& options, I obj) {
    if (options.from_pandas) {
      return internal::PandasObjectIsNull(obj);
    } else {
      return obj == Py_None;
    }
  }

  // Used for post-conversion numpy NaT sentinel checking
  static bool IsNaT(const TimestampType*, int64_t value) {
    return internal::npy_traits<NPY_DATETIME>::isnull(value);
  }

  // Used for post-conversion numpy NaT sentinel checking
  static bool IsNaT(const DurationType*, int64_t value) {
    return internal::npy_traits<NPY_TIMEDELTA>::isnull(value);
  }

  static Result<std::nullptr_t> Convert(const NullType*, const O&, I obj) {
    if (obj == Py_None) {
      return nullptr;
    } else {
      return Status::Invalid("Invalid null value");
    }
  }

  static Result<bool> Convert(const BooleanType*, const O&, I obj) {
    if (obj == Py_True) {
      return true;
    } else if (obj == Py_False) {
      return false;
    } else if (PyArray_IsScalar(obj, Bool)) {
      return reinterpret_cast<PyBoolScalarObject*>(obj)->obval == NPY_TRUE;
    } else {
      return internal::InvalidValue(obj, "tried to convert to boolean");
    }
  }

  template <typename T>
  static enable_if_integer<T, Result<typename T::c_type>> Convert(const T* type, const O&,
                                                                  I obj) {
    typename T::c_type value;
    auto status = internal::CIntFromPython(obj, &value);
    if (ARROW_PREDICT_TRUE(status.ok())) {
      return value;
    } else if (!internal::PyIntScalar_Check(obj)) {
      std::stringstream ss;
      ss << "tried to convert to " << type->ToString();
      return internal::InvalidValue(obj, ss.str());
    } else {
      return status;
    }
  }

  static Result<uint16_t> Convert(const HalfFloatType*, const O&, I obj) {
    uint16_t value;
    RETURN_NOT_OK(PyFloat_AsHalf(obj, &value));
    return value;
  }

  static Result<float> Convert(const FloatType*, const O&, I obj) {
    float value;
    if (internal::PyFloatScalar_Check(obj)) {
      value = static_cast<float>(PyFloat_AsDouble(obj));
      RETURN_IF_PYERROR();
    } else if (internal::PyIntScalar_Check(obj)) {
      RETURN_NOT_OK(internal::IntegerScalarToFloat32Safe(obj, &value));
    } else {
      return internal::InvalidValue(obj, "tried to convert to float32");
    }
    return value;
  }

  static Result<double> Convert(const DoubleType*, const O&, I obj) {
    double value;
    if (PyFloat_Check(obj)) {
      value = PyFloat_AS_DOUBLE(obj);
    } else if (internal::PyFloatScalar_Check(obj)) {
      // Other kinds of float-y things
      value = PyFloat_AsDouble(obj);
      RETURN_IF_PYERROR();
    } else if (internal::PyIntScalar_Check(obj)) {
      RETURN_NOT_OK(internal::IntegerScalarToDoubleSafe(obj, &value));
    } else {
      return internal::InvalidValue(obj, "tried to convert to double");
    }
    return value;
  }

  static Result<Decimal128> Convert(const Decimal128Type* type, const O&, I obj) {
    Decimal128 value;
    RETURN_NOT_OK(internal::DecimalFromPyObject(obj, *type, &value));
    return value;
  }

  static Result<Decimal256> Convert(const Decimal256Type* type, const O&, I obj) {
    Decimal256 value;
    RETURN_NOT_OK(internal::DecimalFromPyObject(obj, *type, &value));
    return value;
  }

  static Result<int32_t> Convert(const Date32Type*, const O&, I obj) {
    int32_t value;
    if (PyDate_Check(obj)) {
      auto pydate = reinterpret_cast<PyDateTime_Date*>(obj);
      value = static_cast<int32_t>(internal::PyDate_to_days(pydate));
    } else {
      RETURN_NOT_OK(
          internal::CIntFromPython(obj, &value, "Integer too large for date32"));
    }
    return value;
  }

  static Result<int64_t> Convert(const Date64Type*, const O&, I obj) {
    int64_t value;
    if (PyDateTime_Check(obj)) {
      auto pydate = reinterpret_cast<PyDateTime_DateTime*>(obj);
      value = internal::PyDateTime_to_ms(pydate);
      // Truncate any intraday milliseconds
      // TODO: introduce an option for this
      value -= value % 86400000LL;
    } else if (PyDate_Check(obj)) {
      auto pydate = reinterpret_cast<PyDateTime_Date*>(obj);
      value = internal::PyDate_to_ms(pydate);
    } else {
      RETURN_NOT_OK(
          internal::CIntFromPython(obj, &value, "Integer too large for date64"));
    }
    return value;
  }

  static Result<int32_t> Convert(const Time32Type* type, const O&, I obj) {
    int32_t value;
    if (PyTime_Check(obj)) {
      switch (type->unit()) {
        case TimeUnit::SECOND:
          value = static_cast<int32_t>(internal::PyTime_to_s(obj));
          break;
        case TimeUnit::MILLI:
          value = static_cast<int32_t>(internal::PyTime_to_ms(obj));
          break;
        default:
          return Status::UnknownError("Invalid time unit");
      }
    } else {
      RETURN_NOT_OK(internal::CIntFromPython(obj, &value, "Integer too large for int32"));
    }
    return value;
  }

  static Result<int64_t> Convert(const Time64Type* type, const O&, I obj) {
    int64_t value;
    if (PyTime_Check(obj)) {
      switch (type->unit()) {
        case TimeUnit::MICRO:
          value = internal::PyTime_to_us(obj);
          break;
        case TimeUnit::NANO:
          value = internal::PyTime_to_ns(obj);
          break;
        default:
          return Status::UnknownError("Invalid time unit");
      }
    } else {
      RETURN_NOT_OK(internal::CIntFromPython(obj, &value, "Integer too large for int64"));
    }
    return value;
  }

  static Result<int64_t> Convert(const TimestampType* type, const O& options, I obj) {
    int64_t value, offset;
    if (PyDateTime_Check(obj)) {
      if (ARROW_PREDICT_FALSE(options.ignore_timezone)) {
        offset = 0;
      } else {
        ARROW_ASSIGN_OR_RAISE(offset, internal::PyDateTime_utcoffset_s(obj));
      }
      auto dt = reinterpret_cast<PyDateTime_DateTime*>(obj);
      switch (type->unit()) {
        case TimeUnit::SECOND:
          value = internal::PyDateTime_to_s(dt) - offset;
          break;
        case TimeUnit::MILLI:
          value = internal::PyDateTime_to_ms(dt) - offset * 1000LL;
          break;
        case TimeUnit::MICRO:
          value = internal::PyDateTime_to_us(dt) - offset * 1000000LL;
          break;
        case TimeUnit::NANO:
          if (internal::IsPandasTimestamp(obj)) {
            // pd.Timestamp value attribute contains the offset from unix epoch
            // so no adjustment for timezone is need.
            OwnedRef nanos(PyObject_GetAttrString(obj, "value"));
            RETURN_IF_PYERROR();
            RETURN_NOT_OK(internal::CIntFromPython(nanos.obj(), &value));
          } else {
            // Conversion to nanoseconds can overflow -> check multiply of microseconds
            value = internal::PyDateTime_to_us(dt);
            if (arrow::internal::MultiplyWithOverflow(value, 1000LL, &value)) {
              return internal::InvalidValue(obj,
                                            "out of bounds for nanosecond resolution");
            }

            // Adjust with offset and check for overflow
            if (arrow::internal::SubtractWithOverflow(value, offset * 1000000000LL,
                                                      &value)) {
              return internal::InvalidValue(obj,
                                            "out of bounds for nanosecond resolution");
            }
          }
          break;
        default:
          return Status::UnknownError("Invalid time unit");
      }
    } else if (PyArray_CheckAnyScalarExact(obj)) {
      // validate that the numpy scalar has np.datetime64 dtype
      std::shared_ptr<DataType> numpy_type;
      RETURN_NOT_OK(NumPyDtypeToArrow(PyArray_DescrFromScalar(obj), &numpy_type));
      if (!numpy_type->Equals(*type)) {
        return Status::NotImplemented("Expected np.datetime64 but got: ",
                                      numpy_type->ToString());
      }
      return reinterpret_cast<PyDatetimeScalarObject*>(obj)->obval;
    } else {
      RETURN_NOT_OK(internal::CIntFromPython(obj, &value));
    }
    return value;
  }

  static Result<MonthDayNanoIntervalType::MonthDayNanos> Convert(
      const MonthDayNanoIntervalType* /*type*/, const O& /*options*/, I obj) {
    MonthDayNanoIntervalType::MonthDayNanos output;
    bool found_attrs = false;
    RETURN_NOT_OK(PopulateMonthDayNano<MonthDayNanoField::kMonths>::Field(
        obj, &output.months, &found_attrs));
    // on relativeoffset weeks is a property calculated from days.  On
    // DateOffset is is a field on its own. timedelta doesn't have a weeks
    // attribute.
    PyObject* pandas_date_offset_type = internal::BorrowPandasDataOffsetType();
    bool is_date_offset = pandas_date_offset_type == (PyObject*)Py_TYPE(obj);
    if (!is_date_offset) {
      RETURN_NOT_OK(PopulateMonthDayNano<MonthDayNanoField::kDaysOnly>::Field(
          obj, &output.days, &found_attrs));
    } else {
      RETURN_NOT_OK(PopulateMonthDayNano<MonthDayNanoField::kWeeksAndDays>::Field(
          obj, &output.days, &found_attrs));
    }
    RETURN_NOT_OK(PopulateMonthDayNano<MonthDayNanoField::kNanoseconds>::Field(
        obj, &output.nanoseconds, &found_attrs));

    // date_offset can have zero fields.
    if (found_attrs || is_date_offset) {
      return output;
    }
    if (PyTuple_Check(obj) && PyTuple_Size(obj) == 3) {
      RETURN_NOT_OK(internal::CIntFromPython(PyTuple_GET_ITEM(obj, 0), &output.months,
                                             "Months (tuple item #0) too large"));
      RETURN_NOT_OK(internal::CIntFromPython(PyTuple_GET_ITEM(obj, 1), &output.days,
                                             "Days (tuple item #1) too large"));
      RETURN_NOT_OK(internal::CIntFromPython(PyTuple_GET_ITEM(obj, 2),
                                             &output.nanoseconds,
                                             "Nanoseconds (tuple item #2) too large"));
      return output;
    }
    return Status::TypeError("No temporal attributes found on object.");
  }

  static Result<int64_t> Convert(const DurationType* type, const O&, I obj) {
    int64_t value;
    if (PyDelta_Check(obj)) {
      auto dt = reinterpret_cast<PyDateTime_Delta*>(obj);
      switch (type->unit()) {
        case TimeUnit::SECOND:
          value = internal::PyDelta_to_s(dt);
          break;
        case TimeUnit::MILLI:
          value = internal::PyDelta_to_ms(dt);
          break;
        case TimeUnit::MICRO: {
          ARROW_ASSIGN_OR_RAISE(value, internal::PyDelta_to_us(dt));
          break;
        }
        case TimeUnit::NANO:
          if (internal::IsPandasTimedelta(obj)) {
            OwnedRef nanos(PyObject_GetAttrString(obj, "value"));
            RETURN_IF_PYERROR();
            RETURN_NOT_OK(internal::CIntFromPython(nanos.obj(), &value));
          } else {
            ARROW_ASSIGN_OR_RAISE(value, internal::PyDelta_to_ns(dt));
          }
          break;
        default:
          return Status::UnknownError("Invalid time unit");
      }
    } else if (PyArray_CheckAnyScalarExact(obj)) {
      // validate that the numpy scalar has np.datetime64 dtype
      std::shared_ptr<DataType> numpy_type;
      RETURN_NOT_OK(NumPyDtypeToArrow(PyArray_DescrFromScalar(obj), &numpy_type));
      if (!numpy_type->Equals(*type)) {
        return Status::NotImplemented("Expected np.timedelta64 but got: ",
                                      numpy_type->ToString());
      }
      return reinterpret_cast<PyTimedeltaScalarObject*>(obj)->obval;
    } else {
      RETURN_NOT_OK(internal::CIntFromPython(obj, &value));
    }
    return value;
  }

  // The binary-like intermediate representation is PyBytesView because it keeps temporary
  // python objects alive (non-contiguous memoryview) and stores whether the original
  // object was unicode encoded or not, which is used for unicode -> bytes coersion if
  // there is a non-unicode object observed.

  static Status Convert(const BaseBinaryType*, const O&, I obj, PyBytesView& view) {
    return view.ParseString(obj);
  }

  static Status Convert(const FixedSizeBinaryType* type, const O&, I obj,
                        PyBytesView& view) {
    ARROW_RETURN_NOT_OK(view.ParseString(obj));
    if (view.size != type->byte_width()) {
      std::stringstream ss;
      ss << "expected to be length " << type->byte_width() << " was " << view.size;
      return internal::InvalidValue(obj, ss.str());
    } else {
      return Status::OK();
    }
  }

  template <typename T>
  static enable_if_string<T, Status> Convert(const T*, const O& options, I obj,
                                             PyBytesView& view) {
    if (options.strict) {
      // Strict conversion, force output to be unicode / utf8 and validate that
      // any binary values are utf8
      ARROW_RETURN_NOT_OK(view.ParseString(obj, true));
      if (!view.is_utf8) {
        return internal::InvalidValue(obj, "was not a utf8 string");
      }
      return Status::OK();
    } else {
      // Non-strict conversion; keep track of whether values are unicode or bytes
      return view.ParseString(obj);
    }
  }

  static Result<bool> Convert(const DataType* type, const O&, I obj) {
    return Status::NotImplemented("PyValue::Convert is not implemented for type ", type);
  }
};

// The base Converter class is a mixin with predefined behavior and constructors.
class PyConverter : public Converter<PyObject*, PyConversionOptions> {
 public:
  // Iterate over the input values and defer the conversion to the Append method
  Status Extend(PyObject* values, int64_t size, int64_t offset = 0) override {
    DCHECK_GE(size, offset);
    /// Ensure we've allocated enough space
    RETURN_NOT_OK(this->Reserve(size - offset));
    // Iterate over the items adding each one
    return internal::VisitSequence(
        values, offset,
        [this](PyObject* item, bool* /* unused */) { return this->Append(item); });
  }

  // Convert and append a sequence of values masked with a numpy array
  Status ExtendMasked(PyObject* values, PyObject* mask, int64_t size,
                      int64_t offset = 0) override {
    DCHECK_GE(size, offset);
    /// Ensure we've allocated enough space
    RETURN_NOT_OK(this->Reserve(size - offset));
    // Iterate over the items adding each one
    return internal::VisitSequenceMasked(
        values, mask, offset, [this](PyObject* item, bool is_masked, bool* /* unused */) {
          if (is_masked) {
            return this->AppendNull();
          } else {
            // This will also apply the null-checking convention in the event
            // that the value is not masked
            return this->Append(item);  // perhaps use AppendValue instead?
          }
        });
  }
};

template <typename T, typename Enable = void>
class PyPrimitiveConverter;

template <typename T>
class PyListConverter;

template <typename U, typename Enable = void>
class PyDictionaryConverter;

class PyStructConverter;

template <typename T, typename Enable = void>
struct PyConverterTrait;

template <typename T>
struct PyConverterTrait<
    T, enable_if_t<(!is_nested_type<T>::value && !is_interval_type<T>::value &&
                    !is_extension_type<T>::value) ||
                   std::is_same<T, MonthDayNanoIntervalType>::value>> {
  using type = PyPrimitiveConverter<T>;
};

template <typename T>
struct PyConverterTrait<T, enable_if_list_like<T>> {
  using type = PyListConverter<T>;
};

template <>
struct PyConverterTrait<StructType> {
  using type = PyStructConverter;
};

template <>
struct PyConverterTrait<DictionaryType> {
  template <typename T>
  using dictionary_type = PyDictionaryConverter<T>;
};

template <typename T>
class PyPrimitiveConverter<T, enable_if_null<T>>
    : public PrimitiveConverter<T, PyConverter> {
 public:
  Status Append(PyObject* value) override {
    if (PyValue::IsNull(this->options_, value)) {
      return this->primitive_builder_->AppendNull();
    } else {
      ARROW_ASSIGN_OR_RAISE(
          auto converted, PyValue::Convert(this->primitive_type_, this->options_, value));
      return this->primitive_builder_->Append(converted);
    }
  }
};

template <typename T>
class PyPrimitiveConverter<
    T, enable_if_t<is_boolean_type<T>::value || is_number_type<T>::value ||
                   is_decimal_type<T>::value || is_date_type<T>::value ||
                   is_time_type<T>::value ||
                   std::is_same<MonthDayNanoIntervalType, T>::value>>
    : public PrimitiveConverter<T, PyConverter> {
 public:
  Status Append(PyObject* value) override {
    // Since the required space has been already allocated in the Extend functions we can
    // rely on the Unsafe builder API which improves the performance.
    if (PyValue::IsNull(this->options_, value)) {
      this->primitive_builder_->UnsafeAppendNull();
    } else {
      ARROW_ASSIGN_OR_RAISE(
          auto converted, PyValue::Convert(this->primitive_type_, this->options_, value));
      this->primitive_builder_->UnsafeAppend(converted);
    }
    return Status::OK();
  }
};

template <typename T>
class PyPrimitiveConverter<
    T, enable_if_t<is_timestamp_type<T>::value || is_duration_type<T>::value>>
    : public PrimitiveConverter<T, PyConverter> {
 public:
  Status Append(PyObject* value) override {
    if (PyValue::IsNull(this->options_, value)) {
      this->primitive_builder_->UnsafeAppendNull();
    } else {
      ARROW_ASSIGN_OR_RAISE(
          auto converted, PyValue::Convert(this->primitive_type_, this->options_, value));
      // Numpy NaT sentinels can be checked after the conversion
      if (PyArray_CheckAnyScalarExact(value) &&
          PyValue::IsNaT(this->primitive_type_, converted)) {
        this->primitive_builder_->UnsafeAppendNull();
      } else {
        this->primitive_builder_->UnsafeAppend(converted);
      }
    }
    return Status::OK();
  }
};

template <typename T>
class PyPrimitiveConverter<T, enable_if_t<std::is_same<T, FixedSizeBinaryType>::value>>
    : public PrimitiveConverter<T, PyConverter> {
 public:
  Status Append(PyObject* value) override {
    if (PyValue::IsNull(this->options_, value)) {
      this->primitive_builder_->UnsafeAppendNull();
    } else {
      ARROW_RETURN_NOT_OK(
          PyValue::Convert(this->primitive_type_, this->options_, value, view_));
      ARROW_RETURN_NOT_OK(this->primitive_builder_->ReserveData(view_.size));
      this->primitive_builder_->UnsafeAppend(view_.bytes);
    }
    return Status::OK();
  }

 protected:
  PyBytesView view_;
};

template <typename T>
class PyPrimitiveConverter<T, enable_if_base_binary<T>>
    : public PrimitiveConverter<T, PyConverter> {
 public:
  using OffsetType = typename T::offset_type;

  Status Append(PyObject* value) override {
    if (PyValue::IsNull(this->options_, value)) {
      this->primitive_builder_->UnsafeAppendNull();
    } else {
      ARROW_RETURN_NOT_OK(
          PyValue::Convert(this->primitive_type_, this->options_, value, view_));
      if (!view_.is_utf8) {
        // observed binary value
        observed_binary_ = true;
      }
      // Since we don't know the varying length input size in advance, we need to
      // reserve space in the value builder one by one. ReserveData raises CapacityError
      // if the value would not fit into the array.
      ARROW_RETURN_NOT_OK(this->primitive_builder_->ReserveData(view_.size));
      this->primitive_builder_->UnsafeAppend(view_.bytes,
                                             static_cast<OffsetType>(view_.size));
    }
    return Status::OK();
  }

  Result<std::shared_ptr<Array>> ToArray() override {
    ARROW_ASSIGN_OR_RAISE(auto array, (PrimitiveConverter<T, PyConverter>::ToArray()));
    if (observed_binary_) {
      // if we saw any non-unicode, cast results to BinaryArray
      auto binary_type = TypeTraits<typename T::PhysicalType>::type_singleton();
      return array->View(binary_type);
    } else {
      return array;
    }
  }

 protected:
  PyBytesView view_;
  bool observed_binary_ = false;
};

template <typename U>
class PyDictionaryConverter<U, enable_if_has_c_type<U>>
    : public DictionaryConverter<U, PyConverter> {
 public:
  Status Append(PyObject* value) override {
    if (PyValue::IsNull(this->options_, value)) {
      return this->value_builder_->AppendNull();
    } else {
      ARROW_ASSIGN_OR_RAISE(auto converted,
                            PyValue::Convert(this->value_type_, this->options_, value));
      return this->value_builder_->Append(converted);
    }
  }
};

template <typename U>
class PyDictionaryConverter<U, enable_if_has_string_view<U>>
    : public DictionaryConverter<U, PyConverter> {
 public:
  Status Append(PyObject* value) override {
    if (PyValue::IsNull(this->options_, value)) {
      return this->value_builder_->AppendNull();
    } else {
      ARROW_RETURN_NOT_OK(
          PyValue::Convert(this->value_type_, this->options_, value, view_));
      return this->value_builder_->Append(view_.bytes, static_cast<int32_t>(view_.size));
    }
  }

 protected:
  PyBytesView view_;
};

template <typename T>
class PyListConverter : public ListConverter<T, PyConverter, PyConverterTrait> {
 public:
  Status Append(PyObject* value) override {
    if (PyValue::IsNull(this->options_, value)) {
      return this->list_builder_->AppendNull();
    }

    RETURN_NOT_OK(this->list_builder_->Append());
    if (PyArray_Check(value)) {
      RETURN_NOT_OK(AppendNdarray(value));
    } else if (PySequence_Check(value)) {
      RETURN_NOT_OK(AppendSequence(value));
    } else if (PySet_Check(value) || (Py_TYPE(value) == &PyDictValues_Type)) {
      RETURN_NOT_OK(AppendIterable(value));
    } else if (PyDict_Check(value) && this->options_.type->id() == Type::MAP) {
      // Branch to support Python Dict with `map` DataType.
      auto items = PyDict_Items(value);
      OwnedRef item_ref(items);
      RETURN_NOT_OK(AppendSequence(items));
    } else {
      return internal::InvalidType(
          value, "was not a sequence or recognized null for conversion to list type");
    }

    return ValidateBuilder(this->list_type_);
  }

 protected:
  Status ValidateBuilder(const MapType*) {
    if (this->list_builder_->key_builder()->null_count() > 0) {
      return Status::Invalid("Invalid Map: key field can not contain null values");
    } else {
      return Status::OK();
    }
  }

  Status ValidateBuilder(const BaseListType*) { return Status::OK(); }

  Status AppendSequence(PyObject* value) {
    int64_t size = static_cast<int64_t>(PySequence_Size(value));
    RETURN_NOT_OK(this->list_builder_->ValidateOverflow(size));
    return this->value_converter_->Extend(value, size);
  }

  Status AppendIterable(PyObject* value) {
    PyObject* iterator = PyObject_GetIter(value);
    OwnedRef iter_ref(iterator);
    while (PyObject* item = PyIter_Next(iterator)) {
      OwnedRef item_ref(item);
      RETURN_NOT_OK(this->value_converter_->Reserve(1));
      RETURN_NOT_OK(this->value_converter_->Append(item));
    }
    return Status::OK();
  }

  Status AppendNdarray(PyObject* value) {
    PyArrayObject* ndarray = reinterpret_cast<PyArrayObject*>(value);
    if (PyArray_NDIM(ndarray) != 1) {
      return Status::Invalid("Can only convert 1-dimensional array values");
    }
    const int64_t size = PyArray_SIZE(ndarray);
    RETURN_NOT_OK(this->list_builder_->ValidateOverflow(size));

    const auto value_type = this->value_converter_->builder()->type();
    switch (value_type->id()) {
// If the value type does not match the expected NumPy dtype, then fall through
// to a slower PySequence-based path
#define LIST_FAST_CASE(TYPE_ID, TYPE, NUMPY_TYPE)         \
  case Type::TYPE_ID: {                                   \
    if (PyArray_DESCR(ndarray)->type_num != NUMPY_TYPE) { \
      return this->value_converter_->Extend(value, size); \
    }                                                     \
    return AppendNdarrayTyped<TYPE, NUMPY_TYPE>(ndarray); \
  }
      LIST_FAST_CASE(BOOL, BooleanType, NPY_BOOL)
      LIST_FAST_CASE(UINT8, UInt8Type, NPY_UINT8)
      LIST_FAST_CASE(INT8, Int8Type, NPY_INT8)
      LIST_FAST_CASE(UINT16, UInt16Type, NPY_UINT16)
      LIST_FAST_CASE(INT16, Int16Type, NPY_INT16)
      LIST_FAST_CASE(UINT32, UInt32Type, NPY_UINT32)
      LIST_FAST_CASE(INT32, Int32Type, NPY_INT32)
      LIST_FAST_CASE(UINT64, UInt64Type, NPY_UINT64)
      LIST_FAST_CASE(INT64, Int64Type, NPY_INT64)
      LIST_FAST_CASE(HALF_FLOAT, HalfFloatType, NPY_FLOAT16)
      LIST_FAST_CASE(FLOAT, FloatType, NPY_FLOAT)
      LIST_FAST_CASE(DOUBLE, DoubleType, NPY_DOUBLE)
      LIST_FAST_CASE(TIMESTAMP, TimestampType, NPY_DATETIME)
      LIST_FAST_CASE(DURATION, DurationType, NPY_TIMEDELTA)
#undef LIST_FAST_CASE
      default: {
        return this->value_converter_->Extend(value, size);
      }
    }
  }

  template <typename ArrowType, int NUMPY_TYPE>
  Status AppendNdarrayTyped(PyArrayObject* ndarray) {
    // no need to go through the conversion
    using NumpyTrait = internal::npy_traits<NUMPY_TYPE>;
    using NumpyType = typename NumpyTrait::value_type;
    using ValueBuilderType = typename TypeTraits<ArrowType>::BuilderType;

    const bool null_sentinels_possible =
        // Always treat Numpy's NaT as null
        NUMPY_TYPE == NPY_DATETIME || NUMPY_TYPE == NPY_TIMEDELTA ||
        // Observing pandas's null sentinels
        (this->options_.from_pandas && NumpyTrait::supports_nulls);

    auto value_builder =
        checked_cast<ValueBuilderType*>(this->value_converter_->builder().get());

    Ndarray1DIndexer<NumpyType> values(ndarray);
    if (null_sentinels_possible) {
      for (int64_t i = 0; i < values.size(); ++i) {
        if (NumpyTrait::isnull(values[i])) {
          RETURN_NOT_OK(value_builder->AppendNull());
        } else {
          RETURN_NOT_OK(value_builder->Append(values[i]));
        }
      }
    } else if (!values.is_strided()) {
      RETURN_NOT_OK(value_builder->AppendValues(values.data(), values.size()));
    } else {
      for (int64_t i = 0; i < values.size(); ++i) {
        RETURN_NOT_OK(value_builder->Append(values[i]));
      }
    }
    return Status::OK();
  }
};

class PyStructConverter : public StructConverter<PyConverter, PyConverterTrait> {
 public:
  Status Append(PyObject* value) override {
    if (PyValue::IsNull(this->options_, value)) {
      return this->struct_builder_->AppendNull();
    }
    switch (input_kind_) {
      case InputKind::DICT:
        RETURN_NOT_OK(this->struct_builder_->Append());
        return AppendDict(value);
      case InputKind::TUPLE:
        RETURN_NOT_OK(this->struct_builder_->Append());
        return AppendTuple(value);
      case InputKind::ITEMS:
        RETURN_NOT_OK(this->struct_builder_->Append());
        return AppendItems(value);
      default:
        RETURN_NOT_OK(InferInputKind(value));
        return Append(value);
    }
  }

 protected:
  Status Init(MemoryPool* pool) override {
    RETURN_NOT_OK((StructConverter<PyConverter, PyConverterTrait>::Init(pool)));

    // Store the field names as a PyObjects for dict matching
    num_fields_ = this->struct_type_->num_fields();
    bytes_field_names_.reset(PyList_New(num_fields_));
    unicode_field_names_.reset(PyList_New(num_fields_));
    RETURN_IF_PYERROR();

    for (int i = 0; i < num_fields_; i++) {
      const auto& field_name = this->struct_type_->field(i)->name();
      PyObject* bytes = PyBytes_FromStringAndSize(field_name.c_str(), field_name.size());
      PyObject* unicode =
          PyUnicode_FromStringAndSize(field_name.c_str(), field_name.size());
      RETURN_IF_PYERROR();
      PyList_SET_ITEM(bytes_field_names_.obj(), i, bytes);
      PyList_SET_ITEM(unicode_field_names_.obj(), i, unicode);
    }
    return Status::OK();
  }

  Status InferInputKind(PyObject* value) {
    // Infer input object's type, note that heterogeneous sequences are not allowed
    if (PyDict_Check(value)) {
      input_kind_ = InputKind::DICT;
    } else if (PyTuple_Check(value)) {
      input_kind_ = InputKind::TUPLE;
    } else if (PySequence_Check(value)) {
      input_kind_ = InputKind::ITEMS;
    } else {
      return internal::InvalidType(value,
                                   "was not a dict, tuple, or recognized null value "
                                   "for conversion to struct type");
    }
    return Status::OK();
  }

  Status InferKeyKind(PyObject* items) {
    for (int i = 0; i < PySequence_Length(items); i++) {
      // retrieve the key from the passed key-value pairs
      ARROW_ASSIGN_OR_RAISE(auto pair, GetKeyValuePair(items, i));

      // check key exists between the unicode field names
      bool do_contain = PySequence_Contains(unicode_field_names_.obj(), pair.first);
      RETURN_IF_PYERROR();
      if (do_contain) {
        key_kind_ = KeyKind::UNICODE;
        return Status::OK();
      }

      // check key exists between the bytes field names
      do_contain = PySequence_Contains(bytes_field_names_.obj(), pair.first);
      RETURN_IF_PYERROR();
      if (do_contain) {
        key_kind_ = KeyKind::BYTES;
        return Status::OK();
      }
    }
    return Status::OK();
  }

  Status AppendEmpty() {
    for (int i = 0; i < num_fields_; i++) {
      RETURN_NOT_OK(this->children_[i]->Append(Py_None));
    }
    return Status::OK();
  }

  Status AppendTuple(PyObject* tuple) {
    if (!PyTuple_Check(tuple)) {
      return internal::InvalidType(tuple, "was expecting a tuple");
    }
    if (PyTuple_GET_SIZE(tuple) != num_fields_) {
      return Status::Invalid("Tuple size must be equal to number of struct fields");
    }
    for (int i = 0; i < num_fields_; i++) {
      PyObject* value = PyTuple_GET_ITEM(tuple, i);
      RETURN_NOT_OK(this->children_[i]->Append(value));
    }
    return Status::OK();
  }

  Status AppendDict(PyObject* dict) {
    if (!PyDict_Check(dict)) {
      return internal::InvalidType(dict, "was expecting a dict");
    }
    switch (key_kind_) {
      case KeyKind::UNICODE:
        return AppendDict(dict, unicode_field_names_.obj());
      case KeyKind::BYTES:
        return AppendDict(dict, bytes_field_names_.obj());
      default:
        RETURN_NOT_OK(InferKeyKind(PyDict_Items(dict)));
        if (key_kind_ == KeyKind::UNKNOWN) {
          // was unable to infer the type which means that all keys are absent
          return AppendEmpty();
        } else {
          return AppendDict(dict);
        }
    }
  }

  Status AppendItems(PyObject* items) {
    if (!PySequence_Check(items)) {
      return internal::InvalidType(items, "was expecting a sequence of key-value items");
    }
    switch (key_kind_) {
      case KeyKind::UNICODE:
        return AppendItems(items, unicode_field_names_.obj());
      case KeyKind::BYTES:
        return AppendItems(items, bytes_field_names_.obj());
      default:
        RETURN_NOT_OK(InferKeyKind(items));
        if (key_kind_ == KeyKind::UNKNOWN) {
          // was unable to infer the type which means that all keys are absent
          return AppendEmpty();
        } else {
          return AppendItems(items);
        }
    }
  }

  Status AppendDict(PyObject* dict, PyObject* field_names) {
    // NOTE we're ignoring any extraneous dict items
    for (int i = 0; i < num_fields_; i++) {
      PyObject* name = PyList_GET_ITEM(field_names, i);  // borrowed
      PyObject* value = PyDict_GetItem(dict, name);      // borrowed
      if (value == NULL) {
        RETURN_IF_PYERROR();
      }
      RETURN_NOT_OK(this->children_[i]->Append(value ? value : Py_None));
    }
    return Status::OK();
  }

  Result<std::pair<PyObject*, PyObject*>> GetKeyValuePair(PyObject* seq, int index) {
    PyObject* pair = PySequence_GetItem(seq, index);
    RETURN_IF_PYERROR();
    if (!PyTuple_Check(pair) || PyTuple_Size(pair) != 2) {
      return internal::InvalidType(pair, "was expecting tuple of (key, value) pair");
    }
    PyObject* key = PyTuple_GetItem(pair, 0);
    RETURN_IF_PYERROR();
    PyObject* value = PyTuple_GetItem(pair, 1);
    RETURN_IF_PYERROR();
    return std::make_pair(key, value);
  }

  Status AppendItems(PyObject* items, PyObject* field_names) {
    auto length = static_cast<int>(PySequence_Size(items));
    RETURN_IF_PYERROR();

    // append the values for the defined fields
    for (int i = 0; i < std::min(num_fields_, length); i++) {
      // retrieve the key-value pair
      ARROW_ASSIGN_OR_RAISE(auto pair, GetKeyValuePair(items, i));

      // validate that the key and the field name are equal
      PyObject* name = PyList_GET_ITEM(field_names, i);
      bool are_equal = PyObject_RichCompareBool(pair.first, name, Py_EQ);
      RETURN_IF_PYERROR();

      // finally append to the respective child builder
      if (are_equal) {
        RETURN_NOT_OK(this->children_[i]->Append(pair.second));
      } else {
        ARROW_ASSIGN_OR_RAISE(auto key_view, PyBytesView::FromString(pair.first));
        ARROW_ASSIGN_OR_RAISE(auto name_view, PyBytesView::FromString(name));
        return Status::Invalid("The expected field name is `", name_view.bytes, "` but `",
                               key_view.bytes, "` was given");
      }
    }
    // insert null values for missing fields
    for (int i = length; i < num_fields_; i++) {
      RETURN_NOT_OK(this->children_[i]->AppendNull());
    }
    return Status::OK();
  }

  // Whether we're converting from a sequence of dicts or tuples or list of pairs
  enum class InputKind { UNKNOWN, DICT, TUPLE, ITEMS } input_kind_ = InputKind::UNKNOWN;
  // Whether the input dictionary keys' type is python bytes or unicode
  enum class KeyKind { UNKNOWN, BYTES, UNICODE } key_kind_ = KeyKind::UNKNOWN;
  // Store the field names as a PyObjects for dict matching
  OwnedRef bytes_field_names_;
  OwnedRef unicode_field_names_;
  // Store the number of fields for later reuse
  int num_fields_;
};

// Convert *obj* to a sequence if necessary
// Fill *size* to its length.  If >= 0 on entry, *size* is an upper size
// bound that may lead to truncation.
Status ConvertToSequenceAndInferSize(PyObject* obj, PyObject** seq, int64_t* size) {
  if (PySequence_Check(obj)) {
    // obj is already a sequence
    int64_t real_size = static_cast<int64_t>(PySequence_Size(obj));
    if (*size < 0) {
      *size = real_size;
    } else {
      *size = std::min(real_size, *size);
    }
    Py_INCREF(obj);
    *seq = obj;
  } else if (*size < 0) {
    // unknown size, exhaust iterator
    *seq = PySequence_List(obj);
    RETURN_IF_PYERROR();
    *size = static_cast<int64_t>(PyList_GET_SIZE(*seq));
  } else {
    // size is known but iterator could be infinite
    Py_ssize_t i, n = *size;
    PyObject* iter = PyObject_GetIter(obj);
    RETURN_IF_PYERROR();
    OwnedRef iter_ref(iter);
    PyObject* lst = PyList_New(n);
    RETURN_IF_PYERROR();
    for (i = 0; i < n; i++) {
      PyObject* item = PyIter_Next(iter);
      if (!item) {
        // either an error occurred or the iterator ended
        RETURN_IF_PYERROR();
        break;
      }
      PyList_SET_ITEM(lst, i, item);
    }
    // Shrink list if len(iterator) < size
    if (i < n && PyList_SetSlice(lst, i, n, NULL)) {
      Py_DECREF(lst);
      RETURN_IF_PYERROR();
    }
    *seq = lst;
    *size = std::min<int64_t>(i, *size);
  }
  return Status::OK();
}

}  // namespace

Result<std::shared_ptr<ChunkedArray>> ConvertPySequence(PyObject* obj, PyObject* mask,
                                                        PyConversionOptions options,
                                                        MemoryPool* pool) {
  PyAcquireGIL lock;

  PyObject* seq = nullptr;
  OwnedRef tmp_seq_nanny;

  ARROW_ASSIGN_OR_RAISE(auto is_pandas_imported, internal::IsModuleImported("pandas"));
  if (is_pandas_imported) {
    // If pandas has been already imported initialize the static pandas objects to
    // support converting from pd.Timedelta and pd.Timestamp objects
    internal::InitPandasStaticData();
  }

  int64_t size = options.size;
  RETURN_NOT_OK(ConvertToSequenceAndInferSize(obj, &seq, &size));
  tmp_seq_nanny.reset(seq);

  // In some cases, type inference may be "loose", like strings. If the user
  // passed pa.string(), then we will error if we encounter any non-UTF8
  // value. If not, then we will allow the result to be a BinaryArray
  if (options.type == nullptr) {
    ARROW_ASSIGN_OR_RAISE(options.type, InferArrowType(seq, mask, options.from_pandas));
    options.strict = false;
  } else {
    options.strict = true;
  }
  DCHECK_GE(size, 0);

  ARROW_ASSIGN_OR_RAISE(auto converter, (MakeConverter<PyConverter, PyConverterTrait>(
                                            options.type, options, pool)));
  if (converter->may_overflow()) {
    // The converter hierarchy contains binary- or list-like builders which can overflow
    // depending on the input values. Wrap the converter with a chunker which detects
    // the overflow and automatically creates new chunks.
    ARROW_ASSIGN_OR_RAISE(auto chunked_converter, MakeChunker(std::move(converter)));
    if (mask != nullptr && mask != Py_None) {
      RETURN_NOT_OK(chunked_converter->ExtendMasked(seq, mask, size));
    } else {
      RETURN_NOT_OK(chunked_converter->Extend(seq, size));
    }
    return chunked_converter->ToChunkedArray();
  } else {
    // If the converter can't overflow spare the capacity error checking on the hot-path,
    // this improves the performance roughly by ~10% for primitive types.
    if (mask != nullptr && mask != Py_None) {
      RETURN_NOT_OK(converter->ExtendMasked(seq, mask, size));
    } else {
      RETURN_NOT_OK(converter->Extend(seq, size));
    }
    return converter->ToChunkedArray();
  }
}

}  // namespace py
}  // namespace arrow

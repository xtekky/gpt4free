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

#include "arrow/python/inference.h"
#include "arrow/python/numpy_interop.h"

#include <datetime.h>

#include <algorithm>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "arrow/status.h"
#include "arrow/util/decimal.h"
#include "arrow/util/logging.h"

#include "arrow/python/datetime.h"
#include "arrow/python/decimal.h"
#include "arrow/python/helpers.h"
#include "arrow/python/iterators.h"
#include "arrow/python/numpy_convert.h"

namespace arrow {
namespace py {
namespace {
// Assigns a tuple to interval_types_tuple containing the nametuple for
// MonthDayNanoIntervalType and if present dateutil's relativedelta and
// pandas DateOffset.
Status ImportPresentIntervalTypes(OwnedRefNoGIL* interval_types_tuple) {
  OwnedRef relative_delta_module;
  // These are Optional imports so swallow errors.
  OwnedRef relative_delta_type;
  // Try to import pandas to get types.
  internal::InitPandasStaticData();
  if (internal::ImportModule("dateutil.relativedelta", &relative_delta_module).ok()) {
    RETURN_NOT_OK(internal::ImportFromModule(relative_delta_module.obj(), "relativedelta",
                                             &relative_delta_type));
  }

  PyObject* date_offset_type = internal::BorrowPandasDataOffsetType();
  interval_types_tuple->reset(
      PyTuple_New(1 + (date_offset_type != nullptr ? 1 : 0) +
                  (relative_delta_type.obj() != nullptr ? 1 : 0)));
  RETURN_IF_PYERROR();
  int index = 0;
  PyTuple_SetItem(interval_types_tuple->obj(), index++,
                  internal::NewMonthDayNanoTupleType());
  RETURN_IF_PYERROR();
  if (date_offset_type != nullptr) {
    Py_XINCREF(date_offset_type);
    PyTuple_SetItem(interval_types_tuple->obj(), index++, date_offset_type);
    RETURN_IF_PYERROR();
  }
  if (relative_delta_type.obj() != nullptr) {
    PyTuple_SetItem(interval_types_tuple->obj(), index++, relative_delta_type.detach());
    RETURN_IF_PYERROR();
  }
  return Status::OK();
}

}  // namespace

#define _NUMPY_UNIFY_NOOP(DTYPE) \
  case NPY_##DTYPE:              \
    return OK;

#define _NUMPY_UNIFY_PROMOTE(DTYPE) \
  case NPY_##DTYPE:                 \
    current_type_num_ = dtype;      \
    current_dtype_ = descr;         \
    return OK;

#define _NUMPY_UNIFY_PROMOTE_TO(DTYPE, NEW_TYPE)               \
  case NPY_##DTYPE:                                            \
    current_type_num_ = NPY_##NEW_TYPE;                        \
    current_dtype_ = PyArray_DescrFromType(current_type_num_); \
    return OK;

// Form a consensus NumPy dtype to use for Arrow conversion for a
// collection of dtype objects observed one at a time
class NumPyDtypeUnifier {
 public:
  enum Action { OK, INVALID };

  NumPyDtypeUnifier() : current_type_num_(-1), current_dtype_(nullptr) {}

  Status InvalidMix(int new_dtype) {
    return Status::Invalid("Cannot mix NumPy dtypes ",
                           GetNumPyTypeName(current_type_num_), " and ",
                           GetNumPyTypeName(new_dtype));
  }

  int Observe_BOOL(PyArray_Descr* descr, int dtype) { return INVALID; }

  int Observe_INT8(PyArray_Descr* descr, int dtype) {
    switch (dtype) {
      _NUMPY_UNIFY_PROMOTE(INT16);
      _NUMPY_UNIFY_PROMOTE(INT32);
      _NUMPY_UNIFY_PROMOTE(INT64);
      _NUMPY_UNIFY_PROMOTE(FLOAT32);
      _NUMPY_UNIFY_PROMOTE(FLOAT64);
      default:
        return INVALID;
    }
  }

  int Observe_INT16(PyArray_Descr* descr, int dtype) {
    switch (dtype) {
      _NUMPY_UNIFY_NOOP(INT8);
      _NUMPY_UNIFY_PROMOTE(INT32);
      _NUMPY_UNIFY_PROMOTE(INT64);
      _NUMPY_UNIFY_NOOP(UINT8);
      _NUMPY_UNIFY_PROMOTE(FLOAT32);
      _NUMPY_UNIFY_PROMOTE(FLOAT64);
      default:
        return INVALID;
    }
  }

  int Observe_INT32(PyArray_Descr* descr, int dtype) {
    switch (dtype) {
      _NUMPY_UNIFY_NOOP(INT8);
      _NUMPY_UNIFY_NOOP(INT16);
      _NUMPY_UNIFY_PROMOTE(INT32);
      _NUMPY_UNIFY_PROMOTE(INT64);
      _NUMPY_UNIFY_NOOP(UINT8);
      _NUMPY_UNIFY_NOOP(UINT16);
      _NUMPY_UNIFY_PROMOTE_TO(FLOAT32, FLOAT64);
      _NUMPY_UNIFY_PROMOTE(FLOAT64);
      default:
        return INVALID;
    }
  }

  int Observe_INT64(PyArray_Descr* descr, int dtype) {
    switch (dtype) {
      _NUMPY_UNIFY_NOOP(INT8);
      _NUMPY_UNIFY_NOOP(INT16);
      _NUMPY_UNIFY_NOOP(INT32);
      _NUMPY_UNIFY_NOOP(INT64);
      _NUMPY_UNIFY_NOOP(UINT8);
      _NUMPY_UNIFY_NOOP(UINT16);
      _NUMPY_UNIFY_NOOP(UINT32);
      _NUMPY_UNIFY_PROMOTE_TO(FLOAT32, FLOAT64);
      _NUMPY_UNIFY_PROMOTE(FLOAT64);
      default:
        return INVALID;
    }
  }

  int Observe_UINT8(PyArray_Descr* descr, int dtype) {
    switch (dtype) {
      _NUMPY_UNIFY_PROMOTE(UINT16);
      _NUMPY_UNIFY_PROMOTE(UINT32);
      _NUMPY_UNIFY_PROMOTE(UINT64);
      _NUMPY_UNIFY_PROMOTE(FLOAT32);
      _NUMPY_UNIFY_PROMOTE(FLOAT64);
      default:
        return INVALID;
    }
  }

  int Observe_UINT16(PyArray_Descr* descr, int dtype) {
    switch (dtype) {
      _NUMPY_UNIFY_NOOP(UINT8);
      _NUMPY_UNIFY_PROMOTE(UINT32);
      _NUMPY_UNIFY_PROMOTE(UINT64);
      _NUMPY_UNIFY_PROMOTE(FLOAT32);
      _NUMPY_UNIFY_PROMOTE(FLOAT64);
      default:
        return INVALID;
    }
  }

  int Observe_UINT32(PyArray_Descr* descr, int dtype) {
    switch (dtype) {
      _NUMPY_UNIFY_NOOP(UINT8);
      _NUMPY_UNIFY_NOOP(UINT16);
      _NUMPY_UNIFY_PROMOTE(UINT64);
      _NUMPY_UNIFY_PROMOTE_TO(FLOAT32, FLOAT64);
      _NUMPY_UNIFY_PROMOTE(FLOAT64);
      default:
        return INVALID;
    }
  }

  int Observe_UINT64(PyArray_Descr* descr, int dtype) {
    switch (dtype) {
      _NUMPY_UNIFY_NOOP(UINT8);
      _NUMPY_UNIFY_NOOP(UINT16);
      _NUMPY_UNIFY_NOOP(UINT32);
      _NUMPY_UNIFY_PROMOTE_TO(FLOAT32, FLOAT64);
      _NUMPY_UNIFY_PROMOTE(FLOAT64);
      default:
        return INVALID;
    }
  }

  int Observe_FLOAT16(PyArray_Descr* descr, int dtype) {
    switch (dtype) {
      _NUMPY_UNIFY_PROMOTE(FLOAT32);
      _NUMPY_UNIFY_PROMOTE(FLOAT64);
      default:
        return INVALID;
    }
  }

  int Observe_FLOAT32(PyArray_Descr* descr, int dtype) {
    switch (dtype) {
      _NUMPY_UNIFY_NOOP(INT8);
      _NUMPY_UNIFY_NOOP(INT16);
      _NUMPY_UNIFY_NOOP(INT32);
      _NUMPY_UNIFY_NOOP(INT64);
      _NUMPY_UNIFY_NOOP(UINT8);
      _NUMPY_UNIFY_NOOP(UINT16);
      _NUMPY_UNIFY_NOOP(UINT32);
      _NUMPY_UNIFY_NOOP(UINT64);
      _NUMPY_UNIFY_PROMOTE(FLOAT64);
      default:
        return INVALID;
    }
  }

  int Observe_FLOAT64(PyArray_Descr* descr, int dtype) {
    switch (dtype) {
      _NUMPY_UNIFY_NOOP(INT8);
      _NUMPY_UNIFY_NOOP(INT16);
      _NUMPY_UNIFY_NOOP(INT32);
      _NUMPY_UNIFY_NOOP(INT64);
      _NUMPY_UNIFY_NOOP(UINT8);
      _NUMPY_UNIFY_NOOP(UINT16);
      _NUMPY_UNIFY_NOOP(UINT32);
      _NUMPY_UNIFY_NOOP(UINT64);
      default:
        return INVALID;
    }
  }

  int Observe_DATETIME(PyArray_Descr* dtype_obj) {
    // TODO: check that units are all the same
    return OK;
  }

  Status Observe(PyArray_Descr* descr) {
    int dtype = fix_numpy_type_num(descr->type_num);

    if (current_type_num_ == -1) {
      current_dtype_ = descr;
      current_type_num_ = dtype;
      return Status::OK();
    } else if (current_type_num_ == dtype) {
      return Status::OK();
    }

#define OBSERVE_CASE(DTYPE)                 \
  case NPY_##DTYPE:                         \
    action = Observe_##DTYPE(descr, dtype); \
    break;

    int action = OK;
    switch (current_type_num_) {
      OBSERVE_CASE(BOOL);
      OBSERVE_CASE(INT8);
      OBSERVE_CASE(INT16);
      OBSERVE_CASE(INT32);
      OBSERVE_CASE(INT64);
      OBSERVE_CASE(UINT8);
      OBSERVE_CASE(UINT16);
      OBSERVE_CASE(UINT32);
      OBSERVE_CASE(UINT64);
      OBSERVE_CASE(FLOAT16);
      OBSERVE_CASE(FLOAT32);
      OBSERVE_CASE(FLOAT64);
      case NPY_DATETIME:
        action = Observe_DATETIME(descr);
        break;
      default:
        return Status::NotImplemented("Unsupported numpy type ", GetNumPyTypeName(dtype));
    }

    if (action == INVALID) {
      return InvalidMix(dtype);
    }
    return Status::OK();
  }

  bool dtype_was_observed() const { return current_type_num_ != -1; }

  PyArray_Descr* current_dtype() const { return current_dtype_; }

  int current_type_num() const { return current_type_num_; }

 private:
  int current_type_num_;
  PyArray_Descr* current_dtype_;
};

class TypeInferrer {
  // A type inference visitor for Python values
 public:
  // \param validate_interval the number of elements to observe before checking
  // whether the data is mixed type or has other problems. This helps avoid
  // excess computation for each element while also making sure we "bail out"
  // early with long sequences that may have problems up front
  // \param make_unions permit mixed-type data by creating union types (not yet
  // implemented)
  explicit TypeInferrer(bool pandas_null_sentinels = false,
                        int64_t validate_interval = 100, bool make_unions = false)
      : pandas_null_sentinels_(pandas_null_sentinels),
        validate_interval_(validate_interval),
        make_unions_(make_unions),
        total_count_(0),
        none_count_(0),
        bool_count_(0),
        int_count_(0),
        date_count_(0),
        time_count_(0),
        timestamp_micro_count_(0),
        duration_count_(0),
        float_count_(0),
        binary_count_(0),
        unicode_count_(0),
        decimal_count_(0),
        list_count_(0),
        struct_count_(0),
        numpy_dtype_count_(0),
        interval_count_(0),
        max_decimal_metadata_(std::numeric_limits<int32_t>::min(),
                              std::numeric_limits<int32_t>::min()),
        decimal_type_() {
    ARROW_CHECK_OK(internal::ImportDecimalType(&decimal_type_));
    ARROW_CHECK_OK(ImportPresentIntervalTypes(&interval_types_));
  }

  /// \param[in] obj a Python object in the sequence
  /// \param[out] keep_going if sufficient information has been gathered to
  /// attempt to begin converting the sequence, *keep_going will be set to true
  /// to signal to the calling visitor loop to terminate
  Status Visit(PyObject* obj, bool* keep_going) {
    ++total_count_;

    if (obj == Py_None || (pandas_null_sentinels_ && internal::PandasObjectIsNull(obj))) {
      ++none_count_;
    } else if (PyBool_Check(obj)) {
      ++bool_count_;
      *keep_going = make_unions_;
    } else if (PyFloat_Check(obj)) {
      ++float_count_;
      *keep_going = make_unions_;
    } else if (internal::IsPyInteger(obj)) {
      ++int_count_;
    } else if (PyDateTime_Check(obj)) {
      // infer timezone from the first encountered datetime object
      if (!timestamp_micro_count_) {
        OwnedRef tzinfo(PyObject_GetAttrString(obj, "tzinfo"));
        if (tzinfo.obj() != nullptr && tzinfo.obj() != Py_None) {
          ARROW_ASSIGN_OR_RAISE(timezone_, internal::TzinfoToString(tzinfo.obj()));
        }
      }
      ++timestamp_micro_count_;
      *keep_going = make_unions_;
    } else if (PyDelta_Check(obj)) {
      ++duration_count_;
      *keep_going = make_unions_;
    } else if (PyDate_Check(obj)) {
      ++date_count_;
      *keep_going = make_unions_;
    } else if (PyTime_Check(obj)) {
      ++time_count_;
      *keep_going = make_unions_;
    } else if (internal::IsPyBinary(obj)) {
      ++binary_count_;
      *keep_going = make_unions_;
    } else if (PyUnicode_Check(obj)) {
      ++unicode_count_;
      *keep_going = make_unions_;
    } else if (PyArray_CheckAnyScalarExact(obj)) {
      RETURN_NOT_OK(VisitDType(PyArray_DescrFromScalar(obj), keep_going));
    } else if (PySet_Check(obj) || (Py_TYPE(obj) == &PyDictValues_Type)) {
      RETURN_NOT_OK(VisitSet(obj, keep_going));
    } else if (PyArray_Check(obj)) {
      RETURN_NOT_OK(VisitNdarray(obj, keep_going));
    } else if (PyDict_Check(obj)) {
      RETURN_NOT_OK(VisitDict(obj));
    } else if (PyList_Check(obj) ||
               (PyTuple_Check(obj) &&
                !PyObject_IsInstance(obj, PyTuple_GetItem(interval_types_.obj(), 0)))) {
      RETURN_NOT_OK(VisitList(obj, keep_going));
    } else if (PyObject_IsInstance(obj, decimal_type_.obj())) {
      RETURN_NOT_OK(max_decimal_metadata_.Update(obj));
      ++decimal_count_;
    } else if (PyObject_IsInstance(obj, interval_types_.obj())) {
      ++interval_count_;
    } else {
      return internal::InvalidValue(obj,
                                    "did not recognize Python value type when inferring "
                                    "an Arrow data type");
    }

    if (total_count_ % validate_interval_ == 0) {
      RETURN_NOT_OK(Validate());
    }

    return Status::OK();
  }

  // Infer value type from a sequence of values
  Status VisitSequence(PyObject* obj, PyObject* mask = nullptr) {
    if (mask == nullptr || mask == Py_None) {
      return internal::VisitSequence(
          obj, /*offset=*/0,
          [this](PyObject* value, bool* keep_going) { return Visit(value, keep_going); });
    } else {
      return internal::VisitSequenceMasked(
          obj, mask, /*offset=*/0,
          [this](PyObject* value, uint8_t masked, bool* keep_going) {
            if (!masked) {
              return Visit(value, keep_going);
            } else {
              return Status::OK();
            }
          });
    }
  }

  // Infer value type from a sequence of values
  Status VisitIterable(PyObject* obj) {
    return internal::VisitIterable(obj, [this](PyObject* value, bool* keep_going) {
      return Visit(value, keep_going);
    });
  }

  Status GetType(std::shared_ptr<DataType>* out) {
    // TODO(wesm): handling forming unions
    if (make_unions_) {
      return Status::NotImplemented("Creating union types not yet supported");
    }

    RETURN_NOT_OK(Validate());

    if (numpy_dtype_count_ > 0) {
      // All NumPy scalars and Nones/nulls
      if (numpy_dtype_count_ + none_count_ == total_count_) {
        std::shared_ptr<DataType> type;
        RETURN_NOT_OK(NumPyDtypeToArrow(numpy_unifier_.current_dtype(), &type));
        *out = type;
        return Status::OK();
      }

      // The "bad path": data contains a mix of NumPy scalars and
      // other kinds of scalars. Note this can happen innocuously
      // because numpy.nan is not a NumPy scalar (it's a built-in
      // PyFloat)

      // TODO(ARROW-5564): Merge together type unification so this
      // hack is not necessary
      switch (numpy_unifier_.current_type_num()) {
        case NPY_BOOL:
          bool_count_ += numpy_dtype_count_;
          break;
        case NPY_INT8:
        case NPY_INT16:
        case NPY_INT32:
        case NPY_INT64:
        case NPY_UINT8:
        case NPY_UINT16:
        case NPY_UINT32:
        case NPY_UINT64:
          int_count_ += numpy_dtype_count_;
          break;
        case NPY_FLOAT32:
        case NPY_FLOAT64:
          float_count_ += numpy_dtype_count_;
          break;
        case NPY_DATETIME:
          return Status::Invalid(
              "numpy.datetime64 scalars cannot be mixed "
              "with other Python scalar values currently");
      }
    }

    if (list_count_) {
      std::shared_ptr<DataType> value_type;
      RETURN_NOT_OK(list_inferrer_->GetType(&value_type));
      *out = list(value_type);
    } else if (struct_count_) {
      RETURN_NOT_OK(GetStructType(out));
    } else if (decimal_count_) {
      if (max_decimal_metadata_.precision() > Decimal128Type::kMaxPrecision) {
        // the default constructor does not validate the precision and scale
        ARROW_ASSIGN_OR_RAISE(*out,
                              Decimal256Type::Make(max_decimal_metadata_.precision(),
                                                   max_decimal_metadata_.scale()));
      } else {
        ARROW_ASSIGN_OR_RAISE(*out,
                              Decimal128Type::Make(max_decimal_metadata_.precision(),
                                                   max_decimal_metadata_.scale()));
      }
    } else if (float_count_) {
      // Prioritize floats before integers
      *out = float64();
    } else if (int_count_) {
      *out = int64();
    } else if (date_count_) {
      *out = date32();
    } else if (time_count_) {
      *out = time64(TimeUnit::MICRO);
    } else if (timestamp_micro_count_) {
      *out = timestamp(TimeUnit::MICRO, timezone_);
    } else if (duration_count_) {
      *out = duration(TimeUnit::MICRO);
    } else if (bool_count_) {
      *out = boolean();
    } else if (binary_count_) {
      *out = binary();
    } else if (unicode_count_) {
      *out = utf8();
    } else if (interval_count_) {
      *out = month_day_nano_interval();
    } else {
      *out = null();
    }
    return Status::OK();
  }

  int64_t total_count() const { return total_count_; }

 protected:
  Status Validate() const {
    if (list_count_ > 0) {
      if (list_count_ + none_count_ != total_count_) {
        return Status::Invalid("cannot mix list and non-list, non-null values");
      }
      RETURN_NOT_OK(list_inferrer_->Validate());
    } else if (struct_count_ > 0) {
      if (struct_count_ + none_count_ != total_count_) {
        return Status::Invalid("cannot mix struct and non-struct, non-null values");
      }
      for (const auto& it : struct_inferrers_) {
        RETURN_NOT_OK(it.second.Validate());
      }
    }
    return Status::OK();
  }

  Status VisitDType(PyArray_Descr* dtype, bool* keep_going) {
    // Continue visiting dtypes for now.
    // TODO(wesm): devise approach for unions
    ++numpy_dtype_count_;
    *keep_going = true;
    return numpy_unifier_.Observe(dtype);
  }

  Status VisitList(PyObject* obj, bool* keep_going /* unused */) {
    if (!list_inferrer_) {
      list_inferrer_.reset(
          new TypeInferrer(pandas_null_sentinels_, validate_interval_, make_unions_));
    }
    ++list_count_;
    return list_inferrer_->VisitSequence(obj);
  }

  Status VisitSet(PyObject* obj, bool* keep_going /* unused */) {
    if (!list_inferrer_) {
      list_inferrer_.reset(
          new TypeInferrer(pandas_null_sentinels_, validate_interval_, make_unions_));
    }
    ++list_count_;
    return list_inferrer_->VisitIterable(obj);
  }

  Status VisitNdarray(PyObject* obj, bool* keep_going) {
    PyArray_Descr* dtype = PyArray_DESCR(reinterpret_cast<PyArrayObject*>(obj));
    if (dtype->type_num == NPY_OBJECT) {
      return VisitList(obj, keep_going);
    }
    // Not an object array: infer child Arrow type from dtype
    if (!list_inferrer_) {
      list_inferrer_.reset(
          new TypeInferrer(pandas_null_sentinels_, validate_interval_, make_unions_));
    }
    ++list_count_;

    // XXX(wesm): In ARROW-4324 I added accounting to check whether
    // all of the non-null values have NumPy dtypes, but the
    // total_count not not being properly incremented here
    ++(*list_inferrer_).total_count_;
    return list_inferrer_->VisitDType(dtype, keep_going);
  }

  Status VisitDict(PyObject* obj) {
    PyObject* key_obj;
    PyObject* value_obj;
    Py_ssize_t pos = 0;

    while (PyDict_Next(obj, &pos, &key_obj, &value_obj)) {
      std::string key;
      if (PyUnicode_Check(key_obj)) {
        RETURN_NOT_OK(internal::PyUnicode_AsStdString(key_obj, &key));
      } else if (PyBytes_Check(key_obj)) {
        key = internal::PyBytes_AsStdString(key_obj);
      } else {
        return Status::TypeError("Expected dict key of type str or bytes, got '",
                                 Py_TYPE(key_obj)->tp_name, "'");
      }
      // Get or create visitor for this key
      auto it = struct_inferrers_.find(key);
      if (it == struct_inferrers_.end()) {
        it = struct_inferrers_
                 .insert(
                     std::make_pair(key, TypeInferrer(pandas_null_sentinels_,
                                                      validate_interval_, make_unions_)))
                 .first;
      }
      TypeInferrer* visitor = &it->second;

      // We ignore termination signals from child visitors for now
      //
      // TODO(wesm): keep track of whether type inference has terminated for
      // the child visitors to avoid doing unneeded work
      bool keep_going = true;
      RETURN_NOT_OK(visitor->Visit(value_obj, &keep_going));
    }

    // We do not terminate visiting dicts since we want the union of all
    // observed keys
    ++struct_count_;
    return Status::OK();
  }

  Status GetStructType(std::shared_ptr<DataType>* out) {
    std::vector<std::shared_ptr<Field>> fields;
    for (auto&& it : struct_inferrers_) {
      std::shared_ptr<DataType> field_type;
      RETURN_NOT_OK(it.second.GetType(&field_type));
      fields.emplace_back(field(it.first, field_type));
    }
    *out = struct_(fields);
    return Status::OK();
  }

 private:
  bool pandas_null_sentinels_;
  int64_t validate_interval_;
  bool make_unions_;
  int64_t total_count_;
  int64_t none_count_;
  int64_t bool_count_;
  int64_t int_count_;
  int64_t date_count_;
  int64_t time_count_;
  int64_t timestamp_micro_count_;
  std::string timezone_;
  int64_t duration_count_;
  int64_t float_count_;
  int64_t binary_count_;
  int64_t unicode_count_;
  int64_t decimal_count_;
  int64_t list_count_;
  int64_t struct_count_;
  int64_t numpy_dtype_count_;
  int64_t interval_count_;
  std::unique_ptr<TypeInferrer> list_inferrer_;
  std::map<std::string, TypeInferrer> struct_inferrers_;

  // If we observe a strongly-typed value in e.g. a NumPy array, we can store
  // it here to skip the type counting logic above
  NumPyDtypeUnifier numpy_unifier_;

  internal::DecimalMetadata max_decimal_metadata_;

  OwnedRefNoGIL decimal_type_;
  OwnedRefNoGIL interval_types_;
};

// Non-exhaustive type inference
Result<std::shared_ptr<DataType>> InferArrowType(PyObject* obj, PyObject* mask,
                                                 bool pandas_null_sentinels) {
  if (pandas_null_sentinels) {
    // ARROW-842: If pandas is not installed then null checks will be less
    // comprehensive, but that is okay.
    internal::InitPandasStaticData();
  }

  std::shared_ptr<DataType> out_type;
  TypeInferrer inferrer(pandas_null_sentinels);
  RETURN_NOT_OK(inferrer.VisitSequence(obj, mask));
  RETURN_NOT_OK(inferrer.GetType(&out_type));
  if (out_type == nullptr) {
    return Status::TypeError("Unable to determine data type");
  } else {
    return std::move(out_type);
  }
}

ARROW_PYTHON_EXPORT
bool IsPyBool(PyObject* obj) { return internal::PyBoolScalar_Check(obj); }

ARROW_PYTHON_EXPORT
bool IsPyInt(PyObject* obj) { return internal::PyIntScalar_Check(obj); }

ARROW_PYTHON_EXPORT
bool IsPyFloat(PyObject* obj) { return internal::PyFloatScalar_Check(obj); }

}  // namespace py
}  // namespace arrow

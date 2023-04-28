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

// Functions for pandas conversion via NumPy

#include "arrow/python/arrow_to_pandas.h"
#include "arrow/python/numpy_interop.h"  // IWYU pragma: expand

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "arrow/array.h"
#include "arrow/buffer.h"
#include "arrow/datum.h"
#include "arrow/status.h"
#include "arrow/table.h"
#include "arrow/type.h"
#include "arrow/type_traits.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/hashing.h"
#include "arrow/util/int_util.h"
#include "arrow/util/logging.h"
#include "arrow/util/macros.h"
#include "arrow/util/parallel.h"
#include "arrow/visit_type_inline.h"

#include "arrow/compute/api.h"

#include "arrow/python/arrow_to_python_internal.h"
#include "arrow/python/common.h"
#include "arrow/python/datetime.h"
#include "arrow/python/decimal.h"
#include "arrow/python/helpers.h"
#include "arrow/python/numpy_convert.h"
#include "arrow/python/numpy_internal.h"
#include "arrow/python/pyarrow.h"
#include "arrow/python/python_to_arrow.h"
#include "arrow/python/type_traits.h"

namespace arrow {

class MemoryPool;

using internal::checked_cast;
using internal::CheckIndexBounds;
using internal::OptionalParallelFor;

namespace py {
namespace {

// Fix options for conversion of an inner (child) array.
PandasOptions MakeInnerOptions(PandasOptions options) {
  // Make sure conversion of inner dictionary arrays always returns an array,
  // not a dict {'indices': array, 'dictionary': array, 'ordered': bool}
  options.decode_dictionaries = true;
  options.categorical_columns.clear();
  options.strings_to_categorical = false;

  // In ARROW-7723, we found as a result of ARROW-3789 that second
  // through microsecond resolution tz-aware timestamps were being promoted to
  // use the DATETIME_NANO_TZ conversion path, yielding a datetime64[ns] NumPy
  // array in this function. PyArray_GETITEM returns datetime.datetime for
  // units second through microsecond but PyLong for nanosecond (because
  // datetime.datetime does not support nanoseconds).
  // We force the object conversion to preserve the value of the timezone.
  // Nanoseconds are returned as integers.
  options.coerce_temporal_nanoseconds = false;

  return options;
}

// ----------------------------------------------------------------------
// PyCapsule code for setting ndarray base to reference C++ object

struct ArrayCapsule {
  std::shared_ptr<Array> array;
};

struct BufferCapsule {
  std::shared_ptr<Buffer> buffer;
};

void ArrayCapsule_Destructor(PyObject* capsule) {
  delete reinterpret_cast<ArrayCapsule*>(PyCapsule_GetPointer(capsule, "arrow::Array"));
}

void BufferCapsule_Destructor(PyObject* capsule) {
  delete reinterpret_cast<BufferCapsule*>(PyCapsule_GetPointer(capsule, "arrow::Buffer"));
}

// ----------------------------------------------------------------------
// pandas 0.x DataFrame conversion internals

using internal::arrow_traits;
using internal::npy_traits;

template <typename T>
struct WrapBytes {};

template <>
struct WrapBytes<StringType> {
  static inline PyObject* Wrap(const char* data, int64_t length) {
    return PyUnicode_FromStringAndSize(data, length);
  }
};

template <>
struct WrapBytes<LargeStringType> {
  static inline PyObject* Wrap(const char* data, int64_t length) {
    return PyUnicode_FromStringAndSize(data, length);
  }
};

template <>
struct WrapBytes<BinaryType> {
  static inline PyObject* Wrap(const char* data, int64_t length) {
    return PyBytes_FromStringAndSize(data, length);
  }
};

template <>
struct WrapBytes<LargeBinaryType> {
  static inline PyObject* Wrap(const char* data, int64_t length) {
    return PyBytes_FromStringAndSize(data, length);
  }
};

template <>
struct WrapBytes<FixedSizeBinaryType> {
  static inline PyObject* Wrap(const char* data, int64_t length) {
    return PyBytes_FromStringAndSize(data, length);
  }
};

static inline bool ListTypeSupported(const DataType& type) {
  switch (type.id()) {
    case Type::BOOL:
    case Type::UINT8:
    case Type::INT8:
    case Type::UINT16:
    case Type::INT16:
    case Type::UINT32:
    case Type::INT32:
    case Type::INT64:
    case Type::UINT64:
    case Type::FLOAT:
    case Type::DOUBLE:
    case Type::DECIMAL128:
    case Type::DECIMAL256:
    case Type::BINARY:
    case Type::LARGE_BINARY:
    case Type::STRING:
    case Type::LARGE_STRING:
    case Type::DATE32:
    case Type::DATE64:
    case Type::STRUCT:
    case Type::TIME32:
    case Type::TIME64:
    case Type::TIMESTAMP:
    case Type::DURATION:
    case Type::DICTIONARY:
    case Type::INTERVAL_MONTH_DAY_NANO:
    case Type::NA:  // empty list
      // The above types are all supported.
      return true;
    case Type::FIXED_SIZE_LIST:
    case Type::LIST:
    case Type::LARGE_LIST: {
      const auto& list_type = checked_cast<const BaseListType&>(type);
      return ListTypeSupported(*list_type.value_type());
    }
    case Type::EXTENSION: {
      const auto& ext = checked_cast<const ExtensionType&>(*type.GetSharedPtr());
      return ListTypeSupported(*(ext.storage_type()));
    }
    default:
      break;
  }
  return false;
}

Status CapsulizeArray(const std::shared_ptr<Array>& arr, PyObject** out) {
  auto capsule = new ArrayCapsule{{arr}};
  *out = PyCapsule_New(reinterpret_cast<void*>(capsule), "arrow::Array",
                       &ArrayCapsule_Destructor);
  if (*out == nullptr) {
    delete capsule;
    RETURN_IF_PYERROR();
  }
  return Status::OK();
}

Status CapsulizeBuffer(const std::shared_ptr<Buffer>& buffer, PyObject** out) {
  auto capsule = new BufferCapsule{{buffer}};
  *out = PyCapsule_New(reinterpret_cast<void*>(capsule), "arrow::Buffer",
                       &BufferCapsule_Destructor);
  if (*out == nullptr) {
    delete capsule;
    RETURN_IF_PYERROR();
  }
  return Status::OK();
}

Status SetNdarrayBase(PyArrayObject* arr, PyObject* base) {
  if (PyArray_SetBaseObject(arr, base) == -1) {
    // Error occurred, trust that SetBaseObject sets the error state
    Py_XDECREF(base);
    RETURN_IF_PYERROR();
  }
  return Status::OK();
}

Status SetBufferBase(PyArrayObject* arr, const std::shared_ptr<Buffer>& buffer) {
  PyObject* base;
  RETURN_NOT_OK(CapsulizeBuffer(buffer, &base));
  return SetNdarrayBase(arr, base);
}

inline void set_numpy_metadata(int type, const DataType* datatype, PyArray_Descr* out) {
  auto metadata = reinterpret_cast<PyArray_DatetimeDTypeMetaData*>(out->c_metadata);
  if (type == NPY_DATETIME) {
    if (datatype->id() == Type::TIMESTAMP) {
      const auto& timestamp_type = checked_cast<const TimestampType&>(*datatype);
      metadata->meta.base = internal::NumPyFrequency(timestamp_type.unit());
    } else {
      DCHECK(false) << "NPY_DATETIME views only supported for Arrow TIMESTAMP types";
    }
  } else if (type == NPY_TIMEDELTA) {
    DCHECK_EQ(datatype->id(), Type::DURATION);
    const auto& duration_type = checked_cast<const DurationType&>(*datatype);
    metadata->meta.base = internal::NumPyFrequency(duration_type.unit());
  }
}

Status PyArray_NewFromPool(int nd, npy_intp* dims, PyArray_Descr* descr, MemoryPool* pool,
                           PyObject** out) {
  // ARROW-6570: Allocate memory from MemoryPool for a couple reasons
  //
  // * Track allocations
  // * Get better performance through custom allocators
  int64_t total_size = descr->elsize;
  for (int i = 0; i < nd; ++i) {
    total_size *= dims[i];
  }

  ARROW_ASSIGN_OR_RAISE(auto buffer, AllocateBuffer(total_size, pool));
  *out = PyArray_NewFromDescr(&PyArray_Type, descr, nd, dims,
                              /*strides=*/nullptr,
                              /*data=*/buffer->mutable_data(),
                              /*flags=*/NPY_ARRAY_CARRAY | NPY_ARRAY_WRITEABLE,
                              /*obj=*/nullptr);
  if (*out == nullptr) {
    RETURN_IF_PYERROR();
    // Trust that error set if NULL returned
  }
  return SetBufferBase(reinterpret_cast<PyArrayObject*>(*out), std::move(buffer));
}

template <typename T = void>
inline const T* GetPrimitiveValues(const Array& arr) {
  if (arr.length() == 0) {
    return nullptr;
  }
  const int elsize = arr.type()->byte_width();
  const auto& prim_arr = checked_cast<const PrimitiveArray&>(arr);
  return reinterpret_cast<const T*>(prim_arr.values()->data() + arr.offset() * elsize);
}

Status MakeNumPyView(std::shared_ptr<Array> arr, PyObject* py_ref, int npy_type, int ndim,
                     npy_intp* dims, PyObject** out) {
  PyAcquireGIL lock;

  PyArray_Descr* descr = internal::GetSafeNumPyDtype(npy_type);
  set_numpy_metadata(npy_type, arr->type().get(), descr);
  PyObject* result = PyArray_NewFromDescr(
      &PyArray_Type, descr, ndim, dims, /*strides=*/nullptr,
      const_cast<void*>(GetPrimitiveValues(*arr)), /*flags=*/0, nullptr);
  PyArrayObject* np_arr = reinterpret_cast<PyArrayObject*>(result);
  if (np_arr == nullptr) {
    // Error occurred, trust that error set
    return Status::OK();
  }

  PyObject* base;
  if (py_ref == nullptr) {
    // Capsule will be owned by the ndarray, no incref necessary. See
    // ARROW-1973
    RETURN_NOT_OK(CapsulizeArray(arr, &base));
  } else {
    Py_INCREF(py_ref);
    base = py_ref;
  }
  RETURN_NOT_OK(SetNdarrayBase(np_arr, base));

  // Do not allow Arrow data to be mutated
  PyArray_CLEARFLAGS(np_arr, NPY_ARRAY_WRITEABLE);
  *out = result;
  return Status::OK();
}

class PandasWriter {
 public:
  enum type {
    OBJECT,
    UINT8,
    INT8,
    UINT16,
    INT16,
    UINT32,
    INT32,
    UINT64,
    INT64,
    HALF_FLOAT,
    FLOAT,
    DOUBLE,
    BOOL,
    DATETIME_DAY,
    DATETIME_SECOND,
    DATETIME_MILLI,
    DATETIME_MICRO,
    DATETIME_NANO,
    DATETIME_NANO_TZ,
    TIMEDELTA_SECOND,
    TIMEDELTA_MILLI,
    TIMEDELTA_MICRO,
    TIMEDELTA_NANO,
    CATEGORICAL,
    EXTENSION
  };

  PandasWriter(const PandasOptions& options, int64_t num_rows, int num_columns)
      : options_(options), num_rows_(num_rows), num_columns_(num_columns) {
    PyAcquireGIL lock;
    internal::InitPandasStaticData();
  }
  virtual ~PandasWriter() {}

  void SetBlockData(PyObject* arr) {
    block_arr_.reset(arr);
    block_data_ =
        reinterpret_cast<uint8_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(arr)));
  }

  /// \brief Either copy or wrap single array to create pandas-compatible array
  /// for Series or DataFrame. num_columns_ can only be 1. Will try to zero
  /// copy if possible (or error if not possible and zero_copy_only=True)
  virtual Status TransferSingle(std::shared_ptr<ChunkedArray> data, PyObject* py_ref) = 0;

  /// \brief Copy ChunkedArray into a multi-column block
  virtual Status CopyInto(std::shared_ptr<ChunkedArray> data, int64_t rel_placement) = 0;

  Status EnsurePlacementAllocated() {
    std::lock_guard<std::mutex> guard(allocation_lock_);
    if (placement_data_ != nullptr) {
      return Status::OK();
    }
    PyAcquireGIL lock;
    npy_intp placement_dims[1] = {num_columns_};
    PyObject* placement_arr = PyArray_SimpleNew(1, placement_dims, NPY_INT64);
    RETURN_IF_PYERROR();
    placement_arr_.reset(placement_arr);
    placement_data_ = reinterpret_cast<int64_t*>(
        PyArray_DATA(reinterpret_cast<PyArrayObject*>(placement_arr)));
    return Status::OK();
  }

  Status EnsureAllocated() {
    std::lock_guard<std::mutex> guard(allocation_lock_);
    if (block_data_ != nullptr) {
      return Status::OK();
    }
    RETURN_NOT_OK(Allocate());
    return Status::OK();
  }

  virtual bool CanZeroCopy(const ChunkedArray& data) const { return false; }

  virtual Status Write(std::shared_ptr<ChunkedArray> data, int64_t abs_placement,
                       int64_t rel_placement) {
    RETURN_NOT_OK(EnsurePlacementAllocated());
    if (num_columns_ == 1 && options_.allow_zero_copy_blocks) {
      RETURN_NOT_OK(TransferSingle(data, /*py_ref=*/nullptr));
    } else {
      RETURN_NOT_OK(
          CheckNoZeroCopy("Cannot do zero copy conversion into "
                          "multi-column DataFrame block"));
      RETURN_NOT_OK(EnsureAllocated());
      RETURN_NOT_OK(CopyInto(data, rel_placement));
    }
    placement_data_[rel_placement] = abs_placement;
    return Status::OK();
  }

  virtual Status GetDataFrameResult(PyObject** out) {
    PyObject* result = PyDict_New();
    RETURN_IF_PYERROR();

    PyObject* block;
    RETURN_NOT_OK(GetResultBlock(&block));

    PyDict_SetItemString(result, "block", block);
    PyDict_SetItemString(result, "placement", placement_arr_.obj());

    RETURN_NOT_OK(AddResultMetadata(result));
    *out = result;
    return Status::OK();
  }

  // Caller steals the reference to this object
  virtual Status GetSeriesResult(PyObject** out) {
    RETURN_NOT_OK(MakeBlock1D());
    // Caller owns the object now
    *out = block_arr_.detach();
    return Status::OK();
  }

 protected:
  virtual Status AddResultMetadata(PyObject* result) { return Status::OK(); }

  Status MakeBlock1D() {
    // For Series or for certain DataFrame block types, we need to shape to a
    // 1D array when there is only one column
    PyAcquireGIL lock;

    DCHECK_EQ(1, num_columns_);

    npy_intp new_dims[1] = {static_cast<npy_intp>(num_rows_)};
    PyArray_Dims dims;
    dims.ptr = new_dims;
    dims.len = 1;

    PyObject* reshaped = PyArray_Newshape(
        reinterpret_cast<PyArrayObject*>(block_arr_.obj()), &dims, NPY_ANYORDER);
    RETURN_IF_PYERROR();

    // ARROW-8801: Here a PyArrayObject is created that is not being managed by
    // any OwnedRef object. This object is then put in the resulting object
    // with PyDict_SetItemString, which increments the reference count, so a
    // memory leak ensues. There are several ways to fix the memory leak but a
    // simple one is to put the reshaped 1D block array in this OwnedRefNoGIL
    // so it will be correctly decref'd when this class is destructed.
    block_arr_.reset(reshaped);
    return Status::OK();
  }

  virtual Status GetResultBlock(PyObject** out) {
    *out = block_arr_.obj();
    return Status::OK();
  }

  Status CheckNoZeroCopy(const std::string& message) {
    if (options_.zero_copy_only) {
      return Status::Invalid(message);
    }
    return Status::OK();
  }

  Status CheckNotZeroCopyOnly(const ChunkedArray& data) {
    if (options_.zero_copy_only) {
      return Status::Invalid("Needed to copy ", data.num_chunks(), " chunks with ",
                             data.null_count(), " nulls, but zero_copy_only was True");
    }
    return Status::OK();
  }

  virtual Status Allocate() {
    return Status::NotImplemented("Override Allocate in subclasses");
  }

  Status AllocateNDArray(int npy_type, int ndim = 2) {
    PyAcquireGIL lock;

    PyObject* block_arr = nullptr;
    npy_intp block_dims[2] = {0, 0};

    if (ndim == 2) {
      block_dims[0] = num_columns_;
      block_dims[1] = num_rows_;
    } else {
      block_dims[0] = num_rows_;
    }
    PyArray_Descr* descr = internal::GetSafeNumPyDtype(npy_type);
    if (PyDataType_REFCHK(descr)) {
      // ARROW-6876: if the array has refcounted items, let Numpy
      // own the array memory so as to decref elements on array destruction
      block_arr = PyArray_SimpleNewFromDescr(ndim, block_dims, descr);
      RETURN_IF_PYERROR();
    } else {
      RETURN_NOT_OK(
          PyArray_NewFromPool(ndim, block_dims, descr, options_.pool, &block_arr));
    }

    SetBlockData(block_arr);
    return Status::OK();
  }

  void SetDatetimeUnit(NPY_DATETIMEUNIT unit) {
    PyAcquireGIL lock;
    auto date_dtype = reinterpret_cast<PyArray_DatetimeDTypeMetaData*>(
        PyArray_DESCR(reinterpret_cast<PyArrayObject*>(block_arr_.obj()))->c_metadata);
    date_dtype->meta.base = unit;
  }

  PandasOptions options_;

  std::mutex allocation_lock_;

  int64_t num_rows_;
  int num_columns_;

  OwnedRefNoGIL block_arr_;
  uint8_t* block_data_ = nullptr;

  // ndarray<int32>
  OwnedRefNoGIL placement_arr_;
  int64_t* placement_data_ = nullptr;

 private:
  ARROW_DISALLOW_COPY_AND_ASSIGN(PandasWriter);
};

template <typename InType, typename OutType>
inline void ConvertIntegerWithNulls(const PandasOptions& options,
                                    const ChunkedArray& data, OutType* out_values) {
  for (int c = 0; c < data.num_chunks(); c++) {
    const auto& arr = *data.chunk(c);
    const InType* in_values = GetPrimitiveValues<InType>(arr);
    // Upcast to double, set NaN as appropriate

    for (int i = 0; i < arr.length(); ++i) {
      *out_values++ =
          arr.IsNull(i) ? static_cast<OutType>(NAN) : static_cast<OutType>(in_values[i]);
    }
  }
}

template <typename T>
inline void ConvertIntegerNoNullsSameType(const PandasOptions& options,
                                          const ChunkedArray& data, T* out_values) {
  for (int c = 0; c < data.num_chunks(); c++) {
    const auto& arr = *data.chunk(c);
    if (arr.length() > 0) {
      const T* in_values = GetPrimitiveValues<T>(arr);
      memcpy(out_values, in_values, sizeof(T) * arr.length());
      out_values += arr.length();
    }
  }
}

template <typename InType, typename OutType>
inline void ConvertIntegerNoNullsCast(const PandasOptions& options,
                                      const ChunkedArray& data, OutType* out_values) {
  for (int c = 0; c < data.num_chunks(); c++) {
    const auto& arr = *data.chunk(c);
    const InType* in_values = GetPrimitiveValues<InType>(arr);
    for (int64_t i = 0; i < arr.length(); ++i) {
      *out_values = in_values[i];
    }
  }
}

template <typename T, typename Enable = void>
struct MemoizationTraits {
  using Scalar = typename T::c_type;
};

template <typename T>
struct MemoizationTraits<T, enable_if_has_string_view<T>> {
  // For binary, we memoize string_view as a scalar value to avoid having to
  // unnecessarily copy the memory into the memo table data structure
  using Scalar = std::string_view;
};

// Generic Array -> PyObject** converter that handles object deduplication, if
// requested
template <typename Type, typename WrapFunction>
inline Status ConvertAsPyObjects(const PandasOptions& options, const ChunkedArray& data,
                                 WrapFunction&& wrap_func, PyObject** out_values) {
  using ArrayType = typename TypeTraits<Type>::ArrayType;
  using Scalar = typename MemoizationTraits<Type>::Scalar;

  ::arrow::internal::ScalarMemoTable<Scalar> memo_table(options.pool);
  std::vector<PyObject*> unique_values;
  int32_t memo_size = 0;

  auto WrapMemoized = [&](const Scalar& value, PyObject** out_values) {
    int32_t memo_index;
    RETURN_NOT_OK(memo_table.GetOrInsert(value, &memo_index));
    if (memo_index == memo_size) {
      // New entry
      RETURN_NOT_OK(wrap_func(value, out_values));
      unique_values.push_back(*out_values);
      ++memo_size;
    } else {
      // Duplicate entry
      Py_INCREF(unique_values[memo_index]);
      *out_values = unique_values[memo_index];
    }
    return Status::OK();
  };

  auto WrapUnmemoized = [&](const Scalar& value, PyObject** out_values) {
    return wrap_func(value, out_values);
  };

  for (int c = 0; c < data.num_chunks(); c++) {
    const auto& arr = arrow::internal::checked_cast<const ArrayType&>(*data.chunk(c));
    if (options.deduplicate_objects) {
      RETURN_NOT_OK(internal::WriteArrayObjects(arr, WrapMemoized, out_values));
    } else {
      RETURN_NOT_OK(internal::WriteArrayObjects(arr, WrapUnmemoized, out_values));
    }
    out_values += arr.length();
  }
  return Status::OK();
}

Status ConvertStruct(PandasOptions options, const ChunkedArray& data,
                     PyObject** out_values) {
  if (data.num_chunks() == 0) {
    return Status::OK();
  }
  // ChunkedArray has at least one chunk
  auto arr = checked_cast<const StructArray*>(data.chunk(0).get());
  // Use it to cache the struct type and number of fields for all chunks
  int32_t num_fields = arr->num_fields();
  auto array_type = arr->type();
  std::vector<OwnedRef> fields_data(num_fields * data.num_chunks());
  OwnedRef dict_item;

  // See notes in MakeInnerOptions.
  options = MakeInnerOptions(std::move(options));
  // Don't blindly convert because timestamps in lists are handled differently.
  options.timestamp_as_object = true;

  for (int c = 0; c < data.num_chunks(); c++) {
    auto fields_data_offset = c * num_fields;
    auto arr = checked_cast<const StructArray*>(data.chunk(c).get());
    // Convert the struct arrays first
    for (int32_t i = 0; i < num_fields; i++) {
      auto field = arr->field(static_cast<int>(i));
      // In case the field is an extension array, use .storage() to convert to Pandas
      if (field->type()->id() == Type::EXTENSION) {
        const ExtensionArray& arr_ext = checked_cast<const ExtensionArray&>(*field);
        field = arr_ext.storage();
      }
      RETURN_NOT_OK(ConvertArrayToPandas(options, field, nullptr,
                                         fields_data[i + fields_data_offset].ref()));
      DCHECK(PyArray_Check(fields_data[i + fields_data_offset].obj()));
    }

    // Construct a dictionary for each row
    const bool has_nulls = data.null_count() > 0;
    for (int64_t i = 0; i < arr->length(); ++i) {
      if (has_nulls && arr->IsNull(i)) {
        Py_INCREF(Py_None);
        *out_values = Py_None;
      } else {
        // Build the new dict object for the row
        dict_item.reset(PyDict_New());
        RETURN_IF_PYERROR();
        for (int32_t field_idx = 0; field_idx < num_fields; ++field_idx) {
          OwnedRef field_value;
          auto name = array_type->field(static_cast<int>(field_idx))->name();
          if (!arr->field(static_cast<int>(field_idx))->IsNull(i)) {
            // Value exists in child array, obtain it
            auto array = reinterpret_cast<PyArrayObject*>(
                fields_data[field_idx + fields_data_offset].obj());
            auto ptr = reinterpret_cast<const char*>(PyArray_GETPTR1(array, i));
            field_value.reset(PyArray_GETITEM(array, ptr));
            RETURN_IF_PYERROR();
          } else {
            // Translate the Null to a None
            Py_INCREF(Py_None);
            field_value.reset(Py_None);
          }
          // PyDict_SetItemString increments reference count
          auto setitem_result =
              PyDict_SetItemString(dict_item.obj(), name.c_str(), field_value.obj());
          RETURN_IF_PYERROR();
          DCHECK_EQ(setitem_result, 0);
        }
        *out_values = dict_item.obj();
        // Grant ownership to the resulting array
        Py_INCREF(*out_values);
      }
      ++out_values;
    }
  }
  return Status::OK();
}

Status DecodeDictionaries(MemoryPool* pool, const std::shared_ptr<DataType>& dense_type,
                          ArrayVector* arrays) {
  compute::ExecContext ctx(pool);
  compute::CastOptions options;
  for (size_t i = 0; i < arrays->size(); ++i) {
    ARROW_ASSIGN_OR_RAISE((*arrays)[i],
                          compute::Cast(*(*arrays)[i], dense_type, options, &ctx));
  }
  return Status::OK();
}

Status DecodeDictionaries(MemoryPool* pool, const std::shared_ptr<DataType>& dense_type,
                          std::shared_ptr<ChunkedArray>* array) {
  auto chunks = (*array)->chunks();
  RETURN_NOT_OK(DecodeDictionaries(pool, dense_type, &chunks));
  *array = std::make_shared<ChunkedArray>(std::move(chunks), dense_type);
  return Status::OK();
}

template <typename ListArrayT>
Status ConvertListsLike(PandasOptions options, const ChunkedArray& data,
                        PyObject** out_values) {
  // Get column of underlying value arrays
  ArrayVector value_arrays;
  for (int c = 0; c < data.num_chunks(); c++) {
    const auto& arr = checked_cast<const ListArrayT&>(*data.chunk(c));
    // values() does not account for offsets, so we need to slice into it.
    // We can't use Flatten(), because it removes the values behind a null list
    // value, and that makes the offsets into original list values and our
    // flattened_values array different.
    std::shared_ptr<Array> flattened_values = arr.values()->Slice(
        arr.value_offset(0), arr.value_offset(arr.length()) - arr.value_offset(0));
    if (arr.value_type()->id() == Type::EXTENSION) {
      const auto& arr_ext = checked_cast<const ExtensionArray&>(*flattened_values);
      value_arrays.emplace_back(arr_ext.storage());
    } else {
      value_arrays.emplace_back(flattened_values);
    }
  }

  using ListArrayType = typename ListArrayT::TypeClass;
  const auto& list_type = checked_cast<const ListArrayType&>(*data.type());
  auto value_type = list_type.value_type();
  if (value_type->id() == Type::EXTENSION) {
    value_type = checked_cast<const ExtensionType&>(*value_type).storage_type();
  }

  auto flat_column = std::make_shared<ChunkedArray>(value_arrays, value_type);

  options = MakeInnerOptions(std::move(options));

  OwnedRefNoGIL owned_numpy_array;
  RETURN_NOT_OK(ConvertChunkedArrayToPandas(options, flat_column, nullptr,
                                            owned_numpy_array.ref()));
  PyObject* numpy_array = owned_numpy_array.obj();
  DCHECK(PyArray_Check(numpy_array));

  int64_t chunk_offset = 0;
  for (int c = 0; c < data.num_chunks(); c++) {
    const auto& arr = checked_cast<const ListArrayT&>(*data.chunk(c));
    const bool has_nulls = data.null_count() > 0;
    for (int64_t i = 0; i < arr.length(); ++i) {
      if (has_nulls && arr.IsNull(i)) {
        Py_INCREF(Py_None);
        *out_values = Py_None;
      } else {
        // Need to subtract value_offset(0) since the original chunk might be a slice
        // into another array.
        OwnedRef start(PyLong_FromLongLong(arr.value_offset(i) + chunk_offset -
                                           arr.value_offset(0)));
        OwnedRef end(PyLong_FromLongLong(arr.value_offset(i + 1) + chunk_offset -
                                         arr.value_offset(0)));
        OwnedRef slice(PySlice_New(start.obj(), end.obj(), nullptr));

        if (ARROW_PREDICT_FALSE(slice.obj() == nullptr)) {
          // Fall out of loop, will return from RETURN_IF_PYERROR
          break;
        }
        *out_values = PyObject_GetItem(numpy_array, slice.obj());

        if (*out_values == nullptr) {
          // Fall out of loop, will return from RETURN_IF_PYERROR
          break;
        }
      }
      ++out_values;
    }
    RETURN_IF_PYERROR();

    chunk_offset += arr.value_offset(arr.length()) - arr.value_offset(0);
  }

  return Status::OK();
}

Status ConvertMap(PandasOptions options, const ChunkedArray& data,
                  PyObject** out_values) {
  // Get columns of underlying key/item arrays
  std::vector<std::shared_ptr<Array>> key_arrays;
  std::vector<std::shared_ptr<Array>> item_arrays;
  for (int c = 0; c < data.num_chunks(); ++c) {
    const auto& map_arr = checked_cast<const MapArray&>(*data.chunk(c));
    key_arrays.emplace_back(map_arr.keys());
    item_arrays.emplace_back(map_arr.items());
  }

  const auto& map_type = checked_cast<const MapType&>(*data.type());
  auto key_type = map_type.key_type();
  auto item_type = map_type.item_type();

  // ARROW-6899: Convert dictionary-encoded children to dense instead of
  // failing below. A more efficient conversion than this could be done later
  if (key_type->id() == Type::DICTIONARY) {
    auto dense_type = checked_cast<const DictionaryType&>(*key_type).value_type();
    RETURN_NOT_OK(DecodeDictionaries(options.pool, dense_type, &key_arrays));
    key_type = dense_type;
  }
  if (item_type->id() == Type::DICTIONARY) {
    auto dense_type = checked_cast<const DictionaryType&>(*item_type).value_type();
    RETURN_NOT_OK(DecodeDictionaries(options.pool, dense_type, &item_arrays));
    item_type = dense_type;
  }

  // See notes in MakeInnerOptions.
  options = MakeInnerOptions(std::move(options));
  // Don't blindly convert because timestamps in lists are handled differently.
  options.timestamp_as_object = true;

  auto flat_keys = std::make_shared<ChunkedArray>(key_arrays, key_type);
  auto flat_items = std::make_shared<ChunkedArray>(item_arrays, item_type);
  OwnedRef list_item;
  OwnedRef key_value;
  OwnedRef item_value;
  OwnedRefNoGIL owned_numpy_keys;
  RETURN_NOT_OK(
      ConvertChunkedArrayToPandas(options, flat_keys, nullptr, owned_numpy_keys.ref()));
  OwnedRefNoGIL owned_numpy_items;
  RETURN_NOT_OK(
      ConvertChunkedArrayToPandas(options, flat_items, nullptr, owned_numpy_items.ref()));
  PyArrayObject* py_keys = reinterpret_cast<PyArrayObject*>(owned_numpy_keys.obj());
  PyArrayObject* py_items = reinterpret_cast<PyArrayObject*>(owned_numpy_items.obj());

  int64_t chunk_offset = 0;
  for (int c = 0; c < data.num_chunks(); ++c) {
    const auto& arr = checked_cast<const MapArray&>(*data.chunk(c));
    const bool has_nulls = data.null_count() > 0;

    // Make a list of key/item pairs for each row in array
    for (int64_t i = 0; i < arr.length(); ++i) {
      if (has_nulls && arr.IsNull(i)) {
        Py_INCREF(Py_None);
        *out_values = Py_None;
      } else {
        int64_t entry_offset = arr.value_offset(i);
        int64_t num_maps = arr.value_offset(i + 1) - entry_offset;

        // Build the new list object for the row of maps
        list_item.reset(PyList_New(num_maps));
        RETURN_IF_PYERROR();

        // Add each key/item pair in the row
        for (int64_t j = 0; j < num_maps; ++j) {
          // Get key value, key is non-nullable for a valid row
          auto ptr_key = reinterpret_cast<const char*>(
              PyArray_GETPTR1(py_keys, chunk_offset + entry_offset + j));
          key_value.reset(PyArray_GETITEM(py_keys, ptr_key));
          RETURN_IF_PYERROR();

          if (item_arrays[c]->IsNull(entry_offset + j)) {
            // Translate the Null to a None
            Py_INCREF(Py_None);
            item_value.reset(Py_None);
          } else {
            // Get valid value from item array
            auto ptr_item = reinterpret_cast<const char*>(
                PyArray_GETPTR1(py_items, chunk_offset + entry_offset + j));
            item_value.reset(PyArray_GETITEM(py_items, ptr_item));
            RETURN_IF_PYERROR();
          }

          // Add the key/item pair to the list for the row
          PyList_SET_ITEM(list_item.obj(), j,
                          PyTuple_Pack(2, key_value.obj(), item_value.obj()));
          RETURN_IF_PYERROR();
        }

        // Pass ownership to the resulting array
        *out_values = list_item.detach();
      }
      ++out_values;
    }
    RETURN_IF_PYERROR();

    chunk_offset += arr.values()->length();
  }

  return Status::OK();
}

template <typename InType, typename OutType>
inline void ConvertNumericNullable(const ChunkedArray& data, InType na_value,
                                   OutType* out_values) {
  for (int c = 0; c < data.num_chunks(); c++) {
    const auto& arr = *data.chunk(c);
    const InType* in_values = GetPrimitiveValues<InType>(arr);

    if (arr.null_count() > 0) {
      for (int64_t i = 0; i < arr.length(); ++i) {
        *out_values++ = arr.IsNull(i) ? na_value : in_values[i];
      }
    } else {
      memcpy(out_values, in_values, sizeof(InType) * arr.length());
      out_values += arr.length();
    }
  }
}

template <typename InType, typename OutType>
inline void ConvertNumericNullableCast(const ChunkedArray& data, InType na_value,
                                       OutType* out_values) {
  for (int c = 0; c < data.num_chunks(); c++) {
    const auto& arr = *data.chunk(c);
    const InType* in_values = GetPrimitiveValues<InType>(arr);

    for (int64_t i = 0; i < arr.length(); ++i) {
      *out_values++ = arr.IsNull(i) ? static_cast<OutType>(na_value)
                                    : static_cast<OutType>(in_values[i]);
    }
  }
}

template <int NPY_TYPE>
class TypedPandasWriter : public PandasWriter {
 public:
  using T = typename npy_traits<NPY_TYPE>::value_type;

  using PandasWriter::PandasWriter;

  Status TransferSingle(std::shared_ptr<ChunkedArray> data, PyObject* py_ref) override {
    if (CanZeroCopy(*data)) {
      PyObject* wrapped;
      npy_intp dims[2] = {static_cast<npy_intp>(num_columns_),
                          static_cast<npy_intp>(num_rows_)};
      RETURN_NOT_OK(
          MakeNumPyView(data->chunk(0), py_ref, NPY_TYPE, /*ndim=*/2, dims, &wrapped));
      SetBlockData(wrapped);
      return Status::OK();
    } else {
      RETURN_NOT_OK(CheckNotZeroCopyOnly(*data));
      RETURN_NOT_OK(EnsureAllocated());
      return CopyInto(data, /*rel_placement=*/0);
    }
  }

  Status CheckTypeExact(const DataType& type, Type::type expected) {
    if (type.id() != expected) {
      // TODO(wesm): stringify NumPy / pandas type
      return Status::NotImplemented("Cannot write Arrow data of type ", type.ToString());
    }
    return Status::OK();
  }

  T* GetBlockColumnStart(int64_t rel_placement) {
    return reinterpret_cast<T*>(block_data_) + rel_placement * num_rows_;
  }

 protected:
  Status Allocate() override { return AllocateNDArray(NPY_TYPE); }
};

struct ObjectWriterVisitor {
  const PandasOptions& options;
  const ChunkedArray& data;
  PyObject** out_values;

  Status Visit(const NullType& type) {
    for (int c = 0; c < data.num_chunks(); c++) {
      std::shared_ptr<Array> arr = data.chunk(c);

      for (int64_t i = 0; i < arr->length(); ++i) {
        // All values are null
        Py_INCREF(Py_None);
        *out_values = Py_None;
        ++out_values;
      }
    }
    return Status::OK();
  }

  Status Visit(const BooleanType& type) {
    for (int c = 0; c < data.num_chunks(); c++) {
      const auto& arr = checked_cast<const BooleanArray&>(*data.chunk(c));

      for (int64_t i = 0; i < arr.length(); ++i) {
        if (arr.IsNull(i)) {
          Py_INCREF(Py_None);
          *out_values++ = Py_None;
        } else if (arr.Value(i)) {
          // True
          Py_INCREF(Py_True);
          *out_values++ = Py_True;
        } else {
          // False
          Py_INCREF(Py_False);
          *out_values++ = Py_False;
        }
      }
    }
    return Status::OK();
  }

  template <typename Type>
  enable_if_integer<Type, Status> Visit(const Type& type) {
    using T = typename Type::c_type;
    auto WrapValue = [](T value, PyObject** out) {
      *out = std::is_signed<T>::value ? PyLong_FromLongLong(value)
                                      : PyLong_FromUnsignedLongLong(value);
      RETURN_IF_PYERROR();
      return Status::OK();
    };
    return ConvertAsPyObjects<Type>(options, data, WrapValue, out_values);
  }

  template <typename Type>
  enable_if_t<is_base_binary_type<Type>::value || is_fixed_size_binary_type<Type>::value,
              Status>
  Visit(const Type& type) {
    auto WrapValue = [](const std::string_view& view, PyObject** out) {
      *out = WrapBytes<Type>::Wrap(view.data(), view.length());
      if (*out == nullptr) {
        PyErr_Clear();
        return Status::UnknownError("Wrapping ", view, " failed");
      }
      return Status::OK();
    };
    return ConvertAsPyObjects<Type>(options, data, WrapValue, out_values);
  }

  template <typename Type>
  enable_if_date<Type, Status> Visit(const Type& type) {
    auto WrapValue = [](typename Type::c_type value, PyObject** out) {
      RETURN_NOT_OK(internal::PyDate_from_int(value, Type::UNIT, out));
      RETURN_IF_PYERROR();
      return Status::OK();
    };
    return ConvertAsPyObjects<Type>(options, data, WrapValue, out_values);
  }

  template <typename Type>
  enable_if_time<Type, Status> Visit(const Type& type) {
    const TimeUnit::type unit = type.unit();
    auto WrapValue = [unit](typename Type::c_type value, PyObject** out) {
      RETURN_NOT_OK(internal::PyTime_from_int(value, unit, out));
      RETURN_IF_PYERROR();
      return Status::OK();
    };
    return ConvertAsPyObjects<Type>(options, data, WrapValue, out_values);
  }

  template <typename Type>
  enable_if_timestamp<Type, Status> Visit(const Type& type) {
    const TimeUnit::type unit = type.unit();
    OwnedRef tzinfo;

    auto ConvertTimezoneNaive = [&](typename Type::c_type value, PyObject** out) {
      RETURN_NOT_OK(internal::PyDateTime_from_int(value, unit, out));
      RETURN_IF_PYERROR();
      return Status::OK();
    };
    auto ConvertTimezoneAware = [&](typename Type::c_type value, PyObject** out) {
      PyObject* naive_datetime;
      RETURN_NOT_OK(ConvertTimezoneNaive(value, &naive_datetime));

      // convert the timezone naive datetime object to timezone aware
      // two step conversion of the datetime mimics Python's code:
      // dt.replace(tzinfo=datetime.timezone.utc).astimezone(tzinfo)
      // first step: replacing timezone with timezone.utc (replace method)
      OwnedRef args(PyTuple_New(0));
      OwnedRef keywords(PyDict_New());
      PyDict_SetItemString(keywords.obj(), "tzinfo", PyDateTime_TimeZone_UTC);
      OwnedRef naive_datetime_replace(PyObject_GetAttrString(naive_datetime, "replace"));
      OwnedRef datetime_utc(
          PyObject_Call(naive_datetime_replace.obj(), args.obj(), keywords.obj()));
      // second step: adjust the datetime to tzinfo timezone (astimezone method)
      *out = PyObject_CallMethod(datetime_utc.obj(), "astimezone", "O", tzinfo.obj());

      // the timezone naive object is no longer required
      Py_DECREF(naive_datetime);
      RETURN_IF_PYERROR();

      return Status::OK();
    };

    if (!type.timezone().empty() && !options.ignore_timezone) {
      // convert timezone aware
      PyObject* tzobj;
      ARROW_ASSIGN_OR_RAISE(tzobj, internal::StringToTzinfo(type.timezone()));
      tzinfo.reset(tzobj);
      RETURN_IF_PYERROR();
      RETURN_NOT_OK(
          ConvertAsPyObjects<Type>(options, data, ConvertTimezoneAware, out_values));
    } else {
      // convert timezone naive
      RETURN_NOT_OK(
          ConvertAsPyObjects<Type>(options, data, ConvertTimezoneNaive, out_values));
    }

    return Status::OK();
  }

  template <typename Type>
  enable_if_t<std::is_same<Type, MonthDayNanoIntervalType>::value, Status> Visit(
      const Type& type) {
    OwnedRef args(PyTuple_New(0));
    OwnedRef kwargs(PyDict_New());
    RETURN_IF_PYERROR();
    auto to_date_offset = [&](const MonthDayNanoIntervalType::MonthDayNanos& interval,
                              PyObject** out) {
      DCHECK(internal::BorrowPandasDataOffsetType() != nullptr);
      // DateOffset objects do not add nanoseconds component to pd.Timestamp.
      // as of  Pandas 1.3.3
      // (https://github.com/pandas-dev/pandas/issues/43892).
      // So convert microseconds and remainder to preserve data
      // but give users more expected results.
      int64_t microseconds = interval.nanoseconds / 1000;
      int64_t nanoseconds;
      if (interval.nanoseconds >= 0) {
        nanoseconds = interval.nanoseconds % 1000;
      } else {
        nanoseconds = -((-interval.nanoseconds) % 1000);
      }

      PyDict_SetItemString(kwargs.obj(), "months", PyLong_FromLong(interval.months));
      PyDict_SetItemString(kwargs.obj(), "days", PyLong_FromLong(interval.days));
      PyDict_SetItemString(kwargs.obj(), "microseconds",
                           PyLong_FromLongLong(microseconds));
      PyDict_SetItemString(kwargs.obj(), "nanoseconds", PyLong_FromLongLong(nanoseconds));
      *out =
          PyObject_Call(internal::BorrowPandasDataOffsetType(), args.obj(), kwargs.obj());
      RETURN_IF_PYERROR();
      return Status::OK();
    };
    return ConvertAsPyObjects<MonthDayNanoIntervalType>(options, data, to_date_offset,
                                                        out_values);
  }

  Status Visit(const Decimal128Type& type) {
    OwnedRef decimal;
    OwnedRef Decimal;
    RETURN_NOT_OK(internal::ImportModule("decimal", &decimal));
    RETURN_NOT_OK(internal::ImportFromModule(decimal.obj(), "Decimal", &Decimal));
    PyObject* decimal_constructor = Decimal.obj();

    for (int c = 0; c < data.num_chunks(); c++) {
      const auto& arr = checked_cast<const arrow::Decimal128Array&>(*data.chunk(c));

      for (int64_t i = 0; i < arr.length(); ++i) {
        if (arr.IsNull(i)) {
          Py_INCREF(Py_None);
          *out_values++ = Py_None;
        } else {
          *out_values++ =
              internal::DecimalFromString(decimal_constructor, arr.FormatValue(i));
          RETURN_IF_PYERROR();
        }
      }
    }

    return Status::OK();
  }

  Status Visit(const Decimal256Type& type) {
    OwnedRef decimal;
    OwnedRef Decimal;
    RETURN_NOT_OK(internal::ImportModule("decimal", &decimal));
    RETURN_NOT_OK(internal::ImportFromModule(decimal.obj(), "Decimal", &Decimal));
    PyObject* decimal_constructor = Decimal.obj();

    for (int c = 0; c < data.num_chunks(); c++) {
      const auto& arr = checked_cast<const arrow::Decimal256Array&>(*data.chunk(c));

      for (int64_t i = 0; i < arr.length(); ++i) {
        if (arr.IsNull(i)) {
          Py_INCREF(Py_None);
          *out_values++ = Py_None;
        } else {
          *out_values++ =
              internal::DecimalFromString(decimal_constructor, arr.FormatValue(i));
          RETURN_IF_PYERROR();
        }
      }
    }

    return Status::OK();
  }

  template <typename T>
  enable_if_t<is_fixed_size_list_type<T>::value || is_var_length_list_type<T>::value,
              Status>
  Visit(const T& type) {
    using ArrayType = typename TypeTraits<T>::ArrayType;
    if (!ListTypeSupported(*type.value_type())) {
      return Status::NotImplemented(
          "Not implemented type for conversion from List to Pandas: ",
          type.value_type()->ToString());
    }
    return ConvertListsLike<ArrayType>(options, data, out_values);
  }

  Status Visit(const MapType& type) { return ConvertMap(options, data, out_values); }

  Status Visit(const StructType& type) {
    return ConvertStruct(options, data, out_values);
  }

  template <typename Type>
  enable_if_t<is_floating_type<Type>::value ||
                  std::is_same<DictionaryType, Type>::value ||
                  std::is_same<DurationType, Type>::value ||
                  std::is_same<ExtensionType, Type>::value ||
                  (std::is_base_of<IntervalType, Type>::value &&
                   !std::is_same<MonthDayNanoIntervalType, Type>::value) ||
                  std::is_base_of<UnionType, Type>::value,
              Status>
  Visit(const Type& type) {
    return Status::NotImplemented("No implemented conversion to object dtype: ",
                                  type.ToString());
  }
};

class ObjectWriter : public TypedPandasWriter<NPY_OBJECT> {
 public:
  using TypedPandasWriter<NPY_OBJECT>::TypedPandasWriter;
  Status CopyInto(std::shared_ptr<ChunkedArray> data, int64_t rel_placement) override {
    PyAcquireGIL lock;
    ObjectWriterVisitor visitor{this->options_, *data,
                                this->GetBlockColumnStart(rel_placement)};
    return VisitTypeInline(*data->type(), &visitor);
  }
};

static inline bool IsNonNullContiguous(const ChunkedArray& data) {
  return data.num_chunks() == 1 && data.null_count() == 0;
}

template <int NPY_TYPE>
class IntWriter : public TypedPandasWriter<NPY_TYPE> {
 public:
  using ArrowType = typename npy_traits<NPY_TYPE>::TypeClass;
  using TypedPandasWriter<NPY_TYPE>::TypedPandasWriter;

  bool CanZeroCopy(const ChunkedArray& data) const override {
    return IsNonNullContiguous(data);
  }

  Status CopyInto(std::shared_ptr<ChunkedArray> data, int64_t rel_placement) override {
    RETURN_NOT_OK(this->CheckTypeExact(*data->type(), ArrowType::type_id));
    ConvertIntegerNoNullsSameType<typename ArrowType::c_type>(
        this->options_, *data, this->GetBlockColumnStart(rel_placement));
    return Status::OK();
  }
};

template <int NPY_TYPE>
class FloatWriter : public TypedPandasWriter<NPY_TYPE> {
 public:
  using ArrowType = typename npy_traits<NPY_TYPE>::TypeClass;
  using TypedPandasWriter<NPY_TYPE>::TypedPandasWriter;
  using T = typename ArrowType::c_type;

  bool CanZeroCopy(const ChunkedArray& data) const override {
    return IsNonNullContiguous(data) && data.type()->id() == ArrowType::type_id;
  }

  Status CopyInto(std::shared_ptr<ChunkedArray> data, int64_t rel_placement) override {
    Type::type in_type = data->type()->id();
    auto out_values = this->GetBlockColumnStart(rel_placement);

#define INTEGER_CASE(IN_TYPE)                                             \
  ConvertIntegerWithNulls<IN_TYPE, T>(this->options_, *data, out_values); \
  break;

    switch (in_type) {
      case Type::UINT8:
        INTEGER_CASE(uint8_t);
      case Type::INT8:
        INTEGER_CASE(int8_t);
      case Type::UINT16:
        INTEGER_CASE(uint16_t);
      case Type::INT16:
        INTEGER_CASE(int16_t);
      case Type::UINT32:
        INTEGER_CASE(uint32_t);
      case Type::INT32:
        INTEGER_CASE(int32_t);
      case Type::UINT64:
        INTEGER_CASE(uint64_t);
      case Type::INT64:
        INTEGER_CASE(int64_t);
      case Type::HALF_FLOAT:
        ConvertNumericNullableCast(*data, npy_traits<NPY_TYPE>::na_sentinel, out_values);
      case Type::FLOAT:
        ConvertNumericNullableCast(*data, npy_traits<NPY_TYPE>::na_sentinel, out_values);
        break;
      case Type::DOUBLE:
        ConvertNumericNullableCast(*data, npy_traits<NPY_TYPE>::na_sentinel, out_values);
        break;
      default:
        return Status::NotImplemented("Cannot write Arrow data of type ",
                                      data->type()->ToString(),
                                      " to a Pandas floating point block");
    }

#undef INTEGER_CASE

    return Status::OK();
  }
};

using UInt8Writer = IntWriter<NPY_UINT8>;
using Int8Writer = IntWriter<NPY_INT8>;
using UInt16Writer = IntWriter<NPY_UINT16>;
using Int16Writer = IntWriter<NPY_INT16>;
using UInt32Writer = IntWriter<NPY_UINT32>;
using Int32Writer = IntWriter<NPY_INT32>;
using UInt64Writer = IntWriter<NPY_UINT64>;
using Int64Writer = IntWriter<NPY_INT64>;
using Float16Writer = FloatWriter<NPY_FLOAT16>;
using Float32Writer = FloatWriter<NPY_FLOAT32>;
using Float64Writer = FloatWriter<NPY_FLOAT64>;

class BoolWriter : public TypedPandasWriter<NPY_BOOL> {
 public:
  using TypedPandasWriter<NPY_BOOL>::TypedPandasWriter;

  Status TransferSingle(std::shared_ptr<ChunkedArray> data, PyObject* py_ref) override {
    RETURN_NOT_OK(
        CheckNoZeroCopy("Zero copy conversions not possible with "
                        "boolean types"));
    RETURN_NOT_OK(EnsureAllocated());
    return CopyInto(data, /*rel_placement=*/0);
  }

  Status CopyInto(std::shared_ptr<ChunkedArray> data, int64_t rel_placement) override {
    RETURN_NOT_OK(this->CheckTypeExact(*data->type(), Type::BOOL));
    auto out_values = this->GetBlockColumnStart(rel_placement);
    for (int c = 0; c < data->num_chunks(); c++) {
      const auto& arr = checked_cast<const BooleanArray&>(*data->chunk(c));
      for (int64_t i = 0; i < arr.length(); ++i) {
        *out_values++ = static_cast<uint8_t>(arr.Value(i));
      }
    }
    return Status::OK();
  }
};

// ----------------------------------------------------------------------
// Date / timestamp types

template <typename T, int64_t SHIFT>
inline void ConvertDatetimeLikeNanos(const ChunkedArray& data, int64_t* out_values) {
  for (int c = 0; c < data.num_chunks(); c++) {
    const auto& arr = *data.chunk(c);
    const T* in_values = GetPrimitiveValues<T>(arr);

    for (int64_t i = 0; i < arr.length(); ++i) {
      *out_values++ = arr.IsNull(i) ? kPandasTimestampNull
                                    : (static_cast<int64_t>(in_values[i]) * SHIFT);
    }
  }
}

template <typename T, int SHIFT>
void ConvertDatesShift(const ChunkedArray& data, int64_t* out_values) {
  for (int c = 0; c < data.num_chunks(); c++) {
    const auto& arr = *data.chunk(c);
    const T* in_values = GetPrimitiveValues<T>(arr);
    for (int64_t i = 0; i < arr.length(); ++i) {
      *out_values++ = arr.IsNull(i) ? kPandasTimestampNull
                                    : static_cast<int64_t>(in_values[i]) / SHIFT;
    }
  }
}

class DatetimeDayWriter : public TypedPandasWriter<NPY_DATETIME> {
 public:
  using TypedPandasWriter<NPY_DATETIME>::TypedPandasWriter;

  Status CopyInto(std::shared_ptr<ChunkedArray> data, int64_t rel_placement) override {
    int64_t* out_values = this->GetBlockColumnStart(rel_placement);
    const auto& type = checked_cast<const DateType&>(*data->type());
    switch (type.unit()) {
      case DateUnit::DAY:
        ConvertDatesShift<int32_t, 1LL>(*data, out_values);
        break;
      case DateUnit::MILLI:
        ConvertDatesShift<int64_t, 86400000LL>(*data, out_values);
        break;
    }
    return Status::OK();
  }

 protected:
  Status Allocate() override {
    RETURN_NOT_OK(this->AllocateNDArray(NPY_DATETIME));
    SetDatetimeUnit(NPY_FR_D);
    return Status::OK();
  }
};

template <TimeUnit::type UNIT>
class DatetimeWriter : public TypedPandasWriter<NPY_DATETIME> {
 public:
  using TypedPandasWriter<NPY_DATETIME>::TypedPandasWriter;

  bool CanZeroCopy(const ChunkedArray& data) const override {
    if (data.type()->id() == Type::TIMESTAMP) {
      const auto& type = checked_cast<const TimestampType&>(*data.type());
      return IsNonNullContiguous(data) && type.unit() == UNIT;
    } else {
      return false;
    }
  }

  Status CopyInto(std::shared_ptr<ChunkedArray> data, int64_t rel_placement) override {
    const auto& ts_type = checked_cast<const TimestampType&>(*data->type());
    DCHECK_EQ(UNIT, ts_type.unit()) << "Should only call instances of this writer "
                                    << "with arrays of the correct unit";
    ConvertNumericNullable<int64_t>(*data, kPandasTimestampNull,
                                    this->GetBlockColumnStart(rel_placement));
    return Status::OK();
  }

 protected:
  Status Allocate() override {
    RETURN_NOT_OK(this->AllocateNDArray(NPY_DATETIME));
    SetDatetimeUnit(internal::NumPyFrequency(UNIT));
    return Status::OK();
  }
};

using DatetimeSecondWriter = DatetimeWriter<TimeUnit::SECOND>;
using DatetimeMilliWriter = DatetimeWriter<TimeUnit::MILLI>;
using DatetimeMicroWriter = DatetimeWriter<TimeUnit::MICRO>;

class DatetimeNanoWriter : public DatetimeWriter<TimeUnit::NANO> {
 public:
  using DatetimeWriter<TimeUnit::NANO>::DatetimeWriter;

  Status CopyInto(std::shared_ptr<ChunkedArray> data, int64_t rel_placement) override {
    Type::type type = data->type()->id();
    int64_t* out_values = this->GetBlockColumnStart(rel_placement);
    compute::ExecContext ctx(options_.pool);
    compute::CastOptions options;
    if (options_.safe_cast) {
      options = compute::CastOptions::Safe();
    } else {
      options = compute::CastOptions::Unsafe();
    }
    Datum out;
    auto target_type = timestamp(TimeUnit::NANO);

    if (type == Type::DATE32) {
      // Convert from days since epoch to datetime64[ns]
      ConvertDatetimeLikeNanos<int32_t, kNanosecondsInDay>(*data, out_values);
    } else if (type == Type::DATE64) {
      // Date64Type is millisecond timestamp stored as int64_t
      // TODO(wesm): Do we want to make sure to zero out the milliseconds?
      ConvertDatetimeLikeNanos<int64_t, 1000000L>(*data, out_values);
    } else if (type == Type::TIMESTAMP) {
      const auto& ts_type = checked_cast<const TimestampType&>(*data->type());

      if (ts_type.unit() == TimeUnit::NANO) {
        ConvertNumericNullable<int64_t>(*data, kPandasTimestampNull, out_values);
      } else if (ts_type.unit() == TimeUnit::MICRO || ts_type.unit() == TimeUnit::MILLI ||
                 ts_type.unit() == TimeUnit::SECOND) {
        ARROW_ASSIGN_OR_RAISE(out, compute::Cast(data, target_type, options, &ctx));
        ConvertNumericNullable<int64_t>(*out.chunked_array(), kPandasTimestampNull,
                                        out_values);
      } else {
        return Status::NotImplemented("Unsupported time unit");
      }
    } else {
      return Status::NotImplemented("Cannot write Arrow data of type ",
                                    data->type()->ToString(),
                                    " to a Pandas datetime block.");
    }
    return Status::OK();
  }
};

class DatetimeTZWriter : public DatetimeNanoWriter {
 public:
  DatetimeTZWriter(const PandasOptions& options, const std::string& timezone,
                   int64_t num_rows)
      : DatetimeNanoWriter(options, num_rows, 1), timezone_(timezone) {}

 protected:
  Status GetResultBlock(PyObject** out) override {
    RETURN_NOT_OK(MakeBlock1D());
    *out = block_arr_.obj();
    return Status::OK();
  }

  Status AddResultMetadata(PyObject* result) override {
    PyObject* py_tz = PyUnicode_FromStringAndSize(
        timezone_.c_str(), static_cast<Py_ssize_t>(timezone_.size()));
    RETURN_IF_PYERROR();
    PyDict_SetItemString(result, "timezone", py_tz);
    Py_DECREF(py_tz);
    return Status::OK();
  }

 private:
  std::string timezone_;
};

template <TimeUnit::type UNIT>
class TimedeltaWriter : public TypedPandasWriter<NPY_TIMEDELTA> {
 public:
  using TypedPandasWriter<NPY_TIMEDELTA>::TypedPandasWriter;

  Status AllocateTimedelta(int ndim) {
    RETURN_NOT_OK(this->AllocateNDArray(NPY_TIMEDELTA, ndim));
    SetDatetimeUnit(internal::NumPyFrequency(UNIT));
    return Status::OK();
  }

  bool CanZeroCopy(const ChunkedArray& data) const override {
    const auto& type = checked_cast<const DurationType&>(*data.type());
    return IsNonNullContiguous(data) && type.unit() == UNIT;
  }

  Status CopyInto(std::shared_ptr<ChunkedArray> data, int64_t rel_placement) override {
    const auto& type = checked_cast<const DurationType&>(*data->type());
    DCHECK_EQ(UNIT, type.unit()) << "Should only call instances of this writer "
                                 << "with arrays of the correct unit";
    ConvertNumericNullable<int64_t>(*data, kPandasTimestampNull,
                                    this->GetBlockColumnStart(rel_placement));
    return Status::OK();
  }

 protected:
  Status Allocate() override { return AllocateTimedelta(2); }
};

using TimedeltaSecondWriter = TimedeltaWriter<TimeUnit::SECOND>;
using TimedeltaMilliWriter = TimedeltaWriter<TimeUnit::MILLI>;
using TimedeltaMicroWriter = TimedeltaWriter<TimeUnit::MICRO>;

class TimedeltaNanoWriter : public TimedeltaWriter<TimeUnit::NANO> {
 public:
  using TimedeltaWriter<TimeUnit::NANO>::TimedeltaWriter;

  Status CopyInto(std::shared_ptr<ChunkedArray> data, int64_t rel_placement) override {
    Type::type type = data->type()->id();
    int64_t* out_values = this->GetBlockColumnStart(rel_placement);
    if (type == Type::DURATION) {
      const auto& ts_type = checked_cast<const DurationType&>(*data->type());
      if (ts_type.unit() == TimeUnit::NANO) {
        ConvertNumericNullable<int64_t>(*data, kPandasTimestampNull, out_values);
      } else if (ts_type.unit() == TimeUnit::MICRO) {
        ConvertDatetimeLikeNanos<int64_t, 1000L>(*data, out_values);
      } else if (ts_type.unit() == TimeUnit::MILLI) {
        ConvertDatetimeLikeNanos<int64_t, 1000000L>(*data, out_values);
      } else if (ts_type.unit() == TimeUnit::SECOND) {
        ConvertDatetimeLikeNanos<int64_t, 1000000000L>(*data, out_values);
      } else {
        return Status::NotImplemented("Unsupported time unit");
      }
    } else {
      return Status::NotImplemented("Cannot write Arrow data of type ",
                                    data->type()->ToString(),
                                    " to a Pandas timedelta block.");
    }
    return Status::OK();
  }
};

Status MakeZeroLengthArray(const std::shared_ptr<DataType>& type,
                           std::shared_ptr<Array>* out) {
  std::unique_ptr<ArrayBuilder> builder;
  RETURN_NOT_OK(MakeBuilder(default_memory_pool(), type, &builder));
  RETURN_NOT_OK(builder->Resize(0));
  return builder->Finish(out);
}

bool NeedDictionaryUnification(const ChunkedArray& data) {
  if (data.num_chunks() < 2) {
    return false;
  }
  const auto& arr_first = checked_cast<const DictionaryArray&>(*data.chunk(0));
  for (int c = 1; c < data.num_chunks(); c++) {
    const auto& arr = checked_cast<const DictionaryArray&>(*data.chunk(c));
    if (!(arr_first.dictionary()->Equals(arr.dictionary()))) {
      return true;
    }
  }
  return false;
}

template <typename IndexType>
class CategoricalWriter
    : public TypedPandasWriter<arrow_traits<IndexType::type_id>::npy_type> {
 public:
  using TRAITS = arrow_traits<IndexType::type_id>;
  using ArrayType = typename TypeTraits<IndexType>::ArrayType;
  using T = typename TRAITS::T;

  explicit CategoricalWriter(const PandasOptions& options, int64_t num_rows)
      : TypedPandasWriter<TRAITS::npy_type>(options, num_rows, 1),
        ordered_(false),
        needs_copy_(false) {}

  Status CopyInto(std::shared_ptr<ChunkedArray> data, int64_t rel_placement) override {
    return Status::NotImplemented("categorical type");
  }

  Status TransferSingle(std::shared_ptr<ChunkedArray> data, PyObject* py_ref) override {
    const auto& dict_type = checked_cast<const DictionaryType&>(*data->type());
    std::shared_ptr<Array> dict;
    if (data->num_chunks() == 0) {
      // no dictionary values => create empty array
      RETURN_NOT_OK(this->AllocateNDArray(TRAITS::npy_type, 1));
      RETURN_NOT_OK(MakeZeroLengthArray(dict_type.value_type(), &dict));
    } else {
      DCHECK_EQ(IndexType::type_id, dict_type.index_type()->id());
      RETURN_NOT_OK(WriteIndices(*data, &dict));
    }

    PyObject* pydict;
    RETURN_NOT_OK(ConvertArrayToPandas(this->options_, dict, nullptr, &pydict));
    dictionary_.reset(pydict);
    ordered_ = dict_type.ordered();
    return Status::OK();
  }

  Status Write(std::shared_ptr<ChunkedArray> data, int64_t abs_placement,
               int64_t rel_placement) override {
    RETURN_NOT_OK(this->EnsurePlacementAllocated());
    RETURN_NOT_OK(TransferSingle(data, /*py_ref=*/nullptr));
    this->placement_data_[rel_placement] = abs_placement;
    return Status::OK();
  }

  Status GetSeriesResult(PyObject** out) override {
    PyAcquireGIL lock;

    PyObject* result = PyDict_New();
    RETURN_IF_PYERROR();

    // Expected single array dictionary layout
    PyDict_SetItemString(result, "indices", this->block_arr_.obj());
    RETURN_IF_PYERROR();
    RETURN_NOT_OK(AddResultMetadata(result));

    *out = result;
    return Status::OK();
  }

 protected:
  Status AddResultMetadata(PyObject* result) override {
    PyDict_SetItemString(result, "dictionary", dictionary_.obj());
    PyObject* py_ordered = ordered_ ? Py_True : Py_False;
    Py_INCREF(py_ordered);
    PyDict_SetItemString(result, "ordered", py_ordered);
    return Status::OK();
  }

  Status WriteIndicesUniform(const ChunkedArray& data) {
    RETURN_NOT_OK(this->AllocateNDArray(TRAITS::npy_type, 1));
    T* out_values = reinterpret_cast<T*>(this->block_data_);

    for (int c = 0; c < data.num_chunks(); c++) {
      const auto& arr = checked_cast<const DictionaryArray&>(*data.chunk(c));
      const auto& indices = checked_cast<const ArrayType&>(*arr.indices());
      auto values = reinterpret_cast<const T*>(indices.raw_values());

      RETURN_NOT_OK(CheckIndexBounds(*indices.data(), arr.dictionary()->length()));
      // Null is -1 in CategoricalBlock
      for (int i = 0; i < arr.length(); ++i) {
        if (indices.IsValid(i)) {
          *out_values++ = values[i];
        } else {
          *out_values++ = -1;
        }
      }
    }
    return Status::OK();
  }

  Status WriteIndicesVarying(const ChunkedArray& data, std::shared_ptr<Array>* out_dict) {
    // Yield int32 indices to allow for dictionary outgrowing the current index
    // type
    RETURN_NOT_OK(this->AllocateNDArray(NPY_INT32, 1));
    auto out_values = reinterpret_cast<int32_t*>(this->block_data_);

    const auto& dict_type = checked_cast<const DictionaryType&>(*data.type());

    ARROW_ASSIGN_OR_RAISE(auto unifier, DictionaryUnifier::Make(dict_type.value_type(),
                                                                this->options_.pool));
    for (int c = 0; c < data.num_chunks(); c++) {
      const auto& arr = checked_cast<const DictionaryArray&>(*data.chunk(c));
      const auto& indices = checked_cast<const ArrayType&>(*arr.indices());
      auto values = reinterpret_cast<const T*>(indices.raw_values());

      std::shared_ptr<Buffer> transpose_buffer;
      RETURN_NOT_OK(unifier->Unify(*arr.dictionary(), &transpose_buffer));

      auto transpose = reinterpret_cast<const int32_t*>(transpose_buffer->data());
      int64_t dict_length = arr.dictionary()->length();

      RETURN_NOT_OK(CheckIndexBounds(*indices.data(), dict_length));

      // Null is -1 in CategoricalBlock
      for (int i = 0; i < arr.length(); ++i) {
        if (indices.IsValid(i)) {
          *out_values++ = transpose[values[i]];
        } else {
          *out_values++ = -1;
        }
      }
    }

    std::shared_ptr<DataType> unused_type;
    return unifier->GetResult(&unused_type, out_dict);
  }

  Status WriteIndices(const ChunkedArray& data, std::shared_ptr<Array>* out_dict) {
    DCHECK_GT(data.num_chunks(), 0);

    // Sniff the first chunk
    const auto& arr_first = checked_cast<const DictionaryArray&>(*data.chunk(0));
    const auto indices_first = std::static_pointer_cast<ArrayType>(arr_first.indices());

    if (data.num_chunks() == 1 && indices_first->null_count() == 0) {
      RETURN_NOT_OK(
          CheckIndexBounds(*indices_first->data(), arr_first.dictionary()->length()));

      PyObject* wrapped;
      npy_intp dims[1] = {static_cast<npy_intp>(this->num_rows_)};
      RETURN_NOT_OK(MakeNumPyView(indices_first, /*py_ref=*/nullptr, TRAITS::npy_type,
                                  /*ndim=*/1, dims, &wrapped));
      this->SetBlockData(wrapped);
      *out_dict = arr_first.dictionary();
    } else {
      RETURN_NOT_OK(this->CheckNotZeroCopyOnly(data));
      if (NeedDictionaryUnification(data)) {
        RETURN_NOT_OK(WriteIndicesVarying(data, out_dict));
      } else {
        RETURN_NOT_OK(WriteIndicesUniform(data));
        *out_dict = arr_first.dictionary();
      }
    }
    return Status::OK();
  }

  OwnedRefNoGIL dictionary_;
  bool ordered_;
  bool needs_copy_;
};

class ExtensionWriter : public PandasWriter {
 public:
  using PandasWriter::PandasWriter;

  Status Allocate() override {
    // no-op
    return Status::OK();
  }

  Status TransferSingle(std::shared_ptr<ChunkedArray> data, PyObject* py_ref) override {
    PyAcquireGIL lock;
    PyObject* py_array;
    py_array = wrap_chunked_array(data);
    py_array_.reset(py_array);

    return Status::OK();
  }

  Status CopyInto(std::shared_ptr<ChunkedArray> data, int64_t rel_placement) override {
    return TransferSingle(data, nullptr);
  }

  Status GetDataFrameResult(PyObject** out) override {
    PyAcquireGIL lock;
    PyObject* result = PyDict_New();
    RETURN_IF_PYERROR();

    PyDict_SetItemString(result, "py_array", py_array_.obj());
    PyDict_SetItemString(result, "placement", placement_arr_.obj());
    *out = result;
    return Status::OK();
  }

  Status GetSeriesResult(PyObject** out) override {
    *out = py_array_.detach();
    return Status::OK();
  }

 protected:
  OwnedRefNoGIL py_array_;
};

Status MakeWriter(const PandasOptions& options, PandasWriter::type writer_type,
                  const DataType& type, int64_t num_rows, int num_columns,
                  std::shared_ptr<PandasWriter>* writer) {
#define BLOCK_CASE(NAME, TYPE)                                        \
  case PandasWriter::NAME:                                            \
    *writer = std::make_shared<TYPE>(options, num_rows, num_columns); \
    break;

#define CATEGORICAL_CASE(TYPE)                                              \
  case TYPE::type_id:                                                       \
    *writer = std::make_shared<CategoricalWriter<TYPE>>(options, num_rows); \
    break;

  switch (writer_type) {
    case PandasWriter::CATEGORICAL: {
      const auto& index_type = *checked_cast<const DictionaryType&>(type).index_type();
      switch (index_type.id()) {
        CATEGORICAL_CASE(Int8Type);
        CATEGORICAL_CASE(Int16Type);
        CATEGORICAL_CASE(Int32Type);
        CATEGORICAL_CASE(Int64Type);
        case Type::UINT8:
        case Type::UINT16:
        case Type::UINT32:
        case Type::UINT64:
          return Status::TypeError(
              "Converting unsigned dictionary indices to pandas",
              " not yet supported, index type: ", index_type.ToString());
        default:
          // Unreachable
          DCHECK(false);
          break;
      }
    } break;
    case PandasWriter::EXTENSION:
      *writer = std::make_shared<ExtensionWriter>(options, num_rows, num_columns);
      break;
      BLOCK_CASE(OBJECT, ObjectWriter);
      BLOCK_CASE(UINT8, UInt8Writer);
      BLOCK_CASE(INT8, Int8Writer);
      BLOCK_CASE(UINT16, UInt16Writer);
      BLOCK_CASE(INT16, Int16Writer);
      BLOCK_CASE(UINT32, UInt32Writer);
      BLOCK_CASE(INT32, Int32Writer);
      BLOCK_CASE(UINT64, UInt64Writer);
      BLOCK_CASE(INT64, Int64Writer);
      BLOCK_CASE(HALF_FLOAT, Float16Writer);
      BLOCK_CASE(FLOAT, Float32Writer);
      BLOCK_CASE(DOUBLE, Float64Writer);
      BLOCK_CASE(BOOL, BoolWriter);
      BLOCK_CASE(DATETIME_DAY, DatetimeDayWriter);
      BLOCK_CASE(DATETIME_SECOND, DatetimeSecondWriter);
      BLOCK_CASE(DATETIME_MILLI, DatetimeMilliWriter);
      BLOCK_CASE(DATETIME_MICRO, DatetimeMicroWriter);
      BLOCK_CASE(DATETIME_NANO, DatetimeNanoWriter);
      BLOCK_CASE(TIMEDELTA_SECOND, TimedeltaSecondWriter);
      BLOCK_CASE(TIMEDELTA_MILLI, TimedeltaMilliWriter);
      BLOCK_CASE(TIMEDELTA_MICRO, TimedeltaMicroWriter);
      BLOCK_CASE(TIMEDELTA_NANO, TimedeltaNanoWriter);
    case PandasWriter::DATETIME_NANO_TZ: {
      const auto& ts_type = checked_cast<const TimestampType&>(type);
      *writer = std::make_shared<DatetimeTZWriter>(options, ts_type.timezone(), num_rows);
    } break;
    default:
      return Status::NotImplemented("Unsupported block type");
  }

#undef BLOCK_CASE
#undef CATEGORICAL_CASE

  return Status::OK();
}

static Status GetPandasWriterType(const ChunkedArray& data, const PandasOptions& options,
                                  PandasWriter::type* output_type) {
#define INTEGER_CASE(NAME)                                                             \
  *output_type =                                                                       \
      data.null_count() > 0                                                            \
          ? options.integer_object_nulls ? PandasWriter::OBJECT : PandasWriter::DOUBLE \
          : PandasWriter::NAME;                                                        \
  break;

  switch (data.type()->id()) {
    case Type::BOOL:
      *output_type = data.null_count() > 0 ? PandasWriter::OBJECT : PandasWriter::BOOL;
      break;
    case Type::UINT8:
      INTEGER_CASE(UINT8);
    case Type::INT8:
      INTEGER_CASE(INT8);
    case Type::UINT16:
      INTEGER_CASE(UINT16);
    case Type::INT16:
      INTEGER_CASE(INT16);
    case Type::UINT32:
      INTEGER_CASE(UINT32);
    case Type::INT32:
      INTEGER_CASE(INT32);
    case Type::UINT64:
      INTEGER_CASE(UINT64);
    case Type::INT64:
      INTEGER_CASE(INT64);
    case Type::HALF_FLOAT:
      *output_type = PandasWriter::HALF_FLOAT;
      break;
    case Type::FLOAT:
      *output_type = PandasWriter::FLOAT;
      break;
    case Type::DOUBLE:
      *output_type = PandasWriter::DOUBLE;
      break;
    case Type::STRING:        // fall through
    case Type::LARGE_STRING:  // fall through
    case Type::BINARY:        // fall through
    case Type::LARGE_BINARY:
    case Type::NA:                       // fall through
    case Type::FIXED_SIZE_BINARY:        // fall through
    case Type::STRUCT:                   // fall through
    case Type::TIME32:                   // fall through
    case Type::TIME64:                   // fall through
    case Type::DECIMAL128:               // fall through
    case Type::DECIMAL256:               // fall through
    case Type::INTERVAL_MONTH_DAY_NANO:  // fall through
      *output_type = PandasWriter::OBJECT;
      break;
    case Type::DATE32:  // fall through
    case Type::DATE64:
      if (options.date_as_object) {
        *output_type = PandasWriter::OBJECT;
      } else {
        *output_type = options.coerce_temporal_nanoseconds ? PandasWriter::DATETIME_NANO
                                                           : PandasWriter::DATETIME_DAY;
      }
      break;
    case Type::TIMESTAMP: {
      const auto& ts_type = checked_cast<const TimestampType&>(*data.type());
      if (options.timestamp_as_object && ts_type.unit() != TimeUnit::NANO) {
        // Nanoseconds are never out of bounds for pandas, so in that case
        // we don't convert to object
        *output_type = PandasWriter::OBJECT;
      } else if (!ts_type.timezone().empty()) {
        *output_type = PandasWriter::DATETIME_NANO_TZ;
      } else if (options.coerce_temporal_nanoseconds) {
        *output_type = PandasWriter::DATETIME_NANO;
      } else {
        switch (ts_type.unit()) {
          case TimeUnit::SECOND:
            *output_type = PandasWriter::DATETIME_SECOND;
            break;
          case TimeUnit::MILLI:
            *output_type = PandasWriter::DATETIME_MILLI;
            break;
          case TimeUnit::MICRO:
            *output_type = PandasWriter::DATETIME_MICRO;
            break;
          case TimeUnit::NANO:
            *output_type = PandasWriter::DATETIME_NANO;
            break;
        }
      }
    } break;
    case Type::DURATION: {
      const auto& dur_type = checked_cast<const DurationType&>(*data.type());
      if (options.coerce_temporal_nanoseconds) {
        *output_type = PandasWriter::TIMEDELTA_NANO;
      } else {
        switch (dur_type.unit()) {
          case TimeUnit::SECOND:
            *output_type = PandasWriter::TIMEDELTA_SECOND;
            break;
          case TimeUnit::MILLI:
            *output_type = PandasWriter::TIMEDELTA_MILLI;
            break;
          case TimeUnit::MICRO:
            *output_type = PandasWriter::TIMEDELTA_MICRO;
            break;
          case TimeUnit::NANO:
            *output_type = PandasWriter::TIMEDELTA_NANO;
            break;
        }
      }
    } break;
    case Type::FIXED_SIZE_LIST:
    case Type::LIST:
    case Type::LARGE_LIST:
    case Type::MAP: {
      auto list_type = std::static_pointer_cast<BaseListType>(data.type());
      if (!ListTypeSupported(*list_type->value_type())) {
        return Status::NotImplemented("Not implemented type for Arrow list to pandas: ",
                                      list_type->value_type()->ToString());
      }
      *output_type = PandasWriter::OBJECT;
    } break;
    case Type::DICTIONARY:
      *output_type = PandasWriter::CATEGORICAL;
      break;
    case Type::EXTENSION:
      *output_type = PandasWriter::EXTENSION;
      break;
    default:
      return Status::NotImplemented(
          "No known equivalent Pandas block for Arrow data of type ",
          data.type()->ToString(), " is known.");
  }
  return Status::OK();
}

// Construct the exact pandas "BlockManager" memory layout
//
// * For each column determine the correct output pandas type
// * Allocate 2D blocks (ncols x nrows) for each distinct data type in output
// * Allocate  block placement arrays
// * Write Arrow columns out into each slice of memory; populate block
// * placement arrays as we go
class PandasBlockCreator {
 public:
  using WriterMap = std::unordered_map<int, std::shared_ptr<PandasWriter>>;

  explicit PandasBlockCreator(const PandasOptions& options, FieldVector fields,
                              ChunkedArrayVector arrays)
      : options_(options), fields_(std::move(fields)), arrays_(std::move(arrays)) {
    num_columns_ = static_cast<int>(arrays_.size());
    if (num_columns_ > 0) {
      num_rows_ = arrays_[0]->length();
    }
    column_block_placement_.resize(num_columns_);
  }
  virtual ~PandasBlockCreator() = default;

  virtual Status Convert(PyObject** out) = 0;

  Status AppendBlocks(const WriterMap& blocks, PyObject* list) {
    for (const auto& it : blocks) {
      PyObject* item;
      RETURN_NOT_OK(it.second->GetDataFrameResult(&item));
      if (PyList_Append(list, item) < 0) {
        RETURN_IF_PYERROR();
      }

      // ARROW-1017; PyList_Append increments object refcount
      Py_DECREF(item);
    }
    return Status::OK();
  }

 protected:
  PandasOptions options_;

  FieldVector fields_;
  ChunkedArrayVector arrays_;
  int num_columns_;
  int64_t num_rows_;

  // column num -> relative placement within internal block
  std::vector<int> column_block_placement_;
};

class ConsolidatedBlockCreator : public PandasBlockCreator {
 public:
  using PandasBlockCreator::PandasBlockCreator;

  Status Convert(PyObject** out) override {
    column_types_.resize(num_columns_);
    RETURN_NOT_OK(CreateBlocks());
    RETURN_NOT_OK(WriteTableToBlocks());
    PyAcquireGIL lock;

    PyObject* result = PyList_New(0);
    RETURN_IF_PYERROR();

    RETURN_NOT_OK(AppendBlocks(blocks_, result));
    RETURN_NOT_OK(AppendBlocks(singleton_blocks_, result));

    *out = result;
    return Status::OK();
  }

  Status GetBlockType(int column_index, PandasWriter::type* out) {
    if (options_.extension_columns.count(fields_[column_index]->name())) {
      *out = PandasWriter::EXTENSION;
      return Status::OK();
    } else {
      return GetPandasWriterType(*arrays_[column_index], options_, out);
    }
  }

  Status CreateBlocks() {
    for (int i = 0; i < num_columns_; ++i) {
      const DataType& type = *arrays_[i]->type();
      PandasWriter::type output_type;
      RETURN_NOT_OK(GetBlockType(i, &output_type));

      int block_placement = 0;
      std::shared_ptr<PandasWriter> writer;
      if (output_type == PandasWriter::CATEGORICAL ||
          output_type == PandasWriter::DATETIME_NANO_TZ ||
          output_type == PandasWriter::EXTENSION) {
        RETURN_NOT_OK(MakeWriter(options_, output_type, type, num_rows_,
                                 /*num_columns=*/1, &writer));
        singleton_blocks_[i] = writer;
      } else {
        auto it = block_sizes_.find(output_type);
        if (it != block_sizes_.end()) {
          block_placement = it->second;
          // Increment count
          ++it->second;
        } else {
          // Add key to map
          block_sizes_[output_type] = 1;
        }
      }
      column_types_[i] = output_type;
      column_block_placement_[i] = block_placement;
    }

    // Create normal non-categorical blocks
    for (const auto& it : this->block_sizes_) {
      PandasWriter::type output_type = static_cast<PandasWriter::type>(it.first);
      std::shared_ptr<PandasWriter> block;
      RETURN_NOT_OK(MakeWriter(this->options_, output_type, /*unused*/ *null(), num_rows_,
                               it.second, &block));
      this->blocks_[output_type] = block;
    }
    return Status::OK();
  }

  Status GetWriter(int i, std::shared_ptr<PandasWriter>* block) {
    PandasWriter::type output_type = this->column_types_[i];
    switch (output_type) {
      case PandasWriter::CATEGORICAL:
      case PandasWriter::DATETIME_NANO_TZ:
      case PandasWriter::EXTENSION: {
        auto it = this->singleton_blocks_.find(i);
        if (it == this->singleton_blocks_.end()) {
          return Status::KeyError("No block allocated");
        }
        *block = it->second;
      } break;
      default:
        auto it = this->blocks_.find(output_type);
        if (it == this->blocks_.end()) {
          return Status::KeyError("No block allocated");
        }
        *block = it->second;
        break;
    }
    return Status::OK();
  }

  Status WriteTableToBlocks() {
    auto WriteColumn = [this](int i) {
      std::shared_ptr<PandasWriter> block;
      RETURN_NOT_OK(this->GetWriter(i, &block));
      // ARROW-3789 Use std::move on the array to permit self-destructing
      return block->Write(std::move(arrays_[i]), i, this->column_block_placement_[i]);
    };

    return OptionalParallelFor(options_.use_threads, num_columns_, WriteColumn);
  }

 private:
  // column num -> block type id
  std::vector<PandasWriter::type> column_types_;

  // block type -> type count
  std::unordered_map<int, int> block_sizes_;
  std::unordered_map<int, const DataType*> block_types_;

  // block type -> block
  WriterMap blocks_;

  WriterMap singleton_blocks_;
};

/// \brief Create blocks for pandas.DataFrame block manager using one block per
/// column strategy. This permits some zero-copy optimizations as well as the
/// ability for the table to "self-destruct" if selected by the user.
class SplitBlockCreator : public PandasBlockCreator {
 public:
  using PandasBlockCreator::PandasBlockCreator;

  Status GetWriter(int i, std::shared_ptr<PandasWriter>* writer) {
    PandasWriter::type output_type = PandasWriter::OBJECT;
    const DataType& type = *arrays_[i]->type();
    if (options_.extension_columns.count(fields_[i]->name())) {
      output_type = PandasWriter::EXTENSION;
    } else {
      // Null count needed to determine output type
      RETURN_NOT_OK(GetPandasWriterType(*arrays_[i], options_, &output_type));
    }
    return MakeWriter(this->options_, output_type, type, num_rows_, 1, writer);
  }

  Status Convert(PyObject** out) override {
    PyAcquireGIL lock;

    PyObject* result = PyList_New(0);
    RETURN_IF_PYERROR();

    for (int i = 0; i < num_columns_; ++i) {
      std::shared_ptr<PandasWriter> writer;
      RETURN_NOT_OK(GetWriter(i, &writer));
      // ARROW-3789 Use std::move on the array to permit self-destructing
      RETURN_NOT_OK(writer->Write(std::move(arrays_[i]), i, /*rel_placement=*/0));

      PyObject* item;
      RETURN_NOT_OK(writer->GetDataFrameResult(&item));
      if (PyList_Append(result, item) < 0) {
        RETURN_IF_PYERROR();
      }
      // PyList_Append increments object refcount
      Py_DECREF(item);
    }

    *out = result;
    return Status::OK();
  }

 private:
  std::vector<std::shared_ptr<PandasWriter>> writers_;
};

Status ConvertCategoricals(const PandasOptions& options, ChunkedArrayVector* arrays,
                           FieldVector* fields) {
  std::vector<int> columns_to_encode;

  // For Categorical conversions
  auto EncodeColumn = [&](int j) {
    int i = columns_to_encode[j];
    if (options.zero_copy_only) {
      return Status::Invalid("Need to dictionary encode a column, but ",
                             "only zero-copy conversions allowed");
    }
    compute::ExecContext ctx(options.pool);
    ARROW_ASSIGN_OR_RAISE(
        Datum out, DictionaryEncode((*arrays)[i],
                                    compute::DictionaryEncodeOptions::Defaults(), &ctx));
    (*arrays)[i] = out.chunked_array();
    (*fields)[i] = (*fields)[i]->WithType((*arrays)[i]->type());
    return Status::OK();
  };

  if (!options.categorical_columns.empty()) {
    for (int i = 0; i < static_cast<int>(arrays->size()); i++) {
      if ((*arrays)[i]->type()->id() != Type::DICTIONARY &&
          options.categorical_columns.count((*fields)[i]->name())) {
        columns_to_encode.push_back(i);
      }
    }
  }
  if (options.strings_to_categorical) {
    for (int i = 0; i < static_cast<int>(arrays->size()); i++) {
      if (is_base_binary_like((*arrays)[i]->type()->id())) {
        columns_to_encode.push_back(i);
      }
    }
  }
  return OptionalParallelFor(options.use_threads,
                             static_cast<int>(columns_to_encode.size()), EncodeColumn);
}

}  // namespace

Status ConvertArrayToPandas(const PandasOptions& options, std::shared_ptr<Array> arr,
                            PyObject* py_ref, PyObject** out) {
  return ConvertChunkedArrayToPandas(
      options, std::make_shared<ChunkedArray>(std::move(arr)), py_ref, out);
}

Status ConvertChunkedArrayToPandas(const PandasOptions& options,
                                   std::shared_ptr<ChunkedArray> arr, PyObject* py_ref,
                                   PyObject** out) {
  if (options.decode_dictionaries && arr->type()->id() == Type::DICTIONARY) {
    const auto& dense_type =
        checked_cast<const DictionaryType&>(*arr->type()).value_type();
    RETURN_NOT_OK(DecodeDictionaries(options.pool, dense_type, &arr));
    DCHECK_NE(arr->type()->id(), Type::DICTIONARY);

    // The original Python DictionaryArray won't own the memory anymore
    // as we actually built a new array when we decoded the DictionaryArray
    // thus let the final resulting numpy array own the memory through a Capsule
    py_ref = nullptr;
  }

  if (options.strings_to_categorical && is_base_binary_like(arr->type()->id())) {
    if (options.zero_copy_only) {
      return Status::Invalid("Need to dictionary encode a column, but ",
                             "only zero-copy conversions allowed");
    }
    compute::ExecContext ctx(options.pool);
    ARROW_ASSIGN_OR_RAISE(
        Datum out,
        DictionaryEncode(arr, compute::DictionaryEncodeOptions::Defaults(), &ctx));
    arr = out.chunked_array();
  }

  PandasOptions modified_options = options;
  modified_options.strings_to_categorical = false;

  // ARROW-7596: We permit the hybrid Series/DataFrame code path to do zero copy
  // optimizations that we do not allow in the default case when converting
  // Table->DataFrame
  modified_options.allow_zero_copy_blocks = true;

  PandasWriter::type output_type;
  RETURN_NOT_OK(GetPandasWriterType(*arr, modified_options, &output_type));
  if (options.decode_dictionaries) {
    DCHECK_NE(output_type, PandasWriter::CATEGORICAL);
  }

  std::shared_ptr<PandasWriter> writer;
  RETURN_NOT_OK(MakeWriter(modified_options, output_type, *arr->type(), arr->length(),
                           /*num_columns=*/1, &writer));
  RETURN_NOT_OK(writer->TransferSingle(std::move(arr), py_ref));
  return writer->GetSeriesResult(out);
}

Status ConvertTableToPandas(const PandasOptions& options, std::shared_ptr<Table> table,
                            PyObject** out) {
  ChunkedArrayVector arrays = table->columns();
  FieldVector fields = table->fields();

  // ARROW-3789: allow "self-destructing" by releasing references to columns as
  // we convert them to pandas
  table = nullptr;

  RETURN_NOT_OK(ConvertCategoricals(options, &arrays, &fields));

  PandasOptions modified_options = options;
  modified_options.strings_to_categorical = false;
  modified_options.categorical_columns.clear();

  if (options.split_blocks) {
    modified_options.allow_zero_copy_blocks = true;
    SplitBlockCreator helper(modified_options, std::move(fields), std::move(arrays));
    return helper.Convert(out);
  } else {
    ConsolidatedBlockCreator helper(modified_options, std::move(fields),
                                    std::move(arrays));
    return helper.Convert(out);
  }
}

}  // namespace py
}  // namespace arrow

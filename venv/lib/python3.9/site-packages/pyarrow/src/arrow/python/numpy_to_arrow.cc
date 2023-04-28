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

#include "arrow/python/numpy_to_arrow.h"
#include "arrow/python/numpy_interop.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "arrow/array.h"
#include "arrow/array/builder_binary.h"
#include "arrow/status.h"
#include "arrow/table.h"
#include "arrow/type_fwd.h"
#include "arrow/type_traits.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/bitmap_generate.h"
#include "arrow/util/bitmap_ops.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/endian.h"
#include "arrow/util/logging.h"
#include "arrow/util/macros.h"
#include "arrow/util/string.h"
#include "arrow/util/utf8.h"
#include "arrow/visit_type_inline.h"

#include "arrow/compute/api_scalar.h"

#include "arrow/python/common.h"
#include "arrow/python/datetime.h"
#include "arrow/python/helpers.h"
#include "arrow/python/iterators.h"
#include "arrow/python/numpy_convert.h"
#include "arrow/python/numpy_internal.h"
#include "arrow/python/python_to_arrow.h"
#include "arrow/python/type_traits.h"

namespace arrow {

using internal::checked_cast;
using internal::CopyBitmap;
using internal::GenerateBitsUnrolled;

namespace py {

using internal::NumPyTypeSize;

// ----------------------------------------------------------------------
// Conversion utilities

namespace {

Status AllocateNullBitmap(MemoryPool* pool, int64_t length,
                          std::shared_ptr<ResizableBuffer>* out) {
  int64_t null_bytes = bit_util::BytesForBits(length);
  ARROW_ASSIGN_OR_RAISE(auto null_bitmap, AllocateResizableBuffer(null_bytes, pool));

  // Padding zeroed by AllocateResizableBuffer
  memset(null_bitmap->mutable_data(), 0, static_cast<size_t>(null_bytes));
  *out = std::move(null_bitmap);
  return Status::OK();
}

// ----------------------------------------------------------------------
// Conversion from NumPy-in-Pandas to Arrow null bitmap

template <int TYPE>
inline int64_t ValuesToBitmap(PyArrayObject* arr, uint8_t* bitmap) {
  typedef internal::npy_traits<TYPE> traits;
  typedef typename traits::value_type T;

  int64_t null_count = 0;

  Ndarray1DIndexer<T> values(arr);
  for (int i = 0; i < values.size(); ++i) {
    if (traits::isnull(values[i])) {
      ++null_count;
    } else {
      bit_util::SetBit(bitmap, i);
    }
  }

  return null_count;
}

class NumPyNullsConverter {
 public:
  /// Convert the given array's null values to a null bitmap.
  /// The null bitmap is only allocated if null values are ever possible.
  static Status Convert(MemoryPool* pool, PyArrayObject* arr, bool from_pandas,
                        std::shared_ptr<ResizableBuffer>* out_null_bitmap_,
                        int64_t* out_null_count) {
    NumPyNullsConverter converter(pool, arr, from_pandas);
    RETURN_NOT_OK(VisitNumpyArrayInline(arr, &converter));
    *out_null_bitmap_ = converter.null_bitmap_;
    *out_null_count = converter.null_count_;
    return Status::OK();
  }

  template <int TYPE>
  Status Visit(PyArrayObject* arr) {
    typedef internal::npy_traits<TYPE> traits;

    const bool null_sentinels_possible =
        // Always treat Numpy's NaT as null
        TYPE == NPY_DATETIME || TYPE == NPY_TIMEDELTA ||
        // Observing pandas's null sentinels
        (from_pandas_ && traits::supports_nulls);

    if (null_sentinels_possible) {
      RETURN_NOT_OK(AllocateNullBitmap(pool_, PyArray_SIZE(arr), &null_bitmap_));
      null_count_ = ValuesToBitmap<TYPE>(arr, null_bitmap_->mutable_data());
    }
    return Status::OK();
  }

 protected:
  NumPyNullsConverter(MemoryPool* pool, PyArrayObject* arr, bool from_pandas)
      : pool_(pool),
        arr_(arr),
        from_pandas_(from_pandas),
        null_bitmap_data_(nullptr),
        null_count_(0) {}

  MemoryPool* pool_;
  PyArrayObject* arr_;
  bool from_pandas_;
  std::shared_ptr<ResizableBuffer> null_bitmap_;
  uint8_t* null_bitmap_data_;
  int64_t null_count_;
};

// Returns null count
int64_t MaskToBitmap(PyArrayObject* mask, int64_t length, uint8_t* bitmap) {
  int64_t null_count = 0;

  if (!PyArray_Check(mask)) return -1;

  Ndarray1DIndexer<uint8_t> mask_values(mask);
  for (int i = 0; i < length; ++i) {
    if (mask_values[i]) {
      ++null_count;
      bit_util::ClearBit(bitmap, i);
    } else {
      bit_util::SetBit(bitmap, i);
    }
  }
  return null_count;
}

}  // namespace

// ----------------------------------------------------------------------
// Conversion from NumPy arrays (possibly originating from pandas) to Arrow
// format. Does not handle NPY_OBJECT dtype arrays; use ConvertPySequence for
// that

class NumPyConverter {
 public:
  NumPyConverter(MemoryPool* pool, PyObject* arr, PyObject* mo,
                 const std::shared_ptr<DataType>& type, bool from_pandas,
                 const compute::CastOptions& cast_options = compute::CastOptions())
      : pool_(pool),
        type_(type),
        arr_(reinterpret_cast<PyArrayObject*>(arr)),
        dtype_(PyArray_DESCR(arr_)),
        mask_(nullptr),
        from_pandas_(from_pandas),
        cast_options_(cast_options),
        null_bitmap_data_(nullptr),
        null_count_(0) {
    if (mo != nullptr && mo != Py_None) {
      mask_ = reinterpret_cast<PyArrayObject*>(mo);
    }
    length_ = static_cast<int64_t>(PyArray_SIZE(arr_));
    itemsize_ = static_cast<int>(PyArray_DESCR(arr_)->elsize);
    stride_ = static_cast<int64_t>(PyArray_STRIDES(arr_)[0]);
  }

  bool is_strided() const { return itemsize_ != stride_; }

  Status Convert();

  const ArrayVector& result() const { return out_arrays_; }

  template <typename T>
  enable_if_primitive_ctype<T, Status> Visit(const T& type) {
    return VisitNative<T>();
  }

  Status Visit(const HalfFloatType& type) { return VisitNative<UInt16Type>(); }

  Status Visit(const Date32Type& type) { return VisitNative<Date32Type>(); }
  Status Visit(const Date64Type& type) { return VisitNative<Date64Type>(); }
  Status Visit(const TimestampType& type) { return VisitNative<TimestampType>(); }
  Status Visit(const Time32Type& type) { return VisitNative<Int32Type>(); }
  Status Visit(const Time64Type& type) { return VisitNative<Int64Type>(); }
  Status Visit(const DurationType& type) { return VisitNative<DurationType>(); }

  Status Visit(const NullType& type) { return TypeNotImplemented(type.ToString()); }

  // NumPy ascii string arrays
  Status Visit(const BinaryType& type);

  // NumPy unicode arrays
  Status Visit(const StringType& type);

  Status Visit(const StructType& type);

  Status Visit(const FixedSizeBinaryType& type);

  // Default case
  Status Visit(const DataType& type) { return TypeNotImplemented(type.ToString()); }

 protected:
  Status InitNullBitmap() {
    RETURN_NOT_OK(AllocateNullBitmap(pool_, length_, &null_bitmap_));
    null_bitmap_data_ = null_bitmap_->mutable_data();
    return Status::OK();
  }

  // Called before ConvertData to ensure Numpy input buffer is in expected
  // Arrow layout
  template <typename ArrowType>
  Status PrepareInputData(std::shared_ptr<Buffer>* data);

  // ----------------------------------------------------------------------
  // Traditional visitor conversion for non-object arrays

  template <typename ArrowType>
  Status ConvertData(std::shared_ptr<Buffer>* data);

  template <typename T>
  Status PushBuilderResult(T* builder) {
    std::shared_ptr<Array> out;
    RETURN_NOT_OK(builder->Finish(&out));
    out_arrays_.emplace_back(out);
    return Status::OK();
  }

  Status PushArray(const std::shared_ptr<ArrayData>& data) {
    out_arrays_.emplace_back(MakeArray(data));
    return Status::OK();
  }

  template <typename ArrowType>
  Status VisitNative() {
    if (mask_ != nullptr) {
      RETURN_NOT_OK(InitNullBitmap());
      null_count_ = MaskToBitmap(mask_, length_, null_bitmap_data_);
      if (null_count_ == -1) return Status::Invalid("Invalid mask type");
    } else {
      RETURN_NOT_OK(NumPyNullsConverter::Convert(pool_, arr_, from_pandas_, &null_bitmap_,
                                                 &null_count_));
    }

    std::shared_ptr<Buffer> data;
    RETURN_NOT_OK(ConvertData<ArrowType>(&data));

    auto arr_data = ArrayData::Make(type_, length_, {null_bitmap_, data}, null_count_, 0);
    return PushArray(arr_data);
  }

  Status TypeNotImplemented(std::string type_name) {
    return Status::NotImplemented("NumPyConverter doesn't implement <", type_name,
                                  "> conversion. ");
  }

  MemoryPool* pool_;
  std::shared_ptr<DataType> type_;
  PyArrayObject* arr_;
  PyArray_Descr* dtype_;
  PyArrayObject* mask_;
  int64_t length_;
  int64_t stride_;
  int itemsize_;

  bool from_pandas_;
  compute::CastOptions cast_options_;

  // Used in visitor pattern
  ArrayVector out_arrays_;

  std::shared_ptr<ResizableBuffer> null_bitmap_;
  uint8_t* null_bitmap_data_;
  int64_t null_count_;
};

Status NumPyConverter::Convert() {
  if (PyArray_NDIM(arr_) != 1) {
    return Status::Invalid("only handle 1-dimensional arrays");
  }

  if (dtype_->type_num == NPY_OBJECT) {
    // If an object array, convert it like a normal Python sequence
    PyConversionOptions py_options;
    py_options.type = type_;
    py_options.from_pandas = from_pandas_;
    ARROW_ASSIGN_OR_RAISE(
        auto chunked_array,
        ConvertPySequence(reinterpret_cast<PyObject*>(arr_),
                          reinterpret_cast<PyObject*>(mask_), py_options, pool_));
    out_arrays_ = chunked_array->chunks();
    return Status::OK();
  }

  if (type_ == nullptr) {
    return Status::Invalid("Must pass data type for non-object arrays");
  }

  // Visit the type to perform conversion
  return VisitTypeInline(*type_, this);
}

namespace {

Status CastBuffer(const std::shared_ptr<DataType>& in_type,
                  const std::shared_ptr<Buffer>& input, const int64_t length,
                  const std::shared_ptr<Buffer>& valid_bitmap, const int64_t null_count,
                  const std::shared_ptr<DataType>& out_type,
                  const compute::CastOptions& cast_options, MemoryPool* pool,
                  std::shared_ptr<Buffer>* out) {
  // Must cast
  auto tmp_data = ArrayData::Make(in_type, length, {valid_bitmap, input}, null_count);
  compute::ExecContext context(pool);
  ARROW_ASSIGN_OR_RAISE(
      std::shared_ptr<Array> casted_array,
      compute::Cast(*MakeArray(tmp_data), out_type, cast_options, &context));
  *out = casted_array->data()->buffers[1];
  return Status::OK();
}

template <typename FromType, typename ToType>
Status StaticCastBuffer(const Buffer& input, const int64_t length, MemoryPool* pool,
                        std::shared_ptr<Buffer>* out) {
  ARROW_ASSIGN_OR_RAISE(auto result, AllocateBuffer(sizeof(ToType) * length, pool));

  auto in_values = reinterpret_cast<const FromType*>(input.data());
  auto out_values = reinterpret_cast<ToType*>(result->mutable_data());
  for (int64_t i = 0; i < length; ++i) {
    *out_values++ = static_cast<ToType>(*in_values++);
  }
  *out = std::move(result);
  return Status::OK();
}

template <typename T>
void CopyStridedBytewise(int8_t* input_data, int64_t length, int64_t stride,
                         T* output_data) {
  // Passing input_data as non-const is a concession to PyObject*
  for (int64_t i = 0; i < length; ++i) {
    memcpy(output_data + i, input_data, sizeof(T));
    input_data += stride;
  }
}

template <typename T>
void CopyStridedNatural(T* input_data, int64_t length, int64_t stride, T* output_data) {
  // Passing input_data as non-const is a concession to PyObject*
  int64_t j = 0;
  for (int64_t i = 0; i < length; ++i) {
    output_data[i] = input_data[j];
    j += stride;
  }
}

class NumPyStridedConverter {
 public:
  static Status Convert(PyArrayObject* arr, int64_t length, MemoryPool* pool,
                        std::shared_ptr<Buffer>* out) {
    NumPyStridedConverter converter(arr, length, pool);
    RETURN_NOT_OK(VisitNumpyArrayInline(arr, &converter));
    *out = converter.buffer_;
    return Status::OK();
  }
  template <int TYPE>
  Status Visit(PyArrayObject* arr) {
    using traits = internal::npy_traits<TYPE>;
    using T = typename traits::value_type;

    ARROW_ASSIGN_OR_RAISE(buffer_, AllocateBuffer(sizeof(T) * length_, pool_));

    const int64_t stride = PyArray_STRIDES(arr)[0];
    // ARROW-16013: convert sizeof(T) to signed int64 first, otherwise dividing by it
    // would do an unsigned division. This cannot be caught by tests without ubsan, since
    // common signed overflow behavior and the fact that the sizeof(T) is currently always
    // a power of two here cause CopyStridedNatural to still produce correct results
    const int64_t element_size = sizeof(T);
    if (stride % element_size == 0) {
      const int64_t stride_elements = stride / element_size;
      CopyStridedNatural(reinterpret_cast<T*>(PyArray_DATA(arr)), length_,
                         stride_elements, reinterpret_cast<T*>(buffer_->mutable_data()));
    } else {
      CopyStridedBytewise(reinterpret_cast<int8_t*>(PyArray_DATA(arr)), length_, stride,
                          reinterpret_cast<T*>(buffer_->mutable_data()));
    }
    return Status::OK();
  }

 protected:
  NumPyStridedConverter(PyArrayObject* arr, int64_t length, MemoryPool* pool)
      : arr_(arr), length_(length), pool_(pool), buffer_(nullptr) {}
  PyArrayObject* arr_;
  int64_t length_;
  MemoryPool* pool_;
  std::shared_ptr<Buffer> buffer_;
};

}  // namespace

template <typename ArrowType>
inline Status NumPyConverter::PrepareInputData(std::shared_ptr<Buffer>* data) {
  if (PyArray_ISBYTESWAPPED(arr_)) {
    // TODO
    return Status::NotImplemented("Byte-swapped arrays not supported");
  }

  if (dtype_->type_num == NPY_BOOL) {
    int64_t nbytes = bit_util::BytesForBits(length_);
    ARROW_ASSIGN_OR_RAISE(auto buffer, AllocateBuffer(nbytes, pool_));

    Ndarray1DIndexer<uint8_t> values(arr_);
    int64_t i = 0;
    const auto generate = [&values, &i]() -> bool { return values[i++] > 0; };
    GenerateBitsUnrolled(buffer->mutable_data(), 0, length_, generate);

    *data = std::move(buffer);
  } else if (is_strided()) {
    RETURN_NOT_OK(NumPyStridedConverter::Convert(arr_, length_, pool_, data));
  } else {
    // Can zero-copy
    *data = std::make_shared<NumPyBuffer>(reinterpret_cast<PyObject*>(arr_));
  }

  return Status::OK();
}

template <typename ArrowType>
inline Status NumPyConverter::ConvertData(std::shared_ptr<Buffer>* data) {
  RETURN_NOT_OK(PrepareInputData<ArrowType>(data));

  std::shared_ptr<DataType> input_type;
  RETURN_NOT_OK(NumPyDtypeToArrow(reinterpret_cast<PyObject*>(dtype_), &input_type));

  if (!input_type->Equals(*type_)) {
    RETURN_NOT_OK(CastBuffer(input_type, *data, length_, null_bitmap_, null_count_, type_,
                             cast_options_, pool_, data));
  }

  return Status::OK();
}

template <>
inline Status NumPyConverter::ConvertData<Date32Type>(std::shared_ptr<Buffer>* data) {
  std::shared_ptr<DataType> input_type;

  RETURN_NOT_OK(PrepareInputData<Date32Type>(data));

  auto date_dtype = reinterpret_cast<PyArray_DatetimeDTypeMetaData*>(dtype_->c_metadata);
  if (dtype_->type_num == NPY_DATETIME) {
    // If we have inbound datetime64[D] data, this needs to be downcasted
    // separately here from int64_t to int32_t, because this data is not
    // supported in compute::Cast
    if (date_dtype->meta.base == NPY_FR_D) {
      // TODO(wesm): How pedantic do we really want to be about checking for int32
      // overflow here?
      Status s = StaticCastBuffer<int64_t, int32_t>(**data, length_, pool_, data);
      RETURN_NOT_OK(s);
    } else {
      RETURN_NOT_OK(NumPyDtypeToArrow(reinterpret_cast<PyObject*>(dtype_), &input_type));
      if (!input_type->Equals(*type_)) {
        // The null bitmap was already computed in VisitNative()
        RETURN_NOT_OK(CastBuffer(input_type, *data, length_, null_bitmap_, null_count_,
                                 type_, cast_options_, pool_, data));
      }
    }
  } else {
    RETURN_NOT_OK(NumPyDtypeToArrow(reinterpret_cast<PyObject*>(dtype_), &input_type));
    if (!input_type->Equals(*type_)) {
      RETURN_NOT_OK(CastBuffer(input_type, *data, length_, null_bitmap_, null_count_,
                               type_, cast_options_, pool_, data));
    }
  }

  return Status::OK();
}

template <>
inline Status NumPyConverter::ConvertData<Date64Type>(std::shared_ptr<Buffer>* data) {
  constexpr int64_t kMillisecondsInDay = 86400000;
  std::shared_ptr<DataType> input_type;

  RETURN_NOT_OK(PrepareInputData<Date64Type>(data));

  auto date_dtype = reinterpret_cast<PyArray_DatetimeDTypeMetaData*>(dtype_->c_metadata);
  if (dtype_->type_num == NPY_DATETIME) {
    // If we have inbound datetime64[D] data, this needs to be downcasted
    // separately here from int64_t to int32_t, because this data is not
    // supported in compute::Cast
    if (date_dtype->meta.base == NPY_FR_D) {
      ARROW_ASSIGN_OR_RAISE(auto result,
                            AllocateBuffer(sizeof(int64_t) * length_, pool_));

      auto in_values = reinterpret_cast<const int64_t*>((*data)->data());
      auto out_values = reinterpret_cast<int64_t*>(result->mutable_data());
      for (int64_t i = 0; i < length_; ++i) {
        *out_values++ = kMillisecondsInDay * (*in_values++);
      }
      *data = std::move(result);
    } else {
      RETURN_NOT_OK(NumPyDtypeToArrow(reinterpret_cast<PyObject*>(dtype_), &input_type));
      if (!input_type->Equals(*type_)) {
        // The null bitmap was already computed in VisitNative()
        RETURN_NOT_OK(CastBuffer(input_type, *data, length_, null_bitmap_, null_count_,
                                 type_, cast_options_, pool_, data));
      }
    }
  } else {
    RETURN_NOT_OK(NumPyDtypeToArrow(reinterpret_cast<PyObject*>(dtype_), &input_type));
    if (!input_type->Equals(*type_)) {
      RETURN_NOT_OK(CastBuffer(input_type, *data, length_, null_bitmap_, null_count_,
                               type_, cast_options_, pool_, data));
    }
  }

  return Status::OK();
}

// Create 16MB chunks for binary data
constexpr int32_t kBinaryChunksize = 1 << 24;

Status NumPyConverter::Visit(const BinaryType& type) {
  ::arrow::internal::ChunkedBinaryBuilder builder(kBinaryChunksize, pool_);

  auto data = reinterpret_cast<const uint8_t*>(PyArray_DATA(arr_));

  auto AppendNotNull = [&builder, this](const uint8_t* data) {
    // This is annoying. NumPy allows strings to have nul-terminators, so
    // we must check for them here
    const size_t item_size =
        strnlen(reinterpret_cast<const char*>(data), static_cast<size_t>(itemsize_));
    return builder.Append(data, static_cast<int32_t>(item_size));
  };

  if (mask_ != nullptr) {
    Ndarray1DIndexer<uint8_t> mask_values(mask_);
    for (int64_t i = 0; i < length_; ++i) {
      if (mask_values[i]) {
        RETURN_NOT_OK(builder.AppendNull());
      } else {
        RETURN_NOT_OK(AppendNotNull(data));
      }
      data += stride_;
    }
  } else {
    for (int64_t i = 0; i < length_; ++i) {
      RETURN_NOT_OK(AppendNotNull(data));
      data += stride_;
    }
  }

  ArrayVector result;
  RETURN_NOT_OK(builder.Finish(&result));
  for (auto arr : result) {
    RETURN_NOT_OK(PushArray(arr->data()));
  }
  return Status::OK();
}

Status NumPyConverter::Visit(const FixedSizeBinaryType& type) {
  auto byte_width = type.byte_width();

  if (itemsize_ != byte_width) {
    return Status::Invalid("Got bytestring of length ", itemsize_, " (expected ",
                           byte_width, ")");
  }

  FixedSizeBinaryBuilder builder(::arrow::fixed_size_binary(byte_width), pool_);
  auto data = reinterpret_cast<const uint8_t*>(PyArray_DATA(arr_));

  if (mask_ != nullptr) {
    Ndarray1DIndexer<uint8_t> mask_values(mask_);
    RETURN_NOT_OK(builder.Reserve(length_));
    for (int64_t i = 0; i < length_; ++i) {
      if (mask_values[i]) {
        RETURN_NOT_OK(builder.AppendNull());
      } else {
        RETURN_NOT_OK(builder.Append(data));
      }
      data += stride_;
    }
  } else {
    for (int64_t i = 0; i < length_; ++i) {
      RETURN_NOT_OK(builder.Append(data));
      data += stride_;
    }
  }

  std::shared_ptr<Array> result;
  RETURN_NOT_OK(builder.Finish(&result));
  return PushArray(result->data());
}

namespace {

// NumPy unicode is UCS4/UTF32 always
constexpr int kNumPyUnicodeSize = 4;

Status AppendUTF32(const char* data, int itemsize, int byteorder,
                   ::arrow::internal::ChunkedStringBuilder* builder) {
  // The binary \x00\x00\x00\x00 indicates a nul terminator in NumPy unicode,
  // so we need to detect that here to truncate if necessary. Yep.
  int actual_length = 0;
  for (; actual_length < itemsize / kNumPyUnicodeSize; ++actual_length) {
    const char* code_point = data + actual_length * kNumPyUnicodeSize;
    if ((*code_point == '\0') && (*(code_point + 1) == '\0') &&
        (*(code_point + 2) == '\0') && (*(code_point + 3) == '\0')) {
      break;
    }
  }

  OwnedRef unicode_obj(PyUnicode_DecodeUTF32(data, actual_length * kNumPyUnicodeSize,
                                             nullptr, &byteorder));
  RETURN_IF_PYERROR();
  OwnedRef utf8_obj(PyUnicode_AsUTF8String(unicode_obj.obj()));
  if (utf8_obj.obj() == NULL) {
    PyErr_Clear();
    return Status::Invalid("failed converting UTF32 to UTF8");
  }

  const int32_t length = static_cast<int32_t>(PyBytes_GET_SIZE(utf8_obj.obj()));
  return builder->Append(
      reinterpret_cast<const uint8_t*>(PyBytes_AS_STRING(utf8_obj.obj())), length);
}

}  // namespace

Status NumPyConverter::Visit(const StringType& type) {
  util::InitializeUTF8();

  ::arrow::internal::ChunkedStringBuilder builder(kBinaryChunksize, pool_);

  auto data = reinterpret_cast<const uint8_t*>(PyArray_DATA(arr_));

  char numpy_byteorder = dtype_->byteorder;

  // For Python C API, -1 is little-endian, 1 is big-endian
#if ARROW_LITTLE_ENDIAN
  // Yield little-endian from both '|' (native) and '<'
  int byteorder = numpy_byteorder == '>' ? 1 : -1;
#else
  // Yield big-endian from both '|' (native) and '>'
  int byteorder = numpy_byteorder == '<' ? -1 : 1;
#endif

  PyAcquireGIL gil_lock;

  const bool is_binary_type = dtype_->type_num == NPY_STRING;
  const bool is_unicode_type = dtype_->type_num == NPY_UNICODE;

  if (!is_binary_type && !is_unicode_type) {
    const bool is_float_type = dtype_->kind == 'f';
    if (from_pandas_ && is_float_type) {
      // in case of from_pandas=True, accept an all-NaN float array as input
      RETURN_NOT_OK(NumPyNullsConverter::Convert(pool_, arr_, from_pandas_, &null_bitmap_,
                                                 &null_count_));
      if (null_count_ == length_) {
        auto arr = std::make_shared<NullArray>(length_);
        compute::ExecContext context(pool_);
        ARROW_ASSIGN_OR_RAISE(
            std::shared_ptr<Array> out,
            compute::Cast(*arr, arrow::utf8(), cast_options_, &context));
        out_arrays_.emplace_back(out);
        return Status::OK();
      }
    }
    std::string dtype_string;
    RETURN_NOT_OK(internal::PyObject_StdStringStr(reinterpret_cast<PyObject*>(dtype_),
                                                  &dtype_string));
    return Status::TypeError("Expected a string or bytes dtype, got ", dtype_string);
  }

  auto AppendNonNullValue = [&](const uint8_t* data) {
    if (is_binary_type) {
      if (ARROW_PREDICT_TRUE(util::ValidateUTF8(data, itemsize_))) {
        return builder.Append(data, itemsize_);
      } else {
        return Status::Invalid("Encountered non-UTF8 binary value: ",
                               HexEncode(data, itemsize_));
      }
    } else {
      // is_unicode_type case
      return AppendUTF32(reinterpret_cast<const char*>(data), itemsize_, byteorder,
                         &builder);
    }
  };

  if (mask_ != nullptr) {
    Ndarray1DIndexer<uint8_t> mask_values(mask_);
    for (int64_t i = 0; i < length_; ++i) {
      if (mask_values[i]) {
        RETURN_NOT_OK(builder.AppendNull());
      } else {
        RETURN_NOT_OK(AppendNonNullValue(data));
      }
      data += stride_;
    }
  } else {
    for (int64_t i = 0; i < length_; ++i) {
      RETURN_NOT_OK(AppendNonNullValue(data));
      data += stride_;
    }
  }

  ArrayVector result;
  RETURN_NOT_OK(builder.Finish(&result));
  for (auto arr : result) {
    RETURN_NOT_OK(PushArray(arr->data()));
  }
  return Status::OK();
}

Status NumPyConverter::Visit(const StructType& type) {
  std::vector<NumPyConverter> sub_converters;
  std::vector<OwnedRefNoGIL> sub_arrays;

  {
    PyAcquireGIL gil_lock;

    // Create converters for each struct type field
    if (dtype_->fields == NULL || !PyDict_Check(dtype_->fields)) {
      return Status::TypeError("Expected struct array");
    }

    for (auto field : type.fields()) {
      PyObject* tup = PyDict_GetItemString(dtype_->fields, field->name().c_str());
      if (tup == NULL) {
        return Status::Invalid("Missing field '", field->name(), "' in struct array");
      }
      PyArray_Descr* sub_dtype =
          reinterpret_cast<PyArray_Descr*>(PyTuple_GET_ITEM(tup, 0));
      DCHECK(PyObject_TypeCheck(sub_dtype, &PyArrayDescr_Type));
      int offset = static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(tup, 1)));
      RETURN_IF_PYERROR();
      Py_INCREF(sub_dtype); /* PyArray_GetField() steals ref */
      PyObject* sub_array = PyArray_GetField(arr_, sub_dtype, offset);
      RETURN_IF_PYERROR();
      sub_arrays.emplace_back(sub_array);
      sub_converters.emplace_back(pool_, sub_array, nullptr /* mask */, field->type(),
                                  from_pandas_);
    }
  }

  std::vector<ArrayVector> groups;
  int64_t null_count = 0;

  // Compute null bitmap and store it as a Boolean Array to include it
  // in the rechunking below
  {
    if (mask_ != nullptr) {
      RETURN_NOT_OK(InitNullBitmap());
      null_count = MaskToBitmap(mask_, length_, null_bitmap_data_);
      if (null_count_ == -1) return Status::Invalid("Invalid mask type");
    }
    groups.push_back({std::make_shared<BooleanArray>(length_, null_bitmap_)});
  }

  // Convert child data
  for (auto& converter : sub_converters) {
    RETURN_NOT_OK(converter.Convert());
    groups.push_back(converter.result());
  }
  // Ensure the different array groups are chunked consistently
  groups = ::arrow::internal::RechunkArraysConsistently(groups);

  // Make struct array chunks by combining groups
  size_t ngroups = groups.size();
  size_t nchunks = groups[0].size();
  for (size_t chunk = 0; chunk < nchunks; chunk++) {
    // First group has the null bitmaps as Boolean Arrays
    const auto& null_data = groups[0][chunk]->data();
    DCHECK_EQ(null_data->type->id(), Type::BOOL);
    DCHECK_EQ(null_data->buffers.size(), 2);
    const auto& null_buffer = null_data->buffers[1];
    // Careful: the rechunked null bitmap may have a non-zero offset
    // to its buffer, and it may not even start on a byte boundary
    int64_t null_offset = null_data->offset;
    std::shared_ptr<Buffer> fixed_null_buffer;

    if (!null_buffer) {
      fixed_null_buffer = null_buffer;
    } else if (null_offset % 8 == 0) {
      fixed_null_buffer =
          std::make_shared<Buffer>(null_buffer,
                                   // byte offset
                                   null_offset / 8,
                                   // byte size
                                   bit_util::BytesForBits(null_data->length));
    } else {
      ARROW_ASSIGN_OR_RAISE(
          fixed_null_buffer,
          CopyBitmap(pool_, null_buffer->data(), null_offset, null_data->length));
    }

    // Create struct array chunk and populate it
    auto arr_data =
        ArrayData::Make(type_, null_data->length, null_count ? kUnknownNullCount : 0, 0);
    arr_data->buffers.push_back(fixed_null_buffer);
    // Append child chunks
    for (size_t i = 1; i < ngroups; i++) {
      arr_data->child_data.push_back(groups[i][chunk]->data());
    }
    RETURN_NOT_OK(PushArray(arr_data));
  }

  return Status::OK();
}

Status NdarrayToArrow(MemoryPool* pool, PyObject* ao, PyObject* mo, bool from_pandas,
                      const std::shared_ptr<DataType>& type,
                      const compute::CastOptions& cast_options,
                      std::shared_ptr<ChunkedArray>* out) {
  if (!PyArray_Check(ao)) {
    // This code path cannot be reached by Python unit tests currently so this
    // is only a sanity check.
    return Status::TypeError("Input object was not a NumPy array");
  }
  if (PyArray_NDIM(reinterpret_cast<PyArrayObject*>(ao)) != 1) {
    return Status::Invalid("only handle 1-dimensional arrays");
  }

  NumPyConverter converter(pool, ao, mo, type, from_pandas, cast_options);
  RETURN_NOT_OK(converter.Convert());
  const auto& output_arrays = converter.result();
  DCHECK_GT(output_arrays.size(), 0);
  *out = std::make_shared<ChunkedArray>(output_arrays);
  return Status::OK();
}

Status NdarrayToArrow(MemoryPool* pool, PyObject* ao, PyObject* mo, bool from_pandas,
                      const std::shared_ptr<DataType>& type,
                      std::shared_ptr<ChunkedArray>* out) {
  return NdarrayToArrow(pool, ao, mo, from_pandas, type, compute::CastOptions(), out);
}

}  // namespace py
}  // namespace arrow

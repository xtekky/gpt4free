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

#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "arrow/util/spaced.h"

#include "parquet/exception.h"
#include "parquet/platform.h"
#include "parquet/types.h"

namespace arrow {

class Array;
class ArrayBuilder;
class BinaryArray;
class BinaryBuilder;
class BooleanBuilder;
class Int32Type;
class Int64Type;
class FloatType;
class DoubleType;
class FixedSizeBinaryType;
template <typename T>
class NumericBuilder;
class FixedSizeBinaryBuilder;
template <typename T>
class Dictionary32Builder;

}  // namespace arrow

namespace parquet {

template <typename DType>
class TypedEncoder;

using BooleanEncoder = TypedEncoder<BooleanType>;
using Int32Encoder = TypedEncoder<Int32Type>;
using Int64Encoder = TypedEncoder<Int64Type>;
using Int96Encoder = TypedEncoder<Int96Type>;
using FloatEncoder = TypedEncoder<FloatType>;
using DoubleEncoder = TypedEncoder<DoubleType>;
using ByteArrayEncoder = TypedEncoder<ByteArrayType>;
using FLBAEncoder = TypedEncoder<FLBAType>;

template <typename DType>
class TypedDecoder;

class BooleanDecoder;
using Int32Decoder = TypedDecoder<Int32Type>;
using Int64Decoder = TypedDecoder<Int64Type>;
using Int96Decoder = TypedDecoder<Int96Type>;
using FloatDecoder = TypedDecoder<FloatType>;
using DoubleDecoder = TypedDecoder<DoubleType>;
using ByteArrayDecoder = TypedDecoder<ByteArrayType>;
class FLBADecoder;

template <typename T>
struct EncodingTraits;

template <>
struct EncodingTraits<BooleanType> {
  using Encoder = BooleanEncoder;
  using Decoder = BooleanDecoder;

  using ArrowType = ::arrow::BooleanType;
  using Accumulator = ::arrow::BooleanBuilder;
  struct DictAccumulator {};
};

template <>
struct EncodingTraits<Int32Type> {
  using Encoder = Int32Encoder;
  using Decoder = Int32Decoder;

  using ArrowType = ::arrow::Int32Type;
  using Accumulator = ::arrow::NumericBuilder<::arrow::Int32Type>;
  using DictAccumulator = ::arrow::Dictionary32Builder<::arrow::Int32Type>;
};

template <>
struct EncodingTraits<Int64Type> {
  using Encoder = Int64Encoder;
  using Decoder = Int64Decoder;

  using ArrowType = ::arrow::Int64Type;
  using Accumulator = ::arrow::NumericBuilder<::arrow::Int64Type>;
  using DictAccumulator = ::arrow::Dictionary32Builder<::arrow::Int64Type>;
};

template <>
struct EncodingTraits<Int96Type> {
  using Encoder = Int96Encoder;
  using Decoder = Int96Decoder;

  struct Accumulator {};
  struct DictAccumulator {};
};

template <>
struct EncodingTraits<FloatType> {
  using Encoder = FloatEncoder;
  using Decoder = FloatDecoder;

  using ArrowType = ::arrow::FloatType;
  using Accumulator = ::arrow::NumericBuilder<::arrow::FloatType>;
  using DictAccumulator = ::arrow::Dictionary32Builder<::arrow::FloatType>;
};

template <>
struct EncodingTraits<DoubleType> {
  using Encoder = DoubleEncoder;
  using Decoder = DoubleDecoder;

  using ArrowType = ::arrow::DoubleType;
  using Accumulator = ::arrow::NumericBuilder<::arrow::DoubleType>;
  using DictAccumulator = ::arrow::Dictionary32Builder<::arrow::DoubleType>;
};

template <>
struct EncodingTraits<ByteArrayType> {
  using Encoder = ByteArrayEncoder;
  using Decoder = ByteArrayDecoder;

  /// \brief Internal helper class for decoding BYTE_ARRAY data where we can
  /// overflow the capacity of a single arrow::BinaryArray
  struct Accumulator {
    std::unique_ptr<::arrow::BinaryBuilder> builder;
    std::vector<std::shared_ptr<::arrow::Array>> chunks;
  };
  using ArrowType = ::arrow::BinaryType;
  using DictAccumulator = ::arrow::Dictionary32Builder<::arrow::BinaryType>;
};

template <>
struct EncodingTraits<FLBAType> {
  using Encoder = FLBAEncoder;
  using Decoder = FLBADecoder;

  using ArrowType = ::arrow::FixedSizeBinaryType;
  using Accumulator = ::arrow::FixedSizeBinaryBuilder;
  using DictAccumulator = ::arrow::Dictionary32Builder<::arrow::FixedSizeBinaryType>;
};

class ColumnDescriptor;

// Untyped base for all encoders
class Encoder {
 public:
  virtual ~Encoder() = default;

  virtual int64_t EstimatedDataEncodedSize() = 0;
  virtual std::shared_ptr<Buffer> FlushValues() = 0;
  virtual Encoding::type encoding() const = 0;

  virtual void Put(const ::arrow::Array& values) = 0;

  virtual MemoryPool* memory_pool() const = 0;
};

// Base class for value encoders. Since encoders may or not have state (e.g.,
// dictionary encoding) we use a class instance to maintain any state.
//
// Encode interfaces are internal, subject to change without deprecation.
template <typename DType>
class TypedEncoder : virtual public Encoder {
 public:
  typedef typename DType::c_type T;

  using Encoder::Put;

  virtual void Put(const T* src, int num_values) = 0;

  virtual void Put(const std::vector<T>& src, int num_values = -1);

  virtual void PutSpaced(const T* src, int num_values, const uint8_t* valid_bits,
                         int64_t valid_bits_offset) = 0;
};

template <typename DType>
void TypedEncoder<DType>::Put(const std::vector<T>& src, int num_values) {
  if (num_values == -1) {
    num_values = static_cast<int>(src.size());
  }
  Put(src.data(), num_values);
}

template <>
inline void TypedEncoder<BooleanType>::Put(const std::vector<bool>& src, int num_values) {
  // NOTE(wesm): This stub is here only to satisfy the compiler; it is
  // overridden later with the actual implementation
}

// Base class for dictionary encoders
template <typename DType>
class DictEncoder : virtual public TypedEncoder<DType> {
 public:
  /// Writes out any buffered indices to buffer preceded by the bit width of this data.
  /// Returns the number of bytes written.
  /// If the supplied buffer is not big enough, returns -1.
  /// buffer must be preallocated with buffer_len bytes. Use EstimatedDataEncodedSize()
  /// to size buffer.
  virtual int WriteIndices(uint8_t* buffer, int buffer_len) = 0;

  virtual int dict_encoded_size() = 0;
  // virtual int dict_encoded_size() { return dict_encoded_size_; }

  virtual int bit_width() const = 0;

  /// Writes out the encoded dictionary to buffer. buffer must be preallocated to
  /// dict_encoded_size() bytes.
  virtual void WriteDict(uint8_t* buffer) = 0;

  virtual int num_entries() const = 0;

  /// \brief EXPERIMENTAL: Append dictionary indices into the encoder. It is
  /// assumed (without any boundschecking) that the indices reference
  /// pre-existing dictionary values
  /// \param[in] indices the dictionary index values. Only Int32Array currently
  /// supported
  virtual void PutIndices(const ::arrow::Array& indices) = 0;

  /// \brief EXPERIMENTAL: Append dictionary into encoder, inserting indices
  /// separately. Currently throws exception if the current dictionary memo is
  /// non-empty
  /// \param[in] values the dictionary values. Only valid for certain
  /// Parquet/Arrow type combinations, like BYTE_ARRAY/BinaryArray
  virtual void PutDictionary(const ::arrow::Array& values) = 0;
};

// ----------------------------------------------------------------------
// Value decoding

class Decoder {
 public:
  virtual ~Decoder() = default;

  // Sets the data for a new page. This will be called multiple times on the same
  // decoder and should reset all internal state.
  virtual void SetData(int num_values, const uint8_t* data, int len) = 0;

  // Returns the number of values left (for the last call to SetData()). This is
  // the number of values left in this page.
  virtual int values_left() const = 0;
  virtual Encoding::type encoding() const = 0;
};

template <typename DType>
class TypedDecoder : virtual public Decoder {
 public:
  using T = typename DType::c_type;

  /// \brief Decode values into a buffer
  ///
  /// Subclasses may override the more specialized Decode methods below.
  ///
  /// \param[in] buffer destination for decoded values
  /// \param[in] max_values maximum number of values to decode
  /// \return The number of values decoded. Should be identical to max_values except
  /// at the end of the current data page.
  virtual int Decode(T* buffer, int max_values) = 0;

  /// \brief Decode the values in this data page but leave spaces for null entries.
  ///
  /// \param[in] buffer destination for decoded values
  /// \param[in] num_values size of the def_levels and buffer arrays including the number
  /// of null slots
  /// \param[in] null_count number of null slots
  /// \param[in] valid_bits bitmap data indicating position of valid slots
  /// \param[in] valid_bits_offset offset into valid_bits
  /// \return The number of values decoded, including nulls.
  virtual int DecodeSpaced(T* buffer, int num_values, int null_count,
                           const uint8_t* valid_bits, int64_t valid_bits_offset) {
    if (null_count > 0) {
      int values_to_read = num_values - null_count;
      int values_read = Decode(buffer, values_to_read);
      if (values_read != values_to_read) {
        throw ParquetException("Number of values / definition_levels read did not match");
      }

      return ::arrow::util::internal::SpacedExpand<T>(buffer, num_values, null_count,
                                                      valid_bits, valid_bits_offset);
    } else {
      return Decode(buffer, num_values);
    }
  }

  /// \brief Decode into an ArrayBuilder or other accumulator
  ///
  /// This function assumes the definition levels were already decoded
  /// as a validity bitmap in the given `valid_bits`.  `null_count`
  /// is the number of 0s in `valid_bits`.
  /// As a space optimization, it is allowed for `valid_bits` to be null
  /// if `null_count` is zero.
  ///
  /// \return number of values decoded
  virtual int DecodeArrow(int num_values, int null_count, const uint8_t* valid_bits,
                          int64_t valid_bits_offset,
                          typename EncodingTraits<DType>::Accumulator* out) = 0;

  /// \brief Decode into an ArrayBuilder or other accumulator ignoring nulls
  ///
  /// \return number of values decoded
  int DecodeArrowNonNull(int num_values,
                         typename EncodingTraits<DType>::Accumulator* out) {
    return DecodeArrow(num_values, 0, /*valid_bits=*/NULLPTR, 0, out);
  }

  /// \brief Decode into a DictionaryBuilder
  ///
  /// This function assumes the definition levels were already decoded
  /// as a validity bitmap in the given `valid_bits`.  `null_count`
  /// is the number of 0s in `valid_bits`.
  /// As a space optimization, it is allowed for `valid_bits` to be null
  /// if `null_count` is zero.
  ///
  /// \return number of values decoded
  virtual int DecodeArrow(int num_values, int null_count, const uint8_t* valid_bits,
                          int64_t valid_bits_offset,
                          typename EncodingTraits<DType>::DictAccumulator* builder) = 0;

  /// \brief Decode into a DictionaryBuilder ignoring nulls
  ///
  /// \return number of values decoded
  int DecodeArrowNonNull(int num_values,
                         typename EncodingTraits<DType>::DictAccumulator* builder) {
    return DecodeArrow(num_values, 0, /*valid_bits=*/NULLPTR, 0, builder);
  }
};

template <typename DType>
class DictDecoder : virtual public TypedDecoder<DType> {
 public:
  using T = typename DType::c_type;

  virtual void SetDict(TypedDecoder<DType>* dictionary) = 0;

  /// \brief Insert dictionary values into the Arrow dictionary builder's memo,
  /// but do not append any indices
  virtual void InsertDictionary(::arrow::ArrayBuilder* builder) = 0;

  /// \brief Decode only dictionary indices and append to dictionary
  /// builder. The builder must have had the dictionary from this decoder
  /// inserted already.
  ///
  /// \warning Remember to reset the builder each time the dict decoder is initialized
  /// with a new dictionary page
  virtual int DecodeIndicesSpaced(int num_values, int null_count,
                                  const uint8_t* valid_bits, int64_t valid_bits_offset,
                                  ::arrow::ArrayBuilder* builder) = 0;

  /// \brief Decode only dictionary indices (no nulls)
  ///
  /// \warning Remember to reset the builder each time the dict decoder is initialized
  /// with a new dictionary page
  virtual int DecodeIndices(int num_values, ::arrow::ArrayBuilder* builder) = 0;

  /// \brief Decode only dictionary indices (no nulls). Same as above
  /// DecodeIndices but target is an array instead of a builder.
  ///
  /// \note API EXPERIMENTAL
  virtual int DecodeIndices(int num_values, int32_t* indices) = 0;

  /// \brief Get dictionary. The reader will call this API when it encounters a
  /// new dictionary.
  ///
  /// @param[out] dictionary The pointer to dictionary values. Dictionary is owned by
  /// the decoder and is destroyed when the decoder is destroyed.
  /// @param[out] dictionary_length The dictionary length.
  ///
  /// \note API EXPERIMENTAL
  virtual void GetDictionary(const T** dictionary, int32_t* dictionary_length) = 0;
};

// ----------------------------------------------------------------------
// TypedEncoder specializations, traits, and factory functions

class BooleanDecoder : virtual public TypedDecoder<BooleanType> {
 public:
  using TypedDecoder<BooleanType>::Decode;

  /// \brief Decode and bit-pack values into a buffer
  ///
  /// \param[in] buffer destination for decoded values
  /// This buffer will contain bit-packed values.
  /// \param[in] max_values max values to decode.
  /// \return The number of values decoded. Should be identical to max_values except
  /// at the end of the current data page.
  virtual int Decode(uint8_t* buffer, int max_values) = 0;
};

class FLBADecoder : virtual public TypedDecoder<FLBAType> {
 public:
  using TypedDecoder<FLBAType>::DecodeSpaced;

  // TODO(wesm): As possible follow-up to PARQUET-1508, we should examine if
  // there is value in adding specialized read methods for
  // FIXED_LEN_BYTE_ARRAY. If only Decimal data can occur with this data type
  // then perhaps not
};

PARQUET_EXPORT
std::unique_ptr<Encoder> MakeEncoder(
    Type::type type_num, Encoding::type encoding, bool use_dictionary = false,
    const ColumnDescriptor* descr = NULLPTR,
    ::arrow::MemoryPool* pool = ::arrow::default_memory_pool());

template <typename DType>
std::unique_ptr<typename EncodingTraits<DType>::Encoder> MakeTypedEncoder(
    Encoding::type encoding, bool use_dictionary = false,
    const ColumnDescriptor* descr = NULLPTR,
    ::arrow::MemoryPool* pool = ::arrow::default_memory_pool()) {
  using OutType = typename EncodingTraits<DType>::Encoder;
  std::unique_ptr<Encoder> base =
      MakeEncoder(DType::type_num, encoding, use_dictionary, descr, pool);
  return std::unique_ptr<OutType>(dynamic_cast<OutType*>(base.release()));
}

PARQUET_EXPORT
std::unique_ptr<Decoder> MakeDecoder(Type::type type_num, Encoding::type encoding,
                                     const ColumnDescriptor* descr = NULLPTR);

namespace detail {

PARQUET_EXPORT
std::unique_ptr<Decoder> MakeDictDecoder(Type::type type_num,
                                         const ColumnDescriptor* descr,
                                         ::arrow::MemoryPool* pool);

}  // namespace detail

template <typename DType>
std::unique_ptr<DictDecoder<DType>> MakeDictDecoder(
    const ColumnDescriptor* descr = NULLPTR,
    ::arrow::MemoryPool* pool = ::arrow::default_memory_pool()) {
  using OutType = DictDecoder<DType>;
  auto decoder = detail::MakeDictDecoder(DType::type_num, descr, pool);
  return std::unique_ptr<OutType>(dynamic_cast<OutType*>(decoder.release()));
}

template <typename DType>
std::unique_ptr<typename EncodingTraits<DType>::Decoder> MakeTypedDecoder(
    Encoding::type encoding, const ColumnDescriptor* descr = NULLPTR) {
  using OutType = typename EncodingTraits<DType>::Decoder;
  std::unique_ptr<Decoder> base = MakeDecoder(DType::type_num, encoding, descr);
  return std::unique_ptr<OutType>(dynamic_cast<OutType*>(base.release()));
}

}  // namespace parquet

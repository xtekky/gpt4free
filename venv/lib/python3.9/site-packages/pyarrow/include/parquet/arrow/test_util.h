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

#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "arrow/array.h"
#include "arrow/array/builder_binary.h"
#include "arrow/array/builder_decimal.h"
#include "arrow/array/builder_primitive.h"
#include "arrow/testing/gtest_util.h"
#include "arrow/testing/random.h"
#include "arrow/type_fwd.h"
#include "arrow/type_traits.h"
#include "arrow/util/decimal.h"
#include "parquet/column_reader.h"

namespace parquet {

using internal::RecordReader;

namespace arrow {

using ::arrow::Array;
using ::arrow::ChunkedArray;
using ::arrow::Status;

template <int32_t PRECISION>
struct DecimalWithPrecisionAndScale {
  static_assert(PRECISION >= 1 && PRECISION <= 38, "Invalid precision value");

  using type = ::arrow::Decimal128Type;
  static constexpr ::arrow::Type::type type_id = ::arrow::Decimal128Type::type_id;
  static constexpr int32_t precision = PRECISION;
  static constexpr int32_t scale = PRECISION - 1;
};

template <int32_t PRECISION>
struct Decimal256WithPrecisionAndScale {
  static_assert(PRECISION >= 1 && PRECISION <= 76, "Invalid precision value");

  using type = ::arrow::Decimal256Type;
  static constexpr ::arrow::Type::type type_id = ::arrow::Decimal256Type::type_id;
  static constexpr int32_t precision = PRECISION;
  static constexpr int32_t scale = PRECISION - 1;
};

template <class ArrowType>
::arrow::enable_if_floating_point<ArrowType, Status> NonNullArray(
    size_t size, std::shared_ptr<Array>* out) {
  using c_type = typename ArrowType::c_type;
  std::vector<c_type> values;
  ::arrow::random_real(size, 0, static_cast<c_type>(0), static_cast<c_type>(1), &values);
  ::arrow::NumericBuilder<ArrowType> builder;
  RETURN_NOT_OK(builder.AppendValues(values.data(), values.size()));
  return builder.Finish(out);
}

template <class ArrowType>
::arrow::enable_if_integer<ArrowType, Status> NonNullArray(size_t size,
                                                           std::shared_ptr<Array>* out) {
  std::vector<typename ArrowType::c_type> values;
  ::arrow::randint(size, 0, 64, &values);

  // Passing data type so this will work with TimestampType too
  ::arrow::NumericBuilder<ArrowType> builder(std::make_shared<ArrowType>(),
                                             ::arrow::default_memory_pool());
  RETURN_NOT_OK(builder.AppendValues(values.data(), values.size()));
  return builder.Finish(out);
}

template <class ArrowType>
::arrow::enable_if_date<ArrowType, Status> NonNullArray(size_t size,
                                                        std::shared_ptr<Array>* out) {
  std::vector<typename ArrowType::c_type> values;
  ::arrow::randint(size, 0, 24, &values);
  for (size_t i = 0; i < size; i++) {
    values[i] *= 86400000;
  }

  // Passing data type so this will work with TimestampType too
  ::arrow::NumericBuilder<ArrowType> builder(std::make_shared<ArrowType>(),
                                             ::arrow::default_memory_pool());
  RETURN_NOT_OK(builder.AppendValues(values.data(), values.size()));
  return builder.Finish(out);
}

template <class ArrowType>
::arrow::enable_if_base_binary<ArrowType, Status> NonNullArray(
    size_t size, std::shared_ptr<Array>* out) {
  using BuilderType = typename ::arrow::TypeTraits<ArrowType>::BuilderType;
  BuilderType builder;
  for (size_t i = 0; i < size; i++) {
    RETURN_NOT_OK(builder.Append("test-string"));
  }
  return builder.Finish(out);
}

template <typename ArrowType>
::arrow::enable_if_fixed_size_binary<ArrowType, Status> NonNullArray(
    size_t size, std::shared_ptr<Array>* out) {
  using BuilderType = typename ::arrow::TypeTraits<ArrowType>::BuilderType;
  // set byte_width to the length of "fixed": 5
  // todo: find a way to generate test data with more diversity.
  BuilderType builder(::arrow::fixed_size_binary(5));
  for (size_t i = 0; i < size; i++) {
    RETURN_NOT_OK(builder.Append("fixed"));
  }
  return builder.Finish(out);
}

template <int32_t byte_width>
static void random_decimals(int64_t n, uint32_t seed, int32_t precision, uint8_t* out) {
  auto gen = ::arrow::random::RandomArrayGenerator(seed);
  std::shared_ptr<Array> decimals;
  if constexpr (byte_width == 16) {
    decimals = gen.Decimal128(::arrow::decimal128(precision, 0), n);
  } else {
    decimals = gen.Decimal256(::arrow::decimal256(precision, 0), n);
  }
  std::memcpy(out, decimals->data()->GetValues<uint8_t>(1, 0), byte_width * n);
}

template <typename ArrowType, int32_t precision = ArrowType::precision>
::arrow::enable_if_t<
    std::is_same<ArrowType, DecimalWithPrecisionAndScale<precision>>::value, Status>
NonNullArray(size_t size, std::shared_ptr<Array>* out) {
  constexpr int32_t kDecimalPrecision = precision;
  constexpr int32_t kDecimalScale = DecimalWithPrecisionAndScale<precision>::scale;

  const auto type = ::arrow::decimal(kDecimalPrecision, kDecimalScale);
  ::arrow::Decimal128Builder builder(type);
  const int32_t byte_width =
      static_cast<const ::arrow::Decimal128Type&>(*type).byte_width();

  constexpr int32_t seed = 0;

  ARROW_ASSIGN_OR_RAISE(auto out_buf, ::arrow::AllocateBuffer(size * byte_width));
  random_decimals<::arrow::Decimal128Type::kByteWidth>(size, seed, kDecimalPrecision,
                                                       out_buf->mutable_data());

  RETURN_NOT_OK(builder.AppendValues(out_buf->data(), size));
  return builder.Finish(out);
}

template <typename ArrowType, int32_t precision = ArrowType::precision>
::arrow::enable_if_t<
    std::is_same<ArrowType, Decimal256WithPrecisionAndScale<precision>>::value, Status>
NonNullArray(size_t size, std::shared_ptr<Array>* out) {
  constexpr int32_t kDecimalPrecision = precision;
  constexpr int32_t kDecimalScale = Decimal256WithPrecisionAndScale<precision>::scale;

  const auto type = ::arrow::decimal256(kDecimalPrecision, kDecimalScale);
  ::arrow::Decimal256Builder builder(type);
  const int32_t byte_width =
      static_cast<const ::arrow::Decimal256Type&>(*type).byte_width();

  constexpr int32_t seed = 0;

  ARROW_ASSIGN_OR_RAISE(auto out_buf, ::arrow::AllocateBuffer(size * byte_width));
  random_decimals<::arrow::Decimal256Type::kByteWidth>(size, seed, kDecimalPrecision,
                                                       out_buf->mutable_data());

  RETURN_NOT_OK(builder.AppendValues(out_buf->data(), size));
  return builder.Finish(out);
}

template <class ArrowType>
::arrow::enable_if_boolean<ArrowType, Status> NonNullArray(size_t size,
                                                           std::shared_ptr<Array>* out) {
  std::vector<uint8_t> values;
  ::arrow::randint(size, 0, 1, &values);
  ::arrow::BooleanBuilder builder;
  RETURN_NOT_OK(builder.AppendValues(values.data(), values.size()));
  return builder.Finish(out);
}

// This helper function only supports (size/2) nulls.
template <typename ArrowType>
::arrow::enable_if_floating_point<ArrowType, Status> NullableArray(
    size_t size, size_t num_nulls, uint32_t seed, std::shared_ptr<Array>* out) {
  using c_type = typename ArrowType::c_type;
  std::vector<c_type> values;
  ::arrow::random_real(size, seed, static_cast<c_type>(-1e10), static_cast<c_type>(1e10),
                       &values);
  std::vector<uint8_t> valid_bytes(size, 1);

  for (size_t i = 0; i < num_nulls; i++) {
    valid_bytes[i * 2] = 0;
  }

  ::arrow::NumericBuilder<ArrowType> builder;
  RETURN_NOT_OK(builder.AppendValues(values.data(), values.size(), valid_bytes.data()));
  return builder.Finish(out);
}

// This helper function only supports (size/2) nulls.
template <typename ArrowType>
::arrow::enable_if_integer<ArrowType, Status> NullableArray(size_t size, size_t num_nulls,
                                                            uint32_t seed,
                                                            std::shared_ptr<Array>* out) {
  std::vector<typename ArrowType::c_type> values;

  // Seed is random in Arrow right now
  (void)seed;
  ::arrow::randint(size, 0, 64, &values);
  std::vector<uint8_t> valid_bytes(size, 1);

  for (size_t i = 0; i < num_nulls; i++) {
    valid_bytes[i * 2] = 0;
  }

  // Passing data type so this will work with TimestampType too
  ::arrow::NumericBuilder<ArrowType> builder(std::make_shared<ArrowType>(),
                                             ::arrow::default_memory_pool());
  RETURN_NOT_OK(builder.AppendValues(values.data(), values.size(), valid_bytes.data()));
  return builder.Finish(out);
}

template <typename ArrowType>
::arrow::enable_if_date<ArrowType, Status> NullableArray(size_t size, size_t num_nulls,
                                                         uint32_t seed,
                                                         std::shared_ptr<Array>* out) {
  std::vector<typename ArrowType::c_type> values;

  // Seed is random in Arrow right now
  (void)seed;
  ::arrow::randint(size, 0, 24, &values);
  for (size_t i = 0; i < size; i++) {
    values[i] *= 86400000;
  }
  std::vector<uint8_t> valid_bytes(size, 1);

  for (size_t i = 0; i < num_nulls; i++) {
    valid_bytes[i * 2] = 0;
  }

  // Passing data type so this will work with TimestampType too
  ::arrow::NumericBuilder<ArrowType> builder(std::make_shared<ArrowType>(),
                                             ::arrow::default_memory_pool());
  RETURN_NOT_OK(builder.AppendValues(values.data(), values.size(), valid_bytes.data()));
  return builder.Finish(out);
}

// This helper function only supports (size/2) nulls yet.
template <typename ArrowType>
::arrow::enable_if_base_binary<ArrowType, Status> NullableArray(
    size_t size, size_t num_nulls, uint32_t seed, std::shared_ptr<::arrow::Array>* out) {
  std::vector<uint8_t> valid_bytes(size, 1);

  for (size_t i = 0; i < num_nulls; i++) {
    valid_bytes[i * 2] = 0;
  }

  using BuilderType = typename ::arrow::TypeTraits<ArrowType>::BuilderType;
  BuilderType builder;

  const int kBufferSize = 10;
  uint8_t buffer[kBufferSize];
  for (size_t i = 0; i < size; i++) {
    if (!valid_bytes[i]) {
      RETURN_NOT_OK(builder.AppendNull());
    } else {
      ::arrow::random_bytes(kBufferSize, seed + static_cast<uint32_t>(i), buffer);
      if (ArrowType::is_utf8) {
        // Trivially force data to be valid UTF8 by making it all ASCII
        for (auto& byte : buffer) {
          byte &= 0x7f;
        }
      }
      RETURN_NOT_OK(builder.Append(buffer, kBufferSize));
    }
  }
  return builder.Finish(out);
}

// This helper function only supports (size/2) nulls yet,
// same as NullableArray<String|Binary>(..)
template <typename ArrowType>
::arrow::enable_if_fixed_size_binary<ArrowType, Status> NullableArray(
    size_t size, size_t num_nulls, uint32_t seed, std::shared_ptr<::arrow::Array>* out) {
  std::vector<uint8_t> valid_bytes(size, 1);

  for (size_t i = 0; i < num_nulls; i++) {
    valid_bytes[i * 2] = 0;
  }

  using BuilderType = typename ::arrow::TypeTraits<ArrowType>::BuilderType;
  const int byte_width = 10;
  BuilderType builder(::arrow::fixed_size_binary(byte_width));

  const int kBufferSize = byte_width;
  uint8_t buffer[kBufferSize];
  for (size_t i = 0; i < size; i++) {
    if (!valid_bytes[i]) {
      RETURN_NOT_OK(builder.AppendNull());
    } else {
      ::arrow::random_bytes(kBufferSize, seed + static_cast<uint32_t>(i), buffer);
      RETURN_NOT_OK(builder.Append(buffer));
    }
  }
  return builder.Finish(out);
}

template <typename ArrowType, int32_t precision = ArrowType::precision>
::arrow::enable_if_t<
    std::is_same<ArrowType, DecimalWithPrecisionAndScale<precision>>::value, Status>
NullableArray(size_t size, size_t num_nulls, uint32_t seed,
              std::shared_ptr<::arrow::Array>* out) {
  std::vector<uint8_t> valid_bytes(size, '\1');

  for (size_t i = 0; i < num_nulls; ++i) {
    valid_bytes[i * 2] = '\0';
  }

  constexpr int32_t kDecimalPrecision = precision;
  constexpr int32_t kDecimalScale = DecimalWithPrecisionAndScale<precision>::scale;
  const auto type = ::arrow::decimal(kDecimalPrecision, kDecimalScale);
  const int32_t byte_width =
      static_cast<const ::arrow::Decimal128Type&>(*type).byte_width();

  ARROW_ASSIGN_OR_RAISE(auto out_buf, ::arrow::AllocateBuffer(size * byte_width));

  random_decimals<::arrow::Decimal128Type::kByteWidth>(size, seed, precision,
                                                       out_buf->mutable_data());

  ::arrow::Decimal128Builder builder(type);
  RETURN_NOT_OK(builder.AppendValues(out_buf->data(), size, valid_bytes.data()));
  return builder.Finish(out);
}

template <typename ArrowType, int32_t precision = ArrowType::precision>
::arrow::enable_if_t<
    std::is_same<ArrowType, Decimal256WithPrecisionAndScale<precision>>::value, Status>
NullableArray(size_t size, size_t num_nulls, uint32_t seed,
              std::shared_ptr<::arrow::Array>* out) {
  std::vector<uint8_t> valid_bytes(size, '\1');

  for (size_t i = 0; i < num_nulls; ++i) {
    valid_bytes[i * 2] = '\0';
  }

  constexpr int32_t kDecimalPrecision = precision;
  constexpr int32_t kDecimalScale = Decimal256WithPrecisionAndScale<precision>::scale;
  const auto type = ::arrow::decimal256(kDecimalPrecision, kDecimalScale);
  const int32_t byte_width =
      static_cast<const ::arrow::Decimal256Type&>(*type).byte_width();

  ARROW_ASSIGN_OR_RAISE(auto out_buf, ::arrow::AllocateBuffer(size * byte_width));

  random_decimals<::arrow::Decimal256Type::kByteWidth>(size, seed, precision,
                                                       out_buf->mutable_data());

  ::arrow::Decimal256Builder builder(type);
  RETURN_NOT_OK(builder.AppendValues(out_buf->data(), size, valid_bytes.data()));
  return builder.Finish(out);
}

// This helper function only supports (size/2) nulls yet.
template <class ArrowType>
::arrow::enable_if_boolean<ArrowType, Status> NullableArray(size_t size, size_t num_nulls,
                                                            uint32_t seed,
                                                            std::shared_ptr<Array>* out) {
  std::vector<uint8_t> values;

  // Seed is random in Arrow right now
  (void)seed;

  ::arrow::randint(size, 0, 1, &values);
  std::vector<uint8_t> valid_bytes(size, 1);

  for (size_t i = 0; i < num_nulls; i++) {
    valid_bytes[i * 2] = 0;
  }

  ::arrow::BooleanBuilder builder;
  RETURN_NOT_OK(builder.AppendValues(values.data(), values.size(), valid_bytes.data()));
  return builder.Finish(out);
}

/// Wrap an Array into a ListArray by splitting it up into size lists.
///
/// This helper function only supports (size/2) nulls.
Status MakeListArray(const std::shared_ptr<Array>& values, int64_t size,
                     int64_t null_count, const std::string& item_name,
                     bool nullable_values, std::shared_ptr<::arrow::ListArray>* out) {
  // We always include an empty list
  int64_t non_null_entries = size - null_count - 1;
  int64_t length_per_entry = values->length() / non_null_entries;

  auto offsets = AllocateBuffer();
  RETURN_NOT_OK(offsets->Resize((size + 1) * sizeof(int32_t)));
  int32_t* offsets_ptr = reinterpret_cast<int32_t*>(offsets->mutable_data());

  auto null_bitmap = AllocateBuffer();
  int64_t bitmap_size = ::arrow::bit_util::BytesForBits(size);
  RETURN_NOT_OK(null_bitmap->Resize(bitmap_size));
  uint8_t* null_bitmap_ptr = null_bitmap->mutable_data();
  memset(null_bitmap_ptr, 0, bitmap_size);

  int32_t current_offset = 0;
  for (int64_t i = 0; i < size; i++) {
    offsets_ptr[i] = current_offset;
    if (!(((i % 2) == 0) && ((i / 2) < null_count))) {
      // Non-null list (list with index 1 is always empty).
      ::arrow::bit_util::SetBit(null_bitmap_ptr, i);
      if (i != 1) {
        current_offset += static_cast<int32_t>(length_per_entry);
      }
    }
  }
  offsets_ptr[size] = static_cast<int32_t>(values->length());

  auto value_field = ::arrow::field(item_name, values->type(), nullable_values);
  *out = std::make_shared<::arrow::ListArray>(::arrow::list(value_field), size, offsets,
                                              values, null_bitmap, null_count);

  return Status::OK();
}

// Make an array containing only empty lists, with a null values array
Status MakeEmptyListsArray(int64_t size, std::shared_ptr<Array>* out_array) {
  // Allocate an offsets buffer containing only zeroes
  const int64_t offsets_nbytes = (size + 1) * sizeof(int32_t);
  ARROW_ASSIGN_OR_RAISE(auto offsets_buffer, ::arrow::AllocateBuffer(offsets_nbytes));
  memset(offsets_buffer->mutable_data(), 0, offsets_nbytes);

  auto value_field =
      ::arrow::field("item", ::arrow::float64(), false /* nullable_values */);
  auto list_type = ::arrow::list(value_field);

  std::vector<std::shared_ptr<Buffer>> child_buffers = {nullptr /* null bitmap */,
                                                        nullptr /* values */};
  auto child_data =
      ::arrow::ArrayData::Make(value_field->type(), 0, std::move(child_buffers));

  std::vector<std::shared_ptr<Buffer>> buffers = {nullptr /* bitmap */,
                                                  std::move(offsets_buffer)};
  auto array_data = ::arrow::ArrayData::Make(list_type, size, std::move(buffers));
  array_data->child_data.push_back(child_data);

  *out_array = ::arrow::MakeArray(array_data);
  return Status::OK();
}

std::shared_ptr<::arrow::Table> MakeSimpleTable(
    const std::shared_ptr<ChunkedArray>& values, bool nullable) {
  auto schema = ::arrow::schema({::arrow::field("col", values->type(), nullable)});
  return ::arrow::Table::Make(schema, {values});
}

std::shared_ptr<::arrow::Table> MakeSimpleTable(const std::shared_ptr<Array>& values,
                                                bool nullable) {
  auto carr = std::make_shared<::arrow::ChunkedArray>(values);
  return MakeSimpleTable(carr, nullable);
}

template <typename T>
void ExpectArray(T* expected, Array* result) {
  auto p_array = static_cast<::arrow::PrimitiveArray*>(result);
  for (int i = 0; i < result->length(); i++) {
    EXPECT_EQ(expected[i], reinterpret_cast<const T*>(p_array->values()->data())[i]);
  }
}

template <typename ArrowType>
void ExpectArrayT(void* expected, Array* result) {
  ::arrow::PrimitiveArray* p_array = static_cast<::arrow::PrimitiveArray*>(result);
  for (int64_t i = 0; i < result->length(); i++) {
    EXPECT_EQ(reinterpret_cast<typename ArrowType::c_type*>(expected)[i],
              reinterpret_cast<const typename ArrowType::c_type*>(
                  p_array->values()->data())[i]);
  }
}

template <>
void ExpectArrayT<::arrow::BooleanType>(void* expected, Array* result) {
  ::arrow::BooleanBuilder builder;
  ARROW_EXPECT_OK(
      builder.AppendValues(reinterpret_cast<uint8_t*>(expected), result->length()));

  std::shared_ptr<Array> expected_array;
  ARROW_EXPECT_OK(builder.Finish(&expected_array));
  EXPECT_TRUE(result->Equals(*expected_array));
}

}  // namespace arrow

}  // namespace parquet

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

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "arrow/testing/uniform_real.h"
#include "arrow/testing/visibility.h"
#include "arrow/type.h"

namespace arrow {

class Array;

namespace random {

using SeedType = int32_t;
constexpr SeedType kSeedMax = std::numeric_limits<SeedType>::max();

class ARROW_TESTING_EXPORT RandomArrayGenerator {
 public:
  explicit RandomArrayGenerator(SeedType seed)
      : seed_distribution_(static_cast<SeedType>(1), kSeedMax), seed_rng_(seed) {}

  /// \brief Generate a null bitmap
  ///
  /// \param[in] size the size of the bitmap to generate
  /// \param[in] null_probability the probability of a bit being zero
  /// \param[in] alignment alignment for memory allocations (in bytes)
  /// \param[in] memory_pool memory pool to allocate memory from
  ///
  /// \return a generated Buffer
  std::shared_ptr<Buffer> NullBitmap(int64_t size, double null_probability = 0,
                                     int64_t alignment = kDefaultBufferAlignment,
                                     MemoryPool* memory_pool = default_memory_pool());

  /// \brief Generate a random BooleanArray
  ///
  /// \param[in] size the size of the array to generate
  /// \param[in] true_probability the probability of a value being 1 / bit-set
  /// \param[in] null_probability the probability of a value being null
  /// \param[in] alignment alignment for memory allocations (in bytes)
  /// \param[in] memory_pool memory pool to allocate memory from
  ///
  /// \return a generated Array
  std::shared_ptr<Array> Boolean(int64_t size, double true_probability,
                                 double null_probability = 0,
                                 int64_t alignment = kDefaultBufferAlignment,
                                 MemoryPool* memory_pool = default_memory_pool());
  /// \brief Generate a random UInt8Array
  ///
  /// \param[in] size the size of the array to generate
  /// \param[in] min the lower bound of the uniform distribution
  /// \param[in] max the upper bound of the uniform distribution
  /// \param[in] null_probability the probability of a value being null
  /// \param[in] alignment alignment for memory allocations (in bytes)
  /// \param[in] memory_pool memory pool to allocate memory from
  ///
  /// \return a generated Array
  std::shared_ptr<Array> UInt8(int64_t size, uint8_t min, uint8_t max,
                               double null_probability = 0,
                               int64_t alignment = kDefaultBufferAlignment,
                               MemoryPool* memory_pool = default_memory_pool());

  /// \brief Generate a random Int8Array
  ///
  /// \param[in] size the size of the array to generate
  /// \param[in] min the lower bound of the uniform distribution
  /// \param[in] max the upper bound of the uniform distribution
  /// \param[in] null_probability the probability of a value being null
  /// \param[in] alignment alignment for memory allocations (in bytes)
  /// \param[in] memory_pool memory pool to allocate memory from
  ///
  /// \return a generated Array
  std::shared_ptr<Array> Int8(int64_t size, int8_t min, int8_t max,
                              double null_probability = 0,
                              int64_t alignment = kDefaultBufferAlignment,
                              MemoryPool* memory_pool = default_memory_pool());

  /// \brief Generate a random UInt16Array
  ///
  /// \param[in] size the size of the array to generate
  /// \param[in] min the lower bound of the uniform distribution
  /// \param[in] max the upper bound of the uniform distribution
  /// \param[in] null_probability the probability of a value being null
  /// \param[in] alignment alignment for memory allocations (in bytes)
  /// \param[in] memory_pool memory pool to allocate memory from
  ///
  /// \return a generated Array
  std::shared_ptr<Array> UInt16(int64_t size, uint16_t min, uint16_t max,
                                double null_probability = 0,
                                int64_t alignment = kDefaultBufferAlignment,
                                MemoryPool* memory_pool = default_memory_pool());

  /// \brief Generate a random Int16Array
  ///
  /// \param[in] size the size of the array to generate
  /// \param[in] min the lower bound of the uniform distribution
  /// \param[in] max the upper bound of the uniform distribution
  /// \param[in] null_probability the probability of a value being null
  /// \param[in] alignment alignment for memory allocations (in bytes)
  /// \param[in] memory_pool memory pool to allocate memory from
  ///
  /// \return a generated Array
  std::shared_ptr<Array> Int16(int64_t size, int16_t min, int16_t max,
                               double null_probability = 0,
                               int64_t alignment = kDefaultBufferAlignment,
                               MemoryPool* memory_pool = default_memory_pool());

  /// \brief Generate a random UInt32Array
  ///
  /// \param[in] size the size of the array to generate
  /// \param[in] min the lower bound of the uniform distribution
  /// \param[in] max the upper bound of the uniform distribution
  /// \param[in] null_probability the probability of a value being null
  /// \param[in] alignment alignment for memory allocations (in bytes)
  /// \param[in] memory_pool memory pool to allocate memory from
  ///
  /// \return a generated Array
  std::shared_ptr<Array> UInt32(int64_t size, uint32_t min, uint32_t max,
                                double null_probability = 0,
                                int64_t alignment = kDefaultBufferAlignment,
                                MemoryPool* memory_pool = default_memory_pool());

  /// \brief Generate a random Int32Array
  ///
  /// \param[in] size the size of the array to generate
  /// \param[in] min the lower bound of the uniform distribution
  /// \param[in] max the upper bound of the uniform distribution
  /// \param[in] null_probability the probability of a value being null
  /// \param[in] alignment alignment for memory allocations (in bytes)
  /// \param[in] memory_pool memory pool to allocate memory from
  ///
  /// \return a generated Array
  std::shared_ptr<Array> Int32(int64_t size, int32_t min, int32_t max,
                               double null_probability = 0,
                               int64_t alignment = kDefaultBufferAlignment,
                               MemoryPool* memory_pool = default_memory_pool());

  /// \brief Generate a random UInt64Array
  ///
  /// \param[in] size the size of the array to generate
  /// \param[in] min the lower bound of the uniform distribution
  /// \param[in] max the upper bound of the uniform distribution
  /// \param[in] null_probability the probability of a value being null
  /// \param[in] alignment alignment for memory allocations (in bytes)
  /// \param[in] memory_pool memory pool to allocate memory from
  ///
  /// \return a generated Array
  std::shared_ptr<Array> UInt64(int64_t size, uint64_t min, uint64_t max,
                                double null_probability = 0,
                                int64_t alignment = kDefaultBufferAlignment,
                                MemoryPool* memory_pool = default_memory_pool());

  /// \brief Generate a random Int64Array
  ///
  /// \param[in] size the size of the array to generate
  /// \param[in] min the lower bound of the uniform distribution
  /// \param[in] max the upper bound of the uniform distribution
  /// \param[in] null_probability the probability of a value being null
  /// \param[in] alignment alignment for memory allocations (in bytes)
  /// \param[in] memory_pool memory pool to allocate memory from
  ///
  /// \return a generated Array
  std::shared_ptr<Array> Int64(int64_t size, int64_t min, int64_t max,
                               double null_probability = 0,
                               int64_t alignment = kDefaultBufferAlignment,
                               MemoryPool* memory_pool = default_memory_pool());

  /// \brief Generate a random HalfFloatArray
  ///
  /// \param[in] size the size of the array to generate
  /// \param[in] min the lower bound of the distribution
  /// \param[in] max the upper bound of the distribution
  /// \param[in] null_probability the probability of a value being null
  /// \param[in] alignment alignment for memory allocations (in bytes)
  /// \param[in] memory_pool memory pool to allocate memory from
  ///
  /// \return a generated Array
  std::shared_ptr<Array> Float16(int64_t size, int16_t min, int16_t max,
                                 double null_probability = 0,
                                 int64_t alignment = kDefaultBufferAlignment,
                                 MemoryPool* memory_pool = default_memory_pool());

  /// \brief Generate a random FloatArray
  ///
  /// \param[in] size the size of the array to generate
  /// \param[in] min the lower bound of the uniform distribution
  /// \param[in] max the upper bound of the uniform distribution
  /// \param[in] null_probability the probability of a value being null
  /// \param[in] nan_probability the probability of a value being NaN
  /// \param[in] alignment alignment for memory allocations (in bytes)
  /// \param[in] memory_pool memory pool to allocate memory from
  ///
  /// \return a generated Array
  std::shared_ptr<Array> Float32(int64_t size, float min, float max,
                                 double null_probability = 0, double nan_probability = 0,
                                 int64_t alignment = kDefaultBufferAlignment,
                                 MemoryPool* memory_pool = default_memory_pool());

  /// \brief Generate a random DoubleArray
  ///
  /// \param[in] size the size of the array to generate
  /// \param[in] min the lower bound of the uniform distribution
  /// \param[in] max the upper bound of the uniform distribution
  /// \param[in] null_probability the probability of a value being null
  /// \param[in] nan_probability the probability of a value being NaN
  /// \param[in] alignment alignment for memory allocations (in bytes)
  /// \param[in] memory_pool memory pool to allocate memory from
  ///
  /// \return a generated Array
  std::shared_ptr<Array> Float64(int64_t size, double min, double max,
                                 double null_probability = 0, double nan_probability = 0,
                                 int64_t alignment = kDefaultBufferAlignment,
                                 MemoryPool* memory_pool = default_memory_pool());

  /// \brief Generate a random Date64Array
  ///
  /// \param[in] size the size of the array to generate
  /// \param[in] min the lower bound of the uniform distribution
  /// \param[in] max the upper bound of the uniform distribution
  /// \param[in] null_probability the probability of a value being null
  /// \param[in] alignment alignment for memory allocations (in bytes)
  /// \param[in] memory_pool memory pool to allocate memory from
  ///
  /// \return a generated Array
  std::shared_ptr<Array> Date64(int64_t size, int64_t min, int64_t max,
                                double null_probability = 0,
                                int64_t alignment = kDefaultBufferAlignment,
                                MemoryPool* memory_pool = default_memory_pool());

  template <typename ArrowType, typename CType = typename ArrowType::c_type>
  std::shared_ptr<Array> Numeric(int64_t size, CType min, CType max,
                                 double null_probability = 0,
                                 int64_t alignment = kDefaultBufferAlignment,
                                 MemoryPool* memory_pool = default_memory_pool()) {
    switch (ArrowType::type_id) {
      case Type::UINT8:
        return UInt8(size, static_cast<uint8_t>(min), static_cast<uint8_t>(max),
                     null_probability, alignment, memory_pool);
      case Type::INT8:
        return Int8(size, static_cast<int8_t>(min), static_cast<int8_t>(max),
                    null_probability, alignment, memory_pool);
      case Type::UINT16:
        return UInt16(size, static_cast<uint16_t>(min), static_cast<uint16_t>(max),
                      null_probability, alignment, memory_pool);
      case Type::INT16:
        return Int16(size, static_cast<int16_t>(min), static_cast<int16_t>(max),
                     null_probability, alignment, memory_pool);
      case Type::UINT32:
        return UInt32(size, static_cast<uint32_t>(min), static_cast<uint32_t>(max),
                      null_probability, alignment, memory_pool);
      case Type::INT32:
        return Int32(size, static_cast<int32_t>(min), static_cast<int32_t>(max),
                     null_probability, alignment, memory_pool);
      case Type::UINT64:
        return UInt64(size, static_cast<uint64_t>(min), static_cast<uint64_t>(max),
                      null_probability, alignment, memory_pool);
      case Type::INT64:
        return Int64(size, static_cast<int64_t>(min), static_cast<int64_t>(max),
                     null_probability, alignment, memory_pool);
      case Type::HALF_FLOAT:
        return Float16(size, static_cast<int16_t>(min), static_cast<int16_t>(max),
                       null_probability, alignment, memory_pool);
      case Type::FLOAT:
        return Float32(size, static_cast<float>(min), static_cast<float>(max),
                       null_probability, /*nan_probability=*/0, alignment, memory_pool);
      case Type::DOUBLE:
        return Float64(size, static_cast<double>(min), static_cast<double>(max),
                       null_probability, /*nan_probability=*/0, alignment, memory_pool);
      case Type::DATE64:
        return Date64(size, static_cast<int64_t>(min), static_cast<int64_t>(max),
                      null_probability, alignment, memory_pool);
      default:
        return nullptr;
    }
  }

  /// \brief Generate a random Decimal128Array
  ///
  /// \param[in] type the type of the array to generate
  ///            (must be an instance of Decimal128Type)
  /// \param[in] size the size of the array to generate
  /// \param[in] null_probability the probability of a value being null
  /// \param[in] alignment alignment for memory allocations (in bytes)
  /// \param[in] memory_pool memory pool to allocate memory from
  ///
  /// \return a generated Array
  std::shared_ptr<Array> Decimal128(std::shared_ptr<DataType> type, int64_t size,
                                    double null_probability = 0,
                                    int64_t alignment = kDefaultBufferAlignment,
                                    MemoryPool* memory_pool = default_memory_pool());

  /// \brief Generate a random Decimal256Array
  ///
  /// \param[in] type the type of the array to generate
  ///            (must be an instance of Decimal256Type)
  /// \param[in] size the size of the array to generate
  /// \param[in] null_probability the probability of a value being null
  /// \param[in] alignment alignment for memory allocations (in bytes)
  /// \param[in] memory_pool memory pool to allocate memory from
  ///
  /// \return a generated Array
  std::shared_ptr<Array> Decimal256(std::shared_ptr<DataType> type, int64_t size,
                                    double null_probability = 0,
                                    int64_t alignment = kDefaultBufferAlignment,
                                    MemoryPool* memory_pool = default_memory_pool());

  /// \brief Generate an array of offsets (for use in e.g. ListArray::FromArrays)
  ///
  /// \param[in] size the size of the array to generate
  /// \param[in] first_offset the first offset value (usually 0)
  /// \param[in] last_offset the last offset value (usually the size of the child array)
  /// \param[in] null_probability the probability of an offset being null
  /// \param[in] force_empty_nulls if true, null offsets must have 0 "length"
  /// \param[in] alignment alignment for memory allocations (in bytes)
  /// \param[in] memory_pool memory pool to allocate memory from
  ///
  /// \return a generated Array
  std::shared_ptr<Array> Offsets(int64_t size, int32_t first_offset, int32_t last_offset,
                                 double null_probability = 0,
                                 bool force_empty_nulls = false,
                                 int64_t alignment = kDefaultBufferAlignment,
                                 MemoryPool* memory_pool = default_memory_pool());

  std::shared_ptr<Array> LargeOffsets(int64_t size, int64_t first_offset,
                                      int64_t last_offset, double null_probability = 0,
                                      bool force_empty_nulls = false,
                                      int64_t alignment = kDefaultBufferAlignment,
                                      MemoryPool* memory_pool = default_memory_pool());

  /// \brief Generate a random StringArray
  ///
  /// \param[in] size the size of the array to generate
  /// \param[in] min_length the lower bound of the string length
  ///            determined by the uniform distribution
  /// \param[in] max_length the upper bound of the string length
  ///            determined by the uniform distribution
  /// \param[in] null_probability the probability of a value being null
  /// \param[in] alignment alignment for memory allocations (in bytes)
  /// \param[in] memory_pool memory pool to allocate memory from
  ///
  /// \return a generated Array
  std::shared_ptr<Array> String(int64_t size, int32_t min_length, int32_t max_length,
                                double null_probability = 0,
                                int64_t alignment = kDefaultBufferAlignment,
                                MemoryPool* memory_pool = default_memory_pool());

  /// \brief Generate a random LargeStringArray
  ///
  /// \param[in] size the size of the array to generate
  /// \param[in] min_length the lower bound of the string length
  ///            determined by the uniform distribution
  /// \param[in] max_length the upper bound of the string length
  ///            determined by the uniform distribution
  /// \param[in] null_probability the probability of a value being null
  /// \param[in] alignment alignment for memory allocations (in bytes)
  /// \param[in] memory_pool memory pool to allocate memory from
  ///
  /// \return a generated Array
  std::shared_ptr<Array> LargeString(int64_t size, int32_t min_length, int32_t max_length,
                                     double null_probability = 0,
                                     int64_t alignment = kDefaultBufferAlignment,
                                     MemoryPool* memory_pool = default_memory_pool());

  /// \brief Generate a random StringArray with repeated values
  ///
  /// \param[in] size the size of the array to generate
  /// \param[in] unique the number of unique string values used
  ///            to populate the array
  /// \param[in] min_length the lower bound of the string length
  ///            determined by the uniform distribution
  /// \param[in] max_length the upper bound of the string length
  ///            determined by the uniform distribution
  /// \param[in] null_probability the probability of a value being null
  /// \param[in] alignment alignment for memory allocations (in bytes)
  /// \param[in] memory_pool memory pool to allocate memory from
  ///
  /// \return a generated Array
  std::shared_ptr<Array> StringWithRepeats(
      int64_t size, int64_t unique, int32_t min_length, int32_t max_length,
      double null_probability = 0, int64_t alignment = kDefaultBufferAlignment,
      MemoryPool* memory_pool = default_memory_pool());

  /// \brief Like StringWithRepeats but return BinaryArray
  std::shared_ptr<Array> BinaryWithRepeats(
      int64_t size, int64_t unique, int32_t min_length, int32_t max_length,
      double null_probability = 0, int64_t alignment = kDefaultBufferAlignment,
      MemoryPool* memory_pool = default_memory_pool());

  /// \brief Generate a random FixedSizeBinaryArray
  ///
  /// \param[in] size the size of the array to generate
  /// \param[in] byte_width the byte width of fixed-size binary items
  /// \param[in] null_probability the probability of a value being null
  /// \param[in] alignment alignment for memory allocations (in bytes)
  /// \param[in] memory_pool memory pool to allocate memory from
  ///
  /// \return a generated Array
  std::shared_ptr<Array> FixedSizeBinary(int64_t size, int32_t byte_width,
                                         double null_probability = 0,
                                         int64_t alignment = kDefaultBufferAlignment,
                                         MemoryPool* memory_pool = default_memory_pool());

  /// \brief Generate a random ListArray
  ///
  /// \param[in] values The underlying values array
  /// \param[in] size The size of the generated list array
  /// \param[in] null_probability the probability of a list value being null
  /// \param[in] force_empty_nulls if true, null list entries must have 0 length
  /// \param[in] alignment alignment for memory allocations (in bytes)
  /// \param[in] memory_pool memory pool to allocate memory from
  ///
  /// \return a generated Array
  std::shared_ptr<Array> List(const Array& values, int64_t size,
                              double null_probability = 0, bool force_empty_nulls = false,
                              int64_t alignment = kDefaultBufferAlignment,
                              MemoryPool* memory_pool = default_memory_pool());

  /// \brief Generate a random MapArray
  ///
  /// \param[in] keys The underlying keys array
  /// \param[in] items The underlying items array
  /// \param[in] size The size of the generated map array
  /// \param[in] null_probability the probability of a map value being null
  /// \param[in] force_empty_nulls if true, null map entries must have 0 length
  /// \param[in] alignment alignment for memory allocations (in bytes)
  /// \param[in] memory_pool memory pool to allocate memory from
  ///
  /// \return a generated Array
  std::shared_ptr<Array> Map(const std::shared_ptr<Array>& keys,
                             const std::shared_ptr<Array>& items, int64_t size,
                             double null_probability = 0, bool force_empty_nulls = false,
                             int64_t alignment = kDefaultBufferAlignment,
                             MemoryPool* memory_pool = default_memory_pool());

  /// \brief Generate a random SparseUnionArray
  ///
  /// The type ids are chosen randomly, according to a uniform distribution,
  /// amongst the given child fields.
  ///
  /// \param[in] fields Vector of Arrays containing the data for each union field
  /// \param[in] size The size of the generated sparse union array
  /// \param[in] alignment alignment for memory allocations (in bytes)
  /// \param[in] memory_pool memory pool to allocate memory from
  std::shared_ptr<Array> SparseUnion(const ArrayVector& fields, int64_t size,
                                     int64_t alignment = kDefaultBufferAlignment,
                                     MemoryPool* memory_pool = default_memory_pool());

  /// \brief Generate a random DenseUnionArray
  ///
  /// The type ids are chosen randomly, according to a uniform distribution,
  /// amongst the given child fields.  The offsets are incremented along
  /// each child field.
  ///
  /// \param[in] fields Vector of Arrays containing the data for each union field
  /// \param[in] size The size of the generated sparse union array
  /// \param[in] alignment alignment for memory allocations (in bytes)
  /// \param[in] memory_pool memory pool to allocate memory from
  std::shared_ptr<Array> DenseUnion(const ArrayVector& fields, int64_t size,
                                    int64_t alignment = kDefaultBufferAlignment,
                                    MemoryPool* memory_pool = default_memory_pool());

  /// \brief Generate a random Array of the specified type, size, and null_probability.
  ///
  /// Generation parameters other than size and null_probability are determined based on
  /// the type of Array to be generated.
  /// If boolean the probabilities of true,false values are 0.25,0.75 respectively.
  /// If numeric min,max will be the least and greatest representable values.
  /// If string min_length,max_length will be 0,sqrt(size) respectively.
  ///
  /// \param[in] type the type of Array to generate
  /// \param[in] size the size of the Array to generate
  /// \param[in] null_probability the probability of a slot being null
  /// \param[in] alignment alignment for memory allocations (in bytes)
  /// \param[in] memory_pool memory pool to allocate memory from
  /// \return a generated Array
  std::shared_ptr<Array> ArrayOf(std::shared_ptr<DataType> type, int64_t size,
                                 double null_probability = 0,
                                 int64_t alignment = kDefaultBufferAlignment,
                                 MemoryPool* memory_pool = default_memory_pool());

  /// \brief Generate an array with random data based on the given field. See BatchOf
  /// for usage info.
  std::shared_ptr<Array> ArrayOf(const Field& field, int64_t size,
                                 int64_t alignment = kDefaultBufferAlignment,
                                 MemoryPool* memory_pool = default_memory_pool());

  /// \brief Generate a record batch with random data of the specified length.
  ///
  /// Generation options are read from key-value metadata for each field, and may be
  /// specified at any nesting level. For example, generation options for the child
  /// values of a list array can be specified by constructing the list type with
  /// list(field("item", int8(), options_metadata))
  ///
  /// The following options are supported:
  ///
  /// For all types except NullType:
  /// - null_probability (double): range [0.0, 1.0] the probability of a null value.
  /// Default/value is 0.0 if the field is marked non-nullable, else it is 0.01
  ///
  /// For all numeric types T:
  /// - min (T::c_type): the minimum value to generate (inclusive), default
  ///   std::numeric_limits<T::c_type>::min()
  /// - max (T::c_type): the maximum value to generate (inclusive), default
  ///   std::numeric_limits<T::c_type>::max()
  /// Note this means that, for example, min/max are int16_t values for HalfFloatType.
  ///
  /// For floating point types T for which is_physical_floating_type<T>:
  /// - nan_probability (double): range [0.0, 1.0] the probability of a NaN value.
  ///
  /// For BooleanType:
  /// - true_probability (double): range [0.0, 1.0] the probability of a true.
  ///
  /// For DictionaryType:
  /// - values (int32_t): the size of the dictionary.
  /// Other properties are passed to the generator for the dictionary indices. However,
  /// min and max cannot be specified. Note it is not possible to otherwise customize
  /// the generation of dictionary values.
  ///
  /// For list, string, and binary types T, including their large variants:
  /// - min_length (T::offset_type): the minimum length of the child to generate,
  ///   default 0
  /// - max_length (T::offset_type): the minimum length of the child to generate,
  ///   default 1024
  ///
  /// For string and binary types T (not including their large variants):
  /// - unique (int32_t): if positive, this many distinct values will be generated
  ///   and all array values will be one of these values, default -1
  ///
  /// For MapType:
  /// - values (int32_t): the number of key-value pairs to generate, which will be
  ///   partitioned among the array values.
  std::shared_ptr<arrow::RecordBatch> BatchOf(
      const FieldVector& fields, int64_t size,
      int64_t alignment = kDefaultBufferAlignment,
      MemoryPool* memory_pool = default_memory_pool());

  SeedType seed() { return seed_distribution_(seed_rng_); }

 private:
  std::uniform_int_distribution<SeedType> seed_distribution_;
  std::default_random_engine seed_rng_;
};

/// Generate an array with random data. See RandomArrayGenerator::BatchOf.
ARROW_TESTING_EXPORT
std::shared_ptr<arrow::RecordBatch> GenerateBatch(
    const FieldVector& fields, int64_t size, SeedType seed,
    int64_t alignment = kDefaultBufferAlignment,
    MemoryPool* memory_pool = default_memory_pool());

/// Generate an array with random data. See RandomArrayGenerator::BatchOf.
ARROW_TESTING_EXPORT
std::shared_ptr<arrow::Array> GenerateArray(
    const Field& field, int64_t size, SeedType seed,
    int64_t alignment = kDefaultBufferAlignment,
    MemoryPool* memory_pool = default_memory_pool());

}  // namespace random

//
// Assorted functions
//

ARROW_TESTING_EXPORT
void rand_day_millis(int64_t N, std::vector<DayTimeIntervalType::DayMilliseconds>* out);
ARROW_TESTING_EXPORT
void rand_month_day_nanos(int64_t N,
                          std::vector<MonthDayNanoIntervalType::MonthDayNanos>* out);

template <typename T, typename U>
void randint(int64_t N, T lower, T upper, std::vector<U>* out) {
  const int random_seed = 0;
  std::default_random_engine gen(random_seed);
  std::uniform_int_distribution<T> d(lower, upper);
  out->resize(N, static_cast<T>(0));
  std::generate(out->begin(), out->end(), [&d, &gen] { return static_cast<U>(d(gen)); });
}

template <typename T, typename U>
void random_real(int64_t n, uint32_t seed, T min_value, T max_value,
                 std::vector<U>* out) {
  std::default_random_engine gen(seed);
  ::arrow::random::uniform_real_distribution<T> d(min_value, max_value);
  out->resize(n, static_cast<T>(0));
  std::generate(out->begin(), out->end(), [&d, &gen] { return static_cast<U>(d(gen)); });
}

template <typename T, typename U>
void rand_uniform_int(int64_t n, uint32_t seed, T min_value, T max_value, U* out) {
  assert(out || (n == 0));
  std::default_random_engine gen(seed);
  std::uniform_int_distribution<T> d(min_value, max_value);
  std::generate(out, out + n, [&d, &gen] { return static_cast<U>(d(gen)); });
}

}  // namespace arrow

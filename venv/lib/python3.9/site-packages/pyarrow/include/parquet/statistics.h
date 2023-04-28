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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "parquet/platform.h"
#include "parquet/types.h"

namespace arrow {

class Array;
class BinaryArray;

}  // namespace arrow

namespace parquet {

class ColumnDescriptor;

// ----------------------------------------------------------------------
// Value comparator interfaces

/// \brief Base class for value comparators. Generally used with
/// TypedComparator<T>
class PARQUET_EXPORT Comparator {
 public:
  virtual ~Comparator() {}

  /// \brief Create a comparator explicitly from physical type and
  /// sort order
  /// \param[in] physical_type the physical type for the typed
  /// comparator
  /// \param[in] sort_order either SortOrder::SIGNED or
  /// SortOrder::UNSIGNED
  /// \param[in] type_length for FIXED_LEN_BYTE_ARRAY only
  static std::shared_ptr<Comparator> Make(Type::type physical_type,
                                          SortOrder::type sort_order,
                                          int type_length = -1);

  /// \brief Create typed comparator inferring default sort order from
  /// ColumnDescriptor
  /// \param[in] descr the Parquet column schema
  static std::shared_ptr<Comparator> Make(const ColumnDescriptor* descr);
};

/// \brief Interface for comparison of physical types according to the
/// semantics of a particular logical type.
template <typename DType>
class TypedComparator : public Comparator {
 public:
  using T = typename DType::c_type;

  /// \brief Scalar comparison of two elements, return true if first
  /// is strictly less than the second
  virtual bool Compare(const T& a, const T& b) = 0;

  /// \brief Compute maximum and minimum elements in a batch of
  /// elements without any nulls
  virtual std::pair<T, T> GetMinMax(const T* values, int64_t length) = 0;

  /// \brief Compute minimum and maximum elements from an Arrow array. Only
  /// valid for certain Parquet Type / Arrow Type combinations, like BYTE_ARRAY
  /// / arrow::BinaryArray
  virtual std::pair<T, T> GetMinMax(const ::arrow::Array& values) = 0;

  /// \brief Compute maximum and minimum elements in a batch of
  /// elements with accompanying bitmap indicating which elements are
  /// included (bit set) and excluded (bit not set)
  ///
  /// \param[in] values the sequence of values
  /// \param[in] length the length of the sequence
  /// \param[in] valid_bits a bitmap indicating which elements are
  /// included (1) or excluded (0)
  /// \param[in] valid_bits_offset the bit offset into the bitmap of
  /// the first element in the sequence
  virtual std::pair<T, T> GetMinMaxSpaced(const T* values, int64_t length,
                                          const uint8_t* valid_bits,
                                          int64_t valid_bits_offset) = 0;
};

/// \brief Typed version of Comparator::Make
template <typename DType>
std::shared_ptr<TypedComparator<DType>> MakeComparator(Type::type physical_type,
                                                       SortOrder::type sort_order,
                                                       int type_length = -1) {
  return std::static_pointer_cast<TypedComparator<DType>>(
      Comparator::Make(physical_type, sort_order, type_length));
}

/// \brief Typed version of Comparator::Make
template <typename DType>
std::shared_ptr<TypedComparator<DType>> MakeComparator(const ColumnDescriptor* descr) {
  return std::static_pointer_cast<TypedComparator<DType>>(Comparator::Make(descr));
}

// ----------------------------------------------------------------------

/// \brief Structure represented encoded statistics to be written to
/// and from Parquet serialized metadata
class PARQUET_EXPORT EncodedStatistics {
  std::shared_ptr<std::string> max_, min_;
  bool is_signed_ = false;

 public:
  EncodedStatistics()
      : max_(std::make_shared<std::string>()), min_(std::make_shared<std::string>()) {}

  const std::string& max() const { return *max_; }
  const std::string& min() const { return *min_; }

  int64_t null_count = 0;
  int64_t distinct_count = 0;

  bool has_min = false;
  bool has_max = false;
  bool has_null_count = false;
  bool has_distinct_count = false;

  // From parquet-mr
  // Don't write stats larger than the max size rather than truncating. The
  // rationale is that some engines may use the minimum value in the page as
  // the true minimum for aggregations and there is no way to mark that a
  // value has been truncated and is a lower bound and not in the page.
  void ApplyStatSizeLimits(size_t length) {
    if (max_->length() > length) {
      has_max = false;
    }
    if (min_->length() > length) {
      has_min = false;
    }
  }

  bool is_set() const {
    return has_min || has_max || has_null_count || has_distinct_count;
  }

  bool is_signed() const { return is_signed_; }

  void set_is_signed(bool is_signed) { is_signed_ = is_signed; }

  EncodedStatistics& set_max(const std::string& value) {
    *max_ = value;
    has_max = true;
    return *this;
  }

  EncodedStatistics& set_min(const std::string& value) {
    *min_ = value;
    has_min = true;
    return *this;
  }

  EncodedStatistics& set_null_count(int64_t value) {
    null_count = value;
    has_null_count = true;
    return *this;
  }

  EncodedStatistics& set_distinct_count(int64_t value) {
    distinct_count = value;
    has_distinct_count = true;
    return *this;
  }
};

/// \brief Base type for computing column statistics while writing a file
class PARQUET_EXPORT Statistics {
 public:
  virtual ~Statistics() {}

  /// \brief Create a new statistics instance given a column schema
  /// definition
  /// \param[in] descr the column schema
  /// \param[in] pool a memory pool to use for any memory allocations, optional
  static std::shared_ptr<Statistics> Make(
      const ColumnDescriptor* descr,
      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool());

  /// \brief Create a new statistics instance given a column schema
  /// definition and pre-existing state
  /// \param[in] descr the column schema
  /// \param[in] encoded_min the encoded minimum value
  /// \param[in] encoded_max the encoded maximum value
  /// \param[in] num_values total number of values
  /// \param[in] null_count number of null values
  /// \param[in] distinct_count number of distinct values
  /// \param[in] has_min_max whether the min/max statistics are set
  /// \param[in] has_null_count whether the null_count statistics are set
  /// \param[in] has_distinct_count whether the distinct_count statistics are set
  /// \param[in] pool a memory pool to use for any memory allocations, optional
  static std::shared_ptr<Statistics> Make(
      const ColumnDescriptor* descr, const std::string& encoded_min,
      const std::string& encoded_max, int64_t num_values, int64_t null_count,
      int64_t distinct_count, bool has_min_max, bool has_null_count,
      bool has_distinct_count,
      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool());

  // Helper function to convert EncodedStatistics to Statistics.
  // EncodedStatistics does not contain number of non-null values, and it can be
  // passed using the num_values parameter.
  static std::shared_ptr<Statistics> Make(
      const ColumnDescriptor* descr, const EncodedStatistics* encoded_statistics,
      int64_t num_values = -1,
      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool());

  /// \brief Return true if the count of null values is set
  virtual bool HasNullCount() const = 0;

  /// \brief The number of null values, may not be set
  virtual int64_t null_count() const = 0;

  /// \brief Return true if the count of distinct values is set
  virtual bool HasDistinctCount() const = 0;

  /// \brief The number of distinct values, may not be set
  virtual int64_t distinct_count() const = 0;

  /// \brief The number of non-null values in the column
  virtual int64_t num_values() const = 0;

  /// \brief Return true if the min and max statistics are set. Obtain
  /// with TypedStatistics<T>::min and max
  virtual bool HasMinMax() const = 0;

  /// \brief Reset state of object to initial (no data observed) state
  virtual void Reset() = 0;

  /// \brief Plain-encoded minimum value
  virtual std::string EncodeMin() const = 0;

  /// \brief Plain-encoded maximum value
  virtual std::string EncodeMax() const = 0;

  /// \brief The finalized encoded form of the statistics for transport
  virtual EncodedStatistics Encode() = 0;

  /// \brief The physical type of the column schema
  virtual Type::type physical_type() const = 0;

  /// \brief The full type descriptor from the column schema
  virtual const ColumnDescriptor* descr() const = 0;

  /// \brief Check two Statistics for equality
  virtual bool Equals(const Statistics& other) const = 0;

 protected:
  static std::shared_ptr<Statistics> Make(Type::type physical_type, const void* min,
                                          const void* max, int64_t num_values,
                                          int64_t null_count, int64_t distinct_count);
};

/// \brief A typed implementation of Statistics
template <typename DType>
class TypedStatistics : public Statistics {
 public:
  using T = typename DType::c_type;

  /// \brief The current minimum value
  virtual const T& min() const = 0;

  /// \brief The current maximum value
  virtual const T& max() const = 0;

  /// \brief Update state with state of another Statistics object
  virtual void Merge(const TypedStatistics<DType>& other) = 0;

  /// \brief Batch statistics update
  virtual void Update(const T* values, int64_t num_values, int64_t null_count) = 0;

  /// \brief Batch statistics update with supplied validity bitmap
  /// \param[in] values pointer to column values
  /// \param[in] valid_bits Pointer to bitmap representing if values are non-null.
  /// \param[in] valid_bits_offset Offset offset into valid_bits where the slice of
  ///                              data begins.
  /// \param[in] num_spaced_values The length of values in values/valid_bits to inspect
  ///                              when calculating statistics. This can be smaller than
  ///                              num_values+null_count as null_count can include nulls
  ///                              from parents while num_spaced_values does not.
  /// \param[in] num_values Number of values that are not null.
  /// \param[in] null_count Number of values that are null.
  virtual void UpdateSpaced(const T* values, const uint8_t* valid_bits,
                            int64_t valid_bits_offset, int64_t num_spaced_values,
                            int64_t num_values, int64_t null_count) = 0;

  /// \brief EXPERIMENTAL: Update statistics with an Arrow array without
  /// conversion to a primitive Parquet C type. Only implemented for certain
  /// Parquet type / Arrow type combinations like BYTE_ARRAY /
  /// arrow::BinaryArray
  ///
  /// If update_counts is true then the null_count and num_values will be updated
  /// based on the null_count of values.  Set to false if these are updated
  /// elsewhere (e.g. when updating a dictionary where the counts are taken from
  /// the indices and not the values)
  virtual void Update(const ::arrow::Array& values, bool update_counts = true) = 0;

  /// \brief Set min and max values to particular values
  virtual void SetMinMax(const T& min, const T& max) = 0;

  /// \brief Increments the null count directly
  /// Use Update to extract the null count from data.  Use this if you determine
  /// the null count through some other means (e.g. dictionary arrays where the
  /// null count is determined from the indices)
  virtual void IncrementNullCount(int64_t n) = 0;

  /// \brief Increments the number ov values directly
  /// The same note on IncrementNullCount applies here
  virtual void IncrementNumValues(int64_t n) = 0;
};

using BoolStatistics = TypedStatistics<BooleanType>;
using Int32Statistics = TypedStatistics<Int32Type>;
using Int64Statistics = TypedStatistics<Int64Type>;
using FloatStatistics = TypedStatistics<FloatType>;
using DoubleStatistics = TypedStatistics<DoubleType>;
using ByteArrayStatistics = TypedStatistics<ByteArrayType>;
using FLBAStatistics = TypedStatistics<FLBAType>;

/// \brief Typed version of Statistics::Make
template <typename DType>
std::shared_ptr<TypedStatistics<DType>> MakeStatistics(
    const ColumnDescriptor* descr,
    ::arrow::MemoryPool* pool = ::arrow::default_memory_pool()) {
  return std::static_pointer_cast<TypedStatistics<DType>>(Statistics::Make(descr, pool));
}

/// \brief Create Statistics initialized to a particular state
/// \param[in] min the minimum value
/// \param[in] max the minimum value
/// \param[in] num_values number of values
/// \param[in] null_count number of null values
/// \param[in] distinct_count number of distinct values
template <typename DType>
std::shared_ptr<TypedStatistics<DType>> MakeStatistics(const typename DType::c_type& min,
                                                       const typename DType::c_type& max,
                                                       int64_t num_values,
                                                       int64_t null_count,
                                                       int64_t distinct_count) {
  return std::static_pointer_cast<TypedStatistics<DType>>(Statistics::Make(
      DType::type_num, &min, &max, num_values, null_count, distinct_count));
}

/// \brief Typed version of Statistics::Make
template <typename DType>
std::shared_ptr<TypedStatistics<DType>> MakeStatistics(
    const ColumnDescriptor* descr, const std::string& encoded_min,
    const std::string& encoded_max, int64_t num_values, int64_t null_count,
    int64_t distinct_count, bool has_min_max, bool has_null_count,
    bool has_distinct_count, ::arrow::MemoryPool* pool = ::arrow::default_memory_pool()) {
  return std::static_pointer_cast<TypedStatistics<DType>>(Statistics::Make(
      descr, encoded_min, encoded_max, num_values, null_count, distinct_count,
      has_min_max, has_null_count, has_distinct_count, pool));
}

}  // namespace parquet

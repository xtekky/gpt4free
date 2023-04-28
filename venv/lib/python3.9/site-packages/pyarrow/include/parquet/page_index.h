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

#include "parquet/types.h"

#include <vector>

namespace parquet {

class ColumnDescriptor;
class ReaderProperties;

/// \brief ColumnIndex is a proxy around format::ColumnIndex.
class PARQUET_EXPORT ColumnIndex {
 public:
  /// \brief Create a ColumnIndex from a serialized thrift message.
  static std::unique_ptr<ColumnIndex> Make(const ColumnDescriptor& descr,
                                           const void* serialized_index,
                                           uint32_t index_len,
                                           const ReaderProperties& properties);

  virtual ~ColumnIndex() = default;

  /// \brief A bitmap with a bit set for each data page that has only null values.
  ///
  /// The length of this vector is equal to the number of data pages in the column.
  virtual const std::vector<bool>& null_pages() const = 0;

  /// \brief A vector of encoded lower bounds for each data page in this column.
  ///
  /// `null_pages` should be inspected first, as only pages with non-null values
  /// may have their lower bounds populated.
  virtual const std::vector<std::string>& encoded_min_values() const = 0;

  /// \brief A vector of encoded upper bounds for each data page in this column.
  ///
  /// `null_pages` should be inspected first, as only pages with non-null values
  /// may have their upper bounds populated.
  virtual const std::vector<std::string>& encoded_max_values() const = 0;

  /// \brief The ordering of lower and upper bounds.
  ///
  /// The boundary order applies accross all lower bounds, and all upper bounds,
  /// respectively. However, the order between lower bounds and upper bounds
  /// cannot be derived from this.
  virtual BoundaryOrder::type boundary_order() const = 0;

  /// \brief Whether per-page null count information is available.
  virtual bool has_null_counts() const = 0;

  /// \brief An optional vector with the number of null values in each data page.
  ///
  /// `has_null_counts` should be called first to determine if this information is
  /// available.
  virtual const std::vector<int64_t>& null_counts() const = 0;

  /// \brief A vector of page indices for non-null pages.
  virtual const std::vector<int32_t>& non_null_page_indices() const = 0;
};

/// \brief Typed implementation of ColumnIndex.
template <typename DType>
class PARQUET_EXPORT TypedColumnIndex : public ColumnIndex {
 public:
  using T = typename DType::c_type;

  /// \brief A vector of lower bounds for each data page in this column.
  ///
  /// This is like `encoded_min_values`, but with the values decoded according to
  /// the column's physical type.
  /// `min_values` and `max_values` can be used together with `boundary_order`
  /// in order to prune some data pages when searching for specific values.
  virtual const std::vector<T>& min_values() const = 0;

  /// \brief A vector of upper bounds for each data page in this column.
  ///
  /// Just like `min_values`, but for upper bounds instead of lower bounds.
  virtual const std::vector<T>& max_values() const = 0;
};

using BoolColumnIndex = TypedColumnIndex<BooleanType>;
using Int32ColumnIndex = TypedColumnIndex<Int32Type>;
using Int64ColumnIndex = TypedColumnIndex<Int64Type>;
using FloatColumnIndex = TypedColumnIndex<FloatType>;
using DoubleColumnIndex = TypedColumnIndex<DoubleType>;
using ByteArrayColumnIndex = TypedColumnIndex<ByteArrayType>;
using FLBAColumnIndex = TypedColumnIndex<FLBAType>;

/// \brief PageLocation is a proxy around format::PageLocation.
struct PARQUET_EXPORT PageLocation {
  /// File offset of the data page.
  int64_t offset;
  /// Total compressed size of the data page and header.
  int32_t compressed_page_size;
  /// Row id of the first row in the page within the row group.
  int64_t first_row_index;
};

/// \brief OffsetIndex is a proxy around format::OffsetIndex.
class PARQUET_EXPORT OffsetIndex {
 public:
  /// \brief Create a OffsetIndex from a serialized thrift message.
  static std::unique_ptr<OffsetIndex> Make(const void* serialized_index,
                                           uint32_t index_len,
                                           const ReaderProperties& properties);

  virtual ~OffsetIndex() = default;

  /// \brief A vector of locations for each data page in this column.
  virtual const std::vector<PageLocation>& page_locations() const = 0;
};

}  // namespace parquet

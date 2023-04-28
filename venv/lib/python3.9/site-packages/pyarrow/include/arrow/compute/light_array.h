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

#include "arrow/array.h"
#include "arrow/compute/exec.h"
#include "arrow/compute/exec/util.h"
#include "arrow/type.h"
#include "arrow/util/cpu_info.h"
#include "arrow/util/logging.h"

/// This file contains lightweight containers for Arrow buffers.  These containers
/// makes compromises in terms of strong ownership and the range of data types supported
/// in order to gain performance and reduced overhead.

namespace arrow {
namespace compute {

/// \brief Context needed by various execution engine operations
///
/// In the execution engine this context is provided by either the node or the
/// plan and the context exists for the lifetime of the plan.  Defining this here
/// allows us to take advantage of these resources without coupling the logic with
/// the execution engine.
struct LightContext {
  bool has_avx2() const { return (hardware_flags & arrow::internal::CpuInfo::AVX2) > 0; }
  int64_t hardware_flags;
  util::TempVectorStack* stack;
};

/// \brief Description of the layout of a "key" column
///
/// A "key" column is a non-nested, non-union column.
/// Every key column has either 0 (null), 2 (e.g. int32) or 3 (e.g. string) buffers
/// and no children.
///
/// This metadata object is a zero-allocation analogue of arrow::DataType
struct ARROW_EXPORT KeyColumnMetadata {
  KeyColumnMetadata() = default;
  KeyColumnMetadata(bool is_fixed_length_in, uint32_t fixed_length_in,
                    bool is_null_type_in = false)
      : is_fixed_length(is_fixed_length_in),
        is_null_type(is_null_type_in),
        fixed_length(fixed_length_in) {}
  /// \brief True if the column is not a varying-length binary type
  ///
  /// If this is true the column will have a validity buffer and
  /// a data buffer and the third buffer will be unused.
  bool is_fixed_length;
  /// \brief True if this column is the null type
  bool is_null_type;
  /// \brief The number of bytes for each item
  ///
  /// Zero has a special meaning, indicating a bit vector with one bit per value if it
  /// isn't a null type column.
  ///
  /// For a varying-length binary column this represents the number of bytes per offset.
  uint32_t fixed_length;
};

/// \brief A lightweight view into a "key" array
///
/// A "key" column is a non-nested, non-union column \see KeyColumnMetadata
///
/// This metadata object is a zero-allocation analogue of arrow::ArrayData
class ARROW_EXPORT KeyColumnArray {
 public:
  /// \brief Create an uninitialized KeyColumnArray
  KeyColumnArray() = default;
  /// \brief Create a read-only view from buffers
  ///
  /// This is a view only and does not take ownership of the buffers.  The lifetime
  /// of the buffers must exceed the lifetime of this view
  KeyColumnArray(const KeyColumnMetadata& metadata, int64_t length,
                 const uint8_t* validity_buffer, const uint8_t* fixed_length_buffer,
                 const uint8_t* var_length_buffer, int bit_offset_validity = 0,
                 int bit_offset_fixed = 0);
  /// \brief Create a mutable view from buffers
  ///
  /// This is a view only and does not take ownership of the buffers.  The lifetime
  /// of the buffers must exceed the lifetime of this view
  KeyColumnArray(const KeyColumnMetadata& metadata, int64_t length,
                 uint8_t* validity_buffer, uint8_t* fixed_length_buffer,
                 uint8_t* var_length_buffer, int bit_offset_validity = 0,
                 int bit_offset_fixed = 0);
  /// \brief Create a sliced view of `this`
  ///
  /// The number of rows used in offset must be divisible by 8
  /// in order to not split bit vectors within a single byte.
  KeyColumnArray Slice(int64_t offset, int64_t length) const;
  /// \brief Create a copy of `this` with a buffer from `other`
  ///
  /// The copy will be identical to `this` except the buffer at buffer_id_to_replace
  /// will be replaced by the corresponding buffer in `other`.
  KeyColumnArray WithBufferFrom(const KeyColumnArray& other,
                                int buffer_id_to_replace) const;

  /// \brief Create a copy of `this` with new metadata
  KeyColumnArray WithMetadata(const KeyColumnMetadata& metadata) const;

  // Constants used for accessing buffers using data() and mutable_data().
  static constexpr int kValidityBuffer = 0;
  static constexpr int kFixedLengthBuffer = 1;
  static constexpr int kVariableLengthBuffer = 2;

  /// \brief Return one of the underlying mutable buffers
  uint8_t* mutable_data(int i) {
    ARROW_DCHECK(i >= 0 && i <= kMaxBuffers);
    return mutable_buffers_[i];
  }
  /// \brief Return one of the underlying read-only buffers
  const uint8_t* data(int i) const {
    ARROW_DCHECK(i >= 0 && i <= kMaxBuffers);
    return buffers_[i];
  }
  /// \brief Return a mutable version of the offsets buffer
  ///
  /// Only valid if this is a view into a varbinary type
  uint32_t* mutable_offsets() {
    DCHECK(!metadata_.is_fixed_length);
    DCHECK_EQ(metadata_.fixed_length, sizeof(uint32_t));
    return reinterpret_cast<uint32_t*>(mutable_data(kFixedLengthBuffer));
  }
  /// \brief Return a read-only version of the offsets buffer
  ///
  /// Only valid if this is a view into a varbinary type
  const uint32_t* offsets() const {
    DCHECK(!metadata_.is_fixed_length);
    DCHECK_EQ(metadata_.fixed_length, sizeof(uint32_t));
    return reinterpret_cast<const uint32_t*>(data(kFixedLengthBuffer));
  }
  /// \brief Return a mutable version of the large-offsets buffer
  ///
  /// Only valid if this is a view into a large varbinary type
  uint64_t* mutable_large_offsets() {
    DCHECK(!metadata_.is_fixed_length);
    DCHECK_EQ(metadata_.fixed_length, sizeof(uint64_t));
    return reinterpret_cast<uint64_t*>(mutable_data(kFixedLengthBuffer));
  }
  /// \brief Return a read-only version of the large-offsets buffer
  ///
  /// Only valid if this is a view into a large varbinary type
  const uint64_t* large_offsets() const {
    DCHECK(!metadata_.is_fixed_length);
    DCHECK_EQ(metadata_.fixed_length, sizeof(uint64_t));
    return reinterpret_cast<const uint64_t*>(data(kFixedLengthBuffer));
  }
  /// \brief Return the type metadata
  const KeyColumnMetadata& metadata() const { return metadata_; }
  /// \brief Return the length (in rows) of the array
  int64_t length() const { return length_; }
  /// \brief Return the bit offset into the corresponding vector
  ///
  /// if i == 1 then this must be a bool array
  int bit_offset(int i) const {
    ARROW_DCHECK(i >= 0 && i < kMaxBuffers);
    return bit_offset_[i];
  }

 private:
  static constexpr int kMaxBuffers = 3;
  const uint8_t* buffers_[kMaxBuffers];
  uint8_t* mutable_buffers_[kMaxBuffers];
  KeyColumnMetadata metadata_;
  int64_t length_;
  // Starting bit offset within the first byte (between 0 and 7)
  // to be used when accessing buffers that store bit vectors.
  int bit_offset_[kMaxBuffers - 1];
};

/// \brief Create KeyColumnMetadata from a DataType
///
/// If `type` is a dictionary type then this will return the KeyColumnMetadata for
/// the indices type
///
/// This should only be called on "key" columns.  Calling this with
/// a non-key column will return Status::TypeError.
ARROW_EXPORT Result<KeyColumnMetadata> ColumnMetadataFromDataType(
    const std::shared_ptr<DataType>& type);

/// \brief Create KeyColumnArray from ArrayData
///
/// If `type` is a dictionary type then this will return the KeyColumnArray for
/// the indices array
///
/// The caller should ensure this is only called on "key" columns.
/// \see ColumnMetadataFromDataType for details
ARROW_EXPORT Result<KeyColumnArray> ColumnArrayFromArrayData(
    const std::shared_ptr<ArrayData>& array_data, int64_t start_row, int64_t num_rows);

/// \brief Create KeyColumnArray from ArrayData and KeyColumnMetadata
///
/// If `type` is a dictionary type then this will return the KeyColumnArray for
/// the indices array
///
/// The caller should ensure this is only called on "key" columns.
/// \see ColumnMetadataFromDataType for details
ARROW_EXPORT KeyColumnArray ColumnArrayFromArrayDataAndMetadata(
    const std::shared_ptr<ArrayData>& array_data, const KeyColumnMetadata& metadata,
    int64_t start_row, int64_t num_rows);

/// \brief Create KeyColumnMetadata instances from an ExecBatch
///
/// column_metadatas will be resized to fit
///
/// All columns in `batch` must be eligible "key" columns and have an array shape
/// \see ColumnMetadataFromDataType for more details
ARROW_EXPORT Status ColumnMetadatasFromExecBatch(
    const ExecBatch& batch, std::vector<KeyColumnMetadata>* column_metadatas);

/// \brief Create KeyColumnArray instances from a slice of an ExecBatch
///
/// column_arrays will be resized to fit
///
/// All columns in `batch` must be eligible "key" columns and have an array shape
/// \see ColumnArrayFromArrayData for more details
ARROW_EXPORT Status ColumnArraysFromExecBatch(const ExecBatch& batch, int64_t start_row,
                                              int64_t num_rows,
                                              std::vector<KeyColumnArray>* column_arrays);

/// \brief Create KeyColumnArray instances from an ExecBatch
///
/// column_arrays will be resized to fit
///
/// All columns in `batch` must be eligible "key" columns and have an array shape
/// \see ColumnArrayFromArrayData for more details
ARROW_EXPORT Status ColumnArraysFromExecBatch(const ExecBatch& batch,
                                              std::vector<KeyColumnArray>* column_arrays);

/// A lightweight resizable array for "key" columns
///
/// Unlike KeyColumnArray this instance owns its buffers
///
/// Resizing is handled by arrow::ResizableBuffer and a doubling approach is
/// used so that resizes will always grow up to the next power of 2
class ARROW_EXPORT ResizableArrayData {
 public:
  /// \brief Create an uninitialized instance
  ///
  /// Init must be called before calling any other operations
  ResizableArrayData()
      : log_num_rows_min_(0),
        pool_(NULLPTR),
        num_rows_(0),
        num_rows_allocated_(0),
        var_len_buf_size_(0) {}

  ~ResizableArrayData() { Clear(true); }

  /// \brief Initialize the array
  /// \param data_type The data type this array is holding data for.
  /// \param pool The pool to make allocations on
  /// \param log_num_rows_min All resize operations will allocate at least enough
  ///                         space for (1 << log_num_rows_min) rows
  void Init(const std::shared_ptr<DataType>& data_type, MemoryPool* pool,
            int log_num_rows_min);

  /// \brief Resets the array back to an empty state
  /// \param release_buffers If true then allocated memory is released and the
  ///                        next resize operation will have to reallocate memory
  void Clear(bool release_buffers);

  /// \brief Resize the fixed length buffers
  ///
  /// The buffers will be resized to hold at least `num_rows_new` rows of data
  Status ResizeFixedLengthBuffers(int num_rows_new);

  /// \brief Resize the varying length buffer if this array is a variable binary type
  ///
  /// This must be called after offsets have been populated and the buffer will be
  /// resized to hold at least as much data as the offsets require
  ///
  /// Does nothing if the array is not a variable binary type
  Status ResizeVaryingLengthBuffer();

  /// \brief The current length (in rows) of the array
  int num_rows() const { return num_rows_; }

  /// \brief A non-owning view into this array
  KeyColumnArray column_array() const;

  /// \brief A lightweight descriptor of the data held by this array
  Result<KeyColumnMetadata> column_metadata() const {
    return ColumnMetadataFromDataType(data_type_);
  }

  /// \brief Convert the data to an arrow::ArrayData
  ///
  /// This is a zero copy operation and the created ArrayData will reference the
  /// buffers held by this instance.
  std::shared_ptr<ArrayData> array_data() const;

  // Constants used for accessing buffers using mutable_data().
  static constexpr int kValidityBuffer = 0;
  static constexpr int kFixedLengthBuffer = 1;
  static constexpr int kVariableLengthBuffer = 2;

  /// \brief A raw pointer to the requested buffer
  ///
  /// If i is 0 (kValidityBuffer) then this returns the validity buffer
  /// If i is 1 (kFixedLengthBuffer) then this returns the buffer used for values (if this
  /// is a fixed
  ///           length data type) or offsets (if this is a variable binary type)
  /// If i is 2 (kVariableLengthBuffer) then this returns the buffer used for variable
  /// length binary data
  uint8_t* mutable_data(int i) { return buffers_[i]->mutable_data(); }

 private:
  static constexpr int64_t kNumPaddingBytes = 64;
  int log_num_rows_min_;
  std::shared_ptr<DataType> data_type_;
  MemoryPool* pool_;
  int num_rows_;
  int num_rows_allocated_;
  int var_len_buf_size_;
  static constexpr int kMaxBuffers = 3;
  std::shared_ptr<ResizableBuffer> buffers_[kMaxBuffers];
};

/// \brief A builder to concatenate batches of data into a larger batch
///
/// Will only store num_rows_max() rows
class ARROW_EXPORT ExecBatchBuilder {
 public:
  /// \brief Add rows from `source` into `target` column
  ///
  /// If `target` is uninitialized or cleared it will be initialized to use
  /// the given pool.
  static Status AppendSelected(const std::shared_ptr<ArrayData>& source,
                               ResizableArrayData* target, int num_rows_to_append,
                               const uint16_t* row_ids, MemoryPool* pool);

  /// \brief Add nulls into `target` column
  ///
  /// If `target` is uninitialized or cleared it will be initialized to use
  /// the given pool.
  static Status AppendNulls(const std::shared_ptr<DataType>& type,
                            ResizableArrayData& target, int num_rows_to_append,
                            MemoryPool* pool);

  /// \brief Add selected rows from `batch`
  ///
  /// If `col_ids` is null then `num_cols` should less than batch.num_values() and
  /// the first `num_cols` columns of batch will be appended.
  ///
  /// All columns in `batch` must have array shape
  Status AppendSelected(MemoryPool* pool, const ExecBatch& batch, int num_rows_to_append,
                        const uint16_t* row_ids, int num_cols,
                        const int* col_ids = NULLPTR);

  /// \brief Add all-null rows
  Status AppendNulls(MemoryPool* pool,
                     const std::vector<std::shared_ptr<DataType>>& types,
                     int num_rows_to_append);

  /// \brief Create an ExecBatch with the data that has been appended so far
  ///        and clear this builder to be used again
  ///
  /// Should only be called if num_rows() returns non-zero.
  ExecBatch Flush();

  int num_rows() const { return values_.empty() ? 0 : values_[0].num_rows(); }

  static int num_rows_max() { return 1 << kLogNumRows; }

 private:
  static constexpr int kLogNumRows = 15;

  // Calculate how many rows to skip from the tail of the
  // sequence of selected rows, such that the total size of skipped rows is at
  // least equal to the size specified by the caller.
  //
  // Skipping of the tail rows
  // is used to allow for faster processing by the caller of remaining rows
  // without checking buffer bounds (useful with SIMD or fixed size memory loads
  // and stores).
  //
  // The sequence of row_ids provided must be non-decreasing.
  //
  static int NumRowsToSkip(const std::shared_ptr<ArrayData>& column, int num_rows,
                           const uint16_t* row_ids, int num_tail_bytes_to_skip);

  // The supplied lambda will be called for each row in the given list of rows.
  // The arguments given to it will be:
  // - index of a row (within the set of selected rows),
  // - pointer to the value,
  // - byte length of the value.
  //
  // The information about nulls (validity bitmap) is not used in this call and
  // has to be processed separately.
  //
  template <class PROCESS_VALUE_FN>
  static void Visit(const std::shared_ptr<ArrayData>& column, int num_rows,
                    const uint16_t* row_ids, PROCESS_VALUE_FN process_value_fn);

  template <bool OUTPUT_BYTE_ALIGNED>
  static void CollectBitsImp(const uint8_t* input_bits, int64_t input_bits_offset,
                             uint8_t* output_bits, int64_t output_bits_offset,
                             int num_rows, const uint16_t* row_ids);
  static void CollectBits(const uint8_t* input_bits, int64_t input_bits_offset,
                          uint8_t* output_bits, int64_t output_bits_offset, int num_rows,
                          const uint16_t* row_ids);

  std::vector<ResizableArrayData> values_;
};

}  // namespace compute
}  // namespace arrow

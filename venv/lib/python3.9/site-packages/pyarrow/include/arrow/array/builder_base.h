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

#include <algorithm>  // IWYU pragma: keep
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "arrow/array/array_base.h"
#include "arrow/array/array_primitive.h"
#include "arrow/buffer.h"
#include "arrow/buffer_builder.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type_fwd.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {

/// \defgroup numeric-builders Concrete builder subclasses for numeric types
/// @{
/// @}

/// \defgroup temporal-builders Concrete builder subclasses for temporal types
/// @{
/// @}

/// \defgroup binary-builders Concrete builder subclasses for binary types
/// @{
/// @}

/// \defgroup nested-builders Concrete builder subclasses for nested types
/// @{
/// @}

/// \defgroup dictionary-builders Concrete builder subclasses for dictionary types
/// @{
/// @}

constexpr int64_t kMinBuilderCapacity = 1 << 5;
constexpr int64_t kListMaximumElements = std::numeric_limits<int32_t>::max() - 1;

/// Base class for all data array builders.
///
/// This class provides a facilities for incrementally building the null bitmap
/// (see Append methods) and as a side effect the current number of slots and
/// the null count.
///
/// \note Users are expected to use builders as one of the concrete types below.
/// For example, ArrayBuilder* pointing to BinaryBuilder should be downcast before use.
class ARROW_EXPORT ArrayBuilder {
 public:
  explicit ArrayBuilder(MemoryPool* pool, int64_t alignment = kDefaultBufferAlignment)
      : pool_(pool), alignment_(alignment), null_bitmap_builder_(pool, alignment) {}

  ARROW_DEFAULT_MOVE_AND_ASSIGN(ArrayBuilder);

  virtual ~ArrayBuilder() = default;

  /// For nested types. Since the objects are owned by this class instance, we
  /// skip shared pointers and just return a raw pointer
  ArrayBuilder* child(int i) { return children_[i].get(); }

  const std::shared_ptr<ArrayBuilder>& child_builder(int i) const { return children_[i]; }

  int num_children() const { return static_cast<int>(children_.size()); }

  virtual int64_t length() const { return length_; }
  int64_t null_count() const { return null_count_; }
  int64_t capacity() const { return capacity_; }

  /// \brief Ensure that enough memory has been allocated to fit the indicated
  /// number of total elements in the builder, including any that have already
  /// been appended. Does not account for reallocations that may be due to
  /// variable size data, like binary values. To make space for incremental
  /// appends, use Reserve instead.
  ///
  /// \param[in] capacity the minimum number of total array values to
  ///            accommodate. Must be greater than the current capacity.
  /// \return Status
  virtual Status Resize(int64_t capacity);

  /// \brief Ensure that there is enough space allocated to append the indicated
  /// number of elements without any further reallocation. Overallocation is
  /// used in order to minimize the impact of incremental Reserve() calls.
  /// Note that additional_capacity is relative to the current number of elements
  /// rather than to the current capacity, so calls to Reserve() which are not
  /// interspersed with addition of new elements may not increase the capacity.
  ///
  /// \param[in] additional_capacity the number of additional array values
  /// \return Status
  Status Reserve(int64_t additional_capacity) {
    auto current_capacity = capacity();
    auto min_capacity = length() + additional_capacity;
    if (min_capacity <= current_capacity) return Status::OK();

    // leave growth factor up to BufferBuilder
    auto new_capacity = BufferBuilder::GrowByFactor(current_capacity, min_capacity);
    return Resize(new_capacity);
  }

  /// Reset the builder.
  virtual void Reset();

  /// \brief Append a null value to builder
  virtual Status AppendNull() = 0;
  /// \brief Append a number of null values to builder
  virtual Status AppendNulls(int64_t length) = 0;

  /// \brief Append a non-null value to builder
  ///
  /// The appended value is an implementation detail, but the corresponding
  /// memory slot is guaranteed to be initialized.
  /// This method is useful when appending a null value to a parent nested type.
  virtual Status AppendEmptyValue() = 0;

  /// \brief Append a number of non-null values to builder
  ///
  /// The appended values are an implementation detail, but the corresponding
  /// memory slot is guaranteed to be initialized.
  /// This method is useful when appending null values to a parent nested type.
  virtual Status AppendEmptyValues(int64_t length) = 0;

  /// \brief Append a value from a scalar
  Status AppendScalar(const Scalar& scalar) { return AppendScalar(scalar, 1); }
  virtual Status AppendScalar(const Scalar& scalar, int64_t n_repeats);
  virtual Status AppendScalars(const ScalarVector& scalars);

  /// \brief Append a range of values from an array.
  ///
  /// The given array must be the same type as the builder.
  virtual Status AppendArraySlice(const ArraySpan& array, int64_t offset,
                                  int64_t length) {
    return Status::NotImplemented("AppendArraySlice for builder for ", *type());
  }

  /// For cases where raw data was memcpy'd into the internal buffers, allows us
  /// to advance the length of the builder. It is your responsibility to use
  /// this function responsibly.
  ARROW_DEPRECATED(
      "Deprecated in 6.0.0. ArrayBuilder::Advance is poorly supported and mostly "
      "untested.\nFor low-level control over buffer construction, use BufferBuilder "
      "or TypedBufferBuilder directly.")
  Status Advance(int64_t elements);

  /// \brief Return result of builder as an internal generic ArrayData
  /// object. Resets builder except for dictionary builder
  ///
  /// \param[out] out the finalized ArrayData object
  /// \return Status
  virtual Status FinishInternal(std::shared_ptr<ArrayData>* out) = 0;

  /// \brief Return result of builder as an Array object.
  ///
  /// The builder is reset except for DictionaryBuilder.
  ///
  /// \param[out] out the finalized Array object
  /// \return Status
  Status Finish(std::shared_ptr<Array>* out);

  /// \brief Return result of builder as an Array object.
  ///
  /// The builder is reset except for DictionaryBuilder.
  ///
  /// \return The finalized Array object
  Result<std::shared_ptr<Array>> Finish();

  /// \brief Return the type of the built Array
  virtual std::shared_ptr<DataType> type() const = 0;

 protected:
  /// Append to null bitmap
  Status AppendToBitmap(bool is_valid);

  /// Vector append. Treat each zero byte as a null.   If valid_bytes is null
  /// assume all of length bits are valid.
  Status AppendToBitmap(const uint8_t* valid_bytes, int64_t length);

  /// Uniform append.  Append N times the same validity bit.
  Status AppendToBitmap(int64_t num_bits, bool value);

  /// Set the next length bits to not null (i.e. valid).
  Status SetNotNull(int64_t length);

  // Unsafe operations (don't check capacity/don't resize)

  void UnsafeAppendNull() { UnsafeAppendToBitmap(false); }

  // Append to null bitmap, update the length
  void UnsafeAppendToBitmap(bool is_valid) {
    null_bitmap_builder_.UnsafeAppend(is_valid);
    ++length_;
    if (!is_valid) ++null_count_;
  }

  // Vector append. Treat each zero byte as a nullzero. If valid_bytes is null
  // assume all of length bits are valid.
  void UnsafeAppendToBitmap(const uint8_t* valid_bytes, int64_t length) {
    if (valid_bytes == NULLPTR) {
      return UnsafeSetNotNull(length);
    }
    null_bitmap_builder_.UnsafeAppend(valid_bytes, length);
    length_ += length;
    null_count_ = null_bitmap_builder_.false_count();
  }

  // Vector append. Copy from a given bitmap. If bitmap is null assume
  // all of length bits are valid.
  void UnsafeAppendToBitmap(const uint8_t* bitmap, int64_t offset, int64_t length) {
    if (bitmap == NULLPTR) {
      return UnsafeSetNotNull(length);
    }
    null_bitmap_builder_.UnsafeAppend(bitmap, offset, length);
    length_ += length;
    null_count_ = null_bitmap_builder_.false_count();
  }

  // Append the same validity value a given number of times.
  void UnsafeAppendToBitmap(const int64_t num_bits, bool value) {
    if (value) {
      UnsafeSetNotNull(num_bits);
    } else {
      UnsafeSetNull(num_bits);
    }
  }

  void UnsafeAppendToBitmap(const std::vector<bool>& is_valid);

  // Set the next validity bits to not null (i.e. valid).
  void UnsafeSetNotNull(int64_t length);

  // Set the next validity bits to null (i.e. invalid).
  void UnsafeSetNull(int64_t length);

  static Status TrimBuffer(const int64_t bytes_filled, ResizableBuffer* buffer);

  /// \brief Finish to an array of the specified ArrayType
  template <typename ArrayType>
  Status FinishTyped(std::shared_ptr<ArrayType>* out) {
    std::shared_ptr<Array> out_untyped;
    ARROW_RETURN_NOT_OK(Finish(&out_untyped));
    *out = std::static_pointer_cast<ArrayType>(std::move(out_untyped));
    return Status::OK();
  }

  // Check the requested capacity for validity
  Status CheckCapacity(int64_t new_capacity) {
    if (ARROW_PREDICT_FALSE(new_capacity < 0)) {
      return Status::Invalid(
          "Resize capacity must be positive (requested: ", new_capacity, ")");
    }

    if (ARROW_PREDICT_FALSE(new_capacity < length_)) {
      return Status::Invalid("Resize cannot downsize (requested: ", new_capacity,
                             ", current length: ", length_, ")");
    }

    return Status::OK();
  }

  // Check for array type
  Status CheckArrayType(const std::shared_ptr<DataType>& expected_type,
                        const Array& array, const char* message);
  Status CheckArrayType(Type::type expected_type, const Array& array,
                        const char* message);

  MemoryPool* pool_;
  int64_t alignment_;

  TypedBufferBuilder<bool> null_bitmap_builder_;
  int64_t null_count_ = 0;

  // Array length, so far. Also, the index of the next element to be added
  int64_t length_ = 0;
  int64_t capacity_ = 0;

  // Child value array builders. These are owned by this class
  std::vector<std::shared_ptr<ArrayBuilder>> children_;

 private:
  ARROW_DISALLOW_COPY_AND_ASSIGN(ArrayBuilder);
};

/// \brief Construct an empty ArrayBuilder corresponding to the data
/// type
/// \param[in] pool the MemoryPool to use for allocations
/// \param[in] type the data type to create the builder for
/// \param[out] out the created ArrayBuilder
ARROW_EXPORT
Status MakeBuilder(MemoryPool* pool, const std::shared_ptr<DataType>& type,
                   std::unique_ptr<ArrayBuilder>* out);

inline Result<std::unique_ptr<ArrayBuilder>> MakeBuilder(
    const std::shared_ptr<DataType>& type, MemoryPool* pool = default_memory_pool()) {
  std::unique_ptr<ArrayBuilder> out;
  ARROW_RETURN_NOT_OK(MakeBuilder(pool, type, &out));
  return std::move(out);
}

/// \brief Construct an empty ArrayBuilder corresponding to the data
/// type, where any top-level or nested dictionary builders return the
/// exact index type specified by the type.
ARROW_EXPORT
Status MakeBuilderExactIndex(MemoryPool* pool, const std::shared_ptr<DataType>& type,
                             std::unique_ptr<ArrayBuilder>* out);

inline Result<std::unique_ptr<ArrayBuilder>> MakeBuilderExactIndex(
    const std::shared_ptr<DataType>& type, MemoryPool* pool = default_memory_pool()) {
  std::unique_ptr<ArrayBuilder> out;
  ARROW_RETURN_NOT_OK(MakeBuilderExactIndex(pool, type, &out));
  return std::move(out);
}

/// \brief Construct an empty DictionaryBuilder initialized optionally
/// with a pre-existing dictionary
/// \param[in] pool the MemoryPool to use for allocations
/// \param[in] type the dictionary type to create the builder for
/// \param[in] dictionary the initial dictionary, if any. May be nullptr
/// \param[out] out the created ArrayBuilder
ARROW_EXPORT
Status MakeDictionaryBuilder(MemoryPool* pool, const std::shared_ptr<DataType>& type,
                             const std::shared_ptr<Array>& dictionary,
                             std::unique_ptr<ArrayBuilder>* out);

inline Result<std::unique_ptr<ArrayBuilder>> MakeDictionaryBuilder(
    const std::shared_ptr<DataType>& type, const std::shared_ptr<Array>& dictionary,
    MemoryPool* pool = default_memory_pool()) {
  std::unique_ptr<ArrayBuilder> out;
  ARROW_RETURN_NOT_OK(MakeDictionaryBuilder(pool, type, dictionary, &out));
  return std::move(out);
}

}  // namespace arrow

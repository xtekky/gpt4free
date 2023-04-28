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

#include <atomic>  // IWYU pragma: export
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "arrow/buffer.h"
#include "arrow/result.h"
#include "arrow/type.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {

class Array;

// When slicing, we do not know the null count of the sliced range without
// doing some computation. To avoid doing this eagerly, we set the null count
// to -1 (any negative number will do). When Array::null_count is called the
// first time, the null count will be computed. See ARROW-33
constexpr int64_t kUnknownNullCount = -1;

// ----------------------------------------------------------------------
// Generic array data container

/// \class ArrayData
/// \brief Mutable container for generic Arrow array data
///
/// This data structure is a self-contained representation of the memory and
/// metadata inside an Arrow array data structure (called vectors in Java). The
/// classes arrow::Array and its subclasses provide strongly-typed accessors
/// with support for the visitor pattern and other affordances.
///
/// This class is designed for easy internal data manipulation, analytical data
/// processing, and data transport to and from IPC messages. For example, we
/// could cast from int64 to float64 like so:
///
/// Int64Array arr = GetMyData();
/// auto new_data = arr.data()->Copy();
/// new_data->type = arrow::float64();
/// DoubleArray double_arr(new_data);
///
/// This object is also useful in an analytics setting where memory may be
/// reused. For example, if we had a group of operations all returning doubles,
/// say:
///
/// Log(Sqrt(Expr(arr)))
///
/// Then the low-level implementations of each of these functions could have
/// the signatures
///
/// void Log(const ArrayData& values, ArrayData* out);
///
/// As another example a function may consume one or more memory buffers in an
/// input array and replace them with newly-allocated data, changing the output
/// data type as well.
struct ARROW_EXPORT ArrayData {
  ArrayData() = default;

  ArrayData(std::shared_ptr<DataType> type, int64_t length,
            int64_t null_count = kUnknownNullCount, int64_t offset = 0)
      : type(std::move(type)), length(length), null_count(null_count), offset(offset) {}

  ArrayData(std::shared_ptr<DataType> type, int64_t length,
            std::vector<std::shared_ptr<Buffer>> buffers,
            int64_t null_count = kUnknownNullCount, int64_t offset = 0)
      : ArrayData(std::move(type), length, null_count, offset) {
    this->buffers = std::move(buffers);
  }

  ArrayData(std::shared_ptr<DataType> type, int64_t length,
            std::vector<std::shared_ptr<Buffer>> buffers,
            std::vector<std::shared_ptr<ArrayData>> child_data,
            int64_t null_count = kUnknownNullCount, int64_t offset = 0)
      : ArrayData(std::move(type), length, null_count, offset) {
    this->buffers = std::move(buffers);
    this->child_data = std::move(child_data);
  }

  static std::shared_ptr<ArrayData> Make(std::shared_ptr<DataType> type, int64_t length,
                                         std::vector<std::shared_ptr<Buffer>> buffers,
                                         int64_t null_count = kUnknownNullCount,
                                         int64_t offset = 0);

  static std::shared_ptr<ArrayData> Make(
      std::shared_ptr<DataType> type, int64_t length,
      std::vector<std::shared_ptr<Buffer>> buffers,
      std::vector<std::shared_ptr<ArrayData>> child_data,
      int64_t null_count = kUnknownNullCount, int64_t offset = 0);

  static std::shared_ptr<ArrayData> Make(
      std::shared_ptr<DataType> type, int64_t length,
      std::vector<std::shared_ptr<Buffer>> buffers,
      std::vector<std::shared_ptr<ArrayData>> child_data,
      std::shared_ptr<ArrayData> dictionary, int64_t null_count = kUnknownNullCount,
      int64_t offset = 0);

  static std::shared_ptr<ArrayData> Make(std::shared_ptr<DataType> type, int64_t length,
                                         int64_t null_count = kUnknownNullCount,
                                         int64_t offset = 0);

  // Move constructor
  ArrayData(ArrayData&& other) noexcept
      : type(std::move(other.type)),
        length(other.length),
        offset(other.offset),
        buffers(std::move(other.buffers)),
        child_data(std::move(other.child_data)),
        dictionary(std::move(other.dictionary)) {
    SetNullCount(other.null_count);
  }

  // Copy constructor
  ArrayData(const ArrayData& other) noexcept
      : type(other.type),
        length(other.length),
        offset(other.offset),
        buffers(other.buffers),
        child_data(other.child_data),
        dictionary(other.dictionary) {
    SetNullCount(other.null_count);
  }

  // Move assignment
  ArrayData& operator=(ArrayData&& other) {
    type = std::move(other.type);
    length = other.length;
    SetNullCount(other.null_count);
    offset = other.offset;
    buffers = std::move(other.buffers);
    child_data = std::move(other.child_data);
    dictionary = std::move(other.dictionary);
    return *this;
  }

  // Copy assignment
  ArrayData& operator=(const ArrayData& other) {
    type = other.type;
    length = other.length;
    SetNullCount(other.null_count);
    offset = other.offset;
    buffers = other.buffers;
    child_data = other.child_data;
    dictionary = other.dictionary;
    return *this;
  }

  std::shared_ptr<ArrayData> Copy() const { return std::make_shared<ArrayData>(*this); }

  bool IsNull(int64_t i) const {
    return ((buffers[0] != NULLPTR) ? !bit_util::GetBit(buffers[0]->data(), i + offset)
                                    : null_count.load() == length);
  }

  // Access a buffer's data as a typed C pointer
  template <typename T>
  inline const T* GetValues(int i, int64_t absolute_offset) const {
    if (buffers[i]) {
      return reinterpret_cast<const T*>(buffers[i]->data()) + absolute_offset;
    } else {
      return NULLPTR;
    }
  }

  template <typename T>
  inline const T* GetValues(int i) const {
    return GetValues<T>(i, offset);
  }

  // Like GetValues, but returns NULLPTR instead of aborting if the underlying
  // buffer is not a CPU buffer.
  template <typename T>
  inline const T* GetValuesSafe(int i, int64_t absolute_offset) const {
    if (buffers[i] && buffers[i]->is_cpu()) {
      return reinterpret_cast<const T*>(buffers[i]->data()) + absolute_offset;
    } else {
      return NULLPTR;
    }
  }

  template <typename T>
  inline const T* GetValuesSafe(int i) const {
    return GetValuesSafe<T>(i, offset);
  }

  // Access a buffer's data as a typed C pointer
  template <typename T>
  inline T* GetMutableValues(int i, int64_t absolute_offset) {
    if (buffers[i]) {
      return reinterpret_cast<T*>(buffers[i]->mutable_data()) + absolute_offset;
    } else {
      return NULLPTR;
    }
  }

  template <typename T>
  inline T* GetMutableValues(int i) {
    return GetMutableValues<T>(i, offset);
  }

  /// \brief Construct a zero-copy slice of the data with the given offset and length
  std::shared_ptr<ArrayData> Slice(int64_t offset, int64_t length) const;

  /// \brief Input-checking variant of Slice
  ///
  /// An Invalid Status is returned if the requested slice falls out of bounds.
  /// Note that unlike Slice, `length` isn't clamped to the available buffer size.
  Result<std::shared_ptr<ArrayData>> SliceSafe(int64_t offset, int64_t length) const;

  void SetNullCount(int64_t v) { null_count.store(v); }

  /// \brief Return null count, or compute and set it if it's not known
  int64_t GetNullCount() const;

  bool MayHaveNulls() const {
    // If an ArrayData is slightly malformed it may have kUnknownNullCount set
    // but no buffer
    return null_count.load() != 0 && buffers[0] != NULLPTR;
  }

  std::shared_ptr<DataType> type;
  int64_t length = 0;
  mutable std::atomic<int64_t> null_count{0};
  // The logical start point into the physical buffers (in values, not bytes).
  // Note that, for child data, this must be *added* to the child data's own offset.
  int64_t offset = 0;
  std::vector<std::shared_ptr<Buffer>> buffers;
  std::vector<std::shared_ptr<ArrayData>> child_data;

  // The dictionary for this Array, if any. Only used for dictionary type
  std::shared_ptr<ArrayData> dictionary;
};

/// \brief A non-owning Buffer reference
struct ARROW_EXPORT BufferSpan {
  // It is the user of this class's responsibility to ensure that
  // buffers that were const originally are not written to
  // accidentally.
  uint8_t* data = NULLPTR;
  int64_t size = 0;
  // Pointer back to buffer that owns this memory
  const std::shared_ptr<Buffer>* owner = NULLPTR;
};

/// \brief EXPERIMENTAL: A non-owning ArrayData reference that is cheaply
/// copyable and does not contain any shared_ptr objects. Do not use in public
/// APIs aside from compute kernels for now
struct ARROW_EXPORT ArraySpan {
  const DataType* type = NULLPTR;
  int64_t length = 0;
  mutable int64_t null_count = kUnknownNullCount;
  int64_t offset = 0;
  BufferSpan buffers[3];

  // 16 bytes of scratch space to enable this ArraySpan to be a view onto
  // scalar values including binary scalars (where we need to create a buffer
  // that looks like two 32-bit or 64-bit offsets)
  uint64_t scratch_space[2];

  ArraySpan() = default;

  explicit ArraySpan(const DataType* type, int64_t length) : type(type), length(length) {}

  ArraySpan(const ArrayData& data) {  // NOLINT implicit conversion
    SetMembers(data);
  }
  explicit ArraySpan(const Scalar& data) { FillFromScalar(data); }

  /// If dictionary-encoded, put dictionary in the first entry
  std::vector<ArraySpan> child_data;

  /// \brief Populate ArraySpan to look like an array of length 1 pointing at
  /// the data members of a Scalar value
  void FillFromScalar(const Scalar& value);

  void SetMembers(const ArrayData& data);

  void SetBuffer(int index, const std::shared_ptr<Buffer>& buffer) {
    this->buffers[index].data = const_cast<uint8_t*>(buffer->data());
    this->buffers[index].size = buffer->size();
    this->buffers[index].owner = &buffer;
  }

  const ArraySpan& dictionary() const { return child_data[0]; }

  /// \brief Return the number of buffers (out of 3) that are used to
  /// constitute this array
  int num_buffers() const;

  // Access a buffer's data as a typed C pointer
  template <typename T>
  inline T* GetValues(int i, int64_t absolute_offset) {
    return reinterpret_cast<T*>(buffers[i].data) + absolute_offset;
  }

  template <typename T>
  inline T* GetValues(int i) {
    return GetValues<T>(i, this->offset);
  }

  // Access a buffer's data as a typed C pointer
  template <typename T>
  inline const T* GetValues(int i, int64_t absolute_offset) const {
    return reinterpret_cast<const T*>(buffers[i].data) + absolute_offset;
  }

  template <typename T>
  inline const T* GetValues(int i) const {
    return GetValues<T>(i, this->offset);
  }

  inline bool IsValid(int64_t i) const {
    return ((this->buffers[0].data != NULLPTR)
                ? bit_util::GetBit(this->buffers[0].data, i + this->offset)
                : this->null_count != this->length);
  }

  inline bool IsNull(int64_t i) const { return !IsValid(i); }

  std::shared_ptr<ArrayData> ToArrayData() const;

  std::shared_ptr<Array> ToArray() const;

  std::shared_ptr<Buffer> GetBuffer(int index) const {
    const BufferSpan& buf = this->buffers[index];
    if (buf.owner) {
      return *buf.owner;
    } else if (buf.data != NULLPTR) {
      // Buffer points to some memory without an owning buffer
      return std::make_shared<Buffer>(buf.data, buf.size);
    } else {
      return NULLPTR;
    }
  }

  void SetSlice(int64_t offset, int64_t length) {
    this->offset = offset;
    this->length = length;
    if (this->type->id() != Type::NA) {
      this->null_count = kUnknownNullCount;
    } else {
      this->null_count = this->length;
    }
  }

  /// \brief Return null count, or compute and set it if it's not known
  int64_t GetNullCount() const;

  bool MayHaveNulls() const {
    // If an ArrayData is slightly malformed it may have kUnknownNullCount set
    // but no buffer
    return null_count != 0 && buffers[0].data != NULLPTR;
  }
};

namespace internal {

void FillZeroLengthArray(const DataType* type, ArraySpan* span);

/// Construct a zero-copy view of this ArrayData with the given type.
///
/// This method checks if the types are layout-compatible.
/// Nested types are traversed in depth-first order. Data buffers must have
/// the same item sizes, even though the logical types may be different.
/// An error is returned if the types are not layout-compatible.
ARROW_EXPORT
Result<std::shared_ptr<ArrayData>> GetArrayView(const std::shared_ptr<ArrayData>& data,
                                                const std::shared_ptr<DataType>& type);

}  // namespace internal
}  // namespace arrow

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
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include "arrow/buffer.h"
#include "arrow/status.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/bitmap_generate.h"
#include "arrow/util/bitmap_ops.h"
#include "arrow/util/macros.h"
#include "arrow/util/ubsan.h"
#include "arrow/util/visibility.h"

namespace arrow {

// ----------------------------------------------------------------------
// Buffer builder classes

/// \class BufferBuilder
/// \brief A class for incrementally building a contiguous chunk of in-memory
/// data
class ARROW_EXPORT BufferBuilder {
 public:
  explicit BufferBuilder(MemoryPool* pool = default_memory_pool(),
                         int64_t alignment = kDefaultBufferAlignment)
      : pool_(pool),
        data_(/*ensure never null to make ubsan happy and avoid check penalties below*/
              util::MakeNonNull<uint8_t>()),
        capacity_(0),
        size_(0),
        alignment_(alignment) {}

  /// \brief Constructs new Builder that will start using
  /// the provided buffer until Finish/Reset are called.
  /// The buffer is not resized.
  explicit BufferBuilder(std::shared_ptr<ResizableBuffer> buffer,
                         MemoryPool* pool = default_memory_pool(),
                         int64_t alignment = kDefaultBufferAlignment)
      : buffer_(std::move(buffer)),
        pool_(pool),
        data_(buffer_->mutable_data()),
        capacity_(buffer_->capacity()),
        size_(buffer_->size()),
        alignment_(alignment) {}

  /// \brief Resize the buffer to the nearest multiple of 64 bytes
  ///
  /// \param new_capacity the new capacity of the of the builder. Will be
  /// rounded up to a multiple of 64 bytes for padding
  /// \param shrink_to_fit if new capacity is smaller than the existing,
  /// reallocate internal buffer. Set to false to avoid reallocations when
  /// shrinking the builder.
  /// \return Status
  Status Resize(const int64_t new_capacity, bool shrink_to_fit = true) {
    if (buffer_ == NULLPTR) {
      ARROW_ASSIGN_OR_RAISE(buffer_,
                            AllocateResizableBuffer(new_capacity, alignment_, pool_));
    } else {
      ARROW_RETURN_NOT_OK(buffer_->Resize(new_capacity, shrink_to_fit));
    }
    capacity_ = buffer_->capacity();
    data_ = buffer_->mutable_data();
    return Status::OK();
  }

  /// \brief Ensure that builder can accommodate the additional number of bytes
  /// without the need to perform allocations
  ///
  /// \param[in] additional_bytes number of additional bytes to make space for
  /// \return Status
  Status Reserve(const int64_t additional_bytes) {
    auto min_capacity = size_ + additional_bytes;
    if (min_capacity <= capacity_) {
      return Status::OK();
    }
    return Resize(GrowByFactor(capacity_, min_capacity), false);
  }

  /// \brief Return a capacity expanded by the desired growth factor
  static int64_t GrowByFactor(int64_t current_capacity, int64_t new_capacity) {
    // Doubling capacity except for large Reserve requests. 2x growth strategy
    // (versus 1.5x) seems to have slightly better performance when using
    // jemalloc, but significantly better performance when using the system
    // allocator. See ARROW-6450 for further discussion
    return std::max(new_capacity, current_capacity * 2);
  }

  /// \brief Append the given data to the buffer
  ///
  /// The buffer is automatically expanded if necessary.
  Status Append(const void* data, const int64_t length) {
    if (ARROW_PREDICT_FALSE(size_ + length > capacity_)) {
      ARROW_RETURN_NOT_OK(Resize(GrowByFactor(capacity_, size_ + length), false));
    }
    UnsafeAppend(data, length);
    return Status::OK();
  }

  /// \brief Append copies of a value to the buffer
  ///
  /// The buffer is automatically expanded if necessary.
  Status Append(const int64_t num_copies, uint8_t value) {
    ARROW_RETURN_NOT_OK(Reserve(num_copies));
    UnsafeAppend(num_copies, value);
    return Status::OK();
  }

  // Advance pointer and zero out memory
  Status Advance(const int64_t length) { return Append(length, 0); }

  // Advance pointer, but don't allocate or zero memory
  void UnsafeAdvance(const int64_t length) { size_ += length; }

  // Unsafe methods don't check existing size
  void UnsafeAppend(const void* data, const int64_t length) {
    memcpy(data_ + size_, data, static_cast<size_t>(length));
    size_ += length;
  }

  void UnsafeAppend(const int64_t num_copies, uint8_t value) {
    memset(data_ + size_, value, static_cast<size_t>(num_copies));
    size_ += num_copies;
  }

  /// \brief Return result of builder as a Buffer object.
  ///
  /// The builder is reset and can be reused afterwards.
  ///
  /// \param[out] out the finalized Buffer object
  /// \param shrink_to_fit if the buffer size is smaller than its capacity,
  /// reallocate to fit more tightly in memory. Set to false to avoid
  /// a reallocation, at the expense of potentially more memory consumption.
  /// \return Status
  Status Finish(std::shared_ptr<Buffer>* out, bool shrink_to_fit = true) {
    ARROW_RETURN_NOT_OK(Resize(size_, shrink_to_fit));
    if (size_ != 0) buffer_->ZeroPadding();
    *out = buffer_;
    if (*out == NULLPTR) {
      ARROW_ASSIGN_OR_RAISE(*out, AllocateBuffer(0, alignment_, pool_));
    }
    Reset();
    return Status::OK();
  }

  Result<std::shared_ptr<Buffer>> Finish(bool shrink_to_fit = true) {
    std::shared_ptr<Buffer> out;
    ARROW_RETURN_NOT_OK(Finish(&out, shrink_to_fit));
    return out;
  }

  /// \brief Like Finish, but override the final buffer size
  ///
  /// This is useful after writing data directly into the builder memory
  /// without calling the Append methods (basically, when using BufferBuilder
  /// mostly for memory allocation).
  Result<std::shared_ptr<Buffer>> FinishWithLength(int64_t final_length,
                                                   bool shrink_to_fit = true) {
    size_ = final_length;
    return Finish(shrink_to_fit);
  }

  void Reset() {
    buffer_ = NULLPTR;
    capacity_ = size_ = 0;
  }

  /// \brief Set size to a smaller value without modifying builder
  /// contents. For reusable BufferBuilder classes
  /// \param[in] position must be non-negative and less than or equal
  /// to the current length()
  void Rewind(int64_t position) { size_ = position; }

  int64_t capacity() const { return capacity_; }
  int64_t length() const { return size_; }
  const uint8_t* data() const { return data_; }
  uint8_t* mutable_data() { return data_; }

 private:
  std::shared_ptr<ResizableBuffer> buffer_;
  MemoryPool* pool_;
  uint8_t* data_;
  int64_t capacity_;
  int64_t size_;
  int64_t alignment_;
};

template <typename T, typename Enable = void>
class TypedBufferBuilder;

/// \brief A BufferBuilder for building a buffer of arithmetic elements
template <typename T>
class TypedBufferBuilder<
    T, typename std::enable_if<std::is_arithmetic<T>::value ||
                               std::is_standard_layout<T>::value>::type> {
 public:
  explicit TypedBufferBuilder(MemoryPool* pool = default_memory_pool(),
                              int64_t alignment = kDefaultBufferAlignment)
      : bytes_builder_(pool, alignment) {}

  explicit TypedBufferBuilder(std::shared_ptr<ResizableBuffer> buffer,
                              MemoryPool* pool = default_memory_pool())
      : bytes_builder_(std::move(buffer), pool) {}

  explicit TypedBufferBuilder(BufferBuilder builder)
      : bytes_builder_(std::move(builder)) {}

  BufferBuilder* bytes_builder() { return &bytes_builder_; }

  Status Append(T value) {
    return bytes_builder_.Append(reinterpret_cast<uint8_t*>(&value), sizeof(T));
  }

  Status Append(const T* values, int64_t num_elements) {
    return bytes_builder_.Append(reinterpret_cast<const uint8_t*>(values),
                                 num_elements * sizeof(T));
  }

  Status Append(const int64_t num_copies, T value) {
    ARROW_RETURN_NOT_OK(Reserve(num_copies + length()));
    UnsafeAppend(num_copies, value);
    return Status::OK();
  }

  void UnsafeAppend(T value) {
    bytes_builder_.UnsafeAppend(reinterpret_cast<uint8_t*>(&value), sizeof(T));
  }

  void UnsafeAppend(const T* values, int64_t num_elements) {
    bytes_builder_.UnsafeAppend(reinterpret_cast<const uint8_t*>(values),
                                num_elements * sizeof(T));
  }

  template <typename Iter>
  void UnsafeAppend(Iter values_begin, Iter values_end) {
    int64_t num_elements = static_cast<int64_t>(std::distance(values_begin, values_end));
    auto data = mutable_data() + length();
    bytes_builder_.UnsafeAdvance(num_elements * sizeof(T));
    std::copy(values_begin, values_end, data);
  }

  void UnsafeAppend(const int64_t num_copies, T value) {
    auto data = mutable_data() + length();
    bytes_builder_.UnsafeAdvance(num_copies * sizeof(T));
    std::fill(data, data + num_copies, value);
  }

  Status Resize(const int64_t new_capacity, bool shrink_to_fit = true) {
    return bytes_builder_.Resize(new_capacity * sizeof(T), shrink_to_fit);
  }

  Status Reserve(const int64_t additional_elements) {
    return bytes_builder_.Reserve(additional_elements * sizeof(T));
  }

  Status Advance(const int64_t length) {
    return bytes_builder_.Advance(length * sizeof(T));
  }

  Status Finish(std::shared_ptr<Buffer>* out, bool shrink_to_fit = true) {
    return bytes_builder_.Finish(out, shrink_to_fit);
  }

  Result<std::shared_ptr<Buffer>> Finish(bool shrink_to_fit = true) {
    std::shared_ptr<Buffer> out;
    ARROW_RETURN_NOT_OK(Finish(&out, shrink_to_fit));
    return out;
  }

  /// \brief Like Finish, but override the final buffer size
  ///
  /// This is useful after writing data directly into the builder memory
  /// without calling the Append methods (basically, when using TypedBufferBuilder
  /// only for memory allocation).
  Result<std::shared_ptr<Buffer>> FinishWithLength(int64_t final_length,
                                                   bool shrink_to_fit = true) {
    return bytes_builder_.FinishWithLength(final_length * sizeof(T), shrink_to_fit);
  }

  void Reset() { bytes_builder_.Reset(); }

  int64_t length() const { return bytes_builder_.length() / sizeof(T); }
  int64_t capacity() const { return bytes_builder_.capacity() / sizeof(T); }
  const T* data() const { return reinterpret_cast<const T*>(bytes_builder_.data()); }
  T* mutable_data() { return reinterpret_cast<T*>(bytes_builder_.mutable_data()); }

 private:
  BufferBuilder bytes_builder_;
};

/// \brief A BufferBuilder for building a buffer containing a bitmap
template <>
class TypedBufferBuilder<bool> {
 public:
  explicit TypedBufferBuilder(MemoryPool* pool = default_memory_pool(),
                              int64_t alignment = kDefaultBufferAlignment)
      : bytes_builder_(pool, alignment) {}

  explicit TypedBufferBuilder(BufferBuilder builder)
      : bytes_builder_(std::move(builder)) {}

  BufferBuilder* bytes_builder() { return &bytes_builder_; }

  Status Append(bool value) {
    ARROW_RETURN_NOT_OK(Reserve(1));
    UnsafeAppend(value);
    return Status::OK();
  }

  Status Append(const uint8_t* valid_bytes, int64_t num_elements) {
    ARROW_RETURN_NOT_OK(Reserve(num_elements));
    UnsafeAppend(valid_bytes, num_elements);
    return Status::OK();
  }

  Status Append(const int64_t num_copies, bool value) {
    ARROW_RETURN_NOT_OK(Reserve(num_copies));
    UnsafeAppend(num_copies, value);
    return Status::OK();
  }

  void UnsafeAppend(bool value) {
    bit_util::SetBitTo(mutable_data(), bit_length_, value);
    if (!value) {
      ++false_count_;
    }
    ++bit_length_;
  }

  /// \brief Append bits from an array of bytes (one value per byte)
  void UnsafeAppend(const uint8_t* bytes, int64_t num_elements) {
    if (num_elements == 0) return;
    int64_t i = 0;
    internal::GenerateBitsUnrolled(mutable_data(), bit_length_, num_elements, [&] {
      bool value = bytes[i++];
      false_count_ += !value;
      return value;
    });
    bit_length_ += num_elements;
  }

  /// \brief Append bits from a packed bitmap
  void UnsafeAppend(const uint8_t* bitmap, int64_t offset, int64_t num_elements) {
    if (num_elements == 0) return;
    internal::CopyBitmap(bitmap, offset, num_elements, mutable_data(), bit_length_);
    false_count_ += num_elements - internal::CountSetBits(bitmap, offset, num_elements);
    bit_length_ += num_elements;
  }

  void UnsafeAppend(const int64_t num_copies, bool value) {
    bit_util::SetBitsTo(mutable_data(), bit_length_, num_copies, value);
    false_count_ += num_copies * !value;
    bit_length_ += num_copies;
  }

  template <bool count_falses, typename Generator>
  void UnsafeAppend(const int64_t num_elements, Generator&& gen) {
    if (num_elements == 0) return;

    if (count_falses) {
      internal::GenerateBitsUnrolled(mutable_data(), bit_length_, num_elements, [&] {
        bool value = gen();
        false_count_ += !value;
        return value;
      });
    } else {
      internal::GenerateBitsUnrolled(mutable_data(), bit_length_, num_elements,
                                     std::forward<Generator>(gen));
    }
    bit_length_ += num_elements;
  }

  Status Resize(const int64_t new_capacity, bool shrink_to_fit = true) {
    const int64_t old_byte_capacity = bytes_builder_.capacity();
    ARROW_RETURN_NOT_OK(
        bytes_builder_.Resize(bit_util::BytesForBits(new_capacity), shrink_to_fit));
    // Resize() may have chosen a larger capacity (e.g. for padding),
    // so ask it again before calling memset().
    const int64_t new_byte_capacity = bytes_builder_.capacity();
    if (new_byte_capacity > old_byte_capacity) {
      // The additional buffer space is 0-initialized for convenience,
      // so that other methods can simply bump the length.
      memset(mutable_data() + old_byte_capacity, 0,
             static_cast<size_t>(new_byte_capacity - old_byte_capacity));
    }
    return Status::OK();
  }

  Status Reserve(const int64_t additional_elements) {
    return Resize(
        BufferBuilder::GrowByFactor(bit_length_, bit_length_ + additional_elements),
        false);
  }

  Status Advance(const int64_t length) {
    ARROW_RETURN_NOT_OK(Reserve(length));
    bit_length_ += length;
    false_count_ += length;
    return Status::OK();
  }

  Status Finish(std::shared_ptr<Buffer>* out, bool shrink_to_fit = true) {
    // set bytes_builder_.size_ == byte size of data
    bytes_builder_.UnsafeAdvance(bit_util::BytesForBits(bit_length_) -
                                 bytes_builder_.length());
    bit_length_ = false_count_ = 0;
    return bytes_builder_.Finish(out, shrink_to_fit);
  }

  Result<std::shared_ptr<Buffer>> Finish(bool shrink_to_fit = true) {
    std::shared_ptr<Buffer> out;
    ARROW_RETURN_NOT_OK(Finish(&out, shrink_to_fit));
    return out;
  }

  /// \brief Like Finish, but override the final buffer size
  ///
  /// This is useful after writing data directly into the builder memory
  /// without calling the Append methods (basically, when using TypedBufferBuilder
  /// only for memory allocation).
  Result<std::shared_ptr<Buffer>> FinishWithLength(int64_t final_length,
                                                   bool shrink_to_fit = true) {
    const auto final_byte_length = bit_util::BytesForBits(final_length);
    bytes_builder_.UnsafeAdvance(final_byte_length - bytes_builder_.length());
    bit_length_ = false_count_ = 0;
    return bytes_builder_.FinishWithLength(final_byte_length, shrink_to_fit);
  }

  void Reset() {
    bytes_builder_.Reset();
    bit_length_ = false_count_ = 0;
  }

  int64_t length() const { return bit_length_; }
  int64_t capacity() const { return bytes_builder_.capacity() * 8; }
  const uint8_t* data() const { return bytes_builder_.data(); }
  uint8_t* mutable_data() { return bytes_builder_.mutable_data(); }
  int64_t false_count() const { return false_count_; }

 private:
  BufferBuilder bytes_builder_;
  int64_t bit_length_ = 0;
  int64_t false_count_ = 0;
};

}  // namespace arrow

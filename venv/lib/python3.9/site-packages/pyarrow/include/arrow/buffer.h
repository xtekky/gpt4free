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
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "arrow/device.h"
#include "arrow/status.h"
#include "arrow/type_fwd.h"
#include "arrow/util/bytes_view.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {

// ----------------------------------------------------------------------
// Buffer classes

/// \class Buffer
/// \brief Object containing a pointer to a piece of contiguous memory with a
/// particular size.
///
/// Buffers have two related notions of length: size and capacity. Size is
/// the number of bytes that might have valid data. Capacity is the number
/// of bytes that were allocated for the buffer in total.
///
/// The Buffer base class does not own its memory, but subclasses often do.
///
/// The following invariant is always true: Size <= Capacity
class ARROW_EXPORT Buffer {
 public:
  /// \brief Construct from buffer and size without copying memory
  ///
  /// \param[in] data a memory buffer
  /// \param[in] size buffer size
  ///
  /// \note The passed memory must be kept alive through some other means
  Buffer(const uint8_t* data, int64_t size)
      : is_mutable_(false), is_cpu_(true), data_(data), size_(size), capacity_(size) {
    SetMemoryManager(default_cpu_memory_manager());
  }

  Buffer(const uint8_t* data, int64_t size, std::shared_ptr<MemoryManager> mm,
         std::shared_ptr<Buffer> parent = NULLPTR)
      : is_mutable_(false), data_(data), size_(size), capacity_(size), parent_(parent) {
    SetMemoryManager(std::move(mm));
  }

  Buffer(uintptr_t address, int64_t size, std::shared_ptr<MemoryManager> mm,
         std::shared_ptr<Buffer> parent = NULLPTR)
      : Buffer(reinterpret_cast<const uint8_t*>(address), size, std::move(mm),
               std::move(parent)) {}

  /// \brief Construct from string_view without copying memory
  ///
  /// \param[in] data a string_view object
  ///
  /// \note The memory viewed by data must not be deallocated in the lifetime of the
  /// Buffer; temporary rvalue strings must be stored in an lvalue somewhere
  explicit Buffer(std::string_view data)
      : Buffer(reinterpret_cast<const uint8_t*>(data.data()),
               static_cast<int64_t>(data.size())) {}

  virtual ~Buffer() = default;

  /// An offset into data that is owned by another buffer, but we want to be
  /// able to retain a valid pointer to it even after other shared_ptr's to the
  /// parent buffer have been destroyed
  ///
  /// This method makes no assertions about alignment or padding of the buffer but
  /// in general we expected buffers to be aligned and padded to 64 bytes.  In the future
  /// we might add utility methods to help determine if a buffer satisfies this contract.
  Buffer(const std::shared_ptr<Buffer>& parent, const int64_t offset, const int64_t size)
      : Buffer(parent->data_ + offset, size) {
    parent_ = parent;
    SetMemoryManager(parent->memory_manager_);
  }

  uint8_t operator[](std::size_t i) const { return data_[i]; }

  /// \brief Construct a new std::string with a hexadecimal representation of the buffer.
  /// \return std::string
  std::string ToHexString();

  /// Return true if both buffers are the same size and contain the same bytes
  /// up to the number of compared bytes
  bool Equals(const Buffer& other, int64_t nbytes) const;

  /// Return true if both buffers are the same size and contain the same bytes
  bool Equals(const Buffer& other) const;

  /// Copy a section of the buffer into a new Buffer.
  Result<std::shared_ptr<Buffer>> CopySlice(
      const int64_t start, const int64_t nbytes,
      MemoryPool* pool = default_memory_pool()) const;

  /// Zero bytes in padding, i.e. bytes between size_ and capacity_.
  void ZeroPadding() {
#ifndef NDEBUG
    CheckMutable();
#endif
    // A zero-capacity buffer can have a null data pointer
    if (capacity_ != 0) {
      memset(mutable_data() + size_, 0, static_cast<size_t>(capacity_ - size_));
    }
  }

  /// \brief Construct an immutable buffer that takes ownership of the contents
  /// of an std::string (without copying it).
  ///
  /// \param[in] data a string to own
  /// \return a new Buffer instance
  static std::shared_ptr<Buffer> FromString(std::string data);

  /// \brief Create buffer referencing typed memory with some length without
  /// copying
  /// \param[in] data the typed memory as C array
  /// \param[in] length the number of values in the array
  /// \return a new shared_ptr<Buffer>
  template <typename T, typename SizeType = int64_t>
  static std::shared_ptr<Buffer> Wrap(const T* data, SizeType length) {
    return std::make_shared<Buffer>(reinterpret_cast<const uint8_t*>(data),
                                    static_cast<int64_t>(sizeof(T) * length));
  }

  /// \brief Create buffer referencing std::vector with some length without
  /// copying
  /// \param[in] data the vector to be referenced. If this vector is changed,
  /// the buffer may become invalid
  /// \return a new shared_ptr<Buffer>
  template <typename T>
  static std::shared_ptr<Buffer> Wrap(const std::vector<T>& data) {
    return std::make_shared<Buffer>(reinterpret_cast<const uint8_t*>(data.data()),
                                    static_cast<int64_t>(sizeof(T) * data.size()));
  }

  /// \brief Copy buffer contents into a new std::string
  /// \return std::string
  /// \note Can throw std::bad_alloc if buffer is large
  std::string ToString() const;

  /// \brief View buffer contents as a std::string_view
  /// \return std::string_view
  explicit operator std::string_view() const {
    return std::string_view(reinterpret_cast<const char*>(data_), size_);
  }

  /// \brief View buffer contents as a util::bytes_view
  /// \return util::bytes_view
  explicit operator util::bytes_view() const { return util::bytes_view(data_, size_); }

  /// \brief Return a pointer to the buffer's data
  ///
  /// The buffer has to be a CPU buffer (`is_cpu()` is true).
  /// Otherwise, an assertion may be thrown or a null pointer may be returned.
  ///
  /// To get the buffer's data address regardless of its device, call `address()`.
  const uint8_t* data() const {
#ifndef NDEBUG
    CheckCPU();
#endif
    return ARROW_PREDICT_TRUE(is_cpu_) ? data_ : NULLPTR;
  }

  /// \brief Return a writable pointer to the buffer's data
  ///
  /// The buffer has to be a mutable CPU buffer (`is_cpu()` and `is_mutable()`
  /// are true).  Otherwise, an assertion may be thrown or a null pointer may
  /// be returned.
  ///
  /// To get the buffer's mutable data address regardless of its device, call
  /// `mutable_address()`.
  uint8_t* mutable_data() {
#ifndef NDEBUG
    CheckCPU();
    CheckMutable();
#endif
    return ARROW_PREDICT_TRUE(is_cpu_ && is_mutable_) ? const_cast<uint8_t*>(data_)
                                                      : NULLPTR;
  }

  /// \brief Return the device address of the buffer's data
  uintptr_t address() const { return reinterpret_cast<uintptr_t>(data_); }

  /// \brief Return a writable device address to the buffer's data
  ///
  /// The buffer has to be a mutable buffer (`is_mutable()` is true).
  /// Otherwise, an assertion may be thrown or 0 may be returned.
  uintptr_t mutable_address() const {
#ifndef NDEBUG
    CheckMutable();
#endif
    return ARROW_PREDICT_TRUE(is_mutable_) ? reinterpret_cast<uintptr_t>(data_) : 0;
  }

  /// \brief Return the buffer's size in bytes
  int64_t size() const { return size_; }

  /// \brief Return the buffer's capacity (number of allocated bytes)
  int64_t capacity() const { return capacity_; }

  /// \brief Whether the buffer is directly CPU-accessible
  ///
  /// If this function returns true, you can read directly from the buffer's
  /// `data()` pointer.  Otherwise, you'll have to `View()` or `Copy()` it.
  bool is_cpu() const { return is_cpu_; }

  /// \brief Whether the buffer is mutable
  ///
  /// If this function returns true, you are allowed to modify buffer contents
  /// using the pointer returned by `mutable_data()` or `mutable_address()`.
  bool is_mutable() const { return is_mutable_; }

  const std::shared_ptr<Device>& device() const { return memory_manager_->device(); }

  const std::shared_ptr<MemoryManager>& memory_manager() const { return memory_manager_; }

  std::shared_ptr<Buffer> parent() const { return parent_; }

  /// \brief Get a RandomAccessFile for reading a buffer
  ///
  /// The returned file object reads from this buffer's underlying memory.
  static Result<std::shared_ptr<io::RandomAccessFile>> GetReader(std::shared_ptr<Buffer>);

  /// \brief Get a OutputStream for writing to a buffer
  ///
  /// The buffer must be mutable.  The returned stream object writes into the buffer's
  /// underlying memory (but it won't resize it).
  static Result<std::shared_ptr<io::OutputStream>> GetWriter(std::shared_ptr<Buffer>);

  /// \brief Copy buffer
  ///
  /// The buffer contents will be copied into a new buffer allocated by the
  /// given MemoryManager.  This function supports cross-device copies.
  static Result<std::shared_ptr<Buffer>> Copy(std::shared_ptr<Buffer> source,
                                              const std::shared_ptr<MemoryManager>& to);

  /// \brief Copy a non-owned buffer
  ///
  /// This is useful for cases where the source memory area is externally managed
  /// (its lifetime not tied to the source Buffer), otherwise please use Copy().
  static Result<std::unique_ptr<Buffer>> CopyNonOwned(
      const Buffer& source, const std::shared_ptr<MemoryManager>& to);

  /// \brief View buffer
  ///
  /// Return a Buffer that reflects this buffer, seen potentially from another
  /// device, without making an explicit copy of the contents.  The underlying
  /// mechanism is typically implemented by the kernel or device driver, and may
  /// involve lazy caching of parts of the buffer contents on the destination
  /// device's memory.
  ///
  /// If a non-copy view is unsupported for the buffer on the given device,
  /// nullptr is returned.  An error can be returned if some low-level
  /// operation fails (such as an out-of-memory condition).
  static Result<std::shared_ptr<Buffer>> View(std::shared_ptr<Buffer> source,
                                              const std::shared_ptr<MemoryManager>& to);

  /// \brief View or copy buffer
  ///
  /// Try to view buffer contents on the given MemoryManager's device, but
  /// fall back to copying if a no-copy view isn't supported.
  static Result<std::shared_ptr<Buffer>> ViewOrCopy(
      std::shared_ptr<Buffer> source, const std::shared_ptr<MemoryManager>& to);

 protected:
  bool is_mutable_;
  bool is_cpu_;
  const uint8_t* data_;
  int64_t size_;
  int64_t capacity_;

  // null by default, but may be set
  std::shared_ptr<Buffer> parent_;

 private:
  // private so that subclasses are forced to call SetMemoryManager()
  std::shared_ptr<MemoryManager> memory_manager_;

 protected:
  void CheckMutable() const;
  void CheckCPU() const;

  void SetMemoryManager(std::shared_ptr<MemoryManager> mm) {
    memory_manager_ = std::move(mm);
    is_cpu_ = memory_manager_->is_cpu();
  }

 private:
  Buffer() = delete;
  ARROW_DISALLOW_COPY_AND_ASSIGN(Buffer);
};

/// \defgroup buffer-slicing-functions Functions for slicing buffers
///
/// @{

/// \brief Construct a view on a buffer at the given offset and length.
///
/// This function cannot fail and does not check for errors (except in debug builds)
static inline std::shared_ptr<Buffer> SliceBuffer(const std::shared_ptr<Buffer>& buffer,
                                                  const int64_t offset,
                                                  const int64_t length) {
  return std::make_shared<Buffer>(buffer, offset, length);
}

/// \brief Construct a view on a buffer at the given offset, up to the buffer's end.
///
/// This function cannot fail and does not check for errors (except in debug builds)
static inline std::shared_ptr<Buffer> SliceBuffer(const std::shared_ptr<Buffer>& buffer,
                                                  const int64_t offset) {
  int64_t length = buffer->size() - offset;
  return SliceBuffer(buffer, offset, length);
}

/// \brief Input-checking version of SliceBuffer
///
/// An Invalid Status is returned if the requested slice falls out of bounds.
ARROW_EXPORT
Result<std::shared_ptr<Buffer>> SliceBufferSafe(const std::shared_ptr<Buffer>& buffer,
                                                int64_t offset);
/// \brief Input-checking version of SliceBuffer
///
/// An Invalid Status is returned if the requested slice falls out of bounds.
/// Note that unlike SliceBuffer, `length` isn't clamped to the available buffer size.
ARROW_EXPORT
Result<std::shared_ptr<Buffer>> SliceBufferSafe(const std::shared_ptr<Buffer>& buffer,
                                                int64_t offset, int64_t length);

/// \brief Like SliceBuffer, but construct a mutable buffer slice.
///
/// If the parent buffer is not mutable, behavior is undefined (it may abort
/// in debug builds).
ARROW_EXPORT
std::shared_ptr<Buffer> SliceMutableBuffer(const std::shared_ptr<Buffer>& buffer,
                                           const int64_t offset, const int64_t length);

/// \brief Like SliceBuffer, but construct a mutable buffer slice.
///
/// If the parent buffer is not mutable, behavior is undefined (it may abort
/// in debug builds).
static inline std::shared_ptr<Buffer> SliceMutableBuffer(
    const std::shared_ptr<Buffer>& buffer, const int64_t offset) {
  int64_t length = buffer->size() - offset;
  return SliceMutableBuffer(buffer, offset, length);
}

/// \brief Input-checking version of SliceMutableBuffer
///
/// An Invalid Status is returned if the requested slice falls out of bounds.
ARROW_EXPORT
Result<std::shared_ptr<Buffer>> SliceMutableBufferSafe(
    const std::shared_ptr<Buffer>& buffer, int64_t offset);
/// \brief Input-checking version of SliceMutableBuffer
///
/// An Invalid Status is returned if the requested slice falls out of bounds.
/// Note that unlike SliceBuffer, `length` isn't clamped to the available buffer size.
ARROW_EXPORT
Result<std::shared_ptr<Buffer>> SliceMutableBufferSafe(
    const std::shared_ptr<Buffer>& buffer, int64_t offset, int64_t length);

/// @}

/// \class MutableBuffer
/// \brief A Buffer whose contents can be mutated. May or may not own its data.
class ARROW_EXPORT MutableBuffer : public Buffer {
 public:
  MutableBuffer(uint8_t* data, const int64_t size) : Buffer(data, size) {
    is_mutable_ = true;
  }

  MutableBuffer(uint8_t* data, const int64_t size, std::shared_ptr<MemoryManager> mm)
      : Buffer(data, size, std::move(mm)) {
    is_mutable_ = true;
  }

  MutableBuffer(const std::shared_ptr<Buffer>& parent, const int64_t offset,
                const int64_t size);

  /// \brief Create buffer referencing typed memory with some length
  /// \param[in] data the typed memory as C array
  /// \param[in] length the number of values in the array
  /// \return a new shared_ptr<Buffer>
  template <typename T, typename SizeType = int64_t>
  static std::shared_ptr<Buffer> Wrap(T* data, SizeType length) {
    return std::make_shared<MutableBuffer>(reinterpret_cast<uint8_t*>(data),
                                           static_cast<int64_t>(sizeof(T) * length));
  }

 protected:
  MutableBuffer() : Buffer(NULLPTR, 0) {}
};

/// \class ResizableBuffer
/// \brief A mutable buffer that can be resized
class ARROW_EXPORT ResizableBuffer : public MutableBuffer {
 public:
  /// Change buffer reported size to indicated size, allocating memory if
  /// necessary.  This will ensure that the capacity of the buffer is a multiple
  /// of 64 bytes as defined in Layout.md.
  /// Consider using ZeroPadding afterwards, to conform to the Arrow layout
  /// specification.
  ///
  /// @param new_size The new size for the buffer.
  /// @param shrink_to_fit Whether to shrink the capacity if new size < current size
  virtual Status Resize(const int64_t new_size, bool shrink_to_fit) = 0;
  Status Resize(const int64_t new_size) {
    return Resize(new_size, /*shrink_to_fit=*/true);
  }

  /// Ensure that buffer has enough memory allocated to fit the indicated
  /// capacity (and meets the 64 byte padding requirement in Layout.md).
  /// It does not change buffer's reported size and doesn't zero the padding.
  virtual Status Reserve(const int64_t new_capacity) = 0;

  template <class T>
  Status TypedResize(const int64_t new_nb_elements, bool shrink_to_fit = true) {
    return Resize(sizeof(T) * new_nb_elements, shrink_to_fit);
  }

  template <class T>
  Status TypedReserve(const int64_t new_nb_elements) {
    return Reserve(sizeof(T) * new_nb_elements);
  }

 protected:
  ResizableBuffer(uint8_t* data, int64_t size) : MutableBuffer(data, size) {}
  ResizableBuffer(uint8_t* data, int64_t size, std::shared_ptr<MemoryManager> mm)
      : MutableBuffer(data, size, std::move(mm)) {}
};

/// \defgroup buffer-allocation-functions Functions for allocating buffers
///
/// @{

/// \brief Allocate a fixed size mutable buffer from a memory pool, zero its padding.
///
/// \param[in] size size of buffer to allocate
/// \param[in] pool a memory pool
ARROW_EXPORT
Result<std::unique_ptr<Buffer>> AllocateBuffer(const int64_t size,
                                               MemoryPool* pool = NULLPTR);
ARROW_EXPORT
Result<std::unique_ptr<Buffer>> AllocateBuffer(const int64_t size, int64_t alignment,
                                               MemoryPool* pool = NULLPTR);

/// \brief Allocate a resizeable buffer from a memory pool, zero its padding.
///
/// \param[in] size size of buffer to allocate
/// \param[in] pool a memory pool
ARROW_EXPORT
Result<std::unique_ptr<ResizableBuffer>> AllocateResizableBuffer(
    const int64_t size, MemoryPool* pool = NULLPTR);
ARROW_EXPORT
Result<std::unique_ptr<ResizableBuffer>> AllocateResizableBuffer(
    const int64_t size, const int64_t alignment, MemoryPool* pool = NULLPTR);

/// \brief Allocate a bitmap buffer from a memory pool
/// no guarantee on values is provided.
///
/// \param[in] length size in bits of bitmap to allocate
/// \param[in] pool memory pool to allocate memory from
ARROW_EXPORT
Result<std::shared_ptr<Buffer>> AllocateBitmap(int64_t length,
                                               MemoryPool* pool = NULLPTR);

/// \brief Allocate a zero-initialized bitmap buffer from a memory pool
///
/// \param[in] length size in bits of bitmap to allocate
/// \param[in] pool memory pool to allocate memory from
ARROW_EXPORT
Result<std::shared_ptr<Buffer>> AllocateEmptyBitmap(int64_t length,
                                                    MemoryPool* pool = NULLPTR);

ARROW_EXPORT
Result<std::shared_ptr<Buffer>> AllocateEmptyBitmap(int64_t length, int64_t alignment,
                                                    MemoryPool* pool = NULLPTR);

/// \brief Concatenate multiple buffers into a single buffer
///
/// \param[in] buffers to be concatenated
/// \param[in] pool memory pool to allocate the new buffer from
ARROW_EXPORT
Result<std::shared_ptr<Buffer>> ConcatenateBuffers(const BufferVector& buffers,
                                                   MemoryPool* pool = NULLPTR);

/// @}

}  // namespace arrow

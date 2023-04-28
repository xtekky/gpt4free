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

// Public API for different memory sharing / IO mechanisms

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "arrow/io/concurrency.h"
#include "arrow/io/interfaces.h"
#include "arrow/type_fwd.h"
#include "arrow/util/visibility.h"

namespace arrow {

class Status;

namespace io {

/// \brief An output stream that writes to a resizable buffer
class ARROW_EXPORT BufferOutputStream : public OutputStream {
 public:
  explicit BufferOutputStream(const std::shared_ptr<ResizableBuffer>& buffer);

  /// \brief Create in-memory output stream with indicated capacity using a
  /// memory pool
  /// \param[in] initial_capacity the initial allocated internal capacity of
  /// the OutputStream
  /// \param[in,out] pool a MemoryPool to use for allocations
  /// \return the created stream
  static Result<std::shared_ptr<BufferOutputStream>> Create(
      int64_t initial_capacity = 4096, MemoryPool* pool = default_memory_pool());

  ~BufferOutputStream() override;

  // Implement the OutputStream interface

  /// Close the stream, preserving the buffer (retrieve it with Finish()).
  Status Close() override;
  bool closed() const override;
  Result<int64_t> Tell() const override;
  Status Write(const void* data, int64_t nbytes) override;

  /// \cond FALSE
  using OutputStream::Write;
  /// \endcond

  /// Close the stream and return the buffer
  Result<std::shared_ptr<Buffer>> Finish();

  /// \brief Initialize state of OutputStream with newly allocated memory and
  /// set position to 0
  /// \param[in] initial_capacity the starting allocated capacity
  /// \param[in,out] pool the memory pool to use for allocations
  /// \return Status
  Status Reset(int64_t initial_capacity = 1024, MemoryPool* pool = default_memory_pool());

  int64_t capacity() const { return capacity_; }

 private:
  BufferOutputStream();

  // Ensures there is sufficient space available to write nbytes
  Status Reserve(int64_t nbytes);

  std::shared_ptr<ResizableBuffer> buffer_;
  bool is_open_;
  int64_t capacity_;
  int64_t position_;
  uint8_t* mutable_data_;
};

/// \brief A helper class to track the size of allocations
///
/// Writes to this stream do not copy or retain any data, they just bump
/// a size counter that can be later used to know exactly which data size
/// needs to be allocated for actual writing.
class ARROW_EXPORT MockOutputStream : public OutputStream {
 public:
  MockOutputStream() : extent_bytes_written_(0), is_open_(true) {}

  // Implement the OutputStream interface
  Status Close() override;
  bool closed() const override;
  Result<int64_t> Tell() const override;
  Status Write(const void* data, int64_t nbytes) override;
  /// \cond FALSE
  using Writable::Write;
  /// \endcond

  int64_t GetExtentBytesWritten() const { return extent_bytes_written_; }

 private:
  int64_t extent_bytes_written_;
  bool is_open_;
};

/// \brief An output stream that writes into a fixed-size mutable buffer
class ARROW_EXPORT FixedSizeBufferWriter : public WritableFile {
 public:
  /// Input buffer must be mutable, will abort if not
  explicit FixedSizeBufferWriter(const std::shared_ptr<Buffer>& buffer);
  ~FixedSizeBufferWriter() override;

  Status Close() override;
  bool closed() const override;
  Status Seek(int64_t position) override;
  Result<int64_t> Tell() const override;
  Status Write(const void* data, int64_t nbytes) override;
  /// \cond FALSE
  using Writable::Write;
  /// \endcond

  Status WriteAt(int64_t position, const void* data, int64_t nbytes) override;

  void set_memcopy_threads(int num_threads);
  void set_memcopy_blocksize(int64_t blocksize);
  void set_memcopy_threshold(int64_t threshold);

 protected:
  class FixedSizeBufferWriterImpl;
  std::unique_ptr<FixedSizeBufferWriterImpl> impl_;
};

/// \class BufferReader
/// \brief Random access zero-copy reads on an arrow::Buffer
class ARROW_EXPORT BufferReader
    : public internal::RandomAccessFileConcurrencyWrapper<BufferReader> {
 public:
  explicit BufferReader(std::shared_ptr<Buffer> buffer);
  explicit BufferReader(const Buffer& buffer);
  BufferReader(const uint8_t* data, int64_t size);

  /// \brief Instantiate from std::string or std::string_view. Does not
  /// own data
  explicit BufferReader(const std::string_view& data);

  bool closed() const override;

  bool supports_zero_copy() const override;

  std::shared_ptr<Buffer> buffer() const { return buffer_; }

  // Synchronous ReadAsync override
  Future<std::shared_ptr<Buffer>> ReadAsync(const IOContext&, int64_t position,
                                            int64_t nbytes) override;
  Status WillNeed(const std::vector<ReadRange>& ranges) override;

 protected:
  friend RandomAccessFileConcurrencyWrapper<BufferReader>;

  Status DoClose();

  Result<int64_t> DoRead(int64_t nbytes, void* buffer);
  Result<std::shared_ptr<Buffer>> DoRead(int64_t nbytes);
  Result<int64_t> DoReadAt(int64_t position, int64_t nbytes, void* out);
  Result<std::shared_ptr<Buffer>> DoReadAt(int64_t position, int64_t nbytes);
  Result<std::string_view> DoPeek(int64_t nbytes) override;

  Result<int64_t> DoTell() const;
  Status DoSeek(int64_t position);
  Result<int64_t> DoGetSize();

  Status CheckClosed() const {
    if (!is_open_) {
      return Status::Invalid("Operation forbidden on closed BufferReader");
    }
    return Status::OK();
  }

  std::shared_ptr<Buffer> buffer_;
  const uint8_t* data_;
  int64_t size_;
  int64_t position_;
  bool is_open_;
};

}  // namespace io
}  // namespace arrow

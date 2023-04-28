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

// Buffered stream implementations

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>

#include "arrow/io/concurrency.h"
#include "arrow/io/interfaces.h"
#include "arrow/util/visibility.h"

namespace arrow {

class Buffer;
class MemoryPool;
class Status;

namespace io {

class ARROW_EXPORT BufferedOutputStream : public OutputStream {
 public:
  ~BufferedOutputStream() override;

  /// \brief Create a buffered output stream wrapping the given output stream.
  /// \param[in] buffer_size the size of the temporary write buffer
  /// \param[in] pool a MemoryPool to use for allocations
  /// \param[in] raw another OutputStream
  /// \return the created BufferedOutputStream
  static Result<std::shared_ptr<BufferedOutputStream>> Create(
      int64_t buffer_size, MemoryPool* pool, std::shared_ptr<OutputStream> raw);

  /// \brief Resize internal buffer
  /// \param[in] new_buffer_size the new buffer size
  /// \return Status
  Status SetBufferSize(int64_t new_buffer_size);

  /// \brief Return the current size of the internal buffer
  int64_t buffer_size() const;

  /// \brief Return the number of remaining bytes that have not been flushed to
  /// the raw OutputStream
  int64_t bytes_buffered() const;

  /// \brief Flush any buffered writes and release the raw
  /// OutputStream. Further operations on this object are invalid
  /// \return the underlying OutputStream
  Result<std::shared_ptr<OutputStream>> Detach();

  // OutputStream interface

  /// \brief Close the buffered output stream.  This implicitly closes the
  /// underlying raw output stream.
  Status Close() override;
  Status Abort() override;
  bool closed() const override;

  Result<int64_t> Tell() const override;
  // Write bytes to the stream. Thread-safe
  Status Write(const void* data, int64_t nbytes) override;
  Status Write(const std::shared_ptr<Buffer>& data) override;

  Status Flush() override;

  /// \brief Return the underlying raw output stream.
  std::shared_ptr<OutputStream> raw() const;

 private:
  explicit BufferedOutputStream(std::shared_ptr<OutputStream> raw, MemoryPool* pool);

  class ARROW_NO_EXPORT Impl;
  std::unique_ptr<Impl> impl_;
};

/// \class BufferedInputStream
/// \brief An InputStream that performs buffered reads from an unbuffered
/// InputStream, which can mitigate the overhead of many small reads in some
/// cases
class ARROW_EXPORT BufferedInputStream
    : public internal::InputStreamConcurrencyWrapper<BufferedInputStream> {
 public:
  ~BufferedInputStream() override;

  /// \brief Create a BufferedInputStream from a raw InputStream
  /// \param[in] buffer_size the size of the temporary read buffer
  /// \param[in] pool a MemoryPool to use for allocations
  /// \param[in] raw a raw InputStream
  /// \param[in] raw_read_bound a bound on the maximum number of bytes
  /// to read from the raw input stream. The default -1 indicates that
  /// it is unbounded
  /// \return the created BufferedInputStream
  static Result<std::shared_ptr<BufferedInputStream>> Create(
      int64_t buffer_size, MemoryPool* pool, std::shared_ptr<InputStream> raw,
      int64_t raw_read_bound = -1);

  /// \brief Resize internal read buffer; calls to Read(...) will read at least
  /// \param[in] new_buffer_size the new read buffer size
  /// \return Status
  Status SetBufferSize(int64_t new_buffer_size);

  /// \brief Return the number of remaining bytes in the read buffer
  int64_t bytes_buffered() const;

  /// \brief Return the current size of the internal buffer
  int64_t buffer_size() const;

  /// \brief Release the raw InputStream. Any data buffered will be
  /// discarded. Further operations on this object are invalid
  /// \return raw the underlying InputStream
  std::shared_ptr<InputStream> Detach();

  /// \brief Return the unbuffered InputStream
  std::shared_ptr<InputStream> raw() const;

  // InputStream APIs

  bool closed() const override;
  Result<std::shared_ptr<const KeyValueMetadata>> ReadMetadata() override;
  Future<std::shared_ptr<const KeyValueMetadata>> ReadMetadataAsync(
      const IOContext& io_context) override;

 private:
  friend InputStreamConcurrencyWrapper<BufferedInputStream>;

  explicit BufferedInputStream(std::shared_ptr<InputStream> raw, MemoryPool* pool,
                               int64_t raw_total_bytes_bound);

  Status DoClose();
  Status DoAbort() override;

  /// \brief Returns the position of the buffered stream, though the position
  /// of the unbuffered stream may be further advanced.
  Result<int64_t> DoTell() const;

  Result<int64_t> DoRead(int64_t nbytes, void* out);

  /// \brief Read into buffer.
  Result<std::shared_ptr<Buffer>> DoRead(int64_t nbytes);

  /// \brief Return a zero-copy string view referencing buffered data,
  /// but do not advance the position of the stream. Buffers data and
  /// expands the buffer size if necessary
  Result<std::string_view> DoPeek(int64_t nbytes) override;

  class ARROW_NO_EXPORT Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace io
}  // namespace arrow

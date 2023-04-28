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

// IO interface implementations for OS files

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "arrow/io/concurrency.h"
#include "arrow/io/interfaces.h"
#include "arrow/util/visibility.h"

namespace arrow {

class Buffer;
class MemoryPool;
class Status;

namespace io {

/// \brief An operating system file open in write-only mode.
class ARROW_EXPORT FileOutputStream : public OutputStream {
 public:
  ~FileOutputStream() override;

  /// \brief Open a local file for writing, truncating any existing file
  /// \param[in] path with UTF8 encoding
  /// \param[in] append append to existing file, otherwise truncate to 0 bytes
  /// \return an open FileOutputStream
  ///
  /// When opening a new file, any existing file with the indicated path is
  /// truncated to 0 bytes, deleting any existing data
  static Result<std::shared_ptr<FileOutputStream>> Open(const std::string& path,
                                                        bool append = false);

  /// \brief Open a file descriptor for writing.  The underlying file isn't
  /// truncated.
  /// \param[in] fd file descriptor
  /// \return an open FileOutputStream
  ///
  /// The file descriptor becomes owned by the OutputStream, and will be closed
  /// on Close() or destruction.
  static Result<std::shared_ptr<FileOutputStream>> Open(int fd);

  // OutputStream interface
  Status Close() override;
  bool closed() const override;
  Result<int64_t> Tell() const override;

  // Write bytes to the stream. Thread-safe
  Status Write(const void* data, int64_t nbytes) override;
  /// \cond FALSE
  using Writable::Write;
  /// \endcond

  int file_descriptor() const;

 private:
  FileOutputStream();

  class ARROW_NO_EXPORT FileOutputStreamImpl;
  std::unique_ptr<FileOutputStreamImpl> impl_;
};

/// \brief An operating system file open in read-only mode.
///
/// Reads through this implementation are unbuffered.  If many small reads
/// need to be issued, it is recommended to use a buffering layer for good
/// performance.
class ARROW_EXPORT ReadableFile
    : public internal::RandomAccessFileConcurrencyWrapper<ReadableFile> {
 public:
  ~ReadableFile() override;

  /// \brief Open a local file for reading
  /// \param[in] path with UTF8 encoding
  /// \param[in] pool a MemoryPool for memory allocations
  /// \return ReadableFile instance
  static Result<std::shared_ptr<ReadableFile>> Open(
      const std::string& path, MemoryPool* pool = default_memory_pool());

  /// \brief Open a local file for reading
  /// \param[in] fd file descriptor
  /// \param[in] pool a MemoryPool for memory allocations
  /// \return ReadableFile instance
  ///
  /// The file descriptor becomes owned by the ReadableFile, and will be closed
  /// on Close() or destruction.
  static Result<std::shared_ptr<ReadableFile>> Open(
      int fd, MemoryPool* pool = default_memory_pool());

  bool closed() const override;

  int file_descriptor() const;

  Status WillNeed(const std::vector<ReadRange>& ranges) override;

 private:
  friend RandomAccessFileConcurrencyWrapper<ReadableFile>;

  explicit ReadableFile(MemoryPool* pool);

  Status DoClose();
  Result<int64_t> DoTell() const;
  Result<int64_t> DoRead(int64_t nbytes, void* buffer);
  Result<std::shared_ptr<Buffer>> DoRead(int64_t nbytes);

  /// \brief Thread-safe implementation of ReadAt
  Result<int64_t> DoReadAt(int64_t position, int64_t nbytes, void* out);

  /// \brief Thread-safe implementation of ReadAt
  Result<std::shared_ptr<Buffer>> DoReadAt(int64_t position, int64_t nbytes);

  Result<int64_t> DoGetSize();
  Status DoSeek(int64_t position);

  class ARROW_NO_EXPORT ReadableFileImpl;
  std::unique_ptr<ReadableFileImpl> impl_;
};

/// \brief A file interface that uses memory-mapped files for memory interactions
///
/// This implementation supports zero-copy reads. The same class is used
/// for both reading and writing.
///
/// If opening a file in a writable mode, it is not truncated first as with
/// FileOutputStream.
class ARROW_EXPORT MemoryMappedFile : public ReadWriteFileInterface {
 public:
  ~MemoryMappedFile() override;

  /// Create new file with indicated size, return in read/write mode
  static Result<std::shared_ptr<MemoryMappedFile>> Create(const std::string& path,
                                                          int64_t size);

  // mmap() with whole file
  static Result<std::shared_ptr<MemoryMappedFile>> Open(const std::string& path,
                                                        FileMode::type mode);

  // mmap() with a region of file, the offset must be a multiple of the page size
  static Result<std::shared_ptr<MemoryMappedFile>> Open(const std::string& path,
                                                        FileMode::type mode,
                                                        const int64_t offset,
                                                        const int64_t length);

  Status Close() override;

  bool closed() const override;

  Result<int64_t> Tell() const override;

  Status Seek(int64_t position) override;

  // Required by RandomAccessFile, copies memory into out. Not thread-safe
  Result<int64_t> Read(int64_t nbytes, void* out) override;

  // Zero copy read, moves position pointer. Not thread-safe
  Result<std::shared_ptr<Buffer>> Read(int64_t nbytes) override;

  // Zero-copy read, leaves position unchanged. Acquires a reader lock
  // for the duration of slice creation (typically very short). Is thread-safe.
  Result<std::shared_ptr<Buffer>> ReadAt(int64_t position, int64_t nbytes) override;

  // Raw copy of the memory at specified position. Thread-safe, but
  // locks out other readers for the duration of memcpy. Prefer the
  // zero copy method
  Result<int64_t> ReadAt(int64_t position, int64_t nbytes, void* out) override;

  // Synchronous ReadAsync override
  Future<std::shared_ptr<Buffer>> ReadAsync(const IOContext&, int64_t position,
                                            int64_t nbytes) override;

  Status WillNeed(const std::vector<ReadRange>& ranges) override;

  bool supports_zero_copy() const override;

  /// Write data at the current position in the file. Thread-safe
  Status Write(const void* data, int64_t nbytes) override;
  /// \cond FALSE
  using Writable::Write;
  /// \endcond

  /// Set the size of the map to new_size.
  Status Resize(int64_t new_size);

  /// Write data at a particular position in the file. Thread-safe
  Status WriteAt(int64_t position, const void* data, int64_t nbytes) override;

  Result<int64_t> GetSize() override;

  int file_descriptor() const;

 private:
  MemoryMappedFile();

  Status WriteInternal(const void* data, int64_t nbytes);

  class ARROW_NO_EXPORT MemoryMap;
  std::shared_ptr<MemoryMap> memory_map_;
};

}  // namespace io
}  // namespace arrow

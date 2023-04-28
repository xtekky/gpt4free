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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "arrow/io/interfaces.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {

class Buffer;
class MemoryPool;
class Status;

namespace io {

class HdfsReadableFile;
class HdfsOutputStream;

/// DEPRECATED.  Use the FileSystem API in arrow::fs instead.
struct ObjectType {
  enum type { FILE, DIRECTORY };
};

/// DEPRECATED.  Use the FileSystem API in arrow::fs instead.
struct ARROW_EXPORT FileStatistics {
  /// Size of file, -1 if finding length is unsupported
  int64_t size;
  ObjectType::type kind;
};

class ARROW_EXPORT FileSystem {
 public:
  virtual ~FileSystem() = default;

  virtual Status MakeDirectory(const std::string& path) = 0;

  virtual Status DeleteDirectory(const std::string& path) = 0;

  virtual Status GetChildren(const std::string& path,
                             std::vector<std::string>* listing) = 0;

  virtual Status Rename(const std::string& src, const std::string& dst) = 0;

  virtual Status Stat(const std::string& path, FileStatistics* stat) = 0;
};

struct HdfsPathInfo {
  ObjectType::type kind;

  std::string name;
  std::string owner;
  std::string group;

  // Access times in UNIX timestamps (seconds)
  int64_t size;
  int64_t block_size;

  int32_t last_modified_time;
  int32_t last_access_time;

  int16_t replication;
  int16_t permissions;
};

struct HdfsConnectionConfig {
  std::string host;
  int port;
  std::string user;
  std::string kerb_ticket;
  std::unordered_map<std::string, std::string> extra_conf;
};

class ARROW_EXPORT HadoopFileSystem : public FileSystem {
 public:
  ~HadoopFileSystem() override;

  // Connect to an HDFS cluster given a configuration
  //
  // @param config (in): configuration for connecting
  // @param fs (out): the created client
  // @returns Status
  static Status Connect(const HdfsConnectionConfig* config,
                        std::shared_ptr<HadoopFileSystem>* fs);

  // Create directory and all parents
  //
  // @param path (in): absolute HDFS path
  // @returns Status
  Status MakeDirectory(const std::string& path) override;

  // Delete file or directory
  // @param path absolute path to data
  // @param recursive if path is a directory, delete contents as well
  // @returns error status on failure
  Status Delete(const std::string& path, bool recursive = false);

  Status DeleteDirectory(const std::string& path) override;

  // Disconnect from cluster
  //
  // @returns Status
  Status Disconnect();

  // @param path (in): absolute HDFS path
  // @returns bool, true if the path exists, false if not (or on error)
  bool Exists(const std::string& path);

  // @param path (in): absolute HDFS path
  // @param info (out)
  // @returns Status
  Status GetPathInfo(const std::string& path, HdfsPathInfo* info);

  // @param nbytes (out): total capacity of the filesystem
  // @returns Status
  Status GetCapacity(int64_t* nbytes);

  // @param nbytes (out): total bytes used of the filesystem
  // @returns Status
  Status GetUsed(int64_t* nbytes);

  Status GetChildren(const std::string& path, std::vector<std::string>* listing) override;

  /// List directory contents
  ///
  /// If path is a relative path, returned values will be absolute paths or URIs
  /// starting from the current working directory.
  Status ListDirectory(const std::string& path, std::vector<HdfsPathInfo>* listing);

  /// Return the filesystem's current working directory.
  ///
  /// The working directory is the base path for all relative paths given to
  /// other APIs.
  /// NOTE: this actually returns a URI.
  Status GetWorkingDirectory(std::string* out);

  /// Change
  ///
  /// @param path file path to change
  /// @param owner pass null for no change
  /// @param group pass null for no change
  Status Chown(const std::string& path, const char* owner, const char* group);

  /// Change path permissions
  ///
  /// \param path Absolute path in file system
  /// \param mode Mode bitset
  /// \return Status
  Status Chmod(const std::string& path, int mode);

  // Move file or directory from source path to destination path within the
  // current filesystem
  Status Rename(const std::string& src, const std::string& dst) override;

  Status Copy(const std::string& src, const std::string& dst);

  Status Move(const std::string& src, const std::string& dst);

  Status Stat(const std::string& path, FileStatistics* stat) override;

  // TODO(wesm): GetWorkingDirectory, SetWorkingDirectory

  // Open an HDFS file in READ mode. Returns error
  // status if the file is not found.
  //
  // @param path complete file path
  Status OpenReadable(const std::string& path, int32_t buffer_size,
                      std::shared_ptr<HdfsReadableFile>* file);

  Status OpenReadable(const std::string& path, int32_t buffer_size,
                      const io::IOContext& io_context,
                      std::shared_ptr<HdfsReadableFile>* file);

  Status OpenReadable(const std::string& path, std::shared_ptr<HdfsReadableFile>* file);

  Status OpenReadable(const std::string& path, const io::IOContext& io_context,
                      std::shared_ptr<HdfsReadableFile>* file);

  // FileMode::WRITE options
  // @param path complete file path
  // @param buffer_size 0 by default
  // @param replication 0 by default
  // @param default_block_size 0 by default
  Status OpenWritable(const std::string& path, bool append, int32_t buffer_size,
                      int16_t replication, int64_t default_block_size,
                      std::shared_ptr<HdfsOutputStream>* file);

  Status OpenWritable(const std::string& path, bool append,
                      std::shared_ptr<HdfsOutputStream>* file);

 private:
  friend class HdfsReadableFile;
  friend class HdfsOutputStream;

  class ARROW_NO_EXPORT HadoopFileSystemImpl;
  std::unique_ptr<HadoopFileSystemImpl> impl_;

  HadoopFileSystem();
  ARROW_DISALLOW_COPY_AND_ASSIGN(HadoopFileSystem);
};

class ARROW_EXPORT HdfsReadableFile : public RandomAccessFile {
 public:
  ~HdfsReadableFile() override;

  Status Close() override;

  bool closed() const override;

  // NOTE: If you wish to read a particular range of a file in a multithreaded
  // context, you may prefer to use ReadAt to avoid locking issues
  Result<int64_t> Read(int64_t nbytes, void* out) override;
  Result<std::shared_ptr<Buffer>> Read(int64_t nbytes) override;
  Result<int64_t> ReadAt(int64_t position, int64_t nbytes, void* out) override;
  Result<std::shared_ptr<Buffer>> ReadAt(int64_t position, int64_t nbytes) override;

  Status Seek(int64_t position) override;
  Result<int64_t> Tell() const override;
  Result<int64_t> GetSize() override;

 private:
  explicit HdfsReadableFile(const io::IOContext&);

  class ARROW_NO_EXPORT HdfsReadableFileImpl;
  std::unique_ptr<HdfsReadableFileImpl> impl_;

  friend class HadoopFileSystem::HadoopFileSystemImpl;

  ARROW_DISALLOW_COPY_AND_ASSIGN(HdfsReadableFile);
};

// Naming this file OutputStream because it does not support seeking (like the
// WritableFile interface)
class ARROW_EXPORT HdfsOutputStream : public OutputStream {
 public:
  ~HdfsOutputStream() override;

  Status Close() override;

  bool closed() const override;

  using OutputStream::Write;
  Status Write(const void* buffer, int64_t nbytes) override;

  Status Flush() override;

  Result<int64_t> Tell() const override;

 private:
  class ARROW_NO_EXPORT HdfsOutputStreamImpl;
  std::unique_ptr<HdfsOutputStreamImpl> impl_;

  friend class HadoopFileSystem::HadoopFileSystemImpl;

  HdfsOutputStream();

  ARROW_DISALLOW_COPY_AND_ASSIGN(HdfsOutputStream);
};

ARROW_EXPORT Status HaveLibHdfs();

}  // namespace io
}  // namespace arrow

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

#include <memory>
#include <string>
#include <vector>

#include "arrow/filesystem/filesystem.h"

namespace arrow {
namespace internal {

class Uri;

}

namespace fs {

/// Options for the LocalFileSystem implementation.
struct ARROW_EXPORT LocalFileSystemOptions {
  static constexpr int32_t kDefaultDirectoryReadahead = 16;
  static constexpr int32_t kDefaultFileInfoBatchSize = 1000;

  /// Whether OpenInputStream and OpenInputFile return a mmap'ed file,
  /// or a regular one.
  bool use_mmap = false;

  /// Options related to `GetFileInfoGenerator` interface.

  /// EXPERIMENTAL: The maximum number of directories processed in parallel
  /// by `GetFileInfoGenerator`.
  int32_t directory_readahead = kDefaultDirectoryReadahead;

  /// EXPERIMENTAL: The maximum number of entries aggregated into each
  /// FileInfoVector chunk by `GetFileInfoGenerator`.
  ///
  /// Since each FileInfo entry needs a separate `stat` system call, a
  /// directory with a very large number of files may take a lot of time to
  /// process entirely. By generating a FileInfoVector after this chunk
  /// size is reached, we ensure FileInfo entries can start being consumed
  /// from the FileInfoGenerator with less initial latency.
  int32_t file_info_batch_size = kDefaultFileInfoBatchSize;

  /// \brief Initialize with defaults
  static LocalFileSystemOptions Defaults();

  bool Equals(const LocalFileSystemOptions& other) const;

  static Result<LocalFileSystemOptions> FromUri(const ::arrow::internal::Uri& uri,
                                                std::string* out_path);
};

/// \brief A FileSystem implementation accessing files on the local machine.
///
/// This class handles only `/`-separated paths.  If desired, conversion
/// from Windows backslash-separated paths should be done by the caller.
/// Details such as symlinks are abstracted away (symlinks are always
/// followed, except when deleting an entry).
class ARROW_EXPORT LocalFileSystem : public FileSystem {
 public:
  explicit LocalFileSystem(const io::IOContext& = io::default_io_context());
  explicit LocalFileSystem(const LocalFileSystemOptions&,
                           const io::IOContext& = io::default_io_context());
  ~LocalFileSystem() override;

  std::string type_name() const override { return "local"; }

  Result<std::string> NormalizePath(std::string path) override;

  bool Equals(const FileSystem& other) const override;

  LocalFileSystemOptions options() const { return options_; }

  /// \cond FALSE
  using FileSystem::GetFileInfo;
  /// \endcond
  Result<FileInfo> GetFileInfo(const std::string& path) override;
  Result<std::vector<FileInfo>> GetFileInfo(const FileSelector& select) override;
  FileInfoGenerator GetFileInfoGenerator(const FileSelector& select) override;

  Status CreateDir(const std::string& path, bool recursive = true) override;

  Status DeleteDir(const std::string& path) override;
  Status DeleteDirContents(const std::string& path, bool missing_dir_ok = false) override;
  Status DeleteRootDirContents() override;

  Status DeleteFile(const std::string& path) override;

  Status Move(const std::string& src, const std::string& dest) override;

  Status CopyFile(const std::string& src, const std::string& dest) override;

  Result<std::shared_ptr<io::InputStream>> OpenInputStream(
      const std::string& path) override;
  Result<std::shared_ptr<io::RandomAccessFile>> OpenInputFile(
      const std::string& path) override;
  Result<std::shared_ptr<io::OutputStream>> OpenOutputStream(
      const std::string& path,
      const std::shared_ptr<const KeyValueMetadata>& metadata = {}) override;
  Result<std::shared_ptr<io::OutputStream>> OpenAppendStream(
      const std::string& path,
      const std::shared_ptr<const KeyValueMetadata>& metadata = {}) override;

 protected:
  LocalFileSystemOptions options_;
};

namespace internal {

// Return whether the string is detected as a local absolute path.
ARROW_EXPORT
bool DetectAbsolutePath(const std::string& s);

}  // namespace internal

}  // namespace fs
}  // namespace arrow

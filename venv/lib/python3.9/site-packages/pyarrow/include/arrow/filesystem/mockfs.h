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

#include <iosfwd>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "arrow/filesystem/filesystem.h"
#include "arrow/util/windows_fixup.h"

namespace arrow {
namespace fs {
namespace internal {

struct MockDirInfo {
  std::string full_path;
  TimePoint mtime;

  bool operator==(const MockDirInfo& other) const {
    return mtime == other.mtime && full_path == other.full_path;
  }

  ARROW_FRIEND_EXPORT friend std::ostream& operator<<(std::ostream&, const MockDirInfo&);
};

struct MockFileInfo {
  std::string full_path;
  TimePoint mtime;
  std::string_view data;

  bool operator==(const MockFileInfo& other) const {
    return mtime == other.mtime && full_path == other.full_path && data == other.data;
  }

  ARROW_FRIEND_EXPORT friend std::ostream& operator<<(std::ostream&, const MockFileInfo&);
};

/// A mock FileSystem implementation that holds its contents in memory.
///
/// Useful for validating the FileSystem API, writing conformance suite,
/// and bootstrapping FileSystem-based APIs.
class ARROW_EXPORT MockFileSystem : public FileSystem {
 public:
  explicit MockFileSystem(TimePoint current_time,
                          const io::IOContext& = io::default_io_context());
  ~MockFileSystem() override;

  std::string type_name() const override { return "mock"; }

  bool Equals(const FileSystem& other) const override;

  // XXX It's not very practical to have to explicitly declare inheritance
  // of default overrides.
  using FileSystem::GetFileInfo;
  Result<FileInfo> GetFileInfo(const std::string& path) override;
  Result<std::vector<FileInfo>> GetFileInfo(const FileSelector& select) override;

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

  // Contents-dumping helpers to ease testing.
  // Output is lexicographically-ordered by full path.
  std::vector<MockDirInfo> AllDirs();
  std::vector<MockFileInfo> AllFiles();

  // Create a File with a content from a string.
  Status CreateFile(const std::string& path, std::string_view content,
                    bool recursive = true);

  // Create a MockFileSystem out of (empty) FileInfo. The content of every
  // file is empty and of size 0. All directories will be created recursively.
  static Result<std::shared_ptr<FileSystem>> Make(TimePoint current_time,
                                                  const std::vector<FileInfo>& infos);

  class Impl;

 protected:
  std::unique_ptr<Impl> impl_;
};

class ARROW_EXPORT MockAsyncFileSystem : public MockFileSystem {
 public:
  explicit MockAsyncFileSystem(TimePoint current_time,
                               const io::IOContext& io_context = io::default_io_context())
      : MockFileSystem(current_time, io_context) {
    default_async_is_sync_ = false;
  }

  FileInfoGenerator GetFileInfoGenerator(const FileSelector& select) override;
};

}  // namespace internal
}  // namespace fs
}  // namespace arrow

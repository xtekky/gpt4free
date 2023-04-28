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

#include <chrono>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "arrow/filesystem/type_fwd.h"
#include "arrow/io/interfaces.h"
#include "arrow/type_fwd.h"
#include "arrow/util/compare.h"
#include "arrow/util/macros.h"
#include "arrow/util/type_fwd.h"
#include "arrow/util/visibility.h"
#include "arrow/util/windows_fixup.h"

namespace arrow {
namespace fs {

// A system clock time point expressed as a 64-bit (or more) number of
// nanoseconds since the epoch.
using TimePoint =
    std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>;

ARROW_EXPORT std::string ToString(FileType);

ARROW_EXPORT std::ostream& operator<<(std::ostream& os, FileType);

static const int64_t kNoSize = -1;
static const TimePoint kNoTime = TimePoint(TimePoint::duration(-1));

/// \brief FileSystem entry info
struct ARROW_EXPORT FileInfo : public util::EqualityComparable<FileInfo> {
  FileInfo() = default;
  FileInfo(FileInfo&&) = default;
  FileInfo& operator=(FileInfo&&) = default;
  FileInfo(const FileInfo&) = default;
  FileInfo& operator=(const FileInfo&) = default;

  explicit FileInfo(std::string path, FileType type = FileType::Unknown)
      : path_(std::move(path)), type_(type) {}

  /// The file type
  FileType type() const { return type_; }
  void set_type(FileType type) { type_ = type; }

  /// The full file path in the filesystem
  const std::string& path() const { return path_; }
  void set_path(std::string path) { path_ = std::move(path); }

  /// The file base name (component after the last directory separator)
  std::string base_name() const;

  // The directory base name (component before the file base name).
  std::string dir_name() const;

  /// The size in bytes, if available
  ///
  /// Only regular files are guaranteed to have a size.
  int64_t size() const { return size_; }
  void set_size(int64_t size) { size_ = size; }

  /// The file extension (excluding the dot)
  std::string extension() const;

  /// The time of last modification, if available
  TimePoint mtime() const { return mtime_; }
  void set_mtime(TimePoint mtime) { mtime_ = mtime; }

  bool IsFile() const { return type_ == FileType::File; }
  bool IsDirectory() const { return type_ == FileType::Directory; }

  bool Equals(const FileInfo& other) const {
    return type() == other.type() && path() == other.path() && size() == other.size() &&
           mtime() == other.mtime();
  }

  std::string ToString() const;

  /// Function object implementing less-than comparison and hashing by
  /// path, to support sorting infos, using them as keys, and other
  /// interactions with the STL.
  struct ByPath {
    bool operator()(const FileInfo& l, const FileInfo& r) const {
      return l.path() < r.path();
    }

    size_t operator()(const FileInfo& i) const {
      return std::hash<std::string>{}(i.path());
    }
  };

 protected:
  std::string path_;
  FileType type_ = FileType::Unknown;
  int64_t size_ = kNoSize;
  TimePoint mtime_ = kNoTime;
};

ARROW_EXPORT std::ostream& operator<<(std::ostream& os, const FileInfo&);

/// \brief File selector for filesystem APIs
struct ARROW_EXPORT FileSelector {
  /// The directory in which to select files.
  /// If the path exists but doesn't point to a directory, this should be an error.
  std::string base_dir;
  /// The behavior if `base_dir` isn't found in the filesystem.  If false,
  /// an error is returned.  If true, an empty selection is returned.
  bool allow_not_found;
  /// Whether to recurse into subdirectories.
  bool recursive;
  /// The maximum number of subdirectories to recurse into.
  int32_t max_recursion;

  FileSelector() : allow_not_found(false), recursive(false), max_recursion(INT32_MAX) {}
};

/// \brief FileSystem, path pair
struct ARROW_EXPORT FileLocator {
  std::shared_ptr<FileSystem> filesystem;
  std::string path;
};

using FileInfoVector = std::vector<FileInfo>;
using FileInfoGenerator = std::function<Future<FileInfoVector>()>;

}  // namespace fs

template <>
struct IterationTraits<fs::FileInfoVector> {
  static fs::FileInfoVector End() { return {}; }
  static bool IsEnd(const fs::FileInfoVector& val) { return val.empty(); }
};

namespace fs {

/// \brief Abstract file system API
class ARROW_EXPORT FileSystem : public std::enable_shared_from_this<FileSystem> {
 public:
  virtual ~FileSystem();

  virtual std::string type_name() const = 0;

  /// EXPERIMENTAL: The IOContext associated with this filesystem.
  const io::IOContext& io_context() const { return io_context_; }

  /// Normalize path for the given filesystem
  ///
  /// The default implementation of this method is a no-op, but subclasses
  /// may allow normalizing irregular path forms (such as Windows local paths).
  virtual Result<std::string> NormalizePath(std::string path);

  virtual bool Equals(const FileSystem& other) const = 0;

  virtual bool Equals(const std::shared_ptr<FileSystem>& other) const {
    return Equals(*other);
  }

  /// Get info for the given target.
  ///
  /// Any symlink is automatically dereferenced, recursively.
  /// A nonexistent or unreachable file returns an Ok status and
  /// has a FileType of value NotFound.  An error status indicates
  /// a truly exceptional condition (low-level I/O error, etc.).
  virtual Result<FileInfo> GetFileInfo(const std::string& path) = 0;
  /// Same, for many targets at once.
  virtual Result<FileInfoVector> GetFileInfo(const std::vector<std::string>& paths);
  /// Same, according to a selector.
  ///
  /// The selector's base directory will not be part of the results, even if
  /// it exists.
  /// If it doesn't exist, see `FileSelector::allow_not_found`.
  virtual Result<FileInfoVector> GetFileInfo(const FileSelector& select) = 0;

  /// Async version of GetFileInfo
  virtual Future<FileInfoVector> GetFileInfoAsync(const std::vector<std::string>& paths);

  /// Streaming async version of GetFileInfo
  ///
  /// The returned generator is not async-reentrant, i.e. you need to wait for
  /// the returned future to complete before calling the generator again.
  virtual FileInfoGenerator GetFileInfoGenerator(const FileSelector& select);

  /// Create a directory and subdirectories.
  ///
  /// This function succeeds if the directory already exists.
  virtual Status CreateDir(const std::string& path, bool recursive = true) = 0;

  /// Delete a directory and its contents, recursively.
  virtual Status DeleteDir(const std::string& path) = 0;

  /// Delete a directory's contents, recursively.
  ///
  /// Like DeleteDir, but doesn't delete the directory itself.
  /// Passing an empty path ("" or "/") is disallowed, see DeleteRootDirContents.
  virtual Status DeleteDirContents(const std::string& path,
                                   bool missing_dir_ok = false) = 0;

  /// Async version of DeleteDirContents.
  virtual Future<> DeleteDirContentsAsync(const std::string& path,
                                          bool missing_dir_ok = false);

  /// EXPERIMENTAL: Delete the root directory's contents, recursively.
  ///
  /// Implementations may decide to raise an error if this operation is
  /// too dangerous.
  // NOTE: may decide to remove this if it's deemed not useful
  virtual Status DeleteRootDirContents() = 0;

  /// Delete a file.
  virtual Status DeleteFile(const std::string& path) = 0;
  /// Delete many files.
  ///
  /// The default implementation issues individual delete operations in sequence.
  virtual Status DeleteFiles(const std::vector<std::string>& paths);

  /// Move / rename a file or directory.
  ///
  /// If the destination exists:
  /// - if it is a non-empty directory, an error is returned
  /// - otherwise, if it has the same type as the source, it is replaced
  /// - otherwise, behavior is unspecified (implementation-dependent).
  virtual Status Move(const std::string& src, const std::string& dest) = 0;

  /// Copy a file.
  ///
  /// If the destination exists and is a directory, an error is returned.
  /// Otherwise, it is replaced.
  virtual Status CopyFile(const std::string& src, const std::string& dest) = 0;

  /// Open an input stream for sequential reading.
  virtual Result<std::shared_ptr<io::InputStream>> OpenInputStream(
      const std::string& path) = 0;
  /// Open an input stream for sequential reading.
  ///
  /// This override assumes the given FileInfo validly represents the file's
  /// characteristics, and may optimize access depending on them (for example
  /// avoid querying the file size or its existence).
  virtual Result<std::shared_ptr<io::InputStream>> OpenInputStream(const FileInfo& info);

  /// Open an input file for random access reading.
  virtual Result<std::shared_ptr<io::RandomAccessFile>> OpenInputFile(
      const std::string& path) = 0;
  /// Open an input file for random access reading.
  ///
  /// This override assumes the given FileInfo validly represents the file's
  /// characteristics, and may optimize access depending on them (for example
  /// avoid querying the file size or its existence).
  virtual Result<std::shared_ptr<io::RandomAccessFile>> OpenInputFile(
      const FileInfo& info);

  /// Async version of OpenInputStream
  virtual Future<std::shared_ptr<io::InputStream>> OpenInputStreamAsync(
      const std::string& path);
  /// Async version of OpenInputStream
  virtual Future<std::shared_ptr<io::InputStream>> OpenInputStreamAsync(
      const FileInfo& info);

  /// Async version of OpenInputFile
  virtual Future<std::shared_ptr<io::RandomAccessFile>> OpenInputFileAsync(
      const std::string& path);
  /// Async version of OpenInputFile
  virtual Future<std::shared_ptr<io::RandomAccessFile>> OpenInputFileAsync(
      const FileInfo& info);

  /// Open an output stream for sequential writing.
  ///
  /// If the target already exists, existing data is truncated.
  virtual Result<std::shared_ptr<io::OutputStream>> OpenOutputStream(
      const std::string& path,
      const std::shared_ptr<const KeyValueMetadata>& metadata) = 0;
  Result<std::shared_ptr<io::OutputStream>> OpenOutputStream(const std::string& path);

  /// Open an output stream for appending.
  ///
  /// If the target doesn't exist, a new empty file is created.
  ///
  /// Note: some filesystem implementations do not support efficient appending
  /// to an existing file, in which case this method will return NotImplemented.
  /// Consider writing to multiple files (using e.g. the dataset layer) instead.
  virtual Result<std::shared_ptr<io::OutputStream>> OpenAppendStream(
      const std::string& path,
      const std::shared_ptr<const KeyValueMetadata>& metadata) = 0;
  Result<std::shared_ptr<io::OutputStream>> OpenAppendStream(const std::string& path);

 protected:
  explicit FileSystem(const io::IOContext& io_context = io::default_io_context())
      : io_context_(io_context) {}

  io::IOContext io_context_;
  // Whether metadata operations (such as GetFileInfo or OpenInputStream)
  // are cheap enough that the default async variants don't bother with
  // a thread pool.
  bool default_async_is_sync_ = true;
};

/// \brief A FileSystem implementation that delegates to another
/// implementation after prepending a fixed base path.
///
/// This is useful to expose a logical view of a subtree of a filesystem,
/// for example a directory in a LocalFileSystem.
/// This works on abstract paths, i.e. paths using forward slashes and
/// and a single root "/".  Windows paths are not guaranteed to work.
/// This makes no security guarantee.  For example, symlinks may allow to
/// "escape" the subtree and access other parts of the underlying filesystem.
class ARROW_EXPORT SubTreeFileSystem : public FileSystem {
 public:
  // This constructor may abort if base_path is invalid.
  explicit SubTreeFileSystem(const std::string& base_path,
                             std::shared_ptr<FileSystem> base_fs);
  ~SubTreeFileSystem() override;

  std::string type_name() const override { return "subtree"; }
  std::string base_path() const { return base_path_; }
  std::shared_ptr<FileSystem> base_fs() const { return base_fs_; }

  Result<std::string> NormalizePath(std::string path) override;

  bool Equals(const FileSystem& other) const override;

  /// \cond FALSE
  using FileSystem::GetFileInfo;
  /// \endcond
  Result<FileInfo> GetFileInfo(const std::string& path) override;
  Result<FileInfoVector> GetFileInfo(const FileSelector& select) override;

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
  Result<std::shared_ptr<io::InputStream>> OpenInputStream(const FileInfo& info) override;
  Result<std::shared_ptr<io::RandomAccessFile>> OpenInputFile(
      const std::string& path) override;
  Result<std::shared_ptr<io::RandomAccessFile>> OpenInputFile(
      const FileInfo& info) override;

  Future<std::shared_ptr<io::InputStream>> OpenInputStreamAsync(
      const std::string& path) override;
  Future<std::shared_ptr<io::InputStream>> OpenInputStreamAsync(
      const FileInfo& info) override;
  Future<std::shared_ptr<io::RandomAccessFile>> OpenInputFileAsync(
      const std::string& path) override;
  Future<std::shared_ptr<io::RandomAccessFile>> OpenInputFileAsync(
      const FileInfo& info) override;

  Result<std::shared_ptr<io::OutputStream>> OpenOutputStream(
      const std::string& path,
      const std::shared_ptr<const KeyValueMetadata>& metadata = {}) override;
  Result<std::shared_ptr<io::OutputStream>> OpenAppendStream(
      const std::string& path,
      const std::shared_ptr<const KeyValueMetadata>& metadata = {}) override;

 protected:
  SubTreeFileSystem() {}

  const std::string base_path_;
  std::shared_ptr<FileSystem> base_fs_;

  Result<std::string> PrependBase(const std::string& s) const;
  Result<std::string> PrependBaseNonEmpty(const std::string& s) const;
  Result<std::string> StripBase(const std::string& s) const;
  Status FixInfo(FileInfo* info) const;

  static Result<std::string> NormalizeBasePath(
      std::string base_path, const std::shared_ptr<FileSystem>& base_fs);
};

/// \brief A FileSystem implementation that delegates to another
/// implementation but inserts latencies at various points.
class ARROW_EXPORT SlowFileSystem : public FileSystem {
 public:
  SlowFileSystem(std::shared_ptr<FileSystem> base_fs,
                 std::shared_ptr<io::LatencyGenerator> latencies);
  SlowFileSystem(std::shared_ptr<FileSystem> base_fs, double average_latency);
  SlowFileSystem(std::shared_ptr<FileSystem> base_fs, double average_latency,
                 int32_t seed);

  std::string type_name() const override { return "slow"; }
  bool Equals(const FileSystem& other) const override;

  using FileSystem::GetFileInfo;
  Result<FileInfo> GetFileInfo(const std::string& path) override;
  Result<FileInfoVector> GetFileInfo(const FileSelector& select) override;

  Status CreateDir(const std::string& path, bool recursive = true) override;

  Status DeleteDir(const std::string& path) override;
  Status DeleteDirContents(const std::string& path, bool missing_dir_ok = false) override;
  Status DeleteRootDirContents() override;

  Status DeleteFile(const std::string& path) override;

  Status Move(const std::string& src, const std::string& dest) override;

  Status CopyFile(const std::string& src, const std::string& dest) override;

  Result<std::shared_ptr<io::InputStream>> OpenInputStream(
      const std::string& path) override;
  Result<std::shared_ptr<io::InputStream>> OpenInputStream(const FileInfo& info) override;
  Result<std::shared_ptr<io::RandomAccessFile>> OpenInputFile(
      const std::string& path) override;
  Result<std::shared_ptr<io::RandomAccessFile>> OpenInputFile(
      const FileInfo& info) override;
  Result<std::shared_ptr<io::OutputStream>> OpenOutputStream(
      const std::string& path,
      const std::shared_ptr<const KeyValueMetadata>& metadata = {}) override;
  Result<std::shared_ptr<io::OutputStream>> OpenAppendStream(
      const std::string& path,
      const std::shared_ptr<const KeyValueMetadata>& metadata = {}) override;

 protected:
  std::shared_ptr<FileSystem> base_fs_;
  std::shared_ptr<io::LatencyGenerator> latencies_;
};

/// \defgroup filesystem-factories Functions for creating FileSystem instances
///
/// @{

/// \brief Create a new FileSystem by URI
///
/// Recognized schemes are "file", "mock", "hdfs", "viewfs", "s3",
/// "gs" and "gcs".
///
/// \param[in] uri a URI-based path, ex: file:///some/local/path
/// \param[out] out_path (optional) Path inside the filesystem.
/// \return out_fs FileSystem instance.
ARROW_EXPORT
Result<std::shared_ptr<FileSystem>> FileSystemFromUri(const std::string& uri,
                                                      std::string* out_path = NULLPTR);

/// \brief Create a new FileSystem by URI with a custom IO context
///
/// Recognized schemes are "file", "mock", "hdfs", "viewfs", "s3",
/// "gs" and "gcs".
///
/// \param[in] uri a URI-based path, ex: file:///some/local/path
/// \param[in] io_context an IOContext which will be associated with the filesystem
/// \param[out] out_path (optional) Path inside the filesystem.
/// \return out_fs FileSystem instance.
ARROW_EXPORT
Result<std::shared_ptr<FileSystem>> FileSystemFromUri(const std::string& uri,
                                                      const io::IOContext& io_context,
                                                      std::string* out_path = NULLPTR);

/// \brief Create a new FileSystem by URI
///
/// Same as FileSystemFromUri, but in addition also recognize non-URIs
/// and treat them as local filesystem paths.  Only absolute local filesystem
/// paths are allowed.
ARROW_EXPORT
Result<std::shared_ptr<FileSystem>> FileSystemFromUriOrPath(
    const std::string& uri, std::string* out_path = NULLPTR);

/// \brief Create a new FileSystem by URI with a custom IO context
///
/// Same as FileSystemFromUri, but in addition also recognize non-URIs
/// and treat them as local filesystem paths.  Only absolute local filesystem
/// paths are allowed.
ARROW_EXPORT
Result<std::shared_ptr<FileSystem>> FileSystemFromUriOrPath(
    const std::string& uri, const io::IOContext& io_context,
    std::string* out_path = NULLPTR);

/// @}

/// \brief Copy files, including from one FileSystem to another
///
/// If a source and destination are resident in the same FileSystem FileSystem::CopyFile
/// will be used, otherwise the file will be opened as a stream in both FileSystems and
/// chunks copied from the source to the destination. No directories will be created.
ARROW_EXPORT
Status CopyFiles(const std::vector<FileLocator>& sources,
                 const std::vector<FileLocator>& destinations,
                 const io::IOContext& io_context = io::default_io_context(),
                 int64_t chunk_size = 1024 * 1024, bool use_threads = true);

/// \brief Copy selected files, including from one FileSystem to another
///
/// Directories will be created under the destination base directory as needed.
ARROW_EXPORT
Status CopyFiles(const std::shared_ptr<FileSystem>& source_fs,
                 const FileSelector& source_sel,
                 const std::shared_ptr<FileSystem>& destination_fs,
                 const std::string& destination_base_dir,
                 const io::IOContext& io_context = io::default_io_context(),
                 int64_t chunk_size = 1024 * 1024, bool use_threads = true);

struct FileSystemGlobalOptions {
  /// Path to a single PEM file holding all TLS CA certificates
  ///
  /// If empty, the underlying TLS library's defaults will be used.
  std::string tls_ca_file_path;

  /// Path to a directory holding TLS CA certificates in individual PEM files
  /// named along the OpenSSL "hashed" format.
  ///
  /// If empty, the underlying TLS library's defaults will be used.
  std::string tls_ca_dir_path;
};

/// EXPERIMENTAL: optional global initialization routine
///
/// This is for environments (such as manylinux) where the path
/// to TLS CA certificates needs to be configured at runtime.
ARROW_EXPORT
Status Initialize(const FileSystemGlobalOptions& options);

}  // namespace fs
}  // namespace arrow

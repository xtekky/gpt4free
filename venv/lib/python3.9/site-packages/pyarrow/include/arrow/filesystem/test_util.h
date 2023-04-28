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
#include <memory>
#include <string>
#include <vector>

#include "arrow/filesystem/filesystem.h"
#include "arrow/filesystem/mockfs.h"
#include "arrow/testing/visibility.h"
#include "arrow/util/counting_semaphore.h"

namespace arrow {
namespace fs {

static constexpr double kTimeSlack = 2.0;  // In seconds

static inline FileInfo File(std::string path) {
  return FileInfo(std::move(path), FileType::File);
}

static inline FileInfo Dir(std::string path) {
  return FileInfo(std::move(path), FileType::Directory);
}

// A subclass of MockFileSystem that blocks operations until an unlock method is
// called.
//
// This is intended for testing fine-grained ordering of filesystem operations.
//
// N.B. Only OpenOutputStream supports gating at the moment but this is simply because
//      it is all that has been needed so far.  Feel free to add support for more methods
//      as required.
class ARROW_TESTING_EXPORT GatedMockFilesystem : public internal::MockFileSystem {
 public:
  GatedMockFilesystem(TimePoint current_time,
                      const io::IOContext& = io::default_io_context());
  ~GatedMockFilesystem() override;

  Result<std::shared_ptr<io::OutputStream>> OpenOutputStream(
      const std::string& path,
      const std::shared_ptr<const KeyValueMetadata>& metadata = {}) override;

  // Wait until at least num_waiters are waiting on OpenOutputStream
  Status WaitForOpenOutputStream(uint32_t num_waiters);
  // Unlock `num_waiters` individual calls to OpenOutputStream
  Status UnlockOpenOutputStream(uint32_t num_waiters);

 private:
  util::CountingSemaphore open_output_sem_;
};

ARROW_TESTING_EXPORT
void CreateFile(FileSystem* fs, const std::string& path, const std::string& data);

// Sort a vector of FileInfo by lexicographic path order
ARROW_TESTING_EXPORT
void SortInfos(FileInfoVector* infos);

ARROW_TESTING_EXPORT
void CollectFileInfoGenerator(FileInfoGenerator gen, FileInfoVector* out_infos);

ARROW_TESTING_EXPORT
void AssertFileInfo(const FileInfo& info, const std::string& path, FileType type);

ARROW_TESTING_EXPORT
void AssertFileInfo(const FileInfo& info, const std::string& path, FileType type,
                    TimePoint mtime);

ARROW_TESTING_EXPORT
void AssertFileInfo(const FileInfo& info, const std::string& path, FileType type,
                    TimePoint mtime, int64_t size);

ARROW_TESTING_EXPORT
void AssertFileInfo(const FileInfo& info, const std::string& path, FileType type,
                    int64_t size);

ARROW_TESTING_EXPORT
void AssertFileInfo(FileSystem* fs, const std::string& path, FileType type);

ARROW_TESTING_EXPORT
void AssertFileInfo(FileSystem* fs, const std::string& path, FileType type,
                    TimePoint mtime);

ARROW_TESTING_EXPORT
void AssertFileInfo(FileSystem* fs, const std::string& path, FileType type,
                    TimePoint mtime, int64_t size);

ARROW_TESTING_EXPORT
void AssertFileInfo(FileSystem* fs, const std::string& path, FileType type, int64_t size);

ARROW_TESTING_EXPORT
void AssertFileContents(FileSystem* fs, const std::string& path,
                        const std::string& expected_data);

template <typename Duration>
void AssertDurationBetween(Duration d, double min_secs, double max_secs) {
  auto seconds = std::chrono::duration_cast<std::chrono::duration<double>>(d);
  ASSERT_GE(seconds.count(), min_secs);
  ASSERT_LE(seconds.count(), max_secs);
}

// Generic tests for FileSystem implementations.
// To use this class, subclass both from it and ::testing::Test,
// implement GetEmptyFileSystem(), and use GENERIC_FS_TEST_FUNCTIONS()
// to define the various tests.
class ARROW_TESTING_EXPORT GenericFileSystemTest {
 public:
  virtual ~GenericFileSystemTest();

  void TestEmpty();
  void TestNormalizePath();
  void TestCreateDir();
  void TestDeleteDir();
  void TestDeleteDirContents();
  void TestDeleteRootDirContents();
  void TestDeleteFile();
  void TestDeleteFiles();
  void TestMoveFile();
  void TestMoveDir();
  void TestCopyFile();
  void TestGetFileInfo();
  void TestGetFileInfoVector();
  void TestGetFileInfoSelector();
  void TestGetFileInfoSelectorWithRecursion();
  void TestGetFileInfoAsync();
  void TestGetFileInfoGenerator();
  void TestOpenOutputStream();
  void TestOpenAppendStream();
  void TestOpenInputStream();
  void TestOpenInputStreamWithFileInfo();
  void TestOpenInputStreamAsync();
  void TestOpenInputFile();
  void TestOpenInputFileWithFileInfo();
  void TestOpenInputFileAsync();
  void TestSpecialChars();

 protected:
  // This function should return the filesystem under test.
  virtual std::shared_ptr<FileSystem> GetEmptyFileSystem() = 0;

  // Override the following functions to specify deviations from expected
  // filesystem semantics.
  // - Whether the filesystem may "implicitly" create intermediate directories
  virtual bool have_implicit_directories() const { return false; }
  // - Whether the filesystem may allow writing a file "over" a directory
  virtual bool allow_write_file_over_dir() const { return false; }
  // - Whether the filesystem allows reading a directory
  virtual bool allow_read_dir_as_file() const { return false; }
  // - Whether the filesystem allows moving a directory
  virtual bool allow_move_dir() const { return true; }
  // - Whether the filesystem allows moving a directory "over" a non-empty destination
  virtual bool allow_move_dir_over_non_empty_dir() const { return false; }
  // - Whether the filesystem allows appending to a file
  virtual bool allow_append_to_file() const { return true; }
  // - Whether the filesystem allows appending to a new (not existent yet) file
  virtual bool allow_append_to_new_file() const { return true; }
  // - Whether the filesystem supports directory modification times
  virtual bool have_directory_mtimes() const { return true; }
  // - Whether some directory tree deletion tests may fail randomly
  virtual bool have_flaky_directory_tree_deletion() const { return false; }
  // - Whether the filesystem stores some metadata alongside files
  virtual bool have_file_metadata() const { return false; }

  void TestEmpty(FileSystem* fs);
  void TestNormalizePath(FileSystem* fs);
  void TestCreateDir(FileSystem* fs);
  void TestDeleteDir(FileSystem* fs);
  void TestDeleteDirContents(FileSystem* fs);
  void TestDeleteRootDirContents(FileSystem* fs);
  void TestDeleteFile(FileSystem* fs);
  void TestDeleteFiles(FileSystem* fs);
  void TestMoveFile(FileSystem* fs);
  void TestMoveDir(FileSystem* fs);
  void TestCopyFile(FileSystem* fs);
  void TestGetFileInfo(FileSystem* fs);
  void TestGetFileInfoVector(FileSystem* fs);
  void TestGetFileInfoSelector(FileSystem* fs);
  void TestGetFileInfoSelectorWithRecursion(FileSystem* fs);
  void TestGetFileInfoAsync(FileSystem* fs);
  void TestGetFileInfoGenerator(FileSystem* fs);
  void TestOpenOutputStream(FileSystem* fs);
  void TestOpenAppendStream(FileSystem* fs);
  void TestOpenInputStream(FileSystem* fs);
  void TestOpenInputStreamWithFileInfo(FileSystem* fs);
  void TestOpenInputStreamAsync(FileSystem* fs);
  void TestOpenInputFile(FileSystem* fs);
  void TestOpenInputFileWithFileInfo(FileSystem* fs);
  void TestOpenInputFileAsync(FileSystem* fs);
  void TestSpecialChars(FileSystem* fs);
};

#define GENERIC_FS_TEST_FUNCTION(TEST_MACRO, TEST_CLASS, NAME) \
  TEST_MACRO(TEST_CLASS, NAME) { this->Test##NAME(); }

#define GENERIC_FS_TEST_FUNCTIONS_MACROS(TEST_MACRO, TEST_CLASS)                     \
  GENERIC_FS_TEST_FUNCTION(TEST_MACRO, TEST_CLASS, Empty)                            \
  GENERIC_FS_TEST_FUNCTION(TEST_MACRO, TEST_CLASS, NormalizePath)                    \
  GENERIC_FS_TEST_FUNCTION(TEST_MACRO, TEST_CLASS, CreateDir)                        \
  GENERIC_FS_TEST_FUNCTION(TEST_MACRO, TEST_CLASS, DeleteDir)                        \
  GENERIC_FS_TEST_FUNCTION(TEST_MACRO, TEST_CLASS, DeleteDirContents)                \
  GENERIC_FS_TEST_FUNCTION(TEST_MACRO, TEST_CLASS, DeleteRootDirContents)            \
  GENERIC_FS_TEST_FUNCTION(TEST_MACRO, TEST_CLASS, DeleteFile)                       \
  GENERIC_FS_TEST_FUNCTION(TEST_MACRO, TEST_CLASS, DeleteFiles)                      \
  GENERIC_FS_TEST_FUNCTION(TEST_MACRO, TEST_CLASS, MoveFile)                         \
  GENERIC_FS_TEST_FUNCTION(TEST_MACRO, TEST_CLASS, MoveDir)                          \
  GENERIC_FS_TEST_FUNCTION(TEST_MACRO, TEST_CLASS, CopyFile)                         \
  GENERIC_FS_TEST_FUNCTION(TEST_MACRO, TEST_CLASS, GetFileInfo)                      \
  GENERIC_FS_TEST_FUNCTION(TEST_MACRO, TEST_CLASS, GetFileInfoVector)                \
  GENERIC_FS_TEST_FUNCTION(TEST_MACRO, TEST_CLASS, GetFileInfoSelector)              \
  GENERIC_FS_TEST_FUNCTION(TEST_MACRO, TEST_CLASS, GetFileInfoSelectorWithRecursion) \
  GENERIC_FS_TEST_FUNCTION(TEST_MACRO, TEST_CLASS, GetFileInfoAsync)                 \
  GENERIC_FS_TEST_FUNCTION(TEST_MACRO, TEST_CLASS, GetFileInfoGenerator)             \
  GENERIC_FS_TEST_FUNCTION(TEST_MACRO, TEST_CLASS, OpenOutputStream)                 \
  GENERIC_FS_TEST_FUNCTION(TEST_MACRO, TEST_CLASS, OpenAppendStream)                 \
  GENERIC_FS_TEST_FUNCTION(TEST_MACRO, TEST_CLASS, OpenInputStream)                  \
  GENERIC_FS_TEST_FUNCTION(TEST_MACRO, TEST_CLASS, OpenInputStreamWithFileInfo)      \
  GENERIC_FS_TEST_FUNCTION(TEST_MACRO, TEST_CLASS, OpenInputStreamAsync)             \
  GENERIC_FS_TEST_FUNCTION(TEST_MACRO, TEST_CLASS, OpenInputFile)                    \
  GENERIC_FS_TEST_FUNCTION(TEST_MACRO, TEST_CLASS, OpenInputFileWithFileInfo)        \
  GENERIC_FS_TEST_FUNCTION(TEST_MACRO, TEST_CLASS, OpenInputFileAsync)               \
  GENERIC_FS_TEST_FUNCTION(TEST_MACRO, TEST_CLASS, SpecialChars)

#define GENERIC_FS_TEST_FUNCTIONS(TEST_CLASS) \
  GENERIC_FS_TEST_FUNCTIONS_MACROS(TEST_F, TEST_CLASS)

#define GENERIC_FS_TYPED_TEST_FUNCTIONS(TEST_CLASS) \
  GENERIC_FS_TEST_FUNCTIONS_MACROS(TYPED_TEST, TEST_CLASS)

}  // namespace fs
}  // namespace arrow

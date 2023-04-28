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
#include "arrow/util/macros.h"
#include "arrow/python/common.h"
#include "arrow/python/visibility.h"

namespace arrow {
namespace py {
namespace fs {

class ARROW_PYTHON_EXPORT PyFileSystemVtable {
 public:
  std::function<void(PyObject*, std::string* out)> get_type_name;
  std::function<bool(PyObject*, const arrow::fs::FileSystem& other)> equals;

  std::function<void(PyObject*, const std::string& path, arrow::fs::FileInfo* out)>
      get_file_info;
  std::function<void(PyObject*, const std::vector<std::string>& paths,
                     std::vector<arrow::fs::FileInfo>* out)>
      get_file_info_vector;
  std::function<void(PyObject*, const arrow::fs::FileSelector&,
                     std::vector<arrow::fs::FileInfo>* out)>
      get_file_info_selector;

  std::function<void(PyObject*, const std::string& path, bool)> create_dir;
  std::function<void(PyObject*, const std::string& path)> delete_dir;
  std::function<void(PyObject*, const std::string& path, bool)> delete_dir_contents;
  std::function<void(PyObject*)> delete_root_dir_contents;
  std::function<void(PyObject*, const std::string& path)> delete_file;
  std::function<void(PyObject*, const std::string& src, const std::string& dest)> move;
  std::function<void(PyObject*, const std::string& src, const std::string& dest)>
      copy_file;

  std::function<void(PyObject*, const std::string& path,
                     std::shared_ptr<io::InputStream>* out)>
      open_input_stream;
  std::function<void(PyObject*, const std::string& path,
                     std::shared_ptr<io::RandomAccessFile>* out)>
      open_input_file;
  std::function<void(PyObject*, const std::string& path,
                     const std::shared_ptr<const KeyValueMetadata>&,
                     std::shared_ptr<io::OutputStream>* out)>
      open_output_stream;
  std::function<void(PyObject*, const std::string& path,
                     const std::shared_ptr<const KeyValueMetadata>&,
                     std::shared_ptr<io::OutputStream>* out)>
      open_append_stream;

  std::function<void(PyObject*, const std::string& path, std::string* out)>
      normalize_path;
};

class ARROW_PYTHON_EXPORT PyFileSystem : public arrow::fs::FileSystem {
 public:
  PyFileSystem(PyObject* handler, PyFileSystemVtable vtable);
  ~PyFileSystem() override;

  static std::shared_ptr<PyFileSystem> Make(PyObject* handler, PyFileSystemVtable vtable);

  std::string type_name() const override;

  bool Equals(const FileSystem& other) const override;

  Result<arrow::fs::FileInfo> GetFileInfo(const std::string& path) override;
  Result<std::vector<arrow::fs::FileInfo>> GetFileInfo(
      const std::vector<std::string>& paths) override;
  Result<std::vector<arrow::fs::FileInfo>> GetFileInfo(
      const arrow::fs::FileSelector& select) override;

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

  Result<std::string> NormalizePath(std::string path) override;

  PyObject* handler() const { return handler_.obj(); }

 private:
  OwnedRefNoGIL handler_;
  PyFileSystemVtable vtable_;
};

}  // namespace fs
}  // namespace py
}  // namespace arrow

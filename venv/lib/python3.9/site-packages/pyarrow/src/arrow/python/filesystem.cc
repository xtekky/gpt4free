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

#include "arrow/util/logging.h"
#include "arrow/python/filesystem.h"

namespace arrow {

using fs::FileInfo;
using fs::FileSelector;

namespace py {
namespace fs {

PyFileSystem::PyFileSystem(PyObject* handler, PyFileSystemVtable vtable)
    : handler_(handler), vtable_(std::move(vtable)) {
  Py_INCREF(handler);
}

PyFileSystem::~PyFileSystem() {}

std::shared_ptr<PyFileSystem> PyFileSystem::Make(PyObject* handler,
                                                 PyFileSystemVtable vtable) {
  return std::make_shared<PyFileSystem>(handler, std::move(vtable));
}

std::string PyFileSystem::type_name() const {
  std::string result;
  auto st = SafeCallIntoPython([&]() -> Status {
    vtable_.get_type_name(handler_.obj(), &result);
    if (PyErr_Occurred()) {
      PyErr_WriteUnraisable(handler_.obj());
    }
    return Status::OK();
  });
  ARROW_UNUSED(st);
  return result;
}

bool PyFileSystem::Equals(const FileSystem& other) const {
  bool result;
  auto st = SafeCallIntoPython([&]() -> Status {
    result = vtable_.equals(handler_.obj(), other);
    if (PyErr_Occurred()) {
      PyErr_WriteUnraisable(handler_.obj());
    }
    return Status::OK();
  });
  ARROW_UNUSED(st);
  return result;
}

Result<FileInfo> PyFileSystem::GetFileInfo(const std::string& path) {
  FileInfo info;

  auto st = SafeCallIntoPython([&]() -> Status {
    vtable_.get_file_info(handler_.obj(), path, &info);
    return CheckPyError();
  });
  RETURN_NOT_OK(st);
  return info;
}

Result<std::vector<FileInfo>> PyFileSystem::GetFileInfo(
    const std::vector<std::string>& paths) {
  std::vector<FileInfo> infos;

  auto st = SafeCallIntoPython([&]() -> Status {
    vtable_.get_file_info_vector(handler_.obj(), paths, &infos);
    return CheckPyError();
  });
  RETURN_NOT_OK(st);
  return infos;
}

Result<std::vector<FileInfo>> PyFileSystem::GetFileInfo(const FileSelector& select) {
  std::vector<FileInfo> infos;

  auto st = SafeCallIntoPython([&]() -> Status {
    vtable_.get_file_info_selector(handler_.obj(), select, &infos);
    return CheckPyError();
  });
  RETURN_NOT_OK(st);
  return infos;
}

Status PyFileSystem::CreateDir(const std::string& path, bool recursive) {
  return SafeCallIntoPython([&]() -> Status {
    vtable_.create_dir(handler_.obj(), path, recursive);
    return CheckPyError();
  });
}

Status PyFileSystem::DeleteDir(const std::string& path) {
  return SafeCallIntoPython([&]() -> Status {
    vtable_.delete_dir(handler_.obj(), path);
    return CheckPyError();
  });
}

Status PyFileSystem::DeleteDirContents(const std::string& path, bool missing_dir_ok) {
  return SafeCallIntoPython([&]() -> Status {
    vtable_.delete_dir_contents(handler_.obj(), path, missing_dir_ok);
    return CheckPyError();
  });
}

Status PyFileSystem::DeleteRootDirContents() {
  return SafeCallIntoPython([&]() -> Status {
    vtable_.delete_root_dir_contents(handler_.obj());
    return CheckPyError();
  });
}

Status PyFileSystem::DeleteFile(const std::string& path) {
  return SafeCallIntoPython([&]() -> Status {
    vtable_.delete_file(handler_.obj(), path);
    return CheckPyError();
  });
}

Status PyFileSystem::Move(const std::string& src, const std::string& dest) {
  return SafeCallIntoPython([&]() -> Status {
    vtable_.move(handler_.obj(), src, dest);
    return CheckPyError();
  });
}

Status PyFileSystem::CopyFile(const std::string& src, const std::string& dest) {
  return SafeCallIntoPython([&]() -> Status {
    vtable_.copy_file(handler_.obj(), src, dest);
    return CheckPyError();
  });
}

Result<std::shared_ptr<io::InputStream>> PyFileSystem::OpenInputStream(
    const std::string& path) {
  std::shared_ptr<io::InputStream> stream;
  auto st = SafeCallIntoPython([&]() -> Status {
    vtable_.open_input_stream(handler_.obj(), path, &stream);
    return CheckPyError();
  });
  RETURN_NOT_OK(st);
  return stream;
}

Result<std::shared_ptr<io::RandomAccessFile>> PyFileSystem::OpenInputFile(
    const std::string& path) {
  std::shared_ptr<io::RandomAccessFile> stream;
  auto st = SafeCallIntoPython([&]() -> Status {
    vtable_.open_input_file(handler_.obj(), path, &stream);
    return CheckPyError();
  });
  RETURN_NOT_OK(st);
  return stream;
}

Result<std::shared_ptr<io::OutputStream>> PyFileSystem::OpenOutputStream(
    const std::string& path, const std::shared_ptr<const KeyValueMetadata>& metadata) {
  std::shared_ptr<io::OutputStream> stream;
  auto st = SafeCallIntoPython([&]() -> Status {
    vtable_.open_output_stream(handler_.obj(), path, metadata, &stream);
    return CheckPyError();
  });
  RETURN_NOT_OK(st);
  return stream;
}

Result<std::shared_ptr<io::OutputStream>> PyFileSystem::OpenAppendStream(
    const std::string& path, const std::shared_ptr<const KeyValueMetadata>& metadata) {
  std::shared_ptr<io::OutputStream> stream;
  auto st = SafeCallIntoPython([&]() -> Status {
    vtable_.open_append_stream(handler_.obj(), path, metadata, &stream);
    return CheckPyError();
  });
  RETURN_NOT_OK(st);
  return stream;
}

Result<std::string> PyFileSystem::NormalizePath(std::string path) {
  std::string normalized;
  auto st = SafeCallIntoPython([&]() -> Status {
    vtable_.normalize_path(handler_.obj(), path, &normalized);
    return CheckPyError();
  });
  RETURN_NOT_OK(st);
  return normalized;
}

}  // namespace fs
}  // namespace py
}  // namespace arrow

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

#include "arrow/io/interfaces.h"
#include "arrow/io/transform.h"

#include "arrow/python/common.h"
#include "arrow/python/visibility.h"

namespace arrow {
namespace py {

class ARROW_NO_EXPORT PythonFile;

class ARROW_PYTHON_EXPORT PyReadableFile : public io::RandomAccessFile {
 public:
  explicit PyReadableFile(PyObject* file);
  ~PyReadableFile() override;

  Status Close() override;
  Status Abort() override;
  bool closed() const override;

  Result<int64_t> Read(int64_t nbytes, void* out) override;
  Result<std::shared_ptr<Buffer>> Read(int64_t nbytes) override;

  // Thread-safe version
  Result<int64_t> ReadAt(int64_t position, int64_t nbytes, void* out) override;

  // Thread-safe version
  Result<std::shared_ptr<Buffer>> ReadAt(int64_t position, int64_t nbytes) override;

  Result<int64_t> GetSize() override;

  Status Seek(int64_t position) override;

  Result<int64_t> Tell() const override;

 private:
  std::unique_ptr<PythonFile> file_;
};

class ARROW_PYTHON_EXPORT PyOutputStream : public io::OutputStream {
 public:
  explicit PyOutputStream(PyObject* file);
  ~PyOutputStream() override;

  Status Close() override;
  Status Abort() override;
  bool closed() const override;
  Result<int64_t> Tell() const override;
  Status Write(const void* data, int64_t nbytes) override;
  Status Write(const std::shared_ptr<Buffer>& buffer) override;

 private:
  std::unique_ptr<PythonFile> file_;
  int64_t position_;
};

// TODO(wesm): seekable output files

// A Buffer subclass that keeps a PyObject reference throughout its
// lifetime, such that the Python object is kept alive as long as the
// C++ buffer is still needed.
// Keeping the reference in a Python wrapper would be incorrect as
// the Python wrapper can get destroyed even though the wrapped C++
// buffer is still alive (ARROW-2270).
class ARROW_PYTHON_EXPORT PyForeignBuffer : public Buffer {
 public:
  static Status Make(const uint8_t* data, int64_t size, PyObject* base,
                     std::shared_ptr<Buffer>* out);

 private:
  PyForeignBuffer(const uint8_t* data, int64_t size, PyObject* base)
      : Buffer(data, size) {
    Py_INCREF(base);
    base_.reset(base);
  }

  OwnedRefNoGIL base_;
};

// All this rigamarole because Cython is really poor with std::function<>

using TransformCallback = std::function<void(
    PyObject*, const std::shared_ptr<Buffer>& src, std::shared_ptr<Buffer>* out)>;

struct TransformInputStreamVTable {
  TransformCallback transform;
};

ARROW_PYTHON_EXPORT
std::shared_ptr<::arrow::io::InputStream> MakeTransformInputStream(
    std::shared_ptr<::arrow::io::InputStream> wrapped, TransformInputStreamVTable vtable,
    PyObject* arg);

using StreamWrapFunc = std::function<Result<std::shared_ptr<io::InputStream>>(
    std::shared_ptr<io::InputStream>)>;
ARROW_PYTHON_EXPORT
std::shared_ptr<StreamWrapFunc> MakeStreamTransformFunc(TransformInputStreamVTable vtable,
                                                        PyObject* handler);
}  // namespace py
}  // namespace arrow

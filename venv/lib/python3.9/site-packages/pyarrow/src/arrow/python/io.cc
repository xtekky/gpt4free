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

#include "io.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>

#include "arrow/io/memory.h"
#include "arrow/memory_pool.h"
#include "arrow/status.h"
#include "arrow/util/logging.h"

#include "arrow/python/common.h"
#include "arrow/python/pyarrow.h"

namespace arrow {

using arrow::io::TransformInputStream;

namespace py {

// ----------------------------------------------------------------------
// Python file

// A common interface to a Python file-like object. Must acquire GIL before
// calling any methods
class PythonFile {
 public:
  explicit PythonFile(PyObject* file) : file_(file), checked_read_buffer_(false) {
    Py_INCREF(file);
  }

  Status CheckClosed() const {
    if (!file_) {
      return Status::Invalid("operation on closed Python file");
    }
    return Status::OK();
  }

  Status Close() {
    if (file_) {
      PyObject* result = cpp_PyObject_CallMethod(file_.obj(), "close", "()");
      Py_XDECREF(result);
      file_.reset();
      PY_RETURN_IF_ERROR(StatusCode::IOError);
    }
    return Status::OK();
  }

  Status Abort() {
    file_.reset();
    return Status::OK();
  }

  bool closed() const {
    if (!file_) {
      return true;
    }
    PyObject* result = PyObject_GetAttrString(file_.obj(), "closed");
    if (result == NULL) {
      // Can't propagate the error, so write it out and return an arbitrary value
      PyErr_WriteUnraisable(NULL);
      return true;
    }
    int ret = PyObject_IsTrue(result);
    Py_XDECREF(result);
    if (ret < 0) {
      PyErr_WriteUnraisable(NULL);
      return true;
    }
    return ret != 0;
  }

  Status Seek(int64_t position, int whence) {
    RETURN_NOT_OK(CheckClosed());

    // whence: 0 for relative to start of file, 2 for end of file
    PyObject* result = cpp_PyObject_CallMethod(file_.obj(), "seek", "(ni)",
                                               static_cast<Py_ssize_t>(position), whence);
    Py_XDECREF(result);
    PY_RETURN_IF_ERROR(StatusCode::IOError);
    return Status::OK();
  }

  Status Read(int64_t nbytes, PyObject** out) {
    RETURN_NOT_OK(CheckClosed());

    PyObject* result = cpp_PyObject_CallMethod(file_.obj(), "read", "(n)",
                                               static_cast<Py_ssize_t>(nbytes));
    PY_RETURN_IF_ERROR(StatusCode::IOError);
    *out = result;
    return Status::OK();
  }

  Status ReadBuffer(int64_t nbytes, PyObject** out) {
    PyObject* result = cpp_PyObject_CallMethod(file_.obj(), "read_buffer", "(n)",
                                               static_cast<Py_ssize_t>(nbytes));
    PY_RETURN_IF_ERROR(StatusCode::IOError);
    *out = result;
    return Status::OK();
  }

  Status Write(const void* data, int64_t nbytes) {
    RETURN_NOT_OK(CheckClosed());

    // Since the data isn't owned, we have to make a copy
    PyObject* py_data =
        PyBytes_FromStringAndSize(reinterpret_cast<const char*>(data), nbytes);
    PY_RETURN_IF_ERROR(StatusCode::IOError);

    PyObject* result = cpp_PyObject_CallMethod(file_.obj(), "write", "(O)", py_data);
    Py_XDECREF(py_data);
    Py_XDECREF(result);
    PY_RETURN_IF_ERROR(StatusCode::IOError);
    return Status::OK();
  }

  Status Write(const std::shared_ptr<Buffer>& buffer) {
    RETURN_NOT_OK(CheckClosed());

    PyObject* py_data = wrap_buffer(buffer);
    PY_RETURN_IF_ERROR(StatusCode::IOError);

    PyObject* result = cpp_PyObject_CallMethod(file_.obj(), "write", "(O)", py_data);
    Py_XDECREF(py_data);
    Py_XDECREF(result);
    PY_RETURN_IF_ERROR(StatusCode::IOError);
    return Status::OK();
  }

  Result<int64_t> Tell() {
    RETURN_NOT_OK(CheckClosed());

    PyObject* result = cpp_PyObject_CallMethod(file_.obj(), "tell", "()");
    PY_RETURN_IF_ERROR(StatusCode::IOError);

    int64_t position = PyLong_AsLongLong(result);
    Py_DECREF(result);

    // PyLong_AsLongLong can raise OverflowError
    PY_RETURN_IF_ERROR(StatusCode::IOError);
    return position;
  }

  std::mutex& lock() { return lock_; }

  bool HasReadBuffer() {
    if (!checked_read_buffer_) {  // we don't want to check this each time
      has_read_buffer_ = PyObject_HasAttrString(file_.obj(), "read_buffer") == 1;
      checked_read_buffer_ = true;
    }
    return has_read_buffer_;
  }

 private:
  std::mutex lock_;
  OwnedRefNoGIL file_;
  bool has_read_buffer_;
  bool checked_read_buffer_;
};

// ----------------------------------------------------------------------
// Seekable input stream

PyReadableFile::PyReadableFile(PyObject* file) { file_.reset(new PythonFile(file)); }

// The destructor does not close the underlying Python file object, as
// there may be multiple references to it.  Instead let the Python
// destructor do its job.
PyReadableFile::~PyReadableFile() {}

Status PyReadableFile::Abort() {
  return SafeCallIntoPython([this]() { return file_->Abort(); });
}

Status PyReadableFile::Close() {
  return SafeCallIntoPython([this]() { return file_->Close(); });
}

bool PyReadableFile::closed() const {
  bool res;
  Status st = SafeCallIntoPython([this, &res]() {
    res = file_->closed();
    return Status::OK();
  });
  return res;
}

Status PyReadableFile::Seek(int64_t position) {
  return SafeCallIntoPython([=] { return file_->Seek(position, 0); });
}

Result<int64_t> PyReadableFile::Tell() const {
  return SafeCallIntoPython([=]() -> Result<int64_t> { return file_->Tell(); });
}

Result<int64_t> PyReadableFile::Read(int64_t nbytes, void* out) {
  return SafeCallIntoPython([=]() -> Result<int64_t> {
    OwnedRef bytes;
    RETURN_NOT_OK(file_->Read(nbytes, bytes.ref()));
    PyObject* bytes_obj = bytes.obj();
    DCHECK(bytes_obj != NULL);

    Py_buffer py_buf;
    if (!PyObject_GetBuffer(bytes_obj, &py_buf, PyBUF_ANY_CONTIGUOUS)) {
      const uint8_t* data = reinterpret_cast<const uint8_t*>(py_buf.buf);
      std::memcpy(out, data, py_buf.len);
      int64_t len = py_buf.len;
      PyBuffer_Release(&py_buf);
      return len;
    } else {
      return Status::TypeError(
          "Python file read() should have returned a bytes object or an object "
          "supporting the buffer protocol, got '",
          Py_TYPE(bytes_obj)->tp_name, "' (did you open the file in binary mode?)");
    }
  });
}

Result<std::shared_ptr<Buffer>> PyReadableFile::Read(int64_t nbytes) {
  return SafeCallIntoPython([=]() -> Result<std::shared_ptr<Buffer>> {
    OwnedRef buffer_obj;
    if (file_->HasReadBuffer()) {
      RETURN_NOT_OK(file_->ReadBuffer(nbytes, buffer_obj.ref()));
    } else {
      RETURN_NOT_OK(file_->Read(nbytes, buffer_obj.ref()));
    }
    DCHECK(buffer_obj.obj() != NULL);

    return PyBuffer::FromPyObject(buffer_obj.obj());
  });
}

Result<int64_t> PyReadableFile::ReadAt(int64_t position, int64_t nbytes, void* out) {
  std::lock_guard<std::mutex> guard(file_->lock());
  return SafeCallIntoPython([=]() -> Result<int64_t> {
    RETURN_NOT_OK(Seek(position));
    return Read(nbytes, out);
  });
}

Result<std::shared_ptr<Buffer>> PyReadableFile::ReadAt(int64_t position, int64_t nbytes) {
  std::lock_guard<std::mutex> guard(file_->lock());
  return SafeCallIntoPython([=]() -> Result<std::shared_ptr<Buffer>> {
    RETURN_NOT_OK(Seek(position));
    return Read(nbytes);
  });
}

Result<int64_t> PyReadableFile::GetSize() {
  return SafeCallIntoPython([=]() -> Result<int64_t> {
    ARROW_ASSIGN_OR_RAISE(int64_t current_position, file_->Tell());
    RETURN_NOT_OK(file_->Seek(0, 2));

    ARROW_ASSIGN_OR_RAISE(int64_t file_size, file_->Tell());
    // Restore previous file position
    RETURN_NOT_OK(file_->Seek(current_position, 0));

    return file_size;
  });
}

// ----------------------------------------------------------------------
// Output stream

PyOutputStream::PyOutputStream(PyObject* file) : position_(0) {
  file_.reset(new PythonFile(file));
}

// The destructor does not close the underlying Python file object, as
// there may be multiple references to it.  Instead let the Python
// destructor do its job.
PyOutputStream::~PyOutputStream() {}

Status PyOutputStream::Abort() {
  return SafeCallIntoPython([=]() { return file_->Abort(); });
}

Status PyOutputStream::Close() {
  return SafeCallIntoPython([=]() { return file_->Close(); });
}

bool PyOutputStream::closed() const {
  bool res;
  Status st = SafeCallIntoPython([this, &res]() {
    res = file_->closed();
    return Status::OK();
  });
  return res;
}

Result<int64_t> PyOutputStream::Tell() const { return position_; }

Status PyOutputStream::Write(const void* data, int64_t nbytes) {
  return SafeCallIntoPython([=]() {
    position_ += nbytes;
    return file_->Write(data, nbytes);
  });
}

Status PyOutputStream::Write(const std::shared_ptr<Buffer>& buffer) {
  return SafeCallIntoPython([=]() {
    position_ += buffer->size();
    return file_->Write(buffer);
  });
}

// ----------------------------------------------------------------------
// Foreign buffer

Status PyForeignBuffer::Make(const uint8_t* data, int64_t size, PyObject* base,
                             std::shared_ptr<Buffer>* out) {
  PyForeignBuffer* buf = new PyForeignBuffer(data, size, base);
  if (buf == NULL) {
    return Status::OutOfMemory("could not allocate foreign buffer object");
  } else {
    *out = std::shared_ptr<Buffer>(buf);
    return Status::OK();
  }
}

// ----------------------------------------------------------------------
// TransformInputStream::TransformFunc wrapper

struct TransformFunctionWrapper {
  TransformFunctionWrapper(TransformCallback cb, PyObject* arg)
      : cb_(std::move(cb)), arg_(std::make_shared<OwnedRefNoGIL>(arg)) {
    Py_INCREF(arg);
  }

  Result<std::shared_ptr<Buffer>> operator()(const std::shared_ptr<Buffer>& src) {
    return SafeCallIntoPython([=]() -> Result<std::shared_ptr<Buffer>> {
      std::shared_ptr<Buffer> dest;
      cb_(arg_->obj(), src, &dest);
      RETURN_NOT_OK(CheckPyError());
      return dest;
    });
  }

 protected:
  // Need to wrap OwnedRefNoGIL because std::function needs the callable
  // to be copy-constructible...
  TransformCallback cb_;
  std::shared_ptr<OwnedRefNoGIL> arg_;
};

std::shared_ptr<::arrow::io::InputStream> MakeTransformInputStream(
    std::shared_ptr<::arrow::io::InputStream> wrapped, TransformInputStreamVTable vtable,
    PyObject* handler) {
  TransformInputStream::TransformFunc transform(
      TransformFunctionWrapper{std::move(vtable.transform), handler});
  return std::make_shared<TransformInputStream>(std::move(wrapped), std::move(transform));
}

std::shared_ptr<StreamWrapFunc> MakeStreamTransformFunc(TransformInputStreamVTable vtable,
                                                        PyObject* handler) {
  TransformInputStream::TransformFunc transform(
      TransformFunctionWrapper{std::move(vtable.transform), handler});
  StreamWrapFunc func = [transform](std::shared_ptr<::arrow::io::InputStream> wrapped) {
    return std::make_shared<TransformInputStream>(wrapped, transform);
  };
  return std::make_shared<StreamWrapFunc>(func);
}

}  // namespace py
}  // namespace arrow

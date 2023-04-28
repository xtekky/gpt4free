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

#include "arrow/python/common.h"

#include <cstdlib>
#include <mutex>
#include <string>

#include "arrow/memory_pool.h"
#include "arrow/status.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/logging.h"

#include "arrow/python/helpers.h"

namespace arrow {

using internal::checked_cast;

namespace py {

static std::mutex memory_pool_mutex;
static MemoryPool* default_python_pool = nullptr;

void set_default_memory_pool(MemoryPool* pool) {
  std::lock_guard<std::mutex> guard(memory_pool_mutex);
  default_python_pool = pool;
}

MemoryPool* get_memory_pool() {
  std::lock_guard<std::mutex> guard(memory_pool_mutex);
  if (default_python_pool) {
    return default_python_pool;
  } else {
    return default_memory_pool();
  }
}

// ----------------------------------------------------------------------
// PythonErrorDetail

namespace {

const char kErrorDetailTypeId[] = "arrow::py::PythonErrorDetail";

// Try to match the Python exception type with an appropriate Status code
StatusCode MapPyError(PyObject* exc_type) {
  StatusCode code;

  if (PyErr_GivenExceptionMatches(exc_type, PyExc_MemoryError)) {
    code = StatusCode::OutOfMemory;
  } else if (PyErr_GivenExceptionMatches(exc_type, PyExc_IndexError)) {
    code = StatusCode::IndexError;
  } else if (PyErr_GivenExceptionMatches(exc_type, PyExc_KeyError)) {
    code = StatusCode::KeyError;
  } else if (PyErr_GivenExceptionMatches(exc_type, PyExc_TypeError)) {
    code = StatusCode::TypeError;
  } else if (PyErr_GivenExceptionMatches(exc_type, PyExc_ValueError) ||
             PyErr_GivenExceptionMatches(exc_type, PyExc_OverflowError)) {
    code = StatusCode::Invalid;
  } else if (PyErr_GivenExceptionMatches(exc_type, PyExc_EnvironmentError)) {
    code = StatusCode::IOError;
  } else if (PyErr_GivenExceptionMatches(exc_type, PyExc_NotImplementedError)) {
    code = StatusCode::NotImplemented;
  } else {
    code = StatusCode::UnknownError;
  }
  return code;
}

// PythonErrorDetail indicates a Python exception was raised.
class PythonErrorDetail : public StatusDetail {
 public:
  const char* type_id() const override { return kErrorDetailTypeId; }

  std::string ToString() const override {
    // This is simple enough not to need the GIL
    const auto ty = reinterpret_cast<const PyTypeObject*>(exc_type_.obj());
    // XXX Should we also print traceback?
    return std::string("Python exception: ") + ty->tp_name;
  }

  void RestorePyError() const {
    Py_INCREF(exc_type_.obj());
    Py_INCREF(exc_value_.obj());
    Py_INCREF(exc_traceback_.obj());
    PyErr_Restore(exc_type_.obj(), exc_value_.obj(), exc_traceback_.obj());
  }

  PyObject* exc_type() const { return exc_type_.obj(); }

  PyObject* exc_value() const { return exc_value_.obj(); }

  static std::shared_ptr<PythonErrorDetail> FromPyError() {
    PyObject* exc_type = nullptr;
    PyObject* exc_value = nullptr;
    PyObject* exc_traceback = nullptr;

    PyErr_Fetch(&exc_type, &exc_value, &exc_traceback);
    PyErr_NormalizeException(&exc_type, &exc_value, &exc_traceback);
    ARROW_CHECK(exc_type)
        << "PythonErrorDetail::FromPyError called without a Python error set";
    DCHECK(PyType_Check(exc_type));
    DCHECK(exc_value);  // Ensured by PyErr_NormalizeException, double-check
    if (exc_traceback == nullptr) {
      // Needed by PyErr_Restore()
      Py_INCREF(Py_None);
      exc_traceback = Py_None;
    }

    std::shared_ptr<PythonErrorDetail> detail(new PythonErrorDetail);
    detail->exc_type_.reset(exc_type);
    detail->exc_value_.reset(exc_value);
    detail->exc_traceback_.reset(exc_traceback);
    return detail;
  }

 protected:
  PythonErrorDetail() = default;

  OwnedRefNoGIL exc_type_, exc_value_, exc_traceback_;
};

}  // namespace

// ----------------------------------------------------------------------
// Python exception <-> Status

Status ConvertPyError(StatusCode code) {
  auto detail = PythonErrorDetail::FromPyError();
  if (code == StatusCode::UnknownError) {
    code = MapPyError(detail->exc_type());
  }

  std::string message;
  RETURN_NOT_OK(internal::PyObject_StdStringStr(detail->exc_value(), &message));
  return Status(code, message, detail);
}

bool IsPyError(const Status& status) {
  if (status.ok()) {
    return false;
  }
  auto detail = status.detail();
  bool result = detail != nullptr && detail->type_id() == kErrorDetailTypeId;
  return result;
}

void RestorePyError(const Status& status) {
  ARROW_CHECK(IsPyError(status));
  const auto& detail = checked_cast<const PythonErrorDetail&>(*status.detail());
  detail.RestorePyError();
}

// ----------------------------------------------------------------------
// PyBuffer

PyBuffer::PyBuffer() : Buffer(nullptr, 0) {}

Status PyBuffer::Init(PyObject* obj) {
  if (!PyObject_GetBuffer(obj, &py_buf_, PyBUF_ANY_CONTIGUOUS)) {
    data_ = reinterpret_cast<const uint8_t*>(py_buf_.buf);
    ARROW_CHECK_NE(data_, nullptr) << "Null pointer in Py_buffer";
    size_ = py_buf_.len;
    capacity_ = py_buf_.len;
    is_mutable_ = !py_buf_.readonly;
    return Status::OK();
  } else {
    return ConvertPyError(StatusCode::Invalid);
  }
}

Result<std::shared_ptr<Buffer>> PyBuffer::FromPyObject(PyObject* obj) {
  PyBuffer* buf = new PyBuffer();
  std::shared_ptr<Buffer> res(buf);
  RETURN_NOT_OK(buf->Init(obj));
  return res;
}

PyBuffer::~PyBuffer() {
  if (data_ != nullptr) {
    PyAcquireGIL lock;
    PyBuffer_Release(&py_buf_);
  }
}

}  // namespace py
}  // namespace arrow

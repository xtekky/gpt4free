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
#include <functional>
#include <utility>

#include "arrow/buffer.h"
#include "arrow/result.h"
#include "arrow/util/macros.h"
#include "arrow/python/pyarrow.h"
#include "arrow/python/visibility.h"

namespace arrow {

class MemoryPool;
template <class T>
class Result;

namespace py {

// Convert current Python error to a Status.  The Python error state is cleared
// and can be restored with RestorePyError().
ARROW_PYTHON_EXPORT Status ConvertPyError(StatusCode code = StatusCode::UnknownError);
// Query whether the given Status is a Python error (as wrapped by ConvertPyError()).
ARROW_PYTHON_EXPORT bool IsPyError(const Status& status);
// Restore a Python error wrapped in a Status.
ARROW_PYTHON_EXPORT void RestorePyError(const Status& status);

// Catch a pending Python exception and return the corresponding Status.
// If no exception is pending, Status::OK() is returned.
inline Status CheckPyError(StatusCode code = StatusCode::UnknownError) {
  if (ARROW_PREDICT_TRUE(!PyErr_Occurred())) {
    return Status::OK();
  } else {
    return ConvertPyError(code);
  }
}

#define RETURN_IF_PYERROR() ARROW_RETURN_NOT_OK(CheckPyError())

#define PY_RETURN_IF_ERROR(CODE) ARROW_RETURN_NOT_OK(CheckPyError(CODE))

// For Cython, as you can't define template C++ functions in Cython, only use them.
// This function can set a Python exception.  It assumes that T has a (cheap)
// default constructor.
template <class T>
T GetResultValue(Result<T> result) {
  if (ARROW_PREDICT_TRUE(result.ok())) {
    return *std::move(result);
  } else {
    int r = internal::check_status(result.status());  // takes the GIL
    assert(r == -1);                                  // should have errored out
    ARROW_UNUSED(r);
    return {};
  }
}

// A RAII-style helper that ensures the GIL is acquired inside a lexical block.
class ARROW_PYTHON_EXPORT PyAcquireGIL {
 public:
  PyAcquireGIL() : acquired_gil_(false) { acquire(); }

  ~PyAcquireGIL() { release(); }

  void acquire() {
    if (!acquired_gil_) {
      state_ = PyGILState_Ensure();
      acquired_gil_ = true;
    }
  }

  // idempotent
  void release() {
    if (acquired_gil_) {
      PyGILState_Release(state_);
      acquired_gil_ = false;
    }
  }

 private:
  bool acquired_gil_;
  PyGILState_STATE state_;
  ARROW_DISALLOW_COPY_AND_ASSIGN(PyAcquireGIL);
};

// A RAII-style helper that releases the GIL until the end of a lexical block
class ARROW_PYTHON_EXPORT PyReleaseGIL {
 public:
  PyReleaseGIL() { saved_state_ = PyEval_SaveThread(); }

  ~PyReleaseGIL() { PyEval_RestoreThread(saved_state_); }

 private:
  PyThreadState* saved_state_;
  ARROW_DISALLOW_COPY_AND_ASSIGN(PyReleaseGIL);
};

// A helper to call safely into the Python interpreter from arbitrary C++ code.
// The GIL is acquired, and the current thread's error status is preserved.
template <typename Function>
auto SafeCallIntoPython(Function&& func) -> decltype(func()) {
  PyAcquireGIL lock;
  PyObject* exc_type;
  PyObject* exc_value;
  PyObject* exc_traceback;
  PyErr_Fetch(&exc_type, &exc_value, &exc_traceback);
  auto maybe_status = std::forward<Function>(func)();
  // If the return Status is a "Python error", the current Python error status
  // describes the error and shouldn't be clobbered.
  if (!IsPyError(::arrow::internal::GenericToStatus(maybe_status)) &&
      exc_type != NULLPTR) {
    PyErr_Restore(exc_type, exc_value, exc_traceback);
  }
  return maybe_status;
}

// A RAII primitive that DECREFs the underlying PyObject* when it
// goes out of scope.
class ARROW_PYTHON_EXPORT OwnedRef {
 public:
  OwnedRef() : obj_(NULLPTR) {}
  OwnedRef(OwnedRef&& other) : OwnedRef(other.detach()) {}
  explicit OwnedRef(PyObject* obj) : obj_(obj) {}

  OwnedRef& operator=(OwnedRef&& other) {
    obj_ = other.detach();
    return *this;
  }

  ~OwnedRef() { reset(); }

  void reset(PyObject* obj) {
    Py_XDECREF(obj_);
    obj_ = obj;
  }

  void reset() { reset(NULLPTR); }

  PyObject* detach() {
    PyObject* result = obj_;
    obj_ = NULLPTR;
    return result;
  }

  PyObject* obj() const { return obj_; }

  PyObject** ref() { return &obj_; }

  operator bool() const { return obj_ != NULLPTR; }

 private:
  ARROW_DISALLOW_COPY_AND_ASSIGN(OwnedRef);

  PyObject* obj_;
};

// Same as OwnedRef, but ensures the GIL is taken when it goes out of scope.
// This is for situations where the GIL is not always known to be held
// (e.g. if it is released in the middle of a function for performance reasons)
class ARROW_PYTHON_EXPORT OwnedRefNoGIL : public OwnedRef {
 public:
  OwnedRefNoGIL() : OwnedRef() {}
  OwnedRefNoGIL(OwnedRefNoGIL&& other) : OwnedRef(other.detach()) {}
  explicit OwnedRefNoGIL(PyObject* obj) : OwnedRef(obj) {}

  ~OwnedRefNoGIL() {
    // This destructor may be called after the Python interpreter is finalized.
    // At least avoid spurious attempts to take the GIL when not necessary.
    if (obj() == NULLPTR) {
      return;
    }
    PyAcquireGIL lock;
    reset();
  }
};

template <typename Fn>
struct BoundFunction;

template <typename... Args>
struct BoundFunction<void(PyObject*, Args...)> {
  // We bind `cdef void fn(object, ...)` to get a `Status(...)`
  // where the Status contains any Python error raised by `fn`
  using Unbound = void(PyObject*, Args...);
  using Bound = Status(Args...);

  BoundFunction(Unbound* unbound, PyObject* bound_arg)
      : bound_arg_(bound_arg), unbound_(unbound) {}

  Status Invoke(Args... args) const {
    PyAcquireGIL lock;
    unbound_(bound_arg_.obj(), std::forward<Args>(args)...);
    RETURN_IF_PYERROR();
    return Status::OK();
  }

  Unbound* unbound_;
  OwnedRefNoGIL bound_arg_;
};

template <typename Return, typename... Args>
struct BoundFunction<Return(PyObject*, Args...)> {
  // We bind `cdef Return fn(object, ...)` to get a `Result<Return>(...)`
  // where the Result contains any Python error raised by `fn` or the
  // return value from `fn`.
  using Unbound = Return(PyObject*, Args...);
  using Bound = Result<Return>(Args...);

  BoundFunction(Unbound* unbound, PyObject* bound_arg)
      : bound_arg_(bound_arg), unbound_(unbound) {}

  Result<Return> Invoke(Args... args) const {
    PyAcquireGIL lock;
    Return ret = unbound_(bound_arg_.obj(), std::forward<Args>(args)...);
    RETURN_IF_PYERROR();
    return ret;
  }

  Unbound* unbound_;
  OwnedRefNoGIL bound_arg_;
};

template <typename OutFn, typename Return, typename... Args>
std::function<OutFn> BindFunction(Return (*unbound)(PyObject*, Args...),
                                  PyObject* bound_arg) {
  using Fn = BoundFunction<Return(PyObject*, Args...)>;

  static_assert(std::is_same<typename Fn::Bound, OutFn>::value,
                "requested bound function of unsupported type");

  Py_XINCREF(bound_arg);
  auto bound_fn = std::make_shared<Fn>(unbound, bound_arg);
  return
      [bound_fn](Args... args) { return bound_fn->Invoke(std::forward<Args>(args)...); };
}

// A temporary conversion of a Python object to a bytes area.
struct PyBytesView {
  const char* bytes;
  Py_ssize_t size;
  bool is_utf8;

  static Result<PyBytesView> FromString(PyObject* obj, bool check_utf8 = false) {
    PyBytesView self;
    ARROW_RETURN_NOT_OK(self.ParseString(obj, check_utf8));
    return std::move(self);
  }

  static Result<PyBytesView> FromUnicode(PyObject* obj) {
    PyBytesView self;
    ARROW_RETURN_NOT_OK(self.ParseUnicode(obj));
    return std::move(self);
  }

  static Result<PyBytesView> FromBinary(PyObject* obj) {
    PyBytesView self;
    ARROW_RETURN_NOT_OK(self.ParseBinary(obj));
    return std::move(self);
  }

  // View the given Python object as string-like, i.e. str or (utf8) bytes
  Status ParseString(PyObject* obj, bool check_utf8 = false) {
    if (PyUnicode_Check(obj)) {
      return ParseUnicode(obj);
    } else {
      ARROW_RETURN_NOT_OK(ParseBinary(obj));
      if (check_utf8) {
        // Check the bytes are utf8 utf-8
        OwnedRef decoded(PyUnicode_FromStringAndSize(bytes, size));
        if (ARROW_PREDICT_TRUE(!PyErr_Occurred())) {
          is_utf8 = true;
        } else {
          PyErr_Clear();
          is_utf8 = false;
        }
      }
      return Status::OK();
    }
  }

  // View the given Python object as unicode string
  Status ParseUnicode(PyObject* obj) {
    // The utf-8 representation is cached on the unicode object
    bytes = PyUnicode_AsUTF8AndSize(obj, &size);
    RETURN_IF_PYERROR();
    is_utf8 = true;
    return Status::OK();
  }

  // View the given Python object as binary-like, i.e. bytes
  Status ParseBinary(PyObject* obj) {
    if (PyBytes_Check(obj)) {
      bytes = PyBytes_AS_STRING(obj);
      size = PyBytes_GET_SIZE(obj);
      is_utf8 = false;
    } else if (PyByteArray_Check(obj)) {
      bytes = PyByteArray_AS_STRING(obj);
      size = PyByteArray_GET_SIZE(obj);
      is_utf8 = false;
    } else if (PyMemoryView_Check(obj)) {
      PyObject* ref = PyMemoryView_GetContiguous(obj, PyBUF_READ, 'C');
      RETURN_IF_PYERROR();
      Py_buffer* buffer = PyMemoryView_GET_BUFFER(ref);
      bytes = reinterpret_cast<const char*>(buffer->buf);
      size = buffer->len;
      is_utf8 = false;
    } else {
      return Status::TypeError("Expected bytes, got a '", Py_TYPE(obj)->tp_name,
                               "' object");
    }
    return Status::OK();
  }

 protected:
  OwnedRef ref;
};

class ARROW_PYTHON_EXPORT PyBuffer : public Buffer {
 public:
  /// While memoryview objects support multi-dimensional buffers, PyBuffer only supports
  /// one-dimensional byte buffers.
  ~PyBuffer();

  static Result<std::shared_ptr<Buffer>> FromPyObject(PyObject* obj);

 private:
  PyBuffer();
  Status Init(PyObject*);

  Py_buffer py_buf_;
};

// Return the common PyArrow memory pool
ARROW_PYTHON_EXPORT void set_default_memory_pool(MemoryPool* pool);
ARROW_PYTHON_EXPORT MemoryPool* get_memory_pool();

// This is annoying: because C++11 does not allow implicit conversion of string
// literals to non-const char*, we need to go through some gymnastics to use
// PyObject_CallMethod without a lot of pain (its arguments are non-const
// char*)
template <typename... ArgTypes>
static inline PyObject* cpp_PyObject_CallMethod(PyObject* obj, const char* method_name,
                                                const char* argspec, ArgTypes... args) {
  return PyObject_CallMethod(obj, const_cast<char*>(method_name),
                             const_cast<char*>(argspec), args...);
}

}  // namespace py
}  // namespace arrow

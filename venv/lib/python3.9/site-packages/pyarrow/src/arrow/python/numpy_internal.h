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

// Internal utilities for dealing with NumPy

#pragma once

#include "arrow/python/numpy_interop.h"

#include "arrow/status.h"

#include "arrow/python/platform.h"

#include <cstdint>
#include <sstream>
#include <string>

namespace arrow {
namespace py {

/// Indexing convenience for interacting with strided 1-dim ndarray objects
template <typename T>
class Ndarray1DIndexer {
 public:
  typedef int64_t size_type;

  Ndarray1DIndexer() : arr_(NULLPTR), data_(NULLPTR) {}

  explicit Ndarray1DIndexer(PyArrayObject* arr) : Ndarray1DIndexer() {
    arr_ = arr;
    DCHECK_EQ(1, PyArray_NDIM(arr)) << "Only works with 1-dimensional arrays";
    data_ = reinterpret_cast<uint8_t*>(PyArray_DATA(arr));
    stride_ = PyArray_STRIDES(arr)[0];
  }

  ~Ndarray1DIndexer() = default;

  int64_t size() const { return PyArray_SIZE(arr_); }

  const T* data() const { return reinterpret_cast<const T*>(data_); }

  bool is_strided() const { return stride_ != sizeof(T); }

  T& operator[](size_type index) {
    return *reinterpret_cast<T*>(data_ + index * stride_);
  }
  const T& operator[](size_type index) const {
    return *reinterpret_cast<const T*>(data_ + index * stride_);
  }

 private:
  PyArrayObject* arr_;
  uint8_t* data_;
  int64_t stride_;
};

// Handling of Numpy Types by their static numbers
// (the NPY_TYPES enum and related defines)

static inline std::string GetNumPyTypeName(int npy_type) {
#define TYPE_CASE(TYPE, NAME) \
  case NPY_##TYPE:            \
    return NAME;

  switch (npy_type) {
    TYPE_CASE(BOOL, "bool")
    TYPE_CASE(INT8, "int8")
    TYPE_CASE(INT16, "int16")
    TYPE_CASE(INT32, "int32")
    TYPE_CASE(INT64, "int64")
#if !NPY_INT32_IS_INT
    TYPE_CASE(INT, "intc")
#endif
#if !NPY_INT64_IS_LONG_LONG
    TYPE_CASE(LONGLONG, "longlong")
#endif
    TYPE_CASE(UINT8, "uint8")
    TYPE_CASE(UINT16, "uint16")
    TYPE_CASE(UINT32, "uint32")
    TYPE_CASE(UINT64, "uint64")
#if !NPY_INT32_IS_INT
    TYPE_CASE(UINT, "uintc")
#endif
#if !NPY_INT64_IS_LONG_LONG
    TYPE_CASE(ULONGLONG, "ulonglong")
#endif
    TYPE_CASE(FLOAT16, "float16")
    TYPE_CASE(FLOAT32, "float32")
    TYPE_CASE(FLOAT64, "float64")
    TYPE_CASE(DATETIME, "datetime64")
    TYPE_CASE(TIMEDELTA, "timedelta64")
    TYPE_CASE(OBJECT, "object")
    TYPE_CASE(VOID, "void")
    default:
      break;
  }

#undef TYPE_CASE
  std::stringstream ss;
  ss << "unrecognized type (" << npy_type << ") in GetNumPyTypeName";
  return ss.str();
}

#define TYPE_VISIT_INLINE(TYPE) \
  case NPY_##TYPE:              \
    return visitor->template Visit<NPY_##TYPE>(arr);

template <typename VISITOR>
inline Status VisitNumpyArrayInline(PyArrayObject* arr, VISITOR* visitor) {
  switch (PyArray_TYPE(arr)) {
    TYPE_VISIT_INLINE(BOOL);
    TYPE_VISIT_INLINE(INT8);
    TYPE_VISIT_INLINE(UINT8);
    TYPE_VISIT_INLINE(INT16);
    TYPE_VISIT_INLINE(UINT16);
    TYPE_VISIT_INLINE(INT32);
    TYPE_VISIT_INLINE(UINT32);
    TYPE_VISIT_INLINE(INT64);
    TYPE_VISIT_INLINE(UINT64);
#if !NPY_INT32_IS_INT
    TYPE_VISIT_INLINE(INT);
    TYPE_VISIT_INLINE(UINT);
#endif
#if !NPY_INT64_IS_LONG_LONG
    TYPE_VISIT_INLINE(LONGLONG);
    TYPE_VISIT_INLINE(ULONGLONG);
#endif
    TYPE_VISIT_INLINE(FLOAT16);
    TYPE_VISIT_INLINE(FLOAT32);
    TYPE_VISIT_INLINE(FLOAT64);
    TYPE_VISIT_INLINE(DATETIME);
    TYPE_VISIT_INLINE(TIMEDELTA);
    TYPE_VISIT_INLINE(OBJECT);
  }
  return Status::NotImplemented("NumPy type not implemented: ",
                                GetNumPyTypeName(PyArray_TYPE(arr)));
}

#undef TYPE_VISIT_INLINE

namespace internal {

inline bool PyFloatScalar_Check(PyObject* obj) {
  return PyFloat_Check(obj) || PyArray_IsScalar(obj, Floating);
}

inline bool PyIntScalar_Check(PyObject* obj) {
  return PyLong_Check(obj) || PyArray_IsScalar(obj, Integer);
}

inline bool PyBoolScalar_Check(PyObject* obj) {
  return PyBool_Check(obj) || PyArray_IsScalar(obj, Bool);
}

static inline PyArray_Descr* GetSafeNumPyDtype(int type) {
  if (type == NPY_DATETIME || type == NPY_TIMEDELTA) {
    // It is not safe to mutate the result of DescrFromType for datetime and
    // timedelta descriptors
    return PyArray_DescrNewFromType(type);
  } else {
    return PyArray_DescrFromType(type);
  }
}

}  // namespace internal

}  // namespace py
}  // namespace arrow

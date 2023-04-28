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

#include <utility>

#include "arrow/array/array_primitive.h"

#include "arrow/python/common.h"
#include "arrow/python/numpy_internal.h"

namespace arrow {
namespace py {
namespace internal {

using arrow::internal::checked_cast;

// Visit the Python sequence, calling the given callable on each element.  If
// the callable returns a non-OK status, iteration stops and the status is
// returned.
//
// The call signature for Visitor must be
//
// Visit(PyObject* obj, int64_t index, bool* keep_going)
//
// If keep_going is set to false, the iteration terminates
template <class VisitorFunc>
inline Status VisitSequenceGeneric(PyObject* obj, int64_t offset, VisitorFunc&& func) {
  // VisitorFunc may set to false to terminate iteration
  bool keep_going = true;

  if (PyArray_Check(obj)) {
    PyArrayObject* arr_obj = reinterpret_cast<PyArrayObject*>(obj);
    if (PyArray_NDIM(arr_obj) != 1) {
      return Status::Invalid("Only 1D arrays accepted");
    }

    if (PyArray_DESCR(arr_obj)->type_num == NPY_OBJECT) {
      // It's an array object, we can fetch object pointers directly
      const Ndarray1DIndexer<PyObject*> objects(arr_obj);
      for (int64_t i = offset; keep_going && i < objects.size(); ++i) {
        RETURN_NOT_OK(func(objects[i], i, &keep_going));
      }
      return Status::OK();
    }
    // It's a non-object array, fall back on regular sequence access.
    // (note PyArray_GETITEM() is slightly different: it returns standard
    //  Python types, not Numpy scalar types)
    // This code path is inefficient: callers should implement dedicated
    // logic for non-object arrays.
  }
  if (PySequence_Check(obj)) {
    if (PyList_Check(obj) || PyTuple_Check(obj)) {
      // Use fast item access
      const Py_ssize_t size = PySequence_Fast_GET_SIZE(obj);
      for (Py_ssize_t i = offset; keep_going && i < size; ++i) {
        PyObject* value = PySequence_Fast_GET_ITEM(obj, i);
        RETURN_NOT_OK(func(value, static_cast<int64_t>(i), &keep_going));
      }
    } else {
      // Regular sequence: avoid making a potentially large copy
      const Py_ssize_t size = PySequence_Size(obj);
      RETURN_IF_PYERROR();
      for (Py_ssize_t i = offset; keep_going && i < size; ++i) {
        OwnedRef value_ref(PySequence_ITEM(obj, i));
        RETURN_IF_PYERROR();
        RETURN_NOT_OK(func(value_ref.obj(), static_cast<int64_t>(i), &keep_going));
      }
    }
  } else {
    return Status::TypeError("Object is not a sequence");
  }
  return Status::OK();
}

// Visit sequence with no null mask
template <class VisitorFunc>
inline Status VisitSequence(PyObject* obj, int64_t offset, VisitorFunc&& func) {
  return VisitSequenceGeneric(
      obj, offset, [&func](PyObject* value, int64_t i /* unused */, bool* keep_going) {
        return func(value, keep_going);
      });
}

/// Visit sequence with null mask
template <class VisitorFunc>
inline Status VisitSequenceMasked(PyObject* obj, PyObject* mo, int64_t offset,
                                  VisitorFunc&& func) {
  if (PyArray_Check(mo)) {
    PyArrayObject* mask = reinterpret_cast<PyArrayObject*>(mo);
    if (PyArray_NDIM(mask) != 1) {
      return Status::Invalid("Mask must be 1D array");
    }
    if (PyArray_SIZE(mask) != static_cast<int64_t>(PySequence_Size(obj))) {
      return Status::Invalid("Mask was a different length from sequence being converted");
    }

    const int dtype = fix_numpy_type_num(PyArray_DESCR(mask)->type_num);
    if (dtype == NPY_BOOL) {
      Ndarray1DIndexer<uint8_t> mask_values(mask);

      return VisitSequenceGeneric(
          obj, offset,
          [&func, &mask_values](PyObject* value, int64_t i, bool* keep_going) {
            return func(value, mask_values[i], keep_going);
          });
    } else {
      return Status::TypeError("Mask must be boolean dtype");
    }
  } else if (py::is_array(mo)) {
    auto unwrap_mask_result = unwrap_array(mo);
    ARROW_RETURN_NOT_OK(unwrap_mask_result);
    std::shared_ptr<Array> mask_ = unwrap_mask_result.ValueOrDie();
    if (mask_->type_id() != Type::type::BOOL) {
      return Status::TypeError("Mask must be an array of booleans");
    }

    if (mask_->length() != PySequence_Size(obj)) {
      return Status::Invalid("Mask was a different length from sequence being converted");
    }

    if (mask_->null_count() != 0) {
      return Status::TypeError("Mask must be an array of booleans");
    }

    BooleanArray* boolmask = checked_cast<BooleanArray*>(mask_.get());
    return VisitSequenceGeneric(
        obj, offset, [&func, &boolmask](PyObject* value, int64_t i, bool* keep_going) {
          return func(value, boolmask->Value(i), keep_going);
        });
  } else if (PySequence_Check(mo)) {
    if (PySequence_Size(mo) != PySequence_Size(obj)) {
      return Status::Invalid("Mask was a different length from sequence being converted");
    }
    RETURN_IF_PYERROR();

    return VisitSequenceGeneric(
        obj, offset, [&func, &mo](PyObject* value, int64_t i, bool* keep_going) {
          OwnedRef value_ref(PySequence_ITEM(mo, i));
          if (!PyBool_Check(value_ref.obj()))
            return Status::TypeError("Mask must be a sequence of booleans");
          return func(value, value_ref.obj() == Py_True, keep_going);
        });
  } else {
    return Status::Invalid("Null mask must be a NumPy array, Arrow array or a Sequence");
  }

  return Status::OK();
}

// Like IterateSequence, but accepts any generic iterable (including
// non-restartable iterators, e.g. generators).
//
// The call signature for VisitorFunc must be Visit(PyObject*, bool*
// keep_going). If keep_going is set to false, the iteration terminates
template <class VisitorFunc>
inline Status VisitIterable(PyObject* obj, VisitorFunc&& func) {
  if (PySequence_Check(obj)) {
    // Numpy arrays fall here as well
    return VisitSequence(obj, /*offset=*/0, std::forward<VisitorFunc>(func));
  }
  // Fall back on the iterator protocol
  OwnedRef iter_ref(PyObject_GetIter(obj));
  PyObject* iter = iter_ref.obj();
  RETURN_IF_PYERROR();
  PyObject* value;

  bool keep_going = true;
  while (keep_going && (value = PyIter_Next(iter))) {
    OwnedRef value_ref(value);
    RETURN_NOT_OK(func(value_ref.obj(), &keep_going));
  }
  RETURN_IF_PYERROR();  // __next__() might have raised
  return Status::OK();
}

}  // namespace internal
}  // namespace py
}  // namespace arrow

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# distutils: language = c++

from libc.stdint cimport *
from libcpp cimport bool as c_bool, nullptr
from libcpp.functional cimport function
from libcpp.memory cimport shared_ptr, unique_ptr, make_shared
from libcpp.string cimport string as c_string
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set

from cpython cimport PyObject
from cpython.datetime cimport PyDateTime_DateTime
cimport cpython


cdef extern from * namespace "std" nogil:
    cdef shared_ptr[T] static_pointer_cast[T, U](shared_ptr[U])


cdef extern from "<optional>" namespace "std" nogil:
    cdef cppclass optional[T]:
        c_bool has_value()
        T value()
        optional(T&)
        optional& operator=[U](U&)


# vendored from the cymove project https://github.com/ozars/cymove
cdef extern from * namespace "cymove" nogil:
    """
    #include <type_traits>
    #include <utility>
    namespace cymove {
    template <typename T>
    inline typename std::remove_reference<T>::type&& cymove(T& t) {
        return std::move(t);
    }
    template <typename T>
    inline typename std::remove_reference<T>::type&& cymove(T&& t) {
        return std::move(t);
    }
    }  // namespace cymove
    """
    cdef T move" cymove::cymove"[T](T)

cdef extern from * namespace "arrow::py" nogil:
    """
    #include <memory>
    #include <utility>

    namespace arrow {
    namespace py {
    template <typename T>
    std::shared_ptr<T> to_shared(std::unique_ptr<T>& t) {
        return std::move(t);
    }
    template <typename T>
    std::shared_ptr<T> to_shared(std::unique_ptr<T>&& t) {
        return std::move(t);
    }
    }  // namespace py
    }  // namespace arrow
    """
    cdef shared_ptr[T] to_shared" arrow::py::to_shared"[T](unique_ptr[T])

cdef extern from "arrow/python/platform.h":
    pass

cdef extern from "<Python.h>":
    void Py_XDECREF(PyObject* o)
    Py_ssize_t Py_REFCNT(PyObject* o)

cdef extern from "numpy/halffloat.h":
    ctypedef uint16_t npy_half

cdef extern from "arrow/api.h" namespace "arrow" nogil:
    # We can later add more of the common status factory methods as needed
    cdef CStatus CStatus_OK "arrow::Status::OK"()

    cdef CStatus CStatus_Invalid "arrow::Status::Invalid"()
    cdef CStatus CStatus_NotImplemented \
        "arrow::Status::NotImplemented"(const c_string& msg)
    cdef CStatus CStatus_UnknownError \
        "arrow::Status::UnknownError"(const c_string& msg)

    cdef cppclass CStatus "arrow::Status":
        CStatus()

        c_string ToString()
        c_string message()
        shared_ptr[CStatusDetail] detail()

        c_bool ok()
        c_bool IsIOError()
        c_bool IsOutOfMemory()
        c_bool IsInvalid()
        c_bool IsKeyError()
        c_bool IsNotImplemented()
        c_bool IsTypeError()
        c_bool IsCapacityError()
        c_bool IsIndexError()
        c_bool IsSerializationError()
        c_bool IsCancelled()

        void Warn()

    cdef cppclass CStatusDetail "arrow::StatusDetail":
        c_string ToString()


cdef extern from "arrow/result.h" namespace "arrow" nogil:
    cdef cppclass CResult "arrow::Result"[T]:
        CResult()
        CResult(CStatus)
        CResult(T)
        c_bool ok()
        CStatus status()
        CStatus Value(T*)
        T operator*()


cdef extern from "arrow/python/common.h" namespace "arrow::py" nogil:
    T GetResultValue[T](CResult[T]) except *
    cdef function[F] BindFunction[F](void* unbound, object bound, ...)


cdef inline object PyObject_to_object(PyObject* o):
    # Cast to "object" increments reference count
    cdef object result = <object> o
    cpython.Py_DECREF(result)
    return result

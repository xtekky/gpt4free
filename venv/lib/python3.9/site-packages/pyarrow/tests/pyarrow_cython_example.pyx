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

# distutils: language=c++
# cython: language_level = 3

from pyarrow.lib cimport *


def get_array_length(obj):
    # An example function accessing both the pyarrow Cython API
    # and the Arrow C++ API
    cdef shared_ptr[CArray] arr = pyarrow_unwrap_array(obj)
    if arr.get() == NULL:
        raise TypeError("not an array")
    return arr.get().length()


def make_null_array(length):
    # An example function that returns a PyArrow object without PyArrow
    # being imported explicitly at the Python level.
    cdef shared_ptr[CArray] null_array
    null_array.reset(new CNullArray(length))
    return pyarrow_wrap_array(null_array)


def cast_scalar(scalar, to_type):
    cdef:
        shared_ptr[CScalar] c_scalar
        shared_ptr[CDataType] c_type
        CResult[shared_ptr[CScalar]] c_result

    c_scalar = pyarrow_unwrap_scalar(scalar)
    if c_scalar.get() == NULL:
        raise TypeError("not a scalar")
    c_type = pyarrow_unwrap_data_type(to_type)
    if c_type.get() == NULL:
        raise TypeError("not a type")
    c_result = c_scalar.get().CastTo(c_type)
    c_scalar = GetResultValue(c_result)
    return pyarrow_wrap_scalar(c_scalar)

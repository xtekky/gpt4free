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

# cython: profile = False
# cython: nonecheck = True
# distutils: language = c++

import datetime
import decimal as _pydecimal
import numpy as np
import os
import sys

from cython.operator cimport dereference as deref
from pyarrow.includes.libarrow cimport *
from pyarrow.includes.libarrow_python cimport *
from pyarrow.includes.common cimport PyObject_to_object
cimport pyarrow.includes.libarrow as libarrow
cimport pyarrow.includes.libarrow_python as libarrow_python
cimport cpython as cp

# Initialize NumPy C API
arrow_init_numpy()
# Initialize PyArrow C++ API
# (used from some of our C++ code, see e.g. ARROW-5260)
import_pyarrow()


MonthDayNano = NewMonthDayNanoTupleType()


def cpu_count():
    """
    Return the number of threads to use in parallel operations.

    The number of threads is determined at startup by inspecting the
    ``OMP_NUM_THREADS`` and ``OMP_THREAD_LIMIT`` environment variables.
    If neither is present, it will default to the number of hardware threads
    on the system. It can be modified at runtime by calling
    :func:`set_cpu_count()`.

    See Also
    --------
    set_cpu_count : Modify the size of this pool.
    io_thread_count : The analogous function for the I/O thread pool.
    """
    return GetCpuThreadPoolCapacity()


def set_cpu_count(int count):
    """
    Set the number of threads to use in parallel operations.

    Parameters
    ----------
    count : int
        The number of concurrent threads that should be used.

    See Also
    --------
    cpu_count : Get the size of this pool.
    set_io_thread_count : The analogous function for the I/O thread pool.
    """
    if count < 1:
        raise ValueError("CPU count must be strictly positive")
    check_status(SetCpuThreadPoolCapacity(count))


Type_NA = _Type_NA
Type_BOOL = _Type_BOOL
Type_UINT8 = _Type_UINT8
Type_INT8 = _Type_INT8
Type_UINT16 = _Type_UINT16
Type_INT16 = _Type_INT16
Type_UINT32 = _Type_UINT32
Type_INT32 = _Type_INT32
Type_UINT64 = _Type_UINT64
Type_INT64 = _Type_INT64
Type_HALF_FLOAT = _Type_HALF_FLOAT
Type_FLOAT = _Type_FLOAT
Type_DOUBLE = _Type_DOUBLE
Type_DECIMAL128 = _Type_DECIMAL128
Type_DECIMAL256 = _Type_DECIMAL256
Type_DATE32 = _Type_DATE32
Type_DATE64 = _Type_DATE64
Type_TIMESTAMP = _Type_TIMESTAMP
Type_TIME32 = _Type_TIME32
Type_TIME64 = _Type_TIME64
Type_DURATION = _Type_DURATION
Type_INTERVAL_MONTH_DAY_NANO = _Type_INTERVAL_MONTH_DAY_NANO
Type_BINARY = _Type_BINARY
Type_STRING = _Type_STRING
Type_LARGE_BINARY = _Type_LARGE_BINARY
Type_LARGE_STRING = _Type_LARGE_STRING
Type_FIXED_SIZE_BINARY = _Type_FIXED_SIZE_BINARY
Type_LIST = _Type_LIST
Type_LARGE_LIST = _Type_LARGE_LIST
Type_MAP = _Type_MAP
Type_FIXED_SIZE_LIST = _Type_FIXED_SIZE_LIST
Type_STRUCT = _Type_STRUCT
Type_SPARSE_UNION = _Type_SPARSE_UNION
Type_DENSE_UNION = _Type_DENSE_UNION
Type_DICTIONARY = _Type_DICTIONARY

UnionMode_SPARSE = _UnionMode_SPARSE
UnionMode_DENSE = _UnionMode_DENSE

__pc = None


def _pc():
    global __pc
    if __pc is None:
        import pyarrow.compute as pc
        try:
            from pyarrow import _exec_plan
            pc._exec_plan = _exec_plan
        except ImportError:
            pass
        __pc = pc
    return __pc


def _gdb_test_session():
    GdbTestSession()


# Assorted compatibility helpers
include "compat.pxi"

# Exception types and Status handling
include "error.pxi"

# Configuration information
include "config.pxi"

# pandas API shim
include "pandas-shim.pxi"

# Memory pools and allocation
include "memory.pxi"

# DataType, Field, Schema
include "types.pxi"

# Array scalar values
include "scalar.pxi"

# Array types
include "array.pxi"

# Builders
include "builder.pxi"

# Column, Table, Record Batch
include "table.pxi"

# Tensors
include "tensor.pxi"

# File IO
include "io.pxi"

# IPC / Messaging
include "ipc.pxi"

# Python serialization
include "serialization.pxi"

# Micro-benchmark routines
include "benchmark.pxi"

# Public API
include "public-api.pxi"

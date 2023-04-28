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

# cython: language_level = 3

from pyarrow.includes.libarrow cimport *
from pyarrow.lib cimport _Weakrefable


cdef class ConvertOptions(_Weakrefable):
    cdef:
        unique_ptr[CCSVConvertOptions] options

    @staticmethod
    cdef ConvertOptions wrap(CCSVConvertOptions options)


cdef class ParseOptions(_Weakrefable):
    cdef:
        unique_ptr[CCSVParseOptions] options
        object _invalid_row_handler

    @staticmethod
    cdef ParseOptions wrap(CCSVParseOptions options)


cdef class ReadOptions(_Weakrefable):
    cdef:
        unique_ptr[CCSVReadOptions] options
        public object encoding

    @staticmethod
    cdef ReadOptions wrap(CCSVReadOptions options)


cdef class WriteOptions(_Weakrefable):
    cdef:
        unique_ptr[CCSVWriteOptions] options

    @staticmethod
    cdef WriteOptions wrap(CCSVWriteOptions options)

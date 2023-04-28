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

from pyarrow.lib cimport *
from pyarrow.includes.common cimport *
from pyarrow.includes.libarrow cimport *

cdef class ScalarUdfContext(_Weakrefable):
    cdef:
        CScalarUdfContext c_context

    cdef void init(self, const CScalarUdfContext& c_context)


cdef class FunctionOptions(_Weakrefable):
    cdef:
        shared_ptr[CFunctionOptions] wrapped

    cdef const CFunctionOptions* get_options(self) except NULL
    cdef void init(self, const shared_ptr[CFunctionOptions]& sp)

    cdef inline shared_ptr[CFunctionOptions] unwrap(self)


cdef class _SortOptions(FunctionOptions):
    pass


cdef CExpression _bind(Expression filter, Schema schema) except *


cdef class Expression(_Weakrefable):

    cdef:
        CExpression expr

    cdef void init(self, const CExpression& sp)

    @staticmethod
    cdef wrap(const CExpression& sp)

    cdef inline CExpression unwrap(self)

    @staticmethod
    cdef Expression _expr_or_scalar(object expr)


cdef CExpression _true

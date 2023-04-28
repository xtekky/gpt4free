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

from libcpp.memory cimport shared_ptr
from pyarrow.includes.libarrow cimport (CArray, CBuffer, CDataType,
                                        CField, CRecordBatch, CSchema,
                                        CTable, CTensor, CSparseCOOTensor,
                                        CSparseCSRMatrix, CSparseCSCMatrix,
                                        CSparseCSFTensor)

cdef extern from "arrow/python/pyarrow.h" namespace "arrow::py":
    cdef int import_pyarrow() except -1
    cdef object wrap_buffer(const shared_ptr[CBuffer]& buffer)
    cdef object wrap_data_type(const shared_ptr[CDataType]& type)
    cdef object wrap_field(const shared_ptr[CField]& field)
    cdef object wrap_schema(const shared_ptr[CSchema]& schema)
    cdef object wrap_array(const shared_ptr[CArray]& sp_array)
    cdef object wrap_tensor(const shared_ptr[CTensor]& sp_tensor)
    cdef object wrap_sparse_tensor_coo(
        const shared_ptr[CSparseCOOTensor]& sp_sparse_tensor)
    cdef object wrap_sparse_tensor_csr(
        const shared_ptr[CSparseCSRMatrix]& sp_sparse_tensor)
    cdef object wrap_sparse_tensor_csc(
        const shared_ptr[CSparseCSCMatrix]& sp_sparse_tensor)
    cdef object wrap_sparse_tensor_csf(
        const shared_ptr[CSparseCSFTensor]& sp_sparse_tensor)
    cdef object wrap_table(const shared_ptr[CTable]& ctable)
    cdef object wrap_batch(const shared_ptr[CRecordBatch]& cbatch)

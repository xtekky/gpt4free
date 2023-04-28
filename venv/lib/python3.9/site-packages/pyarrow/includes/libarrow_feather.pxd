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

from pyarrow.includes.libarrow cimport (CCompressionType, CStatus, CTable,
                                        COutputStream, CResult, shared_ptr,
                                        vector, CRandomAccessFile, CSchema,
                                        c_string, CIpcReadOptions)


cdef extern from "arrow/ipc/api.h" namespace "arrow::ipc" nogil:
    int kFeatherV1Version" arrow::ipc::feather::kFeatherV1Version"
    int kFeatherV2Version" arrow::ipc::feather::kFeatherV2Version"

    cdef cppclass CFeatherProperties" arrow::ipc::feather::WriteProperties":
        int version
        int chunksize
        CCompressionType compression
        int compression_level

    CStatus WriteFeather" arrow::ipc::feather::WriteTable" \
        (const CTable& table, COutputStream* out,
         CFeatherProperties properties)

    cdef cppclass CFeatherReader" arrow::ipc::feather::Reader":
        @staticmethod
        CResult[shared_ptr[CFeatherReader]] Open(
            const shared_ptr[CRandomAccessFile]& file,
            const CIpcReadOptions& options)
        int version()
        shared_ptr[CSchema] schema()

        CStatus Read(shared_ptr[CTable]* out)
        CStatus Read(const vector[int] indices, shared_ptr[CTable]* out)
        CStatus Read(const vector[c_string] names, shared_ptr[CTable]* out)

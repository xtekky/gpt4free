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

from pyarrow.includes.libarrow cimport *

cdef extern from "arrow/gpu/cuda_api.h" namespace "arrow::cuda" nogil:

    cdef cppclass CCudaDeviceManager" arrow::cuda::CudaDeviceManager":
        @staticmethod
        CResult[CCudaDeviceManager*] Instance()
        CResult[shared_ptr[CCudaContext]] GetContext(int gpu_number)
        CResult[shared_ptr[CCudaContext]] GetSharedContext(int gpu_number,
                                                           void* handle)
        CStatus AllocateHost(int device_number, int64_t nbytes,
                             shared_ptr[CCudaHostBuffer]* buffer)
        int num_devices() const

    cdef cppclass CCudaContext" arrow::cuda::CudaContext":
        CResult[shared_ptr[CCudaBuffer]] Allocate(int64_t nbytes)
        CResult[shared_ptr[CCudaBuffer]] View(uint8_t* data, int64_t nbytes)
        CResult[shared_ptr[CCudaBuffer]] OpenIpcBuffer(
            const CCudaIpcMemHandle& ipc_handle)
        CStatus Synchronize()
        int64_t bytes_allocated() const
        const void* handle() const
        int device_number() const
        CResult[uintptr_t] GetDeviceAddress(uintptr_t addr)

    cdef cppclass CCudaIpcMemHandle" arrow::cuda::CudaIpcMemHandle":
        @staticmethod
        CResult[shared_ptr[CCudaIpcMemHandle]] FromBuffer(
            const void* opaque_handle)
        CResult[shared_ptr[CBuffer]] Serialize(CMemoryPool* pool) const

    cdef cppclass CCudaBuffer" arrow::cuda::CudaBuffer"(CBuffer):
        CCudaBuffer(uint8_t* data, int64_t size,
                    const shared_ptr[CCudaContext]& context,
                    c_bool own_data=false, c_bool is_ipc=false)
        CCudaBuffer(const shared_ptr[CCudaBuffer]& parent,
                    const int64_t offset, const int64_t size)

        @staticmethod
        CResult[shared_ptr[CCudaBuffer]] FromBuffer(shared_ptr[CBuffer] buf)

        CStatus CopyToHost(const int64_t position, const int64_t nbytes,
                           void* out) const
        CStatus CopyFromHost(const int64_t position, const void* data,
                             int64_t nbytes)
        CStatus CopyFromDevice(const int64_t position, const void* data,
                               int64_t nbytes)
        CStatus CopyFromAnotherDevice(const shared_ptr[CCudaContext]& src_ctx,
                                      const int64_t position, const void* data,
                                      int64_t nbytes)
        CResult[shared_ptr[CCudaIpcMemHandle]] ExportForIpc()
        shared_ptr[CCudaContext] context() const

    cdef cppclass \
            CCudaHostBuffer" arrow::cuda::CudaHostBuffer"(CMutableBuffer):
        pass

    cdef cppclass \
            CCudaBufferReader" arrow::cuda::CudaBufferReader"(CBufferReader):
        CCudaBufferReader(const shared_ptr[CBuffer]& buffer)
        CResult[int64_t] Read(int64_t nbytes, void* buffer)
        CResult[shared_ptr[CBuffer]] Read(int64_t nbytes)

    cdef cppclass \
            CCudaBufferWriter" arrow::cuda::CudaBufferWriter"(WritableFile):
        CCudaBufferWriter(const shared_ptr[CCudaBuffer]& buffer)
        CStatus Close()
        CStatus Write(const void* data, int64_t nbytes)
        CStatus WriteAt(int64_t position, const void* data, int64_t nbytes)
        CStatus SetBufferSize(const int64_t buffer_size)
        int64_t buffer_size()
        int64_t num_bytes_buffered() const

    CResult[shared_ptr[CCudaHostBuffer]] AllocateCudaHostBuffer(
        int device_number, const int64_t size)

    # Cuda prefix is added to avoid picking up arrow::cuda functions
    # from arrow namespace.
    CResult[shared_ptr[CCudaBuffer]] \
        CudaSerializeRecordBatch" arrow::cuda::SerializeRecordBatch"\
        (const CRecordBatch& batch,
         CCudaContext* ctx)
    CResult[shared_ptr[CRecordBatch]] \
        CudaReadRecordBatch" arrow::cuda::ReadRecordBatch"\
        (const shared_ptr[CSchema]& schema,
         CDictionaryMemo* dictionary_memo,
         const shared_ptr[CCudaBuffer]& buffer,
         CMemoryPool* pool)

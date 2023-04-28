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

from pyarrow.includes.common cimport *
from pyarrow.includes.libarrow cimport *


ctypedef CInvalidRowResult PyInvalidRowCallback(object,
                                                const CCSVInvalidRow&)


cdef extern from "arrow/python/csv.h" namespace "arrow::py::csv":

    function[CInvalidRowHandler] MakeInvalidRowHandler(
        function[PyInvalidRowCallback], object handler)


cdef extern from "arrow/python/api.h" namespace "arrow::py":
    # Requires GIL
    CResult[shared_ptr[CDataType]] InferArrowType(
        object obj, object mask, c_bool pandas_null_sentinels)


cdef extern from "arrow/python/api.h" namespace "arrow::py::internal":
    object NewMonthDayNanoTupleType()
    CResult[PyObject*] MonthDayNanoIntervalArrayToPyList(
        const CMonthDayNanoIntervalArray& array)
    CResult[PyObject*] MonthDayNanoIntervalScalarToPyObject(
        const CMonthDayNanoIntervalScalar& scalar)


cdef extern from "arrow/python/api.h" namespace "arrow::py" nogil:
    shared_ptr[CDataType] GetPrimitiveType(Type type)

    object PyHalf_FromHalf(npy_half value)

    cdef cppclass PyConversionOptions:
        PyConversionOptions()

        shared_ptr[CDataType] type
        int64_t size
        CMemoryPool* pool
        c_bool from_pandas
        c_bool ignore_timezone
        c_bool strict

    # TODO Some functions below are not actually "nogil"

    CResult[shared_ptr[CChunkedArray]] ConvertPySequence(
        object obj, object mask, const PyConversionOptions& options,
        CMemoryPool* pool)

    CStatus NumPyDtypeToArrow(object dtype, shared_ptr[CDataType]* type)

    CStatus NdarrayToArrow(CMemoryPool* pool, object ao, object mo,
                           c_bool from_pandas,
                           const shared_ptr[CDataType]& type,
                           shared_ptr[CChunkedArray]* out)

    CStatus NdarrayToArrow(CMemoryPool* pool, object ao, object mo,
                           c_bool from_pandas,
                           const shared_ptr[CDataType]& type,
                           const CCastOptions& cast_options,
                           shared_ptr[CChunkedArray]* out)

    CStatus NdarrayToTensor(CMemoryPool* pool, object ao,
                            const vector[c_string]& dim_names,
                            shared_ptr[CTensor]* out)

    CStatus TensorToNdarray(const shared_ptr[CTensor]& tensor, object base,
                            PyObject** out)

    CStatus SparseCOOTensorToNdarray(
        const shared_ptr[CSparseCOOTensor]& sparse_tensor, object base,
        PyObject** out_data, PyObject** out_coords)

    CStatus SparseCSRMatrixToNdarray(
        const shared_ptr[CSparseCSRMatrix]& sparse_tensor, object base,
        PyObject** out_data, PyObject** out_indptr, PyObject** out_indices)

    CStatus SparseCSCMatrixToNdarray(
        const shared_ptr[CSparseCSCMatrix]& sparse_tensor, object base,
        PyObject** out_data, PyObject** out_indptr, PyObject** out_indices)

    CStatus SparseCSFTensorToNdarray(
        const shared_ptr[CSparseCSFTensor]& sparse_tensor, object base,
        PyObject** out_data, PyObject** out_indptr, PyObject** out_indices)

    CStatus NdarraysToSparseCOOTensor(CMemoryPool* pool, object data_ao,
                                      object coords_ao,
                                      const vector[int64_t]& shape,
                                      const vector[c_string]& dim_names,
                                      shared_ptr[CSparseCOOTensor]* out)

    CStatus NdarraysToSparseCSRMatrix(CMemoryPool* pool, object data_ao,
                                      object indptr_ao, object indices_ao,
                                      const vector[int64_t]& shape,
                                      const vector[c_string]& dim_names,
                                      shared_ptr[CSparseCSRMatrix]* out)

    CStatus NdarraysToSparseCSCMatrix(CMemoryPool* pool, object data_ao,
                                      object indptr_ao, object indices_ao,
                                      const vector[int64_t]& shape,
                                      const vector[c_string]& dim_names,
                                      shared_ptr[CSparseCSCMatrix]* out)

    CStatus NdarraysToSparseCSFTensor(CMemoryPool* pool, object data_ao,
                                      object indptr_ao, object indices_ao,
                                      const vector[int64_t]& shape,
                                      const vector[int64_t]& axis_order,
                                      const vector[c_string]& dim_names,
                                      shared_ptr[CSparseCSFTensor]* out)

    CStatus TensorToSparseCOOTensor(shared_ptr[CTensor],
                                    shared_ptr[CSparseCOOTensor]* out)

    CStatus TensorToSparseCSRMatrix(shared_ptr[CTensor],
                                    shared_ptr[CSparseCSRMatrix]* out)

    CStatus TensorToSparseCSCMatrix(shared_ptr[CTensor],
                                    shared_ptr[CSparseCSCMatrix]* out)

    CStatus TensorToSparseCSFTensor(shared_ptr[CTensor],
                                    shared_ptr[CSparseCSFTensor]* out)

    CStatus ConvertArrayToPandas(const PandasOptions& options,
                                 shared_ptr[CArray] arr,
                                 object py_ref, PyObject** out)

    CStatus ConvertChunkedArrayToPandas(const PandasOptions& options,
                                        shared_ptr[CChunkedArray] arr,
                                        object py_ref, PyObject** out)

    CStatus ConvertTableToPandas(const PandasOptions& options,
                                 shared_ptr[CTable] table,
                                 PyObject** out)

    void c_set_default_memory_pool \
        " arrow::py::set_default_memory_pool"(CMemoryPool* pool)\

    CMemoryPool* c_get_memory_pool \
        " arrow::py::get_memory_pool"()

    cdef cppclass PyBuffer(CBuffer):
        @staticmethod
        CResult[shared_ptr[CBuffer]] FromPyObject(object obj)

    cdef cppclass PyForeignBuffer(CBuffer):
        @staticmethod
        CStatus Make(const uint8_t* data, int64_t size, object base,
                     shared_ptr[CBuffer]* out)

    cdef cppclass PyReadableFile(CRandomAccessFile):
        PyReadableFile(object fo)

    cdef cppclass PyOutputStream(COutputStream):
        PyOutputStream(object fo)

    cdef cppclass PandasOptions:
        CMemoryPool* pool
        c_bool strings_to_categorical
        c_bool zero_copy_only
        c_bool integer_object_nulls
        c_bool date_as_object
        c_bool timestamp_as_object
        c_bool use_threads
        c_bool coerce_temporal_nanoseconds
        c_bool ignore_timezone
        c_bool deduplicate_objects
        c_bool safe_cast
        c_bool split_blocks
        c_bool self_destruct
        c_bool decode_dictionaries
        unordered_set[c_string] categorical_columns
        unordered_set[c_string] extension_columns

    cdef cppclass CSerializedPyObject" arrow::py::SerializedPyObject":
        shared_ptr[CRecordBatch] batch
        vector[shared_ptr[CTensor]] tensors

        CStatus WriteTo(COutputStream* dst)
        CStatus GetComponents(CMemoryPool* pool, PyObject** dst)

    CStatus SerializeObject(object context, object sequence,
                            CSerializedPyObject* out)

    CStatus DeserializeObject(object context,
                              const CSerializedPyObject& obj,
                              PyObject* base, PyObject** out)

    CStatus ReadSerializedObject(CRandomAccessFile* src,
                                 CSerializedPyObject* out)

    cdef cppclass SparseTensorCounts:
        SparseTensorCounts()
        int coo
        int csr
        int csc
        int csf
        int ndim_csf
        int num_total_tensors() const
        int num_total_buffers() const

    CStatus GetSerializedFromComponents(
        int num_tensors,
        const SparseTensorCounts& num_sparse_tensors,
        int num_ndarrays,
        int num_buffers,
        object buffers,
        CSerializedPyObject* out)


cdef extern from "arrow/python/api.h" namespace "arrow::py::internal" nogil:
    cdef cppclass CTimePoint "arrow::py::internal::TimePoint":
        pass

    CTimePoint PyDateTime_to_TimePoint(PyDateTime_DateTime* pydatetime)
    int64_t TimePoint_to_ns(CTimePoint val)
    CTimePoint TimePoint_from_s(double val)
    CTimePoint TimePoint_from_ns(int64_t val)

    CResult[c_string] TzinfoToString(PyObject* pytzinfo)
    CResult[PyObject*] StringToTzinfo(c_string)


cdef extern from "arrow/python/init.h":
    int arrow_init_numpy() except -1


cdef extern from "arrow/python/pyarrow.h" namespace "arrow::py":
    int import_pyarrow() except -1


cdef extern from "arrow/python/common.h" namespace "arrow::py":
    c_bool IsPyError(const CStatus& status)
    void RestorePyError(const CStatus& status)


cdef extern from "arrow/python/inference.h" namespace "arrow::py":
    c_bool IsPyBool(object o)
    c_bool IsPyInt(object o)
    c_bool IsPyFloat(object o)


cdef extern from "arrow/python/ipc.h" namespace "arrow::py":
    cdef cppclass CPyRecordBatchReader" arrow::py::PyRecordBatchReader" \
            (CRecordBatchReader):
        @staticmethod
        CResult[shared_ptr[CRecordBatchReader]] Make(shared_ptr[CSchema],
                                                     object)


cdef extern from "arrow/python/extension_type.h" namespace "arrow::py":
    cdef cppclass CPyExtensionType \
            " arrow::py::PyExtensionType"(CExtensionType):
        @staticmethod
        CStatus FromClass(const shared_ptr[CDataType] storage_type,
                          const c_string extension_name, object typ,
                          shared_ptr[CExtensionType]* out)

        @staticmethod
        CStatus FromInstance(shared_ptr[CDataType] storage_type,
                             object inst, shared_ptr[CExtensionType]* out)

        object GetInstance()
        CStatus SetInstance(object)

    c_string PyExtensionName()
    CStatus RegisterPyExtensionType(shared_ptr[CDataType])
    CStatus UnregisterPyExtensionType(c_string type_name)


cdef extern from "arrow/python/benchmark.h" namespace "arrow::py::benchmark":
    void Benchmark_PandasObjectIsNull(object lst) except *


cdef extern from "arrow/python/gdb.h" namespace "arrow::gdb" nogil:
    void GdbTestSession "arrow::gdb::TestSession"()

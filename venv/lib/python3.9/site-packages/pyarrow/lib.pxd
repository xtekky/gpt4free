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

from cpython cimport PyObject
from libcpp cimport nullptr, bool as c_bool
from libcpp.cast cimport dynamic_cast
from libcpp.memory cimport dynamic_pointer_cast
from pyarrow.includes.common cimport *
from pyarrow.includes.libarrow cimport *
from pyarrow.includes.libarrow_python cimport *

# Will be available in Cython 3, not backported
# ref: https://github.com/cython/cython/issues/3293#issuecomment-1223058101
cdef extern from "<optional>" namespace "std" nogil:
    cdef cppclass nullopt_t:
        nullopt_t()

    cdef nullopt_t nullopt

    cdef cppclass optional[T]:
        ctypedef T value_type
        optional()
        optional(nullopt_t)
        optional(optional&) except +
        optional(T&) except +
        c_bool has_value()
        T& value()
        T& value_or[U](U& default_value)
        void swap(optional&)
        void reset()
        T& emplace(...)
        T& operator*()
        # T* operator->() # Not Supported
        optional& operator=(optional&)
        optional& operator=[U](U&)
        c_bool operator bool()
        c_bool operator!()
        c_bool operator==[U](optional&, U&)
        c_bool operator!=[U](optional&, U&)
        c_bool operator<[U](optional&, U&)
        c_bool operator>[U](optional&, U&)
        c_bool operator<=[U](optional&, U&)
        c_bool operator>=[U](optional&, U&)

    optional[T] make_optional[T](...) except +

cdef extern from "Python.h":
    int PySlice_Check(object)


cdef int check_status(const CStatus& status) nogil except -1


cdef class _Weakrefable:
    cdef object __weakref__


cdef class IpcWriteOptions(_Weakrefable):
    cdef:
        CIpcWriteOptions c_options


cdef class IpcReadOptions(_Weakrefable):
    cdef:
        CIpcReadOptions c_options


cdef class Message(_Weakrefable):
    cdef:
        unique_ptr[CMessage] message


cdef class MemoryPool(_Weakrefable):
    cdef:
        CMemoryPool* pool

    cdef void init(self, CMemoryPool* pool)


cdef CMemoryPool* maybe_unbox_memory_pool(MemoryPool memory_pool)


cdef object box_memory_pool(CMemoryPool* pool)


cdef class DataType(_Weakrefable):
    cdef:
        shared_ptr[CDataType] sp_type
        CDataType* type
        bytes pep3118_format

    cdef void init(self, const shared_ptr[CDataType]& type) except *
    cpdef Field field(self, i)


cdef class ListType(DataType):
    cdef:
        const CListType* list_type


cdef class LargeListType(DataType):
    cdef:
        const CLargeListType* list_type


cdef class MapType(DataType):
    cdef:
        const CMapType* map_type


cdef class FixedSizeListType(DataType):
    cdef:
        const CFixedSizeListType* list_type


cdef class StructType(DataType):
    cdef:
        const CStructType* struct_type

    cdef Field field_by_name(self, name)


cdef class DictionaryMemo(_Weakrefable):
    cdef:
        # Even though the CDictionaryMemo instance is private, we allocate
        # it on the heap so as to avoid C++ ABI issues with Python wheels.
        shared_ptr[CDictionaryMemo] sp_memo
        CDictionaryMemo* memo


cdef class DictionaryType(DataType):
    cdef:
        const CDictionaryType* dict_type


cdef class TimestampType(DataType):
    cdef:
        const CTimestampType* ts_type


cdef class Time32Type(DataType):
    cdef:
        const CTime32Type* time_type


cdef class Time64Type(DataType):
    cdef:
        const CTime64Type* time_type


cdef class DurationType(DataType):
    cdef:
        const CDurationType* duration_type


cdef class FixedSizeBinaryType(DataType):
    cdef:
        const CFixedSizeBinaryType* fixed_size_binary_type


cdef class Decimal128Type(FixedSizeBinaryType):
    cdef:
        const CDecimal128Type* decimal128_type


cdef class Decimal256Type(FixedSizeBinaryType):
    cdef:
        const CDecimal256Type* decimal256_type


cdef class BaseExtensionType(DataType):
    cdef:
        const CExtensionType* ext_type


cdef class ExtensionType(BaseExtensionType):
    cdef:
        const CPyExtensionType* cpy_ext_type


cdef class PyExtensionType(ExtensionType):
    pass


cdef class _Metadata(_Weakrefable):
    # required because KeyValueMetadata also extends collections.abc.Mapping
    # and the first parent class must be an extension type
    pass


cdef class KeyValueMetadata(_Metadata):
    cdef:
        shared_ptr[const CKeyValueMetadata] wrapped
        const CKeyValueMetadata* metadata

    cdef void init(self, const shared_ptr[const CKeyValueMetadata]& wrapped)

    @staticmethod
    cdef wrap(const shared_ptr[const CKeyValueMetadata]& sp)
    cdef inline shared_ptr[const CKeyValueMetadata] unwrap(self) nogil


cdef class Field(_Weakrefable):
    cdef:
        shared_ptr[CField] sp_field
        CField* field

    cdef readonly:
        DataType type

    cdef void init(self, const shared_ptr[CField]& field)


cdef class Schema(_Weakrefable):
    cdef:
        shared_ptr[CSchema] sp_schema
        CSchema* schema

    cdef void init(self, const vector[shared_ptr[CField]]& fields)
    cdef void init_schema(self, const shared_ptr[CSchema]& schema)


cdef class Scalar(_Weakrefable):
    cdef:
        shared_ptr[CScalar] wrapped

    cdef void init(self, const shared_ptr[CScalar]& wrapped)

    @staticmethod
    cdef wrap(const shared_ptr[CScalar]& wrapped)

    cdef inline shared_ptr[CScalar] unwrap(self) nogil


cdef class _PandasConvertible(_Weakrefable):
    pass


cdef class Array(_PandasConvertible):
    cdef:
        shared_ptr[CArray] sp_array
        CArray* ap

    cdef readonly:
        DataType type
        # To allow Table to propagate metadata to pandas.Series
        object _name

    cdef void init(self, const shared_ptr[CArray]& sp_array) except *
    cdef getitem(self, int64_t i)
    cdef int64_t length(self)


cdef class Tensor(_Weakrefable):
    cdef:
        shared_ptr[CTensor] sp_tensor
        CTensor* tp

    cdef readonly:
        DataType type

    cdef void init(self, const shared_ptr[CTensor]& sp_tensor)


cdef class SparseCSRMatrix(_Weakrefable):
    cdef:
        shared_ptr[CSparseCSRMatrix] sp_sparse_tensor
        CSparseCSRMatrix* stp

    cdef readonly:
        DataType type

    cdef void init(self, const shared_ptr[CSparseCSRMatrix]& sp_sparse_tensor)


cdef class SparseCSCMatrix(_Weakrefable):
    cdef:
        shared_ptr[CSparseCSCMatrix] sp_sparse_tensor
        CSparseCSCMatrix* stp

    cdef readonly:
        DataType type

    cdef void init(self, const shared_ptr[CSparseCSCMatrix]& sp_sparse_tensor)


cdef class SparseCOOTensor(_Weakrefable):
    cdef:
        shared_ptr[CSparseCOOTensor] sp_sparse_tensor
        CSparseCOOTensor* stp

    cdef readonly:
        DataType type

    cdef void init(self, const shared_ptr[CSparseCOOTensor]& sp_sparse_tensor)


cdef class SparseCSFTensor(_Weakrefable):
    cdef:
        shared_ptr[CSparseCSFTensor] sp_sparse_tensor
        CSparseCSFTensor* stp

    cdef readonly:
        DataType type

    cdef void init(self, const shared_ptr[CSparseCSFTensor]& sp_sparse_tensor)


cdef class NullArray(Array):
    pass


cdef class BooleanArray(Array):
    pass


cdef class NumericArray(Array):
    pass


cdef class IntegerArray(NumericArray):
    pass


cdef class FloatingPointArray(NumericArray):
    pass


cdef class Int8Array(IntegerArray):
    pass


cdef class UInt8Array(IntegerArray):
    pass


cdef class Int16Array(IntegerArray):
    pass


cdef class UInt16Array(IntegerArray):
    pass


cdef class Int32Array(IntegerArray):
    pass


cdef class UInt32Array(IntegerArray):
    pass


cdef class Int64Array(IntegerArray):
    pass


cdef class UInt64Array(IntegerArray):
    pass


cdef class HalfFloatArray(FloatingPointArray):
    pass


cdef class FloatArray(FloatingPointArray):
    pass


cdef class DoubleArray(FloatingPointArray):
    pass


cdef class FixedSizeBinaryArray(Array):
    pass


cdef class Decimal128Array(FixedSizeBinaryArray):
    pass


cdef class Decimal256Array(FixedSizeBinaryArray):
    pass


cdef class StructArray(Array):
    pass


cdef class BaseListArray(Array):
    pass


cdef class ListArray(BaseListArray):
    pass


cdef class LargeListArray(BaseListArray):
    pass


cdef class MapArray(ListArray):
    pass


cdef class FixedSizeListArray(BaseListArray):
    pass


cdef class UnionArray(Array):
    pass


cdef class StringArray(Array):
    pass


cdef class BinaryArray(Array):
    pass


cdef class DictionaryArray(Array):
    cdef:
        object _indices, _dictionary


cdef class ExtensionArray(Array):
    pass


cdef class MonthDayNanoIntervalArray(Array):
    pass


cdef wrap_array_output(PyObject* output)
cdef wrap_datum(const CDatum& datum)


cdef class ChunkedArray(_PandasConvertible):
    cdef:
        shared_ptr[CChunkedArray] sp_chunked_array
        CChunkedArray* chunked_array

    cdef readonly:
        # To allow Table to propagate metadata to pandas.Series
        object _name

    cdef void init(self, const shared_ptr[CChunkedArray]& chunked_array)
    cdef getitem(self, int64_t i)


cdef class Table(_PandasConvertible):
    cdef:
        shared_ptr[CTable] sp_table
        CTable* table

    cdef void init(self, const shared_ptr[CTable]& table)


cdef class RecordBatch(_PandasConvertible):
    cdef:
        shared_ptr[CRecordBatch] sp_batch
        CRecordBatch* batch
        Schema _schema

    cdef void init(self, const shared_ptr[CRecordBatch]& table)


cdef class Buffer(_Weakrefable):
    cdef:
        shared_ptr[CBuffer] buffer
        Py_ssize_t shape[1]
        Py_ssize_t strides[1]

    cdef void init(self, const shared_ptr[CBuffer]& buffer)
    cdef getitem(self, int64_t i)


cdef class ResizableBuffer(Buffer):

    cdef void init_rz(self, const shared_ptr[CResizableBuffer]& buffer)


cdef class NativeFile(_Weakrefable):
    cdef:
        shared_ptr[CInputStream] input_stream
        shared_ptr[CRandomAccessFile] random_access
        shared_ptr[COutputStream] output_stream
        bint is_readable
        bint is_writable
        bint is_seekable
        bint own_file

    # By implementing these "virtual" functions (all functions in Cython
    # extension classes are technically virtual in the C++ sense) we can expose
    # the arrow::io abstract file interfaces to other components throughout the
    # suite of Arrow C++ libraries
    cdef set_random_access_file(self, shared_ptr[CRandomAccessFile] handle)
    cdef set_input_stream(self, shared_ptr[CInputStream] handle)
    cdef set_output_stream(self, shared_ptr[COutputStream] handle)

    cdef shared_ptr[CRandomAccessFile] get_random_access_file(self) except *
    cdef shared_ptr[CInputStream] get_input_stream(self) except *
    cdef shared_ptr[COutputStream] get_output_stream(self) except *


cdef class BufferedInputStream(NativeFile):
    pass


cdef class BufferedOutputStream(NativeFile):
    pass


cdef class CompressedInputStream(NativeFile):
    pass


cdef class CompressedOutputStream(NativeFile):
    pass


cdef class _CRecordBatchWriter(_Weakrefable):
    cdef:
        shared_ptr[CRecordBatchWriter] writer


cdef class RecordBatchReader(_Weakrefable):
    cdef:
        shared_ptr[CRecordBatchReader] reader


cdef class Codec(_Weakrefable):
    cdef:
        shared_ptr[CCodec] wrapped

    cdef inline CCodec* unwrap(self) nogil


# This class is only used internally for now
cdef class StopToken:
    cdef:
        CStopToken stop_token

    cdef void init(self, CStopToken stop_token)


cdef get_input_stream(object source, c_bool use_memory_map,
                      shared_ptr[CInputStream]* reader)
cdef get_reader(object source, c_bool use_memory_map,
                shared_ptr[CRandomAccessFile]* reader)
cdef get_writer(object source, shared_ptr[COutputStream]* writer)
cdef NativeFile get_native_file(object source, c_bool use_memory_map)

cdef shared_ptr[CInputStream] native_transcoding_input_stream(
    shared_ptr[CInputStream] stream, src_encoding,
    dest_encoding) except *

cdef shared_ptr[function[StreamWrapFunc]] make_streamwrap_func(
    src_encoding, dest_encoding) except *

# Default is allow_none=False
cpdef DataType ensure_type(object type, bint allow_none=*)

cdef timeunit_to_string(TimeUnit unit)
cdef TimeUnit string_to_timeunit(unit) except *

# Exceptions may be raised when converting dict values, so need to
# check exception state on return
cdef shared_ptr[const CKeyValueMetadata] pyarrow_unwrap_metadata(
    object meta) except *
cdef object pyarrow_wrap_metadata(
    const shared_ptr[const CKeyValueMetadata]& meta)

#
# Public Cython API for 3rd party code
#
# If you add functions to this list, please also update
# `cpp/src/arrow/python/pyarrow.{h, cc}`
#

# Wrapping C++ -> Python

cdef public object pyarrow_wrap_buffer(const shared_ptr[CBuffer]& buf)
cdef public object pyarrow_wrap_resizable_buffer(
    const shared_ptr[CResizableBuffer]& buf)

cdef public object pyarrow_wrap_data_type(const shared_ptr[CDataType]& type)
cdef public object pyarrow_wrap_field(const shared_ptr[CField]& field)
cdef public object pyarrow_wrap_schema(const shared_ptr[CSchema]& type)

cdef public object pyarrow_wrap_scalar(const shared_ptr[CScalar]& sp_scalar)

cdef public object pyarrow_wrap_array(const shared_ptr[CArray]& sp_array)
cdef public object pyarrow_wrap_chunked_array(
    const shared_ptr[CChunkedArray]& sp_array)

cdef public object pyarrow_wrap_sparse_coo_tensor(
    const shared_ptr[CSparseCOOTensor]& sp_sparse_tensor)
cdef public object pyarrow_wrap_sparse_csc_matrix(
    const shared_ptr[CSparseCSCMatrix]& sp_sparse_tensor)
cdef public object pyarrow_wrap_sparse_csf_tensor(
    const shared_ptr[CSparseCSFTensor]& sp_sparse_tensor)
cdef public object pyarrow_wrap_sparse_csr_matrix(
    const shared_ptr[CSparseCSRMatrix]& sp_sparse_tensor)
cdef public object pyarrow_wrap_tensor(const shared_ptr[CTensor]& sp_tensor)

cdef public object pyarrow_wrap_batch(const shared_ptr[CRecordBatch]& cbatch)
cdef public object pyarrow_wrap_table(const shared_ptr[CTable]& ctable)

# Unwrapping Python -> C++

cdef public shared_ptr[CBuffer] pyarrow_unwrap_buffer(object buffer)

cdef public shared_ptr[CDataType] pyarrow_unwrap_data_type(object data_type)
cdef public shared_ptr[CField] pyarrow_unwrap_field(object field)
cdef public shared_ptr[CSchema] pyarrow_unwrap_schema(object schema)

cdef public shared_ptr[CScalar] pyarrow_unwrap_scalar(object scalar)

cdef public shared_ptr[CArray] pyarrow_unwrap_array(object array)
cdef public shared_ptr[CChunkedArray] pyarrow_unwrap_chunked_array(
    object array)

cdef public shared_ptr[CSparseCOOTensor] pyarrow_unwrap_sparse_coo_tensor(
    object sparse_tensor)
cdef public shared_ptr[CSparseCSCMatrix] pyarrow_unwrap_sparse_csc_matrix(
    object sparse_tensor)
cdef public shared_ptr[CSparseCSFTensor] pyarrow_unwrap_sparse_csf_tensor(
    object sparse_tensor)
cdef public shared_ptr[CSparseCSRMatrix] pyarrow_unwrap_sparse_csr_matrix(
    object sparse_tensor)
cdef public shared_ptr[CTensor] pyarrow_unwrap_tensor(object tensor)

cdef public shared_ptr[CRecordBatch] pyarrow_unwrap_batch(object batch)
cdef public shared_ptr[CTable] pyarrow_unwrap_table(object table)

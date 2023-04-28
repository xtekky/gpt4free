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


cdef extern from "arrow/util/key_value_metadata.h" namespace "arrow" nogil:
    cdef cppclass CKeyValueMetadata" arrow::KeyValueMetadata":
        CKeyValueMetadata()
        CKeyValueMetadata(const unordered_map[c_string, c_string]&)
        CKeyValueMetadata(const vector[c_string]& keys,
                          const vector[c_string]& values)

        void reserve(int64_t n)
        int64_t size() const
        c_string key(int64_t i) const
        c_string value(int64_t i) const
        int FindKey(const c_string& key) const

        shared_ptr[CKeyValueMetadata] Copy() const
        c_bool Equals(const CKeyValueMetadata& other)
        void Append(const c_string& key, const c_string& value)
        void ToUnorderedMap(unordered_map[c_string, c_string]*) const
        c_string ToString() const

        CResult[c_string] Get(const c_string& key) const
        CStatus Delete(const c_string& key)
        CStatus Set(const c_string& key, const c_string& value)
        c_bool Contains(const c_string& key) const


cdef extern from "arrow/util/decimal.h" namespace "arrow" nogil:
    cdef cppclass CDecimal128" arrow::Decimal128":
        c_string ToString(int32_t scale) const


cdef extern from "arrow/util/decimal.h" namespace "arrow" nogil:
    cdef cppclass CDecimal256" arrow::Decimal256":
        c_string ToString(int32_t scale) const


cdef extern from "arrow/config.h" namespace "arrow" nogil:
    cdef cppclass CBuildInfo" arrow::BuildInfo":
        int version
        int version_major
        int version_minor
        int version_patch
        c_string version_string
        c_string so_version
        c_string full_so_version
        c_string compiler_id
        c_string compiler_version
        c_string compiler_flags
        c_string git_id
        c_string git_description
        c_string package_kind
        c_string build_type

    const CBuildInfo& GetBuildInfo()

    cdef cppclass CRuntimeInfo" arrow::RuntimeInfo":
        c_string simd_level
        c_string detected_simd_level

    CRuntimeInfo GetRuntimeInfo()


cdef extern from "arrow/util/future.h" namespace "arrow" nogil:
    cdef cppclass CFuture_Void" arrow::Future<>":
        CStatus status()


cdef extern from "arrow/api.h" namespace "arrow" nogil:
    cdef enum Type" arrow::Type::type":
        _Type_NA" arrow::Type::NA"

        _Type_BOOL" arrow::Type::BOOL"

        _Type_UINT8" arrow::Type::UINT8"
        _Type_INT8" arrow::Type::INT8"
        _Type_UINT16" arrow::Type::UINT16"
        _Type_INT16" arrow::Type::INT16"
        _Type_UINT32" arrow::Type::UINT32"
        _Type_INT32" arrow::Type::INT32"
        _Type_UINT64" arrow::Type::UINT64"
        _Type_INT64" arrow::Type::INT64"

        _Type_HALF_FLOAT" arrow::Type::HALF_FLOAT"
        _Type_FLOAT" arrow::Type::FLOAT"
        _Type_DOUBLE" arrow::Type::DOUBLE"

        _Type_DECIMAL128" arrow::Type::DECIMAL128"
        _Type_DECIMAL256" arrow::Type::DECIMAL256"

        _Type_DATE32" arrow::Type::DATE32"
        _Type_DATE64" arrow::Type::DATE64"
        _Type_TIMESTAMP" arrow::Type::TIMESTAMP"
        _Type_TIME32" arrow::Type::TIME32"
        _Type_TIME64" arrow::Type::TIME64"
        _Type_DURATION" arrow::Type::DURATION"
        _Type_INTERVAL_MONTH_DAY_NANO" arrow::Type::INTERVAL_MONTH_DAY_NANO"

        _Type_BINARY" arrow::Type::BINARY"
        _Type_STRING" arrow::Type::STRING"
        _Type_LARGE_BINARY" arrow::Type::LARGE_BINARY"
        _Type_LARGE_STRING" arrow::Type::LARGE_STRING"
        _Type_FIXED_SIZE_BINARY" arrow::Type::FIXED_SIZE_BINARY"

        _Type_LIST" arrow::Type::LIST"
        _Type_LARGE_LIST" arrow::Type::LARGE_LIST"
        _Type_FIXED_SIZE_LIST" arrow::Type::FIXED_SIZE_LIST"
        _Type_STRUCT" arrow::Type::STRUCT"
        _Type_SPARSE_UNION" arrow::Type::SPARSE_UNION"
        _Type_DENSE_UNION" arrow::Type::DENSE_UNION"
        _Type_DICTIONARY" arrow::Type::DICTIONARY"
        _Type_MAP" arrow::Type::MAP"

        _Type_EXTENSION" arrow::Type::EXTENSION"

    cdef enum UnionMode" arrow::UnionMode::type":
        _UnionMode_SPARSE" arrow::UnionMode::SPARSE"
        _UnionMode_DENSE" arrow::UnionMode::DENSE"

    cdef enum TimeUnit" arrow::TimeUnit::type":
        TimeUnit_SECOND" arrow::TimeUnit::SECOND"
        TimeUnit_MILLI" arrow::TimeUnit::MILLI"
        TimeUnit_MICRO" arrow::TimeUnit::MICRO"
        TimeUnit_NANO" arrow::TimeUnit::NANO"

    cdef cppclass CBufferSpec" arrow::DataTypeLayout::BufferSpec":
        pass

    cdef cppclass CDataTypeLayout" arrow::DataTypeLayout":
        vector[CBufferSpec] buffers
        c_bool has_dictionary

    cdef cppclass CDataType" arrow::DataType":
        Type id()

        c_bool Equals(const CDataType& other, c_bool check_metadata)
        c_bool Equals(const shared_ptr[CDataType]& other, c_bool check_metadata)

        shared_ptr[CField] field(int i)
        const vector[shared_ptr[CField]] fields()
        int num_fields()
        CDataTypeLayout layout()
        c_string ToString()

    c_bool is_primitive(Type type)

    cdef cppclass CArrayData" arrow::ArrayData":
        shared_ptr[CDataType] type
        int64_t length
        int64_t null_count
        int64_t offset
        vector[shared_ptr[CBuffer]] buffers
        vector[shared_ptr[CArrayData]] child_data
        shared_ptr[CArrayData] dictionary

        @staticmethod
        shared_ptr[CArrayData] Make(const shared_ptr[CDataType]& type,
                                    int64_t length,
                                    vector[shared_ptr[CBuffer]]& buffers,
                                    int64_t null_count,
                                    int64_t offset)

        @staticmethod
        shared_ptr[CArrayData] MakeWithChildren" Make"(
            const shared_ptr[CDataType]& type,
            int64_t length,
            vector[shared_ptr[CBuffer]]& buffers,
            vector[shared_ptr[CArrayData]]& child_data,
            int64_t null_count,
            int64_t offset)

        @staticmethod
        shared_ptr[CArrayData] MakeWithChildrenAndDictionary" Make"(
            const shared_ptr[CDataType]& type,
            int64_t length,
            vector[shared_ptr[CBuffer]]& buffers,
            vector[shared_ptr[CArrayData]]& child_data,
            shared_ptr[CArrayData]& dictionary,
            int64_t null_count,
            int64_t offset)

    cdef cppclass CArray" arrow::Array":
        shared_ptr[CDataType] type()

        int64_t length()
        int64_t null_count()
        int64_t offset()
        Type type_id()

        int num_fields()

        CResult[shared_ptr[CScalar]] GetScalar(int64_t i) const

        c_string Diff(const CArray& other)
        c_bool Equals(const CArray& arr)
        c_bool IsNull(int i)

        shared_ptr[CArrayData] data()

        shared_ptr[CArray] Slice(int64_t offset)
        shared_ptr[CArray] Slice(int64_t offset, int64_t length)

        CStatus Validate() const
        CStatus ValidateFull() const
        CResult[shared_ptr[CArray]] View(const shared_ptr[CDataType]& type)

    shared_ptr[CArray] MakeArray(const shared_ptr[CArrayData]& data)
    CResult[shared_ptr[CArray]] MakeArrayOfNull(
        const shared_ptr[CDataType]& type, int64_t length, CMemoryPool* pool)

    CResult[shared_ptr[CArray]] MakeArrayFromScalar(
        const CScalar& scalar, int64_t length, CMemoryPool* pool)

    CStatus DebugPrint(const CArray& arr, int indent)

    cdef cppclass CFixedWidthType" arrow::FixedWidthType"(CDataType):
        int bit_width()

    cdef cppclass CNullArray" arrow::NullArray"(CArray):
        CNullArray(int64_t length)

    cdef cppclass CDictionaryArray" arrow::DictionaryArray"(CArray):
        CDictionaryArray(const shared_ptr[CDataType]& type,
                         const shared_ptr[CArray]& indices,
                         const shared_ptr[CArray]& dictionary)
        CDictionaryArray(const shared_ptr[CArrayData]& data)

        @staticmethod
        CResult[shared_ptr[CArray]] FromArrays(
            const shared_ptr[CDataType]& type,
            const shared_ptr[CArray]& indices,
            const shared_ptr[CArray]& dictionary)

        shared_ptr[CArray] indices()
        shared_ptr[CArray] dictionary()

    cdef cppclass CDate32Type" arrow::Date32Type"(CFixedWidthType):
        pass

    cdef cppclass CDate64Type" arrow::Date64Type"(CFixedWidthType):
        pass

    cdef cppclass CTimestampType" arrow::TimestampType"(CFixedWidthType):
        CTimestampType(TimeUnit unit)
        TimeUnit unit()
        const c_string& timezone()

    cdef cppclass CTime32Type" arrow::Time32Type"(CFixedWidthType):
        TimeUnit unit()

    cdef cppclass CTime64Type" arrow::Time64Type"(CFixedWidthType):
        TimeUnit unit()

    shared_ptr[CDataType] ctime32" arrow::time32"(TimeUnit unit)
    shared_ptr[CDataType] ctime64" arrow::time64"(TimeUnit unit)

    cdef cppclass CDurationType" arrow::DurationType"(CFixedWidthType):
        TimeUnit unit()

    shared_ptr[CDataType] cduration" arrow::duration"(TimeUnit unit)

    cdef cppclass CDictionaryType" arrow::DictionaryType"(CFixedWidthType):
        CDictionaryType(const shared_ptr[CDataType]& index_type,
                        const shared_ptr[CDataType]& value_type,
                        c_bool ordered)

        shared_ptr[CDataType] index_type()
        shared_ptr[CDataType] value_type()
        c_bool ordered()

    shared_ptr[CDataType] ctimestamp" arrow::timestamp"(TimeUnit unit)
    shared_ptr[CDataType] ctimestamp" arrow::timestamp"(
        TimeUnit unit, const c_string& timezone)

    cdef cppclass CMemoryPool" arrow::MemoryPool":
        int64_t bytes_allocated()
        int64_t max_memory()
        c_string backend_name()
        void ReleaseUnused()

    cdef cppclass CLoggingMemoryPool" arrow::LoggingMemoryPool"(CMemoryPool):
        CLoggingMemoryPool(CMemoryPool*)

    cdef cppclass CProxyMemoryPool" arrow::ProxyMemoryPool"(CMemoryPool):
        CProxyMemoryPool(CMemoryPool*)

    cdef cppclass CBuffer" arrow::Buffer":
        CBuffer(const uint8_t* data, int64_t size)
        const uint8_t* data()
        uint8_t* mutable_data()
        uintptr_t address()
        uintptr_t mutable_address()
        int64_t size()
        shared_ptr[CBuffer] parent()
        c_bool is_cpu() const
        c_bool is_mutable() const
        c_string ToHexString()
        c_bool Equals(const CBuffer& other)

    CResult[shared_ptr[CBuffer]] SliceBufferSafe(
        const shared_ptr[CBuffer]& buffer, int64_t offset)
    CResult[shared_ptr[CBuffer]] SliceBufferSafe(
        const shared_ptr[CBuffer]& buffer, int64_t offset, int64_t length)

    cdef cppclass CMutableBuffer" arrow::MutableBuffer"(CBuffer):
        CMutableBuffer(const uint8_t* data, int64_t size)

    cdef cppclass CResizableBuffer" arrow::ResizableBuffer"(CMutableBuffer):
        CStatus Resize(const int64_t new_size, c_bool shrink_to_fit)
        CStatus Reserve(const int64_t new_size)

    CResult[unique_ptr[CBuffer]] AllocateBuffer(const int64_t size,
                                                CMemoryPool* pool)

    CResult[unique_ptr[CResizableBuffer]] AllocateResizableBuffer(
        const int64_t size, CMemoryPool* pool)

    cdef CMemoryPool* c_default_memory_pool" arrow::default_memory_pool"()
    cdef CMemoryPool* c_system_memory_pool" arrow::system_memory_pool"()
    cdef CStatus c_jemalloc_memory_pool" arrow::jemalloc_memory_pool"(
        CMemoryPool** out)
    cdef CStatus c_mimalloc_memory_pool" arrow::mimalloc_memory_pool"(
        CMemoryPool** out)
    cdef vector[c_string] c_supported_memory_backends \
        " arrow::SupportedMemoryBackendNames"()

    CStatus c_jemalloc_set_decay_ms" arrow::jemalloc_set_decay_ms"(int ms)

    cdef cppclass CListType" arrow::ListType"(CDataType):
        CListType(const shared_ptr[CDataType]& value_type)
        CListType(const shared_ptr[CField]& field)
        shared_ptr[CDataType] value_type()
        shared_ptr[CField] value_field()

    cdef cppclass CLargeListType" arrow::LargeListType"(CDataType):
        CLargeListType(const shared_ptr[CDataType]& value_type)
        CLargeListType(const shared_ptr[CField]& field)
        shared_ptr[CDataType] value_type()
        shared_ptr[CField] value_field()

    cdef cppclass CMapType" arrow::MapType"(CDataType):
        CMapType(const shared_ptr[CField]& key_field,
                 const shared_ptr[CField]& item_field, c_bool keys_sorted)
        shared_ptr[CDataType] key_type()
        shared_ptr[CField] key_field()
        shared_ptr[CDataType] item_type()
        shared_ptr[CField] item_field()
        c_bool keys_sorted()

    cdef cppclass CFixedSizeListType" arrow::FixedSizeListType"(CDataType):
        CFixedSizeListType(const shared_ptr[CDataType]& value_type,
                           int32_t list_size)
        CFixedSizeListType(const shared_ptr[CField]& field, int32_t list_size)
        shared_ptr[CDataType] value_type()
        shared_ptr[CField] value_field()
        int32_t list_size()

    cdef cppclass CStringType" arrow::StringType"(CDataType):
        pass

    cdef cppclass CFixedSizeBinaryType \
            " arrow::FixedSizeBinaryType"(CFixedWidthType):
        CFixedSizeBinaryType(int byte_width)
        int byte_width()
        int bit_width()

    cdef cppclass CDecimal128Type \
            " arrow::Decimal128Type"(CFixedSizeBinaryType):
        CDecimal128Type(int precision, int scale)
        int precision()
        int scale()

    cdef cppclass CDecimal256Type \
            " arrow::Decimal256Type"(CFixedSizeBinaryType):
        CDecimal256Type(int precision, int scale)
        int precision()
        int scale()

    cdef cppclass CField" arrow::Field":
        cppclass CMergeOptions "arrow::Field::MergeOptions":
            c_bool promote_nullability

            @staticmethod
            CMergeOptions Defaults()

        const c_string& name()
        shared_ptr[CDataType] type()
        c_bool nullable()

        c_string ToString()
        c_bool Equals(const CField& other, c_bool check_metadata)

        shared_ptr[const CKeyValueMetadata] metadata()

        CField(const c_string& name, const shared_ptr[CDataType]& type,
               c_bool nullable)

        CField(const c_string& name, const shared_ptr[CDataType]& type,
               c_bool nullable, const shared_ptr[CKeyValueMetadata]& metadata)

        # Removed const in Cython so don't have to cast to get code to generate
        shared_ptr[CField] AddMetadata(
            const shared_ptr[CKeyValueMetadata]& metadata)
        shared_ptr[CField] WithMetadata(
            const shared_ptr[CKeyValueMetadata]& metadata)
        shared_ptr[CField] RemoveMetadata()
        shared_ptr[CField] WithType(const shared_ptr[CDataType]& type)
        shared_ptr[CField] WithName(const c_string& name)
        shared_ptr[CField] WithNullable(c_bool nullable)
        vector[shared_ptr[CField]] Flatten()

    cdef cppclass CFieldRef" arrow::FieldRef":
        CFieldRef()
        CFieldRef(c_string name)
        CFieldRef(int index)
        CFieldRef(vector[CFieldRef])

        @staticmethod
        CResult[CFieldRef] FromDotPath(c_string& dot_path)
        const c_string* name() const

    cdef cppclass CFieldRefHash" arrow::FieldRef::Hash":
        pass

    cdef cppclass CStructType" arrow::StructType"(CDataType):
        CStructType(const vector[shared_ptr[CField]]& fields)

        shared_ptr[CField] GetFieldByName(const c_string& name)
        vector[shared_ptr[CField]] GetAllFieldsByName(const c_string& name)
        int GetFieldIndex(const c_string& name)
        vector[int] GetAllFieldIndices(const c_string& name)

    cdef cppclass CUnionType" arrow::UnionType"(CDataType):
        UnionMode mode()
        const vector[int8_t]& type_codes()
        const vector[int]& child_ids()

    cdef shared_ptr[CDataType] CMakeSparseUnionType" arrow::sparse_union"(
        vector[shared_ptr[CField]] fields,
        vector[int8_t] type_codes)

    cdef shared_ptr[CDataType] CMakeDenseUnionType" arrow::dense_union"(
        vector[shared_ptr[CField]] fields,
        vector[int8_t] type_codes)

    cdef cppclass CSchema" arrow::Schema":
        CSchema(const vector[shared_ptr[CField]]& fields)
        CSchema(const vector[shared_ptr[CField]]& fields,
                const shared_ptr[const CKeyValueMetadata]& metadata)

        # Does not actually exist, but gets Cython to not complain
        CSchema(const vector[shared_ptr[CField]]& fields,
                const shared_ptr[CKeyValueMetadata]& metadata)

        c_bool Equals(const CSchema& other, c_bool check_metadata)

        shared_ptr[CField] field(int i)
        shared_ptr[const CKeyValueMetadata] metadata()
        shared_ptr[CField] GetFieldByName(const c_string& name)
        vector[shared_ptr[CField]] GetAllFieldsByName(const c_string& name)
        int GetFieldIndex(const c_string& name)
        vector[int] GetAllFieldIndices(const c_string& name)
        int num_fields()
        c_string ToString()

        CResult[shared_ptr[CSchema]] AddField(int i,
                                              const shared_ptr[CField]& field)
        CResult[shared_ptr[CSchema]] RemoveField(int i)
        CResult[shared_ptr[CSchema]] SetField(int i,
                                              const shared_ptr[CField]& field)

        # Removed const in Cython so don't have to cast to get code to generate
        shared_ptr[CSchema] AddMetadata(
            const shared_ptr[CKeyValueMetadata]& metadata)
        shared_ptr[CSchema] WithMetadata(
            const shared_ptr[CKeyValueMetadata]& metadata)
        shared_ptr[CSchema] RemoveMetadata()

    CResult[shared_ptr[CSchema]] UnifySchemas(
        const vector[shared_ptr[CSchema]]& schemas)

    cdef cppclass PrettyPrintOptions:
        PrettyPrintOptions()
        PrettyPrintOptions(int indent_arg)
        PrettyPrintOptions(int indent_arg, int window_arg)
        int indent
        int indent_size
        int window
        int container_window
        c_string null_rep
        c_bool skip_new_lines
        c_bool truncate_metadata
        c_bool show_field_metadata
        c_bool show_schema_metadata

        @staticmethod
        PrettyPrintOptions Defaults()

    CStatus PrettyPrint(const CArray& schema,
                        const PrettyPrintOptions& options,
                        c_string* result)
    CStatus PrettyPrint(const CChunkedArray& schema,
                        const PrettyPrintOptions& options,
                        c_string* result)
    CStatus PrettyPrint(const CSchema& schema,
                        const PrettyPrintOptions& options,
                        c_string* result)

    cdef cppclass CBooleanArray" arrow::BooleanArray"(CArray):
        c_bool Value(int i)
        int64_t false_count()
        int64_t true_count()

    cdef cppclass CUInt8Array" arrow::UInt8Array"(CArray):
        uint8_t Value(int i)

    cdef cppclass CInt8Array" arrow::Int8Array"(CArray):
        int8_t Value(int i)

    cdef cppclass CUInt16Array" arrow::UInt16Array"(CArray):
        uint16_t Value(int i)

    cdef cppclass CInt16Array" arrow::Int16Array"(CArray):
        int16_t Value(int i)

    cdef cppclass CUInt32Array" arrow::UInt32Array"(CArray):
        uint32_t Value(int i)

    cdef cppclass CInt32Array" arrow::Int32Array"(CArray):
        int32_t Value(int i)

    cdef cppclass CUInt64Array" arrow::UInt64Array"(CArray):
        uint64_t Value(int i)

    cdef cppclass CInt64Array" arrow::Int64Array"(CArray):
        int64_t Value(int i)

    cdef cppclass CDate32Array" arrow::Date32Array"(CArray):
        int32_t Value(int i)

    cdef cppclass CDate64Array" arrow::Date64Array"(CArray):
        int64_t Value(int i)

    cdef cppclass CTime32Array" arrow::Time32Array"(CArray):
        int32_t Value(int i)

    cdef cppclass CTime64Array" arrow::Time64Array"(CArray):
        int64_t Value(int i)

    cdef cppclass CTimestampArray" arrow::TimestampArray"(CArray):
        int64_t Value(int i)

    cdef cppclass CDurationArray" arrow::DurationArray"(CArray):
        int64_t Value(int i)

    cdef cppclass CMonthDayNanoIntervalArray \
            "arrow::MonthDayNanoIntervalArray"(CArray):
        pass

    cdef cppclass CHalfFloatArray" arrow::HalfFloatArray"(CArray):
        uint16_t Value(int i)

    cdef cppclass CFloatArray" arrow::FloatArray"(CArray):
        float Value(int i)

    cdef cppclass CDoubleArray" arrow::DoubleArray"(CArray):
        double Value(int i)

    cdef cppclass CFixedSizeBinaryArray" arrow::FixedSizeBinaryArray"(CArray):
        const uint8_t* GetValue(int i)

    cdef cppclass CDecimal128Array" arrow::Decimal128Array"(
        CFixedSizeBinaryArray
    ):
        c_string FormatValue(int i)

    cdef cppclass CDecimal256Array" arrow::Decimal256Array"(
        CFixedSizeBinaryArray
    ):
        c_string FormatValue(int i)

    cdef cppclass CListArray" arrow::ListArray"(CArray):
        @staticmethod
        CResult[shared_ptr[CArray]] FromArrays(
            const CArray& offsets,
            const CArray& values,
            CMemoryPool* pool,
            shared_ptr[CBuffer] null_bitmap,
        )

        @staticmethod
        CResult[shared_ptr[CArray]] FromArraysAndType" FromArrays"(
            shared_ptr[CDataType],
            const CArray& offsets,
            const CArray& values,
            CMemoryPool* pool,
            shared_ptr[CBuffer] null_bitmap,
        )

        const int32_t* raw_value_offsets()
        int32_t value_offset(int i)
        int32_t value_length(int i)
        shared_ptr[CArray] values()
        shared_ptr[CArray] offsets()
        shared_ptr[CDataType] value_type()

    cdef cppclass CLargeListArray" arrow::LargeListArray"(CArray):
        @staticmethod
        CResult[shared_ptr[CArray]] FromArrays(
            const CArray& offsets,
            const CArray& values,
            CMemoryPool* pool,
            shared_ptr[CBuffer] null_bitmap
        )

        @staticmethod
        CResult[shared_ptr[CArray]] FromArraysAndType" FromArrays"(
            shared_ptr[CDataType],
            const CArray& offsets,
            const CArray& values,
            CMemoryPool* pool,
            shared_ptr[CBuffer] null_bitmap
        )

        int64_t value_offset(int i)
        int64_t value_length(int i)
        shared_ptr[CArray] values()
        shared_ptr[CArray] offsets()
        shared_ptr[CDataType] value_type()

    cdef cppclass CFixedSizeListArray" arrow::FixedSizeListArray"(CArray):
        @staticmethod
        CResult[shared_ptr[CArray]] FromArrays(
            const shared_ptr[CArray]& values, int32_t list_size)

        @staticmethod
        CResult[shared_ptr[CArray]] FromArraysAndType" FromArrays"(
            const shared_ptr[CArray]& values, shared_ptr[CDataType])

        int64_t value_offset(int i)
        int64_t value_length(int i)
        shared_ptr[CArray] values()
        shared_ptr[CDataType] value_type()

    cdef cppclass CMapArray" arrow::MapArray"(CArray):
        @staticmethod
        CResult[shared_ptr[CArray]] FromArrays(
            const shared_ptr[CArray]& offsets,
            const shared_ptr[CArray]& keys,
            const shared_ptr[CArray]& items,
            CMemoryPool* pool)

        shared_ptr[CArray] keys()
        shared_ptr[CArray] items()
        CMapType* map_type()
        int64_t value_offset(int i)
        int64_t value_length(int i)
        shared_ptr[CArray] values()
        shared_ptr[CDataType] value_type()

    cdef cppclass CUnionArray" arrow::UnionArray"(CArray):
        shared_ptr[CBuffer] type_codes()
        int8_t* raw_type_codes()
        int child_id(int64_t index)
        shared_ptr[CArray] field(int pos)
        const CArray* UnsafeField(int pos)
        UnionMode mode()

    cdef cppclass CSparseUnionArray" arrow::SparseUnionArray"(CUnionArray):
        @staticmethod
        CResult[shared_ptr[CArray]] Make(
            const CArray& type_codes,
            const vector[shared_ptr[CArray]]& children,
            const vector[c_string]& field_names,
            const vector[int8_t]& type_codes)

    cdef cppclass CDenseUnionArray" arrow::DenseUnionArray"(CUnionArray):
        @staticmethod
        CResult[shared_ptr[CArray]] Make(
            const CArray& type_codes,
            const CArray& value_offsets,
            const vector[shared_ptr[CArray]]& children,
            const vector[c_string]& field_names,
            const vector[int8_t]& type_codes)

        int32_t value_offset(int i)
        shared_ptr[CBuffer] value_offsets()

    cdef cppclass CBinaryArray" arrow::BinaryArray"(CArray):
        const uint8_t* GetValue(int i, int32_t* length)
        shared_ptr[CBuffer] value_data()
        int32_t value_offset(int64_t i)
        int32_t value_length(int64_t i)
        int32_t total_values_length()

    cdef cppclass CLargeBinaryArray" arrow::LargeBinaryArray"(CArray):
        const uint8_t* GetValue(int i, int64_t* length)
        shared_ptr[CBuffer] value_data()
        int64_t value_offset(int64_t i)
        int64_t value_length(int64_t i)
        int64_t total_values_length()

    cdef cppclass CStringArray" arrow::StringArray"(CBinaryArray):
        CStringArray(int64_t length, shared_ptr[CBuffer] value_offsets,
                     shared_ptr[CBuffer] data,
                     shared_ptr[CBuffer] null_bitmap,
                     int64_t null_count,
                     int64_t offset)

        c_string GetString(int i)

    cdef cppclass CLargeStringArray" arrow::LargeStringArray" \
            (CLargeBinaryArray):
        CLargeStringArray(int64_t length, shared_ptr[CBuffer] value_offsets,
                          shared_ptr[CBuffer] data,
                          shared_ptr[CBuffer] null_bitmap,
                          int64_t null_count,
                          int64_t offset)

        c_string GetString(int i)

    cdef cppclass CStructArray" arrow::StructArray"(CArray):
        CStructArray(shared_ptr[CDataType]& type, int64_t length,
                     vector[shared_ptr[CArray]]& children,
                     shared_ptr[CBuffer] null_bitmap=nullptr,
                     int64_t null_count=-1,
                     int64_t offset=0)

        # XXX Cython crashes if default argument values are declared here
        # https://github.com/cython/cython/issues/2167
        @staticmethod
        CResult[shared_ptr[CArray]] MakeFromFieldNames "Make"(
            vector[shared_ptr[CArray]] children,
            vector[c_string] field_names,
            shared_ptr[CBuffer] null_bitmap,
            int64_t null_count,
            int64_t offset)

        @staticmethod
        CResult[shared_ptr[CArray]] MakeFromFields "Make"(
            vector[shared_ptr[CArray]] children,
            vector[shared_ptr[CField]] fields,
            shared_ptr[CBuffer] null_bitmap,
            int64_t null_count,
            int64_t offset)

        shared_ptr[CArray] field(int pos)
        shared_ptr[CArray] GetFieldByName(const c_string& name) const
        CResult[shared_ptr[CArray]] GetFlattenedField(int index, CMemoryPool* pool) const

        CResult[vector[shared_ptr[CArray]]] Flatten(CMemoryPool* pool)

    cdef cppclass CChunkedArray" arrow::ChunkedArray":
        CChunkedArray(const vector[shared_ptr[CArray]]& arrays)
        CChunkedArray(const vector[shared_ptr[CArray]]& arrays,
                      const shared_ptr[CDataType]& type)

        @staticmethod
        CResult[shared_ptr[CChunkedArray]] Make(vector[shared_ptr[CArray]] chunks,
                                                shared_ptr[CDataType] type)
        int64_t length()
        int64_t null_count()
        int num_chunks()
        c_bool Equals(const CChunkedArray& other)

        shared_ptr[CArray] chunk(int i)
        shared_ptr[CDataType] type()
        CResult[shared_ptr[CScalar]] GetScalar(int64_t index) const
        shared_ptr[CChunkedArray] Slice(int64_t offset, int64_t length) const
        shared_ptr[CChunkedArray] Slice(int64_t offset) const

        CResult[vector[shared_ptr[CChunkedArray]]] Flatten(CMemoryPool* pool)

        CStatus Validate() const
        CStatus ValidateFull() const

    cdef cppclass CRecordBatch" arrow::RecordBatch":
        @staticmethod
        shared_ptr[CRecordBatch] Make(
            const shared_ptr[CSchema]& schema, int64_t num_rows,
            const vector[shared_ptr[CArray]]& columns)

        @staticmethod
        CResult[shared_ptr[CRecordBatch]] FromStructArray(
            const shared_ptr[CArray]& array)

        c_bool Equals(const CRecordBatch& other, c_bool check_metadata)

        shared_ptr[CSchema] schema()
        shared_ptr[CArray] column(int i)
        const c_string& column_name(int i)

        const vector[shared_ptr[CArray]]& columns()

        int num_columns()
        int64_t num_rows()

        CStatus Validate() const
        CStatus ValidateFull() const

        shared_ptr[CRecordBatch] ReplaceSchemaMetadata(
            const shared_ptr[CKeyValueMetadata]& metadata)

        shared_ptr[CRecordBatch] Slice(int64_t offset)
        shared_ptr[CRecordBatch] Slice(int64_t offset, int64_t length)

    cdef cppclass CRecordBatchWithMetadata" arrow::RecordBatchWithMetadata":
        shared_ptr[CRecordBatch] batch
        # The struct in C++ does not actually have these two `const` qualifiers, but adding `const` gets Cython to not complain
        const shared_ptr[const CKeyValueMetadata] custom_metadata

    cdef cppclass CTable" arrow::Table":
        CTable(const shared_ptr[CSchema]& schema,
               const vector[shared_ptr[CChunkedArray]]& columns)

        @staticmethod
        shared_ptr[CTable] Make(
            const shared_ptr[CSchema]& schema,
            const vector[shared_ptr[CChunkedArray]]& columns)

        @staticmethod
        shared_ptr[CTable] MakeWithRows "Make"(
            const shared_ptr[CSchema]& schema,
            const vector[shared_ptr[CChunkedArray]]& columns,
            int64_t num_rows)

        @staticmethod
        shared_ptr[CTable] MakeFromArrays" Make"(
            const shared_ptr[CSchema]& schema,
            const vector[shared_ptr[CArray]]& arrays)

        @staticmethod
        CResult[shared_ptr[CTable]] FromRecordBatchReader(
            CRecordBatchReader *reader)

        @staticmethod
        CResult[shared_ptr[CTable]] FromRecordBatches(
            const shared_ptr[CSchema]& schema,
            const vector[shared_ptr[CRecordBatch]]& batches)

        int num_columns()
        int64_t num_rows()

        c_bool Equals(const CTable& other, c_bool check_metadata)

        shared_ptr[CSchema] schema()
        shared_ptr[CChunkedArray] column(int i)
        shared_ptr[CField] field(int i)

        CResult[shared_ptr[CTable]] AddColumn(
            int i, shared_ptr[CField] field, shared_ptr[CChunkedArray] column)
        CResult[shared_ptr[CTable]] RemoveColumn(int i)
        CResult[shared_ptr[CTable]] SetColumn(
            int i, shared_ptr[CField] field, shared_ptr[CChunkedArray] column)

        vector[c_string] ColumnNames()
        CResult[shared_ptr[CTable]] RenameColumns(const vector[c_string]&)
        CResult[shared_ptr[CTable]] SelectColumns(const vector[int]&)

        CResult[shared_ptr[CTable]] Flatten(CMemoryPool* pool)

        CResult[shared_ptr[CTable]] CombineChunks(CMemoryPool* pool)

        CStatus Validate() const
        CStatus ValidateFull() const

        shared_ptr[CTable] ReplaceSchemaMetadata(
            const shared_ptr[CKeyValueMetadata]& metadata)

        shared_ptr[CTable] Slice(int64_t offset)
        shared_ptr[CTable] Slice(int64_t offset, int64_t length)

    cdef cppclass CRecordBatchReader" arrow::RecordBatchReader":
        shared_ptr[CSchema] schema()
        CStatus Close()
        CResult[CRecordBatchWithMetadata] ReadNext()
        CStatus ReadNext(shared_ptr[CRecordBatch]* batch)
        CResult[shared_ptr[CTable]] ToTable()

    cdef cppclass TableBatchReader(CRecordBatchReader):
        TableBatchReader(const CTable& table)
        TableBatchReader(shared_ptr[CTable] table)
        void set_chunksize(int64_t chunksize)

    cdef cppclass CTensor" arrow::Tensor":
        shared_ptr[CDataType] type()
        shared_ptr[CBuffer] data()

        const vector[int64_t]& shape()
        const vector[int64_t]& strides()
        int64_t size()

        int ndim()
        const vector[c_string]& dim_names()
        const c_string& dim_name(int i)

        c_bool is_mutable()
        c_bool is_contiguous()
        Type type_id()
        c_bool Equals(const CTensor& other)

    cdef cppclass CSparseIndex" arrow::SparseIndex":
        pass

    cdef cppclass CSparseCOOIndex" arrow::SparseCOOIndex":
        c_bool is_canonical()

    cdef cppclass CSparseCOOTensor" arrow::SparseCOOTensor":
        shared_ptr[CDataType] type()
        shared_ptr[CBuffer] data()
        CResult[shared_ptr[CTensor]] ToTensor()

        shared_ptr[CSparseIndex] sparse_index()

        const vector[int64_t]& shape()
        int64_t size()
        int64_t non_zero_length()

        int ndim()
        const vector[c_string]& dim_names()
        const c_string& dim_name(int i)

        c_bool is_mutable()
        Type type_id()
        c_bool Equals(const CSparseCOOTensor& other)

    cdef cppclass CSparseCSRMatrix" arrow::SparseCSRMatrix":
        shared_ptr[CDataType] type()
        shared_ptr[CBuffer] data()
        CResult[shared_ptr[CTensor]] ToTensor()

        const vector[int64_t]& shape()
        int64_t size()
        int64_t non_zero_length()

        int ndim()
        const vector[c_string]& dim_names()
        const c_string& dim_name(int i)

        c_bool is_mutable()
        Type type_id()
        c_bool Equals(const CSparseCSRMatrix& other)

    cdef cppclass CSparseCSCMatrix" arrow::SparseCSCMatrix":
        shared_ptr[CDataType] type()
        shared_ptr[CBuffer] data()
        CResult[shared_ptr[CTensor]] ToTensor()

        const vector[int64_t]& shape()
        int64_t size()
        int64_t non_zero_length()

        int ndim()
        const vector[c_string]& dim_names()
        const c_string& dim_name(int i)

        c_bool is_mutable()
        Type type_id()
        c_bool Equals(const CSparseCSCMatrix& other)

    cdef cppclass CSparseCSFTensor" arrow::SparseCSFTensor":
        shared_ptr[CDataType] type()
        shared_ptr[CBuffer] data()
        CResult[shared_ptr[CTensor]] ToTensor()

        const vector[int64_t]& shape()
        int64_t size()
        int64_t non_zero_length()

        int ndim()
        const vector[c_string]& dim_names()
        const c_string& dim_name(int i)

        c_bool is_mutable()
        Type type_id()
        c_bool Equals(const CSparseCSFTensor& other)

    cdef cppclass CScalar" arrow::Scalar":
        CScalar(shared_ptr[CDataType])

        shared_ptr[CDataType] type
        c_bool is_valid

        c_string ToString() const
        c_bool Equals(const CScalar& other) const
        CStatus Validate() const
        CStatus ValidateFull() const
        CResult[shared_ptr[CScalar]] CastTo(shared_ptr[CDataType] to) const

    cdef cppclass CScalarHash" arrow::Scalar::Hash":
        size_t operator()(const shared_ptr[CScalar]& scalar) const

    cdef cppclass CNullScalar" arrow::NullScalar"(CScalar):
        CNullScalar()

    cdef cppclass CBooleanScalar" arrow::BooleanScalar"(CScalar):
        CBooleanScalar(c_bool value)
        c_bool value

    cdef cppclass CInt8Scalar" arrow::Int8Scalar"(CScalar):
        int8_t value

    cdef cppclass CUInt8Scalar" arrow::UInt8Scalar"(CScalar):
        uint8_t value

    cdef cppclass CInt16Scalar" arrow::Int16Scalar"(CScalar):
        int16_t value

    cdef cppclass CUInt16Scalar" arrow::UInt16Scalar"(CScalar):
        uint16_t value

    cdef cppclass CInt32Scalar" arrow::Int32Scalar"(CScalar):
        int32_t value

    cdef cppclass CUInt32Scalar" arrow::UInt32Scalar"(CScalar):
        uint32_t value

    cdef cppclass CInt64Scalar" arrow::Int64Scalar"(CScalar):
        int64_t value

    cdef cppclass CUInt64Scalar" arrow::UInt64Scalar"(CScalar):
        uint64_t value

    cdef cppclass CHalfFloatScalar" arrow::HalfFloatScalar"(CScalar):
        npy_half value

    cdef cppclass CFloatScalar" arrow::FloatScalar"(CScalar):
        float value

    cdef cppclass CDoubleScalar" arrow::DoubleScalar"(CScalar):
        double value

    cdef cppclass CDecimal128Scalar" arrow::Decimal128Scalar"(CScalar):
        CDecimal128 value

    cdef cppclass CDecimal256Scalar" arrow::Decimal256Scalar"(CScalar):
        CDecimal256 value

    cdef cppclass CDate32Scalar" arrow::Date32Scalar"(CScalar):
        int32_t value

    cdef cppclass CDate64Scalar" arrow::Date64Scalar"(CScalar):
        int64_t value

    cdef cppclass CTime32Scalar" arrow::Time32Scalar"(CScalar):
        int32_t value

    cdef cppclass CTime64Scalar" arrow::Time64Scalar"(CScalar):
        int64_t value

    cdef cppclass CTimestampScalar" arrow::TimestampScalar"(CScalar):
        int64_t value

    cdef cppclass CDurationScalar" arrow::DurationScalar"(CScalar):
        int64_t value

    cdef cppclass CMonthDayNanoIntervalScalar \
            "arrow::MonthDayNanoIntervalScalar"(CScalar):
        pass

    cdef cppclass CBaseBinaryScalar" arrow::BaseBinaryScalar"(CScalar):
        shared_ptr[CBuffer] value

    cdef cppclass CBaseListScalar" arrow::BaseListScalar"(CScalar):
        shared_ptr[CArray] value

    cdef cppclass CListScalar" arrow::ListScalar"(CBaseListScalar):
        pass

    cdef cppclass CMapScalar" arrow::MapScalar"(CListScalar):
        pass

    cdef cppclass CStructScalar" arrow::StructScalar"(CScalar):
        vector[shared_ptr[CScalar]] value
        CResult[shared_ptr[CScalar]] field(CFieldRef ref) const

    cdef cppclass CDictionaryScalarIndexAndDictionary \
            "arrow::DictionaryScalar::ValueType":
        shared_ptr[CScalar] index
        shared_ptr[CArray] dictionary

    cdef cppclass CDictionaryScalar" arrow::DictionaryScalar"(CScalar):
        CDictionaryScalar(CDictionaryScalarIndexAndDictionary value,
                          shared_ptr[CDataType], c_bool is_valid)
        CDictionaryScalarIndexAndDictionary value

        CResult[shared_ptr[CScalar]] GetEncodedValue()

    cdef cppclass CUnionScalar" arrow::UnionScalar"(CScalar):
        int8_t type_code

    cdef cppclass CDenseUnionScalar" arrow::DenseUnionScalar"(CUnionScalar):
        shared_ptr[CScalar] value

    cdef cppclass CSparseUnionScalar" arrow::SparseUnionScalar"(CUnionScalar):
        vector[shared_ptr[CScalar]] value
        int child_id

    cdef cppclass CExtensionScalar" arrow::ExtensionScalar"(CScalar):
        CExtensionScalar(shared_ptr[CScalar] storage,
                         shared_ptr[CDataType], c_bool is_valid)
        shared_ptr[CScalar] value

    shared_ptr[CScalar] MakeScalar[Value](Value value)

    cdef cppclass CConcatenateTablesOptions" arrow::ConcatenateTablesOptions":
        c_bool unify_schemas
        CField.CMergeOptions field_merge_options

        @staticmethod
        CConcatenateTablesOptions Defaults()

    CResult[shared_ptr[CTable]] ConcatenateTables(
        const vector[shared_ptr[CTable]]& tables,
        CConcatenateTablesOptions options,
        CMemoryPool* memory_pool)

    cdef cppclass CDictionaryUnifier" arrow::DictionaryUnifier":
        @staticmethod
        CResult[shared_ptr[CChunkedArray]] UnifyChunkedArray(
            shared_ptr[CChunkedArray] array, CMemoryPool* pool)

        @staticmethod
        CResult[shared_ptr[CTable]] UnifyTable(
            const CTable& table, CMemoryPool* pool)

    shared_ptr[CScalar] MakeNullScalar(shared_ptr[CDataType] type)


cdef extern from "arrow/builder.h" namespace "arrow" nogil:

    cdef cppclass CArrayBuilder" arrow::ArrayBuilder":
        CArrayBuilder(shared_ptr[CDataType], CMemoryPool* pool)

        int64_t length()
        int64_t null_count()
        CStatus AppendNull()
        CStatus Finish(shared_ptr[CArray]* out)
        CStatus Reserve(int64_t additional_capacity)

    cdef cppclass CBooleanBuilder" arrow::BooleanBuilder"(CArrayBuilder):
        CBooleanBuilder(CMemoryPool* pool)
        CStatus Append(const c_bool val)
        CStatus Append(const uint8_t val)

    cdef cppclass CInt8Builder" arrow::Int8Builder"(CArrayBuilder):
        CInt8Builder(CMemoryPool* pool)
        CStatus Append(const int8_t value)

    cdef cppclass CInt16Builder" arrow::Int16Builder"(CArrayBuilder):
        CInt16Builder(CMemoryPool* pool)
        CStatus Append(const int16_t value)

    cdef cppclass CInt32Builder" arrow::Int32Builder"(CArrayBuilder):
        CInt32Builder(CMemoryPool* pool)
        CStatus Append(const int32_t value)

    cdef cppclass CInt64Builder" arrow::Int64Builder"(CArrayBuilder):
        CInt64Builder(CMemoryPool* pool)
        CStatus Append(const int64_t value)

    cdef cppclass CUInt8Builder" arrow::UInt8Builder"(CArrayBuilder):
        CUInt8Builder(CMemoryPool* pool)
        CStatus Append(const uint8_t value)

    cdef cppclass CUInt16Builder" arrow::UInt16Builder"(CArrayBuilder):
        CUInt16Builder(CMemoryPool* pool)
        CStatus Append(const uint16_t value)

    cdef cppclass CUInt32Builder" arrow::UInt32Builder"(CArrayBuilder):
        CUInt32Builder(CMemoryPool* pool)
        CStatus Append(const uint32_t value)

    cdef cppclass CUInt64Builder" arrow::UInt64Builder"(CArrayBuilder):
        CUInt64Builder(CMemoryPool* pool)
        CStatus Append(const uint64_t value)

    cdef cppclass CHalfFloatBuilder" arrow::HalfFloatBuilder"(CArrayBuilder):
        CHalfFloatBuilder(CMemoryPool* pool)

    cdef cppclass CFloatBuilder" arrow::FloatBuilder"(CArrayBuilder):
        CFloatBuilder(CMemoryPool* pool)
        CStatus Append(const float value)

    cdef cppclass CDoubleBuilder" arrow::DoubleBuilder"(CArrayBuilder):
        CDoubleBuilder(CMemoryPool* pool)
        CStatus Append(const double value)

    cdef cppclass CBinaryBuilder" arrow::BinaryBuilder"(CArrayBuilder):
        CArrayBuilder(shared_ptr[CDataType], CMemoryPool* pool)
        CStatus Append(const char* value, int32_t length)

    cdef cppclass CStringBuilder" arrow::StringBuilder"(CBinaryBuilder):
        CStringBuilder(CMemoryPool* pool)

        CStatus Append(const c_string& value)

    cdef cppclass CTimestampBuilder "arrow::TimestampBuilder"(CArrayBuilder):
        CTimestampBuilder(const shared_ptr[CDataType] typ, CMemoryPool* pool)
        CStatus Append(const int64_t value)

    cdef cppclass CDate32Builder "arrow::Date32Builder"(CArrayBuilder):
        CDate32Builder(CMemoryPool* pool)
        CStatus Append(const int32_t value)

    cdef cppclass CDate64Builder "arrow::Date64Builder"(CArrayBuilder):
        CDate64Builder(CMemoryPool* pool)
        CStatus Append(const int64_t value)


# Use typedef to emulate syntax for std::function<void(..)>
ctypedef void CallbackTransform(object, const shared_ptr[CBuffer]& src,
                                shared_ptr[CBuffer]* dest)

ctypedef CResult[shared_ptr[CInputStream]] StreamWrapFunc(
    shared_ptr[CInputStream])


cdef extern from "arrow/util/cancel.h" namespace "arrow" nogil:
    cdef cppclass CStopToken "arrow::StopToken":
        CStatus Poll()
        c_bool IsStopRequested()

    cdef cppclass CStopSource "arrow::StopSource":
        CStopToken token()

    CResult[CStopSource*] SetSignalStopSource()
    void ResetSignalStopSource()

    CStatus RegisterCancellingSignalHandler(vector[int] signals)
    void UnregisterCancellingSignalHandler()


cdef extern from "arrow/io/api.h" namespace "arrow::io" nogil:
    cdef enum FileMode" arrow::io::FileMode::type":
        FileMode_READ" arrow::io::FileMode::READ"
        FileMode_WRITE" arrow::io::FileMode::WRITE"
        FileMode_READWRITE" arrow::io::FileMode::READWRITE"

    cdef enum ObjectType" arrow::io::ObjectType::type":
        ObjectType_FILE" arrow::io::ObjectType::FILE"
        ObjectType_DIRECTORY" arrow::io::ObjectType::DIRECTORY"

    cdef cppclass CIOContext" arrow::io::IOContext":
        CIOContext()
        CIOContext(CStopToken)
        CIOContext(CMemoryPool*)
        CIOContext(CMemoryPool*, CStopToken)

    CIOContext c_default_io_context "arrow::io::default_io_context"()
    int GetIOThreadPoolCapacity()
    CStatus SetIOThreadPoolCapacity(int threads)

    cdef cppclass FileStatistics:
        int64_t size
        ObjectType kind

    cdef cppclass FileInterface:
        CStatus Close()
        CResult[int64_t] Tell()
        FileMode mode()
        c_bool closed()

    cdef cppclass Readable:
        # put overload under a different name to avoid cython bug with multiple
        # layers of inheritance
        CResult[shared_ptr[CBuffer]] ReadBuffer" Read"(int64_t nbytes)
        CResult[int64_t] Read(int64_t nbytes, uint8_t* out)

    cdef cppclass Seekable:
        CStatus Seek(int64_t position)

    cdef cppclass Writable:
        CStatus WriteBuffer" Write"(shared_ptr[CBuffer] data)
        CStatus Write(const uint8_t* data, int64_t nbytes)
        CStatus Flush()

    cdef cppclass COutputStream" arrow::io::OutputStream"(FileInterface,
                                                          Writable):
        pass

    cdef cppclass CInputStream" arrow::io::InputStream"(FileInterface,
                                                        Readable):
        CResult[shared_ptr[const CKeyValueMetadata]] ReadMetadata()

    cdef cppclass CRandomAccessFile" arrow::io::RandomAccessFile"(CInputStream,
                                                                  Seekable):
        CResult[int64_t] GetSize()

        @staticmethod
        CResult[shared_ptr[CInputStream]] GetStream(
            shared_ptr[CRandomAccessFile] file,
            int64_t file_offset,
            int64_t nbytes)

        CResult[int64_t] ReadAt(int64_t position, int64_t nbytes,
                                uint8_t* buffer)
        CResult[shared_ptr[CBuffer]] ReadAt(int64_t position, int64_t nbytes)
        c_bool supports_zero_copy()

    cdef cppclass WritableFile(COutputStream, Seekable):
        CStatus WriteAt(int64_t position, const uint8_t* data,
                        int64_t nbytes)

    cdef cppclass ReadWriteFileInterface(CRandomAccessFile,
                                         WritableFile):
        pass

    cdef cppclass CIOFileSystem" arrow::io::FileSystem":
        CStatus Stat(const c_string& path, FileStatistics* stat)

    cdef cppclass FileOutputStream(COutputStream):
        @staticmethod
        CResult[shared_ptr[COutputStream]] Open(const c_string& path)

        int file_descriptor()

    cdef cppclass ReadableFile(CRandomAccessFile):
        @staticmethod
        CResult[shared_ptr[ReadableFile]] Open(const c_string& path)

        @staticmethod
        CResult[shared_ptr[ReadableFile]] Open(const c_string& path,
                                               CMemoryPool* memory_pool)

        int file_descriptor()

    cdef cppclass CMemoryMappedFile \
            " arrow::io::MemoryMappedFile"(ReadWriteFileInterface):

        @staticmethod
        CResult[shared_ptr[CMemoryMappedFile]] Create(const c_string& path,
                                                      int64_t size)

        @staticmethod
        CResult[shared_ptr[CMemoryMappedFile]] Open(const c_string& path,
                                                    FileMode mode)

        CStatus Resize(int64_t size)

        int file_descriptor()

    cdef cppclass CCompressedInputStream \
            " arrow::io::CompressedInputStream"(CInputStream):
        @staticmethod
        CResult[shared_ptr[CCompressedInputStream]] Make(
            CCodec* codec, shared_ptr[CInputStream] raw)

    cdef cppclass CCompressedOutputStream \
            " arrow::io::CompressedOutputStream"(COutputStream):
        @staticmethod
        CResult[shared_ptr[CCompressedOutputStream]] Make(
            CCodec* codec, shared_ptr[COutputStream] raw)

    cdef cppclass CBufferedInputStream \
            " arrow::io::BufferedInputStream"(CInputStream):

        @staticmethod
        CResult[shared_ptr[CBufferedInputStream]] Create(
            int64_t buffer_size, CMemoryPool* pool,
            shared_ptr[CInputStream] raw)

        CResult[shared_ptr[CInputStream]] Detach()

    cdef cppclass CBufferedOutputStream \
            " arrow::io::BufferedOutputStream"(COutputStream):

        @staticmethod
        CResult[shared_ptr[CBufferedOutputStream]] Create(
            int64_t buffer_size, CMemoryPool* pool,
            shared_ptr[COutputStream] raw)

        CResult[shared_ptr[COutputStream]] Detach()

    cdef cppclass CTransformInputStreamVTable \
            "arrow::py::TransformInputStreamVTable":
        CTransformInputStreamVTable()
        function[CallbackTransform] transform

    shared_ptr[CInputStream] MakeTransformInputStream \
        "arrow::py::MakeTransformInputStream"(
        shared_ptr[CInputStream] wrapped, CTransformInputStreamVTable vtable,
        object method_arg)

    shared_ptr[function[StreamWrapFunc]] MakeStreamTransformFunc \
        "arrow::py::MakeStreamTransformFunc"(
        CTransformInputStreamVTable vtable,
        object method_arg)

    # ----------------------------------------------------------------------
    # HDFS

    CStatus HaveLibHdfs()
    CStatus HaveLibHdfs3()

    cdef enum HdfsDriver" arrow::io::HdfsDriver":
        HdfsDriver_LIBHDFS" arrow::io::HdfsDriver::LIBHDFS"
        HdfsDriver_LIBHDFS3" arrow::io::HdfsDriver::LIBHDFS3"

    cdef cppclass HdfsConnectionConfig:
        c_string host
        int port
        c_string user
        c_string kerb_ticket
        unordered_map[c_string, c_string] extra_conf
        HdfsDriver driver

    cdef cppclass HdfsPathInfo:
        ObjectType kind
        c_string name
        c_string owner
        c_string group
        int32_t last_modified_time
        int32_t last_access_time
        int64_t size
        int16_t replication
        int64_t block_size
        int16_t permissions

    cdef cppclass HdfsReadableFile(CRandomAccessFile):
        pass

    cdef cppclass HdfsOutputStream(COutputStream):
        pass

    cdef cppclass CIOHadoopFileSystem \
            "arrow::io::HadoopFileSystem"(CIOFileSystem):
        @staticmethod
        CStatus Connect(const HdfsConnectionConfig* config,
                        shared_ptr[CIOHadoopFileSystem]* client)

        CStatus MakeDirectory(const c_string& path)

        CStatus Delete(const c_string& path, c_bool recursive)

        CStatus Disconnect()

        c_bool Exists(const c_string& path)

        CStatus Chmod(const c_string& path, int mode)
        CStatus Chown(const c_string& path, const char* owner,
                      const char* group)

        CStatus GetCapacity(int64_t* nbytes)
        CStatus GetUsed(int64_t* nbytes)

        CStatus ListDirectory(const c_string& path,
                              vector[HdfsPathInfo]* listing)

        CStatus GetPathInfo(const c_string& path, HdfsPathInfo* info)

        CStatus Rename(const c_string& src, const c_string& dst)

        CStatus OpenReadable(const c_string& path,
                             shared_ptr[HdfsReadableFile]* handle)

        CStatus OpenWritable(const c_string& path, c_bool append,
                             int32_t buffer_size, int16_t replication,
                             int64_t default_block_size,
                             shared_ptr[HdfsOutputStream]* handle)

    cdef cppclass CBufferReader \
            " arrow::io::BufferReader"(CRandomAccessFile):
        CBufferReader(const shared_ptr[CBuffer]& buffer)
        CBufferReader(const uint8_t* data, int64_t nbytes)

    cdef cppclass CBufferOutputStream \
            " arrow::io::BufferOutputStream"(COutputStream):
        CBufferOutputStream(const shared_ptr[CResizableBuffer]& buffer)

    cdef cppclass CMockOutputStream \
            " arrow::io::MockOutputStream"(COutputStream):
        CMockOutputStream()
        int64_t GetExtentBytesWritten()

    cdef cppclass CFixedSizeBufferWriter \
            " arrow::io::FixedSizeBufferWriter"(WritableFile):
        CFixedSizeBufferWriter(const shared_ptr[CBuffer]& buffer)

        void set_memcopy_threads(int num_threads)
        void set_memcopy_blocksize(int64_t blocksize)
        void set_memcopy_threshold(int64_t threshold)


cdef extern from "arrow/ipc/api.h" namespace "arrow::ipc" nogil:
    cdef enum MessageType" arrow::ipc::MessageType":
        MessageType_SCHEMA" arrow::ipc::MessageType::SCHEMA"
        MessageType_RECORD_BATCH" arrow::ipc::MessageType::RECORD_BATCH"
        MessageType_DICTIONARY_BATCH \
            " arrow::ipc::MessageType::DICTIONARY_BATCH"

    # TODO: use "cpdef enum class" to automatically get a Python wrapper?
    # See
    # https://github.com/cython/cython/commit/2c7c22f51405299a4e247f78edf52957d30cf71d#diff-61c1365c0f761a8137754bb3a73bfbf7
    ctypedef enum CMetadataVersion" arrow::ipc::MetadataVersion":
        CMetadataVersion_V1" arrow::ipc::MetadataVersion::V1"
        CMetadataVersion_V2" arrow::ipc::MetadataVersion::V2"
        CMetadataVersion_V3" arrow::ipc::MetadataVersion::V3"
        CMetadataVersion_V4" arrow::ipc::MetadataVersion::V4"
        CMetadataVersion_V5" arrow::ipc::MetadataVersion::V5"

    cdef cppclass CIpcWriteOptions" arrow::ipc::IpcWriteOptions":
        c_bool allow_64bit
        int max_recursion_depth
        int32_t alignment
        c_bool write_legacy_ipc_format
        CMemoryPool* memory_pool
        CMetadataVersion metadata_version
        shared_ptr[CCodec] codec
        c_bool use_threads
        c_bool emit_dictionary_deltas
        c_bool unify_dictionaries

        CIpcWriteOptions()
        CIpcWriteOptions(CIpcWriteOptions&&)

        @staticmethod
        CIpcWriteOptions Defaults()

    cdef cppclass CIpcReadOptions" arrow::ipc::IpcReadOptions":
        int max_recursion_depth
        CMemoryPool* memory_pool
        vector[int] included_fields
        c_bool use_threads
        c_bool ensure_native_endian

        @staticmethod
        CIpcReadOptions Defaults()

    cdef cppclass CIpcWriteStats" arrow::ipc::WriteStats":
        int64_t num_messages
        int64_t num_record_batches
        int64_t num_dictionary_batches
        int64_t num_dictionary_deltas
        int64_t num_replaced_dictionaries

    cdef cppclass CIpcReadStats" arrow::ipc::ReadStats":
        int64_t num_messages
        int64_t num_record_batches
        int64_t num_dictionary_batches
        int64_t num_dictionary_deltas
        int64_t num_replaced_dictionaries

    cdef cppclass CDictionaryMemo" arrow::ipc::DictionaryMemo":
        pass

    cdef cppclass CIpcPayload" arrow::ipc::IpcPayload":
        MessageType type
        shared_ptr[CBuffer] metadata
        vector[shared_ptr[CBuffer]] body_buffers
        int64_t body_length

    cdef cppclass CMessage" arrow::ipc::Message":
        CResult[unique_ptr[CMessage]] Open(shared_ptr[CBuffer] metadata,
                                           shared_ptr[CBuffer] body)

        shared_ptr[CBuffer] body()

        c_bool Equals(const CMessage& other)

        shared_ptr[CBuffer] metadata()
        CMetadataVersion metadata_version()
        MessageType type()

        CStatus SerializeTo(COutputStream* stream,
                            const CIpcWriteOptions& options,
                            int64_t* output_length)

    c_string FormatMessageType(MessageType type)

    cdef cppclass CMessageReader" arrow::ipc::MessageReader":
        @staticmethod
        unique_ptr[CMessageReader] Open(const shared_ptr[CInputStream]& stream)

        CResult[unique_ptr[CMessage]] ReadNextMessage()

    cdef cppclass CRecordBatchWriter" arrow::ipc::RecordBatchWriter":
        CStatus Close()
        CStatus WriteRecordBatch(const CRecordBatch& batch)
        CStatus WriteRecordBatch(
            const CRecordBatch& batch,
            const shared_ptr[const CKeyValueMetadata]& metadata)
        CStatus WriteTable(const CTable& table, int64_t max_chunksize)

        CIpcWriteStats stats()

    cdef cppclass CRecordBatchStreamReader \
            " arrow::ipc::RecordBatchStreamReader"(CRecordBatchReader):
        @staticmethod
        CResult[shared_ptr[CRecordBatchReader]] Open(
            const shared_ptr[CInputStream], const CIpcReadOptions&)

        @staticmethod
        CResult[shared_ptr[CRecordBatchReader]] Open2" Open"(
            unique_ptr[CMessageReader] message_reader,
            const CIpcReadOptions& options)

        CIpcReadStats stats()

    cdef cppclass CRecordBatchFileReader \
            " arrow::ipc::RecordBatchFileReader":
        @staticmethod
        CResult[shared_ptr[CRecordBatchFileReader]] Open(
            CRandomAccessFile* file,
            const CIpcReadOptions& options)

        @staticmethod
        CResult[shared_ptr[CRecordBatchFileReader]] Open2" Open"(
            CRandomAccessFile* file, int64_t footer_offset,
            const CIpcReadOptions& options)

        shared_ptr[CSchema] schema()

        int num_record_batches()

        CResult[shared_ptr[CRecordBatch]] ReadRecordBatch(int i)

        CResult[CRecordBatchWithMetadata] ReadRecordBatchWithCustomMetadata(int i)

        CIpcReadStats stats()

    CResult[shared_ptr[CRecordBatchWriter]] MakeStreamWriter(
        shared_ptr[COutputStream] sink, const shared_ptr[CSchema]& schema,
        CIpcWriteOptions& options)

    CResult[shared_ptr[CRecordBatchWriter]] MakeFileWriter(
        shared_ptr[COutputStream] sink, const shared_ptr[CSchema]& schema,
        CIpcWriteOptions& options)

    CResult[unique_ptr[CMessage]] ReadMessage(CInputStream* stream,
                                              CMemoryPool* pool)

    CStatus GetRecordBatchSize(const CRecordBatch& batch, int64_t* size)
    CStatus GetTensorSize(const CTensor& tensor, int64_t* size)

    CStatus WriteTensor(const CTensor& tensor, COutputStream* dst,
                        int32_t* metadata_length,
                        int64_t* body_length)

    CResult[shared_ptr[CTensor]] ReadTensor(CInputStream* stream)

    CResult[shared_ptr[CRecordBatch]] ReadRecordBatch(
        const CMessage& message, const shared_ptr[CSchema]& schema,
        CDictionaryMemo* dictionary_memo,
        const CIpcReadOptions& options)

    CResult[shared_ptr[CBuffer]] SerializeSchema(
        const CSchema& schema, CMemoryPool* pool)

    CResult[shared_ptr[CBuffer]] SerializeRecordBatch(
        const CRecordBatch& schema, const CIpcWriteOptions& options)

    CResult[shared_ptr[CSchema]] ReadSchema(const CMessage& message,
                                            CDictionaryMemo* dictionary_memo)

    CResult[shared_ptr[CSchema]] ReadSchema(CInputStream* stream,
                                            CDictionaryMemo* dictionary_memo)

    CResult[shared_ptr[CRecordBatch]] ReadRecordBatch(
        const shared_ptr[CSchema]& schema,
        CDictionaryMemo* dictionary_memo,
        const CIpcReadOptions& options,
        CInputStream* stream)

    CStatus AlignStream(CInputStream* stream, int64_t alignment)
    CStatus AlignStream(COutputStream* stream, int64_t alignment)

    cdef CStatus GetRecordBatchPayload \
        " arrow::ipc::GetRecordBatchPayload"(
            const CRecordBatch& batch,
            const CIpcWriteOptions& options,
            CIpcPayload* out)


cdef extern from "arrow/util/value_parsing.h" namespace "arrow" nogil:
    cdef cppclass CTimestampParser" arrow::TimestampParser":
        const char* kind() const
        const char* format() const

        @staticmethod
        shared_ptr[CTimestampParser] MakeStrptime(c_string format)

        @staticmethod
        shared_ptr[CTimestampParser] MakeISO8601()


cdef extern from "arrow/csv/api.h" namespace "arrow::csv" nogil:

    cdef cppclass CCSVInvalidRow" arrow::csv::InvalidRow":
        int32_t expected_columns
        int32_t actual_columns
        int64_t number
        c_string text

    ctypedef enum CInvalidRowResult" arrow::csv::InvalidRowResult":
        CInvalidRowResult_Error" arrow::csv::InvalidRowResult::Error"
        CInvalidRowResult_Skip" arrow::csv::InvalidRowResult::Skip"

    ctypedef CInvalidRowResult CInvalidRowHandler(const CCSVInvalidRow&)


cdef extern from "arrow/csv/api.h" namespace "arrow::csv" nogil:

    ctypedef enum CQuotingStyle "arrow::csv::QuotingStyle":
        CQuotingStyle_Needed "arrow::csv::QuotingStyle::Needed"
        CQuotingStyle_AllValid "arrow::csv::QuotingStyle::AllValid"
        CQuotingStyle_None "arrow::csv::QuotingStyle::None"

    cdef cppclass CCSVParseOptions" arrow::csv::ParseOptions":
        unsigned char delimiter
        c_bool quoting
        unsigned char quote_char
        c_bool double_quote
        c_bool escaping
        unsigned char escape_char
        c_bool newlines_in_values
        c_bool ignore_empty_lines
        function[CInvalidRowHandler] invalid_row_handler

        CCSVParseOptions()
        CCSVParseOptions(CCSVParseOptions&&)

        @staticmethod
        CCSVParseOptions Defaults()

        CStatus Validate()

    cdef cppclass CCSVConvertOptions" arrow::csv::ConvertOptions":
        c_bool check_utf8
        unordered_map[c_string, shared_ptr[CDataType]] column_types
        vector[c_string] null_values
        vector[c_string] true_values
        vector[c_string] false_values
        c_bool strings_can_be_null
        c_bool quoted_strings_can_be_null
        vector[shared_ptr[CTimestampParser]] timestamp_parsers

        c_bool auto_dict_encode
        int32_t auto_dict_max_cardinality
        unsigned char decimal_point

        vector[c_string] include_columns
        c_bool include_missing_columns

        CCSVConvertOptions()
        CCSVConvertOptions(CCSVConvertOptions&&)

        @staticmethod
        CCSVConvertOptions Defaults()

        CStatus Validate()

    cdef cppclass CCSVReadOptions" arrow::csv::ReadOptions":
        c_bool use_threads
        int32_t block_size
        int32_t skip_rows
        int32_t skip_rows_after_names
        vector[c_string] column_names
        c_bool autogenerate_column_names

        CCSVReadOptions()
        CCSVReadOptions(CCSVReadOptions&&)

        @staticmethod
        CCSVReadOptions Defaults()

        CStatus Validate()

    cdef cppclass CCSVWriteOptions" arrow::csv::WriteOptions":
        c_bool include_header
        int32_t batch_size
        unsigned char delimiter
        CQuotingStyle quoting_style
        CIOContext io_context

        CCSVWriteOptions()
        CCSVWriteOptions(CCSVWriteOptions&&)

        @staticmethod
        CCSVWriteOptions Defaults()

        CStatus Validate()

    cdef cppclass CCSVReader" arrow::csv::TableReader":
        @staticmethod
        CResult[shared_ptr[CCSVReader]] Make(
            CIOContext, shared_ptr[CInputStream],
            CCSVReadOptions, CCSVParseOptions, CCSVConvertOptions)

        CResult[shared_ptr[CTable]] Read()

    cdef cppclass CCSVStreamingReader" arrow::csv::StreamingReader"(
            CRecordBatchReader):
        @staticmethod
        CResult[shared_ptr[CCSVStreamingReader]] Make(
            CIOContext, shared_ptr[CInputStream],
            CCSVReadOptions, CCSVParseOptions, CCSVConvertOptions)

    cdef CStatus WriteCSV(CTable&, CCSVWriteOptions& options, COutputStream*)
    cdef CStatus WriteCSV(
        CRecordBatch&, CCSVWriteOptions& options, COutputStream*)
    cdef CResult[shared_ptr[CRecordBatchWriter]] MakeCSVWriter(
        shared_ptr[COutputStream], shared_ptr[CSchema],
        CCSVWriteOptions& options)


cdef extern from "arrow/json/options.h" nogil:

    ctypedef enum CUnexpectedFieldBehavior \
            "arrow::json::UnexpectedFieldBehavior":
        CUnexpectedFieldBehavior_Ignore \
            "arrow::json::UnexpectedFieldBehavior::Ignore"
        CUnexpectedFieldBehavior_Error \
            "arrow::json::UnexpectedFieldBehavior::Error"
        CUnexpectedFieldBehavior_InferType \
            "arrow::json::UnexpectedFieldBehavior::InferType"

    cdef cppclass CJSONReadOptions" arrow::json::ReadOptions":
        c_bool use_threads
        int32_t block_size

        @staticmethod
        CJSONReadOptions Defaults()

    cdef cppclass CJSONParseOptions" arrow::json::ParseOptions":
        shared_ptr[CSchema] explicit_schema
        c_bool newlines_in_values
        CUnexpectedFieldBehavior unexpected_field_behavior

        @staticmethod
        CJSONParseOptions Defaults()


cdef extern from "arrow/json/reader.h" namespace "arrow::json" nogil:

    cdef cppclass CJSONReader" arrow::json::TableReader":
        @staticmethod
        CResult[shared_ptr[CJSONReader]] Make(
            CMemoryPool*, shared_ptr[CInputStream],
            CJSONReadOptions, CJSONParseOptions)

        CResult[shared_ptr[CTable]] Read()


cdef extern from "arrow/util/thread_pool.h" namespace "arrow::internal" nogil:

    cdef cppclass CExecutor "arrow::internal::Executor":
        pass

    cdef cppclass CThreadPool "arrow::internal::ThreadPool"(CExecutor):
        @staticmethod
        CResult[shared_ptr[CThreadPool]] Make(int threads)

    CThreadPool* GetCpuThreadPool()


cdef extern from "arrow/compute/api.h" namespace "arrow::compute" nogil:

    cdef cppclass CExecContext" arrow::compute::ExecContext":
        CExecContext()
        CExecContext(CMemoryPool* pool)
        CExecContext(CMemoryPool* pool, CExecutor* exc)

        CMemoryPool* memory_pool() const
        CExecutor* executor()

    cdef cppclass CKernelSignature" arrow::compute::KernelSignature":
        c_string ToString() const

    cdef cppclass CKernel" arrow::compute::Kernel":
        shared_ptr[CKernelSignature] signature

    cdef cppclass CArrayKernel" arrow::compute::ArrayKernel"(CKernel):
        pass

    cdef cppclass CScalarKernel" arrow::compute::ScalarKernel"(CArrayKernel):
        pass

    cdef cppclass CVectorKernel" arrow::compute::VectorKernel"(CArrayKernel):
        pass

    cdef cppclass CScalarAggregateKernel \
            " arrow::compute::ScalarAggregateKernel"(CKernel):
        pass

    cdef cppclass CHashAggregateKernel \
            " arrow::compute::HashAggregateKernel"(CKernel):
        pass

    cdef cppclass CArity" arrow::compute::Arity":
        int num_args
        c_bool is_varargs

        CArity()

        CArity(int num_args, c_bool is_varargs)

    cdef enum FunctionKind" arrow::compute::Function::Kind":
        FunctionKind_SCALAR" arrow::compute::Function::SCALAR"
        FunctionKind_VECTOR" arrow::compute::Function::VECTOR"
        FunctionKind_SCALAR_AGGREGATE \
            " arrow::compute::Function::SCALAR_AGGREGATE"
        FunctionKind_HASH_AGGREGATE \
            " arrow::compute::Function::HASH_AGGREGATE"
        FunctionKind_META \
            " arrow::compute::Function::META"

    cdef cppclass CFunctionDoc" arrow::compute::FunctionDoc":
        c_string summary
        c_string description
        vector[c_string] arg_names
        c_string options_class
        c_bool options_required

    cdef cppclass CFunctionOptionsType" arrow::compute::FunctionOptionsType":
        const char* type_name() const

    cdef cppclass CFunctionOptions" arrow::compute::FunctionOptions":
        const CFunctionOptionsType* options_type() const
        const char* type_name() const
        c_bool Equals(const CFunctionOptions& other) const
        c_string ToString() const
        unique_ptr[CFunctionOptions] Copy() const
        CResult[shared_ptr[CBuffer]] Serialize() const

        @staticmethod
        CResult[unique_ptr[CFunctionOptions]] Deserialize(
            const c_string& type_name, const CBuffer& buffer)

    cdef cppclass CFunction" arrow::compute::Function":
        const c_string& name() const
        FunctionKind kind() const
        const CArity& arity() const
        const CFunctionDoc& doc() const
        int num_kernels() const
        CResult[CDatum] Execute(const vector[CDatum]& args,
                                const CFunctionOptions* options,
                                CExecContext* ctx) const
        CResult[CDatum] Execute(const CExecBatch& args,
                                const CFunctionOptions* options,
                                CExecContext* ctx) const

    cdef cppclass CScalarFunction" arrow::compute::ScalarFunction"(CFunction):
        vector[const CScalarKernel*] kernels() const

    cdef cppclass CVectorFunction" arrow::compute::VectorFunction"(CFunction):
        vector[const CVectorKernel*] kernels() const

    cdef cppclass CScalarAggregateFunction \
            " arrow::compute::ScalarAggregateFunction"(CFunction):
        vector[const CScalarAggregateKernel*] kernels() const

    cdef cppclass CHashAggregateFunction \
            " arrow::compute::HashAggregateFunction"(CFunction):
        vector[const CHashAggregateKernel*] kernels() const

    cdef cppclass CMetaFunction" arrow::compute::MetaFunction"(CFunction):
        pass

    cdef cppclass CFunctionRegistry" arrow::compute::FunctionRegistry":
        CResult[shared_ptr[CFunction]] GetFunction(
            const c_string& name) const
        vector[c_string] GetFunctionNames() const
        int num_functions() const

    CFunctionRegistry* GetFunctionRegistry()

    cdef cppclass CElementWiseAggregateOptions \
            "arrow::compute::ElementWiseAggregateOptions"(CFunctionOptions):
        CElementWiseAggregateOptions(c_bool skip_nulls)
        c_bool skip_nulls

    ctypedef enum CRoundMode \
            "arrow::compute::RoundMode":
        CRoundMode_DOWN \
            "arrow::compute::RoundMode::DOWN"
        CRoundMode_UP \
            "arrow::compute::RoundMode::UP"
        CRoundMode_TOWARDS_ZERO \
            "arrow::compute::RoundMode::TOWARDS_ZERO"
        CRoundMode_TOWARDS_INFINITY \
            "arrow::compute::RoundMode::TOWARDS_INFINITY"
        CRoundMode_HALF_DOWN \
            "arrow::compute::RoundMode::HALF_DOWN"
        CRoundMode_HALF_UP \
            "arrow::compute::RoundMode::HALF_UP"
        CRoundMode_HALF_TOWARDS_ZERO \
            "arrow::compute::RoundMode::HALF_TOWARDS_ZERO"
        CRoundMode_HALF_TOWARDS_INFINITY \
            "arrow::compute::RoundMode::HALF_TOWARDS_INFINITY"
        CRoundMode_HALF_TO_EVEN \
            "arrow::compute::RoundMode::HALF_TO_EVEN"
        CRoundMode_HALF_TO_ODD \
            "arrow::compute::RoundMode::HALF_TO_ODD"

    cdef cppclass CRoundOptions \
            "arrow::compute::RoundOptions"(CFunctionOptions):
        CRoundOptions(int64_t ndigits, CRoundMode round_mode)
        int64_t ndigits
        CRoundMode round_mode

    ctypedef enum CCalendarUnit \
            "arrow::compute::CalendarUnit":
        CCalendarUnit_NANOSECOND \
            "arrow::compute::CalendarUnit::NANOSECOND"
        CCalendarUnit_MICROSECOND \
            "arrow::compute::CalendarUnit::MICROSECOND"
        CCalendarUnit_MILLISECOND \
            "arrow::compute::CalendarUnit::MILLISECOND"
        CCalendarUnit_SECOND \
            "arrow::compute::CalendarUnit::SECOND"
        CCalendarUnit_MINUTE \
            "arrow::compute::CalendarUnit::MINUTE"
        CCalendarUnit_HOUR \
            "arrow::compute::CalendarUnit::HOUR"
        CCalendarUnit_DAY \
            "arrow::compute::CalendarUnit::DAY"
        CCalendarUnit_WEEK \
            "arrow::compute::CalendarUnit::WEEK"
        CCalendarUnit_MONTH \
            "arrow::compute::CalendarUnit::MONTH"
        CCalendarUnit_QUARTER \
            "arrow::compute::CalendarUnit::QUARTER"
        CCalendarUnit_YEAR \
            "arrow::compute::CalendarUnit::YEAR"

    cdef cppclass CRoundTemporalOptions \
            "arrow::compute::RoundTemporalOptions"(CFunctionOptions):
        CRoundTemporalOptions(int multiple, CCalendarUnit unit,
                              c_bool week_starts_monday,
                              c_bool ceil_is_strictly_greater,
                              c_bool calendar_based_origin)
        int multiple
        CCalendarUnit unit
        c_bool week_starts_monday
        c_bool ceil_is_strictly_greater
        c_bool calendar_based_origin

    cdef cppclass CRoundToMultipleOptions \
            "arrow::compute::RoundToMultipleOptions"(CFunctionOptions):
        CRoundToMultipleOptions(shared_ptr[CScalar] multiple, CRoundMode round_mode)
        shared_ptr[CScalar] multiple
        CRoundMode round_mode

    cdef enum CJoinNullHandlingBehavior \
            "arrow::compute::JoinOptions::NullHandlingBehavior":
        CJoinNullHandlingBehavior_EMIT_NULL \
            "arrow::compute::JoinOptions::EMIT_NULL"
        CJoinNullHandlingBehavior_SKIP \
            "arrow::compute::JoinOptions::SKIP"
        CJoinNullHandlingBehavior_REPLACE \
            "arrow::compute::JoinOptions::REPLACE"

    cdef cppclass CJoinOptions \
            "arrow::compute::JoinOptions"(CFunctionOptions):
        CJoinOptions(CJoinNullHandlingBehavior null_handling,
                     c_string null_replacement)
        CJoinNullHandlingBehavior null_handling
        c_string null_replacement

    cdef cppclass CMatchSubstringOptions \
            "arrow::compute::MatchSubstringOptions"(CFunctionOptions):
        CMatchSubstringOptions(c_string pattern, c_bool ignore_case)
        c_string pattern
        c_bool ignore_case

    cdef cppclass CTrimOptions \
            "arrow::compute::TrimOptions"(CFunctionOptions):
        CTrimOptions(c_string characters)
        c_string characters

    cdef cppclass CPadOptions \
            "arrow::compute::PadOptions"(CFunctionOptions):
        CPadOptions(int64_t width, c_string padding)
        int64_t width
        c_string padding

    cdef cppclass CSliceOptions \
            "arrow::compute::SliceOptions"(CFunctionOptions):
        CSliceOptions(int64_t start, int64_t stop, int64_t step)
        int64_t start
        int64_t stop
        int64_t step

    cdef cppclass CListSliceOptions \
            "arrow::compute::ListSliceOptions"(CFunctionOptions):
        CListSliceOptions(int64_t start, optional[int64_t] stop,
                          int64_t step,
                          optional[c_bool] return_fixed_size_list)
        int64_t start
        optional[int64_t] stop
        int64_t step
        optional[c_bool] return_fixed_size_list

    cdef cppclass CSplitOptions \
            "arrow::compute::SplitOptions"(CFunctionOptions):
        CSplitOptions(int64_t max_splits, c_bool reverse)
        int64_t max_splits
        c_bool reverse

    cdef cppclass CSplitPatternOptions \
            "arrow::compute::SplitPatternOptions"(CFunctionOptions):
        CSplitPatternOptions(c_string pattern, int64_t max_splits,
                             c_bool reverse)
        int64_t max_splits
        c_bool reverse
        c_string pattern

    cdef cppclass CReplaceSliceOptions \
            "arrow::compute::ReplaceSliceOptions"(CFunctionOptions):
        CReplaceSliceOptions(int64_t start, int64_t stop, c_string replacement)
        int64_t start
        int64_t stop
        c_string replacement

    cdef cppclass CReplaceSubstringOptions \
            "arrow::compute::ReplaceSubstringOptions"(CFunctionOptions):
        CReplaceSubstringOptions(c_string pattern, c_string replacement,
                                 int64_t max_replacements)
        c_string pattern
        c_string replacement
        int64_t max_replacements

    cdef cppclass CExtractRegexOptions \
            "arrow::compute::ExtractRegexOptions"(CFunctionOptions):
        CExtractRegexOptions(c_string pattern)
        c_string pattern

    cdef cppclass CCastOptions" arrow::compute::CastOptions"(CFunctionOptions):
        CCastOptions()
        CCastOptions(c_bool safe)
        CCastOptions(CCastOptions&& options)

        @staticmethod
        CCastOptions Safe()

        @staticmethod
        CCastOptions Unsafe()
        shared_ptr[CDataType] to_type
        c_bool allow_int_overflow
        c_bool allow_time_truncate
        c_bool allow_time_overflow
        c_bool allow_decimal_truncate
        c_bool allow_float_truncate
        c_bool allow_invalid_utf8

    cdef enum CFilterNullSelectionBehavior \
            "arrow::compute::FilterOptions::NullSelectionBehavior":
        CFilterNullSelectionBehavior_DROP \
            "arrow::compute::FilterOptions::DROP"
        CFilterNullSelectionBehavior_EMIT_NULL \
            "arrow::compute::FilterOptions::EMIT_NULL"

    cdef cppclass CFilterOptions \
            " arrow::compute::FilterOptions"(CFunctionOptions):
        CFilterOptions()
        CFilterOptions(CFilterNullSelectionBehavior null_selection_behavior)
        CFilterNullSelectionBehavior null_selection_behavior

    cdef enum CDictionaryEncodeNullEncodingBehavior \
            "arrow::compute::DictionaryEncodeOptions::NullEncodingBehavior":
        CDictionaryEncodeNullEncodingBehavior_ENCODE \
            "arrow::compute::DictionaryEncodeOptions::ENCODE"
        CDictionaryEncodeNullEncodingBehavior_MASK \
            "arrow::compute::DictionaryEncodeOptions::MASK"

    cdef cppclass CDictionaryEncodeOptions \
            "arrow::compute::DictionaryEncodeOptions"(CFunctionOptions):
        CDictionaryEncodeOptions(
            CDictionaryEncodeNullEncodingBehavior null_encoding)
        CDictionaryEncodeNullEncodingBehavior null_encoding

    cdef cppclass CTakeOptions \
            " arrow::compute::TakeOptions"(CFunctionOptions):
        CTakeOptions(c_bool boundscheck)
        c_bool boundscheck

    cdef cppclass CStrptimeOptions \
            "arrow::compute::StrptimeOptions"(CFunctionOptions):
        CStrptimeOptions(c_string format, TimeUnit unit, c_bool raise_error)
        c_string format
        TimeUnit unit
        c_bool raise_error

    cdef cppclass CStrftimeOptions \
            "arrow::compute::StrftimeOptions"(CFunctionOptions):
        CStrftimeOptions(c_string format, c_string locale)
        c_string format
        c_string locale

    cdef cppclass CDayOfWeekOptions \
            "arrow::compute::DayOfWeekOptions"(CFunctionOptions):
        CDayOfWeekOptions(c_bool count_from_zero, uint32_t week_start)
        c_bool count_from_zero
        uint32_t week_start

    cdef enum CAssumeTimezoneAmbiguous \
            "arrow::compute::AssumeTimezoneOptions::Ambiguous":
        CAssumeTimezoneAmbiguous_AMBIGUOUS_RAISE \
            "arrow::compute::AssumeTimezoneOptions::AMBIGUOUS_RAISE"
        CAssumeTimezoneAmbiguous_AMBIGUOUS_EARLIEST \
            "arrow::compute::AssumeTimezoneOptions::AMBIGUOUS_EARLIEST"
        CAssumeTimezoneAmbiguous_AMBIGUOUS_LATEST \
            "arrow::compute::AssumeTimezoneOptions::AMBIGUOUS_LATEST"

    cdef enum CAssumeTimezoneNonexistent \
            "arrow::compute::AssumeTimezoneOptions::Nonexistent":
        CAssumeTimezoneNonexistent_NONEXISTENT_RAISE \
            "arrow::compute::AssumeTimezoneOptions::NONEXISTENT_RAISE"
        CAssumeTimezoneNonexistent_NONEXISTENT_EARLIEST \
            "arrow::compute::AssumeTimezoneOptions::NONEXISTENT_EARLIEST"
        CAssumeTimezoneNonexistent_NONEXISTENT_LATEST \
            "arrow::compute::AssumeTimezoneOptions::NONEXISTENT_LATEST"

    cdef cppclass CAssumeTimezoneOptions \
            "arrow::compute::AssumeTimezoneOptions"(CFunctionOptions):
        CAssumeTimezoneOptions(c_string timezone,
                               CAssumeTimezoneAmbiguous ambiguous,
                               CAssumeTimezoneNonexistent nonexistent)
        c_string timezone
        CAssumeTimezoneAmbiguous ambiguous
        CAssumeTimezoneNonexistent nonexistent

    cdef cppclass CWeekOptions \
            "arrow::compute::WeekOptions"(CFunctionOptions):
        CWeekOptions(c_bool week_starts_monday, c_bool count_from_zero,
                     c_bool first_week_is_fully_in_year)
        c_bool week_starts_monday
        c_bool count_from_zero
        c_bool first_week_is_fully_in_year

    cdef cppclass CNullOptions \
            "arrow::compute::NullOptions"(CFunctionOptions):
        CNullOptions(c_bool nan_is_null)
        c_bool nan_is_null

    cdef cppclass CVarianceOptions \
            "arrow::compute::VarianceOptions"(CFunctionOptions):
        CVarianceOptions(int ddof, c_bool skip_nulls, uint32_t min_count)
        int ddof
        c_bool skip_nulls
        uint32_t min_count

    cdef cppclass CScalarAggregateOptions \
            "arrow::compute::ScalarAggregateOptions"(CFunctionOptions):
        CScalarAggregateOptions(c_bool skip_nulls, uint32_t min_count)
        c_bool skip_nulls
        uint32_t min_count

    cdef enum CCountMode "arrow::compute::CountOptions::CountMode":
        CCountMode_ONLY_VALID "arrow::compute::CountOptions::ONLY_VALID"
        CCountMode_ONLY_NULL "arrow::compute::CountOptions::ONLY_NULL"
        CCountMode_ALL "arrow::compute::CountOptions::ALL"

    cdef cppclass CCountOptions \
            "arrow::compute::CountOptions"(CFunctionOptions):
        CCountOptions(CCountMode mode)
        CCountMode mode

    cdef cppclass CModeOptions \
            "arrow::compute::ModeOptions"(CFunctionOptions):
        CModeOptions(int64_t n, c_bool skip_nulls, uint32_t min_count)
        int64_t n
        c_bool skip_nulls
        uint32_t min_count

    cdef cppclass CIndexOptions \
            "arrow::compute::IndexOptions"(CFunctionOptions):
        CIndexOptions(shared_ptr[CScalar] value)
        shared_ptr[CScalar] value

    cdef enum CMapLookupOccurrence \
            "arrow::compute::MapLookupOptions::Occurrence":
        CMapLookupOccurrence_ALL "arrow::compute::MapLookupOptions::ALL"
        CMapLookupOccurrence_FIRST "arrow::compute::MapLookupOptions::FIRST"
        CMapLookupOccurrence_LAST "arrow::compute::MapLookupOptions::LAST"

    cdef cppclass CMapLookupOptions \
            "arrow::compute::MapLookupOptions"(CFunctionOptions):
        CMapLookupOptions(shared_ptr[CScalar] query_key,
                          CMapLookupOccurrence occurrence)
        CMapLookupOccurrence occurrence
        shared_ptr[CScalar] query_key

    cdef cppclass CMakeStructOptions \
            "arrow::compute::MakeStructOptions"(CFunctionOptions):
        CMakeStructOptions(vector[c_string] n,
                           vector[c_bool] r,
                           vector[shared_ptr[const CKeyValueMetadata]] m)
        CMakeStructOptions(vector[c_string] n)
        vector[c_string] field_names
        vector[c_bool] field_nullability
        vector[shared_ptr[const CKeyValueMetadata]] field_metadata

    cdef cppclass CStructFieldOptions \
            "arrow::compute::StructFieldOptions"(CFunctionOptions):
        CStructFieldOptions(vector[int] indices)
        CStructFieldOptions(CFieldRef field_ref)
        vector[int] indices
        CFieldRef field_ref

    ctypedef enum CSortOrder" arrow::compute::SortOrder":
        CSortOrder_Ascending \
            "arrow::compute::SortOrder::Ascending"
        CSortOrder_Descending \
            "arrow::compute::SortOrder::Descending"

    ctypedef enum CNullPlacement" arrow::compute::NullPlacement":
        CNullPlacement_AtStart \
            "arrow::compute::NullPlacement::AtStart"
        CNullPlacement_AtEnd \
            "arrow::compute::NullPlacement::AtEnd"

    cdef cppclass CPartitionNthOptions \
            "arrow::compute::PartitionNthOptions"(CFunctionOptions):
        CPartitionNthOptions(int64_t pivot, CNullPlacement)
        int64_t pivot
        CNullPlacement null_placement

    cdef cppclass CCumulativeSumOptions \
            "arrow::compute::CumulativeSumOptions"(CFunctionOptions):
        CCumulativeSumOptions(shared_ptr[CScalar] start, c_bool skip_nulls)
        shared_ptr[CScalar] start
        c_bool skip_nulls

    cdef cppclass CArraySortOptions \
            "arrow::compute::ArraySortOptions"(CFunctionOptions):
        CArraySortOptions(CSortOrder, CNullPlacement)
        CSortOrder order
        CNullPlacement null_placement

    cdef cppclass CSortKey" arrow::compute::SortKey":
        CSortKey(c_string name, CSortOrder order)
        c_string name
        CSortOrder order

    cdef cppclass CSortOptions \
            "arrow::compute::SortOptions"(CFunctionOptions):
        CSortOptions(vector[CSortKey] sort_keys, CNullPlacement)
        vector[CSortKey] sort_keys
        CNullPlacement null_placement

    cdef cppclass CSelectKOptions \
            "arrow::compute::SelectKOptions"(CFunctionOptions):
        CSelectKOptions(int64_t k, vector[CSortKey] sort_keys)
        int64_t k
        vector[CSortKey] sort_keys

    cdef enum CQuantileInterp \
            "arrow::compute::QuantileOptions::Interpolation":
        CQuantileInterp_LINEAR "arrow::compute::QuantileOptions::LINEAR"
        CQuantileInterp_LOWER "arrow::compute::QuantileOptions::LOWER"
        CQuantileInterp_HIGHER "arrow::compute::QuantileOptions::HIGHER"
        CQuantileInterp_NEAREST "arrow::compute::QuantileOptions::NEAREST"
        CQuantileInterp_MIDPOINT "arrow::compute::QuantileOptions::MIDPOINT"

    cdef cppclass CQuantileOptions \
            "arrow::compute::QuantileOptions"(CFunctionOptions):
        CQuantileOptions(vector[double] q, CQuantileInterp interpolation,
                         c_bool skip_nulls, uint32_t min_count)
        vector[double] q
        CQuantileInterp interpolation
        c_bool skip_nulls
        uint32_t min_count

    cdef cppclass CTDigestOptions \
            "arrow::compute::TDigestOptions"(CFunctionOptions):
        CTDigestOptions(vector[double] q,
                        uint32_t delta, uint32_t buffer_size,
                        c_bool skip_nulls, uint32_t min_count)
        vector[double] q
        uint32_t delta
        uint32_t buffer_size
        c_bool skip_nulls
        uint32_t min_count

    cdef enum CUtf8NormalizeForm \
            "arrow::compute::Utf8NormalizeOptions::Form":
        CUtf8NormalizeForm_NFC "arrow::compute::Utf8NormalizeOptions::NFC"
        CUtf8NormalizeForm_NFKC "arrow::compute::Utf8NormalizeOptions::NFKC"
        CUtf8NormalizeForm_NFD "arrow::compute::Utf8NormalizeOptions::NFD"
        CUtf8NormalizeForm_NFKD "arrow::compute::Utf8NormalizeOptions::NFKD"

    cdef cppclass CUtf8NormalizeOptions \
            "arrow::compute::Utf8NormalizeOptions"(CFunctionOptions):
        CUtf8NormalizeOptions(CUtf8NormalizeForm form)
        CUtf8NormalizeForm form

    cdef cppclass CSetLookupOptions \
            "arrow::compute::SetLookupOptions"(CFunctionOptions):
        CSetLookupOptions(CDatum value_set, c_bool skip_nulls)
        CDatum value_set
        c_bool skip_nulls

    cdef cppclass CRandomOptions \
            "arrow::compute::RandomOptions"(CFunctionOptions):
        CRandomOptions(CRandomOptions)

        @staticmethod
        CRandomOptions FromSystemRandom()

        @staticmethod
        CRandomOptions FromSeed(uint64_t seed)

    cdef enum CRankOptionsTiebreaker \
            "arrow::compute::RankOptions::Tiebreaker":
        CRankOptionsTiebreaker_Min "arrow::compute::RankOptions::Min"
        CRankOptionsTiebreaker_Max "arrow::compute::RankOptions::Max"
        CRankOptionsTiebreaker_First "arrow::compute::RankOptions::First"
        CRankOptionsTiebreaker_Dense "arrow::compute::RankOptions::Dense"

    cdef cppclass CRankOptions \
            "arrow::compute::RankOptions"(CFunctionOptions):
        CRankOptions(vector[CSortKey] sort_keys, CNullPlacement,
                     CRankOptionsTiebreaker tiebreaker)
        vector[CSortKey] sort_keys
        CNullPlacement null_placement
        CRankOptionsTiebreaker tiebreaker

    cdef enum DatumType" arrow::Datum::type":
        DatumType_NONE" arrow::Datum::NONE"
        DatumType_SCALAR" arrow::Datum::SCALAR"
        DatumType_ARRAY" arrow::Datum::ARRAY"
        DatumType_CHUNKED_ARRAY" arrow::Datum::CHUNKED_ARRAY"
        DatumType_RECORD_BATCH" arrow::Datum::RECORD_BATCH"
        DatumType_TABLE" arrow::Datum::TABLE"
        DatumType_COLLECTION" arrow::Datum::COLLECTION"

    cdef cppclass CDatum" arrow::Datum":
        CDatum()
        CDatum(const shared_ptr[CArray]& value)
        CDatum(const shared_ptr[CChunkedArray]& value)
        CDatum(const shared_ptr[CScalar]& value)
        CDatum(const shared_ptr[CRecordBatch]& value)
        CDatum(const shared_ptr[CTable]& value)

        DatumType kind() const
        c_string ToString() const

        const shared_ptr[CArrayData]& array() const
        const shared_ptr[CChunkedArray]& chunked_array() const
        const shared_ptr[CRecordBatch]& record_batch() const
        const shared_ptr[CTable]& table() const
        const shared_ptr[CScalar]& scalar() const

    cdef c_string ToString(DatumType kind)

cdef extern from * namespace "arrow::compute":
    # inlined from compute/function_internal.h to avoid exposing
    # implementation details
    """
    #include "arrow/compute/function.h"
    namespace arrow {
    namespace compute {
    namespace internal {
    Result<std::unique_ptr<FunctionOptions>> DeserializeFunctionOptions(
        const Buffer& buffer);
    } //  namespace internal
    } //  namespace compute
    } //  namespace arrow
    """
    CResult[unique_ptr[CFunctionOptions]] DeserializeFunctionOptions \
        " arrow::compute::internal::DeserializeFunctionOptions"(
            const CBuffer& buffer)


cdef extern from "arrow/compute/exec/aggregate.h" namespace \
        "arrow::compute::internal" nogil:
    cdef cppclass CAggregate "arrow::compute::Aggregate":
        c_string function
        shared_ptr[CFunctionOptions] options

    CResult[CDatum] GroupBy(const vector[CDatum]& arguments,
                            const vector[CDatum]& keys,
                            const vector[CAggregate]& aggregates)


cdef extern from * namespace "arrow::compute":
    # inlined from expression_internal.h to avoid
    # proliferation of #include <unordered_map>
    """
    #include <unordered_map>

    #include "arrow/type.h"
    #include "arrow/datum.h"

    namespace arrow {
    namespace compute {
    struct KnownFieldValues {
      std::unordered_map<FieldRef, Datum, FieldRef::Hash> map;
    };
    } //  namespace compute
    } //  namespace arrow
    """
    cdef struct CKnownFieldValues "arrow::compute::KnownFieldValues":
        unordered_map[CFieldRef, CDatum, CFieldRefHash] map

cdef extern from "arrow/compute/exec/expression.h" \
        namespace "arrow::compute" nogil:

    cdef cppclass CExpression "arrow::compute::Expression":
        c_bool Equals(const CExpression& other) const
        c_string ToString() const
        CResult[CExpression] Bind(const CSchema&)
        const CFieldRef* field_ref() const

    cdef CExpression CMakeScalarExpression \
        "arrow::compute::literal"(shared_ptr[CScalar] value)

    cdef CExpression CMakeFieldExpression \
        "arrow::compute::field_ref"(CFieldRef)

    cdef CExpression CMakeFieldExpressionByIndex \
        "arrow::compute::field_ref"(int idx)

    cdef CExpression CMakeCallExpression \
        "arrow::compute::call"(c_string function,
                               vector[CExpression] arguments,
                               shared_ptr[CFunctionOptions] options)

    cdef CResult[shared_ptr[CBuffer]] CSerializeExpression \
        "arrow::compute::Serialize"(const CExpression&)

    cdef CResult[CExpression] CDeserializeExpression \
        "arrow::compute::Deserialize"(shared_ptr[CBuffer])

    cdef CResult[CKnownFieldValues] \
        CExtractKnownFieldValues "arrow::compute::ExtractKnownFieldValues"(
            const CExpression& partition_expression)


cdef extern from "arrow/compute/exec/options.h" namespace "arrow::compute" nogil:
    cdef enum CJoinType "arrow::compute::JoinType":
        CJoinType_LEFT_SEMI "arrow::compute::JoinType::LEFT_SEMI"
        CJoinType_RIGHT_SEMI "arrow::compute::JoinType::RIGHT_SEMI"
        CJoinType_LEFT_ANTI "arrow::compute::JoinType::LEFT_ANTI"
        CJoinType_RIGHT_ANTI "arrow::compute::JoinType::RIGHT_ANTI"
        CJoinType_INNER "arrow::compute::JoinType::INNER"
        CJoinType_LEFT_OUTER "arrow::compute::JoinType::LEFT_OUTER"
        CJoinType_RIGHT_OUTER "arrow::compute::JoinType::RIGHT_OUTER"
        CJoinType_FULL_OUTER "arrow::compute::JoinType::FULL_OUTER"

    cdef cppclass CAsyncExecBatchGenerator "arrow::compute::AsyncExecBatchGenerator":
        pass

    cdef cppclass CExecNodeOptions "arrow::compute::ExecNodeOptions":
        pass

    cdef cppclass CSourceNodeOptions "arrow::compute::SourceNodeOptions"(CExecNodeOptions):
        pass

    cdef cppclass CTableSourceNodeOptions "arrow::compute::TableSourceNodeOptions"(CExecNodeOptions):
        CTableSourceNodeOptions(shared_ptr[CTable] table, int64_t max_batch_size)

    cdef cppclass CSinkNodeOptions "arrow::compute::SinkNodeOptions"(CExecNodeOptions):
        pass

    cdef cppclass CFilterNodeOptions "arrow::compute::FilterNodeOptions"(CExecNodeOptions):
        CFilterNodeOptions(CExpression)

    cdef cppclass CProjectNodeOptions "arrow::compute::ProjectNodeOptions"(CExecNodeOptions):
        CProjectNodeOptions(vector[CExpression] expressions)
        CProjectNodeOptions(vector[CExpression] expressions,
                            vector[c_string] names)

    cdef cppclass COrderBySinkNodeOptions "arrow::compute::OrderBySinkNodeOptions"(CExecNodeOptions):
        COrderBySinkNodeOptions(vector[CSortOptions] options,
                                CAsyncExecBatchGenerator generator)

    cdef cppclass CHashJoinNodeOptions "arrow::compute::HashJoinNodeOptions"(CExecNodeOptions):
        CHashJoinNodeOptions(CJoinType, vector[CFieldRef] in_left_keys,
                             vector[CFieldRef] in_right_keys)
        CHashJoinNodeOptions(CJoinType, vector[CFieldRef] in_left_keys,
                             vector[CFieldRef] in_right_keys,
                             CExpression filter,
                             c_string output_suffix_for_left,
                             c_string output_suffix_for_right)
        CHashJoinNodeOptions(CJoinType join_type,
                             vector[CFieldRef] left_keys,
                             vector[CFieldRef] right_keys,
                             vector[CFieldRef] left_output,
                             vector[CFieldRef] right_output,
                             CExpression filter,
                             c_string output_suffix_for_left,
                             c_string output_suffix_for_right)


cdef extern from "arrow/compute/exec/exec_plan.h" namespace "arrow::compute" nogil:
    cdef cppclass CDeclaration "arrow::compute::Declaration":
        cppclass Input:
            Input(CExecNode*)
            Input(CDeclaration)

        c_string label
        vector[Input] inputs

        CDeclaration()
        CDeclaration(c_string factory_name, CExecNodeOptions options)
        CDeclaration(c_string factory_name, vector[Input] inputs, shared_ptr[CExecNodeOptions] options)

        @staticmethod
        CDeclaration Sequence(vector[CDeclaration] decls)

        CResult[CExecNode*] AddToPlan(CExecPlan* plan) const

    cdef cppclass CExecPlan "arrow::compute::ExecPlan":
        @staticmethod
        CResult[shared_ptr[CExecPlan]] Make(CExecContext* exec_context)

        CStatus StartProducing()
        CStatus Validate()
        CStatus StopProducing()

        CFuture_Void finished()

        vector[CExecNode*] sinks() const
        vector[CExecNode*] sources() const

    cdef cppclass CExecNode "arrow::compute::ExecNode":
        const vector[CExecNode*]& inputs() const
        const shared_ptr[CSchema]& output_schema() const

    cdef cppclass CExecBatch "arrow::compute::ExecBatch":
        vector[CDatum] values
        int64_t length

    shared_ptr[CRecordBatchReader] MakeGeneratorReader(
        shared_ptr[CSchema] schema,
        CAsyncExecBatchGenerator gen,
        CMemoryPool* memory_pool
    )
    CResult[CExecNode*] MakeExecNode(c_string factory_name, CExecPlan* plan,
                                     vector[CExecNode*] inputs,
                                     const CExecNodeOptions& options)


cdef extern from "arrow/extension_type.h" namespace "arrow":
    cdef cppclass CExtensionTypeRegistry" arrow::ExtensionTypeRegistry":
        @staticmethod
        shared_ptr[CExtensionTypeRegistry] GetGlobalRegistry()

    cdef cppclass CExtensionType" arrow::ExtensionType"(CDataType):
        c_string extension_name()
        shared_ptr[CDataType] storage_type()

        @staticmethod
        shared_ptr[CArray] WrapArray(shared_ptr[CDataType] ext_type,
                                     shared_ptr[CArray] storage)

        @staticmethod
        shared_ptr[CChunkedArray] WrapArray(shared_ptr[CDataType] ext_type,
                                            shared_ptr[CChunkedArray] storage)

    cdef cppclass CExtensionArray" arrow::ExtensionArray"(CArray):
        CExtensionArray(shared_ptr[CDataType], shared_ptr[CArray] storage)

        shared_ptr[CArray] storage()


cdef extern from "arrow/util/compression.h" namespace "arrow" nogil:
    cdef enum CCompressionType" arrow::Compression::type":
        CCompressionType_UNCOMPRESSED" arrow::Compression::UNCOMPRESSED"
        CCompressionType_SNAPPY" arrow::Compression::SNAPPY"
        CCompressionType_GZIP" arrow::Compression::GZIP"
        CCompressionType_BROTLI" arrow::Compression::BROTLI"
        CCompressionType_ZSTD" arrow::Compression::ZSTD"
        CCompressionType_LZ4" arrow::Compression::LZ4"
        CCompressionType_LZ4_FRAME" arrow::Compression::LZ4_FRAME"
        CCompressionType_BZ2" arrow::Compression::BZ2"

    cdef cppclass CCodec" arrow::util::Codec":
        @staticmethod
        CResult[unique_ptr[CCodec]] Create(CCompressionType codec)

        @staticmethod
        CResult[unique_ptr[CCodec]] CreateWithLevel" Create"(
            CCompressionType codec,
            int compression_level)

        @staticmethod
        c_bool SupportsCompressionLevel(CCompressionType codec)

        @staticmethod
        CResult[int] MinimumCompressionLevel(CCompressionType codec)

        @staticmethod
        CResult[int] MaximumCompressionLevel(CCompressionType codec)

        @staticmethod
        CResult[int] DefaultCompressionLevel(CCompressionType codec)

        @staticmethod
        c_bool IsAvailable(CCompressionType codec)

        CResult[int64_t] Decompress(int64_t input_len, const uint8_t* input,
                                    int64_t output_len,
                                    uint8_t* output_buffer)
        CResult[int64_t] Compress(int64_t input_len, const uint8_t* input,
                                  int64_t output_buffer_len,
                                  uint8_t* output_buffer)
        c_string name() const
        int compression_level() const
        int64_t MaxCompressedLen(int64_t input_len, const uint8_t* input)


cdef extern from "arrow/util/io_util.h" namespace "arrow::internal" nogil:
    int ErrnoFromStatus(CStatus status)
    int WinErrorFromStatus(CStatus status)
    int SignalFromStatus(CStatus status)

    CStatus SendSignal(int signum)
    CStatus SendSignalToThread(int signum, uint64_t thread_id)


cdef extern from "arrow/util/iterator.h" namespace "arrow" nogil:
    cdef cppclass CIterator" arrow::Iterator"[T]:
        CResult[T] Next()
        CStatus Visit[Visitor](Visitor&& visitor)
        cppclass RangeIterator:
            CResult[T] operator*()
            RangeIterator& operator++()
            c_bool operator!=(RangeIterator) const
        RangeIterator begin()
        RangeIterator end()
    CIterator[T] MakeVectorIterator[T](vector[T] v)

cdef extern from "arrow/util/thread_pool.h" namespace "arrow" nogil:
    int GetCpuThreadPoolCapacity()
    CStatus SetCpuThreadPoolCapacity(int threads)

cdef extern from "arrow/array/concatenate.h" namespace "arrow" nogil:
    CResult[shared_ptr[CArray]] Concatenate(
        const vector[shared_ptr[CArray]]& arrays,
        CMemoryPool* pool)

cdef extern from "arrow/c/abi.h":
    cdef struct ArrowSchema:
        pass

    cdef struct ArrowArray:
        pass

    cdef struct ArrowArrayStream:
        pass

cdef extern from "arrow/c/bridge.h" namespace "arrow" nogil:
    CStatus ExportType(CDataType&, ArrowSchema* out)
    CResult[shared_ptr[CDataType]] ImportType(ArrowSchema*)

    CStatus ExportField(CField&, ArrowSchema* out)
    CResult[shared_ptr[CField]] ImportField(ArrowSchema*)

    CStatus ExportSchema(CSchema&, ArrowSchema* out)
    CResult[shared_ptr[CSchema]] ImportSchema(ArrowSchema*)

    CStatus ExportArray(CArray&, ArrowArray* out)
    CStatus ExportArray(CArray&, ArrowArray* out, ArrowSchema* out_schema)
    CResult[shared_ptr[CArray]] ImportArray(ArrowArray*,
                                            shared_ptr[CDataType])
    CResult[shared_ptr[CArray]] ImportArray(ArrowArray*, ArrowSchema*)

    CStatus ExportRecordBatch(CRecordBatch&, ArrowArray* out)
    CStatus ExportRecordBatch(CRecordBatch&, ArrowArray* out,
                              ArrowSchema* out_schema)
    CResult[shared_ptr[CRecordBatch]] ImportRecordBatch(ArrowArray*,
                                                        shared_ptr[CSchema])
    CResult[shared_ptr[CRecordBatch]] ImportRecordBatch(ArrowArray*,
                                                        ArrowSchema*)

    CStatus ExportRecordBatchReader(shared_ptr[CRecordBatchReader],
                                    ArrowArrayStream*)
    CResult[shared_ptr[CRecordBatchReader]] ImportRecordBatchReader(
        ArrowArrayStream*)


cdef extern from "arrow/util/byte_size.h" namespace "arrow::util" nogil:
    CResult[int64_t] ReferencedBufferSize(const CArray& array_data)
    CResult[int64_t] ReferencedBufferSize(const CRecordBatch& record_batch)
    CResult[int64_t] ReferencedBufferSize(const CChunkedArray& chunked_array)
    CResult[int64_t] ReferencedBufferSize(const CTable& table)
    int64_t TotalBufferSize(const CArray& array)
    int64_t TotalBufferSize(const CChunkedArray& array)
    int64_t TotalBufferSize(const CRecordBatch& record_batch)
    int64_t TotalBufferSize(const CTable& table)

ctypedef PyObject* CallbackUdf(object user_function, const CScalarUdfContext& context, object inputs)

cdef extern from "arrow/python/udf.h" namespace "arrow::py":
    cdef cppclass CScalarUdfContext" arrow::py::ScalarUdfContext":
        CMemoryPool *pool
        int64_t batch_length

    cdef cppclass CScalarUdfOptions" arrow::py::ScalarUdfOptions":
        c_string func_name
        CArity arity
        CFunctionDoc func_doc
        vector[shared_ptr[CDataType]] input_types
        shared_ptr[CDataType] output_type

    CStatus RegisterScalarFunction(PyObject* function,
                                   function[CallbackUdf] wrapper, const CScalarUdfOptions& options)

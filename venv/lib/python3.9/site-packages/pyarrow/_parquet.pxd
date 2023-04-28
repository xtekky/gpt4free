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
# cython: language_level = 3

from pyarrow.includes.common cimport *
from pyarrow.includes.libarrow cimport (CChunkedArray, CScalar, CSchema, CStatus,
                                        CTable, CMemoryPool, CBuffer,
                                        CKeyValueMetadata,
                                        CRandomAccessFile, COutputStream,
                                        TimeUnit, CRecordBatchReader)
from pyarrow.lib cimport _Weakrefable


cdef extern from "parquet/api/schema.h" namespace "parquet::schema" nogil:
    cdef cppclass Node:
        pass

    cdef cppclass GroupNode(Node):
        pass

    cdef cppclass PrimitiveNode(Node):
        pass

    cdef cppclass ColumnPath:
        c_string ToDotString()
        vector[c_string] ToDotVector()


cdef extern from "parquet/api/schema.h" namespace "parquet" nogil:
    enum ParquetType" parquet::Type::type":
        ParquetType_BOOLEAN" parquet::Type::BOOLEAN"
        ParquetType_INT32" parquet::Type::INT32"
        ParquetType_INT64" parquet::Type::INT64"
        ParquetType_INT96" parquet::Type::INT96"
        ParquetType_FLOAT" parquet::Type::FLOAT"
        ParquetType_DOUBLE" parquet::Type::DOUBLE"
        ParquetType_BYTE_ARRAY" parquet::Type::BYTE_ARRAY"
        ParquetType_FIXED_LEN_BYTE_ARRAY" parquet::Type::FIXED_LEN_BYTE_ARRAY"

    enum ParquetLogicalTypeId" parquet::LogicalType::Type::type":
        ParquetLogicalType_UNDEFINED" parquet::LogicalType::Type::UNDEFINED"
        ParquetLogicalType_STRING" parquet::LogicalType::Type::STRING"
        ParquetLogicalType_MAP" parquet::LogicalType::Type::MAP"
        ParquetLogicalType_LIST" parquet::LogicalType::Type::LIST"
        ParquetLogicalType_ENUM" parquet::LogicalType::Type::ENUM"
        ParquetLogicalType_DECIMAL" parquet::LogicalType::Type::DECIMAL"
        ParquetLogicalType_DATE" parquet::LogicalType::Type::DATE"
        ParquetLogicalType_TIME" parquet::LogicalType::Type::TIME"
        ParquetLogicalType_TIMESTAMP" parquet::LogicalType::Type::TIMESTAMP"
        ParquetLogicalType_INT" parquet::LogicalType::Type::INT"
        ParquetLogicalType_JSON" parquet::LogicalType::Type::JSON"
        ParquetLogicalType_BSON" parquet::LogicalType::Type::BSON"
        ParquetLogicalType_UUID" parquet::LogicalType::Type::UUID"
        ParquetLogicalType_NONE" parquet::LogicalType::Type::NONE"

    enum ParquetTimeUnit" parquet::LogicalType::TimeUnit::unit":
        ParquetTimeUnit_UNKNOWN" parquet::LogicalType::TimeUnit::UNKNOWN"
        ParquetTimeUnit_MILLIS" parquet::LogicalType::TimeUnit::MILLIS"
        ParquetTimeUnit_MICROS" parquet::LogicalType::TimeUnit::MICROS"
        ParquetTimeUnit_NANOS" parquet::LogicalType::TimeUnit::NANOS"

    enum ParquetConvertedType" parquet::ConvertedType::type":
        ParquetConvertedType_NONE" parquet::ConvertedType::NONE"
        ParquetConvertedType_UTF8" parquet::ConvertedType::UTF8"
        ParquetConvertedType_MAP" parquet::ConvertedType::MAP"
        ParquetConvertedType_MAP_KEY_VALUE \
            " parquet::ConvertedType::MAP_KEY_VALUE"
        ParquetConvertedType_LIST" parquet::ConvertedType::LIST"
        ParquetConvertedType_ENUM" parquet::ConvertedType::ENUM"
        ParquetConvertedType_DECIMAL" parquet::ConvertedType::DECIMAL"
        ParquetConvertedType_DATE" parquet::ConvertedType::DATE"
        ParquetConvertedType_TIME_MILLIS" parquet::ConvertedType::TIME_MILLIS"
        ParquetConvertedType_TIME_MICROS" parquet::ConvertedType::TIME_MICROS"
        ParquetConvertedType_TIMESTAMP_MILLIS \
            " parquet::ConvertedType::TIMESTAMP_MILLIS"
        ParquetConvertedType_TIMESTAMP_MICROS \
            " parquet::ConvertedType::TIMESTAMP_MICROS"
        ParquetConvertedType_UINT_8" parquet::ConvertedType::UINT_8"
        ParquetConvertedType_UINT_16" parquet::ConvertedType::UINT_16"
        ParquetConvertedType_UINT_32" parquet::ConvertedType::UINT_32"
        ParquetConvertedType_UINT_64" parquet::ConvertedType::UINT_64"
        ParquetConvertedType_INT_8" parquet::ConvertedType::INT_8"
        ParquetConvertedType_INT_16" parquet::ConvertedType::INT_16"
        ParquetConvertedType_INT_32" parquet::ConvertedType::INT_32"
        ParquetConvertedType_INT_64" parquet::ConvertedType::INT_64"
        ParquetConvertedType_JSON" parquet::ConvertedType::JSON"
        ParquetConvertedType_BSON" parquet::ConvertedType::BSON"
        ParquetConvertedType_INTERVAL" parquet::ConvertedType::INTERVAL"

    enum ParquetRepetition" parquet::Repetition::type":
        ParquetRepetition_REQUIRED" parquet::REPETITION::REQUIRED"
        ParquetRepetition_OPTIONAL" parquet::REPETITION::OPTIONAL"
        ParquetRepetition_REPEATED" parquet::REPETITION::REPEATED"

    enum ParquetEncoding" parquet::Encoding::type":
        ParquetEncoding_PLAIN" parquet::Encoding::PLAIN"
        ParquetEncoding_PLAIN_DICTIONARY" parquet::Encoding::PLAIN_DICTIONARY"
        ParquetEncoding_RLE" parquet::Encoding::RLE"
        ParquetEncoding_BIT_PACKED" parquet::Encoding::BIT_PACKED"
        ParquetEncoding_DELTA_BINARY_PACKED \
            " parquet::Encoding::DELTA_BINARY_PACKED"
        ParquetEncoding_DELTA_LENGTH_BYTE_ARRAY \
            " parquet::Encoding::DELTA_LENGTH_BYTE_ARRAY"
        ParquetEncoding_DELTA_BYTE_ARRAY" parquet::Encoding::DELTA_BYTE_ARRAY"
        ParquetEncoding_RLE_DICTIONARY" parquet::Encoding::RLE_DICTIONARY"
        ParquetEncoding_BYTE_STREAM_SPLIT \
            " parquet::Encoding::BYTE_STREAM_SPLIT"

    enum ParquetCompression" parquet::Compression::type":
        ParquetCompression_UNCOMPRESSED" parquet::Compression::UNCOMPRESSED"
        ParquetCompression_SNAPPY" parquet::Compression::SNAPPY"
        ParquetCompression_GZIP" parquet::Compression::GZIP"
        ParquetCompression_LZO" parquet::Compression::LZO"
        ParquetCompression_BROTLI" parquet::Compression::BROTLI"
        ParquetCompression_LZ4" parquet::Compression::LZ4"
        ParquetCompression_ZSTD" parquet::Compression::ZSTD"

    enum ParquetVersion" parquet::ParquetVersion::type":
        ParquetVersion_V1" parquet::ParquetVersion::PARQUET_1_0"
        ParquetVersion_V2_0" parquet::ParquetVersion::PARQUET_2_0"
        ParquetVersion_V2_4" parquet::ParquetVersion::PARQUET_2_4"
        ParquetVersion_V2_6" parquet::ParquetVersion::PARQUET_2_6"

    enum ParquetSortOrder" parquet::SortOrder::type":
        ParquetSortOrder_SIGNED" parquet::SortOrder::SIGNED"
        ParquetSortOrder_UNSIGNED" parquet::SortOrder::UNSIGNED"
        ParquetSortOrder_UNKNOWN" parquet::SortOrder::UNKNOWN"

    cdef cppclass CParquetLogicalType" parquet::LogicalType":
        c_string ToString() const
        c_string ToJSON() const
        ParquetLogicalTypeId type() const

    cdef cppclass CParquetDecimalType \
            " parquet::DecimalLogicalType"(CParquetLogicalType):
        int32_t precision() const
        int32_t scale() const

    cdef cppclass CParquetIntType \
            " parquet::IntLogicalType"(CParquetLogicalType):
        int bit_width() const
        c_bool is_signed() const

    cdef cppclass CParquetTimeType \
            " parquet::TimeLogicalType"(CParquetLogicalType):
        c_bool is_adjusted_to_utc() const
        ParquetTimeUnit time_unit() const

    cdef cppclass CParquetTimestampType \
            " parquet::TimestampLogicalType"(CParquetLogicalType):
        c_bool is_adjusted_to_utc() const
        ParquetTimeUnit time_unit() const

    cdef cppclass ColumnDescriptor" parquet::ColumnDescriptor":
        c_bool Equals(const ColumnDescriptor& other)

        shared_ptr[ColumnPath] path()
        int16_t max_definition_level()
        int16_t max_repetition_level()

        ParquetType physical_type()
        const shared_ptr[const CParquetLogicalType]& logical_type()
        ParquetConvertedType converted_type()
        const c_string& name()
        int type_length()
        int type_precision()
        int type_scale()

    cdef cppclass SchemaDescriptor:
        const ColumnDescriptor* Column(int i)
        shared_ptr[Node] schema()
        GroupNode* group()
        c_bool Equals(const SchemaDescriptor& other)
        c_string ToString()
        int num_columns()

    cdef c_string FormatStatValue(ParquetType parquet_type, c_string val)

    enum ParquetCipher" parquet::ParquetCipher::type":
        ParquetCipher_AES_GCM_V1" parquet::ParquetCipher::AES_GCM_V1"
        ParquetCipher_AES_GCM_CTR_V1" parquet::ParquetCipher::AES_GCM_CTR_V1"

    struct AadMetadata:
        c_string aad_prefix
        c_string aad_file_unique
        c_bool supply_aad_prefix

    struct EncryptionAlgorithm:
        ParquetCipher algorithm
        AadMetadata aad

cdef extern from "parquet/api/reader.h" namespace "parquet" nogil:
    cdef cppclass ColumnReader:
        pass

    cdef cppclass BoolReader(ColumnReader):
        pass

    cdef cppclass Int32Reader(ColumnReader):
        pass

    cdef cppclass Int64Reader(ColumnReader):
        pass

    cdef cppclass Int96Reader(ColumnReader):
        pass

    cdef cppclass FloatReader(ColumnReader):
        pass

    cdef cppclass DoubleReader(ColumnReader):
        pass

    cdef cppclass ByteArrayReader(ColumnReader):
        pass

    cdef cppclass RowGroupReader:
        pass

    cdef cppclass CEncodedStatistics" parquet::EncodedStatistics":
        const c_string& max() const
        const c_string& min() const
        int64_t null_count
        int64_t distinct_count
        bint has_min
        bint has_max
        bint has_null_count
        bint has_distinct_count

    cdef cppclass ParquetByteArray" parquet::ByteArray":
        uint32_t len
        const uint8_t* ptr

    cdef cppclass ParquetFLBA" parquet::FLBA":
        const uint8_t* ptr

    cdef cppclass CStatistics" parquet::Statistics":
        int64_t null_count() const
        int64_t distinct_count() const
        int64_t num_values() const
        bint HasMinMax()
        bint HasNullCount()
        bint HasDistinctCount()
        c_bool Equals(const CStatistics&) const
        void Reset()
        c_string EncodeMin()
        c_string EncodeMax()
        CEncodedStatistics Encode()
        void SetComparator()
        ParquetType physical_type() const
        const ColumnDescriptor* descr() const

    cdef cppclass CBoolStatistics" parquet::BoolStatistics"(CStatistics):
        c_bool min()
        c_bool max()

    cdef cppclass CInt32Statistics" parquet::Int32Statistics"(CStatistics):
        int32_t min()
        int32_t max()

    cdef cppclass CInt64Statistics" parquet::Int64Statistics"(CStatistics):
        int64_t min()
        int64_t max()

    cdef cppclass CFloatStatistics" parquet::FloatStatistics"(CStatistics):
        float min()
        float max()

    cdef cppclass CDoubleStatistics" parquet::DoubleStatistics"(CStatistics):
        double min()
        double max()

    cdef cppclass CByteArrayStatistics \
            " parquet::ByteArrayStatistics"(CStatistics):
        ParquetByteArray min()
        ParquetByteArray max()

    cdef cppclass CFLBAStatistics" parquet::FLBAStatistics"(CStatistics):
        ParquetFLBA min()
        ParquetFLBA max()

    cdef cppclass CColumnCryptoMetaData" parquet::ColumnCryptoMetaData":
        shared_ptr[ColumnPath] path_in_schema() const
        c_bool encrypted_with_footer_key() const
        const c_string& key_metadata() const

    cdef cppclass CColumnChunkMetaData" parquet::ColumnChunkMetaData":
        int64_t file_offset() const
        const c_string& file_path() const

        c_bool is_metadata_set() const
        ParquetType type() const
        int64_t num_values() const
        shared_ptr[ColumnPath] path_in_schema() const
        bint is_stats_set() const
        shared_ptr[CStatistics] statistics() const
        ParquetCompression compression() const
        const vector[ParquetEncoding]& encodings() const
        c_bool Equals(const CColumnChunkMetaData&) const

        int64_t has_dictionary_page() const
        int64_t dictionary_page_offset() const
        int64_t data_page_offset() const
        int64_t index_page_offset() const
        int64_t total_compressed_size() const
        int64_t total_uncompressed_size() const
        unique_ptr[CColumnCryptoMetaData] crypto_metadata() const

    cdef cppclass CRowGroupMetaData" parquet::RowGroupMetaData":
        c_bool Equals(const CRowGroupMetaData&) const
        int num_columns()
        int64_t num_rows()
        int64_t total_byte_size()
        unique_ptr[CColumnChunkMetaData] ColumnChunk(int i) const

    cdef cppclass CFileMetaData" parquet::FileMetaData":
        c_bool Equals(const CFileMetaData&) const
        uint32_t size()
        int num_columns()
        int64_t num_rows()
        int num_row_groups()
        ParquetVersion version()
        const c_string created_by()
        int num_schema_elements()

        void set_file_path(const c_string& path)
        void AppendRowGroups(const CFileMetaData& other) except +

        unique_ptr[CRowGroupMetaData] RowGroup(int i)
        const SchemaDescriptor* schema()
        shared_ptr[const CKeyValueMetadata] key_value_metadata() const
        void WriteTo(COutputStream* dst) const

        inline c_bool is_encryption_algorithm_set() const
        inline EncryptionAlgorithm encryption_algorithm() const
        inline const c_string& footer_signing_key_metadata() const

    cdef shared_ptr[CFileMetaData] CFileMetaData_Make \
        " parquet::FileMetaData::Make"(const void* serialized_metadata,
                                       uint32_t* metadata_len)

    cdef cppclass CReaderProperties" parquet::ReaderProperties":
        c_bool is_buffered_stream_enabled() const
        void enable_buffered_stream()
        void disable_buffered_stream()

        void set_buffer_size(int64_t buf_size)
        int64_t buffer_size() const

        void set_thrift_string_size_limit(int32_t size)
        int32_t thrift_string_size_limit() const

        void set_thrift_container_size_limit(int32_t size)
        int32_t thrift_container_size_limit() const

        void file_decryption_properties(shared_ptr[CFileDecryptionProperties]
                                        decryption)
        shared_ptr[CFileDecryptionProperties] file_decryption_properties() \
            const

    CReaderProperties default_reader_properties()

    cdef cppclass ArrowReaderProperties:
        ArrowReaderProperties()
        void set_read_dictionary(int column_index, c_bool read_dict)
        c_bool read_dictionary()
        void set_batch_size(int64_t batch_size)
        int64_t batch_size()
        void set_pre_buffer(c_bool pre_buffer)
        c_bool pre_buffer() const
        void set_coerce_int96_timestamp_unit(TimeUnit unit)
        TimeUnit coerce_int96_timestamp_unit() const

    ArrowReaderProperties default_arrow_reader_properties()

    cdef cppclass ParquetFileReader:
        shared_ptr[CFileMetaData] metadata()


cdef extern from "parquet/api/writer.h" namespace "parquet" nogil:
    cdef cppclass WriterProperties:
        cppclass Builder:
            Builder* data_page_version(ParquetDataPageVersion version)
            Builder* version(ParquetVersion version)
            Builder* compression(ParquetCompression codec)
            Builder* compression(const c_string& path,
                                 ParquetCompression codec)
            Builder* compression_level(int compression_level)
            Builder* compression_level(const c_string& path,
                                       int compression_level)
            Builder* encryption(
                shared_ptr[CFileEncryptionProperties]
                file_encryption_properties)
            Builder* disable_dictionary()
            Builder* enable_dictionary()
            Builder* enable_dictionary(const c_string& path)
            Builder* disable_statistics()
            Builder* enable_statistics()
            Builder* enable_statistics(const c_string& path)
            Builder* data_pagesize(int64_t size)
            Builder* encoding(ParquetEncoding encoding)
            Builder* encoding(const c_string& path,
                              ParquetEncoding encoding)
            Builder* write_batch_size(int64_t batch_size)
            Builder* dictionary_pagesize_limit(int64_t dictionary_pagesize_limit)
            shared_ptr[WriterProperties] build()

    cdef cppclass ArrowWriterProperties:
        cppclass Builder:
            Builder()
            Builder* disable_deprecated_int96_timestamps()
            Builder* enable_deprecated_int96_timestamps()
            Builder* coerce_timestamps(TimeUnit unit)
            Builder* allow_truncated_timestamps()
            Builder* disallow_truncated_timestamps()
            Builder* store_schema()
            Builder* enable_compliant_nested_types()
            Builder* disable_compliant_nested_types()
            Builder* set_engine_version(ArrowWriterEngineVersion version)
            shared_ptr[ArrowWriterProperties] build()
        c_bool support_deprecated_int96_timestamps()


cdef extern from "parquet/arrow/reader.h" namespace "parquet::arrow" nogil:
    cdef cppclass FileReader:
        FileReader(CMemoryPool* pool, unique_ptr[ParquetFileReader] reader)

        CStatus GetSchema(shared_ptr[CSchema]* out)

        CStatus ReadColumn(int i, shared_ptr[CChunkedArray]* out)
        CStatus ReadSchemaField(int i, shared_ptr[CChunkedArray]* out)

        int num_row_groups()
        CStatus ReadRowGroup(int i, shared_ptr[CTable]* out)
        CStatus ReadRowGroup(int i, const vector[int]& column_indices,
                             shared_ptr[CTable]* out)

        CStatus ReadRowGroups(const vector[int]& row_groups,
                              shared_ptr[CTable]* out)
        CStatus ReadRowGroups(const vector[int]& row_groups,
                              const vector[int]& column_indices,
                              shared_ptr[CTable]* out)

        CStatus GetRecordBatchReader(const vector[int]& row_group_indices,
                                     const vector[int]& column_indices,
                                     unique_ptr[CRecordBatchReader]* out)
        CStatus GetRecordBatchReader(const vector[int]& row_group_indices,
                                     unique_ptr[CRecordBatchReader]* out)

        CStatus ReadTable(shared_ptr[CTable]* out)
        CStatus ReadTable(const vector[int]& column_indices,
                          shared_ptr[CTable]* out)

        CStatus ScanContents(vector[int] columns, int32_t column_batch_size,
                             int64_t* num_rows)

        const ParquetFileReader* parquet_reader()

        void set_use_threads(c_bool use_threads)

        void set_batch_size(int64_t batch_size)

    cdef cppclass FileReaderBuilder:
        FileReaderBuilder()
        CStatus Open(const shared_ptr[CRandomAccessFile]& file,
                     const CReaderProperties& properties,
                     const shared_ptr[CFileMetaData]& metadata)

        ParquetFileReader* raw_reader()
        FileReaderBuilder* memory_pool(CMemoryPool*)
        FileReaderBuilder* properties(const ArrowReaderProperties&)
        CStatus Build(unique_ptr[FileReader]* out)

    CStatus FromParquetSchema(
        const SchemaDescriptor* parquet_schema,
        const ArrowReaderProperties& properties,
        const shared_ptr[const CKeyValueMetadata]& key_value_metadata,
        shared_ptr[CSchema]* out)

    CStatus StatisticsAsScalars(const CStatistics& Statistics,
                                shared_ptr[CScalar]* min,
                                shared_ptr[CScalar]* max)

cdef extern from "parquet/arrow/schema.h" namespace "parquet::arrow" nogil:

    CStatus ToParquetSchema(
        const CSchema* arrow_schema,
        const ArrowReaderProperties& properties,
        const shared_ptr[const CKeyValueMetadata]& key_value_metadata,
        shared_ptr[SchemaDescriptor]* out)


cdef extern from "parquet/properties.h" namespace "parquet" nogil:
    cdef enum ArrowWriterEngineVersion:
        V1 "parquet::ArrowWriterProperties::V1",
        V2 "parquet::ArrowWriterProperties::V2"

    cdef cppclass ParquetDataPageVersion:
        pass

    cdef ParquetDataPageVersion ParquetDataPageVersion_V1 \
        " parquet::ParquetDataPageVersion::V1"
    cdef ParquetDataPageVersion ParquetDataPageVersion_V2 \
        " parquet::ParquetDataPageVersion::V2"

cdef extern from "parquet/arrow/writer.h" namespace "parquet::arrow" nogil:
    cdef cppclass FileWriter:

        @staticmethod
        CResult[unique_ptr[FileWriter]] Open(const CSchema& schema, CMemoryPool* pool,
                                             const shared_ptr[COutputStream]& sink,
                                             const shared_ptr[WriterProperties]& properties,
                                             const shared_ptr[ArrowWriterProperties]& arrow_properties)

        CStatus WriteTable(const CTable& table, int64_t chunk_size)
        CStatus NewRowGroup(int64_t chunk_size)
        CStatus Close()

        const shared_ptr[CFileMetaData] metadata() const

    CStatus WriteMetaDataFile(
        const CFileMetaData& file_metadata,
        const COutputStream* sink)

cdef class FileEncryptionProperties:
    """File-level encryption properties for the low-level API"""
    cdef:
        shared_ptr[CFileEncryptionProperties] properties

    @staticmethod
    cdef inline FileEncryptionProperties wrap(
            shared_ptr[CFileEncryptionProperties] properties):

        result = FileEncryptionProperties()
        result.properties = properties
        return result

    cdef inline shared_ptr[CFileEncryptionProperties] unwrap(self):
        return self.properties

cdef shared_ptr[WriterProperties] _create_writer_properties(
    use_dictionary=*,
    compression=*,
    version=*,
    write_statistics=*,
    data_page_size=*,
    compression_level=*,
    use_byte_stream_split=*,
    column_encoding=*,
    data_page_version=*,
    FileEncryptionProperties encryption_properties=*,
    write_batch_size=*,
    dictionary_pagesize_limit=*) except *


cdef shared_ptr[ArrowWriterProperties] _create_arrow_writer_properties(
    use_deprecated_int96_timestamps=*,
    coerce_timestamps=*,
    allow_truncated_timestamps=*,
    writer_engine_version=*,
    use_compliant_nested_type=*,
    store_schema=*) except *

cdef class ParquetSchema(_Weakrefable):
    cdef:
        FileMetaData parent  # the FileMetaData owning the SchemaDescriptor
        const SchemaDescriptor* schema

cdef class FileMetaData(_Weakrefable):
    cdef:
        shared_ptr[CFileMetaData] sp_metadata
        CFileMetaData* _metadata
        ParquetSchema _schema

    cdef inline init(self, const shared_ptr[CFileMetaData]& metadata):
        self.sp_metadata = metadata
        self._metadata = metadata.get()

cdef class RowGroupMetaData(_Weakrefable):
    cdef:
        int index  # for pickling support
        unique_ptr[CRowGroupMetaData] up_metadata
        CRowGroupMetaData* metadata
        FileMetaData parent

cdef class ColumnChunkMetaData(_Weakrefable):
    cdef:
        unique_ptr[CColumnChunkMetaData] up_metadata
        CColumnChunkMetaData* metadata
        RowGroupMetaData parent

    cdef inline init(self, RowGroupMetaData parent, int i):
        self.up_metadata = parent.metadata.ColumnChunk(i)
        self.metadata = self.up_metadata.get()
        self.parent = parent

cdef class Statistics(_Weakrefable):
    cdef:
        shared_ptr[CStatistics] statistics
        ColumnChunkMetaData parent

    cdef inline init(self, const shared_ptr[CStatistics]& statistics,
                     ColumnChunkMetaData parent):
        self.statistics = statistics
        self.parent = parent

cdef extern from "parquet/encryption/encryption.h" namespace "parquet" nogil:
    cdef cppclass CFileDecryptionProperties\
            " parquet::FileDecryptionProperties":
        pass

    cdef cppclass CFileEncryptionProperties\
            " parquet::FileEncryptionProperties":
        pass

cdef class FileDecryptionProperties:
    """File-level decryption properties for the low-level API"""
    cdef:
        shared_ptr[CFileDecryptionProperties] properties

    @staticmethod
    cdef inline FileDecryptionProperties wrap(
            shared_ptr[CFileDecryptionProperties] properties):

        result = FileDecryptionProperties()
        result.properties = properties
        return result

    cdef inline shared_ptr[CFileDecryptionProperties] unwrap(self):
        return self.properties

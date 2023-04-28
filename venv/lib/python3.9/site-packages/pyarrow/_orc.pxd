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

from libcpp cimport bool as c_bool
from libc.string cimport const_char
from libcpp.vector cimport vector as std_vector
from pyarrow.includes.common cimport *
from pyarrow.includes.libarrow cimport (CArray, CSchema, CStatus,
                                        CResult, CTable, CMemoryPool,
                                        CKeyValueMetadata,
                                        CRecordBatch,
                                        CTable, CCompressionType,
                                        CRandomAccessFile, COutputStream,
                                        TimeUnit)

cdef extern from "arrow/adapters/orc/options.h" \
        namespace "arrow::adapters::orc" nogil:
    cdef enum CompressionStrategy \
            " arrow::adapters::orc::CompressionStrategy":
        _CompressionStrategy_SPEED \
            " arrow::adapters::orc::CompressionStrategy::kSpeed"
        _CompressionStrategy_COMPRESSION \
            " arrow::adapters::orc::CompressionStrategy::kCompression"

    cdef enum WriterId" arrow::adapters::orc::WriterId":
        _WriterId_ORC_JAVA_WRITER" arrow::adapters::orc::WriterId::kOrcJava"
        _WriterId_ORC_CPP_WRITER" arrow::adapters::orc::WriterId::kOrcCpp"
        _WriterId_PRESTO_WRITER" arrow::adapters::orc::WriterId::kPresto"
        _WriterId_SCRITCHLEY_GO \
            " arrow::adapters::orc::WriterId::kScritchleyGo"
        _WriterId_TRINO_WRITER" arrow::adapters::orc::WriterId::kTrino"
        _WriterId_UNKNOWN_WRITER" arrow::adapters::orc::WriterId::kUnknown"

    cdef enum WriterVersion" arrow::adapters::orc::WriterVersion":
        _WriterVersion_ORIGINAL \
            " arrow::adapters::orc::WriterVersion::kOriginal"
        _WriterVersion_HIVE_8732 \
            " arrow::adapters::orc::WriterVersion::kHive8732"
        _WriterVersion_HIVE_4243 \
            " arrow::adapters::orc::WriterVersion::kHive4243"
        _WriterVersion_HIVE_12055 \
            " arrow::adapters::orc::WriterVersion::kHive12055"
        _WriterVersion_HIVE_13083 \
            " arrow::adapters::orc::WriterVersion::kHive13083"
        _WriterVersion_ORC_101" arrow::adapters::orc::WriterVersion::kOrc101"
        _WriterVersion_ORC_135" arrow::adapters::orc::WriterVersion::kOrc135"
        _WriterVersion_ORC_517" arrow::adapters::orc::WriterVersion::kOrc517"
        _WriterVersion_ORC_203" arrow::adapters::orc::WriterVersion::kOrc203"
        _WriterVersion_ORC_14" arrow::adapters::orc::WriterVersion::kOrc14"
        _WriterVersion_MAX" arrow::adapters::orc::WriterVersion::kMax"

    cdef cppclass FileVersion" arrow::adapters::orc::FileVersion":
        FileVersion(uint32_t major_version, uint32_t minor_version)
        uint32_t major_version()
        uint32_t minor_version()
        c_string ToString()

    cdef struct WriteOptions" arrow::adapters::orc::WriteOptions":
        int64_t batch_size
        FileVersion file_version
        int64_t stripe_size
        CCompressionType compression
        int64_t compression_block_size
        CompressionStrategy compression_strategy
        int64_t row_index_stride
        double padding_tolerance
        double dictionary_key_size_threshold
        std_vector[int64_t] bloom_filter_columns
        double bloom_filter_fpp


cdef extern from "arrow/adapters/orc/adapter.h" \
        namespace "arrow::adapters::orc" nogil:

    cdef cppclass ORCFileReader:
        @staticmethod
        CResult[unique_ptr[ORCFileReader]] Open(
            const shared_ptr[CRandomAccessFile]& file,
            CMemoryPool* pool)

        CResult[shared_ptr[const CKeyValueMetadata]] ReadMetadata()

        CResult[shared_ptr[CSchema]] ReadSchema()

        CResult[shared_ptr[CRecordBatch]] ReadStripe(int64_t stripe)
        CResult[shared_ptr[CRecordBatch]] ReadStripe(
            int64_t stripe, std_vector[c_string])

        CResult[shared_ptr[CTable]] Read()
        CResult[shared_ptr[CTable]] Read(std_vector[c_string])

        int64_t NumberOfStripes()
        int64_t NumberOfRows()
        FileVersion GetFileVersion()
        c_string GetSoftwareVersion()
        CResult[CCompressionType] GetCompression()
        int64_t GetCompressionSize()
        int64_t GetRowIndexStride()
        WriterId GetWriterId()
        int32_t GetWriterIdValue()
        WriterVersion GetWriterVersion()
        int64_t GetNumberOfStripeStatistics()
        int64_t GetContentLength()
        int64_t GetStripeStatisticsLength()
        int64_t GetFileFooterLength()
        int64_t GetFilePostscriptLength()
        int64_t GetFileLength()
        c_string GetSerializedFileTail()

    cdef cppclass ORCFileWriter:
        @staticmethod
        CResult[unique_ptr[ORCFileWriter]] Open(
            COutputStream* output_stream, const WriteOptions& writer_options)

        CStatus Write(const CTable& table)

        CStatus Close()

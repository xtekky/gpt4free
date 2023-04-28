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

from pyarrow.includes.libarrow_dataset cimport *
from pyarrow._parquet cimport *


cdef extern from "arrow/dataset/api.h" namespace "arrow::dataset" nogil:

    cdef cppclass CParquetFileWriter \
            "arrow::dataset::ParquetFileWriter"(CFileWriter):
        const shared_ptr[FileWriter]& parquet_writer() const

    cdef cppclass CParquetFileWriteOptions \
            "arrow::dataset::ParquetFileWriteOptions"(CFileWriteOptions):
        shared_ptr[WriterProperties] writer_properties
        shared_ptr[ArrowWriterProperties] arrow_writer_properties

    cdef cppclass CParquetFileFragment "arrow::dataset::ParquetFileFragment"(
            CFileFragment):
        const vector[int]& row_groups() const
        shared_ptr[CFileMetaData] metadata() const
        CResult[vector[shared_ptr[CFragment]]] SplitByRowGroup(
            CExpression predicate)
        CResult[shared_ptr[CFragment]] SubsetWithFilter "Subset"(
            CExpression predicate)
        CResult[shared_ptr[CFragment]] SubsetWithIds "Subset"(
            vector[int] row_group_ids)
        CStatus EnsureCompleteMetadata()

    cdef cppclass CParquetFileFormatReaderOptions \
            "arrow::dataset::ParquetFileFormat::ReaderOptions":
        unordered_set[c_string] dict_columns
        TimeUnit coerce_int96_timestamp_unit

    cdef cppclass CParquetFileFormat "arrow::dataset::ParquetFileFormat"(
            CFileFormat):
        CParquetFileFormatReaderOptions reader_options
        CResult[shared_ptr[CFileFragment]] MakeFragment(
            CFileSource source,
            CExpression partition_expression,
            shared_ptr[CSchema] physical_schema,
            vector[int] row_groups)

    cdef cppclass CParquetFragmentScanOptions \
            "arrow::dataset::ParquetFragmentScanOptions"(CFragmentScanOptions):
        shared_ptr[CReaderProperties] reader_properties
        shared_ptr[ArrowReaderProperties] arrow_reader_properties

    cdef cppclass CParquetFactoryOptions \
            "arrow::dataset::ParquetFactoryOptions":
        CPartitioningOrFactory partitioning
        c_string partition_base_dir
        c_bool validate_column_chunk_paths

    cdef cppclass CParquetDatasetFactory \
            "arrow::dataset::ParquetDatasetFactory"(CDatasetFactory):
        @staticmethod
        CResult[shared_ptr[CDatasetFactory]] MakeFromMetaDataPath "Make"(
            const c_string& metadata_path,
            shared_ptr[CFileSystem] filesystem,
            shared_ptr[CParquetFileFormat] format,
            CParquetFactoryOptions options
        )

        @staticmethod
        CResult[shared_ptr[CDatasetFactory]] MakeFromMetaDataSource "Make"(
            const CFileSource& metadata_path,
            const c_string& base_path,
            shared_ptr[CFileSystem] filesystem,
            shared_ptr[CParquetFileFormat] format,
            CParquetFactoryOptions options
        )

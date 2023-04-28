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

# cython: profile=False
# distutils: language = c++

from cython.operator cimport dereference as deref
from libcpp.vector cimport vector as std_vector
from libcpp.utility cimport move
from pyarrow.includes.common cimport *
from pyarrow.includes.libarrow cimport *
from pyarrow.lib cimport (check_status, _Weakrefable,
                          MemoryPool, maybe_unbox_memory_pool,
                          Schema, pyarrow_wrap_schema,
                          KeyValueMetadata,
                          pyarrow_wrap_batch,
                          RecordBatch,
                          Table,
                          pyarrow_wrap_table,
                          pyarrow_unwrap_schema,
                          pyarrow_wrap_metadata,
                          pyarrow_unwrap_table,
                          get_reader,
                          get_writer)
from pyarrow.lib import frombytes, tobytes
from pyarrow.util import _stringify_path


cdef compression_type_from_enum(CCompressionType compression_type):
    compression_map = {
        CCompressionType_UNCOMPRESSED: 'UNCOMPRESSED',
        CCompressionType_GZIP: 'ZLIB',
        CCompressionType_SNAPPY: 'SNAPPY',
        CCompressionType_LZ4: 'LZ4',
        CCompressionType_ZSTD: 'ZSTD',
    }
    if compression_type in compression_map:
        return compression_map[compression_type]
    raise ValueError('Unsupported compression')


cdef CCompressionType compression_type_from_name(name) except *:
    if not isinstance(name, str):
        raise TypeError('compression must be a string')
    name = name.upper()
    if name == 'ZLIB':
        return CCompressionType_GZIP
    elif name == 'SNAPPY':
        return CCompressionType_SNAPPY
    elif name == 'LZ4':
        return CCompressionType_LZ4
    elif name == 'ZSTD':
        return CCompressionType_ZSTD
    elif name == 'UNCOMPRESSED':
        return CCompressionType_UNCOMPRESSED
    raise ValueError(f'Unknown CompressionKind: {name}')


cdef compression_strategy_from_enum(
    CompressionStrategy compression_strategy
):
    compression_strategy_map = {
        _CompressionStrategy_SPEED: 'SPEED',
        _CompressionStrategy_COMPRESSION: 'COMPRESSION',
    }
    if compression_strategy in compression_strategy_map:
        return compression_strategy_map[compression_strategy]
    raise ValueError('Unsupported compression strategy')


cdef CompressionStrategy compression_strategy_from_name(name) except *:
    if not isinstance(name, str):
        raise TypeError('compression strategy must be a string')
    name = name.upper()
    if name == 'COMPRESSION':
        return _CompressionStrategy_COMPRESSION
    elif name == 'SPEED':
        return _CompressionStrategy_SPEED
    raise ValueError(f'Unknown CompressionStrategy: {name}')


cdef file_version_from_class(FileVersion file_version):
    return frombytes(file_version.ToString())


cdef writer_id_from_enum(WriterId writer_id):
    writer_id_map = {
        _WriterId_ORC_JAVA_WRITER: 'ORC_JAVA',
        _WriterId_ORC_CPP_WRITER: 'ORC_CPP',
        _WriterId_PRESTO_WRITER: 'PRESTO',
        _WriterId_SCRITCHLEY_GO: 'SCRITCHLEY_GO',
        _WriterId_TRINO_WRITER: 'TRINO',
    }
    if writer_id in writer_id_map:
        return writer_id_map[writer_id]
    raise ValueError('Unsupported writer ID')


cdef writer_version_from_enum(WriterVersion writer_version):
    writer_version_map = {
        _WriterVersion_ORIGINAL: 'ORIGINAL',
        _WriterVersion_HIVE_8732: 'HIVE_8732',
        _WriterVersion_HIVE_4243: 'HIVE_4243',
        _WriterVersion_HIVE_12055: 'HIVE_12055',
        _WriterVersion_HIVE_13083: 'HIVE_13083',
        _WriterVersion_ORC_101: 'ORC_101',
        _WriterVersion_ORC_135: 'ORC_135',
        _WriterVersion_ORC_517: 'ORC_517',
        _WriterVersion_ORC_203: 'ORC_203',
        _WriterVersion_ORC_14: 'ORC_14',
    }
    if writer_version in writer_version_map:
        return writer_version_map[writer_version]
    raise ValueError('Unsupported writer version')


cdef shared_ptr[WriteOptions] _create_write_options(
    file_version=None,
    batch_size=None,
    stripe_size=None,
    compression=None,
    compression_block_size=None,
    compression_strategy=None,
    row_index_stride=None,
    padding_tolerance=None,
    dictionary_key_size_threshold=None,
    bloom_filter_columns=None,
    bloom_filter_fpp=None
) except *:
    """General writer options"""
    cdef:
        shared_ptr[WriteOptions] options
    options = make_shared[WriteOptions]()
    # batch_size
    if batch_size is not None:
        if isinstance(batch_size, int) and batch_size > 0:
            deref(options).batch_size = batch_size
        else:
            raise ValueError(f"Invalid ORC writer batch size: {batch_size}")
    # file_version
    if file_version is not None:
        if file_version == "0.12":
            deref(options).file_version = FileVersion(0, 12)
        elif file_version == "0.11":
            deref(options).file_version = FileVersion(0, 11)
        else:
            raise ValueError(f"Unsupported ORC file version: {file_version}")
    # stripe_size
    if stripe_size is not None:
        if isinstance(stripe_size, int) and stripe_size > 0:
            deref(options).stripe_size = stripe_size
        else:
            raise ValueError(f"Invalid ORC stripe size: {stripe_size}")
    # compression
    if compression is not None:
        if isinstance(compression, str):
            deref(options).compression = compression_type_from_name(
                compression)
        else:
            raise TypeError("Unsupported ORC compression type: "
                            f"{compression}")
    # compression_block_size
    if compression_block_size is not None:
        if (isinstance(compression_block_size, int) and
                compression_block_size > 0):
            deref(options).compression_block_size = compression_block_size
        else:
            raise ValueError("Invalid ORC compression block size: "
                             f"{compression_block_size}")
    # compression_strategy
    if compression_strategy is not None:
        if isinstance(compression, str):
            deref(options).compression_strategy = \
                compression_strategy_from_name(compression_strategy)
        else:
            raise TypeError("Unsupported ORC compression strategy: "
                            f"{compression_strategy}")
    # row_index_stride
    if row_index_stride is not None:
        if isinstance(row_index_stride, int) and row_index_stride > 0:
            deref(options).row_index_stride = row_index_stride
        else:
            raise ValueError("Invalid ORC row index stride: "
                             f"{row_index_stride}")
    # padding_tolerance
    if padding_tolerance is not None:
        try:
            padding_tolerance = float(padding_tolerance)
            deref(options).padding_tolerance = padding_tolerance
        except Exception:
            raise ValueError("Invalid ORC padding tolerance: "
                             f"{padding_tolerance}")
    # dictionary_key_size_threshold
    if dictionary_key_size_threshold is not None:
        try:
            dictionary_key_size_threshold = float(
                dictionary_key_size_threshold)
            assert 0 <= dictionary_key_size_threshold <= 1
            deref(options).dictionary_key_size_threshold = \
                dictionary_key_size_threshold
        except Exception:
            raise ValueError("Invalid ORC dictionary key size threshold: "
                             f"{dictionary_key_size_threshold}")
    # bloom_filter_columns
    if bloom_filter_columns is not None:
        try:
            bloom_filter_columns = list(bloom_filter_columns)
            for col in bloom_filter_columns:
                assert isinstance(col, int) and col >= 0
            deref(options).bloom_filter_columns = bloom_filter_columns
        except Exception:
            raise ValueError("Invalid ORC BloomFilter columns: "
                             f"{bloom_filter_columns}")
    # Max false positive rate of the Bloom Filter
    if bloom_filter_fpp is not None:
        try:
            bloom_filter_fpp = float(bloom_filter_fpp)
            assert 0 <= bloom_filter_fpp <= 1
            deref(options).bloom_filter_fpp = bloom_filter_fpp
        except Exception:
            raise ValueError("Invalid ORC BloomFilter false positive rate: "
                             f"{bloom_filter_fpp}")
    return options


cdef class ORCReader(_Weakrefable):
    cdef:
        object source
        CMemoryPool* allocator
        unique_ptr[ORCFileReader] reader

    def __cinit__(self, MemoryPool memory_pool=None):
        self.allocator = maybe_unbox_memory_pool(memory_pool)

    def open(self, object source, c_bool use_memory_map=True):
        cdef:
            shared_ptr[CRandomAccessFile] rd_handle

        self.source = source

        get_reader(source, use_memory_map, &rd_handle)
        with nogil:
            self.reader = move(GetResultValue(
                ORCFileReader.Open(rd_handle, self.allocator)
            ))

    def metadata(self):
        """
        The arrow metadata for this file.

        Returns
        -------
        metadata : pyarrow.KeyValueMetadata
        """
        cdef:
            shared_ptr[const CKeyValueMetadata] sp_arrow_metadata

        with nogil:
            sp_arrow_metadata = GetResultValue(
                deref(self.reader).ReadMetadata()
            )

        return pyarrow_wrap_metadata(sp_arrow_metadata)

    def schema(self):
        """
        The arrow schema for this file.

        Returns
        -------
        schema : pyarrow.Schema
        """
        cdef:
            shared_ptr[CSchema] sp_arrow_schema

        with nogil:
            sp_arrow_schema = GetResultValue(deref(self.reader).ReadSchema())

        return pyarrow_wrap_schema(sp_arrow_schema)

    def nrows(self):
        return deref(self.reader).NumberOfRows()

    def nstripes(self):
        return deref(self.reader).NumberOfStripes()

    def file_version(self):
        return file_version_from_class(deref(self.reader).GetFileVersion())

    def software_version(self):
        return frombytes(deref(self.reader).GetSoftwareVersion())

    def compression(self):
        return compression_type_from_enum(
            GetResultValue(deref(self.reader).GetCompression()))

    def compression_size(self):
        return deref(self.reader).GetCompressionSize()

    def row_index_stride(self):
        return deref(self.reader).GetRowIndexStride()

    def writer(self):
        writer_name = writer_id_from_enum(deref(self.reader).GetWriterId())
        if writer_name == 'UNKNOWN':
            return deref(self.reader).GetWriterIdValue()
        else:
            return writer_name

    def writer_version(self):
        return writer_version_from_enum(deref(self.reader).GetWriterVersion())

    def nstripe_statistics(self):
        return deref(self.reader).GetNumberOfStripeStatistics()

    def content_length(self):
        return deref(self.reader).GetContentLength()

    def stripe_statistics_length(self):
        return deref(self.reader).GetStripeStatisticsLength()

    def file_footer_length(self):
        return deref(self.reader).GetFileFooterLength()

    def file_postscript_length(self):
        return deref(self.reader).GetFilePostscriptLength()

    def file_length(self):
        return deref(self.reader).GetFileLength()

    def serialized_file_tail(self):
        return deref(self.reader).GetSerializedFileTail()

    def read_stripe(self, n, columns=None):
        cdef:
            shared_ptr[CRecordBatch] sp_record_batch
            RecordBatch batch
            int64_t stripe
            std_vector[c_string] c_names

        stripe = n

        if columns is None:
            with nogil:
                sp_record_batch = GetResultValue(
                    deref(self.reader).ReadStripe(stripe)
                )
        else:
            c_names = [tobytes(name) for name in columns]
            with nogil:
                sp_record_batch = GetResultValue(
                    deref(self.reader).ReadStripe(stripe, c_names)
                )

        return pyarrow_wrap_batch(sp_record_batch)

    def read(self, columns=None):
        cdef:
            shared_ptr[CTable] sp_table
            std_vector[c_string] c_names

        if columns is None:
            with nogil:
                sp_table = GetResultValue(deref(self.reader).Read())
        else:
            c_names = [tobytes(name) for name in columns]
            with nogil:
                sp_table = GetResultValue(deref(self.reader).Read(c_names))

        return pyarrow_wrap_table(sp_table)


cdef class ORCWriter(_Weakrefable):
    cdef:
        unique_ptr[ORCFileWriter] writer
        shared_ptr[COutputStream] sink
        c_bool own_sink

    def open(self, object where, *,
             file_version=None,
             batch_size=None,
             stripe_size=None,
             compression=None,
             compression_block_size=None,
             compression_strategy=None,
             row_index_stride=None,
             padding_tolerance=None,
             dictionary_key_size_threshold=None,
             bloom_filter_columns=None,
             bloom_filter_fpp=None):
        cdef:
            shared_ptr[WriteOptions] write_options
            c_string c_where
        try:
            where = _stringify_path(where)
        except TypeError:
            get_writer(where, &self.sink)
            self.own_sink = False
        else:
            c_where = tobytes(where)
            with nogil:
                self.sink = GetResultValue(FileOutputStream.Open(c_where))
                self.own_sink = True

        write_options = _create_write_options(
            file_version=file_version,
            batch_size=batch_size,
            stripe_size=stripe_size,
            compression=compression,
            compression_block_size=compression_block_size,
            compression_strategy=compression_strategy,
            row_index_stride=row_index_stride,
            padding_tolerance=padding_tolerance,
            dictionary_key_size_threshold=dictionary_key_size_threshold,
            bloom_filter_columns=bloom_filter_columns,
            bloom_filter_fpp=bloom_filter_fpp
        )

        with nogil:
            self.writer = move(GetResultValue(
                ORCFileWriter.Open(self.sink.get(),
                                   deref(write_options))))

    def write(self, Table table):
        cdef:
            shared_ptr[CTable] sp_table
        sp_table = pyarrow_unwrap_table(table)
        with nogil:
            check_status(deref(self.writer).Write(deref(sp_table)))

    def close(self):
        with nogil:
            check_status(deref(self.writer).Close())
            if self.own_sink:
                check_status(deref(self.sink).Close())

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

import io
from textwrap import indent
import warnings

import numpy as np

from libcpp cimport nullptr

from cython.operator cimport dereference as deref
from pyarrow.includes.common cimport *
from pyarrow.includes.libarrow cimport *
from pyarrow.lib cimport (_Weakrefable, Buffer, Array, Schema,
                          check_status,
                          MemoryPool, maybe_unbox_memory_pool,
                          Table, NativeFile,
                          pyarrow_wrap_chunked_array,
                          pyarrow_wrap_schema,
                          pyarrow_wrap_table,
                          pyarrow_wrap_buffer,
                          pyarrow_wrap_batch,
                          pyarrow_wrap_scalar,
                          NativeFile, get_reader, get_writer,
                          string_to_timeunit)

from pyarrow.lib import (ArrowException, NativeFile, BufferOutputStream,
                         _stringify_path, _datetime_from_int,
                         tobytes, frombytes)

cimport cpython as cp


cdef class Statistics(_Weakrefable):
    """Statistics for a single column in a single row group."""

    def __cinit__(self):
        pass

    def __repr__(self):
        return """{}
  has_min_max: {}
  min: {}
  max: {}
  null_count: {}
  distinct_count: {}
  num_values: {}
  physical_type: {}
  logical_type: {}
  converted_type (legacy): {}""".format(object.__repr__(self),
                                        self.has_min_max,
                                        self.min,
                                        self.max,
                                        self.null_count,
                                        self.distinct_count,
                                        self.num_values,
                                        self.physical_type,
                                        str(self.logical_type),
                                        self.converted_type)

    def to_dict(self):
        """
        Get dictionary represenation of statistics.

        Returns
        -------
        dict
            Dictionary with a key for each attribute of this class.
        """
        d = dict(
            has_min_max=self.has_min_max,
            min=self.min,
            max=self.max,
            null_count=self.null_count,
            distinct_count=self.distinct_count,
            num_values=self.num_values,
            physical_type=self.physical_type
        )
        return d

    def __eq__(self, other):
        try:
            return self.equals(other)
        except TypeError:
            return NotImplemented

    def equals(self, Statistics other):
        """
        Return whether the two column statistics objects are equal.

        Parameters
        ----------
        other : Statistics
            Statistics to compare against.

        Returns
        -------
        are_equal : bool
        """
        return self.statistics.get().Equals(deref(other.statistics.get()))

    @property
    def has_min_max(self):
        """Whether min and max are present (bool)."""
        return self.statistics.get().HasMinMax()

    @property
    def has_null_count(self):
        """Whether null count is present (bool)."""
        return self.statistics.get().HasNullCount()

    @property
    def has_distinct_count(self):
        """Whether distinct count is preset (bool)."""
        return self.statistics.get().HasDistinctCount()

    @property
    def min_raw(self):
        """Min value as physical type (bool, int, float, or bytes)."""
        if self.has_min_max:
            return _cast_statistic_raw_min(self.statistics.get())
        else:
            return None

    @property
    def max_raw(self):
        """Max value as physical type (bool, int, float, or bytes)."""
        if self.has_min_max:
            return _cast_statistic_raw_max(self.statistics.get())
        else:
            return None

    @property
    def min(self):
        """
        Min value as logical type.

        Returned as the Python equivalent of logical type, such as datetime.date
        for dates and decimal.Decimal for decimals.
        """
        if self.has_min_max:
            min_scalar, _ = _cast_statistics(self.statistics.get())
            return min_scalar.as_py()
        else:
            return None

    @property
    def max(self):
        """
        Max value as logical type.

        Returned as the Python equivalent of logical type, such as datetime.date
        for dates and decimal.Decimal for decimals.
        """
        if self.has_min_max:
            _, max_scalar = _cast_statistics(self.statistics.get())
            return max_scalar.as_py()
        else:
            return None

    @property
    def null_count(self):
        """Number of null values in chunk (int)."""
        return self.statistics.get().null_count()

    @property
    def distinct_count(self):
        """
        Distinct number of values in chunk (int).

        If this is not set, will return 0.
        """
        # This seems to be zero if not set. See: ARROW-11793
        return self.statistics.get().distinct_count()

    @property
    def num_values(self):
        """Number of non-null values (int)."""
        return self.statistics.get().num_values()

    @property
    def physical_type(self):
        """Physical type of column (str)."""
        raw_physical_type = self.statistics.get().physical_type()
        return physical_type_name_from_enum(raw_physical_type)

    @property
    def logical_type(self):
        """Logical type of column (:class:`ParquetLogicalType`)."""
        return wrap_logical_type(self.statistics.get().descr().logical_type())

    @property
    def converted_type(self):
        """Legacy converted type (str or None)."""
        raw_converted_type = self.statistics.get().descr().converted_type()
        return converted_type_name_from_enum(raw_converted_type)


cdef class ParquetLogicalType(_Weakrefable):
    """Logical type of parquet type."""
    cdef:
        shared_ptr[const CParquetLogicalType] type

    def __cinit__(self):
        pass

    cdef init(self, const shared_ptr[const CParquetLogicalType]& type):
        self.type = type

    def __repr__(self):
        return "{}\n  {}".format(object.__repr__(self), str(self))

    def __str__(self):
        return frombytes(self.type.get().ToString(), safe=True)

    def to_json(self):
        """
        Get a JSON string containing type and type parameters.

        Returns
        -------
        json : str
            JSON representation of type, with at least a field called 'Type'
            which contains the type name. If the type is parameterized, such
            as a decimal with scale and precision, will contain those as fields
            as well.
        """
        return frombytes(self.type.get().ToJSON())

    @property
    def type(self):
        """Name of the logical type (str)."""
        return logical_type_name_from_enum(self.type.get().type())


cdef wrap_logical_type(const shared_ptr[const CParquetLogicalType]& type):
    cdef ParquetLogicalType out = ParquetLogicalType()
    out.init(type)
    return out


cdef _cast_statistic_raw_min(CStatistics* statistics):
    cdef ParquetType physical_type = statistics.physical_type()
    cdef uint32_t type_length = statistics.descr().type_length()
    if physical_type == ParquetType_BOOLEAN:
        return (<CBoolStatistics*> statistics).min()
    elif physical_type == ParquetType_INT32:
        return (<CInt32Statistics*> statistics).min()
    elif physical_type == ParquetType_INT64:
        return (<CInt64Statistics*> statistics).min()
    elif physical_type == ParquetType_FLOAT:
        return (<CFloatStatistics*> statistics).min()
    elif physical_type == ParquetType_DOUBLE:
        return (<CDoubleStatistics*> statistics).min()
    elif physical_type == ParquetType_BYTE_ARRAY:
        return _box_byte_array((<CByteArrayStatistics*> statistics).min())
    elif physical_type == ParquetType_FIXED_LEN_BYTE_ARRAY:
        return _box_flba((<CFLBAStatistics*> statistics).min(), type_length)


cdef _cast_statistic_raw_max(CStatistics* statistics):
    cdef ParquetType physical_type = statistics.physical_type()
    cdef uint32_t type_length = statistics.descr().type_length()
    if physical_type == ParquetType_BOOLEAN:
        return (<CBoolStatistics*> statistics).max()
    elif physical_type == ParquetType_INT32:
        return (<CInt32Statistics*> statistics).max()
    elif physical_type == ParquetType_INT64:
        return (<CInt64Statistics*> statistics).max()
    elif physical_type == ParquetType_FLOAT:
        return (<CFloatStatistics*> statistics).max()
    elif physical_type == ParquetType_DOUBLE:
        return (<CDoubleStatistics*> statistics).max()
    elif physical_type == ParquetType_BYTE_ARRAY:
        return _box_byte_array((<CByteArrayStatistics*> statistics).max())
    elif physical_type == ParquetType_FIXED_LEN_BYTE_ARRAY:
        return _box_flba((<CFLBAStatistics*> statistics).max(), type_length)


cdef _cast_statistics(CStatistics* statistics):
    cdef:
        shared_ptr[CScalar] c_min
        shared_ptr[CScalar] c_max
    check_status(StatisticsAsScalars(statistics[0], &c_min, &c_max))
    return (pyarrow_wrap_scalar(c_min), pyarrow_wrap_scalar(c_max))


cdef _box_byte_array(ParquetByteArray val):
    return cp.PyBytes_FromStringAndSize(<char*> val.ptr, <Py_ssize_t> val.len)


cdef _box_flba(ParquetFLBA val, uint32_t len):
    return cp.PyBytes_FromStringAndSize(<char*> val.ptr, <Py_ssize_t> len)


cdef class ColumnChunkMetaData(_Weakrefable):
    """Column metadata for a single row group."""

    def __cinit__(self):
        pass

    def __repr__(self):
        statistics = indent(repr(self.statistics), 4 * ' ')
        return """{0}
  file_offset: {1}
  file_path: {2}
  physical_type: {3}
  num_values: {4}
  path_in_schema: {5}
  is_stats_set: {6}
  statistics:
{7}
  compression: {8}
  encodings: {9}
  has_dictionary_page: {10}
  dictionary_page_offset: {11}
  data_page_offset: {12}
  total_compressed_size: {13}
  total_uncompressed_size: {14}""".format(object.__repr__(self),
                                          self.file_offset,
                                          self.file_path,
                                          self.physical_type,
                                          self.num_values,
                                          self.path_in_schema,
                                          self.is_stats_set,
                                          statistics,
                                          self.compression,
                                          self.encodings,
                                          self.has_dictionary_page,
                                          self.dictionary_page_offset,
                                          self.data_page_offset,
                                          self.total_compressed_size,
                                          self.total_uncompressed_size)

    def to_dict(self):
        """
        Get dictionary represenation of the column chunk metadata.

        Returns
        -------
        dict
            Dictionary with a key for each attribute of this class.
        """
        statistics = self.statistics.to_dict() if self.is_stats_set else None
        d = dict(
            file_offset=self.file_offset,
            file_path=self.file_path,
            physical_type=self.physical_type,
            num_values=self.num_values,
            path_in_schema=self.path_in_schema,
            is_stats_set=self.is_stats_set,
            statistics=statistics,
            compression=self.compression,
            encodings=self.encodings,
            has_dictionary_page=self.has_dictionary_page,
            dictionary_page_offset=self.dictionary_page_offset,
            data_page_offset=self.data_page_offset,
            total_compressed_size=self.total_compressed_size,
            total_uncompressed_size=self.total_uncompressed_size
        )
        return d

    def __eq__(self, other):
        try:
            return self.equals(other)
        except TypeError:
            return NotImplemented

    def equals(self, ColumnChunkMetaData other):
        """
        Return whether the two column chunk metadata objects are equal.

        Parameters
        ----------
        other : ColumnChunkMetaData
            Metadata to compare against.

        Returns
        -------
        are_equal : bool
        """
        return self.metadata.Equals(deref(other.metadata))

    @property
    def file_offset(self):
        """Offset into file where column chunk is located (int)."""
        return self.metadata.file_offset()

    @property
    def file_path(self):
        """Optional file path if set (str or None)."""
        return frombytes(self.metadata.file_path())

    @property
    def physical_type(self):
        """Physical type of column (str)."""
        return physical_type_name_from_enum(self.metadata.type())

    @property
    def num_values(self):
        """Total number of values (int)."""
        return self.metadata.num_values()

    @property
    def path_in_schema(self):
        """Nested path to field, separated by periods (str)."""
        path = self.metadata.path_in_schema().get().ToDotString()
        return frombytes(path)

    @property
    def is_stats_set(self):
        """Whether or not statistics are present in metadata (bool)."""
        return self.metadata.is_stats_set()

    @property
    def statistics(self):
        """Statistics for column chunk (:class:`Statistics`)."""
        if not self.metadata.is_stats_set():
            return None
        statistics = Statistics()
        statistics.init(self.metadata.statistics(), self)
        return statistics

    @property
    def compression(self):
        """
        Type of compression used for column (str).

        One of 'UNCOMPRESSED', 'SNAPPY', 'GZIP', 'LZO', 'BROTLI', 'LZ4', 'ZSTD',
        or 'UNKNOWN'.
        """
        return compression_name_from_enum(self.metadata.compression())

    @property
    def encodings(self):
        """
        Encodings used for column (tuple of str).

        One of 'PLAIN', 'BIT_PACKED', 'RLE', 'BYTE_STREAM_SPLIT', 'DELTA_BINARY_PACKED',
        'DELTA_BYTE_ARRAY'.
        """
        return tuple(map(encoding_name_from_enum, self.metadata.encodings()))

    @property
    def has_dictionary_page(self):
        """Whether there is dictionary data present in the column chunk (bool)."""
        return bool(self.metadata.has_dictionary_page())

    @property
    def dictionary_page_offset(self):
        """Offset of dictionary page reglative to column chunk offset (int)."""
        if self.has_dictionary_page:
            return self.metadata.dictionary_page_offset()
        else:
            return None

    @property
    def data_page_offset(self):
        """Offset of data page reglative to column chunk offset (int)."""
        return self.metadata.data_page_offset()

    @property
    def has_index_page(self):
        """Not yet supported."""
        raise NotImplementedError('not supported in parquet-cpp')

    @property
    def index_page_offset(self):
        """Not yet supported."""
        raise NotImplementedError("parquet-cpp doesn't return valid values")

    @property
    def total_compressed_size(self):
        """Compresssed size in bytes (int)."""
        return self.metadata.total_compressed_size()

    @property
    def total_uncompressed_size(self):
        """Uncompressed size in bytes (int)."""
        return self.metadata.total_uncompressed_size()


cdef class RowGroupMetaData(_Weakrefable):
    """Metadata for a single row group."""

    def __cinit__(self, FileMetaData parent, int index):
        if index < 0 or index >= parent.num_row_groups:
            raise IndexError('{0} out of bounds'.format(index))
        self.up_metadata = parent._metadata.RowGroup(index)
        self.metadata = self.up_metadata.get()
        self.parent = parent
        self.index = index

    def __reduce__(self):
        return RowGroupMetaData, (self.parent, self.index)

    def __eq__(self, other):
        try:
            return self.equals(other)
        except TypeError:
            return NotImplemented

    def equals(self, RowGroupMetaData other):
        """
        Return whether the two row group metadata objects are equal.

        Parameters
        ----------
        other : RowGroupMetaData
            Metadata to compare against.

        Returns
        -------
        are_equal : bool
        """
        return self.metadata.Equals(deref(other.metadata))

    def column(self, int i):
        """
        Get column metadata at given index.

        Parameters
        ----------
        i : int
            Index of column to get metadata for.

        Returns
        -------
        ColumnChunkMetaData
            Metadata for column within this chunk.
        """
        if i < 0 or i >= self.num_columns:
            raise IndexError('{0} out of bounds'.format(i))
        chunk = ColumnChunkMetaData()
        chunk.init(self, i)
        return chunk

    def __repr__(self):
        return """{0}
  num_columns: {1}
  num_rows: {2}
  total_byte_size: {3}""".format(object.__repr__(self),
                                 self.num_columns,
                                 self.num_rows,
                                 self.total_byte_size)

    def to_dict(self):
        """
        Get dictionary represenation of the row group metadata.

        Returns
        -------
        dict
            Dictionary with a key for each attribute of this class.
        """
        columns = []
        d = dict(
            num_columns=self.num_columns,
            num_rows=self.num_rows,
            total_byte_size=self.total_byte_size,
            columns=columns,
        )
        for i in range(self.num_columns):
            columns.append(self.column(i).to_dict())
        return d

    @property
    def num_columns(self):
        """Number of columns in this row group (int)."""
        return self.metadata.num_columns()

    @property
    def num_rows(self):
        """Number of rows in this row group (int)."""
        return self.metadata.num_rows()

    @property
    def total_byte_size(self):
        """Total byte size of all the uncompressed column data in this row group (int)."""
        return self.metadata.total_byte_size()


def _reconstruct_filemetadata(Buffer serialized):
    cdef:
        FileMetaData metadata = FileMetaData.__new__(FileMetaData)
        CBuffer *buffer = serialized.buffer.get()
        uint32_t metadata_len = <uint32_t>buffer.size()

    metadata.init(CFileMetaData_Make(buffer.data(), &metadata_len))

    return metadata


cdef class FileMetaData(_Weakrefable):
    """Parquet metadata for a single file."""

    def __cinit__(self):
        pass

    def __reduce__(self):
        cdef:
            NativeFile sink = BufferOutputStream()
            COutputStream* c_sink = sink.get_output_stream().get()
        with nogil:
            self._metadata.WriteTo(c_sink)

        cdef Buffer buffer = sink.getvalue()
        return _reconstruct_filemetadata, (buffer,)

    def __repr__(self):
        return """{0}
  created_by: {1}
  num_columns: {2}
  num_rows: {3}
  num_row_groups: {4}
  format_version: {5}
  serialized_size: {6}""".format(object.__repr__(self),
                                 self.created_by, self.num_columns,
                                 self.num_rows, self.num_row_groups,
                                 self.format_version,
                                 self.serialized_size)

    def to_dict(self):
        """
        Get dictionary represenation of the file metadata.

        Returns
        -------
        dict
            Dictionary with a key for each attribute of this class.
        """
        row_groups = []
        d = dict(
            created_by=self.created_by,
            num_columns=self.num_columns,
            num_rows=self.num_rows,
            num_row_groups=self.num_row_groups,
            row_groups=row_groups,
            format_version=self.format_version,
            serialized_size=self.serialized_size
        )
        for i in range(self.num_row_groups):
            row_groups.append(self.row_group(i).to_dict())
        return d

    def __eq__(self, other):
        try:
            return self.equals(other)
        except TypeError:
            return NotImplemented

    def equals(self, FileMetaData other not None):
        """
        Return whether the two file metadata objects are equal.

        Parameters
        ----------
        other : FileMetaData
            Metadata to compare against.

        Returns
        -------
        are_equal : bool
        """
        return self._metadata.Equals(deref(other._metadata))

    @property
    def schema(self):
        """Schema of the file (:class:`ParquetSchema`)."""
        if self._schema is None:
            self._schema = ParquetSchema(self)
        return self._schema

    @property
    def serialized_size(self):
        """Size of the original thrift encoded metadata footer (int)."""
        return self._metadata.size()

    @property
    def num_columns(self):
        """Number of columns in file (int)."""
        return self._metadata.num_columns()

    @property
    def num_rows(self):
        """Total number of rows in file (int)."""
        return self._metadata.num_rows()

    @property
    def num_row_groups(self):
        """Number of row groups in file (int)."""
        return self._metadata.num_row_groups()

    @property
    def format_version(self):
        """
        Parquet format version used in file (str, such as '1.0', '2.4').

        If version is missing or unparsable, will default to assuming '2.4'.
        """
        cdef ParquetVersion version = self._metadata.version()
        if version == ParquetVersion_V1:
            return '1.0'
        elif version == ParquetVersion_V2_0:
            return 'pseudo-2.0'
        elif version == ParquetVersion_V2_4:
            return '2.4'
        elif version == ParquetVersion_V2_6:
            return '2.6'
        else:
            warnings.warn('Unrecognized file version, assuming 2.4: {}'
                          .format(version))
            return '2.4'

    @property
    def created_by(self):
        """
        String describing source of the parquet file (str).

        This typically includes library name and version number. For example, Arrow 7.0's
        writer returns 'parquet-cpp-arrow version 7.0.0'.
        """
        return frombytes(self._metadata.created_by())

    @property
    def metadata(self):
        """Additional metadata as key value pairs (dict[bytes, bytes])."""
        cdef:
            unordered_map[c_string, c_string] metadata
            const CKeyValueMetadata* underlying_metadata
        underlying_metadata = self._metadata.key_value_metadata().get()
        if underlying_metadata != NULL:
            underlying_metadata.ToUnorderedMap(&metadata)
            return metadata
        else:
            return None

    def row_group(self, int i):
        """
        Get metadata for row group at index i.

        Parameters
        ----------
        i : int
            Row group index to get.

        Returns
        -------
        row_group_metadata : RowGroupMetaData
        """
        return RowGroupMetaData(self, i)

    def set_file_path(self, path):
        """
        Set ColumnChunk file paths to the given value.

        This method modifies the ``file_path`` field of each ColumnChunk
        in the FileMetaData to be a particular value.

        Parameters
        ----------
        path : str
            The file path to set on all ColumnChunks.
        """
        cdef:
            c_string c_path = tobytes(path)
        self._metadata.set_file_path(c_path)

    def append_row_groups(self, FileMetaData other):
        """
        Append row groups from other FileMetaData object.

        Parameters
        ----------
        other : FileMetaData
            Other metadata to append row groups from.
        """
        cdef shared_ptr[CFileMetaData] c_metadata

        c_metadata = other.sp_metadata
        self._metadata.AppendRowGroups(deref(c_metadata))

    def write_metadata_file(self, where):
        """
        Write the metadata to a metadata-only Parquet file.

        Parameters
        ----------
        where : path or file-like object
            Where to write the metadata.  Should be a writable path on
            the local filesystem, or a writable file-like object.
        """
        cdef:
            shared_ptr[COutputStream] sink
            c_string c_where

        try:
            where = _stringify_path(where)
        except TypeError:
            get_writer(where, &sink)
        else:
            c_where = tobytes(where)
            with nogil:
                sink = GetResultValue(FileOutputStream.Open(c_where))

        with nogil:
            check_status(
                WriteMetaDataFile(deref(self._metadata), sink.get()))


cdef class ParquetSchema(_Weakrefable):
    """A Parquet schema."""

    def __cinit__(self, FileMetaData container):
        self.parent = container
        self.schema = container._metadata.schema()

    def __repr__(self):
        return "{0}\n{1}".format(
            object.__repr__(self),
            frombytes(self.schema.ToString(), safe=True))

    def __reduce__(self):
        return ParquetSchema, (self.parent,)

    def __len__(self):
        return self.schema.num_columns()

    def __getitem__(self, i):
        return self.column(i)

    @property
    def names(self):
        """Name of each field (list of str)."""
        return [self[i].name for i in range(len(self))]

    def to_arrow_schema(self):
        """
        Convert Parquet schema to effective Arrow schema.

        Returns
        -------
        schema : Schema
        """
        cdef shared_ptr[CSchema] sp_arrow_schema

        with nogil:
            check_status(FromParquetSchema(
                self.schema, default_arrow_reader_properties(),
                self.parent._metadata.key_value_metadata(),
                &sp_arrow_schema))

        return pyarrow_wrap_schema(sp_arrow_schema)

    def __eq__(self, other):
        try:
            return self.equals(other)
        except TypeError:
            return NotImplemented

    def equals(self, ParquetSchema other):
        """
        Return whether the two schemas are equal.

        Parameters
        ----------
        other : ParquetSchema
            Schema to compare against.

        Returns
        -------
        are_equal : bool
        """
        return self.schema.Equals(deref(other.schema))

    def column(self, i):
        """
        Return the schema for a single column.

        Parameters
        ----------
        i : int
            Index of column in schema.

        Returns
        -------
        column_schema : ColumnSchema
        """
        if i < 0 or i >= len(self):
            raise IndexError('{0} out of bounds'.format(i))

        return ColumnSchema(self, i)


cdef class ColumnSchema(_Weakrefable):
    """Schema for a single column."""
    cdef:
        int index
        ParquetSchema parent
        const ColumnDescriptor* descr

    def __cinit__(self, ParquetSchema schema, int index):
        self.parent = schema
        self.index = index  # for pickling support
        self.descr = schema.schema.Column(index)

    def __eq__(self, other):
        try:
            return self.equals(other)
        except TypeError:
            return NotImplemented

    def __reduce__(self):
        return ColumnSchema, (self.parent, self.index)

    def equals(self, ColumnSchema other):
        """
        Return whether the two column schemas are equal.

        Parameters
        ----------
        other : ColumnSchema
            Schema to compare against.

        Returns
        -------
        are_equal : bool
        """
        return self.descr.Equals(deref(other.descr))

    def __repr__(self):
        physical_type = self.physical_type
        converted_type = self.converted_type
        if converted_type == 'DECIMAL':
            converted_type = 'DECIMAL({0}, {1})'.format(self.precision,
                                                        self.scale)
        elif physical_type == 'FIXED_LEN_BYTE_ARRAY':
            converted_type = ('FIXED_LEN_BYTE_ARRAY(length={0})'
                              .format(self.length))

        return """<ParquetColumnSchema>
  name: {0}
  path: {1}
  max_definition_level: {2}
  max_repetition_level: {3}
  physical_type: {4}
  logical_type: {5}
  converted_type (legacy): {6}""".format(self.name, self.path,
                                         self.max_definition_level,
                                         self.max_repetition_level,
                                         physical_type,
                                         str(self.logical_type),
                                         converted_type)

    @property
    def name(self):
        """Name of field (str)."""
        return frombytes(self.descr.name())

    @property
    def path(self):
        """Nested path to field, separated by periods (str)."""
        return frombytes(self.descr.path().get().ToDotString())

    @property
    def max_definition_level(self):
        """Maximum definition level (int)."""
        return self.descr.max_definition_level()

    @property
    def max_repetition_level(self):
        """Maximum repetition level (int)."""
        return self.descr.max_repetition_level()

    @property
    def physical_type(self):
        """Name of physical type (str)."""
        return physical_type_name_from_enum(self.descr.physical_type())

    @property
    def logical_type(self):
        """Logical type of column (:class:`ParquetLogicalType`)."""
        return wrap_logical_type(self.descr.logical_type())

    @property
    def converted_type(self):
        """Legacy converted type (str or None)."""
        return converted_type_name_from_enum(self.descr.converted_type())

    # FIXED_LEN_BYTE_ARRAY attribute
    @property
    def length(self):
        """Array length if fixed length byte array type, None otherwise (int or None)."""
        return self.descr.type_length()

    # Decimal attributes
    @property
    def precision(self):
        """Precision if decimal type, None otherwise (int or None)."""
        return self.descr.type_precision()

    @property
    def scale(self):
        """Scale if decimal type, None otherwise (int or None)."""
        return self.descr.type_scale()


cdef physical_type_name_from_enum(ParquetType type_):
    return {
        ParquetType_BOOLEAN: 'BOOLEAN',
        ParquetType_INT32: 'INT32',
        ParquetType_INT64: 'INT64',
        ParquetType_INT96: 'INT96',
        ParquetType_FLOAT: 'FLOAT',
        ParquetType_DOUBLE: 'DOUBLE',
        ParquetType_BYTE_ARRAY: 'BYTE_ARRAY',
        ParquetType_FIXED_LEN_BYTE_ARRAY: 'FIXED_LEN_BYTE_ARRAY',
    }.get(type_, 'UNKNOWN')


cdef logical_type_name_from_enum(ParquetLogicalTypeId type_):
    return {
        ParquetLogicalType_UNDEFINED: 'UNDEFINED',
        ParquetLogicalType_STRING: 'STRING',
        ParquetLogicalType_MAP: 'MAP',
        ParquetLogicalType_LIST: 'LIST',
        ParquetLogicalType_ENUM: 'ENUM',
        ParquetLogicalType_DECIMAL: 'DECIMAL',
        ParquetLogicalType_DATE: 'DATE',
        ParquetLogicalType_TIME: 'TIME',
        ParquetLogicalType_TIMESTAMP: 'TIMESTAMP',
        ParquetLogicalType_INT: 'INT',
        ParquetLogicalType_JSON: 'JSON',
        ParquetLogicalType_BSON: 'BSON',
        ParquetLogicalType_UUID: 'UUID',
        ParquetLogicalType_NONE: 'NONE',
    }.get(type_, 'UNKNOWN')


cdef converted_type_name_from_enum(ParquetConvertedType type_):
    return {
        ParquetConvertedType_NONE: 'NONE',
        ParquetConvertedType_UTF8: 'UTF8',
        ParquetConvertedType_MAP: 'MAP',
        ParquetConvertedType_MAP_KEY_VALUE: 'MAP_KEY_VALUE',
        ParquetConvertedType_LIST: 'LIST',
        ParquetConvertedType_ENUM: 'ENUM',
        ParquetConvertedType_DECIMAL: 'DECIMAL',
        ParquetConvertedType_DATE: 'DATE',
        ParquetConvertedType_TIME_MILLIS: 'TIME_MILLIS',
        ParquetConvertedType_TIME_MICROS: 'TIME_MICROS',
        ParquetConvertedType_TIMESTAMP_MILLIS: 'TIMESTAMP_MILLIS',
        ParquetConvertedType_TIMESTAMP_MICROS: 'TIMESTAMP_MICROS',
        ParquetConvertedType_UINT_8: 'UINT_8',
        ParquetConvertedType_UINT_16: 'UINT_16',
        ParquetConvertedType_UINT_32: 'UINT_32',
        ParquetConvertedType_UINT_64: 'UINT_64',
        ParquetConvertedType_INT_8: 'INT_8',
        ParquetConvertedType_INT_16: 'INT_16',
        ParquetConvertedType_INT_32: 'INT_32',
        ParquetConvertedType_INT_64: 'INT_64',
        ParquetConvertedType_JSON: 'JSON',
        ParquetConvertedType_BSON: 'BSON',
        ParquetConvertedType_INTERVAL: 'INTERVAL',
    }.get(type_, 'UNKNOWN')


cdef encoding_name_from_enum(ParquetEncoding encoding_):
    return {
        ParquetEncoding_PLAIN: 'PLAIN',
        ParquetEncoding_PLAIN_DICTIONARY: 'PLAIN_DICTIONARY',
        ParquetEncoding_RLE: 'RLE',
        ParquetEncoding_BIT_PACKED: 'BIT_PACKED',
        ParquetEncoding_DELTA_BINARY_PACKED: 'DELTA_BINARY_PACKED',
        ParquetEncoding_DELTA_LENGTH_BYTE_ARRAY: 'DELTA_LENGTH_BYTE_ARRAY',
        ParquetEncoding_DELTA_BYTE_ARRAY: 'DELTA_BYTE_ARRAY',
        ParquetEncoding_RLE_DICTIONARY: 'RLE_DICTIONARY',
        ParquetEncoding_BYTE_STREAM_SPLIT: 'BYTE_STREAM_SPLIT',
    }.get(encoding_, 'UNKNOWN')


cdef encoding_enum_from_name(str encoding_name):
    enc = {
        'PLAIN': ParquetEncoding_PLAIN,
        'BIT_PACKED': ParquetEncoding_BIT_PACKED,
        'RLE': ParquetEncoding_RLE,
        'BYTE_STREAM_SPLIT': ParquetEncoding_BYTE_STREAM_SPLIT,
        'DELTA_BINARY_PACKED': ParquetEncoding_DELTA_BINARY_PACKED,
        'DELTA_BYTE_ARRAY': ParquetEncoding_DELTA_BYTE_ARRAY,
        'RLE_DICTIONARY': 'dict',
        'PLAIN_DICTIONARY': 'dict',
    }.get(encoding_name, None)
    if enc is None:
        raise ValueError(f"Unsupported column encoding: {encoding_name!r}")
    elif enc == 'dict':
        raise ValueError(f"{encoding_name!r} is already used by default.")
    else:
        return enc


cdef compression_name_from_enum(ParquetCompression compression_):
    return {
        ParquetCompression_UNCOMPRESSED: 'UNCOMPRESSED',
        ParquetCompression_SNAPPY: 'SNAPPY',
        ParquetCompression_GZIP: 'GZIP',
        ParquetCompression_LZO: 'LZO',
        ParquetCompression_BROTLI: 'BROTLI',
        ParquetCompression_LZ4: 'LZ4',
        ParquetCompression_ZSTD: 'ZSTD',
    }.get(compression_, 'UNKNOWN')


cdef int check_compression_name(name) except -1:
    if name.upper() not in {'NONE', 'SNAPPY', 'GZIP', 'LZO', 'BROTLI', 'LZ4',
                            'ZSTD'}:
        raise ArrowException("Unsupported compression: " + name)
    return 0


cdef ParquetCompression compression_from_name(name):
    name = name.upper()
    if name == 'SNAPPY':
        return ParquetCompression_SNAPPY
    elif name == 'GZIP':
        return ParquetCompression_GZIP
    elif name == 'LZO':
        return ParquetCompression_LZO
    elif name == 'BROTLI':
        return ParquetCompression_BROTLI
    elif name == 'LZ4':
        return ParquetCompression_LZ4
    elif name == 'ZSTD':
        return ParquetCompression_ZSTD
    else:
        return ParquetCompression_UNCOMPRESSED


cdef class ParquetReader(_Weakrefable):
    cdef:
        object source
        CMemoryPool* pool
        unique_ptr[FileReader] reader
        FileMetaData _metadata
        shared_ptr[CRandomAccessFile] rd_handle

    cdef public:
        _column_idx_map

    def __cinit__(self, MemoryPool memory_pool=None):
        self.pool = maybe_unbox_memory_pool(memory_pool)
        self._metadata = None

    def open(self, object source not None, *, bint use_memory_map=False,
             read_dictionary=None, FileMetaData metadata=None,
             int buffer_size=0, bint pre_buffer=False,
             coerce_int96_timestamp_unit=None,
             FileDecryptionProperties decryption_properties=None,
             thrift_string_size_limit=None,
             thrift_container_size_limit=None):
        cdef:
            shared_ptr[CFileMetaData] c_metadata
            CReaderProperties properties = default_reader_properties()
            ArrowReaderProperties arrow_props = (
                default_arrow_reader_properties())
            c_string path
            FileReaderBuilder builder
            TimeUnit int96_timestamp_unit_code

        if metadata is not None:
            c_metadata = metadata.sp_metadata

        if buffer_size > 0:
            properties.enable_buffered_stream()
            properties.set_buffer_size(buffer_size)
        elif buffer_size == 0:
            properties.disable_buffered_stream()
        else:
            raise ValueError('Buffer size must be larger than zero')

        if thrift_string_size_limit is not None:
            if thrift_string_size_limit <= 0:
                raise ValueError("thrift_string_size_limit "
                                 "must be larger than zero")
            properties.set_thrift_string_size_limit(thrift_string_size_limit)
        if thrift_container_size_limit is not None:
            if thrift_container_size_limit <= 0:
                raise ValueError("thrift_container_size_limit "
                                 "must be larger than zero")
            properties.set_thrift_container_size_limit(
                thrift_container_size_limit)

        if decryption_properties is not None:
            properties.file_decryption_properties(
                decryption_properties.unwrap())

        arrow_props.set_pre_buffer(pre_buffer)

        if coerce_int96_timestamp_unit is None:
            # use the default defined in default_arrow_reader_properties()
            pass
        else:
            arrow_props.set_coerce_int96_timestamp_unit(
                string_to_timeunit(coerce_int96_timestamp_unit))

        self.source = source
        get_reader(source, use_memory_map, &self.rd_handle)

        with nogil:
            check_status(builder.Open(self.rd_handle, properties, c_metadata))

        # Set up metadata
        with nogil:
            c_metadata = builder.raw_reader().metadata()
        self._metadata = result = FileMetaData()
        result.init(c_metadata)

        if read_dictionary is not None:
            self._set_read_dictionary(read_dictionary, &arrow_props)

        with nogil:
            check_status(builder.memory_pool(self.pool)
                         .properties(arrow_props)
                         .Build(&self.reader))

    cdef _set_read_dictionary(self, read_dictionary,
                              ArrowReaderProperties* props):
        for column in read_dictionary:
            if not isinstance(column, int):
                column = self.column_name_idx(column)
            props.set_read_dictionary(column, True)

    @property
    def column_paths(self):
        cdef:
            FileMetaData container = self.metadata
            const CFileMetaData* metadata = container._metadata
            vector[c_string] path
            int i = 0

        paths = []
        for i in range(0, metadata.num_columns()):
            path = (metadata.schema().Column(i)
                    .path().get().ToDotVector())
            paths.append([frombytes(x) for x in path])

        return paths

    @property
    def metadata(self):
        return self._metadata

    @property
    def schema_arrow(self):
        cdef shared_ptr[CSchema] out
        with nogil:
            check_status(self.reader.get().GetSchema(&out))
        return pyarrow_wrap_schema(out)

    @property
    def num_row_groups(self):
        return self.reader.get().num_row_groups()

    def set_use_threads(self, bint use_threads):
        self.reader.get().set_use_threads(use_threads)

    def set_batch_size(self, int64_t batch_size):
        self.reader.get().set_batch_size(batch_size)

    def iter_batches(self, int64_t batch_size, row_groups, column_indices=None,
                     bint use_threads=True):
        cdef:
            vector[int] c_row_groups
            vector[int] c_column_indices
            shared_ptr[CRecordBatch] record_batch
            shared_ptr[TableBatchReader] batch_reader
            unique_ptr[CRecordBatchReader] recordbatchreader

        self.set_batch_size(batch_size)

        if use_threads:
            self.set_use_threads(use_threads)

        for row_group in row_groups:
            c_row_groups.push_back(row_group)

        if column_indices is not None:
            for index in column_indices:
                c_column_indices.push_back(index)
            with nogil:
                check_status(
                    self.reader.get().GetRecordBatchReader(
                        c_row_groups, c_column_indices, &recordbatchreader
                    )
                )
        else:
            with nogil:
                check_status(
                    self.reader.get().GetRecordBatchReader(
                        c_row_groups, &recordbatchreader
                    )
                )

        while True:
            with nogil:
                check_status(
                    recordbatchreader.get().ReadNext(&record_batch)
                )

            if record_batch.get() == NULL:
                break

            yield pyarrow_wrap_batch(record_batch)

    def read_row_group(self, int i, column_indices=None,
                       bint use_threads=True):
        return self.read_row_groups([i], column_indices, use_threads)

    def read_row_groups(self, row_groups not None, column_indices=None,
                        bint use_threads=True):
        cdef:
            shared_ptr[CTable] ctable
            vector[int] c_row_groups
            vector[int] c_column_indices

        self.set_use_threads(use_threads)

        for row_group in row_groups:
            c_row_groups.push_back(row_group)

        if column_indices is not None:
            for index in column_indices:
                c_column_indices.push_back(index)

            with nogil:
                check_status(self.reader.get()
                             .ReadRowGroups(c_row_groups, c_column_indices,
                                            &ctable))
        else:
            # Read all columns
            with nogil:
                check_status(self.reader.get()
                             .ReadRowGroups(c_row_groups, &ctable))
        return pyarrow_wrap_table(ctable)

    def read_all(self, column_indices=None, bint use_threads=True):
        cdef:
            shared_ptr[CTable] ctable
            vector[int] c_column_indices

        self.set_use_threads(use_threads)

        if column_indices is not None:
            for index in column_indices:
                c_column_indices.push_back(index)

            with nogil:
                check_status(self.reader.get()
                             .ReadTable(c_column_indices, &ctable))
        else:
            # Read all columns
            with nogil:
                check_status(self.reader.get()
                             .ReadTable(&ctable))
        return pyarrow_wrap_table(ctable)

    def scan_contents(self, column_indices=None, batch_size=65536):
        cdef:
            vector[int] c_column_indices
            int32_t c_batch_size
            int64_t c_num_rows

        if column_indices is not None:
            for index in column_indices:
                c_column_indices.push_back(index)

        c_batch_size = batch_size

        with nogil:
            check_status(self.reader.get()
                         .ScanContents(c_column_indices, c_batch_size,
                                       &c_num_rows))

        return c_num_rows

    def column_name_idx(self, column_name):
        """
        Find the index of a column by its name.

        Parameters
        ----------
        column_name : str
            Name of the column; separation of nesting levels is done via ".".

        Returns
        -------
        column_idx : int
            Integer index of the column in the schema.
        """
        cdef:
            FileMetaData container = self.metadata
            const CFileMetaData* metadata = container._metadata
            int i = 0

        if self._column_idx_map is None:
            self._column_idx_map = {}
            for i in range(0, metadata.num_columns()):
                col_bytes = tobytes(metadata.schema().Column(i)
                                    .path().get().ToDotString())
                self._column_idx_map[col_bytes] = i

        return self._column_idx_map[tobytes(column_name)]

    def read_column(self, int column_index):
        cdef shared_ptr[CChunkedArray] out
        with nogil:
            check_status(self.reader.get()
                         .ReadColumn(column_index, &out))
        return pyarrow_wrap_chunked_array(out)

    def close(self):
        if not self.closed:
            with nogil:
                check_status(self.rd_handle.get().Close())

    @property
    def closed(self):
        if self.rd_handle == NULL:
            return True
        with nogil:
            closed = self.rd_handle.get().closed()
        return closed


cdef shared_ptr[WriterProperties] _create_writer_properties(
        use_dictionary=None,
        compression=None,
        version=None,
        write_statistics=None,
        data_page_size=None,
        compression_level=None,
        use_byte_stream_split=False,
        column_encoding=None,
        data_page_version=None,
        FileEncryptionProperties encryption_properties=None,
        write_batch_size=None,
        dictionary_pagesize_limit=None) except *:
    """General writer properties"""
    cdef:
        shared_ptr[WriterProperties] properties
        WriterProperties.Builder props

    # data_page_version

    if data_page_version is not None:
        if data_page_version == "1.0":
            props.data_page_version(ParquetDataPageVersion_V1)
        elif data_page_version == "2.0":
            props.data_page_version(ParquetDataPageVersion_V2)
        else:
            raise ValueError("Unsupported Parquet data page version: {0}"
                             .format(data_page_version))

    # version

    if version is not None:
        if version == "1.0":
            props.version(ParquetVersion_V1)
        elif version in ("2.0", "pseudo-2.0"):
            warnings.warn(
                "Parquet format '2.0' pseudo version is deprecated, use "
                "'2.4' or '2.6' for fine-grained feature selection",
                FutureWarning, stacklevel=2)
            props.version(ParquetVersion_V2_0)
        elif version == "2.4":
            props.version(ParquetVersion_V2_4)
        elif version == "2.6":
            props.version(ParquetVersion_V2_6)
        else:
            raise ValueError("Unsupported Parquet format version: {0}"
                             .format(version))

    # compression

    if isinstance(compression, basestring):
        check_compression_name(compression)
        props.compression(compression_from_name(compression))
    elif compression is not None:
        for column, codec in compression.iteritems():
            check_compression_name(codec)
            props.compression(tobytes(column), compression_from_name(codec))

    if isinstance(compression_level, int):
        props.compression_level(compression_level)
    elif compression_level is not None:
        for column, level in compression_level.iteritems():
            props.compression_level(tobytes(column), level)

    # use_dictionary

    if isinstance(use_dictionary, bool):
        if use_dictionary:
            props.enable_dictionary()
            if column_encoding is not None:
                raise ValueError(
                    "To use 'column_encoding' set 'use_dictionary' to False")
        else:
            props.disable_dictionary()
    elif use_dictionary is not None:
        # Deactivate dictionary encoding by default
        props.disable_dictionary()
        for column in use_dictionary:
            props.enable_dictionary(tobytes(column))
            if (column_encoding is not None and
                    column_encoding.get(column) is not None):
                raise ValueError(
                    "To use 'column_encoding' set 'use_dictionary' to False")

    # write_statistics

    if isinstance(write_statistics, bool):
        if write_statistics:
            props.enable_statistics()
        else:
            props.disable_statistics()
    elif write_statistics is not None:
        # Deactivate statistics by default and enable for specified columns
        props.disable_statistics()
        for column in write_statistics:
            props.enable_statistics(tobytes(column))

    # use_byte_stream_split

    if isinstance(use_byte_stream_split, bool):
        if use_byte_stream_split:
            if column_encoding is not None:
                raise ValueError(
                    "'use_byte_stream_split' can not be passed"
                    "together with 'column_encoding'")
            else:
                props.encoding(ParquetEncoding_BYTE_STREAM_SPLIT)
    elif use_byte_stream_split is not None:
        for column in use_byte_stream_split:
            if column_encoding is None:
                column_encoding = {column: 'BYTE_STREAM_SPLIT'}
            elif column_encoding.get(column, None) is None:
                column_encoding[column] = 'BYTE_STREAM_SPLIT'
            else:
                raise ValueError(
                    "'use_byte_stream_split' can not be passed"
                    "together with 'column_encoding'")

    # column_encoding
    # encoding map - encode individual columns

    if column_encoding is not None:
        if isinstance(column_encoding, dict):
            for column, _encoding in column_encoding.items():
                props.encoding(tobytes(column),
                               encoding_enum_from_name(_encoding))
        elif isinstance(column_encoding, str):
            props.encoding(encoding_enum_from_name(column_encoding))
        else:
            raise TypeError(
                "'column_encoding' should be a dictionary or a string")

    if data_page_size is not None:
        props.data_pagesize(data_page_size)

    if write_batch_size is not None:
        props.write_batch_size(write_batch_size)

    if dictionary_pagesize_limit is not None:
        props.dictionary_pagesize_limit(dictionary_pagesize_limit)

    # encryption

    if encryption_properties is not None:
        props.encryption(
            (<FileEncryptionProperties>encryption_properties).unwrap())

    properties = props.build()

    return properties


cdef shared_ptr[ArrowWriterProperties] _create_arrow_writer_properties(
        use_deprecated_int96_timestamps=False,
        coerce_timestamps=None,
        allow_truncated_timestamps=False,
        writer_engine_version=None,
        use_compliant_nested_type=False,
        store_schema=True) except *:
    """Arrow writer properties"""
    cdef:
        shared_ptr[ArrowWriterProperties] arrow_properties
        ArrowWriterProperties.Builder arrow_props

    # Store the original Arrow schema so things like dictionary types can
    # be automatically reconstructed
    if store_schema:
        arrow_props.store_schema()

    # int96 support

    if use_deprecated_int96_timestamps:
        arrow_props.enable_deprecated_int96_timestamps()
    else:
        arrow_props.disable_deprecated_int96_timestamps()

    # coerce_timestamps

    if coerce_timestamps == 'ms':
        arrow_props.coerce_timestamps(TimeUnit_MILLI)
    elif coerce_timestamps == 'us':
        arrow_props.coerce_timestamps(TimeUnit_MICRO)
    elif coerce_timestamps is not None:
        raise ValueError('Invalid value for coerce_timestamps: {0}'
                         .format(coerce_timestamps))

    # allow_truncated_timestamps

    if allow_truncated_timestamps:
        arrow_props.allow_truncated_timestamps()
    else:
        arrow_props.disallow_truncated_timestamps()

    # use_compliant_nested_type

    if use_compliant_nested_type:
        arrow_props.enable_compliant_nested_types()
    else:
        arrow_props.disable_compliant_nested_types()

    # writer_engine_version

    if writer_engine_version == "V1":
        warnings.warn("V1 parquet writer engine is a no-op.  Use V2.")
        arrow_props.set_engine_version(ArrowWriterEngineVersion.V1)
    elif writer_engine_version != "V2":
        raise ValueError("Unsupported Writer Engine Version: {0}"
                         .format(writer_engine_version))

    arrow_properties = arrow_props.build()

    return arrow_properties


cdef class ParquetWriter(_Weakrefable):
    cdef:
        unique_ptr[FileWriter] writer
        shared_ptr[COutputStream] sink
        bint own_sink

    cdef readonly:
        object use_dictionary
        object use_deprecated_int96_timestamps
        object use_byte_stream_split
        object column_encoding
        object coerce_timestamps
        object allow_truncated_timestamps
        object compression
        object compression_level
        object data_page_version
        object use_compliant_nested_type
        object version
        object write_statistics
        object writer_engine_version
        int row_group_size
        int64_t data_page_size
        FileEncryptionProperties encryption_properties
        int64_t write_batch_size
        int64_t dictionary_pagesize_limit
        object store_schema

    def __cinit__(self, where, Schema schema, use_dictionary=None,
                  compression=None, version=None,
                  write_statistics=None,
                  MemoryPool memory_pool=None,
                  use_deprecated_int96_timestamps=False,
                  coerce_timestamps=None,
                  data_page_size=None,
                  allow_truncated_timestamps=False,
                  compression_level=None,
                  use_byte_stream_split=False,
                  column_encoding=None,
                  writer_engine_version=None,
                  data_page_version=None,
                  use_compliant_nested_type=False,
                  encryption_properties=None,
                  write_batch_size=None,
                  dictionary_pagesize_limit=None,
                  store_schema=True):
        cdef:
            shared_ptr[WriterProperties] properties
            shared_ptr[ArrowWriterProperties] arrow_properties
            c_string c_where
            CMemoryPool* pool

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

        properties = _create_writer_properties(
            use_dictionary=use_dictionary,
            compression=compression,
            version=version,
            write_statistics=write_statistics,
            data_page_size=data_page_size,
            compression_level=compression_level,
            use_byte_stream_split=use_byte_stream_split,
            column_encoding=column_encoding,
            data_page_version=data_page_version,
            encryption_properties=encryption_properties,
            write_batch_size=write_batch_size,
            dictionary_pagesize_limit=dictionary_pagesize_limit
        )
        arrow_properties = _create_arrow_writer_properties(
            use_deprecated_int96_timestamps=use_deprecated_int96_timestamps,
            coerce_timestamps=coerce_timestamps,
            allow_truncated_timestamps=allow_truncated_timestamps,
            writer_engine_version=writer_engine_version,
            use_compliant_nested_type=use_compliant_nested_type,
            store_schema=store_schema,
        )

        pool = maybe_unbox_memory_pool(memory_pool)
        with nogil:
            self.writer = move(GetResultValue(
                FileWriter.Open(deref(schema.schema), pool,
                                self.sink, properties, arrow_properties)))

    def close(self):
        with nogil:
            check_status(self.writer.get().Close())
            if self.own_sink:
                check_status(self.sink.get().Close())

    def write_table(self, Table table, row_group_size=None):
        cdef:
            CTable* ctable = table.table
            int64_t c_row_group_size

        if row_group_size is None or row_group_size == -1:
            c_row_group_size = ctable.num_rows()
        elif row_group_size == 0:
            raise ValueError('Row group size cannot be 0')
        else:
            c_row_group_size = row_group_size

        with nogil:
            check_status(self.writer.get()
                         .WriteTable(deref(ctable), c_row_group_size))

    @property
    def metadata(self):
        cdef:
            shared_ptr[CFileMetaData] metadata
            FileMetaData result
        with nogil:
            metadata = self.writer.get().metadata()
        if metadata:
            result = FileMetaData()
            result.init(metadata)
            return result
        raise RuntimeError(
            'file metadata is only available after writer close')

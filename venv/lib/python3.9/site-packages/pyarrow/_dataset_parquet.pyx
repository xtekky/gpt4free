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

"""Dataset support for Parquest file format."""

from cython.operator cimport dereference as deref

import os
import warnings

import pyarrow as pa
from pyarrow.lib cimport *
from pyarrow.lib import frombytes, tobytes
from pyarrow.includes.libarrow cimport *
from pyarrow.includes.libarrow_dataset cimport *
from pyarrow.includes.libarrow_dataset_parquet cimport *
from pyarrow._fs cimport FileSystem
from pyarrow.util import _is_path_like, _stringify_path

from pyarrow._compute cimport Expression, _bind
from pyarrow._dataset cimport (
    _make_file_source,
    DatasetFactory,
    FileFormat,
    FileFragment,
    FileWriteOptions,
    Fragment,
    FragmentScanOptions,
    Partitioning,
    PartitioningFactory,
    WrittenFile
)


from pyarrow._parquet cimport (
    _create_writer_properties, _create_arrow_writer_properties,
    FileMetaData, RowGroupMetaData, ColumnChunkMetaData
)


cdef Expression _true = Expression._scalar(True)


ctypedef CParquetFileWriter* _CParquetFileWriterPtr


cdef class ParquetFileFormat(FileFormat):
    """
    FileFormat for Parquet

    Parameters
    ----------
    read_options : ParquetReadOptions
        Read options for the file.
    default_fragment_scan_options : ParquetFragmentScanOptions
        Scan Options for the file.
    **kwargs : dict
        Additional options for read option or scan option
    """

    cdef:
        CParquetFileFormat* parquet_format

    def __init__(self, read_options=None,
                 default_fragment_scan_options=None, **kwargs):
        cdef:
            shared_ptr[CParquetFileFormat] wrapped
            CParquetFileFormatReaderOptions* options

        # Read/scan options
        read_options_args = {option: kwargs[option] for option in kwargs
                             if option in _PARQUET_READ_OPTIONS}
        scan_args = {option: kwargs[option] for option in kwargs
                     if option not in _PARQUET_READ_OPTIONS}
        if read_options and read_options_args:
            duplicates = ', '.join(sorted(read_options_args))
            raise ValueError(f'If `read_options` is given, '
                             f'cannot specify {duplicates}')
        if default_fragment_scan_options and scan_args:
            duplicates = ', '.join(sorted(scan_args))
            raise ValueError(f'If `default_fragment_scan_options` is given, '
                             f'cannot specify {duplicates}')

        if read_options is None:
            read_options = ParquetReadOptions(**read_options_args)
        elif isinstance(read_options, dict):
            # For backwards compatibility
            duplicates = []
            for option, value in read_options.items():
                if option in _PARQUET_READ_OPTIONS:
                    read_options_args[option] = value
                else:
                    duplicates.append(option)
                    scan_args[option] = value
            if duplicates:
                duplicates = ", ".join(duplicates)
                warnings.warn(f'The scan options {duplicates} should be '
                              'specified directly as keyword arguments')
            read_options = ParquetReadOptions(**read_options_args)
        elif not isinstance(read_options, ParquetReadOptions):
            raise TypeError('`read_options` must be either a dictionary or an '
                            'instance of ParquetReadOptions')

        if default_fragment_scan_options is None:
            default_fragment_scan_options = ParquetFragmentScanOptions(
                **scan_args)
        elif isinstance(default_fragment_scan_options, dict):
            default_fragment_scan_options = ParquetFragmentScanOptions(
                **default_fragment_scan_options)
        elif not isinstance(default_fragment_scan_options,
                            ParquetFragmentScanOptions):
            raise TypeError('`default_fragment_scan_options` must be either a '
                            'dictionary or an instance of '
                            'ParquetFragmentScanOptions')

        wrapped = make_shared[CParquetFileFormat]()
        options = &(wrapped.get().reader_options)
        if read_options.dictionary_columns is not None:
            for column in read_options.dictionary_columns:
                options.dict_columns.insert(tobytes(column))
        options.coerce_int96_timestamp_unit = \
            read_options._coerce_int96_timestamp_unit

        self.init(<shared_ptr[CFileFormat]> wrapped)
        self.default_fragment_scan_options = default_fragment_scan_options

    cdef void init(self, const shared_ptr[CFileFormat]& sp):
        FileFormat.init(self, sp)
        self.parquet_format = <CParquetFileFormat*> sp.get()

    cdef WrittenFile _finish_write(self, path, base_dir,
                                   CFileWriter* file_writer):
        cdef:
            FileMetaData parquet_metadata
            CParquetFileWriter* parquet_file_writer

        parquet_metadata = None
        parquet_file_writer = dynamic_cast[_CParquetFileWriterPtr](file_writer)
        with nogil:
            metadata = deref(
                deref(parquet_file_writer).parquet_writer()).metadata()
        if metadata:
            parquet_metadata = FileMetaData()
            parquet_metadata.init(metadata)
            parquet_metadata.set_file_path(os.path.relpath(path, base_dir))

        size = GetResultValue(file_writer.GetBytesWritten())

        return WrittenFile(path, parquet_metadata, size)

    @property
    def read_options(self):
        cdef CParquetFileFormatReaderOptions* options
        options = &self.parquet_format.reader_options
        parquet_read_options = ParquetReadOptions(
            dictionary_columns={frombytes(col)
                                for col in options.dict_columns},
        )
        # Read options getter/setter works with strings so setting
        # the private property which uses the C Type
        parquet_read_options._coerce_int96_timestamp_unit = \
            options.coerce_int96_timestamp_unit
        return parquet_read_options

    def make_write_options(self, **kwargs):
        opts = FileFormat.make_write_options(self)
        (<ParquetFileWriteOptions> opts).update(**kwargs)
        return opts

    cdef _set_default_fragment_scan_options(self, FragmentScanOptions options):
        if options.type_name == 'parquet':
            self.parquet_format.default_fragment_scan_options = options.wrapped
        else:
            super()._set_default_fragment_scan_options(options)

    def equals(self, ParquetFileFormat other):
        return (
            self.read_options.equals(other.read_options) and
            self.default_fragment_scan_options ==
            other.default_fragment_scan_options
        )

    @property
    def default_extname(self):
        return "parquet"

    def __reduce__(self):
        return ParquetFileFormat, (self.read_options,
                                   self.default_fragment_scan_options)

    def __repr__(self):
        return f"<ParquetFileFormat read_options={self.read_options}>"

    def make_fragment(self, file, filesystem=None,
                      Expression partition_expression=None, row_groups=None):
        cdef:
            vector[int] c_row_groups

        if partition_expression is None:
            partition_expression = _true

        if row_groups is None:
            return super().make_fragment(file, filesystem,
                                         partition_expression)

        c_source = _make_file_source(file, filesystem)
        c_row_groups = [<int> row_group for row_group in set(row_groups)]

        c_fragment = <shared_ptr[CFragment]> GetResultValue(
            self.parquet_format.MakeFragment(move(c_source),
                                             partition_expression.unwrap(),
                                             <shared_ptr[CSchema]>nullptr,
                                             move(c_row_groups)))
        return Fragment.wrap(move(c_fragment))


class RowGroupInfo:
    """
    A wrapper class for RowGroup information

    Parameters
    ----------
    id : integer
        The group ID.
    metadata : FileMetaData
        The rowgroup metadata.
    schema : Schema
        Schema of the rows.
    """

    def __init__(self, id, metadata, schema):
        self.id = id
        self.metadata = metadata
        self.schema = schema

    @property
    def num_rows(self):
        return self.metadata.num_rows

    @property
    def total_byte_size(self):
        return self.metadata.total_byte_size

    @property
    def statistics(self):
        def name_stats(i):
            col = self.metadata.column(i)

            stats = col.statistics
            if stats is None or not stats.has_min_max:
                return None, None

            name = col.path_in_schema
            field_index = self.schema.get_field_index(name)
            if field_index < 0:
                return None, None

            typ = self.schema.field(field_index).type
            return col.path_in_schema, {
                'min': pa.scalar(stats.min, type=typ).as_py(),
                'max': pa.scalar(stats.max, type=typ).as_py()
            }

        return {
            name: stats for name, stats
            in map(name_stats, range(self.metadata.num_columns))
            if stats is not None
        }

    def __repr__(self):
        return "RowGroupInfo({})".format(self.id)

    def __eq__(self, other):
        if isinstance(other, int):
            return self.id == other
        if not isinstance(other, RowGroupInfo):
            return False
        return self.id == other.id


cdef class ParquetFileFragment(FileFragment):
    """A Fragment representing a parquet file."""

    cdef:
        CParquetFileFragment* parquet_file_fragment

    cdef void init(self, const shared_ptr[CFragment]& sp):
        FileFragment.init(self, sp)
        self.parquet_file_fragment = <CParquetFileFragment*> sp.get()

    def __reduce__(self):
        buffer = self.buffer
        # parquet_file_fragment.row_groups() is empty if the metadata
        # information of the file is not yet populated
        if not bool(self.parquet_file_fragment.row_groups()):
            row_groups = None
        else:
            row_groups = [row_group.id for row_group in self.row_groups]

        return self.format.make_fragment, (
            self.path if buffer is None else buffer,
            self.filesystem,
            self.partition_expression,
            row_groups
        )

    def ensure_complete_metadata(self):
        """
        Ensure that all metadata (statistics, physical schema, ...) have
        been read and cached in this fragment.
        """
        with nogil:
            check_status(self.parquet_file_fragment.EnsureCompleteMetadata())

    @property
    def row_groups(self):
        metadata = self.metadata
        cdef vector[int] row_groups = self.parquet_file_fragment.row_groups()
        return [RowGroupInfo(i, metadata.row_group(i), self.physical_schema)
                for i in row_groups]

    @property
    def metadata(self):
        self.ensure_complete_metadata()
        cdef FileMetaData metadata = FileMetaData()
        metadata.init(self.parquet_file_fragment.metadata())
        return metadata

    @property
    def num_row_groups(self):
        """
        Return the number of row groups viewed by this fragment (not the
        number of row groups in the origin file).
        """
        self.ensure_complete_metadata()
        return self.parquet_file_fragment.row_groups().size()

    def split_by_row_group(self, Expression filter=None,
                           Schema schema=None):
        """
        Split the fragment into multiple fragments.

        Yield a Fragment wrapping each row group in this ParquetFileFragment.
        Row groups will be excluded whose metadata contradicts the optional
        filter.

        Parameters
        ----------
        filter : Expression, default None
            Only include the row groups which satisfy this predicate (using
            the Parquet RowGroup statistics).
        schema : Schema, default None
            Schema to use when filtering row groups. Defaults to the
            Fragment's phsyical schema

        Returns
        -------
        A list of Fragments
        """
        cdef:
            vector[shared_ptr[CFragment]] c_fragments
            CExpression c_filter
            shared_ptr[CFragment] c_fragment

        schema = schema or self.physical_schema
        c_filter = _bind(filter, schema)
        with nogil:
            c_fragments = move(GetResultValue(
                self.parquet_file_fragment.SplitByRowGroup(move(c_filter))))

        return [Fragment.wrap(c_fragment) for c_fragment in c_fragments]

    def subset(self, Expression filter=None, Schema schema=None,
               object row_group_ids=None):
        """
        Create a subset of the fragment (viewing a subset of the row groups).

        Subset can be specified by either a filter predicate (with optional
        schema) or by a list of row group IDs. Note that when using a filter,
        the resulting fragment can be empty (viewing no row groups).

        Parameters
        ----------
        filter : Expression, default None
            Only include the row groups which satisfy this predicate (using
            the Parquet RowGroup statistics).
        schema : Schema, default None
            Schema to use when filtering row groups. Defaults to the
            Fragment's phsyical schema
        row_group_ids : list of ints
            The row group IDs to include in the subset. Can only be specified
            if `filter` is None.

        Returns
        -------
        ParquetFileFragment
        """
        cdef:
            CExpression c_filter
            vector[int] c_row_group_ids
            shared_ptr[CFragment] c_fragment

        if filter is not None and row_group_ids is not None:
            raise ValueError(
                "Cannot specify both 'filter' and 'row_group_ids'."
            )

        if filter is not None:
            schema = schema or self.physical_schema
            c_filter = _bind(filter, schema)
            with nogil:
                c_fragment = move(GetResultValue(
                    self.parquet_file_fragment.SubsetWithFilter(
                        move(c_filter))))
        elif row_group_ids is not None:
            c_row_group_ids = [
                <int> row_group for row_group in sorted(set(row_group_ids))
            ]
            with nogil:
                c_fragment = move(GetResultValue(
                    self.parquet_file_fragment.SubsetWithIds(
                        move(c_row_group_ids))))
        else:
            raise ValueError(
                "Need to specify one of 'filter' or 'row_group_ids'"
            )

        return Fragment.wrap(c_fragment)


cdef class ParquetReadOptions(_Weakrefable):
    """
    Parquet format specific options for reading.

    Parameters
    ----------
    dictionary_columns : list of string, default None
        Names of columns which should be dictionary encoded as
        they are read
    coerce_int96_timestamp_unit : str, default None
        Cast timestamps that are stored in INT96 format to a particular
        resolution (e.g. 'ms'). Setting to None is equivalent to 'ns'
        and therefore INT96 timestamps will be inferred as timestamps
        in nanoseconds
    """

    cdef public:
        set dictionary_columns
        TimeUnit _coerce_int96_timestamp_unit

    # Also see _PARQUET_READ_OPTIONS
    def __init__(self, dictionary_columns=None,
                 coerce_int96_timestamp_unit=None):
        self.dictionary_columns = set(dictionary_columns or set())
        self.coerce_int96_timestamp_unit = coerce_int96_timestamp_unit

    @property
    def coerce_int96_timestamp_unit(self):
        return timeunit_to_string(self._coerce_int96_timestamp_unit)

    @coerce_int96_timestamp_unit.setter
    def coerce_int96_timestamp_unit(self, unit):
        if unit is not None:
            self._coerce_int96_timestamp_unit = string_to_timeunit(unit)
        else:
            self._coerce_int96_timestamp_unit = TimeUnit_NANO

    def equals(self, ParquetReadOptions other):
        return (self.dictionary_columns == other.dictionary_columns and
                self.coerce_int96_timestamp_unit ==
                other.coerce_int96_timestamp_unit)

    def __eq__(self, other):
        try:
            return self.equals(other)
        except TypeError:
            return False

    def __repr__(self):
        return (
            f"<ParquetReadOptions"
            f" dictionary_columns={self.dictionary_columns}"
            f" coerce_int96_timestamp_unit={self.coerce_int96_timestamp_unit}>"
        )


cdef class ParquetFileWriteOptions(FileWriteOptions):

    cdef:
        CParquetFileWriteOptions* parquet_options
        object _properties

    def update(self, **kwargs):
        arrow_fields = {
            "use_deprecated_int96_timestamps",
            "coerce_timestamps",
            "allow_truncated_timestamps",
        }

        setters = set()
        for name, value in kwargs.items():
            if name not in self._properties:
                raise TypeError("unexpected parquet write option: " + name)
            self._properties[name] = value
            if name in arrow_fields:
                setters.add(self._set_arrow_properties)
            else:
                setters.add(self._set_properties)

        for setter in setters:
            setter()

    def _set_properties(self):
        cdef CParquetFileWriteOptions* opts = self.parquet_options

        opts.writer_properties = _create_writer_properties(
            use_dictionary=self._properties["use_dictionary"],
            compression=self._properties["compression"],
            version=self._properties["version"],
            write_statistics=self._properties["write_statistics"],
            data_page_size=self._properties["data_page_size"],
            compression_level=self._properties["compression_level"],
            use_byte_stream_split=(
                self._properties["use_byte_stream_split"]
            ),
            column_encoding=self._properties["column_encoding"],
            data_page_version=self._properties["data_page_version"],
        )

    def _set_arrow_properties(self):
        cdef CParquetFileWriteOptions* opts = self.parquet_options

        opts.arrow_writer_properties = _create_arrow_writer_properties(
            use_deprecated_int96_timestamps=(
                self._properties["use_deprecated_int96_timestamps"]
            ),
            coerce_timestamps=self._properties["coerce_timestamps"],
            allow_truncated_timestamps=(
                self._properties["allow_truncated_timestamps"]
            ),
            writer_engine_version="V2",
            use_compliant_nested_type=(
                self._properties["use_compliant_nested_type"]
            )
        )

    cdef void init(self, const shared_ptr[CFileWriteOptions]& sp):
        FileWriteOptions.init(self, sp)
        self.parquet_options = <CParquetFileWriteOptions*> sp.get()
        self._properties = dict(
            use_dictionary=True,
            compression="snappy",
            version="1.0",
            write_statistics=None,
            data_page_size=None,
            compression_level=None,
            use_byte_stream_split=False,
            column_encoding=None,
            data_page_version="1.0",
            use_deprecated_int96_timestamps=False,
            coerce_timestamps=None,
            allow_truncated_timestamps=False,
            use_compliant_nested_type=False,
        )
        self._set_properties()
        self._set_arrow_properties()


cdef set _PARQUET_READ_OPTIONS = {
    'dictionary_columns', 'coerce_int96_timestamp_unit'
}


cdef class ParquetFragmentScanOptions(FragmentScanOptions):
    """
    Scan-specific options for Parquet fragments.

    Parameters
    ----------
    use_buffered_stream : bool, default False
        Read files through buffered input streams rather than loading entire
        row groups at once. This may be enabled to reduce memory overhead.
        Disabled by default.
    buffer_size : int, default 8192
        Size of buffered stream, if enabled. Default is 8KB.
    pre_buffer : bool, default False
        If enabled, pre-buffer the raw Parquet data instead of issuing one
        read per column chunk. This can improve performance on high-latency
        filesystems.
    thrift_string_size_limit : int, default None
        If not None, override the maximum total string size allocated
        when decoding Thrift structures. The default limit should be
        sufficient for most Parquet files.
    thrift_container_size_limit : int, default None
        If not None, override the maximum total size of containers allocated
        when decoding Thrift structures. The default limit should be
        sufficient for most Parquet files.
    """

    cdef:
        CParquetFragmentScanOptions* parquet_options

    # Avoid mistakingly creating attributes
    __slots__ = ()

    def __init__(self, *, bint use_buffered_stream=False,
                 buffer_size=8192,
                 bint pre_buffer=False,
                 thrift_string_size_limit=None,
                 thrift_container_size_limit=None):
        self.init(shared_ptr[CFragmentScanOptions](
            new CParquetFragmentScanOptions()))
        self.use_buffered_stream = use_buffered_stream
        self.buffer_size = buffer_size
        self.pre_buffer = pre_buffer
        if thrift_string_size_limit is not None:
            self.thrift_string_size_limit = thrift_string_size_limit
        if thrift_container_size_limit is not None:
            self.thrift_container_size_limit = thrift_container_size_limit

    cdef void init(self, const shared_ptr[CFragmentScanOptions]& sp):
        FragmentScanOptions.init(self, sp)
        self.parquet_options = <CParquetFragmentScanOptions*> sp.get()

    cdef CReaderProperties* reader_properties(self):
        return self.parquet_options.reader_properties.get()

    cdef ArrowReaderProperties* arrow_reader_properties(self):
        return self.parquet_options.arrow_reader_properties.get()

    @property
    def use_buffered_stream(self):
        return self.reader_properties().is_buffered_stream_enabled()

    @use_buffered_stream.setter
    def use_buffered_stream(self, bint use_buffered_stream):
        if use_buffered_stream:
            self.reader_properties().enable_buffered_stream()
        else:
            self.reader_properties().disable_buffered_stream()

    @property
    def buffer_size(self):
        return self.reader_properties().buffer_size()

    @buffer_size.setter
    def buffer_size(self, buffer_size):
        if buffer_size <= 0:
            raise ValueError("Buffer size must be larger than zero")
        self.reader_properties().set_buffer_size(buffer_size)

    @property
    def pre_buffer(self):
        return self.arrow_reader_properties().pre_buffer()

    @pre_buffer.setter
    def pre_buffer(self, bint pre_buffer):
        self.arrow_reader_properties().set_pre_buffer(pre_buffer)

    @property
    def thrift_string_size_limit(self):
        return self.reader_properties().thrift_string_size_limit()

    @thrift_string_size_limit.setter
    def thrift_string_size_limit(self, size):
        if size <= 0:
            raise ValueError("size must be larger than zero")
        self.reader_properties().set_thrift_string_size_limit(size)

    @property
    def thrift_container_size_limit(self):
        return self.reader_properties().thrift_container_size_limit()

    @thrift_container_size_limit.setter
    def thrift_container_size_limit(self, size):
        if size <= 0:
            raise ValueError("size must be larger than zero")
        self.reader_properties().set_thrift_container_size_limit(size)

    def equals(self, ParquetFragmentScanOptions other):
        attrs = (
            self.use_buffered_stream, self.buffer_size, self.pre_buffer,
            self.thrift_string_size_limit, self.thrift_container_size_limit)
        other_attrs = (
            other.use_buffered_stream, other.buffer_size, other.pre_buffer,
            other.thrift_string_size_limit,
            other.thrift_container_size_limit)
        return attrs == other_attrs

    @classmethod
    def _reconstruct(cls, kwargs):
        return cls(**kwargs)

    def __reduce__(self):
        kwargs = dict(
            use_buffered_stream=self.use_buffered_stream,
            buffer_size=self.buffer_size,
            pre_buffer=self.pre_buffer,
            thrift_string_size_limit=self.thrift_string_size_limit,
            thrift_container_size_limit=self.thrift_container_size_limit,
        )
        return type(self)._reconstruct, (kwargs,)


cdef class ParquetFactoryOptions(_Weakrefable):
    """
    Influences the discovery of parquet dataset.

    Parameters
    ----------
    partition_base_dir : str, optional
        For the purposes of applying the partitioning, paths will be
        stripped of the partition_base_dir. Files not matching the
        partition_base_dir prefix will be skipped for partitioning discovery.
        The ignored files will still be part of the Dataset, but will not
        have partition information.
    partitioning : Partitioning, PartitioningFactory, optional
        The partitioning scheme applied to fragments, see ``Partitioning``.
    validate_column_chunk_paths : bool, default False
        Assert that all ColumnChunk paths are consistent. The parquet spec
        allows for ColumnChunk data to be stored in multiple files, but
        ParquetDatasetFactory supports only a single file with all ColumnChunk
        data. If this flag is set construction of a ParquetDatasetFactory will
        raise an error if ColumnChunk data is not resident in a single file.
    """

    cdef:
        CParquetFactoryOptions options

    __slots__ = ()  # avoid mistakingly creating attributes

    def __init__(self, partition_base_dir=None, partitioning=None,
                 validate_column_chunk_paths=False):
        if isinstance(partitioning, PartitioningFactory):
            self.partitioning_factory = partitioning
        elif isinstance(partitioning, Partitioning):
            self.partitioning = partitioning

        if partition_base_dir is not None:
            self.partition_base_dir = partition_base_dir

        self.options.validate_column_chunk_paths = validate_column_chunk_paths

    cdef inline CParquetFactoryOptions unwrap(self):
        return self.options

    @property
    def partitioning(self):
        """Partitioning to apply to discovered files.

        NOTE: setting this property will overwrite partitioning_factory.
        """
        c_partitioning = self.options.partitioning.partitioning()
        if c_partitioning.get() == nullptr:
            return None
        return Partitioning.wrap(c_partitioning)

    @partitioning.setter
    def partitioning(self, Partitioning value):
        self.options.partitioning = (<Partitioning> value).unwrap()

    @property
    def partitioning_factory(self):
        """PartitioningFactory to apply to discovered files and
        discover a Partitioning.

        NOTE: setting this property will overwrite partitioning.
        """
        c_factory = self.options.partitioning.factory()
        if c_factory.get() == nullptr:
            return None
        return PartitioningFactory.wrap(c_factory)

    @partitioning_factory.setter
    def partitioning_factory(self, PartitioningFactory value):
        self.options.partitioning = (<PartitioningFactory> value).unwrap()

    @property
    def partition_base_dir(self):
        """
        Base directory to strip paths before applying the partitioning.
        """
        return frombytes(self.options.partition_base_dir)

    @partition_base_dir.setter
    def partition_base_dir(self, value):
        self.options.partition_base_dir = tobytes(value)

    @property
    def validate_column_chunk_paths(self):
        """
        Base directory to strip paths before applying the partitioning.
        """
        return self.options.validate_column_chunk_paths

    @validate_column_chunk_paths.setter
    def validate_column_chunk_paths(self, value):
        self.options.validate_column_chunk_paths = value


cdef class ParquetDatasetFactory(DatasetFactory):
    """
    Create a ParquetDatasetFactory from a Parquet `_metadata` file.

    Parameters
    ----------
    metadata_path : str
        Path to the `_metadata` parquet metadata-only file generated with
        `pyarrow.parquet.write_metadata`.
    filesystem : pyarrow.fs.FileSystem
        Filesystem to read the metadata_path from, and subsequent parquet
        files.
    format : ParquetFileFormat
        Parquet format options.
    options : ParquetFactoryOptions, optional
        Various flags influencing the discovery of filesystem paths.
    """

    cdef:
        CParquetDatasetFactory* parquet_factory

    def __init__(self, metadata_path, FileSystem filesystem not None,
                 FileFormat format not None,
                 ParquetFactoryOptions options=None):
        cdef:
            c_string c_path
            shared_ptr[CFileSystem] c_filesystem
            shared_ptr[CParquetFileFormat] c_format
            CResult[shared_ptr[CDatasetFactory]] result
            CParquetFactoryOptions c_options

        c_path = tobytes(metadata_path)
        c_filesystem = filesystem.unwrap()
        c_format = static_pointer_cast[CParquetFileFormat, CFileFormat](
            format.unwrap())
        options = options or ParquetFactoryOptions()
        c_options = options.unwrap()

        with nogil:
            result = CParquetDatasetFactory.MakeFromMetaDataPath(
                c_path, c_filesystem, c_format, c_options)
        self.init(GetResultValue(result))

    cdef init(self, shared_ptr[CDatasetFactory]& sp):
        DatasetFactory.init(self, sp)
        self.parquet_factory = <CParquetDatasetFactory*> sp.get()

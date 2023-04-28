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

"""Dataset is currently unstable. APIs subject to change without notice."""

import pyarrow as pa
from pyarrow.util import _is_iterable, _stringify_path, _is_path_like

from pyarrow._dataset import (  # noqa
    CsvFileFormat,
    CsvFragmentScanOptions,
    Dataset,
    DatasetFactory,
    DirectoryPartitioning,
    FeatherFileFormat,
    FilenamePartitioning,
    FileFormat,
    FileFragment,
    FileSystemDataset,
    FileSystemDatasetFactory,
    FileSystemFactoryOptions,
    FileWriteOptions,
    Fragment,
    FragmentScanOptions,
    HivePartitioning,
    IpcFileFormat,
    IpcFileWriteOptions,
    InMemoryDataset,
    Partitioning,
    PartitioningFactory,
    Scanner,
    TaggedRecordBatch,
    UnionDataset,
    UnionDatasetFactory,
    WrittenFile,
    _get_partition_keys,
    _filesystemdataset_write,
)
# keep Expression functionality exposed here for backwards compatibility
from pyarrow.compute import Expression, scalar, field  # noqa


_orc_available = False
_orc_msg = (
    "The pyarrow installation is not built with support for the ORC file "
    "format."
)

try:
    from pyarrow._dataset_orc import OrcFileFormat
    _orc_available = True
except ImportError:
    pass

_parquet_available = False
_parquet_msg = (
    "The pyarrow installation is not built with support for the Parquet file "
    "format."
)

try:
    from pyarrow._dataset_parquet import (  # noqa
        ParquetDatasetFactory,
        ParquetFactoryOptions,
        ParquetFileFormat,
        ParquetFileFragment,
        ParquetFileWriteOptions,
        ParquetFragmentScanOptions,
        ParquetReadOptions,
        RowGroupInfo,
    )
    _parquet_available = True
except ImportError:
    pass


def __getattr__(name):
    if name == "OrcFileFormat" and not _orc_available:
        raise ImportError(_orc_msg)

    if name == "ParquetFileFormat" and not _parquet_available:
        raise ImportError(_parquet_msg)

    raise AttributeError(
        "module 'pyarrow.dataset' has no attribute '{0}'".format(name)
    )


def partitioning(schema=None, field_names=None, flavor=None,
                 dictionaries=None):
    """
    Specify a partitioning scheme.

    The supported schemes include:

    - "DirectoryPartitioning": this scheme expects one segment in the file path
      for each field in the specified schema (all fields are required to be
      present). For example given schema<year:int16, month:int8> the path
      "/2009/11" would be parsed to ("year"_ == 2009 and "month"_ == 11).
    - "HivePartitioning": a scheme for "/$key=$value/" nested directories as
      found in Apache Hive. This is a multi-level, directory based partitioning
      scheme. Data is partitioned by static values of a particular column in
      the schema. Partition keys are represented in the form $key=$value in
      directory names. Field order is ignored, as are missing or unrecognized
      field names.
      For example, given schema<year:int16, month:int8, day:int8>, a possible
      path would be "/year=2009/month=11/day=15" (but the field order does not
      need to match).
    - "FilenamePartitioning": this scheme expects the partitions will have
      filenames containing the field values separated by "_".
      For example, given schema<year:int16, month:int8, day:int8>, a possible
      partition filename "2009_11_part-0.parquet" would be parsed
      to ("year"_ == 2009 and "month"_ == 11).

    Parameters
    ----------
    schema : pyarrow.Schema, default None
        The schema that describes the partitions present in the file path.
        If not specified, and `field_names` and/or `flavor` are specified,
        the schema will be inferred from the file path (and a
        PartitioningFactory is returned).
    field_names :  list of str, default None
        A list of strings (field names). If specified, the schema's types are
        inferred from the file paths (only valid for DirectoryPartitioning).
    flavor : str, default None
        The default is DirectoryPartitioning. Specify ``flavor="hive"`` for
        a HivePartitioning, and ``flavor="filename"`` for a
        FilenamePartitioning.
    dictionaries : dict[str, Array]
        If the type of any field of `schema` is a dictionary type, the
        corresponding entry of `dictionaries` must be an array containing
        every value which may be taken by the corresponding column or an
        error will be raised in parsing. Alternatively, pass `infer` to have
        Arrow discover the dictionary values, in which case a
        PartitioningFactory is returned.

    Returns
    -------
    Partitioning or PartitioningFactory
        The partioning scheme

    Examples
    --------

    Specify the Schema for paths like "/2009/June":

    >>> import pyarrow as pa
    >>> import pyarrow.dataset as ds
    >>> part = ds.partitioning(pa.schema([("year", pa.int16()),
    ...                                   ("month", pa.string())]))

    or let the types be inferred by only specifying the field names:

    >>> part =  ds.partitioning(field_names=["year", "month"])

    For paths like "/2009/June", the year will be inferred as int32 while month
    will be inferred as string.

    Specify a Schema with dictionary encoding, providing dictionary values:

    >>> part = ds.partitioning(
    ...     pa.schema([
    ...         ("year", pa.int16()),
    ...         ("month", pa.dictionary(pa.int8(), pa.string()))
    ...     ]),
    ...     dictionaries={
    ...         "month": pa.array(["January", "February", "March"]),
    ...     })

    Alternatively, specify a Schema with dictionary encoding, but have Arrow
    infer the dictionary values:

    >>> part = ds.partitioning(
    ...     pa.schema([
    ...         ("year", pa.int16()),
    ...         ("month", pa.dictionary(pa.int8(), pa.string()))
    ...     ]),
    ...     dictionaries="infer")

    Create a Hive scheme for a path like "/year=2009/month=11":

    >>> part = ds.partitioning(
    ...     pa.schema([("year", pa.int16()), ("month", pa.int8())]),
    ...     flavor="hive")

    A Hive scheme can also be discovered from the directory structure (and
    types will be inferred):

    >>> part = ds.partitioning(flavor="hive")
    """
    if flavor is None:
        # default flavor
        if schema is not None:
            if field_names is not None:
                raise ValueError(
                    "Cannot specify both 'schema' and 'field_names'")
            if dictionaries == 'infer':
                return DirectoryPartitioning.discover(schema=schema)
            return DirectoryPartitioning(schema, dictionaries)
        elif field_names is not None:
            if isinstance(field_names, list):
                return DirectoryPartitioning.discover(field_names)
            else:
                raise ValueError(
                    "Expected list of field names, got {}".format(
                        type(field_names)))
        else:
            raise ValueError(
                "For the default directory flavor, need to specify "
                "a Schema or a list of field names")
    if flavor == "filename":
        if schema is not None:
            if field_names is not None:
                raise ValueError(
                    "Cannot specify both 'schema' and 'field_names'")
            if dictionaries == 'infer':
                return FilenamePartitioning.discover(schema=schema)
            return FilenamePartitioning(schema, dictionaries)
        elif field_names is not None:
            if isinstance(field_names, list):
                return FilenamePartitioning.discover(field_names)
            else:
                raise ValueError(
                    "Expected list of field names, got {}".format(
                        type(field_names)))
        else:
            raise ValueError(
                "For the filename flavor, need to specify "
                "a Schema or a list of field names")
    elif flavor == 'hive':
        if field_names is not None:
            raise ValueError("Cannot specify 'field_names' for flavor 'hive'")
        elif schema is not None:
            if isinstance(schema, pa.Schema):
                if dictionaries == 'infer':
                    return HivePartitioning.discover(schema=schema)
                return HivePartitioning(schema, dictionaries)
            else:
                raise ValueError(
                    "Expected Schema for 'schema', got {}".format(
                        type(schema)))
        else:
            return HivePartitioning.discover()
    else:
        raise ValueError("Unsupported flavor")


def _ensure_partitioning(scheme):
    """
    Validate input and return a Partitioning(Factory).

    It passes None through if no partitioning scheme is defined.
    """
    if scheme is None:
        pass
    elif isinstance(scheme, str):
        scheme = partitioning(flavor=scheme)
    elif isinstance(scheme, list):
        scheme = partitioning(field_names=scheme)
    elif isinstance(scheme, (Partitioning, PartitioningFactory)):
        pass
    else:
        ValueError("Expected Partitioning or PartitioningFactory, got {}"
                   .format(type(scheme)))
    return scheme


def _ensure_format(obj):
    if isinstance(obj, FileFormat):
        return obj
    elif obj == "parquet":
        if not _parquet_available:
            raise ValueError(_parquet_msg)
        return ParquetFileFormat()
    elif obj in {"ipc", "arrow"}:
        return IpcFileFormat()
    elif obj == "feather":
        return FeatherFileFormat()
    elif obj == "csv":
        return CsvFileFormat()
    elif obj == "orc":
        if not _orc_available:
            raise ValueError(_orc_msg)
        return OrcFileFormat()
    else:
        raise ValueError("format '{}' is not supported".format(obj))


def _ensure_multiple_sources(paths, filesystem=None):
    """
    Treat a list of paths as files belonging to a single file system

    If the file system is local then also validates that all paths
    are referencing existing *files* otherwise any non-file paths will be
    silently skipped (for example on a remote filesystem).

    Parameters
    ----------
    paths : list of path-like
        Note that URIs are not allowed.
    filesystem : FileSystem or str, optional
        If an URI is passed, then its path component will act as a prefix for
        the file paths.

    Returns
    -------
    (FileSystem, list of str)
        File system object and a list of normalized paths.

    Raises
    ------
    TypeError
        If the passed filesystem has wrong type.
    IOError
        If the file system is local and a referenced path is not available or
        not a file.
    """
    from pyarrow.fs import (
        LocalFileSystem, SubTreeFileSystem, _MockFileSystem, FileType,
        _ensure_filesystem
    )

    if filesystem is None:
        # fall back to local file system as the default
        filesystem = LocalFileSystem()
    else:
        # construct a filesystem if it is a valid URI
        filesystem = _ensure_filesystem(filesystem)

    is_local = (
        isinstance(filesystem, (LocalFileSystem, _MockFileSystem)) or
        (isinstance(filesystem, SubTreeFileSystem) and
         isinstance(filesystem.base_fs, LocalFileSystem))
    )

    # allow normalizing irregular paths such as Windows local paths
    paths = [filesystem.normalize_path(_stringify_path(p)) for p in paths]

    # validate that all of the paths are pointing to existing *files*
    # possible improvement is to group the file_infos by type and raise for
    # multiple paths per error category
    if is_local:
        for info in filesystem.get_file_info(paths):
            file_type = info.type
            if file_type == FileType.File:
                continue
            elif file_type == FileType.NotFound:
                raise FileNotFoundError(info.path)
            elif file_type == FileType.Directory:
                raise IsADirectoryError(
                    'Path {} points to a directory, but only file paths are '
                    'supported. To construct a nested or union dataset pass '
                    'a list of dataset objects instead.'.format(info.path)
                )
            else:
                raise IOError(
                    'Path {} exists but its type is unknown (could be a '
                    'special file such as a Unix socket or character device, '
                    'or Windows NUL / CON / ...)'.format(info.path)
                )

    return filesystem, paths


def _ensure_single_source(path, filesystem=None):
    """
    Treat path as either a recursively traversable directory or a single file.

    Parameters
    ----------
    path : path-like
    filesystem : FileSystem or str, optional
        If an URI is passed, then its path component will act as a prefix for
        the file paths.

    Returns
    -------
    (FileSystem, list of str or fs.Selector)
        File system object and either a single item list pointing to a file or
        an fs.Selector object pointing to a directory.

    Raises
    ------
    TypeError
        If the passed filesystem has wrong type.
    FileNotFoundError
        If the referenced file or directory doesn't exist.
    """
    from pyarrow.fs import FileType, FileSelector, _resolve_filesystem_and_path

    # at this point we already checked that `path` is a path-like
    filesystem, path = _resolve_filesystem_and_path(path, filesystem)

    # ensure that the path is normalized before passing to dataset discovery
    path = filesystem.normalize_path(path)

    # retrieve the file descriptor
    file_info = filesystem.get_file_info(path)

    # depending on the path type either return with a recursive
    # directory selector or as a list containing a single file
    if file_info.type == FileType.Directory:
        paths_or_selector = FileSelector(path, recursive=True)
    elif file_info.type == FileType.File:
        paths_or_selector = [path]
    else:
        raise FileNotFoundError(path)

    return filesystem, paths_or_selector


def _filesystem_dataset(source, schema=None, filesystem=None,
                        partitioning=None, format=None,
                        partition_base_dir=None, exclude_invalid_files=None,
                        selector_ignore_prefixes=None):
    """
    Create a FileSystemDataset which can be used to build a Dataset.

    Parameters are documented in the dataset function.

    Returns
    -------
    FileSystemDataset
    """
    format = _ensure_format(format or 'parquet')
    partitioning = _ensure_partitioning(partitioning)

    if isinstance(source, (list, tuple)):
        fs, paths_or_selector = _ensure_multiple_sources(source, filesystem)
    else:
        fs, paths_or_selector = _ensure_single_source(source, filesystem)

    options = FileSystemFactoryOptions(
        partitioning=partitioning,
        partition_base_dir=partition_base_dir,
        exclude_invalid_files=exclude_invalid_files,
        selector_ignore_prefixes=selector_ignore_prefixes
    )
    factory = FileSystemDatasetFactory(fs, paths_or_selector, format, options)

    return factory.finish(schema)


def _in_memory_dataset(source, schema=None, **kwargs):
    if any(v is not None for v in kwargs.values()):
        raise ValueError(
            "For in-memory datasets, you cannot pass any additional arguments")
    return InMemoryDataset(source, schema)


def _union_dataset(children, schema=None, **kwargs):
    if any(v is not None for v in kwargs.values()):
        raise ValueError(
            "When passing a list of Datasets, you cannot pass any additional "
            "arguments"
        )

    if schema is None:
        # unify the children datasets' schemas
        schema = pa.unify_schemas([child.schema for child in children])

    for child in children:
        if getattr(child, "_scan_options", None):
            raise ValueError(
                "Creating an UnionDataset from filtered or projected Datasets "
                "is currently not supported. Union the unfiltered datasets "
                "and apply the filter to the resulting union."
            )

    # create datasets with the requested schema
    children = [child.replace_schema(schema) for child in children]

    return UnionDataset(schema, children)


def parquet_dataset(metadata_path, schema=None, filesystem=None, format=None,
                    partitioning=None, partition_base_dir=None):
    """
    Create a FileSystemDataset from a `_metadata` file created via
    `pyarrrow.parquet.write_metadata`.

    Parameters
    ----------
    metadata_path : path,
        Path pointing to a single file parquet metadata file
    schema : Schema, optional
        Optionally provide the Schema for the Dataset, in which case it will
        not be inferred from the source.
    filesystem : FileSystem or URI string, default None
        If a single path is given as source and filesystem is None, then the
        filesystem will be inferred from the path.
        If an URI string is passed, then a filesystem object is constructed
        using the URI's optional path component as a directory prefix. See the
        examples below.
        Note that the URIs on Windows must follow 'file:///C:...' or
        'file:/C:...' patterns.
    format : ParquetFileFormat
        An instance of a ParquetFileFormat if special options needs to be
        passed.
    partitioning : Partitioning, PartitioningFactory, str, list of str
        The partitioning scheme specified with the ``partitioning()``
        function. A flavor string can be used as shortcut, and with a list of
        field names a DirectionaryPartitioning will be inferred.
    partition_base_dir : str, optional
        For the purposes of applying the partitioning, paths will be
        stripped of the partition_base_dir. Files not matching the
        partition_base_dir prefix will be skipped for partitioning discovery.
        The ignored files will still be part of the Dataset, but will not
        have partition information.

    Returns
    -------
    FileSystemDataset
        The dataset corresponding to the given metadata
    """
    from pyarrow.fs import LocalFileSystem, _ensure_filesystem

    if format is None:
        format = ParquetFileFormat()
    elif not isinstance(format, ParquetFileFormat):
        raise ValueError("format argument must be a ParquetFileFormat")

    if filesystem is None:
        filesystem = LocalFileSystem()
    else:
        filesystem = _ensure_filesystem(filesystem)

    metadata_path = filesystem.normalize_path(_stringify_path(metadata_path))
    options = ParquetFactoryOptions(
        partition_base_dir=partition_base_dir,
        partitioning=_ensure_partitioning(partitioning)
    )

    factory = ParquetDatasetFactory(
        metadata_path, filesystem, format, options=options)
    return factory.finish(schema)


def dataset(source, schema=None, format=None, filesystem=None,
            partitioning=None, partition_base_dir=None,
            exclude_invalid_files=None, ignore_prefixes=None):
    """
    Open a dataset.

    Datasets provides functionality to efficiently work with tabular,
    potentially larger than memory and multi-file dataset.

    - A unified interface for different sources, like Parquet and Feather
    - Discovery of sources (crawling directories, handle directory-based
      partitioned datasets, basic schema normalization)
    - Optimized reading with predicate pushdown (filtering rows), projection
      (selecting columns), parallel reading or fine-grained managing of tasks.

    Note that this is the high-level API, to have more control over the dataset
    construction use the low-level API classes (FileSystemDataset,
    FilesystemDatasetFactory, etc.)

    Parameters
    ----------
    source : path, list of paths, dataset, list of datasets, (list of) \
RecordBatch or Table, iterable of RecordBatch, RecordBatchReader, or URI
        Path pointing to a single file:
            Open a FileSystemDataset from a single file.
        Path pointing to a directory:
            The directory gets discovered recursively according to a
            partitioning scheme if given.
        List of file paths:
            Create a FileSystemDataset from explicitly given files. The files
            must be located on the same filesystem given by the filesystem
            parameter.
            Note that in contrary of construction from a single file, passing
            URIs as paths is not allowed.
        List of datasets:
            A nested UnionDataset gets constructed, it allows arbitrary
            composition of other datasets.
            Note that additional keyword arguments are not allowed.
        (List of) batches or tables, iterable of batches, or RecordBatchReader:
            Create an InMemoryDataset. If an iterable or empty list is given,
            a schema must also be given. If an iterable or RecordBatchReader
            is given, the resulting dataset can only be scanned once; further
            attempts will raise an error.
    schema : Schema, optional
        Optionally provide the Schema for the Dataset, in which case it will
        not be inferred from the source.
    format : FileFormat or str
        Currently "parquet", "ipc"/"arrow"/"feather", "csv", and "orc" are
        supported. For Feather, only version 2 files are supported.
    filesystem : FileSystem or URI string, default None
        If a single path is given as source and filesystem is None, then the
        filesystem will be inferred from the path.
        If an URI string is passed, then a filesystem object is constructed
        using the URI's optional path component as a directory prefix. See the
        examples below.
        Note that the URIs on Windows must follow 'file:///C:...' or
        'file:/C:...' patterns.
    partitioning : Partitioning, PartitioningFactory, str, list of str
        The partitioning scheme specified with the ``partitioning()``
        function. A flavor string can be used as shortcut, and with a list of
        field names a DirectionaryPartitioning will be inferred.
    partition_base_dir : str, optional
        For the purposes of applying the partitioning, paths will be
        stripped of the partition_base_dir. Files not matching the
        partition_base_dir prefix will be skipped for partitioning discovery.
        The ignored files will still be part of the Dataset, but will not
        have partition information.
    exclude_invalid_files : bool, optional (default True)
        If True, invalid files will be excluded (file format specific check).
        This will incur IO for each files in a serial and single threaded
        fashion. Disabling this feature will skip the IO, but unsupported
        files may be present in the Dataset (resulting in an error at scan
        time).
    ignore_prefixes : list, optional
        Files matching any of these prefixes will be ignored by the
        discovery process. This is matched to the basename of a path.
        By default this is ['.', '_'].
        Note that discovery happens only if a directory is passed as source.

    Returns
    -------
    dataset : Dataset
        Either a FileSystemDataset or a UnionDataset depending on the source
        parameter.

    Examples
    --------
    Creating an example Table:

    >>> import pyarrow as pa
    >>> import pyarrow.parquet as pq
    >>> table = pa.table({'year': [2020, 2022, 2021, 2022, 2019, 2021],
    ...                   'n_legs': [2, 2, 4, 4, 5, 100],
    ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
    ...                              "Brittle stars", "Centipede"]})
    >>> pq.write_table(table, "file.parquet")

    Opening a single file:

    >>> import pyarrow.dataset as ds
    >>> dataset = ds.dataset("file.parquet", format="parquet")
    >>> dataset.to_table()
    pyarrow.Table
    year: int64
    n_legs: int64
    animal: string
    ----
    year: [[2020,2022,2021,2022,2019,2021]]
    n_legs: [[2,2,4,4,5,100]]
    animal: [["Flamingo","Parrot","Dog","Horse","Brittle stars","Centipede"]]

    Opening a single file with an explicit schema:

    >>> myschema = pa.schema([
    ...     ('n_legs', pa.int64()),
    ...     ('animal', pa.string())])
    >>> dataset = ds.dataset("file.parquet", schema=myschema, format="parquet")
    >>> dataset.to_table()
    pyarrow.Table
    n_legs: int64
    animal: string
    ----
    n_legs: [[2,2,4,4,5,100]]
    animal: [["Flamingo","Parrot","Dog","Horse","Brittle stars","Centipede"]]

    Opening a dataset for a single directory:

    >>> ds.write_dataset(table, "partitioned_dataset", format="parquet",
    ...                  partitioning=['year'])
    >>> dataset = ds.dataset("partitioned_dataset", format="parquet")
    >>> dataset.to_table()
    pyarrow.Table
    n_legs: int64
    animal: string
    ----
    n_legs: [[5],[2],[4,100],[2,4]]
    animal: [["Brittle stars"],["Flamingo"],...["Parrot","Horse"]]

    For a single directory from a S3 bucket:

    >>> ds.dataset("s3://mybucket/nyc-taxi/",
    ...            format="parquet") # doctest: +SKIP

    Opening a dataset from a list of relatives local paths:

    >>> dataset = ds.dataset([
    ...     "partitioned_dataset/2019/part-0.parquet",
    ...     "partitioned_dataset/2020/part-0.parquet",
    ...     "partitioned_dataset/2021/part-0.parquet",
    ... ], format='parquet')
    >>> dataset.to_table()
    pyarrow.Table
    n_legs: int64
    animal: string
    ----
    n_legs: [[5],[2],[4,100]]
    animal: [["Brittle stars"],["Flamingo"],["Dog","Centipede"]]

    With filesystem provided:

    >>> paths = [
    ...     'part0/data.parquet',
    ...     'part1/data.parquet',
    ...     'part3/data.parquet',
    ... ]
    >>> ds.dataset(paths, filesystem='file:///directory/prefix,
    ...            format='parquet') # doctest: +SKIP

    Which is equivalent with:

    >>> fs = SubTreeFileSystem("/directory/prefix",
    ...                        LocalFileSystem()) # doctest: +SKIP
    >>> ds.dataset(paths, filesystem=fs, format='parquet') # doctest: +SKIP

    With a remote filesystem URI:

    >>> paths = [
    ...     'nested/directory/part0/data.parquet',
    ...     'nested/directory/part1/data.parquet',
    ...     'nested/directory/part3/data.parquet',
    ... ]
    >>> ds.dataset(paths, filesystem='s3://bucket/',
    ...            format='parquet') # doctest: +SKIP

    Similarly to the local example, the directory prefix may be included in the
    filesystem URI:

    >>> ds.dataset(paths, filesystem='s3://bucket/nested/directory',
    ...         format='parquet') # doctest: +SKIP

    Construction of a nested dataset:

    >>> ds.dataset([
    ...     dataset("s3://old-taxi-data", format="parquet"),
    ...     dataset("local/path/to/data", format="ipc")
    ... ]) # doctest: +SKIP
    """
    # collect the keyword arguments for later reuse
    kwargs = dict(
        schema=schema,
        filesystem=filesystem,
        partitioning=partitioning,
        format=format,
        partition_base_dir=partition_base_dir,
        exclude_invalid_files=exclude_invalid_files,
        selector_ignore_prefixes=ignore_prefixes
    )

    if _is_path_like(source):
        return _filesystem_dataset(source, **kwargs)
    elif isinstance(source, (tuple, list)):
        if all(_is_path_like(elem) for elem in source):
            return _filesystem_dataset(source, **kwargs)
        elif all(isinstance(elem, Dataset) for elem in source):
            return _union_dataset(source, **kwargs)
        elif all(isinstance(elem, (pa.RecordBatch, pa.Table))
                 for elem in source):
            return _in_memory_dataset(source, **kwargs)
        else:
            unique_types = set(type(elem).__name__ for elem in source)
            type_names = ', '.join('{}'.format(t) for t in unique_types)
            raise TypeError(
                'Expected a list of path-like or dataset objects, or a list '
                'of batches or tables. The given list contains the following '
                'types: {}'.format(type_names)
            )
    elif isinstance(source, (pa.RecordBatch, pa.Table)):
        return _in_memory_dataset(source, **kwargs)
    else:
        raise TypeError(
            'Expected a path-like, list of path-likes or a list of Datasets '
            'instead of the given type: {}'.format(type(source).__name__)
        )


def _ensure_write_partitioning(part, schema, flavor):
    if isinstance(part, PartitioningFactory):
        raise ValueError("A PartitioningFactory cannot be used. "
                         "Did you call the partitioning function "
                         "without supplying a schema?")

    if isinstance(part, Partitioning) and flavor:
        raise ValueError(
            "Providing a partitioning_flavor with "
            "a Partitioning object is not supported"
        )
    elif isinstance(part, (tuple, list)):
        # Name of fields were provided instead of a partitioning object.
        # Create a partitioning factory with those field names.
        part = partitioning(
            schema=pa.schema([schema.field(f) for f in part]),
            flavor=flavor
        )
    elif part is None:
        part = partitioning(pa.schema([]), flavor=flavor)

    if not isinstance(part, Partitioning):
        raise ValueError(
            "partitioning must be a Partitioning object or "
            "a list of column names"
        )

    return part


def write_dataset(data, base_dir, *, basename_template=None, format=None,
                  partitioning=None, partitioning_flavor=None, schema=None,
                  filesystem=None, file_options=None, use_threads=True,
                  max_partitions=None, max_open_files=None,
                  max_rows_per_file=None, min_rows_per_group=None,
                  max_rows_per_group=None, file_visitor=None,
                  existing_data_behavior='error', create_dir=True):
    """
    Write a dataset to a given format and partitioning.

    Parameters
    ----------
    data : Dataset, Table/RecordBatch, RecordBatchReader, list of \
Table/RecordBatch, or iterable of RecordBatch
        The data to write. This can be a Dataset instance or
        in-memory Arrow data. If an iterable is given, the schema must
        also be given.
    base_dir : str
        The root directory where to write the dataset.
    basename_template : str, optional
        A template string used to generate basenames of written data files.
        The token '{i}' will be replaced with an automatically incremented
        integer. If not specified, it defaults to
        "part-{i}." + format.default_extname
    format : FileFormat or str
        The format in which to write the dataset. Currently supported:
        "parquet", "ipc"/"arrow"/"feather", and "csv". If a FileSystemDataset
        is being written and `format` is not specified, it defaults to the
        same format as the specified FileSystemDataset. When writing a
        Table or RecordBatch, this keyword is required.
    partitioning : Partitioning or list[str], optional
        The partitioning scheme specified with the ``partitioning()``
        function or a list of field names. When providing a list of
        field names, you can use ``partitioning_flavor`` to drive which
        partitioning type should be used.
    partitioning_flavor : str, optional
        One of the partitioning flavors supported by
        ``pyarrow.dataset.partitioning``. If omitted will use the
        default of ``partitioning()`` which is directory partitioning.
    schema : Schema, optional
    filesystem : FileSystem, optional
    file_options : pyarrow.dataset.FileWriteOptions, optional
        FileFormat specific write options, created using the
        ``FileFormat.make_write_options()`` function.
    use_threads : bool, default True
        Write files in parallel. If enabled, then maximum parallelism will be
        used determined by the number of available CPU cores.
    max_partitions : int, default 1024
        Maximum number of partitions any batch may be written into.
    max_open_files : int, default 1024
        If greater than 0 then this will limit the maximum number of
        files that can be left open. If an attempt is made to open
        too many files then the least recently used file will be closed.
        If this setting is set too low you may end up fragmenting your
        data into many small files.
    max_rows_per_file : int, default 0
        Maximum number of rows per file. If greater than 0 then this will
        limit how many rows are placed in any single file. Otherwise there
        will be no limit and one file will be created in each output
        directory unless files need to be closed to respect max_open_files
    min_rows_per_group : int, default 0
        Minimum number of rows per group. When the value is greater than 0,
        the dataset writer will batch incoming data and only write the row
        groups to the disk when sufficient rows have accumulated.
    max_rows_per_group : int, default 1024 * 1024
        Maximum number of rows per group. If the value is greater than 0,
        then the dataset writer may split up large incoming batches into
        multiple row groups.  If this value is set, then min_rows_per_group
        should also be set. Otherwise it could end up with very small row
        groups.
    file_visitor : function
        If set, this function will be called with a WrittenFile instance
        for each file created during the call.  This object will have both
        a path attribute and a metadata attribute.

        The path attribute will be a string containing the path to
        the created file.

        The metadata attribute will be the parquet metadata of the file.
        This metadata will have the file path attribute set and can be used
        to build a _metadata file.  The metadata attribute will be None if
        the format is not parquet.

        Example visitor which simple collects the filenames created::

            visited_paths = []

            def file_visitor(written_file):
                visited_paths.append(written_file.path)
    existing_data_behavior : 'error' | 'overwrite_or_ignore' | \
'delete_matching'
        Controls how the dataset will handle data that already exists in
        the destination.  The default behavior ('error') is to raise an error
        if any data exists in the destination.

        'overwrite_or_ignore' will ignore any existing data and will
        overwrite files with the same name as an output file.  Other
        existing files will be ignored.  This behavior, in combination
        with a unique basename_template for each write, will allow for
        an append workflow.

        'delete_matching' is useful when you are writing a partitioned
        dataset.  The first time each partition directory is encountered
        the entire directory will be deleted.  This allows you to overwrite
        old partitions completely.
    create_dir : bool, default True
        If False, directories will not be created.  This can be useful for
        filesystems that do not require directories.
    """
    from pyarrow.fs import _resolve_filesystem_and_path

    if isinstance(data, (list, tuple)):
        schema = schema or data[0].schema
        data = InMemoryDataset(data, schema=schema)
    elif isinstance(data, (pa.RecordBatch, pa.Table)):
        schema = schema or data.schema
        data = InMemoryDataset(data, schema=schema)
    elif isinstance(data, pa.ipc.RecordBatchReader) or _is_iterable(data):
        data = Scanner.from_batches(data, schema=schema)
        schema = None
    elif not isinstance(data, (Dataset, Scanner)):
        raise ValueError(
            "Only Dataset, Scanner, Table/RecordBatch, RecordBatchReader, "
            "a list of Tables/RecordBatches, or iterable of batches are "
            "supported."
        )

    if format is None and isinstance(data, FileSystemDataset):
        format = data.format
    else:
        format = _ensure_format(format)

    if file_options is None:
        file_options = format.make_write_options()

    if format != file_options.format:
        raise TypeError("Supplied FileWriteOptions have format {}, "
                        "which doesn't match supplied FileFormat {}".format(
                            format, file_options))

    if basename_template is None:
        basename_template = "part-{i}." + format.default_extname

    if max_partitions is None:
        max_partitions = 1024

    if max_open_files is None:
        max_open_files = 1024

    if max_rows_per_file is None:
        max_rows_per_file = 0

    if max_rows_per_group is None:
        max_rows_per_group = 1 << 20

    if min_rows_per_group is None:
        min_rows_per_group = 0

    # at this point data is a Scanner or a Dataset, anything else
    # was converted to one of those two. So we can grab the schema
    # to build the partitioning object from Dataset.
    if isinstance(data, Scanner):
        partitioning_schema = data.projected_schema
    else:
        partitioning_schema = data.schema
    partitioning = _ensure_write_partitioning(partitioning,
                                              schema=partitioning_schema,
                                              flavor=partitioning_flavor)

    filesystem, base_dir = _resolve_filesystem_and_path(base_dir, filesystem)

    if isinstance(data, Dataset):
        scanner = data.scanner(use_threads=use_threads)
    else:
        # scanner was passed directly by the user, in which case a schema
        # cannot be passed
        if schema is not None:
            raise ValueError("Cannot specify a schema when writing a Scanner")
        scanner = data

    _filesystemdataset_write(
        scanner, base_dir, basename_template, filesystem, partitioning,
        file_options, max_partitions, file_visitor, existing_data_behavior,
        max_open_files, max_rows_per_file,
        min_rows_per_group, max_rows_per_group, create_dir
    )

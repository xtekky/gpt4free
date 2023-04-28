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

"""Dataset is currently unstable. APIs subject to change without notice."""

from cython.operator cimport dereference as deref

import codecs
import collections
import os
import warnings
from libcpp cimport bool

import pyarrow as pa
from pyarrow.lib cimport *
from pyarrow.lib import ArrowTypeError, frombytes, tobytes, _pc
from pyarrow.includes.libarrow_dataset cimport *
from pyarrow._compute cimport Expression, _bind
from pyarrow._fs cimport FileSystem, FileInfo, FileSelector
from pyarrow._csv cimport (
    ConvertOptions, ParseOptions, ReadOptions, WriteOptions)
from pyarrow.util import _is_iterable, _is_path_like, _stringify_path


def _forbid_instantiation(klass, subclasses_instead=True):
    msg = '{} is an abstract class thus cannot be initialized.'.format(
        klass.__name__
    )
    if subclasses_instead:
        subclasses = [cls.__name__ for cls in klass.__subclasses__]
        msg += ' Use one of the subclasses instead: {}'.format(
            ', '.join(subclasses)
        )
    raise TypeError(msg)


_orc_fileformat = None
_orc_imported = False


def _get_orc_fileformat():
    """
    Import OrcFileFormat on first usage (to avoid circular import issue
    when `pyarrow._dataset_orc` would be imported first)
    """
    global _orc_fileformat
    global _orc_imported
    if not _orc_imported:
        try:
            from pyarrow._dataset_orc import OrcFileFormat
            _orc_fileformat = OrcFileFormat
        except ImportError as e:
            _orc_fileformat = None
        finally:
            _orc_imported = True
    return _orc_fileformat


_dataset_pq = False


def _get_parquet_classes():
    """
    Import Parquet class files on first usage (to avoid circular import issue
    when `pyarrow._dataset_parquet` would be imported first)
    """
    global _dataset_pq
    if _dataset_pq is False:
        try:
            import pyarrow._dataset_parquet as _dataset_pq
        except ImportError:
            _dataset_pq = None


def _get_parquet_symbol(name):
    """
    Get a symbol from pyarrow.parquet if the latter is importable, otherwise
    return None.
    """
    _get_parquet_classes()
    return _dataset_pq and getattr(_dataset_pq, name)


cdef CFileSource _make_file_source(object file, FileSystem filesystem=None):

    cdef:
        CFileSource c_source
        shared_ptr[CFileSystem] c_filesystem
        c_string c_path
        shared_ptr[CRandomAccessFile] c_file
        shared_ptr[CBuffer] c_buffer

    if isinstance(file, Buffer):
        c_buffer = pyarrow_unwrap_buffer(file)
        c_source = CFileSource(move(c_buffer))

    elif _is_path_like(file):
        if filesystem is None:
            raise ValueError("cannot construct a FileSource from "
                             "a path without a FileSystem")
        c_filesystem = filesystem.unwrap()
        c_path = tobytes(_stringify_path(file))
        c_source = CFileSource(move(c_path), move(c_filesystem))

    elif hasattr(file, 'read'):
        # Optimistically hope this is file-like
        c_file = get_native_file(file, False).get_random_access_file()
        c_source = CFileSource(move(c_file))

    else:
        raise TypeError("cannot construct a FileSource "
                        "from " + str(file))

    return c_source


cdef CSegmentEncoding _get_segment_encoding(str segment_encoding):
    if segment_encoding == "none":
        return CSegmentEncodingNone
    elif segment_encoding == "uri":
        return CSegmentEncodingUri
    raise ValueError(f"Unknown segment encoding: {segment_encoding}")


cdef Expression _true = Expression._scalar(True)


cdef class Dataset(_Weakrefable):
    """
    Collection of data fragments and potentially child datasets.

    Arrow Datasets allow you to query against data that has been split across
    multiple files. This sharding of data may indicate partitioning, which
    can accelerate queries that only touch some partitions (files).
    """

    def __init__(self):
        _forbid_instantiation(self.__class__)

    cdef void init(self, const shared_ptr[CDataset]& sp):
        self.wrapped = sp
        self.dataset = sp.get()
        self._scan_options = dict()

    @staticmethod
    cdef wrap(const shared_ptr[CDataset]& sp):
        type_name = frombytes(sp.get().type_name())

        classes = {
            'union': UnionDataset,
            'filesystem': FileSystemDataset,
            'in-memory': InMemoryDataset,
        }

        class_ = classes.get(type_name, None)
        if class_ is None:
            raise TypeError(type_name)

        cdef Dataset self = class_.__new__(class_)
        self.init(sp)
        return self

    cdef shared_ptr[CDataset] unwrap(self) nogil:
        return self.wrapped

    @property
    def partition_expression(self):
        """
        An Expression which evaluates to true for all data viewed by this
        Dataset.
        """
        return Expression.wrap(self.dataset.partition_expression())

    def replace_schema(self, Schema schema not None):
        """
        Return a copy of this Dataset with a different schema.

        The copy will view the same Fragments. If the new schema is not
        compatible with the original dataset's schema then an error will
        be raised.

        Parameters
        ----------
        schema : Schema
            The new dataset schema.
        """
        cdef shared_ptr[CDataset] copy = GetResultValue(
            self.dataset.ReplaceSchema(pyarrow_unwrap_schema(schema))
        )

        d = Dataset.wrap(move(copy))
        if self._scan_options:
            # Preserve scan options if set.
            d._scan_options = self._scan_options.copy()
        return d

    def get_fragments(self, Expression filter=None):
        """Returns an iterator over the fragments in this dataset.

        Parameters
        ----------
        filter : Expression, default None
            Return fragments matching the optional filter, either using the
            partition_expression or internal information like Parquet's
            statistics.

        Returns
        -------
        fragments : iterator of Fragment
        """
        if self._scan_options.get("filter") is not None:
            # Accessing fragments of a filtered dataset is not supported.
            # It would be unclear if you wanted to filter the fragments
            # or the rows in those fragments.
            raise ValueError(
                "Retrieving fragments of a filtered or projected "
                "dataset is not allowed. Remove the filtering."
            )

        return self._get_fragments(filter)

    def _get_fragments(self, Expression filter):
        cdef:
            CExpression c_filter
            CFragmentIterator c_iterator

        if filter is None:
            c_fragments = move(GetResultValue(self.dataset.GetFragments()))
        else:
            c_filter = _bind(filter, self.schema)
            c_fragments = move(GetResultValue(
                self.dataset.GetFragments(c_filter)))

        for maybe_fragment in c_fragments:
            yield Fragment.wrap(GetResultValue(move(maybe_fragment)))

    def _scanner_options(self, options):
        """Returns the default options to create a new Scanner.

        This is automatically invoked by :meth:`Dataset.scanner`
        and there is no need to use it.
        """
        new_options = options.copy()

        # at the moment only support filter
        requested_filter = options.get("filter")
        current_filter = self._scan_options.get("filter")
        if requested_filter is not None and current_filter is not None:
            new_options["filter"] = current_filter & requested_filter
        elif current_filter is not None:
            new_options["filter"] = current_filter

        return new_options

    def scanner(self, **kwargs):
        """
        Build a scan operation against the dataset.

        Data is not loaded immediately. Instead, this produces a Scanner,
        which exposes further operations (e.g. loading all data as a
        table, counting rows).

        See the :meth:`Scanner.from_dataset` method for further information.

        Parameters
        ----------
        **kwargs : dict, optional
            Arguments for `Scanner.from_dataset`.

        Returns
        -------
        scanner : Scanner

        Examples
        --------
        >>> import pyarrow as pa
        >>> table = pa.table({'year': [2020, 2022, 2021, 2022, 2019, 2021],
        ...                   'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>>
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, "dataset_scanner.parquet")

        >>> import pyarrow.dataset as ds
        >>> dataset = ds.dataset("dataset_scanner.parquet")

        Selecting a subset of the columns:

        >>> dataset.scanner(columns=["year", "n_legs"]).to_table()
        pyarrow.Table
        year: int64
        n_legs: int64
        ----
        year: [[2020,2022,2021,2022,2019,2021]]
        n_legs: [[2,2,4,4,5,100]]

        Projecting selected columns using an expression:

        >>> dataset.scanner(columns={
        ...     "n_legs_uint": ds.field("n_legs").cast("uint8"),
        ... }).to_table()
        pyarrow.Table
        n_legs_uint: uint8
        ----
        n_legs_uint: [[2,2,4,4,5,100]]

        Filtering rows while scanning:

        >>> dataset.scanner(filter=ds.field("year") > 2020).to_table()
        pyarrow.Table
        year: int64
        n_legs: int64
        animal: string
        ----
        year: [[2022,2021,2022,2021]]
        n_legs: [[2,4,4,100]]
        animal: [["Parrot","Dog","Horse","Centipede"]]
        """
        return Scanner.from_dataset(self, **kwargs)

    def to_batches(self, **kwargs):
        """
        Read the dataset as materialized record batches.

        Parameters
        ----------
        **kwargs : dict, optional
            Arguments for `Scanner.from_dataset`.

        Returns
        -------
        record_batches : iterator of RecordBatch
        """
        return self.scanner(**kwargs).to_batches()

    def to_table(self, **kwargs):
        """
        Read the dataset to an Arrow table.

        Note that this method reads all the selected data from the dataset
        into memory.

        Parameters
        ----------
        **kwargs : dict, optional
            Arguments for `Scanner.from_dataset`.

        Returns
        -------
        table : Table
        """
        return self.scanner(**kwargs).to_table()

    def take(self, object indices, **kwargs):
        """
        Select rows of data by index.

        Parameters
        ----------
        indices : Array or array-like
            indices of rows to select in the dataset.
        **kwargs : dict, optional
            See scanner() method for full parameter description.

        Returns
        -------
        table : Table
        """
        return self.scanner(**kwargs).take(indices)

    def head(self, int num_rows, **kwargs):
        """
        Load the first N rows of the dataset.

        Parameters
        ----------
        num_rows : int
            The number of rows to load.
        **kwargs : dict, optional
            See scanner() method for full parameter description.

        Returns
        -------
        table : Table
        """
        return self.scanner(**kwargs).head(num_rows)

    def count_rows(self, **kwargs):
        """
        Count rows matching the scanner filter.

        Parameters
        ----------
        **kwargs : dict, optional
            See scanner() method for full parameter description.

        Returns
        -------
        count : int
        """
        return self.scanner(**kwargs).count_rows()

    @property
    def schema(self):
        """The common schema of the full Dataset"""
        return pyarrow_wrap_schema(self.dataset.schema())

    def filter(self, expression not None):
        """
        Apply a row filter to the dataset.

        Parameters
        ----------
        expression : Expression
            The filter that should be applied to the dataset.

        Returns
        -------
        Dataset
        """
        cdef:
            Dataset filtered_dataset

        new_filter = expression
        current_filter = self._scan_options.get("filter")
        if current_filter is not None and new_filter is not None:
            new_filter = current_filter & new_filter

        filtered_dataset = self.__class__.__new__(self.__class__)
        filtered_dataset.init(self.wrapped)
        filtered_dataset._scan_options = dict(filter=new_filter)
        return filtered_dataset

    def sort_by(self, sorting, **kwargs):
        """
        Sort the Dataset by one or multiple columns.

        Parameters
        ----------
        sorting : str or list[tuple(name, order)]
            Name of the column to use to sort (ascending), or
            a list of multiple sorting conditions where
            each entry is a tuple with column name
            and sorting order ("ascending" or "descending")
        **kwargs : dict, optional
            Additional sorting options.
            As allowed by :class:`SortOptions`

        Returns
        -------
        InMemoryDataset
            A new dataset sorted according to the sort keys.
        """
        if isinstance(sorting, str):
            sorting = [(sorting, "ascending")]

        res = _pc()._exec_plan._sort_source(self, output_type=InMemoryDataset,
                                            sort_options=_pc().SortOptions(
                                                sort_keys=sorting, **kwargs
                                            ))
        return res

    def join(self, right_dataset, keys, right_keys=None, join_type="left outer",
             left_suffix=None, right_suffix=None, coalesce_keys=True,
             use_threads=True):
        """
        Perform a join between this dataset and another one.

        Result of the join will be a new dataset, where further
        operations can be applied.

        Parameters
        ----------
        right_dataset : dataset
            The dataset to join to the current one, acting as the right dataset
            in the join operation.
        keys : str or list[str]
            The columns from current dataset that should be used as keys
            of the join operation left side.
        right_keys : str or list[str], default None
            The columns from the right_dataset that should be used as keys
            on the join operation right side.
            When ``None`` use the same key names as the left dataset.
        join_type : str, default "left outer"
            The kind of join that should be performed, one of
            ("left semi", "right semi", "left anti", "right anti",
            "inner", "left outer", "right outer", "full outer")
        left_suffix : str, default None
            Which suffix to add to right column names. This prevents confusion
            when the columns in left and right datasets have colliding names.
        right_suffix : str, default None
            Which suffic to add to the left column names. This prevents confusion
            when the columns in left and right datasets have colliding names.
        coalesce_keys : bool, default True
            If the duplicated keys should be omitted from one of the sides
            in the join result.
        use_threads : bool, default True
            Whenever to use multithreading or not.

        Returns
        -------
        InMemoryDataset
        """
        if right_keys is None:
            right_keys = keys
        return _pc()._exec_plan._perform_join(join_type, self, keys, right_dataset, right_keys,
                                              left_suffix=left_suffix, right_suffix=right_suffix,
                                              use_threads=use_threads, coalesce_keys=coalesce_keys,
                                              output_type=InMemoryDataset)


cdef class InMemoryDataset(Dataset):
    """
    A Dataset wrapping in-memory data.

    Parameters
    ----------
    source : RecordBatch, Table, list, tuple
        The data for this dataset. Can be a RecordBatch, Table, list of
        RecordBatch/Table, iterable of RecordBatch, or a RecordBatchReader
        If an iterable is provided, the schema must also be provided.
    schema : Schema, optional
        Only required if passing an iterable as the source
    """

    cdef:
        CInMemoryDataset* in_memory_dataset

    def __init__(self, source, Schema schema=None):
        cdef:
            RecordBatchReader reader
            shared_ptr[CInMemoryDataset] in_memory_dataset

        if isinstance(source, (pa.RecordBatch, pa.Table)):
            source = [source]

        if isinstance(source, (list, tuple)):
            batches = []
            for item in source:
                if isinstance(item, pa.RecordBatch):
                    batches.append(item)
                elif isinstance(item, pa.Table):
                    batches.extend(item.to_batches())
                else:
                    raise TypeError(
                        'Expected a list of tables or batches. The given list '
                        'contains a ' + type(item).__name__)
                if schema is None:
                    schema = item.schema
                elif not schema.equals(item.schema):
                    raise ArrowTypeError(
                        f'Item has schema\n{item.schema}\nwhich does not '
                        f'match expected schema\n{schema}')
            if not batches and schema is None:
                raise ValueError('Must provide schema to construct in-memory '
                                 'dataset from an empty list')
            table = pa.Table.from_batches(batches, schema=schema)
            in_memory_dataset = make_shared[CInMemoryDataset](
                pyarrow_unwrap_table(table))
        else:
            raise TypeError(
                'Expected a table, batch, or list of tables/batches '
                'instead of the given type: ' +
                type(source).__name__
            )

        self.init(<shared_ptr[CDataset]> in_memory_dataset)

    cdef void init(self, const shared_ptr[CDataset]& sp):
        Dataset.init(self, sp)
        self.in_memory_dataset = <CInMemoryDataset*> sp.get()


cdef class UnionDataset(Dataset):
    """
    A Dataset wrapping child datasets.

    Children's schemas must agree with the provided schema.

    Parameters
    ----------
    schema : Schema
        A known schema to conform to.
    children : list of Dataset
        One or more input children
    """

    cdef:
        CUnionDataset* union_dataset

    def __init__(self, Schema schema not None, children):
        cdef:
            Dataset child
            CDatasetVector c_children
            shared_ptr[CUnionDataset] union_dataset

        for child in children:
            c_children.push_back(child.wrapped)

        union_dataset = GetResultValue(CUnionDataset.Make(
            pyarrow_unwrap_schema(schema), move(c_children)))
        self.init(<shared_ptr[CDataset]> union_dataset)

    cdef void init(self, const shared_ptr[CDataset]& sp):
        Dataset.init(self, sp)
        self.union_dataset = <CUnionDataset*> sp.get()

    def __reduce__(self):
        return UnionDataset, (self.schema, self.children)

    @property
    def children(self):
        cdef CDatasetVector children = self.union_dataset.children()
        return [Dataset.wrap(children[i]) for i in range(children.size())]


cdef class FileSystemDataset(Dataset):
    """
    A Dataset of file fragments.

    A FileSystemDataset is composed of one or more FileFragment.

    Parameters
    ----------
    fragments : list[Fragments]
        List of fragments to consume.
    schema : Schema
        The top-level schema of the Dataset.
    format : FileFormat
        File format of the fragments, currently only ParquetFileFormat,
        IpcFileFormat, and CsvFileFormat are supported.
    filesystem : FileSystem
        FileSystem of the fragments.
    root_partition : Expression, optional
        The top-level partition of the DataDataset.
    """

    cdef:
        CFileSystemDataset* filesystem_dataset

    def __init__(self, fragments, Schema schema, FileFormat format,
                 FileSystem filesystem=None, root_partition=None):
        cdef:
            FileFragment fragment=None
            vector[shared_ptr[CFileFragment]] c_fragments
            CResult[shared_ptr[CDataset]] result
            shared_ptr[CFileSystem] c_filesystem

        if root_partition is None:
            root_partition = _true
        elif not isinstance(root_partition, Expression):
            raise TypeError(
                "Argument 'root_partition' has incorrect type (expected "
                "Epression, got {0})".format(type(root_partition))
            )

        for fragment in fragments:
            c_fragments.push_back(
                static_pointer_cast[CFileFragment, CFragment](
                    fragment.unwrap()))

            if filesystem is None:
                filesystem = fragment.filesystem

        if filesystem is not None:
            c_filesystem = filesystem.unwrap()

        result = CFileSystemDataset.Make(
            pyarrow_unwrap_schema(schema),
            (<Expression> root_partition).unwrap(),
            format.unwrap(),
            c_filesystem,
            c_fragments
        )
        self.init(GetResultValue(result))

    @property
    def filesystem(self):
        return FileSystem.wrap(self.filesystem_dataset.filesystem())

    @property
    def partitioning(self):
        """
        The partitioning of the Dataset source, if discovered.

        If the FileSystemDataset is created using the ``dataset()`` factory
        function with a partitioning specified, this will return the
        finalized Partitioning object from the dataset discovery. In all
        other cases, this returns None.
        """
        c_partitioning = self.filesystem_dataset.partitioning()
        if c_partitioning.get() == nullptr:
            return None
        try:
            return Partitioning.wrap(c_partitioning)
        except TypeError:
            # e.g. type_name "default"
            return None

    cdef void init(self, const shared_ptr[CDataset]& sp):
        Dataset.init(self, sp)
        self.filesystem_dataset = <CFileSystemDataset*> sp.get()

    def __reduce__(self):
        return FileSystemDataset, (
            list(self.get_fragments()),
            self.schema,
            self.format,
            self.filesystem,
            self.partition_expression
        )

    @classmethod
    def from_paths(cls, paths, schema=None, format=None,
                   filesystem=None, partitions=None, root_partition=None):
        """A Dataset created from a list of paths on a particular filesystem.

        Parameters
        ----------
        paths : list of str
            List of file paths to create the fragments from.
        schema : Schema
            The top-level schema of the DataDataset.
        format : FileFormat
            File format to create fragments from, currently only
            ParquetFileFormat, IpcFileFormat, and CsvFileFormat are supported.
        filesystem : FileSystem
            The filesystem which files are from.
        partitions : list[Expression], optional
            Attach additional partition information for the file paths.
        root_partition : Expression, optional
            The top-level partition of the DataDataset.
        """
        cdef:
            FileFragment fragment

        if root_partition is None:
            root_partition = _true

        for arg, class_, name in [
            (schema, Schema, 'schema'),
            (format, FileFormat, 'format'),
            (filesystem, FileSystem, 'filesystem'),
            (root_partition, Expression, 'root_partition')
        ]:
            if not isinstance(arg, class_):
                raise TypeError(
                    "Argument '{0}' has incorrect type (expected {1}, "
                    "got {2})".format(name, class_.__name__, type(arg))
                )

        partitions = partitions or [_true] * len(paths)

        if len(paths) != len(partitions):
            raise ValueError(
                'The number of files resulting from paths_or_selector '
                'must be equal to the number of partitions.'
            )

        fragments = [
            format.make_fragment(path, filesystem, partitions[i])
            for i, path in enumerate(paths)
        ]
        return FileSystemDataset(fragments, schema, format,
                                 filesystem, root_partition)

    @property
    def files(self):
        """List of the files"""
        cdef vector[c_string] files = self.filesystem_dataset.files()
        return [frombytes(f) for f in files]

    @property
    def format(self):
        """The FileFormat of this source."""
        return FileFormat.wrap(self.filesystem_dataset.format())


cdef class FileWriteOptions(_Weakrefable):

    def __init__(self):
        _forbid_instantiation(self.__class__)

    cdef void init(self, const shared_ptr[CFileWriteOptions]& sp):
        self.wrapped = sp
        self.c_options = sp.get()

    @staticmethod
    cdef wrap(const shared_ptr[CFileWriteOptions]& sp):
        type_name = frombytes(sp.get().type_name())

        classes = {
            'csv': CsvFileWriteOptions,
            'ipc': IpcFileWriteOptions,
            'parquet': _get_parquet_symbol('ParquetFileWriteOptions'),
        }

        class_ = classes.get(type_name, None)
        if class_ is None:
            raise TypeError(type_name)

        cdef FileWriteOptions self = class_.__new__(class_)
        self.init(sp)
        return self

    @property
    def format(self):
        return FileFormat.wrap(self.c_options.format())

    cdef inline shared_ptr[CFileWriteOptions] unwrap(self):
        return self.wrapped


cdef class FileFormat(_Weakrefable):

    def __init__(self):
        _forbid_instantiation(self.__class__)

    cdef void init(self, const shared_ptr[CFileFormat]& sp):
        self.wrapped = sp
        self.format = sp.get()

    @staticmethod
    cdef wrap(const shared_ptr[CFileFormat]& sp):
        type_name = frombytes(sp.get().type_name())

        classes = {
            'ipc': IpcFileFormat,
            'csv': CsvFileFormat,
            'parquet': _get_parquet_symbol('ParquetFileFormat'),
            'orc': _get_orc_fileformat(),
        }

        class_ = classes.get(type_name, None)
        if class_ is None:
            raise TypeError(type_name)

        cdef FileFormat self = class_.__new__(class_)
        self.init(sp)
        return self

    cdef WrittenFile _finish_write(self, path, base_dir,
                                   CFileWriter* file_writer):
        parquet_metadata = None
        size = GetResultValue(file_writer.GetBytesWritten())
        return WrittenFile(path, parquet_metadata, size)

    cdef inline shared_ptr[CFileFormat] unwrap(self):
        return self.wrapped

    def inspect(self, file, filesystem=None):
        """
        Infer the schema of a file.

        Parameters
        ----------
        file : file-like object, path-like or str
            The file or file path to infer a schema from.
        filesystem : Filesystem, optional
            If `filesystem` is given, `file` must be a string and specifies
            the path of the file to read from the filesystem.

        Returns
        -------
        schema : Schema
            The schema inferred from the file
        """
        cdef:
            CFileSource c_source = _make_file_source(file, filesystem)
            CResult[shared_ptr[CSchema]] c_result
        with nogil:
            c_result = self.format.Inspect(c_source)
        c_schema = GetResultValue(c_result)
        return pyarrow_wrap_schema(move(c_schema))

    def make_fragment(self, file, filesystem=None,
                      Expression partition_expression=None):
        """
        Make a FileFragment from a given file.

        Parameters
        ----------
        file : file-like object, path-like or str
            The file or file path to make a fragment from.
        filesystem : Filesystem, optional
            If `filesystem` is given, `file` must be a string and specifies
            the path of the file to read from the filesystem.
        partition_expression : Expression
            The filter expression.
        """
        if partition_expression is None:
            partition_expression = _true

        c_source = _make_file_source(file, filesystem)
        c_fragment = <shared_ptr[CFragment]> GetResultValue(
            self.format.MakeFragment(move(c_source),
                                     partition_expression.unwrap(),
                                     <shared_ptr[CSchema]>nullptr))
        return Fragment.wrap(move(c_fragment))

    def make_write_options(self):
        sp_write_options = self.format.DefaultWriteOptions()
        if sp_write_options.get() == nullptr:
            # DefaultWriteOptions() may return `nullptr` which means that
            # the format does not yet support writing datasets.
            raise NotImplementedError(
                "Writing datasets not yet implemented for this file format."
            )
        return FileWriteOptions.wrap(sp_write_options)

    @property
    def default_extname(self):
        return frombytes(self.format.type_name())

    @property
    def default_fragment_scan_options(self):
        dfso = FragmentScanOptions.wrap(
            self.wrapped.get().default_fragment_scan_options)
        # CsvFileFormat stores a Python-specific encoding field that needs
        # to be restored because it does not exist in the C++ struct
        if isinstance(self, CsvFileFormat):
            if self._read_options_py is not None:
                dfso.read_options = self._read_options_py
        return dfso

    @default_fragment_scan_options.setter
    def default_fragment_scan_options(self, FragmentScanOptions options):
        if options is None:
            self.wrapped.get().default_fragment_scan_options =\
                <shared_ptr[CFragmentScanOptions]>nullptr
        else:
            self._set_default_fragment_scan_options(options)

    cdef _set_default_fragment_scan_options(self, FragmentScanOptions options):
        raise ValueError(f"Cannot set fragment scan options for "
                         f"'{options.type_name}' on {self.__class__.__name__}")

    def __eq__(self, other):
        try:
            return self.equals(other)
        except TypeError:
            return False


cdef class Fragment(_Weakrefable):
    """Fragment of data from a Dataset."""

    def __init__(self):
        _forbid_instantiation(self.__class__)

    cdef void init(self, const shared_ptr[CFragment]& sp):
        self.wrapped = sp
        self.fragment = sp.get()

    @staticmethod
    cdef wrap(const shared_ptr[CFragment]& sp):
        type_name = frombytes(sp.get().type_name())

        classes = {
            # IpcFileFormat, CsvFileFormat and OrcFileFormat do not have
            # corresponding subclasses of FileFragment
            'ipc': FileFragment,
            'csv': FileFragment,
            'orc': FileFragment,
            'parquet': _get_parquet_symbol('ParquetFileFragment'),
        }

        class_ = classes.get(type_name, None)
        if class_ is None:
            class_ = Fragment

        cdef Fragment self = class_.__new__(class_)
        self.init(sp)
        return self

    cdef inline shared_ptr[CFragment] unwrap(self):
        return self.wrapped

    @property
    def physical_schema(self):
        """Return the physical schema of this Fragment. This schema can be
        different from the dataset read schema."""
        cdef:
            CResult[shared_ptr[CSchema]] maybe_schema
        with nogil:
            maybe_schema = self.fragment.ReadPhysicalSchema()
        return pyarrow_wrap_schema(GetResultValue(maybe_schema))

    @property
    def partition_expression(self):
        """An Expression which evaluates to true for all data viewed by this
        Fragment.
        """
        return Expression.wrap(self.fragment.partition_expression())

    def scanner(self, Schema schema=None, **kwargs):
        """
        Build a scan operation against the fragment.

        Data is not loaded immediately. Instead, this produces a Scanner,
        which exposes further operations (e.g. loading all data as a
        table, counting rows).

        Parameters
        ----------
        schema : Schema
            Schema to use for scanning. This is used to unify a Fragment to
            it's Dataset's schema. If not specified this will use the
            Fragment's physical schema which might differ for each Fragment.
        **kwargs : dict, optional
            Arguments for `Scanner.from_fragment`.

        Returns
        -------
        scanner : Scanner
        """
        return Scanner.from_fragment(self, schema=schema, **kwargs)

    def to_batches(self, Schema schema=None, **kwargs):
        """
        Read the fragment as materialized record batches.

        Parameters
        ----------
        schema : Schema, optional
            Concrete schema to use for scanning.
        **kwargs : dict, optional
            Arguments for `Scanner.from_fragment`.

        Returns
        -------
        record_batches : iterator of RecordBatch
        """
        return self.scanner(schema=schema, **kwargs).to_batches()

    def to_table(self, Schema schema=None, **kwargs):
        """
        Convert this Fragment into a Table.

        Use this convenience utility with care. This will serially materialize
        the Scan result in memory before creating the Table.

        Parameters
        ----------
        schema : Schema, optional
            Concrete schema to use for scanning.
        **kwargs : dict, optional
            Arguments for `Scanner.from_fragment`.

        Returns
        -------
        table : Table
        """
        return self.scanner(schema=schema, **kwargs).to_table()

    def take(self, object indices, **kwargs):
        """
        Select rows of data by index.

        Parameters
        ----------
        indices : Array or array-like
            The indices of row to select in the dataset.
        **kwargs : dict, optional
            Arguments for `Scanner.from_fragment`.

        Returns
        -------
        Table
        """
        return self.scanner(**kwargs).take(indices)

    def head(self, int num_rows, **kwargs):
        """
        Load the first N rows of the fragment.

        Parameters
        ----------
        num_rows : int
            The number of rows to load.
        **kwargs : dict, optional
            Arguments for `Scanner.from_fragment`.

        Returns
        -------
        Table
        """
        return self.scanner(**kwargs).head(num_rows)

    def count_rows(self, **kwargs):
        """
        Count rows matching the scanner filter.

        Parameters
        ----------
        **kwargs : dict, optional
            Arguments for `Scanner.from_fragment`.

        Returns
        -------
        count : int
        """
        return self.scanner(**kwargs).count_rows()


cdef class FileFragment(Fragment):
    """A Fragment representing a data file."""

    cdef void init(self, const shared_ptr[CFragment]& sp):
        Fragment.init(self, sp)
        self.file_fragment = <CFileFragment*> sp.get()

    def __repr__(self):
        type_name = frombytes(self.fragment.type_name())
        if type_name != "parquet":
            typ = f" type={type_name}"
        else:
            # parquet has a subclass -> type embedded in class name
            typ = ""
        partition_dict = _get_partition_keys(self.partition_expression)
        partition = ", ".join(
            [f"{key}={val}" for key, val in partition_dict.items()]
        )
        if partition:
            partition = f" partition=[{partition}]"
        return "<pyarrow.dataset.{0}{1} path={2}{3}>".format(
            self.__class__.__name__, typ, self.path, partition
        )

    def __reduce__(self):
        buffer = self.buffer
        return self.format.make_fragment, (
            self.path if buffer is None else buffer,
            self.filesystem,
            self.partition_expression
        )

    def open(self):
        """
        Open a NativeFile of the buffer or file viewed by this fragment.
        """
        cdef:
            shared_ptr[CFileSystem] c_filesystem
            shared_ptr[CRandomAccessFile] opened
            c_string c_path
            NativeFile out = NativeFile()

        if self.buffer is not None:
            return pa.BufferReader(self.buffer)

        c_path = tobytes(self.file_fragment.source().path())
        with nogil:
            c_filesystem = self.file_fragment.source().filesystem()
            opened = GetResultValue(c_filesystem.get().OpenInputFile(c_path))

        out.set_random_access_file(opened)
        out.is_readable = True
        return out

    @property
    def path(self):
        """
        The path of the data file viewed by this fragment, if it views a
        file. If instead it views a buffer, this will be "<Buffer>".
        """
        return frombytes(self.file_fragment.source().path())

    @property
    def filesystem(self):
        """
        The FileSystem containing the data file viewed by this fragment, if
        it views a file. If instead it views a buffer, this will be None.
        """
        cdef:
            shared_ptr[CFileSystem] c_fs
        c_fs = self.file_fragment.source().filesystem()

        if c_fs.get() == nullptr:
            return None

        return FileSystem.wrap(c_fs)

    @property
    def buffer(self):
        """
        The buffer viewed by this fragment, if it views a buffer. If
        instead it views a file, this will be None.
        """
        cdef:
            shared_ptr[CBuffer] c_buffer
        c_buffer = self.file_fragment.source().buffer()

        if c_buffer.get() == nullptr:
            return None

        return pyarrow_wrap_buffer(c_buffer)

    @property
    def format(self):
        """
        The format of the data file viewed by this fragment.
        """
        return FileFormat.wrap(self.file_fragment.format())


cdef class FragmentScanOptions(_Weakrefable):
    """Scan options specific to a particular fragment and scan operation."""

    def __init__(self):
        _forbid_instantiation(self.__class__)

    cdef void init(self, const shared_ptr[CFragmentScanOptions]& sp):
        self.wrapped = sp

    @staticmethod
    cdef wrap(const shared_ptr[CFragmentScanOptions]& sp):
        if not sp:
            return None

        type_name = frombytes(sp.get().type_name())

        classes = {
            'csv': CsvFragmentScanOptions,
            'parquet': _get_parquet_symbol('ParquetFragmentScanOptions'),
        }

        class_ = classes.get(type_name, None)
        if class_ is None:
            raise TypeError(type_name)

        cdef FragmentScanOptions self = class_.__new__(class_)
        self.init(sp)
        return self

    @property
    def type_name(self):
        return frombytes(self.wrapped.get().type_name())

    def __eq__(self, other):
        try:
            return self.equals(other)
        except TypeError:
            return False


cdef class IpcFileWriteOptions(FileWriteOptions):
    cdef:
        CIpcFileWriteOptions* ipc_options

    def __init__(self):
        _forbid_instantiation(self.__class__)

    @property
    def write_options(self):
        out = IpcWriteOptions()
        out.c_options = CIpcWriteOptions(deref(self.ipc_options.options))
        return out

    @write_options.setter
    def write_options(self, IpcWriteOptions write_options not None):
        self.ipc_options.options.reset(
            new CIpcWriteOptions(write_options.c_options))

    cdef void init(self, const shared_ptr[CFileWriteOptions]& sp):
        FileWriteOptions.init(self, sp)
        self.ipc_options = <CIpcFileWriteOptions*> sp.get()


cdef class IpcFileFormat(FileFormat):

    def __init__(self):
        self.init(shared_ptr[CFileFormat](new CIpcFileFormat()))

    def equals(self, IpcFileFormat other):
        return True

    def make_write_options(self, **kwargs):
        cdef IpcFileWriteOptions opts = \
            <IpcFileWriteOptions> FileFormat.make_write_options(self)
        opts.write_options = IpcWriteOptions(**kwargs)
        return opts

    @property
    def default_extname(self):
        return "arrow"

    def __reduce__(self):
        return IpcFileFormat, tuple()


cdef class FeatherFileFormat(IpcFileFormat):

    @property
    def default_extname(self):
        return "feather"


cdef class CsvFileFormat(FileFormat):
    """
    FileFormat for CSV files.

    Parameters
    ----------
    parse_options : pyarrow.csv.ParseOptions
        Options regarding CSV parsing.
    default_fragment_scan_options : CsvFragmentScanOptions
        Default options for fragments scan.
    convert_options : pyarrow.csv.ConvertOptions
        Options regarding value conversion.
    read_options : pyarrow.csv.ReadOptions
        General read options.
    """
    cdef:
        CCsvFileFormat* csv_format
        # The encoding field in ReadOptions does not exist in the C++ struct.
        # We need to store it here and override it when reading
        # default_fragment_scan_options.read_options
        public ReadOptions _read_options_py

    # Avoid mistakingly creating attributes
    __slots__ = ()

    def __init__(self, ParseOptions parse_options=None,
                 default_fragment_scan_options=None,
                 ConvertOptions convert_options=None,
                 ReadOptions read_options=None):
        self.init(shared_ptr[CFileFormat](new CCsvFileFormat()))
        if parse_options is not None:
            self.parse_options = parse_options
        if convert_options is not None or read_options is not None:
            if default_fragment_scan_options:
                raise ValueError('If `default_fragment_scan_options` is '
                                 'given, cannot specify convert_options '
                                 'or read_options')
            self.default_fragment_scan_options = CsvFragmentScanOptions(
                convert_options=convert_options, read_options=read_options)
        elif isinstance(default_fragment_scan_options, dict):
            self.default_fragment_scan_options = CsvFragmentScanOptions(
                **default_fragment_scan_options)
        elif isinstance(default_fragment_scan_options, CsvFragmentScanOptions):
            self.default_fragment_scan_options = default_fragment_scan_options
        elif default_fragment_scan_options is not None:
            raise TypeError('`default_fragment_scan_options` must be either '
                            'a dictionary or an instance of '
                            'CsvFragmentScanOptions')
        if read_options is not None:
            self._read_options_py = read_options

    cdef void init(self, const shared_ptr[CFileFormat]& sp):
        FileFormat.init(self, sp)
        self.csv_format = <CCsvFileFormat*> sp.get()

    def make_write_options(self, **kwargs):
        cdef CsvFileWriteOptions opts = \
            <CsvFileWriteOptions> FileFormat.make_write_options(self)
        opts.write_options = WriteOptions(**kwargs)
        return opts

    @property
    def parse_options(self):
        return ParseOptions.wrap(self.csv_format.parse_options)

    @parse_options.setter
    def parse_options(self, ParseOptions parse_options not None):
        self.csv_format.parse_options = deref(parse_options.options)

    cdef _set_default_fragment_scan_options(self, FragmentScanOptions options):
        if options.type_name == 'csv':
            self.csv_format.default_fragment_scan_options = options.wrapped
            self.default_fragment_scan_options.read_options = options.read_options
            self._read_options_py = options.read_options
        else:
            super()._set_default_fragment_scan_options(options)

    def equals(self, CsvFileFormat other):
        return (
            self.parse_options.equals(other.parse_options) and
            self.default_fragment_scan_options ==
            other.default_fragment_scan_options)

    def __reduce__(self):
        return CsvFileFormat, (self.parse_options,
                               self.default_fragment_scan_options)

    def __repr__(self):
        return f"<CsvFileFormat parse_options={self.parse_options}>"


cdef class CsvFragmentScanOptions(FragmentScanOptions):
    """
    Scan-specific options for CSV fragments.

    Parameters
    ----------
    convert_options : pyarrow.csv.ConvertOptions
        Options regarding value conversion.
    read_options : pyarrow.csv.ReadOptions
        General read options.
    """

    cdef:
        CCsvFragmentScanOptions* csv_options
        # The encoding field in ReadOptions does not exist in the C++ struct.
        # We need to store it here and override it when reading read_options
        ReadOptions _read_options_py

    # Avoid mistakingly creating attributes
    __slots__ = ()

    def __init__(self, ConvertOptions convert_options=None,
                 ReadOptions read_options=None):
        self.init(shared_ptr[CFragmentScanOptions](
            new CCsvFragmentScanOptions()))
        if convert_options is not None:
            self.convert_options = convert_options
        if read_options is not None:
            self.read_options = read_options
            self._read_options_py = read_options

    cdef void init(self, const shared_ptr[CFragmentScanOptions]& sp):
        FragmentScanOptions.init(self, sp)
        self.csv_options = <CCsvFragmentScanOptions*> sp.get()

    @property
    def convert_options(self):
        return ConvertOptions.wrap(self.csv_options.convert_options)

    @convert_options.setter
    def convert_options(self, ConvertOptions convert_options not None):
        self.csv_options.convert_options = deref(convert_options.options)

    @property
    def read_options(self):
        read_options = ReadOptions.wrap(self.csv_options.read_options)
        if self._read_options_py is not None:
            read_options.encoding = self._read_options_py.encoding
        return read_options

    @read_options.setter
    def read_options(self, ReadOptions read_options not None):
        self.csv_options.read_options = deref(read_options.options)
        self._read_options_py = read_options
        if codecs.lookup(read_options.encoding).name != 'utf-8':
            self.csv_options.stream_transform_func = deref(
                make_streamwrap_func(read_options.encoding, 'utf-8'))

    def equals(self, CsvFragmentScanOptions other):
        return (
            other and
            self.convert_options.equals(other.convert_options) and
            self.read_options.equals(other.read_options))

    def __reduce__(self):
        return CsvFragmentScanOptions, (self.convert_options,
                                        self.read_options)


cdef class CsvFileWriteOptions(FileWriteOptions):
    cdef:
        CCsvFileWriteOptions* csv_options
        object _properties

    def __init__(self):
        _forbid_instantiation(self.__class__)

    @property
    def write_options(self):
        return WriteOptions.wrap(deref(self.csv_options.write_options))

    @write_options.setter
    def write_options(self, WriteOptions write_options not None):
        self.csv_options.write_options.reset(
            new CCSVWriteOptions(deref(write_options.options)))

    cdef void init(self, const shared_ptr[CFileWriteOptions]& sp):
        FileWriteOptions.init(self, sp)
        self.csv_options = <CCsvFileWriteOptions*> sp.get()


cdef class Partitioning(_Weakrefable):

    def __init__(self):
        _forbid_instantiation(self.__class__)

    cdef init(self, const shared_ptr[CPartitioning]& sp):
        self.wrapped = sp
        self.partitioning = sp.get()

    @staticmethod
    cdef wrap(const shared_ptr[CPartitioning]& sp):
        type_name = frombytes(sp.get().type_name())

        classes = {
            'directory': DirectoryPartitioning,
            'hive': HivePartitioning,
            'filename': FilenamePartitioning,
        }

        class_ = classes.get(type_name, None)
        if class_ is None:
            raise TypeError(type_name)

        cdef Partitioning self = class_.__new__(class_)
        self.init(sp)
        return self

    cdef inline shared_ptr[CPartitioning] unwrap(self):
        return self.wrapped

    def parse(self, path):
        cdef CResult[CExpression] result
        result = self.partitioning.Parse(tobytes(path))
        return Expression.wrap(GetResultValue(result))

    @property
    def schema(self):
        """The arrow Schema attached to the partitioning."""
        return pyarrow_wrap_schema(self.partitioning.schema())


cdef class PartitioningFactory(_Weakrefable):

    def __init__(self):
        _forbid_instantiation(self.__class__)

    cdef init(self, const shared_ptr[CPartitioningFactory]& sp):
        self.wrapped = sp
        self.factory = sp.get()

    @staticmethod
    cdef wrap(const shared_ptr[CPartitioningFactory]& sp):
        cdef PartitioningFactory self = PartitioningFactory.__new__(
            PartitioningFactory
        )
        self.init(sp)
        return self

    cdef inline shared_ptr[CPartitioningFactory] unwrap(self):
        return self.wrapped

    @property
    def type_name(self):
        return frombytes(self.factory.type_name())


cdef vector[shared_ptr[CArray]] _partitioning_dictionaries(
        Schema schema, dictionaries) except *:
    cdef:
        vector[shared_ptr[CArray]] c_dictionaries

    dictionaries = dictionaries or {}

    for field in schema:
        dictionary = dictionaries.get(field.name)

        if (isinstance(field.type, pa.DictionaryType) and
                dictionary is not None):
            c_dictionaries.push_back(pyarrow_unwrap_array(dictionary))
        else:
            c_dictionaries.push_back(<shared_ptr[CArray]> nullptr)

    return c_dictionaries

cdef class KeyValuePartitioning(Partitioning):

    cdef:
        CKeyValuePartitioning* keyvalue_partitioning

    def __init__(self):
        _forbid_instantiation(self.__class__)

    cdef init(self, const shared_ptr[CPartitioning]& sp):
        Partitioning.init(self, sp)
        self.keyvalue_partitioning = <CKeyValuePartitioning*> sp.get()
        self.wrapped = sp
        self.partitioning = sp.get()

    @property
    def dictionaries(self):
        """
        The unique values for each partition field, if available.

        Those values are only available if the Partitioning object was
        created through dataset discovery from a PartitioningFactory, or
        if the dictionaries were manually specified in the constructor.
        If no dictionary field is available, this returns an empty list.
        """
        cdef vector[shared_ptr[CArray]] c_arrays
        c_arrays = self.keyvalue_partitioning.dictionaries()
        res = []
        for arr in c_arrays:
            if arr.get() == nullptr:
                # Partitioning object has not been created through
                # inspected Factory
                res.append(None)
            else:
                res.append(pyarrow_wrap_array(arr))
        return res


cdef class DirectoryPartitioning(KeyValuePartitioning):
    """
    A Partitioning based on a specified Schema.

    The DirectoryPartitioning expects one segment in the file path for each
    field in the schema (all fields are required to be present).
    For example given schema<year:int16, month:int8> the path "/2009/11" would
    be parsed to ("year"_ == 2009 and "month"_ == 11).

    Parameters
    ----------
    schema : Schema
        The schema that describes the partitions present in the file path.
    dictionaries : dict[str, Array]
        If the type of any field of `schema` is a dictionary type, the
        corresponding entry of `dictionaries` must be an array containing
        every value which may be taken by the corresponding column or an
        error will be raised in parsing.
    segment_encoding : str, default "uri"
        After splitting paths into segments, decode the segments. Valid
        values are "uri" (URI-decode segments) and "none" (leave as-is).

    Returns
    -------
    DirectoryPartitioning

    Examples
    --------
    >>> from pyarrow.dataset import DirectoryPartitioning
    >>> partitioning = DirectoryPartitioning(
    ...     pa.schema([("year", pa.int16()), ("month", pa.int8())]))
    >>> print(partitioning.parse("/2009/11/"))
    ((year == 2009) and (month == 11))
    """

    cdef:
        CDirectoryPartitioning* directory_partitioning

    def __init__(self, Schema schema not None, dictionaries=None,
                 segment_encoding="uri"):
        cdef:
            shared_ptr[CDirectoryPartitioning] c_partitioning
            CKeyValuePartitioningOptions c_options

        c_options.segment_encoding = _get_segment_encoding(segment_encoding)
        c_partitioning = make_shared[CDirectoryPartitioning](
            pyarrow_unwrap_schema(schema),
            _partitioning_dictionaries(schema, dictionaries),
            c_options,
        )
        self.init(<shared_ptr[CPartitioning]> c_partitioning)

    cdef init(self, const shared_ptr[CPartitioning]& sp):
        KeyValuePartitioning.init(self, sp)
        self.directory_partitioning = <CDirectoryPartitioning*> sp.get()

    @staticmethod
    def discover(field_names=None, infer_dictionary=False,
                 max_partition_dictionary_size=0,
                 schema=None, segment_encoding="uri"):
        """
        Discover a DirectoryPartitioning.

        Parameters
        ----------
        field_names : list of str
            The names to associate with the values from the subdirectory names.
            If schema is given, will be populated from the schema.
        infer_dictionary : bool, default False
            When inferring a schema for partition fields, yield dictionary
            encoded types instead of plain types. This can be more efficient
            when materializing virtual columns, and Expressions parsed by the
            finished Partitioning will include dictionaries of all unique
            inspected values for each field.
        max_partition_dictionary_size : int, default 0
            Synonymous with infer_dictionary for backwards compatibility with
            1.0: setting this to -1 or None is equivalent to passing
            infer_dictionary=True.
        schema : Schema, default None
            Use this schema instead of inferring a schema from partition
            values. Partition values will be validated against this schema
            before accumulation into the Partitioning's dictionary.
        segment_encoding : str, default "uri"
            After splitting paths into segments, decode the segments. Valid
            values are "uri" (URI-decode segments) and "none" (leave as-is).

        Returns
        -------
        PartitioningFactory
            To be used in the FileSystemFactoryOptions.
        """
        cdef:
            CPartitioningFactoryOptions c_options
            vector[c_string] c_field_names

        if max_partition_dictionary_size in {-1, None}:
            infer_dictionary = True
        elif max_partition_dictionary_size != 0:
            raise NotImplementedError("max_partition_dictionary_size must be "
                                      "0, -1, or None")

        if infer_dictionary:
            c_options.infer_dictionary = True

        if schema:
            c_options.schema = pyarrow_unwrap_schema(schema)
            c_field_names = [tobytes(f.name) for f in schema]
        elif not field_names:
            raise ValueError(
                "Neither field_names nor schema was passed; "
                "cannot infer field_names")
        else:
            c_field_names = [tobytes(s) for s in field_names]

        c_options.segment_encoding = _get_segment_encoding(segment_encoding)

        return PartitioningFactory.wrap(
            CDirectoryPartitioning.MakeFactory(c_field_names, c_options))


cdef class HivePartitioning(KeyValuePartitioning):
    """
    A Partitioning for "/$key=$value/" nested directories as found in
    Apache Hive.

    Multi-level, directory based partitioning scheme originating from
    Apache Hive with all data files stored in the leaf directories. Data is
    partitioned by static values of a particular column in the schema.
    Partition keys are represented in the form $key=$value in directory names.
    Field order is ignored, as are missing or unrecognized field names.

    For example, given schema<year:int16, month:int8, day:int8>, a possible
    path would be "/year=2009/month=11/day=15".

    Parameters
    ----------
    schema : Schema
        The schema that describes the partitions present in the file path.
    dictionaries : dict[str, Array]
        If the type of any field of `schema` is a dictionary type, the
        corresponding entry of `dictionaries` must be an array containing
        every value which may be taken by the corresponding column or an
        error will be raised in parsing.
    null_fallback : str, default "__HIVE_DEFAULT_PARTITION__"
        If any field is None then this fallback will be used as a label
    segment_encoding : str, default "uri"
        After splitting paths into segments, decode the segments. Valid
        values are "uri" (URI-decode segments) and "none" (leave as-is).

    Returns
    -------
    HivePartitioning

    Examples
    --------
    >>> from pyarrow.dataset import HivePartitioning
    >>> partitioning = HivePartitioning(
    ...     pa.schema([("year", pa.int16()), ("month", pa.int8())]))
    >>> print(partitioning.parse("/year=2009/month=11/"))
    ((year == 2009) and (month == 11))

    """

    cdef:
        CHivePartitioning* hive_partitioning

    def __init__(self,
                 Schema schema not None,
                 dictionaries=None,
                 null_fallback="__HIVE_DEFAULT_PARTITION__",
                 segment_encoding="uri"):

        cdef:
            shared_ptr[CHivePartitioning] c_partitioning
            CHivePartitioningOptions c_options

        c_options.null_fallback = tobytes(null_fallback)
        c_options.segment_encoding = _get_segment_encoding(segment_encoding)

        c_partitioning = make_shared[CHivePartitioning](
            pyarrow_unwrap_schema(schema),
            _partitioning_dictionaries(schema, dictionaries),
            c_options,
        )
        self.init(<shared_ptr[CPartitioning]> c_partitioning)

    cdef init(self, const shared_ptr[CPartitioning]& sp):
        KeyValuePartitioning.init(self, sp)
        self.hive_partitioning = <CHivePartitioning*> sp.get()

    @staticmethod
    def discover(infer_dictionary=False,
                 max_partition_dictionary_size=0,
                 null_fallback="__HIVE_DEFAULT_PARTITION__",
                 schema=None,
                 segment_encoding="uri"):
        """
        Discover a HivePartitioning.

        Parameters
        ----------
        infer_dictionary : bool, default False
            When inferring a schema for partition fields, yield dictionary
            encoded types instead of plain. This can be more efficient when
            materializing virtual columns, and Expressions parsed by the
            finished Partitioning will include dictionaries of all unique
            inspected values for each field.
        max_partition_dictionary_size : int, default 0
            Synonymous with infer_dictionary for backwards compatibility with
            1.0: setting this to -1 or None is equivalent to passing
            infer_dictionary=True.
        null_fallback : str, default "__HIVE_DEFAULT_PARTITION__"
            When inferring a schema for partition fields this value will be
            replaced by null.  The default is set to __HIVE_DEFAULT_PARTITION__
            for compatibility with Spark
        schema : Schema, default None
            Use this schema instead of inferring a schema from partition
            values. Partition values will be validated against this schema
            before accumulation into the Partitioning's dictionary.
        segment_encoding : str, default "uri"
            After splitting paths into segments, decode the segments. Valid
            values are "uri" (URI-decode segments) and "none" (leave as-is).

        Returns
        -------
        PartitioningFactory
            To be used in the FileSystemFactoryOptions.
        """
        cdef:
            CHivePartitioningFactoryOptions c_options

        if max_partition_dictionary_size in {-1, None}:
            infer_dictionary = True
        elif max_partition_dictionary_size != 0:
            raise NotImplementedError("max_partition_dictionary_size must be "
                                      "0, -1, or None")

        if infer_dictionary:
            c_options.infer_dictionary = True

        c_options.null_fallback = tobytes(null_fallback)

        if schema:
            c_options.schema = pyarrow_unwrap_schema(schema)

        c_options.segment_encoding = _get_segment_encoding(segment_encoding)

        return PartitioningFactory.wrap(
            CHivePartitioning.MakeFactory(c_options))


cdef class FilenamePartitioning(KeyValuePartitioning):
    """
    A Partitioning based on a specified Schema.

    The FilenamePartitioning expects one segment in the file name for each
    field in the schema (all fields are required to be present) separated
    by '_'. For example given schema<year:int16, month:int8> the name
    ``"2009_11_"`` would be parsed to ("year" == 2009 and "month" == 11).

    Parameters
    ----------
    schema : Schema
        The schema that describes the partitions present in the file path.
    dictionaries : dict[str, Array]
        If the type of any field of `schema` is a dictionary type, the
        corresponding entry of `dictionaries` must be an array containing
        every value which may be taken by the corresponding column or an
        error will be raised in parsing.
    segment_encoding : str, default "uri"
        After splitting paths into segments, decode the segments. Valid
        values are "uri" (URI-decode segments) and "none" (leave as-is).

    Returns
    -------
    FilenamePartitioning

    Examples
    --------
    >>> from pyarrow.dataset import FilenamePartitioning
    >>> partitioning = FilenamePartitioning(
    ...     pa.schema([("year", pa.int16()), ("month", pa.int8())]))
    >>> print(partitioning.parse("2009_11_data.parquet"))
    ((year == 2009) and (month == 11))
    """

    cdef:
        CFilenamePartitioning* filename_partitioning

    def __init__(self, Schema schema not None, dictionaries=None,
                 segment_encoding="uri"):
        cdef:
            shared_ptr[CFilenamePartitioning] c_partitioning
            CKeyValuePartitioningOptions c_options

        c_options.segment_encoding = _get_segment_encoding(segment_encoding)
        c_partitioning = make_shared[CFilenamePartitioning](
            pyarrow_unwrap_schema(schema),
            _partitioning_dictionaries(schema, dictionaries),
            c_options,
        )
        self.init(<shared_ptr[CPartitioning]> c_partitioning)

    cdef init(self, const shared_ptr[CPartitioning]& sp):
        KeyValuePartitioning.init(self, sp)
        self.filename_partitioning = <CFilenamePartitioning*> sp.get()

    @staticmethod
    def discover(field_names=None, infer_dictionary=False,
                 schema=None, segment_encoding="uri"):
        """
        Discover a FilenamePartitioning.

        Parameters
        ----------
        field_names : list of str
            The names to associate with the values from the subdirectory names.
            If schema is given, will be populated from the schema.
        infer_dictionary : bool, default False
            When inferring a schema for partition fields, yield dictionary
            encoded types instead of plain types. This can be more efficient
            when materializing virtual columns, and Expressions parsed by the
            finished Partitioning will include dictionaries of all unique
            inspected values for each field.
        schema : Schema, default None
            Use this schema instead of inferring a schema from partition
            values. Partition values will be validated against this schema
            before accumulation into the Partitioning's dictionary.
        segment_encoding : str, default "uri"
            After splitting paths into segments, decode the segments. Valid
            values are "uri" (URI-decode segments) and "none" (leave as-is).

        Returns
        -------
        PartitioningFactory
            To be used in the FileSystemFactoryOptions.
        """
        cdef:
            CPartitioningFactoryOptions c_options
            vector[c_string] c_field_names

        if infer_dictionary:
            c_options.infer_dictionary = True

        if schema:
            c_options.schema = pyarrow_unwrap_schema(schema)
            c_field_names = [tobytes(f.name) for f in schema]
        elif not field_names:
            raise TypeError(
                "Neither field_names nor schema was passed; "
                "cannot infer field_names")
        else:
            c_field_names = [tobytes(s) for s in field_names]

        c_options.segment_encoding = _get_segment_encoding(segment_encoding)

        return PartitioningFactory.wrap(
            CFilenamePartitioning.MakeFactory(c_field_names, c_options))


cdef class DatasetFactory(_Weakrefable):
    """
    DatasetFactory is used to create a Dataset, inspect the Schema
    of the fragments contained in it, and declare a partitioning.
    """

    def __init__(self):
        _forbid_instantiation(self.__class__)

    cdef init(self, const shared_ptr[CDatasetFactory]& sp):
        self.wrapped = sp
        self.factory = sp.get()

    @staticmethod
    cdef wrap(const shared_ptr[CDatasetFactory]& sp):
        cdef DatasetFactory self = \
            DatasetFactory.__new__(DatasetFactory)
        self.init(sp)
        return self

    cdef inline shared_ptr[CDatasetFactory] unwrap(self) nogil:
        return self.wrapped

    @property
    def root_partition(self):
        return Expression.wrap(self.factory.root_partition())

    @root_partition.setter
    def root_partition(self, Expression expr):
        check_status(self.factory.SetRootPartition(expr.unwrap()))

    def inspect_schemas(self):
        cdef CResult[vector[shared_ptr[CSchema]]] result
        cdef CInspectOptions options
        with nogil:
            result = self.factory.InspectSchemas(options)

        schemas = []
        for s in GetResultValue(result):
            schemas.append(pyarrow_wrap_schema(s))
        return schemas

    def inspect(self):
        """
        Inspect all data fragments and return a common Schema.

        Returns
        -------
        Schema
        """
        cdef:
            CInspectOptions options
            CResult[shared_ptr[CSchema]] result
        with nogil:
            result = self.factory.Inspect(options)
        return pyarrow_wrap_schema(GetResultValue(result))

    def finish(self, Schema schema=None):
        """
        Create a Dataset using the inspected schema or an explicit schema
        (if given).

        Parameters
        ----------
        schema : Schema, default None
            The schema to conform the source to.  If None, the inspected
            schema is used.

        Returns
        -------
        Dataset
        """
        cdef:
            shared_ptr[CSchema] sp_schema
            CResult[shared_ptr[CDataset]] result

        if schema is not None:
            sp_schema = pyarrow_unwrap_schema(schema)
            with nogil:
                result = self.factory.FinishWithSchema(sp_schema)
        else:
            with nogil:
                result = self.factory.Finish()

        return Dataset.wrap(GetResultValue(result))


cdef class FileSystemFactoryOptions(_Weakrefable):
    """
    Influences the discovery of filesystem paths.

    Parameters
    ----------
    partition_base_dir : str, optional
        For the purposes of applying the partitioning, paths will be
        stripped of the partition_base_dir. Files not matching the
        partition_base_dir prefix will be skipped for partitioning discovery.
        The ignored files will still be part of the Dataset, but will not
        have partition information.
    partitioning : Partitioning/PartitioningFactory, optional
       Apply the Partitioning to every discovered Fragment. See Partitioning or
       PartitioningFactory documentation.
    exclude_invalid_files : bool, optional (default True)
        If True, invalid files will be excluded (file format specific check).
        This will incur IO for each files in a serial and single threaded
        fashion. Disabling this feature will skip the IO, but unsupported
        files may be present in the Dataset (resulting in an error at scan
        time).
    selector_ignore_prefixes : list, optional
        When discovering from a Selector (and not from an explicit file list),
        ignore files and directories matching any of these prefixes.
        By default this is ['.', '_'].
    """

    cdef:
        CFileSystemFactoryOptions options

    __slots__ = ()  # avoid mistakingly creating attributes

    def __init__(self, partition_base_dir=None, partitioning=None,
                 exclude_invalid_files=None,
                 list selector_ignore_prefixes=None):
        if isinstance(partitioning, PartitioningFactory):
            self.partitioning_factory = partitioning
        elif isinstance(partitioning, Partitioning):
            self.partitioning = partitioning

        if partition_base_dir is not None:
            self.partition_base_dir = partition_base_dir
        if exclude_invalid_files is not None:
            self.exclude_invalid_files = exclude_invalid_files
        if selector_ignore_prefixes is not None:
            self.selector_ignore_prefixes = selector_ignore_prefixes

    cdef inline CFileSystemFactoryOptions unwrap(self):
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
    def exclude_invalid_files(self):
        """Whether to exclude invalid files."""
        return self.options.exclude_invalid_files

    @exclude_invalid_files.setter
    def exclude_invalid_files(self, bint value):
        self.options.exclude_invalid_files = value

    @property
    def selector_ignore_prefixes(self):
        """
        List of prefixes. Files matching one of those prefixes will be
        ignored by the discovery process.
        """
        return [frombytes(p) for p in self.options.selector_ignore_prefixes]

    @selector_ignore_prefixes.setter
    def selector_ignore_prefixes(self, values):
        self.options.selector_ignore_prefixes = [tobytes(v) for v in values]


cdef class FileSystemDatasetFactory(DatasetFactory):
    """
    Create a DatasetFactory from a list of paths with schema inspection.

    Parameters
    ----------
    filesystem : pyarrow.fs.FileSystem
        Filesystem to discover.
    paths_or_selector : pyarrow.fs.FileSelector or list of path-likes
        Either a Selector object or a list of path-like objects.
    format : FileFormat
        Currently only ParquetFileFormat and IpcFileFormat are supported.
    options : FileSystemFactoryOptions, optional
        Various flags influencing the discovery of filesystem paths.
    """

    cdef:
        CFileSystemDatasetFactory* filesystem_factory

    def __init__(self, FileSystem filesystem not None, paths_or_selector,
                 FileFormat format not None,
                 FileSystemFactoryOptions options=None):
        cdef:
            vector[c_string] paths
            CFileSelector c_selector
            CResult[shared_ptr[CDatasetFactory]] result
            shared_ptr[CFileSystem] c_filesystem
            shared_ptr[CFileFormat] c_format
            CFileSystemFactoryOptions c_options

        options = options or FileSystemFactoryOptions()
        c_options = options.unwrap()
        c_filesystem = filesystem.unwrap()
        c_format = format.unwrap()

        if isinstance(paths_or_selector, FileSelector):
            with nogil:
                c_selector = (<FileSelector> paths_or_selector).selector
                result = CFileSystemDatasetFactory.MakeFromSelector(
                    c_filesystem,
                    c_selector,
                    c_format,
                    c_options
                )
        elif isinstance(paths_or_selector, (list, tuple)):
            paths = [tobytes(s) for s in paths_or_selector]
            with nogil:
                result = CFileSystemDatasetFactory.MakeFromPaths(
                    c_filesystem,
                    paths,
                    c_format,
                    c_options
                )
        else:
            raise TypeError('Must pass either paths or a FileSelector, but '
                            'passed {}'.format(type(paths_or_selector)))

        self.init(GetResultValue(result))

    cdef init(self, shared_ptr[CDatasetFactory]& sp):
        DatasetFactory.init(self, sp)
        self.filesystem_factory = <CFileSystemDatasetFactory*> sp.get()


cdef class UnionDatasetFactory(DatasetFactory):
    """
    Provides a way to inspect/discover a Dataset's expected schema before
    materialization.

    Parameters
    ----------
    factories : list of DatasetFactory
    """

    cdef:
        CUnionDatasetFactory* union_factory

    def __init__(self, list factories):
        cdef:
            DatasetFactory factory
            vector[shared_ptr[CDatasetFactory]] c_factories
        for factory in factories:
            c_factories.push_back(factory.unwrap())
        self.init(GetResultValue(CUnionDatasetFactory.Make(c_factories)))

    cdef init(self, const shared_ptr[CDatasetFactory]& sp):
        DatasetFactory.init(self, sp)
        self.union_factory = <CUnionDatasetFactory*> sp.get()


cdef class RecordBatchIterator(_Weakrefable):
    """An iterator over a sequence of record batches."""
    cdef:
        # An object that must be kept alive with the iterator.
        object iterator_owner
        # Iterator is a non-POD type and Cython uses offsetof, leading
        # to a compiler warning unless wrapped like so
        shared_ptr[CRecordBatchIterator] iterator

    def __init__(self):
        _forbid_instantiation(self.__class__, subclasses_instead=False)

    @staticmethod
    cdef wrap(object owner, CRecordBatchIterator iterator):
        cdef RecordBatchIterator self = \
            RecordBatchIterator.__new__(RecordBatchIterator)
        self.iterator_owner = owner
        self.iterator = make_shared[CRecordBatchIterator](move(iterator))
        return self

    cdef inline shared_ptr[CRecordBatchIterator] unwrap(self) nogil:
        return self.iterator

    def __iter__(self):
        return self

    def __next__(self):
        cdef shared_ptr[CRecordBatch] record_batch
        with nogil:
            record_batch = GetResultValue(move(self.iterator.get().Next()))
        if record_batch == NULL:
            raise StopIteration
        return pyarrow_wrap_batch(record_batch)


class TaggedRecordBatch(collections.namedtuple(
        "TaggedRecordBatch", ["record_batch", "fragment"])):
    """
    A combination of a record batch and the fragment it came from.

    Parameters
    ----------
    record_batch : RecordBatch
        The record batch.
    fragment : Fragment
        Fragment of the record batch.
    """


cdef class TaggedRecordBatchIterator(_Weakrefable):
    """An iterator over a sequence of record batches with fragments."""
    cdef:
        object iterator_owner
        shared_ptr[CTaggedRecordBatchIterator] iterator

    def __init__(self):
        _forbid_instantiation(self.__class__, subclasses_instead=False)

    @staticmethod
    cdef wrap(object owner, CTaggedRecordBatchIterator iterator):
        cdef TaggedRecordBatchIterator self = \
            TaggedRecordBatchIterator.__new__(TaggedRecordBatchIterator)
        self.iterator_owner = owner
        self.iterator = make_shared[CTaggedRecordBatchIterator](
            move(iterator))
        return self

    def __iter__(self):
        return self

    def __next__(self):
        cdef CTaggedRecordBatch batch
        with nogil:
            batch = GetResultValue(move(self.iterator.get().Next()))
        if batch.record_batch == NULL:
            raise StopIteration
        return TaggedRecordBatch(
            record_batch=pyarrow_wrap_batch(batch.record_batch),
            fragment=Fragment.wrap(batch.fragment))


_DEFAULT_BATCH_SIZE = 2**17
_DEFAULT_BATCH_READAHEAD = 16
_DEFAULT_FRAGMENT_READAHEAD = 4

cdef void _populate_builder(const shared_ptr[CScannerBuilder]& ptr,
                            object columns=None, Expression filter=None,
                            int batch_size=_DEFAULT_BATCH_SIZE,
                            int batch_readahead=_DEFAULT_BATCH_READAHEAD,
                            int fragment_readahead=_DEFAULT_FRAGMENT_READAHEAD,
                            bint use_threads=True, MemoryPool memory_pool=None,
                            FragmentScanOptions fragment_scan_options=None)\
        except *:
    cdef:
        CScannerBuilder *builder
        vector[CExpression] c_exprs

    builder = ptr.get()

    check_status(builder.Filter(_bind(
        filter, pyarrow_wrap_schema(builder.schema()))))

    if columns is not None:
        if isinstance(columns, dict):
            for expr in columns.values():
                if not isinstance(expr, Expression):
                    raise TypeError(
                        "Expected an Expression for a 'column' dictionary "
                        "value, got {} instead".format(type(expr))
                    )
                c_exprs.push_back((<Expression> expr).unwrap())

            check_status(
                builder.Project(c_exprs, [tobytes(c) for c in columns.keys()])
            )
        elif isinstance(columns, list):
            check_status(builder.ProjectColumns([tobytes(c) for c in columns]))
        else:
            raise ValueError(
                "Expected a list or a dict for 'columns', "
                "got {} instead.".format(type(columns))
            )

    check_status(builder.BatchSize(batch_size))
    check_status(builder.BatchReadahead(batch_readahead))
    check_status(builder.FragmentReadahead(fragment_readahead))
    check_status(builder.UseThreads(use_threads))
    check_status(builder.Pool(maybe_unbox_memory_pool(memory_pool)))
    if fragment_scan_options:
        check_status(
            builder.FragmentScanOptions(fragment_scan_options.wrapped))


cdef class Scanner(_Weakrefable):
    """A materialized scan operation with context and options bound.

    A scanner is the class that glues the scan tasks, data fragments and data
    sources together.

    Parameters
    ----------
    dataset : Dataset
        Dataset to scan.
    columns : list of str or dict, default None
        The columns to project. This can be a list of column names to
        include (order and duplicates will be preserved), or a dictionary
        with {{new_column_name: expression}} values for more advanced
        projections.

        The list of columns or expressions may use the special fields
        `__batch_index` (the index of the batch within the fragment),
        `__fragment_index` (the index of the fragment within the dataset),
        `__last_in_fragment` (whether the batch is last in fragment), and
        `__filename` (the name of the source file or a description of the
        source fragment).

        The columns will be passed down to Datasets and corresponding data
        fragments to avoid loading, copying, and deserializing columns
        that will not be required further down the compute chain.
        By default all of the available columns are projected.
        Raises an exception if any of the referenced column names does
        not exist in the dataset's Schema.
    filter : Expression, default None
        Scan will return only the rows matching the filter.
        If possible the predicate will be pushed down to exploit the
        partition information or internal metadata found in the data
        source, e.g. Parquet statistics. Otherwise filters the loaded
        RecordBatches before yielding them.
    batch_size : int, default 128Ki
        The maximum row count for scanned record batches. If scanned
        record batches are overflowing memory then this method can be
        called to reduce their size.
    batch_readahead : int, default 16
        The number of batches to read ahead in a file. This might not work
        for all file formats. Increasing this number will increase
        RAM usage but could also improve IO utilization.
    fragment_readahead : int, default 4
        The number of files to read ahead. Increasing this number will increase
        RAM usage but could also improve IO utilization.
    use_threads : bool, default True
        If enabled, then maximum parallelism will be used determined by
        the number of available CPU cores.
    use_async : bool, default True
        This flag is deprecated and is being kept for this release for
        backwards compatibility.  It will be removed in the next release.
    memory_pool : MemoryPool, default None
        For memory allocations, if required. If not specified, uses the
        default pool.
    """

    def __init__(self):
        _forbid_instantiation(self.__class__)

    cdef void init(self, const shared_ptr[CScanner]& sp):
        self.wrapped = sp
        self.scanner = sp.get()

    @staticmethod
    cdef wrap(const shared_ptr[CScanner]& sp):
        cdef Scanner self = Scanner.__new__(Scanner)
        self.init(sp)
        return self

    cdef inline shared_ptr[CScanner] unwrap(self):
        return self.wrapped

    @staticmethod
    cdef shared_ptr[CScanOptions] _make_scan_options(Dataset dataset, dict py_scanoptions) except *:
        cdef:
            shared_ptr[CScannerBuilder] builder = make_shared[CScannerBuilder](dataset.unwrap())

        py_scanoptions = dataset._scanner_options(py_scanoptions)

        # Need to explicitly expand the arguments as Cython doesn't support
        # keyword expansion in cdef functions.
        _populate_builder(
            builder,
            columns=py_scanoptions.get("columns"),
            filter=py_scanoptions.get("filter"),
            batch_size=py_scanoptions.get("batch_size", _DEFAULT_BATCH_SIZE),
            batch_readahead=py_scanoptions.get(
                "batch_readahead", _DEFAULT_BATCH_READAHEAD),
            fragment_readahead=py_scanoptions.get(
                "fragment_readahead", _DEFAULT_FRAGMENT_READAHEAD),
            use_threads=py_scanoptions.get("use_threads", True),
            memory_pool=py_scanoptions.get("memory_pool"),
            fragment_scan_options=py_scanoptions.get("fragment_scan_options"))

        return GetResultValue(deref(builder).GetScanOptions())

    @staticmethod
    def from_dataset(Dataset dataset not None, *,
                     object columns=None,
                     Expression filter=None,
                     int batch_size=_DEFAULT_BATCH_SIZE,
                     int batch_readahead=_DEFAULT_BATCH_READAHEAD,
                     int fragment_readahead=_DEFAULT_FRAGMENT_READAHEAD,
                     FragmentScanOptions fragment_scan_options=None,
                     bint use_threads=True, object use_async=None,
                     MemoryPool memory_pool=None):
        """
        Create Scanner from Dataset,

        Parameters
        ----------
        dataset : Dataset
            Dataset to scan.
        columns : list of str, default None
            The columns to project. This can be a list of column names to
            include (order and duplicates will be preserved), or a dictionary
            with {new_column_name: expression} values for more advanced
            projections.

            The list of columns or expressions may use the special fields
            `__batch_index` (the index of the batch within the fragment),
            `__fragment_index` (the index of the fragment within the dataset),
            `__last_in_fragment` (whether the batch is last in fragment), and
            `__filename` (the name of the source file or a description of the
            source fragment).

            The columns will be passed down to Datasets and corresponding data
            fragments to avoid loading, copying, and deserializing columns
            that will not be required further down the compute chain.
            By default all of the available columns are projected. Raises
            an exception if any of the referenced column names does not exist
            in the dataset's Schema.
        filter : Expression, default None
            Scan will return only the rows matching the filter.
            If possible the predicate will be pushed down to exploit the
            partition information or internal metadata found in the data
            source, e.g. Parquet statistics. Otherwise filters the loaded
            RecordBatches before yielding them.
        batch_size : int, default 128Ki
            The maximum row count for scanned record batches. If scanned
            record batches are overflowing memory then this method can be
            called to reduce their size.
        batch_readahead : int, default 16
            The number of batches to read ahead in a file. This might not work
            for all file formats. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_readahead : int, default 4
            The number of files to read ahead. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_scan_options : FragmentScanOptions, default None
            Options specific to a particular scan and fragment type, which
            can change between different scans of the same dataset.
        use_threads : bool, default True
            If enabled, then maximum parallelism will be used determined by
            the number of available CPU cores.
        use_async : bool, default True
            This flag is deprecated and is being kept for this release for
            backwards compatibility.  It will be removed in the next
            release.
        memory_pool : MemoryPool, default None
            For memory allocations, if required. If not specified, uses the
            default pool.
        """
        cdef:
            shared_ptr[CScanOptions] options
            shared_ptr[CScannerBuilder] builder
            shared_ptr[CScanner] scanner

        if use_async is not None:
            warnings.warn('The use_async flag is deprecated and has no '
                          'effect.  It will be removed in the next release.',
                          FutureWarning)

        options = Scanner._make_scan_options(
            dataset,
            dict(columns=columns, filter=filter, batch_size=batch_size,
                 batch_readahead=batch_readahead,
                 fragment_readahead=fragment_readahead, use_threads=use_threads,
                 memory_pool=memory_pool, fragment_scan_options=fragment_scan_options)
        )
        builder = make_shared[CScannerBuilder](dataset.unwrap(), options)
        scanner = GetResultValue(builder.get().Finish())
        return Scanner.wrap(scanner)

    @staticmethod
    def from_fragment(Fragment fragment not None, *, Schema schema=None,
                      object columns=None, Expression filter=None,
                      int batch_size=_DEFAULT_BATCH_SIZE,
                      int batch_readahead=_DEFAULT_BATCH_READAHEAD,
                      FragmentScanOptions fragment_scan_options=None,
                      bint use_threads=True, object use_async=None,
                      MemoryPool memory_pool=None,):
        """
        Create Scanner from Fragment,

        Parameters
        ----------
        fragment : Fragment
            fragment to scan.
        schema : Schema, optional
            The schema of the fragment.
        columns : list of str, default None
            The columns to project. This can be a list of column names to
            include (order and duplicates will be preserved), or a dictionary
            with {new_column_name: expression} values for more advanced
            projections.

            The list of columns or expressions may use the special fields
            `__batch_index` (the index of the batch within the fragment),
            `__fragment_index` (the index of the fragment within the dataset),
            `__last_in_fragment` (whether the batch is last in fragment), and
            `__filename` (the name of the source file or a description of the
            source fragment).

            The columns will be passed down to Datasets and corresponding data
            fragments to avoid loading, copying, and deserializing columns
            that will not be required further down the compute chain.
            By default all of the available columns are projected. Raises
            an exception if any of the referenced column names does not exist
            in the dataset's Schema.
        filter : Expression, default None
            Scan will return only the rows matching the filter.
            If possible the predicate will be pushed down to exploit the
            partition information or internal metadata found in the data
            source, e.g. Parquet statistics. Otherwise filters the loaded
            RecordBatches before yielding them.
        batch_size : int, default 128Ki
            The maximum row count for scanned record batches. If scanned
            record batches are overflowing memory then this method can be
            called to reduce their size.
        batch_readahead : int, default 16
            The number of batches to read ahead in a file. This might not work
            for all file formats. Increasing this number will increase
            RAM usage but could also improve IO utilization.
        fragment_scan_options : FragmentScanOptions, default None
            Options specific to a particular scan and fragment type, which
            can change between different scans of the same dataset.
        use_threads : bool, default True
            If enabled, then maximum parallelism will be used determined by
            the number of available CPU cores.
        use_async : bool, default True
            This flag is deprecated and is being kept for this release for
            backwards compatibility.  It will be removed in the next
            release.
        memory_pool : MemoryPool, default None
            For memory allocations, if required. If not specified, uses the
            default pool.
        use_threads : bool, default True
            If enabled, then maximum parallelism will be used determined by
            the number of available CPU cores.
        use_async : bool, default True
            This flag is deprecated and is being kept for this release for
            backwards compatibility.  It will be removed in the next
            release.
        memory_pool : MemoryPool, default None
            For memory allocations, if required. If not specified, uses the
            default pool.
        """
        cdef:
            shared_ptr[CScanOptions] options = make_shared[CScanOptions]()
            shared_ptr[CScannerBuilder] builder
            shared_ptr[CScanner] scanner

        schema = schema or fragment.physical_schema

        if use_async is not None:
            warnings.warn('The use_async flag is deprecated and has no '
                          'effect.  It will be removed in the next release.',
                          FutureWarning)

        builder = make_shared[CScannerBuilder](pyarrow_unwrap_schema(schema),
                                               fragment.unwrap(), options)
        _populate_builder(builder, columns=columns, filter=filter,
                          batch_size=batch_size, batch_readahead=batch_readahead,
                          fragment_readahead=_DEFAULT_FRAGMENT_READAHEAD,
                          use_threads=use_threads,
                          memory_pool=memory_pool,
                          fragment_scan_options=fragment_scan_options)

        scanner = GetResultValue(builder.get().Finish())
        return Scanner.wrap(scanner)

    @staticmethod
    def from_batches(source, *, Schema schema=None, object columns=None,
                     Expression filter=None, int batch_size=_DEFAULT_BATCH_SIZE,
                     FragmentScanOptions fragment_scan_options=None,
                     bint use_threads=True, object use_async=None,
                     MemoryPool memory_pool=None):
        """
        Create a Scanner from an iterator of batches.

        This creates a scanner which can be used only once. It is
        intended to support writing a dataset (which takes a scanner)
        from a source which can be read only once (e.g. a
        RecordBatchReader or generator).

        Parameters
        ----------
        source : Iterator
            The iterator of Batches.
        schema : Schema
            The schema of the batches.
        columns : list of str or dict, default None
                The columns to project.
        filter : Expression, default None
            Scan will return only the rows matching the filter.
        batch_size : int, default 128Ki
            The maximum row count for scanned record batches.
        fragment_scan_options : FragmentScanOptions
            The fragment scan options.
        use_threads : bool, default True
            If enabled, then maximum parallelism will be used determined by
            the number of available CPU cores.
        use_async : bool, default True
            This flag is deprecated and is being kept for this release for
            backwards compatibility.  It will be removed in the next
            release.
        memory_pool : MemoryPool, default None
            For memory allocations, if required. If not specified, uses the
            default pool.
        """
        cdef:
            shared_ptr[CScanOptions] options = make_shared[CScanOptions]()
            shared_ptr[CScannerBuilder] builder
            shared_ptr[CScanner] scanner
            RecordBatchReader reader
        if isinstance(source, pa.ipc.RecordBatchReader):
            if schema:
                raise ValueError('Cannot specify a schema when providing '
                                 'a RecordBatchReader')
            reader = source
        elif _is_iterable(source):
            if schema is None:
                raise ValueError('Must provide schema to construct scanner '
                                 'from an iterable')
            reader = pa.ipc.RecordBatchReader.from_batches(schema, source)
        else:
            raise TypeError('Expected a RecordBatchReader or an iterable of '
                            'batches instead of the given type: ' +
                            type(source).__name__)
        builder = CScannerBuilder.FromRecordBatchReader(reader.reader)

        if use_async is not None:
            warnings.warn('The use_async flag is deprecated and has no '
                          'effect.  It will be removed in the next release.',
                          FutureWarning)

        _populate_builder(builder, columns=columns, filter=filter,
                          batch_size=batch_size, batch_readahead=_DEFAULT_BATCH_READAHEAD,
                          fragment_readahead=_DEFAULT_FRAGMENT_READAHEAD, use_threads=use_threads,
                          memory_pool=memory_pool,
                          fragment_scan_options=fragment_scan_options)
        scanner = GetResultValue(builder.get().Finish())
        return Scanner.wrap(scanner)

    @property
    def dataset_schema(self):
        """The schema with which batches will be read from fragments."""
        return pyarrow_wrap_schema(
            self.scanner.options().get().dataset_schema)

    @property
    def projected_schema(self):
        """
        The materialized schema of the data, accounting for projections.

        This is the schema of any data returned from the scanner.
        """
        return pyarrow_wrap_schema(
            self.scanner.options().get().projected_schema)

    def to_batches(self):
        """
        Consume a Scanner in record batches.

        Returns
        -------
        record_batches : iterator of RecordBatch
        """
        def _iterator(batch_iter):
            for batch in batch_iter:
                yield batch.record_batch
        # Don't make ourselves a generator so errors are raised immediately
        return _iterator(self.scan_batches())

    def scan_batches(self):
        """
        Consume a Scanner in record batches with corresponding fragments.

        Returns
        -------
        record_batches : iterator of TaggedRecordBatch
        """
        cdef CTaggedRecordBatchIterator iterator
        with nogil:
            iterator = move(GetResultValue(self.scanner.ScanBatches()))
        # Don't make ourselves a generator so errors are raised immediately
        return TaggedRecordBatchIterator.wrap(self, move(iterator))

    def to_table(self):
        """
        Convert a Scanner into a Table.

        Use this convenience utility with care. This will serially materialize
        the Scan result in memory before creating the Table.

        Returns
        -------
        Table
        """
        cdef CResult[shared_ptr[CTable]] result

        with nogil:
            result = self.scanner.ToTable()

        return pyarrow_wrap_table(GetResultValue(result))

    def take(self, object indices):
        """
        Select rows of data by index.

        Will only consume as many batches of the underlying dataset as
        needed. Otherwise, this is equivalent to
        ``to_table().take(indices)``.

        Parameters
        ----------
        indices : Array or array-like
            indices of rows to select in the dataset.

        Returns
        -------
        Table
        """
        cdef CResult[shared_ptr[CTable]] result
        cdef shared_ptr[CArray] c_indices

        if not isinstance(indices, pa.Array):
            indices = pa.array(indices)
        c_indices = pyarrow_unwrap_array(indices)

        with nogil:
            result = self.scanner.TakeRows(deref(c_indices))
        return pyarrow_wrap_table(GetResultValue(result))

    def head(self, int num_rows):
        """
        Load the first N rows of the dataset.

        Parameters
        ----------
        num_rows : int
            The number of rows to load.

        Returns
        -------
        Table
        """
        cdef CResult[shared_ptr[CTable]] result
        with nogil:
            result = self.scanner.Head(num_rows)
        return pyarrow_wrap_table(GetResultValue(result))

    def count_rows(self):
        """
        Count rows matching the scanner filter.

        Returns
        -------
        count : int
        """
        cdef CResult[int64_t] result
        with nogil:
            result = self.scanner.CountRows()
        return GetResultValue(result)

    def to_reader(self):
        """Consume this scanner as a RecordBatchReader.

        Returns
        -------
        RecordBatchReader
        """
        cdef RecordBatchReader reader
        reader = RecordBatchReader.__new__(RecordBatchReader)
        reader.reader = GetResultValue(self.scanner.ToRecordBatchReader())
        return reader


def _get_partition_keys(Expression partition_expression):
    """
    Extract partition keys (equality constraints between a field and a scalar)
    from an expression as a dict mapping the field's name to its value.

    NB: All expressions yielded by a HivePartitioning or DirectoryPartitioning
    will be conjunctions of equality conditions and are accessible through this
    function. Other subexpressions will be ignored.

    For example, an expression of
    <pyarrow.dataset.Expression ((part == A:string) and (year == 2016:int32))>
    is converted to {'part': 'A', 'year': 2016}
    """
    cdef:
        CExpression expr = partition_expression.unwrap()
        pair[CFieldRef, CDatum] ref_val

    out = {}
    for ref_val in GetResultValue(CExtractKnownFieldValues(expr)).map:
        assert ref_val.first.name() != nullptr
        assert ref_val.second.kind() == DatumType_SCALAR
        val = pyarrow_wrap_scalar(ref_val.second.scalar())
        out[frombytes(deref(ref_val.first.name()))] = val.as_py()
    return out


cdef class WrittenFile(_Weakrefable):
    """
    Metadata information about files written as
    part of a dataset write operation

    Parameters
    ----------
    path : str
        Path to the file.
    metadata : pyarrow.parquet.FileMetaData, optional
        For Parquet files, the Parquet file metadata.
    size : int
        The size of the file in bytes.
    """

    def __init__(self, path, metadata, size):
        self.path = path
        self.metadata = metadata
        self.size = size


cdef void _filesystemdataset_write_visitor(
        dict visit_args,
        CFileWriter* file_writer):
    cdef:
        str path
        str base_dir
        WrittenFile written_file
        object parquet_metadata
        FileFormat file_format

    parquet_metadata = None
    path = frombytes(deref(file_writer).destination().path)
    base_dir = frombytes(visit_args['base_dir'])
    file_format = FileFormat.wrap(file_writer.format())
    written_file = file_format._finish_write(path, base_dir, file_writer)
    visit_args['file_visitor'](written_file)


def _filesystemdataset_write(
    Scanner data not None,
    object base_dir not None,
    str basename_template not None,
    FileSystem filesystem not None,
    Partitioning partitioning not None,
    FileWriteOptions file_options not None,
    int max_partitions,
    object file_visitor,
    str existing_data_behavior not None,
    int max_open_files,
    int max_rows_per_file,
    int min_rows_per_group,
    int max_rows_per_group,
    bool create_dir
):
    """
    CFileSystemDataset.Write wrapper
    """
    cdef:
        CFileSystemDatasetWriteOptions c_options
        shared_ptr[CScanner] c_scanner
        vector[shared_ptr[CRecordBatch]] c_batches
        dict visit_args

    c_options.file_write_options = file_options.unwrap()
    c_options.filesystem = filesystem.unwrap()
    c_options.base_dir = tobytes(_stringify_path(base_dir))
    c_options.partitioning = partitioning.unwrap()
    c_options.max_partitions = max_partitions
    c_options.max_open_files = max_open_files
    c_options.max_rows_per_file = max_rows_per_file
    c_options.max_rows_per_group = max_rows_per_group
    c_options.min_rows_per_group = min_rows_per_group
    c_options.basename_template = tobytes(basename_template)
    if existing_data_behavior == 'error':
        c_options.existing_data_behavior = ExistingDataBehavior_ERROR
    elif existing_data_behavior == 'overwrite_or_ignore':
        c_options.existing_data_behavior =\
            ExistingDataBehavior_OVERWRITE_OR_IGNORE
    elif existing_data_behavior == 'delete_matching':
        c_options.existing_data_behavior = ExistingDataBehavior_DELETE_MATCHING
    else:
        raise ValueError(
            ("existing_data_behavior must be one of 'error', ",
             "'overwrite_or_ignore' or 'delete_matching'")
        )
    c_options.create_dir = create_dir

    if file_visitor is not None:
        visit_args = {'base_dir': c_options.base_dir,
                      'file_visitor': file_visitor}
        # Need to use post_finish because parquet metadata is not available
        # until after Finish has been called
        c_options.writer_post_finish = BindFunction[cb_writer_finish_internal](
            &_filesystemdataset_write_visitor, visit_args)

    c_scanner = data.unwrap()
    with nogil:
        check_status(CFileSystemDataset.Write(c_options, c_scanner))

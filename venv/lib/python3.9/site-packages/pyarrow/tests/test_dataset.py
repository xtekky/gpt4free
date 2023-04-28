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

import contextlib
import os
import posixpath
import datetime
import pathlib
import pickle
import sys
import textwrap
import tempfile
import threading
import time

from urllib.parse import quote

import numpy as np
import pytest

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv
import pyarrow.feather
import pyarrow.fs as fs
from pyarrow.tests.util import (change_cwd, _filesystem_uri,
                                FSProtocolClass, ProxyHandler,
                                _configure_s3_limited_user)

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import pyarrow.dataset as ds
except ImportError:
    ds = None

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

# Marks all of the tests in this module
# Ignore these with pytest ... -m 'not dataset'
pytestmark = pytest.mark.dataset


def _generate_data(n):
    import datetime
    import itertools

    day = datetime.datetime(2000, 1, 1)
    interval = datetime.timedelta(days=5)
    colors = itertools.cycle(['green', 'blue', 'yellow', 'red', 'orange'])

    data = []
    for i in range(n):
        data.append((day, i, float(i), next(colors)))
        day += interval

    return pd.DataFrame(data, columns=['date', 'index', 'value', 'color'])


def _table_from_pandas(df):
    schema = pa.schema([
        pa.field('date', pa.date32()),
        pa.field('index', pa.int64()),
        pa.field('value', pa.float64()),
        pa.field('color', pa.string()),
    ])
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    return table.replace_schema_metadata()


def assert_dataset_fragment_convenience_methods(dataset):
    # FileFragment convenience methods
    for fragment in dataset.get_fragments():
        with fragment.open() as nf:
            assert isinstance(nf, pa.NativeFile)
            assert not nf.closed
            assert nf.seekable()
            assert nf.readable()
            assert not nf.writable()


@pytest.fixture
@pytest.mark.parquet
def mockfs():
    mockfs = fs._MockFileSystem()

    directories = [
        'subdir/1/xxx',
        'subdir/2/yyy',
    ]

    for i, directory in enumerate(directories):
        path = '{}/file{}.parquet'.format(directory, i)
        mockfs.create_dir(directory)
        with mockfs.open_output_stream(path) as out:
            data = [
                list(range(5)),
                list(map(float, range(5))),
                list(map(str, range(5))),
                [i] * 5,
                [{'a': j % 3, 'b': str(j % 3)} for j in range(5)],
            ]
            schema = pa.schema([
                ('i64', pa.int64()),
                ('f64', pa.float64()),
                ('str', pa.string()),
                ('const', pa.int64()),
                ('struct', pa.struct({'a': pa.int64(), 'b': pa.string()})),
            ])
            batch = pa.record_batch(data, schema=schema)
            table = pa.Table.from_batches([batch])

            pq.write_table(table, out)

    return mockfs


@pytest.fixture
def open_logging_fs(monkeypatch):
    from pyarrow.fs import PyFileSystem, LocalFileSystem
    from .test_fs import ProxyHandler

    localfs = LocalFileSystem()

    def normalized(paths):
        return {localfs.normalize_path(str(p)) for p in paths}

    opened = set()

    def open_input_file(self, path):
        path = localfs.normalize_path(str(path))
        opened.add(path)
        return self._fs.open_input_file(path)

    # patch proxyhandler to log calls to open_input_file
    monkeypatch.setattr(ProxyHandler, "open_input_file", open_input_file)
    fs = PyFileSystem(ProxyHandler(localfs))

    @contextlib.contextmanager
    def assert_opens(expected_opened):
        opened.clear()
        try:
            yield
        finally:
            assert normalized(opened) == normalized(expected_opened)

    return fs, assert_opens


@pytest.fixture(scope='module')
def multisourcefs(request):
    request.config.pyarrow.requires('pandas')
    request.config.pyarrow.requires('parquet')

    df = _generate_data(1000)
    mockfs = fs._MockFileSystem()

    # simply split the dataframe into four chunks to construct a data source
    # from each chunk into its own directory
    df_a, df_b, df_c, df_d = np.array_split(df, 4)

    # create a directory containing a flat sequence of parquet files without
    # any partitioning involved
    mockfs.create_dir('plain')
    for i, chunk in enumerate(np.array_split(df_a, 10)):
        path = 'plain/chunk-{}.parquet'.format(i)
        with mockfs.open_output_stream(path) as out:
            pq.write_table(_table_from_pandas(chunk), out)

    # create one with schema partitioning by weekday and color
    mockfs.create_dir('schema')
    for part, chunk in df_b.groupby([df_b.date.dt.dayofweek, df_b.color]):
        folder = 'schema/{}/{}'.format(*part)
        path = '{}/chunk.parquet'.format(folder)
        mockfs.create_dir(folder)
        with mockfs.open_output_stream(path) as out:
            pq.write_table(_table_from_pandas(chunk), out)

    # create one with hive partitioning by year and month
    mockfs.create_dir('hive')
    for part, chunk in df_c.groupby([df_c.date.dt.year, df_c.date.dt.month]):
        folder = 'hive/year={}/month={}'.format(*part)
        path = '{}/chunk.parquet'.format(folder)
        mockfs.create_dir(folder)
        with mockfs.open_output_stream(path) as out:
            pq.write_table(_table_from_pandas(chunk), out)

    # create one with hive partitioning by color
    mockfs.create_dir('hive_color')
    for part, chunk in df_d.groupby("color"):
        folder = 'hive_color/color={}'.format(part)
        path = '{}/chunk.parquet'.format(folder)
        mockfs.create_dir(folder)
        with mockfs.open_output_stream(path) as out:
            pq.write_table(_table_from_pandas(chunk), out)

    return mockfs


@pytest.fixture
@pytest.mark.parquet
def dataset(mockfs):
    format = ds.ParquetFileFormat()
    selector = fs.FileSelector('subdir', recursive=True)
    options = ds.FileSystemFactoryOptions('subdir')
    options.partitioning = ds.DirectoryPartitioning(
        pa.schema([
            pa.field('group', pa.int32()),
            pa.field('key', pa.string())
        ])
    )
    factory = ds.FileSystemDatasetFactory(mockfs, selector, format, options)
    return factory.finish()


@pytest.fixture(params=[
    (True),
    (False)
], ids=['threaded', 'serial'])
def dataset_reader(request):
    '''
    Fixture which allows dataset scanning operations to be
    run with/without threads
    '''
    use_threads = request.param

    class reader:

        def __init__(self):
            self.use_threads = use_threads

        def _patch_kwargs(self, kwargs):
            if 'use_threads' in kwargs:
                raise Exception(
                    ('Invalid use of dataset_reader, do not specify'
                     ' use_threads'))
            kwargs['use_threads'] = use_threads

        def to_table(self, dataset, **kwargs):
            self._patch_kwargs(kwargs)
            return dataset.to_table(**kwargs)

        def to_batches(self, dataset, **kwargs):
            self._patch_kwargs(kwargs)
            return dataset.to_batches(**kwargs)

        def scanner(self, dataset, **kwargs):
            self._patch_kwargs(kwargs)
            return dataset.scanner(**kwargs)

        def head(self, dataset, num_rows, **kwargs):
            self._patch_kwargs(kwargs)
            return dataset.head(num_rows, **kwargs)

        def take(self, dataset, indices, **kwargs):
            self._patch_kwargs(kwargs)
            return dataset.take(indices, **kwargs)

        def count_rows(self, dataset, **kwargs):
            self._patch_kwargs(kwargs)
            return dataset.count_rows(**kwargs)

    return reader()


@pytest.mark.parquet
def test_filesystem_dataset(mockfs):
    schema = pa.schema([
        pa.field('const', pa.int64())
    ])
    file_format = ds.ParquetFileFormat()
    paths = ['subdir/1/xxx/file0.parquet', 'subdir/2/yyy/file1.parquet']
    partitions = [ds.field('part') == x for x in range(1, 3)]
    fragments = [file_format.make_fragment(path, mockfs, part)
                 for path, part in zip(paths, partitions)]
    root_partition = ds.field('level') == ds.scalar(1337)

    dataset_from_fragments = ds.FileSystemDataset(
        fragments, schema=schema, format=file_format,
        filesystem=mockfs, root_partition=root_partition,
    )
    dataset_from_paths = ds.FileSystemDataset.from_paths(
        paths, schema=schema, format=file_format, filesystem=mockfs,
        partitions=partitions, root_partition=root_partition,
    )

    for dataset in [dataset_from_fragments, dataset_from_paths]:
        assert isinstance(dataset, ds.FileSystemDataset)
        assert isinstance(dataset.format, ds.ParquetFileFormat)
        assert dataset.partition_expression.equals(root_partition)
        assert set(dataset.files) == set(paths)

        fragments = list(dataset.get_fragments())
        for fragment, partition, path in zip(fragments, partitions, paths):
            assert fragment.partition_expression.equals(partition)
            assert fragment.path == path
            assert isinstance(fragment.format, ds.ParquetFileFormat)
            assert isinstance(fragment, ds.ParquetFileFragment)
            assert fragment.row_groups == [0]
            assert fragment.num_row_groups == 1

            row_group_fragments = list(fragment.split_by_row_group())
            assert fragment.num_row_groups == len(row_group_fragments) == 1
            assert isinstance(row_group_fragments[0], ds.ParquetFileFragment)
            assert row_group_fragments[0].path == path
            assert row_group_fragments[0].row_groups == [0]
            assert row_group_fragments[0].num_row_groups == 1

        fragments = list(dataset.get_fragments(filter=ds.field("const") == 0))
        assert len(fragments) == 2

    # the root_partition keyword has a default
    dataset = ds.FileSystemDataset(
        fragments, schema=schema, format=file_format, filesystem=mockfs
    )
    assert dataset.partition_expression.equals(ds.scalar(True))

    # from_paths partitions have defaults
    dataset = ds.FileSystemDataset.from_paths(
        paths, schema=schema, format=file_format, filesystem=mockfs
    )
    assert dataset.partition_expression.equals(ds.scalar(True))
    for fragment in dataset.get_fragments():
        assert fragment.partition_expression.equals(ds.scalar(True))

    # validation of required arguments
    with pytest.raises(TypeError, match="incorrect type"):
        ds.FileSystemDataset(fragments, file_format, schema)
    # validation of root_partition
    with pytest.raises(TypeError, match="incorrect type"):
        ds.FileSystemDataset(fragments, schema=schema,
                             format=file_format, root_partition=1)
    # missing required argument in from_paths
    with pytest.raises(TypeError, match="incorrect type"):
        ds.FileSystemDataset.from_paths(fragments, format=file_format)


def test_filesystem_dataset_no_filesystem_interaction(dataset_reader):
    # ARROW-8283
    schema = pa.schema([
        pa.field('f1', pa.int64())
    ])
    file_format = ds.IpcFileFormat()
    paths = ['nonexistingfile.arrow']

    # creating the dataset itself doesn't raise
    dataset = ds.FileSystemDataset.from_paths(
        paths, schema=schema, format=file_format,
        filesystem=fs.LocalFileSystem(),
    )

    # getting fragments also doesn't raise
    dataset.get_fragments()

    # scanning does raise
    with pytest.raises(FileNotFoundError):
        dataset_reader.to_table(dataset)


@pytest.mark.parquet
def test_dataset(dataset, dataset_reader):
    assert isinstance(dataset, ds.Dataset)
    assert isinstance(dataset.schema, pa.Schema)

    # TODO(kszucs): test non-boolean Exprs for filter do raise
    expected_i64 = pa.array([0, 1, 2, 3, 4], type=pa.int64())
    expected_f64 = pa.array([0, 1, 2, 3, 4], type=pa.float64())

    for batch in dataset_reader.to_batches(dataset):
        assert isinstance(batch, pa.RecordBatch)
        assert batch.column(0).equals(expected_i64)
        assert batch.column(1).equals(expected_f64)

    for batch in dataset_reader.scanner(dataset).scan_batches():
        assert isinstance(batch, ds.TaggedRecordBatch)
        assert isinstance(batch.fragment, ds.Fragment)

    table = dataset_reader.to_table(dataset)
    assert isinstance(table, pa.Table)
    assert len(table) == 10

    condition = ds.field('i64') == 1
    result = dataset.to_table(use_threads=True, filter=condition)
    # Don't rely on the scanning order
    result = result.sort_by('group').to_pydict()

    assert result['i64'] == [1, 1]
    assert result['f64'] == [1., 1.]
    assert sorted(result['group']) == [1, 2]
    assert sorted(result['key']) == ['xxx', 'yyy']

    # Filtering on a nested field ref
    condition = ds.field(('struct', 'b')) == '1'
    result = dataset.to_table(use_threads=True, filter=condition)
    result = result.sort_by('group').to_pydict()

    assert result['i64'] == [1, 4, 1, 4]
    assert result['f64'] == [1.0, 4.0, 1.0, 4.0]
    assert result['group'] == [1, 1, 2, 2]
    assert result['key'] == ['xxx', 'xxx', 'yyy', 'yyy']

    # Projecting on a nested field ref expression
    projection = {
        'i64': ds.field('i64'),
        'f64': ds.field('f64'),
        'new': ds.field(('struct', 'b')) == '1',
    }
    result = dataset.to_table(use_threads=True, columns=projection)
    result = result.sort_by('i64').to_pydict()

    assert list(result) == ['i64', 'f64', 'new']
    assert result['i64'] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert result['f64'] == [0.0, 0.0, 1.0, 1.0,
                             2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
    assert result['new'] == [False, False, True, True, False, False,
                             False, False, True, True]
    assert_dataset_fragment_convenience_methods(dataset)


@pytest.mark.parquet
def test_scanner_options(dataset):
    scanner = dataset.to_batches(fragment_readahead=16, batch_readahead=8)
    batch = next(scanner)
    assert batch.num_columns == 7


@pytest.mark.parquet
def test_scanner(dataset, dataset_reader):
    scanner = dataset_reader.scanner(
        dataset, memory_pool=pa.default_memory_pool())
    assert isinstance(scanner, ds.Scanner)

    with pytest.raises(pa.ArrowInvalid):
        dataset_reader.scanner(dataset, columns=['unknown'])

    scanner = dataset_reader.scanner(dataset, columns=['i64'],
                                     memory_pool=pa.default_memory_pool())
    assert scanner.dataset_schema == dataset.schema
    assert scanner.projected_schema == pa.schema([("i64", pa.int64())])

    assert isinstance(scanner, ds.Scanner)
    table = scanner.to_table()
    for batch in scanner.to_batches():
        assert batch.schema == scanner.projected_schema
        assert batch.num_columns == 1
    assert table == scanner.to_reader().read_all()

    assert table.schema == scanner.projected_schema
    for i in range(table.num_rows):
        indices = pa.array([i])
        assert table.take(indices) == scanner.take(indices)
    with pytest.raises(pa.ArrowIndexError):
        scanner.take(pa.array([table.num_rows]))

    assert table.num_rows == scanner.count_rows()

    scanner = dataset_reader.scanner(dataset, columns=['__filename',
                                                       '__fragment_index',
                                                       '__batch_index',
                                                       '__last_in_fragment'],
                                     memory_pool=pa.default_memory_pool())
    table = scanner.to_table()
    expected_names = ['__filename', '__fragment_index',
                      '__batch_index', '__last_in_fragment']
    assert table.column_names == expected_names

    sorted_table = table.sort_by('__fragment_index')
    assert sorted_table['__filename'].to_pylist() == (
        ['subdir/1/xxx/file0.parquet'] * 5 +
        ['subdir/2/yyy/file1.parquet'] * 5)
    assert sorted_table['__fragment_index'].to_pylist() == ([0] * 5 + [1] * 5)
    assert sorted_table['__batch_index'].to_pylist() == [0] * 10
    assert sorted_table['__last_in_fragment'].to_pylist() == [True] * 10


@pytest.mark.parquet
def test_scanner_memory_pool(dataset):
    # honor default pool - https://issues.apache.org/jira/browse/ARROW-18164
    old_pool = pa.default_memory_pool()
    # TODO(ARROW-18293) we should be able to use the proxy memory pool for
    # for testing, but this crashes
    # pool = pa.proxy_memory_pool(old_pool)
    pool = pa.system_memory_pool()
    pa.set_memory_pool(pool)

    try:
        allocated_before = pool.bytes_allocated()
        scanner = ds.Scanner.from_dataset(dataset)
        _ = scanner.to_table()
        assert pool.bytes_allocated() > allocated_before
    finally:
        pa.set_memory_pool(old_pool)


@pytest.mark.parquet
def test_scanner_async_deprecated(dataset):
    with pytest.warns(FutureWarning):
        dataset.scanner(use_async=False)
    with pytest.warns(FutureWarning):
        dataset.scanner(use_async=True)
    with pytest.warns(FutureWarning):
        dataset.to_table(use_async=False)
    with pytest.warns(FutureWarning):
        dataset.to_table(use_async=True)
    with pytest.warns(FutureWarning):
        dataset.head(1, use_async=False)
    with pytest.warns(FutureWarning):
        dataset.head(1, use_async=True)
    with pytest.warns(FutureWarning):
        ds.Scanner.from_dataset(dataset, use_async=False)
    with pytest.warns(FutureWarning):
        ds.Scanner.from_dataset(dataset, use_async=True)
    with pytest.warns(FutureWarning):
        ds.Scanner.from_fragment(
            next(dataset.get_fragments()), use_async=False)
    with pytest.warns(FutureWarning):
        ds.Scanner.from_fragment(
            next(dataset.get_fragments()), use_async=True)


@pytest.mark.parquet
def test_head(dataset, dataset_reader):
    result = dataset_reader.head(dataset, 0)
    assert result == pa.Table.from_batches([], schema=dataset.schema)

    result = dataset_reader.head(dataset, 1, columns=['i64']).to_pydict()
    assert result == {'i64': [0]}

    result = dataset_reader.head(dataset, 2, columns=['i64'],
                                 filter=ds.field('i64') > 1).to_pydict()
    assert result == {'i64': [2, 3]}

    result = dataset_reader.head(dataset, 1024, columns=['i64']).to_pydict()
    assert result == {'i64': list(range(5)) * 2}

    fragment = next(dataset.get_fragments())
    result = fragment.head(1, columns=['i64']).to_pydict()
    assert result == {'i64': [0]}

    result = fragment.head(1024, columns=['i64']).to_pydict()
    assert result == {'i64': list(range(5))}


@pytest.mark.parquet
def test_take(dataset, dataset_reader):
    fragment = next(dataset.get_fragments())
    for indices in [[1, 3], pa.array([1, 3])]:
        expected = dataset_reader.to_table(fragment).take(indices)
        assert dataset_reader.take(fragment, indices) == expected
    with pytest.raises(IndexError):
        dataset_reader.take(fragment, pa.array([5]))

    for indices in [[1, 7], pa.array([1, 7])]:
        assert dataset_reader.take(
            dataset, indices) == dataset_reader.to_table(dataset).take(indices)
    with pytest.raises(IndexError):
        dataset_reader.take(dataset, pa.array([10]))


@pytest.mark.parquet
def test_count_rows(dataset, dataset_reader):
    fragment = next(dataset.get_fragments())
    assert dataset_reader.count_rows(fragment) == 5
    assert dataset_reader.count_rows(
        fragment, filter=ds.field("i64") == 4) == 1

    assert dataset_reader.count_rows(dataset) == 10
    # Filter on partition key
    assert dataset_reader.count_rows(
        dataset, filter=ds.field("group") == 1) == 5
    # Filter on data
    assert dataset_reader.count_rows(dataset, filter=ds.field("i64") >= 3) == 4
    assert dataset_reader.count_rows(dataset, filter=ds.field("i64") < 0) == 0


def test_abstract_classes():
    classes = [
        ds.FileFormat,
        ds.Scanner,
        ds.Partitioning,
    ]
    for klass in classes:
        with pytest.raises(TypeError):
            klass()


def test_partitioning():
    schema = pa.schema([
        pa.field('i64', pa.int64()),
        pa.field('f64', pa.float64())
    ])
    for klass in [ds.DirectoryPartitioning, ds.HivePartitioning,
                  ds.FilenamePartitioning]:
        partitioning = klass(schema)
        assert isinstance(partitioning, ds.Partitioning)

    partitioning = ds.DirectoryPartitioning(
        pa.schema([
            pa.field('group', pa.int64()),
            pa.field('key', pa.float64())
        ])
    )
    assert len(partitioning.dictionaries) == 2
    assert all(x is None for x in partitioning.dictionaries)
    expr = partitioning.parse('/3/3.14/')
    assert isinstance(expr, ds.Expression)

    expected = (ds.field('group') == 3) & (ds.field('key') == 3.14)
    assert expr.equals(expected)

    with pytest.raises(pa.ArrowInvalid):
        partitioning.parse('/prefix/3/aaa')

    expr = partitioning.parse('/3/')
    expected = ds.field('group') == 3
    assert expr.equals(expected)

    partitioning = ds.HivePartitioning(
        pa.schema([
            pa.field('alpha', pa.int64()),
            pa.field('beta', pa.int64())
        ]),
        null_fallback='xyz'
    )
    assert len(partitioning.dictionaries) == 2
    assert all(x is None for x in partitioning.dictionaries)
    expr = partitioning.parse('/alpha=0/beta=3/')
    expected = (
        (ds.field('alpha') == ds.scalar(0)) &
        (ds.field('beta') == ds.scalar(3))
    )
    assert expr.equals(expected)

    expr = partitioning.parse('/alpha=xyz/beta=3/')
    expected = (
        (ds.field('alpha').is_null() & (ds.field('beta') == ds.scalar(3)))
    )
    assert expr.equals(expected)

    for shouldfail in ['/alpha=one/beta=2/', '/alpha=one/', '/beta=two/']:
        with pytest.raises(pa.ArrowInvalid):
            partitioning.parse(shouldfail)

    partitioning = ds.FilenamePartitioning(
        pa.schema([
            pa.field('group', pa.int64()),
            pa.field('key', pa.float64())
        ])
    )
    assert len(partitioning.dictionaries) == 2
    assert all(x is None for x in partitioning.dictionaries)
    expr = partitioning.parse('3_3.14_')
    assert isinstance(expr, ds.Expression)

    expected = (ds.field('group') == 3) & (ds.field('key') == 3.14)
    assert expr.equals(expected)

    with pytest.raises(pa.ArrowInvalid):
        partitioning.parse('prefix_3_aaa_')

    partitioning = ds.DirectoryPartitioning(
        pa.schema([
            pa.field('group', pa.int64()),
            pa.field('key', pa.dictionary(pa.int8(), pa.string()))
        ]),
        dictionaries={
            "key": pa.array(["first", "second", "third"]),
        })
    assert partitioning.dictionaries[0] is None
    assert partitioning.dictionaries[1].to_pylist() == [
        "first", "second", "third"]

    partitioning = ds.FilenamePartitioning(
        pa.schema([
            pa.field('group', pa.int64()),
            pa.field('key', pa.dictionary(pa.int8(), pa.string()))
        ]),
        dictionaries={
            "key": pa.array(["first", "second", "third"]),
        })
    assert partitioning.dictionaries[0] is None
    assert partitioning.dictionaries[1].to_pylist() == [
        "first", "second", "third"]

    # test partitioning roundtrip
    table = pa.table([
        pa.array(range(20)), pa.array(np.random.randn(20)),
        pa.array(np.repeat(['a', 'b'], 10))],
        names=["f1", "f2", "part"]
    )
    partitioning_schema = pa.schema([("part", pa.string())])
    for klass in [ds.DirectoryPartitioning, ds.HivePartitioning,
                  ds.FilenamePartitioning]:
        with tempfile.TemporaryDirectory() as tempdir:
            partitioning = klass(partitioning_schema)
            ds.write_dataset(table, tempdir,
                             format='ipc', partitioning=partitioning)
            load_back = ds.dataset(tempdir, format='ipc',
                                   partitioning=partitioning)
            load_back_table = load_back.to_table()
            assert load_back_table.equals(table)


def test_expression_arithmetic_operators():
    dataset = ds.dataset(pa.table({'a': [1, 2, 3], 'b': [2, 2, 2]}))
    a = ds.field("a")
    b = ds.field("b")
    result = dataset.to_table(columns={
        "a+1": a + 1,
        "b-a": b - a,
        "a*2": a * 2,
        "a/b": a.cast("float64") / b,
    })
    expected = pa.table({
        "a+1": [2, 3, 4], "b-a": [1, 0, -1],
        "a*2": [2, 4, 6], "a/b": [0.5, 1.0, 1.5],
    })
    assert result.equals(expected)


def test_partition_keys():
    a, b, c = [ds.field(f) == f for f in 'abc']
    assert ds._get_partition_keys(a) == {'a': 'a'}
    assert ds._get_partition_keys(a & b & c) == {f: f for f in 'abc'}

    nope = ds.field('d') >= 3
    assert ds._get_partition_keys(nope) == {}
    assert ds._get_partition_keys(a & nope) == {'a': 'a'}

    null = ds.field('a').is_null()
    assert ds._get_partition_keys(null) == {'a': None}


@pytest.mark.parquet
def test_parquet_read_options():
    opts1 = ds.ParquetReadOptions()
    opts2 = ds.ParquetReadOptions(dictionary_columns=['a', 'b'])
    opts3 = ds.ParquetReadOptions(coerce_int96_timestamp_unit="ms")

    assert opts1.dictionary_columns == set()

    assert opts2.dictionary_columns == {'a', 'b'}

    assert opts1.coerce_int96_timestamp_unit == "ns"
    assert opts3.coerce_int96_timestamp_unit == "ms"

    assert opts1 == opts1
    assert opts1 != opts2
    assert opts1 != opts3


@pytest.mark.parquet
def test_parquet_file_format_read_options():
    pff1 = ds.ParquetFileFormat()
    pff2 = ds.ParquetFileFormat(dictionary_columns={'a'})
    pff3 = ds.ParquetFileFormat(coerce_int96_timestamp_unit="s")

    assert pff1.read_options == ds.ParquetReadOptions()
    assert pff2.read_options == ds.ParquetReadOptions(dictionary_columns=['a'])
    assert pff3.read_options == ds.ParquetReadOptions(
        coerce_int96_timestamp_unit="s")


@pytest.mark.parquet
def test_parquet_scan_options():
    opts1 = ds.ParquetFragmentScanOptions()
    opts2 = ds.ParquetFragmentScanOptions(buffer_size=4096)
    opts3 = ds.ParquetFragmentScanOptions(
        buffer_size=2**13, use_buffered_stream=True)
    opts4 = ds.ParquetFragmentScanOptions(buffer_size=2**13, pre_buffer=True)
    opts5 = ds.ParquetFragmentScanOptions(
        thrift_string_size_limit=123456,
        thrift_container_size_limit=987654,)

    assert opts1.use_buffered_stream is False
    assert opts1.buffer_size == 2**13
    assert opts1.pre_buffer is False
    assert opts1.thrift_string_size_limit == 100_000_000  # default in C++
    assert opts1.thrift_container_size_limit == 1_000_000  # default in C++

    assert opts2.use_buffered_stream is False
    assert opts2.buffer_size == 2**12
    assert opts2.pre_buffer is False

    assert opts3.use_buffered_stream is True
    assert opts3.buffer_size == 2**13
    assert opts3.pre_buffer is False

    assert opts4.use_buffered_stream is False
    assert opts4.buffer_size == 2**13
    assert opts4.pre_buffer is True

    assert opts5.thrift_string_size_limit == 123456
    assert opts5.thrift_container_size_limit == 987654

    assert opts1 == opts1
    assert opts1 != opts2
    assert opts2 != opts3
    assert opts3 != opts4
    assert opts5 != opts1


def test_file_format_pickling():
    formats = [
        ds.IpcFileFormat(),
        ds.CsvFileFormat(),
        ds.CsvFileFormat(pa.csv.ParseOptions(delimiter='\t',
                                             ignore_empty_lines=True)),
        ds.CsvFileFormat(read_options=pa.csv.ReadOptions(
            skip_rows=3, column_names=['foo'])),
        ds.CsvFileFormat(read_options=pa.csv.ReadOptions(
            skip_rows=3, block_size=2**20)),
    ]
    try:
        formats.append(ds.OrcFileFormat())
    except ImportError:
        pass

    if pq is not None:
        formats.extend([
            ds.ParquetFileFormat(),
            ds.ParquetFileFormat(dictionary_columns={'a'}),
            ds.ParquetFileFormat(use_buffered_stream=True),
            ds.ParquetFileFormat(
                use_buffered_stream=True,
                buffer_size=4096,
                thrift_string_size_limit=123,
                thrift_container_size_limit=456,
            ),
        ])

    for file_format in formats:
        assert pickle.loads(pickle.dumps(file_format)) == file_format


def test_fragment_scan_options_pickling():
    options = [
        ds.CsvFragmentScanOptions(),
        ds.CsvFragmentScanOptions(
            convert_options=pa.csv.ConvertOptions(strings_can_be_null=True)),
        ds.CsvFragmentScanOptions(
            read_options=pa.csv.ReadOptions(block_size=2**16)),
    ]

    if pq is not None:
        options.extend([
            ds.ParquetFragmentScanOptions(buffer_size=4096),
            ds.ParquetFragmentScanOptions(pre_buffer=True),
        ])

    for option in options:
        assert pickle.loads(pickle.dumps(option)) == option


@pytest.mark.parametrize('paths_or_selector', [
    fs.FileSelector('subdir', recursive=True),
    [
        'subdir/1/xxx/file0.parquet',
        'subdir/2/yyy/file1.parquet',
    ]
])
@pytest.mark.parametrize('pre_buffer', [False, True])
@pytest.mark.parquet
def test_filesystem_factory(mockfs, paths_or_selector, pre_buffer):
    format = ds.ParquetFileFormat(
        read_options=ds.ParquetReadOptions(dictionary_columns={"str"}),
        pre_buffer=pre_buffer
    )

    options = ds.FileSystemFactoryOptions('subdir')
    options.partitioning = ds.DirectoryPartitioning(
        pa.schema([
            pa.field('group', pa.int32()),
            pa.field('key', pa.string())
        ])
    )
    assert options.partition_base_dir == 'subdir'
    assert options.selector_ignore_prefixes == ['.', '_']
    assert options.exclude_invalid_files is False

    factory = ds.FileSystemDatasetFactory(
        mockfs, paths_or_selector, format, options
    )
    inspected_schema = factory.inspect()

    assert factory.inspect().equals(pa.schema([
        pa.field('i64', pa.int64()),
        pa.field('f64', pa.float64()),
        pa.field('str', pa.dictionary(pa.int32(), pa.string())),
        pa.field('const', pa.int64()),
        pa.field('struct', pa.struct({'a': pa.int64(),
                                      'b': pa.string()})),
        pa.field('group', pa.int32()),
        pa.field('key', pa.string()),
    ]), check_metadata=False)

    assert isinstance(factory.inspect_schemas(), list)
    assert isinstance(factory.finish(inspected_schema),
                      ds.FileSystemDataset)
    assert factory.root_partition.equals(ds.scalar(True))

    dataset = factory.finish()
    assert isinstance(dataset, ds.FileSystemDataset)

    scanner = dataset.scanner()
    expected_i64 = pa.array([0, 1, 2, 3, 4], type=pa.int64())
    expected_f64 = pa.array([0, 1, 2, 3, 4], type=pa.float64())
    expected_str = pa.DictionaryArray.from_arrays(
        pa.array([0, 1, 2, 3, 4], type=pa.int32()),
        pa.array("0 1 2 3 4".split(), type=pa.string())
    )
    expected_struct = pa.array([{'a': i % 3, 'b': str(i % 3)}
                                for i in range(5)])
    iterator = scanner.scan_batches()
    for (batch, fragment), group, key in zip(iterator, [1, 2], ['xxx', 'yyy']):
        expected_group = pa.array([group] * 5, type=pa.int32())
        expected_key = pa.array([key] * 5, type=pa.string())
        expected_const = pa.array([group - 1] * 5, type=pa.int64())
        # Can't compare or really introspect expressions from Python
        assert fragment.partition_expression is not None
        assert batch.num_columns == 7
        assert batch[0].equals(expected_i64)
        assert batch[1].equals(expected_f64)
        assert batch[2].equals(expected_str)
        assert batch[3].equals(expected_const)
        assert batch[4].equals(expected_struct)
        assert batch[5].equals(expected_group)
        assert batch[6].equals(expected_key)

    table = dataset.to_table()
    assert isinstance(table, pa.Table)
    assert len(table) == 10
    assert table.num_columns == 7


@pytest.mark.parquet
def test_make_fragment(multisourcefs):
    parquet_format = ds.ParquetFileFormat()
    dataset = ds.dataset('/plain', filesystem=multisourcefs,
                         format=parquet_format)

    for path in dataset.files:
        fragment = parquet_format.make_fragment(path, multisourcefs)
        assert fragment.row_groups == [0]

        row_group_fragment = parquet_format.make_fragment(path, multisourcefs,
                                                          row_groups=[0])
        for f in [fragment, row_group_fragment]:
            assert isinstance(f, ds.ParquetFileFragment)
            assert f.path == path
            assert isinstance(f.filesystem, type(multisourcefs))
        assert row_group_fragment.row_groups == [0]


def test_make_csv_fragment_from_buffer(dataset_reader):
    content = textwrap.dedent("""
        alpha,num,animal
        a,12,dog
        b,11,cat
        c,10,rabbit
    """)
    buffer = pa.py_buffer(content.encode('utf-8'))

    csv_format = ds.CsvFileFormat()
    fragment = csv_format.make_fragment(buffer)

    # When buffer, fragment open returns a BufferReader, not NativeFile
    assert isinstance(fragment.open(), pa.BufferReader)

    expected = pa.table([['a', 'b', 'c'],
                         [12, 11, 10],
                         ['dog', 'cat', 'rabbit']],
                        names=['alpha', 'num', 'animal'])
    assert dataset_reader.to_table(fragment).equals(expected)

    pickled = pickle.loads(pickle.dumps(fragment))
    assert dataset_reader.to_table(pickled).equals(fragment.to_table())


@pytest.mark.parquet
def test_make_parquet_fragment_from_buffer(dataset_reader):
    arrays = [
        pa.array(['a', 'b', 'c']),
        pa.array([12, 11, 10]),
        pa.array(['dog', 'cat', 'rabbit'])
    ]
    dictionary_arrays = [
        arrays[0].dictionary_encode(),
        arrays[1],
        arrays[2].dictionary_encode()
    ]
    dictionary_format = ds.ParquetFileFormat(
        read_options=ds.ParquetReadOptions(
            dictionary_columns=['alpha', 'animal']
        ),
        use_buffered_stream=True,
        buffer_size=4096,
    )

    cases = [
        (arrays, ds.ParquetFileFormat()),
        (dictionary_arrays, dictionary_format)
    ]
    for arrays, format_ in cases:
        table = pa.table(arrays, names=['alpha', 'num', 'animal'])

        out = pa.BufferOutputStream()
        pq.write_table(table, out)
        buffer = out.getvalue()

        fragment = format_.make_fragment(buffer)
        assert dataset_reader.to_table(fragment).equals(table)

        pickled = pickle.loads(pickle.dumps(fragment))
        assert dataset_reader.to_table(pickled).equals(table)


@pytest.mark.parquet
def _create_dataset_for_fragments(tempdir, chunk_size=None, filesystem=None):
    table = pa.table(
        [range(8), [1] * 8, ['a'] * 4 + ['b'] * 4],
        names=['f1', 'f2', 'part']
    )

    path = str(tempdir / "test_parquet_dataset")

    # write_to_dataset currently requires pandas
    pq.write_to_dataset(table, path,
                        partition_cols=["part"], chunk_size=chunk_size)
    dataset = ds.dataset(
        path, format="parquet", partitioning="hive", filesystem=filesystem
    )

    return table, dataset


@pytest.mark.pandas
@pytest.mark.parquet
@pytest.mark.filterwarnings(
    "ignore:Passing 'use_legacy_dataset=True':FutureWarning")
def test_fragments(tempdir, dataset_reader):
    table, dataset = _create_dataset_for_fragments(tempdir)

    # list fragments
    fragments = list(dataset.get_fragments())
    assert len(fragments) == 2
    f = fragments[0]

    physical_names = ['f1', 'f2']
    # file's schema does not include partition column
    assert f.physical_schema.names == physical_names
    assert f.format.inspect(f.path, f.filesystem) == f.physical_schema
    assert f.partition_expression.equals(ds.field('part') == 'a')

    # By default, the partition column is not part of the schema.
    result = dataset_reader.to_table(f)
    assert result.column_names == physical_names
    assert result.equals(table.remove_column(2).slice(0, 4))

    # scanning fragment includes partition columns when given the proper
    # schema.
    result = dataset_reader.to_table(f, schema=dataset.schema)
    assert result.column_names == ['f1', 'f2', 'part']
    assert result.equals(table.slice(0, 4))
    assert f.physical_schema == result.schema.remove(2)

    # scanning fragments follow filter predicate
    result = dataset_reader.to_table(
        f, schema=dataset.schema, filter=ds.field('f1') < 2)
    assert result.column_names == ['f1', 'f2', 'part']


@pytest.mark.pandas
@pytest.mark.parquet
def test_fragments_implicit_cast(tempdir):
    # ARROW-8693
    table = pa.table([range(8), [1] * 4 + [2] * 4], names=['col', 'part'])
    path = str(tempdir / "test_parquet_dataset")
    pq.write_to_dataset(table, path, partition_cols=["part"])

    part = ds.partitioning(pa.schema([('part', 'int8')]), flavor="hive")
    dataset = ds.dataset(path, format="parquet", partitioning=part)
    fragments = dataset.get_fragments(filter=ds.field("part") >= 2)
    assert len(list(fragments)) == 1


@pytest.mark.pandas
@pytest.mark.parquet
@pytest.mark.filterwarnings(
    "ignore:Passing 'use_legacy_dataset=True':FutureWarning")
def test_fragments_reconstruct(tempdir, dataset_reader):
    table, dataset = _create_dataset_for_fragments(tempdir)

    def assert_yields_projected(fragment, row_slice,
                                columns=None, filter=None):
        actual = fragment.to_table(
            schema=table.schema, columns=columns, filter=filter)
        column_names = columns if columns else table.column_names
        assert actual.column_names == column_names

        expected = table.slice(*row_slice).select(column_names)
        assert actual.equals(expected)

    fragment = list(dataset.get_fragments())[0]
    parquet_format = fragment.format

    # test pickle roundtrip
    pickled_fragment = pickle.loads(pickle.dumps(fragment))
    assert dataset_reader.to_table(
        pickled_fragment) == dataset_reader.to_table(fragment)

    # manually re-construct a fragment, with explicit schema
    new_fragment = parquet_format.make_fragment(
        fragment.path, fragment.filesystem,
        partition_expression=fragment.partition_expression)
    assert dataset_reader.to_table(new_fragment).equals(
        dataset_reader.to_table(fragment))
    assert_yields_projected(new_fragment, (0, 4))

    # filter / column projection, inspected schema
    new_fragment = parquet_format.make_fragment(
        fragment.path, fragment.filesystem,
        partition_expression=fragment.partition_expression)
    assert_yields_projected(new_fragment, (0, 2), filter=ds.field('f1') < 2)

    # filter requiring cast / column projection, inspected schema
    new_fragment = parquet_format.make_fragment(
        fragment.path, fragment.filesystem,
        partition_expression=fragment.partition_expression)
    assert_yields_projected(new_fragment, (0, 2),
                            columns=['f1'], filter=ds.field('f1') < 2.0)

    # filter on the partition column
    new_fragment = parquet_format.make_fragment(
        fragment.path, fragment.filesystem,
        partition_expression=fragment.partition_expression)
    assert_yields_projected(new_fragment, (0, 4),
                            filter=ds.field('part') == 'a')

    # Fragments don't contain the partition's columns if not provided to the
    # `to_table(schema=...)` method.
    pattern = (r'No match for FieldRef.Name\(part\) in ' +
               fragment.physical_schema.to_string(False, False, False))
    with pytest.raises(ValueError, match=pattern):
        new_fragment = parquet_format.make_fragment(
            fragment.path, fragment.filesystem,
            partition_expression=fragment.partition_expression)
        dataset_reader.to_table(new_fragment, filter=ds.field('part') == 'a')


@pytest.mark.pandas
@pytest.mark.parquet
@pytest.mark.filterwarnings(
    "ignore:Passing 'use_legacy_dataset=True':FutureWarning")
def test_fragments_parquet_row_groups(tempdir, dataset_reader):
    table, dataset = _create_dataset_for_fragments(tempdir, chunk_size=2)

    fragment = list(dataset.get_fragments())[0]

    # list and scan row group fragments
    row_group_fragments = list(fragment.split_by_row_group())
    assert len(row_group_fragments) == fragment.num_row_groups == 2
    result = dataset_reader.to_table(
        row_group_fragments[0], schema=dataset.schema)
    assert result.column_names == ['f1', 'f2', 'part']
    assert len(result) == 2
    assert result.equals(table.slice(0, 2))

    assert row_group_fragments[0].row_groups is not None
    assert row_group_fragments[0].num_row_groups == 1
    assert row_group_fragments[0].row_groups[0].statistics == {
        'f1': {'min': 0, 'max': 1},
        'f2': {'min': 1, 'max': 1},
    }

    fragment = list(dataset.get_fragments(filter=ds.field('f1') < 1))[0]
    row_group_fragments = list(fragment.split_by_row_group(ds.field('f1') < 1))
    assert len(row_group_fragments) == 1
    result = dataset_reader.to_table(
        row_group_fragments[0], filter=ds.field('f1') < 1)
    assert len(result) == 1


@pytest.mark.parquet
def test_fragments_parquet_num_row_groups(tempdir):
    table = pa.table({'a': range(8)})
    pq.write_table(table, tempdir / "test.parquet", row_group_size=2)
    dataset = ds.dataset(tempdir / "test.parquet", format="parquet")
    original_fragment = list(dataset.get_fragments())[0]

    # create fragment with subset of row groups
    fragment = original_fragment.format.make_fragment(
        original_fragment.path, original_fragment.filesystem,
        row_groups=[1, 3])
    assert fragment.num_row_groups == 2
    # ensure that parsing metadata preserves correct number of row groups
    fragment.ensure_complete_metadata()
    assert fragment.num_row_groups == 2
    assert len(fragment.row_groups) == 2


@pytest.mark.pandas
@pytest.mark.parquet
def test_fragments_parquet_row_groups_dictionary(tempdir, dataset_reader):
    import pandas as pd

    df = pd.DataFrame(dict(col1=['a', 'b'], col2=[1, 2]))
    df['col1'] = df['col1'].astype("category")

    pq.write_table(pa.table(df), tempdir / "test_filter_dictionary.parquet")

    import pyarrow.dataset as ds
    dataset = ds.dataset(tempdir / 'test_filter_dictionary.parquet')
    result = dataset_reader.to_table(dataset, filter=ds.field("col1") == "a")

    assert (df.iloc[0] == result.to_pandas()).all().all()


@pytest.mark.pandas
@pytest.mark.parquet
@pytest.mark.filterwarnings(
    "ignore:Passing 'use_legacy_dataset=True':FutureWarning")
def test_fragments_parquet_ensure_metadata(tempdir, open_logging_fs):
    fs, assert_opens = open_logging_fs
    _, dataset = _create_dataset_for_fragments(
        tempdir, chunk_size=2, filesystem=fs
    )
    fragment = list(dataset.get_fragments())[0]

    # with default discovery, no metadata loaded
    with assert_opens([fragment.path]):
        fragment.ensure_complete_metadata()
    assert fragment.row_groups == [0, 1]

    # second time -> use cached / no file IO
    with assert_opens([]):
        fragment.ensure_complete_metadata()

    assert isinstance(fragment.metadata, pq.FileMetaData)

    # recreate fragment with row group ids
    new_fragment = fragment.format.make_fragment(
        fragment.path, fragment.filesystem, row_groups=[0, 1]
    )
    assert new_fragment.row_groups == fragment.row_groups

    # collect metadata
    new_fragment.ensure_complete_metadata()
    row_group = new_fragment.row_groups[0]
    assert row_group.id == 0
    assert row_group.num_rows == 2
    assert row_group.statistics is not None

    # pickling preserves row group ids
    pickled_fragment = pickle.loads(pickle.dumps(new_fragment))
    with assert_opens([fragment.path]):
        assert pickled_fragment.row_groups == [0, 1]
        row_group = pickled_fragment.row_groups[0]
        assert row_group.id == 0
        assert row_group.statistics is not None


@pytest.mark.pandas
@pytest.mark.parquet
def test_fragments_parquet_pickle_no_metadata(tempdir, open_logging_fs):
    # https://issues.apache.org/jira/browse/ARROW-15796
    fs, assert_opens = open_logging_fs
    _, dataset = _create_dataset_for_fragments(tempdir, filesystem=fs)
    fragment = list(dataset.get_fragments())[1]

    # second fragment hasn't yet loaded the metadata,
    # and pickling it also should not read the metadata
    with assert_opens([]):
        pickled_fragment = pickle.loads(pickle.dumps(fragment))

    # then accessing the row group info reads the metadata
    with assert_opens([pickled_fragment.path]):
        row_groups = pickled_fragment.row_groups
    assert row_groups == [0]


def _create_dataset_all_types(tempdir, chunk_size=None):
    table = pa.table(
        [
            pa.array([True, None, False], pa.bool_()),
            pa.array([1, 10, 42], pa.int8()),
            pa.array([1, 10, 42], pa.uint8()),
            pa.array([1, 10, 42], pa.int16()),
            pa.array([1, 10, 42], pa.uint16()),
            pa.array([1, 10, 42], pa.int32()),
            pa.array([1, 10, 42], pa.uint32()),
            pa.array([1, 10, 42], pa.int64()),
            pa.array([1, 10, 42], pa.uint64()),
            pa.array([1.0, 10.0, 42.0], pa.float32()),
            pa.array([1.0, 10.0, 42.0], pa.float64()),
            pa.array(['a', None, 'z'], pa.utf8()),
            pa.array(['a', None, 'z'], pa.binary()),
            pa.array([1, 10, 42], pa.timestamp('s')),
            pa.array([1, 10, 42], pa.timestamp('ms')),
            pa.array([1, 10, 42], pa.timestamp('us')),
            pa.array([1, 10, 42], pa.date32()),
            pa.array([1, 10, 4200000000], pa.date64()),
            pa.array([1, 10, 42], pa.time32('s')),
            pa.array([1, 10, 42], pa.time64('us')),
        ],
        names=[
            'boolean',
            'int8',
            'uint8',
            'int16',
            'uint16',
            'int32',
            'uint32',
            'int64',
            'uint64',
            'float',
            'double',
            'utf8',
            'binary',
            'ts[s]',
            'ts[ms]',
            'ts[us]',
            'date32',
            'date64',
            'time32',
            'time64',
        ]
    )

    path = str(tempdir / "test_parquet_dataset_all_types")

    # write_to_dataset currently requires pandas
    pq.write_to_dataset(table, path, use_legacy_dataset=True,
                        chunk_size=chunk_size)

    return table, ds.dataset(path, format="parquet", partitioning="hive")


@pytest.mark.pandas
@pytest.mark.parquet
@pytest.mark.filterwarnings(
    "ignore:Passing 'use_legacy_dataset=True':FutureWarning")
def test_parquet_fragment_statistics(tempdir):
    table, dataset = _create_dataset_all_types(tempdir)

    fragment = list(dataset.get_fragments())[0]

    import datetime
    def dt_s(x): return datetime.datetime(1970, 1, 1, 0, 0, x)
    def dt_ms(x): return datetime.datetime(1970, 1, 1, 0, 0, 0, x*1000)
    def dt_us(x): return datetime.datetime(1970, 1, 1, 0, 0, 0, x)
    date = datetime.date
    time = datetime.time

    # list and scan row group fragments
    row_group_fragments = list(fragment.split_by_row_group())
    assert row_group_fragments[0].row_groups is not None
    row_group = row_group_fragments[0].row_groups[0]
    assert row_group.num_rows == 3
    assert row_group.total_byte_size > 1000
    assert row_group.statistics == {
        'boolean': {'min': False, 'max': True},
        'int8': {'min': 1, 'max': 42},
        'uint8': {'min': 1, 'max': 42},
        'int16': {'min': 1, 'max': 42},
        'uint16': {'min': 1, 'max': 42},
        'int32': {'min': 1, 'max': 42},
        'uint32': {'min': 1, 'max': 42},
        'int64': {'min': 1, 'max': 42},
        'uint64': {'min': 1, 'max': 42},
        'float': {'min': 1.0, 'max': 42.0},
        'double': {'min': 1.0, 'max': 42.0},
        'utf8': {'min': 'a', 'max': 'z'},
        'binary': {'min': b'a', 'max': b'z'},
        'ts[s]': {'min': dt_s(1), 'max': dt_s(42)},
        'ts[ms]': {'min': dt_ms(1), 'max': dt_ms(42)},
        'ts[us]': {'min': dt_us(1), 'max': dt_us(42)},
        'date32': {'min': date(1970, 1, 2), 'max': date(1970, 2, 12)},
        'date64': {'min': date(1970, 1, 1), 'max': date(1970, 2, 18)},
        'time32': {'min': time(0, 0, 1), 'max': time(0, 0, 42)},
        'time64': {'min': time(0, 0, 0, 1), 'max': time(0, 0, 0, 42)},
    }


@pytest.mark.parquet
def test_parquet_fragment_statistics_nulls(tempdir):
    table = pa.table({'a': [0, 1, None, None], 'b': ['a', 'b', None, None]})
    pq.write_table(table, tempdir / "test.parquet", row_group_size=2)

    dataset = ds.dataset(tempdir / "test.parquet", format="parquet")
    fragments = list(dataset.get_fragments())[0].split_by_row_group()
    # second row group has all nulls -> no statistics
    assert fragments[1].row_groups[0].statistics == {}


@pytest.mark.pandas
@pytest.mark.parquet
def test_parquet_empty_row_group_statistics(tempdir):
    df = pd.DataFrame({"a": ["a", "b", "b"], "b": [4, 5, 6]})[:0]
    df.to_parquet(tempdir / "test.parquet", engine="pyarrow")

    dataset = ds.dataset(tempdir / "test.parquet", format="parquet")
    fragments = list(dataset.get_fragments())[0].split_by_row_group()
    # Only row group is empty
    assert fragments[0].row_groups[0].statistics == {}


@pytest.mark.pandas
@pytest.mark.parquet
@pytest.mark.filterwarnings(
    "ignore:Passing 'use_legacy_dataset=True':FutureWarning")
def test_fragments_parquet_row_groups_predicate(tempdir):
    table, dataset = _create_dataset_for_fragments(tempdir, chunk_size=2)

    fragment = list(dataset.get_fragments())[0]
    assert fragment.partition_expression.equals(ds.field('part') == 'a')

    # predicate may reference a partition field not present in the
    # physical_schema if an explicit schema is provided to split_by_row_group

    # filter matches partition_expression: all row groups
    row_group_fragments = list(
        fragment.split_by_row_group(filter=ds.field('part') == 'a',
                                    schema=dataset.schema))
    assert len(row_group_fragments) == 2

    # filter contradicts partition_expression: no row groups
    row_group_fragments = list(
        fragment.split_by_row_group(filter=ds.field('part') == 'b',
                                    schema=dataset.schema))
    assert len(row_group_fragments) == 0


@pytest.mark.pandas
@pytest.mark.parquet
@pytest.mark.filterwarnings(
    "ignore:Passing 'use_legacy_dataset=True':FutureWarning")
def test_fragments_parquet_row_groups_reconstruct(tempdir, dataset_reader):
    table, dataset = _create_dataset_for_fragments(tempdir, chunk_size=2)

    fragment = list(dataset.get_fragments())[0]
    parquet_format = fragment.format
    row_group_fragments = list(fragment.split_by_row_group())

    # test pickle roundtrip
    pickled_fragment = pickle.loads(pickle.dumps(fragment))
    assert dataset_reader.to_table(
        pickled_fragment) == dataset_reader.to_table(fragment)

    # manually re-construct row group fragments
    new_fragment = parquet_format.make_fragment(
        fragment.path, fragment.filesystem,
        partition_expression=fragment.partition_expression,
        row_groups=[0])
    result = dataset_reader.to_table(new_fragment)
    assert result.equals(dataset_reader.to_table(row_group_fragments[0]))

    # manually re-construct a row group fragment with filter/column projection
    new_fragment = parquet_format.make_fragment(
        fragment.path, fragment.filesystem,
        partition_expression=fragment.partition_expression,
        row_groups={1})
    result = dataset_reader.to_table(
        new_fragment, schema=table.schema, columns=['f1', 'part'],
        filter=ds.field('f1') < 3, )
    assert result.column_names == ['f1', 'part']
    assert len(result) == 1

    # out of bounds row group index
    new_fragment = parquet_format.make_fragment(
        fragment.path, fragment.filesystem,
        partition_expression=fragment.partition_expression,
        row_groups={2})
    with pytest.raises(IndexError, match="references row group 2"):
        dataset_reader.to_table(new_fragment)


@pytest.mark.pandas
@pytest.mark.parquet
@pytest.mark.filterwarnings(
    "ignore:Passing 'use_legacy_dataset=True':FutureWarning")
def test_fragments_parquet_subset_ids(tempdir, open_logging_fs,
                                      dataset_reader):
    fs, assert_opens = open_logging_fs
    table, dataset = _create_dataset_for_fragments(tempdir, chunk_size=1,
                                                   filesystem=fs)
    fragment = list(dataset.get_fragments())[0]

    # select with row group ids
    subfrag = fragment.subset(row_group_ids=[0, 3])
    with assert_opens([]):
        assert subfrag.num_row_groups == 2
        assert subfrag.row_groups == [0, 3]
        assert subfrag.row_groups[0].statistics is not None

    # check correct scan result of subset
    result = dataset_reader.to_table(subfrag)
    assert result.to_pydict() == {"f1": [0, 3], "f2": [1, 1]}

    # empty list of ids
    subfrag = fragment.subset(row_group_ids=[])
    assert subfrag.num_row_groups == 0
    assert subfrag.row_groups == []
    result = dataset_reader.to_table(subfrag, schema=dataset.schema)
    assert result.num_rows == 0
    assert result.equals(table[:0])


@pytest.mark.pandas
@pytest.mark.parquet
@pytest.mark.filterwarnings(
    "ignore:Passing 'use_legacy_dataset=True':FutureWarning")
def test_fragments_parquet_subset_filter(tempdir, open_logging_fs,
                                         dataset_reader):
    fs, assert_opens = open_logging_fs
    table, dataset = _create_dataset_for_fragments(tempdir, chunk_size=1,
                                                   filesystem=fs)
    fragment = list(dataset.get_fragments())[0]

    # select with filter
    subfrag = fragment.subset(ds.field("f1") >= 1)
    with assert_opens([]):
        assert subfrag.num_row_groups == 3
        assert len(subfrag.row_groups) == 3
        assert subfrag.row_groups[0].statistics is not None

    # check correct scan result of subset
    result = dataset_reader.to_table(subfrag)
    assert result.to_pydict() == {"f1": [1, 2, 3], "f2": [1, 1, 1]}

    # filter that results in empty selection
    subfrag = fragment.subset(ds.field("f1") > 5)
    assert subfrag.num_row_groups == 0
    assert subfrag.row_groups == []
    result = dataset_reader.to_table(subfrag, schema=dataset.schema)
    assert result.num_rows == 0
    assert result.equals(table[:0])

    # passing schema to ensure filter on partition expression works
    subfrag = fragment.subset(ds.field("part") == "a", schema=dataset.schema)
    assert subfrag.num_row_groups == 4


@pytest.mark.pandas
@pytest.mark.parquet
@pytest.mark.filterwarnings(
    "ignore:Passing 'use_legacy_dataset=True':FutureWarning")
def test_fragments_parquet_subset_invalid(tempdir):
    _, dataset = _create_dataset_for_fragments(tempdir, chunk_size=1)
    fragment = list(dataset.get_fragments())[0]

    # passing none or both of filter / row_group_ids
    with pytest.raises(ValueError):
        fragment.subset(ds.field("f1") >= 1, row_group_ids=[1, 2])

    with pytest.raises(ValueError):
        fragment.subset()


@pytest.mark.pandas
@pytest.mark.parquet
def test_fragments_repr(tempdir, dataset):
    # partitioned parquet dataset
    fragment = list(dataset.get_fragments())[0]
    assert (
        repr(fragment) ==
        "<pyarrow.dataset.ParquetFileFragment path=subdir/1/xxx/file0.parquet "
        "partition=[key=xxx, group=1]>"
    )

    # single-file parquet dataset (no partition information in repr)
    table, path = _create_single_file(tempdir)
    dataset = ds.dataset(path, format="parquet")
    fragment = list(dataset.get_fragments())[0]
    assert (
        repr(fragment) ==
        "<pyarrow.dataset.ParquetFileFragment path={}>".format(
            dataset.filesystem.normalize_path(str(path)))
    )

    # non-parquet format
    path = tempdir / "data.feather"
    pa.feather.write_feather(table, path)
    dataset = ds.dataset(path, format="feather")
    fragment = list(dataset.get_fragments())[0]
    assert (
        repr(fragment) ==
        "<pyarrow.dataset.FileFragment type=ipc path={}>".format(
            dataset.filesystem.normalize_path(str(path)))
    )


@pytest.mark.parquet
def test_partitioning_factory(mockfs):
    paths_or_selector = fs.FileSelector('subdir', recursive=True)
    format = ds.ParquetFileFormat()

    options = ds.FileSystemFactoryOptions('subdir')
    partitioning_factory = ds.DirectoryPartitioning.discover(['group', 'key'])
    assert isinstance(partitioning_factory, ds.PartitioningFactory)
    options.partitioning_factory = partitioning_factory

    factory = ds.FileSystemDatasetFactory(
        mockfs, paths_or_selector, format, options
    )
    inspected_schema = factory.inspect()
    # i64/f64 from data, group/key from "/1/xxx" and "/2/yyy" paths
    expected_schema = pa.schema([
        ("i64", pa.int64()),
        ("f64", pa.float64()),
        ("str", pa.string()),
        ("const", pa.int64()),
        ("struct", pa.struct({'a': pa.int64(), 'b': pa.string()})),
        ("group", pa.int32()),
        ("key", pa.string()),
    ])
    assert inspected_schema.equals(expected_schema)

    hive_partitioning_factory = ds.HivePartitioning.discover()
    assert isinstance(hive_partitioning_factory, ds.PartitioningFactory)


@pytest.mark.parquet
@pytest.mark.parametrize('infer_dictionary', [False, True])
def test_partitioning_factory_dictionary(mockfs, infer_dictionary):
    paths_or_selector = fs.FileSelector('subdir', recursive=True)
    format = ds.ParquetFileFormat()
    options = ds.FileSystemFactoryOptions('subdir')

    options.partitioning_factory = ds.DirectoryPartitioning.discover(
        ['group', 'key'], infer_dictionary=infer_dictionary)

    factory = ds.FileSystemDatasetFactory(
        mockfs, paths_or_selector, format, options)

    inferred_schema = factory.inspect()
    if infer_dictionary:
        expected_type = pa.dictionary(pa.int32(), pa.string())
        assert inferred_schema.field('key').type == expected_type

        table = factory.finish().to_table().combine_chunks()
        actual = table.column('key').chunk(0)
        expected = pa.array(['xxx'] * 5 + ['yyy'] * 5).dictionary_encode()
        assert actual.equals(expected)

        # ARROW-9345 ensure filtering on the partition field works
        table = factory.finish().to_table(filter=ds.field('key') == 'xxx')
        actual = table.column('key').chunk(0)
        expected = expected.slice(0, 5)
        assert actual.equals(expected)
    else:
        assert inferred_schema.field('key').type == pa.string()


def test_partitioning_factory_segment_encoding():
    mockfs = fs._MockFileSystem()
    format = ds.IpcFileFormat()
    schema = pa.schema([("i64", pa.int64())])
    table = pa.table([pa.array(range(10))], schema=schema)
    partition_schema = pa.schema(
        [("date", pa.timestamp("s")), ("string", pa.string())])
    string_partition_schema = pa.schema(
        [("date", pa.string()), ("string", pa.string())])
    full_schema = pa.schema(list(schema) + list(partition_schema))
    for directory in [
            "directory/2021-05-04 00%3A00%3A00/%24",
            "hive/date=2021-05-04 00%3A00%3A00/string=%24",
    ]:
        mockfs.create_dir(directory)
        with mockfs.open_output_stream(directory + "/0.feather") as sink:
            with pa.ipc.new_file(sink, schema) as writer:
                writer.write_table(table)
                writer.close()

    # Directory
    selector = fs.FileSelector("directory", recursive=True)
    options = ds.FileSystemFactoryOptions("directory")
    options.partitioning_factory = ds.DirectoryPartitioning.discover(
        schema=partition_schema)
    factory = ds.FileSystemDatasetFactory(mockfs, selector, format, options)
    inferred_schema = factory.inspect()
    assert inferred_schema == full_schema
    actual = factory.finish().to_table(columns={
        "date_int": ds.field("date").cast(pa.int64()),
    })
    assert actual[0][0].as_py() == 1620086400

    options.partitioning_factory = ds.DirectoryPartitioning.discover(
        ["date", "string"], segment_encoding="none")
    factory = ds.FileSystemDatasetFactory(mockfs, selector, format, options)
    fragments = list(factory.finish().get_fragments())
    assert fragments[0].partition_expression.equals(
        (ds.field("date") == "2021-05-04 00%3A00%3A00") &
        (ds.field("string") == "%24"))

    options.partitioning = ds.DirectoryPartitioning(
        string_partition_schema, segment_encoding="none")
    factory = ds.FileSystemDatasetFactory(mockfs, selector, format, options)
    fragments = list(factory.finish().get_fragments())
    assert fragments[0].partition_expression.equals(
        (ds.field("date") == "2021-05-04 00%3A00%3A00") &
        (ds.field("string") == "%24"))

    options.partitioning_factory = ds.DirectoryPartitioning.discover(
        schema=partition_schema, segment_encoding="none")
    factory = ds.FileSystemDatasetFactory(mockfs, selector, format, options)
    with pytest.raises(pa.ArrowInvalid,
                       match="Could not cast segments for partition field"):
        inferred_schema = factory.inspect()

    # Hive
    selector = fs.FileSelector("hive", recursive=True)
    options = ds.FileSystemFactoryOptions("hive")
    options.partitioning_factory = ds.HivePartitioning.discover(
        schema=partition_schema)
    factory = ds.FileSystemDatasetFactory(mockfs, selector, format, options)
    inferred_schema = factory.inspect()
    assert inferred_schema == full_schema
    actual = factory.finish().to_table(columns={
        "date_int": ds.field("date").cast(pa.int64()),
    })
    assert actual[0][0].as_py() == 1620086400

    options.partitioning_factory = ds.HivePartitioning.discover(
        segment_encoding="none")
    factory = ds.FileSystemDatasetFactory(mockfs, selector, format, options)
    fragments = list(factory.finish().get_fragments())
    assert fragments[0].partition_expression.equals(
        (ds.field("date") == "2021-05-04 00%3A00%3A00") &
        (ds.field("string") == "%24"))

    options.partitioning = ds.HivePartitioning(
        string_partition_schema, segment_encoding="none")
    factory = ds.FileSystemDatasetFactory(mockfs, selector, format, options)
    fragments = list(factory.finish().get_fragments())
    assert fragments[0].partition_expression.equals(
        (ds.field("date") == "2021-05-04 00%3A00%3A00") &
        (ds.field("string") == "%24"))

    options.partitioning_factory = ds.HivePartitioning.discover(
        schema=partition_schema, segment_encoding="none")
    factory = ds.FileSystemDatasetFactory(mockfs, selector, format, options)
    with pytest.raises(pa.ArrowInvalid,
                       match="Could not cast segments for partition field"):
        inferred_schema = factory.inspect()


def test_partitioning_factory_hive_segment_encoding_key_encoded():
    mockfs = fs._MockFileSystem()
    format = ds.IpcFileFormat()
    schema = pa.schema([("i64", pa.int64())])
    table = pa.table([pa.array(range(10))], schema=schema)
    partition_schema = pa.schema(
        [("test'; date", pa.timestamp("s")), ("test';[ string'", pa.string())])
    string_partition_schema = pa.schema(
        [("test'; date", pa.string()), ("test';[ string'", pa.string())])
    full_schema = pa.schema(list(schema) + list(partition_schema))

    partition_schema_en = pa.schema(
        [("test%27%3B%20date", pa.timestamp("s")),
         ("test%27%3B%5B%20string%27", pa.string())])
    string_partition_schema_en = pa.schema(
        [("test%27%3B%20date", pa.string()),
         ("test%27%3B%5B%20string%27", pa.string())])

    directory = ("hive/test%27%3B%20date=2021-05-04 00%3A00%3A00/"
                 "test%27%3B%5B%20string%27=%24")
    mockfs.create_dir(directory)
    with mockfs.open_output_stream(directory + "/0.feather") as sink:
        with pa.ipc.new_file(sink, schema) as writer:
            writer.write_table(table)
            writer.close()

    # Hive
    selector = fs.FileSelector("hive", recursive=True)
    options = ds.FileSystemFactoryOptions("hive")
    options.partitioning_factory = ds.HivePartitioning.discover(
        schema=partition_schema)
    factory = ds.FileSystemDatasetFactory(mockfs, selector, format, options)
    inferred_schema = factory.inspect()
    assert inferred_schema == full_schema
    actual = factory.finish().to_table(columns={
        "date_int": ds.field("test'; date").cast(pa.int64()),
    })
    assert actual[0][0].as_py() == 1620086400

    options.partitioning_factory = ds.HivePartitioning.discover(
        segment_encoding="uri")
    factory = ds.FileSystemDatasetFactory(mockfs, selector, format, options)
    fragments = list(factory.finish().get_fragments())
    assert fragments[0].partition_expression.equals(
        (ds.field("test'; date") == "2021-05-04 00:00:00") &
        (ds.field("test';[ string'") == "$"))

    options.partitioning = ds.HivePartitioning(
        string_partition_schema, segment_encoding="uri")
    factory = ds.FileSystemDatasetFactory(mockfs, selector, format, options)
    fragments = list(factory.finish().get_fragments())
    assert fragments[0].partition_expression.equals(
        (ds.field("test'; date") == "2021-05-04 00:00:00") &
        (ds.field("test';[ string'") == "$"))

    options.partitioning_factory = ds.HivePartitioning.discover(
        segment_encoding="none")
    factory = ds.FileSystemDatasetFactory(mockfs, selector, format, options)
    fragments = list(factory.finish().get_fragments())
    assert fragments[0].partition_expression.equals(
        (ds.field("test%27%3B%20date") == "2021-05-04 00%3A00%3A00") &
        (ds.field("test%27%3B%5B%20string%27") == "%24"))

    options.partitioning = ds.HivePartitioning(
        string_partition_schema_en, segment_encoding="none")
    factory = ds.FileSystemDatasetFactory(mockfs, selector, format, options)
    fragments = list(factory.finish().get_fragments())
    assert fragments[0].partition_expression.equals(
        (ds.field("test%27%3B%20date") == "2021-05-04 00%3A00%3A00") &
        (ds.field("test%27%3B%5B%20string%27") == "%24"))

    options.partitioning_factory = ds.HivePartitioning.discover(
        schema=partition_schema_en, segment_encoding="none")
    factory = ds.FileSystemDatasetFactory(mockfs, selector, format, options)
    with pytest.raises(pa.ArrowInvalid,
                       match="Could not cast segments for partition field"):
        inferred_schema = factory.inspect()


def test_dictionary_partitioning_outer_nulls_raises(tempdir):
    table = pa.table({'a': ['x', 'y', None], 'b': ['x', 'y', 'z']})
    part = ds.partitioning(
        pa.schema([pa.field('a', pa.string()), pa.field('b', pa.string())]))
    with pytest.raises(pa.ArrowInvalid):
        ds.write_dataset(table, tempdir, format='ipc', partitioning=part)


def test_positional_keywords_raises(tempdir):
    table = pa.table({'a': ['x', 'y', None], 'b': ['x', 'y', 'z']})
    with pytest.raises(TypeError):
        ds.write_dataset(table, tempdir, "basename-{i}.arrow")


@pytest.mark.parquet
@pytest.mark.pandas
def test_read_partition_keys_only(tempdir):
    BATCH_SIZE = 2 ** 15
    # This is a regression test for ARROW-15318 which saw issues
    # reading only the partition keys from files with batches larger
    # than the default batch size (e.g. so we need to return two chunks)
    table = pa.table({
        'key': pa.repeat(0, BATCH_SIZE + 1),
        'value': np.arange(BATCH_SIZE + 1)})
    pq.write_to_dataset(
        table[:BATCH_SIZE],
        tempdir / 'one', partition_cols=['key'])
    pq.write_to_dataset(
        table[:BATCH_SIZE + 1],
        tempdir / 'two', partition_cols=['key'])

    table = pq.read_table(tempdir / 'one', columns=['key'])
    assert table['key'].num_chunks == 1

    table = pq.read_table(tempdir / 'two', columns=['key', 'value'])
    assert table['key'].num_chunks == 2

    table = pq.read_table(tempdir / 'two', columns=['key'])
    assert table['key'].num_chunks == 2


def _has_subdirs(basedir):
    elements = os.listdir(basedir)
    return any([os.path.isdir(os.path.join(basedir, el)) for el in elements])


def _do_list_all_dirs(basedir, path_so_far, result):
    for f in os.listdir(basedir):
        true_nested = os.path.join(basedir, f)
        if os.path.isdir(true_nested):
            norm_nested = posixpath.join(path_so_far, f)
            if _has_subdirs(true_nested):
                _do_list_all_dirs(true_nested, norm_nested, result)
            else:
                result.append(norm_nested)


def _list_all_dirs(basedir):
    result = []
    _do_list_all_dirs(basedir, '', result)
    return result


def _check_dataset_directories(tempdir, expected_directories):
    actual_directories = set(_list_all_dirs(tempdir))
    assert actual_directories == set(expected_directories)


def test_dictionary_partitioning_inner_nulls(tempdir):
    table = pa.table({'a': ['x', 'y', 'z'], 'b': ['x', 'y', None]})
    part = ds.partitioning(
        pa.schema([pa.field('a', pa.string()), pa.field('b', pa.string())]))
    ds.write_dataset(table, tempdir, format='ipc', partitioning=part)
    _check_dataset_directories(tempdir, ['x/x', 'y/y', 'z'])


def test_hive_partitioning_nulls(tempdir):
    table = pa.table({'a': ['x', None, 'z'], 'b': ['x', 'y', None]})
    part = ds.HivePartitioning(pa.schema(
        [pa.field('a', pa.string()), pa.field('b', pa.string())]), None, 'xyz')
    ds.write_dataset(table, tempdir, format='ipc', partitioning=part)
    _check_dataset_directories(tempdir, ['a=x/b=x', 'a=xyz/b=y', 'a=z/b=xyz'])


def test_partitioning_function():
    schema = pa.schema([("year", pa.int16()), ("month", pa.int8())])
    names = ["year", "month"]

    # default DirectoryPartitioning
    part = ds.partitioning(schema)
    assert isinstance(part, ds.DirectoryPartitioning)
    part = ds.partitioning(schema, dictionaries="infer")
    assert isinstance(part, ds.PartitioningFactory)
    part = ds.partitioning(field_names=names)
    assert isinstance(part, ds.PartitioningFactory)
    # needs schema or list of names
    with pytest.raises(ValueError):
        ds.partitioning()
    with pytest.raises(ValueError, match="Expected list"):
        ds.partitioning(field_names=schema)
    with pytest.raises(ValueError, match="Cannot specify both"):
        ds.partitioning(schema, field_names=schema)

    # Hive partitioning
    part = ds.partitioning(schema, flavor="hive")
    assert isinstance(part, ds.HivePartitioning)
    part = ds.partitioning(schema, dictionaries="infer", flavor="hive")
    assert isinstance(part, ds.PartitioningFactory)
    part = ds.partitioning(flavor="hive")
    assert isinstance(part, ds.PartitioningFactory)
    # cannot pass list of names
    with pytest.raises(ValueError):
        ds.partitioning(names, flavor="hive")
    with pytest.raises(ValueError, match="Cannot specify 'field_names'"):
        ds.partitioning(field_names=names, flavor="hive")

    # unsupported flavor
    with pytest.raises(ValueError):
        ds.partitioning(schema, flavor="unsupported")


@pytest.mark.parquet
def test_directory_partitioning_dictionary_key(mockfs):
    # ARROW-8088 specifying partition key as dictionary type
    schema = pa.schema([
        pa.field('group', pa.dictionary(pa.int8(), pa.int32())),
        pa.field('key', pa.dictionary(pa.int8(), pa.string()))
    ])
    part = ds.DirectoryPartitioning.discover(schema=schema)

    dataset = ds.dataset(
        "subdir", format="parquet", filesystem=mockfs, partitioning=part
    )
    assert dataset.partitioning.schema == schema
    table = dataset.to_table()

    assert table.column('group').type.equals(schema.types[0])
    assert table.column('group').to_pylist() == [1] * 5 + [2] * 5
    assert table.column('key').type.equals(schema.types[1])
    assert table.column('key').to_pylist() == ['xxx'] * 5 + ['yyy'] * 5


def test_hive_partitioning_dictionary_key(multisourcefs):
    # ARROW-8088 specifying partition key as dictionary type
    schema = pa.schema([
        pa.field('year', pa.dictionary(pa.int8(), pa.int16())),
        pa.field('month', pa.dictionary(pa.int8(), pa.int16()))
    ])
    part = ds.HivePartitioning.discover(schema=schema)

    dataset = ds.dataset(
        "hive", format="parquet", filesystem=multisourcefs, partitioning=part
    )
    assert dataset.partitioning.schema == schema
    table = dataset.to_table()

    year_dictionary = list(range(2006, 2011))
    month_dictionary = list(range(1, 13))
    assert table.column('year').type.equals(schema.types[0])
    for chunk in table.column('year').chunks:
        actual = chunk.dictionary.to_pylist()
        actual.sort()
        assert actual == year_dictionary
    assert table.column('month').type.equals(schema.types[1])
    for chunk in table.column('month').chunks:
        actual = chunk.dictionary.to_pylist()
        actual.sort()
        assert actual == month_dictionary


def _create_single_file(base_dir, table=None, row_group_size=None):
    if table is None:
        table = pa.table({'a': range(9), 'b': [0.] * 4 + [1.] * 5})
    path = base_dir / "test.parquet"
    pq.write_table(table, path, row_group_size=row_group_size)
    return table, path


def _create_directory_of_files(base_dir):
    table1 = pa.table({'a': range(9), 'b': [0.] * 4 + [1.] * 5})
    path1 = base_dir / "test1.parquet"
    pq.write_table(table1, path1)
    table2 = pa.table({'a': range(9, 18), 'b': [0.] * 4 + [1.] * 5})
    path2 = base_dir / "test2.parquet"
    pq.write_table(table2, path2)
    return (table1, table2), (path1, path2)


def _check_dataset(dataset, table, dataset_reader):
    # also test that pickle roundtrip keeps the functionality
    for d in [dataset, pickle.loads(pickle.dumps(dataset))]:
        assert dataset.schema.equals(table.schema)
        assert dataset_reader.to_table(dataset).equals(table)


def _check_dataset_from_path(path, table, dataset_reader, **kwargs):
    # pathlib object
    assert isinstance(path, pathlib.Path)

    # accept Path, str, List[Path], List[str]
    for p in [path, str(path), [path], [str(path)]]:
        dataset = ds.dataset(path, **kwargs)
        assert isinstance(dataset, ds.FileSystemDataset)
        _check_dataset(dataset, table, dataset_reader)

    # relative string path
    with change_cwd(path.parent):
        dataset = ds.dataset(path.name, **kwargs)
        assert isinstance(dataset, ds.FileSystemDataset)
        _check_dataset(dataset, table, dataset_reader)


@pytest.mark.parquet
def test_open_dataset_single_file(tempdir, dataset_reader):
    table, path = _create_single_file(tempdir)
    _check_dataset_from_path(path, table, dataset_reader)


@pytest.mark.parquet
def test_deterministic_row_order(tempdir, dataset_reader):
    # ARROW-8447 Ensure that dataset.to_table (and Scanner::ToTable) returns a
    # deterministic row ordering. This is achieved by constructing a single
    # parquet file with one row per RowGroup.
    table, path = _create_single_file(tempdir, row_group_size=1)
    _check_dataset_from_path(path, table, dataset_reader)


@pytest.mark.parquet
def test_open_dataset_directory(tempdir, dataset_reader):
    tables, _ = _create_directory_of_files(tempdir)
    table = pa.concat_tables(tables)
    _check_dataset_from_path(tempdir, table, dataset_reader)


@pytest.mark.parquet
def test_open_dataset_list_of_files(tempdir, dataset_reader):
    tables, (path1, path2) = _create_directory_of_files(tempdir)
    table = pa.concat_tables(tables)

    datasets = [
        ds.dataset([path1, path2]),
        ds.dataset([str(path1), str(path2)])
    ]
    datasets += [
        pickle.loads(pickle.dumps(d)) for d in datasets
    ]

    for dataset in datasets:
        assert dataset.schema.equals(table.schema)
        result = dataset_reader.to_table(dataset)
        assert result.equals(table)


@pytest.mark.parquet
def test_open_dataset_filesystem_fspath(tempdir):
    # single file
    table, path = _create_single_file(tempdir)

    fspath = FSProtocolClass(path)

    # filesystem inferred from path
    dataset1 = ds.dataset(fspath)
    assert dataset1.schema.equals(table.schema)

    # filesystem specified
    dataset2 = ds.dataset(fspath, filesystem=fs.LocalFileSystem())
    assert dataset2.schema.equals(table.schema)

    # passing different filesystem
    with pytest.raises(TypeError):
        ds.dataset(fspath, filesystem=fs._MockFileSystem())


@pytest.mark.parquet
def test_construct_from_single_file(tempdir, dataset_reader):
    directory = tempdir / 'single-file'
    directory.mkdir()
    table, path = _create_single_file(directory)
    relative_path = path.relative_to(directory)

    # instantiate from a single file
    d1 = ds.dataset(path)
    # instantiate from a single file with a filesystem object
    d2 = ds.dataset(path, filesystem=fs.LocalFileSystem())
    # instantiate from a single file with prefixed filesystem URI
    d3 = ds.dataset(str(relative_path), filesystem=_filesystem_uri(directory))
    # pickle roundtrip
    d4 = pickle.loads(pickle.dumps(d1))

    assert dataset_reader.to_table(d1) == dataset_reader.to_table(
        d2) == dataset_reader.to_table(d3) == dataset_reader.to_table(d4)


@pytest.mark.parquet
def test_construct_from_single_directory(tempdir, dataset_reader):
    directory = tempdir / 'single-directory'
    directory.mkdir()
    tables, paths = _create_directory_of_files(directory)

    d1 = ds.dataset(directory)
    d2 = ds.dataset(directory, filesystem=fs.LocalFileSystem())
    d3 = ds.dataset(directory.name, filesystem=_filesystem_uri(tempdir))
    t1 = dataset_reader.to_table(d1)
    t2 = dataset_reader.to_table(d2)
    t3 = dataset_reader.to_table(d3)
    assert t1 == t2 == t3

    # test pickle roundtrip
    for d in [d1, d2, d3]:
        restored = pickle.loads(pickle.dumps(d))
        assert dataset_reader.to_table(restored) == t1


@pytest.mark.parquet
def test_construct_from_list_of_files(tempdir, dataset_reader):
    # instantiate from a list of files
    directory = tempdir / 'list-of-files'
    directory.mkdir()
    tables, paths = _create_directory_of_files(directory)

    relative_paths = [p.relative_to(tempdir) for p in paths]
    with change_cwd(tempdir):
        d1 = ds.dataset(relative_paths)
        t1 = dataset_reader.to_table(d1)
        assert len(t1) == sum(map(len, tables))

    d2 = ds.dataset(relative_paths, filesystem=_filesystem_uri(tempdir))
    t2 = dataset_reader.to_table(d2)
    d3 = ds.dataset(paths)
    t3 = dataset_reader.to_table(d3)
    d4 = ds.dataset(paths, filesystem=fs.LocalFileSystem())
    t4 = dataset_reader.to_table(d4)

    assert t1 == t2 == t3 == t4


@pytest.mark.parquet
def test_construct_from_list_of_mixed_paths_fails(mockfs):
    # isntantiate from a list of mixed paths
    files = [
        'subdir/1/xxx/file0.parquet',
        'subdir/1/xxx/doesnt-exist.parquet',
    ]
    with pytest.raises(FileNotFoundError, match='doesnt-exist'):
        ds.dataset(files, filesystem=mockfs)


@pytest.mark.parquet
def test_construct_from_mixed_child_datasets(mockfs):
    # isntantiate from a list of mixed paths
    a = ds.dataset(['subdir/1/xxx/file0.parquet',
                    'subdir/2/yyy/file1.parquet'], filesystem=mockfs)
    b = ds.dataset('subdir', filesystem=mockfs)

    dataset = ds.dataset([a, b])

    assert isinstance(dataset, ds.UnionDataset)
    assert len(list(dataset.get_fragments())) == 4

    table = dataset.to_table()
    assert len(table) == 20
    assert table.num_columns == 5

    assert len(dataset.children) == 2
    for child in dataset.children:
        assert child.files == ['subdir/1/xxx/file0.parquet',
                               'subdir/2/yyy/file1.parquet']


def test_construct_empty_dataset():
    empty = ds.dataset([], format='ipc')
    table = empty.to_table()
    assert table.num_rows == 0
    assert table.num_columns == 0


def test_construct_dataset_with_invalid_schema():
    empty = ds.dataset([], format='ipc', schema=pa.schema([
        ('a', pa.int64()),
        ('a', pa.string())
    ]))
    with pytest.raises(ValueError, match='Multiple matches for .*a.* in '):
        empty.to_table()


def test_construct_from_invalid_sources_raise(multisourcefs):
    child1 = ds.FileSystemDatasetFactory(
        multisourcefs,
        fs.FileSelector('/plain'),
        format=ds.ParquetFileFormat()
    )
    child2 = ds.FileSystemDatasetFactory(
        multisourcefs,
        fs.FileSelector('/schema'),
        format=ds.ParquetFileFormat()
    )
    batch1 = pa.RecordBatch.from_arrays([pa.array(range(10))], names=["a"])
    batch2 = pa.RecordBatch.from_arrays([pa.array(range(10))], names=["b"])

    with pytest.raises(TypeError, match='Expected.*FileSystemDatasetFactory'):
        ds.dataset([child1, child2])

    expected = (
        "Expected a list of path-like or dataset objects, or a list "
        "of batches or tables. The given list contains the following "
        "types: int"
    )
    with pytest.raises(TypeError, match=expected):
        ds.dataset([1, 2, 3])

    expected = (
        "Expected a path-like, list of path-likes or a list of Datasets "
        "instead of the given type: NoneType"
    )
    with pytest.raises(TypeError, match=expected):
        ds.dataset(None)

    expected = (
        "Expected a path-like, list of path-likes or a list of Datasets "
        "instead of the given type: generator"
    )
    with pytest.raises(TypeError, match=expected):
        ds.dataset((batch1 for _ in range(3)))

    expected = (
        "Must provide schema to construct in-memory dataset from an empty list"
    )
    with pytest.raises(ValueError, match=expected):
        ds.InMemoryDataset([])

    expected = (
        "Item has schema\nb: int64\nwhich does not match expected schema\n"
        "a: int64"
    )
    with pytest.raises(TypeError, match=expected):
        ds.dataset([batch1, batch2])

    expected = (
        "Expected a list of path-like or dataset objects, or a list of "
        "batches or tables. The given list contains the following types:"
    )
    with pytest.raises(TypeError, match=expected):
        ds.dataset([batch1, 0])

    expected = (
        "Expected a list of tables or batches. The given list contains a int"
    )
    with pytest.raises(TypeError, match=expected):
        ds.InMemoryDataset([batch1, 0])


def test_construct_in_memory(dataset_reader):
    batch = pa.RecordBatch.from_arrays([pa.array(range(10))], names=["a"])
    table = pa.Table.from_batches([batch])

    dataset_table = ds.dataset([], format='ipc', schema=pa.schema([])
                               ).to_table()
    assert dataset_table == pa.table([])

    for source in (batch, table, [batch], [table]):
        dataset = ds.dataset(source)
        assert dataset_reader.to_table(dataset) == table
        assert len(list(dataset.get_fragments())) == 1
        assert next(dataset.get_fragments()).to_table() == table
        assert pa.Table.from_batches(list(dataset.to_batches())) == table


@pytest.mark.parametrize('use_threads', [False, True])
def test_scan_iterator(use_threads):
    batch = pa.RecordBatch.from_arrays([pa.array(range(10))], names=["a"])
    table = pa.Table.from_batches([batch])
    # When constructed from readers/iterators, should be one-shot
    match = "OneShotFragment was already scanned"
    for factory, schema in (
            (lambda: pa.RecordBatchReader.from_batches(
                batch.schema, [batch]), None),
            (lambda: (batch for _ in range(1)), batch.schema),
    ):
        # Scanning the fragment consumes the underlying iterator
        scanner = ds.Scanner.from_batches(
            factory(), schema=schema, use_threads=use_threads)
        assert scanner.to_table() == table
        with pytest.raises(pa.ArrowInvalid, match=match):
            scanner.to_table()


def _create_partitioned_dataset(basedir):
    table = pa.table({'a': range(9), 'b': [0.] * 4 + [1.] * 5})

    path = basedir / "dataset-partitioned"
    path.mkdir()

    for i in range(3):
        part = path / "part={}".format(i)
        part.mkdir()
        pq.write_table(table.slice(3*i, 3), part / "test.parquet")

    full_table = table.append_column(
        "part", pa.array(np.repeat([0, 1, 2], 3), type=pa.int32()))

    return full_table, path


@pytest.mark.parquet
def test_open_dataset_partitioned_directory(tempdir, dataset_reader):
    full_table, path = _create_partitioned_dataset(tempdir)

    # no partitioning specified, just read all individual files
    table = full_table.select(['a', 'b'])
    _check_dataset_from_path(path, table, dataset_reader)

    # specify partition scheme with discovery
    dataset = ds.dataset(
        str(path), partitioning=ds.partitioning(flavor="hive"))
    assert dataset.schema.equals(full_table.schema)

    # specify partition scheme with discovery and relative path
    with change_cwd(tempdir):
        dataset = ds.dataset("dataset-partitioned/",
                             partitioning=ds.partitioning(flavor="hive"))
        assert dataset.schema.equals(full_table.schema)

    # specify partition scheme with string short-cut
    dataset = ds.dataset(str(path), partitioning="hive")
    assert dataset.schema.equals(full_table.schema)

    # specify partition scheme with explicit scheme
    dataset = ds.dataset(
        str(path),
        partitioning=ds.partitioning(
            pa.schema([("part", pa.int8())]), flavor="hive"))
    expected_schema = table.schema.append(pa.field("part", pa.int8()))
    assert dataset.schema.equals(expected_schema)

    result = dataset.to_table()
    expected = table.append_column(
        "part", pa.array(np.repeat([0, 1, 2], 3), type=pa.int8()))
    assert result.equals(expected)


@pytest.mark.parquet
def test_open_dataset_filesystem(tempdir):
    # single file
    table, path = _create_single_file(tempdir)

    # filesystem inferred from path
    dataset1 = ds.dataset(str(path))
    assert dataset1.schema.equals(table.schema)

    # filesystem specified
    dataset2 = ds.dataset(str(path), filesystem=fs.LocalFileSystem())
    assert dataset2.schema.equals(table.schema)

    # local filesystem specified with relative path
    with change_cwd(tempdir):
        dataset3 = ds.dataset("test.parquet", filesystem=fs.LocalFileSystem())
    assert dataset3.schema.equals(table.schema)

    # passing different filesystem
    with pytest.raises(FileNotFoundError):
        ds.dataset(str(path), filesystem=fs._MockFileSystem())


@pytest.mark.parquet
def test_open_dataset_unsupported_format(tempdir):
    _, path = _create_single_file(tempdir)
    with pytest.raises(ValueError, match="format 'blabla' is not supported"):
        ds.dataset([path], format="blabla")


@pytest.mark.parquet
def test_open_union_dataset(tempdir, dataset_reader):
    _, path = _create_single_file(tempdir)
    dataset = ds.dataset(path)

    union = ds.dataset([dataset, dataset])
    assert isinstance(union, ds.UnionDataset)

    pickled = pickle.loads(pickle.dumps(union))
    assert dataset_reader.to_table(pickled) == dataset_reader.to_table(union)


def test_open_union_dataset_with_additional_kwargs(multisourcefs):
    child = ds.dataset('/plain', filesystem=multisourcefs, format='parquet')
    with pytest.raises(ValueError, match="cannot pass any additional"):
        ds.dataset([child], format="parquet")


def test_open_dataset_non_existing_file():
    # ARROW-8213: Opening a dataset with a local incorrect path gives confusing
    #             error message
    with pytest.raises(FileNotFoundError):
        ds.dataset('i-am-not-existing.arrow', format='ipc')

    with pytest.raises(pa.ArrowInvalid, match='cannot be relative'):
        ds.dataset('file:i-am-not-existing.arrow', format='ipc')


@pytest.mark.parquet
@pytest.mark.parametrize('partitioning', ["directory", "hive"])
@pytest.mark.parametrize('null_fallback', ['xyz', None])
@pytest.mark.parametrize('infer_dictionary', [False, True])
@pytest.mark.parametrize('partition_keys', [
    (["A", "B", "C"], [1, 2, 3]),
    ([1, 2, 3], ["A", "B", "C"]),
    (["A", "B", "C"], ["D", "E", "F"]),
    ([1, 2, 3], [4, 5, 6]),
    ([1, None, 3], ["A", "B", "C"]),
    ([1, 2, 3], ["A", None, "C"]),
    ([None, 2, 3], [None, 2, 3]),
])
def test_partition_discovery(
    tempdir, partitioning, null_fallback, infer_dictionary, partition_keys
):
    # ARROW-9288 / ARROW-9476
    table = pa.table({'a': range(9), 'b': [0.0] * 4 + [1.0] * 5})

    has_null = None in partition_keys[0] or None in partition_keys[1]
    if partitioning == "directory" and has_null:
        # Directory partitioning can't handle the first part being null
        return

    if partitioning == "directory":
        partitioning = ds.DirectoryPartitioning.discover(
            ["part1", "part2"], infer_dictionary=infer_dictionary)
        fmt = "{0}/{1}"
        null_value = None
    else:
        if null_fallback:
            partitioning = ds.HivePartitioning.discover(
                infer_dictionary=infer_dictionary, null_fallback=null_fallback
            )
        else:
            partitioning = ds.HivePartitioning.discover(
                infer_dictionary=infer_dictionary)
        fmt = "part1={0}/part2={1}"
        if null_fallback:
            null_value = null_fallback
        else:
            null_value = "__HIVE_DEFAULT_PARTITION__"

    basepath = tempdir / "dataset"
    basepath.mkdir()

    part_keys1, part_keys2 = partition_keys
    for part1 in part_keys1:
        for part2 in part_keys2:
            path = basepath / \
                fmt.format(part1 or null_value, part2 or null_value)
            path.mkdir(parents=True)
            pq.write_table(table, path / "test.parquet")

    dataset = ds.dataset(str(basepath), partitioning=partitioning)

    def expected_type(key):
        if infer_dictionary:
            value_type = pa.string() if isinstance(key, str) else pa.int32()
            return pa.dictionary(pa.int32(), value_type)
        else:
            return pa.string() if isinstance(key, str) else pa.int32()
    expected_schema = table.schema.append(
        pa.field("part1", expected_type(part_keys1[0]))
    ).append(
        pa.field("part2", expected_type(part_keys2[0]))
    )
    assert dataset.schema.equals(expected_schema)


@pytest.mark.pandas
def test_dataset_partitioned_dictionary_type_reconstruct(tempdir):
    # https://issues.apache.org/jira/browse/ARROW-11400
    table = pa.table({'part': np.repeat(['A', 'B'], 5), 'col': range(10)})
    part = ds.partitioning(table.select(['part']).schema, flavor="hive")
    ds.write_dataset(table, tempdir, partitioning=part, format="feather")

    dataset = ds.dataset(
        tempdir, format="feather",
        partitioning=ds.HivePartitioning.discover(infer_dictionary=True)
    )
    expected = pa.table(
        {'col': table['col'], 'part': table['part'].dictionary_encode()}
    )
    assert dataset.to_table().equals(expected)
    fragment = list(dataset.get_fragments())[0]
    assert fragment.to_table(schema=dataset.schema).equals(expected[:5])
    part_expr = fragment.partition_expression

    restored = pickle.loads(pickle.dumps(dataset))
    assert restored.to_table().equals(expected)

    restored = pickle.loads(pickle.dumps(fragment))
    assert restored.to_table(schema=dataset.schema).equals(expected[:5])
    # to_pandas call triggers computation of the actual dictionary values
    assert restored.to_table(schema=dataset.schema).to_pandas().equals(
        expected[:5].to_pandas()
    )
    assert restored.partition_expression.equals(part_expr)


@pytest.fixture
@pytest.mark.parquet
def s3_example_simple(s3_server):
    from pyarrow.fs import FileSystem

    host, port, access_key, secret_key = s3_server['connection']
    uri = (
        "s3://{}:{}@mybucket/data.parquet?scheme=http&endpoint_override={}:{}"
        "&allow_bucket_creation=True"
        .format(access_key, secret_key, host, port)
    )

    fs, path = FileSystem.from_uri(uri)

    fs.create_dir("mybucket")
    table = pa.table({'a': [1, 2, 3]})
    with fs.open_output_stream("mybucket/data.parquet") as out:
        pq.write_table(table, out)

    return table, path, fs, uri, host, port, access_key, secret_key


@pytest.mark.parquet
@pytest.mark.s3
def test_open_dataset_from_uri_s3(s3_example_simple, dataset_reader):
    # open dataset from non-localfs string path
    table, path, fs, uri, _, _, _, _ = s3_example_simple

    # full string URI
    dataset = ds.dataset(uri, format="parquet")
    assert dataset_reader.to_table(dataset).equals(table)

    # passing filesystem object
    dataset = ds.dataset(path, format="parquet", filesystem=fs)
    assert dataset_reader.to_table(dataset).equals(table)


@pytest.mark.parquet
@pytest.mark.s3  # still needed to create the data
def test_open_dataset_from_uri_s3_fsspec(s3_example_simple):
    table, path, _, _, host, port, access_key, secret_key = s3_example_simple
    s3fs = pytest.importorskip("s3fs")

    from pyarrow.fs import PyFileSystem, FSSpecHandler

    fs = s3fs.S3FileSystem(
        key=access_key,
        secret=secret_key,
        client_kwargs={
            'endpoint_url': 'http://{}:{}'.format(host, port)
        }
    )

    # passing as fsspec filesystem
    dataset = ds.dataset(path, format="parquet", filesystem=fs)
    assert dataset.to_table().equals(table)

    # directly passing the fsspec-handler
    fs = PyFileSystem(FSSpecHandler(fs))
    dataset = ds.dataset(path, format="parquet", filesystem=fs)
    assert dataset.to_table().equals(table)


@pytest.mark.parquet
@pytest.mark.s3
def test_open_dataset_from_s3_with_filesystem_uri(s3_server):
    from pyarrow.fs import FileSystem

    host, port, access_key, secret_key = s3_server['connection']
    bucket = 'theirbucket'
    path = 'nested/folder/data.parquet'
    uri = "s3://{}:{}@{}/{}?scheme=http&endpoint_override={}:{}"\
        "&allow_bucket_creation=true".format(
            access_key, secret_key, bucket, path, host, port
        )

    fs, path = FileSystem.from_uri(uri)
    assert path == 'theirbucket/nested/folder/data.parquet'

    fs.create_dir(bucket)

    table = pa.table({'a': [1, 2, 3]})
    with fs.open_output_stream(path) as out:
        pq.write_table(table, out)

    # full string URI
    dataset = ds.dataset(uri, format="parquet")
    assert dataset.to_table().equals(table)

    # passing filesystem as an uri
    template = (
        "s3://{}:{}@{{}}?scheme=http&endpoint_override={}:{}".format(
            access_key, secret_key, host, port
        )
    )
    cases = [
        ('theirbucket/nested/folder/', '/data.parquet'),
        ('theirbucket/nested/folder', 'data.parquet'),
        ('theirbucket/nested/', 'folder/data.parquet'),
        ('theirbucket/nested', 'folder/data.parquet'),
        ('theirbucket', '/nested/folder/data.parquet'),
        ('theirbucket', 'nested/folder/data.parquet'),
    ]
    for prefix, path in cases:
        uri = template.format(prefix)
        dataset = ds.dataset(path, filesystem=uri, format="parquet")
        assert dataset.to_table().equals(table)

    with pytest.raises(pa.ArrowInvalid, match='Missing bucket name'):
        uri = template.format('/')
        ds.dataset('/theirbucket/nested/folder/data.parquet', filesystem=uri)

    error = (
        "The path component of the filesystem URI must point to a directory "
        "but it has a type: `{}`. The path component is `{}` and the given "
        "filesystem URI is `{}`"
    )

    path = 'theirbucket/doesnt/exist'
    uri = template.format(path)
    with pytest.raises(ValueError) as exc:
        ds.dataset('data.parquet', filesystem=uri)
    assert str(exc.value) == error.format('NotFound', path, uri)

    path = 'theirbucket/nested/folder/data.parquet'
    uri = template.format(path)
    with pytest.raises(ValueError) as exc:
        ds.dataset('data.parquet', filesystem=uri)
    assert str(exc.value) == error.format('File', path, uri)


@pytest.mark.parquet
def test_open_dataset_from_fsspec(tempdir):
    table, path = _create_single_file(tempdir)

    fsspec = pytest.importorskip("fsspec")

    localfs = fsspec.filesystem("file")
    dataset = ds.dataset(path, filesystem=localfs)
    assert dataset.schema.equals(table.schema)


@pytest.mark.parquet
def test_file_format_inspect_fsspec(tempdir):
    # https://issues.apache.org/jira/browse/ARROW-16413
    fsspec = pytest.importorskip("fsspec")

    # create bucket + file with pyarrow
    table = pa.table({'a': [1, 2, 3]})
    path = tempdir / "data.parquet"
    pq.write_table(table, path)

    # read using fsspec filesystem
    fsspec_fs = fsspec.filesystem("file")
    assert fsspec_fs.ls(tempdir)[0].endswith("data.parquet")

    # inspect using dataset file format
    format = ds.ParquetFileFormat()
    # manually creating a PyFileSystem instead of using fs._ensure_filesystem
    # which would convert an fsspec local filesystem to a native one
    filesystem = fs.PyFileSystem(fs.FSSpecHandler(fsspec_fs))
    schema = format.inspect(path, filesystem)
    assert schema.equals(table.schema)

    fragment = format.make_fragment(path, filesystem)
    assert fragment.physical_schema.equals(table.schema)


@pytest.mark.pandas
def test_filter_timestamp(tempdir, dataset_reader):
    # ARROW-11379
    path = tempdir / "test_partition_timestamps"

    table = pa.table({
        "dates": ['2012-01-01', '2012-01-02'] * 5,
        "id": range(10)})

    # write dataset partitioned on dates (as strings)
    part = ds.partitioning(table.select(['dates']).schema, flavor="hive")
    ds.write_dataset(table, path, partitioning=part, format="feather")

    # read dataset partitioned on dates (as timestamps)
    part = ds.partitioning(pa.schema([("dates", pa.timestamp("s"))]),
                           flavor="hive")
    dataset = ds.dataset(path, format="feather", partitioning=part)

    condition = ds.field("dates") > pd.Timestamp("2012-01-01")
    table = dataset_reader.to_table(dataset, filter=condition)
    assert table.column('id').to_pylist() == [1, 3, 5, 7, 9]

    import datetime
    condition = ds.field("dates") > datetime.datetime(2012, 1, 1)
    table = dataset_reader.to_table(dataset, filter=condition)
    assert table.column('id').to_pylist() == [1, 3, 5, 7, 9]


@pytest.mark.parquet
def test_filter_implicit_cast(tempdir, dataset_reader):
    # ARROW-7652
    table = pa.table({'a': pa.array([0, 1, 2, 3, 4, 5], type=pa.int8())})
    _, path = _create_single_file(tempdir, table)
    dataset = ds.dataset(str(path))

    filter_ = ds.field('a') > 2
    assert len(dataset_reader.to_table(dataset, filter=filter_)) == 3


@pytest.mark.parquet
def test_filter_equal_null(tempdir, dataset_reader):
    # ARROW-12066 equality with null, although not useful, should not crash
    table = pa.table({"A": ["a", "b", None]})
    _, path = _create_single_file(tempdir, table)
    dataset = ds.dataset(str(path))

    table = dataset_reader.to_table(
        dataset, filter=ds.field("A") == ds.scalar(None)
    )
    assert table.num_rows == 0


@pytest.mark.parquet
def test_filter_compute_expression(tempdir, dataset_reader):
    table = pa.table({
        "A": ["a", "b", None, "a", "c"],
        "B": [datetime.datetime(2022, 1, 1, i) for i in range(5)],
        "C": [datetime.datetime(2022, 1, i) for i in range(1, 6)],
    })
    _, path = _create_single_file(tempdir, table)
    dataset = ds.dataset(str(path))

    filter_ = pc.is_in(ds.field('A'), pa.array(["a", "b"]))
    assert dataset_reader.to_table(dataset, filter=filter_).num_rows == 3

    filter_ = pc.hour(ds.field('B')) >= 3
    assert dataset_reader.to_table(dataset, filter=filter_).num_rows == 2

    days = pc.days_between(ds.field('B'), ds.field("C"))
    result = dataset_reader.to_table(dataset, columns={"days": days})
    assert result["days"].to_pylist() == [0, 1, 2, 3, 4]


def test_dataset_union(multisourcefs):
    child = ds.FileSystemDatasetFactory(
        multisourcefs, fs.FileSelector('/plain'),
        format=ds.ParquetFileFormat()
    )
    factory = ds.UnionDatasetFactory([child])

    # TODO(bkietz) reintroduce factory.children property
    assert len(factory.inspect_schemas()) == 1
    assert all(isinstance(s, pa.Schema) for s in factory.inspect_schemas())
    assert factory.inspect_schemas()[0].equals(child.inspect())
    assert factory.inspect().equals(child.inspect())
    assert isinstance(factory.finish(), ds.Dataset)


def test_union_dataset_from_other_datasets(tempdir, multisourcefs):
    child1 = ds.dataset('/plain', filesystem=multisourcefs, format='parquet')
    child2 = ds.dataset('/schema', filesystem=multisourcefs, format='parquet',
                        partitioning=['week', 'color'])
    child3 = ds.dataset('/hive', filesystem=multisourcefs, format='parquet',
                        partitioning='hive')

    assert child1.schema != child2.schema != child3.schema

    assembled = ds.dataset([child1, child2, child3])
    assert isinstance(assembled, ds.UnionDataset)

    msg = 'cannot pass any additional arguments'
    with pytest.raises(ValueError, match=msg):
        ds.dataset([child1, child2], filesystem=multisourcefs)

    expected_schema = pa.schema([
        ('date', pa.date32()),
        ('index', pa.int64()),
        ('value', pa.float64()),
        ('color', pa.string()),
        ('week', pa.int32()),
        ('year', pa.int32()),
        ('month', pa.int32()),
    ])
    assert assembled.schema.equals(expected_schema)
    assert assembled.to_table().schema.equals(expected_schema)

    assembled = ds.dataset([child1, child3])
    expected_schema = pa.schema([
        ('date', pa.date32()),
        ('index', pa.int64()),
        ('value', pa.float64()),
        ('color', pa.string()),
        ('year', pa.int32()),
        ('month', pa.int32()),
    ])
    assert assembled.schema.equals(expected_schema)
    assert assembled.to_table().schema.equals(expected_schema)

    expected_schema = pa.schema([
        ('month', pa.int32()),
        ('color', pa.string()),
        ('date', pa.date32()),
    ])
    assembled = ds.dataset([child1, child3], schema=expected_schema)
    assert assembled.to_table().schema.equals(expected_schema)

    expected_schema = pa.schema([
        ('month', pa.int32()),
        ('color', pa.string()),
        ('unknown', pa.string())  # fill with nulls
    ])
    assembled = ds.dataset([child1, child3], schema=expected_schema)
    assert assembled.to_table().schema.equals(expected_schema)

    # incompatible schemas, date and index columns have conflicting types
    table = pa.table([range(9), [0.] * 4 + [1.] * 5, 'abcdefghj'],
                     names=['date', 'value', 'index'])
    _, path = _create_single_file(tempdir, table=table)
    child4 = ds.dataset(path)

    with pytest.raises(pa.ArrowInvalid, match='Unable to merge'):
        ds.dataset([child1, child4])


def test_dataset_from_a_list_of_local_directories_raises(multisourcefs):
    msg = 'points to a directory, but only file paths are supported'
    with pytest.raises(IsADirectoryError, match=msg):
        ds.dataset(['/plain', '/schema', '/hive'], filesystem=multisourcefs)


def test_union_dataset_filesystem_datasets(multisourcefs):
    # without partitioning
    dataset = ds.dataset([
        ds.dataset('/plain', filesystem=multisourcefs),
        ds.dataset('/schema', filesystem=multisourcefs),
        ds.dataset('/hive', filesystem=multisourcefs),
    ])
    expected_schema = pa.schema([
        ('date', pa.date32()),
        ('index', pa.int64()),
        ('value', pa.float64()),
        ('color', pa.string()),
    ])
    assert dataset.schema.equals(expected_schema)

    # with hive partitioning for two hive sources
    dataset = ds.dataset([
        ds.dataset('/plain', filesystem=multisourcefs),
        ds.dataset('/schema', filesystem=multisourcefs),
        ds.dataset('/hive', filesystem=multisourcefs, partitioning='hive')
    ])
    expected_schema = pa.schema([
        ('date', pa.date32()),
        ('index', pa.int64()),
        ('value', pa.float64()),
        ('color', pa.string()),
        ('year', pa.int32()),
        ('month', pa.int32()),
    ])
    assert dataset.schema.equals(expected_schema)


@pytest.mark.parquet
def test_specified_schema(tempdir, dataset_reader):
    table = pa.table({'a': [1, 2, 3], 'b': [.1, .2, .3]})
    pq.write_table(table, tempdir / "data.parquet")

    def _check_dataset(schema, expected, expected_schema=None):
        dataset = ds.dataset(str(tempdir / "data.parquet"), schema=schema)
        if expected_schema is not None:
            assert dataset.schema.equals(expected_schema)
        else:
            assert dataset.schema.equals(schema)
        result = dataset_reader.to_table(dataset)
        assert result.equals(expected)

    # no schema specified
    schema = None
    expected = table
    _check_dataset(schema, expected, expected_schema=table.schema)

    # identical schema specified
    schema = table.schema
    expected = table
    _check_dataset(schema, expected)

    # Specifying schema with change column order
    schema = pa.schema([('b', 'float64'), ('a', 'int64')])
    expected = pa.table([[.1, .2, .3], [1, 2, 3]], names=['b', 'a'])
    _check_dataset(schema, expected)

    # Specifying schema with missing column
    schema = pa.schema([('a', 'int64')])
    expected = pa.table([[1, 2, 3]], names=['a'])
    _check_dataset(schema, expected)

    # Specifying schema with additional column
    schema = pa.schema([('a', 'int64'), ('c', 'int32')])
    expected = pa.table([[1, 2, 3],
                         pa.array([None, None, None], type='int32')],
                        names=['a', 'c'])
    _check_dataset(schema, expected)

    # Specifying with differing field types
    schema = pa.schema([('a', 'int32'), ('b', 'float64')])
    dataset = ds.dataset(str(tempdir / "data.parquet"), schema=schema)
    expected = pa.table([table['a'].cast('int32'),
                         table['b']],
                        names=['a', 'b'])
    _check_dataset(schema, expected)

    # Specifying with incompatible schema
    schema = pa.schema([('a', pa.list_(pa.int32())), ('b', 'float64')])
    dataset = ds.dataset(str(tempdir / "data.parquet"), schema=schema)
    assert dataset.schema.equals(schema)
    with pytest.raises(NotImplementedError,
                       match='Unsupported cast from int64 to list'):
        dataset_reader.to_table(dataset)


@pytest.mark.parquet
def test_incompatible_schema_hang(tempdir, dataset_reader):
    # ARROW-13480: deadlock when reading past an errored fragment

    fn = tempdir / "data.parquet"
    table = pa.table({'a': [1, 2, 3]})
    pq.write_table(table, fn)

    schema = pa.schema([('a', pa.null())])
    dataset = ds.dataset([str(fn)] * 100, schema=schema)
    assert dataset.schema.equals(schema)
    scanner = dataset_reader.scanner(dataset)
    with pytest.raises(NotImplementedError,
                       match='Unsupported cast from int64 to null'):
        reader = scanner.to_reader()
        reader.read_all()


def test_ipc_format(tempdir, dataset_reader):
    table = pa.table({'a': pa.array([1, 2, 3], type="int8"),
                      'b': pa.array([.1, .2, .3], type="float64")})

    path = str(tempdir / 'test.arrow')
    with pa.output_stream(path) as sink:
        writer = pa.RecordBatchFileWriter(sink, table.schema)
        writer.write_batch(table.to_batches()[0])
        writer.close()

    dataset = ds.dataset(path, format=ds.IpcFileFormat())
    result = dataset_reader.to_table(dataset)
    assert result.equals(table)

    assert_dataset_fragment_convenience_methods(dataset)

    for format_str in ["ipc", "arrow"]:
        dataset = ds.dataset(path, format=format_str)
        result = dataset_reader.to_table(dataset)
        assert result.equals(table)


@pytest.mark.orc
def test_orc_format(tempdir, dataset_reader):
    from pyarrow import orc
    table = pa.table({'a': pa.array([1, 2, 3], type="int8"),
                      'b': pa.array([.1, .2, .3], type="float64")})

    path = str(tempdir / 'test.orc')
    orc.write_table(table, path)

    dataset = ds.dataset(path, format=ds.OrcFileFormat())
    fragments = list(dataset.get_fragments())
    assert isinstance(fragments[0], ds.FileFragment)
    result = dataset_reader.to_table(dataset)
    result.validate(full=True)
    assert result.equals(table)

    assert_dataset_fragment_convenience_methods(dataset)

    dataset = ds.dataset(path, format="orc")
    result = dataset_reader.to_table(dataset)
    result.validate(full=True)
    assert result.equals(table)

    result = dataset_reader.to_table(dataset, columns=["b"])
    result.validate(full=True)
    assert result.equals(table.select(["b"]))

    result = dataset_reader.to_table(
        dataset, columns={"b2": ds.field("b") * 2}
    )
    result.validate(full=True)
    assert result.equals(
        pa.table({'b2': pa.array([.2, .4, .6], type="float64")})
    )

    assert dataset_reader.count_rows(dataset) == 3
    assert dataset_reader.count_rows(dataset, filter=ds.field("a") > 2) == 1


@pytest.mark.orc
def test_orc_scan_options(tempdir, dataset_reader):
    from pyarrow import orc
    table = pa.table({'a': pa.array([1, 2, 3], type="int8"),
                      'b': pa.array([.1, .2, .3], type="float64")})

    path = str(tempdir / 'test.orc')
    orc.write_table(table, path)

    dataset = ds.dataset(path, format="orc")
    result = list(dataset_reader.to_batches(dataset))
    assert len(result) == 1
    assert result[0].num_rows == 3
    assert result[0].equals(table.to_batches()[0])
    # TODO batch_size is not yet supported (ARROW-14153)
    # result = list(dataset_reader.to_batches(dataset, batch_size=2))
    # assert len(result) == 2
    # assert result[0].num_rows == 2
    # assert result[0].equals(table.slice(0, 2).to_batches()[0])
    # assert result[1].num_rows == 1
    # assert result[1].equals(table.slice(2, 1).to_batches()[0])


def test_orc_format_not_supported():
    try:
        from pyarrow.dataset import OrcFileFormat  # noqa
    except ImportError:
        # ORC is not available, test error message
        with pytest.raises(
            ValueError, match="not built with support for the ORC file"
        ):
            ds.dataset(".", format="orc")


@pytest.mark.orc
def test_orc_writer_not_implemented_for_dataset():
    with pytest.raises(
        NotImplementedError,
        match="Writing datasets not yet implemented for this file format"
    ):
        ds.write_dataset(
            pa.table({"a": range(10)}), format='orc', base_dir='/tmp'
        )

    of = ds.OrcFileFormat()
    with pytest.raises(
        NotImplementedError,
        match="Writing datasets not yet implemented for this file format"
    ):
        of.make_write_options()


@pytest.mark.pandas
def test_csv_format(tempdir, dataset_reader):
    table = pa.table({'a': pa.array([1, 2, 3], type="int64"),
                      'b': pa.array([.1, .2, .3], type="float64")})

    path = str(tempdir / 'test.csv')
    table.to_pandas().to_csv(path, index=False)

    dataset = ds.dataset(path, format=ds.CsvFileFormat())
    result = dataset_reader.to_table(dataset)
    assert result.equals(table)

    assert_dataset_fragment_convenience_methods(dataset)

    dataset = ds.dataset(path, format='csv')
    result = dataset_reader.to_table(dataset)
    assert result.equals(table)


@pytest.mark.pandas
@pytest.mark.parametrize("compression", [
    "bz2",
    "gzip",
    "lz4",
    "zstd",
])
def test_csv_format_compressed(tempdir, compression, dataset_reader):
    if not pyarrow.Codec.is_available(compression):
        pytest.skip("{} support is not built".format(compression))
    table = pa.table({'a': pa.array([1, 2, 3], type="int64"),
                      'b': pa.array([.1, .2, .3], type="float64")})
    filesystem = fs.LocalFileSystem()
    suffix = compression if compression != 'gzip' else 'gz'
    path = str(tempdir / f'test.csv.{suffix}')
    with filesystem.open_output_stream(path, compression=compression) as sink:
        # https://github.com/pandas-dev/pandas/issues/23854
        # With CI version of Pandas (anything < 1.2), Pandas tries to write
        # str to the sink
        csv_str = table.to_pandas().to_csv(index=False)
        sink.write(csv_str.encode('utf-8'))

    dataset = ds.dataset(path, format=ds.CsvFileFormat())
    result = dataset_reader.to_table(dataset)
    assert result.equals(table)


def test_csv_format_options(tempdir, dataset_reader):
    path = str(tempdir / 'test.csv')
    with open(path, 'w') as sink:
        sink.write('skipped\ncol0\nfoo\nbar\n')
    dataset = ds.dataset(path, format='csv')
    result = dataset_reader.to_table(dataset)
    assert result.equals(
        pa.table({'skipped': pa.array(['col0', 'foo', 'bar'])}))

    dataset = ds.dataset(path, format=ds.CsvFileFormat(
        read_options=pa.csv.ReadOptions(skip_rows=1)))
    result = dataset_reader.to_table(dataset)
    assert result.equals(pa.table({'col0': pa.array(['foo', 'bar'])}))

    dataset = ds.dataset(path, format=ds.CsvFileFormat(
        read_options=pa.csv.ReadOptions(column_names=['foo'])))
    result = dataset_reader.to_table(dataset)
    assert result.equals(
        pa.table({'foo': pa.array(['skipped', 'col0', 'foo', 'bar'])}))


def test_csv_format_options_generate_columns(tempdir, dataset_reader):
    path = str(tempdir / 'test.csv')
    with open(path, 'w') as sink:
        sink.write('1,a,true,1\n')

    dataset = ds.dataset(path, format=ds.CsvFileFormat(
        read_options=pa.csv.ReadOptions(autogenerate_column_names=True)))
    result = dataset_reader.to_table(dataset)
    expected_column_names = ["f0", "f1", "f2", "f3"]
    assert result.column_names == expected_column_names
    assert result.equals(pa.table({'f0': pa.array([1]),
                                   'f1': pa.array(["a"]),
                                   'f2': pa.array([True]),
                                   'f3': pa.array([1])}))


def test_csv_fragment_options(tempdir, dataset_reader):
    path = str(tempdir / 'test.csv')
    with open(path, 'w') as sink:
        sink.write('col0\nfoo\nspam\nMYNULL\n')
    dataset = ds.dataset(path, format='csv')
    convert_options = pyarrow.csv.ConvertOptions(null_values=['MYNULL'],
                                                 strings_can_be_null=True)
    options = ds.CsvFragmentScanOptions(
        convert_options=convert_options,
        read_options=pa.csv.ReadOptions(block_size=2**16))
    result = dataset_reader.to_table(dataset, fragment_scan_options=options)
    assert result.equals(pa.table({'col0': pa.array(['foo', 'spam', None])}))

    csv_format = ds.CsvFileFormat(convert_options=convert_options)
    dataset = ds.dataset(path, format=csv_format)
    result = dataset_reader.to_table(dataset)
    assert result.equals(pa.table({'col0': pa.array(['foo', 'spam', None])}))

    options = ds.CsvFragmentScanOptions()
    result = dataset_reader.to_table(dataset, fragment_scan_options=options)
    assert result.equals(
        pa.table({'col0': pa.array(['foo', 'spam', 'MYNULL'])}))


def test_encoding(tempdir, dataset_reader):
    path = str(tempdir / 'test.csv')

    for encoding, input_rows in [
        ('latin-1', b"a,b\nun,\xe9l\xe9phant"),
        ('utf16', b'\xff\xfea\x00,\x00b\x00\n\x00u\x00n\x00,'
         b'\x00\xe9\x00l\x00\xe9\x00p\x00h\x00a\x00n\x00t\x00'),
    ]:

        with open(path, 'wb') as sink:
            sink.write(input_rows)

        # Interpret as utf8:
        expected_schema = pa.schema([("a", pa.string()), ("b", pa.string())])
        expected_table = pa.table({'a': ["un"],
                                   'b': ["éléphant"]}, schema=expected_schema)

        read_options = pa.csv.ReadOptions(encoding=encoding)
        file_format = ds.CsvFileFormat(read_options=read_options)
        dataset_transcoded = ds.dataset(path, format=file_format)
        assert dataset_transcoded.schema.equals(expected_schema)
        assert dataset_transcoded.to_table().equals(expected_table)


# Test if a dataset with non-utf8 chars in the column names is properly handled
def test_column_names_encoding(tempdir, dataset_reader):
    path = str(tempdir / 'test.csv')

    with open(path, 'wb') as sink:
        sink.write(b"\xe9,b\nun,\xe9l\xe9phant")

    # Interpret as utf8:
    expected_schema = pa.schema([("é", pa.string()), ("b", pa.string())])
    expected_table = pa.table({'é': ["un"],
                               'b': ["éléphant"]}, schema=expected_schema)

    # Reading as string without specifying encoding should produce an error
    dataset = ds.dataset(path, format='csv', schema=expected_schema)
    with pytest.raises(pyarrow.lib.ArrowInvalid, match="invalid UTF8"):
        dataset_reader.to_table(dataset)

    # Setting the encoding in the read_options should transcode the data
    read_options = pa.csv.ReadOptions(encoding='latin-1')
    file_format = ds.CsvFileFormat(read_options=read_options)
    dataset_transcoded = ds.dataset(path, format=file_format)
    assert dataset_transcoded.schema.equals(expected_schema)
    assert dataset_transcoded.to_table().equals(expected_table)


def test_feather_format(tempdir, dataset_reader):
    from pyarrow.feather import write_feather

    table = pa.table({'a': pa.array([1, 2, 3], type="int8"),
                      'b': pa.array([.1, .2, .3], type="float64")})

    basedir = tempdir / "feather_dataset"
    basedir.mkdir()
    write_feather(table, str(basedir / "data.feather"))

    dataset = ds.dataset(basedir, format=ds.IpcFileFormat())
    result = dataset_reader.to_table(dataset)
    assert result.equals(table)

    assert_dataset_fragment_convenience_methods(dataset)

    dataset = ds.dataset(basedir, format="feather")
    result = dataset_reader.to_table(dataset)
    assert result.equals(table)

    # ARROW-8641 - column selection order
    result = dataset_reader.to_table(dataset, columns=["b", "a"])
    assert result.column_names == ["b", "a"]
    result = dataset_reader.to_table(dataset, columns=["a", "a"])
    assert result.column_names == ["a", "a"]

    # error with Feather v1 files
    write_feather(table, str(basedir / "data1.feather"), version=1)
    with pytest.raises(ValueError):
        dataset_reader.to_table(ds.dataset(basedir, format="feather"))


@pytest.mark.pandas
@pytest.mark.parametrize("compression", [
    "lz4",
    "zstd",
    "brotli"  # not supported
])
def test_feather_format_compressed(tempdir, compression, dataset_reader):
    table = pa.table({'a': pa.array([0]*300, type="int8"),
                      'b': pa.array([.1, .2, .3]*100, type="float64")})
    if not pa.Codec.is_available(compression):
        pytest.skip()

    basedir = tempdir / "feather_dataset_compressed"
    basedir.mkdir()
    file_format = ds.IpcFileFormat()

    uncompressed_basedir = tempdir / "feather_dataset_uncompressed"
    uncompressed_basedir.mkdir()
    ds.write_dataset(
        table,
        str(uncompressed_basedir / "data.arrow"),
        format=file_format,
        file_options=file_format.make_write_options(compression=None)
    )

    if compression == "brotli":
        with pytest.raises(ValueError, match="Compression type"):
            write_options = file_format.make_write_options(
                compression=compression)
        with pytest.raises(ValueError, match="Compression type"):
            codec = pa.Codec(compression)
            write_options = file_format.make_write_options(compression=codec)
        return

    write_options = file_format.make_write_options(compression=compression)
    ds.write_dataset(
        table,
        str(basedir / "data.arrow"),
        format=file_format,
        file_options=write_options
    )

    dataset = ds.dataset(basedir, format=ds.IpcFileFormat())
    result = dataset_reader.to_table(dataset)
    assert result.equals(table)

    compressed_file = basedir / "data.arrow" / "part-0.arrow"
    compressed_size = compressed_file.stat().st_size
    uncompressed_file = uncompressed_basedir / "data.arrow" / "part-0.arrow"
    uncompressed_size = uncompressed_file.stat().st_size
    assert compressed_size < uncompressed_size


def _create_parquet_dataset_simple(root_path):
    """
    Creates a simple (flat files, no nested partitioning) Parquet dataset
    """

    metadata_collector = []

    for i in range(4):
        table = pa.table({'f1': [i] * 10, 'f2': np.random.randn(10)})
        pq.write_to_dataset(
            table, str(root_path), metadata_collector=metadata_collector
        )

    metadata_path = str(root_path / '_metadata')
    # write _metadata file
    pq.write_metadata(
        table.schema, metadata_path,
        metadata_collector=metadata_collector
    )
    return metadata_path, table


@pytest.mark.parquet
@pytest.mark.pandas  # write_to_dataset currently requires pandas
def test_parquet_dataset_factory(tempdir):
    root_path = tempdir / "test_parquet_dataset"
    metadata_path, table = _create_parquet_dataset_simple(root_path)
    dataset = ds.parquet_dataset(metadata_path)
    assert dataset.schema.equals(table.schema)
    assert len(dataset.files) == 4
    result = dataset.to_table()
    assert result.num_rows == 40


@pytest.mark.parquet
@pytest.mark.pandas  # write_to_dataset currently requires pandas
@pytest.mark.skipif(sys.platform == 'win32',
                    reason="Results in FileNotFoundError on Windows")
def test_parquet_dataset_factory_fsspec(tempdir):
    # https://issues.apache.org/jira/browse/ARROW-16413
    fsspec = pytest.importorskip("fsspec")

    # create dataset with pyarrow
    root_path = tempdir / "test_parquet_dataset"
    metadata_path, table = _create_parquet_dataset_simple(root_path)

    # read using fsspec filesystem
    fsspec_fs = fsspec.filesystem("file")
    # manually creating a PyFileSystem, because passing the local fsspec
    # filesystem would internally be converted to native LocalFileSystem
    filesystem = fs.PyFileSystem(fs.FSSpecHandler(fsspec_fs))
    dataset = ds.parquet_dataset(metadata_path, filesystem=filesystem)
    assert dataset.schema.equals(table.schema)
    assert len(dataset.files) == 4
    result = dataset.to_table()
    assert result.num_rows == 40


@pytest.mark.parquet
@pytest.mark.pandas  # write_to_dataset currently requires pandas
@pytest.mark.parametrize('use_legacy_dataset', [False, True])
@pytest.mark.filterwarnings(
    "ignore:Passing 'use_legacy_dataset=True':FutureWarning")
def test_parquet_dataset_factory_roundtrip(tempdir, use_legacy_dataset):
    # Simple test to ensure we can roundtrip dataset to
    # _metadata/common_metadata and back.  A more complex test
    # using partitioning will have to wait for ARROW-13269.  The
    # above test (test_parquet_dataset_factory) will not work
    # when legacy is False as there is no "append" equivalent in
    # the new dataset until ARROW-12358
    root_path = tempdir / "test_parquet_dataset"
    table = pa.table({'f1': [0] * 10, 'f2': np.random.randn(10)})
    metadata_collector = []
    pq.write_to_dataset(
        table, str(root_path), metadata_collector=metadata_collector,
        use_legacy_dataset=use_legacy_dataset
    )
    metadata_path = str(root_path / '_metadata')
    # write _metadata file
    pq.write_metadata(
        table.schema, metadata_path,
        metadata_collector=metadata_collector
    )
    dataset = ds.parquet_dataset(metadata_path)
    assert dataset.schema.equals(table.schema)
    result = dataset.to_table()
    assert result.num_rows == 10


@pytest.mark.parquet
def test_parquet_dataset_factory_order(tempdir):
    # The order of the fragments in the dataset should match the order of the
    # row groups in the _metadata file.
    metadatas = []
    # Create a dataset where f1 is incrementing from 0 to 100 spread across
    # 10 files.  Put the row groups in the correct order in _metadata
    for i in range(10):
        table = pa.table(
            {'f1': list(range(i*10, (i+1)*10))})
        table_path = tempdir / f'{i}.parquet'
        pq.write_table(table, table_path, metadata_collector=metadatas)
        metadatas[-1].set_file_path(f'{i}.parquet')
    metadata_path = str(tempdir / '_metadata')
    pq.write_metadata(table.schema, metadata_path, metadatas)
    dataset = ds.parquet_dataset(metadata_path)
    # Ensure the table contains values from 0-100 in the right order
    scanned_table = dataset.to_table()
    scanned_col = scanned_table.column('f1').to_pylist()
    assert scanned_col == list(range(0, 100))


@pytest.mark.parquet
@pytest.mark.pandas
def test_parquet_dataset_factory_invalid(tempdir):
    root_path = tempdir / "test_parquet_dataset_invalid"
    metadata_path, table = _create_parquet_dataset_simple(root_path)
    # remove one of the files
    list(root_path.glob("*.parquet"))[0].unlink()
    dataset = ds.parquet_dataset(metadata_path)
    assert dataset.schema.equals(table.schema)
    assert len(dataset.files) == 4
    with pytest.raises(FileNotFoundError):
        dataset.to_table()


def _create_metadata_file(root_path):
    # create _metadata file from existing parquet dataset
    parquet_paths = list(sorted(root_path.rglob("*.parquet")))
    schema = pq.ParquetFile(parquet_paths[0]).schema.to_arrow_schema()

    metadata_collector = []
    for path in parquet_paths:
        metadata = pq.ParquetFile(path).metadata
        metadata.set_file_path(str(path.relative_to(root_path)))
        metadata_collector.append(metadata)

    metadata_path = root_path / "_metadata"
    pq.write_metadata(
        schema, metadata_path, metadata_collector=metadata_collector
    )
    return metadata_path


def _create_parquet_dataset_partitioned(root_path):
    table = pa.table([
        pa.array(range(20)), pa.array(np.random.randn(20)),
        pa.array(np.repeat(['a', 'b'], 10))],
        names=["f1", "f2", "part"]
    )
    table = table.replace_schema_metadata({"key": "value"})
    pq.write_to_dataset(table, str(root_path), partition_cols=['part'])
    return _create_metadata_file(root_path), table


@pytest.mark.parquet
@pytest.mark.pandas
def test_parquet_dataset_factory_partitioned(tempdir):
    root_path = tempdir / "test_parquet_dataset_factory_partitioned"
    metadata_path, table = _create_parquet_dataset_partitioned(root_path)

    partitioning = ds.partitioning(flavor="hive")
    dataset = ds.parquet_dataset(metadata_path, partitioning=partitioning)

    assert dataset.schema.equals(table.schema)
    assert len(dataset.files) == 2
    result = dataset.to_table()
    assert result.num_rows == 20

    # the partitioned dataset does not preserve order
    result = result.to_pandas().sort_values("f1").reset_index(drop=True)
    expected = table.to_pandas()
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parquet
@pytest.mark.pandas
def test_parquet_dataset_factory_metadata(tempdir):
    # ensure ParquetDatasetFactory preserves metadata (ARROW-9363)
    root_path = tempdir / "test_parquet_dataset_factory_metadata"
    metadata_path, table = _create_parquet_dataset_partitioned(root_path)

    dataset = ds.parquet_dataset(metadata_path, partitioning="hive")
    assert dataset.schema.equals(table.schema)
    assert b"key" in dataset.schema.metadata

    fragments = list(dataset.get_fragments())
    assert b"key" in fragments[0].physical_schema.metadata


@pytest.mark.parquet
@pytest.mark.pandas
def test_parquet_dataset_lazy_filtering(tempdir, open_logging_fs):
    fs, assert_opens = open_logging_fs

    # Test to ensure that no IO happens when filtering a dataset
    # created with ParquetDatasetFactory from a _metadata file

    root_path = tempdir / "test_parquet_dataset_lazy_filtering"
    metadata_path, _ = _create_parquet_dataset_simple(root_path)

    # creating the dataset should only open the metadata file
    with assert_opens([metadata_path]):
        dataset = ds.parquet_dataset(
            metadata_path,
            partitioning=ds.partitioning(flavor="hive"),
            filesystem=fs)

    # materializing fragments should not open any file
    with assert_opens([]):
        fragments = list(dataset.get_fragments())

    # filtering fragments should not open any file
    with assert_opens([]):
        list(dataset.get_fragments(ds.field("f1") > 15))

    # splitting by row group should still not open any file
    with assert_opens([]):
        fragments[0].split_by_row_group(ds.field("f1") > 15)

    # ensuring metadata of split fragment should also not open any file
    with assert_opens([]):
        rg_fragments = fragments[0].split_by_row_group()
        rg_fragments[0].ensure_complete_metadata()

    # FIXME(bkietz) on Windows this results in FileNotFoundErrors.
    # but actually scanning does open files
    # with assert_opens([f.path for f in fragments]):
    #    dataset.to_table()


@pytest.mark.parquet
@pytest.mark.pandas
def test_dataset_schema_metadata(tempdir, dataset_reader):
    # ARROW-8802
    df = pd.DataFrame({'a': [1, 2, 3]})
    path = tempdir / "test.parquet"
    df.to_parquet(path)
    dataset = ds.dataset(path)

    schema = dataset_reader.to_table(dataset).schema
    projected_schema = dataset_reader.to_table(dataset, columns=["a"]).schema

    # ensure the pandas metadata is included in the schema
    assert b"pandas" in schema.metadata
    # ensure it is still there in a projected schema (with column selection)
    assert schema.equals(projected_schema, check_metadata=True)


@pytest.mark.parquet
def test_filter_mismatching_schema(tempdir, dataset_reader):
    # ARROW-9146
    table = pa.table({"col": pa.array([1, 2, 3, 4], type='int32')})
    pq.write_table(table, str(tempdir / "data.parquet"))

    # specifying explicit schema, but that mismatches the schema of the data
    schema = pa.schema([("col", pa.int64())])
    dataset = ds.dataset(
        tempdir / "data.parquet", format="parquet", schema=schema)

    # filtering on a column with such type mismatch should implicitly
    # cast the column
    filtered = dataset_reader.to_table(dataset, filter=ds.field("col") > 2)
    assert filtered["col"].equals(table["col"].cast('int64').slice(2))

    fragment = list(dataset.get_fragments())[0]
    filtered = dataset_reader.to_table(
        fragment, filter=ds.field("col") > 2, schema=schema)
    assert filtered["col"].equals(table["col"].cast('int64').slice(2))


@pytest.mark.parquet
@pytest.mark.pandas
def test_dataset_project_only_partition_columns(tempdir, dataset_reader):
    # ARROW-8729
    table = pa.table({'part': 'a a b b'.split(), 'col': list(range(4))})

    path = str(tempdir / 'test_dataset')
    pq.write_to_dataset(table, path, partition_cols=['part'])
    dataset = ds.dataset(path, partitioning='hive')

    all_cols = dataset_reader.to_table(dataset)
    part_only = dataset_reader.to_table(dataset, columns=['part'])

    assert all_cols.column('part').equals(part_only.column('part'))


@pytest.mark.parquet
@pytest.mark.pandas
def test_dataset_project_null_column(tempdir, dataset_reader):
    import pandas as pd
    df = pd.DataFrame({"col": np.array([None, None, None], dtype='object')})

    f = tempdir / "test_dataset_project_null_column.parquet"
    df.to_parquet(f, engine="pyarrow")

    dataset = ds.dataset(f, format="parquet",
                         schema=pa.schema([("col", pa.int64())]))
    expected = pa.table({'col': pa.array([None, None, None], pa.int64())})
    assert dataset_reader.to_table(dataset).equals(expected)


def test_dataset_project_columns(tempdir, dataset_reader):
    # basic column re-projection with expressions
    from pyarrow import feather
    table = pa.table({"A": [1, 2, 3], "B": [1., 2., 3.], "C": ["a", "b", "c"]})
    feather.write_feather(table, tempdir / "data.feather")

    dataset = ds.dataset(tempdir / "data.feather", format="feather")
    result = dataset_reader.to_table(dataset, columns={
        'A_renamed': ds.field('A'),
        'B_as_int': ds.field('B').cast("int32", safe=False),
        'C_is_a': ds.field('C') == 'a'
    })
    expected = pa.table({
        "A_renamed": [1, 2, 3],
        "B_as_int": pa.array([1, 2, 3], type="int32"),
        "C_is_a": [True, False, False],
    })
    assert result.equals(expected)

    # raise proper error when not passing an expression
    with pytest.raises(TypeError, match="Expected an Expression"):
        dataset_reader.to_table(dataset, columns={"A": "A"})


@pytest.mark.pandas
@pytest.mark.parquet
def test_dataset_preserved_partitioning(tempdir):
    # ARROW-8655

    # through discovery, but without partitioning
    _, path = _create_single_file(tempdir)
    dataset = ds.dataset(path)
    assert dataset.partitioning is None

    # through discovery, with hive partitioning but not specified
    full_table, path = _create_partitioned_dataset(tempdir)
    dataset = ds.dataset(path)
    assert dataset.partitioning is None

    # through discovery, with hive partitioning (from a partitioning factory)
    dataset = ds.dataset(path, partitioning="hive")
    part = dataset.partitioning
    assert part is not None
    assert isinstance(part, ds.HivePartitioning)
    assert part.schema == pa.schema([("part", pa.int32())])
    assert len(part.dictionaries) == 1
    assert part.dictionaries[0] == pa.array([0, 1, 2], pa.int32())

    # through discovery, with hive partitioning (from a partitioning object)
    part = ds.partitioning(pa.schema([("part", pa.int32())]), flavor="hive")
    assert isinstance(part, ds.HivePartitioning)  # not a factory
    assert len(part.dictionaries) == 1
    assert all(x is None for x in part.dictionaries)
    dataset = ds.dataset(path, partitioning=part)
    part = dataset.partitioning
    assert isinstance(part, ds.HivePartitioning)
    assert part.schema == pa.schema([("part", pa.int32())])
    # TODO is this expected?
    assert len(part.dictionaries) == 1
    assert all(x is None for x in part.dictionaries)

    # through manual creation -> not available
    dataset = ds.dataset(path, partitioning="hive")
    dataset2 = ds.FileSystemDataset(
        list(dataset.get_fragments()), schema=dataset.schema,
        format=dataset.format, filesystem=dataset.filesystem
    )
    assert dataset2.partitioning is None

    # through discovery with ParquetDatasetFactory
    root_path = tempdir / "data-partitioned-metadata"
    metadata_path, _ = _create_parquet_dataset_partitioned(root_path)
    dataset = ds.parquet_dataset(metadata_path, partitioning="hive")
    part = dataset.partitioning
    assert part is not None
    assert isinstance(part, ds.HivePartitioning)
    assert part.schema == pa.schema([("part", pa.string())])
    assert len(part.dictionaries) == 1
    # will be fixed by ARROW-13153 (order is not preserved at the moment)
    # assert part.dictionaries[0] == pa.array(["a", "b"], pa.string())
    assert set(part.dictionaries[0].to_pylist()) == {"a", "b"}


@pytest.mark.parquet
@pytest.mark.pandas
def test_write_to_dataset_given_null_just_works(tempdir):
    schema = pa.schema([
        pa.field('col', pa.int64()),
        pa.field('part', pa.dictionary(pa.int32(), pa.string()))
    ])
    table = pa.table({'part': [None, None, 'a', 'a'],
                      'col': list(range(4))}, schema=schema)

    path = str(tempdir / 'test_dataset')
    pq.write_to_dataset(table, path, partition_cols=[
                        'part'], use_legacy_dataset=False)

    actual_table = pq.read_table(tempdir / 'test_dataset')
    # column.equals can handle the difference in chunking but not the fact
    # that `part` will have different dictionaries for the two chunks
    assert actual_table.column('part').to_pylist(
    ) == table.column('part').to_pylist()
    assert actual_table.column('col').equals(table.column('col'))


@pytest.mark.parquet
@pytest.mark.pandas
@pytest.mark.filterwarnings(
    "ignore:Passing 'use_legacy_dataset=True':FutureWarning")
def test_legacy_write_to_dataset_drops_null(tempdir):
    schema = pa.schema([
        pa.field('col', pa.int64()),
        pa.field('part', pa.dictionary(pa.int32(), pa.string()))
    ])
    table = pa.table({'part': ['a', 'a', None, None],
                      'col': list(range(4))}, schema=schema)
    expected = pa.table(
        {'part': ['a', 'a'], 'col': list(range(2))}, schema=schema)

    path = str(tempdir / 'test_dataset')
    pq.write_to_dataset(table, path, partition_cols=[
                        'part'], use_legacy_dataset=True)

    actual = pq.read_table(tempdir / 'test_dataset')
    assert actual == expected


def _sort_table(tab, sort_col):
    import pyarrow.compute as pc
    sorted_indices = pc.sort_indices(
        tab, options=pc.SortOptions([(sort_col, 'ascending')]))
    return pc.take(tab, sorted_indices)


def _check_dataset_roundtrip(dataset, base_dir, expected_files, sort_col,
                             base_dir_path=None, partitioning=None):
    base_dir_path = base_dir_path or base_dir

    ds.write_dataset(dataset, base_dir, format="arrow",
                     partitioning=partitioning, use_threads=False)

    # check that all files are present
    file_paths = list(base_dir_path.rglob("*"))
    assert set(file_paths) == set(expected_files)

    # check that reading back in as dataset gives the same result
    dataset2 = ds.dataset(
        base_dir_path, format="arrow", partitioning=partitioning)

    assert _sort_table(dataset2.to_table(), sort_col).equals(
        _sort_table(dataset.to_table(), sort_col))


@pytest.mark.parquet
def test_write_dataset(tempdir):
    # manually create a written dataset and read as dataset object
    directory = tempdir / 'single-file'
    directory.mkdir()
    _ = _create_single_file(directory)
    dataset = ds.dataset(directory)

    # full string path
    target = tempdir / 'single-file-target'
    expected_files = [target / "part-0.arrow"]
    _check_dataset_roundtrip(dataset, str(target), expected_files, 'a', target)

    # pathlib path object
    target = tempdir / 'single-file-target2'
    expected_files = [target / "part-0.arrow"]
    _check_dataset_roundtrip(dataset, target, expected_files, 'a', target)

    # TODO
    # # relative path
    # target = tempdir / 'single-file-target3'
    # expected_files = [target / "part-0.ipc"]
    # _check_dataset_roundtrip(
    #     dataset, './single-file-target3', expected_files, target)

    # Directory of files
    directory = tempdir / 'single-directory'
    directory.mkdir()
    _ = _create_directory_of_files(directory)
    dataset = ds.dataset(directory)

    target = tempdir / 'single-directory-target'
    expected_files = [target / "part-0.arrow"]
    _check_dataset_roundtrip(dataset, str(target), expected_files, 'a', target)


@pytest.mark.parquet
@pytest.mark.pandas
def test_write_dataset_partitioned(tempdir):
    directory = tempdir / "partitioned"
    _ = _create_parquet_dataset_partitioned(directory)
    partitioning = ds.partitioning(flavor="hive")
    dataset = ds.dataset(directory, partitioning=partitioning)

    # hive partitioning
    target = tempdir / 'partitioned-hive-target'
    expected_paths = [
        target / "part=a", target / "part=a" / "part-0.arrow",
        target / "part=b", target / "part=b" / "part-0.arrow"
    ]
    partitioning_schema = ds.partitioning(
        pa.schema([("part", pa.string())]), flavor="hive")
    _check_dataset_roundtrip(
        dataset, str(target), expected_paths, 'f1', target,
        partitioning=partitioning_schema)

    # directory partitioning
    target = tempdir / 'partitioned-dir-target'
    expected_paths = [
        target / "a", target / "a" / "part-0.arrow",
        target / "b", target / "b" / "part-0.arrow"
    ]
    partitioning_schema = ds.partitioning(
        pa.schema([("part", pa.string())]))
    _check_dataset_roundtrip(
        dataset, str(target), expected_paths, 'f1', target,
        partitioning=partitioning_schema)


def test_write_dataset_with_field_names(tempdir):
    table = pa.table({'a': ['x', 'y', None], 'b': ['x', 'y', 'z']})

    ds.write_dataset(table, tempdir, format='ipc',
                     partitioning=["b"])

    load_back = ds.dataset(tempdir, format='ipc', partitioning=["b"])
    files = load_back.files
    partitioning_dirs = {
        str(pathlib.Path(f).relative_to(tempdir).parent) for f in files
    }
    assert partitioning_dirs == {"x", "y", "z"}

    load_back_table = load_back.to_table()
    assert load_back_table.equals(table)


def test_write_dataset_with_field_names_hive(tempdir):
    table = pa.table({'a': ['x', 'y', None], 'b': ['x', 'y', 'z']})

    ds.write_dataset(table, tempdir, format='ipc',
                     partitioning=["b"], partitioning_flavor="hive")

    load_back = ds.dataset(tempdir, format='ipc', partitioning="hive")
    files = load_back.files
    partitioning_dirs = {
        str(pathlib.Path(f).relative_to(tempdir).parent) for f in files
    }
    assert partitioning_dirs == {"b=x", "b=y", "b=z"}

    load_back_table = load_back.to_table()
    assert load_back_table.equals(table)


def test_write_dataset_with_scanner(tempdir):
    table = pa.table({'a': ['x', 'y', None], 'b': ['x', 'y', 'z'],
                      'c': [1, 2, 3]})

    ds.write_dataset(table, tempdir, format='ipc',
                     partitioning=["b"])

    dataset = ds.dataset(tempdir, format='ipc', partitioning=["b"])

    with tempfile.TemporaryDirectory() as tempdir2:
        ds.write_dataset(dataset.scanner(columns=["b", "c"]),
                         tempdir2, format='ipc', partitioning=["b"])

        load_back = ds.dataset(tempdir2, format='ipc', partitioning=["b"])
        load_back_table = load_back.to_table()
        assert dict(load_back_table.to_pydict()
                    ) == table.drop(["a"]).to_pydict()


@pytest.mark.parquet
def test_write_dataset_with_backpressure(tempdir):
    consumer_gate = threading.Event()

    # A filesystem that blocks all writes so that we can build
    # up backpressure.  The writes are released at the end of
    # the test.
    class GatingFs(ProxyHandler):
        def open_output_stream(self, path, metadata):
            # Block until the end of the test
            consumer_gate.wait()
            return self._fs.open_output_stream(path, metadata=metadata)
    gating_fs = fs.PyFileSystem(GatingFs(fs.LocalFileSystem()))

    schema = pa.schema([pa.field('data', pa.int32())])
    # The scanner should queue ~ 8Mi rows (~8 batches) but due to ARROW-16258
    # it always queues 32 batches.
    batch = pa.record_batch([pa.array(list(range(1_000_000)))], schema=schema)
    batches_read = 0
    min_backpressure = 32
    end = 200
    keep_going = True

    def counting_generator():
        nonlocal batches_read
        while batches_read < end:
            if not keep_going:
                return
            time.sleep(0.01)
            batches_read += 1
            yield batch

    scanner = ds.Scanner.from_batches(
        counting_generator(), schema=schema, use_threads=True)

    write_thread = threading.Thread(
        target=lambda: ds.write_dataset(
            scanner, str(tempdir), format='parquet', filesystem=gating_fs))
    write_thread.start()

    try:
        start = time.time()

        def duration():
            return time.time() - start

        # This test is timing dependent.  There is no signal from the C++
        # when backpressure has been hit.  We don't know exactly when
        # backpressure will be hit because it may take some time for the
        # signal to get from the sink to the scanner.
        #
        # The test may emit false positives on slow systems.  It could
        # theoretically emit a false negative if the scanner managed to read
        # and emit all 200 batches before the backpressure signal had a chance
        # to propagate but the 0.01s delay in the generator should make that
        # scenario unlikely.
        last_value = 0
        backpressure_probably_hit = False
        while duration() < 10:
            if batches_read > min_backpressure:
                if batches_read == last_value:
                    backpressure_probably_hit = True
                    break
                last_value = batches_read
            time.sleep(0.5)

        assert backpressure_probably_hit

    finally:
        # If any batches remain to be generated go ahead and
        # skip them
        keep_going = False
        consumer_gate.set()
        write_thread.join()


def test_write_dataset_with_dataset(tempdir):
    table = pa.table({'b': ['x', 'y', 'z'], 'c': [1, 2, 3]})

    ds.write_dataset(table, tempdir, format='ipc',
                     partitioning=["b"])

    dataset = ds.dataset(tempdir, format='ipc', partitioning=["b"])

    with tempfile.TemporaryDirectory() as tempdir2:
        ds.write_dataset(dataset, tempdir2,
                         format='ipc', partitioning=["b"])

        load_back = ds.dataset(tempdir2, format='ipc', partitioning=["b"])
        load_back_table = load_back.to_table()
        assert dict(load_back_table.to_pydict()) == table.to_pydict()


@pytest.mark.pandas
def test_write_dataset_existing_data(tempdir):
    directory = tempdir / 'ds'
    table = pa.table({'b': ['x', 'y', 'z'], 'c': [1, 2, 3]})
    partitioning = ds.partitioning(schema=pa.schema(
        [pa.field('c', pa.int64())]), flavor='hive')

    def compare_tables_ignoring_order(t1, t2):
        df1 = t1.to_pandas().sort_values('b').reset_index(drop=True)
        df2 = t2.to_pandas().sort_values('b').reset_index(drop=True)
        assert df1.equals(df2)

    # First write is ok
    ds.write_dataset(table, directory, partitioning=partitioning, format='ipc')

    table = pa.table({'b': ['a', 'b', 'c'], 'c': [2, 3, 4]})

    # Second write should fail
    with pytest.raises(pa.ArrowInvalid):
        ds.write_dataset(table, directory,
                         partitioning=partitioning, format='ipc')

    extra_table = pa.table({'b': ['e']})
    extra_file = directory / 'c=2' / 'foo.arrow'
    pyarrow.feather.write_feather(extra_table, extra_file)

    # Should be ok and overwrite with overwrite behavior
    ds.write_dataset(table, directory, partitioning=partitioning,
                     format='ipc',
                     existing_data_behavior='overwrite_or_ignore')

    overwritten = pa.table(
        {'b': ['e', 'x', 'a', 'b', 'c'], 'c': [2, 1, 2, 3, 4]})
    readback = ds.dataset(tempdir, format='ipc',
                          partitioning=partitioning).to_table()
    compare_tables_ignoring_order(readback, overwritten)
    assert extra_file.exists()

    # Should be ok and delete matching with delete_matching
    ds.write_dataset(table, directory, partitioning=partitioning,
                     format='ipc', existing_data_behavior='delete_matching')

    overwritten = pa.table({'b': ['x', 'a', 'b', 'c'], 'c': [1, 2, 3, 4]})
    readback = ds.dataset(tempdir, format='ipc',
                          partitioning=partitioning).to_table()
    compare_tables_ignoring_order(readback, overwritten)
    assert not extra_file.exists()


def _generate_random_int_array(size=4, min=1, max=10):
    return np.random.randint(min, max, size)


def _generate_data_and_columns(num_of_columns, num_of_records):
    data = []
    column_names = []
    for i in range(num_of_columns):
        data.append(_generate_random_int_array(size=num_of_records,
                                               min=1,
                                               max=num_of_records))
        column_names.append("c" + str(i))
    record_batch = pa.record_batch(data=data, names=column_names)
    return record_batch


def _get_num_of_files_generated(base_directory, file_format):
    return len(list(pathlib.Path(base_directory).glob(f'**/*.{file_format}')))


@pytest.mark.parquet
def test_write_dataset_max_rows_per_file(tempdir):
    directory = tempdir / 'ds'
    max_rows_per_file = 10
    max_rows_per_group = 10
    num_of_columns = 2
    num_of_records = 35

    record_batch = _generate_data_and_columns(num_of_columns,
                                              num_of_records)

    ds.write_dataset(record_batch, directory, format="parquet",
                     max_rows_per_file=max_rows_per_file,
                     max_rows_per_group=max_rows_per_group)

    files_in_dir = os.listdir(directory)

    # number of partitions with max_rows and the partition with the remainder
    expected_partitions = num_of_records // max_rows_per_file + 1

    # test whether the expected amount of files are written
    assert len(files_in_dir) == expected_partitions

    # compute the number of rows per each file written
    result_row_combination = []
    for _, f_file in enumerate(files_in_dir):
        f_path = directory / str(f_file)
        dataset = ds.dataset(f_path, format="parquet")
        result_row_combination.append(dataset.to_table().shape[0])

    # test whether the generated files have the expected number of rows
    assert expected_partitions == len(result_row_combination)
    assert num_of_records == sum(result_row_combination)
    assert all(file_rowcount <= max_rows_per_file
               for file_rowcount in result_row_combination)


@pytest.mark.parquet
def test_write_dataset_min_rows_per_group(tempdir):
    directory = tempdir / 'ds'
    min_rows_per_group = 6
    max_rows_per_group = 8
    num_of_columns = 2

    record_sizes = [5, 5, 5, 5, 5, 4, 4, 4, 4, 4]

    record_batches = [_generate_data_and_columns(num_of_columns,
                                                 num_of_records)
                      for num_of_records in record_sizes]

    data_source = directory / "min_rows_group"

    ds.write_dataset(data=record_batches, base_dir=data_source,
                     min_rows_per_group=min_rows_per_group,
                     max_rows_per_group=max_rows_per_group,
                     format="parquet")

    files_in_dir = os.listdir(data_source)
    for _, f_file in enumerate(files_in_dir):
        f_path = data_source / str(f_file)
        dataset = ds.dataset(f_path, format="parquet")
        table = dataset.to_table()
        batches = table.to_batches()

        for id, batch in enumerate(batches):
            rows_per_batch = batch.num_rows
            if id < len(batches) - 1:
                assert rows_per_batch >= min_rows_per_group and \
                    rows_per_batch <= max_rows_per_group
            else:
                assert rows_per_batch <= max_rows_per_group


@pytest.mark.parquet
def test_write_dataset_max_rows_per_group(tempdir):
    directory = tempdir / 'ds'
    max_rows_per_group = 18
    num_of_columns = 2
    num_of_records = 30

    record_batch = _generate_data_and_columns(num_of_columns,
                                              num_of_records)

    data_source = directory / "max_rows_group"

    ds.write_dataset(data=record_batch, base_dir=data_source,
                     max_rows_per_group=max_rows_per_group,
                     format="parquet")

    files_in_dir = os.listdir(data_source)
    batched_data = []
    for f_file in files_in_dir:
        f_path = data_source / str(f_file)
        dataset = ds.dataset(f_path, format="parquet")
        table = dataset.to_table()
        batches = table.to_batches()
        for batch in batches:
            batched_data.append(batch.num_rows)

    assert batched_data == [18, 12]


@pytest.mark.parquet
def test_write_dataset_max_open_files(tempdir):
    directory = tempdir / 'ds'
    file_format = "parquet"
    partition_column_id = 1
    column_names = ['c1', 'c2']
    record_batch_1 = pa.record_batch(data=[[1, 2, 3, 4, 0, 10],
                                           ['a', 'b', 'c', 'd', 'e', 'a']],
                                     names=column_names)
    record_batch_2 = pa.record_batch(data=[[5, 6, 7, 8, 0, 1],
                                           ['a', 'b', 'c', 'd', 'e', 'c']],
                                     names=column_names)
    record_batch_3 = pa.record_batch(data=[[9, 10, 11, 12, 0, 1],
                                           ['a', 'b', 'c', 'd', 'e', 'd']],
                                     names=column_names)
    record_batch_4 = pa.record_batch(data=[[13, 14, 15, 16, 0, 1],
                                           ['a', 'b', 'c', 'd', 'e', 'b']],
                                     names=column_names)

    table = pa.Table.from_batches([record_batch_1, record_batch_2,
                                   record_batch_3, record_batch_4])

    partitioning = ds.partitioning(
        pa.schema([(column_names[partition_column_id], pa.string())]),
        flavor="hive")

    data_source_1 = directory / "default"

    ds.write_dataset(data=table, base_dir=data_source_1,
                     partitioning=partitioning, format=file_format)

    # Here we consider the number of unique partitions created when
    # partitioning column contains duplicate records.
    #   Returns: (number_of_files_generated, number_of_partitions)
    def _get_compare_pair(data_source, record_batch, file_format, col_id):
        num_of_files_generated = _get_num_of_files_generated(
            base_directory=data_source, file_format=file_format)
        number_of_partitions = len(pa.compute.unique(record_batch[col_id]))
        return num_of_files_generated, number_of_partitions

    # CASE 1: when max_open_files=default & max_open_files >= num_of_partitions
    #         In case of a writing to disk via partitioning based on a
    #         particular column (considering row labels in that column),
    #         the number of unique rows must be equal
    #         to the number of files generated

    num_of_files_generated, number_of_partitions \
        = _get_compare_pair(data_source_1, record_batch_1, file_format,
                            partition_column_id)
    assert num_of_files_generated == number_of_partitions

    # CASE 2: when max_open_files > 0 & max_open_files < num_of_partitions
    #         the number of files generated must be greater than the number of
    #         partitions

    data_source_2 = directory / "max_1"

    max_open_files = 3

    ds.write_dataset(data=table, base_dir=data_source_2,
                     partitioning=partitioning, format=file_format,
                     max_open_files=max_open_files, use_threads=False)

    num_of_files_generated, number_of_partitions \
        = _get_compare_pair(data_source_2, record_batch_1, file_format,
                            partition_column_id)
    assert num_of_files_generated > number_of_partitions


@pytest.mark.parquet
@pytest.mark.pandas
def test_write_dataset_partitioned_dict(tempdir):
    directory = tempdir / "partitioned"
    _ = _create_parquet_dataset_partitioned(directory)

    # directory partitioning, dictionary partition columns
    dataset = ds.dataset(
        directory,
        partitioning=ds.HivePartitioning.discover(infer_dictionary=True))
    target = tempdir / 'partitioned-dir-target'
    expected_paths = [
        target / "a", target / "a" / "part-0.arrow",
        target / "b", target / "b" / "part-0.arrow"
    ]
    partitioning = ds.partitioning(pa.schema([
        dataset.schema.field('part')]),
        dictionaries={'part': pa.array(['a', 'b'])})
    # NB: dictionaries required here since we use partitioning to parse
    # directories in _check_dataset_roundtrip (not currently required for
    # the formatting step)
    _check_dataset_roundtrip(
        dataset, str(target), expected_paths, 'f1', target,
        partitioning=partitioning)


@pytest.mark.parquet
@pytest.mark.pandas
def test_write_dataset_use_threads(tempdir):
    directory = tempdir / "partitioned"
    _ = _create_parquet_dataset_partitioned(directory)
    dataset = ds.dataset(directory, partitioning="hive")

    partitioning = ds.partitioning(
        pa.schema([("part", pa.string())]), flavor="hive")

    target1 = tempdir / 'partitioned1'
    paths_written = []

    def file_visitor(written_file):
        paths_written.append(written_file.path)

    ds.write_dataset(
        dataset, target1, format="feather", partitioning=partitioning,
        use_threads=True, file_visitor=file_visitor
    )

    expected_paths = {
        target1 / 'part=a' / 'part-0.feather',
        target1 / 'part=b' / 'part-0.feather'
    }
    paths_written_set = set(map(pathlib.Path, paths_written))
    assert paths_written_set == expected_paths

    target2 = tempdir / 'partitioned2'
    ds.write_dataset(
        dataset, target2, format="feather", partitioning=partitioning,
        use_threads=False
    )

    # check that reading in gives same result
    result1 = ds.dataset(target1, format="feather", partitioning=partitioning)
    result2 = ds.dataset(target2, format="feather", partitioning=partitioning)
    assert result1.to_table().equals(result2.to_table())


def test_write_table(tempdir):
    table = pa.table([
        pa.array(range(20)), pa.array(np.random.randn(20)),
        pa.array(np.repeat(['a', 'b'], 10))
    ], names=["f1", "f2", "part"])

    base_dir = tempdir / 'single'
    ds.write_dataset(table, base_dir,
                     basename_template='dat_{i}.arrow', format="feather")
    # check that all files are present
    file_paths = list(base_dir.rglob("*"))
    expected_paths = [base_dir / "dat_0.arrow"]
    assert set(file_paths) == set(expected_paths)
    # check Table roundtrip
    result = ds.dataset(base_dir, format="ipc").to_table()
    assert result.equals(table)

    # with partitioning
    base_dir = tempdir / 'partitioned'
    expected_paths = [
        base_dir / "part=a", base_dir / "part=a" / "dat_0.arrow",
        base_dir / "part=b", base_dir / "part=b" / "dat_0.arrow"
    ]

    visited_paths = []
    visited_sizes = []

    def file_visitor(written_file):
        visited_paths.append(written_file.path)
        visited_sizes.append(written_file.size)

    partitioning = ds.partitioning(
        pa.schema([("part", pa.string())]), flavor="hive")
    ds.write_dataset(table, base_dir, format="feather",
                     basename_template='dat_{i}.arrow',
                     partitioning=partitioning, file_visitor=file_visitor)
    file_paths = list(base_dir.rglob("*"))
    assert set(file_paths) == set(expected_paths)
    actual_sizes = [os.path.getsize(path) for path in visited_paths]
    assert visited_sizes == actual_sizes
    result = ds.dataset(base_dir, format="ipc", partitioning=partitioning)
    assert result.to_table().equals(table)
    assert len(visited_paths) == 2
    for visited_path in visited_paths:
        assert pathlib.Path(visited_path) in expected_paths


def test_write_table_multiple_fragments(tempdir):
    table = pa.table([
        pa.array(range(10)), pa.array(np.random.randn(10)),
        pa.array(np.repeat(['a', 'b'], 5))
    ], names=["f1", "f2", "part"])
    table = pa.concat_tables([table]*2)

    # Table with multiple batches written as single Fragment by default
    base_dir = tempdir / 'single'
    ds.write_dataset(table, base_dir, format="feather")
    assert set(base_dir.rglob("*")) == set([base_dir / "part-0.feather"])
    assert ds.dataset(base_dir, format="ipc").to_table().equals(table)

    # Same for single-element list of Table
    base_dir = tempdir / 'single-list'
    ds.write_dataset([table], base_dir, format="feather")
    assert set(base_dir.rglob("*")) == set([base_dir / "part-0.feather"])
    assert ds.dataset(base_dir, format="ipc").to_table().equals(table)

    # Provide list of batches to write multiple fragments
    base_dir = tempdir / 'multiple'
    ds.write_dataset(table.to_batches(), base_dir, format="feather")
    assert set(base_dir.rglob("*")) == set(
        [base_dir / "part-0.feather"])
    assert ds.dataset(base_dir, format="ipc").to_table().equals(table)

    # Provide list of tables to write multiple fragments
    base_dir = tempdir / 'multiple-table'
    ds.write_dataset([table, table], base_dir, format="feather")
    assert set(base_dir.rglob("*")) == set(
        [base_dir / "part-0.feather"])
    assert ds.dataset(base_dir, format="ipc").to_table().equals(
        pa.concat_tables([table]*2)
    )


def test_write_iterable(tempdir):
    table = pa.table([
        pa.array(range(20)), pa.array(np.random.randn(20)),
        pa.array(np.repeat(['a', 'b'], 10))
    ], names=["f1", "f2", "part"])

    base_dir = tempdir / 'inmemory_iterable'
    ds.write_dataset((batch for batch in table.to_batches()), base_dir,
                     schema=table.schema,
                     basename_template='dat_{i}.arrow', format="feather")
    result = ds.dataset(base_dir, format="ipc").to_table()
    assert result.equals(table)

    base_dir = tempdir / 'inmemory_reader'
    reader = pa.RecordBatchReader.from_batches(table.schema,
                                               table.to_batches())
    ds.write_dataset(reader, base_dir,
                     basename_template='dat_{i}.arrow', format="feather")
    result = ds.dataset(base_dir, format="ipc").to_table()
    assert result.equals(table)


def test_write_scanner(tempdir, dataset_reader):
    table = pa.table([
        pa.array(range(20)), pa.array(np.random.randn(20)),
        pa.array(np.repeat(['a', 'b'], 10))
    ], names=["f1", "f2", "part"])
    dataset = ds.dataset(table)

    base_dir = tempdir / 'dataset_from_scanner'
    ds.write_dataset(dataset_reader.scanner(
        dataset), base_dir, format="feather")
    result = dataset_reader.to_table(ds.dataset(base_dir, format="ipc"))
    assert result.equals(table)

    # scanner with different projected_schema
    base_dir = tempdir / 'dataset_from_scanner2'
    ds.write_dataset(dataset_reader.scanner(dataset, columns=["f1"]),
                     base_dir, format="feather")
    result = dataset_reader.to_table(ds.dataset(base_dir, format="ipc"))
    assert result.equals(table.select(["f1"]))

    # schema not allowed when writing a scanner
    with pytest.raises(ValueError, match="Cannot specify a schema"):
        ds.write_dataset(dataset_reader.scanner(dataset), base_dir,
                         schema=table.schema, format="feather")


def test_write_table_partitioned_dict(tempdir):
    # ensure writing table partitioned on a dictionary column works without
    # specifying the dictionary values explicitly
    table = pa.table([
        pa.array(range(20)),
        pa.array(np.repeat(['a', 'b'], 10)).dictionary_encode(),
    ], names=['col', 'part'])

    partitioning = ds.partitioning(table.select(["part"]).schema)

    base_dir = tempdir / "dataset"
    ds.write_dataset(
        table, base_dir, format="feather", partitioning=partitioning
    )

    # check roundtrip
    partitioning_read = ds.DirectoryPartitioning.discover(
        ["part"], infer_dictionary=True)
    result = ds.dataset(
        base_dir, format="ipc", partitioning=partitioning_read
    ).to_table()
    assert result.equals(table)


@pytest.mark.parquet
def test_write_dataset_parquet(tempdir):
    table = pa.table([
        pa.array(range(20)), pa.array(np.random.randn(20)),
        pa.array(np.repeat(['a', 'b'], 10))
    ], names=["f1", "f2", "part"])

    # using default "parquet" format string

    base_dir = tempdir / 'parquet_dataset'
    ds.write_dataset(table, base_dir, format="parquet")
    # check that all files are present
    file_paths = list(base_dir.rglob("*"))
    expected_paths = [base_dir / "part-0.parquet"]
    assert set(file_paths) == set(expected_paths)
    # check Table roundtrip
    result = ds.dataset(base_dir, format="parquet").to_table()
    assert result.equals(table)

    # using custom options
    for version in ["1.0", "2.4", "2.6"]:
        format = ds.ParquetFileFormat()
        opts = format.make_write_options(version=version)
        base_dir = tempdir / 'parquet_dataset_version{0}'.format(version)
        ds.write_dataset(table, base_dir, format=format, file_options=opts)
        meta = pq.read_metadata(base_dir / "part-0.parquet")
        expected_version = "1.0" if version == "1.0" else "2.6"
        assert meta.format_version == expected_version


def test_write_dataset_csv(tempdir):
    table = pa.table([
        pa.array(range(20)), pa.array(np.random.randn(20)),
        pa.array(np.repeat(['a', 'b'], 10))
    ], names=["f1", "f2", "chr1"])

    base_dir = tempdir / 'csv_dataset'
    ds.write_dataset(table, base_dir, format="csv")
    # check that all files are present
    file_paths = list(base_dir.rglob("*"))
    expected_paths = [base_dir / "part-0.csv"]
    assert set(file_paths) == set(expected_paths)
    # check Table roundtrip
    result = ds.dataset(base_dir, format="csv").to_table()
    assert result.equals(table)

    # using custom options
    format = ds.CsvFileFormat(read_options=pyarrow.csv.ReadOptions(
        column_names=table.schema.names))
    opts = format.make_write_options(include_header=False)
    base_dir = tempdir / 'csv_dataset_noheader'
    ds.write_dataset(table, base_dir, format=format, file_options=opts)
    result = ds.dataset(base_dir, format=format).to_table()
    assert result.equals(table)


@pytest.mark.parquet
def test_write_dataset_parquet_file_visitor(tempdir):
    table = pa.table([
        pa.array(range(20)), pa.array(np.random.randn(20)),
        pa.array(np.repeat(['a', 'b'], 10))
    ], names=["f1", "f2", "part"])

    visitor_called = False

    def file_visitor(written_file):
        nonlocal visitor_called
        if (written_file.metadata is not None and
                written_file.metadata.num_columns == 3):
            visitor_called = True

    base_dir = tempdir / 'parquet_dataset'
    ds.write_dataset(table, base_dir, format="parquet",
                     file_visitor=file_visitor)

    assert visitor_called


@pytest.mark.parquet
def test_partition_dataset_parquet_file_visitor(tempdir):
    f1_vals = [item for chunk in range(4) for item in [chunk] * 10]
    f2_vals = [item*10 for chunk in range(4) for item in [chunk] * 10]
    table = pa.table({'f1': f1_vals, 'f2': f2_vals,
                      'part': np.repeat(['a', 'b'], 20)})

    root_path = tempdir / 'partitioned'
    partitioning = ds.partitioning(
        pa.schema([("part", pa.string())]), flavor="hive")

    paths_written = []

    sample_metadata = None

    def file_visitor(written_file):
        nonlocal sample_metadata
        if written_file.metadata:
            sample_metadata = written_file.metadata
        paths_written.append(written_file.path)

    ds.write_dataset(
        table, root_path, format="parquet", partitioning=partitioning,
        use_threads=True, file_visitor=file_visitor
    )

    expected_paths = {
        root_path / 'part=a' / 'part-0.parquet',
        root_path / 'part=b' / 'part-0.parquet'
    }
    paths_written_set = set(map(pathlib.Path, paths_written))
    assert paths_written_set == expected_paths
    assert sample_metadata is not None
    assert sample_metadata.num_columns == 2


@pytest.mark.parquet
@pytest.mark.pandas
def test_write_dataset_arrow_schema_metadata(tempdir):
    # ensure we serialize ARROW schema in the parquet metadata, to have a
    # correct roundtrip (e.g. preserve non-UTC timezone)
    table = pa.table({"a": [pd.Timestamp("2012-01-01", tz="Europe/Brussels")]})
    assert table["a"].type.tz == "Europe/Brussels"

    ds.write_dataset(table, tempdir, format="parquet")
    result = pq.read_table(tempdir / "part-0.parquet")
    assert result["a"].type.tz == "Europe/Brussels"


def test_write_dataset_schema_metadata(tempdir):
    # ensure that schema metadata gets written
    from pyarrow import feather

    table = pa.table({'a': [1, 2, 3]})
    table = table.replace_schema_metadata({b'key': b'value'})
    ds.write_dataset(table, tempdir, format="feather")

    schema = feather.read_table(tempdir / "part-0.feather").schema
    assert schema.metadata == {b'key': b'value'}


@pytest.mark.parquet
def test_write_dataset_schema_metadata_parquet(tempdir):
    # ensure that schema metadata gets written
    table = pa.table({'a': [1, 2, 3]})
    table = table.replace_schema_metadata({b'key': b'value'})
    ds.write_dataset(table, tempdir, format="parquet")

    schema = pq.read_table(tempdir / "part-0.parquet").schema
    assert schema.metadata == {b'key': b'value'}


@pytest.mark.parquet
@pytest.mark.s3
def test_write_dataset_s3(s3_example_simple):
    # write dataset with s3 filesystem
    _, _, fs, _, host, port, access_key, secret_key = s3_example_simple
    uri_template = (
        "s3://{}:{}@{{}}?scheme=http&endpoint_override={}:{}".format(
            access_key, secret_key, host, port)
    )

    table = pa.table([
        pa.array(range(20)), pa.array(np.random.randn(20)),
        pa.array(np.repeat(['a', 'b'], 10))],
        names=["f1", "f2", "part"]
    )
    part = ds.partitioning(pa.schema([("part", pa.string())]), flavor="hive")

    # writing with filesystem object
    ds.write_dataset(
        table, "mybucket/dataset", filesystem=fs, format="feather",
        partitioning=part
    )
    # check roundtrip
    result = ds.dataset(
        "mybucket/dataset", filesystem=fs, format="ipc", partitioning="hive"
    ).to_table()
    assert result.equals(table)

    # writing with URI
    uri = uri_template.format("mybucket/dataset2")
    ds.write_dataset(table, uri, format="feather", partitioning=part)
    # check roundtrip
    result = ds.dataset(
        "mybucket/dataset2", filesystem=fs, format="ipc", partitioning="hive"
    ).to_table()
    assert result.equals(table)

    # writing with path + URI as filesystem
    uri = uri_template.format("mybucket")
    ds.write_dataset(
        table, "dataset3", filesystem=uri, format="feather", partitioning=part
    )
    # check roundtrip
    result = ds.dataset(
        "mybucket/dataset3", filesystem=fs, format="ipc", partitioning="hive"
    ).to_table()
    assert result.equals(table)


_minio_put_only_policy = """{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:ListBucket",
                "s3:GetObjectVersion"
            ],
            "Resource": [
                "arn:aws:s3:::*"
            ]
        }
    ]
}"""


@pytest.mark.parquet
@pytest.mark.s3
def test_write_dataset_s3_put_only(s3_server):
    # [ARROW-15892] Testing the create_dir flag which will restrict
    # creating a new directory for writing a dataset. This is
    # required while writing a dataset in s3 where we have very
    # limited permissions and thus we can directly write the dataset
    # without creating a directory.
    from pyarrow.fs import S3FileSystem

    # write dataset with s3 filesystem
    host, port, _, _ = s3_server['connection']
    fs = S3FileSystem(
        access_key='limited',
        secret_key='limited123',
        endpoint_override='{}:{}'.format(host, port),
        scheme='http'
    )
    _configure_s3_limited_user(s3_server, _minio_put_only_policy)

    table = pa.table([
        pa.array(range(20)), pa.array(np.random.randn(20)),
        pa.array(np.repeat(['a', 'b'], 10))],
        names=["f1", "f2", "part"]
    )
    part = ds.partitioning(pa.schema([("part", pa.string())]), flavor="hive")

    # writing with filesystem object with create_dir flag set to false
    ds.write_dataset(
        table, "existing-bucket", filesystem=fs,
        format="feather", create_dir=False, partitioning=part,
        existing_data_behavior='overwrite_or_ignore'
    )
    # check roundtrip
    result = ds.dataset(
        "existing-bucket", filesystem=fs, format="ipc", partitioning="hive"
    ).to_table()
    assert result.equals(table)

    # Passing create_dir is fine if the bucket already exists
    ds.write_dataset(
        table, "existing-bucket", filesystem=fs,
        format="feather", create_dir=True, partitioning=part,
        existing_data_behavior='overwrite_or_ignore'
    )
    # check roundtrip
    result = ds.dataset(
        "existing-bucket", filesystem=fs, format="ipc", partitioning="hive"
    ).to_table()
    assert result.equals(table)

    # Error enforced by filesystem
    with pytest.raises(OSError,
                       match="Bucket 'non-existing-bucket' not found"):
        ds.write_dataset(
            table, "non-existing-bucket", filesystem=fs,
            format="feather", create_dir=True,
            existing_data_behavior='overwrite_or_ignore'
        )

    # Error enforced by minio / S3 service
    fs = S3FileSystem(
        access_key='limited',
        secret_key='limited123',
        endpoint_override='{}:{}'.format(host, port),
        scheme='http',
        allow_bucket_creation=True,
    )
    with pytest.raises(OSError, match="Access Denied"):
        ds.write_dataset(
            table, "non-existing-bucket", filesystem=fs,
            format="feather", create_dir=True,
            existing_data_behavior='overwrite_or_ignore'
        )


@pytest.mark.parquet
def test_dataset_null_to_dictionary_cast(tempdir, dataset_reader):
    # ARROW-12420
    table = pa.table({"a": [None, None]})
    pq.write_table(table, tempdir / "test.parquet")

    schema = pa.schema([
        pa.field("a", pa.dictionary(pa.int32(), pa.string()))
    ])
    fsds = ds.FileSystemDataset.from_paths(
        paths=[tempdir / "test.parquet"],
        schema=schema,
        format=ds.ParquetFileFormat(),
        filesystem=fs.LocalFileSystem(),
    )
    table = dataset_reader.to_table(fsds)
    assert table.schema == schema


@pytest.mark.dataset
def test_dataset_join(tempdir):
    t1 = pa.table({
        "colA": [1, 2, 6],
        "col2": ["a", "b", "f"]
    })
    ds.write_dataset(t1, tempdir / "t1", format="ipc")
    ds1 = ds.dataset(tempdir / "t1", format="ipc")

    t2 = pa.table({
        "colB": [99, 2, 1],
        "col3": ["Z", "B", "A"]
    })
    ds.write_dataset(t2, tempdir / "t2", format="ipc")
    ds2 = ds.dataset(tempdir / "t2", format="ipc")

    result = ds1.join(ds2, "colA", "colB")
    assert result.to_table() == pa.table({
        "colA": [1, 2, 6],
        "col2": ["a", "b", "f"],
        "col3": ["A", "B", None]
    })

    result = ds1.join(ds2, "colA", "colB", join_type="full outer")
    assert result.to_table().sort_by("colA") == pa.table({
        "colA": [1, 2, 6, 99],
        "col2": ["a", "b", "f", None],
        "col3": ["A", "B", None, "Z"]
    })


@pytest.mark.dataset
def test_dataset_join_unique_key(tempdir):
    t1 = pa.table({
        "colA": [1, 2, 6],
        "col2": ["a", "b", "f"]
    })
    ds.write_dataset(t1, tempdir / "t1", format="ipc")
    ds1 = ds.dataset(tempdir / "t1", format="ipc")

    t2 = pa.table({
        "colA": [99, 2, 1],
        "col3": ["Z", "B", "A"]
    })
    ds.write_dataset(t2, tempdir / "t2", format="ipc")
    ds2 = ds.dataset(tempdir / "t2", format="ipc")

    result = ds1.join(ds2, "colA")
    assert result.to_table() == pa.table({
        "colA": [1, 2, 6],
        "col2": ["a", "b", "f"],
        "col3": ["A", "B", None]
    })

    result = ds1.join(ds2, "colA", join_type="full outer", right_suffix="_r")
    assert result.to_table().sort_by("colA") == pa.table({
        "colA": [1, 2, 6, 99],
        "col2": ["a", "b", "f", None],
        "col3": ["A", "B", None, "Z"]
    })


@pytest.mark.dataset
def test_dataset_join_collisions(tempdir):
    t1 = pa.table({
        "colA": [1, 2, 6],
        "colB": [10, 20, 60],
        "colVals": ["a", "b", "f"]
    })
    ds.write_dataset(t1, tempdir / "t1", format="ipc")
    ds1 = ds.dataset(tempdir / "t1", format="ipc")

    t2 = pa.table({
        "colA": [99, 2, 1],
        "colB": [99, 20, 10],
        "colVals": ["Z", "B", "A"]
    })
    ds.write_dataset(t2, tempdir / "t2", format="ipc")
    ds2 = ds.dataset(tempdir / "t2", format="ipc")

    result = ds1.join(ds2, "colA", join_type="full outer", right_suffix="_r")
    assert result.to_table().sort_by("colA") == pa.table([
        [1, 2, 6, 99],
        [10, 20, 60, None],
        ["a", "b", "f", None],
        [10, 20, None, 99],
        ["A", "B", None, "Z"],
    ], names=["colA", "colB", "colVals", "colB_r", "colVals_r"])


@pytest.mark.parametrize('dstype', [
    "fs", "mem"
])
def test_dataset_filter(tempdir, dstype):
    t1 = pa.table({
        "colA": [1, 2, 6, 8],
        "col2": ["a", "b", "f", "g"]
    })
    if dstype == "fs":
        ds.write_dataset(t1, tempdir / "t1", format="ipc")
        ds1 = ds.dataset(tempdir / "t1", format="ipc")
    elif dstype == "mem":
        ds1 = ds.dataset(t1)
    else:
        raise NotImplementedError

    # Ensure chained filtering works.
    result = ds1.filter(pc.field("colA") < 3).filter(pc.field("col2") == "a")
    assert type(result) == (ds.FileSystemDataset if dstype ==
                            "fs" else ds.InMemoryDataset)

    assert result.to_table() == pa.table({
        "colA": [1],
        "col2": ["a"]
    })

    assert result.head(5) == pa.table({
        "colA": [1],
        "col2": ["a"]
    })

    # Ensure that further filtering with scanners works too
    r2 = ds1.filter(pc.field("colA") < 8).filter(
        pc.field("colA") > 1).scanner(filter=pc.field("colA") != 6)
    assert r2.to_table() == pa.table({
        "colA": [2],
        "col2": ["b"]
    })

    # Ensure that writing back to disk works.
    ds.write_dataset(result, tempdir / "filtered", format="ipc")
    filtered = ds.dataset(tempdir / "filtered", format="ipc")
    assert filtered.to_table() == pa.table({
        "colA": [1],
        "col2": ["a"]
    })

    # Ensure that joining to a filtered Dataset works.
    joined = result.join(ds.dataset(pa.table({
        "colB": [10, 20],
        "col2": ["a", "b"]
    })), keys="col2", join_type="right outer")
    assert joined.to_table().sort_by("colB") == pa.table({
        "colA": [1, None],
        "colB": [10, 20],
        "col2": ["a", "b"]
    })

    # Filter with None doesn't work for now
    with pytest.raises(TypeError):
        ds1.filter(None)

    # Can't get fragments of a filtered dataset
    with pytest.raises(ValueError):
        result.get_fragments()

    # Ensure replacing schema preserves the filter.
    schema_without_col2 = ds1.schema.remove(1)
    newschema = ds1.filter(
        pc.field("colA") < 3
    ).replace_schema(schema_without_col2)
    assert newschema.to_table() == pa.table({
        "colA": [1, 2],
    })
    with pytest.raises(pa.ArrowInvalid):
        # The schema might end up being replaced with
        # something that makes the filter invalid.
        # Let's make sure we error nicely.
        result.replace_schema(schema_without_col2).to_table()


@pytest.mark.parametrize('dstype', [
    "fs", "mem"
])
def test_union_dataset_filter(tempdir, dstype):
    t1 = pa.table({
        "colA": [1, 2, 6, 8],
        "col2": ["a", "b", "f", "g"]
    })
    t2 = pa.table({
        "colA": [9, 10, 11],
        "col2": ["h", "i", "l"]
    })
    if dstype == "fs":
        ds.write_dataset(t1, tempdir / "t1", format="ipc")
        ds1 = ds.dataset(tempdir / "t1", format="ipc")
        ds.write_dataset(t2, tempdir / "t2", format="ipc")
        ds2 = ds.dataset(tempdir / "t2", format="ipc")
    elif dstype == "mem":
        ds1 = ds.dataset(t1)
        ds2 = ds.dataset(t2)
    else:
        raise NotImplementedError

    filtered_union_ds = ds.dataset((ds1, ds2)).filter(
        (pc.field("colA") < 3) | (pc.field("colA") == 9)
    )
    assert filtered_union_ds.to_table() == pa.table({
        "colA": [1, 2, 9],
        "col2": ["a", "b", "h"]
    })

    joined = filtered_union_ds.join(ds.dataset(pa.table({
        "colB": [10, 20],
        "col2": ["a", "b"]
    })), keys="col2", join_type="left outer")
    assert joined.to_table().sort_by("colA") == pa.table({
        "colA": [1, 2, 9],
        "col2": ["a", "b", "h"],
        "colB": [10, 20, None]
    })

    filtered_ds1 = ds1.filter(pc.field("colA") < 3)
    filtered_ds2 = ds2.filter(pc.field("colA") < 10)

    with pytest.raises(ValueError, match="currently not supported"):
        ds.dataset((filtered_ds1, filtered_ds2))


def test_parquet_dataset_filter(tempdir):
    root_path = tempdir / "test_parquet_dataset_filter"
    metadata_path, _ = _create_parquet_dataset_simple(root_path)
    dataset = ds.parquet_dataset(metadata_path)

    result = dataset.to_table()
    assert result.num_rows == 40

    filtered_ds = dataset.filter(pc.field("f1") < 2)
    assert filtered_ds.to_table().num_rows == 20

    with pytest.raises(ValueError):
        filtered_ds.get_fragments()


def test_write_dataset_with_scanner_use_projected_schema(tempdir):
    """
    Ensure the projected schema is used to validate partitions for scanner

    https://issues.apache.org/jira/browse/ARROW-17228
    """
    table = pa.table([pa.array(range(20))], names=["original_column"])
    table_dataset = ds.dataset(table)
    columns = {
        "renamed_column": ds.field("original_column"),
    }
    scanner = table_dataset.scanner(columns=columns)

    ds.write_dataset(
        scanner, tempdir, partitioning=["renamed_column"], format="ipc")
    with (
        pytest.raises(
            KeyError, match=r"'Column original_column does not exist in schema"
        )
    ):
        ds.write_dataset(
            scanner, tempdir, partitioning=["original_column"], format="ipc"
        )


@pytest.mark.parametrize("format", ("ipc", "parquet"))
def test_read_table_nested_columns(tempdir, format):
    if format == "parquet":
        pytest.importorskip("pyarrow.parquet")

    table = pa.table({"user_id": ["abc123", "qrs456"],
                      "a.dotted.field": [1, 2],
                      "interaction": [
        {"type": None, "element": "button",
         "values": [1, 2], "structs":[{"foo": "bar"}, None]},
        {"type": "scroll", "element": "window",
         "values": [None, 3, 4], "structs":[{"fizz": "buzz"}]}
    ]})
    ds.write_dataset(table, tempdir / "table", format=format)
    ds1 = ds.dataset(tempdir / "table", format=format)

    # Dot path to read subsets of nested data
    table = ds1.to_table(
        columns=["user_id", "interaction.type", "interaction.values",
                 "interaction.structs", "a.dotted.field"])
    assert table.to_pylist() == [
        {'user_id': 'abc123', 'type': None, 'values': [1, 2],
         'structs': [{'fizz': None, 'foo': 'bar'}, None], 'a.dotted.field': 1},
        {'user_id': 'qrs456', 'type': 'scroll', 'values': [None, 3, 4],
         'structs': [{'fizz': 'buzz', 'foo': None}], 'a.dotted.field': 2}
    ]


def test_dataset_partition_with_slash(tmpdir):
    from pyarrow import dataset as ds

    path = tmpdir / "slash-writer-x"

    dt_table = pa.Table.from_arrays([
        pa.array([1, 2, 3, 4, 5], pa.int32()),
        pa.array(["experiment/A/f.csv", "experiment/B/f.csv",
                  "experiment/A/f.csv", "experiment/C/k.csv",
                  "experiment/M/i.csv"], pa.utf8())], ["exp_id", "exp_meta"])

    ds.write_dataset(
        data=dt_table,
        base_dir=path,
        format='ipc',
        partitioning=['exp_meta'],
        partitioning_flavor='hive',
    )

    read_table = ds.dataset(
        source=path,
        format='ipc',
        partitioning='hive',
        schema=pa.schema([pa.field("exp_id", pa.int32()),
                          pa.field("exp_meta", pa.utf8())])
    ).to_table().combine_chunks()

    assert dt_table == read_table.sort_by("exp_id")

    exp_meta = dt_table.column(1).to_pylist()
    exp_meta = sorted(set(exp_meta))  # take unique
    encoded_paths = ["exp_meta=" + quote(path, safe='') for path in exp_meta]
    file_paths = sorted(os.listdir(path))

    assert encoded_paths == file_paths


@pytest.mark.parametrize('dstype', [
    "fs", "mem"
])
def test_dataset_sort_by(tempdir, dstype):
    table = pa.table([
        pa.array([3, 1, 4, 2, 5]),
        pa.array(["b", "a", "b", "a", "c"]),
    ], names=["values", "keys"])

    if dstype == "fs":
        ds.write_dataset(table, tempdir / "t1", format="ipc")
        dt = ds.dataset(tempdir / "t1", format="ipc")
    elif dstype == "mem":
        dt = ds.dataset(table)
    else:
        raise NotImplementedError

    assert dt.sort_by("values").to_table().to_pydict() == {
        "keys": ["a", "a", "b", "b", "c"],
        "values": [1, 2, 3, 4, 5]
    }

    assert dt.sort_by([("values", "descending")]).to_table().to_pydict() == {
        "keys": ["c", "b", "b", "a", "a"],
        "values": [5, 4, 3, 2, 1]
    }

    assert dt.filter((pc.field("values") < 4)).sort_by(
        "values"
    ).to_table().to_pydict() == {
        "keys": ["a", "a", "b"],
        "values": [1, 2, 3]
    }

    table = pa.Table.from_arrays([
        pa.array([5, 7, 7, 35], type=pa.int64()),
        pa.array(["foo", "car", "bar", "foobar"])
    ], names=["a", "b"])
    dt = ds.dataset(table)

    sorted_tab = dt.sort_by([("a", "descending")])
    sorted_tab_dict = sorted_tab.to_table().to_pydict()
    assert sorted_tab_dict["a"] == [35, 7, 7, 5]
    assert sorted_tab_dict["b"] == ["foobar", "car", "bar", "foo"]

    sorted_tab = dt.sort_by([("a", "ascending")])
    sorted_tab_dict = sorted_tab.to_table().to_pydict()
    assert sorted_tab_dict["a"] == [5, 7, 7, 35]
    assert sorted_tab_dict["b"] == ["foo", "car", "bar", "foobar"]

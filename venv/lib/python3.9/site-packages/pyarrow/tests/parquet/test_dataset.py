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

import datetime
import inspect
import os
import pathlib

import numpy as np
import pytest
import unittest.mock as mock

import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem
from pyarrow.tests import util
from pyarrow.tests.parquet.common import (
    parametrize_legacy_dataset, parametrize_legacy_dataset_fixed,
    parametrize_legacy_dataset_not_supported)
from pyarrow.util import guid
from pyarrow.vendored.version import Version

try:
    import pyarrow.parquet as pq
    from pyarrow.tests.parquet.common import (
        _read_table, _test_dataframe, _write_table)
except ImportError:
    pq = None


try:
    import pandas as pd
    import pandas.testing as tm

except ImportError:
    pd = tm = None


# Marks all of the tests in this module
# Ignore these with pytest ... -m 'not parquet'
pytestmark = pytest.mark.parquet


@pytest.mark.pandas
def test_parquet_piece_read(tempdir):
    df = _test_dataframe(1000)
    table = pa.Table.from_pandas(df)

    path = tempdir / 'parquet_piece_read.parquet'
    _write_table(table, path, version='2.6')

    with pytest.warns(FutureWarning):
        piece1 = pq.ParquetDatasetPiece(path)

    result = piece1.read()
    assert result.equals(table)


@pytest.mark.pandas
def test_parquet_piece_open_and_get_metadata(tempdir):
    df = _test_dataframe(100)
    table = pa.Table.from_pandas(df)

    path = tempdir / 'parquet_piece_read.parquet'
    _write_table(table, path, version='2.6')

    with pytest.warns(FutureWarning):
        piece = pq.ParquetDatasetPiece(path)

    table1 = piece.read()
    assert isinstance(table1, pa.Table)
    meta1 = piece.get_metadata()
    assert isinstance(meta1, pq.FileMetaData)

    assert table.equals(table1)


@pytest.mark.filterwarnings("ignore:ParquetDatasetPiece:FutureWarning")
def test_parquet_piece_basics():
    path = '/baz.parq'

    piece1 = pq.ParquetDatasetPiece(path)
    piece2 = pq.ParquetDatasetPiece(path, row_group=1)
    piece3 = pq.ParquetDatasetPiece(
        path, row_group=1, partition_keys=[('foo', 0), ('bar', 1)])

    assert str(piece1) == path
    assert str(piece2) == '/baz.parq | row_group=1'
    assert str(piece3) == 'partition[foo=0, bar=1] /baz.parq | row_group=1'

    assert piece1 == piece1
    assert piece2 == piece2
    assert piece3 == piece3
    assert piece1 != piece3


def test_partition_set_dictionary_type():
    set1 = pq.PartitionSet('key1', ['foo', 'bar', 'baz'])
    set2 = pq.PartitionSet('key2', [2007, 2008, 2009])

    assert isinstance(set1.dictionary, pa.StringArray)
    assert isinstance(set2.dictionary, pa.IntegerArray)

    set3 = pq.PartitionSet('key2', [datetime.datetime(2007, 1, 1)])
    with pytest.raises(TypeError):
        set3.dictionary


@parametrize_legacy_dataset_fixed
def test_filesystem_uri(tempdir, use_legacy_dataset):
    table = pa.table({"a": [1, 2, 3]})

    directory = tempdir / "data_dir"
    directory.mkdir()
    path = directory / "data.parquet"
    pq.write_table(table, str(path))

    # filesystem object
    result = pq.read_table(
        path, filesystem=fs.LocalFileSystem(),
        use_legacy_dataset=use_legacy_dataset)
    assert result.equals(table)

    # filesystem URI
    result = pq.read_table(
        "data_dir/data.parquet", filesystem=util._filesystem_uri(tempdir),
        use_legacy_dataset=use_legacy_dataset)
    assert result.equals(table)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_read_partitioned_directory(tempdir, use_legacy_dataset):
    fs = LocalFileSystem._get_instance()
    _partition_test_for_filesystem(fs, tempdir, use_legacy_dataset)


@pytest.mark.filterwarnings("ignore:'ParquetDataset:FutureWarning")
@pytest.mark.pandas
def test_create_parquet_dataset_multi_threaded(tempdir):
    fs = LocalFileSystem._get_instance()
    base_path = tempdir

    _partition_test_for_filesystem(fs, base_path)

    manifest = pq.ParquetManifest(base_path, filesystem=fs,
                                  metadata_nthreads=1)
    with pytest.warns(
        FutureWarning, match="Specifying the 'metadata_nthreads'"
    ):
        dataset = pq.ParquetDataset(
            base_path, filesystem=fs, metadata_nthreads=16,
            use_legacy_dataset=True
        )
    assert len(dataset.pieces) > 0
    partitions = dataset.partitions
    assert len(partitions.partition_names) > 0
    assert partitions.partition_names == manifest.partitions.partition_names
    assert len(partitions.levels) == len(manifest.partitions.levels)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_read_partitioned_columns_selection(tempdir, use_legacy_dataset):
    # ARROW-3861 - do not include partition columns in resulting table when
    # `columns` keyword was passed without those columns
    fs = LocalFileSystem._get_instance()
    base_path = tempdir
    _partition_test_for_filesystem(fs, base_path)

    dataset = pq.ParquetDataset(
        base_path, use_legacy_dataset=use_legacy_dataset)
    result = dataset.read(columns=["values"])
    if use_legacy_dataset:
        # ParquetDataset implementation always includes the partition columns
        # automatically, and we can't easily "fix" this since dask relies on
        # this behaviour (ARROW-8644)
        assert result.column_names == ["values", "foo", "bar"]
    else:
        assert result.column_names == ["values"]


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_filters_equivalency(tempdir, use_legacy_dataset):
    fs = LocalFileSystem._get_instance()
    base_path = tempdir

    integer_keys = [0, 1]
    string_keys = ['a', 'b', 'c']
    boolean_keys = [True, False]
    partition_spec = [
        ['integer', integer_keys],
        ['string', string_keys],
        ['boolean', boolean_keys]
    ]

    df = pd.DataFrame({
        'integer': np.array(integer_keys, dtype='i4').repeat(15),
        'string': np.tile(np.tile(np.array(string_keys, dtype=object), 5), 2),
        'boolean': np.tile(np.tile(np.array(boolean_keys, dtype='bool'), 5),
                           3),
    }, columns=['integer', 'string', 'boolean'])

    _generate_partition_directories(fs, base_path, partition_spec, df)

    # Old filters syntax:
    #  integer == 1 AND string != b AND boolean == True
    dataset = pq.ParquetDataset(
        base_path, filesystem=fs,
        filters=[('integer', '=', 1), ('string', '!=', 'b'),
                 ('boolean', '==', 'True')],
        use_legacy_dataset=use_legacy_dataset,
    )
    table = dataset.read()
    result_df = (table.to_pandas().reset_index(drop=True))

    assert 0 not in result_df['integer'].values
    assert 'b' not in result_df['string'].values
    assert False not in result_df['boolean'].values

    # filters in disjunctive normal form:
    #  (integer == 1 AND string != b AND boolean == True) OR
    #  (integer == 2 AND boolean == False)
    # TODO(ARROW-3388): boolean columns are reconstructed as string
    filters = [
        [
            ('integer', '=', 1),
            ('string', '!=', 'b'),
            ('boolean', '==', 'True')
        ],
        [('integer', '=', 0), ('boolean', '==', 'False')]
    ]
    dataset = pq.ParquetDataset(
        base_path, filesystem=fs, filters=filters,
        use_legacy_dataset=use_legacy_dataset)
    table = dataset.read()
    result_df = table.to_pandas().reset_index(drop=True)

    # Check that all rows in the DF fulfill the filter
    df_filter_1 = (result_df['integer'] == 1) \
        & (result_df['string'] != 'b') \
        & (result_df['boolean'] == 'True')
    df_filter_2 = (np.array(result_df['integer']) == 0) \
        & (result_df['boolean'] == 'False')
    assert df_filter_1.sum() > 0
    assert df_filter_2.sum() > 0
    assert result_df.shape[0] == (df_filter_1.sum() + df_filter_2.sum())

    if use_legacy_dataset:
        # Check for \0 in predicate values. Until they are correctly
        # implemented in ARROW-3391, they would otherwise lead to weird
        # results with the current code.
        with pytest.raises(NotImplementedError):
            filters = [[('string', '==', b'1\0a')]]
            pq.ParquetDataset(base_path, filesystem=fs, filters=filters,
                              use_legacy_dataset=True)
        with pytest.raises(NotImplementedError):
            filters = [[('string', '==', '1\0a')]]
            pq.ParquetDataset(base_path, filesystem=fs, filters=filters,
                              use_legacy_dataset=True)
    else:
        for filters in [[[('string', '==', b'1\0a')]],
                        [[('string', '==', '1\0a')]]]:
            dataset = pq.ParquetDataset(
                base_path, filesystem=fs, filters=filters,
                use_legacy_dataset=False)
            assert dataset.read().num_rows == 0


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_filters_cutoff_exclusive_integer(tempdir, use_legacy_dataset):
    fs = LocalFileSystem._get_instance()
    base_path = tempdir

    integer_keys = [0, 1, 2, 3, 4]
    partition_spec = [
        ['integers', integer_keys],
    ]
    N = 5

    df = pd.DataFrame({
        'index': np.arange(N),
        'integers': np.array(integer_keys, dtype='i4'),
    }, columns=['index', 'integers'])

    _generate_partition_directories(fs, base_path, partition_spec, df)

    dataset = pq.ParquetDataset(
        base_path, filesystem=fs,
        filters=[
            ('integers', '<', 4),
            ('integers', '>', 1),
        ],
        use_legacy_dataset=use_legacy_dataset
    )
    table = dataset.read()
    result_df = (table.to_pandas()
                      .sort_values(by='index')
                      .reset_index(drop=True))

    result_list = [x for x in map(int, result_df['integers'].values)]
    assert result_list == [2, 3]


@pytest.mark.pandas
@parametrize_legacy_dataset
@pytest.mark.xfail(
    # different error with use_legacy_datasets because result_df is no longer
    # categorical
    raises=(TypeError, AssertionError),
    reason='Loss of type information in creation of categoricals.'
)
def test_filters_cutoff_exclusive_datetime(tempdir, use_legacy_dataset):
    fs = LocalFileSystem._get_instance()
    base_path = tempdir

    date_keys = [
        datetime.date(2018, 4, 9),
        datetime.date(2018, 4, 10),
        datetime.date(2018, 4, 11),
        datetime.date(2018, 4, 12),
        datetime.date(2018, 4, 13)
    ]
    partition_spec = [
        ['dates', date_keys]
    ]
    N = 5

    df = pd.DataFrame({
        'index': np.arange(N),
        'dates': np.array(date_keys, dtype='datetime64'),
    }, columns=['index', 'dates'])

    _generate_partition_directories(fs, base_path, partition_spec, df)

    dataset = pq.ParquetDataset(
        base_path, filesystem=fs,
        filters=[
            ('dates', '<', "2018-04-12"),
            ('dates', '>', "2018-04-10")
        ],
        use_legacy_dataset=use_legacy_dataset
    )
    table = dataset.read()
    result_df = (table.to_pandas()
                      .sort_values(by='index')
                      .reset_index(drop=True))

    expected = pd.Categorical(
        np.array([datetime.date(2018, 4, 11)], dtype='datetime64'),
        categories=np.array(date_keys, dtype='datetime64'))

    assert result_df['dates'].values == expected


@pytest.mark.pandas
@pytest.mark.dataset
def test_filters_inclusive_datetime(tempdir):
    # ARROW-11480
    path = tempdir / 'timestamps.parquet'

    pd.DataFrame({
        "dates": pd.date_range("2020-01-01", periods=10, freq="D"),
        "id": range(10)
    }).to_parquet(path, use_deprecated_int96_timestamps=True)

    table = pq.read_table(path, filters=[
        ("dates", "<=", datetime.datetime(2020, 1, 5))
    ])

    assert table.column('id').to_pylist() == [0, 1, 2, 3, 4]


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_filters_inclusive_integer(tempdir, use_legacy_dataset):
    fs = LocalFileSystem._get_instance()
    base_path = tempdir

    integer_keys = [0, 1, 2, 3, 4]
    partition_spec = [
        ['integers', integer_keys],
    ]
    N = 5

    df = pd.DataFrame({
        'index': np.arange(N),
        'integers': np.array(integer_keys, dtype='i4'),
    }, columns=['index', 'integers'])

    _generate_partition_directories(fs, base_path, partition_spec, df)

    dataset = pq.ParquetDataset(
        base_path, filesystem=fs,
        filters=[
            ('integers', '<=', 3),
            ('integers', '>=', 2),
        ],
        use_legacy_dataset=use_legacy_dataset
    )
    table = dataset.read()
    result_df = (table.to_pandas()
                 .sort_values(by='index')
                 .reset_index(drop=True))

    result_list = [int(x) for x in map(int, result_df['integers'].values)]
    assert result_list == [2, 3]


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_filters_inclusive_set(tempdir, use_legacy_dataset):
    fs = LocalFileSystem._get_instance()
    base_path = tempdir

    integer_keys = [0, 1]
    string_keys = ['a', 'b', 'c']
    boolean_keys = [True, False]
    partition_spec = [
        ['integer', integer_keys],
        ['string', string_keys],
        ['boolean', boolean_keys]
    ]

    df = pd.DataFrame({
        'integer': np.array(integer_keys, dtype='i4').repeat(15),
        'string': np.tile(np.tile(np.array(string_keys, dtype=object), 5), 2),
        'boolean': np.tile(np.tile(np.array(boolean_keys, dtype='bool'), 5),
                           3),
    }, columns=['integer', 'string', 'boolean'])

    _generate_partition_directories(fs, base_path, partition_spec, df)

    dataset = pq.ParquetDataset(
        base_path, filesystem=fs,
        filters=[('string', 'in', 'ab')],
        use_legacy_dataset=use_legacy_dataset
    )
    table = dataset.read()
    result_df = (table.to_pandas().reset_index(drop=True))

    assert 'a' in result_df['string'].values
    assert 'b' in result_df['string'].values
    assert 'c' not in result_df['string'].values

    dataset = pq.ParquetDataset(
        base_path, filesystem=fs,
        filters=[('integer', 'in', [1]), ('string', 'in', ('a', 'b')),
                 ('boolean', 'not in', {'False'})],
        use_legacy_dataset=use_legacy_dataset
    )
    table = dataset.read()
    result_df = (table.to_pandas().reset_index(drop=True))

    assert 0 not in result_df['integer'].values
    assert 'c' not in result_df['string'].values
    assert False not in result_df['boolean'].values


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_filters_invalid_pred_op(tempdir, use_legacy_dataset):
    fs = LocalFileSystem._get_instance()
    base_path = tempdir

    integer_keys = [0, 1, 2, 3, 4]
    partition_spec = [
        ['integers', integer_keys],
    ]
    N = 5

    df = pd.DataFrame({
        'index': np.arange(N),
        'integers': np.array(integer_keys, dtype='i4'),
    }, columns=['index', 'integers'])

    _generate_partition_directories(fs, base_path, partition_spec, df)

    with pytest.raises(TypeError):
        pq.ParquetDataset(base_path,
                          filesystem=fs,
                          filters=[('integers', 'in', 3), ],
                          use_legacy_dataset=use_legacy_dataset)

    with pytest.raises(ValueError):
        pq.ParquetDataset(base_path,
                          filesystem=fs,
                          filters=[('integers', '=<', 3), ],
                          use_legacy_dataset=use_legacy_dataset)

    if use_legacy_dataset:
        with pytest.raises(ValueError):
            pq.ParquetDataset(base_path,
                              filesystem=fs,
                              filters=[('integers', 'in', set()), ],
                              use_legacy_dataset=use_legacy_dataset)
    else:
        # Dataset API returns empty table instead
        dataset = pq.ParquetDataset(base_path,
                                    filesystem=fs,
                                    filters=[('integers', 'in', set()), ],
                                    use_legacy_dataset=use_legacy_dataset)
        assert dataset.read().num_rows == 0

    if use_legacy_dataset:
        with pytest.raises(ValueError):
            pq.ParquetDataset(base_path,
                              filesystem=fs,
                              filters=[('integers', '!=', {3})],
                              use_legacy_dataset=use_legacy_dataset)
    else:
        dataset = pq.ParquetDataset(base_path,
                                    filesystem=fs,
                                    filters=[('integers', '!=', {3})],
                                    use_legacy_dataset=use_legacy_dataset)
        with pytest.raises(NotImplementedError):
            assert dataset.read().num_rows == 0


@pytest.mark.pandas
@parametrize_legacy_dataset_fixed
def test_filters_invalid_column(tempdir, use_legacy_dataset):
    # ARROW-5572 - raise error on invalid name in filter specification
    # works with new dataset / xfail with legacy implementation
    fs = LocalFileSystem._get_instance()
    base_path = tempdir

    integer_keys = [0, 1, 2, 3, 4]
    partition_spec = [['integers', integer_keys]]
    N = 5

    df = pd.DataFrame({
        'index': np.arange(N),
        'integers': np.array(integer_keys, dtype='i4'),
    }, columns=['index', 'integers'])

    _generate_partition_directories(fs, base_path, partition_spec, df)

    msg = r"No match for FieldRef.Name\(non_existent_column\)"
    with pytest.raises(ValueError, match=msg):
        pq.ParquetDataset(base_path, filesystem=fs,
                          filters=[('non_existent_column', '<', 3), ],
                          use_legacy_dataset=use_legacy_dataset).read()


@pytest.mark.pandas
@parametrize_legacy_dataset
@pytest.mark.parametrize("filters",
                         ([('integers', '<', 3)],
                          [[('integers', '<', 3)]],
                          pc.field('integers') < 3,
                          pc.field('nested', 'a') < 3,
                          pc.field('nested', 'b').cast(pa.int64()) < 3))
@pytest.mark.parametrize("read_method", ("read_table", "read_pandas"))
def test_filters_read_table(tempdir, use_legacy_dataset, filters, read_method):
    read = getattr(pq, read_method)
    # test that filters keyword is passed through in read_table
    fs = LocalFileSystem._get_instance()
    base_path = tempdir

    integer_keys = [0, 1, 2, 3, 4]
    partition_spec = [
        ['integers', integer_keys],
    ]
    N = len(integer_keys)

    df = pd.DataFrame({
        'index': np.arange(N),
        'integers': np.array(integer_keys, dtype='i4'),
        'nested': np.array([{'a': i, 'b': str(i)} for i in range(N)])
    })

    _generate_partition_directories(fs, base_path, partition_spec, df)

    kwargs = dict(filesystem=fs, filters=filters,
                  use_legacy_dataset=use_legacy_dataset)

    # Using Expression in legacy dataset not supported
    if use_legacy_dataset and isinstance(filters, pc.Expression):
        msg = "Expressions as filter not supported for legacy dataset"
        with pytest.raises(TypeError, match=msg):
            read(base_path, **kwargs)
    else:
        table = read(base_path, **kwargs)
        assert table.num_rows == 3


@pytest.mark.pandas
@parametrize_legacy_dataset_fixed
def test_partition_keys_with_underscores(tempdir, use_legacy_dataset):
    # ARROW-5666 - partition field values with underscores preserve underscores
    # xfail with legacy dataset -> they get interpreted as integers
    fs = LocalFileSystem._get_instance()
    base_path = tempdir

    string_keys = ["2019_2", "2019_3"]
    partition_spec = [
        ['year_week', string_keys],
    ]
    N = 2

    df = pd.DataFrame({
        'index': np.arange(N),
        'year_week': np.array(string_keys, dtype='object'),
    }, columns=['index', 'year_week'])

    _generate_partition_directories(fs, base_path, partition_spec, df)

    dataset = pq.ParquetDataset(
        base_path, use_legacy_dataset=use_legacy_dataset)
    result = dataset.read()
    assert result.column("year_week").to_pylist() == string_keys


@pytest.mark.s3
@parametrize_legacy_dataset
def test_read_s3fs(s3_example_s3fs, use_legacy_dataset):
    fs, path = s3_example_s3fs
    path = path + "/test.parquet"
    table = pa.table({"a": [1, 2, 3]})
    _write_table(table, path, filesystem=fs)

    result = _read_table(
        path, filesystem=fs, use_legacy_dataset=use_legacy_dataset
    )
    assert result.equals(table)


@pytest.mark.s3
@parametrize_legacy_dataset
def test_read_directory_s3fs(s3_example_s3fs, use_legacy_dataset):
    fs, directory = s3_example_s3fs
    path = directory + "/test.parquet"
    table = pa.table({"a": [1, 2, 3]})
    _write_table(table, path, filesystem=fs)

    result = _read_table(
        directory, filesystem=fs, use_legacy_dataset=use_legacy_dataset
    )
    assert result.equals(table)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_read_single_file_list(tempdir, use_legacy_dataset):
    data_path = str(tempdir / 'data.parquet')

    table = pa.table({"a": [1, 2, 3]})
    _write_table(table, data_path)

    result = pq.ParquetDataset(
        [data_path], use_legacy_dataset=use_legacy_dataset
    ).read()
    assert result.equals(table)


@pytest.mark.pandas
@pytest.mark.s3
@parametrize_legacy_dataset
def test_read_partitioned_directory_s3fs_wrapper(
    s3_example_s3fs, use_legacy_dataset
):
    import s3fs

    from pyarrow.filesystem import S3FSWrapper

    if Version(s3fs.__version__) >= Version("0.5"):
        pytest.skip("S3FSWrapper no longer working for s3fs 0.5+")

    fs, path = s3_example_s3fs
    with pytest.warns(FutureWarning):
        wrapper = S3FSWrapper(fs)
    _partition_test_for_filesystem(wrapper, path)

    # Check that we can auto-wrap
    dataset = pq.ParquetDataset(
        path, filesystem=fs, use_legacy_dataset=use_legacy_dataset
    )
    dataset.read()


@pytest.mark.pandas
@pytest.mark.s3
@parametrize_legacy_dataset
def test_read_partitioned_directory_s3fs(s3_example_s3fs, use_legacy_dataset):
    fs, path = s3_example_s3fs
    _partition_test_for_filesystem(
        fs, path, use_legacy_dataset=use_legacy_dataset
    )


def _partition_test_for_filesystem(fs, base_path, use_legacy_dataset=True):
    foo_keys = [0, 1]
    bar_keys = ['a', 'b', 'c']
    partition_spec = [
        ['foo', foo_keys],
        ['bar', bar_keys]
    ]
    N = 30

    df = pd.DataFrame({
        'index': np.arange(N),
        'foo': np.array(foo_keys, dtype='i4').repeat(15),
        'bar': np.tile(np.tile(np.array(bar_keys, dtype=object), 5), 2),
        'values': np.random.randn(N)
    }, columns=['index', 'foo', 'bar', 'values'])

    _generate_partition_directories(fs, base_path, partition_spec, df)

    dataset = pq.ParquetDataset(
        base_path, filesystem=fs, use_legacy_dataset=use_legacy_dataset)
    table = dataset.read()
    result_df = (table.to_pandas()
                 .sort_values(by='index')
                 .reset_index(drop=True))

    expected_df = (df.sort_values(by='index')
                   .reset_index(drop=True)
                   .reindex(columns=result_df.columns))

    expected_df['foo'] = pd.Categorical(df['foo'], categories=foo_keys)
    expected_df['bar'] = pd.Categorical(df['bar'], categories=bar_keys)

    assert (result_df.columns == ['index', 'values', 'foo', 'bar']).all()

    tm.assert_frame_equal(result_df, expected_df)


def _generate_partition_directories(fs, base_dir, partition_spec, df):
    # partition_spec : list of lists, e.g. [['foo', [0, 1, 2],
    #                                       ['bar', ['a', 'b', 'c']]
    # part_table : a pyarrow.Table to write to each partition
    DEPTH = len(partition_spec)

    pathsep = getattr(fs, "pathsep", getattr(fs, "sep", "/"))

    def _visit_level(base_dir, level, part_keys):
        name, values = partition_spec[level]
        for value in values:
            this_part_keys = part_keys + [(name, value)]

            level_dir = pathsep.join([
                str(base_dir),
                '{}={}'.format(name, value)
            ])
            fs.mkdir(level_dir)

            if level == DEPTH - 1:
                # Generate example data
                file_path = pathsep.join([level_dir, guid()])
                filtered_df = _filter_partition(df, this_part_keys)
                part_table = pa.Table.from_pandas(filtered_df)
                with fs.open(file_path, 'wb') as f:
                    _write_table(part_table, f)
                assert fs.exists(file_path)

                file_success = pathsep.join([level_dir, '_SUCCESS'])
                with fs.open(file_success, 'wb') as f:
                    pass
            else:
                _visit_level(level_dir, level + 1, this_part_keys)
                file_success = pathsep.join([level_dir, '_SUCCESS'])
                with fs.open(file_success, 'wb') as f:
                    pass

    _visit_level(base_dir, 0, [])


def _test_read_common_metadata_files(fs, base_path):
    import pandas as pd

    import pyarrow.parquet as pq

    N = 100
    df = pd.DataFrame({
        'index': np.arange(N),
        'values': np.random.randn(N)
    }, columns=['index', 'values'])

    base_path = str(base_path)
    data_path = os.path.join(base_path, 'data.parquet')

    table = pa.Table.from_pandas(df)

    with fs.open(data_path, 'wb') as f:
        _write_table(table, f)

    metadata_path = os.path.join(base_path, '_common_metadata')
    with fs.open(metadata_path, 'wb') as f:
        pq.write_metadata(table.schema, f)

    dataset = pq.ParquetDataset(base_path, filesystem=fs,
                                use_legacy_dataset=True)
    with pytest.warns(FutureWarning):
        assert dataset.common_metadata_path == str(metadata_path)

    with fs.open(data_path) as f:
        common_schema = pq.read_metadata(f).schema
    assert dataset.schema.equals(common_schema)

    # handle list of one directory
    dataset2 = pq.ParquetDataset([base_path], filesystem=fs,
                                 use_legacy_dataset=True)
    assert dataset2.schema.equals(dataset.schema)


@pytest.mark.pandas
@pytest.mark.filterwarnings("ignore:'ParquetDataset.schema:FutureWarning")
def test_read_common_metadata_files(tempdir):
    fs = LocalFileSystem._get_instance()
    _test_read_common_metadata_files(fs, tempdir)


@pytest.mark.pandas
@pytest.mark.filterwarnings("ignore:'ParquetDataset.schema:FutureWarning")
def test_read_metadata_files(tempdir):
    fs = LocalFileSystem._get_instance()

    N = 100
    df = pd.DataFrame({
        'index': np.arange(N),
        'values': np.random.randn(N)
    }, columns=['index', 'values'])

    data_path = tempdir / 'data.parquet'

    table = pa.Table.from_pandas(df)

    with fs.open(data_path, 'wb') as f:
        _write_table(table, f)

    metadata_path = tempdir / '_metadata'
    with fs.open(metadata_path, 'wb') as f:
        pq.write_metadata(table.schema, f)

    dataset = pq.ParquetDataset(tempdir, filesystem=fs,
                                use_legacy_dataset=True)
    with pytest.warns(FutureWarning):
        assert dataset.metadata_path == str(metadata_path)

    with fs.open(data_path) as f:
        metadata_schema = pq.read_metadata(f).schema
    assert dataset.schema.equals(metadata_schema)


def _filter_partition(df, part_keys):
    predicate = np.ones(len(df), dtype=bool)

    to_drop = []
    for name, value in part_keys:
        to_drop.append(name)

        # to avoid pandas warning
        if isinstance(value, (datetime.date, datetime.datetime)):
            value = pd.Timestamp(value)

        predicate &= df[name] == value

    return df[predicate].drop(to_drop, axis=1)


@parametrize_legacy_dataset
@pytest.mark.pandas
def test_filter_before_validate_schema(tempdir, use_legacy_dataset):
    # ARROW-4076 apply filter before schema validation
    # to avoid checking unneeded schemas

    # create partitioned dataset with mismatching schemas which would
    # otherwise raise if first validation all schemas
    dir1 = tempdir / 'A=0'
    dir1.mkdir()
    table1 = pa.Table.from_pandas(pd.DataFrame({'B': [1, 2, 3]}))
    pq.write_table(table1, dir1 / 'data.parquet')

    dir2 = tempdir / 'A=1'
    dir2.mkdir()
    table2 = pa.Table.from_pandas(pd.DataFrame({'B': ['a', 'b', 'c']}))
    pq.write_table(table2, dir2 / 'data.parquet')

    # read single file using filter
    table = pq.read_table(tempdir, filters=[[('A', '==', 0)]],
                          use_legacy_dataset=use_legacy_dataset)
    assert table.column('B').equals(pa.chunked_array([[1, 2, 3]]))


@pytest.mark.pandas
@pytest.mark.filterwarnings(
    "ignore:Specifying the 'metadata':FutureWarning")
@parametrize_legacy_dataset
def test_read_multiple_files(tempdir, use_legacy_dataset):
    nfiles = 10
    size = 5

    dirpath = tempdir / guid()
    dirpath.mkdir()

    test_data = []
    paths = []
    for i in range(nfiles):
        df = _test_dataframe(size, seed=i)

        # Hack so that we don't have a dtype cast in v1 files
        df['uint32'] = df['uint32'].astype(np.int64)

        path = dirpath / '{}.parquet'.format(i)

        table = pa.Table.from_pandas(df)
        _write_table(table, path)

        test_data.append(table)
        paths.append(path)

    # Write a _SUCCESS.crc file
    (dirpath / '_SUCCESS.crc').touch()

    def read_multiple_files(paths, columns=None, use_threads=True, **kwargs):
        dataset = pq.ParquetDataset(
            paths, use_legacy_dataset=use_legacy_dataset, **kwargs)
        return dataset.read(columns=columns, use_threads=use_threads)

    result = read_multiple_files(paths)
    expected = pa.concat_tables(test_data)

    assert result.equals(expected)

    # Read with provided metadata
    # TODO(dataset) specifying metadata not yet supported
    metadata = pq.read_metadata(paths[0])
    if use_legacy_dataset:
        result2 = read_multiple_files(paths, metadata=metadata)
        assert result2.equals(expected)

        with pytest.warns(FutureWarning, match="Specifying the 'schema'"):
            result3 = pq.ParquetDataset(dirpath, schema=metadata.schema,
                                        use_legacy_dataset=True).read()
        assert result3.equals(expected)
    else:
        with pytest.raises(ValueError, match="no longer supported"):
            pq.read_table(paths, metadata=metadata, use_legacy_dataset=False)

    # Read column subset
    to_read = [0, 2, 6, result.num_columns - 1]

    col_names = [result.field(i).name for i in to_read]
    out = pq.read_table(
        dirpath, columns=col_names, use_legacy_dataset=use_legacy_dataset
    )
    expected = pa.Table.from_arrays([result.column(i) for i in to_read],
                                    names=col_names,
                                    metadata=result.schema.metadata)
    assert out.equals(expected)

    # Read with multiple threads
    pq.read_table(
        dirpath, use_threads=True, use_legacy_dataset=use_legacy_dataset
    )

    # Test failure modes with non-uniform metadata
    bad_apple = _test_dataframe(size, seed=i).iloc[:, :4]
    bad_apple_path = tempdir / '{}.parquet'.format(guid())

    t = pa.Table.from_pandas(bad_apple)
    _write_table(t, bad_apple_path)

    if not use_legacy_dataset:
        # TODO(dataset) Dataset API skips bad files
        return

    bad_meta = pq.read_metadata(bad_apple_path)

    with pytest.raises(ValueError):
        read_multiple_files(paths + [bad_apple_path])

    with pytest.raises(ValueError):
        read_multiple_files(paths, metadata=bad_meta)

    mixed_paths = [bad_apple_path, paths[0]]

    with pytest.raises(ValueError):
        with pytest.warns(FutureWarning, match="Specifying the 'schema'"):
            read_multiple_files(mixed_paths, schema=bad_meta.schema)

    with pytest.raises(ValueError):
        read_multiple_files(mixed_paths)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_dataset_read_pandas(tempdir, use_legacy_dataset):
    nfiles = 5
    size = 5

    dirpath = tempdir / guid()
    dirpath.mkdir()

    test_data = []
    frames = []
    paths = []
    for i in range(nfiles):
        df = _test_dataframe(size, seed=i)
        df.index = np.arange(i * size, (i + 1) * size)
        df.index.name = 'index'

        path = dirpath / '{}.parquet'.format(i)

        table = pa.Table.from_pandas(df)
        _write_table(table, path)
        test_data.append(table)
        frames.append(df)
        paths.append(path)

    dataset = pq.ParquetDataset(dirpath, use_legacy_dataset=use_legacy_dataset)
    columns = ['uint8', 'strings']
    result = dataset.read_pandas(columns=columns).to_pandas()
    expected = pd.concat([x[columns] for x in frames])

    tm.assert_frame_equal(result, expected)

    # also be able to pass the columns as a set (ARROW-12314)
    result = dataset.read_pandas(columns=set(columns)).to_pandas()
    assert result.shape == expected.shape
    # column order can be different because of using a set
    tm.assert_frame_equal(result.reindex(columns=expected.columns), expected)


@pytest.mark.filterwarnings("ignore:'ParquetDataset:FutureWarning")
@pytest.mark.pandas
@parametrize_legacy_dataset
def test_dataset_memory_map(tempdir, use_legacy_dataset):
    # ARROW-2627: Check that we can use ParquetDataset with memory-mapping
    dirpath = tempdir / guid()
    dirpath.mkdir()

    df = _test_dataframe(10, seed=0)
    path = dirpath / '{}.parquet'.format(0)
    table = pa.Table.from_pandas(df)
    _write_table(table, path, version='2.6')

    dataset = pq.ParquetDataset(
        dirpath, memory_map=True, use_legacy_dataset=use_legacy_dataset)
    assert dataset.read().equals(table)
    if use_legacy_dataset:
        assert dataset.pieces[0].read().equals(table)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_dataset_enable_buffered_stream(tempdir, use_legacy_dataset):
    dirpath = tempdir / guid()
    dirpath.mkdir()

    df = _test_dataframe(10, seed=0)
    path = dirpath / '{}.parquet'.format(0)
    table = pa.Table.from_pandas(df)
    _write_table(table, path, version='2.6')

    with pytest.raises(ValueError):
        pq.ParquetDataset(
            dirpath, buffer_size=-64,
            use_legacy_dataset=use_legacy_dataset)

    for buffer_size in [128, 1024]:
        dataset = pq.ParquetDataset(
            dirpath, buffer_size=buffer_size,
            use_legacy_dataset=use_legacy_dataset)
        assert dataset.read().equals(table)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_dataset_enable_pre_buffer(tempdir, use_legacy_dataset):
    dirpath = tempdir / guid()
    dirpath.mkdir()

    df = _test_dataframe(10, seed=0)
    path = dirpath / '{}.parquet'.format(0)
    table = pa.Table.from_pandas(df)
    _write_table(table, path, version='2.6')

    for pre_buffer in (True, False):
        dataset = pq.ParquetDataset(
            dirpath, pre_buffer=pre_buffer,
            use_legacy_dataset=use_legacy_dataset)
        assert dataset.read().equals(table)
        actual = pq.read_table(dirpath, pre_buffer=pre_buffer,
                               use_legacy_dataset=use_legacy_dataset)
        assert actual.equals(table)


def _make_example_multifile_dataset(base_path, nfiles=10, file_nrows=5):
    test_data = []
    paths = []
    for i in range(nfiles):
        df = _test_dataframe(file_nrows, seed=i)
        path = base_path / '{}.parquet'.format(i)

        test_data.append(_write_table(df, path))
        paths.append(path)
    return paths


def _assert_dataset_paths(dataset, paths, use_legacy_dataset):
    if use_legacy_dataset:
        assert set(map(str, paths)) == {x.path for x in dataset._pieces}
    else:
        paths = [str(path.as_posix()) for path in paths]
        assert set(paths) == set(dataset._dataset.files)


@pytest.mark.pandas
@parametrize_legacy_dataset
@pytest.mark.parametrize('dir_prefix', ['_', '.'])
def test_ignore_private_directories(tempdir, dir_prefix, use_legacy_dataset):
    dirpath = tempdir / guid()
    dirpath.mkdir()

    paths = _make_example_multifile_dataset(dirpath, nfiles=10,
                                            file_nrows=5)

    # private directory
    (dirpath / '{}staging'.format(dir_prefix)).mkdir()

    dataset = pq.ParquetDataset(dirpath, use_legacy_dataset=use_legacy_dataset)

    _assert_dataset_paths(dataset, paths, use_legacy_dataset)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_ignore_hidden_files_dot(tempdir, use_legacy_dataset):
    dirpath = tempdir / guid()
    dirpath.mkdir()

    paths = _make_example_multifile_dataset(dirpath, nfiles=10,
                                            file_nrows=5)

    with (dirpath / '.DS_Store').open('wb') as f:
        f.write(b'gibberish')

    with (dirpath / '.private').open('wb') as f:
        f.write(b'gibberish')

    dataset = pq.ParquetDataset(dirpath, use_legacy_dataset=use_legacy_dataset)

    _assert_dataset_paths(dataset, paths, use_legacy_dataset)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_ignore_hidden_files_underscore(tempdir, use_legacy_dataset):
    dirpath = tempdir / guid()
    dirpath.mkdir()

    paths = _make_example_multifile_dataset(dirpath, nfiles=10,
                                            file_nrows=5)

    with (dirpath / '_committed_123').open('wb') as f:
        f.write(b'abcd')

    with (dirpath / '_started_321').open('wb') as f:
        f.write(b'abcd')

    dataset = pq.ParquetDataset(dirpath, use_legacy_dataset=use_legacy_dataset)

    _assert_dataset_paths(dataset, paths, use_legacy_dataset)


@pytest.mark.pandas
@parametrize_legacy_dataset
@pytest.mark.parametrize('dir_prefix', ['_', '.'])
def test_ignore_no_private_directories_in_base_path(
    tempdir, dir_prefix, use_legacy_dataset
):
    # ARROW-8427 - don't ignore explicitly listed files if parent directory
    # is a private directory
    dirpath = tempdir / "{0}data".format(dir_prefix) / guid()
    dirpath.mkdir(parents=True)

    paths = _make_example_multifile_dataset(dirpath, nfiles=10,
                                            file_nrows=5)

    dataset = pq.ParquetDataset(paths, use_legacy_dataset=use_legacy_dataset)
    _assert_dataset_paths(dataset, paths, use_legacy_dataset)

    # ARROW-9644 - don't ignore full directory with underscore in base path
    dataset = pq.ParquetDataset(dirpath, use_legacy_dataset=use_legacy_dataset)
    _assert_dataset_paths(dataset, paths, use_legacy_dataset)


@pytest.mark.pandas
@parametrize_legacy_dataset_fixed
def test_ignore_custom_prefixes(tempdir, use_legacy_dataset):
    # ARROW-9573 - allow override of default ignore_prefixes
    part = ["xxx"] * 3 + ["yyy"] * 3
    table = pa.table([
        pa.array(range(len(part))),
        pa.array(part).dictionary_encode(),
    ], names=['index', '_part'])

    # TODO use_legacy_dataset ARROW-10247
    pq.write_to_dataset(table, str(tempdir), partition_cols=['_part'])

    private_duplicate = tempdir / '_private_duplicate'
    private_duplicate.mkdir()
    pq.write_to_dataset(table, str(private_duplicate),
                        partition_cols=['_part'])

    read = pq.read_table(
        tempdir, use_legacy_dataset=use_legacy_dataset,
        ignore_prefixes=['_private'])

    assert read.equals(table)


@parametrize_legacy_dataset_fixed
def test_empty_directory(tempdir, use_legacy_dataset):
    # ARROW-5310 - reading empty directory
    # fails with legacy implementation
    empty_dir = tempdir / 'dataset'
    empty_dir.mkdir()

    dataset = pq.ParquetDataset(
        empty_dir, use_legacy_dataset=use_legacy_dataset)
    result = dataset.read()
    assert result.num_rows == 0
    assert result.num_columns == 0


@pytest.mark.filterwarnings("ignore:'ParquetDataset.schema:FutureWarning")
def _test_write_to_dataset_with_partitions(base_path,
                                           use_legacy_dataset=True,
                                           filesystem=None,
                                           schema=None,
                                           index_name=None):
    import pandas as pd
    import pandas.testing as tm

    import pyarrow.parquet as pq

    # ARROW-1400
    output_df = pd.DataFrame({'group1': list('aaabbbbccc'),
                              'group2': list('eefeffgeee'),
                              'num': list(range(10)),
                              'nan': [np.nan] * 10,
                              'date': np.arange('2017-01-01', '2017-01-11',
                                                dtype='datetime64[D]')})
    output_df["date"] = output_df["date"].astype('datetime64[ns]')
    cols = output_df.columns.tolist()
    partition_by = ['group1', 'group2']
    output_table = pa.Table.from_pandas(output_df, schema=schema, safe=False,
                                        preserve_index=False)
    pq.write_to_dataset(output_table, base_path, partition_by,
                        filesystem=filesystem,
                        use_legacy_dataset=use_legacy_dataset)

    metadata_path = os.path.join(str(base_path), '_common_metadata')

    if filesystem is not None:
        with filesystem.open(metadata_path, 'wb') as f:
            pq.write_metadata(output_table.schema, f)
    else:
        pq.write_metadata(output_table.schema, metadata_path)

    # ARROW-2891: Ensure the output_schema is preserved when writing a
    # partitioned dataset
    dataset = pq.ParquetDataset(base_path,
                                filesystem=filesystem,
                                validate_schema=True,
                                use_legacy_dataset=use_legacy_dataset)
    # ARROW-2209: Ensure the dataset schema also includes the partition columns
    if use_legacy_dataset:
        with pytest.warns(FutureWarning, match="'ParquetDataset.schema'"):
            dataset_cols = set(dataset.schema.to_arrow_schema().names)
    else:
        # NB schema property is an arrow and not parquet schema
        dataset_cols = set(dataset.schema.names)

    assert dataset_cols == set(output_table.schema.names)

    input_table = dataset.read()
    input_df = input_table.to_pandas()

    # Read data back in and compare with original DataFrame
    # Partitioned columns added to the end of the DataFrame when read
    input_df_cols = input_df.columns.tolist()
    assert partition_by == input_df_cols[-1 * len(partition_by):]

    input_df = input_df[cols]
    # Partitioned columns become 'categorical' dtypes
    for col in partition_by:
        output_df[col] = output_df[col].astype('category')
    tm.assert_frame_equal(output_df, input_df)


def _test_write_to_dataset_no_partitions(base_path,
                                         use_legacy_dataset=True,
                                         filesystem=None):
    import pandas as pd

    import pyarrow.parquet as pq

    # ARROW-1400
    output_df = pd.DataFrame({'group1': list('aaabbbbccc'),
                              'group2': list('eefeffgeee'),
                              'num': list(range(10)),
                              'date': np.arange('2017-01-01', '2017-01-11',
                                                dtype='datetime64[D]')})
    output_df["date"] = output_df["date"].astype('datetime64[ns]')
    cols = output_df.columns.tolist()
    output_table = pa.Table.from_pandas(output_df)

    if filesystem is None:
        filesystem = LocalFileSystem._get_instance()

    # Without partitions, append files to root_path
    n = 5
    for i in range(n):
        pq.write_to_dataset(output_table, base_path,
                            use_legacy_dataset=use_legacy_dataset,
                            filesystem=filesystem)
    output_files = [file for file in filesystem.ls(str(base_path))
                    if file.endswith(".parquet")]
    assert len(output_files) == n

    # Deduplicated incoming DataFrame should match
    # original outgoing Dataframe
    input_table = pq.ParquetDataset(
        base_path, filesystem=filesystem,
        use_legacy_dataset=use_legacy_dataset
    ).read()
    input_df = input_table.to_pandas()
    input_df = input_df.drop_duplicates()
    input_df = input_df[cols]
    assert output_df.equals(input_df)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_write_to_dataset_with_partitions(tempdir, use_legacy_dataset):
    _test_write_to_dataset_with_partitions(str(tempdir), use_legacy_dataset)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_write_to_dataset_with_partitions_and_schema(
    tempdir, use_legacy_dataset
):
    schema = pa.schema([pa.field('group1', type=pa.string()),
                        pa.field('group2', type=pa.string()),
                        pa.field('num', type=pa.int64()),
                        pa.field('nan', type=pa.int32()),
                        pa.field('date', type=pa.timestamp(unit='us'))])
    _test_write_to_dataset_with_partitions(
        str(tempdir), use_legacy_dataset, schema=schema)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_write_to_dataset_with_partitions_and_index_name(
    tempdir, use_legacy_dataset
):
    _test_write_to_dataset_with_partitions(
        str(tempdir), use_legacy_dataset, index_name='index_name')


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_write_to_dataset_no_partitions(tempdir, use_legacy_dataset):
    _test_write_to_dataset_no_partitions(str(tempdir), use_legacy_dataset)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_write_to_dataset_pathlib(tempdir, use_legacy_dataset):
    _test_write_to_dataset_with_partitions(
        tempdir / "test1", use_legacy_dataset)
    _test_write_to_dataset_no_partitions(
        tempdir / "test2", use_legacy_dataset)


@pytest.mark.pandas
@pytest.mark.s3
@parametrize_legacy_dataset
def test_write_to_dataset_pathlib_nonlocal(
    tempdir, s3_example_s3fs, use_legacy_dataset
):
    # pathlib paths are only accepted for local files
    fs, _ = s3_example_s3fs

    with pytest.raises(TypeError, match="path-like objects are only allowed"):
        _test_write_to_dataset_with_partitions(
            tempdir / "test1", use_legacy_dataset, filesystem=fs)

    with pytest.raises(TypeError, match="path-like objects are only allowed"):
        _test_write_to_dataset_no_partitions(
            tempdir / "test2", use_legacy_dataset, filesystem=fs)


@pytest.mark.pandas
@pytest.mark.s3
@parametrize_legacy_dataset
def test_write_to_dataset_with_partitions_s3fs(
    s3_example_s3fs, use_legacy_dataset
):
    fs, path = s3_example_s3fs

    _test_write_to_dataset_with_partitions(
        path, use_legacy_dataset, filesystem=fs)


@pytest.mark.pandas
@pytest.mark.s3
@parametrize_legacy_dataset
def test_write_to_dataset_no_partitions_s3fs(
    s3_example_s3fs, use_legacy_dataset
):
    fs, path = s3_example_s3fs

    _test_write_to_dataset_no_partitions(
        path, use_legacy_dataset, filesystem=fs)


@pytest.mark.filterwarnings(
    "ignore:'ParquetDataset:FutureWarning",
    "ignore:'partition_filename_cb':FutureWarning")
@pytest.mark.pandas
@parametrize_legacy_dataset_not_supported
def test_write_to_dataset_with_partitions_and_custom_filenames(
    tempdir, use_legacy_dataset
):
    output_df = pd.DataFrame({'group1': list('aaabbbbccc'),
                              'group2': list('eefeffgeee'),
                              'num': list(range(10)),
                              'nan': [np.nan] * 10,
                              'date': np.arange('2017-01-01', '2017-01-11',
                                                dtype='datetime64[D]')})
    output_df["date"] = output_df["date"].astype('datetime64[ns]')
    partition_by = ['group1', 'group2']
    output_table = pa.Table.from_pandas(output_df)
    path = str(tempdir)

    def partition_filename_callback(keys):
        return "{}-{}.parquet".format(*keys)

    pq.write_to_dataset(output_table, path,
                        partition_by, partition_filename_callback,
                        use_legacy_dataset=use_legacy_dataset)

    dataset = pq.ParquetDataset(path)

    # ARROW-3538: Ensure partition filenames match the given pattern
    # defined in the local function partition_filename_callback
    expected_basenames = [
        'a-e.parquet', 'a-f.parquet',
        'b-e.parquet', 'b-f.parquet',
        'b-g.parquet', 'c-e.parquet'
    ]
    output_basenames = [os.path.basename(p.path) for p in dataset.pieces]

    assert sorted(expected_basenames) == sorted(output_basenames)


@pytest.mark.dataset
@pytest.mark.pandas
def test_write_to_dataset_filesystem(tempdir):
    df = pd.DataFrame({'A': [1, 2, 3]})
    table = pa.Table.from_pandas(df)
    path = str(tempdir)

    pq.write_to_dataset(table, path, filesystem=fs.LocalFileSystem())
    result = pq.read_table(path)
    assert result.equals(table)


def _make_dataset_for_pickling(tempdir, use_legacy_dataset=False, N=100):
    path = tempdir / 'data.parquet'
    fs = LocalFileSystem._get_instance()

    df = pd.DataFrame({
        'index': np.arange(N),
        'values': np.random.randn(N)
    }, columns=['index', 'values'])
    table = pa.Table.from_pandas(df)

    num_groups = 3
    with pq.ParquetWriter(path, table.schema) as writer:
        for i in range(num_groups):
            writer.write_table(table)

    reader = pq.ParquetFile(path)
    assert reader.metadata.num_row_groups == num_groups

    metadata_path = tempdir / '_metadata'
    with fs.open(metadata_path, 'wb') as f:
        pq.write_metadata(table.schema, f)

    dataset = pq.ParquetDataset(
        tempdir, filesystem=fs, use_legacy_dataset=use_legacy_dataset)
    if use_legacy_dataset:
        with pytest.warns(FutureWarning):
            assert dataset.metadata_path == str(metadata_path)

    return dataset


def _assert_dataset_is_picklable(dataset, pickler, use_legacy_dataset=False):
    def is_pickleable(obj):
        return obj == pickler.loads(pickler.dumps(obj))

    assert is_pickleable(dataset)
    if use_legacy_dataset:
        with pytest.warns(FutureWarning):
            metadata = dataset.metadata
        assert is_pickleable(metadata)
        assert is_pickleable(metadata.schema)
        assert len(metadata.schema)
        for column in metadata.schema:
            assert is_pickleable(column)

        for piece in dataset._pieces:
            assert is_pickleable(piece)
            metadata = piece.get_metadata()
            assert metadata.num_row_groups
            for i in range(metadata.num_row_groups):
                assert is_pickleable(metadata.row_group(i))


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_builtin_pickle_dataset(tempdir, datadir, use_legacy_dataset):
    import pickle
    dataset = _make_dataset_for_pickling(tempdir, use_legacy_dataset)
    _assert_dataset_is_picklable(
        dataset, pickler=pickle, use_legacy_dataset=use_legacy_dataset)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_cloudpickle_dataset(tempdir, datadir, use_legacy_dataset):
    cp = pytest.importorskip('cloudpickle')
    dataset = _make_dataset_for_pickling(tempdir, use_legacy_dataset)
    _assert_dataset_is_picklable(
        dataset, pickler=cp, use_legacy_dataset=use_legacy_dataset)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_partitioned_dataset(tempdir, use_legacy_dataset):
    # ARROW-3208: Segmentation fault when reading a Parquet partitioned dataset
    # to a Parquet file
    path = tempdir / "ARROW-3208"
    df = pd.DataFrame({
        'one': [-1, 10, 2.5, 100, 1000, 1, 29.2],
        'two': [-1, 10, 2, 100, 1000, 1, 11],
        'three': [0, 0, 0, 0, 0, 0, 0]
    })
    table = pa.Table.from_pandas(df)
    pq.write_to_dataset(table, root_path=str(path),
                        partition_cols=['one', 'two'],
                        use_legacy_dataset=use_legacy_dataset)
    table = pq.ParquetDataset(
        path, use_legacy_dataset=use_legacy_dataset).read()
    pq.write_table(table, path / "output.parquet")


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_dataset_read_dictionary(tempdir, use_legacy_dataset):
    path = tempdir / "ARROW-3325-dataset"
    t1 = pa.table([[util.rands(10) for i in range(5)] * 10], names=['f0'])
    t2 = pa.table([[util.rands(10) for i in range(5)] * 10], names=['f0'])
    pq.write_to_dataset(t1, root_path=str(path),
                        use_legacy_dataset=use_legacy_dataset)
    pq.write_to_dataset(t2, root_path=str(path),
                        use_legacy_dataset=use_legacy_dataset)

    result = pq.ParquetDataset(
        path, read_dictionary=['f0'],
        use_legacy_dataset=use_legacy_dataset).read()

    # The order of the chunks is non-deterministic
    ex_chunks = [t1[0].chunk(0).dictionary_encode(),
                 t2[0].chunk(0).dictionary_encode()]

    assert result[0].num_chunks == 2
    c0, c1 = result[0].chunk(0), result[0].chunk(1)
    if c0.equals(ex_chunks[0]):
        assert c1.equals(ex_chunks[1])
    else:
        assert c0.equals(ex_chunks[1])
        assert c1.equals(ex_chunks[0])


@pytest.mark.dataset
@pytest.mark.pandas
@pytest.mark.filterwarnings("ignore:Passing 'use_legacy:FutureWarning")
def test_read_table_schema(tempdir):
    # test that schema keyword is passed through in read_table
    table = pa.table({'a': pa.array([1, 2, 3], pa.int32())})
    pq.write_table(table, tempdir / "data1.parquet")
    pq.write_table(table, tempdir / "data2.parquet")

    schema = pa.schema([('a', 'int64')])

    # reading single file (which is special cased in the code)
    result = pq.read_table(tempdir / "data1.parquet", schema=schema)
    expected = pa.table({'a': [1, 2, 3]}, schema=schema)
    assert result.equals(expected)

    # reading multiple fiels
    result = pq.read_table(tempdir, schema=schema)
    expected = pa.table({'a': [1, 2, 3, 1, 2, 3]}, schema=schema)
    assert result.equals(expected)

    # don't allow it with the legacy reader
    with pytest.raises(
        ValueError, match="The 'schema' argument is only supported"
    ):
        pq.read_table(tempdir / "data.parquet", schema=schema,
                      use_legacy_dataset=True)

    # using ParquetDataset directory with non-legacy implementation
    result = pq.ParquetDataset(
        tempdir, schema=schema, use_legacy_dataset=False
    )
    expected = pa.table({'a': [1, 2, 3, 1, 2, 3]}, schema=schema)
    assert result.read().equals(expected)


@pytest.mark.dataset
def test_dataset_unsupported_keywords():

    with pytest.raises(ValueError, match="not yet supported with the new"):
        pq.ParquetDataset("", use_legacy_dataset=False, metadata=pa.schema([]))

    with pytest.raises(ValueError, match="not yet supported with the new"):
        pq.ParquetDataset("", use_legacy_dataset=False, validate_schema=False)

    with pytest.raises(ValueError, match="not yet supported with the new"):
        pq.ParquetDataset("", use_legacy_dataset=False, split_row_groups=True)

    with pytest.raises(ValueError, match="not yet supported with the new"):
        pq.ParquetDataset("", use_legacy_dataset=False, metadata_nthreads=4)

    with pytest.raises(ValueError, match="no longer supported"):
        pq.read_table("", use_legacy_dataset=False, metadata=pa.schema([]))


@pytest.mark.dataset
@pytest.mark.filterwarnings("ignore:Passing 'use_legacy:FutureWarning")
def test_dataset_partitioning(tempdir):
    import pyarrow.dataset as ds

    # create small dataset with directory partitioning
    root_path = tempdir / "test_partitioning"
    (root_path / "2012" / "10" / "01").mkdir(parents=True)

    table = pa.table({'a': [1, 2, 3]})
    pq.write_table(
        table, str(root_path / "2012" / "10" / "01" / "data.parquet"))

    # This works with new dataset API

    # read_table
    part = ds.partitioning(field_names=["year", "month", "day"])
    result = pq.read_table(
        str(root_path), partitioning=part, use_legacy_dataset=False)
    assert result.column_names == ["a", "year", "month", "day"]

    result = pq.ParquetDataset(
        str(root_path), partitioning=part, use_legacy_dataset=False).read()
    assert result.column_names == ["a", "year", "month", "day"]

    # This raises an error for legacy dataset
    with pytest.raises(ValueError):
        pq.read_table(
            str(root_path), partitioning=part, use_legacy_dataset=True)

    with pytest.raises(ValueError):
        pq.ParquetDataset(
            str(root_path), partitioning=part, use_legacy_dataset=True)


@pytest.mark.dataset
def test_parquet_dataset_new_filesystem(tempdir):
    # Ensure we can pass new FileSystem object to ParquetDataset
    # (use new implementation automatically without specifying
    #  use_legacy_dataset=False)
    table = pa.table({'a': [1, 2, 3]})
    pq.write_table(table, tempdir / 'data.parquet')
    # don't use simple LocalFileSystem (as that gets mapped to legacy one)
    filesystem = fs.SubTreeFileSystem(str(tempdir), fs.LocalFileSystem())
    dataset = pq.ParquetDataset('.', filesystem=filesystem)
    result = dataset.read()
    assert result.equals(table)


@pytest.mark.filterwarnings("ignore:'ParquetDataset:FutureWarning")
@parametrize_legacy_dataset
def test_parquet_dataset_partitions_piece_path_with_fsspec(
    tempdir, use_legacy_dataset
):
    # ARROW-10462 ensure that on Windows we properly use posix-style paths
    # as used by fsspec
    fsspec = pytest.importorskip("fsspec")
    filesystem = fsspec.filesystem('file')
    table = pa.table({'a': [1, 2, 3]})
    pq.write_table(table, tempdir / 'data.parquet')

    # pass a posix-style path (using "/" also on Windows)
    path = str(tempdir).replace("\\", "/")
    dataset = pq.ParquetDataset(
        path, filesystem=filesystem, use_legacy_dataset=use_legacy_dataset)
    # ensure the piece path is also posix-style
    expected = path + "/data.parquet"
    assert dataset.pieces[0].path == expected


@pytest.mark.dataset
def test_parquet_dataset_deprecated_properties(tempdir):
    table = pa.table({'a': [1, 2, 3]})
    path = tempdir / 'data.parquet'
    pq.write_table(table, path)
    dataset = pq.ParquetDataset(path, use_legacy_dataset=True)

    with pytest.warns(FutureWarning, match="'ParquetDataset.pieces"):
        dataset.pieces

    with pytest.warns(FutureWarning, match="'ParquetDataset.partitions"):
        dataset.partitions

    with pytest.warns(FutureWarning, match="'ParquetDataset.memory_map"):
        dataset.memory_map

    with pytest.warns(FutureWarning, match="'ParquetDataset.read_dictio"):
        dataset.read_dictionary

    with pytest.warns(FutureWarning, match="'ParquetDataset.buffer_size"):
        dataset.buffer_size

    with pytest.warns(FutureWarning, match="'ParquetDataset.fs"):
        dataset.fs

    with pytest.warns(FutureWarning, match="'ParquetDataset.schema'"):
        dataset.schema

    with pytest.warns(FutureWarning, match="'ParquetDataset.common_metadata'"):
        dataset.common_metadata

    with pytest.warns(FutureWarning, match="'ParquetDataset.metadata"):
        dataset.metadata

    with pytest.warns(FutureWarning, match="'ParquetDataset.metadata_path"):
        dataset.metadata_path

    with pytest.warns(FutureWarning,
                      match="'ParquetDataset.common_metadata_path"):
        dataset.common_metadata_path

    dataset2 = pq.ParquetDataset(path, use_legacy_dataset=False)

    with pytest.warns(FutureWarning, match="'ParquetDataset.pieces"):
        dataset2.pieces


@pytest.mark.dataset
def test_parquet_write_to_dataset_deprecated_properties(tempdir):
    table = pa.table({'a': [1, 2, 3]})
    path = tempdir / 'data.parquet'

    with pytest.warns(FutureWarning,
                      match="Passing 'use_legacy_dataset=True'"):
        pq.write_to_dataset(table, path, use_legacy_dataset=True)

    # check also that legacy implementation is set when
    # partition_filename_cb is specified
    with pytest.warns(FutureWarning,
                      match="Passing 'use_legacy_dataset=True'"):
        pq.write_to_dataset(table, path,
                            partition_filename_cb=lambda x: 'filename.parquet')


@pytest.mark.dataset
def test_parquet_write_to_dataset_unsupported_keywards_in_legacy(tempdir):
    table = pa.table({'a': [1, 2, 3]})
    path = tempdir / 'data.parquet'

    with pytest.raises(ValueError, match="schema"):
        pq.write_to_dataset(table, path, use_legacy_dataset=True,
                            schema=pa.schema([
                                ('a', pa.int32())
                            ]))

    with pytest.raises(ValueError, match="partitioning"):
        pq.write_to_dataset(table, path, use_legacy_dataset=True,
                            partitioning=["a"])

    with pytest.raises(ValueError, match="use_threads"):
        pq.write_to_dataset(table, path, use_legacy_dataset=True,
                            use_threads=False)

    with pytest.raises(ValueError, match="file_visitor"):
        pq.write_to_dataset(table, path, use_legacy_dataset=True,
                            file_visitor=lambda x: x)

    with pytest.raises(ValueError, match="existing_data_behavior"):
        pq.write_to_dataset(table, path, use_legacy_dataset=True,
                            existing_data_behavior='error')

    with pytest.raises(ValueError, match="basename_template"):
        pq.write_to_dataset(table, path, use_legacy_dataset=True,
                            basename_template='part-{i}.parquet')


@pytest.mark.dataset
def test_parquet_write_to_dataset_exposed_keywords(tempdir):
    table = pa.table({'a': [1, 2, 3]})
    path = tempdir / 'partitioning'

    paths_written = []

    def file_visitor(written_file):
        paths_written.append(written_file.path)

    basename_template = 'part-{i}.parquet'

    pq.write_to_dataset(table, path, partitioning=["a"],
                        file_visitor=file_visitor,
                        basename_template=basename_template,
                        use_legacy_dataset=False)

    expected_paths = {
        path / '1' / 'part-0.parquet',
        path / '2' / 'part-0.parquet',
        path / '3' / 'part-0.parquet'
    }
    paths_written_set = set(map(pathlib.Path, paths_written))
    assert paths_written_set == expected_paths


@pytest.mark.dataset
def test_write_to_dataset_conflicting_keywords(tempdir):
    table = pa.table({'a': [1, 2, 3]})
    path = tempdir / 'data.parquet'

    with pytest.raises(ValueError, match="'basename_template' argument "
                       "is not supported by use_legacy_dataset=True"):
        pq.write_to_dataset(table, path,
                            use_legacy_dataset=True,
                            partition_filename_cb=lambda x: 'filename.parquet',
                            basename_template='file-{i}.parquet')
    with pytest.raises(ValueError, match="'partition_filename_cb' argument "
                       "is not supported by use_legacy_dataset=False"):
        pq.write_to_dataset(table, path,
                            use_legacy_dataset=False,
                            partition_filename_cb=lambda x: 'filename.parquet',
                            basename_template='file-{i}.parquet')

    with pytest.raises(ValueError, match="'partitioning' argument "
                       "is not supported by use_legacy_dataset=True"):
        pq.write_to_dataset(table, path,
                            use_legacy_dataset=True,
                            partition_cols=["a"],
                            partitioning=["a"])

    with pytest.raises(ValueError, match="'partition_cols' argument "
                       "is not supported by use_legacy_dataset=False"):
        pq.write_to_dataset(table, path,
                            use_legacy_dataset=False,
                            partition_cols=["a"],
                            partitioning=["a"])

    with pytest.raises(ValueError, match="'file_visitor' argument "
                       "is not supported by use_legacy_dataset=True"):
        pq.write_to_dataset(table, path,
                            use_legacy_dataset=True,
                            metadata_collector=[],
                            file_visitor=lambda x: x)
    with pytest.raises(ValueError, match="'metadata_collector' argument "
                       "is not supported by use_legacy_dataset=False"):
        pq.write_to_dataset(table, path,
                            use_legacy_dataset=False,
                            metadata_collector=[],
                            file_visitor=lambda x: x)


@pytest.mark.dataset
@pytest.mark.parametrize("write_dataset_kwarg", (
    ("create_dir", True),
    ("create_dir", False),
))
def test_write_to_dataset_kwargs_passed(tempdir, write_dataset_kwarg):
    """Verify kwargs in pq.write_to_dataset are passed onto ds.write_dataset"""
    import pyarrow.dataset as ds

    table = pa.table({"a": [1, 2, 3]})
    path = tempdir / 'out.parquet'

    signature = inspect.signature(ds.write_dataset)
    key, arg = write_dataset_kwarg

    # kwarg not in pq.write_to_dataset, but will be passed to ds.write_dataset
    assert key not in inspect.signature(pq.write_to_dataset).parameters
    assert key in signature.parameters

    with mock.patch.object(ds, "write_dataset", autospec=True)\
            as mock_write_dataset:
        pq.write_to_dataset(table, path, **{key: arg})
        _name, _args, kwargs = mock_write_dataset.mock_calls[0]
        assert kwargs[key] == arg

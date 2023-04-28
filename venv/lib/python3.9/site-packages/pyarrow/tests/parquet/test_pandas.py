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

import io
import json

import numpy as np
import pytest

import pyarrow as pa
from pyarrow.fs import LocalFileSystem, SubTreeFileSystem
from pyarrow.tests.parquet.common import (
    parametrize_legacy_dataset, parametrize_legacy_dataset_not_supported)
from pyarrow.util import guid

try:
    import pyarrow.parquet as pq
    from pyarrow.tests.parquet.common import (_read_table, _test_dataframe,
                                              _write_table)
except ImportError:
    pq = None


try:
    import pandas as pd
    import pandas.testing as tm

    from pyarrow.tests.parquet.common import (_roundtrip_pandas_dataframe,
                                              alltypes_sample)
except ImportError:
    pd = tm = None


# Marks all of the tests in this module
# Ignore these with pytest ... -m 'not parquet'
pytestmark = pytest.mark.parquet


@pytest.mark.pandas
def test_pandas_parquet_custom_metadata(tempdir):
    df = alltypes_sample(size=10000)

    filename = tempdir / 'pandas_roundtrip.parquet'
    arrow_table = pa.Table.from_pandas(df)
    assert b'pandas' in arrow_table.schema.metadata

    _write_table(arrow_table, filename, version='2.6', coerce_timestamps='ms')

    metadata = pq.read_metadata(filename).metadata
    assert b'pandas' in metadata

    js = json.loads(metadata[b'pandas'].decode('utf8'))
    assert js['index_columns'] == [{'kind': 'range',
                                    'name': None,
                                    'start': 0, 'stop': 10000,
                                    'step': 1}]


@pytest.mark.pandas
def test_merging_parquet_tables_with_different_pandas_metadata(tempdir):
    # ARROW-3728: Merging Parquet Files - Pandas Meta in Schema Mismatch
    schema = pa.schema([
        pa.field('int', pa.int16()),
        pa.field('float', pa.float32()),
        pa.field('string', pa.string())
    ])
    df1 = pd.DataFrame({
        'int': np.arange(3, dtype=np.uint8),
        'float': np.arange(3, dtype=np.float32),
        'string': ['ABBA', 'EDDA', 'ACDC']
    })
    df2 = pd.DataFrame({
        'int': [4, 5],
        'float': [1.1, None],
        'string': [None, None]
    })
    table1 = pa.Table.from_pandas(df1, schema=schema, preserve_index=False)
    table2 = pa.Table.from_pandas(df2, schema=schema, preserve_index=False)

    assert not table1.schema.equals(table2.schema, check_metadata=True)
    assert table1.schema.equals(table2.schema)

    writer = pq.ParquetWriter(tempdir / 'merged.parquet', schema=schema)
    writer.write_table(table1)
    writer.write_table(table2)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_pandas_parquet_column_multiindex(tempdir, use_legacy_dataset):
    df = alltypes_sample(size=10)
    df.columns = pd.MultiIndex.from_tuples(
        list(zip(df.columns, df.columns[::-1])),
        names=['level_1', 'level_2']
    )

    filename = tempdir / 'pandas_roundtrip.parquet'
    arrow_table = pa.Table.from_pandas(df)
    assert arrow_table.schema.pandas_metadata is not None

    _write_table(arrow_table, filename, version='2.6', coerce_timestamps='ms')

    table_read = pq.read_pandas(
        filename, use_legacy_dataset=use_legacy_dataset)
    df_read = table_read.to_pandas()
    tm.assert_frame_equal(df, df_read)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_pandas_parquet_2_0_roundtrip_read_pandas_no_index_written(
    tempdir, use_legacy_dataset
):
    df = alltypes_sample(size=10000)

    filename = tempdir / 'pandas_roundtrip.parquet'
    arrow_table = pa.Table.from_pandas(df, preserve_index=False)
    js = arrow_table.schema.pandas_metadata
    assert not js['index_columns']
    # ARROW-2170
    # While index_columns should be empty, columns needs to be filled still.
    assert js['columns']

    _write_table(arrow_table, filename, version='2.6', coerce_timestamps='ms')
    table_read = pq.read_pandas(
        filename, use_legacy_dataset=use_legacy_dataset)

    js = table_read.schema.pandas_metadata
    assert not js['index_columns']

    read_metadata = table_read.schema.metadata
    assert arrow_table.schema.metadata == read_metadata

    df_read = table_read.to_pandas()
    tm.assert_frame_equal(df, df_read)


# TODO(dataset) duplicate column selection actually gives duplicate columns now
@pytest.mark.pandas
@parametrize_legacy_dataset_not_supported
def test_pandas_column_selection(tempdir, use_legacy_dataset):
    size = 10000
    np.random.seed(0)
    df = pd.DataFrame({
        'uint8': np.arange(size, dtype=np.uint8),
        'uint16': np.arange(size, dtype=np.uint16)
    })
    filename = tempdir / 'pandas_roundtrip.parquet'
    arrow_table = pa.Table.from_pandas(df)
    _write_table(arrow_table, filename)
    table_read = _read_table(
        filename, columns=['uint8'], use_legacy_dataset=use_legacy_dataset)
    df_read = table_read.to_pandas()

    tm.assert_frame_equal(df[['uint8']], df_read)

    # ARROW-4267: Selection of duplicate columns still leads to these columns
    # being read uniquely.
    table_read = _read_table(
        filename, columns=['uint8', 'uint8'],
        use_legacy_dataset=use_legacy_dataset)
    df_read = table_read.to_pandas()

    tm.assert_frame_equal(df[['uint8']], df_read)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_pandas_parquet_native_file_roundtrip(tempdir, use_legacy_dataset):
    df = _test_dataframe(10000)
    arrow_table = pa.Table.from_pandas(df)
    imos = pa.BufferOutputStream()
    _write_table(arrow_table, imos, version='2.6')
    buf = imos.getvalue()
    reader = pa.BufferReader(buf)
    df_read = _read_table(
        reader, use_legacy_dataset=use_legacy_dataset).to_pandas()
    tm.assert_frame_equal(df, df_read)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_read_pandas_column_subset(tempdir, use_legacy_dataset):
    df = _test_dataframe(10000)
    arrow_table = pa.Table.from_pandas(df)
    imos = pa.BufferOutputStream()
    _write_table(arrow_table, imos, version='2.6')
    buf = imos.getvalue()
    reader = pa.BufferReader(buf)
    df_read = pq.read_pandas(
        reader, columns=['strings', 'uint8'],
        use_legacy_dataset=use_legacy_dataset
    ).to_pandas()
    tm.assert_frame_equal(df[['strings', 'uint8']], df_read)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_pandas_parquet_empty_roundtrip(tempdir, use_legacy_dataset):
    df = _test_dataframe(0)
    arrow_table = pa.Table.from_pandas(df)
    imos = pa.BufferOutputStream()
    _write_table(arrow_table, imos, version='2.6')
    buf = imos.getvalue()
    reader = pa.BufferReader(buf)
    df_read = _read_table(
        reader, use_legacy_dataset=use_legacy_dataset).to_pandas()
    tm.assert_frame_equal(df, df_read)


@pytest.mark.pandas
def test_pandas_can_write_nested_data(tempdir):
    data = {
        "agg_col": [
            {"page_type": 1},
            {"record_type": 1},
            {"non_consecutive_home": 0},
        ],
        "uid_first": "1001"
    }
    df = pd.DataFrame(data=data)
    arrow_table = pa.Table.from_pandas(df)
    imos = pa.BufferOutputStream()
    # This succeeds under V2
    _write_table(arrow_table, imos)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_pandas_parquet_pyfile_roundtrip(tempdir, use_legacy_dataset):
    filename = tempdir / 'pandas_pyfile_roundtrip.parquet'
    size = 5
    df = pd.DataFrame({
        'int64': np.arange(size, dtype=np.int64),
        'float32': np.arange(size, dtype=np.float32),
        'float64': np.arange(size, dtype=np.float64),
        'bool': np.random.randn(size) > 0,
        'strings': ['foo', 'bar', None, 'baz', 'qux']
    })

    arrow_table = pa.Table.from_pandas(df)

    with filename.open('wb') as f:
        _write_table(arrow_table, f, version="2.4")

    data = io.BytesIO(filename.read_bytes())

    table_read = _read_table(data, use_legacy_dataset=use_legacy_dataset)
    df_read = table_read.to_pandas()
    tm.assert_frame_equal(df, df_read)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_pandas_parquet_configuration_options(tempdir, use_legacy_dataset):
    size = 10000
    np.random.seed(0)
    df = pd.DataFrame({
        'uint8': np.arange(size, dtype=np.uint8),
        'uint16': np.arange(size, dtype=np.uint16),
        'uint32': np.arange(size, dtype=np.uint32),
        'uint64': np.arange(size, dtype=np.uint64),
        'int8': np.arange(size, dtype=np.int16),
        'int16': np.arange(size, dtype=np.int16),
        'int32': np.arange(size, dtype=np.int32),
        'int64': np.arange(size, dtype=np.int64),
        'float32': np.arange(size, dtype=np.float32),
        'float64': np.arange(size, dtype=np.float64),
        'bool': np.random.randn(size) > 0
    })
    filename = tempdir / 'pandas_roundtrip.parquet'
    arrow_table = pa.Table.from_pandas(df)

    for use_dictionary in [True, False]:
        _write_table(arrow_table, filename, version='2.6',
                     use_dictionary=use_dictionary)
        table_read = _read_table(
            filename, use_legacy_dataset=use_legacy_dataset)
        df_read = table_read.to_pandas()
        tm.assert_frame_equal(df, df_read)

    for write_statistics in [True, False]:
        _write_table(arrow_table, filename, version='2.6',
                     write_statistics=write_statistics)
        table_read = _read_table(filename,
                                 use_legacy_dataset=use_legacy_dataset)
        df_read = table_read.to_pandas()
        tm.assert_frame_equal(df, df_read)

    for compression in ['NONE', 'SNAPPY', 'GZIP', 'LZ4', 'ZSTD']:
        if (compression != 'NONE' and
                not pa.lib.Codec.is_available(compression)):
            continue
        _write_table(arrow_table, filename, version='2.6',
                     compression=compression)
        table_read = _read_table(
            filename, use_legacy_dataset=use_legacy_dataset)
        df_read = table_read.to_pandas()
        tm.assert_frame_equal(df, df_read)


@pytest.mark.pandas
@pytest.mark.filterwarnings("ignore:Parquet format '2.0':FutureWarning")
def test_spark_flavor_preserves_pandas_metadata():
    df = _test_dataframe(size=100)
    df.index = np.arange(0, 10 * len(df), 10)
    df.index.name = 'foo'

    result = _roundtrip_pandas_dataframe(df, {'version': '2.0',
                                              'flavor': 'spark'})
    tm.assert_frame_equal(result, df)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_index_column_name_duplicate(tempdir, use_legacy_dataset):
    data = {
        'close': {
            pd.Timestamp('2017-06-30 01:31:00'): 154.99958999999998,
            pd.Timestamp('2017-06-30 01:32:00'): 154.99958999999998,
        },
        'time': {
            pd.Timestamp('2017-06-30 01:31:00'): pd.Timestamp(
                '2017-06-30 01:31:00'
            ),
            pd.Timestamp('2017-06-30 01:32:00'): pd.Timestamp(
                '2017-06-30 01:32:00'
            ),
        }
    }
    path = str(tempdir / 'data.parquet')
    dfx = pd.DataFrame(data).set_index('time', drop=False)
    tdfx = pa.Table.from_pandas(dfx)
    _write_table(tdfx, path)
    arrow_table = _read_table(path, use_legacy_dataset=use_legacy_dataset)
    result_df = arrow_table.to_pandas()
    tm.assert_frame_equal(result_df, dfx)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_multiindex_duplicate_values(tempdir, use_legacy_dataset):
    num_rows = 3
    numbers = list(range(num_rows))
    index = pd.MultiIndex.from_arrays(
        [['foo', 'foo', 'bar'], numbers],
        names=['foobar', 'some_numbers'],
    )

    df = pd.DataFrame({'numbers': numbers}, index=index)
    table = pa.Table.from_pandas(df)

    filename = tempdir / 'dup_multi_index_levels.parquet'

    _write_table(table, filename)
    result_table = _read_table(filename, use_legacy_dataset=use_legacy_dataset)
    assert table.equals(result_table)

    result_df = result_table.to_pandas()
    tm.assert_frame_equal(result_df, df)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_backwards_compatible_index_naming(datadir, use_legacy_dataset):
    expected_string = b"""\
carat        cut  color  clarity  depth  table  price     x     y     z
 0.23      Ideal      E      SI2   61.5   55.0    326  3.95  3.98  2.43
 0.21    Premium      E      SI1   59.8   61.0    326  3.89  3.84  2.31
 0.23       Good      E      VS1   56.9   65.0    327  4.05  4.07  2.31
 0.29    Premium      I      VS2   62.4   58.0    334  4.20  4.23  2.63
 0.31       Good      J      SI2   63.3   58.0    335  4.34  4.35  2.75
 0.24  Very Good      J     VVS2   62.8   57.0    336  3.94  3.96  2.48
 0.24  Very Good      I     VVS1   62.3   57.0    336  3.95  3.98  2.47
 0.26  Very Good      H      SI1   61.9   55.0    337  4.07  4.11  2.53
 0.22       Fair      E      VS2   65.1   61.0    337  3.87  3.78  2.49
 0.23  Very Good      H      VS1   59.4   61.0    338  4.00  4.05  2.39"""
    expected = pd.read_csv(io.BytesIO(expected_string), sep=r'\s{2,}',
                           index_col=None, header=0, engine='python')
    table = _read_table(
        datadir / 'v0.7.1.parquet', use_legacy_dataset=use_legacy_dataset)
    result = table.to_pandas()
    tm.assert_frame_equal(result, expected)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_backwards_compatible_index_multi_level_named(
    datadir, use_legacy_dataset
):
    expected_string = b"""\
carat        cut  color  clarity  depth  table  price     x     y     z
 0.23      Ideal      E      SI2   61.5   55.0    326  3.95  3.98  2.43
 0.21    Premium      E      SI1   59.8   61.0    326  3.89  3.84  2.31
 0.23       Good      E      VS1   56.9   65.0    327  4.05  4.07  2.31
 0.29    Premium      I      VS2   62.4   58.0    334  4.20  4.23  2.63
 0.31       Good      J      SI2   63.3   58.0    335  4.34  4.35  2.75
 0.24  Very Good      J     VVS2   62.8   57.0    336  3.94  3.96  2.48
 0.24  Very Good      I     VVS1   62.3   57.0    336  3.95  3.98  2.47
 0.26  Very Good      H      SI1   61.9   55.0    337  4.07  4.11  2.53
 0.22       Fair      E      VS2   65.1   61.0    337  3.87  3.78  2.49
 0.23  Very Good      H      VS1   59.4   61.0    338  4.00  4.05  2.39"""
    expected = pd.read_csv(
        io.BytesIO(expected_string), sep=r'\s{2,}',
        index_col=['cut', 'color', 'clarity'],
        header=0, engine='python'
    ).sort_index()

    table = _read_table(datadir / 'v0.7.1.all-named-index.parquet',
                        use_legacy_dataset=use_legacy_dataset)
    result = table.to_pandas()
    tm.assert_frame_equal(result, expected)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_backwards_compatible_index_multi_level_some_named(
        datadir, use_legacy_dataset
):
    expected_string = b"""\
carat        cut  color  clarity  depth  table  price     x     y     z
 0.23      Ideal      E      SI2   61.5   55.0    326  3.95  3.98  2.43
 0.21    Premium      E      SI1   59.8   61.0    326  3.89  3.84  2.31
 0.23       Good      E      VS1   56.9   65.0    327  4.05  4.07  2.31
 0.29    Premium      I      VS2   62.4   58.0    334  4.20  4.23  2.63
 0.31       Good      J      SI2   63.3   58.0    335  4.34  4.35  2.75
 0.24  Very Good      J     VVS2   62.8   57.0    336  3.94  3.96  2.48
 0.24  Very Good      I     VVS1   62.3   57.0    336  3.95  3.98  2.47
 0.26  Very Good      H      SI1   61.9   55.0    337  4.07  4.11  2.53
 0.22       Fair      E      VS2   65.1   61.0    337  3.87  3.78  2.49
 0.23  Very Good      H      VS1   59.4   61.0    338  4.00  4.05  2.39"""
    expected = pd.read_csv(
        io.BytesIO(expected_string),
        sep=r'\s{2,}', index_col=['cut', 'color', 'clarity'],
        header=0, engine='python'
    ).sort_index()
    expected.index = expected.index.set_names(['cut', None, 'clarity'])

    table = _read_table(datadir / 'v0.7.1.some-named-index.parquet',
                        use_legacy_dataset=use_legacy_dataset)
    result = table.to_pandas()
    tm.assert_frame_equal(result, expected)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_backwards_compatible_column_metadata_handling(
    datadir, use_legacy_dataset
):
    expected = pd.DataFrame(
        {'a': [1, 2, 3], 'b': [.1, .2, .3],
         'c': pd.date_range("2017-01-01", periods=3, tz='Europe/Brussels')})
    expected.index = pd.MultiIndex.from_arrays(
        [['a', 'b', 'c'],
         pd.date_range("2017-01-01", periods=3, tz='Europe/Brussels')],
        names=['index', None])

    path = datadir / 'v0.7.1.column-metadata-handling.parquet'
    table = _read_table(path, use_legacy_dataset=use_legacy_dataset)
    result = table.to_pandas()
    tm.assert_frame_equal(result, expected)

    table = _read_table(
        path, columns=['a'], use_legacy_dataset=use_legacy_dataset)
    result = table.to_pandas()
    tm.assert_frame_equal(result, expected[['a']].reset_index(drop=True))


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_categorical_index_survives_roundtrip(use_legacy_dataset):
    # ARROW-3652, addressed by ARROW-3246
    df = pd.DataFrame([['a', 'b'], ['c', 'd']], columns=['c1', 'c2'])
    df['c1'] = df['c1'].astype('category')
    df = df.set_index(['c1'])

    table = pa.Table.from_pandas(df)
    bos = pa.BufferOutputStream()
    pq.write_table(table, bos)
    ref_df = pq.read_pandas(
        bos.getvalue(), use_legacy_dataset=use_legacy_dataset).to_pandas()
    assert isinstance(ref_df.index, pd.CategoricalIndex)
    assert ref_df.index.equals(df.index)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_categorical_order_survives_roundtrip(use_legacy_dataset):
    # ARROW-6302
    df = pd.DataFrame({"a": pd.Categorical(
        ["a", "b", "c", "a"], categories=["b", "c", "d"], ordered=True)})

    table = pa.Table.from_pandas(df)
    bos = pa.BufferOutputStream()
    pq.write_table(table, bos)

    contents = bos.getvalue()
    result = pq.read_pandas(
        contents, use_legacy_dataset=use_legacy_dataset).to_pandas()

    tm.assert_frame_equal(result, df)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_pandas_categorical_na_type_row_groups(use_legacy_dataset):
    # ARROW-5085
    df = pd.DataFrame({"col": [None] * 100, "int": [1.0] * 100})
    df_category = df.astype({"col": "category", "int": "category"})
    table = pa.Table.from_pandas(df)
    table_cat = pa.Table.from_pandas(df_category)
    buf = pa.BufferOutputStream()

    # it works
    pq.write_table(table_cat, buf, version='2.6', chunk_size=10)
    result = pq.read_table(
        buf.getvalue(), use_legacy_dataset=use_legacy_dataset)

    # Result is non-categorical
    assert result[0].equals(table[0])
    assert result[1].equals(table[1])


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_pandas_categorical_roundtrip(use_legacy_dataset):
    # ARROW-5480, this was enabled by ARROW-3246

    # Have one of the categories unobserved and include a null (-1)
    codes = np.array([2, 0, 0, 2, 0, -1, 2], dtype='int32')
    categories = ['foo', 'bar', 'baz']
    df = pd.DataFrame({'x': pd.Categorical.from_codes(
        codes, categories=categories)})

    buf = pa.BufferOutputStream()
    pq.write_table(pa.table(df), buf)

    result = pq.read_table(
        buf.getvalue(), use_legacy_dataset=use_legacy_dataset).to_pandas()
    assert result.x.dtype == 'category'
    assert (result.x.cat.categories == categories).all()
    tm.assert_frame_equal(result, df)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_write_to_dataset_pandas_preserve_extensiondtypes(
    tempdir, use_legacy_dataset
):
    df = pd.DataFrame({'part': 'a', "col": [1, 2, 3]})
    df['col'] = df['col'].astype("Int64")
    table = pa.table(df)

    pq.write_to_dataset(
        table, str(tempdir / "case1"), partition_cols=['part'],
        use_legacy_dataset=use_legacy_dataset
    )
    result = pq.read_table(
        str(tempdir / "case1"), use_legacy_dataset=use_legacy_dataset
    ).to_pandas()
    tm.assert_frame_equal(result[["col"]], df[["col"]])

    pq.write_to_dataset(
        table, str(tempdir / "case2"), use_legacy_dataset=use_legacy_dataset
    )
    result = pq.read_table(
        str(tempdir / "case2"), use_legacy_dataset=use_legacy_dataset
    ).to_pandas()
    tm.assert_frame_equal(result[["col"]], df[["col"]])

    pq.write_table(table, str(tempdir / "data.parquet"))
    result = pq.read_table(
        str(tempdir / "data.parquet"), use_legacy_dataset=use_legacy_dataset
    ).to_pandas()
    tm.assert_frame_equal(result[["col"]], df[["col"]])


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_write_to_dataset_pandas_preserve_index(tempdir, use_legacy_dataset):
    # ARROW-8251 - preserve pandas index in roundtrip

    df = pd.DataFrame({'part': ['a', 'a', 'b'], "col": [1, 2, 3]})
    df.index = pd.Index(['a', 'b', 'c'], name="idx")
    table = pa.table(df)
    df_cat = df[["col", "part"]].copy()
    df_cat["part"] = df_cat["part"].astype("category")

    pq.write_to_dataset(
        table, str(tempdir / "case1"), partition_cols=['part'],
        use_legacy_dataset=use_legacy_dataset
    )
    result = pq.read_table(
        str(tempdir / "case1"), use_legacy_dataset=use_legacy_dataset
    ).to_pandas()
    tm.assert_frame_equal(result, df_cat)

    pq.write_to_dataset(
        table, str(tempdir / "case2"), use_legacy_dataset=use_legacy_dataset
    )
    result = pq.read_table(
        str(tempdir / "case2"), use_legacy_dataset=use_legacy_dataset
    ).to_pandas()
    tm.assert_frame_equal(result, df)

    pq.write_table(table, str(tempdir / "data.parquet"))
    result = pq.read_table(
        str(tempdir / "data.parquet"), use_legacy_dataset=use_legacy_dataset
    ).to_pandas()
    tm.assert_frame_equal(result, df)


@pytest.mark.pandas
@parametrize_legacy_dataset
@pytest.mark.parametrize('preserve_index', [True, False, None])
@pytest.mark.parametrize('metadata_fname', ["_metadata", "_common_metadata"])
def test_dataset_read_pandas_common_metadata(
    tempdir, use_legacy_dataset, preserve_index, metadata_fname
):
    # ARROW-1103
    nfiles = 5
    size = 5

    dirpath = tempdir / guid()
    dirpath.mkdir()

    test_data = []
    frames = []
    paths = []
    for i in range(nfiles):
        df = _test_dataframe(size, seed=i)
        df.index = pd.Index(np.arange(i * size, (i + 1) * size), name='index')

        path = dirpath / '{}.parquet'.format(i)

        table = pa.Table.from_pandas(df, preserve_index=preserve_index)

        # Obliterate metadata
        table = table.replace_schema_metadata(None)
        assert table.schema.metadata is None

        _write_table(table, path)
        test_data.append(table)
        frames.append(df)
        paths.append(path)

    # Write _metadata common file
    table_for_metadata = pa.Table.from_pandas(
        df, preserve_index=preserve_index
    )
    pq.write_metadata(table_for_metadata.schema, dirpath / metadata_fname)

    dataset = pq.ParquetDataset(dirpath, use_legacy_dataset=use_legacy_dataset)
    columns = ['uint8', 'strings']
    result = dataset.read_pandas(columns=columns).to_pandas()
    expected = pd.concat([x[columns] for x in frames])
    expected.index.name = (
        df.index.name if preserve_index is not False else None)
    tm.assert_frame_equal(result, expected)


@pytest.mark.pandas
def test_read_pandas_passthrough_keywords(tempdir):
    # ARROW-11464 - previously not all keywords were passed through (such as
    # the filesystem keyword)
    df = pd.DataFrame({'a': [1, 2, 3]})

    filename = tempdir / 'data.parquet'
    _write_table(df, filename)

    result = pq.read_pandas(
        'data.parquet',
        filesystem=SubTreeFileSystem(str(tempdir), LocalFileSystem())
    )
    assert result.equals(pa.table(df))


@pytest.mark.pandas
def test_read_pandas_map_fields(tempdir):
    # ARROW-10140 - table created from Pandas with mapping fields
    df = pd.DataFrame({
        'col1': pd.Series([
            [('id', 'something'), ('value2', 'else')],
            [('id', 'something2'), ('value', 'else2')],
        ]),
        'col2': pd.Series(['foo', 'bar'])
    })

    filename = tempdir / 'data.parquet'

    udt = pa.map_(pa.string(), pa.string())
    schema = pa.schema([pa.field('col1', udt), pa.field('col2', pa.string())])
    arrow_table = pa.Table.from_pandas(df, schema)

    _write_table(arrow_table, filename)

    result = pq.read_pandas(filename).to_pandas()
    tm.assert_frame_equal(result, df)

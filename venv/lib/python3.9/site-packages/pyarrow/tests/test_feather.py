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
import os
import sys
import tempfile
import pytest
import hypothesis as h
import hypothesis.strategies as st

import numpy as np

import pyarrow as pa
import pyarrow.tests.strategies as past
from pyarrow.feather import (read_feather, write_feather, read_table,
                             FeatherDataset)

try:
    from pandas.testing import assert_frame_equal
    import pandas as pd
    import pyarrow.pandas_compat
except ImportError:
    pass


@pytest.fixture(scope='module')
def datadir(base_datadir):
    return base_datadir / 'feather'


def random_path(prefix='feather_'):
    return tempfile.mktemp(prefix=prefix)


@pytest.fixture(scope="module", params=[1, 2])
def version(request):
    yield request.param


@pytest.fixture(scope="module", params=[None, "uncompressed", "lz4", "zstd"])
def compression(request):
    if request.param in ['lz4', 'zstd'] and not pa.Codec.is_available(
            request.param):
        pytest.skip(f'{request.param} is not available')
    yield request.param


TEST_FILES = None


def setup_module(module):
    global TEST_FILES
    TEST_FILES = []


def teardown_module(module):
    for path in TEST_FILES:
        try:
            os.remove(path)
        except os.error:
            pass


@pytest.mark.pandas
def test_file_not_exist():
    with pytest.raises(pa.ArrowIOError):
        read_feather('test_invalid_file')


def _check_pandas_roundtrip(df, expected=None, path=None,
                            columns=None, use_threads=False,
                            version=None, compression=None,
                            compression_level=None):
    if path is None:
        path = random_path()

    if version is None:
        version = 2

    TEST_FILES.append(path)
    write_feather(df, path, compression=compression,
                  compression_level=compression_level, version=version)

    if not os.path.exists(path):
        raise Exception('file not written')

    result = read_feather(path, columns, use_threads=use_threads)

    if expected is None:
        expected = df

    assert_frame_equal(result, expected)


def _check_arrow_roundtrip(table, path=None, compression=None):
    if path is None:
        path = random_path()

    TEST_FILES.append(path)
    write_feather(table, path, compression=compression)
    if not os.path.exists(path):
        raise Exception('file not written')

    result = read_table(path)
    assert result.equals(table)


def _assert_error_on_write(df, exc, path=None, version=2):
    # check that we are raising the exception
    # on writing

    if path is None:
        path = random_path()

    TEST_FILES.append(path)

    def f():
        write_feather(df, path, version=version)

    pytest.raises(exc, f)


def test_dataset(version):
    num_values = (100, 100)
    num_files = 5
    paths = [random_path() for i in range(num_files)]
    data = {
        "col_" + str(i): np.random.randn(num_values[0])
        for i in range(num_values[1])
    }
    table = pa.table(data)

    TEST_FILES.extend(paths)
    for index, path in enumerate(paths):
        rows = (
            index * (num_values[0] // num_files),
            (index + 1) * (num_values[0] // num_files),
        )

        write_feather(table[rows[0]: rows[1]], path, version=version)

    data = FeatherDataset(paths).read_table()
    assert data.equals(table)


@pytest.mark.pandas
def test_float_no_nulls(version):
    data = {}
    numpy_dtypes = ['f4', 'f8']
    num_values = 100

    for dtype in numpy_dtypes:
        values = np.random.randn(num_values)
        data[dtype] = values.astype(dtype)

    df = pd.DataFrame(data)
    _check_pandas_roundtrip(df, version=version)


@pytest.mark.pandas
def test_read_table(version):
    num_values = (100, 100)
    path = random_path()

    TEST_FILES.append(path)

    values = np.random.randint(0, 100, size=num_values)
    columns = ['col_' + str(i) for i in range(100)]
    table = pa.Table.from_arrays(values, columns)

    write_feather(table, path, version=version)

    result = read_table(path)
    assert result.equals(table)

    # Test without memory mapping
    result = read_table(path, memory_map=False)
    assert result.equals(table)

    result = read_feather(path, memory_map=False)
    assert_frame_equal(table.to_pandas(), result)


@pytest.mark.pandas
def test_use_threads(version):
    # ARROW-14470
    num_values = (10, 10)
    path = random_path()

    TEST_FILES.append(path)

    values = np.random.randint(0, 10, size=num_values)
    columns = ['col_' + str(i) for i in range(10)]
    table = pa.Table.from_arrays(values, columns)

    write_feather(table, path, version=version)

    result = read_feather(path)
    assert_frame_equal(table.to_pandas(), result)

    # Test read_feather with use_threads=False
    result = read_feather(path, use_threads=False)
    assert_frame_equal(table.to_pandas(), result)

    # Test read_table with use_threads=False
    result = read_table(path, use_threads=False)
    assert result.equals(table)


@pytest.mark.pandas
def test_float_nulls(version):
    num_values = 100

    path = random_path()
    TEST_FILES.append(path)

    null_mask = np.random.randint(0, 10, size=num_values) < 3
    dtypes = ['f4', 'f8']
    expected_cols = []

    arrays = []
    for name in dtypes:
        values = np.random.randn(num_values).astype(name)
        arrays.append(pa.array(values, mask=null_mask))

        values[null_mask] = np.nan

        expected_cols.append(values)

    table = pa.table(arrays, names=dtypes)
    _check_arrow_roundtrip(table)

    df = table.to_pandas()
    _check_pandas_roundtrip(df, version=version)


@pytest.mark.pandas
def test_integer_no_nulls(version):
    data, arr = {}, []

    numpy_dtypes = ['i1', 'i2', 'i4', 'i8',
                    'u1', 'u2', 'u4', 'u8']
    num_values = 100

    for dtype in numpy_dtypes:
        values = np.random.randint(0, 100, size=num_values)
        data[dtype] = values.astype(dtype)
        arr.append(values.astype(dtype))

    df = pd.DataFrame(data)
    _check_pandas_roundtrip(df, version=version)

    table = pa.table(arr, names=numpy_dtypes)
    _check_arrow_roundtrip(table)


@pytest.mark.pandas
def test_platform_numpy_integers(version):
    data = {}

    numpy_dtypes = ['longlong']
    num_values = 100

    for dtype in numpy_dtypes:
        values = np.random.randint(0, 100, size=num_values)
        data[dtype] = values.astype(dtype)

    df = pd.DataFrame(data)
    _check_pandas_roundtrip(df, version=version)


@pytest.mark.pandas
def test_integer_with_nulls(version):
    # pandas requires upcast to float dtype
    path = random_path()
    TEST_FILES.append(path)

    int_dtypes = ['i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8']
    num_values = 100

    arrays = []
    null_mask = np.random.randint(0, 10, size=num_values) < 3
    expected_cols = []
    for name in int_dtypes:
        values = np.random.randint(0, 100, size=num_values)
        arrays.append(pa.array(values, mask=null_mask))

        expected = values.astype('f8')
        expected[null_mask] = np.nan

        expected_cols.append(expected)

    table = pa.table(arrays, names=int_dtypes)
    _check_arrow_roundtrip(table)

    df = table.to_pandas()
    _check_pandas_roundtrip(df, version=version)


@pytest.mark.pandas
def test_boolean_no_nulls(version):
    num_values = 100

    np.random.seed(0)

    df = pd.DataFrame({'bools': np.random.randn(num_values) > 0})
    _check_pandas_roundtrip(df, version=version)


@pytest.mark.pandas
def test_boolean_nulls(version):
    # pandas requires upcast to object dtype
    path = random_path()
    TEST_FILES.append(path)

    num_values = 100
    np.random.seed(0)

    mask = np.random.randint(0, 10, size=num_values) < 3
    values = np.random.randint(0, 10, size=num_values) < 5

    table = pa.table([pa.array(values, mask=mask)], names=['bools'])
    _check_arrow_roundtrip(table)

    df = table.to_pandas()
    _check_pandas_roundtrip(df, version=version)


def test_buffer_bounds_error(version):
    # ARROW-1676
    path = random_path()
    TEST_FILES.append(path)

    for i in range(16, 256):
        table = pa.Table.from_arrays(
            [pa.array([None] + list(range(i)), type=pa.float64())],
            names=["arr"]
        )
        _check_arrow_roundtrip(table)


def test_boolean_object_nulls(version):
    repeats = 100
    table = pa.Table.from_arrays(
        [np.array([False, None, True] * repeats, dtype=object)],
        names=["arr"]
    )
    _check_arrow_roundtrip(table)


@pytest.mark.pandas
def test_delete_partial_file_on_error(version):
    if sys.platform == 'win32':
        pytest.skip('Windows hangs on to file handle for some reason')

    class CustomClass:
        pass

    # strings will fail
    df = pd.DataFrame(
        {
            'numbers': range(5),
            'strings': [b'foo', None, 'bar', CustomClass(), np.nan]},
        columns=['numbers', 'strings'])

    path = random_path()
    try:
        write_feather(df, path, version=version)
    except Exception:
        pass

    assert not os.path.exists(path)


@pytest.mark.pandas
def test_strings(version):
    repeats = 1000

    # Mixed bytes, unicode, strings coerced to binary
    values = [b'foo', None, 'bar', 'qux', np.nan]
    df = pd.DataFrame({'strings': values * repeats})

    ex_values = [b'foo', None, b'bar', b'qux', np.nan]
    expected = pd.DataFrame({'strings': ex_values * repeats})
    _check_pandas_roundtrip(df, expected, version=version)

    # embedded nulls are ok
    values = ['foo', None, 'bar', 'qux', None]
    df = pd.DataFrame({'strings': values * repeats})
    expected = pd.DataFrame({'strings': values * repeats})
    _check_pandas_roundtrip(df, expected, version=version)

    values = ['foo', None, 'bar', 'qux', np.nan]
    df = pd.DataFrame({'strings': values * repeats})
    expected = pd.DataFrame({'strings': values * repeats})
    _check_pandas_roundtrip(df, expected, version=version)


@pytest.mark.pandas
def test_empty_strings(version):
    df = pd.DataFrame({'strings': [''] * 10})
    _check_pandas_roundtrip(df, version=version)


@pytest.mark.pandas
def test_all_none(version):
    df = pd.DataFrame({'all_none': [None] * 10})
    _check_pandas_roundtrip(df, version=version)


@pytest.mark.pandas
def test_all_null_category(version):
    # ARROW-1188
    df = pd.DataFrame({"A": (1, 2, 3), "B": (None, None, None)})
    df = df.assign(B=df.B.astype("category"))
    _check_pandas_roundtrip(df, version=version)


@pytest.mark.pandas
def test_multithreaded_read(version):
    data = {'c{}'.format(i): [''] * 10
            for i in range(100)}
    df = pd.DataFrame(data)
    _check_pandas_roundtrip(df, use_threads=True, version=version)


@pytest.mark.pandas
def test_nan_as_null(version):
    # Create a nan that is not numpy.nan
    values = np.array(['foo', np.nan, np.nan * 2, 'bar'] * 10)
    df = pd.DataFrame({'strings': values})
    _check_pandas_roundtrip(df, version=version)


@pytest.mark.pandas
def test_category(version):
    repeats = 1000
    values = ['foo', None, 'bar', 'qux', np.nan]
    df = pd.DataFrame({'strings': values * repeats})
    df['strings'] = df['strings'].astype('category')

    values = ['foo', None, 'bar', 'qux', None]
    expected = pd.DataFrame({'strings': pd.Categorical(values * repeats)})
    _check_pandas_roundtrip(df, expected, version=version)


@pytest.mark.pandas
def test_timestamp(version):
    df = pd.DataFrame({'naive': pd.date_range('2016-03-28', periods=10)})
    df['with_tz'] = (df.naive.dt.tz_localize('utc')
                     .dt.tz_convert('America/Los_Angeles'))

    _check_pandas_roundtrip(df, version=version)


@pytest.mark.pandas
def test_timestamp_with_nulls(version):
    df = pd.DataFrame({'test': [pd.Timestamp(2016, 1, 1),
                                None,
                                pd.Timestamp(2016, 1, 3)]})
    df['with_tz'] = df.test.dt.tz_localize('utc')

    _check_pandas_roundtrip(df, version=version)


@pytest.mark.pandas
@pytest.mark.xfail(reason="not supported", raises=TypeError)
def test_timedelta_with_nulls_v1():
    df = pd.DataFrame({'test': [pd.Timedelta('1 day'),
                                None,
                                pd.Timedelta('3 day')]})
    _check_pandas_roundtrip(df, version=1)


@pytest.mark.pandas
def test_timedelta_with_nulls():
    df = pd.DataFrame({'test': [pd.Timedelta('1 day'),
                                None,
                                pd.Timedelta('3 day')]})
    _check_pandas_roundtrip(df, version=2)


@pytest.mark.pandas
def test_out_of_float64_timestamp_with_nulls(version):
    df = pd.DataFrame(
        {'test': pd.DatetimeIndex([1451606400000000001,
                                   None, 14516064000030405])})
    df['with_tz'] = df.test.dt.tz_localize('utc')
    _check_pandas_roundtrip(df, version=version)


@pytest.mark.pandas
def test_non_string_columns(version):
    df = pd.DataFrame({0: [1, 2, 3, 4],
                       1: [True, False, True, False]})
    expected = df

    if version == 1:
        expected = df.rename(columns=str)
    _check_pandas_roundtrip(df, expected, version=version)


@pytest.mark.pandas
@pytest.mark.skipif(not os.path.supports_unicode_filenames,
                    reason='unicode filenames not supported')
def test_unicode_filename(version):
    # GH #209
    name = (b'Besa_Kavaj\xc3\xab.feather').decode('utf-8')
    df = pd.DataFrame({'foo': [1, 2, 3, 4]})
    _check_pandas_roundtrip(df, path=random_path(prefix=name),
                            version=version)


@pytest.mark.pandas
def test_read_columns(version):
    df = pd.DataFrame({
        'foo': [1, 2, 3, 4],
        'boo': [5, 6, 7, 8],
        'woo': [1, 3, 5, 7]
    })
    expected = df[['boo', 'woo']]

    _check_pandas_roundtrip(df, expected, version=version,
                            columns=['boo', 'woo'])


def test_overwritten_file(version):
    path = random_path()
    TEST_FILES.append(path)

    num_values = 100
    np.random.seed(0)

    values = np.random.randint(0, 10, size=num_values)

    table = pa.table({'ints': values})
    write_feather(table, path)

    table = pa.table({'more_ints': values[0:num_values//2]})
    _check_arrow_roundtrip(table, path=path)


@pytest.mark.pandas
def test_filelike_objects(version):
    buf = io.BytesIO()

    # the copy makes it non-strided
    df = pd.DataFrame(np.arange(12).reshape(4, 3),
                      columns=['a', 'b', 'c']).copy()
    write_feather(df, buf, version=version)

    buf.seek(0)

    result = read_feather(buf)
    assert_frame_equal(result, df)


@pytest.mark.pandas
@pytest.mark.filterwarnings("ignore:Sparse:FutureWarning")
@pytest.mark.filterwarnings("ignore:DataFrame.to_sparse:FutureWarning")
def test_sparse_dataframe(version):
    if not pa.pandas_compat._pandas_api.has_sparse:
        pytest.skip("version of pandas does not support SparseDataFrame")
    # GH #221
    data = {'A': [0, 1, 2],
            'B': [1, 0, 1]}
    df = pd.DataFrame(data).to_sparse(fill_value=1)
    expected = df.to_dense()
    _check_pandas_roundtrip(df, expected, version=version)


@pytest.mark.pandas
def test_duplicate_columns_pandas():

    # https://github.com/wesm/feather/issues/53
    # not currently able to handle duplicate columns
    df = pd.DataFrame(np.arange(12).reshape(4, 3),
                      columns=list('aaa')).copy()
    _assert_error_on_write(df, ValueError)


def test_duplicate_columns():
    # only works for version 2
    table = pa.table([[1, 2, 3], [4, 5, 6], [7, 8, 9]], names=['a', 'a', 'b'])
    _check_arrow_roundtrip(table)
    _assert_error_on_write(table, ValueError, version=1)


@pytest.mark.pandas
def test_unsupported():
    # https://github.com/wesm/feather/issues/240
    # serializing actual python objects

    # custom python objects
    class A:
        pass

    df = pd.DataFrame({'a': [A(), A()]})
    _assert_error_on_write(df, ValueError)

    # non-strings
    df = pd.DataFrame({'a': ['a', 1, 2.0]})
    _assert_error_on_write(df, TypeError)


@pytest.mark.pandas
def test_v2_set_chunksize():
    df = pd.DataFrame({'A': np.arange(1000)})
    table = pa.table(df)

    buf = io.BytesIO()
    write_feather(table, buf, chunksize=250, version=2)

    result = buf.getvalue()

    ipc_file = pa.ipc.open_file(pa.BufferReader(result))
    assert ipc_file.num_record_batches == 4
    assert len(ipc_file.get_batch(0)) == 250


@pytest.mark.pandas
@pytest.mark.lz4
@pytest.mark.snappy
@pytest.mark.zstd
def test_v2_compression_options():
    df = pd.DataFrame({'A': np.arange(1000)})

    cases = [
        # compression, compression_level
        ('uncompressed', None),
        ('lz4', None),
        ('lz4', 1),
        ('lz4', 12),
        ('zstd', 1),
        ('zstd', 10)
    ]

    for compression, compression_level in cases:
        _check_pandas_roundtrip(df, compression=compression,
                                compression_level=compression_level)

    buf = io.BytesIO()

    # Trying to compress with V1
    with pytest.raises(
            ValueError,
            match="Feather V1 files do not support compression option"):
        write_feather(df, buf, compression='lz4', version=1)

    # Trying to set chunksize with V1
    with pytest.raises(
            ValueError,
            match="Feather V1 files do not support chunksize option"):
        write_feather(df, buf, chunksize=4096, version=1)

    # Unsupported compressor
    with pytest.raises(ValueError,
                       match='compression="snappy" not supported'):
        write_feather(df, buf, compression='snappy')


def test_v2_lz4_default_compression():
    # ARROW-8750: Make sure that the compression=None option selects lz4 if
    # it's available
    if not pa.Codec.is_available('lz4_frame'):
        pytest.skip("LZ4 compression support is not built in C++")

    # some highly compressible data
    t = pa.table([np.repeat(0, 100000)], names=['f0'])

    buf = io.BytesIO()
    write_feather(t, buf)
    default_result = buf.getvalue()

    buf = io.BytesIO()
    write_feather(t, buf, compression='uncompressed')
    uncompressed_result = buf.getvalue()

    assert len(default_result) < len(uncompressed_result)


def test_v1_unsupported_types():
    table = pa.table([pa.array([[1, 2, 3], [], None])], names=['f0'])

    buf = io.BytesIO()
    with pytest.raises(TypeError,
                       match=("Unsupported Feather V1 type: "
                              "list<item: int64>. "
                              "Use V2 format to serialize all Arrow types.")):
        write_feather(table, buf, version=1)


@pytest.mark.slow
@pytest.mark.pandas
def test_large_dataframe(version):
    df = pd.DataFrame({'A': np.arange(400000000)})
    _check_pandas_roundtrip(df, version=version)


@pytest.mark.large_memory
@pytest.mark.pandas
def test_chunked_binary_error_message():
    # ARROW-3058: As Feather does not yet support chunked columns, we at least
    # make sure it's clear to the user what is going on

    # 2^31 + 1 bytes
    values = [b'x'] + [
        b'x' * (1 << 20)
    ] * 2 * (1 << 10)
    df = pd.DataFrame({'byte_col': values})

    # Works fine with version 2
    buf = io.BytesIO()
    write_feather(df, buf, version=2)
    result = read_feather(pa.BufferReader(buf.getvalue()))
    assert_frame_equal(result, df)

    with pytest.raises(ValueError, match="'byte_col' exceeds 2GB maximum "
                       "capacity of a Feather binary column. This restriction "
                       "may be lifted in the future"):
        write_feather(df, io.BytesIO(), version=1)


def test_feather_without_pandas(tempdir, version):
    # ARROW-8345
    table = pa.table([pa.array([1, 2, 3])], names=['f0'])
    path = str(tempdir / "data.feather")
    _check_arrow_roundtrip(table, path)


@pytest.mark.pandas
def test_read_column_selection(version):
    # ARROW-8641
    df = pd.DataFrame(np.arange(12).reshape(4, 3), columns=['a', 'b', 'c'])

    # select columns as string names or integer indices
    _check_pandas_roundtrip(
        df, columns=['a', 'c'], expected=df[['a', 'c']], version=version)
    _check_pandas_roundtrip(
        df, columns=[0, 2], expected=df[['a', 'c']], version=version)

    # different order is followed
    _check_pandas_roundtrip(
        df, columns=['b', 'a'], expected=df[['b', 'a']], version=version)
    _check_pandas_roundtrip(
        df, columns=[1, 0], expected=df[['b', 'a']], version=version)


def test_read_column_duplicated_selection(tempdir, version):
    # duplicated columns in the column selection
    table = pa.table([[1, 2, 3], [4, 5, 6], [7, 8, 9]], names=['a', 'b', 'c'])
    path = str(tempdir / "data.feather")
    write_feather(table, path, version=version)

    expected = pa.table([[1, 2, 3], [4, 5, 6], [1, 2, 3]],
                        names=['a', 'b', 'a'])
    for col_selection in [['a', 'b', 'a'], [0, 1, 0]]:
        result = read_table(path, columns=col_selection)
        assert result.equals(expected)


def test_read_column_duplicated_in_file(tempdir):
    # duplicated columns in feather file (only works for feather v2)
    table = pa.table([[1, 2, 3], [4, 5, 6], [7, 8, 9]], names=['a', 'b', 'a'])
    path = str(tempdir / "data.feather")
    write_feather(table, path, version=2)

    # no selection works fine
    result = read_table(path)
    assert result.equals(table)

    # selection with indices works
    result = read_table(path, columns=[0, 2])
    assert result.column_names == ['a', 'a']

    # selection with column names errors
    with pytest.raises(ValueError):
        read_table(path, columns=['a', 'b'])


def test_nested_types(compression):
    # https://issues.apache.org/jira/browse/ARROW-8860
    table = pa.table({'col': pa.StructArray.from_arrays(
        [[0, 1, 2], [1, 2, 3]], names=["f1", "f2"])})
    _check_arrow_roundtrip(table, compression=compression)

    table = pa.table({'col': pa.array([[1, 2], [3, 4]])})
    _check_arrow_roundtrip(table, compression=compression)

    table = pa.table({'col': pa.array([[[1, 2], [3, 4]], [[5, 6], None]])})
    _check_arrow_roundtrip(table, compression=compression)


@h.given(past.all_tables, st.sampled_from(["uncompressed", "lz4", "zstd"]))
def test_roundtrip(table, compression):
    _check_arrow_roundtrip(table, compression=compression)


@pytest.mark.lz4
def test_feather_v017_experimental_compression_backward_compatibility(datadir):
    # ARROW-11163 - ensure newer pyarrow versions can read the old feather
    # files from version 0.17.0 with experimental compression support (before
    # it was officially added to IPC format in 1.0.0)

    # file generated with:
    #     table = pa.table({'a': range(5)})
    #     from pyarrow import feather
    #     feather.write_feather(
    #         table, "v0.17.0.version.2-compression.lz4.feather",
    #         compression="lz4", version=2)
    expected = pa.table({'a': range(5)})
    result = read_table(datadir / "v0.17.0.version.2-compression.lz4.feather")
    assert result.equals(expected)


@pytest.mark.pandas
def test_preserve_index_pandas(version):
    df = pd.DataFrame({'a': [1, 2, 3]}, index=['a', 'b', 'c'])

    if version == 1:
        expected = df.reset_index(drop=True).rename(columns=str)
    else:
        expected = df

    _check_pandas_roundtrip(df, expected, version=version)


@pytest.mark.pandas
def test_feather_datetime_resolution_arrow_to_pandas(tempdir):
    # ARROW-17192 - ensure timestamp_as_object=True (together with other
    # **kwargs) can be passed in read_feather to to_pandas.

    from datetime import datetime
    df = pd.DataFrame({"date": [
        datetime.fromisoformat("1654-01-01"),
        datetime.fromisoformat("1920-01-01"), ],
    })
    write_feather(df, tempdir / "test_resolution.feather")

    expected_0 = datetime.fromisoformat("1654-01-01")
    expected_1 = datetime.fromisoformat("1920-01-01")

    result = read_feather(tempdir / "test_resolution.feather",
                          timestamp_as_object=True)

    assert expected_0 == result['date'][0]
    assert expected_1 == result['date'][1]

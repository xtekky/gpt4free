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

from collections import OrderedDict
import io
import warnings

import numpy as np
import pytest

import pyarrow as pa
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem, FileSystem
from pyarrow.tests import util
from pyarrow.tests.parquet.common import (_check_roundtrip, _roundtrip_table,
                                          parametrize_legacy_dataset,
                                          _test_dataframe)

try:
    import pyarrow.parquet as pq
    from pyarrow.tests.parquet.common import _read_table, _write_table
except ImportError:
    pq = None


try:
    import pandas as pd
    import pandas.testing as tm

    from pyarrow.tests.pandas_examples import dataframe_with_lists
    from pyarrow.tests.parquet.common import alltypes_sample
except ImportError:
    pd = tm = None


# Marks all of the tests in this module
# Ignore these with pytest ... -m 'not parquet'
pytestmark = pytest.mark.parquet


def test_parquet_invalid_version(tempdir):
    table = pa.table({'a': [1, 2, 3]})
    with pytest.raises(ValueError, match="Unsupported Parquet format version"):
        _write_table(table, tempdir / 'test_version.parquet', version="2.2")
    with pytest.raises(ValueError, match="Unsupported Parquet data page " +
                       "version"):
        _write_table(table, tempdir / 'test_version.parquet',
                     data_page_version="2.2")


@parametrize_legacy_dataset
def test_set_data_page_size(use_legacy_dataset):
    arr = pa.array([1, 2, 3] * 100000)
    t = pa.Table.from_arrays([arr], names=['f0'])

    # 128K, 512K
    page_sizes = [2 << 16, 2 << 18]
    for target_page_size in page_sizes:
        _check_roundtrip(t, data_page_size=target_page_size,
                         use_legacy_dataset=use_legacy_dataset)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_set_write_batch_size(use_legacy_dataset):
    df = _test_dataframe(100)
    table = pa.Table.from_pandas(df, preserve_index=False)

    _check_roundtrip(
        table, data_page_size=10, write_batch_size=1, version='2.4'
    )


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_set_dictionary_pagesize_limit(use_legacy_dataset):
    df = _test_dataframe(100)
    table = pa.Table.from_pandas(df, preserve_index=False)

    _check_roundtrip(table, dictionary_pagesize_limit=1,
                     data_page_size=10, version='2.4')

    with pytest.raises(TypeError):
        _check_roundtrip(table, dictionary_pagesize_limit="a",
                         data_page_size=10, version='2.4')


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_chunked_table_write(use_legacy_dataset):
    # ARROW-232
    tables = []
    batch = pa.RecordBatch.from_pandas(alltypes_sample(size=10))
    tables.append(pa.Table.from_batches([batch] * 3))
    df, _ = dataframe_with_lists()
    batch = pa.RecordBatch.from_pandas(df)
    tables.append(pa.Table.from_batches([batch] * 3))

    for data_page_version in ['1.0', '2.0']:
        for use_dictionary in [True, False]:
            for table in tables:
                _check_roundtrip(
                    table, version='2.6',
                    use_legacy_dataset=use_legacy_dataset,
                    data_page_version=data_page_version,
                    use_dictionary=use_dictionary)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_memory_map(tempdir, use_legacy_dataset):
    df = alltypes_sample(size=10)

    table = pa.Table.from_pandas(df)
    _check_roundtrip(table, read_table_kwargs={'memory_map': True},
                     version='2.6', use_legacy_dataset=use_legacy_dataset)

    filename = str(tempdir / 'tmp_file')
    with open(filename, 'wb') as f:
        _write_table(table, f, version='2.6')
    table_read = pq.read_pandas(filename, memory_map=True,
                                use_legacy_dataset=use_legacy_dataset)
    assert table_read.equals(table)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_enable_buffered_stream(tempdir, use_legacy_dataset):
    df = alltypes_sample(size=10)

    table = pa.Table.from_pandas(df)
    _check_roundtrip(table, read_table_kwargs={'buffer_size': 1025},
                     version='2.6', use_legacy_dataset=use_legacy_dataset)

    filename = str(tempdir / 'tmp_file')
    with open(filename, 'wb') as f:
        _write_table(table, f, version='2.6')
    table_read = pq.read_pandas(filename, buffer_size=4096,
                                use_legacy_dataset=use_legacy_dataset)
    assert table_read.equals(table)


@parametrize_legacy_dataset
def test_special_chars_filename(tempdir, use_legacy_dataset):
    table = pa.Table.from_arrays([pa.array([42])], ["ints"])
    filename = "foo # bar"
    path = tempdir / filename
    assert not path.exists()
    _write_table(table, str(path))
    assert path.exists()
    table_read = _read_table(str(path), use_legacy_dataset=use_legacy_dataset)
    assert table_read.equals(table)


@parametrize_legacy_dataset
def test_invalid_source(use_legacy_dataset):
    # Test that we provide an helpful error message pointing out
    # that None wasn't expected when trying to open a Parquet None file.
    #
    # Depending on use_legacy_dataset the message changes slightly
    # but in both cases it should point out that None wasn't expected.
    with pytest.raises(TypeError, match="None"):
        pq.read_table(None, use_legacy_dataset=use_legacy_dataset)

    with pytest.raises(TypeError, match="None"):
        pq.ParquetFile(None)


@pytest.mark.slow
def test_file_with_over_int16_max_row_groups():
    # PARQUET-1857: Parquet encryption support introduced a INT16_MAX upper
    # limit on the number of row groups, but this limit only impacts files with
    # encrypted row group metadata because of the int16 row group ordinal used
    # in the Parquet Thrift metadata. Unencrypted files are not impacted, so
    # this test checks that it works (even if it isn't a good idea)
    t = pa.table([list(range(40000))], names=['f0'])
    _check_roundtrip(t, row_group_size=1)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_empty_table_roundtrip(use_legacy_dataset):
    df = alltypes_sample(size=10)

    # Create a non-empty table to infer the types correctly, then slice to 0
    table = pa.Table.from_pandas(df)
    table = pa.Table.from_arrays(
        [col.chunk(0)[:0] for col in table.itercolumns()],
        names=table.schema.names)

    assert table.schema.field('null').type == pa.null()
    assert table.schema.field('null_list').type == pa.list_(pa.null())
    _check_roundtrip(
        table, version='2.6', use_legacy_dataset=use_legacy_dataset)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_empty_table_no_columns(use_legacy_dataset):
    df = pd.DataFrame()
    empty = pa.Table.from_pandas(df, preserve_index=False)
    _check_roundtrip(empty, use_legacy_dataset=use_legacy_dataset)


@parametrize_legacy_dataset
def test_write_nested_zero_length_array_chunk_failure(use_legacy_dataset):
    # Bug report in ARROW-3792
    cols = OrderedDict(
        int32=pa.int32(),
        list_string=pa.list_(pa.string())
    )
    data = [[], [OrderedDict(int32=1, list_string=('G',)), ]]

    # This produces a table with a column like
    # <Column name='list_string' type=ListType(list<item: string>)>
    # [
    #   [],
    #   [
    #     [
    #       "G"
    #     ]
    #   ]
    # ]
    #
    # Each column is a ChunkedArray with 2 elements
    my_arrays = [pa.array(batch, type=pa.struct(cols)).flatten()
                 for batch in data]
    my_batches = [pa.RecordBatch.from_arrays(batch, schema=pa.schema(cols))
                  for batch in my_arrays]
    tbl = pa.Table.from_batches(my_batches, pa.schema(cols))
    _check_roundtrip(tbl, use_legacy_dataset=use_legacy_dataset)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_multiple_path_types(tempdir, use_legacy_dataset):
    # Test compatibility with PEP 519 path-like objects
    path = tempdir / 'zzz.parquet'
    df = pd.DataFrame({'x': np.arange(10, dtype=np.int64)})
    _write_table(df, path)
    table_read = _read_table(path, use_legacy_dataset=use_legacy_dataset)
    df_read = table_read.to_pandas()
    tm.assert_frame_equal(df, df_read)

    # Test compatibility with plain string paths
    path = str(tempdir) + 'zzz.parquet'
    df = pd.DataFrame({'x': np.arange(10, dtype=np.int64)})
    _write_table(df, path)
    table_read = _read_table(path, use_legacy_dataset=use_legacy_dataset)
    df_read = table_read.to_pandas()
    tm.assert_frame_equal(df, df_read)


@parametrize_legacy_dataset
def test_fspath(tempdir, use_legacy_dataset):
    # ARROW-12472 support __fspath__ objects without using str()
    path = tempdir / "test.parquet"
    table = pa.table({"a": [1, 2, 3]})
    _write_table(table, path)

    fs_protocol_obj = util.FSProtocolClass(path)

    result = _read_table(
        fs_protocol_obj, use_legacy_dataset=use_legacy_dataset
    )
    assert result.equals(table)

    # combined with non-local filesystem raises
    with pytest.raises(TypeError):
        _read_table(fs_protocol_obj, filesystem=FileSystem())


@pytest.mark.dataset
@parametrize_legacy_dataset
@pytest.mark.parametrize("filesystem", [
    None, fs.LocalFileSystem(), LocalFileSystem._get_instance()
])
@pytest.mark.parametrize("name", ("data.parquet", "例.parquet"))
def test_relative_paths(tempdir, use_legacy_dataset, filesystem, name):
    if use_legacy_dataset and isinstance(filesystem, fs.FileSystem):
        pytest.skip("Passing new filesystem not supported for legacy reader")
    # reading and writing from relative paths
    table = pa.table({"a": [1, 2, 3]})
    path = tempdir / name

    # reading
    pq.write_table(table, str(path))
    with util.change_cwd(tempdir):
        result = pq.read_table(name, filesystem=filesystem,
                               use_legacy_dataset=use_legacy_dataset)
    assert result.equals(table)

    path.unlink()
    assert not path.exists()

    # writing
    with util.change_cwd(tempdir):
        pq.write_table(table, name, filesystem=filesystem)
    result = pq.read_table(path)
    assert result.equals(table)


def test_read_non_existing_file():
    # ensure we have a proper error message
    with pytest.raises(FileNotFoundError):
        pq.read_table('i-am-not-existing.parquet')


def test_file_error_python_exception():
    class BogusFile(io.BytesIO):
        def read(self, *args):
            raise ZeroDivisionError("zorglub")

        def seek(self, *args):
            raise ZeroDivisionError("zorglub")

    # ensure the Python exception is restored
    with pytest.raises(ZeroDivisionError, match="zorglub"):
        pq.read_table(BogusFile(b""))


@parametrize_legacy_dataset
def test_parquet_read_from_buffer(tempdir, use_legacy_dataset):
    # reading from a buffer from python's open()
    table = pa.table({"a": [1, 2, 3]})
    pq.write_table(table, str(tempdir / "data.parquet"))

    with open(str(tempdir / "data.parquet"), "rb") as f:
        result = pq.read_table(f, use_legacy_dataset=use_legacy_dataset)
    assert result.equals(table)

    with open(str(tempdir / "data.parquet"), "rb") as f:
        result = pq.read_table(pa.PythonFile(f),
                               use_legacy_dataset=use_legacy_dataset)
    assert result.equals(table)


@parametrize_legacy_dataset
def test_byte_stream_split(use_legacy_dataset):
    # This is only a smoke test.
    arr_float = pa.array(list(map(float, range(100))))
    arr_int = pa.array(list(map(int, range(100))))
    data_float = [arr_float, arr_float]
    table = pa.Table.from_arrays(data_float, names=['a', 'b'])

    # Check with byte_stream_split for both columns.
    _check_roundtrip(table, expected=table, compression="gzip",
                     use_dictionary=False, use_byte_stream_split=True)

    # Check with byte_stream_split for column 'b' and dictionary
    # for column 'a'.
    _check_roundtrip(table, expected=table, compression="gzip",
                     use_dictionary=['a'],
                     use_byte_stream_split=['b'])

    # Check with a collision for both columns.
    _check_roundtrip(table, expected=table, compression="gzip",
                     use_dictionary=['a', 'b'],
                     use_byte_stream_split=['a', 'b'])

    # Check with mixed column types.
    mixed_table = pa.Table.from_arrays([arr_float, arr_int],
                                       names=['a', 'b'])
    _check_roundtrip(mixed_table, expected=mixed_table,
                     use_dictionary=['b'],
                     use_byte_stream_split=['a'])

    # Try to use the wrong data type with the byte_stream_split encoding.
    # This should throw an exception.
    table = pa.Table.from_arrays([arr_int], names=['tmp'])
    with pytest.raises(IOError):
        _check_roundtrip(table, expected=table, use_byte_stream_split=True,
                         use_dictionary=False,
                         use_legacy_dataset=use_legacy_dataset)


@parametrize_legacy_dataset
def test_column_encoding(use_legacy_dataset):
    arr_float = pa.array(list(map(float, range(100))))
    arr_int = pa.array(list(map(int, range(100))))
    mixed_table = pa.Table.from_arrays([arr_float, arr_int],
                                       names=['a', 'b'])

    # Check "BYTE_STREAM_SPLIT" for column 'a' and "PLAIN" column_encoding for
    # column 'b'.
    _check_roundtrip(mixed_table, expected=mixed_table, use_dictionary=False,
                     column_encoding={'a': "BYTE_STREAM_SPLIT", 'b': "PLAIN"},
                     use_legacy_dataset=use_legacy_dataset)

    # Check "PLAIN" for all columns.
    _check_roundtrip(mixed_table, expected=mixed_table,
                     use_dictionary=False,
                     column_encoding="PLAIN",
                     use_legacy_dataset=use_legacy_dataset)

    # Check "DELTA_BINARY_PACKED" for integer columns.
    _check_roundtrip(mixed_table, expected=mixed_table,
                     use_dictionary=False,
                     column_encoding={'a': "PLAIN",
                                      'b': "DELTA_BINARY_PACKED"},
                     use_legacy_dataset=use_legacy_dataset)

    # Try to pass "BYTE_STREAM_SPLIT" column encoding for integer column 'b'.
    # This should throw an error as it is only supports FLOAT and DOUBLE.
    with pytest.raises(IOError,
                       match="BYTE_STREAM_SPLIT only supports FLOAT and"
                             " DOUBLE"):
        _check_roundtrip(mixed_table, expected=mixed_table,
                         use_dictionary=False,
                         column_encoding={'b': "BYTE_STREAM_SPLIT"},
                         use_legacy_dataset=use_legacy_dataset)

    # Try to pass use "DELTA_BINARY_PACKED" encoding on float column.
    # This should throw an error as only integers are supported.
    with pytest.raises(OSError):
        _check_roundtrip(mixed_table, expected=mixed_table,
                         use_dictionary=False,
                         column_encoding={'a': "DELTA_BINARY_PACKED"},
                         use_legacy_dataset=use_legacy_dataset)

    # Try to pass "RLE_DICTIONARY".
    # This should throw an error as dictionary encoding is already used by
    # default and not supported to be specified as "fallback" encoding
    with pytest.raises(ValueError):
        _check_roundtrip(mixed_table, expected=mixed_table,
                         use_dictionary=False,
                         column_encoding="RLE_DICTIONARY",
                         use_legacy_dataset=use_legacy_dataset)

    # Try to pass unsupported encoding.
    with pytest.raises(ValueError):
        _check_roundtrip(mixed_table, expected=mixed_table,
                         use_dictionary=False,
                         column_encoding={'a': "MADE_UP_ENCODING"},
                         use_legacy_dataset=use_legacy_dataset)

    # Try to pass column_encoding and use_dictionary.
    # This should throw an error.
    with pytest.raises(ValueError):
        _check_roundtrip(mixed_table, expected=mixed_table,
                         use_dictionary=['b'],
                         column_encoding={'b': "PLAIN"},
                         use_legacy_dataset=use_legacy_dataset)

    # Try to pass column_encoding and use_dictionary=True (default value).
    # This should throw an error.
    with pytest.raises(ValueError):
        _check_roundtrip(mixed_table, expected=mixed_table,
                         column_encoding={'b': "PLAIN"},
                         use_legacy_dataset=use_legacy_dataset)

    # Try to pass column_encoding and use_byte_stream_split on same column.
    # This should throw an error.
    with pytest.raises(ValueError):
        _check_roundtrip(mixed_table, expected=mixed_table,
                         use_dictionary=False,
                         use_byte_stream_split=['a'],
                         column_encoding={'a': "RLE",
                                          'b': "BYTE_STREAM_SPLIT"},
                         use_legacy_dataset=use_legacy_dataset)

    # Try to pass column_encoding and use_byte_stream_split=True.
    # This should throw an error.
    with pytest.raises(ValueError):
        _check_roundtrip(mixed_table, expected=mixed_table,
                         use_dictionary=False,
                         use_byte_stream_split=True,
                         column_encoding={'a': "RLE",
                                          'b': "BYTE_STREAM_SPLIT"},
                         use_legacy_dataset=use_legacy_dataset)

    # Try to pass column_encoding=True.
    # This should throw an error.
    with pytest.raises(TypeError):
        _check_roundtrip(mixed_table, expected=mixed_table,
                         use_dictionary=False,
                         column_encoding=True,
                         use_legacy_dataset=use_legacy_dataset)


@parametrize_legacy_dataset
def test_compression_level(use_legacy_dataset):
    arr = pa.array(list(map(int, range(1000))))
    data = [arr, arr]
    table = pa.Table.from_arrays(data, names=['a', 'b'])

    # Check one compression level.
    _check_roundtrip(table, expected=table, compression="gzip",
                     compression_level=1,
                     use_legacy_dataset=use_legacy_dataset)

    # Check another one to make sure that compression_level=1 does not
    # coincide with the default one in Arrow.
    _check_roundtrip(table, expected=table, compression="gzip",
                     compression_level=5,
                     use_legacy_dataset=use_legacy_dataset)

    # Check that the user can provide a compression per column
    _check_roundtrip(table, expected=table,
                     compression={'a': "gzip", 'b': "snappy"},
                     use_legacy_dataset=use_legacy_dataset)

    # Check that the user can provide a compression level per column
    _check_roundtrip(table, expected=table, compression="gzip",
                     compression_level={'a': 2, 'b': 3},
                     use_legacy_dataset=use_legacy_dataset)

    # Check if both LZ4 compressors are working
    # (level < 3 -> fast, level >= 3 -> HC)
    _check_roundtrip(table, expected=table, compression="lz4",
                     compression_level=1,
                     use_legacy_dataset=use_legacy_dataset)

    _check_roundtrip(table, expected=table, compression="lz4",
                     compression_level=9,
                     use_legacy_dataset=use_legacy_dataset)

    # Check that specifying a compression level for a codec which does allow
    # specifying one, results into an error.
    # Uncompressed, snappy and lzo do not support specifying a compression
    # level.
    # GZIP (zlib) allows for specifying a compression level but as of up
    # to version 1.2.11 the valid range is [-1, 9].
    invalid_combinations = [("snappy", 4), ("gzip", -1337),
                            ("None", 444), ("lzo", 14)]
    buf = io.BytesIO()
    for (codec, level) in invalid_combinations:
        with pytest.raises((ValueError, OSError)):
            _write_table(table, buf, compression=codec,
                         compression_level=level)


def test_sanitized_spark_field_names():
    a0 = pa.array([0, 1, 2, 3, 4])
    name = 'prohib; ,\t{}'
    table = pa.Table.from_arrays([a0], [name])

    result = _roundtrip_table(table, write_table_kwargs={'flavor': 'spark'})

    expected_name = 'prohib______'
    assert result.schema[0].name == expected_name


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_multithreaded_read(use_legacy_dataset):
    df = alltypes_sample(size=10000)

    table = pa.Table.from_pandas(df)

    buf = io.BytesIO()
    _write_table(table, buf, compression='SNAPPY', version='2.6')

    buf.seek(0)
    table1 = _read_table(
        buf, use_threads=True, use_legacy_dataset=use_legacy_dataset)

    buf.seek(0)
    table2 = _read_table(
        buf, use_threads=False, use_legacy_dataset=use_legacy_dataset)

    assert table1.equals(table2)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_min_chunksize(use_legacy_dataset):
    data = pd.DataFrame([np.arange(4)], columns=['A', 'B', 'C', 'D'])
    table = pa.Table.from_pandas(data.reset_index())

    buf = io.BytesIO()
    _write_table(table, buf, chunk_size=-1)

    buf.seek(0)
    result = _read_table(buf, use_legacy_dataset=use_legacy_dataset)

    assert result.equals(table)

    with pytest.raises(ValueError):
        _write_table(table, buf, chunk_size=0)


@pytest.mark.pandas
def test_write_error_deletes_incomplete_file(tempdir):
    # ARROW-1285
    df = pd.DataFrame({'a': list('abc'),
                       'b': list(range(1, 4)),
                       'c': np.arange(3, 6).astype('u1'),
                       'd': np.arange(4.0, 7.0, dtype='float64'),
                       'e': [True, False, True],
                       'f': pd.Categorical(list('abc')),
                       'g': pd.date_range('20130101', periods=3),
                       'h': pd.date_range('20130101', periods=3,
                                          tz='US/Eastern'),
                       'i': pd.date_range('20130101', periods=3, freq='ns')})

    pdf = pa.Table.from_pandas(df)

    filename = tempdir / 'tmp_file'
    try:
        _write_table(pdf, filename)
    except pa.ArrowException:
        pass

    assert not filename.exists()


@parametrize_legacy_dataset
def test_read_non_existent_file(tempdir, use_legacy_dataset):
    path = 'non-existent-file.parquet'
    try:
        pq.read_table(path, use_legacy_dataset=use_legacy_dataset)
    except Exception as e:
        assert path in e.args[0]


@parametrize_legacy_dataset
def test_read_table_doesnt_warn(datadir, use_legacy_dataset):
    if use_legacy_dataset:
        msg = "Passing 'use_legacy_dataset=True'"
        with pytest.warns(FutureWarning, match=msg):
            pq.read_table(datadir / 'v0.7.1.parquet',
                          use_legacy_dataset=use_legacy_dataset)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter(action="error")
            pq.read_table(datadir / 'v0.7.1.parquet',
                          use_legacy_dataset=use_legacy_dataset)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_zlib_compression_bug(use_legacy_dataset):
    # ARROW-3514: "zlib deflate failed, output buffer too small"
    table = pa.Table.from_arrays([pa.array(['abc', 'def'])], ['some_col'])
    f = io.BytesIO()
    pq.write_table(table, f, compression='gzip')

    f.seek(0)
    roundtrip = pq.read_table(f, use_legacy_dataset=use_legacy_dataset)
    tm.assert_frame_equal(roundtrip.to_pandas(), table.to_pandas())


@parametrize_legacy_dataset
def test_parquet_file_too_small(tempdir, use_legacy_dataset):
    path = str(tempdir / "test.parquet")
    # TODO(dataset) with datasets API it raises OSError instead
    with pytest.raises((pa.ArrowInvalid, OSError),
                       match='size is 0 bytes'):
        with open(path, 'wb') as f:
            pass
        pq.read_table(path, use_legacy_dataset=use_legacy_dataset)

    with pytest.raises((pa.ArrowInvalid, OSError),
                       match='size is 4 bytes'):
        with open(path, 'wb') as f:
            f.write(b'ffff')
        pq.read_table(path, use_legacy_dataset=use_legacy_dataset)


@pytest.mark.pandas
@pytest.mark.fastparquet
@pytest.mark.filterwarnings("ignore:RangeIndex:FutureWarning")
@pytest.mark.filterwarnings("ignore:tostring:DeprecationWarning:fastparquet")
def test_fastparquet_cross_compatibility(tempdir):
    fp = pytest.importorskip('fastparquet')

    df = pd.DataFrame(
        {
            "a": list("abc"),
            "b": list(range(1, 4)),
            "c": np.arange(4.0, 7.0, dtype="float64"),
            "d": [True, False, True],
            "e": pd.date_range("20130101", periods=3),
            "f": pd.Categorical(["a", "b", "a"]),
            # fastparquet writes list as BYTE_ARRAY JSON, so no roundtrip
            # "g": [[1, 2], None, [1, 2, 3]],
        }
    )
    table = pa.table(df)

    # Arrow -> fastparquet
    file_arrow = str(tempdir / "cross_compat_arrow.parquet")
    pq.write_table(table, file_arrow, compression=None)

    fp_file = fp.ParquetFile(file_arrow)
    df_fp = fp_file.to_pandas()
    tm.assert_frame_equal(df, df_fp)

    # Fastparquet -> arrow
    file_fastparquet = str(tempdir / "cross_compat_fastparquet.parquet")
    fp.write(file_fastparquet, df)

    table_fp = pq.read_pandas(file_fastparquet)
    # for fastparquet written file, categoricals comes back as strings
    # (no arrow schema in parquet metadata)
    df['f'] = df['f'].astype(object)
    tm.assert_frame_equal(table_fp.to_pandas(), df)


@parametrize_legacy_dataset
@pytest.mark.parametrize('array_factory', [
    lambda: pa.array([0, None] * 10),
    lambda: pa.array([0, None] * 10).dictionary_encode(),
    lambda: pa.array(["", None] * 10),
    lambda: pa.array(["", None] * 10).dictionary_encode(),
])
@pytest.mark.parametrize('use_dictionary', [False, True])
@pytest.mark.parametrize('read_dictionary', [False, True])
def test_buffer_contents(
        array_factory, use_dictionary, read_dictionary, use_legacy_dataset
):
    # Test that null values are deterministically initialized to zero
    # after a roundtrip through Parquet.
    # See ARROW-8006 and ARROW-8011.
    orig_table = pa.Table.from_pydict({"col": array_factory()})
    bio = io.BytesIO()
    pq.write_table(orig_table, bio, use_dictionary=True)
    bio.seek(0)
    read_dictionary = ['col'] if read_dictionary else None
    table = pq.read_table(bio, use_threads=False,
                          read_dictionary=read_dictionary,
                          use_legacy_dataset=use_legacy_dataset)

    for col in table.columns:
        [chunk] = col.chunks
        buf = chunk.buffers()[1]
        assert buf.to_pybytes() == buf.size * b"\0"


def test_parquet_compression_roundtrip(tempdir):
    # ARROW-10480: ensure even with nonstandard Parquet file naming
    # conventions, writing and then reading a file works. In
    # particular, ensure that we don't automatically double-compress
    # the stream due to auto-detecting the extension in the filename
    table = pa.table([pa.array(range(4))], names=["ints"])
    path = tempdir / "arrow-10480.pyarrow.gz"
    pq.write_table(table, path, compression="GZIP")
    result = pq.read_table(path)
    assert result.equals(table)


def test_empty_row_groups(tempdir):
    # ARROW-3020
    table = pa.Table.from_arrays([pa.array([], type='int32')], ['f0'])

    path = tempdir / 'empty_row_groups.parquet'

    num_groups = 3
    with pq.ParquetWriter(path, table.schema) as writer:
        for i in range(num_groups):
            writer.write_table(table)

    reader = pq.ParquetFile(path)
    assert reader.metadata.num_row_groups == num_groups

    for i in range(num_groups):
        assert reader.read_row_group(i).equals(table)


def test_reads_over_batch(tempdir):
    data = [None] * (1 << 20)
    data.append([1])
    # Large list<int64> with mostly nones and one final
    # value.  This should force batched reads when
    # reading back.
    table = pa.Table.from_arrays([data], ['column'])

    path = tempdir / 'arrow-11607.parquet'
    pq.write_table(table, path)
    table2 = pq.read_table(path)
    assert table == table2


@pytest.mark.dataset
def test_permutation_of_column_order(tempdir):
    # ARROW-2366
    case = tempdir / "dataset_column_order_permutation"
    case.mkdir(exist_ok=True)

    data1 = pa.table([[1, 2, 3], [.1, .2, .3]], names=['a', 'b'])
    pq.write_table(data1, case / "data1.parquet")

    data2 = pa.table([[.4, .5, .6], [4, 5, 6]], names=['b', 'a'])
    pq.write_table(data2, case / "data2.parquet")

    table = pq.read_table(str(case))
    table2 = pa.table([[1, 2, 3, 4, 5, 6],
                       [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
                      names=['a', 'b'])

    assert table == table2


def test_read_table_legacy_deprecated(tempdir):
    # ARROW-15870
    table = pa.table({'a': [1, 2, 3]})
    path = tempdir / 'data.parquet'
    pq.write_table(table, path)

    with pytest.warns(
        FutureWarning, match="Passing 'use_legacy_dataset=True'"
    ):
        pq.read_table(path, use_legacy_dataset=True)


def test_thrift_size_limits(tempdir):
    path = tempdir / 'largethrift.parquet'

    array = pa.array(list(range(10)))
    num_cols = 1000
    table = pa.table(
        [array] * num_cols,
        names=[f'some_long_column_name_{i}' for i in range(num_cols)])
    pq.write_table(table, path)

    with pytest.raises(
            OSError,
            match="Couldn't deserialize thrift:.*Exceeded size limit"):
        pq.read_table(path, thrift_string_size_limit=50 * num_cols)
    with pytest.raises(
            OSError,
            match="Couldn't deserialize thrift:.*Exceeded size limit"):
        pq.read_table(path, thrift_container_size_limit=num_cols)

    got = pq.read_table(path, thrift_string_size_limit=100 * num_cols)
    assert got == table
    got = pq.read_table(path, thrift_container_size_limit=2 * num_cols)
    assert got == table
    got = pq.read_table(path)
    assert got == table

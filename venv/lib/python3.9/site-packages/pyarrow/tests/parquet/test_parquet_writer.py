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

import pytest

import pyarrow as pa
from pyarrow import fs
from pyarrow.filesystem import FileSystem, LocalFileSystem
from pyarrow.tests.parquet.common import parametrize_legacy_dataset

try:
    import pyarrow.parquet as pq
    from pyarrow.tests.parquet.common import _read_table, _test_dataframe
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
@parametrize_legacy_dataset
def test_parquet_incremental_file_build(tempdir, use_legacy_dataset):
    df = _test_dataframe(100)
    df['unique_id'] = 0

    arrow_table = pa.Table.from_pandas(df, preserve_index=False)
    out = pa.BufferOutputStream()

    writer = pq.ParquetWriter(out, arrow_table.schema, version='2.6')

    frames = []
    for i in range(10):
        df['unique_id'] = i
        arrow_table = pa.Table.from_pandas(df, preserve_index=False)
        writer.write_table(arrow_table)

        frames.append(df.copy())

    writer.close()

    buf = out.getvalue()
    result = _read_table(
        pa.BufferReader(buf), use_legacy_dataset=use_legacy_dataset)

    expected = pd.concat(frames, ignore_index=True)
    tm.assert_frame_equal(result.to_pandas(), expected)


def test_validate_schema_write_table(tempdir):
    # ARROW-2926
    simple_fields = [
        pa.field('POS', pa.uint32()),
        pa.field('desc', pa.string())
    ]

    simple_schema = pa.schema(simple_fields)

    # simple_table schema does not match simple_schema
    simple_from_array = [pa.array([1]), pa.array(['bla'])]
    simple_table = pa.Table.from_arrays(simple_from_array, ['POS', 'desc'])

    path = tempdir / 'simple_validate_schema.parquet'

    with pq.ParquetWriter(path, simple_schema,
                          version='2.6',
                          compression='snappy', flavor='spark') as w:
        with pytest.raises(ValueError):
            w.write_table(simple_table)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_parquet_writer_context_obj(tempdir, use_legacy_dataset):
    df = _test_dataframe(100)
    df['unique_id'] = 0

    arrow_table = pa.Table.from_pandas(df, preserve_index=False)
    out = pa.BufferOutputStream()

    with pq.ParquetWriter(out, arrow_table.schema, version='2.6') as writer:

        frames = []
        for i in range(10):
            df['unique_id'] = i
            arrow_table = pa.Table.from_pandas(df, preserve_index=False)
            writer.write_table(arrow_table)

            frames.append(df.copy())

    buf = out.getvalue()
    result = _read_table(
        pa.BufferReader(buf), use_legacy_dataset=use_legacy_dataset)

    expected = pd.concat(frames, ignore_index=True)
    tm.assert_frame_equal(result.to_pandas(), expected)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_parquet_writer_context_obj_with_exception(
    tempdir, use_legacy_dataset
):
    df = _test_dataframe(100)
    df['unique_id'] = 0

    arrow_table = pa.Table.from_pandas(df, preserve_index=False)
    out = pa.BufferOutputStream()
    error_text = 'Artificial Error'

    try:
        with pq.ParquetWriter(out,
                              arrow_table.schema,
                              version='2.6') as writer:

            frames = []
            for i in range(10):
                df['unique_id'] = i
                arrow_table = pa.Table.from_pandas(df, preserve_index=False)
                writer.write_table(arrow_table)
                frames.append(df.copy())
                if i == 5:
                    raise ValueError(error_text)
    except Exception as e:
        assert str(e) == error_text

    buf = out.getvalue()
    result = _read_table(
        pa.BufferReader(buf), use_legacy_dataset=use_legacy_dataset)

    expected = pd.concat(frames, ignore_index=True)
    tm.assert_frame_equal(result.to_pandas(), expected)


@pytest.mark.pandas
@pytest.mark.parametrize("filesystem", [
    None,
    LocalFileSystem._get_instance(),
    fs.LocalFileSystem(),
])
def test_parquet_writer_write_wrappers(tempdir, filesystem):
    df = _test_dataframe(100)
    table = pa.Table.from_pandas(df, preserve_index=False)
    batch = pa.RecordBatch.from_pandas(df, preserve_index=False)
    path_table = str(tempdir / 'data_table.parquet')
    path_batch = str(tempdir / 'data_batch.parquet')

    with pq.ParquetWriter(
        path_table, table.schema, filesystem=filesystem, version='2.6'
    ) as writer:
        writer.write_table(table)

    result = _read_table(path_table).to_pandas()
    tm.assert_frame_equal(result, df)

    with pq.ParquetWriter(
        path_batch, table.schema, filesystem=filesystem, version='2.6'
    ) as writer:
        writer.write_batch(batch)

    result = _read_table(path_batch).to_pandas()
    tm.assert_frame_equal(result, df)

    with pq.ParquetWriter(
        path_table, table.schema, filesystem=filesystem, version='2.6'
    ) as writer:
        writer.write(table)

    result = _read_table(path_table).to_pandas()
    tm.assert_frame_equal(result, df)

    with pq.ParquetWriter(
        path_batch, table.schema, filesystem=filesystem, version='2.6'
    ) as writer:
        writer.write(batch)

    result = _read_table(path_batch).to_pandas()
    tm.assert_frame_equal(result, df)


@pytest.mark.pandas
@pytest.mark.parametrize("filesystem", [
    None,
    LocalFileSystem._get_instance(),
    fs.LocalFileSystem(),
])
def test_parquet_writer_filesystem_local(tempdir, filesystem):
    df = _test_dataframe(100)
    table = pa.Table.from_pandas(df, preserve_index=False)
    path = str(tempdir / 'data.parquet')

    with pq.ParquetWriter(
        path, table.schema, filesystem=filesystem, version='2.6'
    ) as writer:
        writer.write_table(table)

    result = _read_table(path).to_pandas()
    tm.assert_frame_equal(result, df)


@pytest.mark.pandas
@pytest.mark.s3
def test_parquet_writer_filesystem_s3(s3_example_fs):
    df = _test_dataframe(100)
    table = pa.Table.from_pandas(df, preserve_index=False)

    fs, uri, path = s3_example_fs

    with pq.ParquetWriter(
        path, table.schema, filesystem=fs, version='2.6'
    ) as writer:
        writer.write_table(table)

    result = _read_table(uri).to_pandas()
    tm.assert_frame_equal(result, df)


@pytest.mark.pandas
@pytest.mark.s3
def test_parquet_writer_filesystem_s3_uri(s3_example_fs):
    df = _test_dataframe(100)
    table = pa.Table.from_pandas(df, preserve_index=False)

    fs, uri, path = s3_example_fs

    with pq.ParquetWriter(uri, table.schema, version='2.6') as writer:
        writer.write_table(table)

    result = _read_table(path, filesystem=fs).to_pandas()
    tm.assert_frame_equal(result, df)


@pytest.mark.pandas
@pytest.mark.s3
def test_parquet_writer_filesystem_s3fs(s3_example_s3fs):
    df = _test_dataframe(100)
    table = pa.Table.from_pandas(df, preserve_index=False)

    fs, directory = s3_example_s3fs
    path = directory + "/test.parquet"

    with pq.ParquetWriter(
        path, table.schema, filesystem=fs, version='2.6'
    ) as writer:
        writer.write_table(table)

    result = _read_table(path, filesystem=fs).to_pandas()
    tm.assert_frame_equal(result, df)


@pytest.mark.pandas
def test_parquet_writer_filesystem_buffer_raises():
    df = _test_dataframe(100)
    table = pa.Table.from_pandas(df, preserve_index=False)
    filesystem = fs.LocalFileSystem()

    # Should raise ValueError when filesystem is passed with file-like object
    with pytest.raises(ValueError, match="specified path is file-like"):
        pq.ParquetWriter(
            pa.BufferOutputStream(), table.schema, filesystem=filesystem
        )


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_parquet_writer_with_caller_provided_filesystem(use_legacy_dataset):
    out = pa.BufferOutputStream()

    class CustomFS(FileSystem):
        def __init__(self):
            self.path = None
            self.mode = None

        def open(self, path, mode='rb'):
            self.path = path
            self.mode = mode
            return out

    fs = CustomFS()
    fname = 'expected_fname.parquet'
    df = _test_dataframe(100)
    table = pa.Table.from_pandas(df, preserve_index=False)

    with pq.ParquetWriter(fname, table.schema, filesystem=fs, version='2.6') \
            as writer:
        writer.write_table(table)

    assert fs.path == fname
    assert fs.mode == 'wb'
    assert out.closed

    buf = out.getvalue()
    table_read = _read_table(
        pa.BufferReader(buf), use_legacy_dataset=use_legacy_dataset)
    df_read = table_read.to_pandas()
    tm.assert_frame_equal(df_read, df)

    # Should raise ValueError when filesystem is passed with file-like object
    with pytest.raises(ValueError) as err_info:
        pq.ParquetWriter(pa.BufferOutputStream(), table.schema, filesystem=fs)
        expected_msg = ("filesystem passed but where is file-like, so"
                        " there is nothing to open with filesystem.")
        assert str(err_info) == expected_msg


def test_parquet_writer_store_schema(tempdir):
    table = pa.table({'a': [1, 2, 3]})

    # default -> write schema information
    path1 = tempdir / 'test_with_schema.parquet'
    with pq.ParquetWriter(path1, table.schema) as writer:
        writer.write_table(table)

    meta = pq.read_metadata(path1)
    assert b'ARROW:schema' in meta.metadata
    assert meta.metadata[b'ARROW:schema']

    # disable adding schema information
    path2 = tempdir / 'test_without_schema.parquet'
    with pq.ParquetWriter(path2, table.schema, store_schema=False) as writer:
        writer.write_table(table)

    meta = pq.read_metadata(path2)
    assert meta.metadata is None

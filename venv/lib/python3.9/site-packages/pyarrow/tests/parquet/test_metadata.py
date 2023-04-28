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
import decimal
from collections import OrderedDict
import io

import numpy as np
import pytest

import pyarrow as pa
from pyarrow.tests.parquet.common import _check_roundtrip, make_sample_file
from pyarrow.fs import LocalFileSystem
from pyarrow.tests import util

try:
    import pyarrow.parquet as pq
    from pyarrow.tests.parquet.common import _write_table
except ImportError:
    pq = None


try:
    import pandas as pd
    import pandas.testing as tm

    from pyarrow.tests.parquet.common import alltypes_sample
except ImportError:
    pd = tm = None


# Marks all of the tests in this module
# Ignore these with pytest ... -m 'not parquet'
pytestmark = pytest.mark.parquet


@pytest.mark.pandas
def test_parquet_metadata_api():
    df = alltypes_sample(size=10000)
    df = df.reindex(columns=sorted(df.columns))
    df.index = np.random.randint(0, 1000000, size=len(df))

    fileh = make_sample_file(df)
    ncols = len(df.columns)

    # Series of sniff tests
    meta = fileh.metadata
    repr(meta)
    assert meta.num_rows == len(df)
    assert meta.num_columns == ncols + 1  # +1 for index
    assert meta.num_row_groups == 1
    assert meta.format_version == '2.6'
    assert 'parquet-cpp' in meta.created_by
    assert isinstance(meta.serialized_size, int)
    assert isinstance(meta.metadata, dict)

    # Schema
    schema = fileh.schema
    assert meta.schema is schema
    assert len(schema) == ncols + 1  # +1 for index
    repr(schema)

    col = schema[0]
    repr(col)
    assert col.name == df.columns[0]
    assert col.max_definition_level == 1
    assert col.max_repetition_level == 0
    assert col.max_repetition_level == 0

    assert col.physical_type == 'BOOLEAN'
    assert col.converted_type == 'NONE'

    with pytest.raises(IndexError):
        schema[ncols + 1]  # +1 for index

    with pytest.raises(IndexError):
        schema[-1]

    # Row group
    for rg in range(meta.num_row_groups):
        rg_meta = meta.row_group(rg)
        assert isinstance(rg_meta, pq.RowGroupMetaData)
        repr(rg_meta)

        for col in range(rg_meta.num_columns):
            col_meta = rg_meta.column(col)
            assert isinstance(col_meta, pq.ColumnChunkMetaData)
            repr(col_meta)

    with pytest.raises(IndexError):
        meta.row_group(-1)

    with pytest.raises(IndexError):
        meta.row_group(meta.num_row_groups + 1)

    rg_meta = meta.row_group(0)
    assert rg_meta.num_rows == len(df)
    assert rg_meta.num_columns == ncols + 1  # +1 for index
    assert rg_meta.total_byte_size > 0

    with pytest.raises(IndexError):
        col_meta = rg_meta.column(-1)

    with pytest.raises(IndexError):
        col_meta = rg_meta.column(ncols + 2)

    col_meta = rg_meta.column(0)
    assert col_meta.file_offset > 0
    assert col_meta.file_path == ''  # created from BytesIO
    assert col_meta.physical_type == 'BOOLEAN'
    assert col_meta.num_values == 10000
    assert col_meta.path_in_schema == 'bool'
    assert col_meta.is_stats_set is True
    assert isinstance(col_meta.statistics, pq.Statistics)
    assert col_meta.compression == 'SNAPPY'
    assert col_meta.encodings == ('PLAIN', 'RLE')
    assert col_meta.has_dictionary_page is False
    assert col_meta.dictionary_page_offset is None
    assert col_meta.data_page_offset > 0
    assert col_meta.total_compressed_size > 0
    assert col_meta.total_uncompressed_size > 0
    with pytest.raises(NotImplementedError):
        col_meta.has_index_page
    with pytest.raises(NotImplementedError):
        col_meta.index_page_offset


def test_parquet_metadata_lifetime(tempdir):
    # ARROW-6642 - ensure that chained access keeps parent objects alive
    table = pa.table({'a': [1, 2, 3]})
    pq.write_table(table, tempdir / 'test_metadata_segfault.parquet')
    parquet_file = pq.ParquetFile(tempdir / 'test_metadata_segfault.parquet')
    parquet_file.metadata.row_group(0).column(0).statistics


@pytest.mark.pandas
@pytest.mark.parametrize(
    (
        'data',
        'type',
        'physical_type',
        'min_value',
        'max_value',
        'null_count',
        'num_values',
        'distinct_count'
    ),
    [
        ([1, 2, 2, None, 4], pa.uint8(), 'INT32', 1, 4, 1, 4, 0),
        ([1, 2, 2, None, 4], pa.uint16(), 'INT32', 1, 4, 1, 4, 0),
        ([1, 2, 2, None, 4], pa.uint32(), 'INT32', 1, 4, 1, 4, 0),
        ([1, 2, 2, None, 4], pa.uint64(), 'INT64', 1, 4, 1, 4, 0),
        ([-1, 2, 2, None, 4], pa.int8(), 'INT32', -1, 4, 1, 4, 0),
        ([-1, 2, 2, None, 4], pa.int16(), 'INT32', -1, 4, 1, 4, 0),
        ([-1, 2, 2, None, 4], pa.int32(), 'INT32', -1, 4, 1, 4, 0),
        ([-1, 2, 2, None, 4], pa.int64(), 'INT64', -1, 4, 1, 4, 0),
        (
            [-1.1, 2.2, 2.3, None, 4.4], pa.float32(),
            'FLOAT', -1.1, 4.4, 1, 4, 0
        ),
        (
            [-1.1, 2.2, 2.3, None, 4.4], pa.float64(),
            'DOUBLE', -1.1, 4.4, 1, 4, 0
        ),
        (
            ['', 'b', chr(1000), None, 'aaa'], pa.binary(),
            'BYTE_ARRAY', b'', chr(1000).encode('utf-8'), 1, 4, 0
        ),
        (
            [True, False, False, True, True], pa.bool_(),
            'BOOLEAN', False, True, 0, 5, 0
        ),
        (
            [b'\x00', b'b', b'12', None, b'aaa'], pa.binary(),
            'BYTE_ARRAY', b'\x00', b'b', 1, 4, 0
        ),
    ]
)
def test_parquet_column_statistics_api(data, type, physical_type, min_value,
                                       max_value, null_count, num_values,
                                       distinct_count):
    df = pd.DataFrame({'data': data})
    schema = pa.schema([pa.field('data', type)])
    table = pa.Table.from_pandas(df, schema=schema, safe=False)
    fileh = make_sample_file(table)

    meta = fileh.metadata

    rg_meta = meta.row_group(0)
    col_meta = rg_meta.column(0)

    stat = col_meta.statistics
    assert stat.has_min_max
    assert _close(type, stat.min, min_value)
    assert _close(type, stat.max, max_value)
    assert stat.null_count == null_count
    assert stat.num_values == num_values
    # TODO(kszucs) until parquet-cpp API doesn't expose HasDistinctCount
    # method, missing distinct_count is represented as zero instead of None
    assert stat.distinct_count == distinct_count
    assert stat.physical_type == physical_type


def _close(type, left, right):
    if type == pa.float32():
        return abs(left - right) < 1E-7
    elif type == pa.float64():
        return abs(left - right) < 1E-13
    else:
        return left == right


# ARROW-6339
@pytest.mark.pandas
def test_parquet_raise_on_unset_statistics():
    df = pd.DataFrame({"t": pd.Series([pd.NaT], dtype="datetime64[ns]")})
    meta = make_sample_file(pa.Table.from_pandas(df)).metadata

    assert not meta.row_group(0).column(0).statistics.has_min_max
    assert meta.row_group(0).column(0).statistics.max is None


def test_statistics_convert_logical_types(tempdir):
    # ARROW-5166, ARROW-4139

    # (min, max, type)
    cases = [(10, 11164359321221007157, pa.uint64()),
             (10, 4294967295, pa.uint32()),
             ("ähnlich", "öffentlich", pa.utf8()),
             (datetime.time(10, 30, 0, 1000), datetime.time(15, 30, 0, 1000),
              pa.time32('ms')),
             (datetime.time(10, 30, 0, 1000), datetime.time(15, 30, 0, 1000),
              pa.time64('us')),
             (datetime.datetime(2019, 6, 24, 0, 0, 0, 1000),
              datetime.datetime(2019, 6, 25, 0, 0, 0, 1000),
              pa.timestamp('ms')),
             (datetime.datetime(2019, 6, 24, 0, 0, 0, 1000),
              datetime.datetime(2019, 6, 25, 0, 0, 0, 1000),
              pa.timestamp('us')),
             (datetime.date(2019, 6, 24),
              datetime.date(2019, 6, 25),
              pa.date32()),
             (decimal.Decimal("20.123"),
              decimal.Decimal("20.124"),
              pa.decimal128(12, 5))]

    for i, (min_val, max_val, typ) in enumerate(cases):
        t = pa.Table.from_arrays([pa.array([min_val, max_val], type=typ)],
                                 ['col'])
        path = str(tempdir / ('example{}.parquet'.format(i)))
        pq.write_table(t, path, version='2.6')
        pf = pq.ParquetFile(path)
        stats = pf.metadata.row_group(0).column(0).statistics
        assert stats.min == min_val
        assert stats.max == max_val


def test_parquet_write_disable_statistics(tempdir):
    table = pa.Table.from_pydict(
        OrderedDict([
            ('a', pa.array([1, 2, 3])),
            ('b', pa.array(['a', 'b', 'c']))
        ])
    )
    _write_table(table, tempdir / 'data.parquet')
    meta = pq.read_metadata(tempdir / 'data.parquet')
    for col in [0, 1]:
        cc = meta.row_group(0).column(col)
        assert cc.is_stats_set is True
        assert cc.statistics is not None

    _write_table(table, tempdir / 'data2.parquet', write_statistics=False)
    meta = pq.read_metadata(tempdir / 'data2.parquet')
    for col in [0, 1]:
        cc = meta.row_group(0).column(col)
        assert cc.is_stats_set is False
        assert cc.statistics is None

    _write_table(table, tempdir / 'data3.parquet', write_statistics=['a'])
    meta = pq.read_metadata(tempdir / 'data3.parquet')
    cc_a = meta.row_group(0).column(0)
    cc_b = meta.row_group(0).column(1)
    assert cc_a.is_stats_set is True
    assert cc_b.is_stats_set is False
    assert cc_a.statistics is not None
    assert cc_b.statistics is None


def test_field_id_metadata():
    # ARROW-7080
    field_id = b'PARQUET:field_id'
    inner = pa.field('inner', pa.int32(), metadata={field_id: b'100'})
    middle = pa.field('middle', pa.struct(
        [inner]), metadata={field_id: b'101'})
    fields = [
        pa.field('basic', pa.int32(), metadata={
                 b'other': b'abc', field_id: b'1'}),
        pa.field(
            'list',
            pa.list_(pa.field('list-inner', pa.int32(),
                              metadata={field_id: b'10'})),
            metadata={field_id: b'11'}),
        pa.field('struct', pa.struct([middle]), metadata={field_id: b'102'}),
        pa.field('no-metadata', pa.int32()),
        pa.field('non-integral-field-id', pa.int32(),
                 metadata={field_id: b'xyz'}),
        pa.field('negative-field-id', pa.int32(),
                 metadata={field_id: b'-1000'})
    ]
    arrs = [[] for _ in fields]
    table = pa.table(arrs, schema=pa.schema(fields))

    bio = pa.BufferOutputStream()
    pq.write_table(table, bio)
    contents = bio.getvalue()

    pf = pq.ParquetFile(pa.BufferReader(contents))
    schema = pf.schema_arrow

    assert schema[0].metadata[field_id] == b'1'
    assert schema[0].metadata[b'other'] == b'abc'

    list_field = schema[1]
    assert list_field.metadata[field_id] == b'11'

    list_item_field = list_field.type.value_field
    assert list_item_field.metadata[field_id] == b'10'

    struct_field = schema[2]
    assert struct_field.metadata[field_id] == b'102'

    struct_middle_field = struct_field.type[0]
    assert struct_middle_field.metadata[field_id] == b'101'

    struct_inner_field = struct_middle_field.type[0]
    assert struct_inner_field.metadata[field_id] == b'100'

    assert schema[3].metadata is None
    # Invalid input is passed through (ok) but does not
    # have field_id in parquet (not tested)
    assert schema[4].metadata[field_id] == b'xyz'
    assert schema[5].metadata[field_id] == b'-1000'


@pytest.mark.pandas
def test_multi_dataset_metadata(tempdir):
    filenames = ["ARROW-1983-dataset.0", "ARROW-1983-dataset.1"]
    metapath = str(tempdir / "_metadata")

    # create a test dataset
    df = pd.DataFrame({
        'one': [1, 2, 3],
        'two': [-1, -2, -3],
        'three': [[1, 2], [2, 3], [3, 4]],
    })
    table = pa.Table.from_pandas(df)

    # write dataset twice and collect/merge metadata
    _meta = None
    for filename in filenames:
        meta = []
        pq.write_table(table, str(tempdir / filename),
                       metadata_collector=meta)
        meta[0].set_file_path(filename)
        if _meta is None:
            _meta = meta[0]
        else:
            _meta.append_row_groups(meta[0])

    # Write merged metadata-only file
    with open(metapath, "wb") as f:
        _meta.write_metadata_file(f)

    # Read back the metadata
    meta = pq.read_metadata(metapath)
    md = meta.to_dict()
    _md = _meta.to_dict()
    for key in _md:
        if key != 'serialized_size':
            assert _md[key] == md[key]
    assert _md['num_columns'] == 3
    assert _md['num_rows'] == 6
    assert _md['num_row_groups'] == 2
    assert _md['serialized_size'] == 0
    assert md['serialized_size'] > 0


@pytest.mark.filterwarnings("ignore:Parquet format:FutureWarning")
def test_write_metadata(tempdir):
    path = str(tempdir / "metadata")
    schema = pa.schema([("a", "int64"), ("b", "float64")])

    # write a pyarrow schema
    pq.write_metadata(schema, path)
    parquet_meta = pq.read_metadata(path)
    schema_as_arrow = parquet_meta.schema.to_arrow_schema()
    assert schema_as_arrow.equals(schema)

    # ARROW-8980: Check that the ARROW:schema metadata key was removed
    if schema_as_arrow.metadata:
        assert b'ARROW:schema' not in schema_as_arrow.metadata

    # pass through writer keyword arguments
    for version in ["1.0", "2.0", "2.4", "2.6"]:
        pq.write_metadata(schema, path, version=version)
        parquet_meta = pq.read_metadata(path)
        # The version is stored as a single integer in the Parquet metadata,
        # so it cannot correctly express dotted format versions
        expected_version = "1.0" if version == "1.0" else "2.6"
        assert parquet_meta.format_version == expected_version

    # metadata_collector: list of FileMetaData objects
    table = pa.table({'a': [1, 2], 'b': [.1, .2]}, schema=schema)
    pq.write_table(table, tempdir / "data.parquet")
    parquet_meta = pq.read_metadata(str(tempdir / "data.parquet"))
    pq.write_metadata(
        schema, path, metadata_collector=[parquet_meta, parquet_meta]
    )
    parquet_meta_mult = pq.read_metadata(path)
    assert parquet_meta_mult.num_row_groups == 2

    # append metadata with different schema raises an error
    msg = ("AppendRowGroups requires equal schemas.\n"
           "The two columns with index 0 differ.")
    with pytest.raises(RuntimeError, match=msg):
        pq.write_metadata(
            pa.schema([("a", "int32"), ("b", "null")]),
            path, metadata_collector=[parquet_meta, parquet_meta]
        )


def test_table_large_metadata():
    # ARROW-8694
    my_schema = pa.schema([pa.field('f0', 'double')],
                          metadata={'large': 'x' * 10000000})

    table = pa.table([np.arange(10)], schema=my_schema)
    _check_roundtrip(table)


@pytest.mark.pandas
def test_compare_schemas():
    df = alltypes_sample(size=10000)

    fileh = make_sample_file(df)
    fileh2 = make_sample_file(df)
    fileh3 = make_sample_file(df[df.columns[::2]])

    # ParquetSchema
    assert isinstance(fileh.schema, pq.ParquetSchema)
    assert fileh.schema.equals(fileh.schema)
    assert fileh.schema == fileh.schema
    assert fileh.schema.equals(fileh2.schema)
    assert fileh.schema == fileh2.schema
    assert fileh.schema != 'arbitrary object'
    assert not fileh.schema.equals(fileh3.schema)
    assert fileh.schema != fileh3.schema

    # ColumnSchema
    assert isinstance(fileh.schema[0], pq.ColumnSchema)
    assert fileh.schema[0].equals(fileh.schema[0])
    assert fileh.schema[0] == fileh.schema[0]
    assert not fileh.schema[0].equals(fileh.schema[1])
    assert fileh.schema[0] != fileh.schema[1]
    assert fileh.schema[0] != 'arbitrary object'


@pytest.mark.pandas
def test_read_schema(tempdir):
    N = 100
    df = pd.DataFrame({
        'index': np.arange(N),
        'values': np.random.randn(N)
    }, columns=['index', 'values'])

    data_path = tempdir / 'test.parquet'

    table = pa.Table.from_pandas(df)
    _write_table(table, data_path)

    read1 = pq.read_schema(data_path)
    read2 = pq.read_schema(data_path, memory_map=True)
    assert table.schema.equals(read1)
    assert table.schema.equals(read2)

    assert table.schema.metadata[b'pandas'] == read1.metadata[b'pandas']


def test_parquet_metadata_empty_to_dict(tempdir):
    # https://issues.apache.org/jira/browse/ARROW-10146
    table = pa.table({"a": pa.array([], type="int64")})
    pq.write_table(table, tempdir / "data.parquet")
    metadata = pq.read_metadata(tempdir / "data.parquet")
    # ensure this doesn't error / statistics set to None
    metadata_dict = metadata.to_dict()
    assert len(metadata_dict["row_groups"]) == 1
    assert len(metadata_dict["row_groups"][0]["columns"]) == 1
    assert metadata_dict["row_groups"][0]["columns"][0]["statistics"] is None


@pytest.mark.slow
@pytest.mark.large_memory
def test_metadata_exceeds_message_size():
    # ARROW-13655: Thrift may enable a default message size that limits
    # the size of Parquet metadata that can be written.
    NCOLS = 1000
    NREPEATS = 4000

    table = pa.table({str(i): np.random.randn(10) for i in range(NCOLS)})

    with pa.BufferOutputStream() as out:
        pq.write_table(table, out)
        buf = out.getvalue()

    original_metadata = pq.read_metadata(pa.BufferReader(buf))
    metadata = pq.read_metadata(pa.BufferReader(buf))
    for i in range(NREPEATS):
        metadata.append_row_groups(original_metadata)

    with pa.BufferOutputStream() as out:
        metadata.write_metadata_file(out)
        buf = out.getvalue()

    metadata = pq.read_metadata(pa.BufferReader(buf))


def test_metadata_schema_filesystem(tempdir):
    table = pa.table({"a": [1, 2, 3]})

    # URI writing to local file.
    fname = "data.parquet"
    file_path = str(tempdir / fname)
    file_uri = 'file:///' + file_path

    pq.write_table(table, file_path)

    # Get expected `metadata` from path.
    metadata = pq.read_metadata(tempdir / fname)
    schema = table.schema

    assert pq.read_metadata(file_uri).equals(metadata)
    assert pq.read_metadata(
        file_path, filesystem=LocalFileSystem()).equals(metadata)
    assert pq.read_metadata(
        fname, filesystem=f'file:///{tempdir}').equals(metadata)

    assert pq.read_schema(file_uri).equals(schema)
    assert pq.read_schema(
        file_path, filesystem=LocalFileSystem()).equals(schema)
    assert pq.read_schema(
        fname, filesystem=f'file:///{tempdir}').equals(schema)

    with util.change_cwd(tempdir):
        # Pass `filesystem` arg
        assert pq.read_metadata(
            fname, filesystem=LocalFileSystem()).equals(metadata)

        assert pq.read_schema(
            fname, filesystem=LocalFileSystem()).equals(schema)


def test_metadata_equals():
    table = pa.table({"a": [1, 2, 3]})
    with pa.BufferOutputStream() as out:
        pq.write_table(table, out)
        buf = out.getvalue()

    original_metadata = pq.read_metadata(pa.BufferReader(buf))
    match = "Argument 'other' has incorrect type"
    with pytest.raises(TypeError, match=match):
        original_metadata.equals(None)


@pytest.mark.parametrize("t1,t2,expected_error", (
    ({'col1': range(10)}, {'col1': range(10)}, None),
    ({'col1': range(10)}, {'col2': range(10)},
     "The two columns with index 0 differ."),
    ({'col1': range(10), 'col2': range(10)}, {'col3': range(10)},
     "This schema has 2 columns, other has 1")
))
def test_metadata_append_row_groups_diff(t1, t2, expected_error):
    table1 = pa.table(t1)
    table2 = pa.table(t2)

    buf1 = io.BytesIO()
    buf2 = io.BytesIO()
    pq.write_table(table1, buf1)
    pq.write_table(table2, buf2)
    buf1.seek(0)
    buf2.seek(0)

    meta1 = pq.ParquetFile(buf1).metadata
    meta2 = pq.ParquetFile(buf2).metadata

    if expected_error:
        # Error clearly defines it's happening at append row groups call
        prefix = "AppendRowGroups requires equal schemas.\n"
        with pytest.raises(RuntimeError, match=prefix + expected_error):
            meta1.append_row_groups(meta2)
    else:
        meta1.append_row_groups(meta2)


@pytest.mark.s3
def test_write_metadata_fs_file_combinations(tempdir, s3_example_s3fs):
    s3_fs, s3_path = s3_example_s3fs

    meta1 = tempdir / "meta1"
    meta2 = tempdir / "meta2"
    meta3 = tempdir / "meta3"
    meta4 = tempdir / "meta4"
    meta5 = f"{s3_path}/meta5"

    table = pa.table({"col": range(5)})

    # plain local path
    pq.write_metadata(table.schema, meta1, [])

    # Used the localfilesystem to resolve opening an output stream
    pq.write_metadata(table.schema, meta2, [], filesystem=LocalFileSystem())

    # Can resolve local file URI
    pq.write_metadata(table.schema, meta3.as_uri(), [])

    # Take a file-like obj all the way thru?
    with meta4.open('wb+') as meta4_stream:
        pq.write_metadata(table.schema, meta4_stream, [])

    # S3FileSystem
    pq.write_metadata(table.schema, meta5, [], filesystem=s3_fs)

    assert meta1.read_bytes() == meta2.read_bytes() \
        == meta3.read_bytes() == meta4.read_bytes() \
        == s3_fs.open(meta5).read()

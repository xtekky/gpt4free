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

from collections import UserList
import io
import pathlib
import pytest
import socket
import threading
import weakref

import numpy as np

import pyarrow as pa
from pyarrow.tests.util import changed_environ, invoke_script


try:
    from pandas.testing import assert_frame_equal, assert_series_equal
    import pandas as pd
except ImportError:
    pass


class IpcFixture:
    write_stats = None

    def __init__(self, sink_factory=lambda: io.BytesIO()):
        self._sink_factory = sink_factory
        self.sink = self.get_sink()

    def get_sink(self):
        return self._sink_factory()

    def get_source(self):
        return self.sink.getvalue()

    def write_batches(self, num_batches=5, as_table=False):
        nrows = 5
        schema = pa.schema([('one', pa.float64()), ('two', pa.utf8())])

        writer = self._get_writer(self.sink, schema)

        batches = []
        for i in range(num_batches):
            batch = pa.record_batch(
                [np.random.randn(nrows),
                 ['foo', None, 'bar', 'bazbaz', 'qux']],
                schema=schema)
            batches.append(batch)

        if as_table:
            table = pa.Table.from_batches(batches)
            writer.write_table(table)
        else:
            for batch in batches:
                writer.write_batch(batch)

        self.write_stats = writer.stats
        writer.close()
        return batches


class FileFormatFixture(IpcFixture):

    is_file = True
    options = None

    def _get_writer(self, sink, schema):
        return pa.ipc.new_file(sink, schema, options=self.options)

    def _check_roundtrip(self, as_table=False):
        batches = self.write_batches(as_table=as_table)
        file_contents = pa.BufferReader(self.get_source())

        reader = pa.ipc.open_file(file_contents)

        assert reader.num_record_batches == len(batches)

        for i, batch in enumerate(batches):
            # it works. Must convert back to DataFrame
            batch = reader.get_batch(i)
            assert batches[i].equals(batch)
            assert reader.schema.equals(batches[0].schema)

        assert isinstance(reader.stats, pa.ipc.ReadStats)
        assert isinstance(self.write_stats, pa.ipc.WriteStats)
        assert tuple(reader.stats) == tuple(self.write_stats)


class StreamFormatFixture(IpcFixture):

    # ARROW-6474, for testing writing old IPC protocol with 4-byte prefix
    use_legacy_ipc_format = False
    # ARROW-9395, for testing writing old metadata version
    options = None
    is_file = False

    def _get_writer(self, sink, schema):
        return pa.ipc.new_stream(
            sink,
            schema,
            use_legacy_format=self.use_legacy_ipc_format,
            options=self.options,
        )


class MessageFixture(IpcFixture):

    def _get_writer(self, sink, schema):
        return pa.RecordBatchStreamWriter(sink, schema)


@pytest.fixture
def ipc_fixture():
    return IpcFixture()


@pytest.fixture
def file_fixture():
    return FileFormatFixture()


@pytest.fixture
def stream_fixture():
    return StreamFormatFixture()


@pytest.fixture(params=[
    pytest.param(
        pytest.lazy_fixture('file_fixture'),
        id='File Format'
    ),
    pytest.param(
        pytest.lazy_fixture('stream_fixture'),
        id='Stream Format'
    )
])
def format_fixture(request):
    return request.param


def test_empty_file():
    buf = b''
    with pytest.raises(pa.ArrowInvalid):
        pa.ipc.open_file(pa.BufferReader(buf))


def test_file_simple_roundtrip(file_fixture):
    file_fixture._check_roundtrip(as_table=False)


def test_file_write_table(file_fixture):
    file_fixture._check_roundtrip(as_table=True)


@pytest.mark.parametrize("sink_factory", [
    lambda: io.BytesIO(),
    lambda: pa.BufferOutputStream()
])
def test_file_read_all(sink_factory):
    fixture = FileFormatFixture(sink_factory)

    batches = fixture.write_batches()
    file_contents = pa.BufferReader(fixture.get_source())

    reader = pa.ipc.open_file(file_contents)

    result = reader.read_all()
    expected = pa.Table.from_batches(batches)
    assert result.equals(expected)


def test_open_file_from_buffer(file_fixture):
    # ARROW-2859; APIs accept the buffer protocol
    file_fixture.write_batches()
    source = file_fixture.get_source()

    reader1 = pa.ipc.open_file(source)
    reader2 = pa.ipc.open_file(pa.BufferReader(source))
    reader3 = pa.RecordBatchFileReader(source)

    result1 = reader1.read_all()
    result2 = reader2.read_all()
    result3 = reader3.read_all()

    assert result1.equals(result2)
    assert result1.equals(result3)

    st1 = reader1.stats
    assert st1.num_messages == 6
    assert st1.num_record_batches == 5
    assert reader2.stats == st1
    assert reader3.stats == st1


@pytest.mark.pandas
def test_file_read_pandas(file_fixture):
    frames = [batch.to_pandas() for batch in file_fixture.write_batches()]

    file_contents = pa.BufferReader(file_fixture.get_source())
    reader = pa.ipc.open_file(file_contents)
    result = reader.read_pandas()

    expected = pd.concat(frames).reset_index(drop=True)
    assert_frame_equal(result, expected)


def test_file_pathlib(file_fixture, tmpdir):
    file_fixture.write_batches()
    source = file_fixture.get_source()

    path = tmpdir.join('file.arrow').strpath
    with open(path, 'wb') as f:
        f.write(source)

    t1 = pa.ipc.open_file(pathlib.Path(path)).read_all()
    t2 = pa.ipc.open_file(pa.OSFile(path)).read_all()

    assert t1.equals(t2)


def test_empty_stream():
    buf = io.BytesIO(b'')
    with pytest.raises(pa.ArrowInvalid):
        pa.ipc.open_stream(buf)


@pytest.mark.pandas
def test_read_year_month_nano_interval(tmpdir):
    """ARROW-15783: Verify to_pandas works for interval types.

    Interval types require static structures to be enabled. This test verifies
    that they are when no other library functions are invoked.
    """
    mdn_interval_type = pa.month_day_nano_interval()
    schema = pa.schema([pa.field('nums', mdn_interval_type)])

    path = tmpdir.join('file.arrow').strpath
    with pa.OSFile(path, 'wb') as sink:
        with pa.ipc.new_file(sink, schema) as writer:
            interval_array = pa.array([(1, 2, 3)], type=mdn_interval_type)
            batch = pa.record_batch([interval_array], schema)
            writer.write(batch)
    invoke_script('read_record_batch.py', path)


@pytest.mark.pandas
def test_stream_categorical_roundtrip(stream_fixture):
    df = pd.DataFrame({
        'one': np.random.randn(5),
        'two': pd.Categorical(['foo', np.nan, 'bar', 'foo', 'foo'],
                              categories=['foo', 'bar'],
                              ordered=True)
    })
    batch = pa.RecordBatch.from_pandas(df)
    with stream_fixture._get_writer(stream_fixture.sink, batch.schema) as wr:
        wr.write_batch(batch)

    table = (pa.ipc.open_stream(pa.BufferReader(stream_fixture.get_source()))
             .read_all())
    assert_frame_equal(table.to_pandas(), df)


def test_open_stream_from_buffer(stream_fixture):
    # ARROW-2859
    stream_fixture.write_batches()
    source = stream_fixture.get_source()

    reader1 = pa.ipc.open_stream(source)
    reader2 = pa.ipc.open_stream(pa.BufferReader(source))
    reader3 = pa.RecordBatchStreamReader(source)

    result1 = reader1.read_all()
    result2 = reader2.read_all()
    result3 = reader3.read_all()

    assert result1.equals(result2)
    assert result1.equals(result3)

    st1 = reader1.stats
    assert st1.num_messages == 6
    assert st1.num_record_batches == 5
    assert reader2.stats == st1
    assert reader3.stats == st1

    assert tuple(st1) == tuple(stream_fixture.write_stats)


@pytest.mark.parametrize('options', [
    pa.ipc.IpcReadOptions(),
    pa.ipc.IpcReadOptions(use_threads=False),
])
def test_open_stream_options(stream_fixture, options):
    stream_fixture.write_batches()
    source = stream_fixture.get_source()

    reader = pa.ipc.open_stream(source, options=options)

    reader.read_all()
    st = reader.stats
    assert st.num_messages == 6
    assert st.num_record_batches == 5

    assert tuple(st) == tuple(stream_fixture.write_stats)


def test_open_stream_with_wrong_options(stream_fixture):
    stream_fixture.write_batches()
    source = stream_fixture.get_source()

    with pytest.raises(TypeError):
        pa.ipc.open_stream(source, options=True)


@pytest.mark.parametrize('options', [
    pa.ipc.IpcReadOptions(),
    pa.ipc.IpcReadOptions(use_threads=False),
])
def test_open_file_options(file_fixture, options):
    file_fixture.write_batches()
    source = file_fixture.get_source()

    reader = pa.ipc.open_file(source, options=options)

    reader.read_all()

    st = reader.stats
    assert st.num_messages == 6
    assert st.num_record_batches == 5


def test_open_file_with_wrong_options(file_fixture):
    file_fixture.write_batches()
    source = file_fixture.get_source()

    with pytest.raises(TypeError):
        pa.ipc.open_file(source, options=True)


@pytest.mark.pandas
def test_stream_write_dispatch(stream_fixture):
    # ARROW-1616
    df = pd.DataFrame({
        'one': np.random.randn(5),
        'two': pd.Categorical(['foo', np.nan, 'bar', 'foo', 'foo'],
                              categories=['foo', 'bar'],
                              ordered=True)
    })
    table = pa.Table.from_pandas(df, preserve_index=False)
    batch = pa.RecordBatch.from_pandas(df, preserve_index=False)
    with stream_fixture._get_writer(stream_fixture.sink, table.schema) as wr:
        wr.write(table)
        wr.write(batch)

    table = (pa.ipc.open_stream(pa.BufferReader(stream_fixture.get_source()))
             .read_all())
    assert_frame_equal(table.to_pandas(),
                       pd.concat([df, df], ignore_index=True))


@pytest.mark.pandas
def test_stream_write_table_batches(stream_fixture):
    # ARROW-504
    df = pd.DataFrame({
        'one': np.random.randn(20),
    })

    b1 = pa.RecordBatch.from_pandas(df[:10], preserve_index=False)
    b2 = pa.RecordBatch.from_pandas(df, preserve_index=False)

    table = pa.Table.from_batches([b1, b2, b1])

    with stream_fixture._get_writer(stream_fixture.sink, table.schema) as wr:
        wr.write_table(table, max_chunksize=15)

    batches = list(pa.ipc.open_stream(stream_fixture.get_source()))

    assert list(map(len, batches)) == [10, 15, 5, 10]
    result_table = pa.Table.from_batches(batches)
    assert_frame_equal(result_table.to_pandas(),
                       pd.concat([df[:10], df, df[:10]],
                                 ignore_index=True))


@pytest.mark.parametrize('use_legacy_ipc_format', [False, True])
def test_stream_simple_roundtrip(stream_fixture, use_legacy_ipc_format):
    stream_fixture.use_legacy_ipc_format = use_legacy_ipc_format
    batches = stream_fixture.write_batches()
    file_contents = pa.BufferReader(stream_fixture.get_source())
    reader = pa.ipc.open_stream(file_contents)

    assert reader.schema.equals(batches[0].schema)

    total = 0
    for i, next_batch in enumerate(reader):
        assert next_batch.equals(batches[i])
        total += 1

    assert total == len(batches)

    with pytest.raises(StopIteration):
        reader.read_next_batch()


@pytest.mark.zstd
def test_compression_roundtrip():
    sink = io.BytesIO()
    values = np.random.randint(0, 3, 10000)
    table = pa.Table.from_arrays([values], names=["values"])

    options = pa.ipc.IpcWriteOptions(compression='zstd')
    with pa.ipc.RecordBatchFileWriter(
            sink, table.schema, options=options) as writer:
        writer.write_table(table)
    len1 = len(sink.getvalue())

    sink2 = io.BytesIO()
    codec = pa.Codec('zstd', compression_level=5)
    options = pa.ipc.IpcWriteOptions(compression=codec)
    with pa.ipc.RecordBatchFileWriter(
            sink2, table.schema, options=options) as writer:
        writer.write_table(table)
    len2 = len(sink2.getvalue())

    # In theory len2 should be less than len1 but for this test we just want
    # to ensure compression_level is being correctly passed down to the C++
    # layer so we don't really care if it makes it worse or better
    assert len2 != len1

    t1 = pa.ipc.open_file(sink).read_all()
    t2 = pa.ipc.open_file(sink2).read_all()

    assert t1 == t2


def test_write_options():
    options = pa.ipc.IpcWriteOptions()
    assert options.allow_64bit is False
    assert options.use_legacy_format is False
    assert options.metadata_version == pa.ipc.MetadataVersion.V5

    options.allow_64bit = True
    assert options.allow_64bit is True

    options.use_legacy_format = True
    assert options.use_legacy_format is True

    options.metadata_version = pa.ipc.MetadataVersion.V4
    assert options.metadata_version == pa.ipc.MetadataVersion.V4
    for value in ('V5', 42):
        with pytest.raises((TypeError, ValueError)):
            options.metadata_version = value

    assert options.compression is None
    for value in ['lz4', 'zstd']:
        if pa.Codec.is_available(value):
            options.compression = value
            assert options.compression == value
            options.compression = value.upper()
            assert options.compression == value
    options.compression = None
    assert options.compression is None

    with pytest.raises(TypeError):
        options.compression = 0

    assert options.use_threads is True
    options.use_threads = False
    assert options.use_threads is False

    if pa.Codec.is_available('lz4'):
        options = pa.ipc.IpcWriteOptions(
            metadata_version=pa.ipc.MetadataVersion.V4,
            allow_64bit=True,
            use_legacy_format=True,
            compression='lz4',
            use_threads=False)
        assert options.metadata_version == pa.ipc.MetadataVersion.V4
        assert options.allow_64bit is True
        assert options.use_legacy_format is True
        assert options.compression == 'lz4'
        assert options.use_threads is False


def test_write_options_legacy_exclusive(stream_fixture):
    with pytest.raises(
            ValueError,
            match="provide at most one of options and use_legacy_format"):
        stream_fixture.use_legacy_ipc_format = True
        stream_fixture.options = pa.ipc.IpcWriteOptions()
        stream_fixture.write_batches()


@pytest.mark.parametrize('options', [
    pa.ipc.IpcWriteOptions(),
    pa.ipc.IpcWriteOptions(allow_64bit=True),
    pa.ipc.IpcWriteOptions(use_legacy_format=True),
    pa.ipc.IpcWriteOptions(metadata_version=pa.ipc.MetadataVersion.V4),
    pa.ipc.IpcWriteOptions(use_legacy_format=True,
                           metadata_version=pa.ipc.MetadataVersion.V4),
])
def test_stream_options_roundtrip(stream_fixture, options):
    stream_fixture.use_legacy_ipc_format = None
    stream_fixture.options = options
    batches = stream_fixture.write_batches()
    file_contents = pa.BufferReader(stream_fixture.get_source())

    message = pa.ipc.read_message(stream_fixture.get_source())
    assert message.metadata_version == options.metadata_version

    reader = pa.ipc.open_stream(file_contents)

    assert reader.schema.equals(batches[0].schema)

    total = 0
    for i, next_batch in enumerate(reader):
        assert next_batch.equals(batches[i])
        total += 1

    assert total == len(batches)

    with pytest.raises(StopIteration):
        reader.read_next_batch()


def test_read_options():
    options = pa.ipc.IpcReadOptions()
    assert options.use_threads is True
    assert options.ensure_native_endian is True
    assert options.included_fields == []

    options.ensure_native_endian = False
    assert options.ensure_native_endian is False

    options.use_threads = False
    assert options.use_threads is False

    options.included_fields = [0, 1]
    assert options.included_fields == [0, 1]

    with pytest.raises(TypeError):
        options.included_fields = None

    options = pa.ipc.IpcReadOptions(
        use_threads=False, ensure_native_endian=False,
        included_fields=[1]
    )
    assert options.use_threads is False
    assert options.ensure_native_endian is False
    assert options.included_fields == [1]


def test_read_options_included_fields(stream_fixture):
    options1 = pa.ipc.IpcReadOptions()
    options2 = pa.ipc.IpcReadOptions(included_fields=[1])
    table = pa.Table.from_arrays([pa.array(['foo', 'bar', 'baz', 'qux']),
                                 pa.array([1, 2, 3, 4])],
                                 names=['a', 'b'])
    with stream_fixture._get_writer(stream_fixture.sink, table.schema) as wr:
        wr.write_table(table)
    source = stream_fixture.get_source()

    reader1 = pa.ipc.open_stream(source, options=options1)
    reader2 = pa.ipc.open_stream(
        source, options=options2, memory_pool=pa.system_memory_pool())

    result1 = reader1.read_all()
    result2 = reader2.read_all()

    assert result1.num_columns == 2
    assert result2.num_columns == 1

    expected = pa.Table.from_arrays([pa.array([1, 2, 3, 4])], names=["b"])
    assert result2 == expected
    assert result1 == table


def test_dictionary_delta(format_fixture):
    ty = pa.dictionary(pa.int8(), pa.utf8())
    data = [["foo", "foo", None],
            ["foo", "bar", "foo"],  # potential delta
            ["foo", "bar"],  # nothing new
            ["foo", None, "bar", "quux"],  # potential delta
            ["bar", "quux"],  # replacement
            ]
    batches = [
        pa.RecordBatch.from_arrays([pa.array(v, type=ty)], names=['dicts'])
        for v in data]
    batches_delta_only = batches[:4]
    schema = batches[0].schema

    def write_batches(batches, as_table=False):
        with format_fixture._get_writer(pa.MockOutputStream(),
                                        schema) as writer:
            if as_table:
                table = pa.Table.from_batches(batches)
                writer.write_table(table)
            else:
                for batch in batches:
                    writer.write_batch(batch)
            return writer.stats

    if format_fixture.is_file:
        # File format cannot handle replacement
        with pytest.raises(pa.ArrowInvalid):
            write_batches(batches)
        # File format cannot handle delta if emit_deltas
        # is not provided
        with pytest.raises(pa.ArrowInvalid):
            write_batches(batches_delta_only)
    else:
        st = write_batches(batches)
        assert st.num_record_batches == 5
        assert st.num_dictionary_batches == 4
        assert st.num_replaced_dictionaries == 3
        assert st.num_dictionary_deltas == 0

    format_fixture.use_legacy_ipc_format = None
    format_fixture.options = pa.ipc.IpcWriteOptions(
        emit_dictionary_deltas=True)
    if format_fixture.is_file:
        # File format cannot handle replacement
        with pytest.raises(pa.ArrowInvalid):
            write_batches(batches)
    else:
        st = write_batches(batches)
        assert st.num_record_batches == 5
        assert st.num_dictionary_batches == 4
        assert st.num_replaced_dictionaries == 1
        assert st.num_dictionary_deltas == 2

    st = write_batches(batches_delta_only)
    assert st.num_record_batches == 4
    assert st.num_dictionary_batches == 3
    assert st.num_replaced_dictionaries == 0
    assert st.num_dictionary_deltas == 2

    format_fixture.options = pa.ipc.IpcWriteOptions(
        unify_dictionaries=True
    )
    st = write_batches(batches, as_table=True)
    assert st.num_record_batches == 5
    if format_fixture.is_file:
        assert st.num_dictionary_batches == 1
        assert st.num_replaced_dictionaries == 0
        assert st.num_dictionary_deltas == 0
    else:
        assert st.num_dictionary_batches == 4
        assert st.num_replaced_dictionaries == 3
        assert st.num_dictionary_deltas == 0


def test_envvar_set_legacy_ipc_format():
    schema = pa.schema([pa.field('foo', pa.int32())])

    writer = pa.ipc.new_stream(pa.BufferOutputStream(), schema)
    assert not writer._use_legacy_format
    assert writer._metadata_version == pa.ipc.MetadataVersion.V5
    writer = pa.ipc.new_file(pa.BufferOutputStream(), schema)
    assert not writer._use_legacy_format
    assert writer._metadata_version == pa.ipc.MetadataVersion.V5

    with changed_environ('ARROW_PRE_0_15_IPC_FORMAT', '1'):
        writer = pa.ipc.new_stream(pa.BufferOutputStream(), schema)
        assert writer._use_legacy_format
        assert writer._metadata_version == pa.ipc.MetadataVersion.V5
        writer = pa.ipc.new_file(pa.BufferOutputStream(), schema)
        assert writer._use_legacy_format
        assert writer._metadata_version == pa.ipc.MetadataVersion.V5

    with changed_environ('ARROW_PRE_1_0_METADATA_VERSION', '1'):
        writer = pa.ipc.new_stream(pa.BufferOutputStream(), schema)
        assert not writer._use_legacy_format
        assert writer._metadata_version == pa.ipc.MetadataVersion.V4
        writer = pa.ipc.new_file(pa.BufferOutputStream(), schema)
        assert not writer._use_legacy_format
        assert writer._metadata_version == pa.ipc.MetadataVersion.V4

    with changed_environ('ARROW_PRE_1_0_METADATA_VERSION', '1'):
        with changed_environ('ARROW_PRE_0_15_IPC_FORMAT', '1'):
            writer = pa.ipc.new_stream(pa.BufferOutputStream(), schema)
            assert writer._use_legacy_format
            assert writer._metadata_version == pa.ipc.MetadataVersion.V4
            writer = pa.ipc.new_file(pa.BufferOutputStream(), schema)
            assert writer._use_legacy_format
            assert writer._metadata_version == pa.ipc.MetadataVersion.V4


def test_stream_read_all(stream_fixture):
    batches = stream_fixture.write_batches()
    file_contents = pa.BufferReader(stream_fixture.get_source())
    reader = pa.ipc.open_stream(file_contents)

    result = reader.read_all()
    expected = pa.Table.from_batches(batches)
    assert result.equals(expected)


@pytest.mark.pandas
def test_stream_read_pandas(stream_fixture):
    frames = [batch.to_pandas() for batch in stream_fixture.write_batches()]
    file_contents = stream_fixture.get_source()
    reader = pa.ipc.open_stream(file_contents)
    result = reader.read_pandas()

    expected = pd.concat(frames).reset_index(drop=True)
    assert_frame_equal(result, expected)


@pytest.fixture
def example_messages(stream_fixture):
    batches = stream_fixture.write_batches()
    file_contents = stream_fixture.get_source()
    buf_reader = pa.BufferReader(file_contents)
    reader = pa.MessageReader.open_stream(buf_reader)
    return batches, list(reader)


def test_message_ctors_no_segfault():
    with pytest.raises(TypeError):
        repr(pa.Message())

    with pytest.raises(TypeError):
        repr(pa.MessageReader())


def test_message_reader(example_messages):
    _, messages = example_messages

    assert len(messages) == 6
    assert messages[0].type == 'schema'
    assert isinstance(messages[0].metadata, pa.Buffer)
    assert isinstance(messages[0].body, pa.Buffer)
    assert messages[0].metadata_version == pa.MetadataVersion.V5

    for msg in messages[1:]:
        assert msg.type == 'record batch'
        assert isinstance(msg.metadata, pa.Buffer)
        assert isinstance(msg.body, pa.Buffer)
        assert msg.metadata_version == pa.MetadataVersion.V5


def test_message_serialize_read_message(example_messages):
    _, messages = example_messages

    msg = messages[0]
    buf = msg.serialize()
    reader = pa.BufferReader(buf.to_pybytes() * 2)

    restored = pa.ipc.read_message(buf)
    restored2 = pa.ipc.read_message(reader)
    restored3 = pa.ipc.read_message(buf.to_pybytes())
    restored4 = pa.ipc.read_message(reader)

    assert msg.equals(restored)
    assert msg.equals(restored2)
    assert msg.equals(restored3)
    assert msg.equals(restored4)

    with pytest.raises(pa.ArrowInvalid, match="Corrupted message"):
        pa.ipc.read_message(pa.BufferReader(b'ab'))

    with pytest.raises(EOFError):
        pa.ipc.read_message(reader)


@pytest.mark.gzip
def test_message_read_from_compressed(example_messages):
    # Part of ARROW-5910
    _, messages = example_messages
    for message in messages:
        raw_out = pa.BufferOutputStream()
        with pa.output_stream(raw_out, compression='gzip') as compressed_out:
            message.serialize_to(compressed_out)

        compressed_buf = raw_out.getvalue()

        result = pa.ipc.read_message(pa.input_stream(compressed_buf,
                                                     compression='gzip'))
        assert result.equals(message)


def test_message_read_schema(example_messages):
    batches, messages = example_messages
    schema = pa.ipc.read_schema(messages[0])
    assert schema.equals(batches[1].schema)


def test_message_read_record_batch(example_messages):
    batches, messages = example_messages

    for batch, message in zip(batches, messages[1:]):
        read_batch = pa.ipc.read_record_batch(message, batch.schema)
        assert read_batch.equals(batch)


def test_read_record_batch_on_stream_error_message():
    # ARROW-5374
    batch = pa.record_batch([pa.array([b"foo"], type=pa.utf8())],
                            names=['strs'])
    stream = pa.BufferOutputStream()
    with pa.ipc.new_stream(stream, batch.schema) as writer:
        writer.write_batch(batch)
    buf = stream.getvalue()
    with pytest.raises(IOError,
                       match="type record batch but got schema"):
        pa.ipc.read_record_batch(buf, batch.schema)


# ----------------------------------------------------------------------
# Socket streaming testa


class StreamReaderServer(threading.Thread):

    def init(self, do_read_all):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(('127.0.0.1', 0))
        self._sock.listen(1)
        host, port = self._sock.getsockname()
        self._do_read_all = do_read_all
        self._schema = None
        self._batches = []
        self._table = None
        return port

    def run(self):
        connection, client_address = self._sock.accept()
        try:
            source = connection.makefile(mode='rb')
            reader = pa.ipc.open_stream(source)
            self._schema = reader.schema
            if self._do_read_all:
                self._table = reader.read_all()
            else:
                for i, batch in enumerate(reader):
                    self._batches.append(batch)
        finally:
            connection.close()
            self._sock.close()

    def get_result(self):
        return (self._schema, self._table if self._do_read_all
                else self._batches)


class SocketStreamFixture(IpcFixture):

    def __init__(self):
        # XXX(wesm): test will decide when to start socket server. This should
        # probably be refactored
        pass

    def start_server(self, do_read_all):
        self._server = StreamReaderServer()
        port = self._server.init(do_read_all)
        self._server.start()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect(('127.0.0.1', port))
        self.sink = self.get_sink()

    def stop_and_get_result(self):
        import struct
        self.sink.write(struct.pack('Q', 0))
        self.sink.flush()
        self._sock.close()
        self._server.join()
        return self._server.get_result()

    def get_sink(self):
        return self._sock.makefile(mode='wb')

    def _get_writer(self, sink, schema):
        return pa.RecordBatchStreamWriter(sink, schema)


@pytest.fixture
def socket_fixture():
    return SocketStreamFixture()


def test_socket_simple_roundtrip(socket_fixture):
    socket_fixture.start_server(do_read_all=False)
    writer_batches = socket_fixture.write_batches()
    reader_schema, reader_batches = socket_fixture.stop_and_get_result()

    assert reader_schema.equals(writer_batches[0].schema)
    assert len(reader_batches) == len(writer_batches)
    for i, batch in enumerate(writer_batches):
        assert reader_batches[i].equals(batch)


def test_socket_read_all(socket_fixture):
    socket_fixture.start_server(do_read_all=True)
    writer_batches = socket_fixture.write_batches()
    _, result = socket_fixture.stop_and_get_result()

    expected = pa.Table.from_batches(writer_batches)
    assert result.equals(expected)


# ----------------------------------------------------------------------
# Miscellaneous IPC tests

@pytest.mark.pandas
def test_ipc_file_stream_has_eos():
    # ARROW-5395
    df = pd.DataFrame({'foo': [1.5]})
    batch = pa.RecordBatch.from_pandas(df)
    sink = pa.BufferOutputStream()
    write_file(batch, sink)
    buffer = sink.getvalue()

    # skip the file magic
    reader = pa.ipc.open_stream(buffer[8:])

    # will fail if encounters footer data instead of eos
    rdf = reader.read_pandas()

    assert_frame_equal(df, rdf)


@pytest.mark.pandas
def test_ipc_zero_copy_numpy():
    df = pd.DataFrame({'foo': [1.5]})

    batch = pa.RecordBatch.from_pandas(df)
    sink = pa.BufferOutputStream()
    write_file(batch, sink)
    buffer = sink.getvalue()
    reader = pa.BufferReader(buffer)

    batches = read_file(reader)

    data = batches[0].to_pandas()
    rdf = pd.DataFrame(data)
    assert_frame_equal(df, rdf)


@pytest.mark.pandas
@pytest.mark.parametrize("ipc_type", ["stream", "file"])
def test_batches_with_custom_metadata_roundtrip(ipc_type):
    df = pd.DataFrame({'foo': [1.5]})

    batch = pa.RecordBatch.from_pandas(df)
    sink = pa.BufferOutputStream()

    batch_count = 2
    file_factory = {"stream": pa.ipc.new_stream,
                    "file": pa.ipc.new_file}[ipc_type]

    with file_factory(sink, batch.schema) as writer:
        for i in range(batch_count):
            writer.write_batch(batch, custom_metadata={"batch_id": str(i)})
        # write a batch without custom metadata
        writer.write_batch(batch)

    buffer = sink.getvalue()

    if ipc_type == "stream":
        with pa.ipc.open_stream(buffer) as reader:
            batch_with_metas = list(reader.iter_batches_with_custom_metadata())
    else:
        with pa.ipc.open_file(buffer) as reader:
            batch_with_metas = [reader.get_batch_with_custom_metadata(i)
                                for i in range(reader.num_record_batches)]

    for i in range(batch_count):
        assert batch_with_metas[i].batch.num_rows == 1
        assert isinstance(
            batch_with_metas[i].custom_metadata, pa.KeyValueMetadata)
        assert batch_with_metas[i].custom_metadata == {"batch_id": str(i)}

    # the last batch has no custom metadata
    assert batch_with_metas[batch_count].batch.num_rows == 1
    assert batch_with_metas[batch_count].custom_metadata is None


def test_ipc_stream_no_batches():
    # ARROW-2307
    table = pa.Table.from_arrays([pa.array([1, 2, 3, 4]),
                                  pa.array(['foo', 'bar', 'baz', 'qux'])],
                                 names=['a', 'b'])

    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema):
        pass

    source = sink.getvalue()
    with pa.ipc.open_stream(source) as reader:
        result = reader.read_all()

    assert result.schema.equals(table.schema)
    assert len(result) == 0


@pytest.mark.pandas
def test_get_record_batch_size():
    N = 10
    itemsize = 8
    df = pd.DataFrame({'foo': np.random.randn(N)})

    batch = pa.RecordBatch.from_pandas(df)
    assert pa.ipc.get_record_batch_size(batch) > (N * itemsize)


@pytest.mark.pandas
def _check_serialize_pandas_round_trip(df, use_threads=False):
    buf = pa.serialize_pandas(df, nthreads=2 if use_threads else 1)
    result = pa.deserialize_pandas(buf, use_threads=use_threads)
    assert_frame_equal(result, df)


@pytest.mark.pandas
def test_pandas_serialize_round_trip():
    index = pd.Index([1, 2, 3], name='my_index')
    columns = ['foo', 'bar']
    df = pd.DataFrame(
        {'foo': [1.5, 1.6, 1.7], 'bar': list('abc')},
        index=index, columns=columns
    )
    _check_serialize_pandas_round_trip(df)


@pytest.mark.pandas
def test_pandas_serialize_round_trip_nthreads():
    index = pd.Index([1, 2, 3], name='my_index')
    columns = ['foo', 'bar']
    df = pd.DataFrame(
        {'foo': [1.5, 1.6, 1.7], 'bar': list('abc')},
        index=index, columns=columns
    )
    _check_serialize_pandas_round_trip(df, use_threads=True)


@pytest.mark.pandas
def test_pandas_serialize_round_trip_multi_index():
    index1 = pd.Index([1, 2, 3], name='level_1')
    index2 = pd.Index(list('def'), name=None)
    index = pd.MultiIndex.from_arrays([index1, index2])

    columns = ['foo', 'bar']
    df = pd.DataFrame(
        {'foo': [1.5, 1.6, 1.7], 'bar': list('abc')},
        index=index,
        columns=columns,
    )
    _check_serialize_pandas_round_trip(df)


@pytest.mark.pandas
def test_serialize_pandas_empty_dataframe():
    df = pd.DataFrame()
    _check_serialize_pandas_round_trip(df)


@pytest.mark.pandas
def test_pandas_serialize_round_trip_not_string_columns():
    df = pd.DataFrame(list(zip([1.5, 1.6, 1.7], 'abc')))
    buf = pa.serialize_pandas(df)
    result = pa.deserialize_pandas(buf)
    assert_frame_equal(result, df)


@pytest.mark.pandas
def test_serialize_pandas_no_preserve_index():
    df = pd.DataFrame({'a': [1, 2, 3]}, index=[1, 2, 3])
    expected = pd.DataFrame({'a': [1, 2, 3]})

    buf = pa.serialize_pandas(df, preserve_index=False)
    result = pa.deserialize_pandas(buf)
    assert_frame_equal(result, expected)

    buf = pa.serialize_pandas(df, preserve_index=True)
    result = pa.deserialize_pandas(buf)
    assert_frame_equal(result, df)


@pytest.mark.pandas
@pytest.mark.filterwarnings("ignore:'pyarrow:FutureWarning")
def test_serialize_with_pandas_objects():
    df = pd.DataFrame({'a': [1, 2, 3]}, index=[1, 2, 3])
    s = pd.Series([1, 2, 3, 4])

    data = {
        'a_series': df['a'],
        'a_frame': df,
        's_series': s
    }

    serialized = pa.serialize(data).to_buffer()
    deserialized = pa.deserialize(serialized)
    assert_frame_equal(deserialized['a_frame'], df)

    assert_series_equal(deserialized['a_series'], df['a'])
    assert deserialized['a_series'].name == 'a'

    assert_series_equal(deserialized['s_series'], s)
    assert deserialized['s_series'].name is None


@pytest.mark.pandas
def test_schema_batch_serialize_methods():
    nrows = 5
    df = pd.DataFrame({
        'one': np.random.randn(nrows),
        'two': ['foo', np.nan, 'bar', 'bazbaz', 'qux']})
    batch = pa.RecordBatch.from_pandas(df)

    s_schema = batch.schema.serialize()
    s_batch = batch.serialize()

    recons_schema = pa.ipc.read_schema(s_schema)
    recons_batch = pa.ipc.read_record_batch(s_batch, recons_schema)
    assert recons_batch.equals(batch)


def test_schema_serialization_with_metadata():
    field_metadata = {b'foo': b'bar', b'kind': b'field'}
    schema_metadata = {b'foo': b'bar', b'kind': b'schema'}

    f0 = pa.field('a', pa.int8())
    f1 = pa.field('b', pa.string(), metadata=field_metadata)

    schema = pa.schema([f0, f1], metadata=schema_metadata)

    s_schema = schema.serialize()
    recons_schema = pa.ipc.read_schema(s_schema)

    assert recons_schema.equals(schema)
    assert recons_schema.metadata == schema_metadata
    assert recons_schema[0].metadata is None
    assert recons_schema[1].metadata == field_metadata


def write_file(batch, sink):
    with pa.ipc.new_file(sink, batch.schema) as writer:
        writer.write_batch(batch)


def read_file(source):
    with pa.ipc.open_file(source) as reader:
        return [reader.get_batch(i) for i in range(reader.num_record_batches)]


def test_write_empty_ipc_file():
    # ARROW-3894: IPC file was not being properly initialized when no record
    # batches are being written
    schema = pa.schema([('field', pa.int64())])

    sink = pa.BufferOutputStream()
    with pa.ipc.new_file(sink, schema):
        pass

    buf = sink.getvalue()
    with pa.RecordBatchFileReader(pa.BufferReader(buf)) as reader:
        table = reader.read_all()
    assert len(table) == 0
    assert table.schema.equals(schema)


def test_py_record_batch_reader():
    def make_schema():
        return pa.schema([('field', pa.int64())])

    def make_batches():
        schema = make_schema()
        batch1 = pa.record_batch([[1, 2, 3]], schema=schema)
        batch2 = pa.record_batch([[4, 5]], schema=schema)
        return [batch1, batch2]

    # With iterable
    batches = UserList(make_batches())  # weakrefable
    wr = weakref.ref(batches)

    with pa.RecordBatchReader.from_batches(make_schema(),
                                           batches) as reader:
        batches = None
        assert wr() is not None
        assert list(reader) == make_batches()
        assert wr() is None

    # With iterator
    batches = iter(UserList(make_batches()))  # weakrefable
    wr = weakref.ref(batches)

    with pa.RecordBatchReader.from_batches(make_schema(),
                                           batches) as reader:
        batches = None
        assert wr() is not None
        assert list(reader) == make_batches()
        assert wr() is None

    # ensure we get proper error when not passing a schema
    # (https://issues.apache.org/jira/browse/ARROW-18229)
    batches = make_batches()
    with pytest.raises(TypeError):
        reader = pa.RecordBatchReader.from_batches(
            [('field', pa.int64())], batches)
        pass

    with pytest.raises(TypeError):
        reader = pa.RecordBatchReader.from_batches(None, batches)
        pass

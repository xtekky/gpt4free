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

import bz2
from contextlib import contextmanager
from io import (BytesIO, StringIO, TextIOWrapper, BufferedIOBase, IOBase)
import itertools
import gc
import gzip
import math
import os
import pathlib
import pickle
import pytest
import sys
import tempfile
import weakref

import numpy as np

from pyarrow.util import guid
from pyarrow import Codec
import pyarrow as pa


def check_large_seeks(file_factory):
    if sys.platform in ('win32', 'darwin'):
        pytest.skip("need sparse file support")
    try:
        filename = tempfile.mktemp(prefix='test_io')
        with open(filename, 'wb') as f:
            f.truncate(2 ** 32 + 10)
            f.seek(2 ** 32 + 5)
            f.write(b'mark\n')
        with file_factory(filename) as f:
            assert f.seek(2 ** 32 + 5) == 2 ** 32 + 5
            assert f.tell() == 2 ** 32 + 5
            assert f.read(5) == b'mark\n'
            assert f.tell() == 2 ** 32 + 10
    finally:
        os.unlink(filename)


@contextmanager
def assert_file_not_found():
    with pytest.raises(FileNotFoundError):
        yield


# ----------------------------------------------------------------------
# Python file-like objects


def test_python_file_write():
    buf = BytesIO()

    f = pa.PythonFile(buf)

    assert f.tell() == 0

    s1 = b'enga\xc3\xb1ado'
    s2 = b'foobar'

    f.write(s1)
    assert f.tell() == len(s1)

    f.write(s2)

    expected = s1 + s2

    result = buf.getvalue()
    assert result == expected

    assert not f.closed
    f.close()
    assert f.closed

    with pytest.raises(TypeError, match="binary file expected"):
        pa.PythonFile(StringIO())


def test_python_file_read():
    data = b'some sample data'

    buf = BytesIO(data)
    f = pa.PythonFile(buf, mode='r')

    assert f.size() == len(data)

    assert f.tell() == 0

    assert f.read(4) == b'some'
    assert f.tell() == 4

    f.seek(0)
    assert f.tell() == 0

    f.seek(5)
    assert f.tell() == 5

    v = f.read(50)
    assert v == b'sample data'
    assert len(v) == 11

    assert f.size() == len(data)

    assert not f.closed
    f.close()
    assert f.closed

    with pytest.raises(TypeError, match="binary file expected"):
        pa.PythonFile(StringIO(), mode='r')


@pytest.mark.parametrize("nbytes", (-1, 0, 1, 5, 100))
@pytest.mark.parametrize("file_offset", (-1, 0, 5, 100))
def test_python_file_get_stream(nbytes, file_offset):

    data = b'data1data2data3data4data5'

    f = pa.PythonFile(BytesIO(data), mode='r')

    # negative nbytes or offsets don't make sense here, raise ValueError
    if nbytes < 0 or file_offset < 0:
        with pytest.raises(pa.ArrowInvalid,
                           match="should be a positive value"):
            f.get_stream(file_offset=file_offset, nbytes=nbytes)
        f.close()
        return
    else:
        stream = f.get_stream(file_offset=file_offset, nbytes=nbytes)

    # Subsequent calls to 'read' should match behavior if same
    # data passed to BytesIO where get_stream should handle if
    # nbytes/file_offset results in no bytes b/c out of bounds.
    start = min(file_offset, len(data))
    end = min(file_offset + nbytes, len(data))
    buf = BytesIO(data[start:end])

    # read some chunks
    assert stream.read(nbytes=4) == buf.read(4)
    assert stream.read(nbytes=6) == buf.read(6)

    # Read to end of each stream
    assert stream.read() == buf.read()

    # Try reading past the stream
    n = len(data) * 2
    assert stream.read(n) == buf.read(n)

    # NativeFile[CInputStream] is not seekable
    with pytest.raises(OSError, match="seekable"):
        stream.seek(0)

    stream.close()
    assert stream.closed


def test_python_file_read_at():
    data = b'some sample data'

    buf = BytesIO(data)
    f = pa.PythonFile(buf, mode='r')

    # test simple read at
    v = f.read_at(nbytes=5, offset=3)
    assert v == b'e sam'
    assert len(v) == 5

    # test reading entire file when nbytes > len(file)
    w = f.read_at(nbytes=50, offset=0)
    assert w == data
    assert len(w) == 16


def test_python_file_readall():
    data = b'some sample data'

    buf = BytesIO(data)
    with pa.PythonFile(buf, mode='r') as f:
        assert f.readall() == data


def test_python_file_readinto():
    length = 10
    data = b'some sample data longer than 10'
    dst_buf = bytearray(length)
    src_buf = BytesIO(data)

    with pa.PythonFile(src_buf, mode='r') as f:
        assert f.readinto(dst_buf) == 10

        assert dst_buf[:length] == data[:length]
        assert len(dst_buf) == length


def test_python_file_read_buffer():
    length = 10
    data = b'0123456798'
    dst_buf = bytearray(data)

    class DuckReader:
        def close(self):
            pass

        @property
        def closed(self):
            return False

        def read_buffer(self, nbytes):
            assert nbytes == length
            return memoryview(dst_buf)[:nbytes]

    duck_reader = DuckReader()
    with pa.PythonFile(duck_reader, mode='r') as f:
        buf = f.read_buffer(length)
        assert len(buf) == length
        assert memoryview(buf).tobytes() == dst_buf[:length]
        # buf should point to the same memory, so modyfing it
        memoryview(buf)[0] = ord(b'x')
        # should modify the original
        assert dst_buf[0] == ord(b'x')


def test_python_file_correct_abc():
    with pa.PythonFile(BytesIO(b''), mode='r') as f:
        assert isinstance(f, BufferedIOBase)
        assert isinstance(f, IOBase)


def test_python_file_iterable():
    data = b'''line1
    line2
    line3
    '''

    buf = BytesIO(data)
    buf2 = BytesIO(data)

    with pa.PythonFile(buf, mode='r') as f:
        for read, expected in zip(f, buf2):
            assert read == expected


def test_python_file_large_seeks():
    def factory(filename):
        return pa.PythonFile(open(filename, 'rb'))

    check_large_seeks(factory)


def test_bytes_reader():
    # Like a BytesIO, but zero-copy underneath for C++ consumers
    data = b'some sample data'
    f = pa.BufferReader(data)
    assert f.tell() == 0

    assert f.size() == len(data)

    assert f.read(4) == b'some'
    assert f.tell() == 4

    f.seek(0)
    assert f.tell() == 0

    f.seek(0, 2)
    assert f.tell() == len(data)

    f.seek(5)
    assert f.tell() == 5

    assert f.read(50) == b'sample data'

    assert not f.closed
    f.close()
    assert f.closed


def test_bytes_reader_non_bytes():
    with pytest.raises(TypeError):
        pa.BufferReader('some sample data')


def test_bytes_reader_retains_parent_reference():
    import gc

    # ARROW-421
    def get_buffer():
        data = b'some sample data' * 1000
        reader = pa.BufferReader(data)
        reader.seek(5)
        return reader.read_buffer(6)

    buf = get_buffer()
    gc.collect()
    assert buf.to_pybytes() == b'sample'
    assert buf.parent is not None


def test_python_file_implicit_mode(tmpdir):
    path = os.path.join(str(tmpdir), 'foo.txt')
    with open(path, 'wb') as f:
        pf = pa.PythonFile(f)
        assert pf.writable()
        assert not pf.readable()
        assert not pf.seekable()  # PyOutputStream isn't seekable
        f.write(b'foobar\n')

    with open(path, 'rb') as f:
        pf = pa.PythonFile(f)
        assert pf.readable()
        assert not pf.writable()
        assert pf.seekable()
        assert pf.read() == b'foobar\n'

    bio = BytesIO()
    pf = pa.PythonFile(bio)
    assert pf.writable()
    assert not pf.readable()
    assert not pf.seekable()
    pf.write(b'foobar\n')
    assert bio.getvalue() == b'foobar\n'


def test_python_file_writelines(tmpdir):
    lines = [b'line1\n', b'line2\n' b'line3']
    path = os.path.join(str(tmpdir), 'foo.txt')
    with open(path, 'wb') as f:
        try:
            f = pa.PythonFile(f, mode='w')
            assert f.writable()
            f.writelines(lines)
        finally:
            f.close()

    with open(path, 'rb') as f:
        try:
            f = pa.PythonFile(f, mode='r')
            assert f.readable()
            assert f.read() == b''.join(lines)
        finally:
            f.close()


def test_python_file_closing():
    bio = BytesIO()
    pf = pa.PythonFile(bio)
    wr = weakref.ref(pf)
    del pf
    assert wr() is None  # object was destroyed
    assert not bio.closed
    pf = pa.PythonFile(bio)
    pf.close()
    assert bio.closed


# ----------------------------------------------------------------------
# Buffers


def check_buffer_pickling(buf):
    # Check that buffer survives a pickle roundtrip
    for protocol in range(0, pickle.HIGHEST_PROTOCOL + 1):
        result = pickle.loads(pickle.dumps(buf, protocol=protocol))
        assert len(result) == len(buf)
        assert memoryview(result) == memoryview(buf)
        assert result.to_pybytes() == buf.to_pybytes()
        assert result.is_mutable == buf.is_mutable


def test_buffer_bytes():
    val = b'some data'

    buf = pa.py_buffer(val)
    assert isinstance(buf, pa.Buffer)
    assert not buf.is_mutable
    assert buf.is_cpu

    result = buf.to_pybytes()
    assert result == val

    check_buffer_pickling(buf)


def test_buffer_null_data():
    null_buff = pa.foreign_buffer(address=0, size=0)
    assert null_buff.to_pybytes() == b""
    assert null_buff.address == 0
    # ARROW-16048: we shouldn't expose a NULL address through the Python
    # buffer protocol.
    m = memoryview(null_buff)
    assert m.tobytes() == b""
    assert pa.py_buffer(m).address != 0

    check_buffer_pickling(null_buff)


def test_buffer_memoryview():
    val = b'some data'

    buf = pa.py_buffer(val)
    assert isinstance(buf, pa.Buffer)
    assert not buf.is_mutable
    assert buf.is_cpu

    result = memoryview(buf)
    assert result == val

    check_buffer_pickling(buf)


def test_buffer_bytearray():
    val = bytearray(b'some data')

    buf = pa.py_buffer(val)
    assert isinstance(buf, pa.Buffer)
    assert buf.is_mutable
    assert buf.is_cpu

    result = bytearray(buf)
    assert result == val

    check_buffer_pickling(buf)


def test_buffer_invalid():
    with pytest.raises(TypeError,
                       match="(bytes-like object|buffer interface)"):
        pa.py_buffer(None)


def test_buffer_weakref():
    buf = pa.py_buffer(b'some data')
    wr = weakref.ref(buf)
    assert wr() is not None
    del buf
    assert wr() is None


@pytest.mark.parametrize('val, expected_hex_buffer',
                         [(b'check', b'636865636B'),
                          (b'\a0', b'0730'),
                          (b'', b'')])
def test_buffer_hex(val, expected_hex_buffer):
    buf = pa.py_buffer(val)
    assert buf.hex() == expected_hex_buffer


def test_buffer_to_numpy():
    # Make sure creating a numpy array from an arrow buffer works
    byte_array = bytearray(20)
    byte_array[0] = 42
    buf = pa.py_buffer(byte_array)
    array = np.frombuffer(buf, dtype="uint8")
    assert array[0] == byte_array[0]
    byte_array[0] += 1
    assert array[0] == byte_array[0]
    assert array.base == buf


def test_buffer_from_numpy():
    # C-contiguous
    arr = np.arange(12, dtype=np.int8).reshape((3, 4))
    buf = pa.py_buffer(arr)
    assert buf.is_cpu
    assert buf.is_mutable
    assert buf.to_pybytes() == arr.tobytes()
    # F-contiguous; note strides information is lost
    buf = pa.py_buffer(arr.T)
    assert buf.is_cpu
    assert buf.is_mutable
    assert buf.to_pybytes() == arr.tobytes()
    # Non-contiguous
    with pytest.raises(ValueError, match="not contiguous"):
        buf = pa.py_buffer(arr.T[::2])


def test_buffer_address():
    b1 = b'some data!'
    b2 = bytearray(b1)
    b3 = bytearray(b1)

    buf1 = pa.py_buffer(b1)
    buf2 = pa.py_buffer(b1)
    buf3 = pa.py_buffer(b2)
    buf4 = pa.py_buffer(b3)

    assert buf1.address > 0
    assert buf1.address == buf2.address
    assert buf3.address != buf2.address
    assert buf4.address != buf3.address

    arr = np.arange(5)
    buf = pa.py_buffer(arr)
    assert buf.address == arr.ctypes.data


def test_buffer_equals():
    # Buffer.equals() returns true iff the buffers have the same contents
    def eq(a, b):
        assert a.equals(b)
        assert a == b
        assert not (a != b)

    def ne(a, b):
        assert not a.equals(b)
        assert not (a == b)
        assert a != b

    b1 = b'some data!'
    b2 = bytearray(b1)
    b3 = bytearray(b1)
    b3[0] = 42
    buf1 = pa.py_buffer(b1)
    buf2 = pa.py_buffer(b2)
    buf3 = pa.py_buffer(b2)
    buf4 = pa.py_buffer(b3)
    buf5 = pa.py_buffer(np.frombuffer(b2, dtype=np.int16))
    eq(buf1, buf1)
    eq(buf1, buf2)
    eq(buf2, buf3)
    ne(buf2, buf4)
    # Data type is indifferent
    eq(buf2, buf5)


def test_buffer_eq_bytes():
    buf = pa.py_buffer(b'some data')
    assert buf == b'some data'
    assert buf == bytearray(b'some data')
    assert buf != b'some dat1'

    with pytest.raises(TypeError):
        buf == 'some data'


def test_buffer_getitem():
    data = bytearray(b'some data!')
    buf = pa.py_buffer(data)

    n = len(data)
    for ix in range(-n, n - 1):
        assert buf[ix] == data[ix]

    with pytest.raises(IndexError):
        buf[n]

    with pytest.raises(IndexError):
        buf[-n - 1]


def test_buffer_slicing():
    data = b'some data!'
    buf = pa.py_buffer(data)

    sliced = buf.slice(2)
    expected = pa.py_buffer(b'me data!')
    assert sliced.equals(expected)

    sliced2 = buf.slice(2, 4)
    expected2 = pa.py_buffer(b'me d')
    assert sliced2.equals(expected2)

    # 0 offset
    assert buf.slice(0).equals(buf)

    # Slice past end of buffer
    assert len(buf.slice(len(buf))) == 0

    with pytest.raises(IndexError):
        buf.slice(-1)

    with pytest.raises(IndexError):
        buf.slice(len(buf) + 1)
    assert buf[11:].to_pybytes() == b""

    # Slice stop exceeds buffer length
    with pytest.raises(IndexError):
        buf.slice(1, len(buf))
    assert buf[1:11].to_pybytes() == buf.to_pybytes()[1:]

    # Negative length
    with pytest.raises(IndexError):
        buf.slice(1, -1)

    # Test slice notation
    assert buf[2:].equals(buf.slice(2))
    assert buf[2:5].equals(buf.slice(2, 3))
    assert buf[-5:].equals(buf.slice(len(buf) - 5))
    assert buf[-5:-2].equals(buf.slice(len(buf) - 5, 3))

    with pytest.raises(IndexError):
        buf[::-1]
    with pytest.raises(IndexError):
        buf[::2]

    n = len(buf)
    for start in range(-n * 2, n * 2):
        for stop in range(-n * 2, n * 2):
            assert buf[start:stop].to_pybytes() == buf.to_pybytes()[start:stop]


def test_buffer_hashing():
    # Buffers are unhashable
    with pytest.raises(TypeError, match="unhashable"):
        hash(pa.py_buffer(b'123'))


def test_buffer_protocol_respects_immutability():
    # ARROW-3228; NumPy's frombuffer ctor determines whether a buffer-like
    # object is mutable by first attempting to get a mutable buffer using
    # PyObject_FromBuffer. If that fails, it assumes that the object is
    # immutable
    a = b'12345'
    arrow_ref = pa.py_buffer(a)
    numpy_ref = np.frombuffer(arrow_ref, dtype=np.uint8)
    assert not numpy_ref.flags.writeable


def test_foreign_buffer():
    obj = np.array([1, 2], dtype=np.int32)
    addr = obj.__array_interface__["data"][0]
    size = obj.nbytes
    buf = pa.foreign_buffer(addr, size, obj)
    wr = weakref.ref(obj)
    del obj
    assert np.frombuffer(buf, dtype=np.int32).tolist() == [1, 2]
    assert wr() is not None
    del buf
    assert wr() is None


def test_allocate_buffer():
    buf = pa.allocate_buffer(100)
    assert buf.size == 100
    assert buf.is_mutable
    assert buf.parent is None

    bit = b'abcde'
    writer = pa.FixedSizeBufferWriter(buf)
    writer.write(bit)

    assert buf.to_pybytes()[:5] == bit


def test_allocate_buffer_resizable():
    buf = pa.allocate_buffer(100, resizable=True)
    assert isinstance(buf, pa.ResizableBuffer)

    buf.resize(200)
    assert buf.size == 200


@pytest.mark.parametrize("compression", [
    pytest.param(
        "bz2", marks=pytest.mark.xfail(raises=pa.lib.ArrowNotImplementedError)
    ),
    "brotli",
    "gzip",
    "lz4",
    "zstd",
    "snappy"
])
def test_compress_decompress(compression):
    if not Codec.is_available(compression):
        pytest.skip("{} support is not built".format(compression))

    INPUT_SIZE = 10000
    test_data = (np.random.randint(0, 255, size=INPUT_SIZE)
                 .astype(np.uint8)
                 .tobytes())
    test_buf = pa.py_buffer(test_data)

    compressed_buf = pa.compress(test_buf, codec=compression)
    compressed_bytes = pa.compress(test_data, codec=compression,
                                   asbytes=True)

    assert isinstance(compressed_bytes, bytes)

    decompressed_buf = pa.decompress(compressed_buf, INPUT_SIZE,
                                     codec=compression)
    decompressed_bytes = pa.decompress(compressed_bytes, INPUT_SIZE,
                                       codec=compression, asbytes=True)

    assert isinstance(decompressed_bytes, bytes)

    assert decompressed_buf.equals(test_buf)
    assert decompressed_bytes == test_data

    with pytest.raises(ValueError):
        pa.decompress(compressed_bytes, codec=compression)


@pytest.mark.parametrize("compression", [
    pytest.param(
        "bz2", marks=pytest.mark.xfail(raises=pa.lib.ArrowNotImplementedError)
    ),
    "brotli",
    "gzip",
    "lz4",
    "zstd",
    "snappy"
])
def test_compression_level(compression):
    if not Codec.is_available(compression):
        pytest.skip("{} support is not built".format(compression))

    codec = Codec(compression)
    if codec.name == "snappy":
        assert codec.compression_level is None
    else:
        assert isinstance(codec.compression_level, int)

    # These codecs do not support a compression level
    no_level = ['snappy']
    if compression in no_level:
        assert not Codec.supports_compression_level(compression)
        with pytest.raises(ValueError):
            Codec(compression, 0)
        with pytest.raises(ValueError):
            Codec.minimum_compression_level(compression)
        with pytest.raises(ValueError):
            Codec.maximum_compression_level(compression)
        with pytest.raises(ValueError):
            Codec.default_compression_level(compression)
        return

    INPUT_SIZE = 10000
    test_data = (np.random.randint(0, 255, size=INPUT_SIZE)
                 .astype(np.uint8)
                 .tobytes())
    test_buf = pa.py_buffer(test_data)

    min_level = Codec.minimum_compression_level(compression)
    max_level = Codec.maximum_compression_level(compression)
    default_level = Codec.default_compression_level(compression)

    assert min_level < max_level
    assert default_level >= min_level
    assert default_level <= max_level

    for compression_level in range(min_level, max_level+1):
        codec = Codec(compression, compression_level)
        compressed_buf = codec.compress(test_buf)
        compressed_bytes = codec.compress(test_data, asbytes=True)
        assert isinstance(compressed_bytes, bytes)
        decompressed_buf = codec.decompress(compressed_buf, INPUT_SIZE)
        decompressed_bytes = codec.decompress(compressed_bytes, INPUT_SIZE,
                                              asbytes=True)

        assert isinstance(decompressed_bytes, bytes)

        assert decompressed_buf.equals(test_buf)
        assert decompressed_bytes == test_data

        with pytest.raises(ValueError):
            codec.decompress(compressed_bytes)

    # The ability to set a seed this way is not present on older versions of
    # numpy (currently in our python 3.6 CI build).  Some inputs might just
    # happen to compress the same between the two levels so using seeded
    # random numbers is necessary to help get more reliable results
    #
    # The goal of this part is to ensure the compression_level is being
    # passed down to the C++ layer, not to verify the compression algs
    # themselves
    if not hasattr(np.random, 'default_rng'):
        pytest.skip('Requires newer version of numpy')
    rng = np.random.default_rng(seed=42)
    values = rng.integers(0, 100, 1000)
    arr = pa.array(values)
    hard_to_compress_buffer = arr.buffers()[1]

    weak_codec = Codec(compression, min_level)
    weakly_compressed_buf = weak_codec.compress(hard_to_compress_buffer)

    strong_codec = Codec(compression, max_level)
    strongly_compressed_buf = strong_codec.compress(hard_to_compress_buffer)

    assert len(weakly_compressed_buf) > len(strongly_compressed_buf)


def test_buffer_memoryview_is_immutable():
    val = b'some data'

    buf = pa.py_buffer(val)
    assert not buf.is_mutable
    assert isinstance(buf, pa.Buffer)

    result = memoryview(buf)
    assert result.readonly

    with pytest.raises(TypeError) as exc:
        result[0] = b'h'
        assert 'cannot modify read-only' in str(exc.value)

    b = bytes(buf)
    with pytest.raises(TypeError) as exc:
        b[0] = b'h'
        assert 'cannot modify read-only' in str(exc.value)


def test_uninitialized_buffer():
    # ARROW-2039: calling Buffer() directly creates an uninitialized object
    # ARROW-2638: prevent calling extension class constructors directly
    with pytest.raises(TypeError):
        pa.Buffer()


def test_memory_output_stream():
    # 10 bytes
    val = b'dataabcdef'
    f = pa.BufferOutputStream()

    K = 1000
    for i in range(K):
        f.write(val)

    buf = f.getvalue()
    assert len(buf) == len(val) * K
    assert buf.to_pybytes() == val * K


def test_inmemory_write_after_closed():
    f = pa.BufferOutputStream()
    f.write(b'ok')
    assert not f.closed
    f.getvalue()
    assert f.closed

    with pytest.raises(ValueError):
        f.write(b'not ok')


def test_buffer_protocol_ref_counting():
    def make_buffer(bytes_obj):
        return bytearray(pa.py_buffer(bytes_obj))

    buf = make_buffer(b'foo')
    gc.collect()
    assert buf == b'foo'

    # ARROW-1053
    val = b'foo'
    refcount_before = sys.getrefcount(val)
    for i in range(10):
        make_buffer(val)
    gc.collect()
    assert refcount_before == sys.getrefcount(val)


def test_nativefile_write_memoryview():
    f = pa.BufferOutputStream()
    data = b'ok'

    arr = np.frombuffer(data, dtype='S1')

    f.write(arr)
    f.write(bytearray(data))
    f.write(pa.py_buffer(data))
    with pytest.raises(TypeError):
        f.write(data.decode('utf8'))

    buf = f.getvalue()

    assert buf.to_pybytes() == data * 3


# ----------------------------------------------------------------------
# Mock output stream


def test_mock_output_stream():
    # Make sure that the MockOutputStream and the BufferOutputStream record the
    # same size

    # 10 bytes
    val = b'dataabcdef'

    f1 = pa.MockOutputStream()
    f2 = pa.BufferOutputStream()

    K = 1000
    for i in range(K):
        f1.write(val)
        f2.write(val)

    assert f1.size() == len(f2.getvalue())

    # Do the same test with a table
    record_batch = pa.RecordBatch.from_arrays([pa.array([1, 2, 3])], ['a'])

    f1 = pa.MockOutputStream()
    f2 = pa.BufferOutputStream()

    stream_writer1 = pa.RecordBatchStreamWriter(f1, record_batch.schema)
    stream_writer2 = pa.RecordBatchStreamWriter(f2, record_batch.schema)

    stream_writer1.write_batch(record_batch)
    stream_writer2.write_batch(record_batch)
    stream_writer1.close()
    stream_writer2.close()

    assert f1.size() == len(f2.getvalue())


# ----------------------------------------------------------------------
# OS files and memory maps


@pytest.fixture
def sample_disk_data(request, tmpdir):
    SIZE = 4096
    arr = np.random.randint(0, 256, size=SIZE).astype('u1')
    data = arr.tobytes()[:SIZE]

    path = os.path.join(str(tmpdir), guid())

    with open(path, 'wb') as f:
        f.write(data)

    def teardown():
        _try_delete(path)

    request.addfinalizer(teardown)
    return path, data


def _check_native_file_reader(FACTORY, sample_data,
                              allow_read_out_of_bounds=True):
    path, data = sample_data

    f = FACTORY(path, mode='r')

    assert f.read(10) == data[:10]
    assert f.read(0) == b''
    assert f.tell() == 10

    assert f.read() == data[10:]

    assert f.size() == len(data)

    f.seek(0)
    assert f.tell() == 0

    # Seeking past end of file not supported in memory maps
    if allow_read_out_of_bounds:
        f.seek(len(data) + 1)
        assert f.tell() == len(data) + 1
        assert f.read(5) == b''

    # Test whence argument of seek, ARROW-1287
    assert f.seek(3) == 3
    assert f.seek(3, os.SEEK_CUR) == 6
    assert f.tell() == 6

    ex_length = len(data) - 2
    assert f.seek(-2, os.SEEK_END) == ex_length
    assert f.tell() == ex_length


def test_memory_map_reader(sample_disk_data):
    _check_native_file_reader(pa.memory_map, sample_disk_data,
                              allow_read_out_of_bounds=False)


def test_memory_map_retain_buffer_reference(sample_disk_data):
    path, data = sample_disk_data

    cases = []
    with pa.memory_map(path, 'rb') as f:
        cases.append((f.read_buffer(100), data[:100]))
        cases.append((f.read_buffer(100), data[100:200]))
        cases.append((f.read_buffer(100), data[200:300]))

    # Call gc.collect() for good measure
    gc.collect()

    for buf, expected in cases:
        assert buf.to_pybytes() == expected


def test_os_file_reader(sample_disk_data):
    _check_native_file_reader(pa.OSFile, sample_disk_data)


def test_os_file_large_seeks():
    check_large_seeks(pa.OSFile)


def _try_delete(path):
    try:
        os.remove(path)
    except os.error:
        pass


def test_memory_map_writer(tmpdir):
    SIZE = 4096
    arr = np.random.randint(0, 256, size=SIZE).astype('u1')
    data = arr.tobytes()[:SIZE]

    path = os.path.join(str(tmpdir), guid())
    with open(path, 'wb') as f:
        f.write(data)

    f = pa.memory_map(path, mode='r+b')

    f.seek(10)
    f.write(b'peekaboo')
    assert f.tell() == 18

    f.seek(10)
    assert f.read(8) == b'peekaboo'

    f2 = pa.memory_map(path, mode='r+b')

    f2.seek(10)
    f2.write(b'booapeak')
    f2.seek(10)

    f.seek(10)
    assert f.read(8) == b'booapeak'

    # Does not truncate file
    f3 = pa.memory_map(path, mode='w')
    f3.write(b'foo')

    with pa.memory_map(path) as f4:
        assert f4.size() == SIZE

    with pytest.raises(IOError):
        f3.read(5)

    f.seek(0)
    assert f.read(3) == b'foo'


def test_memory_map_resize(tmpdir):
    SIZE = 4096
    arr = np.random.randint(0, 256, size=SIZE).astype(np.uint8)
    data1 = arr.tobytes()[:(SIZE // 2)]
    data2 = arr.tobytes()[(SIZE // 2):]

    path = os.path.join(str(tmpdir), guid())

    mmap = pa.create_memory_map(path, SIZE / 2)
    mmap.write(data1)

    mmap.resize(SIZE)
    mmap.write(data2)

    mmap.close()

    with open(path, 'rb') as f:
        assert f.read() == arr.tobytes()


def test_memory_zero_length(tmpdir):
    path = os.path.join(str(tmpdir), guid())
    f = open(path, 'wb')
    f.close()
    with pa.memory_map(path, mode='r+b') as memory_map:
        assert memory_map.size() == 0


def test_memory_map_large_seeks():
    check_large_seeks(pa.memory_map)


def test_memory_map_close_remove(tmpdir):
    # ARROW-6740: should be able to delete closed memory-mapped file (Windows)
    path = os.path.join(str(tmpdir), guid())
    mmap = pa.create_memory_map(path, 4096)
    mmap.close()
    assert mmap.closed
    os.remove(path)  # Shouldn't fail


def test_memory_map_deref_remove(tmpdir):
    path = os.path.join(str(tmpdir), guid())
    pa.create_memory_map(path, 4096)
    os.remove(path)  # Shouldn't fail


def test_os_file_writer(tmpdir):
    SIZE = 4096
    arr = np.random.randint(0, 256, size=SIZE).astype('u1')
    data = arr.tobytes()[:SIZE]

    path = os.path.join(str(tmpdir), guid())
    with open(path, 'wb') as f:
        f.write(data)

    # Truncates file
    f2 = pa.OSFile(path, mode='w')
    f2.write(b'foo')

    with pa.OSFile(path) as f3:
        assert f3.size() == 3

    with pytest.raises(IOError):
        f2.read(5)


def test_native_file_write_reject_unicode():
    # ARROW-3227
    nf = pa.BufferOutputStream()
    with pytest.raises(TypeError):
        nf.write('foo')


def test_native_file_modes(tmpdir):
    path = os.path.join(str(tmpdir), guid())
    with open(path, 'wb') as f:
        f.write(b'foooo')

    with pa.OSFile(path, mode='r') as f:
        assert f.mode == 'rb'
        assert f.readable()
        assert not f.writable()
        assert f.seekable()

    with pa.OSFile(path, mode='rb') as f:
        assert f.mode == 'rb'
        assert f.readable()
        assert not f.writable()
        assert f.seekable()

    with pa.OSFile(path, mode='w') as f:
        assert f.mode == 'wb'
        assert not f.readable()
        assert f.writable()
        assert not f.seekable()

    with pa.OSFile(path, mode='wb') as f:
        assert f.mode == 'wb'
        assert not f.readable()
        assert f.writable()
        assert not f.seekable()

    with open(path, 'wb') as f:
        f.write(b'foooo')

    with pa.memory_map(path, 'r') as f:
        assert f.mode == 'rb'
        assert f.readable()
        assert not f.writable()
        assert f.seekable()

    with pa.memory_map(path, 'r+') as f:
        assert f.mode == 'rb+'
        assert f.readable()
        assert f.writable()
        assert f.seekable()

    with pa.memory_map(path, 'r+b') as f:
        assert f.mode == 'rb+'
        assert f.readable()
        assert f.writable()
        assert f.seekable()


def test_native_file_permissions(tmpdir):
    # ARROW-10124: permissions of created files should follow umask
    cur_umask = os.umask(0o002)
    os.umask(cur_umask)

    path = os.path.join(str(tmpdir), guid())
    with pa.OSFile(path, mode='w'):
        pass
    assert os.stat(path).st_mode & 0o777 == 0o666 & ~cur_umask

    path = os.path.join(str(tmpdir), guid())
    with pa.memory_map(path, 'w'):
        pass
    assert os.stat(path).st_mode & 0o777 == 0o666 & ~cur_umask


def test_native_file_raises_ValueError_after_close(tmpdir):
    path = os.path.join(str(tmpdir), guid())
    with open(path, 'wb') as f:
        f.write(b'foooo')

    with pa.OSFile(path, mode='rb') as os_file:
        assert not os_file.closed
    assert os_file.closed

    with pa.memory_map(path, mode='rb') as mmap_file:
        assert not mmap_file.closed
    assert mmap_file.closed

    files = [os_file,
             mmap_file]

    methods = [('tell', ()),
               ('seek', (0,)),
               ('size', ()),
               ('flush', ()),
               ('readable', ()),
               ('writable', ()),
               ('seekable', ())]

    for f in files:
        for method, args in methods:
            with pytest.raises(ValueError):
                getattr(f, method)(*args)


def test_native_file_TextIOWrapper(tmpdir):
    data = ('foooo\n'
            'barrr\n'
            'bazzz\n')

    path = os.path.join(str(tmpdir), guid())
    with open(path, 'wb') as f:
        f.write(data.encode('utf-8'))

    with TextIOWrapper(pa.OSFile(path, mode='rb')) as fil:
        assert fil.readable()
        res = fil.read()
        assert res == data
    assert fil.closed

    with TextIOWrapper(pa.OSFile(path, mode='rb')) as fil:
        # Iteration works
        lines = list(fil)
        assert ''.join(lines) == data

    # Writing
    path2 = os.path.join(str(tmpdir), guid())
    with TextIOWrapper(pa.OSFile(path2, mode='wb')) as fil:
        assert fil.writable()
        fil.write(data)

    with TextIOWrapper(pa.OSFile(path2, mode='rb')) as fil:
        res = fil.read()
        assert res == data


def test_native_file_TextIOWrapper_perf(tmpdir):
    # ARROW-16272: TextIOWrapper.readline() shouldn't exhaust a large
    # Arrow input stream.
    data = b'foo\nquux\n'
    path = str(tmpdir / 'largefile.txt')
    with open(path, 'wb') as f:
        f.write(data * 100_000)

    binary_file = pa.OSFile(path, mode='rb')
    with TextIOWrapper(binary_file) as f:
        assert binary_file.tell() == 0
        nbytes = 20_000
        lines = f.readlines(nbytes)
        assert len(lines) == math.ceil(2 * nbytes / len(data))
        assert nbytes <= binary_file.tell() <= nbytes * 2


def test_native_file_read1(tmpdir):
    # ARROW-16272: read1() should not exhaust the input stream if there
    # is a large amount of data remaining.
    data = b'123\n' * 1_000_000
    path = str(tmpdir / 'largefile.txt')
    with open(path, 'wb') as f:
        f.write(data)

    chunks = []
    with pa.OSFile(path, mode='rb') as f:
        while True:
            b = f.read1()
            assert len(b) < len(data)
            chunks.append(b)
            b = f.read1(30_000)
            assert len(b) <= 30_000
            chunks.append(b)
            if not b:
                break

    assert b"".join(chunks) == data


@pytest.mark.pandas
def test_native_file_pandas_text_reader(tmpdir):
    # ARROW-16272: Pandas' read_csv() should not exhaust an Arrow
    # input stream when a small nrows is passed.
    import pandas as pd
    import pandas.testing as tm
    data = b'a,b\n' * 10_000_000
    path = str(tmpdir / 'largefile.txt')
    with open(path, 'wb') as f:
        f.write(data)

    with pa.OSFile(path, mode='rb') as f:
        df = pd.read_csv(f, nrows=10)
        expected = pd.DataFrame({'a': ['a'] * 10, 'b': ['b'] * 10})
        tm.assert_frame_equal(df, expected)
        # Some readahead occurred, but not up to the end of file
        assert f.tell() <= 256 * 1024


def test_native_file_open_error():
    with assert_file_not_found():
        pa.OSFile('non_existent_file', 'rb')
    with assert_file_not_found():
        pa.memory_map('non_existent_file', 'rb')


# ----------------------------------------------------------------------
# Buffered streams

def test_buffered_input_stream():
    raw = pa.BufferReader(b"123456789")
    f = pa.BufferedInputStream(raw, buffer_size=4)
    assert f.read(2) == b"12"
    assert raw.tell() == 4
    f.close()
    assert f.closed
    assert raw.closed


def test_buffered_input_stream_detach_seekable():
    # detach() to a seekable file (io::RandomAccessFile in C++)
    f = pa.BufferedInputStream(pa.BufferReader(b"123456789"), buffer_size=4)
    assert f.read(2) == b"12"
    raw = f.detach()
    assert f.closed
    assert not raw.closed
    assert raw.seekable()
    assert raw.read(4) == b"5678"
    raw.seek(2)
    assert raw.read(4) == b"3456"


def test_buffered_input_stream_detach_non_seekable():
    # detach() to a non-seekable file (io::InputStream in C++)
    f = pa.BufferedInputStream(
        pa.BufferedInputStream(pa.BufferReader(b"123456789"), buffer_size=4),
        buffer_size=4)
    assert f.read(2) == b"12"
    raw = f.detach()
    assert f.closed
    assert not raw.closed
    assert not raw.seekable()
    assert raw.read(4) == b"5678"
    with pytest.raises(EnvironmentError):
        raw.seek(2)


def test_buffered_output_stream():
    np_buf = np.zeros(100, dtype=np.int8)  # zero-initialized buffer
    buf = pa.py_buffer(np_buf)

    raw = pa.FixedSizeBufferWriter(buf)
    f = pa.BufferedOutputStream(raw, buffer_size=4)
    f.write(b"12")
    assert np_buf[:4].tobytes() == b'\0\0\0\0'
    f.flush()
    assert np_buf[:4].tobytes() == b'12\0\0'
    f.write(b"3456789")
    f.close()
    assert f.closed
    assert raw.closed
    assert np_buf[:10].tobytes() == b'123456789\0'


def test_buffered_output_stream_detach():
    np_buf = np.zeros(100, dtype=np.int8)  # zero-initialized buffer
    buf = pa.py_buffer(np_buf)

    f = pa.BufferedOutputStream(pa.FixedSizeBufferWriter(buf), buffer_size=4)
    f.write(b"12")
    assert np_buf[:4].tobytes() == b'\0\0\0\0'
    raw = f.detach()
    assert f.closed
    assert not raw.closed
    assert np_buf[:4].tobytes() == b'12\0\0'


# ----------------------------------------------------------------------
# Compressed input and output streams

def check_compressed_input(data, fn, compression):
    raw = pa.OSFile(fn, mode="rb")
    with pa.CompressedInputStream(raw, compression) as compressed:
        assert not compressed.closed
        assert compressed.readable()
        assert not compressed.writable()
        assert not compressed.seekable()
        got = compressed.read()
        assert got == data
    assert compressed.closed
    assert raw.closed

    # Same with read_buffer()
    raw = pa.OSFile(fn, mode="rb")
    with pa.CompressedInputStream(raw, compression) as compressed:
        buf = compressed.read_buffer()
        assert isinstance(buf, pa.Buffer)
        assert buf.to_pybytes() == data


@pytest.mark.gzip
def test_compressed_input_gzip(tmpdir):
    data = b"some test data\n" * 10 + b"eof\n"
    fn = str(tmpdir / "compressed_input_test.gz")
    with gzip.open(fn, "wb") as f:
        f.write(data)
    check_compressed_input(data, fn, "gzip")


def test_compressed_input_bz2(tmpdir):
    data = b"some test data\n" * 10 + b"eof\n"
    fn = str(tmpdir / "compressed_input_test.bz2")
    with bz2.BZ2File(fn, "w") as f:
        f.write(data)
    try:
        check_compressed_input(data, fn, "bz2")
    except NotImplementedError as e:
        pytest.skip(str(e))


@pytest.mark.gzip
def test_compressed_input_openfile(tmpdir):
    if not Codec.is_available("gzip"):
        pytest.skip("gzip support is not built")

    data = b"some test data\n" * 10 + b"eof\n"
    fn = str(tmpdir / "test_compressed_input_openfile.gz")
    with gzip.open(fn, "wb") as f:
        f.write(data)

    with pa.CompressedInputStream(fn, "gzip") as compressed:
        buf = compressed.read_buffer()
        assert buf.to_pybytes() == data
    assert compressed.closed

    with pa.CompressedInputStream(pathlib.Path(fn), "gzip") as compressed:
        buf = compressed.read_buffer()
        assert buf.to_pybytes() == data
    assert compressed.closed

    f = open(fn, "rb")
    with pa.CompressedInputStream(f, "gzip") as compressed:
        buf = compressed.read_buffer()
        assert buf.to_pybytes() == data
    assert f.closed


def check_compressed_concatenated(data, fn, compression):
    raw = pa.OSFile(fn, mode="rb")
    with pa.CompressedInputStream(raw, compression) as compressed:
        got = compressed.read()
        assert got == data


@pytest.mark.gzip
def test_compressed_concatenated_gzip(tmpdir):
    data = b"some test data\n" * 10 + b"eof\n"
    fn = str(tmpdir / "compressed_input_test2.gz")
    with gzip.open(fn, "wb") as f:
        f.write(data[:50])
    with gzip.open(fn, "ab") as f:
        f.write(data[50:])
    check_compressed_concatenated(data, fn, "gzip")


@pytest.mark.gzip
def test_compressed_input_invalid():
    data = b"foo" * 10
    raw = pa.BufferReader(data)
    with pytest.raises(ValueError):
        pa.CompressedInputStream(raw, "unknown_compression")
    with pytest.raises(TypeError):
        pa.CompressedInputStream(raw, None)

    with pa.CompressedInputStream(raw, "gzip") as compressed:
        with pytest.raises(IOError, match="zlib inflate failed"):
            compressed.read()


def make_compressed_output(data, fn, compression):
    raw = pa.BufferOutputStream()
    with pa.CompressedOutputStream(raw, compression) as compressed:
        assert not compressed.closed
        assert not compressed.readable()
        assert compressed.writable()
        assert not compressed.seekable()
        compressed.write(data)
    assert compressed.closed
    assert raw.closed
    with open(fn, "wb") as f:
        f.write(raw.getvalue())


@pytest.mark.gzip
def test_compressed_output_gzip(tmpdir):
    data = b"some test data\n" * 10 + b"eof\n"
    fn = str(tmpdir / "compressed_output_test.gz")
    make_compressed_output(data, fn, "gzip")
    with gzip.open(fn, "rb") as f:
        got = f.read()
        assert got == data


def test_compressed_output_bz2(tmpdir):
    data = b"some test data\n" * 10 + b"eof\n"
    fn = str(tmpdir / "compressed_output_test.bz2")
    try:
        make_compressed_output(data, fn, "bz2")
    except NotImplementedError as e:
        pytest.skip(str(e))
    with bz2.BZ2File(fn, "r") as f:
        got = f.read()
        assert got == data


def test_output_stream_constructor(tmpdir):
    if not Codec.is_available("gzip"):
        pytest.skip("gzip support is not built")
    with pa.CompressedOutputStream(tmpdir / "ctor.gz", "gzip") as stream:
        stream.write(b"test")
    with (tmpdir / "ctor2.gz").open("wb") as f:
        with pa.CompressedOutputStream(f, "gzip") as stream:
            stream.write(b"test")


@pytest.mark.parametrize(("path", "expected_compression"), [
    ("file.bz2", "bz2"),
    ("file.lz4", "lz4"),
    (pathlib.Path("file.gz"), "gzip"),
    (pathlib.Path("path/to/file.zst"), "zstd"),
])
def test_compression_detection(path, expected_compression):
    if not Codec.is_available(expected_compression):
        with pytest.raises(pa.lib.ArrowNotImplementedError):
            Codec.detect(path)
    else:
        codec = Codec.detect(path)
        assert isinstance(codec, Codec)
        assert codec.name == expected_compression


def test_unknown_compression_raises():
    with pytest.raises(ValueError):
        Codec.is_available('unknown')
    with pytest.raises(TypeError):
        Codec(None)
    with pytest.raises(ValueError):
        Codec('unknown')


@pytest.mark.parametrize("compression", [
    "bz2",
    "brotli",
    "gzip",
    "lz4",
    "zstd",
    pytest.param(
        "snappy",
        marks=pytest.mark.xfail(raises=pa.lib.ArrowNotImplementedError)
    )
])
def test_compressed_roundtrip(compression):
    if not Codec.is_available(compression):
        pytest.skip("{} support is not built".format(compression))

    data = b"some test data\n" * 10 + b"eof\n"
    raw = pa.BufferOutputStream()
    with pa.CompressedOutputStream(raw, compression) as compressed:
        compressed.write(data)

    cdata = raw.getvalue()
    assert len(cdata) < len(data)
    raw = pa.BufferReader(cdata)
    with pa.CompressedInputStream(raw, compression) as compressed:
        got = compressed.read()
        assert got == data


@pytest.mark.parametrize(
    "compression",
    ["bz2", "brotli", "gzip", "lz4", "zstd"]
)
def test_compressed_recordbatch_stream(compression):
    if not Codec.is_available(compression):
        pytest.skip("{} support is not built".format(compression))

    # ARROW-4836: roundtrip a RecordBatch through a compressed stream
    table = pa.Table.from_arrays([pa.array([1, 2, 3, 4, 5])], ['a'])
    raw = pa.BufferOutputStream()
    stream = pa.CompressedOutputStream(raw, compression)
    writer = pa.RecordBatchStreamWriter(stream, table.schema)
    writer.write_table(table, max_chunksize=3)
    writer.close()
    stream.close()  # Flush data
    buf = raw.getvalue()
    stream = pa.CompressedInputStream(pa.BufferReader(buf), compression)
    got_table = pa.RecordBatchStreamReader(stream).read_all()
    assert got_table == table


# ----------------------------------------------------------------------
# Transform input streams

unicode_transcoding_example = (
    "Dès Noël où un zéphyr haï me vêt de glaçons würmiens "
    "je dîne d’exquis rôtis de bœuf au kir à l’aÿ d’âge mûr & cætera !"
)


def check_transcoding(data, src_encoding, dest_encoding, chunk_sizes):
    chunk_sizes = iter(chunk_sizes)
    stream = pa.transcoding_input_stream(
        pa.BufferReader(data.encode(src_encoding)),
        src_encoding, dest_encoding)
    out = []
    while True:
        buf = stream.read(next(chunk_sizes))
        out.append(buf)
        if not buf:
            break
    out = b''.join(out)
    assert out.decode(dest_encoding) == data


@pytest.mark.parametrize('src_encoding, dest_encoding',
                         [('utf-8', 'utf-16'),
                          ('utf-16', 'utf-8'),
                          ('utf-8', 'utf-32-le'),
                          ('utf-8', 'utf-32-be'),
                          ])
def test_transcoding_input_stream(src_encoding, dest_encoding):
    # All at once
    check_transcoding(unicode_transcoding_example,
                      src_encoding, dest_encoding, [1000, 0])
    # Incremental
    check_transcoding(unicode_transcoding_example,
                      src_encoding, dest_encoding,
                      itertools.cycle([1, 2, 3, 5]))


@pytest.mark.parametrize('src_encoding, dest_encoding',
                         [('utf-8', 'utf-8'),
                          ('utf-8', 'UTF8')])
def test_transcoding_no_ops(src_encoding, dest_encoding):
    # No indirection is wasted when a trivial transcoding is requested
    stream = pa.BufferReader(b"abc123")
    assert pa.transcoding_input_stream(
        stream, src_encoding, dest_encoding) is stream


@pytest.mark.parametrize('src_encoding, dest_encoding',
                         [('utf-8', 'ascii'),
                          ('utf-8', 'latin-1'),
                          ])
def test_transcoding_encoding_error(src_encoding, dest_encoding):
    # Character \u0100 cannot be represented in the destination encoding
    stream = pa.transcoding_input_stream(
        pa.BufferReader("\u0100".encode(src_encoding)),
        src_encoding,
        dest_encoding)
    with pytest.raises(UnicodeEncodeError):
        stream.read(1)


@pytest.mark.parametrize('src_encoding, dest_encoding',
                         [('utf-8', 'utf-16'),
                          ('utf-16', 'utf-8'),
                          ])
def test_transcoding_decoding_error(src_encoding, dest_encoding):
    # The given bytestring is not valid in the source encoding
    stream = pa.transcoding_input_stream(
        pa.BufferReader(b"\xff\xff\xff\xff"),
        src_encoding,
        dest_encoding)
    with pytest.raises(UnicodeError):
        stream.read(1)


# ----------------------------------------------------------------------
# High-level API

@pytest.mark.gzip
def test_input_stream_buffer():
    data = b"some test data\n" * 10 + b"eof\n"
    for arg in [pa.py_buffer(data), memoryview(data)]:
        stream = pa.input_stream(arg)
        assert stream.read() == data

    gz_data = gzip.compress(data)
    stream = pa.input_stream(memoryview(gz_data))
    assert stream.read() == gz_data
    stream = pa.input_stream(memoryview(gz_data), compression='gzip')
    assert stream.read() == data


def test_input_stream_duck_typing():
    # Accept objects having the right file-like methods...
    class DuckReader:

        def close(self):
            pass

        @property
        def closed(self):
            return False

        def read(self, nbytes=None):
            return b'hello'

    stream = pa.input_stream(DuckReader())
    assert stream.read(5) == b'hello'


def test_input_stream_file_path(tmpdir):
    data = b"some test data\n" * 10 + b"eof\n"
    file_path = tmpdir / 'input_stream'
    with open(str(file_path), 'wb') as f:
        f.write(data)

    stream = pa.input_stream(file_path)
    assert stream.read() == data
    stream = pa.input_stream(str(file_path))
    assert stream.read() == data
    stream = pa.input_stream(pathlib.Path(str(file_path)))
    assert stream.read() == data


@pytest.mark.gzip
def test_input_stream_file_path_compressed(tmpdir):
    data = b"some test data\n" * 10 + b"eof\n"
    gz_data = gzip.compress(data)
    file_path = tmpdir / 'input_stream.gz'
    with open(str(file_path), 'wb') as f:
        f.write(gz_data)

    stream = pa.input_stream(file_path)
    assert stream.read() == data
    stream = pa.input_stream(str(file_path))
    assert stream.read() == data
    stream = pa.input_stream(pathlib.Path(str(file_path)))
    assert stream.read() == data

    stream = pa.input_stream(file_path, compression='gzip')
    assert stream.read() == data
    stream = pa.input_stream(file_path, compression=None)
    assert stream.read() == gz_data


def test_input_stream_file_path_buffered(tmpdir):
    data = b"some test data\n" * 10 + b"eof\n"
    file_path = tmpdir / 'input_stream.buffered'
    with open(str(file_path), 'wb') as f:
        f.write(data)

    stream = pa.input_stream(file_path, buffer_size=32)
    assert isinstance(stream, pa.BufferedInputStream)
    assert stream.read() == data
    stream = pa.input_stream(str(file_path), buffer_size=64)
    assert isinstance(stream, pa.BufferedInputStream)
    assert stream.read() == data
    stream = pa.input_stream(pathlib.Path(str(file_path)), buffer_size=1024)
    assert isinstance(stream, pa.BufferedInputStream)
    assert stream.read() == data

    unbuffered_stream = pa.input_stream(file_path, buffer_size=0)
    assert isinstance(unbuffered_stream, pa.OSFile)

    msg = 'Buffer size must be larger than zero'
    with pytest.raises(ValueError, match=msg):
        pa.input_stream(file_path, buffer_size=-1)
    with pytest.raises(TypeError):
        pa.input_stream(file_path, buffer_size='million')


@pytest.mark.gzip
def test_input_stream_file_path_compressed_and_buffered(tmpdir):
    data = b"some test data\n" * 100 + b"eof\n"
    gz_data = gzip.compress(data)
    file_path = tmpdir / 'input_stream_compressed_and_buffered.gz'
    with open(str(file_path), 'wb') as f:
        f.write(gz_data)

    stream = pa.input_stream(file_path, buffer_size=32, compression='gzip')
    assert stream.read() == data
    stream = pa.input_stream(str(file_path), buffer_size=64)
    assert stream.read() == data
    stream = pa.input_stream(pathlib.Path(str(file_path)), buffer_size=1024)
    assert stream.read() == data


@pytest.mark.gzip
def test_input_stream_python_file(tmpdir):
    data = b"some test data\n" * 10 + b"eof\n"
    bio = BytesIO(data)

    stream = pa.input_stream(bio)
    assert stream.read() == data

    gz_data = gzip.compress(data)
    bio = BytesIO(gz_data)
    stream = pa.input_stream(bio)
    assert stream.read() == gz_data
    bio.seek(0)
    stream = pa.input_stream(bio, compression='gzip')
    assert stream.read() == data

    file_path = tmpdir / 'input_stream'
    with open(str(file_path), 'wb') as f:
        f.write(data)
    with open(str(file_path), 'rb') as f:
        stream = pa.input_stream(f)
        assert stream.read() == data


@pytest.mark.gzip
def test_input_stream_native_file():
    data = b"some test data\n" * 10 + b"eof\n"
    gz_data = gzip.compress(data)
    reader = pa.BufferReader(gz_data)
    stream = pa.input_stream(reader)
    assert stream is reader
    reader = pa.BufferReader(gz_data)
    stream = pa.input_stream(reader, compression='gzip')
    assert stream.read() == data


def test_input_stream_errors(tmpdir):
    buf = memoryview(b"")
    with pytest.raises(ValueError):
        pa.input_stream(buf, compression="foo")

    for arg in [bytearray(), StringIO()]:
        with pytest.raises(TypeError):
            pa.input_stream(arg)

    with assert_file_not_found():
        pa.input_stream("non_existent_file")

    with open(str(tmpdir / 'new_file'), 'wb') as f:
        with pytest.raises(TypeError, match="readable file expected"):
            pa.input_stream(f)


def test_output_stream_buffer():
    data = b"some test data\n" * 10 + b"eof\n"
    buf = bytearray(len(data))
    stream = pa.output_stream(pa.py_buffer(buf))
    stream.write(data)
    assert buf == data

    buf = bytearray(len(data))
    stream = pa.output_stream(memoryview(buf))
    stream.write(data)
    assert buf == data


def test_output_stream_duck_typing():
    # Accept objects having the right file-like methods...
    class DuckWriter:
        def __init__(self):
            self.buf = pa.BufferOutputStream()

        def close(self):
            pass

        @property
        def closed(self):
            return False

        def write(self, data):
            self.buf.write(data)

    duck_writer = DuckWriter()
    stream = pa.output_stream(duck_writer)
    assert stream.write(b'hello')
    assert duck_writer.buf.getvalue().to_pybytes() == b'hello'


def test_output_stream_file_path(tmpdir):
    data = b"some test data\n" * 10 + b"eof\n"
    file_path = tmpdir / 'output_stream'

    def check_data(file_path, data):
        with pa.output_stream(file_path) as stream:
            stream.write(data)
        with open(str(file_path), 'rb') as f:
            assert f.read() == data

    check_data(file_path, data)
    check_data(str(file_path), data)
    check_data(pathlib.Path(str(file_path)), data)


@pytest.mark.gzip
def test_output_stream_file_path_compressed(tmpdir):
    data = b"some test data\n" * 10 + b"eof\n"
    file_path = tmpdir / 'output_stream.gz'

    def check_data(file_path, data, **kwargs):
        with pa.output_stream(file_path, **kwargs) as stream:
            stream.write(data)
        with open(str(file_path), 'rb') as f:
            return f.read()

    assert gzip.decompress(check_data(file_path, data)) == data
    assert gzip.decompress(check_data(str(file_path), data)) == data
    assert gzip.decompress(
        check_data(pathlib.Path(str(file_path)), data)) == data

    assert gzip.decompress(
        check_data(file_path, data, compression='gzip')) == data
    assert check_data(file_path, data, compression=None) == data

    with pytest.raises(ValueError, match='Invalid value for compression'):
        assert check_data(file_path, data, compression='rabbit') == data


def test_output_stream_file_path_buffered(tmpdir):
    data = b"some test data\n" * 10 + b"eof\n"
    file_path = tmpdir / 'output_stream.buffered'

    def check_data(file_path, data, **kwargs):
        with pa.output_stream(file_path, **kwargs) as stream:
            if kwargs.get('buffer_size', 0) > 0:
                assert isinstance(stream, pa.BufferedOutputStream)
            stream.write(data)
        with open(str(file_path), 'rb') as f:
            return f.read()

    unbuffered_stream = pa.output_stream(file_path, buffer_size=0)
    assert isinstance(unbuffered_stream, pa.OSFile)

    msg = 'Buffer size must be larger than zero'
    with pytest.raises(ValueError, match=msg):
        assert check_data(file_path, data, buffer_size=-128) == data

    assert check_data(file_path, data, buffer_size=32) == data
    assert check_data(file_path, data, buffer_size=1024) == data
    assert check_data(str(file_path), data, buffer_size=32) == data

    result = check_data(pathlib.Path(str(file_path)), data, buffer_size=32)
    assert result == data


@pytest.mark.gzip
def test_output_stream_file_path_compressed_and_buffered(tmpdir):
    data = b"some test data\n" * 100 + b"eof\n"
    file_path = tmpdir / 'output_stream_compressed_and_buffered.gz'

    def check_data(file_path, data, **kwargs):
        with pa.output_stream(file_path, **kwargs) as stream:
            stream.write(data)
        with open(str(file_path), 'rb') as f:
            return f.read()

    result = check_data(file_path, data, buffer_size=32)
    assert gzip.decompress(result) == data

    result = check_data(file_path, data, buffer_size=1024)
    assert gzip.decompress(result) == data

    result = check_data(file_path, data, buffer_size=1024, compression='gzip')
    assert gzip.decompress(result) == data


def test_output_stream_destructor(tmpdir):
    # The wrapper returned by pa.output_stream() should respect Python
    # file semantics, i.e. destroying it should close the underlying
    # file cleanly.
    data = b"some test data\n"
    file_path = tmpdir / 'output_stream.buffered'

    def check_data(file_path, data, **kwargs):
        stream = pa.output_stream(file_path, **kwargs)
        stream.write(data)
        del stream
        gc.collect()
        with open(str(file_path), 'rb') as f:
            return f.read()

    assert check_data(file_path, data, buffer_size=0) == data
    assert check_data(file_path, data, buffer_size=1024) == data


@pytest.mark.gzip
def test_output_stream_python_file(tmpdir):
    data = b"some test data\n" * 10 + b"eof\n"

    def check_data(data, **kwargs):
        # XXX cannot use BytesIO because stream.close() is necessary
        # to finish writing compressed data, but it will also close the
        # underlying BytesIO
        fn = str(tmpdir / 'output_stream_file')
        with open(fn, 'wb') as f:
            with pa.output_stream(f, **kwargs) as stream:
                stream.write(data)
        with open(fn, 'rb') as f:
            return f.read()

    assert check_data(data) == data
    assert gzip.decompress(check_data(data, compression='gzip')) == data


def test_output_stream_errors(tmpdir):
    buf = memoryview(bytearray())
    with pytest.raises(ValueError):
        pa.output_stream(buf, compression="foo")

    for arg in [bytearray(), StringIO()]:
        with pytest.raises(TypeError):
            pa.output_stream(arg)

    fn = str(tmpdir / 'new_file')
    with open(fn, 'wb') as f:
        pass
    with open(fn, 'rb') as f:
        with pytest.raises(TypeError, match="writable file expected"):
            pa.output_stream(f)

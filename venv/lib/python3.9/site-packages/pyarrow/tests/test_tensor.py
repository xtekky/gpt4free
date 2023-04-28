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

import os
import sys
import pytest
import weakref

import numpy as np
import pyarrow as pa


tensor_type_pairs = [
    ('i1', pa.int8()),
    ('i2', pa.int16()),
    ('i4', pa.int32()),
    ('i8', pa.int64()),
    ('u1', pa.uint8()),
    ('u2', pa.uint16()),
    ('u4', pa.uint32()),
    ('u8', pa.uint64()),
    ('f2', pa.float16()),
    ('f4', pa.float32()),
    ('f8', pa.float64())
]


def test_tensor_attrs():
    data = np.random.randn(10, 4)

    tensor = pa.Tensor.from_numpy(data)

    assert tensor.ndim == 2
    assert tensor.dim_names == []
    assert tensor.size == 40
    assert tensor.shape == data.shape
    assert tensor.strides == data.strides

    assert tensor.is_contiguous
    assert tensor.is_mutable

    # not writeable
    data2 = data.copy()
    data2.flags.writeable = False
    tensor = pa.Tensor.from_numpy(data2)
    assert not tensor.is_mutable

    # With dim_names
    tensor = pa.Tensor.from_numpy(data, dim_names=('x', 'y'))
    assert tensor.ndim == 2
    assert tensor.dim_names == ['x', 'y']
    assert tensor.dim_name(0) == 'x'
    assert tensor.dim_name(1) == 'y'

    wr = weakref.ref(tensor)
    assert wr() is not None
    del tensor
    assert wr() is None


def test_tensor_base_object():
    tensor = pa.Tensor.from_numpy(np.random.randn(10, 4))
    n = sys.getrefcount(tensor)
    array = tensor.to_numpy()  # noqa
    assert sys.getrefcount(tensor) == n + 1


@pytest.mark.parametrize('dtype_str,arrow_type', tensor_type_pairs)
def test_tensor_numpy_roundtrip(dtype_str, arrow_type):
    dtype = np.dtype(dtype_str)
    data = (100 * np.random.randn(10, 4)).astype(dtype)

    tensor = pa.Tensor.from_numpy(data)
    assert tensor.type == arrow_type

    repr(tensor)

    result = tensor.to_numpy()
    assert (data == result).all()


def test_tensor_ipc_roundtrip(tmpdir):
    data = np.random.randn(10, 4)
    tensor = pa.Tensor.from_numpy(data)

    path = os.path.join(str(tmpdir), 'pyarrow-tensor-ipc-roundtrip')
    mmap = pa.create_memory_map(path, 1024)

    pa.ipc.write_tensor(tensor, mmap)

    mmap.seek(0)
    result = pa.ipc.read_tensor(mmap)

    assert result.equals(tensor)


@pytest.mark.gzip
def test_tensor_ipc_read_from_compressed(tempdir):
    # ARROW-5910
    data = np.random.randn(10, 4)
    tensor = pa.Tensor.from_numpy(data)

    path = tempdir / 'tensor-compressed-file'

    out_stream = pa.output_stream(path, compression='gzip')
    pa.ipc.write_tensor(tensor, out_stream)
    out_stream.close()

    result = pa.ipc.read_tensor(pa.input_stream(path, compression='gzip'))
    assert result.equals(tensor)


def test_tensor_ipc_strided(tmpdir):
    data1 = np.random.randn(10, 4)
    tensor1 = pa.Tensor.from_numpy(data1[::2])

    data2 = np.random.randn(10, 6, 4)
    tensor2 = pa.Tensor.from_numpy(data2[::, ::2, ::])

    path = os.path.join(str(tmpdir), 'pyarrow-tensor-ipc-strided')
    mmap = pa.create_memory_map(path, 2048)

    for tensor in [tensor1, tensor2]:
        mmap.seek(0)
        pa.ipc.write_tensor(tensor, mmap)

        mmap.seek(0)
        result = pa.ipc.read_tensor(mmap)

        assert result.equals(tensor)


def test_tensor_equals():
    def eq(a, b):
        assert a.equals(b)
        assert a == b
        assert not (a != b)

    def ne(a, b):
        assert not a.equals(b)
        assert not (a == b)
        assert a != b

    data = np.random.randn(10, 6, 4)[::, ::2, ::]
    tensor1 = pa.Tensor.from_numpy(data)
    tensor2 = pa.Tensor.from_numpy(np.ascontiguousarray(data))
    eq(tensor1, tensor2)
    data = data.copy()
    data[9, 0, 0] = 1.0
    tensor2 = pa.Tensor.from_numpy(np.ascontiguousarray(data))
    ne(tensor1, tensor2)


def test_tensor_hashing():
    # Tensors are unhashable
    with pytest.raises(TypeError, match="unhashable"):
        hash(pa.Tensor.from_numpy(np.arange(10)))


def test_tensor_size():
    data = np.random.randn(10, 4)
    tensor = pa.Tensor.from_numpy(data)
    assert pa.ipc.get_tensor_size(tensor) > (data.size * 8)


def test_read_tensor(tmpdir):
    # Create and write tensor tensor
    data = np.random.randn(10, 4)
    tensor = pa.Tensor.from_numpy(data)
    data_size = pa.ipc.get_tensor_size(tensor)
    path = os.path.join(str(tmpdir), 'pyarrow-tensor-ipc-read-tensor')
    write_mmap = pa.create_memory_map(path, data_size)
    pa.ipc.write_tensor(tensor, write_mmap)
    # Try to read tensor
    read_mmap = pa.memory_map(path, mode='r')
    array = pa.ipc.read_tensor(read_mmap).to_numpy()
    np.testing.assert_equal(data, array)


def test_tensor_memoryview():
    # Tensors support the PEP 3118 buffer protocol
    for dtype, expected_format in [(np.int8, '=b'),
                                   (np.int64, '=q'),
                                   (np.uint64, '=Q'),
                                   (np.float16, 'e'),
                                   (np.float64, 'd'),
                                   ]:
        data = np.arange(10, dtype=dtype)
        dtype = data.dtype
        lst = data.tolist()
        tensor = pa.Tensor.from_numpy(data)
        m = memoryview(tensor)
        assert m.format == expected_format
        assert m.shape == data.shape
        assert m.strides == data.strides
        assert m.ndim == 1
        assert m.nbytes == data.nbytes
        assert m.itemsize == data.itemsize
        assert m.itemsize * 8 == tensor.type.bit_width
        assert np.frombuffer(m, dtype).tolist() == lst
        del tensor, data
        assert np.frombuffer(m, dtype).tolist() == lst

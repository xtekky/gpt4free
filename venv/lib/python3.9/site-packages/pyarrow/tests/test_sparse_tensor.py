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
import sys
import weakref

import numpy as np
import pyarrow as pa

try:
    from scipy.sparse import csr_matrix, coo_matrix
except ImportError:
    coo_matrix = None
    csr_matrix = None

try:
    import sparse
except ImportError:
    sparse = None


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


@pytest.mark.parametrize('sparse_tensor_type', [
    pa.SparseCSRMatrix,
    pa.SparseCSCMatrix,
    pa.SparseCOOTensor,
    pa.SparseCSFTensor,
])
def test_sparse_tensor_attrs(sparse_tensor_type):
    data = np.array([
        [8, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 5],
        [3, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 6],
    ])
    dim_names = ('x', 'y')
    sparse_tensor = sparse_tensor_type.from_dense_numpy(data, dim_names)

    assert sparse_tensor.ndim == 2
    assert sparse_tensor.size == 24
    assert sparse_tensor.shape == data.shape
    assert sparse_tensor.is_mutable
    assert sparse_tensor.dim_name(0) == dim_names[0]
    assert sparse_tensor.dim_names == dim_names
    assert sparse_tensor.non_zero_length == 6

    wr = weakref.ref(sparse_tensor)
    assert wr() is not None
    del sparse_tensor
    assert wr() is None


def test_sparse_coo_tensor_base_object():
    expected_data = np.array([[8, 2, 5, 3, 4, 6]]).T
    expected_coords = np.array([
        [0, 0, 1, 2, 3, 3],
        [0, 2, 5, 0, 4, 5],
    ]).T
    array = np.array([
        [8, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 5],
        [3, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 6],
    ])
    sparse_tensor = pa.SparseCOOTensor.from_dense_numpy(array)
    n = sys.getrefcount(sparse_tensor)
    result_data, result_coords = sparse_tensor.to_numpy()
    assert sparse_tensor.has_canonical_format
    assert sys.getrefcount(sparse_tensor) == n + 2

    sparse_tensor = None
    assert np.array_equal(expected_data, result_data)
    assert np.array_equal(expected_coords, result_coords)
    assert result_coords.flags.c_contiguous  # row-major


def test_sparse_csr_matrix_base_object():
    data = np.array([[8, 2, 5, 3, 4, 6]]).T
    indptr = np.array([0, 2, 3, 4, 6])
    indices = np.array([0, 2, 5, 0, 4, 5])
    array = np.array([
        [8, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 5],
        [3, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 6],
    ])
    sparse_tensor = pa.SparseCSRMatrix.from_dense_numpy(array)
    n = sys.getrefcount(sparse_tensor)
    result_data, result_indptr, result_indices = sparse_tensor.to_numpy()
    assert sys.getrefcount(sparse_tensor) == n + 3

    sparse_tensor = None
    assert np.array_equal(data, result_data)
    assert np.array_equal(indptr, result_indptr)
    assert np.array_equal(indices, result_indices)


def test_sparse_csf_tensor_base_object():
    data = np.array([[8, 2, 5, 3, 4, 6]]).T
    indptr = [np.array([0, 2, 3, 4, 6])]
    indices = [
        np.array([0, 1, 2, 3]),
        np.array([0, 2, 5, 0, 4, 5])
    ]
    array = np.array([
        [8, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 5],
        [3, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 6],
    ])
    sparse_tensor = pa.SparseCSFTensor.from_dense_numpy(array)
    n = sys.getrefcount(sparse_tensor)
    result_data, result_indptr, result_indices = sparse_tensor.to_numpy()
    assert sys.getrefcount(sparse_tensor) == n + 4

    sparse_tensor = None
    assert np.array_equal(data, result_data)
    assert np.array_equal(indptr[0], result_indptr[0])
    assert np.array_equal(indices[0], result_indices[0])
    assert np.array_equal(indices[1], result_indices[1])


@pytest.mark.parametrize('sparse_tensor_type', [
    pa.SparseCSRMatrix,
    pa.SparseCSCMatrix,
    pa.SparseCOOTensor,
    pa.SparseCSFTensor,
])
def test_sparse_tensor_equals(sparse_tensor_type):
    def eq(a, b):
        assert a.equals(b)
        assert a == b
        assert not (a != b)

    def ne(a, b):
        assert not a.equals(b)
        assert not (a == b)
        assert a != b

    data = np.random.randn(10, 6)[::, ::2]
    sparse_tensor1 = sparse_tensor_type.from_dense_numpy(data)
    sparse_tensor2 = sparse_tensor_type.from_dense_numpy(
        np.ascontiguousarray(data))
    eq(sparse_tensor1, sparse_tensor2)
    data = data.copy()
    data[9, 0] = 1.0
    sparse_tensor2 = sparse_tensor_type.from_dense_numpy(
        np.ascontiguousarray(data))
    ne(sparse_tensor1, sparse_tensor2)


@pytest.mark.parametrize('dtype_str,arrow_type', tensor_type_pairs)
def test_sparse_coo_tensor_from_dense(dtype_str, arrow_type):
    dtype = np.dtype(dtype_str)
    expected_data = np.array([[8, 2, 5, 3, 4, 6]]).T.astype(dtype)
    expected_coords = np.array([
        [0, 0, 1, 2, 3, 3],
        [0, 2, 5, 0, 4, 5],
    ]).T
    array = np.array([
        [8, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 5],
        [3, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 6],
    ]).astype(dtype)
    tensor = pa.Tensor.from_numpy(array)

    # Test from numpy array
    sparse_tensor = pa.SparseCOOTensor.from_dense_numpy(array)
    repr(sparse_tensor)
    result_data, result_coords = sparse_tensor.to_numpy()
    assert sparse_tensor.type == arrow_type
    assert np.array_equal(expected_data, result_data)
    assert np.array_equal(expected_coords, result_coords)

    # Test from Tensor
    sparse_tensor = pa.SparseCOOTensor.from_tensor(tensor)
    repr(sparse_tensor)
    result_data, result_coords = sparse_tensor.to_numpy()
    assert sparse_tensor.type == arrow_type
    assert np.array_equal(expected_data, result_data)
    assert np.array_equal(expected_coords, result_coords)


@pytest.mark.parametrize('dtype_str,arrow_type', tensor_type_pairs)
def test_sparse_csr_matrix_from_dense(dtype_str, arrow_type):
    dtype = np.dtype(dtype_str)
    data = np.array([[8, 2, 5, 3, 4, 6]]).T.astype(dtype)
    indptr = np.array([0, 2, 3, 4, 6])
    indices = np.array([0, 2, 5, 0, 4, 5])
    array = np.array([
        [8, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 5],
        [3, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 6],
    ]).astype(dtype)
    tensor = pa.Tensor.from_numpy(array)

    # Test from numpy array
    sparse_tensor = pa.SparseCSRMatrix.from_dense_numpy(array)
    repr(sparse_tensor)
    result_data, result_indptr, result_indices = sparse_tensor.to_numpy()
    assert sparse_tensor.type == arrow_type
    assert np.array_equal(data, result_data)
    assert np.array_equal(indptr, result_indptr)
    assert np.array_equal(indices, result_indices)

    # Test from Tensor
    sparse_tensor = pa.SparseCSRMatrix.from_tensor(tensor)
    repr(sparse_tensor)
    result_data, result_indptr, result_indices = sparse_tensor.to_numpy()
    assert sparse_tensor.type == arrow_type
    assert np.array_equal(data, result_data)
    assert np.array_equal(indptr, result_indptr)
    assert np.array_equal(indices, result_indices)


@pytest.mark.parametrize('dtype_str,arrow_type', tensor_type_pairs)
def test_sparse_csf_tensor_from_dense_numpy(dtype_str, arrow_type):
    dtype = np.dtype(dtype_str)
    data = np.array([[8, 2, 5, 3, 4, 6]]).T.astype(dtype)
    indptr = [np.array([0, 2, 3, 4, 6])]
    indices = [
        np.array([0, 1, 2, 3]),
        np.array([0, 2, 5, 0, 4, 5])
    ]
    array = np.array([
        [8, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 5],
        [3, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 6],
    ]).astype(dtype)

    # Test from numpy array
    sparse_tensor = pa.SparseCSFTensor.from_dense_numpy(array)
    repr(sparse_tensor)
    result_data, result_indptr, result_indices = sparse_tensor.to_numpy()
    assert sparse_tensor.type == arrow_type
    assert np.array_equal(data, result_data)
    assert np.array_equal(indptr[0], result_indptr[0])
    assert np.array_equal(indices[0], result_indices[0])
    assert np.array_equal(indices[1], result_indices[1])


@pytest.mark.parametrize('dtype_str,arrow_type', tensor_type_pairs)
def test_sparse_csf_tensor_from_dense_tensor(dtype_str, arrow_type):
    dtype = np.dtype(dtype_str)
    data = np.array([[8, 2, 5, 3, 4, 6]]).T.astype(dtype)
    indptr = [np.array([0, 2, 3, 4, 6])]
    indices = [
        np.array([0, 1, 2, 3]),
        np.array([0, 2, 5, 0, 4, 5])
    ]
    array = np.array([
        [8, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 5],
        [3, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 6],
    ]).astype(dtype)
    tensor = pa.Tensor.from_numpy(array)

    # Test from Tensor
    sparse_tensor = pa.SparseCSFTensor.from_tensor(tensor)
    repr(sparse_tensor)
    result_data, result_indptr, result_indices = sparse_tensor.to_numpy()
    assert sparse_tensor.type == arrow_type
    assert np.array_equal(data, result_data)
    assert np.array_equal(indptr[0], result_indptr[0])
    assert np.array_equal(indices[0], result_indices[0])
    assert np.array_equal(indices[1], result_indices[1])


@pytest.mark.parametrize('dtype_str,arrow_type', tensor_type_pairs)
def test_sparse_coo_tensor_numpy_roundtrip(dtype_str, arrow_type):
    dtype = np.dtype(dtype_str)
    data = np.array([[1, 2, 3, 4, 5, 6]]).T.astype(dtype)
    coords = np.array([
        [0, 0, 2, 3, 1, 3],
        [0, 2, 0, 4, 5, 5],
    ]).T
    shape = (4, 6)
    dim_names = ('x', 'y')

    sparse_tensor = pa.SparseCOOTensor.from_numpy(data, coords, shape,
                                                  dim_names)
    repr(sparse_tensor)
    result_data, result_coords = sparse_tensor.to_numpy()
    assert sparse_tensor.type == arrow_type
    assert np.array_equal(data, result_data)
    assert np.array_equal(coords, result_coords)
    assert sparse_tensor.dim_names == dim_names


@pytest.mark.parametrize('dtype_str,arrow_type', tensor_type_pairs)
def test_sparse_csr_matrix_numpy_roundtrip(dtype_str, arrow_type):
    dtype = np.dtype(dtype_str)
    data = np.array([[8, 2, 5, 3, 4, 6]]).T.astype(dtype)
    indptr = np.array([0, 2, 3, 4, 6])
    indices = np.array([0, 2, 5, 0, 4, 5])
    shape = (4, 6)
    dim_names = ('x', 'y')

    sparse_tensor = pa.SparseCSRMatrix.from_numpy(data, indptr, indices,
                                                  shape, dim_names)
    repr(sparse_tensor)
    result_data, result_indptr, result_indices = sparse_tensor.to_numpy()
    assert sparse_tensor.type == arrow_type
    assert np.array_equal(data, result_data)
    assert np.array_equal(indptr, result_indptr)
    assert np.array_equal(indices, result_indices)
    assert sparse_tensor.dim_names == dim_names


@pytest.mark.parametrize('dtype_str,arrow_type', tensor_type_pairs)
def test_sparse_csf_tensor_numpy_roundtrip(dtype_str, arrow_type):
    dtype = np.dtype(dtype_str)
    data = np.array([[8, 2, 5, 3, 4, 6]]).T.astype(dtype)
    indptr = [np.array([0, 2, 3, 4, 6])]
    indices = [
        np.array([0, 1, 2, 3]),
        np.array([0, 2, 5, 0, 4, 5])
    ]
    axis_order = (0, 1)
    shape = (4, 6)
    dim_names = ('x', 'y')

    sparse_tensor = pa.SparseCSFTensor.from_numpy(data, indptr, indices,
                                                  shape, axis_order,
                                                  dim_names)
    repr(sparse_tensor)
    result_data, result_indptr, result_indices = sparse_tensor.to_numpy()
    assert sparse_tensor.type == arrow_type
    assert np.array_equal(data, result_data)
    assert np.array_equal(indptr[0], result_indptr[0])
    assert np.array_equal(indices[0], result_indices[0])
    assert np.array_equal(indices[1], result_indices[1])
    assert sparse_tensor.dim_names == dim_names


@pytest.mark.parametrize('sparse_tensor_type', [
    pa.SparseCSRMatrix,
    pa.SparseCSCMatrix,
    pa.SparseCOOTensor,
    pa.SparseCSFTensor,
])
@pytest.mark.parametrize('dtype_str,arrow_type', tensor_type_pairs)
def test_dense_to_sparse_tensor(dtype_str, arrow_type, sparse_tensor_type):
    dtype = np.dtype(dtype_str)
    array = np.array([[4, 0, 9, 0],
                      [0, 7, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 5]]).astype(dtype)
    dim_names = ('x', 'y')

    sparse_tensor = sparse_tensor_type.from_dense_numpy(array, dim_names)
    tensor = sparse_tensor.to_tensor()
    result_array = tensor.to_numpy()

    assert sparse_tensor.type == arrow_type
    assert tensor.type == arrow_type
    assert sparse_tensor.dim_names == dim_names
    assert np.array_equal(array, result_array)


@pytest.mark.skipif(not coo_matrix, reason="requires scipy")
@pytest.mark.parametrize('dtype_str,arrow_type', tensor_type_pairs)
def test_sparse_coo_tensor_scipy_roundtrip(dtype_str, arrow_type):
    dtype = np.dtype(dtype_str)
    data = np.array([1, 2, 3, 4, 5, 6]).astype(dtype)
    row = np.array([0, 0, 2, 3, 1, 3])
    col = np.array([0, 2, 0, 4, 5, 5])
    shape = (4, 6)
    dim_names = ('x', 'y')

    # non-canonical sparse coo matrix
    scipy_matrix = coo_matrix((data, (row, col)), shape=shape)
    sparse_tensor = pa.SparseCOOTensor.from_scipy(scipy_matrix,
                                                  dim_names=dim_names)
    out_scipy_matrix = sparse_tensor.to_scipy()

    assert not scipy_matrix.has_canonical_format
    assert not sparse_tensor.has_canonical_format
    assert not out_scipy_matrix.has_canonical_format
    assert sparse_tensor.type == arrow_type
    assert sparse_tensor.dim_names == dim_names
    assert scipy_matrix.dtype == out_scipy_matrix.dtype
    assert np.array_equal(scipy_matrix.data, out_scipy_matrix.data)
    assert np.array_equal(scipy_matrix.row, out_scipy_matrix.row)
    assert np.array_equal(scipy_matrix.col, out_scipy_matrix.col)

    if dtype_str == 'f2':
        dense_array = \
            scipy_matrix.astype(np.float32).toarray().astype(np.float16)
    else:
        dense_array = scipy_matrix.toarray()
    assert np.array_equal(dense_array, sparse_tensor.to_tensor().to_numpy())

    # canonical sparse coo matrix
    scipy_matrix.sum_duplicates()
    sparse_tensor = pa.SparseCOOTensor.from_scipy(scipy_matrix,
                                                  dim_names=dim_names)
    out_scipy_matrix = sparse_tensor.to_scipy()

    assert scipy_matrix.has_canonical_format
    assert sparse_tensor.has_canonical_format
    assert out_scipy_matrix.has_canonical_format


@pytest.mark.skipif(not csr_matrix, reason="requires scipy")
@pytest.mark.parametrize('dtype_str,arrow_type', tensor_type_pairs)
def test_sparse_csr_matrix_scipy_roundtrip(dtype_str, arrow_type):
    dtype = np.dtype(dtype_str)
    data = np.array([8, 2, 5, 3, 4, 6]).astype(dtype)
    indptr = np.array([0, 2, 3, 4, 6])
    indices = np.array([0, 2, 5, 0, 4, 5])
    shape = (4, 6)
    dim_names = ('x', 'y')

    sparse_array = csr_matrix((data, indices, indptr), shape=shape)
    sparse_tensor = pa.SparseCSRMatrix.from_scipy(sparse_array,
                                                  dim_names=dim_names)
    out_sparse_array = sparse_tensor.to_scipy()

    assert sparse_tensor.type == arrow_type
    assert sparse_tensor.dim_names == dim_names
    assert sparse_array.dtype == out_sparse_array.dtype
    assert np.array_equal(sparse_array.data, out_sparse_array.data)
    assert np.array_equal(sparse_array.indptr, out_sparse_array.indptr)
    assert np.array_equal(sparse_array.indices, out_sparse_array.indices)

    if dtype_str == 'f2':
        dense_array = \
            sparse_array.astype(np.float32).toarray().astype(np.float16)
    else:
        dense_array = sparse_array.toarray()
    assert np.array_equal(dense_array, sparse_tensor.to_tensor().to_numpy())


@pytest.mark.skipif(not sparse, reason="requires pydata/sparse")
@pytest.mark.parametrize('dtype_str,arrow_type', tensor_type_pairs)
def test_pydata_sparse_sparse_coo_tensor_roundtrip(dtype_str, arrow_type):
    dtype = np.dtype(dtype_str)
    data = np.array([1, 2, 3, 4, 5, 6]).astype(dtype)
    coords = np.array([
        [0, 0, 2, 3, 1, 3],
        [0, 2, 0, 4, 5, 5],
    ])
    shape = (4, 6)
    dim_names = ("x", "y")

    sparse_array = sparse.COO(data=data, coords=coords, shape=shape)
    sparse_tensor = pa.SparseCOOTensor.from_pydata_sparse(sparse_array,
                                                          dim_names=dim_names)
    out_sparse_array = sparse_tensor.to_pydata_sparse()

    assert sparse_tensor.type == arrow_type
    assert sparse_tensor.dim_names == dim_names
    assert sparse_array.dtype == out_sparse_array.dtype
    assert np.array_equal(sparse_array.data, out_sparse_array.data)
    assert np.array_equal(sparse_array.coords, out_sparse_array.coords)
    assert np.array_equal(sparse_array.todense(),
                          sparse_tensor.to_tensor().to_numpy())

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

import collections
import warnings

import numpy as np

import pyarrow as pa
from pyarrow.lib import SerializationContext, py_buffer, builtin_pickle

try:
    import cloudpickle
except ImportError:
    cloudpickle = builtin_pickle


try:
    # This function is available after numpy-0.16.0.
    # See also: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    from numpy.lib.format import descr_to_dtype
except ImportError:
    def descr_to_dtype(descr):
        '''
        descr may be stored as dtype.descr, which is a list of (name, format,
        [shape]) tuples where format may be a str or a tuple.  Offsets are not
        explicitly saved, rather empty fields with name, format == '', '|Vn'
        are added as padding.  This function reverses the process, eliminating
        the empty padding fields.
        '''
        if isinstance(descr, str):
            # No padding removal needed
            return np.dtype(descr)
        elif isinstance(descr, tuple):
            # subtype, will always have a shape descr[1]
            dt = descr_to_dtype(descr[0])
            return np.dtype((dt, descr[1]))
        fields = []
        offset = 0
        for field in descr:
            if len(field) == 2:
                name, descr_str = field
                dt = descr_to_dtype(descr_str)
            else:
                name, descr_str, shape = field
                dt = np.dtype((descr_to_dtype(descr_str), shape))

            # Ignore padding bytes, which will be void bytes with '' as name
            # Once support for blank names is removed, only "if name == ''"
            # needed)
            is_pad = (name == '' and dt.type is np.void and dt.names is None)
            if not is_pad:
                fields.append((name, dt, offset))

            offset += dt.itemsize

        names, formats, offsets = zip(*fields)
        # names may be (title, names) tuples
        nametups = (n if isinstance(n, tuple) else (None, n) for n in names)
        titles, names = zip(*nametups)
        return np.dtype({'names': names, 'formats': formats, 'titles': titles,
                         'offsets': offsets, 'itemsize': offset})


def _deprecate_serialization(name):
    msg = (
        "'pyarrow.{}' is deprecated as of 2.0.0 and will be removed in a "
        "future version. Use pickle or the pyarrow IPC functionality instead."
    ).format(name)
    warnings.warn(msg, FutureWarning, stacklevel=3)


# ----------------------------------------------------------------------
# Set up serialization for numpy with dtype object (primitive types are
# handled efficiently with Arrow's Tensor facilities, see
# python_to_arrow.cc)

def _serialize_numpy_array_list(obj):
    if obj.dtype.str != '|O':
        # Make the array c_contiguous if necessary so that we can call change
        # the view.
        if not obj.flags.c_contiguous:
            obj = np.ascontiguousarray(obj)
        return obj.view('uint8'), np.lib.format.dtype_to_descr(obj.dtype)
    else:
        return obj.tolist(), np.lib.format.dtype_to_descr(obj.dtype)


def _deserialize_numpy_array_list(data):
    if data[1] != '|O':
        assert data[0].dtype == np.uint8
        return data[0].view(descr_to_dtype(data[1]))
    else:
        return np.array(data[0], dtype=np.dtype(data[1]))


def _serialize_numpy_matrix(obj):
    if obj.dtype.str != '|O':
        # Make the array c_contiguous if necessary so that we can call change
        # the view.
        if not obj.flags.c_contiguous:
            obj = np.ascontiguousarray(obj.A)
        return obj.A.view('uint8'), np.lib.format.dtype_to_descr(obj.dtype)
    else:
        return obj.A.tolist(), np.lib.format.dtype_to_descr(obj.dtype)


def _deserialize_numpy_matrix(data):
    if data[1] != '|O':
        assert data[0].dtype == np.uint8
        return np.matrix(data[0].view(descr_to_dtype(data[1])),
                         copy=False)
    else:
        return np.matrix(data[0], dtype=np.dtype(data[1]), copy=False)


# ----------------------------------------------------------------------
# pyarrow.RecordBatch-specific serialization matters

def _serialize_pyarrow_recordbatch(batch):
    output_stream = pa.BufferOutputStream()
    with pa.RecordBatchStreamWriter(output_stream, schema=batch.schema) as wr:
        wr.write_batch(batch)
    return output_stream.getvalue()  # This will also close the stream.


def _deserialize_pyarrow_recordbatch(buf):
    with pa.RecordBatchStreamReader(buf) as reader:
        return reader.read_next_batch()


# ----------------------------------------------------------------------
# pyarrow.Array-specific serialization matters

def _serialize_pyarrow_array(array):
    # TODO(suquark): implement more effcient array serialization.
    batch = pa.RecordBatch.from_arrays([array], [''])
    return _serialize_pyarrow_recordbatch(batch)


def _deserialize_pyarrow_array(buf):
    # TODO(suquark): implement more effcient array deserialization.
    batch = _deserialize_pyarrow_recordbatch(buf)
    return batch.columns[0]


# ----------------------------------------------------------------------
# pyarrow.Table-specific serialization matters

def _serialize_pyarrow_table(table):
    output_stream = pa.BufferOutputStream()
    with pa.RecordBatchStreamWriter(output_stream, schema=table.schema) as wr:
        wr.write_table(table)
    return output_stream.getvalue()  # This will also close the stream.


def _deserialize_pyarrow_table(buf):
    with pa.RecordBatchStreamReader(buf) as reader:
        return reader.read_all()


def _pickle_to_buffer(x):
    pickled = builtin_pickle.dumps(x, protocol=builtin_pickle.HIGHEST_PROTOCOL)
    return py_buffer(pickled)


def _load_pickle_from_buffer(data):
    as_memoryview = memoryview(data)
    return builtin_pickle.loads(as_memoryview)


# ----------------------------------------------------------------------
# pandas-specific serialization matters

def _register_custom_pandas_handlers(context):
    # ARROW-1784, faster path for pandas-only visibility

    try:
        import pandas as pd
    except ImportError:
        return

    import pyarrow.pandas_compat as pdcompat

    sparse_type_error_msg = (
        '{0} serialization is not supported.\n'
        'Note that {0} is planned to be deprecated '
        'in pandas future releases.\n'
        'See https://github.com/pandas-dev/pandas/issues/19239 '
        'for more information.'
    )

    def _serialize_pandas_dataframe(obj):
        if (pdcompat._pandas_api.has_sparse and
                isinstance(obj, pd.SparseDataFrame)):
            raise NotImplementedError(
                sparse_type_error_msg.format('SparseDataFrame')
            )

        return pdcompat.dataframe_to_serialized_dict(obj)

    def _deserialize_pandas_dataframe(data):
        return pdcompat.serialized_dict_to_dataframe(data)

    def _serialize_pandas_series(obj):
        if (pdcompat._pandas_api.has_sparse and
                isinstance(obj, pd.SparseSeries)):
            raise NotImplementedError(
                sparse_type_error_msg.format('SparseSeries')
            )

        return _serialize_pandas_dataframe(pd.DataFrame({obj.name: obj}))

    def _deserialize_pandas_series(data):
        deserialized = _deserialize_pandas_dataframe(data)
        return deserialized[deserialized.columns[0]]

    context.register_type(
        pd.Series, 'pd.Series',
        custom_serializer=_serialize_pandas_series,
        custom_deserializer=_deserialize_pandas_series)

    context.register_type(
        pd.Index, 'pd.Index',
        custom_serializer=_pickle_to_buffer,
        custom_deserializer=_load_pickle_from_buffer)

    if hasattr(pd.core, 'arrays'):
        if hasattr(pd.core.arrays, 'interval'):
            context.register_type(
                pd.core.arrays.interval.IntervalArray,
                'pd.core.arrays.interval.IntervalArray',
                custom_serializer=_pickle_to_buffer,
                custom_deserializer=_load_pickle_from_buffer)

        if hasattr(pd.core.arrays, 'period'):
            context.register_type(
                pd.core.arrays.period.PeriodArray,
                'pd.core.arrays.period.PeriodArray',
                custom_serializer=_pickle_to_buffer,
                custom_deserializer=_load_pickle_from_buffer)

        if hasattr(pd.core.arrays, 'datetimes'):
            context.register_type(
                pd.core.arrays.datetimes.DatetimeArray,
                'pd.core.arrays.datetimes.DatetimeArray',
                custom_serializer=_pickle_to_buffer,
                custom_deserializer=_load_pickle_from_buffer)

    context.register_type(
        pd.DataFrame, 'pd.DataFrame',
        custom_serializer=_serialize_pandas_dataframe,
        custom_deserializer=_deserialize_pandas_dataframe)


def register_torch_serialization_handlers(serialization_context):
    # ----------------------------------------------------------------------
    # Set up serialization for pytorch tensors
    _deprecate_serialization("register_torch_serialization_handlers")

    try:
        import torch

        def _serialize_torch_tensor(obj):
            if obj.is_sparse:
                return pa.SparseCOOTensor.from_numpy(
                    obj._values().detach().numpy(),
                    obj._indices().detach().numpy().T,
                    shape=list(obj.shape))
            else:
                return obj.detach().numpy()

        def _deserialize_torch_tensor(data):
            if isinstance(data, pa.SparseCOOTensor):
                return torch.sparse_coo_tensor(
                    indices=data.to_numpy()[1].T,
                    values=data.to_numpy()[0][:, 0],
                    size=data.shape)
            else:
                return torch.from_numpy(data)

        for t in [torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor,
                  torch.ByteTensor, torch.CharTensor, torch.ShortTensor,
                  torch.IntTensor, torch.LongTensor, torch.Tensor]:
            serialization_context.register_type(
                t, "torch." + t.__name__,
                custom_serializer=_serialize_torch_tensor,
                custom_deserializer=_deserialize_torch_tensor)
    except ImportError:
        # no torch
        pass


def _register_collections_serialization_handlers(serialization_context):
    def _serialize_deque(obj):
        return list(obj)

    def _deserialize_deque(data):
        return collections.deque(data)

    serialization_context.register_type(
        collections.deque, "collections.deque",
        custom_serializer=_serialize_deque,
        custom_deserializer=_deserialize_deque)

    def _serialize_ordered_dict(obj):
        return list(obj.keys()), list(obj.values())

    def _deserialize_ordered_dict(data):
        return collections.OrderedDict(zip(data[0], data[1]))

    serialization_context.register_type(
        collections.OrderedDict, "collections.OrderedDict",
        custom_serializer=_serialize_ordered_dict,
        custom_deserializer=_deserialize_ordered_dict)

    def _serialize_default_dict(obj):
        return list(obj.keys()), list(obj.values()), obj.default_factory

    def _deserialize_default_dict(data):
        return collections.defaultdict(data[2], zip(data[0], data[1]))

    serialization_context.register_type(
        collections.defaultdict, "collections.defaultdict",
        custom_serializer=_serialize_default_dict,
        custom_deserializer=_deserialize_default_dict)

    def _serialize_counter(obj):
        return list(obj.keys()), list(obj.values())

    def _deserialize_counter(data):
        return collections.Counter(dict(zip(data[0], data[1])))

    serialization_context.register_type(
        collections.Counter, "collections.Counter",
        custom_serializer=_serialize_counter,
        custom_deserializer=_deserialize_counter)


# ----------------------------------------------------------------------
# Set up serialization for scipy sparse matrices. Primitive types are handled
# efficiently with Arrow's SparseTensor facilities, see numpy_convert.cc)

def _register_scipy_handlers(serialization_context):
    try:
        from scipy.sparse import (csr_matrix, csc_matrix, coo_matrix,
                                  isspmatrix_coo, isspmatrix_csr,
                                  isspmatrix_csc, isspmatrix)

        def _serialize_scipy_sparse(obj):
            if isspmatrix_coo(obj):
                return 'coo', pa.SparseCOOTensor.from_scipy(obj)

            elif isspmatrix_csr(obj):
                return 'csr', pa.SparseCSRMatrix.from_scipy(obj)

            elif isspmatrix_csc(obj):
                return 'csc', pa.SparseCSCMatrix.from_scipy(obj)

            elif isspmatrix(obj):
                return 'csr', pa.SparseCOOTensor.from_scipy(obj.to_coo())

            else:
                raise NotImplementedError(
                    "Serialization of {} is not supported.".format(obj[0]))

        def _deserialize_scipy_sparse(data):
            if data[0] == 'coo':
                return data[1].to_scipy()

            elif data[0] == 'csr':
                return data[1].to_scipy()

            elif data[0] == 'csc':
                return data[1].to_scipy()

            else:
                return data[1].to_scipy()

        serialization_context.register_type(
            coo_matrix, 'scipy.sparse.coo.coo_matrix',
            custom_serializer=_serialize_scipy_sparse,
            custom_deserializer=_deserialize_scipy_sparse)

        serialization_context.register_type(
            csr_matrix, 'scipy.sparse.csr.csr_matrix',
            custom_serializer=_serialize_scipy_sparse,
            custom_deserializer=_deserialize_scipy_sparse)

        serialization_context.register_type(
            csc_matrix, 'scipy.sparse.csc.csc_matrix',
            custom_serializer=_serialize_scipy_sparse,
            custom_deserializer=_deserialize_scipy_sparse)

    except ImportError:
        # no scipy
        pass


# ----------------------------------------------------------------------
# Set up serialization for pydata/sparse tensors.

def _register_pydata_sparse_handlers(serialization_context):
    try:
        import sparse

        def _serialize_pydata_sparse(obj):
            if isinstance(obj, sparse.COO):
                return 'coo', pa.SparseCOOTensor.from_pydata_sparse(obj)
            else:
                raise NotImplementedError(
                    "Serialization of {} is not supported.".format(sparse.COO))

        def _deserialize_pydata_sparse(data):
            if data[0] == 'coo':
                data_array, coords = data[1].to_numpy()
                return sparse.COO(
                    data=data_array[:, 0],
                    coords=coords.T, shape=data[1].shape)

        serialization_context.register_type(
            sparse.COO, 'sparse.COO',
            custom_serializer=_serialize_pydata_sparse,
            custom_deserializer=_deserialize_pydata_sparse)

    except ImportError:
        # no pydata/sparse
        pass


def _register_default_serialization_handlers(serialization_context):

    # ----------------------------------------------------------------------
    # Set up serialization for primitive datatypes

    # TODO(pcm): This is currently a workaround until arrow supports
    # arbitrary precision integers. This is only called on long integers,
    # see the associated case in the append method in python_to_arrow.cc
    serialization_context.register_type(
        int, "int",
        custom_serializer=lambda obj: str(obj),
        custom_deserializer=lambda data: int(data))

    serialization_context.register_type(
        type(lambda: 0), "function",
        pickle=True)

    serialization_context.register_type(type, "type", pickle=True)

    serialization_context.register_type(
        np.matrix, 'np.matrix',
        custom_serializer=_serialize_numpy_matrix,
        custom_deserializer=_deserialize_numpy_matrix)

    serialization_context.register_type(
        np.ndarray, 'np.array',
        custom_serializer=_serialize_numpy_array_list,
        custom_deserializer=_deserialize_numpy_array_list)

    serialization_context.register_type(
        pa.Array, 'pyarrow.Array',
        custom_serializer=_serialize_pyarrow_array,
        custom_deserializer=_deserialize_pyarrow_array)

    serialization_context.register_type(
        pa.RecordBatch, 'pyarrow.RecordBatch',
        custom_serializer=_serialize_pyarrow_recordbatch,
        custom_deserializer=_deserialize_pyarrow_recordbatch)

    serialization_context.register_type(
        pa.Table, 'pyarrow.Table',
        custom_serializer=_serialize_pyarrow_table,
        custom_deserializer=_deserialize_pyarrow_table)

    _register_collections_serialization_handlers(serialization_context)
    _register_custom_pandas_handlers(serialization_context)
    _register_scipy_handlers(serialization_context)
    _register_pydata_sparse_handlers(serialization_context)


def register_default_serialization_handlers(serialization_context):
    _deprecate_serialization("register_default_serialization_handlers")
    _register_default_serialization_handlers(serialization_context)


def default_serialization_context():
    _deprecate_serialization("default_serialization_context")
    context = SerializationContext()
    _register_default_serialization_handlers(context)
    return context

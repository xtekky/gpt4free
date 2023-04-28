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

import ctypes
import hypothesis as h
import hypothesis.strategies as st

import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
import pytest


all_types = st.deferred(
    lambda: (
        past.signed_integer_types |
        past.unsigned_integer_types |
        past.floating_types |
        past.bool_type |
        past.string_type |
        past.large_string_type
    )
)


# datetime is tested in test_extra.py
# dictionary is tested in test_categorical()
@h.given(past.arrays(all_types, size=3))
def test_dtypes(arr):
    table = pa.table([arr], names=["a"])
    df = table.__dataframe__()

    null_count = df.get_column(0).null_count
    assert null_count == arr.null_count
    assert isinstance(null_count, int)
    assert df.get_column(0).size() == 3
    assert df.get_column(0).offset == 0


@pytest.mark.parametrize(
    "uint, uint_bw",
    [
        (pa.uint8(), 8),
        (pa.uint16(), 16),
        (pa.uint32(), 32)
    ]
)
@pytest.mark.parametrize(
    "int, int_bw", [
        (pa.int8(), 8),
        (pa.int16(), 16),
        (pa.int32(), 32),
        (pa.int64(), 64)
    ]
)
@pytest.mark.parametrize(
    "float, float_bw, np_float", [
        (pa.float16(), 16, np.float16),
        (pa.float32(), 32, np.float32),
        (pa.float64(), 64, np.float64)
    ]
)
@pytest.mark.parametrize("unit", ['s', 'ms', 'us', 'ns'])
@pytest.mark.parametrize("tz", ['', 'America/New_York', '+07:30', '-04:30'])
def test_mixed_dtypes(uint, uint_bw, int, int_bw,
                      float, float_bw, np_float, unit, tz):
    from datetime import datetime as dt
    arr = [1, 2, 3]
    dt_arr = [dt(2007, 7, 13), dt(2007, 7, 14), dt(2007, 7, 15)]
    table = pa.table(
        {
            "a": pa.array(arr, type=uint),
            "b": pa.array(arr, type=int),
            "c": pa.array(np.array(arr, dtype=np_float), type=float),
            "d": [True, False, True],
            "e": ["a", "", "c"],
            "f": pa.array(dt_arr, type=pa.timestamp(unit, tz=tz))
        }
    )
    df = table.__dataframe__()
    # 0 = DtypeKind.INT, 1 = DtypeKind.UINT, 2 = DtypeKind.FLOAT,
    # 20 = DtypeKind.BOOL, 21 = DtypeKind.STRING, 22 = DtypeKind.DATETIME
    # see DtypeKind class in column.py
    columns = {"a": 1, "b": 0, "c": 2, "d": 20, "e": 21, "f": 22}

    for column, kind in columns.items():
        col = df.get_column_by_name(column)

        assert col.null_count == 0
        assert col.size() == 3
        assert col.offset == 0
        assert col.dtype[0] == kind

    assert df.get_column_by_name("a").dtype[1] == uint_bw
    assert df.get_column_by_name("b").dtype[1] == int_bw
    assert df.get_column_by_name("c").dtype[1] == float_bw


def test_na_float():
    table = pa.table({"a": [1.0, None, 2.0]})
    df = table.__dataframe__()
    col = df.get_column_by_name("a")
    assert col.null_count == 1
    assert isinstance(col.null_count, int)


def test_noncategorical():
    table = pa.table({"a": [1, 2, 3]})
    df = table.__dataframe__()
    col = df.get_column_by_name("a")
    with pytest.raises(TypeError, match=".*categorical.*"):
        col.describe_categorical


def test_categorical():
    import pyarrow as pa
    arr = ["Mon", "Tue", "Mon", "Wed", "Mon", "Thu", "Fri", "Sat", None]
    table = pa.table(
        {"weekday": pa.array(arr).dictionary_encode()}
    )

    col = table.__dataframe__().get_column_by_name("weekday")
    categorical = col.describe_categorical
    assert isinstance(categorical["is_ordered"], bool)
    assert isinstance(categorical["is_dictionary"], bool)


def test_dataframe():
    n = pa.chunked_array([[2, 2, 4], [4, 5, 100]])
    a = pa.chunked_array([["Flamingo", "Parrot", "Cow"],
                         ["Horse", "Brittle stars", "Centipede"]])
    table = pa.table([n, a], names=['n_legs', 'animals'])
    df = table.__dataframe__()

    assert df.num_columns() == 2
    assert df.num_rows() == 6
    assert df.num_chunks() == 2
    assert list(df.column_names()) == ['n_legs', 'animals']
    assert list(df.select_columns((1,)).column_names()) == list(
        df.select_columns_by_name(("animals",)).column_names()
    )


@pytest.mark.parametrize(["size", "n_chunks"], [(10, 3), (12, 3), (12, 5)])
def test_df_get_chunks(size, n_chunks):
    table = pa.table({"x": list(range(size))})
    df = table.__dataframe__()
    chunks = list(df.get_chunks(n_chunks))
    assert len(chunks) == n_chunks
    assert sum(chunk.num_rows() for chunk in chunks) == size


@pytest.mark.parametrize(["size", "n_chunks"], [(10, 3), (12, 3), (12, 5)])
def test_column_get_chunks(size, n_chunks):
    table = pa.table({"x": list(range(size))})
    df = table.__dataframe__()
    chunks = list(df.get_column(0).get_chunks(n_chunks))
    assert len(chunks) == n_chunks
    assert sum(chunk.size() for chunk in chunks) == size


@pytest.mark.pandas
@pytest.mark.parametrize(
    "uint", [pa.uint8(), pa.uint16(), pa.uint32()]
)
@pytest.mark.parametrize(
    "int", [pa.int8(), pa.int16(), pa.int32(), pa.int64()]
)
@pytest.mark.parametrize(
    "float, np_float", [
        (pa.float16(), np.float16),
        (pa.float32(), np.float32),
        (pa.float64(), np.float64)
    ]
)
def test_get_columns(uint, int, float, np_float):
    arr = [[1, 2, 3], [4, 5]]
    arr_float = np.array([1, 2, 3, 4, 5], dtype=np_float)
    table = pa.table(
        {
            "a": pa.chunked_array(arr, type=uint),
            "b": pa.chunked_array(arr, type=int),
            "c": pa.array(arr_float, type=float)
        }
    )
    df = table.__dataframe__()
    for col in df.get_columns():
        assert col.size() == 5
        assert col.num_chunks() == 1

    # 0 = DtypeKind.INT, 1 = DtypeKind.UINT, 2 = DtypeKind.FLOAT,
    # see DtypeKind class in column.py
    assert df.get_column(0).dtype[0] == 1  # UINT
    assert df.get_column(1).dtype[0] == 0  # INT
    assert df.get_column(2).dtype[0] == 2  # FLOAT


@pytest.mark.parametrize(
    "int", [pa.int8(), pa.int16(), pa.int32(), pa.int64()]
)
def test_buffer(int):
    arr = [0, 1, -1]
    table = pa.table({"a": pa.array(arr, type=int)})
    df = table.__dataframe__()
    col = df.get_column(0)
    buf = col.get_buffers()

    dataBuf, dataDtype = buf["data"]

    assert dataBuf.bufsize > 0
    assert dataBuf.ptr != 0
    device, _ = dataBuf.__dlpack_device__()

    # 0 = DtypeKind.INT
    # see DtypeKind class in column.py
    assert dataDtype[0] == 0

    if device == 1:  # CPU-only as we're going to directly read memory here
        bitwidth = dataDtype[1]
        ctype = {
            8: ctypes.c_int8,
            16: ctypes.c_int16,
            32: ctypes.c_int32,
            64: ctypes.c_int64,
        }[bitwidth]

        for idx, truth in enumerate(arr):
            val = ctype.from_address(dataBuf.ptr + idx * (bitwidth // 8)).value
            assert val == truth, f"Buffer at index {idx} mismatch"

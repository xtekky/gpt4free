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

from datetime import datetime as dt
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
import pytest

import pyarrow.interchange as pi
from pyarrow.interchange.column import (
    _PyArrowColumn,
    ColumnNullType,
    DtypeKind,
)
from pyarrow.interchange.from_dataframe import _from_dataframe

try:
    import pandas as pd
    # import pandas.testing as tm
except ImportError:
    pass


@pytest.mark.parametrize("unit", ['s', 'ms', 'us', 'ns'])
@pytest.mark.parametrize("tz", ['', 'America/New_York', '+07:30', '-04:30'])
def test_datetime(unit, tz):
    dt_arr = [dt(2007, 7, 13), dt(2007, 7, 14), None]
    table = pa.table({"A": pa.array(dt_arr, type=pa.timestamp(unit, tz=tz))})
    col = table.__dataframe__().get_column_by_name("A")

    assert col.size() == 3
    assert col.offset == 0
    assert col.null_count == 1
    assert col.dtype[0] == DtypeKind.DATETIME
    assert col.describe_null == (ColumnNullType.USE_BITMASK, 0)


@pytest.mark.parametrize(
    ["test_data", "kind"],
    [
        (["foo", "bar"], 21),
        ([1.5, 2.5, 3.5], 2),
        ([1, 2, 3, 4], 0),
    ],
)
def test_array_to_pyarrowcolumn(test_data, kind):
    arr = pa.array(test_data)
    arr_column = _PyArrowColumn(arr)

    assert arr_column._col == arr
    assert arr_column.size() == len(test_data)
    assert arr_column.dtype[0] == kind
    assert arr_column.num_chunks() == 1
    assert arr_column.null_count == 0
    assert arr_column.get_buffers()["validity"] is None
    assert len(list(arr_column.get_chunks())) == 1

    for chunk in arr_column.get_chunks():
        assert chunk == arr_column


def test_offset_of_sliced_array():
    arr = pa.array([1, 2, 3, 4])
    arr_sliced = arr.slice(2, 2)

    table = pa.table([arr], names=["arr"])
    table_sliced = pa.table([arr_sliced], names=["arr_sliced"])

    col = table_sliced.__dataframe__().get_column(0)
    assert col.offset == 2

    result = _from_dataframe(table_sliced.__dataframe__())
    assert table_sliced.equals(result)
    assert not table.equals(result)

    # pandas hardcodes offset to 0:
    # https://github.com/pandas-dev/pandas/blob/5c66e65d7b9fef47ccb585ce2fd0b3ea18dc82ea/pandas/core/interchange/from_dataframe.py#L247
    # so conversion to pandas can't be tested currently

    # df = pandas_from_dataframe(table)
    # df_sliced = pandas_from_dataframe(table_sliced)

    # tm.assert_series_equal(df["arr"][2:4], df_sliced["arr_sliced"],
    #                        check_index=False, check_names=False)


# Currently errors due to string conversion
# as col.size is called as a property not method in pandas
# see L255-L257 in pandas/core/interchange/from_dataframe.py
@pytest.mark.pandas
def test_categorical_roundtrip():
    pytest.skip("Bug in pandas implementation")

    if Version(pd.__version__) < Version("1.5.0"):
        pytest.skip("__dataframe__ added to pandas in 1.5.0")
    arr = ["Mon", "Tue", "Mon", "Wed", "Mon", "Thu", "Fri", "Sat", "Sun"]
    table = pa.table(
        {"weekday": pa.array(arr).dictionary_encode()}
    )

    pandas_df = table.to_pandas()
    result = pi.from_dataframe(pandas_df)

    # Checking equality for the values
    # As the dtype of the indices is changed from int32 in pa.Table
    # to int64 in pandas interchange protocol implementation
    assert result[0].chunk(0).dictionary == table[0].chunk(0).dictionary

    table_protocol = table.__dataframe__()
    result_protocol = result.__dataframe__()

    assert table_protocol.num_columns() == result_protocol.num_columns()
    assert table_protocol.num_rows() == result_protocol.num_rows()
    assert table_protocol.num_chunks() == result_protocol.num_chunks()
    assert table_protocol.column_names() == result_protocol.column_names()

    col_table = table_protocol.get_column(0)
    col_result = result_protocol.get_column(0)

    assert col_result.dtype[0] == DtypeKind.CATEGORICAL
    assert col_result.dtype[0] == col_table.dtype[0]
    assert col_result.size == col_table.size
    assert col_result.offset == col_table.offset

    desc_cat_table = col_result.describe_categorical
    desc_cat_result = col_result.describe_categorical

    assert desc_cat_table["is_ordered"] == desc_cat_result["is_ordered"]
    assert desc_cat_table["is_dictionary"] == desc_cat_result["is_dictionary"]
    assert isinstance(desc_cat_result["categories"]._col, pa.Array)


@pytest.mark.pandas
@pytest.mark.parametrize(
    "uint", [pa.uint8(), pa.uint16(), pa.uint32()]
)
@pytest.mark.parametrize(
    "int", [pa.int8(), pa.int16(), pa.int32(), pa.int64()]
)
@pytest.mark.parametrize(
    "float, np_float", [
        # (pa.float16(), np.float16),   #not supported by pandas
        (pa.float32(), np.float32),
        (pa.float64(), np.float64)
    ]
)
def test_pandas_roundtrip(uint, int, float, np_float):
    if Version(pd.__version__) < Version("1.5.0"):
        pytest.skip("__dataframe__ added to pandas in 1.5.0")

    arr = [1, 2, 3]
    table = pa.table(
        {
            "a": pa.array(arr, type=uint),
            "b": pa.array(arr, type=int),
            "c": pa.array(np.array(arr, dtype=np_float), type=float),
        }
    )
    from pandas.api.interchange import (
        from_dataframe as pandas_from_dataframe
    )
    pandas_df = pandas_from_dataframe(table)
    result = pi.from_dataframe(pandas_df)
    assert table.equals(result)

    table_protocol = table.__dataframe__()
    result_protocol = result.__dataframe__()

    assert table_protocol.num_columns() == result_protocol.num_columns()
    assert table_protocol.num_rows() == result_protocol.num_rows()
    assert table_protocol.num_chunks() == result_protocol.num_chunks()
    assert table_protocol.column_names() == result_protocol.column_names()


@pytest.mark.pandas
def test_roundtrip_pandas_string():
    # See https://github.com/pandas-dev/pandas/issues/50554
    if Version(pd.__version__) < Version("1.6"):
        pytest.skip(" Column.size() called as a method in pandas 2.0.0")

    # large string is not supported by pandas implementation
    table = pa.table({"a": pa.array(["a", "", "c"])})

    from pandas.api.interchange import (
        from_dataframe as pandas_from_dataframe
    )
    pandas_df = pandas_from_dataframe(table)
    result = pi.from_dataframe(pandas_df)

    assert result[0].to_pylist() == table[0].to_pylist()
    assert pa.types.is_string(table[0].type)
    assert pa.types.is_large_string(result[0].type)

    table_protocol = table.__dataframe__()
    result_protocol = result.__dataframe__()

    assert table_protocol.num_columns() == result_protocol.num_columns()
    assert table_protocol.num_rows() == result_protocol.num_rows()
    assert table_protocol.num_chunks() == result_protocol.num_chunks()
    assert table_protocol.column_names() == result_protocol.column_names()


@pytest.mark.pandas
def test_roundtrip_pandas_boolean():
    if Version(pd.__version__) < Version("1.5.0"):
        pytest.skip("__dataframe__ added to pandas in 1.5.0")

    table = pa.table({"a": [True, False, True]})

    from pandas.api.interchange import (
        from_dataframe as pandas_from_dataframe
    )
    pandas_df = pandas_from_dataframe(table)
    result = pi.from_dataframe(pandas_df)

    assert table.equals(result)

    table_protocol = table.__dataframe__()
    result_protocol = result.__dataframe__()

    assert table_protocol.num_columns() == result_protocol.num_columns()
    assert table_protocol.num_rows() == result_protocol.num_rows()
    assert table_protocol.num_chunks() == result_protocol.num_chunks()
    assert table_protocol.column_names() == result_protocol.column_names()


@pytest.mark.pandas
@pytest.mark.parametrize("unit", ['s', 'ms', 'us', 'ns'])
def test_roundtrip_pandas_datetime(unit):
    if Version(pd.__version__) < Version("1.5.0"):
        pytest.skip("__dataframe__ added to pandas in 1.5.0")
    from datetime import datetime as dt

    # timezones not included as they are not yet supported in
    # the pandas implementation
    dt_arr = [dt(2007, 7, 13), dt(2007, 7, 14), dt(2007, 7, 15)]
    table = pa.table({"a": pa.array(dt_arr, type=pa.timestamp(unit))})

    if Version(pd.__version__) < Version("1.6"):
        # pandas < 2.0 always creates datetime64 in "ns"
        # resolution
        expected = pa.table({"a": pa.array(dt_arr, type=pa.timestamp('ns'))})
    else:
        expected = table

    from pandas.api.interchange import (
        from_dataframe as pandas_from_dataframe
    )
    pandas_df = pandas_from_dataframe(table)
    result = pi.from_dataframe(pandas_df)

    assert expected.equals(result)

    expected_protocol = expected.__dataframe__()
    result_protocol = result.__dataframe__()

    assert expected_protocol.num_columns() == result_protocol.num_columns()
    assert expected_protocol.num_rows() == result_protocol.num_rows()
    assert expected_protocol.num_chunks() == result_protocol.num_chunks()
    assert expected_protocol.column_names() == result_protocol.column_names()


@pytest.mark.large_memory
@pytest.mark.pandas
def test_pandas_assertion_error_large_string():
    # Test AssertionError as pandas does not support "U" type strings
    if Version(pd.__version__) < Version("1.5.0"):
        pytest.skip("__dataframe__ added to pandas in 1.5.0")

    data = np.array([b'x'*1024]*(3*1024**2), dtype='object')  # 3GB bytes data
    arr = pa.array(data, type=pa.large_string())
    table = pa.table([arr], names=["large_string"])

    from pandas.api.interchange import (
        from_dataframe as pandas_from_dataframe
    )

    with pytest.raises(AssertionError):
        pandas_from_dataframe(table)


@pytest.mark.pandas
@pytest.mark.parametrize(
    "np_float", [np.float32, np.float64]
)
def test_pandas_to_pyarrow_with_missing(np_float):
    if Version(pd.__version__) < Version("1.5.0"):
        pytest.skip("__dataframe__ added to pandas in 1.5.0")

    np_array = np.array([0, np.nan, 2], dtype=np_float)
    datetime_array = [None, dt(2007, 7, 14), dt(2007, 7, 15)]
    df = pd.DataFrame({
        "a": np_array,   # float, ColumnNullType.USE_NAN
        "dt": datetime_array  # ColumnNullType.USE_SENTINEL
    })
    expected = pa.table({
        "a": pa.array(np_array, from_pandas=True),
        "dt": pa.array(datetime_array, type=pa.timestamp("ns"))
    })
    result = pi.from_dataframe(df)

    assert result.equals(expected)


@pytest.mark.pandas
def test_pandas_to_pyarrow_float16_with_missing():
    if Version(pd.__version__) < Version("1.5.0"):
        pytest.skip("__dataframe__ added to pandas in 1.5.0")

    # np.float16 errors if ps.is_nan is used
    # pyarrow.lib.ArrowNotImplementedError: Function 'is_nan' has no kernel
    # matching input types (halffloat)
    np_array = np.array([0, np.nan, 2], dtype=np.float16)
    df = pd.DataFrame({"a": np_array})

    with pytest.raises(NotImplementedError):
        pi.from_dataframe(df)


@pytest.mark.pandas
def test_pandas_to_pyarrow_string_with_missing():
    if Version(pd.__version__) < Version("1.5.0"):
        pytest.skip("__dataframe__ added to pandas in 1.5.0")

    # pandas is using int64 offsets for string dtype so the constructed
    # pyarrow string column will always be a large_string data type
    arr = {
        "Y": ["a", "b", None],  # bool, ColumnNullType.USE_BYTEMASK,
    }
    df = pd.DataFrame(arr)
    expected = pa.table(arr)
    result = pi.from_dataframe(df)

    assert result[0].to_pylist() == expected[0].to_pylist()
    assert pa.types.is_string(expected[0].type)
    assert pa.types.is_large_string(result[0].type)


@pytest.mark.pandas
def test_pandas_to_pyarrow_categorical_with_missing():
    if Version(pd.__version__) < Version("1.5.0"):
        pytest.skip("__dataframe__ added to pandas in 1.5.0")

    arr = ["Mon", "Tue", "Mon", "Wed", "Mon", "Thu", "Fri", "Sat", None]
    df = pd.DataFrame(
        {"weekday": arr}
    )
    df = df.astype("category")
    result = pi.from_dataframe(df)

    expected_dictionary = ["Fri", "Mon", "Sat", "Thu", "Tue", "Wed"]
    expected_indices = pa.array([1, 4, 1, 5, 1, 3, 0, 2, None], type=pa.int8())

    assert result[0].to_pylist() == arr
    assert result[0].chunk(0).dictionary.to_pylist() == expected_dictionary
    assert result[0].chunk(0).indices.equals(expected_indices)


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
@pytest.mark.parametrize("unit", ['s', 'ms', 'us', 'ns'])
@pytest.mark.parametrize("tz", ['America/New_York', '+07:30', '-04:30'])
@pytest.mark.parametrize("offset, length", [(0, 3), (0, 2), (1, 2), (2, 1)])
def test_pyarrow_roundtrip(uint, int, float, np_float,
                           unit, tz, offset, length):

    from datetime import datetime as dt
    arr = [1, 2, None]
    dt_arr = [dt(2007, 7, 13), None, dt(2007, 7, 15)]

    table = pa.table(
        {
            "a": pa.array(arr, type=uint),
            "b": pa.array(arr, type=int),
            "c": pa.array(np.array(arr, dtype=np_float),
                          type=float, from_pandas=True),
            "d": [True, False, True],
            "e": [True, False, None],
            "f": ["a", None, "c"],
            "g": pa.array(dt_arr, type=pa.timestamp(unit, tz=tz))
        }
    )
    table = table.slice(offset, length)
    result = _from_dataframe(table.__dataframe__())

    assert table.equals(result)

    table_protocol = table.__dataframe__()
    result_protocol = result.__dataframe__()

    assert table_protocol.num_columns() == result_protocol.num_columns()
    assert table_protocol.num_rows() == result_protocol.num_rows()
    assert table_protocol.num_chunks() == result_protocol.num_chunks()
    assert table_protocol.column_names() == result_protocol.column_names()


@pytest.mark.parametrize("offset, length", [(0, 10), (0, 2), (7, 3), (2, 1)])
def test_pyarrow_roundtrip_categorical(offset, length):
    arr = ["Mon", "Tue", "Mon", "Wed", "Mon", "Thu", "Fri", None, "Sun"]
    table = pa.table(
        {"weekday": pa.array(arr).dictionary_encode()}
    )
    table = table.slice(offset, length)
    result = _from_dataframe(table.__dataframe__())

    assert table.equals(result)

    table_protocol = table.__dataframe__()
    result_protocol = result.__dataframe__()

    assert table_protocol.num_columns() == result_protocol.num_columns()
    assert table_protocol.num_rows() == result_protocol.num_rows()
    assert table_protocol.num_chunks() == result_protocol.num_chunks()
    assert table_protocol.column_names() == result_protocol.column_names()

    col_table = table_protocol.get_column(0)
    col_result = result_protocol.get_column(0)

    assert col_result.dtype[0] == DtypeKind.CATEGORICAL
    assert col_result.dtype[0] == col_table.dtype[0]
    assert col_result.size() == col_table.size()
    assert col_result.offset == col_table.offset

    desc_cat_table = col_result.describe_categorical
    desc_cat_result = col_result.describe_categorical

    assert desc_cat_table["is_ordered"] == desc_cat_result["is_ordered"]
    assert desc_cat_table["is_dictionary"] == desc_cat_result["is_dictionary"]
    assert isinstance(desc_cat_result["categories"]._col, pa.Array)


@pytest.mark.large_memory
def test_pyarrow_roundtrip_large_string():

    data = np.array([b'x'*1024]*(3*1024**2), dtype='object')  # 3GB bytes data
    arr = pa.array(data, type=pa.large_string())
    table = pa.table([arr], names=["large_string"])

    result = _from_dataframe(table.__dataframe__())
    col = result.__dataframe__().get_column(0)

    assert col.size() == 3*1024**2
    assert pa.types.is_large_string(table[0].type)
    assert pa.types.is_large_string(result[0].type)

    assert table.equals(result)


def test_nan_as_null():
    table = pa.table({"a": [1, 2, 3, 4]})
    with pytest.raises(RuntimeError):
        table.__dataframe__(nan_as_null=True)


@pytest.mark.pandas
def test_allow_copy_false():
    if Version(pd.__version__) < Version("1.5.0"):
        pytest.skip("__dataframe__ added to pandas in 1.5.0")

    # Test that an error is raised when a copy is needed
    # to create a bitmask

    df = pd.DataFrame({"a": [0, 1.0, 2.0]})
    with pytest.raises(RuntimeError):
        pi.from_dataframe(df, allow_copy=False)

    df = pd.DataFrame({
        "dt": [None, dt(2007, 7, 14), dt(2007, 7, 15)]
    })
    with pytest.raises(RuntimeError):
        pi.from_dataframe(df, allow_copy=False)


@pytest.mark.pandas
def test_allow_copy_false_bool_categorical():
    if Version(pd.__version__) < Version("1.5.0"):
        pytest.skip("__dataframe__ added to pandas in 1.5.0")

    # Test that an error is raised for boolean
    # and categorical dtype (copy is always made)

    df = pd.DataFrame({"a": [None, False, True]})
    with pytest.raises(RuntimeError):
        pi.from_dataframe(df, allow_copy=False)

    df = pd.DataFrame({"a": [True, False, True]})
    with pytest.raises(RuntimeError):
        pi.from_dataframe(df, allow_copy=False)

    df = pd.DataFrame({"weekday": ["a", "b", None]})
    df = df.astype("category")
    with pytest.raises(RuntimeError):
        pi.from_dataframe(df, allow_copy=False)

    df = pd.DataFrame({"weekday": ["a", "b", "c"]})
    df = df.astype("category")
    with pytest.raises(RuntimeError):
        pi.from_dataframe(df, allow_copy=False)

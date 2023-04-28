import numpy as np

from pandas import (
    DataFrame,
    Index,
    RangeIndex,
    Series,
)
import pandas._testing as tm

# -----------------------------------------------------------------------------
# Copy/view behaviour for the values that are set in a DataFrame


def test_set_column_with_array():
    # Case: setting an array as a new column (df[col] = arr) copies that data
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    arr = np.array([1, 2, 3], dtype="int64")

    df["c"] = arr

    # the array data is copied
    assert not np.shares_memory(df["c"].values, arr)
    # and thus modifying the array does not modify the DataFrame
    arr[0] = 0
    tm.assert_series_equal(df["c"], Series([1, 2, 3], name="c"))


def test_set_column_with_series(using_copy_on_write):
    # Case: setting a series as a new column (df[col] = s) copies that data
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    ser = Series([1, 2, 3])

    df["c"] = ser

    if using_copy_on_write:
        # TODO(CoW) with CoW we can delay the copy
        # assert np.shares_memory(df["c"].values, ser.values)
        assert not np.shares_memory(df["c"].values, ser.values)
    else:
        # the series data is copied
        assert not np.shares_memory(df["c"].values, ser.values)

    # and modifying the series does not modify the DataFrame
    ser.iloc[0] = 0
    assert ser.iloc[0] == 0
    tm.assert_series_equal(df["c"], Series([1, 2, 3], name="c"))


def test_set_column_with_index(using_copy_on_write):
    # Case: setting an index as a new column (df[col] = idx) copies that data
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    idx = Index([1, 2, 3])

    df["c"] = idx

    # the index data is copied
    assert not np.shares_memory(df["c"].values, idx.values)

    # and thus modifying the index does not modify the DataFrame
    idx.values[0] = 0
    tm.assert_series_equal(df["c"], Series([1, 2, 3], name="c"))

    idx = RangeIndex(1, 4)
    arr = idx.values

    df["d"] = idx

    assert not np.shares_memory(df["d"].values, arr)
    arr[0] = 0
    tm.assert_series_equal(df["d"], Series([1, 2, 3], name="d"))


def test_set_columns_with_dataframe(using_copy_on_write):
    # Case: setting a DataFrame as new columns copies that data
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = DataFrame({"c": [7, 8, 9], "d": [10, 11, 12]})

    df[["c", "d"]] = df2

    if using_copy_on_write:
        # TODO(CoW) with CoW we can delay the copy
        # assert np.shares_memory(df["c"].values, df2["c"].values)
        assert not np.shares_memory(df["c"].values, df2["c"].values)
    else:
        # the data is copied
        assert not np.shares_memory(df["c"].values, df2["c"].values)

    # and modifying the set DataFrame does not modify the original DataFrame
    df2.iloc[0, 0] = 0
    tm.assert_series_equal(df["c"], Series([7, 8, 9], name="c"))

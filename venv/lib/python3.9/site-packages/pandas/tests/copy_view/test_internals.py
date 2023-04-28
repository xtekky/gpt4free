import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
from pandas import DataFrame
from pandas.tests.copy_view.util import get_array


@td.skip_array_manager_invalid_test
def test_consolidate(using_copy_on_write):

    # create unconsolidated DataFrame
    df = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    df["c"] = [4, 5, 6]

    # take a viewing subset
    subset = df[:]

    # each block of subset references a block of df
    assert subset._mgr.refs is not None and all(
        ref is not None for ref in subset._mgr.refs
    )

    # consolidate the two int64 blocks
    subset._consolidate_inplace()

    # the float64 block still references the parent one because it still a view
    assert subset._mgr.refs[0] is not None
    # equivalent of assert np.shares_memory(df["b"].values, subset["b"].values)
    # but avoids caching df["b"]
    assert np.shares_memory(get_array(df, "b"), get_array(subset, "b"))

    # the new consolidated int64 block does not reference another
    assert subset._mgr.refs[1] is None

    # the parent dataframe now also only is linked for the float column
    assert df._mgr._has_no_reference(0)
    assert not df._mgr._has_no_reference(1)
    assert df._mgr._has_no_reference(2)

    # and modifying subset still doesn't modify parent
    if using_copy_on_write:
        subset.iloc[0, 1] = 0.0
        assert df._mgr._has_no_reference(1)
        assert df.loc[0, "b"] == 0.1


@td.skip_array_manager_invalid_test
def test_clear_parent(using_copy_on_write):
    # ensure to clear parent reference if we are no longer viewing data from parent
    if not using_copy_on_write:
        pytest.skip("test only relevant when using copy-on-write")

    df = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    subset = df[:]
    assert subset._mgr.parent is not None

    # replacing existing columns loses the references to the parent df
    subset["a"] = 0
    assert subset._mgr.parent is not None
    # when losing the last reference, also the parent should be reset
    subset["b"] = 0
    assert subset._mgr.parent is None


@pytest.mark.single_cpu
@td.skip_array_manager_invalid_test
def test_switch_options():
    # ensure we can switch the value of the option within one session
    # (assuming data is constructed after switching)

    # using the option_context to ensure we set back to global option value
    # after running the test
    with pd.option_context("mode.copy_on_write", False):
        df = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
        subset = df[:]
        subset.iloc[0, 0] = 0
        # df updated with CoW disabled
        assert df.iloc[0, 0] == 0

        pd.options.mode.copy_on_write = True
        df = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
        subset = df[:]
        subset.iloc[0, 0] = 0
        # df not updated with CoW enabled
        assert df.iloc[0, 0] == 1

        pd.options.mode.copy_on_write = False
        df = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
        subset = df[:]
        subset.iloc[0, 0] = 0
        # df updated with CoW disabled
        assert df.iloc[0, 0] == 0

"""
Tests for behavior if an author does *not* implement EA methods.
"""
import numpy as np
import pytest

import pandas._testing as tm
from pandas.core.arrays import ExtensionArray


class MyEA(ExtensionArray):
    def __init__(self, values) -> None:
        self._values = values


@pytest.fixture
def data():
    arr = np.arange(10)
    return MyEA(arr)


class TestExtensionArray:
    def test_errors(self, data, all_arithmetic_operators):
        # invalid ops
        op_name = all_arithmetic_operators
        with pytest.raises(AttributeError):
            getattr(data, op_name)


def test_depr_na_sentinel():
    # GH#46910
    msg = "The `na_sentinel` argument of `MyEA.factorize` is deprecated"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):

        class MyEA(ExtensionArray):
            def factorize(self, na_sentinel=-1):
                pass

        with tm.assert_produces_warning(None):
            MyEA()

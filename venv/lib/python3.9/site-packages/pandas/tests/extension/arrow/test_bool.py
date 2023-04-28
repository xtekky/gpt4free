import numpy as np
import pytest

from pandas.compat import (
    is_ci_environment,
    is_platform_windows,
)

import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_bool_dtype
from pandas.tests.extension import base

pytest.importorskip("pyarrow", minversion="1.0.1")

from pandas.tests.extension.arrow.arrays import (  # isort:skip
    ArrowBoolArray,
    ArrowBoolDtype,
)


@pytest.fixture
def dtype():
    return ArrowBoolDtype()


@pytest.fixture
def data():
    values = np.random.randint(0, 2, size=100, dtype=bool)
    values[1] = ~values[0]
    return ArrowBoolArray._from_sequence(values)


@pytest.fixture
def data_missing():
    return ArrowBoolArray._from_sequence([None, True])


def test_basic_equals(data):
    # https://github.com/pandas-dev/pandas/issues/34660
    assert pd.Series(data).equals(pd.Series(data))


class BaseArrowTests:
    pass


class TestDtype(BaseArrowTests, base.BaseDtypeTests):
    pass


class TestInterface(BaseArrowTests, base.BaseInterfaceTests):
    def test_copy(self, data):
        # __setitem__ does not work, so we only have a smoke-test
        data.copy()

    def test_view(self, data):
        # __setitem__ does not work, so we only have a smoke-test
        data.view()

    @pytest.mark.xfail(
        raises=AssertionError,
        reason="Doesn't recognize data._na_value as NA",
    )
    def test_contains(self, data, data_missing):
        super().test_contains(data, data_missing)


class TestConstructors(BaseArrowTests, base.BaseConstructorsTests):
    @pytest.mark.xfail(reason="pa.NULL is not recognised as scalar, GH-33899")
    def test_series_constructor_no_data_with_index(self, dtype, na_value):
        # pyarrow.lib.ArrowInvalid: only handle 1-dimensional arrays
        super().test_series_constructor_no_data_with_index(dtype, na_value)

    @pytest.mark.xfail(reason="pa.NULL is not recognised as scalar, GH-33899")
    def test_series_constructor_scalar_na_with_index(self, dtype, na_value):
        # pyarrow.lib.ArrowInvalid: only handle 1-dimensional arrays
        super().test_series_constructor_scalar_na_with_index(dtype, na_value)

    @pytest.mark.xfail(reason="_from_sequence ignores dtype keyword")
    def test_empty(self, dtype):
        super().test_empty(dtype)


class TestReduce(base.BaseNoReduceTests):
    def test_reduce_series_boolean(self):
        pass


@pytest.mark.skipif(
    is_ci_environment() and is_platform_windows(),
    reason="Causes stack overflow on Windows CI",
)
class TestReduceBoolean(base.BaseBooleanReduceTests):
    pass


def test_is_bool_dtype(data):
    assert is_bool_dtype(data)
    assert pd.core.common.is_bool_indexer(data)
    s = pd.Series(range(len(data)))
    result = s[data]
    expected = s[np.asarray(data)]
    tm.assert_series_equal(result, expected)

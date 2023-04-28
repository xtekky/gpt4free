import pytest

import altair as alt
from altair.utils import AltairDeprecationWarning
from altair.utils.deprecation import _deprecate, deprecated


def test_deprecated_class():
    OldChart = _deprecate(alt.Chart, "OldChart")
    with pytest.warns(AltairDeprecationWarning) as record:
        OldChart()
    assert "alt.OldChart" in record[0].message.args[0]
    assert "alt.Chart" in record[0].message.args[0]


def test_deprecation_decorator():
    @deprecated(message="func is deprecated")
    def func(x):
        return x + 1

    with pytest.warns(AltairDeprecationWarning) as record:
        y = func(1)
    assert y == 2
    assert record[0].message.args[0] == "func is deprecated"

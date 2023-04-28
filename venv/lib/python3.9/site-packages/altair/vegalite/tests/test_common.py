"""Tests of functionality that should work in all vegalite versions"""

import pytest

import pandas as pd

from .. import v3, v4


@pytest.fixture
def basic_spec():
    return {
        "data": {"url": "data.csv"},
        "mark": "line",
        "encoding": {
            "color": {"type": "nominal", "field": "color"},
            "x": {"type": "quantitative", "field": "xval"},
            "y": {"type": "ordinal", "field": "yval"},
        },
    }


def make_final_spec(alt, basic_spec):
    theme = alt.themes.get()
    spec = theme()
    spec.update(basic_spec)
    return spec


def make_basic_chart(alt):
    data = pd.DataFrame(
        {
            "a": ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
            "b": [28, 55, 43, 91, 81, 53, 19, 87, 52],
        }
    )

    return alt.Chart(data).mark_bar().encode(x="a", y="b")


@pytest.mark.parametrize("alt", [v3, v4])
def test_basic_chart_to_dict(alt, basic_spec):
    chart = (
        alt.Chart("data.csv")
        .mark_line()
        .encode(alt.X("xval:Q"), y=alt.Y("yval:O"), color="color:N")
    )
    dct = chart.to_dict()

    # schema should be in the top level
    assert dct.pop("$schema").startswith("http")

    # remainder of spec should match the basic spec
    assert dct == make_final_spec(alt, basic_spec)


@pytest.mark.parametrize("alt", [v3, v4])
def test_basic_chart_from_dict(alt, basic_spec):
    chart = alt.Chart.from_dict(basic_spec)
    dct = chart.to_dict()

    # schema should be in the top level
    assert dct.pop("$schema").startswith("http")

    # remainder of spec should match the basic spec
    assert dct == make_final_spec(alt, basic_spec)


@pytest.mark.parametrize("alt", [v3, v4])
def test_theme_enable(alt, basic_spec):
    active_theme = alt.themes.active

    try:
        alt.themes.enable("none")

        chart = alt.Chart.from_dict(basic_spec)
        dct = chart.to_dict()

        # schema should be in the top level
        assert dct.pop("$schema").startswith("http")

        # remainder of spec should match the basic spec
        # without any theme settings
        assert dct == basic_spec
    finally:
        # reset the theme to its initial value
        alt.themes.enable(active_theme)


@pytest.mark.parametrize("alt", [v3, v4])
def test_max_rows(alt):
    basic_chart = make_basic_chart(alt)

    with alt.data_transformers.enable("default"):
        basic_chart.to_dict()  # this should not fail

    with alt.data_transformers.enable("default", max_rows=5):
        with pytest.raises(alt.MaxRowsError):
            basic_chart.to_dict()  # this should not fail

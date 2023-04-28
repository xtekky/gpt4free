import pytest

import altair as alt
from ..mimebundle import spec_to_mimebundle


@pytest.fixture
def require_altair_saver():
    try:
        import altair_saver  # noqa: F401
    except ImportError:
        pytest.skip("altair_saver not importable; cannot run saver tests")


@pytest.fixture
def vegalite_spec():
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v4.json",
        "description": "A simple bar chart with embedded data.",
        "data": {
            "values": [
                {"a": "A", "b": 28},
                {"a": "B", "b": 55},
                {"a": "C", "b": 43},
                {"a": "D", "b": 91},
                {"a": "E", "b": 81},
                {"a": "F", "b": 53},
                {"a": "G", "b": 19},
                {"a": "H", "b": 87},
                {"a": "I", "b": 52},
            ]
        },
        "mark": "bar",
        "encoding": {
            "x": {"field": "a", "type": "ordinal"},
            "y": {"field": "b", "type": "quantitative"},
        },
    }


@pytest.fixture
def vega_spec():
    return {
        "$schema": "https://vega.github.io/schema/vega/v5.json",
        "axes": [
            {
                "aria": False,
                "domain": False,
                "grid": True,
                "gridScale": "x",
                "labels": False,
                "maxExtent": 0,
                "minExtent": 0,
                "orient": "left",
                "scale": "y",
                "tickCount": {"signal": "ceil(height/40)"},
                "ticks": False,
                "zindex": 0,
            },
            {
                "grid": False,
                "labelAlign": "right",
                "labelAngle": 270,
                "labelBaseline": "middle",
                "orient": "bottom",
                "scale": "x",
                "title": "a",
                "zindex": 0,
            },
            {
                "grid": False,
                "labelOverlap": True,
                "orient": "left",
                "scale": "y",
                "tickCount": {"signal": "ceil(height/40)"},
                "title": "b",
                "zindex": 0,
            },
        ],
        "background": "white",
        "data": [
            {
                "name": "source_0",
                "values": [
                    {"a": "A", "b": 28},
                    {"a": "B", "b": 55},
                    {"a": "C", "b": 43},
                    {"a": "D", "b": 91},
                    {"a": "E", "b": 81},
                    {"a": "F", "b": 53},
                    {"a": "G", "b": 19},
                    {"a": "H", "b": 87},
                    {"a": "I", "b": 52},
                ],
            },
            {
                "name": "data_0",
                "source": "source_0",
                "transform": [
                    {
                        "expr": 'isValid(datum["b"]) && isFinite(+datum["b"])',
                        "type": "filter",
                    }
                ],
            },
        ],
        "description": "A simple bar chart with embedded data.",
        "height": 200,
        "marks": [
            {
                "encode": {
                    "update": {
                        "ariaRoleDescription": {"value": "bar"},
                        "description": {
                            "signal": '"a: " + (isValid(datum["a"]) ? datum["a"] : ""+datum["a"]) + "; b: " + (format(datum["b"], ""))'
                        },
                        "fill": {"value": "#4c78a8"},
                        "width": {"band": 1, "scale": "x"},
                        "x": {"field": "a", "scale": "x"},
                        "y": {"field": "b", "scale": "y"},
                        "y2": {"scale": "y", "value": 0},
                    }
                },
                "from": {"data": "data_0"},
                "name": "marks",
                "style": ["bar"],
                "type": "rect",
            }
        ],
        "padding": 5,
        "scales": [
            {
                "domain": {"data": "data_0", "field": "a", "sort": True},
                "name": "x",
                "paddingInner": 0.1,
                "paddingOuter": 0.05,
                "range": {"step": {"signal": "x_step"}},
                "type": "band",
            },
            {
                "domain": {"data": "data_0", "field": "b"},
                "name": "y",
                "nice": True,
                "range": [{"signal": "height"}, 0],
                "type": "linear",
                "zero": True,
            },
        ],
        "signals": [
            {"name": "x_step", "value": 20},
            {
                "name": "width",
                "update": "bandspace(domain('x').length, 0.1, 0.05) * x_step",
            },
        ],
        "style": "cell",
    }


def test_vegalite_to_vega_mimebundle(require_altair_saver, vegalite_spec, vega_spec):
    # temporay fix for https://github.com/vega/vega-lite/issues/7776
    def delete_none(axes):
        for axis in axes:
            for key, value in list(axis.items()):
                if value is None:
                    del axis[key]
        return axes

    bundle = spec_to_mimebundle(
        spec=vegalite_spec,
        format="vega",
        mode="vega-lite",
        vega_version=alt.VEGA_VERSION,
        vegalite_version=alt.VEGALITE_VERSION,
        vegaembed_version=alt.VEGAEMBED_VERSION,
    )

    bundle["application/vnd.vega.v5+json"]["axes"] = delete_none(
        bundle["application/vnd.vega.v5+json"]["axes"]
    )
    assert bundle == {"application/vnd.vega.v5+json": vega_spec}


def test_spec_to_vegalite_mimebundle(vegalite_spec):
    bundle = spec_to_mimebundle(
        spec=vegalite_spec,
        mode="vega-lite",
        format="vega-lite",
        vegalite_version=alt.VEGALITE_VERSION,
    )
    assert bundle == {"application/vnd.vegalite.v4+json": vegalite_spec}


def test_spec_to_vega_mimebundle(vega_spec):
    bundle = spec_to_mimebundle(
        spec=vega_spec, mode="vega", format="vega", vega_version=alt.VEGA_VERSION
    )
    assert bundle == {"application/vnd.vega.v5+json": vega_spec}


def test_spec_to_json_mimebundle():
    bundle = spec_to_mimebundle(
        spec=vegalite_spec,
        mode="vega-lite",
        format="json",
    )
    assert bundle == {"application/json": vegalite_spec}

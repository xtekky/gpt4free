import json
import pytest

try:
    from IPython import InteractiveShell

    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False
    pass

from altair.vegalite.v4 import VegaLite
from altair.vega.v5 import Vega


DATA_RECORDS = [
    {"amount": 28, "category": "A"},
    {"amount": 55, "category": "B"},
    {"amount": 43, "category": "C"},
    {"amount": 91, "category": "D"},
    {"amount": 81, "category": "E"},
    {"amount": 53, "category": "F"},
    {"amount": 19, "category": "G"},
    {"amount": 87, "category": "H"},
]

if IPYTHON_AVAILABLE:
    _ipshell = InteractiveShell.instance()
    _ipshell.run_cell("%load_ext altair")
    _ipshell.run_cell(
        """
import pandas as pd
table = pd.DataFrame.from_records({})
the_data = table
""".format(
            DATA_RECORDS
        )
    )


VEGA_SPEC = {
    "$schema": "https://vega.github.io/schema/vega/v5.json",
    "axes": [
        {"orient": "bottom", "scale": "xscale"},
        {"orient": "left", "scale": "yscale"},
    ],
    "data": [{"name": "table", "values": DATA_RECORDS}],
    "height": 200,
    "marks": [
        {
            "encode": {
                "enter": {
                    "width": {"band": 1, "scale": "xscale"},
                    "x": {"field": "category", "scale": "xscale"},
                    "y": {"field": "amount", "scale": "yscale"},
                    "y2": {"scale": "yscale", "value": 0},
                },
                "hover": {"fill": {"value": "red"}},
                "update": {"fill": {"value": "steelblue"}},
            },
            "from": {"data": "table"},
            "type": "rect",
        },
        {
            "encode": {
                "enter": {
                    "align": {"value": "center"},
                    "baseline": {"value": "bottom"},
                    "fill": {"value": "#333"},
                },
                "update": {
                    "fillOpacity": [
                        {"test": "datum === tooltip", "value": 0},
                        {"value": 1},
                    ],
                    "text": {"signal": "tooltip.amount"},
                    "x": {"band": 0.5, "scale": "xscale", "signal": "tooltip.category"},
                    "y": {"offset": -2, "scale": "yscale", "signal": "tooltip.amount"},
                },
            },
            "type": "text",
        },
    ],
    "padding": 5,
    "scales": [
        {
            "domain": {"data": "table", "field": "category"},
            "name": "xscale",
            "padding": 0.05,
            "range": "width",
            "round": True,
            "type": "band",
        },
        {
            "domain": {"data": "table", "field": "amount"},
            "name": "yscale",
            "nice": True,
            "range": "height",
        },
    ],
    "signals": [
        {
            "name": "tooltip",
            "on": [
                {"events": "rect:mouseover", "update": "datum"},
                {"events": "rect:mouseout", "update": "{}"},
            ],
            "value": {},
        }
    ],
    "width": 400,
}


VEGALITE_SPEC = {
    "$schema": "https://vega.github.io/schema/vega-lite/v4.json",
    "data": {"values": DATA_RECORDS},
    "description": "A simple bar chart with embedded data.",
    "encoding": {
        "x": {"field": "category", "type": "ordinal"},
        "y": {"field": "amount", "type": "quantitative"},
    },
    "mark": "bar",
}


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="requires ipython")
def test_vegalite_magic_data_included():
    result = _ipshell.run_cell("%%vegalite\n" + json.dumps(VEGALITE_SPEC))
    assert isinstance(result.result, VegaLite)
    assert VEGALITE_SPEC == result.result.spec


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="requires ipython")
def test_vegalite_magic_json_flag():
    result = _ipshell.run_cell("%%vegalite --json\n" + json.dumps(VEGALITE_SPEC))
    assert isinstance(result.result, VegaLite)
    assert VEGALITE_SPEC == result.result.spec


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="requires ipython")
def test_vegalite_magic_pandas_data():
    spec = {key: val for key, val in VEGALITE_SPEC.items() if key != "data"}
    result = _ipshell.run_cell("%%vegalite table\n" + json.dumps(spec))
    assert isinstance(result.result, VegaLite)
    assert VEGALITE_SPEC == result.result.spec


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="requires ipython")
def test_vega_magic_data_included():
    result = _ipshell.run_cell("%%vega\n" + json.dumps(VEGA_SPEC))
    assert isinstance(result.result, Vega)
    assert VEGA_SPEC == result.result.spec


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="requires ipython")
def test_vega_magic_json_flag():
    result = _ipshell.run_cell("%%vega --json\n" + json.dumps(VEGA_SPEC))
    assert isinstance(result.result, Vega)
    assert VEGA_SPEC == result.result.spec


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="requires ipython")
def test_vega_magic_pandas_data():
    spec = {key: val for key, val in VEGA_SPEC.items() if key != "data"}
    result = _ipshell.run_cell("%%vega table\n" + json.dumps(spec))
    assert isinstance(result.result, Vega)
    assert VEGA_SPEC == result.result.spec


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="requires ipython")
def test_vega_magic_pandas_data_renamed():
    spec = {key: val for key, val in VEGA_SPEC.items() if key != "data"}
    result = _ipshell.run_cell("%%vega table:the_data\n" + json.dumps(spec))
    assert isinstance(result.result, Vega)
    assert VEGA_SPEC == result.result.spec

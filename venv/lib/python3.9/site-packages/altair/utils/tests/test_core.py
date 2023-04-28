import types

import numpy as np
import pandas as pd
import pytest

import altair as alt
from .. import parse_shorthand, update_nested, infer_encoding_types
from ..core import infer_dtype

FAKE_CHANNELS_MODULE = '''
"""Fake channels module for utility tests."""

from altair.utils import schemapi


class FieldChannel(object):
    def __init__(self, shorthand, **kwargs):
        kwargs['shorthand'] = shorthand
        return super(FieldChannel, self).__init__(**kwargs)


class ValueChannel(object):
    def __init__(self, value, **kwargs):
        kwargs['value'] = value
        return super(ValueChannel, self).__init__(**kwargs)


class X(FieldChannel, schemapi.SchemaBase):
    _schema = {}
    _encoding_name = "x"


class XValue(ValueChannel, schemapi.SchemaBase):
    _schema = {}
    _encoding_name = "x"


class Y(FieldChannel, schemapi.SchemaBase):
    _schema = {}
    _encoding_name = "y"


class YValue(ValueChannel, schemapi.SchemaBase):
    _schema = {}
    _encoding_name = "y"


class StrokeWidth(FieldChannel, schemapi.SchemaBase):
    _schema = {}
    _encoding_name = "strokeWidth"


class StrokeWidthValue(ValueChannel, schemapi.SchemaBase):
    _schema = {}
    _encoding_name = "strokeWidth"
'''


@pytest.mark.parametrize(
    "value,expected_type",
    [
        ([1, 2, 3], "integer"),
        ([1.0, 2.0, 3.0], "floating"),
        ([1, 2.0, 3], "mixed-integer-float"),
        (["a", "b", "c"], "string"),
        (["a", "b", np.nan], "mixed"),
    ],
)
def test_infer_dtype(value, expected_type):
    assert infer_dtype(value) == expected_type


def test_parse_shorthand():
    def check(s, **kwargs):
        assert parse_shorthand(s) == kwargs

    check("")

    # Fields alone
    check("foobar", field="foobar")
    check("blah:(fd ", field="blah:(fd ")

    # Fields with type
    check("foobar:quantitative", type="quantitative", field="foobar")
    check("foobar:nominal", type="nominal", field="foobar")
    check("foobar:ordinal", type="ordinal", field="foobar")
    check("foobar:temporal", type="temporal", field="foobar")
    check("foobar:geojson", type="geojson", field="foobar")

    check("foobar:Q", type="quantitative", field="foobar")
    check("foobar:N", type="nominal", field="foobar")
    check("foobar:O", type="ordinal", field="foobar")
    check("foobar:T", type="temporal", field="foobar")
    check("foobar:G", type="geojson", field="foobar")

    # Fields with aggregate and/or type
    check("average(foobar)", field="foobar", aggregate="average")
    check("min(foobar):temporal", type="temporal", field="foobar", aggregate="min")
    check("sum(foobar):Q", type="quantitative", field="foobar", aggregate="sum")

    # check that invalid arguments are not split-out
    check("invalid(blah)", field="invalid(blah)")
    check("blah:invalid", field="blah:invalid")
    check("invalid(blah):invalid", field="invalid(blah):invalid")

    # check parsing in presence of strange characters
    check(
        "average(a b:(c\nd):Q",
        aggregate="average",
        field="a b:(c\nd",
        type="quantitative",
    )

    # special case: count doesn't need an argument
    check("count()", aggregate="count", type="quantitative")
    check("count():O", aggregate="count", type="ordinal")

    # time units:
    check("month(x)", field="x", timeUnit="month", type="temporal")
    check("year(foo):O", field="foo", timeUnit="year", type="ordinal")
    check("date(date):quantitative", field="date", timeUnit="date", type="quantitative")
    check(
        "yearmonthdate(field)", field="field", timeUnit="yearmonthdate", type="temporal"
    )


def test_parse_shorthand_with_data():
    def check(s, data, **kwargs):
        assert parse_shorthand(s, data) == kwargs

    data = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": ["A", "B", "C", "D", "E"],
            "z": pd.date_range("2018-01-01", periods=5, freq="D"),
            "t": pd.date_range("2018-01-01", periods=5, freq="D").tz_localize("UTC"),
        }
    )

    check("x", data, field="x", type="quantitative")
    check("y", data, field="y", type="nominal")
    check("z", data, field="z", type="temporal")
    check("t", data, field="t", type="temporal")
    check("count(x)", data, field="x", aggregate="count", type="quantitative")
    check("count()", data, aggregate="count", type="quantitative")
    check("month(z)", data, timeUnit="month", field="z", type="temporal")
    check("month(t)", data, timeUnit="month", field="t", type="temporal")


def test_parse_shorthand_all_aggregates():
    aggregates = alt.Root._schema["definitions"]["AggregateOp"]["enum"]
    for aggregate in aggregates:
        shorthand = "{aggregate}(field):Q".format(aggregate=aggregate)
        assert parse_shorthand(shorthand) == {
            "aggregate": aggregate,
            "field": "field",
            "type": "quantitative",
        }


def test_parse_shorthand_all_timeunits():
    timeUnits = []
    for loc in ["Local", "Utc"]:
        for typ in ["Single", "Multi"]:
            defn = loc + typ + "TimeUnit"
            timeUnits.extend(alt.Root._schema["definitions"][defn]["enum"])
    for timeUnit in timeUnits:
        shorthand = "{timeUnit}(field):Q".format(timeUnit=timeUnit)
        assert parse_shorthand(shorthand) == {
            "timeUnit": timeUnit,
            "field": "field",
            "type": "quantitative",
        }


def test_parse_shorthand_window_count():
    shorthand = "count()"
    dct = parse_shorthand(
        shorthand,
        parse_aggregates=False,
        parse_window_ops=True,
        parse_timeunits=False,
        parse_types=False,
    )
    assert dct == {"op": "count"}


def test_parse_shorthand_all_window_ops():
    window_ops = alt.Root._schema["definitions"]["WindowOnlyOp"]["enum"]
    aggregates = alt.Root._schema["definitions"]["AggregateOp"]["enum"]
    for op in window_ops + aggregates:
        shorthand = "{op}(field)".format(op=op)
        dct = parse_shorthand(
            shorthand,
            parse_aggregates=False,
            parse_window_ops=True,
            parse_timeunits=False,
            parse_types=False,
        )
        assert dct == {"field": "field", "op": op}


def test_update_nested():
    original = {"x": {"b": {"foo": 2}, "c": 4}}
    update = {"x": {"b": {"foo": 5}, "d": 6}, "y": 40}

    output = update_nested(original, update, copy=True)
    assert output is not original
    assert output == {"x": {"b": {"foo": 5}, "c": 4, "d": 6}, "y": 40}

    output2 = update_nested(original, update)
    assert output2 is original
    assert output == output2


@pytest.fixture
def channels():
    channels = types.ModuleType("channels")
    exec(FAKE_CHANNELS_MODULE, channels.__dict__)
    return channels


def _getargs(*args, **kwargs):
    return args, kwargs


def test_infer_encoding_types(channels):
    expected = dict(
        x=channels.X("xval"),
        y=channels.YValue("yval"),
        strokeWidth=channels.StrokeWidthValue(value=4),
    )

    # All positional args
    args, kwds = _getargs(
        channels.X("xval"), channels.YValue("yval"), channels.StrokeWidthValue(4)
    )
    assert infer_encoding_types(args, kwds, channels) == expected

    # All keyword args
    args, kwds = _getargs(x="xval", y=alt.value("yval"), strokeWidth=alt.value(4))
    assert infer_encoding_types(args, kwds, channels) == expected

    # Mixed positional & keyword
    args, kwds = _getargs(
        channels.X("xval"), channels.YValue("yval"), strokeWidth=alt.value(4)
    )
    assert infer_encoding_types(args, kwds, channels) == expected


def test_infer_encoding_types_with_condition(channels):
    args, kwds = _getargs(
        x=alt.condition("pred1", alt.value(1), alt.value(2)),
        y=alt.condition("pred2", alt.value(1), "yval"),
        strokeWidth=alt.condition("pred3", "sval", alt.value(2)),
    )
    expected = dict(
        x=channels.XValue(2, condition=channels.XValue(1, test="pred1")),
        y=channels.Y("yval", condition=channels.YValue(1, test="pred2")),
        strokeWidth=channels.StrokeWidthValue(
            2, condition=channels.StrokeWidth("sval", test="pred3")
        ),
    )
    assert infer_encoding_types(args, kwds, channels) == expected

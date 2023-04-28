"""Unit tests for altair API"""

import io
import json
import operator
import os
import tempfile

import jsonschema
import pytest
import pandas as pd

import altair.vegalite.v3 as alt
from altair.utils import AltairDeprecationWarning

try:
    import altair_saver  # noqa: F401
except ImportError:
    altair_saver = None


def getargs(*args, **kwargs):
    return args, kwargs


OP_DICT = {
    "layer": operator.add,
    "hconcat": operator.or_,
    "vconcat": operator.and_,
}


def _make_chart_type(chart_type):
    data = pd.DataFrame(
        {
            "x": [28, 55, 43, 91, 81, 53, 19, 87],
            "y": [43, 91, 81, 53, 19, 87, 52, 28],
            "color": list("AAAABBBB"),
        }
    )
    base = (
        alt.Chart(data)
        .mark_point()
        .encode(
            x="x",
            y="y",
            color="color",
        )
    )

    if chart_type in ["layer", "hconcat", "vconcat", "concat"]:
        func = getattr(alt, chart_type)
        return func(base.mark_square(), base.mark_circle())
    elif chart_type == "facet":
        return base.facet("color")
    elif chart_type == "facet_encoding":
        return base.encode(facet="color")
    elif chart_type == "repeat":
        return base.encode(alt.X(alt.repeat(), type="quantitative")).repeat(["x", "y"])
    elif chart_type == "chart":
        return base
    else:
        raise ValueError("chart_type='{}' is not recognized".format(chart_type))


@pytest.fixture
def basic_chart():
    data = pd.DataFrame(
        {
            "a": ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
            "b": [28, 55, 43, 91, 81, 53, 19, 87, 52],
        }
    )

    return alt.Chart(data).mark_bar().encode(x="a", y="b")


def test_chart_data_types():
    def Chart(data):
        return alt.Chart(data).mark_point().encode(x="x:Q", y="y:Q")

    # Url Data
    data = "/path/to/my/data.csv"
    dct = Chart(data).to_dict()
    assert dct["data"] == {"url": data}

    # Dict Data
    data = {"values": [{"x": 1, "y": 2}, {"x": 2, "y": 3}]}
    with alt.data_transformers.enable(consolidate_datasets=False):
        dct = Chart(data).to_dict()
    assert dct["data"] == data

    with alt.data_transformers.enable(consolidate_datasets=True):
        dct = Chart(data).to_dict()
    name = dct["data"]["name"]
    assert dct["datasets"][name] == data["values"]

    # DataFrame data
    data = pd.DataFrame({"x": range(5), "y": range(5)})
    with alt.data_transformers.enable(consolidate_datasets=False):
        dct = Chart(data).to_dict()
    assert dct["data"]["values"] == data.to_dict(orient="records")

    with alt.data_transformers.enable(consolidate_datasets=True):
        dct = Chart(data).to_dict()
    name = dct["data"]["name"]
    assert dct["datasets"][name] == data.to_dict(orient="records")

    # Named data object
    data = alt.NamedData(name="Foo")
    dct = Chart(data).to_dict()
    assert dct["data"] == {"name": "Foo"}


def test_chart_infer_types():
    data = pd.DataFrame(
        {
            "x": pd.date_range("2012", periods=10, freq="Y"),
            "y": range(10),
            "c": list("abcabcabca"),
        }
    )

    def _check_encodings(chart):
        dct = chart.to_dict()
        assert dct["encoding"]["x"]["type"] == "temporal"
        assert dct["encoding"]["x"]["field"] == "x"
        assert dct["encoding"]["y"]["type"] == "quantitative"
        assert dct["encoding"]["y"]["field"] == "y"
        assert dct["encoding"]["color"]["type"] == "nominal"
        assert dct["encoding"]["color"]["field"] == "c"

    # Pass field names by keyword
    chart = alt.Chart(data).mark_point().encode(x="x", y="y", color="c")
    _check_encodings(chart)

    # pass Channel objects by keyword
    chart = (
        alt.Chart(data)
        .mark_point()
        .encode(x=alt.X("x"), y=alt.Y("y"), color=alt.Color("c"))
    )
    _check_encodings(chart)

    # pass Channel objects by value
    chart = alt.Chart(data).mark_point().encode(alt.X("x"), alt.Y("y"), alt.Color("c"))
    _check_encodings(chart)

    # override default types
    chart = (
        alt.Chart(data)
        .mark_point()
        .encode(alt.X("x", type="nominal"), alt.Y("y", type="ordinal"))
    )
    dct = chart.to_dict()
    assert dct["encoding"]["x"]["type"] == "nominal"
    assert dct["encoding"]["y"]["type"] == "ordinal"


@pytest.mark.parametrize(
    "args, kwargs",
    [
        getargs(detail=["value:Q", "name:N"], tooltip=["value:Q", "name:N"]),
        getargs(detail=["value", "name"], tooltip=["value", "name"]),
        getargs(alt.Detail(["value:Q", "name:N"]), alt.Tooltip(["value:Q", "name:N"])),
        getargs(alt.Detail(["value", "name"]), alt.Tooltip(["value", "name"])),
        getargs(
            [alt.Detail("value:Q"), alt.Detail("name:N")],
            [alt.Tooltip("value:Q"), alt.Tooltip("name:N")],
        ),
        getargs(
            [alt.Detail("value"), alt.Detail("name")],
            [alt.Tooltip("value"), alt.Tooltip("name")],
        ),
    ],
)
def test_multiple_encodings(args, kwargs):
    df = pd.DataFrame({"value": [1, 2, 3], "name": ["A", "B", "C"]})
    encoding_dct = [
        {"field": "value", "type": "quantitative"},
        {"field": "name", "type": "nominal"},
    ]
    chart = alt.Chart(df).mark_point().encode(*args, **kwargs)
    dct = chart.to_dict()
    assert dct["encoding"]["detail"] == encoding_dct
    assert dct["encoding"]["tooltip"] == encoding_dct


def test_chart_operations():
    data = pd.DataFrame(
        {
            "x": pd.date_range("2012", periods=10, freq="Y"),
            "y": range(10),
            "c": list("abcabcabca"),
        }
    )
    chart1 = alt.Chart(data).mark_line().encode(x="x", y="y", color="c")
    chart2 = chart1.mark_point()
    chart3 = chart1.mark_circle()
    chart4 = chart1.mark_square()

    chart = chart1 + chart2 + chart3
    assert isinstance(chart, alt.LayerChart)
    assert len(chart.layer) == 3
    chart += chart4
    assert len(chart.layer) == 4

    chart = chart1 | chart2 | chart3
    assert isinstance(chart, alt.HConcatChart)
    assert len(chart.hconcat) == 3
    chart |= chart4
    assert len(chart.hconcat) == 4

    chart = chart1 & chart2 & chart3
    assert isinstance(chart, alt.VConcatChart)
    assert len(chart.vconcat) == 3
    chart &= chart4
    assert len(chart.vconcat) == 4


def test_selection_to_dict():
    brush = alt.selection(type="interval")

    # test some value selections
    # Note: X and Y cannot have conditions
    alt.Chart("path/to/data.json").mark_point().encode(
        color=alt.condition(brush, alt.ColorValue("red"), alt.ColorValue("blue")),
        opacity=alt.condition(brush, alt.value(0.5), alt.value(1.0)),
        text=alt.condition(brush, alt.TextValue("foo"), alt.value("bar")),
    ).to_dict()

    # test some field selections
    # Note: X and Y cannot have conditions
    # Conditions cannot both be fields
    alt.Chart("path/to/data.json").mark_point().encode(
        color=alt.condition(brush, alt.Color("col1:N"), alt.value("blue")),
        opacity=alt.condition(brush, "col1:N", alt.value(0.5)),
        text=alt.condition(brush, alt.value("abc"), alt.Text("col2:N")),
        size=alt.condition(brush, alt.value(20), "col2:N"),
    ).to_dict()


def test_selection_expression():
    selection = alt.selection_single(fields=["value"])

    assert isinstance(selection.value, alt.expr.Expression)
    assert selection.value.to_dict() == "{0}.value".format(selection.name)

    assert isinstance(selection["value"], alt.expr.Expression)
    assert selection["value"].to_dict() == "{0}['value']".format(selection.name)

    with pytest.raises(AttributeError):
        selection.__magic__


@pytest.mark.parametrize("format", ["html", "json", "png", "svg", "pdf"])
def test_save(format, basic_chart):
    if format in ["pdf", "png"]:
        out = io.BytesIO()
        mode = "rb"
    else:
        out = io.StringIO()
        mode = "r"

    if format in ["svg", "png", "pdf"]:
        if not altair_saver:
            with pytest.raises(ValueError) as err:
                basic_chart.save(out, format=format)
            assert "github.com/altair-viz/altair_saver" in str(err.value)
            return
        elif format not in altair_saver.available_formats():
            with pytest.raises(ValueError) as err:
                basic_chart.save(out, format=format)
            assert f"No enabled saver found that supports format='{format}'" in str(
                err.value
            )
            return

    basic_chart.save(out, format=format)
    out.seek(0)
    content = out.read()

    if format == "json":
        assert "$schema" in json.loads(content)
    if format == "html":
        assert content.startswith("<!DOCTYPE html>")

    fid, filename = tempfile.mkstemp(suffix="." + format)
    os.close(fid)

    try:
        basic_chart.save(filename)
        with open(filename, mode) as f:
            assert f.read()[:1000] == content[:1000]
    finally:
        os.remove(filename)


def test_facet_basic():
    # wrapped facet
    chart1 = (
        alt.Chart("data.csv")
        .mark_point()
        .encode(
            x="x:Q",
            y="y:Q",
        )
        .facet("category:N", columns=2)
    )

    dct1 = chart1.to_dict()

    assert dct1["facet"] == alt.Facet("category:N").to_dict()
    assert dct1["columns"] == 2
    assert dct1["data"] == alt.UrlData("data.csv").to_dict()

    # explicit row/col facet
    chart2 = (
        alt.Chart("data.csv")
        .mark_point()
        .encode(
            x="x:Q",
            y="y:Q",
        )
        .facet(row="category1:Q", column="category2:Q")
    )

    dct2 = chart2.to_dict()

    assert dct2["facet"]["row"] == alt.Facet("category1:Q").to_dict()
    assert dct2["facet"]["column"] == alt.Facet("category2:Q").to_dict()
    assert "columns" not in dct2
    assert dct2["data"] == alt.UrlData("data.csv").to_dict()


def test_facet_parse():
    chart = (
        alt.Chart("data.csv")
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .facet(row="row:N", column="column:O")
    )
    dct = chart.to_dict()
    assert dct["data"] == {"url": "data.csv"}
    assert "data" not in dct["spec"]
    assert dct["facet"] == {
        "column": {"field": "column", "type": "ordinal"},
        "row": {"field": "row", "type": "nominal"},
    }


def test_facet_parse_data():
    data = pd.DataFrame({"x": range(5), "y": range(5), "row": list("abcab")})
    chart = (
        alt.Chart(data)
        .mark_point()
        .encode(x="x", y="y:O")
        .facet(row="row", column="column:O")
    )
    with alt.data_transformers.enable(consolidate_datasets=False):
        dct = chart.to_dict()
    assert "values" in dct["data"]
    assert "data" not in dct["spec"]
    assert dct["facet"] == {
        "column": {"field": "column", "type": "ordinal"},
        "row": {"field": "row", "type": "nominal"},
    }

    with alt.data_transformers.enable(consolidate_datasets=True):
        dct = chart.to_dict()
    assert "datasets" in dct
    assert "name" in dct["data"]
    assert "data" not in dct["spec"]
    assert dct["facet"] == {
        "column": {"field": "column", "type": "ordinal"},
        "row": {"field": "row", "type": "nominal"},
    }


def test_selection():
    # test instantiation of selections
    interval = alt.selection_interval(name="selec_1")
    assert interval.selection.type == "interval"
    assert interval.name == "selec_1"

    single = alt.selection_single(name="selec_2")
    assert single.selection.type == "single"
    assert single.name == "selec_2"

    multi = alt.selection_multi(name="selec_3")
    assert multi.selection.type == "multi"
    assert multi.name == "selec_3"

    # test adding to chart
    chart = alt.Chart().add_selection(single)
    chart = chart.add_selection(multi, interval)
    assert set(chart.selection.keys()) == {"selec_1", "selec_2", "selec_3"}

    # test logical operations
    assert isinstance(single & multi, alt.Selection)
    assert isinstance(single | multi, alt.Selection)
    assert isinstance(~single, alt.Selection)
    assert isinstance((single & multi)[0].group, alt.SelectionAnd)
    assert isinstance((single | multi)[0].group, alt.SelectionOr)
    assert isinstance((~single)[0].group, alt.SelectionNot)

    # test that default names increment (regression for #1454)
    sel1 = alt.selection_single()
    sel2 = alt.selection_multi()
    sel3 = alt.selection_interval()
    names = {s.name for s in (sel1, sel2, sel3)}
    assert len(names) == 3


def test_transforms():
    # aggregate transform
    agg1 = alt.AggregatedFieldDef(**{"as": "x1", "op": "mean", "field": "y"})
    agg2 = alt.AggregatedFieldDef(**{"as": "x2", "op": "median", "field": "z"})
    chart = alt.Chart().transform_aggregate([agg1], ["foo"], x2="median(z)")
    kwds = dict(aggregate=[agg1, agg2], groupby=["foo"])
    assert chart.transform == [alt.AggregateTransform(**kwds)]

    # bin transform
    chart = alt.Chart().transform_bin("binned", field="field", bin=True)
    kwds = {"as": "binned", "field": "field", "bin": True}
    assert chart.transform == [alt.BinTransform(**kwds)]

    # calcualte transform
    chart = alt.Chart().transform_calculate("calc", "datum.a * 4")
    kwds = {"as": "calc", "calculate": "datum.a * 4"}
    assert chart.transform == [alt.CalculateTransform(**kwds)]

    # impute transform
    chart = alt.Chart().transform_impute("field", "key", groupby=["x"])
    kwds = {"impute": "field", "key": "key", "groupby": ["x"]}
    assert chart.transform == [alt.ImputeTransform(**kwds)]

    # joinaggregate transform
    chart = alt.Chart().transform_joinaggregate(min="min(x)", groupby=["key"])
    kwds = {
        "joinaggregate": [
            alt.JoinAggregateFieldDef(field="x", op="min", **{"as": "min"})
        ],
        "groupby": ["key"],
    }
    assert chart.transform == [alt.JoinAggregateTransform(**kwds)]

    # filter transform
    chart = alt.Chart().transform_filter("datum.a < 4")
    assert chart.transform == [alt.FilterTransform(filter="datum.a < 4")]

    # flatten transform
    chart = alt.Chart().transform_flatten(["A", "B"], ["X", "Y"])
    kwds = {"as": ["X", "Y"], "flatten": ["A", "B"]}
    assert chart.transform == [alt.FlattenTransform(**kwds)]

    # fold transform
    chart = alt.Chart().transform_fold(["A", "B", "C"], as_=["key", "val"])
    kwds = {"as": ["key", "val"], "fold": ["A", "B", "C"]}
    assert chart.transform == [alt.FoldTransform(**kwds)]

    # lookup transform
    lookup_data = alt.LookupData(alt.UrlData("foo.csv"), "id", ["rate"])
    chart = alt.Chart().transform_lookup(
        from_=lookup_data, as_="a", lookup="a", default="b"
    )
    kwds = {"from": lookup_data, "as": "a", "lookup": "a", "default": "b"}
    assert chart.transform == [alt.LookupTransform(**kwds)]

    # sample transform
    chart = alt.Chart().transform_sample()
    assert chart.transform == [alt.SampleTransform(1000)]

    # stack transform
    chart = alt.Chart().transform_stack("stacked", "x", groupby=["y"])
    assert chart.transform == [
        alt.StackTransform(stack="x", groupby=["y"], **{"as": "stacked"})
    ]

    # timeUnit transform
    chart = alt.Chart().transform_timeunit("foo", field="x", timeUnit="date")
    kwds = {"as": "foo", "field": "x", "timeUnit": "date"}
    assert chart.transform == [alt.TimeUnitTransform(**kwds)]

    # window transform
    chart = alt.Chart().transform_window(xsum="sum(x)", ymin="min(y)", frame=[None, 0])
    window = [
        alt.WindowFieldDef(**{"as": "xsum", "field": "x", "op": "sum"}),
        alt.WindowFieldDef(**{"as": "ymin", "field": "y", "op": "min"}),
    ]

    # kwargs don't maintain order in Python < 3.6, so window list can
    # be reversed
    assert chart.transform == [
        alt.WindowTransform(frame=[None, 0], window=window)
    ] or chart.transform == [alt.WindowTransform(frame=[None, 0], window=window[::-1])]


def test_filter_transform_selection_predicates():
    selector1 = alt.selection_interval(name="s1")
    selector2 = alt.selection_interval(name="s2")
    base = alt.Chart("data.txt").mark_point()

    chart = base.transform_filter(selector1)
    assert chart.to_dict()["transform"] == [{"filter": {"selection": "s1"}}]

    chart = base.transform_filter(~selector1)
    assert chart.to_dict()["transform"] == [{"filter": {"selection": {"not": "s1"}}}]

    chart = base.transform_filter(selector1 & selector2)
    assert chart.to_dict()["transform"] == [
        {"filter": {"selection": {"and": ["s1", "s2"]}}}
    ]

    chart = base.transform_filter(selector1 | selector2)
    assert chart.to_dict()["transform"] == [
        {"filter": {"selection": {"or": ["s1", "s2"]}}}
    ]

    chart = base.transform_filter(selector1 | ~selector2)
    assert chart.to_dict()["transform"] == [
        {"filter": {"selection": {"or": ["s1", {"not": "s2"}]}}}
    ]

    chart = base.transform_filter(~selector1 | ~selector2)
    assert chart.to_dict()["transform"] == [
        {"filter": {"selection": {"or": [{"not": "s1"}, {"not": "s2"}]}}}
    ]

    chart = base.transform_filter(~(selector1 & selector2))
    assert chart.to_dict()["transform"] == [
        {"filter": {"selection": {"not": {"and": ["s1", "s2"]}}}}
    ]


def test_resolve_methods():
    chart = alt.LayerChart().resolve_axis(x="shared", y="independent")
    assert chart.resolve == alt.Resolve(
        axis=alt.AxisResolveMap(x="shared", y="independent")
    )

    chart = alt.LayerChart().resolve_legend(color="shared", fill="independent")
    assert chart.resolve == alt.Resolve(
        legend=alt.LegendResolveMap(color="shared", fill="independent")
    )

    chart = alt.LayerChart().resolve_scale(x="shared", y="independent")
    assert chart.resolve == alt.Resolve(
        scale=alt.ScaleResolveMap(x="shared", y="independent")
    )


def test_layer_encodings():
    chart = alt.LayerChart().encode(x="column:Q")
    assert chart.encoding.x == alt.X(shorthand="column:Q")


def test_add_selection():
    selections = [
        alt.selection_interval(),
        alt.selection_single(),
        alt.selection_multi(),
    ]
    chart = (
        alt.Chart()
        .mark_point()
        .add_selection(selections[0])
        .add_selection(selections[1], selections[2])
    )
    expected = {s.name: s.selection for s in selections}
    assert chart.selection == expected


def test_repeat_add_selections():
    base = alt.Chart("data.csv").mark_point()
    selection = alt.selection_single()
    chart1 = base.add_selection(selection).repeat(list("ABC"))
    chart2 = base.repeat(list("ABC")).add_selection(selection)
    assert chart1.to_dict() == chart2.to_dict()


def test_facet_add_selections():
    base = alt.Chart("data.csv").mark_point()
    selection = alt.selection_single()
    chart1 = base.add_selection(selection).facet("val:Q")
    chart2 = base.facet("val:Q").add_selection(selection)
    assert chart1.to_dict() == chart2.to_dict()


def test_layer_add_selection():
    base = alt.Chart("data.csv").mark_point()
    selection = alt.selection_single()
    chart1 = alt.layer(base.add_selection(selection), base)
    chart2 = alt.layer(base, base).add_selection(selection)
    assert chart1.to_dict() == chart2.to_dict()


@pytest.mark.parametrize("charttype", [alt.concat, alt.hconcat, alt.vconcat])
def test_compound_add_selections(charttype):
    base = alt.Chart("data.csv").mark_point()
    selection = alt.selection_single()
    chart1 = charttype(base.add_selection(selection), base.add_selection(selection))
    chart2 = charttype(base, base).add_selection(selection)
    assert chart1.to_dict() == chart2.to_dict()


def test_selection_property():
    sel = alt.selection_interval()
    chart = alt.Chart("data.csv").mark_point().properties(selection=sel)

    assert list(chart["selection"].keys()) == [sel.name]


def test_LookupData():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    lookup = alt.LookupData(data=df, key="x")

    dct = lookup.to_dict()
    assert dct["key"] == "x"
    assert dct["data"] == {
        "values": [{"x": 1, "y": 4}, {"x": 2, "y": 5}, {"x": 3, "y": 6}]
    }


def test_themes():
    chart = alt.Chart("foo.txt").mark_point()
    active = alt.themes.active

    try:
        alt.themes.enable("default")
        assert chart.to_dict()["config"] == {
            "mark": {"tooltip": None},
            "view": {"width": 400, "height": 300},
        }

        alt.themes.enable("opaque")
        assert chart.to_dict()["config"] == {
            "background": "white",
            "mark": {"tooltip": None},
            "view": {"width": 400, "height": 300},
        }

        alt.themes.enable("none")
        assert "config" not in chart.to_dict()

    finally:
        # re-enable the original active theme
        alt.themes.enable(active)


def test_chart_from_dict():
    base = alt.Chart("data.csv").mark_point().encode(x="x:Q", y="y:Q")

    charts = [
        base,
        base + base,
        base | base,
        base & base,
        base.facet("c:N"),
        (base + base).facet(row="c:N", data="data.csv"),
        base.repeat(["c", "d"]),
        (base + base).repeat(row=["c", "d"]),
    ]

    for chart in charts:
        print(chart)
        chart_out = alt.Chart.from_dict(chart.to_dict())
        assert type(chart_out) is type(chart)

    # test that an invalid spec leads to a schema validation error
    with pytest.raises(jsonschema.ValidationError):
        alt.Chart.from_dict({"invalid": "spec"})


def test_consolidate_datasets(basic_chart):
    subchart1 = basic_chart
    subchart2 = basic_chart.copy()
    subchart2.data = basic_chart.data.copy()
    chart = subchart1 | subchart2

    with alt.data_transformers.enable(consolidate_datasets=True):
        dct_consolidated = chart.to_dict()

    with alt.data_transformers.enable(consolidate_datasets=False):
        dct_standard = chart.to_dict()

    assert "datasets" in dct_consolidated
    assert "datasets" not in dct_standard

    datasets = dct_consolidated["datasets"]

    # two dataset copies should be recognized as duplicates
    assert len(datasets) == 1

    # make sure data matches original & names are correct
    name, data = datasets.popitem()

    for spec in dct_standard["hconcat"]:
        assert spec["data"]["values"] == data

    for spec in dct_consolidated["hconcat"]:
        assert spec["data"] == {"name": name}


def test_consolidate_InlineData():
    data = alt.InlineData(
        values=[{"a": 1, "b": 1}, {"a": 2, "b": 2}], format={"type": "csv"}
    )
    chart = alt.Chart(data).mark_point()

    with alt.data_transformers.enable(consolidate_datasets=False):
        dct = chart.to_dict()
    assert dct["data"]["format"] == data.format
    assert dct["data"]["values"] == data.values

    with alt.data_transformers.enable(consolidate_datasets=True):
        dct = chart.to_dict()
    assert dct["data"]["format"] == data.format
    assert list(dct["datasets"].values())[0] == data.values

    data = alt.InlineData(values=[], name="runtime_data")
    chart = alt.Chart(data).mark_point()

    with alt.data_transformers.enable(consolidate_datasets=False):
        dct = chart.to_dict()
    assert dct["data"] == data.to_dict()

    with alt.data_transformers.enable(consolidate_datasets=True):
        dct = chart.to_dict()
    assert dct["data"] == data.to_dict()


def test_deprecated_encodings():
    base = alt.Chart("data.txt").mark_point()

    with pytest.warns(AltairDeprecationWarning) as record:
        chart1 = base.encode(strokeOpacity=alt.Strokeopacity("x:Q")).to_dict()
    assert "alt.StrokeOpacity" in record[0].message.args[0]
    chart2 = base.encode(strokeOpacity=alt.StrokeOpacity("x:Q")).to_dict()

    assert chart1 == chart2


def test_repeat():
    # wrapped repeat
    chart1 = (
        alt.Chart("data.csv")
        .mark_point()
        .encode(
            x=alt.X(alt.repeat(), type="quantitative"),
            y="y:Q",
        )
        .repeat(["A", "B", "C", "D"], columns=2)
    )

    dct1 = chart1.to_dict()

    assert dct1["repeat"] == ["A", "B", "C", "D"]
    assert dct1["columns"] == 2
    assert dct1["spec"]["encoding"]["x"]["field"] == {"repeat": "repeat"}

    # explicit row/col repeat
    chart2 = (
        alt.Chart("data.csv")
        .mark_point()
        .encode(
            x=alt.X(alt.repeat("row"), type="quantitative"),
            y=alt.Y(alt.repeat("column"), type="quantitative"),
        )
        .repeat(row=["A", "B", "C"], column=["C", "B", "A"])
    )

    dct2 = chart2.to_dict()

    assert dct2["repeat"] == {"row": ["A", "B", "C"], "column": ["C", "B", "A"]}
    assert "columns" not in dct2
    assert dct2["spec"]["encoding"]["x"]["field"] == {"repeat": "row"}
    assert dct2["spec"]["encoding"]["y"]["field"] == {"repeat": "column"}


def test_data_property():
    data = pd.DataFrame({"x": [1, 2, 3], "y": list("ABC")})
    chart1 = alt.Chart(data).mark_point()
    chart2 = alt.Chart().mark_point().properties(data=data)

    assert chart1.to_dict() == chart2.to_dict()


@pytest.mark.parametrize("method", ["layer", "hconcat", "vconcat", "concat"])
@pytest.mark.parametrize(
    "data", ["data.json", pd.DataFrame({"x": range(3), "y": list("abc")})]
)
def test_subcharts_with_same_data(method, data):
    func = getattr(alt, method)

    point = alt.Chart(data).mark_point().encode(x="x:Q", y="y:Q")
    line = point.mark_line()
    text = point.mark_text()

    chart1 = func(point, line, text)
    assert chart1.data is not alt.Undefined
    assert all(c.data is alt.Undefined for c in getattr(chart1, method))

    if method != "concat":
        op = OP_DICT[method]
        chart2 = op(op(point, line), text)
        assert chart2.data is not alt.Undefined
        assert all(c.data is alt.Undefined for c in getattr(chart2, method))


@pytest.mark.parametrize("method", ["layer", "hconcat", "vconcat", "concat"])
@pytest.mark.parametrize(
    "data", ["data.json", pd.DataFrame({"x": range(3), "y": list("abc")})]
)
def test_subcharts_different_data(method, data):
    func = getattr(alt, method)

    point = alt.Chart(data).mark_point().encode(x="x:Q", y="y:Q")
    otherdata = alt.Chart("data.csv").mark_point().encode(x="x:Q", y="y:Q")
    nodata = alt.Chart().mark_point().encode(x="x:Q", y="y:Q")

    chart1 = func(point, otherdata)
    assert chart1.data is alt.Undefined
    assert getattr(chart1, method)[0].data is data

    chart2 = func(point, nodata)
    assert chart2.data is alt.Undefined
    assert getattr(chart2, method)[0].data is data


def test_layer_facet(basic_chart):
    chart = (basic_chart + basic_chart).facet(row="row:Q")
    assert chart.data is not alt.Undefined
    assert chart.spec.data is alt.Undefined
    for layer in chart.spec.layer:
        assert layer.data is alt.Undefined

    dct = chart.to_dict()
    assert "data" in dct


def test_layer_errors():
    toplevel_chart = alt.Chart("data.txt").mark_point().configure_legend(columns=2)

    facet_chart1 = alt.Chart("data.txt").mark_point().encode(facet="row:Q")

    facet_chart2 = alt.Chart("data.txt").mark_point().facet("row:Q")

    repeat_chart = alt.Chart("data.txt").mark_point().repeat(["A", "B", "C"])

    simple_chart = alt.Chart("data.txt").mark_point()

    with pytest.raises(ValueError) as err:
        toplevel_chart + simple_chart
    assert str(err.value).startswith(
        'Objects with "config" attribute cannot be used within LayerChart.'
    )

    with pytest.raises(ValueError) as err:
        repeat_chart + simple_chart
    assert str(err.value) == "Repeat charts cannot be layered."

    with pytest.raises(ValueError) as err:
        facet_chart1 + simple_chart
    assert str(err.value) == "Faceted charts cannot be layered."

    with pytest.raises(ValueError) as err:
        alt.layer(simple_chart) + facet_chart2
    assert str(err.value) == "Faceted charts cannot be layered."


@pytest.mark.parametrize(
    "chart_type",
    ["layer", "hconcat", "vconcat", "concat", "facet", "facet_encoding", "repeat"],
)
def test_resolve(chart_type):
    chart = _make_chart_type(chart_type)
    chart = (
        chart.resolve_scale(
            x="independent",
        )
        .resolve_legend(color="independent")
        .resolve_axis(y="independent")
    )
    dct = chart.to_dict()
    assert dct["resolve"] == {
        "scale": {"x": "independent"},
        "legend": {"color": "independent"},
        "axis": {"y": "independent"},
    }


# TODO: test vconcat, hconcat, concat when schema allows them.
# This is blocked by https://github.com/vega/vega-lite/issues/5261
@pytest.mark.parametrize("chart_type", ["chart", "layer", "facet_encoding"])
@pytest.mark.parametrize("facet_arg", [None, "facet", "row", "column"])
def test_facet(chart_type, facet_arg):
    chart = _make_chart_type(chart_type)
    if facet_arg is None:
        chart = chart.facet("color:N", columns=2)
    else:
        chart = chart.facet(**{facet_arg: "color:N", "columns": 2})
    dct = chart.to_dict()

    assert "spec" in dct
    assert dct["columns"] == 2
    expected = {"field": "color", "type": "nominal"}
    if facet_arg is None or facet_arg == "facet":
        assert dct["facet"] == expected
    else:
        assert dct["facet"][facet_arg] == expected


def test_sequence():
    data = alt.sequence(100)
    assert data.to_dict() == {"sequence": {"start": 0, "stop": 100}}

    data = alt.sequence(5, 10)
    assert data.to_dict() == {"sequence": {"start": 5, "stop": 10}}

    data = alt.sequence(0, 1, 0.1, as_="x")
    assert data.to_dict() == {
        "sequence": {"start": 0, "stop": 1, "step": 0.1, "as": "x"}
    }


def test_graticule():
    data = alt.graticule()
    assert data.to_dict() == {"graticule": True}

    data = alt.graticule(step=[15, 15])
    assert data.to_dict() == {"graticule": {"step": [15, 15]}}


def test_sphere():
    data = alt.sphere()
    assert data.to_dict() == {"sphere": True}

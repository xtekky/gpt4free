"""Tests of various renderers"""

import json

import pytest

import altair.vegalite.v4 as alt


@pytest.fixture
def chart():
    return alt.Chart("data.csv").mark_point()


def test_html_renderer_embed_options(chart, renderer="html"):
    """Test that embed_options in renderer metadata are correctly manifest in html"""
    # Short of parsing the javascript, it's difficult to parse out the
    # actions. So we use string matching

    def assert_has_options(chart, **opts):
        html = chart._repr_mimebundle_(None, None)["text/html"]
        for key, val in opts.items():
            assert json.dumps({key: val})[1:-1] in html

    with alt.renderers.enable(renderer):
        assert_has_options(chart, mode="vega-lite")

        with alt.renderers.enable(embed_options=dict(actions={"export": True})):
            assert_has_options(chart, mode="vega-lite", actions={"export": True})

        with alt.renderers.set_embed_options(actions=True):
            assert_has_options(chart, mode="vega-lite", actions=True)


def test_mimetype_renderer_embed_options(chart, renderer="mimetype"):
    # check that metadata is passed appropriately
    mimetype = alt.display.VEGALITE_MIME_TYPE
    spec = chart.to_dict()
    with alt.renderers.enable(renderer):
        # Sanity check: no metadata specified
        bundle, metadata = chart._repr_mimebundle_(None, None)
        assert bundle[mimetype] == spec
        assert metadata == {}
        with alt.renderers.set_embed_options(actions=False):
            bundle, metadata = chart._repr_mimebundle_(None, None)
            assert set(bundle.keys()) == {mimetype, "text/plain"}
            assert bundle[mimetype] == spec
            assert metadata == {mimetype: {"embed_options": {"actions": False}}}


def test_json_renderer_embed_options(chart, renderer="json"):
    """Test that embed_options in renderer metadata are correctly manifest in html"""
    mimetype = "application/json"
    spec = chart.to_dict()
    with alt.renderers.enable(renderer):
        # Sanity check: no options specified
        bundle, metadata = chart._repr_mimebundle_(None, None)
        assert bundle[mimetype] == spec
        assert metadata == {}

        with alt.renderers.enable(option="foo"):
            bundle, metadata = chart._repr_mimebundle_(None, None)
            assert set(bundle.keys()) == {mimetype, "text/plain"}
            assert bundle[mimetype] == spec
            assert metadata == {mimetype: {"option": "foo"}}

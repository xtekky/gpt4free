import pytest

from ..html import spec_to_html


@pytest.fixture
def spec():
    return {
        "data": {"url": "data.json"},
        "mark": "point",
        "encoding": {
            "x": {"field": "x", "type": "quantitative"},
            "y": {"field": "y", "type": "quantitative"},
        },
    }


@pytest.mark.parametrize("requirejs", [True, False])
@pytest.mark.parametrize("fullhtml", [True, False])
def test_spec_to_html(requirejs, fullhtml, spec):
    # We can't test that the html actually renders, but we'll test aspects of
    # it to make certain that the keywords are respected.
    vegaembed_version = ("3.12",)
    vegalite_version = ("3.0",)
    vega_version = "4.0"

    html = spec_to_html(
        spec,
        mode="vega-lite",
        requirejs=requirejs,
        fullhtml=fullhtml,
        vegalite_version=vegalite_version,
        vegaembed_version=vegaembed_version,
        vega_version=vega_version,
    )
    html = html.strip()

    if fullhtml:
        assert html.startswith("<!DOCTYPE html>")
        assert html.endswith("</html>")
    else:
        assert html.startswith("<style>")
        assert html.endswith("</script>")

    if requirejs:
        assert "require(" in html
    else:
        assert "require(" not in html

    assert "vega-lite@{}".format(vegalite_version) in html
    assert "vega@{}".format(vega_version) in html
    assert "vega-embed@{}".format(vegaembed_version) in html

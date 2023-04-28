from contextlib import contextmanager

import pytest

import altair.vegalite.v4 as alt


@contextmanager
def check_render_options(**options):
    """
    Context manager that will assert that alt.renderers.options are equivalent
    to the given options in the IPython.display.display call
    """
    import IPython.display

    def check_options(obj):
        assert alt.renderers.options == options

    _display = IPython.display.display
    IPython.display.display = check_options
    try:
        yield
    finally:
        IPython.display.display = _display


def test_check_renderer_options():
    # this test should pass
    with check_render_options():
        from IPython.display import display

        display(None)

    # check that an error is appropriately raised if the test fails
    with pytest.raises(AssertionError):
        with check_render_options(foo="bar"):
            from IPython.display import display

            display(None)


def test_display_options():
    chart = alt.Chart("data.csv").mark_point().encode(x="foo:Q")

    # check that there are no options by default
    with check_render_options():
        chart.display()

    # check that display options are passed
    with check_render_options(embed_options={"tooltip": False, "renderer": "canvas"}):
        chart.display("canvas", tooltip=False)

    # check that above options do not persist
    with check_render_options():
        chart.display()

    # check that display options augment rather than overwrite pre-set options
    with alt.renderers.enable(embed_options={"tooltip": True, "renderer": "svg"}):
        with check_render_options(embed_options={"tooltip": True, "renderer": "svg"}):
            chart.display()

        with check_render_options(
            embed_options={"tooltip": True, "renderer": "canvas"}
        ):
            chart.display("canvas")

    # check that above options do not persist
    with check_render_options():
        chart.display()

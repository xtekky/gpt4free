import os

from ...utils.mimebundle import spec_to_mimebundle
from ...utils.deprecation import deprecated
from ..display import Displayable
from ..display import default_renderer_base
from ..display import json_renderer_base
from ..display import RendererRegistry
from ..display import HTMLRenderer

from .schema import SCHEMA_VERSION

VEGALITE_VERSION = SCHEMA_VERSION.lstrip("v")
VEGA_VERSION = "5"
VEGAEMBED_VERSION = "5"


# ==============================================================================
# VegaLite v3 renderer logic
# ==============================================================================


# The MIME type for Vega-Lite 3.x releases.
VEGALITE_MIME_TYPE = "application/vnd.vegalite.v3+json"  # type: str

# The entry point group that can be used by other packages to declare other
# renderers that will be auto-detected. Explicit registration is also
# allowed by the PluginRegistery API.
ENTRY_POINT_GROUP = "altair.vegalite.v3.renderer"  # type: str

# The display message when rendering fails
DEFAULT_DISPLAY = """\
<VegaLite 3 object>

If you see this message, it means the renderer has not been properly enabled
for the frontend that you are using. For more information, see
https://altair-viz.github.io/user_guide/troubleshooting.html
"""

renderers = RendererRegistry(entry_point_group=ENTRY_POINT_GROUP)

here = os.path.dirname(os.path.realpath(__file__))


def default_renderer(spec, **metadata):
    return default_renderer_base(spec, VEGALITE_MIME_TYPE, DEFAULT_DISPLAY, **metadata)


def json_renderer(spec, **metadata):
    return json_renderer_base(spec, DEFAULT_DISPLAY, **metadata)


def png_renderer(spec, **metadata):
    return spec_to_mimebundle(
        spec,
        format="png",
        mode="vega-lite",
        vega_version=VEGA_VERSION,
        vegaembed_version=VEGAEMBED_VERSION,
        vegalite_version=VEGALITE_VERSION,
        **metadata,
    )


def svg_renderer(spec, **metadata):
    return spec_to_mimebundle(
        spec,
        format="svg",
        mode="vega-lite",
        vega_version=VEGA_VERSION,
        vegaembed_version=VEGAEMBED_VERSION,
        vegalite_version=VEGALITE_VERSION,
        **metadata,
    )


colab_renderer = HTMLRenderer(
    mode="vega-lite",
    fullhtml=True,
    requirejs=False,
    output_div="altair-viz",
    vega_version=VEGA_VERSION,
    vegaembed_version=VEGAEMBED_VERSION,
    vegalite_version=VEGALITE_VERSION,
)

zeppelin_renderer = HTMLRenderer(
    mode="vega-lite",
    fullhtml=True,
    requirejs=False,
    vega_version=VEGA_VERSION,
    vegaembed_version=VEGAEMBED_VERSION,
    vegalite_version=VEGALITE_VERSION,
)

kaggle_renderer = HTMLRenderer(
    mode="vega-lite",
    fullhtml=False,
    requirejs=True,
    vega_version=VEGA_VERSION,
    vegaembed_version=VEGAEMBED_VERSION,
    vegalite_version=VEGALITE_VERSION,
)

html_renderer = HTMLRenderer(
    mode="vega-lite",
    template="universal",
    vega_version=VEGA_VERSION,
    vegaembed_version=VEGAEMBED_VERSION,
    vegalite_version=VEGALITE_VERSION,
)

renderers.register("default", default_renderer)
renderers.register("html", html_renderer)
renderers.register("jupyterlab", default_renderer)
renderers.register("nteract", default_renderer)
renderers.register("colab", colab_renderer)
renderers.register("kaggle", kaggle_renderer)
renderers.register("json", json_renderer)
renderers.register("png", png_renderer)
renderers.register("svg", svg_renderer)
renderers.register("zeppelin", zeppelin_renderer)
renderers.enable("default")


class VegaLite(Displayable):
    """An IPython/Jupyter display class for rendering VegaLite 3."""

    renderers = renderers
    schema_path = (__name__, "schema/vega-lite-schema.json")


@deprecated(
    "Rendering VegaLite 3 specifications is deprecated and will be removed in Altair 5. "
    "Use `import altair as alt` instead of `import altair.vegalite.v3 as alt`."
)
def vegalite(spec, validate=True):
    """Render and optionally validate a VegaLite 3 spec.

    This will use the currently enabled renderer to render the spec.

    Parameters
    ==========
    spec: dict
        A fully compliant VegaLite 3 spec, with the data portion fully processed.
    validate: bool
        Should the spec be validated against the VegaLite 3 schema?
    """
    from IPython.display import display

    display(VegaLite(spec, validate=validate))

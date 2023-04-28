import os

from ...utils.mimebundle import spec_to_mimebundle
from ..display import Displayable
from ..display import default_renderer_base
from ..display import json_renderer_base
from ..display import RendererRegistry
from ..display import HTMLRenderer

from .schema import SCHEMA_VERSION

VEGALITE_VERSION = SCHEMA_VERSION.lstrip("v")
VEGA_VERSION = "5"
VEGAEMBED_VERSION = "6"


# ==============================================================================
# VegaLite v4 renderer logic
# ==============================================================================


# The MIME type for Vega-Lite 4.x releases.
VEGALITE_MIME_TYPE = "application/vnd.vegalite.v4+json"  # type: str

# The entry point group that can be used by other packages to declare other
# renderers that will be auto-detected. Explicit registration is also
# allowed by the PluginRegistery API.
ENTRY_POINT_GROUP = "altair.vegalite.v4.renderer"  # type: str

# The display message when rendering fails
DEFAULT_DISPLAY = """\
<VegaLite 4 object>

If you see this message, it means the renderer has not been properly enabled
for the frontend that you are using. For more information, see
https://altair-viz.github.io/user_guide/troubleshooting.html
"""

renderers = RendererRegistry(entry_point_group=ENTRY_POINT_GROUP)

here = os.path.dirname(os.path.realpath(__file__))


def mimetype_renderer(spec, **metadata):
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


html_renderer = HTMLRenderer(
    mode="vega-lite",
    template="universal",
    vega_version=VEGA_VERSION,
    vegaembed_version=VEGAEMBED_VERSION,
    vegalite_version=VEGALITE_VERSION,
)

renderers.register("default", html_renderer)
renderers.register("html", html_renderer)
renderers.register("colab", html_renderer)
renderers.register("kaggle", html_renderer)
renderers.register("zeppelin", html_renderer)
renderers.register("mimetype", mimetype_renderer)
renderers.register("jupyterlab", mimetype_renderer)
renderers.register("nteract", mimetype_renderer)
renderers.register("json", json_renderer)
renderers.register("png", png_renderer)
renderers.register("svg", svg_renderer)
renderers.enable("default")


class VegaLite(Displayable):
    """An IPython/Jupyter display class for rendering VegaLite 4."""

    renderers = renderers
    schema_path = (__name__, "schema/vega-lite-schema.json")


def vegalite(spec, validate=True):
    """Render and optionally validate a VegaLite 4 spec.

    This will use the currently enabled renderer to render the spec.

    Parameters
    ==========
    spec: dict
        A fully compliant VegaLite 4 spec, with the data portion fully processed.
    validate: bool
        Should the spec be validated against the VegaLite 4 schema?
    """
    from IPython.display import display

    display(VegaLite(spec, validate=validate))

import os

from ...utils import PluginRegistry
from ...utils.deprecation import deprecated
from ..display import Displayable
from ..display import default_renderer_base
from ..display import json_renderer_base
from ..display import RendererType
from ..display import HTMLRenderer

from .schema import SCHEMA_VERSION

VEGA_VERSION = SCHEMA_VERSION.lstrip("v")
VEGAEMBED_VERSION = "3"


# ==============================================================================
# Vega 5 renderer logic
# ==============================================================================


# The MIME type for Vega 5 releases.
VEGA_MIME_TYPE = "application/vnd.vega.v5+json"  # type: str

# The entry point group that can be used by other packages to declare other
# renderers that will be auto-detected. Explicit registration is also
# allowed by the PluginRegistery API.
ENTRY_POINT_GROUP = "altair.vega.v5.renderer"  # type: str

# The display message when rendering fails
DEFAULT_DISPLAY = """\
<Vega 5 object>

If you see this message, it means the renderer has not been properly enabled
for the frontend that you are using. For more information, see
https://altair-viz.github.io/user_guide/troubleshooting.html
"""

renderers = PluginRegistry[RendererType](entry_point_group=ENTRY_POINT_GROUP)


here = os.path.dirname(os.path.realpath(__file__))


def default_renderer(spec):
    return default_renderer_base(spec, VEGA_MIME_TYPE, DEFAULT_DISPLAY)


def json_renderer(spec):
    return json_renderer_base(spec, DEFAULT_DISPLAY)


colab_renderer = HTMLRenderer(
    mode="vega",
    fullhtml=True,
    requirejs=False,
    output_div="altair-viz",
    vega_version=VEGA_VERSION,
    vegaembed_version=VEGAEMBED_VERSION,
)


kaggle_renderer = HTMLRenderer(
    mode="vega",
    fullhtml=False,
    requirejs=True,
    vega_version=VEGA_VERSION,
    vegaembed_version=VEGAEMBED_VERSION,
)


html_renderer = HTMLRenderer(
    mode="vega",
    template="universal",
    vega_version=VEGA_VERSION,
    vegaembed_version=VEGAEMBED_VERSION,
)


renderers.register("default", default_renderer)
renderers.register("html", html_renderer)
renderers.register("jupyterlab", default_renderer)
renderers.register("nteract", default_renderer)
renderers.register("colab", colab_renderer)
renderers.register("kaggle", kaggle_renderer)
renderers.register("json", json_renderer)
renderers.enable("default")


class Vega(Displayable):
    """An IPython/Jupyter display class for rendering Vega 5."""

    renderers = renderers
    schema_path = (__name__, "schema/vega-schema.json")


@deprecated(
    "Rendering Vega specifications is deprecated and will be removed in Altair 5."
)
def vega(spec, validate=True):
    """Render and optionally validate a Vega 5 spec.

    This will use the currently enabled renderer to render the spec.

    Parameters
    ==========
    spec: dict
        A fully compliant Vega 5 spec, with the data portion fully processed.
    validate: bool
        Should the spec be validated against the Vega 5 schema?
    """
    from IPython.display import display

    display(Vega(spec, validate=validate))

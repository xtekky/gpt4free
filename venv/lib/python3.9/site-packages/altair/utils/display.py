import json
import pkgutil
import textwrap
from typing import Callable, Dict
import uuid

from .plugin_registry import PluginRegistry
from .mimebundle import spec_to_mimebundle
from .schemapi import validate_jsonschema


# ==============================================================================
# Renderer registry
# ==============================================================================
MimeBundleType = Dict[str, object]
RendererType = Callable[..., MimeBundleType]


class RendererRegistry(PluginRegistry[RendererType]):
    entrypoint_err_messages = {
        "notebook": textwrap.dedent(
            """
            To use the 'notebook' renderer, you must install the vega package
            and the associated Jupyter extension.
            See https://altair-viz.github.io/getting_started/installation.html
            for more information.
            """
        ),
        "altair_viewer": textwrap.dedent(
            """
            To use the 'altair_viewer' renderer, you must install the altair_viewer
            package; see http://github.com/altair-viz/altair_viewer/
            for more information.
            """
        ),
    }

    def set_embed_options(
        self,
        defaultStyle=None,
        renderer=None,
        width=None,
        height=None,
        padding=None,
        scaleFactor=None,
        actions=None,
        **kwargs,
    ):
        """Set options for embeddings of Vega & Vega-Lite charts.

        Options are fully documented at https://github.com/vega/vega-embed.
        Similar to the `enable()` method, this can be used as either
        a persistent global switch, or as a temporary local setting using
        a context manager (i.e. a `with` statement).

        Parameters
        ----------
        defaultStyle : bool or string
            Specify a default stylesheet for embed actions.
        renderer : string
            The renderer to use for the view. One of "canvas" (default) or "svg"
        width : integer
            The view width in pixels
        height : integer
            The view height in pixels
        padding : integer
            The view padding in pixels
        scaleFactor : number
            The number by which to multiply the width and height (default 1)
            of an exported PNG or SVG image.
        actions : bool or dict
            Determines if action links ("Export as PNG/SVG", "View Source",
            "View Vega" (only for Vega-Lite), "Open in Vega Editor") are
            included with the embedded view. If the value is true, all action
            links will be shown and none if the value is false. This property
            can take a key-value mapping object that maps keys (export, source,
            compiled, editor) to boolean values for determining if
            each action link should be shown.
        **kwargs :
            Additional options are passed directly to embed options.
        """
        options = {
            "defaultStyle": defaultStyle,
            "renderer": renderer,
            "width": width,
            "height": height,
            "padding": padding,
            "scaleFactor": scaleFactor,
            "actions": actions,
        }
        kwargs.update({key: val for key, val in options.items() if val is not None})
        return self.enable(None, embed_options=kwargs)


# ==============================================================================
# VegaLite v1/v2 renderer logic
# ==============================================================================


class Displayable(object):
    """A base display class for VegaLite v1/v2.

    This class takes a VegaLite v1/v2 spec and does the following:

    1. Optionally validates the spec against a schema.
    2. Uses the RendererPlugin to grab a renderer and call it when the
       IPython/Jupyter display method (_repr_mimebundle_) is called.

    The spec passed to this class must be fully schema compliant and already
    have the data portion of the spec fully processed and ready to serialize.
    In practice, this means, the data portion of the spec should have been passed
    through appropriate data model transformers.
    """

    renderers = None
    schema_path = ("altair", "")

    def __init__(self, spec, validate=False):
        # type: (dict, bool) -> None
        self.spec = spec
        self.validate = validate
        self._validate()

    def _validate(self):
        # type: () -> None
        """Validate the spec against the schema."""
        schema_dict = json.loads(pkgutil.get_data(*self.schema_path).decode("utf-8"))
        validate_jsonschema(self.spec, schema_dict)

    def _repr_mimebundle_(self, include=None, exclude=None):
        """Return a MIME bundle for display in Jupyter frontends."""
        if self.renderers is not None:
            return self.renderers.get()(self.spec)
        else:
            return {}


def default_renderer_base(spec, mime_type, str_repr, **options):
    """A default renderer for Vega or VegaLite that works for modern frontends.

    This renderer works with modern frontends (JupyterLab, nteract) that know
    how to render the custom VegaLite MIME type listed above.
    """
    assert isinstance(spec, dict)
    bundle = {}
    metadata = {}

    bundle[mime_type] = spec
    bundle["text/plain"] = str_repr
    if options:
        metadata[mime_type] = options
    return bundle, metadata


def json_renderer_base(spec, str_repr, **options):
    """A renderer that returns a MIME type of application/json.

    In JupyterLab/nteract this is rendered as a nice JSON tree.
    """
    return default_renderer_base(
        spec, mime_type="application/json", str_repr=str_repr, **options
    )


class HTMLRenderer(object):
    """Object to render charts as HTML, with a unique output div each time"""

    def __init__(self, output_div="altair-viz-{}", **kwargs):
        self._output_div = output_div
        self.kwargs = kwargs

    @property
    def output_div(self):
        return self._output_div.format(uuid.uuid4().hex)

    def __call__(self, spec, **metadata):
        kwargs = self.kwargs.copy()
        kwargs.update(metadata)
        return spec_to_mimebundle(
            spec, format="html", output_div=self.output_div, **kwargs
        )

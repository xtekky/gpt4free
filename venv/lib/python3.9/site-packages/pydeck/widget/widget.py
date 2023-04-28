from ast import literal_eval
import json

from ipywidgets import register, CallbackDispatcher, DOMWidget
from traitlets import Any, Bool, Int, Unicode

from ..data_utils.binary_transfer import data_buffer_serialization
from ._frontend import module_name, module_version
from .debounce import debounce


def store_selection(widget_instance, payload):
    """Callback for storing data on click"""
    try:
        if payload.get("data") and payload["data"].get("object"):
            datum = payload["data"]["object"]
            widget_instance.selected_data.append(datum)
        else:
            widget_instance.selected_data = []
    except Exception as e:
        widget_instance.handler_exception = e


@register
class DeckGLWidget(DOMWidget):
    """
    Jupyter environment widget that takes JSON and
    renders a deck.gl visualization based on provided properties.

    You may set a Mapbox API key as an environment variable to use Mapbox maps in your visualization

    Attributes
    ----------
        json_input : str, default ''
            JSON as a string meant for reading into deck.gl JSON API
        mapbox_key : str, default ''
            API key for Mapbox map tiles
        height : int, default 500
            Height of Jupyter notebook cell, in pixels
        width : int or str, default "100%"
            Width of Jupyter notebook cell, in pixels or, if a string, a CSS width
        tooltip : bool or dict of {str: str}, default True
            See the ``Deck`` constructor.
        google_maps_key : str, default ''
            API key for Google Maps
        selected_data : list of dict, default []
            Data selected on click, if the pydeck Jupyter widget is enabled for server use
    """

    _model_name = Unicode("JupyterTransportModel").tag(sync=True)
    _model_module = Unicode(module_name).tag(sync=True)
    _model_module_version = Unicode(module_version).tag(sync=True)
    _view_name = Unicode("JupyterTransportView").tag(sync=True)
    _view_module = Unicode(module_name).tag(sync=True)
    _view_module_version = Unicode(module_version).tag(sync=True)

    carto_key = Unicode("", allow_none=True).tag(sync=True)
    mapbox_key = Unicode("", allow_none=True).tag(sync=True)
    google_maps_key = Unicode("", allow_none=True).tag(sync=True)

    json_input = Unicode("").tag(sync=True)
    data_buffer = Any(default_value=None, allow_none=True).tag(sync=True, **data_buffer_serialization)
    custom_libraries = Any(allow_none=True).tag(sync=True)
    configuration = Any(allow_none=True).tag(sync=True)
    tooltip = Any(True).tag(sync=True)
    height = Int(500).tag(sync=True)
    width = Any("100%").tag(sync=True)

    def __init__(self, **kwargs):
        super(DeckGLWidget, self).__init__(**kwargs)
        self._hover_handlers = CallbackDispatcher()
        self._click_handlers = CallbackDispatcher()
        self._resize_handlers = CallbackDispatcher()
        self._view_state_handlers = CallbackDispatcher()
        self._drag_handlers = CallbackDispatcher()
        self._drag_start_handlers = CallbackDispatcher()
        self._drag_end_handlers = CallbackDispatcher()
        self.on_msg(self._handle_custom_msgs)

        self.handler_exception = None
        self.selected_data = []
        self.on_click(store_selection)

    def on_hover(self, callback, remove=False):
        self._hover_handlers.register_callback(callback, remove=remove)

    def on_resize(self, callback, remove=False):
        self._resize_handlers.register_callback(callback, remove=remove)

    def on_view_state_change(self, callback, debounce_seconds=0.2, remove=False):
        callback = debounce(debounce_seconds)(callback) if debounce_seconds > 0 else callback
        self._view_state_handlers.register_callback(callback, remove=remove)

    def on_click(self, callback, remove=False):
        self._click_handlers.register_callback(callback, remove=remove)

    def on_drag_start(self, callback, remove=False):
        self._drag_start_handlers.register_callback(callback, remove=remove)

    def on_drag(self, callback, remove=False):
        self._drag_handlers.register_callback(callback, remove=remove)

    def on_drag_end(self, callback, remove=False):
        self._drag_end_handlers.register_callback(callback, remove=remove)

    def _handle_custom_msgs(self, _, content, buffers=None):
        content = json.loads(content)
        event_type = content.get("type", "")
        if event_type == "deck-hover-event":
            self._hover_handlers(self, content)
        elif event_type == "deck-resize-event":
            self._resize_handlers(self, content)
        elif event_type == "deck-view-state-change-event":
            self._view_state_handlers(self, content)
        elif event_type == "deck-click-event":
            self._click_handlers(self, content)
        elif event_type == "deck-drag-start-event":
            self._drag_start_handlers(self, content)
        elif event_type == "deck-drag-event":
            self._drag_handlers(self, content)
        elif event_type == "deck-drag-end-event":
            self._drag_end_handlers(self, content)

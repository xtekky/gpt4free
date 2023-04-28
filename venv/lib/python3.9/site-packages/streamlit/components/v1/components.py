# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import json
import os
import threading
from typing import Any, Dict, Optional, Type, Union

import streamlit
from streamlit import type_util, util
from streamlit.elements.form import current_form_id
from streamlit.errors import StreamlitAPIException
from streamlit.logger import get_logger
from streamlit.proto.Components_pb2 import ArrowTable as ArrowTableProto
from streamlit.proto.Components_pb2 import SpecialArg
from streamlit.proto.Element_pb2 import Element
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.runtime.state import NoValue, register_widget
from streamlit.type_util import to_bytes

LOGGER = get_logger(__name__)


class MarshallComponentException(StreamlitAPIException):
    """Class for exceptions generated during custom component marshalling."""

    pass


class CustomComponent:
    """A Custom Component declaration."""

    def __init__(
        self,
        name: str,
        path: Optional[str] = None,
        url: Optional[str] = None,
    ):
        if (path is None and url is None) or (path is not None and url is not None):
            raise StreamlitAPIException(
                "Either 'path' or 'url' must be set, but not both."
            )

        self.name = name
        self.path = path
        self.url = url

    def __repr__(self) -> str:
        return util.repr_(self)

    @property
    def abspath(self) -> Optional[str]:
        """The absolute path that the component is served from."""
        if self.path is None:
            return None
        return os.path.abspath(self.path)

    def __call__(
        self,
        *args,
        default: Any = None,
        key: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """An alias for create_instance."""
        return self.create_instance(*args, default=default, key=key, **kwargs)

    @gather_metrics("create_instance")
    def create_instance(
        self,
        *args,
        default: Any = None,
        key: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Create a new instance of the component.

        Parameters
        ----------
        *args
            Must be empty; all args must be named. (This parameter exists to
            enforce correct use of the function.)
        default: any or None
            The default return value for the component. This is returned when
            the component's frontend hasn't yet specified a value with
            `setComponentValue`.
        key: str or None
            If not None, this is the user key we use to generate the
            component's "widget ID".
        **kwargs
            Keyword args to pass to the component.

        Returns
        -------
        any or None
            The component's widget value.

        """
        if len(args) > 0:
            raise MarshallComponentException(f"Argument '{args[0]}' needs a label")

        try:
            import pyarrow

            from streamlit.components.v1 import component_arrow
        except ImportError:
            raise StreamlitAPIException(
                """To use Custom Components in Streamlit, you need to install
PyArrow. To do so locally:

`pip install pyarrow`

And if you're using Streamlit Cloud, add "pyarrow" to your requirements.txt."""
            )

        # In addition to the custom kwargs passed to the component, we also
        # send the special 'default' and 'key' params to the component
        # frontend.
        all_args = dict(kwargs, **{"default": default, "key": key})

        json_args = {}
        special_args = []
        for arg_name, arg_val in all_args.items():
            if type_util.is_bytes_like(arg_val):
                bytes_arg = SpecialArg()
                bytes_arg.key = arg_name
                bytes_arg.bytes = to_bytes(arg_val)
                special_args.append(bytes_arg)
            elif type_util.is_dataframe_like(arg_val):
                dataframe_arg = SpecialArg()
                dataframe_arg.key = arg_name
                component_arrow.marshall(dataframe_arg.arrow_dataframe.data, arg_val)
                special_args.append(dataframe_arg)
            else:
                json_args[arg_name] = arg_val

        try:
            serialized_json_args = json.dumps(json_args)
        except Exception as ex:
            raise MarshallComponentException(
                "Could not convert component args to JSON", ex
            )

        def marshall_component(dg, element: Element) -> Union[Any, Type[NoValue]]:
            element.component_instance.component_name = self.name
            element.component_instance.form_id = current_form_id(dg)
            if self.url is not None:
                element.component_instance.url = self.url

            # Normally, a widget's element_hash (which determines
            # its identity across multiple runs of an app) is computed
            # by hashing the entirety of its protobuf. This means that,
            # if any of the arguments to the widget are changed, Streamlit
            # considers it a new widget instance and it loses its previous
            # state.
            #
            # However! If a *component* has a `key` argument, then the
            # component's hash identity is determined by entirely by
            # `component_name + url + key`. This means that, when `key`
            # exists, the component will maintain its identity even when its
            # other arguments change, and the component's iframe won't be
            # remounted on the frontend.
            #
            # So: if `key` is None, we marshall the element's arguments
            # *before* computing its widget_ui_value (which creates its hash).
            # If `key` is not None, we marshall the arguments *after*.

            def marshall_element_args():
                element.component_instance.json_args = serialized_json_args
                element.component_instance.special_args.extend(special_args)

            if key is None:
                marshall_element_args()

            def deserialize_component(ui_value, widget_id=""):
                # ui_value is an object from json, an ArrowTable proto, or a bytearray
                return ui_value

            ctx = get_script_run_ctx()
            component_state = register_widget(
                element_type="component_instance",
                element_proto=element.component_instance,
                user_key=key,
                widget_func_name=self.name,
                deserializer=deserialize_component,
                serializer=lambda x: x,
                ctx=ctx,
            )
            widget_value = component_state.value

            if key is not None:
                marshall_element_args()

            if widget_value is None:
                widget_value = default
            elif isinstance(widget_value, ArrowTableProto):
                widget_value = component_arrow.arrow_proto_to_dataframe(widget_value)

            # widget_value will be either None or whatever the component's most
            # recent setWidgetValue value is. We coerce None -> NoValue,
            # because that's what DeltaGenerator._enqueue expects.
            return widget_value if widget_value is not None else NoValue

        # We currently only support writing to st._main, but this will change
        # when we settle on an improved API in a post-layout world.
        dg = streamlit._main

        element = Element()
        return_value = marshall_component(dg, element)
        result = dg._enqueue(
            "component_instance", element.component_instance, return_value
        )

        return result

    def __eq__(self, other) -> bool:
        """Equality operator."""
        return (
            isinstance(other, CustomComponent)
            and self.name == other.name
            and self.path == other.path
            and self.url == other.url
        )

    def __ne__(self, other) -> bool:
        """Inequality operator."""
        return not self == other

    def __str__(self) -> str:
        return f"'{self.name}': {self.path if self.path is not None else self.url}"


def declare_component(
    name: str,
    path: Optional[str] = None,
    url: Optional[str] = None,
) -> CustomComponent:
    """Create and register a custom component.

    Parameters
    ----------
    name: str
        A short, descriptive name for the component. Like, "slider".
    path: str or None
        The path to serve the component's frontend files from. Either
        `path` or `url` must be specified, but not both.
    url: str or None
        The URL that the component is served from. Either `path` or `url`
        must be specified, but not both.

    Returns
    -------
    CustomComponent
        A CustomComponent that can be called like a function.
        Calling the component will create a new instance of the component
        in the Streamlit app.

    """

    # Get our stack frame.
    current_frame = inspect.currentframe()
    assert current_frame is not None

    # Get the stack frame of our calling function.
    caller_frame = current_frame.f_back
    assert caller_frame is not None

    # Get the caller's module name. `__name__` gives us the module's
    # fully-qualified name, which includes its package.
    module = inspect.getmodule(caller_frame)
    assert module is not None
    module_name = module.__name__

    # If the caller was the main module that was executed (that is, if the
    # user executed `python my_component.py`), then this name will be
    # "__main__" instead of the actual package name. In this case, we use
    # the main module's filename, sans `.py` extension, as the component name.
    if module_name == "__main__":
        file_path = inspect.getfile(caller_frame)
        filename = os.path.basename(file_path)
        module_name, _ = os.path.splitext(filename)

    # Build the component name.
    component_name = f"{module_name}.{name}"

    # Create our component object, and register it.
    component = CustomComponent(name=component_name, path=path, url=url)
    ComponentRegistry.instance().register_component(component)

    return component


class ComponentRegistry:
    _instance_lock: threading.Lock = threading.Lock()
    _instance: Optional["ComponentRegistry"] = None

    @classmethod
    def instance(cls) -> "ComponentRegistry":
        """Returns the singleton ComponentRegistry"""
        # We use a double-checked locking optimization to avoid the overhead
        # of acquiring the lock in the common case:
        # https://en.wikipedia.org/wiki/Double-checked_locking
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = ComponentRegistry()
        return cls._instance

    def __init__(self):
        self._components: Dict[str, CustomComponent] = {}
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return util.repr_(self)

    def register_component(self, component: CustomComponent) -> None:
        """Register a CustomComponent.

        Parameters
        ----------
        component : CustomComponent
            The component to register.
        """

        # Validate the component's path
        abspath = component.abspath
        if abspath is not None and not os.path.isdir(abspath):
            raise StreamlitAPIException(f"No such component directory: '{abspath}'")

        with self._lock:
            existing = self._components.get(component.name)
            self._components[component.name] = component

        if existing is not None and component != existing:
            LOGGER.warning(
                "%s overriding previously-registered %s",
                component,
                existing,
            )

        LOGGER.debug("Registered component %s", component)

    def get_component_path(self, name: str) -> Optional[str]:
        """Return the filesystem path for the component with the given name.

        If no such component is registered, or if the component exists but is
        being served from a URL, return None instead.
        """
        component = self._components.get(name, None)
        return component.abspath if component is not None else None

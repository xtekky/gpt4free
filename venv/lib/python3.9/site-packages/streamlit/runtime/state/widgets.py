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

import textwrap
from types import MappingProxyType
from typing import TYPE_CHECKING, Dict, Mapping, Optional

from typing_extensions import Final, TypeAlias

from streamlit.errors import DuplicateWidgetID
from streamlit.proto.WidgetStates_pb2 import WidgetState, WidgetStates
from streamlit.runtime.state.common import (
    RegisterWidgetResult,
    T,
    WidgetArgs,
    WidgetCallback,
    WidgetDeserializer,
    WidgetKwargs,
    WidgetMetadata,
    WidgetProto,
    WidgetSerializer,
    compute_widget_id,
    user_key_from_widget_id,
)
from streamlit.type_util import ValueFieldName

if TYPE_CHECKING:
    from streamlit.runtime.scriptrunner import ScriptRunContext

ElementType: TypeAlias = str

# NOTE: We use this table to start with a best-effort guess for the value_type
# of each widget. Once we actually receive a proto for a widget from the
# frontend, the guess is updated to be the correct type. Unfortunately, we're
# not able to always rely on the proto as the type may be needed earlier.
# Thankfully, in these cases (when value_type == "trigger_value"), the static
# table here being slightly inaccurate should never pose a problem.
ELEMENT_TYPE_TO_VALUE_TYPE: Final[
    Mapping[ElementType, ValueFieldName]
] = MappingProxyType(
    {
        "button": "trigger_value",
        "download_button": "trigger_value",
        "checkbox": "bool_value",
        "camera_input": "file_uploader_state_value",
        "color_picker": "string_value",
        "date_input": "string_array_value",
        "file_uploader": "file_uploader_state_value",
        "multiselect": "int_array_value",
        "number_input": "double_value",
        "radio": "int_value",
        "selectbox": "int_value",
        "slider": "double_array_value",
        "text_area": "string_value",
        "text_input": "string_value",
        "time_input": "string_value",
        "component_instance": "json_value",
        "data_editor": "string_value",
    }
)


class NoValue:
    """Return this from DeltaGenerator.foo_widget() when you want the st.foo_widget()
    call to return None. This is needed because `DeltaGenerator._enqueue`
    replaces `None` with a `DeltaGenerator` (for use in non-widget elements).
    """

    pass


def register_widget(
    element_type: ElementType,
    element_proto: WidgetProto,
    deserializer: WidgetDeserializer[T],
    serializer: WidgetSerializer[T],
    ctx: Optional["ScriptRunContext"],
    user_key: Optional[str] = None,
    widget_func_name: Optional[str] = None,
    on_change_handler: Optional[WidgetCallback] = None,
    args: Optional[WidgetArgs] = None,
    kwargs: Optional[WidgetKwargs] = None,
) -> RegisterWidgetResult[T]:
    """Register a widget with Streamlit, and return its current value.
    NOTE: This function should be called after the proto has been filled.

    Parameters
    ----------
    element_type : ElementType
        The type of the element as stored in proto.
    element_proto : WidgetProto
        The proto of the specified type (e.g. Button/Multiselect/Slider proto)
    deserializer : WidgetDeserializer[T]
        Called to convert a widget's protobuf value to the value returned by
        its st.<widget_name> function.
    serializer : WidgetSerializer[T]
        Called to convert a widget's value to its protobuf representation.
    ctx : Optional[ScriptRunContext]
        Used to ensure uniqueness of widget IDs, and to look up widget values.
    user_key : Optional[str]
        Optional user-specified string to use as the widget ID.
        If this is None, we'll generate an ID by hashing the element.
    widget_func_name : Optional[str]
        The widget's DeltaGenerator function name, if it's different from
        its element_type. Custom components are a special case: they all have
        the element_type "component_instance", but are instantiated with
        dynamically-named functions.
    on_change_handler : Optional[WidgetCallback]
        An optional callback invoked when the widget's value changes.
    args : Optional[WidgetArgs]
        args to pass to on_change_handler when invoked
    kwargs : Optional[WidgetKwargs]
        kwargs to pass to on_change_handler when invoked

    Returns
    -------
    register_widget_result : RegisterWidgetResult[T]
        Provides information on which value to return to the widget caller,
        and whether the UI needs updating.

        - Unhappy path:
            - Our ScriptRunContext doesn't exist (meaning that we're running
            as a "bare script" outside streamlit).
            - We are disconnected from the SessionState instance.
            In both cases we'll return a fallback RegisterWidgetResult[T].
        - Happy path:
            - The widget has already been registered on a previous run but the
            user hasn't interacted with it on the client. The widget will have
            the default value it was first created with. We then return a
            RegisterWidgetResult[T], containing this value.
            - The widget has already been registered and the user *has*
            interacted with it. The widget will have that most recent
            user-specified value. We then return a RegisterWidgetResult[T],
            containing this value.

        For both paths a widget return value is provided, allowing the widgets
        to be used in a non-streamlit setting.
    """
    widget_id = compute_widget_id(element_type, element_proto, user_key)
    element_proto.id = widget_id

    # Create the widget's updated metadata, and register it with session_state.
    metadata = WidgetMetadata(
        widget_id,
        deserializer,
        serializer,
        value_type=ELEMENT_TYPE_TO_VALUE_TYPE[element_type],
        callback=on_change_handler,
        callback_args=args,
        callback_kwargs=kwargs,
    )
    return register_widget_from_metadata(metadata, ctx, widget_func_name, element_type)


def register_widget_from_metadata(
    metadata: WidgetMetadata[T],
    ctx: Optional["ScriptRunContext"],
    widget_func_name: Optional[str],
    element_type: ElementType,
) -> RegisterWidgetResult[T]:
    """Register a widget and return its value, using an already constructed
    `WidgetMetadata`.

    This is split out from `register_widget` to allow caching code to replay
    widgets by saving and reusing the completed metadata.

    See `register_widget` for details on what this returns.
    """
    # Local import to avoid import cycle
    import streamlit.runtime.caching as caching

    if ctx is None:
        # Early-out if we don't have a script run context (which probably means
        # we're running as a "bare" Python script, and not via `streamlit run`).
        return RegisterWidgetResult.failure(deserializer=metadata.deserializer)

    widget_id = metadata.id
    user_key = user_key_from_widget_id(widget_id)

    # Ensure another widget with the same user key hasn't already been registered.
    if user_key is not None:
        if user_key not in ctx.widget_user_keys_this_run:
            ctx.widget_user_keys_this_run.add(user_key)
        else:
            raise DuplicateWidgetID(
                _build_duplicate_widget_message(
                    widget_func_name if widget_func_name is not None else element_type,
                    user_key,
                )
            )

    # Ensure another widget with the same id hasn't already been registered.
    new_widget = widget_id not in ctx.widget_ids_this_run
    if new_widget:
        ctx.widget_ids_this_run.add(widget_id)
    else:
        raise DuplicateWidgetID(
            _build_duplicate_widget_message(
                widget_func_name if widget_func_name is not None else element_type,
                user_key,
            )
        )
    # Save the widget metadata for cached result replay
    caching.save_widget_metadata(metadata)
    return ctx.session_state.register_widget(metadata, user_key)


def coalesce_widget_states(
    old_states: WidgetStates, new_states: WidgetStates
) -> WidgetStates:
    """Coalesce an older WidgetStates into a newer one, and return a new
    WidgetStates containing the result.

    For most widget values, we just take the latest version.

    However, any trigger_values (which are set by buttons) that are True in
    `old_states` will be set to True in the coalesced result, so that button
    presses don't go missing.
    """
    states_by_id: Dict[str, WidgetState] = {
        wstate.id: wstate for wstate in new_states.widgets
    }

    for old_state in old_states.widgets:
        if old_state.WhichOneof("value") == "trigger_value" and old_state.trigger_value:

            # Ensure the corresponding new_state is also a trigger;
            # otherwise, a widget that was previously a button but no longer is
            # could get a bad value.
            new_trigger_val = states_by_id.get(old_state.id)
            if (
                new_trigger_val
                and new_trigger_val.WhichOneof("value") == "trigger_value"
            ):
                states_by_id[old_state.id] = old_state

    coalesced = WidgetStates()
    coalesced.widgets.extend(states_by_id.values())

    return coalesced


def _build_duplicate_widget_message(
    widget_func_name: str, user_key: Optional[str] = None
) -> str:
    if user_key is not None:
        message = textwrap.dedent(
            """
            There are multiple widgets with the same `key='{user_key}'`.

            To fix this, please make sure that the `key` argument is unique for each
            widget you create.
            """
        )
    else:
        message = textwrap.dedent(
            """
            There are multiple identical `st.{widget_type}` widgets with the
            same generated key.

            When a widget is created, it's assigned an internal key based on
            its structure. Multiple widgets with an identical structure will
            result in the same internal key, which causes this error.

            To fix this error, please pass a unique `key` argument to
            `st.{widget_type}`.
            """
        )

    return message.strip("\n").format(widget_type=widget_func_name, user_key=user_key)

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
from __future__ import annotations

import json
import pickle
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    KeysView,
    List,
    MutableMapping,
    Union,
    cast,
)

from pympler.asizeof import asizeof
from typing_extensions import Final, TypeAlias

import streamlit as st
from streamlit import config
from streamlit.errors import StreamlitAPIException, UnserializableSessionStateError
from streamlit.proto.WidgetStates_pb2 import WidgetState as WidgetStateProto
from streamlit.proto.WidgetStates_pb2 import WidgetStates as WidgetStatesProto
from streamlit.runtime.state.common import (
    RegisterWidgetResult,
    T,
    WidgetMetadata,
    is_keyed_widget_id,
    is_widget_id,
)
from streamlit.runtime.stats import CacheStat, CacheStatsProvider
from streamlit.type_util import ValueFieldName, is_array_value_field_name

if TYPE_CHECKING:
    from streamlit.runtime.session_manager import SessionManager


STREAMLIT_INTERNAL_KEY_PREFIX: Final = "$$STREAMLIT_INTERNAL_KEY"
SCRIPT_RUN_WITHOUT_ERRORS_KEY: Final = (
    f"{STREAMLIT_INTERNAL_KEY_PREFIX}_SCRIPT_RUN_WITHOUT_ERRORS"
)


@dataclass(frozen=True)
class Serialized:
    """A widget value that's serialized to a protobuf. Immutable."""

    value: WidgetStateProto


@dataclass(frozen=True)
class Value:
    """A widget value that's not serialized. Immutable."""

    value: Any


WState: TypeAlias = Union[Value, Serialized]


@dataclass
class WStates(MutableMapping[str, Any]):
    """A mapping of widget IDs to values. Widget values can be stored in
    serialized or deserialized form, but when values are retrieved from the
    mapping, they'll always be deserialized.
    """

    states: dict[str, WState] = field(default_factory=dict)
    widget_metadata: dict[str, WidgetMetadata[Any]] = field(default_factory=dict)

    def __getitem__(self, k: str) -> Any:
        """Return the value of the widget with the given key.
        If the widget's value is currently stored in serialized form, it
        will be deserialized first.
        """
        wstate = self.states.get(k)
        if wstate is None:
            raise KeyError(k)

        if isinstance(wstate, Value):
            # The widget's value is already deserialized - return it directly.
            return wstate.value

        # The widget's value is serialized. We deserialize it, and return
        # the deserialized value.

        metadata = self.widget_metadata.get(k)
        if metadata is None:
            # No deserializer, which should only happen if state is
            # gotten from a reconnecting browser and the script is
            # trying to access it. Pretend it doesn't exist.
            raise KeyError(k)
        value_field_name = cast(
            ValueFieldName,
            wstate.value.WhichOneof("value"),
        )
        value = wstate.value.__getattribute__(value_field_name)

        if is_array_value_field_name(value_field_name):
            # Array types are messages with data in a `data` field
            value = value.data
        elif value_field_name == "json_value":
            value = json.loads(value)

        deserialized = metadata.deserializer(value, metadata.id)

        # Update metadata to reflect information from WidgetState proto
        self.set_widget_metadata(
            replace(
                metadata,
                value_type=value_field_name,
            )
        )

        self.states[k] = Value(deserialized)
        return deserialized

    def __setitem__(self, k: str, v: WState) -> None:
        self.states[k] = v

    def __delitem__(self, k: str) -> None:
        del self.states[k]

    def __len__(self) -> int:
        return len(self.states)

    def __iter__(self):
        # For this and many other methods, we can't simply delegate to the
        # states field, because we need to invoke `__getitem__` for any
        # values, to handle deserialization and unwrapping of values.
        for key in self.states:
            yield key

    def keys(self) -> KeysView[str]:
        return KeysView(self.states)

    def items(self) -> set[tuple[str, Any]]:  # type: ignore[override]
        return {(k, self[k]) for k in self}

    def values(self) -> set[Any]:  # type: ignore[override]
        return {self[wid] for wid in self}

    def update(self, other: "WStates") -> None:  # type: ignore[override]
        """Copy all widget values and metadata from 'other' into this mapping,
        overwriting any data in this mapping that's also present in 'other'.
        """
        self.states.update(other.states)
        self.widget_metadata.update(other.widget_metadata)

    def set_widget_from_proto(self, widget_state: WidgetStateProto) -> None:
        """Set a widget's serialized value, overwriting any existing value it has."""
        self[widget_state.id] = Serialized(widget_state)

    def set_from_value(self, k: str, v: Any) -> None:
        """Set a widget's deserialized value, overwriting any existing value it has."""
        self[k] = Value(v)

    def set_widget_metadata(self, widget_meta: WidgetMetadata[Any]) -> None:
        """Set a widget's metadata, overwriting any existing metadata it has."""
        self.widget_metadata[widget_meta.id] = widget_meta

    def remove_stale_widgets(self, active_widget_ids: set[str]) -> None:
        """Remove widget state for widgets whose ids aren't in `active_widget_ids`."""
        self.states = {k: v for k, v in self.states.items() if k in active_widget_ids}

    def get_serialized(self, k: str) -> WidgetStateProto | None:
        """Get the serialized value of the widget with the given id.

        If the widget doesn't exist, return None. If the widget exists but
        is not in serialized form, it will be serialized first.
        """

        item = self.states.get(k)
        if item is None:
            # No such widget: return None.
            return None

        if isinstance(item, Serialized):
            # Widget value is serialized: return it directly.
            return item.value

        # Widget value is not serialized: serialize it first!
        metadata = self.widget_metadata.get(k)
        if metadata is None:
            # We're missing the widget's metadata. (Can this happen?)
            return None

        widget = WidgetStateProto()
        widget.id = k

        field = metadata.value_type
        serialized = metadata.serializer(item.value)
        if is_array_value_field_name(field):
            arr = getattr(widget, field)
            arr.data.extend(serialized)
        elif field == "json_value":
            setattr(widget, field, json.dumps(serialized))
        elif field == "file_uploader_state_value":
            widget.file_uploader_state_value.CopyFrom(serialized)
        else:
            setattr(widget, field, serialized)

        return widget

    def as_widget_states(self) -> list[WidgetStateProto]:
        """Return a list of serialized widget values for each widget with a value."""
        states = [
            self.get_serialized(widget_id)
            for widget_id in self.states.keys()
            if self.get_serialized(widget_id)
        ]
        states = cast(List[WidgetStateProto], states)
        return states

    def call_callback(self, widget_id: str) -> None:
        """Call the given widget's callback and return the callback's
        return value. If the widget has no callback, return None.

        If the widget doesn't exist, raise an Exception.
        """
        metadata = self.widget_metadata.get(widget_id)
        assert metadata is not None
        callback = metadata.callback
        if callback is None:
            return

        args = metadata.callback_args or ()
        kwargs = metadata.callback_kwargs or {}
        callback(*args, **kwargs)


def _missing_key_error_message(key: str) -> str:
    return (
        f'st.session_state has no key "{key}". Did you forget to initialize it? '
        f"More info: https://docs.streamlit.io/library/advanced-features/session-state#initialization"
    )


@dataclass
class SessionState:
    """SessionState allows users to store values that persist between app
    reruns.

    Example
    -------
    >>> if "num_script_runs" not in st.session_state:
    ...     st.session_state.num_script_runs = 0
    >>> st.session_state.num_script_runs += 1
    >>> st.write(st.session_state.num_script_runs)  # writes 1

    The next time your script runs, the value of
    st.session_state.num_script_runs will be preserved.
    >>> st.session_state.num_script_runs += 1
    >>> st.write(st.session_state.num_script_runs)  # writes 2
    """

    # All the values from previous script runs, squished together to save memory
    _old_state: dict[str, Any] = field(default_factory=dict)

    # Values set in session state during the current script run, possibly for
    # setting a widget's value. Keyed by a user provided string.
    _new_session_state: dict[str, Any] = field(default_factory=dict)

    # Widget values from the frontend, usually one changing prompted the script rerun
    _new_widget_state: WStates = field(default_factory=WStates)

    # Keys used for widgets will be eagerly converted to the matching widget id
    _key_id_mapping: dict[str, str] = field(default_factory=dict)

    # is it possible for a value to get through this without being deserialized?
    def _compact_state(self) -> None:
        """Copy all current session_state and widget_state values into our
        _old_state dict, and then clear our current session_state and
        widget_state.
        """
        for key_or_wid in self:
            self._old_state[key_or_wid] = self[key_or_wid]
        self._new_session_state.clear()
        self._new_widget_state.clear()

    def clear(self) -> None:
        """Reset self completely, clearing all current and old values."""
        self._old_state.clear()
        self._new_session_state.clear()
        self._new_widget_state.clear()
        self._key_id_mapping.clear()

    @property
    def filtered_state(self) -> dict[str, Any]:
        """The combined session and widget state, excluding keyless widgets."""

        wid_key_map = self._reverse_key_wid_map

        state: dict[str, Any] = {}

        # We can't write `for k, v in self.items()` here because doing so will
        # run into a `KeyError` if widget metadata has been cleared (which
        # happens when the streamlit server restarted or the cache was cleared),
        # then we receive a widget's state from a browser.
        for k in self._keys():
            if not is_widget_id(k) and not _is_internal_key(k):
                state[k] = self[k]
            elif is_keyed_widget_id(k):
                try:
                    key = wid_key_map[k]
                    state[key] = self[k]
                except KeyError:
                    # Widget id no longer maps to a key, it is a not yet
                    # cleared value in old state for a reset widget
                    pass

        return state

    @property
    def _reverse_key_wid_map(self) -> dict[str, str]:
        """Return a mapping of widget_id : widget_key."""
        wid_key_map = {v: k for k, v in self._key_id_mapping.items()}
        return wid_key_map

    def _keys(self) -> set[str]:
        """All keys active in Session State, with widget keys converted
        to widget ids when one is known. (This includes autogenerated keys
        for widgets that don't have user_keys defined, and which aren't
        exposed to user code.)
        """
        old_keys = {self._get_widget_id(k) for k in self._old_state.keys()}
        new_widget_keys = set(self._new_widget_state.keys())
        new_session_state_keys = {
            self._get_widget_id(k) for k in self._new_session_state.keys()
        }
        return old_keys | new_widget_keys | new_session_state_keys

    def is_new_state_value(self, user_key: str) -> bool:
        """True if a value with the given key is in the current session state."""
        return user_key in self._new_session_state

    def __iter__(self) -> Iterator[Any]:
        """Return an iterator over the keys of the SessionState.
        This is a shortcut for `iter(self.keys())`
        """
        return iter(self._keys())

    def __len__(self) -> int:
        """Return the number of items in SessionState."""
        return len(self._keys())

    def __getitem__(self, key: str) -> Any:
        wid_key_map = self._reverse_key_wid_map
        widget_id = self._get_widget_id(key)

        if widget_id in wid_key_map and widget_id == key:
            # the "key" is a raw widget id, so get its associated user key for lookup
            key = wid_key_map[widget_id]
        try:
            return self._getitem(widget_id, key)
        except KeyError:
            raise KeyError(_missing_key_error_message(key))

    def _getitem(self, widget_id: str | None, user_key: str | None) -> Any:
        """Get the value of an entry in Session State, using either the
        user-provided key or a widget id as appropriate for the internal dict
        being accessed.

        At least one of the arguments must have a value.
        """
        assert user_key is not None or widget_id is not None

        if user_key is not None:
            try:
                return self._new_session_state[user_key]
            except KeyError:
                pass

        if widget_id is not None:
            try:
                return self._new_widget_state[widget_id]
            except KeyError:
                pass

        # Typically, there won't be both a widget id and an associated state key in
        # old state at the same time, so the order we check is arbitrary.
        # The exception is if session state is set and then a later run has
        # a widget created, so the widget id entry should be newer.
        # The opposite case shouldn't happen, because setting the value of a widget
        # through session state will result in the next widget state reflecting that
        # value.
        if widget_id is not None:
            try:
                return self._old_state[widget_id]
            except KeyError:
                pass

        if user_key is not None:
            try:
                return self._old_state[user_key]
            except KeyError:
                pass

        # We'll never get here
        raise KeyError

    def __setitem__(self, user_key: str, value: Any) -> None:
        """Set the value of the session_state entry with the given user_key.

        If the key corresponds to a widget or form that's been instantiated
        during the current script run, raise a StreamlitAPIException instead.
        """
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx()

        if ctx is not None:
            widget_id = self._key_id_mapping.get(user_key, None)
            widget_ids = ctx.widget_ids_this_run
            form_ids = ctx.form_ids_this_run

            if widget_id in widget_ids or user_key in form_ids:
                raise StreamlitAPIException(
                    f"`st.session_state.{user_key}` cannot be modified after the widget"
                    f" with key `{user_key}` is instantiated."
                )

        self._new_session_state[user_key] = value

    def __delitem__(self, key: str) -> None:
        widget_id = self._get_widget_id(key)

        if not (key in self or widget_id in self):
            raise KeyError(_missing_key_error_message(key))

        if key in self._new_session_state:
            del self._new_session_state[key]

        if key in self._old_state:
            del self._old_state[key]

        if key in self._key_id_mapping:
            del self._key_id_mapping[key]

        if widget_id in self._new_widget_state:
            del self._new_widget_state[widget_id]

        if widget_id in self._old_state:
            del self._old_state[widget_id]

    def set_widgets_from_proto(self, widget_states: WidgetStatesProto) -> None:
        """Set the value of all widgets represented in the given WidgetStatesProto."""
        for state in widget_states.widgets:
            self._new_widget_state.set_widget_from_proto(state)

    def on_script_will_rerun(self, latest_widget_states: WidgetStatesProto) -> None:
        """Called by ScriptRunner before its script re-runs.

        Update widget data and call callbacks on widgets whose value changed
        between the previous and current script runs.
        """
        # Update ourselves with the new widget_states. The old widget states,
        # used to skip callbacks if values haven't changed, are also preserved.
        self._compact_state()
        self.set_widgets_from_proto(latest_widget_states)
        self._call_callbacks()

    def _call_callbacks(self) -> None:
        """Call any callback associated with each widget whose value
        changed between the previous and current script runs.
        """
        from streamlit.runtime.scriptrunner import RerunException

        changed_widget_ids = [
            wid for wid in self._new_widget_state if self._widget_changed(wid)
        ]
        for wid in changed_widget_ids:
            try:
                self._new_widget_state.call_callback(wid)
            except RerunException:
                st.warning(
                    "Calling st.experimental_rerun() within a callback is a no-op."
                )

    def _widget_changed(self, widget_id: str) -> bool:
        """True if the given widget's value changed between the previous
        script run and the current script run.
        """
        new_value = self._new_widget_state.get(widget_id)
        old_value = self._old_state.get(widget_id)
        changed: bool = new_value != old_value
        return changed

    def on_script_finished(self, widget_ids_this_run: set[str]) -> None:
        """Called by ScriptRunner after its script finishes running.
         Updates widgets to prepare for the next script run.

        Parameters
        ----------
        widget_ids_this_run: set[str]
            The IDs of the widgets that were accessed during the script
            run. Any widget state whose ID does *not* appear in this set
            is considered "stale" and will be removed.
        """
        self._reset_triggers()
        self._remove_stale_widgets(widget_ids_this_run)

    def _reset_triggers(self) -> None:
        """Set all trigger values in our state dictionary to False."""
        for state_id in self._new_widget_state:
            metadata = self._new_widget_state.widget_metadata.get(state_id)
            if metadata is not None and metadata.value_type == "trigger_value":
                self._new_widget_state[state_id] = Value(False)

        for state_id in self._old_state:
            metadata = self._new_widget_state.widget_metadata.get(state_id)
            if metadata is not None and metadata.value_type == "trigger_value":
                self._old_state[state_id] = False

    def _remove_stale_widgets(self, active_widget_ids: set[str]) -> None:
        """Remove widget state for widgets whose ids aren't in `active_widget_ids`."""
        self._new_widget_state.remove_stale_widgets(active_widget_ids)

        # Remove entries from _old_state corresponding to
        # widgets not in widget_ids.
        self._old_state = {
            k: v
            for k, v in self._old_state.items()
            if (k in active_widget_ids or not is_widget_id(k))
        }

    def _set_widget_metadata(self, widget_metadata: WidgetMetadata[Any]) -> None:
        """Set a widget's metadata."""
        widget_id = widget_metadata.id
        self._new_widget_state.widget_metadata[widget_id] = widget_metadata

    def get_widget_states(self) -> list[WidgetStateProto]:
        """Return a list of serialized widget values for each widget with a value."""
        return self._new_widget_state.as_widget_states()

    def _get_widget_id(self, k: str) -> str:
        """Turns a value that might be a widget id or a user provided key into
        an appropriate widget id.
        """
        return self._key_id_mapping.get(k, k)

    def _set_key_widget_mapping(self, widget_id: str, user_key: str) -> None:
        self._key_id_mapping[user_key] = widget_id

    def register_widget(
        self, metadata: WidgetMetadata[T], user_key: str | None
    ) -> RegisterWidgetResult[T]:
        """Register a widget with the SessionState.

        Returns
        -------
        RegisterWidgetResult[T]
            Contains the widget's current value, and a bool that will be True
            if the frontend needs to be updated with the current value.
        """
        widget_id = metadata.id

        self._set_widget_metadata(metadata)
        if user_key is not None:
            # If the widget has a user_key, update its user_key:widget_id mapping
            self._set_key_widget_mapping(widget_id, user_key)

        if widget_id not in self and (user_key is None or user_key not in self):
            # This is the first time the widget is registered, so we save its
            # value in widget state.
            deserializer = metadata.deserializer
            initial_widget_value = deepcopy(deserializer(None, metadata.id))
            self._new_widget_state.set_from_value(widget_id, initial_widget_value)

        # Get the current value of the widget for use as its return value.
        # We return a copy, so that reference types can't be accidentally
        # mutated by user code.
        widget_value = cast(T, self[widget_id])
        widget_value = deepcopy(widget_value)

        # widget_value_changed indicates to the caller that the widget's
        # current value is different from what is in the frontend.
        widget_value_changed = user_key is not None and self.is_new_state_value(
            user_key
        )

        return RegisterWidgetResult(widget_value, widget_value_changed)

    def __contains__(self, key: str) -> bool:
        try:
            self[key]
        except KeyError:
            return False
        else:
            return True

    def get_stats(self) -> list[CacheStat]:
        stat = CacheStat("st_session_state", "", asizeof(self))
        return [stat]

    def _check_serializable(self) -> None:
        """Verify that everything added to session state can be serialized.
        We use pickleability as the metric for serializability, and test for
        pickleability by just trying it.
        """
        for k in self:
            try:
                pickle.dumps(self[k])
            except Exception as e:
                err_msg = f"""Cannot serialize the value (of type `{type(self[k])}`) of '{k}' in st.session_state.
                Streamlit has been configured to use [pickle](https://docs.python.org/3/library/pickle.html) to
                serialize session_state values. Please convert the value to a pickle-serializable type. To learn
                more about this behavior, see [our docs](https://docs.streamlit.io/knowledge-base/using-streamlit/serializable-session-state). """
                raise UnserializableSessionStateError(err_msg) from e

    def maybe_check_serializable(self) -> None:
        """Verify that session state can be serialized, if the relevant config
        option is set.

        See `_check_serializable` for details."""
        if config.get_option("runner.enforceSerializableSessionState"):
            self._check_serializable()


def _is_internal_key(key: str) -> bool:
    return key.startswith(STREAMLIT_INTERNAL_KEY_PREFIX)


@dataclass
class SessionStateStatProvider(CacheStatsProvider):
    _session_mgr: "SessionManager"

    def get_stats(self) -> list[CacheStat]:
        stats: list[CacheStat] = []
        for session_info in self._session_mgr.list_active_sessions():
            session_state = session_info.session.session_state
            stats.extend(session_state.get_stats())
        return stats

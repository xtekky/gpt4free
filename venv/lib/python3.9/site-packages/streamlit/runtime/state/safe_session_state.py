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

import threading
from typing import Any, Dict, List, Optional, Set

from streamlit.proto.WidgetStates_pb2 import WidgetState as WidgetStateProto
from streamlit.proto.WidgetStates_pb2 import WidgetStates as WidgetStatesProto
from streamlit.runtime.state.common import RegisterWidgetResult, T, WidgetMetadata
from streamlit.runtime.state.session_state import SessionState


class SafeSessionState:
    """Thread-safe wrapper around SessionState.

    When AppSession gets a re-run request, it can interrupt its existing
    ScriptRunner and spin up a new ScriptRunner to handle the request.
    When this happens, the existing ScriptRunner will continue executing
    its script until it reaches a yield point - but during this time, it
    must not mutate its SessionState. An interrupted ScriptRunner assigns
    a dummy SessionState instance to its wrapper to prevent further mutation.
    """

    def __init__(self, state: SessionState):
        self._state = state
        # TODO: we'd prefer this be a threading.Lock instead of RLock -
        #  but `call_callbacks` first needs to be rewritten.
        self._lock = threading.RLock()
        self._disconnected = False

    def disconnect(self) -> None:
        """Disconnect the wrapper from its underlying SessionState.
        ScriptRunner calls this when it gets a stop request. After this
        function is called, all future SessionState interactions are no-ops.
        """
        with self._lock:
            self._disconnected = True

    def register_widget(
        self, metadata: WidgetMetadata[T], user_key: Optional[str]
    ) -> RegisterWidgetResult[T]:
        with self._lock:
            if self._disconnected:
                return RegisterWidgetResult.failure(metadata.deserializer)

            return self._state.register_widget(metadata, user_key)

    def on_script_will_rerun(self, latest_widget_states: WidgetStatesProto) -> None:
        with self._lock:
            if self._disconnected:
                return

            # TODO: rewrite this to copy the callbacks list into a local
            #  variable so that we don't need to hold our lock for the
            #  duration. (This will also allow us to downgrade our RLock
            #  to a Lock.)
            self._state.on_script_will_rerun(latest_widget_states)

    def on_script_finished(self, widget_ids_this_run: Set[str]) -> None:
        with self._lock:
            if self._disconnected:
                return

            self._state.on_script_finished(widget_ids_this_run)

    def maybe_check_serializable(self) -> None:
        with self._lock:
            if self._disconnected:
                return

            self._state.maybe_check_serializable()

    def get_widget_states(self) -> List[WidgetStateProto]:
        """Return a list of serialized widget values for each widget with a value."""
        with self._lock:
            if self._disconnected:
                return []

            return self._state.get_widget_states()

    def is_new_state_value(self, user_key: str) -> bool:
        with self._lock:
            if self._disconnected:
                return False

            return self._state.is_new_state_value(user_key)

    @property
    def filtered_state(self) -> Dict[str, Any]:
        """The combined session and widget state, excluding keyless widgets."""
        with self._lock:
            if self._disconnected:
                return {}

            return self._state.filtered_state

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            if self._disconnected:
                raise KeyError(key)

            return self._state[key]

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            if self._disconnected:
                return

            self._state[key] = value

    def __delitem__(self, key: str) -> None:
        with self._lock:
            if self._disconnected:
                raise KeyError(key)

            del self._state[key]

    def __contains__(self, key: str) -> bool:
        with self._lock:
            if self._disconnected:
                return False

            return key in self._state

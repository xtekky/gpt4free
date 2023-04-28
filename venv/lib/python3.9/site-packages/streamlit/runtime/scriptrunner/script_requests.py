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
from dataclasses import dataclass
from enum import Enum
from typing import Optional, cast

from streamlit.proto.WidgetStates_pb2 import WidgetStates
from streamlit.runtime.state import coalesce_widget_states


class ScriptRequestType(Enum):
    # The ScriptRunner should continue running its script.
    CONTINUE = "CONTINUE"

    # If the script is running, it should be stopped as soon
    # as the ScriptRunner reaches an interrupt point.
    # This is a terminal state.
    STOP = "STOP"

    # A script rerun has been requested. The ScriptRunner should
    # handle this request as soon as it reaches an interrupt point.
    RERUN = "RERUN"


@dataclass(frozen=True)
class RerunData:
    """Data attached to RERUN requests. Immutable."""

    query_string: str = ""
    widget_states: Optional[WidgetStates] = None
    page_script_hash: str = ""
    page_name: str = ""


@dataclass(frozen=True)
class ScriptRequest:
    """A STOP or RERUN request and associated data."""

    type: ScriptRequestType
    _rerun_data: Optional[RerunData] = None

    @property
    def rerun_data(self) -> RerunData:
        if self.type is not ScriptRequestType.RERUN:
            raise RuntimeError("RerunData is only set for RERUN requests.")
        return cast(RerunData, self._rerun_data)


class ScriptRequests:
    """An interface for communicating with a ScriptRunner. Thread-safe.

    AppSession makes requests of a ScriptRunner through this class, and
    ScriptRunner handles those requests.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._state = ScriptRequestType.CONTINUE
        self._rerun_data = RerunData()

    def request_stop(self) -> None:
        """Request that the ScriptRunner stop running. A stopped ScriptRunner
        can't be used anymore. STOP requests succeed unconditionally.
        """
        with self._lock:
            self._state = ScriptRequestType.STOP

    def request_rerun(self, new_data: RerunData) -> bool:
        """Request that the ScriptRunner rerun its script.

        If the ScriptRunner has been stopped, this request can't be honored:
        return False.

        Otherwise, record the request and return True. The ScriptRunner will
        handle the rerun request as soon as it reaches an interrupt point.
        """

        with self._lock:
            if self._state == ScriptRequestType.STOP:
                # We can't rerun after being stopped.
                return False

            if self._state == ScriptRequestType.CONTINUE:
                # If we're running, we can handle a rerun request
                # unconditionally.
                self._state = ScriptRequestType.RERUN
                self._rerun_data = new_data
                return True

            if self._state == ScriptRequestType.RERUN:
                # If we have an existing Rerun request, we coalesce this
                # new request into it.
                if self._rerun_data.widget_states is None:
                    # The existing request's widget_states is None, which
                    # means it wants to rerun with whatever the most
                    # recent script execution's widget state was.
                    # We have no meaningful state to merge with, and
                    # so we simply overwrite the existing request.
                    self._rerun_data = new_data
                    return True

                if new_data.widget_states is not None:
                    # Both the existing and the new request have
                    # non-null widget_states. Merge them together.
                    coalesced_states = coalesce_widget_states(
                        self._rerun_data.widget_states, new_data.widget_states
                    )
                    self._rerun_data = RerunData(
                        query_string=new_data.query_string,
                        widget_states=coalesced_states,
                        page_script_hash=new_data.page_script_hash,
                        page_name=new_data.page_name,
                    )
                    return True

                # If old widget_states is NOT None, and new widget_states IS
                # None, then this new request is entirely redundant. Leave
                # our existing rerun_data as is.
                return True

            # We'll never get here
            raise RuntimeError(f"Unrecognized ScriptRunnerState: {self._state}")

    def on_scriptrunner_yield(self) -> Optional[ScriptRequest]:
        """Called by the ScriptRunner when it's at a yield point.

        If we have no request, return None.

        If we have a RERUN request, return the request and set our internal
        state to CONTINUE.

        If we have a STOP request, return the request and remain stopped.
        """
        if self._state == ScriptRequestType.CONTINUE:
            # We avoid taking a lock in the common case. If a STOP or RERUN
            # request is received between the `if` and `return`, it will be
            # handled at the next `on_scriptrunner_yield`, or when
            # `on_scriptrunner_ready` is called.
            return None

        with self._lock:
            if self._state == ScriptRequestType.RERUN:
                self._state = ScriptRequestType.CONTINUE
                return ScriptRequest(ScriptRequestType.RERUN, self._rerun_data)

            assert self._state == ScriptRequestType.STOP
            return ScriptRequest(ScriptRequestType.STOP)

    def on_scriptrunner_ready(self) -> ScriptRequest:
        """Called by the ScriptRunner when it's about to run its script for
        the first time, and also after its script has successfully completed.

        If we have a RERUN request, return the request and set
        our internal state to CONTINUE.

        If we have a STOP request or no request, set our internal state
        to STOP.
        """
        with self._lock:
            if self._state == ScriptRequestType.RERUN:
                self._state = ScriptRequestType.CONTINUE
                return ScriptRequest(ScriptRequestType.RERUN, self._rerun_data)

            # If we don't have a rerun request, unconditionally change our
            # state to STOP.
            self._state = ScriptRequestType.STOP
            return ScriptRequest(ScriptRequestType.STOP)

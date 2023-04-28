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

import collections
import threading
from dataclasses import dataclass, field
from typing import Callable, Counter, Dict, List, Optional, Set

from typing_extensions import Final, TypeAlias

from streamlit import runtime
from streamlit.errors import StreamlitAPIException
from streamlit.logger import get_logger
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.proto.PageProfile_pb2 import Command
from streamlit.runtime.state import SafeSessionState
from streamlit.runtime.uploaded_file_manager import UploadedFileManager

LOGGER: Final = get_logger(__name__)

UserInfo: TypeAlias = Dict[str, Optional[str]]


@dataclass
class ScriptRunContext:
    """A context object that contains data for a "script run" - that is,
    data that's scoped to a single ScriptRunner execution (and therefore also
    scoped to a single connected "session").

    ScriptRunContext is used internally by virtually every `st.foo()` function.
    It is accessed only from the script thread that's created by ScriptRunner.

    Streamlit code typically retrieves the active ScriptRunContext via the
    `get_script_run_ctx` function.
    """

    session_id: str
    _enqueue: Callable[[ForwardMsg], None]
    query_string: str
    session_state: SafeSessionState
    uploaded_file_mgr: UploadedFileManager
    page_script_hash: str
    user_info: UserInfo

    gather_usage_stats: bool = False
    command_tracking_deactivated: bool = False
    tracked_commands: List[Command] = field(default_factory=list)
    tracked_commands_counter: Counter[str] = field(default_factory=collections.Counter)
    _set_page_config_allowed: bool = True
    _has_script_started: bool = False
    widget_ids_this_run: Set[str] = field(default_factory=set)
    widget_user_keys_this_run: Set[str] = field(default_factory=set)
    form_ids_this_run: Set[str] = field(default_factory=set)
    cursors: Dict[int, "streamlit.cursor.RunningCursor"] = field(default_factory=dict)
    dg_stack: List["streamlit.delta_generator.DeltaGenerator"] = field(
        default_factory=list
    )

    def reset(self, query_string: str = "", page_script_hash: str = "") -> None:
        self.cursors = {}
        self.widget_ids_this_run = set()
        self.widget_user_keys_this_run = set()
        self.form_ids_this_run = set()
        self.query_string = query_string
        self.page_script_hash = page_script_hash
        # Permit set_page_config when the ScriptRunContext is reused on a rerun
        self._set_page_config_allowed = True
        self._has_script_started = False
        self.command_tracking_deactivated: bool = False
        self.tracked_commands = []
        self.tracked_commands_counter = collections.Counter()

    def on_script_start(self) -> None:
        self._has_script_started = True

    def enqueue(self, msg: ForwardMsg) -> None:
        """Enqueue a ForwardMsg for this context's session."""
        if msg.HasField("page_config_changed") and not self._set_page_config_allowed:
            raise StreamlitAPIException(
                "`set_page_config()` can only be called once per app, "
                + "and must be called as the first Streamlit command in your script.\n\n"
                + "For more information refer to the [docs]"
                + "(https://docs.streamlit.io/library/api-reference/utilities/st.set_page_config)."
            )

        # We want to disallow set_page config if one of the following occurs:
        # - set_page_config was called on this message
        # - The script has already started and a different st call occurs (a delta)
        if msg.HasField("page_config_changed") or (
            msg.HasField("delta") and self._has_script_started
        ):
            self._set_page_config_allowed = False

        # Pass the message up to our associated ScriptRunner.
        self._enqueue(msg)


SCRIPT_RUN_CONTEXT_ATTR_NAME: Final = "streamlit_script_run_ctx"


def add_script_run_ctx(
    thread: Optional[threading.Thread] = None, ctx: Optional[ScriptRunContext] = None
):
    """Adds the current ScriptRunContext to a newly-created thread.

    This should be called from this thread's parent thread,
    before the new thread starts.

    Parameters
    ----------
    thread : threading.Thread
        The thread to attach the current ScriptRunContext to.
    ctx : ScriptRunContext or None
        The ScriptRunContext to add, or None to use the current thread's
        ScriptRunContext.

    Returns
    -------
    threading.Thread
        The same thread that was passed in, for chaining.

    """
    if thread is None:
        thread = threading.current_thread()
    if ctx is None:
        ctx = get_script_run_ctx()
    if ctx is not None:
        setattr(thread, SCRIPT_RUN_CONTEXT_ATTR_NAME, ctx)
    return thread


def get_script_run_ctx(suppress_warning: bool = False) -> Optional[ScriptRunContext]:
    """
    Parameters
    ----------
    suppress_warning : bool
        If True, don't log a warning if there's no ScriptRunContext.
    Returns
    -------
    ScriptRunContext | None
        The current thread's ScriptRunContext, or None if it doesn't have one.

    """
    thread = threading.current_thread()
    ctx: Optional[ScriptRunContext] = getattr(
        thread, SCRIPT_RUN_CONTEXT_ATTR_NAME, None
    )
    if ctx is None and runtime.exists() and not suppress_warning:
        # Only warn about a missing ScriptRunContext if suppress_warning is False, and
        # we were started via `streamlit run`. Otherwise, the user is likely running a
        # script "bare", and doesn't need to be warned about streamlit
        # bits that are irrelevant when not connected to a session.
        LOGGER.warning("Thread '%s': missing ScriptRunContext", thread.name)

    return ctx


# Needed to avoid circular dependencies while running tests.
import streamlit

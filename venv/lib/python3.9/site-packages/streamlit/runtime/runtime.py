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

import asyncio
import sys
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Awaitable, Dict, NamedTuple, Optional, Tuple, Type

from typing_extensions import Final

from streamlit import config
from streamlit.logger import get_logger
from streamlit.proto.BackMsg_pb2 import BackMsg
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.app_session import AppSession
from streamlit.runtime.caching import (
    get_data_cache_stats_provider,
    get_resource_cache_stats_provider,
)
from streamlit.runtime.caching.storage.local_disk_cache_storage import (
    LocalDiskCacheStorageManager,
)
from streamlit.runtime.forward_msg_cache import (
    ForwardMsgCache,
    create_reference_msg,
    populate_hash_if_needed,
)
from streamlit.runtime.legacy_caching.caching import _mem_caches
from streamlit.runtime.media_file_manager import MediaFileManager
from streamlit.runtime.media_file_storage import MediaFileStorage
from streamlit.runtime.memory_session_storage import MemorySessionStorage
from streamlit.runtime.runtime_util import is_cacheable_msg
from streamlit.runtime.script_data import ScriptData
from streamlit.runtime.session_manager import (
    ActiveSessionInfo,
    SessionClient,
    SessionClientDisconnectedError,
    SessionManager,
    SessionStorage,
)
from streamlit.runtime.state import (
    SCRIPT_RUN_WITHOUT_ERRORS_KEY,
    SessionStateStatProvider,
)
from streamlit.runtime.stats import StatsManager
from streamlit.runtime.uploaded_file_manager import UploadedFileManager
from streamlit.runtime.websocket_session_manager import WebsocketSessionManager
from streamlit.watcher import LocalSourcesWatcher

if TYPE_CHECKING:
    from streamlit.runtime.caching.storage import CacheStorageManager

# Wait for the script run result for 60s and if no result is available give up
SCRIPT_RUN_CHECK_TIMEOUT: Final = 60

LOGGER: Final = get_logger(__name__)


class RuntimeStoppedError(Exception):
    """Raised by operations on a Runtime instance that is stopped."""


@dataclass(frozen=True)
class RuntimeConfig:
    """Config options for StreamlitRuntime."""

    # The filesystem path of the Streamlit script to run.
    script_path: str

    # The (optional) command line that Streamlit was started with
    # (e.g. "streamlit run app.py")
    command_line: Optional[str]

    # The storage backend for Streamlit's MediaFileManager.
    media_file_storage: MediaFileStorage

    # The cache storage backend for Streamlit's st.cache_data.
    cache_storage_manager: CacheStorageManager = field(
        default_factory=LocalDiskCacheStorageManager
    )

    # The SessionManager class to be used.
    session_manager_class: Type[SessionManager] = WebsocketSessionManager

    # The SessionStorage instance for the SessionManager to use.
    session_storage: SessionStorage = field(default_factory=MemorySessionStorage)


class RuntimeState(Enum):
    INITIAL = "INITIAL"
    NO_SESSIONS_CONNECTED = "NO_SESSIONS_CONNECTED"
    ONE_OR_MORE_SESSIONS_CONNECTED = "ONE_OR_MORE_SESSIONS_CONNECTED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"


class AsyncObjects(NamedTuple):
    """Container for all asyncio objects that Runtime manages.
    These cannot be initialized until the Runtime's eventloop is assigned.
    """

    # The eventloop that Runtime is running on.
    eventloop: asyncio.AbstractEventLoop

    # Set after Runtime.stop() is called. Never cleared.
    must_stop: asyncio.Event

    # Set when a client connects; cleared when we have no connected clients.
    has_connection: asyncio.Event

    # Set after a ForwardMsg is enqueued; cleared when we flush ForwardMsgs.
    need_send_data: asyncio.Event

    # Completed when the Runtime has started.
    started: asyncio.Future[None]

    # Completed when the Runtime has stopped.
    stopped: asyncio.Future[None]


class Runtime:
    _instance: Optional[Runtime] = None

    @classmethod
    def instance(cls) -> Runtime:
        """Return the singleton Runtime instance. Raise an Error if the
        Runtime hasn't been created yet.
        """
        if cls._instance is None:
            raise RuntimeError("Runtime hasn't been created!")
        return cls._instance

    @classmethod
    def exists(cls) -> bool:
        """True if the singleton Runtime instance has been created.

        When a Streamlit app is running in "raw mode" - that is, when the
        app is run via `python app.py` instead of `streamlit run app.py` -
        the Runtime will not exist, and various Streamlit functions need
        to adapt.
        """
        return cls._instance is not None

    def __init__(self, config: RuntimeConfig):
        """Create a Runtime instance. It won't be started yet.

        Runtime is *not* thread-safe. Its public methods are generally
        safe to call only on the same thread that its event loop runs on.

        Parameters
        ----------
        config
            Config options.
        """
        if Runtime._instance is not None:
            raise RuntimeError("Runtime instance already exists!")
        Runtime._instance = self

        # Will be created when we start.
        self._async_objs: Optional[AsyncObjects] = None

        # The task that runs our main loop. We need to save a reference
        # to it so that it doesn't get garbage collected while running.
        self._loop_coroutine_task: Optional[asyncio.Task[None]] = None

        self._main_script_path = config.script_path
        self._command_line = config.command_line or ""

        self._state = RuntimeState.INITIAL

        # Initialize managers
        self._message_cache = ForwardMsgCache()
        self._uploaded_file_mgr = UploadedFileManager()
        self._uploaded_file_mgr.on_files_updated.connect(self._on_files_updated)
        self._media_file_mgr = MediaFileManager(storage=config.media_file_storage)
        self._cache_storage_manager = config.cache_storage_manager

        self._session_mgr = config.session_manager_class(
            session_storage=config.session_storage,
            uploaded_file_manager=self._uploaded_file_mgr,
            message_enqueued_callback=self._enqueued_some_message,
        )

        self._stats_mgr = StatsManager()
        self._stats_mgr.register_provider(get_data_cache_stats_provider())
        self._stats_mgr.register_provider(get_resource_cache_stats_provider())
        self._stats_mgr.register_provider(_mem_caches)
        self._stats_mgr.register_provider(self._message_cache)
        self._stats_mgr.register_provider(self._uploaded_file_mgr)
        self._stats_mgr.register_provider(SessionStateStatProvider(self._session_mgr))

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def message_cache(self) -> ForwardMsgCache:
        return self._message_cache

    @property
    def uploaded_file_mgr(self) -> UploadedFileManager:
        return self._uploaded_file_mgr

    @property
    def cache_storage_manager(self) -> CacheStorageManager:
        return self._cache_storage_manager

    @property
    def media_file_mgr(self) -> MediaFileManager:
        return self._media_file_mgr

    @property
    def stats_mgr(self) -> StatsManager:
        return self._stats_mgr

    @property
    def stopped(self) -> Awaitable[None]:
        """A Future that completes when the Runtime's run loop has exited."""
        return self._get_async_objs().stopped

    # NOTE: A few Runtime methods listed as threadsafe (get_client, _on_files_updated,
    # and is_active_session) currently rely on the implementation detail that
    # WebsocketSessionManager's get_active_session_info and is_active_session methods
    # happen to be threadsafe. This may change with future SessionManager implementations,
    # at which point we'll need to formalize our thread safety rules for each
    # SessionManager method.
    def get_client(self, session_id: str) -> Optional[SessionClient]:
        """Get the SessionClient for the given session_id, or None
        if no such session exists.

        Notes
        -----
        Threading: SAFE. May be called on any thread.
        """
        session_info = self._session_mgr.get_active_session_info(session_id)
        if session_info is None:
            return None
        return session_info.client

    def _on_files_updated(self, session_id: str) -> None:
        """Event handler for UploadedFileManager.on_file_added.
        Ensures that uploaded files from stale sessions get deleted.

        Notes
        -----
        Threading: SAFE. May be called on any thread.
        """
        if not self._session_mgr.is_active_session(session_id):
            # If an uploaded file doesn't belong to an active session,
            # remove it so it doesn't stick around forever.
            self._uploaded_file_mgr.remove_session_files(session_id)

    async def start(self) -> None:
        """Start the runtime. This must be called only once, before
        any other functions are called.

        When this coroutine returns, Streamlit is ready to accept new sessions.

        Notes
        -----
        Threading: UNSAFE. Must be called on the eventloop thread.
        """

        # Create our AsyncObjects. We need to have a running eventloop to
        # instantiate our various synchronization primitives.
        async_objs = AsyncObjects(
            eventloop=asyncio.get_running_loop(),
            must_stop=asyncio.Event(),
            has_connection=asyncio.Event(),
            need_send_data=asyncio.Event(),
            started=asyncio.Future(),
            stopped=asyncio.Future(),
        )
        self._async_objs = async_objs

        if sys.version_info >= (3, 8, 0):
            # Python 3.8+ supports a create_task `name` parameter, which can
            # make debugging a bit easier.
            self._loop_coroutine_task = asyncio.create_task(
                self._loop_coroutine(), name="Runtime.loop_coroutine"
            )
        else:
            self._loop_coroutine_task = asyncio.create_task(self._loop_coroutine())

        await async_objs.started

    def stop(self) -> None:
        """Request that Streamlit close all sessions and stop running.
        Note that Streamlit won't stop running immediately.

        Notes
        -----
        Threading: SAFE. May be called from any thread.
        """

        async_objs = self._get_async_objs()

        def stop_on_eventloop():
            if self._state in (RuntimeState.STOPPING, RuntimeState.STOPPED):
                return

            LOGGER.debug("Runtime stopping...")
            self._set_state(RuntimeState.STOPPING)
            async_objs.must_stop.set()

        async_objs.eventloop.call_soon_threadsafe(stop_on_eventloop)

    def is_active_session(self, session_id: str) -> bool:
        """True if the session_id belongs to an active session.

        Notes
        -----
        Threading: SAFE. May be called on any thread.
        """
        return self._session_mgr.is_active_session(session_id)

    def connect_session(
        self,
        client: SessionClient,
        user_info: Dict[str, Optional[str]],
        existing_session_id: Optional[str] = None,
    ) -> str:
        """Create a new session (or connect to an existing one) and return its unique ID.

        Parameters
        ----------
        client
            A concrete SessionClient implementation for communicating with
            the session's client.
        user_info
            A dict that contains information about the session's user. For now,
            it only (optionally) contains the user's email address.

            {
                "email": "example@example.com"
            }

        Returns
        -------
        str
            The session's unique string ID.

        Notes
        -----
        Threading: UNSAFE. Must be called on the eventloop thread.
        """
        if self._state in (RuntimeState.STOPPING, RuntimeState.STOPPED):
            raise RuntimeStoppedError(f"Can't connect_session (state={self._state})")

        session_id = self._session_mgr.connect_session(
            client=client,
            script_data=ScriptData(self._main_script_path, self._command_line or ""),
            user_info=user_info,
            existing_session_id=existing_session_id,
        )
        self._set_state(RuntimeState.ONE_OR_MORE_SESSIONS_CONNECTED)
        self._get_async_objs().has_connection.set()

        return session_id

    def create_session(
        self,
        client: SessionClient,
        user_info: Dict[str, Optional[str]],
        existing_session_id: Optional[str] = None,
    ) -> str:
        """Create a new session (or connect to an existing one) and return its unique ID.

        Notes
        -----
        This method is simply an alias for connect_session added for backwards
        compatibility.
        """
        LOGGER.warning("create_session is deprecated! Use connect_session instead.")
        return self.connect_session(
            client=client, user_info=user_info, existing_session_id=existing_session_id
        )

    def close_session(self, session_id: str) -> None:
        """Close and completely shut down a session.

        This differs from disconnect_session in that it always completely shuts down a
        session, permanently losing any associated state (session state, uploaded files,
        etc.).

        This function may be called multiple times for the same session,
        which is not an error. (Subsequent calls just no-op.)

        Parameters
        ----------
        session_id
            The session's unique ID.

        Notes
        -----
        Threading: UNSAFE. Must be called on the eventloop thread.
        """
        self._session_mgr.close_session(session_id)
        self._on_session_disconnected()

    def disconnect_session(self, session_id: str) -> None:
        """Disconnect a session. It will stop producing ForwardMsgs.

        Differs from close_session because disconnected sessions can be reconnected to
        for a brief window (depending on the SessionManager/SessionStorage
        implementations used by the runtime).

        This function may be called multiple times for the same session,
        which is not an error. (Subsequent calls just no-op.)

        Parameters
        ----------
        session_id
            The session's unique ID.

        Notes
        -----
        Threading: UNSAFE. Must be called on the eventloop thread.
        """
        self._session_mgr.disconnect_session(session_id)
        self._on_session_disconnected()

    def handle_backmsg(self, session_id: str, msg: BackMsg) -> None:
        """Send a BackMsg to an active session.

        Parameters
        ----------
        session_id
            The session's unique ID.
        msg
            The BackMsg to deliver to the session.

        Notes
        -----
        Threading: UNSAFE. Must be called on the eventloop thread.
        """
        if self._state in (RuntimeState.STOPPING, RuntimeState.STOPPED):
            raise RuntimeStoppedError(f"Can't handle_backmsg (state={self._state})")

        session_info = self._session_mgr.get_active_session_info(session_id)
        if session_info is None:
            LOGGER.debug(
                "Discarding BackMsg for disconnected session (id=%s)", session_id
            )
            return

        session_info.session.handle_backmsg(msg)

    def handle_backmsg_deserialization_exception(
        self, session_id: str, exc: BaseException
    ) -> None:
        """Handle an Exception raised during deserialization of a BackMsg.

        Parameters
        ----------
        session_id
            The session's unique ID.
        exc
            The Exception.

        Notes
        -----
        Threading: UNSAFE. Must be called on the eventloop thread.
        """
        if self._state in (RuntimeState.STOPPING, RuntimeState.STOPPED):
            raise RuntimeStoppedError(
                f"Can't handle_backmsg_deserialization_exception (state={self._state})"
            )

        session_info = self._session_mgr.get_active_session_info(session_id)
        if session_info is None:
            LOGGER.debug(
                "Discarding BackMsg Exception for disconnected session (id=%s)",
                session_id,
            )
            return

        session_info.session.handle_backmsg_exception(exc)

    @property
    async def is_ready_for_browser_connection(self) -> Tuple[bool, str]:
        if self._state not in (
            RuntimeState.INITIAL,
            RuntimeState.STOPPING,
            RuntimeState.STOPPED,
        ):
            return True, "ok"

        return False, "unavailable"

    async def does_script_run_without_error(self) -> Tuple[bool, str]:
        """Load and execute the app's script to verify it runs without an error.

        Returns
        -------
        (True, "ok") if the script completes without error, or (False, err_msg)
        if the script raises an exception.

        Notes
        -----
        Threading: UNSAFE. Must be called on the eventloop thread.
        """
        # NOTE: We create an AppSession directly here instead of using the
        # SessionManager intentionally. This isn't a "real" session and is only being
        # used to test that the script runs without error.
        session = AppSession(
            script_data=ScriptData(self._main_script_path, self._command_line),
            uploaded_file_manager=self._uploaded_file_mgr,
            message_enqueued_callback=self._enqueued_some_message,
            local_sources_watcher=LocalSourcesWatcher(self._main_script_path),
            user_info={"email": "test@test.com"},
        )

        try:
            session.request_rerun(None)

            now = time.perf_counter()
            while (
                SCRIPT_RUN_WITHOUT_ERRORS_KEY not in session.session_state
                and (time.perf_counter() - now) < SCRIPT_RUN_CHECK_TIMEOUT
            ):
                await asyncio.sleep(0.1)

            if SCRIPT_RUN_WITHOUT_ERRORS_KEY not in session.session_state:
                return False, "timeout"

            ok = session.session_state[SCRIPT_RUN_WITHOUT_ERRORS_KEY]
            msg = "ok" if ok else "error"

            return ok, msg
        finally:
            session.shutdown()

    def _set_state(self, new_state: RuntimeState) -> None:
        LOGGER.debug("Runtime state: %s -> %s", self._state, new_state)
        self._state = new_state

    async def _loop_coroutine(self) -> None:
        """The main Runtime loop.

        This function won't exit until `stop` is called.

        Notes
        -----
        Threading: UNSAFE. Must be called on the eventloop thread.
        """

        async_objs = self._get_async_objs()

        try:
            if self._state == RuntimeState.INITIAL:
                self._set_state(RuntimeState.NO_SESSIONS_CONNECTED)
            elif self._state == RuntimeState.ONE_OR_MORE_SESSIONS_CONNECTED:
                pass
            else:
                raise RuntimeError(f"Bad Runtime state at start: {self._state}")

            # Signal that we're started and ready to accept sessions
            async_objs.started.set_result(None)

            while not async_objs.must_stop.is_set():
                if self._state == RuntimeState.NO_SESSIONS_CONNECTED:
                    await asyncio.wait(
                        (
                            asyncio.create_task(async_objs.must_stop.wait()),
                            asyncio.create_task(async_objs.has_connection.wait()),
                        ),
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                elif self._state == RuntimeState.ONE_OR_MORE_SESSIONS_CONNECTED:
                    async_objs.need_send_data.clear()

                    for active_session_info in self._session_mgr.list_active_sessions():
                        msg_list = active_session_info.session.flush_browser_queue()
                        for msg in msg_list:
                            try:
                                self._send_message(active_session_info, msg)
                            except SessionClientDisconnectedError:
                                self._session_mgr.disconnect_session(
                                    active_session_info.session.id
                                )

                            # Yield for a tick after sending a message.
                            await asyncio.sleep(0)

                    # Yield for a few milliseconds between session message
                    # flushing.
                    await asyncio.sleep(0.01)

                else:
                    # Break out of the thread loop if we encounter any other state.
                    break

                await asyncio.wait(
                    (
                        asyncio.create_task(async_objs.must_stop.wait()),
                        asyncio.create_task(async_objs.need_send_data.wait()),
                    ),
                    return_when=asyncio.FIRST_COMPLETED,
                )

            # Shut down all AppSessions.
            for session_info in self._session_mgr.list_sessions():
                # NOTE: We want to fully shut down sessions when the runtime stops for
                # now, but this may change in the future if/when our notion of a session
                # is no longer so tightly coupled to a browser tab.
                self._session_mgr.close_session(session_info.session.id)

            self._set_state(RuntimeState.STOPPED)
            async_objs.stopped.set_result(None)

        except Exception as e:
            async_objs.stopped.set_exception(e)
            traceback.print_exc()
            LOGGER.info(
                """
Please report this bug at https://github.com/streamlit/streamlit/issues.
"""
            )

    def _send_message(self, session_info: ActiveSessionInfo, msg: ForwardMsg) -> None:
        """Send a message to a client.

        If the client is likely to have already cached the message, we may
        instead send a "reference" message that contains only the hash of the
        message.

        Parameters
        ----------
        session_info : ActiveSessionInfo
            The ActiveSessionInfo associated with websocket
        msg : ForwardMsg
            The message to send to the client

        Notes
        -----
        Threading: UNSAFE. Must be called on the eventloop thread.
        """
        msg.metadata.cacheable = is_cacheable_msg(msg)
        msg_to_send = msg
        if msg.metadata.cacheable:
            populate_hash_if_needed(msg)

            if self._message_cache.has_message_reference(
                msg, session_info.session, session_info.script_run_count
            ):
                # This session has probably cached this message. Send
                # a reference instead.
                LOGGER.debug("Sending cached message ref (hash=%s)", msg.hash)
                msg_to_send = create_reference_msg(msg)

            # Cache the message so it can be referenced in the future.
            # If the message is already cached, this will reset its
            # age.
            LOGGER.debug("Caching message (hash=%s)", msg.hash)
            self._message_cache.add_message(
                msg, session_info.session, session_info.script_run_count
            )

        # If this was a `script_finished` message, we increment the
        # script_run_count for this session, and update the cache
        if (
            msg.WhichOneof("type") == "script_finished"
            and msg.script_finished == ForwardMsg.FINISHED_SUCCESSFULLY
        ):
            LOGGER.debug(
                "Script run finished successfully; "
                "removing expired entries from MessageCache "
                "(max_age=%s)",
                config.get_option("global.maxCachedMessageAge"),
            )
            session_info.script_run_count += 1
            self._message_cache.remove_expired_session_entries(
                session_info.session, session_info.script_run_count
            )

        # Ship it off!
        session_info.client.write_forward_msg(msg_to_send)

    def _enqueued_some_message(self) -> None:
        """Callback called by AppSession after the AppSession has enqueued a
        message. Sets the "needs_send_data" event, which causes our core
        loop to wake up and flush client message queues.

        Notes
        -----
        Threading: SAFE. May be called on any thread.
        """
        async_objs = self._get_async_objs()
        async_objs.eventloop.call_soon_threadsafe(async_objs.need_send_data.set)

    def _get_async_objs(self) -> AsyncObjects:
        """Return our AsyncObjects instance. If the Runtime hasn't been
        started, this will raise an error.
        """
        if self._async_objs is None:
            raise RuntimeError("Runtime hasn't started yet!")
        return self._async_objs

    def _on_session_disconnected(self) -> None:
        """Set the runtime state to NO_SESSIONS_CONNECTED if the last active
        session was disconnected.
        """
        if (
            self._state == RuntimeState.ONE_OR_MORE_SESSIONS_CONNECTED
            and self._session_mgr.num_active_sessions() == 0
        ):
            self._get_async_objs().has_connection.clear()
            self._set_state(RuntimeState.NO_SESSIONS_CONNECTED)

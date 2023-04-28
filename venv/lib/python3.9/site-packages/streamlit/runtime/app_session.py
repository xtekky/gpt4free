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
import asyncio
import sys
import uuid
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

import streamlit.elements.exception as exception_utils
from streamlit import config, runtime, source_util
from streamlit.case_converters import to_snake_case
from streamlit.logger import get_logger
from streamlit.proto.BackMsg_pb2 import BackMsg
from streamlit.proto.ClientState_pb2 import ClientState
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.proto.GitInfo_pb2 import GitInfo
from streamlit.proto.NewSession_pb2 import (
    Config,
    CustomThemeConfig,
    NewSession,
    UserInfo,
)
from streamlit.proto.PagesChanged_pb2 import PagesChanged
from streamlit.runtime import caching, legacy_caching
from streamlit.runtime.credentials import Credentials
from streamlit.runtime.forward_msg_queue import ForwardMsgQueue
from streamlit.runtime.metrics_util import Installation
from streamlit.runtime.script_data import ScriptData
from streamlit.runtime.scriptrunner import RerunData, ScriptRunner, ScriptRunnerEvent
from streamlit.runtime.secrets import secrets_singleton
from streamlit.runtime.uploaded_file_manager import UploadedFileManager
from streamlit.version import STREAMLIT_VERSION_STRING
from streamlit.watcher import LocalSourcesWatcher

LOGGER = get_logger(__name__)
if TYPE_CHECKING:
    from streamlit.runtime.state import SessionState


class AppSessionState(Enum):
    APP_NOT_RUNNING = "APP_NOT_RUNNING"
    APP_IS_RUNNING = "APP_IS_RUNNING"
    SHUTDOWN_REQUESTED = "SHUTDOWN_REQUESTED"


def _generate_scriptrun_id() -> str:
    """Randomly generate a unique ID for a script execution."""
    return str(uuid.uuid4())


class AppSession:
    """
    Contains session data for a single "user" of an active app
    (that is, a connected browser tab).

    Each AppSession has its own ScriptData, root DeltaGenerator, ScriptRunner,
    and widget state.

    An AppSession is attached to each thread involved in running its script.

    """

    def __init__(
        self,
        script_data: ScriptData,
        uploaded_file_manager: UploadedFileManager,
        message_enqueued_callback: Optional[Callable[[], None]],
        local_sources_watcher: LocalSourcesWatcher,
        user_info: Dict[str, Optional[str]],
    ) -> None:
        """Initialize the AppSession.

        Parameters
        ----------
        script_data : ScriptData
            Object storing parameters related to running a script

        uploaded_file_manager : UploadedFileManager
            Used to manage files uploaded by users via the Streamlit web client.

        message_enqueued_callback : Callable[[], None]
            After enqueuing a message, this callable notification will be invoked.

        local_sources_watcher: LocalSourcesWatcher
            The file watcher that lets the session know local files have changed.

        user_info: Dict
            A dict that contains information about the current user. For now,
            it only contains the user's email address.

            {
                "email": "example@example.com"
            }

            Information about the current user is optionally provided when a
            websocket connection is initialized via the "X-Streamlit-User" header.

        """
        # Each AppSession has a unique string ID.
        self.id = str(uuid.uuid4())

        self._event_loop = asyncio.get_running_loop()
        self._script_data = script_data
        self._uploaded_file_mgr = uploaded_file_manager

        # The browser queue contains messages that haven't yet been
        # delivered to the browser. Periodically, the server flushes
        # this queue and delivers its contents to the browser.
        self._browser_queue = ForwardMsgQueue()
        self._message_enqueued_callback = message_enqueued_callback

        self._state = AppSessionState.APP_NOT_RUNNING

        # Need to remember the client state here because when a script reruns
        # due to the source code changing we need to pass in the previous client state.
        self._client_state = ClientState()

        self._local_sources_watcher: Optional[
            LocalSourcesWatcher
        ] = local_sources_watcher
        self._stop_config_listener: Optional[Callable[[], bool]] = None
        self._stop_pages_listener: Optional[Callable[[], bool]] = None

        self.register_file_watchers()

        self._run_on_save = config.get_option("server.runOnSave")

        self._scriptrunner: Optional[ScriptRunner] = None

        # This needs to be lazily imported to avoid a dependency cycle.
        from streamlit.runtime.state import SessionState

        self._session_state = SessionState()
        self._user_info = user_info

        self._debug_last_backmsg_id: Optional[str] = None

        LOGGER.debug("AppSession initialized (id=%s)", self.id)

    def __del__(self) -> None:
        """Ensure that we call shutdown() when an AppSession is garbage collected."""
        self.shutdown()

    def register_file_watchers(self) -> None:
        """Register handlers to be called when various files are changed.

        Files that we watch include:
          * source files that already exist (for edits)
          * `.py` files in the the main script's `pages/` directory (for file additions
            and deletions)
          * project and user-level config.toml files
          * the project-level secrets.toml files

        This method is called automatically on AppSession construction, but it may be
        called again in the case when a session is disconnected and is being reconnect
        to.
        """
        if self._local_sources_watcher is None:
            self._local_sources_watcher = LocalSourcesWatcher(
                self._script_data.main_script_path
            )

        self._local_sources_watcher.register_file_change_callback(
            self._on_source_file_changed
        )
        self._stop_config_listener = config.on_config_parsed(
            self._on_source_file_changed, force_connect=True
        )
        self._stop_pages_listener = source_util.register_pages_changed_callback(
            self._on_pages_changed
        )
        secrets_singleton.file_change_listener.connect(self._on_secrets_file_changed)

    def disconnect_file_watchers(self) -> None:
        """Disconnect the file watcher handlers registered by register_file_watchers."""
        if self._local_sources_watcher is not None:
            self._local_sources_watcher.close()
        if self._stop_config_listener is not None:
            self._stop_config_listener()
        if self._stop_pages_listener is not None:
            self._stop_pages_listener()

        secrets_singleton.file_change_listener.disconnect(self._on_secrets_file_changed)

        self._local_sources_watcher = None
        self._stop_config_listener = None
        self._stop_pages_listener = None

    def flush_browser_queue(self) -> List[ForwardMsg]:
        """Clear the forward message queue and return the messages it contained.

        The Server calls this periodically to deliver new messages
        to the browser connected to this app.

        Returns
        -------
        list[ForwardMsg]
            The messages that were removed from the queue and should
            be delivered to the browser.

        """
        return self._browser_queue.flush()

    def shutdown(self) -> None:
        """Shut down the AppSession.

        It's an error to use a AppSession after it's been shut down.

        """
        if self._state != AppSessionState.SHUTDOWN_REQUESTED:
            LOGGER.debug("Shutting down (id=%s)", self.id)
            # Clear any unused session files in upload file manager and media
            # file manager
            self._uploaded_file_mgr.remove_session_files(self.id)

            if runtime.exists():
                runtime.get_instance().media_file_mgr.clear_session_refs(self.id)
                runtime.get_instance().media_file_mgr.remove_orphaned_files()

            # Shut down the ScriptRunner, if one is active.
            # self._state must not be set to SHUTDOWN_REQUESTED until
            # *after* this is called.
            self.request_script_stop()

            self._state = AppSessionState.SHUTDOWN_REQUESTED

            # Disconnect all file watchers if we haven't already, although we will have
            # generally already done so by the time we get here.
            self.disconnect_file_watchers()

    def _enqueue_forward_msg(self, msg: ForwardMsg) -> None:
        """Enqueue a new ForwardMsg to our browser queue.

        This can be called on both the main thread and a ScriptRunner
        run thread.

        Parameters
        ----------
        msg : ForwardMsg
            The message to enqueue

        """
        if not config.get_option("client.displayEnabled"):
            return

        if self._debug_last_backmsg_id:
            msg.debug_last_backmsg_id = self._debug_last_backmsg_id

        self._browser_queue.enqueue(msg)
        if self._message_enqueued_callback:
            self._message_enqueued_callback()

    def handle_backmsg(self, msg: BackMsg) -> None:
        """Process a BackMsg."""
        try:
            msg_type = msg.WhichOneof("type")

            if msg_type == "rerun_script":
                if msg.debug_last_backmsg_id:
                    self._debug_last_backmsg_id = msg.debug_last_backmsg_id

                self._handle_rerun_script_request(msg.rerun_script)
            elif msg_type == "load_git_info":
                self._handle_git_information_request()
            elif msg_type == "clear_cache":
                self._handle_clear_cache_request()
            elif msg_type == "set_run_on_save":
                self._handle_set_run_on_save_request(msg.set_run_on_save)
            elif msg_type == "stop_script":
                self._handle_stop_script_request()
            else:
                LOGGER.warning('No handler for "%s"', msg_type)

        except Exception as ex:
            LOGGER.error(ex)
            self.handle_backmsg_exception(ex)

    def handle_backmsg_exception(self, e: BaseException) -> None:
        """Handle an Exception raised while processing a BackMsg from the browser."""
        # This does a few things:
        # 1) Clears the current app in the browser.
        # 2) Marks the current app as "stopped" in the browser.
        # 3) HACK: Resets any script params that may have been broken (e.g. the
        # command-line when rerunning with wrong argv[0])

        self._on_scriptrunner_event(
            self._scriptrunner, ScriptRunnerEvent.SCRIPT_STOPPED_WITH_SUCCESS
        )
        self._on_scriptrunner_event(
            self._scriptrunner,
            ScriptRunnerEvent.SCRIPT_STARTED,
            page_script_hash="",
        )
        self._on_scriptrunner_event(
            self._scriptrunner, ScriptRunnerEvent.SCRIPT_STOPPED_WITH_SUCCESS
        )

        # Send an Exception message to the frontend.
        # Because _on_scriptrunner_event does its work in an eventloop callback,
        # this exception ForwardMsg *must* also be enqueued in a callback,
        # so that it will be enqueued *after* the various ForwardMsgs that
        # _on_scriptrunner_event sends.
        self._event_loop.call_soon_threadsafe(
            lambda: self._enqueue_forward_msg(self._create_exception_message(e))
        )

    def request_rerun(self, client_state: Optional[ClientState]) -> None:
        """Signal that we're interested in running the script.

        If the script is not already running, it will be started immediately.
        Otherwise, a rerun will be requested.

        Parameters
        ----------
        client_state : streamlit.proto.ClientState_pb2.ClientState | None
            The ClientState protobuf to run the script with, or None
            to use previous client state.

        """
        if self._state == AppSessionState.SHUTDOWN_REQUESTED:
            LOGGER.warning("Discarding rerun request after shutdown")
            return

        if client_state:
            rerun_data = RerunData(
                client_state.query_string,
                client_state.widget_states,
                client_state.page_script_hash,
                client_state.page_name,
            )
        else:
            rerun_data = RerunData()

        if self._scriptrunner is not None:
            if bool(config.get_option("runner.fastReruns")):
                # If fastReruns is enabled, we don't send rerun requests to our
                # existing ScriptRunner. Instead, we tell it to shut down. We'll
                # then spin up a new ScriptRunner, below, to handle the rerun
                # immediately.
                self._scriptrunner.request_stop()
                self._scriptrunner = None
            else:
                # fastReruns is not enabled. Send our ScriptRunner a rerun
                # request. If the request is accepted, we're done.
                success = self._scriptrunner.request_rerun(rerun_data)
                if success:
                    return

        # If we are here, then either we have no ScriptRunner, or our
        # current ScriptRunner is shutting down and cannot handle a rerun
        # request - so we'll create and start a new ScriptRunner.
        self._create_scriptrunner(rerun_data)

    def request_script_stop(self) -> None:
        """Request that the scriptrunner stop execution.

        Does nothing if no scriptrunner exists.
        """
        if self._scriptrunner is not None:
            self._scriptrunner.request_stop()

    def _create_scriptrunner(self, initial_rerun_data: RerunData) -> None:
        """Create and run a new ScriptRunner with the given RerunData."""
        self._scriptrunner = ScriptRunner(
            session_id=self.id,
            main_script_path=self._script_data.main_script_path,
            client_state=self._client_state,
            session_state=self._session_state,
            uploaded_file_mgr=self._uploaded_file_mgr,
            initial_rerun_data=initial_rerun_data,
            user_info=self._user_info,
        )
        self._scriptrunner.on_event.connect(self._on_scriptrunner_event)
        self._scriptrunner.start()

    @property
    def session_state(self) -> "SessionState":
        return self._session_state

    def _should_rerun_on_file_change(self, filepath: str) -> bool:
        main_script_path = self._script_data.main_script_path
        pages = source_util.get_pages(main_script_path)

        changed_page_script_hash = next(
            filter(lambda k: pages[k]["script_path"] == filepath, pages),
            None,
        )

        if changed_page_script_hash is not None:
            current_page_script_hash = self._client_state.page_script_hash
            return changed_page_script_hash == current_page_script_hash

        return True

    def _on_source_file_changed(self, filepath: Optional[str] = None) -> None:
        """One of our source files changed. Schedule a rerun if appropriate."""
        if filepath is not None and not self._should_rerun_on_file_change(filepath):
            return

        if self._run_on_save:
            self.request_rerun(self._client_state)
        else:
            self._enqueue_forward_msg(self._create_file_change_message())

    def _on_secrets_file_changed(self, _) -> None:
        """Called when `secrets.file_change_listener` emits a Signal."""

        # NOTE: At the time of writing, this function only calls `_on_source_file_changed`.
        # The reason behind creating this function instead of just passing `_on_source_file_changed`
        # to `connect` / `disconnect` directly is that every function that is passed to `connect` / `disconnect`
        # must have at least one argument for `sender` (in this case we don't really care about it, thus `_`),
        # and introducing an unnecessary argument to `_on_source_file_changed` just for this purpose sounded finicky.
        self._on_source_file_changed()

    def _on_pages_changed(self, _) -> None:
        msg = ForwardMsg()
        _populate_app_pages(msg.pages_changed, self._script_data.main_script_path)
        self._enqueue_forward_msg(msg)

    def _clear_queue(self) -> None:
        self._browser_queue.clear()

    def _on_scriptrunner_event(
        self,
        sender: Optional[ScriptRunner],
        event: ScriptRunnerEvent,
        forward_msg: Optional[ForwardMsg] = None,
        exception: Optional[BaseException] = None,
        client_state: Optional[ClientState] = None,
        page_script_hash: Optional[str] = None,
    ) -> None:
        """Called when our ScriptRunner emits an event.

        This is generally called from the sender ScriptRunner's script thread.
        We forward the event on to _handle_scriptrunner_event_on_event_loop,
        which will be called on the main thread.
        """
        self._event_loop.call_soon_threadsafe(
            lambda: self._handle_scriptrunner_event_on_event_loop(
                sender, event, forward_msg, exception, client_state, page_script_hash
            )
        )

    def _handle_scriptrunner_event_on_event_loop(
        self,
        sender: Optional[ScriptRunner],
        event: ScriptRunnerEvent,
        forward_msg: Optional[ForwardMsg] = None,
        exception: Optional[BaseException] = None,
        client_state: Optional[ClientState] = None,
        page_script_hash: Optional[str] = None,
    ) -> None:
        """Handle a ScriptRunner event.

        This function must only be called on our eventloop thread.

        Parameters
        ----------
        sender : ScriptRunner | None
            The ScriptRunner that emitted the event. (This may be set to
            None when called from `handle_backmsg_exception`, if no
            ScriptRunner was active when the backmsg exception was raised.)

        event : ScriptRunnerEvent
            The event type.

        forward_msg : ForwardMsg | None
            The ForwardMsg to send to the frontend. Set only for the
            ENQUEUE_FORWARD_MSG event.

        exception : BaseException | None
            An exception thrown during compilation. Set only for the
            SCRIPT_STOPPED_WITH_COMPILE_ERROR event.

        client_state : streamlit.proto.ClientState_pb2.ClientState | None
            The ScriptRunner's final ClientState. Set only for the
            SHUTDOWN event.

        page_script_hash : str | None
            A hash of the script path corresponding to the page currently being
            run. Set only for the SCRIPT_STARTED event.
        """

        assert (
            self._event_loop == asyncio.get_running_loop()
        ), "This function must only be called on the eventloop thread the AppSession was created on."

        if sender is not self._scriptrunner:
            # This event was sent by a non-current ScriptRunner; ignore it.
            # This can happen after sppinng up a new ScriptRunner (to handle a
            # rerun request, for example) while another ScriptRunner is still
            # shutting down. The shutting-down ScriptRunner may still
            # emit events.
            LOGGER.debug("Ignoring event from non-current ScriptRunner: %s", event)
            return

        prev_state = self._state

        if event == ScriptRunnerEvent.SCRIPT_STARTED:
            if self._state != AppSessionState.SHUTDOWN_REQUESTED:
                self._state = AppSessionState.APP_IS_RUNNING

            assert (
                page_script_hash is not None
            ), "page_script_hash must be set for the SCRIPT_STARTED event"

            self._clear_queue()
            self._enqueue_forward_msg(
                self._create_new_session_message(page_script_hash)
            )

        elif (
            event == ScriptRunnerEvent.SCRIPT_STOPPED_WITH_SUCCESS
            or event == ScriptRunnerEvent.SCRIPT_STOPPED_WITH_COMPILE_ERROR
        ):
            if self._state != AppSessionState.SHUTDOWN_REQUESTED:
                self._state = AppSessionState.APP_NOT_RUNNING

            script_succeeded = event == ScriptRunnerEvent.SCRIPT_STOPPED_WITH_SUCCESS

            script_finished_msg = self._create_script_finished_message(
                ForwardMsg.FINISHED_SUCCESSFULLY
                if script_succeeded
                else ForwardMsg.FINISHED_WITH_COMPILE_ERROR
            )
            self._enqueue_forward_msg(script_finished_msg)

            self._debug_last_backmsg_id = None

            if script_succeeded:
                # The script completed successfully: update our
                # LocalSourcesWatcher to account for any source code changes
                # that change which modules should be watched.
                if self._local_sources_watcher:
                    self._local_sources_watcher.update_watched_modules()
            else:
                # The script didn't complete successfully: send the exception
                # to the frontend.
                assert (
                    exception is not None
                ), "exception must be set for the SCRIPT_STOPPED_WITH_COMPILE_ERROR event"
                msg = ForwardMsg()
                exception_utils.marshall(
                    msg.session_event.script_compilation_exception, exception
                )
                self._enqueue_forward_msg(msg)

        elif event == ScriptRunnerEvent.SCRIPT_STOPPED_FOR_RERUN:
            script_finished_msg = self._create_script_finished_message(
                ForwardMsg.FINISHED_EARLY_FOR_RERUN
            )
            self._enqueue_forward_msg(script_finished_msg)
            if self._local_sources_watcher:
                self._local_sources_watcher.update_watched_modules()

        elif event == ScriptRunnerEvent.SHUTDOWN:
            assert (
                client_state is not None
            ), "client_state must be set for the SHUTDOWN event"

            if self._state == AppSessionState.SHUTDOWN_REQUESTED:
                # Only clear media files if the script is done running AND the
                # session is actually shutting down.
                runtime.get_instance().media_file_mgr.clear_session_refs(self.id)

            self._client_state = client_state
            self._scriptrunner = None

        elif event == ScriptRunnerEvent.ENQUEUE_FORWARD_MSG:
            assert (
                forward_msg is not None
            ), "null forward_msg in ENQUEUE_FORWARD_MSG event"
            self._enqueue_forward_msg(forward_msg)

        # Send a message if our run state changed
        app_was_running = prev_state == AppSessionState.APP_IS_RUNNING
        app_is_running = self._state == AppSessionState.APP_IS_RUNNING
        if app_is_running != app_was_running:
            self._enqueue_forward_msg(self._create_session_status_changed_message())

    def _create_session_status_changed_message(self) -> ForwardMsg:
        """Create and return a session_status_changed ForwardMsg."""
        msg = ForwardMsg()
        msg.session_status_changed.run_on_save = self._run_on_save
        msg.session_status_changed.script_is_running = (
            self._state == AppSessionState.APP_IS_RUNNING
        )
        return msg

    def _create_file_change_message(self) -> ForwardMsg:
        """Create and return a 'script_changed_on_disk' ForwardMsg."""
        msg = ForwardMsg()
        msg.session_event.script_changed_on_disk = True
        return msg

    def _create_new_session_message(self, page_script_hash: str) -> ForwardMsg:
        """Create and return a new_session ForwardMsg."""
        msg = ForwardMsg()

        msg.new_session.script_run_id = _generate_scriptrun_id()
        msg.new_session.name = self._script_data.name
        msg.new_session.main_script_path = self._script_data.main_script_path
        msg.new_session.page_script_hash = page_script_hash

        _populate_app_pages(msg.new_session, self._script_data.main_script_path)
        _populate_config_msg(msg.new_session.config)
        _populate_theme_msg(msg.new_session.custom_theme)

        # Immutable session data. We send this every time a new session is
        # started, to avoid having to track whether the client has already
        # received it. It does not change from run to run; it's up to the
        # to perform one-time initialization only once.
        imsg = msg.new_session.initialize

        _populate_user_info_msg(imsg.user_info)

        imsg.environment_info.streamlit_version = STREAMLIT_VERSION_STRING
        imsg.environment_info.python_version = ".".join(map(str, sys.version_info))

        imsg.session_status.run_on_save = self._run_on_save
        imsg.session_status.script_is_running = (
            self._state == AppSessionState.APP_IS_RUNNING
        )

        imsg.command_line = self._script_data.command_line
        imsg.session_id = self.id

        return msg

    def _create_script_finished_message(
        self, status: "ForwardMsg.ScriptFinishedStatus.ValueType"
    ) -> ForwardMsg:
        """Create and return a script_finished ForwardMsg."""
        msg = ForwardMsg()
        msg.script_finished = status
        return msg

    def _create_exception_message(self, e: BaseException) -> ForwardMsg:
        """Create and return an Exception ForwardMsg."""
        msg = ForwardMsg()
        exception_utils.marshall(msg.delta.new_element.exception, e)
        return msg

    def _handle_git_information_request(self) -> None:
        msg = ForwardMsg()

        try:
            from streamlit.git_util import GitRepo

            repo = GitRepo(self._script_data.main_script_path)

            repo_info = repo.get_repo_info()
            if repo_info is None:
                return

            repository_name, branch, module = repo_info

            msg.git_info_changed.repository = repository_name
            msg.git_info_changed.branch = branch
            msg.git_info_changed.module = module

            msg.git_info_changed.untracked_files[:] = repo.untracked_files
            msg.git_info_changed.uncommitted_files[:] = repo.uncommitted_files

            if repo.is_head_detached:
                msg.git_info_changed.state = GitInfo.GitStates.HEAD_DETACHED
            elif len(repo.ahead_commits) > 0:
                msg.git_info_changed.state = GitInfo.GitStates.AHEAD_OF_REMOTE
            else:
                msg.git_info_changed.state = GitInfo.GitStates.DEFAULT

            self._enqueue_forward_msg(msg)
        except Exception as ex:
            # Users may never even install Git in the first place, so this
            # error requires no action. It can be useful for debugging.
            LOGGER.debug("Obtaining Git information produced an error", exc_info=ex)

    def _handle_rerun_script_request(
        self, client_state: Optional[ClientState] = None
    ) -> None:
        """Tell the ScriptRunner to re-run its script.

        Parameters
        ----------
        client_state : streamlit.proto.ClientState_pb2.ClientState | None
            The ClientState protobuf to run the script with, or None
            to use previous client state.

        """
        self.request_rerun(client_state)

    def _handle_stop_script_request(self) -> None:
        """Tell the ScriptRunner to stop running its script."""
        self.request_script_stop()

    def _handle_clear_cache_request(self) -> None:
        """Clear this app's cache.

        Because this cache is global, it will be cleared for all users.

        """
        legacy_caching.clear_cache()
        caching.cache_data.clear()
        caching.cache_resource.clear()
        self._session_state.clear()

    def _handle_set_run_on_save_request(self, new_value: bool) -> None:
        """Change our run_on_save flag to the given value.

        The browser will be notified of the change.

        Parameters
        ----------
        new_value : bool
            New run_on_save value

        """
        self._run_on_save = new_value
        self._enqueue_forward_msg(self._create_session_status_changed_message())


def _populate_config_msg(msg: Config) -> None:
    msg.gather_usage_stats = config.get_option("browser.gatherUsageStats")
    msg.max_cached_message_age = config.get_option("global.maxCachedMessageAge")
    msg.mapbox_token = config.get_option("mapbox.token")
    msg.allow_run_on_save = config.get_option("server.allowRunOnSave")
    msg.hide_top_bar = config.get_option("ui.hideTopBar")
    msg.hide_sidebar_nav = config.get_option("ui.hideSidebarNav")


def _populate_theme_msg(msg: CustomThemeConfig) -> None:
    enum_encoded_options = {"base", "font"}
    theme_opts = config.get_options_for_section("theme")

    if not any(theme_opts.values()):
        return

    for option_name, option_val in theme_opts.items():
        if option_name not in enum_encoded_options and option_val is not None:
            setattr(msg, to_snake_case(option_name), option_val)

    # NOTE: If unset, base and font will default to the protobuf enum zero
    # values, which are BaseTheme.LIGHT and FontFamily.SANS_SERIF,
    # respectively. This is why we both don't handle the cases explicitly and
    # also only log a warning when receiving invalid base/font options.
    base_map = {
        "light": msg.BaseTheme.LIGHT,
        "dark": msg.BaseTheme.DARK,
    }
    base = theme_opts["base"]
    if base is not None:
        if base not in base_map:
            LOGGER.warning(
                f'"{base}" is an invalid value for theme.base.'
                f" Allowed values include {list(base_map.keys())}."
                ' Setting theme.base to "light".'
            )
        else:
            msg.base = base_map[base]

    font_map = {
        "sans serif": msg.FontFamily.SANS_SERIF,
        "serif": msg.FontFamily.SERIF,
        "monospace": msg.FontFamily.MONOSPACE,
    }
    font = theme_opts["font"]
    if font is not None:
        if font not in font_map:
            LOGGER.warning(
                f'"{font}" is an invalid value for theme.font.'
                f" Allowed values include {list(font_map.keys())}."
                ' Setting theme.font to "sans serif".'
            )
        else:
            msg.font = font_map[font]


def _populate_user_info_msg(msg: UserInfo) -> None:
    msg.installation_id = Installation.instance().installation_id
    msg.installation_id_v3 = Installation.instance().installation_id_v3
    if Credentials.get_current().activation:
        msg.email = Credentials.get_current().activation.email
    else:
        msg.email = ""


def _populate_app_pages(
    msg: Union[NewSession, PagesChanged], main_script_path: str
) -> None:
    for page_script_hash, page_info in source_util.get_pages(main_script_path).items():
        page_proto = msg.app_pages.add()

        page_proto.page_script_hash = page_script_hash
        page_proto.page_name = page_info["page_name"]
        page_proto.icon = page_info["icon"]

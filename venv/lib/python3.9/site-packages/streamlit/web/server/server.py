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

import errno
import logging
import os
import socket
import ssl
import sys
from pathlib import Path
from typing import Any, Awaitable, List, Optional, Union

import click
import tornado.concurrent
import tornado.locks
import tornado.netutil
import tornado.web
import tornado.websocket
from tornado.httpserver import HTTPServer
from typing_extensions import Final

from streamlit import config, file_util, source_util, util
from streamlit.components.v1.components import ComponentRegistry
from streamlit.config_option import ConfigOption
from streamlit.logger import get_logger
from streamlit.runtime import Runtime, RuntimeConfig, RuntimeState
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage
from streamlit.runtime.runtime_util import get_max_message_size_bytes
from streamlit.web.cache_storage_manager_config import (
    create_default_cache_storage_manager,
)
from streamlit.web.server.app_static_file_handler import AppStaticFileHandler
from streamlit.web.server.browser_websocket_handler import BrowserWebSocketHandler
from streamlit.web.server.component_request_handler import ComponentRequestHandler
from streamlit.web.server.media_file_handler import MediaFileHandler
from streamlit.web.server.routes import (
    AddSlashHandler,
    AllowedMessageOriginsHandler,
    AssetsFileHandler,
    HealthHandler,
    MessageCacheHandler,
    StaticFileHandler,
)
from streamlit.web.server.server_util import make_url_path_regex
from streamlit.web.server.stats_request_handler import StatsRequestHandler
from streamlit.web.server.upload_file_request_handler import (
    UPLOAD_FILE_ROUTE,
    UploadFileRequestHandler,
)

LOGGER = get_logger(__name__)

TORNADO_SETTINGS = {
    # Gzip HTTP responses.
    "compress_response": True,
    # Ping every 1s to keep WS alive.
    # 2021.06.22: this value was previously 20s, and was causing
    # connection instability for a small number of users. This smaller
    # ping_interval fixes that instability.
    # https://github.com/streamlit/streamlit/issues/3196
    "websocket_ping_interval": 1,
    # If we don't get a ping response within 30s, the connection
    # is timed out.
    "websocket_ping_timeout": 30,
}

# When server.port is not available it will look for the next available port
# up to MAX_PORT_SEARCH_RETRIES.
MAX_PORT_SEARCH_RETRIES = 100

# When server.address starts with this prefix, the server will bind
# to an unix socket.
UNIX_SOCKET_PREFIX = "unix://"

MEDIA_ENDPOINT: Final = "/media"
STREAM_ENDPOINT: Final = r"_stcore/stream"
METRIC_ENDPOINT: Final = r"(?:st-metrics|_stcore/metrics)"
MESSAGE_ENDPOINT: Final = r"_stcore/message"
HEALTH_ENDPOINT: Final = r"(?:healthz|_stcore/health)"
ALLOWED_MESSAGE_ORIGIN_ENDPOINT: Final = r"_stcore/allowed-message-origins"
SCRIPT_HEALTH_CHECK_ENDPOINT: Final = (
    r"(?:script-health-check|_stcore/script-health-check)"
)


class RetriesExceeded(Exception):
    pass


def server_port_is_manually_set() -> bool:
    return config.is_manually_set("server.port")


def server_address_is_unix_socket() -> bool:
    address = config.get_option("server.address")
    return address is not None and address.startswith(UNIX_SOCKET_PREFIX)


def start_listening(app: tornado.web.Application) -> None:
    """Makes the server start listening at the configured port.

    In case the port is already taken it tries listening to the next available
    port.  It will error after MAX_PORT_SEARCH_RETRIES attempts.

    """
    cert_file = config.get_option("server.sslCertFile")
    key_file = config.get_option("server.sslKeyFile")
    ssl_options = _get_ssl_options(cert_file, key_file)

    http_server = HTTPServer(
        app,
        max_buffer_size=config.get_option("server.maxUploadSize") * 1024 * 1024,
        ssl_options=ssl_options,
    )

    if server_address_is_unix_socket():
        start_listening_unix_socket(http_server)
    else:
        start_listening_tcp_socket(http_server)


def _get_ssl_options(
    cert_file: Optional[str], key_file: Optional[str]
) -> Union[ssl.SSLContext, None]:
    if bool(cert_file) != bool(key_file):
        LOGGER.error(
            "Options 'server.sslCertFile' and 'server.sslKeyFile' must "
            "be set together. Set missing options or delete existing options."
        )
        sys.exit(1)
    if cert_file and key_file:
        # ssl_ctx.load_cert_chain raise exception as below, but it is not
        # sufficiently user-friendly
        # FileNotFoundError: [Errno 2] No such file or directory
        if not Path(cert_file).exists():
            LOGGER.error("Cert file '%s' does not exist.", cert_file)
            sys.exit(1)
        if not Path(key_file).exists():
            LOGGER.error("Key file '%s' does not exist.", key_file)
            sys.exit(1)

        ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        # When the SSL certificate fails to load, an exception is raised as below,
        # but it is not sufficiently user-friendly.
        # ssl.SSLError: [SSL] PEM lib (_ssl.c:4067)
        try:
            ssl_ctx.load_cert_chain(cert_file, key_file)
        except ssl.SSLError:
            LOGGER.error(
                "Failed to load SSL certificate. Make sure "
                "cert file '%s' and key file '%s' are correct.",
                cert_file,
                key_file,
            )
            sys.exit(1)

        return ssl_ctx
    return None


def start_listening_unix_socket(http_server: HTTPServer) -> None:
    address = config.get_option("server.address")
    file_name = os.path.expanduser(address[len(UNIX_SOCKET_PREFIX) :])

    unix_socket = tornado.netutil.bind_unix_socket(file_name)
    http_server.add_socket(unix_socket)


def start_listening_tcp_socket(http_server: HTTPServer) -> None:
    call_count = 0

    port = None
    while call_count < MAX_PORT_SEARCH_RETRIES:
        address = config.get_option("server.address")
        port = config.get_option("server.port")

        try:
            http_server.listen(port, address)
            break  # It worked! So let's break out of the loop.

        except (OSError, socket.error) as e:
            if e.errno == errno.EADDRINUSE:
                if server_port_is_manually_set():
                    LOGGER.error("Port %s is already in use", port)
                    sys.exit(1)
                else:
                    LOGGER.debug(
                        "Port %s already in use, trying to use the next one.", port
                    )
                    port += 1
                    # Save port 3000 because it is used for the development
                    # server in the front end.
                    if port == 3000:
                        port += 1

                    config.set_option(
                        "server.port", port, ConfigOption.STREAMLIT_DEFINITION
                    )
                    call_count += 1
            else:
                raise

    if call_count >= MAX_PORT_SEARCH_RETRIES:
        raise RetriesExceeded(
            f"Cannot start Streamlit server. Port {port} is already in use, and "
            f"Streamlit was unable to find a free port after {MAX_PORT_SEARCH_RETRIES} attempts.",
        )


class Server:
    def __init__(self, main_script_path: str, command_line: Optional[str]):
        """Create the server. It won't be started yet."""
        _set_tornado_log_levels()

        self._main_script_path = main_script_path

        # Initialize MediaFileStorage and its associated endpoint
        media_file_storage = MemoryMediaFileStorage(MEDIA_ENDPOINT)
        MediaFileHandler.initialize_storage(media_file_storage)

        self._runtime = Runtime(
            RuntimeConfig(
                script_path=main_script_path,
                command_line=command_line,
                media_file_storage=media_file_storage,
                cache_storage_manager=create_default_cache_storage_manager(),
            ),
        )

        self._runtime.stats_mgr.register_provider(media_file_storage)

    def __repr__(self) -> str:
        return util.repr_(self)

    @property
    def main_script_path(self) -> str:
        return self._main_script_path

    async def start(self) -> None:
        """Start the server.

        When this returns, Streamlit is ready to accept new sessions.
        """

        LOGGER.debug("Starting server...")

        app = self._create_app()
        start_listening(app)

        port = config.get_option("server.port")
        LOGGER.debug("Server started on port %s", port)

        await self._runtime.start()

    @property
    def stopped(self) -> Awaitable[None]:
        """A Future that completes when the Server's run loop has exited."""
        return self._runtime.stopped

    def _create_app(self) -> tornado.web.Application:
        """Create our tornado web app."""
        base = config.get_option("server.baseUrlPath")

        routes: List[Any] = [
            (
                make_url_path_regex(base, STREAM_ENDPOINT),
                BrowserWebSocketHandler,
                dict(runtime=self._runtime),
            ),
            (
                make_url_path_regex(base, HEALTH_ENDPOINT),
                HealthHandler,
                dict(callback=lambda: self._runtime.is_ready_for_browser_connection),
            ),
            (
                make_url_path_regex(base, MESSAGE_ENDPOINT),
                MessageCacheHandler,
                dict(cache=self._runtime.message_cache),
            ),
            (
                make_url_path_regex(base, METRIC_ENDPOINT),
                StatsRequestHandler,
                dict(stats_manager=self._runtime.stats_mgr),
            ),
            (
                make_url_path_regex(base, ALLOWED_MESSAGE_ORIGIN_ENDPOINT),
                AllowedMessageOriginsHandler,
            ),
            (
                make_url_path_regex(
                    base,
                    UPLOAD_FILE_ROUTE,
                ),
                UploadFileRequestHandler,
                dict(
                    file_mgr=self._runtime.uploaded_file_mgr,
                    is_active_session=self._runtime.is_active_session,
                ),
            ),
            (
                make_url_path_regex(base, "assets/(.*)"),
                AssetsFileHandler,
                {"path": "%s/" % file_util.get_assets_dir()},
            ),
            (
                make_url_path_regex(base, f"{MEDIA_ENDPOINT}/(.*)"),
                MediaFileHandler,
                {"path": ""},
            ),
            (
                make_url_path_regex(base, "component/(.*)"),
                ComponentRequestHandler,
                dict(registry=ComponentRegistry.instance()),
            ),
        ]

        if config.get_option("server.scriptHealthCheckEnabled"):
            routes.extend(
                [
                    (
                        make_url_path_regex(base, SCRIPT_HEALTH_CHECK_ENDPOINT),
                        HealthHandler,
                        dict(
                            callback=lambda: self._runtime.does_script_run_without_error()
                        ),
                    )
                ]
            )

        if config.get_option("server.enableStaticServing"):
            routes.extend(
                [
                    (
                        make_url_path_regex(base, "app/static/(.*)"),
                        AppStaticFileHandler,
                        {"path": file_util.get_app_static_dir(self.main_script_path)},
                    ),
                ]
            )

        if config.get_option("global.developmentMode"):
            LOGGER.debug("Serving static content from the Node dev server")
        else:
            static_path = file_util.get_static_dir()
            LOGGER.debug("Serving static content from %s", static_path)

            routes.extend(
                [
                    (
                        make_url_path_regex(base, "(.*)"),
                        StaticFileHandler,
                        {
                            "path": "%s/" % static_path,
                            "default_filename": "index.html",
                            "get_pages": lambda: set(
                                [
                                    page_info["page_name"]
                                    for page_info in source_util.get_pages(
                                        self.main_script_path
                                    ).values()
                                ]
                            ),
                        },
                    ),
                    (make_url_path_regex(base, trailing_slash=False), AddSlashHandler),
                ]
            )

        return tornado.web.Application(
            routes,
            cookie_secret=config.get_option("server.cookieSecret"),
            xsrf_cookies=config.get_option("server.enableXsrfProtection"),
            # Set the websocket message size. The default value is too low.
            websocket_max_message_size=get_max_message_size_bytes(),
            **TORNADO_SETTINGS,  # type: ignore[arg-type]
        )

    @property
    def browser_is_connected(self) -> bool:
        return self._runtime.state == RuntimeState.ONE_OR_MORE_SESSIONS_CONNECTED

    @property
    def is_running_hello(self) -> bool:
        from streamlit.hello import Hello

        return self._main_script_path == Hello.__file__

    def stop(self) -> None:
        click.secho("  Stopping...", fg="blue")
        self._runtime.stop()


def _set_tornado_log_levels() -> None:
    if not config.get_option("global.developmentMode"):
        # Hide logs unless they're super important.
        # Example of stuff we don't care about: 404 about .js.map files.
        logging.getLogger("tornado.access").setLevel(logging.ERROR)
        logging.getLogger("tornado.application").setLevel(logging.ERROR)
        logging.getLogger("tornado.general").setLevel(logging.ERROR)

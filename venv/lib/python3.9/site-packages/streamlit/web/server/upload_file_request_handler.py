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

from typing import Any, Callable, Dict, List

import tornado.httputil
import tornado.web

from streamlit import config
from streamlit.logger import get_logger
from streamlit.runtime.uploaded_file_manager import UploadedFileManager, UploadedFileRec
from streamlit.web.server import routes, server_util

# /_stcore/upload_file/(optional session id)/(optional widget id)
UPLOAD_FILE_ROUTE = (
    r"/_stcore/upload_file/?(?P<session_id>[^/]*)?/?(?P<widget_id>[^/]*)?"
)
LOGGER = get_logger(__name__)


class UploadFileRequestHandler(tornado.web.RequestHandler):
    """Implements the POST /upload_file endpoint."""

    def initialize(
        self, file_mgr: UploadedFileManager, is_active_session: Callable[[str], bool]
    ):
        """
        Parameters
        ----------
        file_mgr : UploadedFileManager
            The server's singleton UploadedFileManager. All file uploads
            go here.
        is_active_session:
            A function that returns true if a session_id belongs to an active
            session.
        """
        self._file_mgr = file_mgr
        self._is_active_session = is_active_session

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.set_header("Access-Control-Allow-Headers", "Content-Type")
        if config.get_option("server.enableXsrfProtection"):
            self.set_header(
                "Access-Control-Allow-Origin",
                server_util.get_url(config.get_option("browser.serverAddress")),
            )
            self.set_header("Access-Control-Allow-Headers", "X-Xsrftoken, Content-Type")
            self.set_header("Vary", "Origin")
            self.set_header("Access-Control-Allow-Credentials", "true")
        elif routes.allow_cross_origin_requests():
            self.set_header("Access-Control-Allow-Origin", "*")

    def options(self, **kwargs):
        """/OPTIONS handler for preflight CORS checks.

        When a browser is making a CORS request, it may sometimes first
        send an OPTIONS request, to check whether the server understands the
        CORS protocol. This is optional, and doesn't happen for every request
        or in every browser. If an OPTIONS request does get sent, and is not
        then handled by the server, the browser will fail the underlying
        request.

        The proper way to handle this is to send a 204 response ("no content")
        with the CORS headers attached. (These headers are automatically added
        to every outgoing response, including OPTIONS responses,
        via set_default_headers().)

        See https://developer.mozilla.org/en-US/docs/Glossary/Preflight_request
        """
        self.set_status(204)
        self.finish()

    @staticmethod
    def _require_arg(args: Dict[str, List[bytes]], name: str) -> str:
        """Return the value of the argument with the given name.

        A human-readable exception will be raised if the argument doesn't
        exist. This will be used as the body for the error response returned
        from the request.
        """
        try:
            arg = args[name]
        except KeyError:
            raise Exception(f"Missing '{name}'")

        if len(arg) != 1:
            raise Exception(f"Expected 1 '{name}' arg, but got {len(arg)}")

        # Convert bytes to string
        return arg[0].decode("utf-8")

    def post(self, **kwargs):
        """Receive an uploaded file and add it to our UploadedFileManager.
        Return the file's ID, so that the client can refer to it.
        """
        args: Dict[str, List[bytes]] = {}
        files: Dict[str, List[Any]] = {}

        tornado.httputil.parse_body_arguments(
            content_type=self.request.headers["Content-Type"],
            body=self.request.body,
            arguments=args,
            files=files,
        )

        try:
            session_id = self._require_arg(args, "sessionId")
            widget_id = self._require_arg(args, "widgetId")
            if not self._is_active_session(session_id):
                raise Exception(f"Invalid session_id: '{session_id}'")

        except Exception as e:
            self.send_error(400, reason=str(e))
            return

        # Create an UploadedFile object for each file.
        # We assign an initial, invalid file_id to each file in this loop.
        # The file_mgr will assign unique file IDs and return in `add_file`,
        # below.
        uploaded_files: List[UploadedFileRec] = []
        for _, flist in files.items():
            for file in flist:
                uploaded_files.append(
                    UploadedFileRec(
                        id=0,
                        name=file["filename"],
                        type=file["content_type"],
                        data=file["body"],
                    )
                )

        if len(uploaded_files) != 1:
            self.send_error(
                400, reason=f"Expected 1 file, but got {len(uploaded_files)}"
            )
            return

        added_file = self._file_mgr.add_file(
            session_id=session_id, widget_id=widget_id, file=uploaded_files[0]
        )

        # Return the file_id to the client. (The client will parse
        # the string back to an int.)
        self.write(str(added_file.id))
        self.set_status(200)

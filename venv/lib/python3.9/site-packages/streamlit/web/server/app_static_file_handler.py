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

import mimetypes
import os
from pathlib import Path
from typing import Optional

import tornado.web

from streamlit.logger import get_logger
from streamlit.web.server.routes import AssetsFileHandler

_LOGGER = get_logger(__name__)


# We agreed on these limitations for the initial release of static file sharing,
# based on security concerns from the SiS and Community Cloud teams
# The maximum possible size of single serving static file.
MAX_APP_STATIC_FILE_SIZE = 200 * 1024 * 1024  # 200 MB
# The list of file extensions that we serve with the corresponding Content-Type header.
# All files with other extensions will be served with Content-Type: text/plain
SAFE_APP_STATIC_FILE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp")


class AppStaticFileHandler(AssetsFileHandler):
    def initialize(self, path: str, default_filename: Optional[str] = None) -> None:
        super().initialize(path, default_filename)
        mimetypes.add_type("image/webp", ".webp")

    def validate_absolute_path(self, root: str, absolute_path: str) -> Optional[str]:
        full_path = os.path.realpath(absolute_path)

        if os.path.isdir(full_path):
            # we don't want to serve directories, and serve only files
            raise tornado.web.HTTPError(404)

        if os.path.commonprefix([full_path, root]) != root:
            # Don't allow misbehaving clients to break out of the static files directory
            _LOGGER.warning(
                "Serving files outside of the static directory is not supported"
            )
            raise tornado.web.HTTPError(404)

        if (
            os.path.exists(full_path)
            and os.path.getsize(full_path) > MAX_APP_STATIC_FILE_SIZE
        ):
            raise tornado.web.HTTPError(
                404,
                "File is too large, its size should not exceed "
                f"{MAX_APP_STATIC_FILE_SIZE} bytes",
                reason="File is too large",
            )

        return super().validate_absolute_path(root, absolute_path)

    def set_extra_headers(self, path: str) -> None:
        if Path(path).suffix not in SAFE_APP_STATIC_FILE_EXTENSIONS:
            self.set_header("Content-Type", "text/plain")
        self.set_header("X-Content-Type-Options", "nosniff")

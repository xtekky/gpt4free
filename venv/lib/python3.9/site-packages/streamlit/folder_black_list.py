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

import os

from streamlit import config, file_util, util

# The files in the folders below should always be blacklisted.
DEFAULT_FOLDER_BLACKLIST = [
    "**/.*",
    "**/anaconda",
    "**/anaconda2",
    "**/anaconda3",
    "**/dist-packages",
    "**/miniconda",
    "**/miniconda2",
    "**/miniconda3",
    "**/node_modules",
    "**/pyenv",
    "**/site-packages",
    "**/venv",
    "**/virtualenv",
]


class FolderBlackList(object):
    """Implement a black list object with globbing.

    Note
    ----
    Blacklist any path that matches a glob in `DEFAULT_FOLDER_BLACKLIST`.

    """

    def __init__(self, folder_blacklist):
        """Constructor.

        Parameters
        ----------
        folder_blacklist : list of str
            list of folder names with globbing to blacklist.

        """
        self._folder_blacklist = list(folder_blacklist)
        self._folder_blacklist.extend(DEFAULT_FOLDER_BLACKLIST)

        # Add the Streamlit lib folder when in dev mode, since otherwise we end
        # up with weird situations where the ID of a class in one run is not
        # the same as in another run.
        if config.get_option("global.developmentMode"):
            self._folder_blacklist.append(os.path.dirname(__file__))

    def __repr__(self) -> str:
        return util.repr_(self)

    def is_blacklisted(self, filepath):
        """Test if filepath is in the blacklist.

        Parameters
        ----------
        filepath : str
            File path that we intend to test.

        """
        return any(
            file_util.file_is_in_folder_glob(filepath, blacklisted_folder)
            for blacklisted_folder in self._folder_blacklist
        )

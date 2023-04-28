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

import shutil
import tempfile

from streamlit import util

# We provide our own context manager for temporary directory that wraps
# tempfile.mkdtemp


class TemporaryDirectory(object):
    """Temporary directory context manager.

    Creates a temporary directory that exists within the context manager scope.
    It returns the path to the created directory.
    Wrapper on top of tempfile.mkdtemp.

    Parameters
    ----------
    suffix : str or None
        Suffix to the filename.
    prefix : str or None
        Prefix to the filename.
    dir : str or None
        Enclosing directory.

    """

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __repr__(self) -> str:
        return util.repr_(self)

    def __enter__(self):
        self._path = tempfile.mkdtemp(*self._args, **self._kwargs)
        return self._path

    def __exit__(self, exc_type, exc_value, exc_traceback):
        shutil.rmtree(self._path)

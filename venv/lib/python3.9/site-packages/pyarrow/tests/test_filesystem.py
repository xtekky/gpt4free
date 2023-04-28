# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import pyarrow as pa
from pyarrow import filesystem

import os
import pytest


def test_filesystem_deprecated():
    with pytest.warns(FutureWarning):
        filesystem.LocalFileSystem()

    with pytest.warns(FutureWarning):
        filesystem.LocalFileSystem.get_instance()


def test_filesystem_deprecated_toplevel():
    with pytest.warns(FutureWarning):
        pa.localfs

    with pytest.warns(FutureWarning):
        pa.FileSystem

    with pytest.warns(FutureWarning):
        pa.LocalFileSystem

    with pytest.warns(FutureWarning):
        pa.HadoopFileSystem


def test_resolve_uri():
    uri = "file:///home/user/myfile.parquet"
    fs, path = filesystem.resolve_filesystem_and_path(uri)
    assert isinstance(fs, filesystem.LocalFileSystem)
    assert path == "/home/user/myfile.parquet"


def test_resolve_local_path():
    for uri in ['/home/user/myfile.parquet',
                'myfile.parquet',
                'my # file ? parquet',
                'C:/Windows/myfile.parquet',
                r'C:\\Windows\\myfile.parquet',
                ]:
        fs, path = filesystem.resolve_filesystem_and_path(uri)
        assert isinstance(fs, filesystem.LocalFileSystem)
        assert path == uri


@pytest.mark.filterwarnings("ignore:pyarrow.filesystem.LocalFileSystem")
def test_resolve_home_directory():
    uri = '~/myfile.parquet'
    fs, path = filesystem.resolve_filesystem_and_path(uri)
    assert isinstance(fs, filesystem.LocalFileSystem)
    assert path == os.path.expanduser(uri)

    local_fs = filesystem.LocalFileSystem()
    fs, path = filesystem.resolve_filesystem_and_path(uri, local_fs)
    assert path == os.path.expanduser(uri)

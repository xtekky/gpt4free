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

from datetime import datetime, timezone, timedelta
import gzip
import os
import pathlib
import pickle

import pytest
import weakref

import pyarrow as pa
from pyarrow.tests.test_io import assert_file_not_found
from pyarrow.tests.util import (_filesystem_uri, ProxyHandler,
                                _configure_s3_limited_user)

from pyarrow.fs import (FileType, FileInfo, FileSelector, FileSystem,
                        LocalFileSystem, SubTreeFileSystem, _MockFileSystem,
                        FileSystemHandler, PyFileSystem, FSSpecHandler,
                        copy_files)


class DummyHandler(FileSystemHandler):
    def __init__(self, value=42):
        self._value = value

    def __eq__(self, other):
        if isinstance(other, FileSystemHandler):
            return self._value == other._value
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, FileSystemHandler):
            return self._value != other._value
        return NotImplemented

    def get_type_name(self):
        return "dummy"

    def normalize_path(self, path):
        return path

    def get_file_info(self, paths):
        info = []
        for path in paths:
            if "file" in path:
                info.append(FileInfo(path, FileType.File))
            elif "dir" in path:
                info.append(FileInfo(path, FileType.Directory))
            elif "notfound" in path:
                info.append(FileInfo(path, FileType.NotFound))
            elif "badtype" in path:
                # Will raise when converting
                info.append(object())
            else:
                raise IOError
        return info

    def get_file_info_selector(self, selector):
        if selector.base_dir != "somedir":
            if selector.allow_not_found:
                return []
            else:
                raise FileNotFoundError(selector.base_dir)
        infos = [
            FileInfo("somedir/file1", FileType.File, size=123),
            FileInfo("somedir/subdir1", FileType.Directory),
        ]
        if selector.recursive:
            infos += [
                FileInfo("somedir/subdir1/file2", FileType.File, size=456),
            ]
        return infos

    def create_dir(self, path, recursive):
        if path == "recursive":
            assert recursive is True
        elif path == "non-recursive":
            assert recursive is False
        else:
            raise IOError

    def delete_dir(self, path):
        assert path == "delete_dir"

    def delete_dir_contents(self, path, missing_dir_ok):
        if not path.strip("/"):
            raise ValueError
        assert path == "delete_dir_contents"

    def delete_root_dir_contents(self):
        pass

    def delete_file(self, path):
        assert path == "delete_file"

    def move(self, src, dest):
        assert src == "move_from"
        assert dest == "move_to"

    def copy_file(self, src, dest):
        assert src == "copy_file_from"
        assert dest == "copy_file_to"

    def open_input_stream(self, path):
        if "notfound" in path:
            raise FileNotFoundError(path)
        data = "{0}:input_stream".format(path).encode('utf8')
        return pa.BufferReader(data)

    def open_input_file(self, path):
        if "notfound" in path:
            raise FileNotFoundError(path)
        data = "{0}:input_file".format(path).encode('utf8')
        return pa.BufferReader(data)

    def open_output_stream(self, path, metadata):
        if "notfound" in path:
            raise FileNotFoundError(path)
        return pa.BufferOutputStream()

    def open_append_stream(self, path, metadata):
        if "notfound" in path:
            raise FileNotFoundError(path)
        return pa.BufferOutputStream()


@pytest.fixture
def localfs(request, tempdir):
    return dict(
        fs=LocalFileSystem(),
        pathfn=lambda p: (tempdir / p).as_posix(),
        allow_move_dir=True,
        allow_append_to_file=True,
    )


@pytest.fixture
def py_localfs(request, tempdir):
    return dict(
        fs=PyFileSystem(ProxyHandler(LocalFileSystem())),
        pathfn=lambda p: (tempdir / p).as_posix(),
        allow_move_dir=True,
        allow_append_to_file=True,
    )


@pytest.fixture
def mockfs(request):
    return dict(
        fs=_MockFileSystem(),
        pathfn=lambda p: p,
        allow_move_dir=True,
        allow_append_to_file=True,
    )


@pytest.fixture
def py_mockfs(request):
    return dict(
        fs=PyFileSystem(ProxyHandler(_MockFileSystem())),
        pathfn=lambda p: p,
        allow_move_dir=True,
        allow_append_to_file=True,
    )


@pytest.fixture
def localfs_with_mmap(request, tempdir):
    return dict(
        fs=LocalFileSystem(use_mmap=True),
        pathfn=lambda p: (tempdir / p).as_posix(),
        allow_move_dir=True,
        allow_append_to_file=True,
    )


@pytest.fixture
def subtree_localfs(request, tempdir, localfs):
    return dict(
        fs=SubTreeFileSystem(str(tempdir), localfs['fs']),
        pathfn=lambda p: p,
        allow_move_dir=True,
        allow_append_to_file=True,
    )


@pytest.fixture
def gcsfs(request, gcs_server):
    request.config.pyarrow.requires('gcs')
    from pyarrow.fs import GcsFileSystem

    host, port = gcs_server['connection']
    bucket = 'pyarrow-filesystem/'

    fs = GcsFileSystem(
        endpoint_override=f'{host}:{port}',
        scheme='http',
        # Mock endpoint doesn't check credentials.
        anonymous=True,
        retry_time_limit=timedelta(seconds=45)
    )
    try:
        fs.create_dir(bucket)
    except OSError as e:
        pytest.skip(f"Could not create directory in {fs}: {e}")

    yield dict(
        fs=fs,
        pathfn=bucket.__add__,
        allow_move_dir=False,
        allow_append_to_file=False,
    )
    fs.delete_dir(bucket)


@pytest.fixture
def s3fs(request, s3_server):
    request.config.pyarrow.requires('s3')
    from pyarrow.fs import S3FileSystem

    host, port, access_key, secret_key = s3_server['connection']
    bucket = 'pyarrow-filesystem/'

    fs = S3FileSystem(
        access_key=access_key,
        secret_key=secret_key,
        endpoint_override='{}:{}'.format(host, port),
        scheme='http',
        allow_bucket_creation=True,
        allow_bucket_deletion=True
    )
    fs.create_dir(bucket)

    yield dict(
        fs=fs,
        pathfn=bucket.__add__,
        allow_move_dir=False,
        allow_append_to_file=False,
    )
    fs.delete_dir(bucket)


@pytest.fixture
def subtree_s3fs(request, s3fs):
    prefix = 'pyarrow-filesystem/prefix/'
    return dict(
        fs=SubTreeFileSystem(prefix, s3fs['fs']),
        pathfn=prefix.__add__,
        allow_move_dir=False,
        allow_append_to_file=False,
    )


_minio_limited_policy = """{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:ListAllMyBuckets",
                "s3:PutObject",
                "s3:GetObject",
                "s3:ListBucket",
                "s3:PutObjectTagging",
                "s3:DeleteObject",
                "s3:GetObjectVersion"
            ],
            "Resource": [
                "arn:aws:s3:::*"
            ]
        }
    ]
}"""


@pytest.fixture
def hdfs(request, hdfs_connection):
    request.config.pyarrow.requires('hdfs')
    if not pa.have_libhdfs():
        pytest.skip('Cannot locate libhdfs')

    from pyarrow.fs import HadoopFileSystem

    host, port, user = hdfs_connection
    fs = HadoopFileSystem(host, port=port, user=user)

    return dict(
        fs=fs,
        pathfn=lambda p: p,
        allow_move_dir=True,
        allow_append_to_file=True,
    )


@pytest.fixture
def py_fsspec_localfs(request, tempdir):
    fsspec = pytest.importorskip("fsspec")
    fs = fsspec.filesystem('file')
    return dict(
        fs=PyFileSystem(FSSpecHandler(fs)),
        pathfn=lambda p: (tempdir / p).as_posix(),
        allow_move_dir=True,
        allow_append_to_file=True,
    )


@pytest.fixture
def py_fsspec_memoryfs(request, tempdir):
    fsspec = pytest.importorskip("fsspec", minversion="0.7.5")
    if fsspec.__version__ == "0.8.5":
        # see https://issues.apache.org/jira/browse/ARROW-10934
        pytest.skip("Bug in fsspec 0.8.5 for in-memory filesystem")
    fs = fsspec.filesystem('memory')
    return dict(
        fs=PyFileSystem(FSSpecHandler(fs)),
        pathfn=lambda p: p,
        allow_move_dir=True,
        allow_append_to_file=True,
    )


@pytest.fixture
def py_fsspec_s3fs(request, s3_server):
    s3fs = pytest.importorskip("s3fs")
    host, port, access_key, secret_key = s3_server['connection']
    bucket = 'pyarrow-filesystem/'

    fs = s3fs.S3FileSystem(
        key=access_key,
        secret=secret_key,
        client_kwargs=dict(endpoint_url='http://{}:{}'.format(host, port))
    )
    fs = PyFileSystem(FSSpecHandler(fs))
    fs.create_dir(bucket)

    yield dict(
        fs=fs,
        pathfn=bucket.__add__,
        allow_move_dir=False,
        allow_append_to_file=True,
    )
    fs.delete_dir(bucket)


@pytest.fixture(params=[
    pytest.param(
        pytest.lazy_fixture('localfs'),
        id='LocalFileSystem()'
    ),
    pytest.param(
        pytest.lazy_fixture('localfs_with_mmap'),
        id='LocalFileSystem(use_mmap=True)'
    ),
    pytest.param(
        pytest.lazy_fixture('subtree_localfs'),
        id='SubTreeFileSystem(LocalFileSystem())'
    ),
    pytest.param(
        pytest.lazy_fixture('s3fs'),
        id='S3FileSystem',
        marks=pytest.mark.s3
    ),
    pytest.param(
        pytest.lazy_fixture('gcsfs'),
        id='GcsFileSystem',
        marks=pytest.mark.gcs
    ),
    pytest.param(
        pytest.lazy_fixture('hdfs'),
        id='HadoopFileSystem',
        marks=pytest.mark.hdfs
    ),
    pytest.param(
        pytest.lazy_fixture('mockfs'),
        id='_MockFileSystem()'
    ),
    pytest.param(
        pytest.lazy_fixture('py_localfs'),
        id='PyFileSystem(ProxyHandler(LocalFileSystem()))'
    ),
    pytest.param(
        pytest.lazy_fixture('py_mockfs'),
        id='PyFileSystem(ProxyHandler(_MockFileSystem()))'
    ),
    pytest.param(
        pytest.lazy_fixture('py_fsspec_localfs'),
        id='PyFileSystem(FSSpecHandler(fsspec.LocalFileSystem()))'
    ),
    pytest.param(
        pytest.lazy_fixture('py_fsspec_memoryfs'),
        id='PyFileSystem(FSSpecHandler(fsspec.filesystem("memory")))'
    ),
    pytest.param(
        pytest.lazy_fixture('py_fsspec_s3fs'),
        id='PyFileSystem(FSSpecHandler(s3fs.S3FileSystem()))',
        marks=pytest.mark.s3
    ),
])
def filesystem_config(request):
    return request.param


@pytest.fixture
def fs(request, filesystem_config):
    return filesystem_config['fs']


@pytest.fixture
def pathfn(request, filesystem_config):
    return filesystem_config['pathfn']


@pytest.fixture
def allow_move_dir(request, filesystem_config):
    return filesystem_config['allow_move_dir']


@pytest.fixture
def allow_append_to_file(request, filesystem_config):
    return filesystem_config['allow_append_to_file']


def check_mtime(file_info):
    assert isinstance(file_info.mtime, datetime)
    assert isinstance(file_info.mtime_ns, int)
    assert file_info.mtime_ns >= 0
    assert file_info.mtime_ns == pytest.approx(
        file_info.mtime.timestamp() * 1e9)
    # It's an aware UTC datetime
    tzinfo = file_info.mtime.tzinfo
    assert tzinfo is not None
    assert tzinfo.utcoffset(None) == timedelta(0)


def check_mtime_absent(file_info):
    assert file_info.mtime is None
    assert file_info.mtime_ns is None


def check_mtime_or_absent(file_info):
    if file_info.mtime is None:
        check_mtime_absent(file_info)
    else:
        check_mtime(file_info)


def skip_fsspec_s3fs(fs):
    if fs.type_name == "py::fsspec+s3":
        pytest.xfail(reason="Not working with fsspec's s3fs")


@pytest.mark.s3
def test_s3fs_limited_permissions_create_bucket(s3_server):
    from pyarrow.fs import S3FileSystem
    _configure_s3_limited_user(s3_server, _minio_limited_policy)
    host, port, _, _ = s3_server['connection']

    fs = S3FileSystem(
        access_key='limited',
        secret_key='limited123',
        endpoint_override='{}:{}'.format(host, port),
        scheme='http'
    )
    fs.create_dir('existing-bucket/test')

    with pytest.raises(pa.ArrowIOError, match="Bucket 'new-bucket' not found"):
        fs.create_dir('new-bucket')

    with pytest.raises(pa.ArrowIOError, match="Would delete bucket"):
        fs.delete_dir('existing-bucket')


def test_file_info_constructor():
    dt = datetime.fromtimestamp(1568799826, timezone.utc)

    info = FileInfo("foo/bar")
    assert info.path == "foo/bar"
    assert info.base_name == "bar"
    assert info.type == FileType.Unknown
    assert info.size is None
    check_mtime_absent(info)

    info = FileInfo("foo/baz.txt", type=FileType.File, size=123,
                    mtime=1568799826.5)
    assert info.path == "foo/baz.txt"
    assert info.base_name == "baz.txt"
    assert info.type == FileType.File
    assert info.size == 123
    assert info.mtime_ns == 1568799826500000000
    check_mtime(info)

    info = FileInfo("foo", type=FileType.Directory, mtime=dt)
    assert info.path == "foo"
    assert info.base_name == "foo"
    assert info.type == FileType.Directory
    assert info.size is None
    assert info.mtime == dt
    assert info.mtime_ns == 1568799826000000000
    check_mtime(info)


def test_cannot_instantiate_base_filesystem():
    with pytest.raises(TypeError):
        FileSystem()


def test_filesystem_equals():
    fs0 = LocalFileSystem()
    fs1 = LocalFileSystem()
    fs2 = _MockFileSystem()

    assert fs0.equals(fs0)
    assert fs0.equals(fs1)
    with pytest.raises(TypeError):
        fs0.equals('string')
    assert fs0 == fs0 == fs1
    assert fs0 != 4

    assert fs2 == fs2
    assert fs2 != _MockFileSystem()

    assert SubTreeFileSystem('/base', fs0) == SubTreeFileSystem('/base', fs0)
    assert SubTreeFileSystem('/base', fs0) != SubTreeFileSystem('/base', fs2)
    assert SubTreeFileSystem('/base', fs0) != SubTreeFileSystem('/other', fs0)


def test_subtree_filesystem():
    localfs = LocalFileSystem()

    subfs = SubTreeFileSystem('/base', localfs)
    assert subfs.base_path == '/base/'
    assert subfs.base_fs == localfs
    assert repr(subfs).startswith('SubTreeFileSystem(base_path=/base/, '
                                  'base_fs=<pyarrow._fs.LocalFileSystem')

    subfs = SubTreeFileSystem('/another/base/', LocalFileSystem())
    assert subfs.base_path == '/another/base/'
    assert subfs.base_fs == localfs
    assert repr(subfs).startswith('SubTreeFileSystem(base_path=/another/base/,'
                                  ' base_fs=<pyarrow._fs.LocalFileSystem')


def test_filesystem_pickling(fs):
    if fs.type_name.split('::')[-1] == 'mock':
        pytest.xfail(reason='MockFileSystem is not serializable')

    serialized = pickle.dumps(fs)
    restored = pickle.loads(serialized)
    assert isinstance(restored, FileSystem)
    assert restored.equals(fs)


def test_filesystem_is_functional_after_pickling(fs, pathfn):
    if fs.type_name.split('::')[-1] == 'mock':
        pytest.xfail(reason='MockFileSystem is not serializable')
    skip_fsspec_s3fs(fs)

    aaa = pathfn('a/aa/aaa/')
    bb = pathfn('a/bb')
    c = pathfn('c.txt')

    fs.create_dir(aaa)
    with fs.open_output_stream(bb):
        pass  # touch
    with fs.open_output_stream(c) as fp:
        fp.write(b'test')

    restored = pickle.loads(pickle.dumps(fs))
    aaa_info, bb_info, c_info = restored.get_file_info([aaa, bb, c])
    assert aaa_info.type == FileType.Directory
    assert bb_info.type == FileType.File
    assert c_info.type == FileType.File


def test_type_name():
    fs = LocalFileSystem()
    assert fs.type_name == "local"
    fs = _MockFileSystem()
    assert fs.type_name == "mock"


def test_normalize_path(fs):
    # Trivial path names (without separators) should generally be
    # already normalized.  Just a sanity check.
    assert fs.normalize_path("foo") == "foo"


def test_non_path_like_input_raises(fs):
    class Path:
        pass

    invalid_paths = [1, 1.1, Path(), tuple(), {}, [], lambda: 1,
                     pathlib.Path()]
    for path in invalid_paths:
        with pytest.raises(TypeError):
            fs.create_dir(path)


def test_get_file_info(fs, pathfn):
    aaa = pathfn('a/aa/aaa/')
    bb = pathfn('a/bb')
    c = pathfn('c.txt')
    zzz = pathfn('zzz')

    fs.create_dir(aaa)
    with fs.open_output_stream(bb):
        pass  # touch
    with fs.open_output_stream(c) as fp:
        fp.write(b'test')

    aaa_info, bb_info, c_info, zzz_info = fs.get_file_info([aaa, bb, c, zzz])

    assert aaa_info.path == aaa
    assert 'aaa' in repr(aaa_info)
    assert aaa_info.extension == ''
    if fs.type_name == "py::fsspec+s3":
        # s3fs doesn't create empty directories
        assert aaa_info.type == FileType.NotFound
    else:
        assert aaa_info.type == FileType.Directory
        assert 'FileType.Directory' in repr(aaa_info)
    assert aaa_info.size is None
    check_mtime_or_absent(aaa_info)

    assert bb_info.path == str(bb)
    assert bb_info.base_name == 'bb'
    assert bb_info.extension == ''
    assert bb_info.type == FileType.File
    assert 'FileType.File' in repr(bb_info)
    assert bb_info.size == 0
    if fs.type_name not in ["py::fsspec+memory", "py::fsspec+s3"]:
        check_mtime(bb_info)

    assert c_info.path == str(c)
    assert c_info.base_name == 'c.txt'
    assert c_info.extension == 'txt'
    assert c_info.type == FileType.File
    assert 'FileType.File' in repr(c_info)
    assert c_info.size == 4
    if fs.type_name not in ["py::fsspec+memory", "py::fsspec+s3"]:
        check_mtime(c_info)

    assert zzz_info.path == str(zzz)
    assert zzz_info.base_name == 'zzz'
    assert zzz_info.extension == ''
    assert zzz_info.type == FileType.NotFound
    assert zzz_info.size is None
    assert zzz_info.mtime is None
    assert 'FileType.NotFound' in repr(zzz_info)
    check_mtime_absent(zzz_info)

    # with single path
    aaa_info2 = fs.get_file_info(aaa)
    assert aaa_info.path == aaa_info2.path
    assert aaa_info.type == aaa_info2.type


def test_get_file_info_with_selector(fs, pathfn):
    base_dir = pathfn('selector-dir/')
    file_a = pathfn('selector-dir/test_file_a')
    file_b = pathfn('selector-dir/test_file_b')
    dir_a = pathfn('selector-dir/test_dir_a')
    file_c = pathfn('selector-dir/test_dir_a/test_file_c')
    dir_b = pathfn('selector-dir/test_dir_b')

    try:
        fs.create_dir(base_dir)
        with fs.open_output_stream(file_a):
            pass
        with fs.open_output_stream(file_b):
            pass
        fs.create_dir(dir_a)
        with fs.open_output_stream(file_c):
            pass
        fs.create_dir(dir_b)

        # recursive selector
        selector = FileSelector(base_dir, allow_not_found=False,
                                recursive=True)
        assert selector.base_dir == base_dir

        infos = fs.get_file_info(selector)
        if fs.type_name == "py::fsspec+s3":
            # s3fs only lists directories if they are not empty, but depending
            # on the s3fs/fsspec version combo, it includes the base_dir
            # (https://github.com/dask/s3fs/issues/393)
            assert (len(infos) == 4) or (len(infos) == 5)
        else:
            assert len(infos) == 5

        for info in infos:
            if (info.path.endswith(file_a) or info.path.endswith(file_b) or
                    info.path.endswith(file_c)):
                assert info.type == FileType.File
            elif (info.path.rstrip("/").endswith(dir_a) or
                  info.path.rstrip("/").endswith(dir_b)):
                assert info.type == FileType.Directory
            elif (fs.type_name == "py::fsspec+s3" and
                  info.path.rstrip("/").endswith("selector-dir")):
                # s3fs can include base dir, see above
                assert info.type == FileType.Directory
            else:
                raise ValueError('unexpected path {}'.format(info.path))
            check_mtime_or_absent(info)

        # non-recursive selector -> not selecting the nested file_c
        selector = FileSelector(base_dir, recursive=False)

        infos = fs.get_file_info(selector)
        if fs.type_name == "py::fsspec+s3":
            # s3fs only lists directories if they are not empty
            # + for s3fs 0.5.2 all directories are dropped because of buggy
            # side-effect of previous find() call
            # (https://github.com/dask/s3fs/issues/410)
            assert (len(infos) == 3) or (len(infos) == 2)
        else:
            assert len(infos) == 4

    finally:
        fs.delete_dir(base_dir)


def test_create_dir(fs, pathfn):
    # s3fs fails deleting dir fails if it is empty
    # (https://github.com/dask/s3fs/issues/317)
    skip_fsspec_s3fs(fs)
    d = pathfn('test-directory/')

    with pytest.raises(pa.ArrowIOError):
        fs.delete_dir(d)

    fs.create_dir(d)
    fs.delete_dir(d)

    d = pathfn('deeply/nested/test-directory/')
    fs.create_dir(d, recursive=True)
    fs.delete_dir(d)


def test_delete_dir(fs, pathfn):
    skip_fsspec_s3fs(fs)

    d = pathfn('directory/')
    nd = pathfn('directory/nested/')

    fs.create_dir(nd)
    fs.delete_dir(d)
    with pytest.raises(pa.ArrowIOError):
        fs.delete_dir(nd)
    with pytest.raises(pa.ArrowIOError):
        fs.delete_dir(d)


def test_delete_dir_contents(fs, pathfn):
    skip_fsspec_s3fs(fs)

    d = pathfn('directory/')
    nd = pathfn('directory/nested/')

    fs.create_dir(nd)
    fs.delete_dir_contents(d)
    with pytest.raises(pa.ArrowIOError):
        fs.delete_dir(nd)
    fs.delete_dir_contents(nd, missing_dir_ok=True)
    with pytest.raises(pa.ArrowIOError):
        fs.delete_dir_contents(nd)
    fs.delete_dir(d)
    with pytest.raises(pa.ArrowIOError):
        fs.delete_dir(d)


def _check_root_dir_contents(config):
    fs = config['fs']
    pathfn = config['pathfn']

    d = pathfn('directory/')
    nd = pathfn('directory/nested/')

    fs.create_dir(nd)
    with pytest.raises(pa.ArrowInvalid):
        fs.delete_dir_contents("")
    with pytest.raises(pa.ArrowInvalid):
        fs.delete_dir_contents("/")
    with pytest.raises(pa.ArrowInvalid):
        fs.delete_dir_contents("//")

    fs.delete_dir_contents("", accept_root_dir=True)
    fs.delete_dir_contents("/", accept_root_dir=True)
    fs.delete_dir_contents("//", accept_root_dir=True)
    with pytest.raises(pa.ArrowIOError):
        fs.delete_dir(d)


def test_delete_root_dir_contents(mockfs, py_mockfs):
    _check_root_dir_contents(mockfs)
    _check_root_dir_contents(py_mockfs)


def test_copy_file(fs, pathfn):
    s = pathfn('test-copy-source-file')
    t = pathfn('test-copy-target-file')

    with fs.open_output_stream(s):
        pass

    fs.copy_file(s, t)
    fs.delete_file(s)
    fs.delete_file(t)


def test_move_directory(fs, pathfn, allow_move_dir):
    # move directory (doesn't work with S3)
    s = pathfn('source-dir/')
    t = pathfn('target-dir/')

    fs.create_dir(s)

    if allow_move_dir:
        fs.move(s, t)
        with pytest.raises(pa.ArrowIOError):
            fs.delete_dir(s)
        fs.delete_dir(t)
    else:
        with pytest.raises(pa.ArrowIOError):
            fs.move(s, t)


def test_move_file(fs, pathfn):
    # s3fs moving a file with recursive=True on latest 0.5 version
    # (https://github.com/dask/s3fs/issues/394)
    skip_fsspec_s3fs(fs)

    s = pathfn('test-move-source-file')
    t = pathfn('test-move-target-file')

    with fs.open_output_stream(s):
        pass

    fs.move(s, t)
    with pytest.raises(pa.ArrowIOError):
        fs.delete_file(s)
    fs.delete_file(t)


def test_delete_file(fs, pathfn):
    p = pathfn('test-delete-target-file')
    with fs.open_output_stream(p):
        pass

    fs.delete_file(p)
    with pytest.raises(pa.ArrowIOError):
        fs.delete_file(p)

    d = pathfn('test-delete-nested')
    fs.create_dir(d)
    f = pathfn('test-delete-nested/target-file')
    with fs.open_output_stream(f) as s:
        s.write(b'data')

    fs.delete_dir(d)


def identity(v):
    return v


@pytest.mark.gzip
@pytest.mark.parametrize(
    ('compression', 'buffer_size', 'compressor'),
    [
        (None, None, identity),
        (None, 64, identity),
        ('gzip', None, gzip.compress),
        ('gzip', 256, gzip.compress),
    ]
)
def test_open_input_stream(fs, pathfn, compression, buffer_size, compressor):
    p = pathfn('open-input-stream')

    data = b'some data for reading\n' * 512
    with fs.open_output_stream(p) as s:
        s.write(compressor(data))

    with fs.open_input_stream(p, compression, buffer_size) as s:
        result = s.read()

    assert result == data


def test_open_input_file(fs, pathfn):
    p = pathfn('open-input-file')

    data = b'some data' * 1024
    with fs.open_output_stream(p) as s:
        s.write(data)

    read_from = len(b'some data') * 512
    with fs.open_input_file(p) as f:
        result = f.read()
    assert result == data

    with fs.open_input_file(p) as f:
        f.seek(read_from)
        result = f.read()

    assert result == data[read_from:]


def test_open_input_stream_not_found(fs, pathfn):
    # The proper exception should be raised for this common case (ARROW-15896)
    p = pathfn('open-input-stream-not-found')
    with pytest.raises(FileNotFoundError):
        fs.open_input_stream(p)


@pytest.mark.gzip
@pytest.mark.parametrize(
    ('compression', 'buffer_size', 'decompressor'),
    [
        (None, None, identity),
        (None, 64, identity),
        ('gzip', None, gzip.decompress),
        ('gzip', 256, gzip.decompress),
    ]
)
def test_open_output_stream(fs, pathfn, compression, buffer_size,
                            decompressor):
    p = pathfn('open-output-stream')

    data = b'some data for writing' * 1024
    with fs.open_output_stream(p, compression, buffer_size) as f:
        f.write(data)

    with fs.open_input_stream(p, compression, buffer_size) as f:
        assert f.read(len(data)) == data


@pytest.mark.gzip
@pytest.mark.parametrize(
    ('compression', 'buffer_size', 'compressor', 'decompressor'),
    [
        (None, None, identity, identity),
        (None, 64, identity, identity),
        ('gzip', None, gzip.compress, gzip.decompress),
        ('gzip', 256, gzip.compress, gzip.decompress),
    ]
)
def test_open_append_stream(fs, pathfn, compression, buffer_size, compressor,
                            decompressor, allow_append_to_file):
    p = pathfn('open-append-stream')

    initial = compressor(b'already existing')
    with fs.open_output_stream(p) as s:
        s.write(initial)

    if allow_append_to_file:
        with fs.open_append_stream(p, compression=compression,
                                   buffer_size=buffer_size) as f:
            f.write(b'\nnewly added')

        with fs.open_input_stream(p) as f:
            result = f.read()

        result = decompressor(result)
        assert result == b'already existing\nnewly added'
    else:
        with pytest.raises(pa.ArrowNotImplementedError):
            fs.open_append_stream(p, compression=compression,
                                  buffer_size=buffer_size)


def test_open_output_stream_metadata(fs, pathfn):
    p = pathfn('open-output-stream-metadata')
    metadata = {'Content-Type': 'x-pyarrow/test'}

    data = b'some data'
    with fs.open_output_stream(p, metadata=metadata) as f:
        f.write(data)

    with fs.open_input_stream(p) as f:
        assert f.read() == data
        got_metadata = f.metadata()

    if fs.type_name in ['s3', 'gcs'] or 'mock' in fs.type_name:
        for k, v in metadata.items():
            assert got_metadata[k] == v.encode()
    else:
        assert got_metadata == {}


def test_localfs_options():
    # LocalFileSystem instantiation
    LocalFileSystem(use_mmap=False)

    with pytest.raises(TypeError):
        LocalFileSystem(xxx=False)


def test_localfs_errors(localfs):
    # Local filesystem errors should raise the right Python exceptions
    # (e.g. FileNotFoundError)
    fs = localfs['fs']
    with assert_file_not_found():
        fs.open_input_stream('/non/existent/file')
    with assert_file_not_found():
        fs.open_output_stream('/non/existent/file')
    with assert_file_not_found():
        fs.create_dir('/non/existent/dir', recursive=False)
    with assert_file_not_found():
        fs.delete_dir('/non/existent/dir')
    with assert_file_not_found():
        fs.delete_file('/non/existent/dir')
    with assert_file_not_found():
        fs.move('/non/existent', '/xxx')
    with assert_file_not_found():
        fs.copy_file('/non/existent', '/xxx')


def test_localfs_file_info(localfs):
    fs = localfs['fs']

    file_path = pathlib.Path(__file__)
    dir_path = file_path.parent
    [file_info, dir_info] = fs.get_file_info([file_path.as_posix(),
                                              dir_path.as_posix()])
    assert file_info.size == file_path.stat().st_size
    assert file_info.mtime_ns == file_path.stat().st_mtime_ns
    check_mtime(file_info)
    assert dir_info.mtime_ns == dir_path.stat().st_mtime_ns
    check_mtime(dir_info)


def test_mockfs_mtime_roundtrip(mockfs):
    dt = datetime.fromtimestamp(1568799826, timezone.utc)
    fs = _MockFileSystem(dt)

    with fs.open_output_stream('foo'):
        pass
    [info] = fs.get_file_info(['foo'])
    assert info.mtime == dt


@pytest.mark.gcs
def test_gcs_options():
    from pyarrow.fs import GcsFileSystem
    dt = datetime.now()
    fs = GcsFileSystem(access_token='abc',
                       target_service_account='service_account@apache',
                       credential_token_expiration=dt,
                       default_bucket_location='us-west2',
                       scheme='https', endpoint_override='localhost:8999')
    assert isinstance(fs, GcsFileSystem)
    assert fs.default_bucket_location == 'us-west2'
    assert pickle.loads(pickle.dumps(fs)) == fs

    fs = GcsFileSystem()
    assert isinstance(fs, GcsFileSystem)
    assert pickle.loads(pickle.dumps(fs)) == fs

    fs = GcsFileSystem(anonymous=True)
    assert isinstance(fs, GcsFileSystem)
    assert pickle.loads(pickle.dumps(fs)) == fs

    fs = GcsFileSystem(default_metadata={"ACL": "authenticated-read",
                                         "Content-Type": "text/plain"})
    assert isinstance(fs, GcsFileSystem)
    assert pickle.loads(pickle.dumps(fs)) == fs

    with pytest.raises(ValueError):
        GcsFileSystem(access_token='access')
    with pytest.raises(ValueError):
        GcsFileSystem(anonymous=True, access_token='secret')
    with pytest.raises(ValueError):
        GcsFileSystem(anonymous=True, target_service_account='acct')
    with pytest.raises(ValueError):
        GcsFileSystem(credential_token_expiration=datetime.now())


@pytest.mark.s3
def test_s3_options():
    from pyarrow.fs import (AwsDefaultS3RetryStrategy,
                            AwsStandardS3RetryStrategy, S3FileSystem,
                            S3RetryStrategy)

    fs = S3FileSystem(access_key='access', secret_key='secret',
                      session_token='token', region='us-east-2',
                      scheme='https', endpoint_override='localhost:8999')
    assert isinstance(fs, S3FileSystem)
    assert fs.region == 'us-east-2'
    assert pickle.loads(pickle.dumps(fs)) == fs

    fs = S3FileSystem(role_arn='role', session_name='session',
                      external_id='id', load_frequency=100)
    assert isinstance(fs, S3FileSystem)
    assert pickle.loads(pickle.dumps(fs)) == fs

    # Note that the retry strategy won't survive pickling for now
    fs = S3FileSystem(
        retry_strategy=AwsStandardS3RetryStrategy(max_attempts=5))
    assert isinstance(fs, S3FileSystem)

    fs = S3FileSystem(
        retry_strategy=AwsDefaultS3RetryStrategy(max_attempts=5))
    assert isinstance(fs, S3FileSystem)

    fs2 = S3FileSystem(role_arn='role')
    assert isinstance(fs2, S3FileSystem)
    assert pickle.loads(pickle.dumps(fs2)) == fs2
    assert fs2 != fs

    fs = S3FileSystem(anonymous=True)
    assert isinstance(fs, S3FileSystem)
    assert pickle.loads(pickle.dumps(fs)) == fs

    fs = S3FileSystem(background_writes=True)
    assert isinstance(fs, S3FileSystem)
    assert pickle.loads(pickle.dumps(fs)) == fs

    fs2 = S3FileSystem(background_writes=True,
                       default_metadata={"ACL": "authenticated-read",
                                         "Content-Type": "text/plain"})
    assert isinstance(fs2, S3FileSystem)
    assert pickle.loads(pickle.dumps(fs2)) == fs2
    assert fs2 != fs

    fs = S3FileSystem(allow_bucket_creation=True, allow_bucket_deletion=True)
    assert isinstance(fs, S3FileSystem)
    assert pickle.loads(pickle.dumps(fs)) == fs

    fs = S3FileSystem(request_timeout=0.5, connect_timeout=0.25)
    assert isinstance(fs, S3FileSystem)
    assert pickle.loads(pickle.dumps(fs)) == fs

    fs2 = S3FileSystem(request_timeout=0.25, connect_timeout=0.5)
    assert isinstance(fs2, S3FileSystem)
    assert pickle.loads(pickle.dumps(fs2)) == fs2
    assert fs2 != fs

    with pytest.raises(ValueError):
        S3FileSystem(access_key='access')
    with pytest.raises(ValueError):
        S3FileSystem(secret_key='secret')
    with pytest.raises(ValueError):
        S3FileSystem(access_key='access', session_token='token')
    with pytest.raises(ValueError):
        S3FileSystem(secret_key='secret', session_token='token')
    with pytest.raises(ValueError):
        S3FileSystem(
            access_key='access', secret_key='secret', role_arn='arn'
        )
    with pytest.raises(ValueError):
        S3FileSystem(
            access_key='access', secret_key='secret', anonymous=True
        )
    with pytest.raises(ValueError):
        S3FileSystem(role_arn="arn", anonymous=True)
    with pytest.raises(ValueError):
        S3FileSystem(default_metadata=["foo", "bar"])
    with pytest.raises(ValueError):
        S3FileSystem(retry_strategy=S3RetryStrategy())


@pytest.mark.s3
def test_s3_proxy_options(monkeypatch):
    from pyarrow.fs import S3FileSystem

    # The following two are equivalent:
    proxy_opts_1_dict = {'scheme': 'http', 'host': 'localhost', 'port': 8999}
    proxy_opts_1_str = 'http://localhost:8999'
    # The following two are equivalent:
    proxy_opts_2_dict = {'scheme': 'https', 'host': 'localhost', 'port': 8080}
    proxy_opts_2_str = 'https://localhost:8080'

    # Check dict case for 'proxy_options'
    fs = S3FileSystem(proxy_options=proxy_opts_1_dict)
    assert isinstance(fs, S3FileSystem)
    assert pickle.loads(pickle.dumps(fs)) == fs

    fs = S3FileSystem(proxy_options=proxy_opts_2_dict)
    assert isinstance(fs, S3FileSystem)
    assert pickle.loads(pickle.dumps(fs)) == fs

    # Check str case for 'proxy_options'
    fs = S3FileSystem(proxy_options=proxy_opts_1_str)
    assert isinstance(fs, S3FileSystem)
    assert pickle.loads(pickle.dumps(fs)) == fs

    fs = S3FileSystem(proxy_options=proxy_opts_2_str)
    assert isinstance(fs, S3FileSystem)
    assert pickle.loads(pickle.dumps(fs)) == fs

    # Check that two FSs using the same proxy_options dict are equal
    fs1 = S3FileSystem(proxy_options=proxy_opts_1_dict)
    fs2 = S3FileSystem(proxy_options=proxy_opts_1_dict)
    assert fs1 == fs2
    assert pickle.loads(pickle.dumps(fs1)) == fs2
    assert pickle.loads(pickle.dumps(fs2)) == fs1

    fs1 = S3FileSystem(proxy_options=proxy_opts_2_dict)
    fs2 = S3FileSystem(proxy_options=proxy_opts_2_dict)
    assert fs1 == fs2
    assert pickle.loads(pickle.dumps(fs1)) == fs2
    assert pickle.loads(pickle.dumps(fs2)) == fs1

    # Check that two FSs using the same proxy_options str are equal
    fs1 = S3FileSystem(proxy_options=proxy_opts_1_str)
    fs2 = S3FileSystem(proxy_options=proxy_opts_1_str)
    assert fs1 == fs2
    assert pickle.loads(pickle.dumps(fs1)) == fs2
    assert pickle.loads(pickle.dumps(fs2)) == fs1

    fs1 = S3FileSystem(proxy_options=proxy_opts_2_str)
    fs2 = S3FileSystem(proxy_options=proxy_opts_2_str)
    assert fs1 == fs2
    assert pickle.loads(pickle.dumps(fs1)) == fs2
    assert pickle.loads(pickle.dumps(fs2)) == fs1

    # Check that two FSs using equivalent proxy_options
    # (one dict, one str) are equal
    fs1 = S3FileSystem(proxy_options=proxy_opts_1_dict)
    fs2 = S3FileSystem(proxy_options=proxy_opts_1_str)
    assert fs1 == fs2
    assert pickle.loads(pickle.dumps(fs1)) == fs2
    assert pickle.loads(pickle.dumps(fs2)) == fs1

    fs1 = S3FileSystem(proxy_options=proxy_opts_2_dict)
    fs2 = S3FileSystem(proxy_options=proxy_opts_2_str)
    assert fs1 == fs2
    assert pickle.loads(pickle.dumps(fs1)) == fs2
    assert pickle.loads(pickle.dumps(fs2)) == fs1

    # Check that two FSs using nonequivalent proxy_options are not equal
    fs1 = S3FileSystem(proxy_options=proxy_opts_1_dict)
    fs2 = S3FileSystem(proxy_options=proxy_opts_2_dict)
    assert fs1 != fs2
    assert pickle.loads(pickle.dumps(fs1)) != fs2
    assert pickle.loads(pickle.dumps(fs2)) != fs1

    fs1 = S3FileSystem(proxy_options=proxy_opts_1_dict)
    fs2 = S3FileSystem(proxy_options=proxy_opts_2_str)
    assert fs1 != fs2
    assert pickle.loads(pickle.dumps(fs1)) != fs2
    assert pickle.loads(pickle.dumps(fs2)) != fs1

    fs1 = S3FileSystem(proxy_options=proxy_opts_1_str)
    fs2 = S3FileSystem(proxy_options=proxy_opts_2_dict)
    assert fs1 != fs2
    assert pickle.loads(pickle.dumps(fs1)) != fs2
    assert pickle.loads(pickle.dumps(fs2)) != fs1

    fs1 = S3FileSystem(proxy_options=proxy_opts_1_str)
    fs2 = S3FileSystem(proxy_options=proxy_opts_2_str)
    assert fs1 != fs2
    assert pickle.loads(pickle.dumps(fs1)) != fs2
    assert pickle.loads(pickle.dumps(fs2)) != fs1

    # Check that two FSs (one using proxy_options and the other not)
    # are not equal
    fs1 = S3FileSystem(proxy_options=proxy_opts_1_dict)
    fs2 = S3FileSystem()
    assert fs1 != fs2
    assert pickle.loads(pickle.dumps(fs1)) != fs2
    assert pickle.loads(pickle.dumps(fs2)) != fs1

    fs1 = S3FileSystem(proxy_options=proxy_opts_1_str)
    fs2 = S3FileSystem()
    assert fs1 != fs2
    assert pickle.loads(pickle.dumps(fs1)) != fs2
    assert pickle.loads(pickle.dumps(fs2)) != fs1

    fs1 = S3FileSystem(proxy_options=proxy_opts_2_dict)
    fs2 = S3FileSystem()
    assert fs1 != fs2
    assert pickle.loads(pickle.dumps(fs1)) != fs2
    assert pickle.loads(pickle.dumps(fs2)) != fs1

    fs1 = S3FileSystem(proxy_options=proxy_opts_2_str)
    fs2 = S3FileSystem()
    assert fs1 != fs2
    assert pickle.loads(pickle.dumps(fs1)) != fs2
    assert pickle.loads(pickle.dumps(fs2)) != fs1

    # Only dict and str are supported
    with pytest.raises(TypeError):
        S3FileSystem(proxy_options=('http', 'localhost', 9090))
    # Missing scheme
    with pytest.raises(KeyError):
        S3FileSystem(proxy_options={'host': 'localhost', 'port': 9090})
    # Missing host
    with pytest.raises(KeyError):
        S3FileSystem(proxy_options={'scheme': 'https', 'port': 9090})
    # Missing port
    with pytest.raises(KeyError):
        S3FileSystem(proxy_options={'scheme': 'http', 'host': 'localhost'})
    # Invalid proxy URI (invalid scheme htttps)
    with pytest.raises(pa.ArrowInvalid):
        S3FileSystem(proxy_options='htttps://localhost:9000')
    # Invalid proxy_options dict (invalid scheme htttps)
    with pytest.raises(pa.ArrowInvalid):
        S3FileSystem(proxy_options={'scheme': 'htttp', 'host': 'localhost',
                                    'port': 8999})


@pytest.mark.s3
def test_s3fs_wrong_region():
    from pyarrow.fs import S3FileSystem

    # wrong region for bucket
    # anonymous=True incase CI/etc has invalid credentials
    fs = S3FileSystem(region='eu-north-1', anonymous=True)

    msg = ("When getting information for bucket 'voltrondata-labs-datasets': "
           r"AWS Error UNKNOWN \(HTTP status 301\) during HeadBucket "
           "operation: No response body. Looks like the configured region is "
           "'eu-north-1' while the bucket is located in 'us-east-2'."
           "|NETWORK_CONNECTION")
    with pytest.raises(OSError, match=msg) as exc:
        fs.get_file_info("voltrondata-labs-datasets")

    # Sometimes fails on unrelated network error, so next call would also fail.
    if 'NETWORK_CONNECTION' in str(exc.value):
        return

    fs = S3FileSystem(region='us-east-2', anonymous=True)
    fs.get_file_info("voltrondata-labs-datasets")


@pytest.mark.hdfs
def test_hdfs_options(hdfs_connection):
    from pyarrow.fs import HadoopFileSystem
    if not pa.have_libhdfs():
        pytest.skip('Cannot locate libhdfs')

    host, port, user = hdfs_connection

    replication = 2
    buffer_size = 64*1024
    default_block_size = 128*1024**2
    uri = ('hdfs://{}:{}/?user={}&replication={}&buffer_size={}'
           '&default_block_size={}')

    hdfs1 = HadoopFileSystem(host, port, user='libhdfs',
                             replication=replication, buffer_size=buffer_size,
                             default_block_size=default_block_size)
    hdfs2 = HadoopFileSystem.from_uri(uri.format(
        host, port, 'libhdfs', replication, buffer_size, default_block_size
    ))
    hdfs3 = HadoopFileSystem.from_uri(uri.format(
        host, port, 'me', replication, buffer_size, default_block_size
    ))
    hdfs4 = HadoopFileSystem.from_uri(uri.format(
        host, port, 'me', replication + 1, buffer_size, default_block_size
    ))
    hdfs5 = HadoopFileSystem(host, port)
    hdfs6 = HadoopFileSystem.from_uri('hdfs://{}:{}'.format(host, port))
    hdfs7 = HadoopFileSystem(host, port, user='localuser')
    hdfs8 = HadoopFileSystem(host, port, user='localuser',
                             kerb_ticket="cache_path")
    hdfs9 = HadoopFileSystem(host, port, user='localuser',
                             kerb_ticket=pathlib.Path("cache_path"))
    hdfs10 = HadoopFileSystem(host, port, user='localuser',
                              kerb_ticket="cache_path2")
    hdfs11 = HadoopFileSystem(host, port, user='localuser',
                              kerb_ticket="cache_path",
                              extra_conf={'hdfs_token': 'abcd'})

    assert hdfs1 == hdfs2
    assert hdfs5 == hdfs6
    assert hdfs6 != hdfs7
    assert hdfs2 != hdfs3
    assert hdfs3 != hdfs4
    assert hdfs7 != hdfs5
    assert hdfs2 != hdfs3
    assert hdfs3 != hdfs4
    assert hdfs7 != hdfs8
    assert hdfs8 == hdfs9
    assert hdfs10 != hdfs9
    assert hdfs11 != hdfs8

    with pytest.raises(TypeError):
        HadoopFileSystem()
    with pytest.raises(TypeError):
        HadoopFileSystem.from_uri(3)

    for fs in [hdfs1, hdfs2, hdfs3, hdfs4, hdfs5, hdfs6, hdfs7, hdfs8,
               hdfs9, hdfs10, hdfs11]:
        assert pickle.loads(pickle.dumps(fs)) == fs

    host, port, user = hdfs_connection

    hdfs = HadoopFileSystem(host, port, user=user)
    assert hdfs.get_file_info(FileSelector('/'))

    hdfs = HadoopFileSystem.from_uri(
        "hdfs://{}:{}/?user={}".format(host, port, user)
    )
    assert hdfs.get_file_info(FileSelector('/'))


@pytest.mark.parametrize(('uri', 'expected_klass', 'expected_path'), [
    # leading slashes are removed intentionally, because MockFileSystem doesn't
    # have a distinction between relative and absolute paths
    ('mock:', _MockFileSystem, ''),
    ('mock:foo/bar', _MockFileSystem, 'foo/bar'),
    ('mock:/foo/bar', _MockFileSystem, 'foo/bar'),
    ('mock:///foo/bar', _MockFileSystem, 'foo/bar'),
    ('mock:///some%20path/%C3%A9', _MockFileSystem, 'some path/é'),
    ('file:/', LocalFileSystem, '/'),
    ('file:///', LocalFileSystem, '/'),
    ('file:/foo/bar', LocalFileSystem, '/foo/bar'),
    ('file:///foo/bar', LocalFileSystem, '/foo/bar'),
    ('file:///some%20path/%C3%A9', LocalFileSystem, '/some path/é'),
    # no %-decoding for non-URI inputs
    ('/', LocalFileSystem, '/'),
    ('/foo/bar', LocalFileSystem, '/foo/bar'),
    ('/some path/%20é', LocalFileSystem, '/some path/%20é'),
])
def test_filesystem_from_uri(uri, expected_klass, expected_path):
    fs, path = FileSystem.from_uri(uri)
    assert isinstance(fs, expected_klass)
    assert path == expected_path


@pytest.mark.parametrize(
    'path',
    ['', '/', 'foo/bar', '/foo/bar', __file__]
)
def test_filesystem_from_path_object(path):
    p = pathlib.Path(path)
    fs, path = FileSystem.from_uri(p)
    assert isinstance(fs, LocalFileSystem)
    assert path == p.resolve().absolute().as_posix()


@pytest.mark.s3
def test_filesystem_from_uri_s3(s3_server):
    from pyarrow.fs import S3FileSystem

    host, port, access_key, secret_key = s3_server['connection']

    uri = "s3://{}:{}@mybucket/foo/bar?scheme=http&endpoint_override={}:{}"\
          "&allow_bucket_creation=True" \
          .format(access_key, secret_key, host, port)

    fs, path = FileSystem.from_uri(uri)
    assert isinstance(fs, S3FileSystem)
    assert path == "mybucket/foo/bar"

    fs.create_dir(path)
    [info] = fs.get_file_info([path])
    assert info.path == path
    assert info.type == FileType.Directory


@pytest.mark.gcs
def test_filesystem_from_uri_gcs(gcs_server):
    from pyarrow.fs import GcsFileSystem

    host, port = gcs_server['connection']

    uri = ("gs://anonymous@" +
           f"mybucket/foo/bar?scheme=http&endpoint_override={host}:{port}&" +
           "retry_limit_seconds=5")

    fs, path = FileSystem.from_uri(uri)
    assert isinstance(fs, GcsFileSystem)
    assert path == "mybucket/foo/bar"

    fs.create_dir(path)
    [info] = fs.get_file_info([path])
    assert info.path == path
    assert info.type == FileType.Directory


def test_py_filesystem():
    handler = DummyHandler()
    fs = PyFileSystem(handler)
    assert isinstance(fs, PyFileSystem)
    assert fs.type_name == "py::dummy"
    assert fs.handler is handler

    with pytest.raises(TypeError):
        PyFileSystem(None)


def test_py_filesystem_equality():
    handler1 = DummyHandler(1)
    handler2 = DummyHandler(2)
    handler3 = DummyHandler(2)
    fs1 = PyFileSystem(handler1)
    fs2 = PyFileSystem(handler1)
    fs3 = PyFileSystem(handler2)
    fs4 = PyFileSystem(handler3)

    assert fs2 is not fs1
    assert fs3 is not fs2
    assert fs4 is not fs3
    assert fs2 == fs1  # Same handler
    assert fs3 != fs2  # Unequal handlers
    assert fs4 == fs3  # Equal handlers

    assert fs1 != LocalFileSystem()
    assert fs1 != object()


def test_py_filesystem_pickling():
    handler = DummyHandler()
    fs = PyFileSystem(handler)

    serialized = pickle.dumps(fs)
    restored = pickle.loads(serialized)
    assert isinstance(restored, FileSystem)
    assert restored == fs
    assert restored.handler == handler
    assert restored.type_name == "py::dummy"


def test_py_filesystem_lifetime():
    handler = DummyHandler()
    fs = PyFileSystem(handler)
    assert isinstance(fs, PyFileSystem)
    wr = weakref.ref(handler)
    handler = None
    assert wr() is not None
    fs = None
    assert wr() is None

    # Taking the .handler attribute doesn't wreck reference counts
    handler = DummyHandler()
    fs = PyFileSystem(handler)
    wr = weakref.ref(handler)
    handler = None
    assert wr() is fs.handler
    assert wr() is not None
    fs = None
    assert wr() is None


def test_py_filesystem_get_file_info():
    handler = DummyHandler()
    fs = PyFileSystem(handler)

    [info] = fs.get_file_info(['some/dir'])
    assert info.path == 'some/dir'
    assert info.type == FileType.Directory

    [info] = fs.get_file_info(['some/file'])
    assert info.path == 'some/file'
    assert info.type == FileType.File

    [info] = fs.get_file_info(['notfound'])
    assert info.path == 'notfound'
    assert info.type == FileType.NotFound

    with pytest.raises(TypeError):
        fs.get_file_info(['badtype'])

    with pytest.raises(IOError):
        fs.get_file_info(['xxx'])


def test_py_filesystem_get_file_info_selector():
    handler = DummyHandler()
    fs = PyFileSystem(handler)

    selector = FileSelector(base_dir="somedir")
    infos = fs.get_file_info(selector)
    assert len(infos) == 2
    assert infos[0].path == "somedir/file1"
    assert infos[0].type == FileType.File
    assert infos[0].size == 123
    assert infos[1].path == "somedir/subdir1"
    assert infos[1].type == FileType.Directory
    assert infos[1].size is None

    selector = FileSelector(base_dir="somedir", recursive=True)
    infos = fs.get_file_info(selector)
    assert len(infos) == 3
    assert infos[0].path == "somedir/file1"
    assert infos[1].path == "somedir/subdir1"
    assert infos[2].path == "somedir/subdir1/file2"

    selector = FileSelector(base_dir="notfound")
    with pytest.raises(FileNotFoundError):
        fs.get_file_info(selector)

    selector = FileSelector(base_dir="notfound", allow_not_found=True)
    assert fs.get_file_info(selector) == []


def test_py_filesystem_ops():
    handler = DummyHandler()
    fs = PyFileSystem(handler)

    fs.create_dir("recursive", recursive=True)
    fs.create_dir("non-recursive", recursive=False)
    with pytest.raises(IOError):
        fs.create_dir("foobar")

    fs.delete_dir("delete_dir")
    fs.delete_dir_contents("delete_dir_contents")
    for path in ("", "/", "//"):
        with pytest.raises(ValueError):
            fs.delete_dir_contents(path)
        fs.delete_dir_contents(path, accept_root_dir=True)
    fs.delete_file("delete_file")
    fs.move("move_from", "move_to")
    fs.copy_file("copy_file_from", "copy_file_to")


def test_py_open_input_stream():
    fs = PyFileSystem(DummyHandler())

    with fs.open_input_stream("somefile") as f:
        assert f.read() == b"somefile:input_stream"
    with pytest.raises(FileNotFoundError):
        fs.open_input_stream("notfound")


def test_py_open_input_file():
    fs = PyFileSystem(DummyHandler())

    with fs.open_input_file("somefile") as f:
        assert f.read() == b"somefile:input_file"
    with pytest.raises(FileNotFoundError):
        fs.open_input_file("notfound")


def test_py_open_output_stream():
    fs = PyFileSystem(DummyHandler())

    with fs.open_output_stream("somefile") as f:
        f.write(b"data")


def test_py_open_append_stream():
    fs = PyFileSystem(DummyHandler())

    with fs.open_append_stream("somefile") as f:
        f.write(b"data")


@pytest.mark.s3
def test_s3_real_aws():
    # Exercise connection code with an AWS-backed S3 bucket.
    # This is a minimal integration check for ARROW-9261 and similar issues.
    from pyarrow.fs import S3FileSystem
    default_region = (os.environ.get('PYARROW_TEST_S3_REGION') or
                      'us-east-1')
    fs = S3FileSystem(anonymous=True)
    assert fs.region == default_region

    fs = S3FileSystem(anonymous=True, region='us-east-2')
    entries = fs.get_file_info(FileSelector(
        'voltrondata-labs-datasets/nyc-taxi'))
    assert len(entries) > 0
    key = 'voltrondata-labs-datasets/nyc-taxi/year=2019/month=6/part-0.parquet'
    with fs.open_input_stream(key) as f:
        md = f.metadata()
        assert 'Content-Type' in md
        assert md['Last-Modified'] == b'2022-07-12T23:32:00Z'
        # For some reason, the header value is quoted
        # (both with AWS and Minio)
        assert md['ETag'] == b'"4c6a76826a695c6ac61592bc30cda3df-16"'


@pytest.mark.s3
def test_s3_real_aws_region_selection():
    # Taken from a registry of open S3-hosted datasets
    # at https://github.com/awslabs/open-data-registry
    fs, path = FileSystem.from_uri('s3://mf-nwp-models/README.txt')
    assert fs.region == 'eu-west-1'
    with fs.open_input_stream(path) as f:
        assert b"Meteo-France Atmospheric models on AWS" in f.read(50)

    # Passing an explicit region disables auto-selection
    fs, path = FileSystem.from_uri(
        's3://mf-nwp-models/README.txt?region=us-east-2')
    assert fs.region == 'us-east-2'
    # Reading from the wrong region may still work for public buckets...

    # Non-existent bucket (hopefully, otherwise need to fix this test)
    with pytest.raises(IOError, match="Bucket '.*' not found"):
        FileSystem.from_uri('s3://x-arrow-non-existent-bucket')
    fs, path = FileSystem.from_uri(
        's3://x-arrow-non-existent-bucket?region=us-east-3')
    assert fs.region == 'us-east-3'


@pytest.mark.s3
def test_resolve_s3_region():
    from pyarrow.fs import resolve_s3_region
    assert resolve_s3_region('voltrondata-labs-datasets') == 'us-east-2'
    assert resolve_s3_region('mf-nwp-models') == 'eu-west-1'

    with pytest.raises(ValueError, match="Not a valid bucket name"):
        resolve_s3_region('foo/bar')
    with pytest.raises(ValueError, match="Not a valid bucket name"):
        resolve_s3_region('s3:bucket')


@pytest.mark.s3
def test_copy_files(s3_connection, s3fs, tempdir):
    fs = s3fs["fs"]
    pathfn = s3fs["pathfn"]

    # create test file on S3 filesystem
    path = pathfn('c.txt')
    with fs.open_output_stream(path) as f:
        f.write(b'test')

    # create URI for created file
    host, port, access_key, secret_key = s3_connection
    source_uri = (
        f"s3://{access_key}:{secret_key}@{path}"
        f"?scheme=http&endpoint_override={host}:{port}"
    )
    # copy from S3 URI to local file
    local_path1 = str(tempdir / "c_copied1.txt")
    copy_files(source_uri, local_path1)

    localfs = LocalFileSystem()
    with localfs.open_input_stream(local_path1) as f:
        assert f.read() == b"test"

    # copy from S3 path+filesystem to local file
    local_path2 = str(tempdir / "c_copied2.txt")
    copy_files(path, local_path2, source_filesystem=fs)
    with localfs.open_input_stream(local_path2) as f:
        assert f.read() == b"test"

    # copy to local file with URI
    local_path3 = str(tempdir / "c_copied3.txt")
    destination_uri = _filesystem_uri(local_path3)  # file://
    copy_files(source_uri, destination_uri)

    with localfs.open_input_stream(local_path3) as f:
        assert f.read() == b"test"

    # copy to local file with path+filesystem
    local_path4 = str(tempdir / "c_copied4.txt")
    copy_files(source_uri, local_path4, destination_filesystem=localfs)

    with localfs.open_input_stream(local_path4) as f:
        assert f.read() == b"test"

    # copy with additional options
    local_path5 = str(tempdir / "c_copied5.txt")
    copy_files(source_uri, local_path5, chunk_size=1, use_threads=False)

    with localfs.open_input_stream(local_path5) as f:
        assert f.read() == b"test"


def test_copy_files_directory(tempdir):
    localfs = LocalFileSystem()

    # create source directory with 2 files
    source_dir = tempdir / "source"
    source_dir.mkdir()
    with localfs.open_output_stream(str(source_dir / "file1")) as f:
        f.write(b'test1')
    with localfs.open_output_stream(str(source_dir / "file2")) as f:
        f.write(b'test2')

    def check_copied_files(destination_dir):
        with localfs.open_input_stream(str(destination_dir / "file1")) as f:
            assert f.read() == b"test1"
        with localfs.open_input_stream(str(destination_dir / "file2")) as f:
            assert f.read() == b"test2"

    # Copy directory with local file paths
    destination_dir1 = tempdir / "destination1"
    # TODO need to create?
    destination_dir1.mkdir()
    copy_files(str(source_dir), str(destination_dir1))
    check_copied_files(destination_dir1)

    # Copy directory with path+filesystem
    destination_dir2 = tempdir / "destination2"
    destination_dir2.mkdir()
    copy_files(str(source_dir), str(destination_dir2),
               source_filesystem=localfs, destination_filesystem=localfs)
    check_copied_files(destination_dir2)

    # Copy directory with URI
    destination_dir3 = tempdir / "destination3"
    destination_dir3.mkdir()
    source_uri = _filesystem_uri(str(source_dir))  # file://
    destination_uri = _filesystem_uri(str(destination_dir3))
    copy_files(source_uri, destination_uri)
    check_copied_files(destination_dir3)

    # Copy directory with Path objects
    destination_dir4 = tempdir / "destination4"
    destination_dir4.mkdir()
    copy_files(source_dir, destination_dir4)
    check_copied_files(destination_dir4)

    # copy with additional non-default options
    destination_dir5 = tempdir / "destination5"
    destination_dir5.mkdir()
    copy_files(source_dir, destination_dir5, chunk_size=1, use_threads=False)
    check_copied_files(destination_dir5)

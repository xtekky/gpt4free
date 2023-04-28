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

import os
import pickle
import random
import unittest
from io import BytesIO
from os.path import join as pjoin

import numpy as np
import pytest

import pyarrow as pa
from pyarrow.pandas_compat import _pandas_api
from pyarrow.tests import util
from pyarrow.tests.parquet.common import _test_dataframe
from pyarrow.tests.parquet.test_dataset import (
    _test_read_common_metadata_files, _test_write_to_dataset_with_partitions,
    _test_write_to_dataset_no_partitions
)
from pyarrow.util import guid

# ----------------------------------------------------------------------
# HDFS tests


def check_libhdfs_present():
    if not pa.have_libhdfs():
        message = 'No libhdfs available on system'
        if os.environ.get('PYARROW_HDFS_TEST_LIBHDFS_REQUIRE'):
            pytest.fail(message)
        else:
            pytest.skip(message)


def hdfs_test_client():
    host = os.environ.get('ARROW_HDFS_TEST_HOST', 'default')
    user = os.environ.get('ARROW_HDFS_TEST_USER', None)
    try:
        port = int(os.environ.get('ARROW_HDFS_TEST_PORT', 0))
    except ValueError:
        raise ValueError('Env variable ARROW_HDFS_TEST_PORT was not '
                         'an integer')

    with pytest.warns(FutureWarning):
        return pa.hdfs.connect(host, port, user)


@pytest.mark.hdfs
class HdfsTestCases:

    def _make_test_file(self, hdfs, test_name, test_path, test_data):
        base_path = pjoin(self.tmp_path, test_name)
        hdfs.mkdir(base_path)

        full_path = pjoin(base_path, test_path)

        with hdfs.open(full_path, 'wb') as f:
            f.write(test_data)

        return full_path

    @classmethod
    def setUpClass(cls):
        cls.check_driver()
        cls.hdfs = hdfs_test_client()
        cls.tmp_path = '/tmp/pyarrow-test-{}'.format(random.randint(0, 1000))
        cls.hdfs.mkdir(cls.tmp_path)

    @classmethod
    def tearDownClass(cls):
        cls.hdfs.delete(cls.tmp_path, recursive=True)
        cls.hdfs.close()

    def test_pickle(self):
        s = pickle.dumps(self.hdfs)
        h2 = pickle.loads(s)
        assert h2.is_open
        assert h2.host == self.hdfs.host
        assert h2.port == self.hdfs.port
        assert h2.user == self.hdfs.user
        assert h2.kerb_ticket == self.hdfs.kerb_ticket
        # smoketest unpickled client works
        h2.ls(self.tmp_path)

    def test_cat(self):
        path = pjoin(self.tmp_path, 'cat-test')

        data = b'foobarbaz'
        with self.hdfs.open(path, 'wb') as f:
            f.write(data)

        contents = self.hdfs.cat(path)
        assert contents == data

    def test_capacity_space(self):
        capacity = self.hdfs.get_capacity()
        space_used = self.hdfs.get_space_used()
        disk_free = self.hdfs.df()

        assert capacity > 0
        assert capacity > space_used
        assert disk_free == (capacity - space_used)

    def test_close(self):
        client = hdfs_test_client()
        assert client.is_open
        client.close()
        assert not client.is_open

        with pytest.raises(Exception):
            client.ls('/')

    def test_mkdir(self):
        path = pjoin(self.tmp_path, 'test-dir/test-dir')
        parent_path = pjoin(self.tmp_path, 'test-dir')

        self.hdfs.mkdir(path)
        assert self.hdfs.exists(path)

        self.hdfs.delete(parent_path, recursive=True)
        assert not self.hdfs.exists(path)

    def test_mv_rename(self):
        path = pjoin(self.tmp_path, 'mv-test')
        new_path = pjoin(self.tmp_path, 'mv-new-test')

        data = b'foobarbaz'
        with self.hdfs.open(path, 'wb') as f:
            f.write(data)

        assert self.hdfs.exists(path)
        self.hdfs.mv(path, new_path)
        assert not self.hdfs.exists(path)
        assert self.hdfs.exists(new_path)

        assert self.hdfs.cat(new_path) == data

        self.hdfs.rename(new_path, path)
        assert self.hdfs.cat(path) == data

    def test_info(self):
        path = pjoin(self.tmp_path, 'info-base')
        file_path = pjoin(path, 'ex')
        self.hdfs.mkdir(path)

        data = b'foobarbaz'
        with self.hdfs.open(file_path, 'wb') as f:
            f.write(data)

        path_info = self.hdfs.info(path)
        file_path_info = self.hdfs.info(file_path)

        assert path_info['kind'] == 'directory'

        assert file_path_info['kind'] == 'file'
        assert file_path_info['size'] == len(data)

    def test_exists_isdir_isfile(self):
        dir_path = pjoin(self.tmp_path, 'info-base')
        file_path = pjoin(dir_path, 'ex')
        missing_path = pjoin(dir_path, 'this-path-is-missing')

        self.hdfs.mkdir(dir_path)
        with self.hdfs.open(file_path, 'wb') as f:
            f.write(b'foobarbaz')

        assert self.hdfs.exists(dir_path)
        assert self.hdfs.exists(file_path)
        assert not self.hdfs.exists(missing_path)

        assert self.hdfs.isdir(dir_path)
        assert not self.hdfs.isdir(file_path)
        assert not self.hdfs.isdir(missing_path)

        assert not self.hdfs.isfile(dir_path)
        assert self.hdfs.isfile(file_path)
        assert not self.hdfs.isfile(missing_path)

    def test_disk_usage(self):
        path = pjoin(self.tmp_path, 'disk-usage-base')
        p1 = pjoin(path, 'p1')
        p2 = pjoin(path, 'p2')

        subdir = pjoin(path, 'subdir')
        p3 = pjoin(subdir, 'p3')

        if self.hdfs.exists(path):
            self.hdfs.delete(path, True)

        self.hdfs.mkdir(path)
        self.hdfs.mkdir(subdir)

        data = b'foobarbaz'

        for file_path in [p1, p2, p3]:
            with self.hdfs.open(file_path, 'wb') as f:
                f.write(data)

        assert self.hdfs.disk_usage(path) == len(data) * 3

    def test_ls(self):
        base_path = pjoin(self.tmp_path, 'ls-test')
        self.hdfs.mkdir(base_path)

        dir_path = pjoin(base_path, 'a-dir')
        f1_path = pjoin(base_path, 'a-file-1')

        self.hdfs.mkdir(dir_path)

        f = self.hdfs.open(f1_path, 'wb')
        f.write(b'a' * 10)

        contents = sorted(self.hdfs.ls(base_path, False))
        assert contents == [dir_path, f1_path]

    def test_chmod_chown(self):
        path = pjoin(self.tmp_path, 'chmod-test')
        with self.hdfs.open(path, 'wb') as f:
            f.write(b'a' * 10)

    def test_download_upload(self):
        base_path = pjoin(self.tmp_path, 'upload-test')

        data = b'foobarbaz'
        buf = BytesIO(data)
        buf.seek(0)

        self.hdfs.upload(base_path, buf)

        out_buf = BytesIO()
        self.hdfs.download(base_path, out_buf)
        out_buf.seek(0)
        assert out_buf.getvalue() == data

    def test_file_context_manager(self):
        path = pjoin(self.tmp_path, 'ctx-manager')

        data = b'foo'
        with self.hdfs.open(path, 'wb') as f:
            f.write(data)

        with self.hdfs.open(path, 'rb') as f:
            assert f.size() == 3
            result = f.read(10)
            assert result == data

    def test_open_not_exist(self):
        path = pjoin(self.tmp_path, 'does-not-exist-123')

        with pytest.raises(FileNotFoundError):
            self.hdfs.open(path)

    def test_open_write_error(self):
        with pytest.raises((FileExistsError, IsADirectoryError)):
            self.hdfs.open('/', 'wb')

    def test_read_whole_file(self):
        path = pjoin(self.tmp_path, 'read-whole-file')

        data = b'foo' * 1000
        with self.hdfs.open(path, 'wb') as f:
            f.write(data)

        with self.hdfs.open(path, 'rb') as f:
            result = f.read()

        assert result == data

    def _write_multiple_hdfs_pq_files(self, tmpdir):
        import pyarrow.parquet as pq
        nfiles = 10
        size = 5
        test_data = []
        for i in range(nfiles):
            df = _test_dataframe(size, seed=i)

            df['index'] = np.arange(i * size, (i + 1) * size)

            # Hack so that we don't have a dtype cast in v1 files
            df['uint32'] = df['uint32'].astype(np.int64)

            path = pjoin(tmpdir, '{}.parquet'.format(i))

            table = pa.Table.from_pandas(df, preserve_index=False)
            with self.hdfs.open(path, 'wb') as f:
                pq.write_table(table, f)

            test_data.append(table)

        expected = pa.concat_tables(test_data)
        return expected

    @pytest.mark.pandas
    @pytest.mark.parquet
    def test_read_multiple_parquet_files(self):

        tmpdir = pjoin(self.tmp_path, 'multi-parquet-' + guid())

        self.hdfs.mkdir(tmpdir)

        expected = self._write_multiple_hdfs_pq_files(tmpdir)
        result = self.hdfs.read_parquet(tmpdir)

        _pandas_api.assert_frame_equal(result.to_pandas()
                                       .sort_values(by='index')
                                       .reset_index(drop=True),
                                       expected.to_pandas())

    @pytest.mark.pandas
    @pytest.mark.parquet
    def test_read_multiple_parquet_files_with_uri(self):
        import pyarrow.parquet as pq

        tmpdir = pjoin(self.tmp_path, 'multi-parquet-uri-' + guid())

        self.hdfs.mkdir(tmpdir)

        expected = self._write_multiple_hdfs_pq_files(tmpdir)
        path = _get_hdfs_uri(tmpdir)
        result = pq.read_table(path)

        _pandas_api.assert_frame_equal(result.to_pandas()
                                       .sort_values(by='index')
                                       .reset_index(drop=True),
                                       expected.to_pandas())

    @pytest.mark.pandas
    @pytest.mark.parquet
    def test_read_write_parquet_files_with_uri(self):
        import pyarrow.parquet as pq

        tmpdir = pjoin(self.tmp_path, 'uri-parquet-' + guid())
        self.hdfs.mkdir(tmpdir)
        path = _get_hdfs_uri(pjoin(tmpdir, 'test.parquet'))

        size = 5
        df = _test_dataframe(size, seed=0)
        # Hack so that we don't have a dtype cast in v1 files
        df['uint32'] = df['uint32'].astype(np.int64)
        table = pa.Table.from_pandas(df, preserve_index=False)

        pq.write_table(table, path, filesystem=self.hdfs)

        result = pq.read_table(
            path, filesystem=self.hdfs, use_legacy_dataset=True
        ).to_pandas()

        _pandas_api.assert_frame_equal(result, df)

    @pytest.mark.parquet
    @pytest.mark.pandas
    def test_read_common_metadata_files(self):
        tmpdir = pjoin(self.tmp_path, 'common-metadata-' + guid())
        self.hdfs.mkdir(tmpdir)
        _test_read_common_metadata_files(self.hdfs, tmpdir)

    @pytest.mark.parquet
    @pytest.mark.pandas
    def test_write_to_dataset_with_partitions(self):
        tmpdir = pjoin(self.tmp_path, 'write-partitions-' + guid())
        self.hdfs.mkdir(tmpdir)
        _test_write_to_dataset_with_partitions(
            tmpdir, filesystem=self.hdfs)

    @pytest.mark.parquet
    @pytest.mark.pandas
    def test_write_to_dataset_no_partitions(self):
        tmpdir = pjoin(self.tmp_path, 'write-no_partitions-' + guid())
        self.hdfs.mkdir(tmpdir)
        _test_write_to_dataset_no_partitions(
            tmpdir, filesystem=self.hdfs)


class TestLibHdfs(HdfsTestCases, unittest.TestCase):

    @classmethod
    def check_driver(cls):
        check_libhdfs_present()

    def test_orphaned_file(self):
        hdfs = hdfs_test_client()
        file_path = self._make_test_file(hdfs, 'orphaned_file_test', 'fname',
                                         b'foobarbaz')

        f = hdfs.open(file_path)
        hdfs = None
        f = None  # noqa


def _get_hdfs_uri(path):
    host = os.environ.get('ARROW_HDFS_TEST_HOST', 'localhost')
    try:
        port = int(os.environ.get('ARROW_HDFS_TEST_PORT', 0))
    except ValueError:
        raise ValueError('Env variable ARROW_HDFS_TEST_PORT was not '
                         'an integer')
    uri = "hdfs://{}:{}{}".format(host, port, path)

    return uri


@pytest.mark.hdfs
@pytest.mark.pandas
@pytest.mark.parquet
@pytest.mark.fastparquet
def test_fastparquet_read_with_hdfs():
    from pandas.testing import assert_frame_equal

    check_libhdfs_present()
    try:
        import snappy  # noqa
    except ImportError:
        pytest.skip('fastparquet test requires snappy')

    import pyarrow.parquet as pq
    fastparquet = pytest.importorskip('fastparquet')

    fs = hdfs_test_client()

    df = util.make_dataframe()

    table = pa.Table.from_pandas(df)

    path = '/tmp/testing.parquet'
    with fs.open(path, 'wb') as f:
        pq.write_table(table, f)

    parquet_file = fastparquet.ParquetFile(path, open_with=fs.open)

    result = parquet_file.to_pandas()
    assert_frame_equal(result, df)

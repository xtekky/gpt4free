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
import posixpath
import sys
import warnings

from pyarrow.util import implements, _DEPR_MSG
from pyarrow.filesystem import FileSystem
import pyarrow._hdfsio as _hdfsio


class HadoopFileSystem(_hdfsio.HadoopFileSystem, FileSystem):
    """
    DEPRECATED: FileSystem interface for HDFS cluster.

    See pyarrow.hdfs.connect for full connection details

    .. deprecated:: 2.0
        ``pyarrow.hdfs.HadoopFileSystem`` is deprecated,
        please use ``pyarrow.fs.HadoopFileSystem`` instead.
    """

    def __init__(self, host="default", port=0, user=None, kerb_ticket=None,
                 driver='libhdfs', extra_conf=None):
        warnings.warn(
            _DEPR_MSG.format(
                "hdfs.HadoopFileSystem", "2.0.0", "fs.HadoopFileSystem"),
            FutureWarning, stacklevel=2)
        if driver == 'libhdfs':
            _maybe_set_hadoop_classpath()

        self._connect(host, port, user, kerb_ticket, extra_conf)

    def __reduce__(self):
        return (HadoopFileSystem, (self.host, self.port, self.user,
                                   self.kerb_ticket, self.extra_conf))

    def _isfilestore(self):
        """
        Return True if this is a Unix-style file store with directories.
        """
        return True

    @implements(FileSystem.isdir)
    def isdir(self, path):
        return super().isdir(path)

    @implements(FileSystem.isfile)
    def isfile(self, path):
        return super().isfile(path)

    @implements(FileSystem.delete)
    def delete(self, path, recursive=False):
        return super().delete(path, recursive)

    def mkdir(self, path, **kwargs):
        """
        Create directory in HDFS.

        Parameters
        ----------
        path : str
            Directory path to create, including any parent directories.

        Notes
        -----
        libhdfs does not support create_parents=False, so we ignore this here
        """
        return super().mkdir(path)

    @implements(FileSystem.rename)
    def rename(self, path, new_path):
        return super().rename(path, new_path)

    @implements(FileSystem.exists)
    def exists(self, path):
        return super().exists(path)

    def ls(self, path, detail=False):
        """
        Retrieve directory contents and metadata, if requested.

        Parameters
        ----------
        path : str
            HDFS path to retrieve contents of.
        detail : bool, default False
            If False, only return list of paths.

        Returns
        -------
        result : list of dicts (detail=True) or strings (detail=False)
        """
        return super().ls(path, detail)

    def walk(self, top_path):
        """
        Directory tree generator for HDFS, like os.walk.

        Parameters
        ----------
        top_path : str
            Root directory for tree traversal.

        Returns
        -------
        Generator yielding 3-tuple (dirpath, dirnames, filename)
        """
        contents = self.ls(top_path, detail=True)

        directories, files = _libhdfs_walk_files_dirs(top_path, contents)
        yield top_path, directories, files
        for dirname in directories:
            yield from self.walk(self._path_join(top_path, dirname))


def _maybe_set_hadoop_classpath():
    import re

    if re.search(r'hadoop-common[^/]+.jar', os.environ.get('CLASSPATH', '')):
        return

    if 'HADOOP_HOME' in os.environ:
        if sys.platform != 'win32':
            classpath = _derive_hadoop_classpath()
        else:
            hadoop_bin = '{}/bin/hadoop'.format(os.environ['HADOOP_HOME'])
            classpath = _hadoop_classpath_glob(hadoop_bin)
    else:
        classpath = _hadoop_classpath_glob('hadoop')

    os.environ['CLASSPATH'] = classpath.decode('utf-8')


def _derive_hadoop_classpath():
    import subprocess

    find_args = ('find', '-L', os.environ['HADOOP_HOME'], '-name', '*.jar')
    find = subprocess.Popen(find_args, stdout=subprocess.PIPE)
    xargs_echo = subprocess.Popen(('xargs', 'echo'),
                                  stdin=find.stdout,
                                  stdout=subprocess.PIPE)
    jars = subprocess.check_output(('tr', "' '", "':'"),
                                   stdin=xargs_echo.stdout)
    hadoop_conf = os.environ["HADOOP_CONF_DIR"] \
        if "HADOOP_CONF_DIR" in os.environ \
        else os.environ["HADOOP_HOME"] + "/etc/hadoop"
    return (hadoop_conf + ":").encode("utf-8") + jars


def _hadoop_classpath_glob(hadoop_bin):
    import subprocess

    hadoop_classpath_args = (hadoop_bin, 'classpath', '--glob')
    return subprocess.check_output(hadoop_classpath_args)


def _libhdfs_walk_files_dirs(top_path, contents):
    files = []
    directories = []
    for c in contents:
        scrubbed_name = posixpath.split(c['name'])[1]
        if c['kind'] == 'file':
            files.append(scrubbed_name)
        else:
            directories.append(scrubbed_name)

    return directories, files


def connect(host="default", port=0, user=None, kerb_ticket=None,
            extra_conf=None):
    """
    DEPRECATED: Connect to an HDFS cluster.

    All parameters are optional and should only be set if the defaults need
    to be overridden.

    Authentication should be automatic if the HDFS cluster uses Kerberos.
    However, if a username is specified, then the ticket cache will likely
    be required.

    .. deprecated:: 2.0
        ``pyarrow.hdfs.connect`` is deprecated,
        please use ``pyarrow.fs.HadoopFileSystem`` instead.

    Parameters
    ----------
    host : NameNode. Set to "default" for fs.defaultFS from core-site.xml.
    port : NameNode's port. Set to 0 for default or logical (HA) nodes.
    user : Username when connecting to HDFS; None implies login user.
    kerb_ticket : Path to Kerberos ticket cache.
    extra_conf : dict, default None
      extra Key/Value pairs for config; Will override any
      hdfs-site.xml properties

    Notes
    -----
    The first time you call this method, it will take longer than usual due
    to JNI spin-up time.

    Returns
    -------
    filesystem : HadoopFileSystem
    """
    warnings.warn(
        _DEPR_MSG.format("hdfs.connect", "2.0.0", "fs.HadoopFileSystem"),
        FutureWarning, stacklevel=2
    )
    return _connect(
        host=host, port=port, user=user, kerb_ticket=kerb_ticket,
        extra_conf=extra_conf
    )


def _connect(host="default", port=0, user=None, kerb_ticket=None,
             extra_conf=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fs = HadoopFileSystem(host=host, port=port, user=user,
                              kerb_ticket=kerb_ticket,
                              extra_conf=extra_conf)
    return fs

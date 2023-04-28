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

# cython: language_level = 3

from pyarrow.lib cimport check_status
from pyarrow.includes.common cimport *
from pyarrow.includes.libarrow cimport *
from pyarrow.includes.libarrow_fs cimport *
from pyarrow._fs cimport FileSystem

from pyarrow.lib import frombytes, tobytes
from pyarrow.util import _stringify_path


cdef class HadoopFileSystem(FileSystem):
    """
    HDFS backed FileSystem implementation

    Parameters
    ----------
    host : str
        HDFS host to connect to. Set to "default" for fs.defaultFS from
        core-site.xml.
    port : int, default 8020
        HDFS port to connect to. Set to 0 for default or logical (HA) nodes.
    user : str, default None
        Username when connecting to HDFS; None implies login user.
    replication : int, default 3
        Number of copies each block will have.
    buffer_size : int, default 0
        If 0, no buffering will happen otherwise the size of the temporary read
        and write buffer.
    default_block_size : int, default None
        None means the default configuration for HDFS, a typical block size is
        128 MB.
    kerb_ticket : string or path, default None
        If not None, the path to the Kerberos ticket cache.
    extra_conf : dict, default None
        Extra key/value pairs for configuration; will override any
        hdfs-site.xml properties.

    Examples
    --------
    >>> from pyarrow import fs
    >>> hdfs = fs.HadoopFileSystem(host, port, user=user, kerb_ticket=ticket_cache_path) # doctest: +SKIP

    For usage of the methods see examples for :func:`~pyarrow.fs.LocalFileSystem`.
    """

    cdef:
        CHadoopFileSystem* hdfs

    def __init__(self, str host, int port=8020, *, str user=None,
                 int replication=3, int buffer_size=0,
                 default_block_size=None, kerb_ticket=None,
                 extra_conf=None):
        cdef:
            CHdfsOptions options
            shared_ptr[CHadoopFileSystem] wrapped

        if not host.startswith(('hdfs://', 'viewfs://')) and host != "default":
            # TODO(kszucs): do more sanitization
            host = 'hdfs://{}'.format(host)

        options.ConfigureEndPoint(tobytes(host), int(port))
        options.ConfigureReplication(replication)
        options.ConfigureBufferSize(buffer_size)

        if user is not None:
            options.ConfigureUser(tobytes(user))
        if default_block_size is not None:
            options.ConfigureBlockSize(default_block_size)
        if kerb_ticket is not None:
            options.ConfigureKerberosTicketCachePath(
                tobytes(_stringify_path(kerb_ticket)))
        if extra_conf is not None:
            for k, v in extra_conf.items():
                options.ConfigureExtraConf(tobytes(k), tobytes(v))

        with nogil:
            wrapped = GetResultValue(CHadoopFileSystem.Make(options))
        self.init(<shared_ptr[CFileSystem]> wrapped)

    cdef init(self, const shared_ptr[CFileSystem]& wrapped):
        FileSystem.init(self, wrapped)
        self.hdfs = <CHadoopFileSystem*> wrapped.get()

    @staticmethod
    def from_uri(uri):
        """
        Instantiate HadoopFileSystem object from an URI string.

        The following two calls are equivalent

        * ``HadoopFileSystem.from_uri('hdfs://localhost:8020/?user=test\
&replication=1')``
        * ``HadoopFileSystem('localhost', port=8020, user='test', \
replication=1)``

        Parameters
        ----------
        uri : str
            A string URI describing the connection to HDFS.
            In order to change the user, replication, buffer_size or
            default_block_size pass the values as query parts.

        Returns
        -------
        HadoopFileSystem
        """
        cdef:
            HadoopFileSystem self = HadoopFileSystem.__new__(HadoopFileSystem)
            shared_ptr[CHadoopFileSystem] wrapped
            CHdfsOptions options

        options = GetResultValue(CHdfsOptions.FromUriString(tobytes(uri)))
        with nogil:
            wrapped = GetResultValue(CHadoopFileSystem.Make(options))

        self.init(<shared_ptr[CFileSystem]> wrapped)
        return self

    @classmethod
    def _reconstruct(cls, kwargs):
        return cls(**kwargs)

    def __reduce__(self):
        cdef CHdfsOptions opts = self.hdfs.options()
        return (
            HadoopFileSystem._reconstruct, (dict(
                host=frombytes(opts.connection_config.host),
                port=opts.connection_config.port,
                user=frombytes(opts.connection_config.user),
                replication=opts.replication,
                buffer_size=opts.buffer_size,
                default_block_size=opts.default_block_size,
                kerb_ticket=frombytes(opts.connection_config.kerb_ticket),
                extra_conf={frombytes(k): frombytes(v)
                            for k, v in opts.connection_config.extra_conf},
            ),)
        )

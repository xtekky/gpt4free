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

import pytest
from pyarrow import Codec
from pyarrow import fs

groups = [
    'brotli',
    'bz2',
    'cython',
    'dataset',
    'hypothesis',
    'fastparquet',
    'gandiva',
    'gcs',
    'gdb',
    'gzip',
    'hdfs',
    'large_memory',
    'lz4',
    'memory_leak',
    'nopandas',
    'orc',
    'pandas',
    'parquet',
    'parquet_encryption',
    'plasma',
    's3',
    'snappy',
    'substrait',
    'tensorflow',
    'flight',
    'slow',
    'requires_testing_data',
    'zstd',
]

defaults = {
    'brotli': Codec.is_available('brotli'),
    'bz2': Codec.is_available('bz2'),
    'cython': False,
    'dataset': False,
    'fastparquet': False,
    'flight': False,
    'gandiva': False,
    'gcs': False,
    'gdb': True,
    'gzip': Codec.is_available('gzip'),
    'hdfs': False,
    'hypothesis': False,
    'large_memory': False,
    'lz4': Codec.is_available('lz4'),
    'memory_leak': False,
    'nopandas': False,
    'orc': False,
    'pandas': False,
    'parquet': False,
    'parquet_encryption': False,
    'plasma': False,
    'requires_testing_data': True,
    's3': False,
    'slow': False,
    'snappy': Codec.is_available('snappy'),
    'substrait': False,
    'tensorflow': False,
    'zstd': Codec.is_available('zstd'),
}

try:
    import cython  # noqa
    defaults['cython'] = True
except ImportError:
    pass

try:
    import fastparquet  # noqa
    defaults['fastparquet'] = True
except ImportError:
    pass

try:
    import pyarrow.gandiva  # noqa
    defaults['gandiva'] = True
except ImportError:
    pass

try:
    import pyarrow.dataset  # noqa
    defaults['dataset'] = True
except ImportError:
    pass

try:
    import pyarrow.orc  # noqa
    defaults['orc'] = True
except ImportError:
    pass

try:
    import pandas  # noqa
    defaults['pandas'] = True
except ImportError:
    defaults['nopandas'] = True

try:
    import pyarrow.parquet  # noqa
    defaults['parquet'] = True
except ImportError:
    pass

try:
    import pyarrow.parquet.encryption  # noqa
    defaults['parquet_encryption'] = True
except ImportError:
    pass


try:
    import pyarrow.plasma  # noqa
    defaults['plasma'] = True
except ImportError:
    pass

try:
    import tensorflow  # noqa
    defaults['tensorflow'] = True
except ImportError:
    pass

try:
    import pyarrow.flight  # noqa
    defaults['flight'] = True
except ImportError:
    pass

try:
    from pyarrow.fs import GcsFileSystem  # noqa
    defaults['gcs'] = True
except ImportError:
    pass


try:
    from pyarrow.fs import S3FileSystem  # noqa
    defaults['s3'] = True
except ImportError:
    pass

try:
    from pyarrow.fs import HadoopFileSystem  # noqa
    defaults['hdfs'] = True
except ImportError:
    pass

try:
    import pyarrow.substrait  # noqa
    defaults['substrait'] = True
except ImportError:
    pass


# Doctest should ignore files for the modules that are not built
def pytest_ignore_collect(path, config):
    if config.option.doctestmodules:
        # don't try to run doctests on the /tests directory
        if "/pyarrow/tests/" in str(path):
            return True

        doctest_groups = [
            'dataset',
            'orc',
            'parquet',
            'plasma',
            'flight',
            'substrait',
        ]

        # handle cuda, flight, etc
        for group in doctest_groups:
            if 'pyarrow/{}'.format(group) in str(path):
                if not defaults[group]:
                    return True

        if 'pyarrow/parquet/encryption' in str(path):
            if not defaults['parquet_encryption']:
                return True

        if 'pyarrow/cuda' in str(path):
            try:
                import pyarrow.cuda  # noqa
                return False
            except ImportError:
                return True

        if 'pyarrow/fs' in str(path):
            try:
                from pyarrow.fs import S3FileSystem  # noqa
                return False
            except ImportError:
                return True

    if getattr(config.option, "doctest_cython", False):
        if "/pyarrow/tests/" in str(path):
            return True
        if "/pyarrow/_parquet_encryption" in str(path):
            return True

    return False


# Save output files from doctest examples into temp dir
@pytest.fixture(autouse=True)
def _docdir(request):

    # Trigger ONLY for the doctests
    doctest_m = request.config.option.doctestmodules
    doctest_c = getattr(request.config.option, "doctest_cython", False)

    if doctest_m or doctest_c:

        # Get the fixture dynamically by its name.
        tmpdir = request.getfixturevalue('tmpdir')

        # Chdir only for the duration of the test.
        with tmpdir.as_cwd():
            yield

    else:
        yield


# Define doctest_namespace for fs module docstring import
@pytest.fixture(autouse=True)
def add_fs(doctest_namespace, request, tmp_path):

    # Trigger ONLY for the doctests
    doctest_m = request.config.option.doctestmodules
    doctest_c = getattr(request.config.option, "doctest_cython", False)

    if doctest_m or doctest_c:
        # fs import
        doctest_namespace["fs"] = fs

        # Creation of an object and file with data
        local = fs.LocalFileSystem()
        path = tmp_path / 'pyarrow-fs-example.dat'
        with local.open_output_stream(str(path)) as stream:
            stream.write(b'data')
        doctest_namespace["local"] = local
        doctest_namespace["local_path"] = str(tmp_path)
        doctest_namespace["path"] = str(path)
    yield

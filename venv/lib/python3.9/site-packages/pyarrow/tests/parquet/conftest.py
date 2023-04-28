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

from pyarrow.util import guid


@pytest.fixture(scope='module')
def datadir(base_datadir):
    return base_datadir / 'parquet'


@pytest.fixture
def s3_bucket(s3_server):
    boto3 = pytest.importorskip('boto3')
    botocore = pytest.importorskip('botocore')

    host, port, access_key, secret_key = s3_server['connection']
    s3 = boto3.resource(
        's3',
        endpoint_url='http://{}:{}'.format(host, port),
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=botocore.client.Config(signature_version='s3v4'),
        region_name='us-east-1'
    )
    bucket = s3.Bucket('test-s3fs')
    try:
        bucket.create()
    except Exception:
        # we get BucketAlreadyOwnedByYou error with fsspec handler
        pass
    return 'test-s3fs'


@pytest.fixture
def s3_example_s3fs(s3_server, s3_bucket):
    s3fs = pytest.importorskip('s3fs')

    host, port, access_key, secret_key = s3_server['connection']
    fs = s3fs.S3FileSystem(
        key=access_key,
        secret=secret_key,
        client_kwargs={
            'endpoint_url': 'http://{}:{}'.format(host, port)
        }
    )

    test_path = '{}/{}'.format(s3_bucket, guid())

    fs.mkdir(test_path)
    yield fs, test_path
    try:
        fs.rm(test_path, recursive=True)
    except FileNotFoundError:
        pass


@pytest.fixture
def s3_example_fs(s3_server):
    from pyarrow.fs import FileSystem

    host, port, access_key, secret_key = s3_server['connection']
    uri = (
        "s3://{}:{}@mybucket/data.parquet?scheme=http&endpoint_override={}:{}"
        .format(access_key, secret_key, host, port)
    )
    fs, path = FileSystem.from_uri(uri)

    fs.create_dir("mybucket")

    yield fs, uri, path

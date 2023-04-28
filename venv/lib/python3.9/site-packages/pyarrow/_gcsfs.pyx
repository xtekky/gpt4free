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

from pyarrow.lib cimport (check_status, pyarrow_wrap_metadata,
                          pyarrow_unwrap_metadata)
from pyarrow.lib import frombytes, tobytes, KeyValueMetadata, ensure_metadata
from pyarrow.includes.common cimport *
from pyarrow.includes.libarrow cimport *
from pyarrow.includes.libarrow_fs cimport *
from pyarrow._fs cimport FileSystem, TimePoint_to_ns, PyDateTime_to_TimePoint
from cython.operator cimport dereference as deref

from datetime import datetime, timedelta, timezone


cdef class GcsFileSystem(FileSystem):
    """
    Google Cloud Storage (GCS) backed FileSystem implementation

    By default uses the process described in https://google.aip.dev/auth/4110
    to resolve credentials. If not running on Google Cloud Platform (GCP),
    this generally requires the environment variable
    GOOGLE_APPLICATION_CREDENTIALS to point to a JSON file
    containing credentials.

    Note: GCS buckets are special and the operations available on them may be
    limited or more expensive than expected compared to local file systems.

    Note: When pickling a GcsFileSystem that uses default credentials, resolution
    credentials are not stored in the serialized data. Therefore, when unpickling
    it is assumed that the necessary credentials are in place for the target
    process.

    Parameters
    ----------
    anonymous : boolean, default False
        Whether to connect anonymously.
        If true, will not attempt to look up credentials using standard GCP
        configuration methods.
    access_token : str, default None
        GCP access token.  If provided, temporary credentials will be fetched by
        assuming this role; also, a `credential_token_expiration` must be
        specified as well.
    target_service_account : str, default None
        An optional service account to try to impersonate when accessing GCS. This
        requires the specified credential user or service account to have the necessary
        permissions.
    credential_token_expiration : datetime, default None
        Expiration for credential generated with an access token. Must be specified
        if `access_token` is specified.
    default_bucket_location : str, default 'US'
        GCP region to create buckets in.
    scheme : str, default 'https'
        GCS connection transport scheme.
    endpoint_override : str, default None
        Override endpoint with a connect string such as "localhost:9000"
    default_metadata : mapping or pyarrow.KeyValueMetadata, default None
        Default metadata for `open_output_stream`.  This will be ignored if
        non-empty metadata is passed to `open_output_stream`.
    retry_time_limit : timedelta, default None
        Set the maximum amount of time the GCS client will attempt to retry
        transient errors. Subsecond granularity is ignored.
    """

    cdef:
        CGcsFileSystem* gcsfs

    def __init__(self, *, bint anonymous=False, access_token=None,
                 target_service_account=None, credential_token_expiration=None,
                 default_bucket_location='US',
                 scheme=None,
                 endpoint_override=None,
                 default_metadata=None,
                 retry_time_limit=None):
        cdef:
            CGcsOptions options
            shared_ptr[CGcsFileSystem] wrapped
            double time_limit_seconds

        # Intentional use of truthiness because empty strings aren't valid and
        # for reconstruction from pickling will give empty strings.
        if anonymous and (target_service_account or access_token):
            raise ValueError(
                'anonymous option is not compatible with target_service_account and '
                'access_token'
            )
        elif bool(access_token) != bool(credential_token_expiration):
            raise ValueError(
                'access_token and credential_token_expiration must be '
                'specified together'
            )

        elif anonymous:
            options = CGcsOptions.Anonymous()
        elif access_token:
            if not isinstance(credential_token_expiration, datetime):
                raise ValueError(
                    "credential_token_expiration must be a datetime")
            options = CGcsOptions.FromAccessToken(
                tobytes(access_token),
                PyDateTime_to_TimePoint(<PyDateTime_DateTime*>credential_token_expiration))
        else:
            options = CGcsOptions.Defaults()

        # Target service account requires base credentials so
        # it is not part of the if/else chain above which only
        # handles base credentials.
        if target_service_account:
            options = CGcsOptions.FromImpersonatedServiceAccount(
                options.credentials, tobytes(target_service_account))

        options.default_bucket_location = tobytes(default_bucket_location)

        if scheme is not None:
            options.scheme = tobytes(scheme)
        if endpoint_override is not None:
            options.endpoint_override = tobytes(endpoint_override)
        if default_metadata is not None:
            options.default_metadata = pyarrow_unwrap_metadata(
                ensure_metadata(default_metadata))
        if retry_time_limit is not None:
            time_limit_seconds = retry_time_limit.total_seconds()
            options.retry_limit_seconds = time_limit_seconds

        with nogil:
            wrapped = GetResultValue(CGcsFileSystem.Make(options))

        self.init(<shared_ptr[CFileSystem]> wrapped)

    cdef init(self, const shared_ptr[CFileSystem]& wrapped):
        FileSystem.init(self, wrapped)
        self.gcsfs = <CGcsFileSystem*> wrapped.get()

    @classmethod
    def _reconstruct(cls, kwargs):
        return cls(**kwargs)

    def _expiration_datetime_from_options(self):
        expiration_ns = TimePoint_to_ns(
            self.gcsfs.options().credentials.expiration())
        if expiration_ns == 0:
            return None
        return datetime.fromtimestamp(expiration_ns / 1.0e9, timezone.utc)

    def __reduce__(self):
        cdef CGcsOptions opts = self.gcsfs.options()
        service_account = frombytes(opts.credentials.target_service_account())
        expiration_dt = self._expiration_datetime_from_options()
        retry_time_limit = None
        if opts.retry_limit_seconds.has_value():
            retry_time_limit = timedelta(
                seconds=opts.retry_limit_seconds.value())
        return (
            GcsFileSystem._reconstruct, (dict(
                access_token=frombytes(opts.credentials.access_token()),
                anonymous=opts.credentials.anonymous(),
                credential_token_expiration=expiration_dt,
                target_service_account=service_account,
                scheme=frombytes(opts.scheme),
                endpoint_override=frombytes(opts.endpoint_override),
                default_bucket_location=frombytes(
                    opts.default_bucket_location),
                default_metadata=pyarrow_wrap_metadata(opts.default_metadata),
                retry_time_limit=retry_time_limit
            ),))

    @property
    def default_bucket_location(self):
        """
        The GCP location this filesystem will write to.
        """
        return frombytes(self.gcsfs.options().default_bucket_location)

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
from pyarrow.lib import frombytes, tobytes, KeyValueMetadata
from pyarrow.includes.common cimport *
from pyarrow.includes.libarrow cimport *
from pyarrow.includes.libarrow_fs cimport *
from pyarrow._fs cimport FileSystem


cpdef enum S3LogLevel:
    Off = <int8_t> CS3LogLevel_Off
    Fatal = <int8_t> CS3LogLevel_Fatal
    Error = <int8_t> CS3LogLevel_Error
    Warn = <int8_t> CS3LogLevel_Warn
    Info = <int8_t> CS3LogLevel_Info
    Debug = <int8_t> CS3LogLevel_Debug
    Trace = <int8_t> CS3LogLevel_Trace


def initialize_s3(S3LogLevel log_level=S3LogLevel.Fatal):
    """
    Initialize S3 support

    Parameters
    ----------
    log_level : S3LogLevel
        level of logging

    Examples
    --------
    >>> fs.initialize_s3(fs.S3LogLevel.Error) # doctest: +SKIP
    """
    cdef CS3GlobalOptions options
    options.log_level = <CS3LogLevel> log_level
    check_status(CInitializeS3(options))


def finalize_s3():
    check_status(CFinalizeS3())


def resolve_s3_region(bucket):
    """
    Resolve the S3 region of a bucket.

    Parameters
    ----------
    bucket : str
        A S3 bucket name

    Returns
    -------
    region : str
        A S3 region name

    Examples
    --------
    >>> fs.resolve_s3_region('voltrondata-labs-datasets')
    'us-east-2'
    """
    cdef:
        c_string c_bucket
        c_string c_region

    c_bucket = tobytes(bucket)
    with nogil:
        c_region = GetResultValue(ResolveS3BucketRegion(c_bucket))

    return frombytes(c_region)


class S3RetryStrategy:
    """
    Base class for AWS retry strategies for use with S3.

    Parameters
    ----------
    max_attempts : int, default 3
        The maximum number of retry attempts to attempt before failing.
    """

    def __init__(self, max_attempts=3):
        self.max_attempts = max_attempts


class AwsStandardS3RetryStrategy(S3RetryStrategy):
    """
    Represents an AWS Standard retry strategy for use with S3.

    Parameters
    ----------
    max_attempts : int, default 3
        The maximum number of retry attempts to attempt before failing.
    """
    pass


class AwsDefaultS3RetryStrategy(S3RetryStrategy):
    """
    Represents an AWS Default retry strategy for use with S3.

    Parameters
    ----------
    max_attempts : int, default 3
        The maximum number of retry attempts to attempt before failing.
    """
    pass


cdef class S3FileSystem(FileSystem):
    """
    S3-backed FileSystem implementation

    If neither access_key nor secret_key are provided, and role_arn is also not
    provided, then attempts to initialize from AWS environment variables,
    otherwise both access_key and secret_key must be provided.

    If role_arn is provided instead of access_key and secret_key, temporary
    credentials will be fetched by issuing a request to STS to assume the
    specified role.

    Note: S3 buckets are special and the operations available on them may be
    limited or more expensive than desired.

    When S3FileSystem creates new buckets (assuming allow_bucket_creation is 
    True), it does not pass any non-default settings. In AWS S3, the bucket and 
    all objects will be not publicly visible, and will have no bucket policies 
    and no resource tags. To have more control over how buckets are created,
    use a different API to create them.

    Parameters
    ----------
    access_key : str, default None
        AWS Access Key ID. Pass None to use the standard AWS environment
        variables and/or configuration file.
    secret_key : str, default None
        AWS Secret Access key. Pass None to use the standard AWS environment
        variables and/or configuration file.
    session_token : str, default None
        AWS Session Token.  An optional session token, required if access_key
        and secret_key are temporary credentials from STS.
    anonymous : boolean, default False
        Whether to connect anonymously if access_key and secret_key are None.
        If true, will not attempt to look up credentials using standard AWS
        configuration methods.
    role_arn : str, default None
        AWS Role ARN.  If provided instead of access_key and secret_key,
        temporary credentials will be fetched by assuming this role.
    session_name : str, default None
        An optional identifier for the assumed role session.
    external_id : str, default None
        An optional unique identifier that might be required when you assume
        a role in another account.
    load_frequency : int, default 900
        The frequency (in seconds) with which temporary credentials from an
        assumed role session will be refreshed.
    region : str, default None
        AWS region to connect to. If not set, the AWS SDK will attempt to
        determine the region using heuristics such as environment variables,
        configuration profile, EC2 metadata, or default to 'us-east-1' when SDK
        version <1.8. One can also use :func:`pyarrow.fs.resolve_s3_region` to
        automatically resolve the region from a bucket name.
    request_timeout : double, default None
        Socket read timeouts on Windows and macOS, in seconds.
        If omitted, the AWS SDK default value is used (typically 3 seconds).
        This option is ignored on non-Windows, non-macOS systems.
    connect_timeout : double, default None
        Socket connection timeout, in seconds.
        If omitted, the AWS SDK default value is used (typically 1 second).
    scheme : str, default 'https'
        S3 connection transport scheme.
    endpoint_override : str, default None
        Override region with a connect string such as "localhost:9000"
    background_writes : boolean, default True
        Whether file writes will be issued in the background, without
        blocking.
    default_metadata : mapping or pyarrow.KeyValueMetadata, default None
        Default metadata for open_output_stream.  This will be ignored if
        non-empty metadata is passed to open_output_stream.
    proxy_options : dict or str, default None
        If a proxy is used, provide the options here. Supported options are:
        'scheme' (str: 'http' or 'https'; required), 'host' (str; required),
        'port' (int; required), 'username' (str; optional),
        'password' (str; optional).
        A proxy URI (str) can also be provided, in which case these options
        will be derived from the provided URI.
        The following are equivalent::

            S3FileSystem(proxy_options='http://username:password@localhost:8020')
            S3FileSystem(proxy_options={'scheme': 'http', 'host': 'localhost',
                                        'port': 8020, 'username': 'username',
                                        'password': 'password'})
    allow_bucket_creation : bool, default False
        Whether to allow CreateDir at the bucket-level. This option may also be 
        passed in a URI query parameter.
    allow_bucket_deletion : bool, default False
        Whether to allow DeleteDir at the bucket-level. This option may also be 
        passed in a URI query parameter.
    retry_strategy : S3RetryStrategy, default AwsStandardS3RetryStrategy(max_attempts=3)
        The retry strategy to use with S3; fail after max_attempts. Available
        strategies are AwsStandardS3RetryStrategy, AwsDefaultS3RetryStrategy.

    Examples
    --------
    >>> from pyarrow import fs
    >>> s3 = fs.S3FileSystem(region='us-west-2')
    >>> s3.get_file_info(fs.FileSelector(
    ...    'power-analysis-ready-datastore/power_901_constants.zarr/FROCEAN', recursive=True
    ... ))
    [<FileInfo for 'power-analysis-ready-datastore/power_901_constants.zarr/FROCEAN/.zarray...

    For usage of the methods see examples for :func:`~pyarrow.fs.LocalFileSystem`.
    """

    cdef:
        CS3FileSystem* s3fs

    def __init__(self, *, access_key=None, secret_key=None, session_token=None,
                 bint anonymous=False, region=None, request_timeout=None,
                 connect_timeout=None, scheme=None, endpoint_override=None,
                 bint background_writes=True, default_metadata=None,
                 role_arn=None, session_name=None, external_id=None,
                 load_frequency=900, proxy_options=None,
                 allow_bucket_creation=False, allow_bucket_deletion=False,
                 retry_strategy: S3RetryStrategy = AwsStandardS3RetryStrategy(max_attempts=3)):
        cdef:
            CS3Options options
            shared_ptr[CS3FileSystem] wrapped

        if access_key is not None and secret_key is None:
            raise ValueError(
                'In order to initialize with explicit credentials both '
                'access_key and secret_key must be provided, '
                '`secret_key` is not set.'
            )
        elif access_key is None and secret_key is not None:
            raise ValueError(
                'In order to initialize with explicit credentials both '
                'access_key and secret_key must be provided, '
                '`access_key` is not set.'
            )

        elif session_token is not None and (access_key is None or
                                            secret_key is None):
            raise ValueError(
                'In order to initialize a session with temporary credentials, '
                'both secret_key and access_key must be provided in addition '
                'to session_token.'
            )

        elif (access_key is not None or secret_key is not None):
            if anonymous:
                raise ValueError(
                    'Cannot pass anonymous=True together with access_key '
                    'and secret_key.')

            if role_arn:
                raise ValueError(
                    'Cannot provide role_arn with access_key and secret_key')

            if session_token is None:
                session_token = ""

            options = CS3Options.FromAccessKey(
                tobytes(access_key),
                tobytes(secret_key),
                tobytes(session_token)
            )
        elif anonymous:
            if role_arn:
                raise ValueError(
                    'Cannot provide role_arn with anonymous=True')

            options = CS3Options.Anonymous()
        elif role_arn:
            if session_name is None:
                session_name = ''
            if external_id is None:
                external_id = ''

            options = CS3Options.FromAssumeRole(
                tobytes(role_arn),
                tobytes(session_name),
                tobytes(external_id),
                load_frequency
            )
        else:
            options = CS3Options.Defaults()

        if region is not None:
            options.region = tobytes(region)
        if request_timeout is not None:
            options.request_timeout = request_timeout
        if connect_timeout is not None:
            options.connect_timeout = connect_timeout
        if scheme is not None:
            options.scheme = tobytes(scheme)
        if endpoint_override is not None:
            options.endpoint_override = tobytes(endpoint_override)
        if background_writes is not None:
            options.background_writes = background_writes
        if default_metadata is not None:
            if not isinstance(default_metadata, KeyValueMetadata):
                default_metadata = KeyValueMetadata(default_metadata)
            options.default_metadata = pyarrow_unwrap_metadata(
                default_metadata)

        if proxy_options is not None:
            if isinstance(proxy_options, dict):
                options.proxy_options.scheme = tobytes(proxy_options["scheme"])
                options.proxy_options.host = tobytes(proxy_options["host"])
                options.proxy_options.port = proxy_options["port"]
                proxy_username = proxy_options.get("username", None)
                if proxy_username:
                    options.proxy_options.username = tobytes(proxy_username)
                proxy_password = proxy_options.get("password", None)
                if proxy_password:
                    options.proxy_options.password = tobytes(proxy_password)
            elif isinstance(proxy_options, str):
                options.proxy_options = GetResultValue(
                    CS3ProxyOptions.FromUriString(tobytes(proxy_options)))
            else:
                raise TypeError(
                    "'proxy_options': expected 'dict' or 'str', "
                    f"got {type(proxy_options)} instead.")

        options.allow_bucket_creation = allow_bucket_creation
        options.allow_bucket_deletion = allow_bucket_deletion

        if isinstance(retry_strategy, AwsStandardS3RetryStrategy):
            options.retry_strategy = CS3RetryStrategy.GetAwsStandardRetryStrategy(
                retry_strategy.max_attempts)
        elif isinstance(retry_strategy, AwsDefaultS3RetryStrategy):
            options.retry_strategy = CS3RetryStrategy.GetAwsDefaultRetryStrategy(
                retry_strategy.max_attempts)
        else:
            raise ValueError(f'Invalid retry_strategy {retry_strategy!r}')

        with nogil:
            wrapped = GetResultValue(CS3FileSystem.Make(options))

        self.init(<shared_ptr[CFileSystem]> wrapped)

    cdef init(self, const shared_ptr[CFileSystem]& wrapped):
        FileSystem.init(self, wrapped)
        self.s3fs = <CS3FileSystem*> wrapped.get()

    @classmethod
    def _reconstruct(cls, kwargs):
        return cls(**kwargs)

    def __reduce__(self):
        cdef CS3Options opts = self.s3fs.options()

        # if creds were explicitly provided, then use them
        # else obtain them as they were last time.
        if opts.credentials_kind == CS3CredentialsKind_Explicit:
            access_key = frombytes(opts.GetAccessKey())
            secret_key = frombytes(opts.GetSecretKey())
            session_token = frombytes(opts.GetSessionToken())
        else:
            access_key = None
            secret_key = None
            session_token = None

        return (
            S3FileSystem._reconstruct, (dict(
                access_key=access_key,
                secret_key=secret_key,
                session_token=session_token,
                anonymous=(opts.credentials_kind ==
                           CS3CredentialsKind_Anonymous),
                region=frombytes(opts.region),
                scheme=frombytes(opts.scheme),
                connect_timeout=opts.connect_timeout,
                request_timeout=opts.request_timeout,
                endpoint_override=frombytes(opts.endpoint_override),
                role_arn=frombytes(opts.role_arn),
                session_name=frombytes(opts.session_name),
                external_id=frombytes(opts.external_id),
                load_frequency=opts.load_frequency,
                background_writes=opts.background_writes,
                allow_bucket_creation=opts.allow_bucket_creation,
                allow_bucket_deletion=opts.allow_bucket_deletion,
                default_metadata=pyarrow_wrap_metadata(opts.default_metadata),
                proxy_options={'scheme': frombytes(opts.proxy_options.scheme),
                               'host': frombytes(opts.proxy_options.host),
                               'port': opts.proxy_options.port,
                               'username': frombytes(
                                   opts.proxy_options.username),
                               'password': frombytes(
                                   opts.proxy_options.password)},
            ),)
        )

    @property
    def region(self):
        """
        The AWS region this filesystem connects to.
        """
        return frombytes(self.s3fs.region())

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

# distutils: language = c++

from pyarrow.includes.common cimport *
from pyarrow.includes.libarrow cimport *
from pyarrow.includes.libarrow_python cimport CTimePoint

cdef extern from "arrow/filesystem/api.h" namespace "arrow::fs" nogil:

    ctypedef enum CFileType "arrow::fs::FileType":
        CFileType_NotFound "arrow::fs::FileType::NotFound"
        CFileType_Unknown "arrow::fs::FileType::Unknown"
        CFileType_File "arrow::fs::FileType::File"
        CFileType_Directory "arrow::fs::FileType::Directory"

    cdef cppclass CFileInfo "arrow::fs::FileInfo":
        CFileInfo()
        CFileInfo(CFileInfo&&)
        CFileInfo& operator=(CFileInfo&&)
        CFileInfo(const CFileInfo&)
        CFileInfo& operator=(const CFileInfo&)

        CFileType type()
        void set_type(CFileType type)
        c_string path()
        void set_path(const c_string& path)
        c_string base_name()
        int64_t size()
        void set_size(int64_t size)
        c_string extension()
        CTimePoint mtime()
        void set_mtime(CTimePoint mtime)

    cdef cppclass CFileSelector "arrow::fs::FileSelector":
        CFileSelector()
        c_string base_dir
        c_bool allow_not_found
        c_bool recursive

    cdef cppclass CFileLocator "arrow::fs::FileLocator":
        shared_ptr[CFileSystem] filesystem
        c_string path

    cdef cppclass CFileSystem "arrow::fs::FileSystem":
        shared_ptr[CFileSystem] shared_from_this()
        c_string type_name() const
        CResult[c_string] NormalizePath(c_string path)
        CResult[CFileInfo] GetFileInfo(const c_string& path)
        CResult[vector[CFileInfo]] GetFileInfo(
            const vector[c_string]& paths)
        CResult[vector[CFileInfo]] GetFileInfo(const CFileSelector& select)
        CStatus CreateDir(const c_string& path, c_bool recursive)
        CStatus DeleteDir(const c_string& path)
        CStatus DeleteDirContents(const c_string& path, c_bool missing_dir_ok)
        CStatus DeleteRootDirContents()
        CStatus DeleteFile(const c_string& path)
        CStatus DeleteFiles(const vector[c_string]& paths)
        CStatus Move(const c_string& src, const c_string& dest)
        CStatus CopyFile(const c_string& src, const c_string& dest)
        CResult[shared_ptr[CInputStream]] OpenInputStream(
            const c_string& path)
        CResult[shared_ptr[CRandomAccessFile]] OpenInputFile(
            const c_string& path)
        CResult[shared_ptr[COutputStream]] OpenOutputStream(
            const c_string& path, const shared_ptr[const CKeyValueMetadata]&)
        CResult[shared_ptr[COutputStream]] OpenAppendStream(
            const c_string& path, const shared_ptr[const CKeyValueMetadata]&)
        c_bool Equals(const CFileSystem& other)
        c_bool Equals(shared_ptr[CFileSystem] other)

    CResult[shared_ptr[CFileSystem]] CFileSystemFromUri \
        "arrow::fs::FileSystemFromUri"(const c_string& uri, c_string* out_path)
    CResult[shared_ptr[CFileSystem]] CFileSystemFromUriOrPath \
        "arrow::fs::FileSystemFromUriOrPath"(const c_string& uri,
                                             c_string* out_path)

    cdef cppclass CFileSystemGlobalOptions \
            "arrow::fs::FileSystemGlobalOptions":
        c_string tls_ca_file_path
        c_string tls_ca_dir_path

    CStatus CFileSystemsInitialize "arrow::fs::Initialize" \
        (const CFileSystemGlobalOptions& options)

    cdef cppclass CLocalFileSystemOptions "arrow::fs::LocalFileSystemOptions":
        c_bool use_mmap

        @staticmethod
        CLocalFileSystemOptions Defaults()

        c_bool Equals(const CLocalFileSystemOptions& other)

    cdef cppclass CLocalFileSystem "arrow::fs::LocalFileSystem"(CFileSystem):
        CLocalFileSystem()
        CLocalFileSystem(CLocalFileSystemOptions)
        CLocalFileSystemOptions options()

    cdef cppclass CSubTreeFileSystem \
            "arrow::fs::SubTreeFileSystem"(CFileSystem):
        CSubTreeFileSystem(const c_string& base_path,
                           shared_ptr[CFileSystem] base_fs)
        c_string base_path()
        shared_ptr[CFileSystem] base_fs()

    ctypedef enum CS3LogLevel "arrow::fs::S3LogLevel":
        CS3LogLevel_Off "arrow::fs::S3LogLevel::Off"
        CS3LogLevel_Fatal "arrow::fs::S3LogLevel::Fatal"
        CS3LogLevel_Error "arrow::fs::S3LogLevel::Error"
        CS3LogLevel_Warn "arrow::fs::S3LogLevel::Warn"
        CS3LogLevel_Info "arrow::fs::S3LogLevel::Info"
        CS3LogLevel_Debug "arrow::fs::S3LogLevel::Debug"
        CS3LogLevel_Trace "arrow::fs::S3LogLevel::Trace"

    cdef struct CS3GlobalOptions "arrow::fs::S3GlobalOptions":
        CS3LogLevel log_level

    cdef cppclass CS3ProxyOptions "arrow::fs::S3ProxyOptions":
        c_string scheme
        c_string host
        int port
        c_string username
        c_string password
        c_bool Equals(const CS3ProxyOptions& other)

        @staticmethod
        CResult[CS3ProxyOptions] FromUriString "FromUri"(
            const c_string& uri_string)

    ctypedef enum CS3CredentialsKind "arrow::fs::S3CredentialsKind":
        CS3CredentialsKind_Anonymous "arrow::fs::S3CredentialsKind::Anonymous"
        CS3CredentialsKind_Default "arrow::fs::S3CredentialsKind::Default"
        CS3CredentialsKind_Explicit "arrow::fs::S3CredentialsKind::Explicit"
        CS3CredentialsKind_Role "arrow::fs::S3CredentialsKind::Role"
        CS3CredentialsKind_WebIdentity \
            "arrow::fs::S3CredentialsKind::WebIdentity"

    cdef cppclass CS3RetryStrategy "arrow::fs::S3RetryStrategy":
        @staticmethod
        shared_ptr[CS3RetryStrategy] GetAwsDefaultRetryStrategy(int64_t max_attempts)

        @staticmethod
        shared_ptr[CS3RetryStrategy] GetAwsStandardRetryStrategy(int64_t max_attempts)

    cdef cppclass CS3Options "arrow::fs::S3Options":
        c_string region
        double connect_timeout
        double request_timeout
        c_string endpoint_override
        c_string scheme
        c_bool background_writes
        c_bool allow_bucket_creation
        c_bool allow_bucket_deletion
        shared_ptr[const CKeyValueMetadata] default_metadata
        c_string role_arn
        c_string session_name
        c_string external_id
        int load_frequency
        CS3ProxyOptions proxy_options
        CS3CredentialsKind credentials_kind
        shared_ptr[CS3RetryStrategy] retry_strategy
        void ConfigureDefaultCredentials()
        void ConfigureAccessKey(const c_string& access_key,
                                const c_string& secret_key,
                                const c_string& session_token)
        c_string GetAccessKey()
        c_string GetSecretKey()
        c_string GetSessionToken()
        c_bool Equals(const CS3Options& other)

        @staticmethod
        CS3Options Defaults()

        @staticmethod
        CS3Options Anonymous()

        @staticmethod
        CS3Options FromAccessKey(const c_string& access_key,
                                 const c_string& secret_key,
                                 const c_string& session_token)

        @staticmethod
        CS3Options FromAssumeRole(const c_string& role_arn,
                                  const c_string& session_name,
                                  const c_string& external_id,
                                  const int load_frequency)

    cdef cppclass CS3FileSystem "arrow::fs::S3FileSystem"(CFileSystem):
        @staticmethod
        CResult[shared_ptr[CS3FileSystem]] Make(const CS3Options& options)
        CS3Options options()
        c_string region()

    cdef CStatus CInitializeS3 "arrow::fs::InitializeS3"(
        const CS3GlobalOptions& options)
    cdef CStatus CFinalizeS3 "arrow::fs::FinalizeS3"()

    cdef CResult[c_string] ResolveS3BucketRegion(const c_string& bucket)

    cdef cppclass CGcsCredentials "arrow::fs::GcsCredentials":
        c_bool anonymous()
        CTimePoint expiration()
        c_string access_token()
        c_string target_service_account()

    cdef cppclass CGcsOptions "arrow::fs::GcsOptions":
        CGcsCredentials credentials
        c_string endpoint_override
        c_string scheme
        c_string default_bucket_location
        optional[double] retry_limit_seconds
        shared_ptr[const CKeyValueMetadata] default_metadata
        c_bool Equals(const CS3Options& other)

        @staticmethod
        CGcsOptions Defaults()

        @staticmethod
        CGcsOptions Anonymous()

        @staticmethod
        CGcsOptions FromAccessToken(const c_string& access_token,
                                    CTimePoint expiration)

        @staticmethod
        CGcsOptions FromImpersonatedServiceAccount(const CGcsCredentials& base_credentials,
                                                   c_string& target_service_account)

    cdef cppclass CGcsFileSystem "arrow::fs::GcsFileSystem":
        @staticmethod
        CResult[shared_ptr[CGcsFileSystem]] Make(const CGcsOptions& options)
        CGcsOptions options()

    cdef cppclass CHdfsOptions "arrow::fs::HdfsOptions":
        HdfsConnectionConfig connection_config
        int32_t buffer_size
        int16_t replication
        int64_t default_block_size

        @staticmethod
        CResult[CHdfsOptions] FromUriString "FromUri"(
            const c_string& uri_string)
        void ConfigureEndPoint(c_string host, int port)
        void ConfigureDriver(c_bool use_hdfs3)
        void ConfigureReplication(int16_t replication)
        void ConfigureUser(c_string user_name)
        void ConfigureBufferSize(int32_t buffer_size)
        void ConfigureBlockSize(int64_t default_block_size)
        void ConfigureKerberosTicketCachePath(c_string path)
        void ConfigureExtraConf(c_string key, c_string value)

    cdef cppclass CHadoopFileSystem "arrow::fs::HadoopFileSystem"(CFileSystem):
        @staticmethod
        CResult[shared_ptr[CHadoopFileSystem]] Make(
            const CHdfsOptions& options)
        CHdfsOptions options()

    cdef cppclass CMockFileSystem "arrow::fs::internal::MockFileSystem"(
            CFileSystem):
        CMockFileSystem(CTimePoint current_time)

    CStatus CCopyFiles "arrow::fs::CopyFiles"(
        const vector[CFileLocator]& sources,
        const vector[CFileLocator]& destinations,
        const CIOContext& io_context,
        int64_t chunk_size, c_bool use_threads)
    CStatus CCopyFilesWithSelector "arrow::fs::CopyFiles"(
        const shared_ptr[CFileSystem]& source_fs,
        const CFileSelector& source_sel,
        const shared_ptr[CFileSystem]& destination_fs,
        const c_string& destination_base_dir,
        const CIOContext& io_context,
        int64_t chunk_size, c_bool use_threads)


# Callbacks for implementing Python filesystems
# Use typedef to emulate syntax for std::function<void(..)>
ctypedef void CallbackGetTypeName(object, c_string*)
ctypedef c_bool CallbackEquals(object, const CFileSystem&)

ctypedef void CallbackGetFileInfo(object, const c_string&, CFileInfo*)
ctypedef void CallbackGetFileInfoVector(object, const vector[c_string]&,
                                        vector[CFileInfo]*)
ctypedef void CallbackGetFileInfoSelector(object, const CFileSelector&,
                                          vector[CFileInfo]*)
ctypedef void CallbackCreateDir(object, const c_string&, c_bool)
ctypedef void CallbackDeleteDir(object, const c_string&)
ctypedef void CallbackDeleteDirContents(object, const c_string&, c_bool)
ctypedef void CallbackDeleteRootDirContents(object)
ctypedef void CallbackDeleteFile(object, const c_string&)
ctypedef void CallbackMove(object, const c_string&, const c_string&)
ctypedef void CallbackCopyFile(object, const c_string&, const c_string&)

ctypedef void CallbackOpenInputStream(object, const c_string&,
                                      shared_ptr[CInputStream]*)
ctypedef void CallbackOpenInputFile(object, const c_string&,
                                    shared_ptr[CRandomAccessFile]*)
ctypedef void CallbackOpenOutputStream(
    object, const c_string&, const shared_ptr[const CKeyValueMetadata]&,
    shared_ptr[COutputStream]*)
ctypedef void CallbackNormalizePath(object, const c_string&, c_string*)

cdef extern from "arrow/python/filesystem.h" namespace "arrow::py::fs" nogil:

    cdef cppclass CPyFileSystemVtable "arrow::py::fs::PyFileSystemVtable":
        PyFileSystemVtable()
        function[CallbackGetTypeName] get_type_name
        function[CallbackEquals] equals
        function[CallbackGetFileInfo] get_file_info
        function[CallbackGetFileInfoVector] get_file_info_vector
        function[CallbackGetFileInfoSelector] get_file_info_selector
        function[CallbackCreateDir] create_dir
        function[CallbackDeleteDir] delete_dir
        function[CallbackDeleteDirContents] delete_dir_contents
        function[CallbackDeleteRootDirContents] delete_root_dir_contents
        function[CallbackDeleteFile] delete_file
        function[CallbackMove] move
        function[CallbackCopyFile] copy_file
        function[CallbackOpenInputStream] open_input_stream
        function[CallbackOpenInputFile] open_input_file
        function[CallbackOpenOutputStream] open_output_stream
        function[CallbackOpenOutputStream] open_append_stream
        function[CallbackNormalizePath] normalize_path

    cdef cppclass CPyFileSystem "arrow::py::fs::PyFileSystem":
        @staticmethod
        shared_ptr[CPyFileSystem] Make(object handler,
                                       CPyFileSystemVtable vtable)

        PyObject* handler()

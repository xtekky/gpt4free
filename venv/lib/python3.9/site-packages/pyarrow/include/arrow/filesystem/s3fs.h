// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "arrow/filesystem/filesystem.h"
#include "arrow/util/macros.h"
#include "arrow/util/uri.h"

namespace Aws {
namespace Auth {

class AWSCredentialsProvider;
class STSAssumeRoleCredentialsProvider;

}  // namespace Auth
namespace STS {
class STSClient;
}
}  // namespace Aws

namespace arrow {
namespace fs {

/// Options for using a proxy for S3
struct ARROW_EXPORT S3ProxyOptions {
  std::string scheme;
  std::string host;
  int port = -1;
  std::string username;
  std::string password;

  /// Initialize from URI such as http://username:password@host:port
  /// or http://host:port
  static Result<S3ProxyOptions> FromUri(const std::string& uri);
  static Result<S3ProxyOptions> FromUri(const ::arrow::internal::Uri& uri);

  bool Equals(const S3ProxyOptions& other) const;
};

enum class S3CredentialsKind : int8_t {
  /// Anonymous access (no credentials used)
  Anonymous,
  /// Use default AWS credentials, configured through environment variables
  Default,
  /// Use explicitly-provided access key pair
  Explicit,
  /// Assume role through a role ARN
  Role,
  /// Use web identity token to assume role, configured through environment variables
  WebIdentity
};

/// Pure virtual class for describing custom S3 retry strategies
class ARROW_EXPORT S3RetryStrategy {
 public:
  virtual ~S3RetryStrategy() = default;

  /// Simple struct where each field corresponds to a field in Aws::Client::AWSError
  struct AWSErrorDetail {
    /// Corresponds to AWSError::GetErrorType()
    int error_type;
    /// Corresponds to AWSError::GetMessage()
    std::string message;
    /// Corresponds to AWSError::GetExceptionName()
    std::string exception_name;
    /// Corresponds to AWSError::ShouldRetry()
    bool should_retry;
  };
  /// Returns true if the S3 request resulting in the provided error should be retried.
  virtual bool ShouldRetry(const AWSErrorDetail& error, int64_t attempted_retries) = 0;
  /// Returns the time in milliseconds the S3 client should sleep for until retrying.
  virtual int64_t CalculateDelayBeforeNextRetry(const AWSErrorDetail& error,
                                                int64_t attempted_retries) = 0;
  /// Returns a stock AWS Default retry strategy.
  static std::shared_ptr<S3RetryStrategy> GetAwsDefaultRetryStrategy(
      int64_t max_attempts);
  /// Returns a stock AWS Standard retry strategy.
  static std::shared_ptr<S3RetryStrategy> GetAwsStandardRetryStrategy(
      int64_t max_attempts);
};

/// Options for the S3FileSystem implementation.
struct ARROW_EXPORT S3Options {
  /// \brief AWS region to connect to.
  ///
  /// If unset, the AWS SDK will choose a default value.  The exact algorithm
  /// depends on the SDK version.  Before 1.8, the default is hardcoded
  /// to "us-east-1".  Since 1.8, several heuristics are used to determine
  /// the region (environment variables, configuration profile, EC2 metadata
  /// server).
  std::string region;

  /// \brief Socket connection timeout, in seconds
  ///
  /// If negative, the AWS SDK default value is used (typically 1 second).
  double connect_timeout = -1;

  /// \brief Socket read timeout on Windows and macOS, in seconds
  ///
  /// If negative, the AWS SDK default value is used (typically 3 seconds).
  /// This option is ignored on non-Windows, non-macOS systems.
  double request_timeout = -1;

  /// If non-empty, override region with a connect string such as "localhost:9000"
  // XXX perhaps instead take a URL like "http://localhost:9000"?
  std::string endpoint_override;
  /// S3 connection transport, default "https"
  std::string scheme = "https";

  /// ARN of role to assume
  std::string role_arn;
  /// Optional identifier for an assumed role session.
  std::string session_name;
  /// Optional external idenitifer to pass to STS when assuming a role
  std::string external_id;
  /// Frequency (in seconds) to refresh temporary credentials from assumed role
  int load_frequency = 900;

  /// If connection is through a proxy, set options here
  S3ProxyOptions proxy_options;

  /// AWS credentials provider
  std::shared_ptr<Aws::Auth::AWSCredentialsProvider> credentials_provider;

  /// Type of credentials being used. Set along with credentials_provider.
  S3CredentialsKind credentials_kind = S3CredentialsKind::Default;

  /// Whether OutputStream writes will be issued in the background, without blocking.
  bool background_writes = true;

  /// Whether to allow creation of buckets
  ///
  /// When S3FileSystem creates new buckets, it does not pass any non-default settings.
  /// In AWS S3, the bucket and all objects will be not publicly visible, and there
  /// will be no bucket policies and no resource tags. To have more control over how
  /// buckets are created, use a different API to create them.
  bool allow_bucket_creation = false;

  /// Whether to allow deletion of buckets
  bool allow_bucket_deletion = false;

  /// \brief Default metadata for OpenOutputStream.
  ///
  /// This will be ignored if non-empty metadata is passed to OpenOutputStream.
  std::shared_ptr<const KeyValueMetadata> default_metadata;

  /// Optional retry strategy to determine which error types should be retried, and the
  /// delay between retries.
  std::shared_ptr<S3RetryStrategy> retry_strategy;

  S3Options();

  /// Configure with the default AWS credentials provider chain.
  void ConfigureDefaultCredentials();

  /// Configure with anonymous credentials.  This will only let you access public buckets.
  void ConfigureAnonymousCredentials();

  /// Configure with explicit access and secret key.
  void ConfigureAccessKey(const std::string& access_key, const std::string& secret_key,
                          const std::string& session_token = "");

  /// Configure with credentials from an assumed role.
  void ConfigureAssumeRoleCredentials(
      const std::string& role_arn, const std::string& session_name = "",
      const std::string& external_id = "", int load_frequency = 900,
      const std::shared_ptr<Aws::STS::STSClient>& stsClient = NULLPTR);

  /// Configure with credentials from role assumed using a web identitiy token
  void ConfigureAssumeRoleWithWebIdentityCredentials();

  std::string GetAccessKey() const;
  std::string GetSecretKey() const;
  std::string GetSessionToken() const;

  bool Equals(const S3Options& other) const;

  /// \brief Initialize with default credentials provider chain
  ///
  /// This is recommended if you use the standard AWS environment variables
  /// and/or configuration file.
  static S3Options Defaults();

  /// \brief Initialize with anonymous credentials.
  ///
  /// This will only let you access public buckets.
  static S3Options Anonymous();

  /// \brief Initialize with explicit access and secret key.
  ///
  /// Optionally, a session token may also be provided for temporary credentials
  /// (from STS).
  static S3Options FromAccessKey(const std::string& access_key,
                                 const std::string& secret_key,
                                 const std::string& session_token = "");

  /// \brief Initialize from an assumed role.
  static S3Options FromAssumeRole(
      const std::string& role_arn, const std::string& session_name = "",
      const std::string& external_id = "", int load_frequency = 900,
      const std::shared_ptr<Aws::STS::STSClient>& stsClient = NULLPTR);

  /// \brief Initialize from an assumed role with web-identity.
  /// Uses the AWS SDK which uses environment variables to
  /// generate temporary credentials.
  static S3Options FromAssumeRoleWithWebIdentity();

  static Result<S3Options> FromUri(const ::arrow::internal::Uri& uri,
                                   std::string* out_path = NULLPTR);
  static Result<S3Options> FromUri(const std::string& uri,
                                   std::string* out_path = NULLPTR);
};

/// S3-backed FileSystem implementation.
///
/// Some implementation notes:
/// - buckets are special and the operations available on them may be limited
///   or more expensive than desired.
class ARROW_EXPORT S3FileSystem : public FileSystem {
 public:
  ~S3FileSystem() override;

  std::string type_name() const override { return "s3"; }

  /// Return the original S3 options when constructing the filesystem
  S3Options options() const;
  /// Return the actual region this filesystem connects to
  std::string region() const;

  bool Equals(const FileSystem& other) const override;

  /// \cond FALSE
  using FileSystem::GetFileInfo;
  /// \endcond
  Result<FileInfo> GetFileInfo(const std::string& path) override;
  Result<std::vector<FileInfo>> GetFileInfo(const FileSelector& select) override;

  FileInfoGenerator GetFileInfoGenerator(const FileSelector& select) override;

  Status CreateDir(const std::string& path, bool recursive = true) override;

  Status DeleteDir(const std::string& path) override;
  Status DeleteDirContents(const std::string& path, bool missing_dir_ok = false) override;
  Future<> DeleteDirContentsAsync(const std::string& path,
                                  bool missing_dir_ok = false) override;
  Status DeleteRootDirContents() override;

  Status DeleteFile(const std::string& path) override;

  Status Move(const std::string& src, const std::string& dest) override;

  Status CopyFile(const std::string& src, const std::string& dest) override;

  /// Create a sequential input stream for reading from a S3 object.
  ///
  /// NOTE: Reads from the stream will be synchronous and unbuffered.
  /// You way want to wrap the stream in a BufferedInputStream or use
  /// a custom readahead strategy to avoid idle waits.
  Result<std::shared_ptr<io::InputStream>> OpenInputStream(
      const std::string& path) override;
  /// Create a sequential input stream for reading from a S3 object.
  ///
  /// This override avoids a HEAD request by assuming the FileInfo
  /// contains correct information.
  Result<std::shared_ptr<io::InputStream>> OpenInputStream(const FileInfo& info) override;

  /// Create a random access file for reading from a S3 object.
  ///
  /// See OpenInputStream for performance notes.
  Result<std::shared_ptr<io::RandomAccessFile>> OpenInputFile(
      const std::string& path) override;
  /// Create a random access file for reading from a S3 object.
  ///
  /// This override avoids a HEAD request by assuming the FileInfo
  /// contains correct information.
  Result<std::shared_ptr<io::RandomAccessFile>> OpenInputFile(
      const FileInfo& info) override;

  /// Create a sequential output stream for writing to a S3 object.
  ///
  /// NOTE: Writes to the stream will be buffered.  Depending on
  /// S3Options.background_writes, they can be synchronous or not.
  /// It is recommended to enable background_writes unless you prefer
  /// implementing your own background execution strategy.
  Result<std::shared_ptr<io::OutputStream>> OpenOutputStream(
      const std::string& path,
      const std::shared_ptr<const KeyValueMetadata>& metadata = {}) override;

  Result<std::shared_ptr<io::OutputStream>> OpenAppendStream(
      const std::string& path,
      const std::shared_ptr<const KeyValueMetadata>& metadata = {}) override;

  /// Create a S3FileSystem instance from the given options.
  static Result<std::shared_ptr<S3FileSystem>> Make(
      const S3Options& options, const io::IOContext& = io::default_io_context());

 protected:
  explicit S3FileSystem(const S3Options& options, const io::IOContext&);

  class Impl;
  std::shared_ptr<Impl> impl_;
};

enum class S3LogLevel : int8_t { Off, Fatal, Error, Warn, Info, Debug, Trace };

struct ARROW_EXPORT S3GlobalOptions {
  S3LogLevel log_level;
};

/// Initialize the S3 APIs.  It is required to call this function at least once
/// before using S3FileSystem.
ARROW_EXPORT
Status InitializeS3(const S3GlobalOptions& options);

/// Ensure the S3 APIs are initialized, but only if not already done.
/// If necessary, this will call InitializeS3() with some default options.
ARROW_EXPORT
Status EnsureS3Initialized();

/// Whether S3 was initialized, and not finalized.
ARROW_EXPORT
bool IsS3Initialized();

/// Shutdown the S3 APIs.
ARROW_EXPORT
Status FinalizeS3();

/// Ensure the S3 APIs are shutdown, but only if not already done.
ARROW_EXPORT
Status EnsureS3Finalized();

ARROW_EXPORT
Result<std::string> ResolveS3BucketRegion(const std::string& bucket);

}  // namespace fs
}  // namespace arrow

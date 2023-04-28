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
#include <optional>
#include <string>
#include <vector>

#include "arrow/filesystem/filesystem.h"
#include "arrow/util/uri.h"

namespace arrow {
namespace fs {

// Opaque wrapper for GCS's library credentials to avoid exposing in Arrow headers.
struct GcsCredentialsHolder;
class GcsFileSystem;

/// \brief Container for GCS Credentials and information necessary to recreate them.
class ARROW_EXPORT GcsCredentials {
 public:
  bool Equals(const GcsCredentials& other) const;
  bool anonymous() const { return anonymous_; }
  const std::string& access_token() const { return access_token_; }
  TimePoint expiration() const { return expiration_; }
  const std::string& target_service_account() const { return target_service_account_; }
  const std::string& json_credentials() const { return json_credentials_; }
  const std::shared_ptr<GcsCredentialsHolder>& holder() const { return holder_; }

 private:
  GcsCredentials() = default;
  bool anonymous_ = false;
  std::string access_token_;
  TimePoint expiration_;
  std::string target_service_account_;
  std::string json_credentials_;
  std::shared_ptr<GcsCredentialsHolder> holder_;
  friend class GcsFileSystem;
  friend struct GcsOptions;
};

/// Options for the GcsFileSystem implementation.
struct ARROW_EXPORT GcsOptions {
  /// \brief Equivalent to GcsOptions::Defaults().
  GcsOptions();
  GcsCredentials credentials;

  std::string endpoint_override;
  std::string scheme;
  /// \brief Location to use for creating buckets.
  std::string default_bucket_location;

  /// \brief If set used to control total time allowed for retrying underlying
  /// errors.
  ///
  /// The default policy is to retry for up to 15 minutes.
  std::optional<double> retry_limit_seconds;

  /// \brief Default metadata for OpenOutputStream.
  ///
  /// This will be ignored if non-empty metadata is passed to OpenOutputStream.
  std::shared_ptr<const KeyValueMetadata> default_metadata;

  bool Equals(const GcsOptions& other) const;

  /// \brief Initialize with Google Default Credentials
  ///
  /// Create options configured to use [Application Default Credentials][aip/4110]. The
  /// details of this mechanism are too involved to describe here, but suffice is to say
  /// that applications can override any defaults using an environment variable
  /// (`GOOGLE_APPLICATION_CREDENTIALS`), and that the defaults work with most Google
  /// Cloud Platform deployment environments (GCE, GKE, Cloud Run, etc.), and that have
  /// the same behavior as the `gcloud` CLI tool on your workstation.
  ///
  /// \see https://cloud.google.com/docs/authentication
  ///
  /// [aip/4110]: https://google.aip.dev/auth/4110
  static GcsOptions Defaults();

  /// \brief Initialize with anonymous credentials
  static GcsOptions Anonymous();

  /// \brief Initialize with access token
  ///
  /// These credentials are useful when using an out-of-band mechanism to fetch access
  /// tokens. Note that access tokens are time limited, you will need to manually refresh
  /// the tokens created by the out-of-band mechanism.
  static GcsOptions FromAccessToken(const std::string& access_token,
                                    TimePoint expiration);

  /// \brief Initialize with service account impersonation
  ///
  /// Service account impersonation allows one principal (a user or service account) to
  /// impersonate a service account. It requires that the calling principal has the
  /// necessary permissions *on* the service account.
  static GcsOptions FromImpersonatedServiceAccount(
      const GcsCredentials& base_credentials, const std::string& target_service_account);

  /// Creates service account credentials from a JSON object in string form.
  ///
  /// The @p json_object  is expected to be in the format described by [aip/4112]. Such an
  /// object contains the identity of a service account, as well as a private key that can
  /// be used to sign tokens, showing the caller was holding the private key.
  ///
  /// In GCP one can create several "keys" for each service account, and these keys are
  /// downloaded as a JSON "key file". The contents of such a file are in the format
  /// required by this function. Remember that key files and their contents should be
  /// treated as any other secret with security implications, think of them as passwords
  /// (because they are!), don't store them or output them where unauthorized persons may
  /// read them.
  ///
  /// Most applications should probably use default credentials, maybe pointing them to a
  /// file with these contents. Using this function may be useful when the json object is
  /// obtained from a Cloud Secret Manager or a similar service.
  ///
  /// [aip/4112]: https://google.aip.dev/auth/4112
  static GcsOptions FromServiceAccountCredentials(const std::string& json_object);

  /// Initialize from URIs such as "gs://bucket/object".
  static Result<GcsOptions> FromUri(const arrow::internal::Uri& uri,
                                    std::string* out_path);
  static Result<GcsOptions> FromUri(const std::string& uri, std::string* out_path);
};

/// \brief GCS-backed FileSystem implementation.
///
/// GCS (Google Cloud Storage - https://cloud.google.com/storage) is a scalable object
/// storage system for any amount of data. The main abstractions in GCS are buckets and
/// objects. A bucket is a namespace for objects, buckets can store any number of objects,
/// tens of millions and even billions is not uncommon.  Each object contains a single
/// blob of data, up to 5TiB in size.  Buckets are typically configured to keep a single
/// version of each object, but versioning can be enabled. Versioning is important because
/// objects are immutable, once created one cannot append data to the object or modify the
/// object data in any way.
///
/// GCS buckets are in a global namespace, if a Google Cloud customer creates a bucket
/// named `foo` no other customer can create a bucket with the same name. Note that a
/// principal (a user or service account) may only list the buckets they are entitled to,
/// and then only within a project. It is not possible to list "all" the buckets.
///
/// Within each bucket objects are in flat namespace. GCS does not have folders or
/// directories. However, following some conventions it is possible to emulate
/// directories. To this end, this class:
///
/// - All buckets are treated as directories at the "root"
/// - Creating a root directory results in a new bucket being created, this may be slower
///   than most GCS operations.
/// - The class creates marker objects for a directory, using a metadata attribute to
///   annotate the file.
/// - GCS can list all the objects with a given prefix, this is used to emulate listing
///   of directories.
/// - In object lists GCS can summarize all the objects with a common prefix as a single
///   entry, this is used to emulate non-recursive lists. Note that GCS list time is
///   proportional to the number of objects in the prefix. Listing recursively takes
///   almost the same time as non-recursive lists.
///
class ARROW_EXPORT GcsFileSystem : public FileSystem {
 public:
  ~GcsFileSystem() override = default;

  std::string type_name() const override;
  const GcsOptions& options() const;

  bool Equals(const FileSystem& other) const override;

  Result<FileInfo> GetFileInfo(const std::string& path) override;
  Result<FileInfoVector> GetFileInfo(const FileSelector& select) override;

  Status CreateDir(const std::string& path, bool recursive) override;

  Status DeleteDir(const std::string& path) override;

  Status DeleteDirContents(const std::string& path, bool missing_dir_ok = false) override;

  /// This is not implemented in GcsFileSystem, as it would be too dangerous.
  Status DeleteRootDirContents() override;

  Status DeleteFile(const std::string& path) override;

  Status Move(const std::string& src, const std::string& dest) override;

  Status CopyFile(const std::string& src, const std::string& dest) override;

  Result<std::shared_ptr<io::InputStream>> OpenInputStream(
      const std::string& path) override;
  Result<std::shared_ptr<io::InputStream>> OpenInputStream(const FileInfo& info) override;

  Result<std::shared_ptr<io::RandomAccessFile>> OpenInputFile(
      const std::string& path) override;
  Result<std::shared_ptr<io::RandomAccessFile>> OpenInputFile(
      const FileInfo& info) override;

  Result<std::shared_ptr<io::OutputStream>> OpenOutputStream(
      const std::string& path,
      const std::shared_ptr<const KeyValueMetadata>& metadata) override;

  ARROW_DEPRECATED(
      "Deprecated. "
      "OpenAppendStream is unsupported on the GCS FileSystem.")
  Result<std::shared_ptr<io::OutputStream>> OpenAppendStream(
      const std::string& path,
      const std::shared_ptr<const KeyValueMetadata>& metadata) override;

  /// Create a GcsFileSystem instance from the given options.
  // TODO(ARROW-16884): make this return Result for consistency
  static std::shared_ptr<GcsFileSystem> Make(
      const GcsOptions& options, const io::IOContext& = io::default_io_context());

 private:
  explicit GcsFileSystem(const GcsOptions& options, const io::IOContext& io_context);

  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace fs
}  // namespace arrow

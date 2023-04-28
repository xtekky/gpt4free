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
#include <unordered_map>

#include "arrow/util/mutex.h"

#include "parquet/exception.h"
#include "parquet/platform.h"

namespace parquet {
namespace encryption {

/// This class wraps the key access token of a KMS server. If your token changes over
/// time, you should keep the reference to the KeyAccessToken object and call Refresh()
/// method every time you have a new token.
class PARQUET_EXPORT KeyAccessToken {
 public:
  KeyAccessToken() = default;

  explicit KeyAccessToken(const std::string value) : value_(value) {}

  void Refresh(const std::string& new_value) {
    auto lock = mutex_.Lock();
    value_ = new_value;
  }

  const std::string& value() const {
    auto lock = mutex_.Lock();
    return value_;
  }

 private:
  std::string value_;
  mutable ::arrow::util::Mutex mutex_;
};

struct PARQUET_EXPORT KmsConnectionConfig {
  std::string kms_instance_id;
  std::string kms_instance_url;
  /// If the access token is changed in the future, you should keep a reference to
  /// this object and call Refresh() on it whenever there is a new access token.
  std::shared_ptr<KeyAccessToken> refreshable_key_access_token;
  std::unordered_map<std::string, std::string> custom_kms_conf;

  KmsConnectionConfig();

  const std::string& key_access_token() const {
    if (refreshable_key_access_token == NULL ||
        refreshable_key_access_token->value().empty()) {
      throw ParquetException("key access token is not set!");
    }
    return refreshable_key_access_token->value();
  }

  void SetDefaultIfEmpty();
};

class PARQUET_EXPORT KmsClient {
 public:
  static constexpr const char kKmsInstanceIdDefault[] = "DEFAULT";
  static constexpr const char kKmsInstanceUrlDefault[] = "DEFAULT";
  static constexpr const char kKeyAccessTokenDefault[] = "DEFAULT";

  /// Wraps a key - encrypts it with the master key, encodes the result
  /// and potentially adds a KMS-specific metadata.
  virtual std::string WrapKey(const std::string& key_bytes,
                              const std::string& master_key_identifier) = 0;

  /// Decrypts (unwraps) a key with the master key.
  virtual std::string UnwrapKey(const std::string& wrapped_key,
                                const std::string& master_key_identifier) = 0;
  virtual ~KmsClient() {}
};

}  // namespace encryption
}  // namespace parquet

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

#include "parquet/encryption/key_encryption_key.h"
#include "parquet/encryption/kms_client.h"
#include "parquet/encryption/kms_client_factory.h"
#include "parquet/encryption/two_level_cache_with_expiration.h"
#include "parquet/platform.h"

namespace parquet {
namespace encryption {

// KeyToolkit is a utility that keeps various tools for key management (such as key
// rotation, kms client instantiation, cache control, etc), plus a number of auxiliary
// classes for internal use.
class PARQUET_EXPORT KeyToolkit {
 public:
  /// KMS client two level cache: token -> KMSInstanceId -> KmsClient
  TwoLevelCacheWithExpiration<std::shared_ptr<KmsClient>>& kms_client_cache_per_token() {
    return kms_client_cache_;
  }
  /// Key encryption key two level cache for wrapping: token -> MasterEncryptionKeyId ->
  /// KeyEncryptionKey
  TwoLevelCacheWithExpiration<KeyEncryptionKey>& kek_write_cache_per_token() {
    return key_encryption_key_write_cache_;
  }

  /// Key encryption key two level cache for unwrapping: token -> KeyEncryptionKeyId ->
  /// KeyEncryptionKeyBytes
  TwoLevelCacheWithExpiration<std::string>& kek_read_cache_per_token() {
    return key_encryption_key_read_cache_;
  }

  std::shared_ptr<KmsClient> GetKmsClient(
      const KmsConnectionConfig& kms_connection_config, double cache_entry_lifetime_ms);

  /// Flush any caches that are tied to the (compromised) access_token
  void RemoveCacheEntriesForToken(const std::string& access_token);

  void RemoveCacheEntriesForAllTokens();

  void RegisterKmsClientFactory(std::shared_ptr<KmsClientFactory> kms_client_factory) {
    if (kms_client_factory_ != NULL) {
      throw ParquetException("KMS client factory has already been registered.");
    }
    kms_client_factory_ = kms_client_factory;
  }

 private:
  TwoLevelCacheWithExpiration<std::shared_ptr<KmsClient>> kms_client_cache_;
  TwoLevelCacheWithExpiration<KeyEncryptionKey> key_encryption_key_write_cache_;
  TwoLevelCacheWithExpiration<std::string> key_encryption_key_read_cache_;
  std::shared_ptr<KmsClientFactory> kms_client_factory_;
};

}  // namespace encryption
}  // namespace parquet

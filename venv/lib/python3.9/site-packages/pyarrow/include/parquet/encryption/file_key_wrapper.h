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

#include "arrow/util/concurrent_map.h"

#include "parquet/encryption/file_key_material_store.h"
#include "parquet/encryption/key_encryption_key.h"
#include "parquet/encryption/key_toolkit.h"
#include "parquet/encryption/kms_client.h"
#include "parquet/platform.h"

namespace parquet {
namespace encryption {

// This class will generate "key metadata" from "data encryption key" and "master key",
// following these steps:
// 1. Wrap "data encryption key". There are 2 modes:
//   1.1. single wrapping: encrypt "data encryption key" directly with "master encryption
//        key"
//   1.2. double wrapping: 2 steps:
//     1.2.1. "key encryption key" is randomized (see KeyEncryptionKey class)
//     1.2.2. "data encryption key" is encrypted with the above "key encryption key"
// 2. Create "key material" (see structure in KeyMaterial class)
// 3. Create "key metadata" with "key material" inside or a reference to outside "key
//    material" (see structure in KeyMetadata class).
//    We don't support the case "key material" stores outside "key metadata" yet.
class PARQUET_EXPORT FileKeyWrapper {
 public:
  static constexpr int kKeyEncryptionKeyLength = 16;
  static constexpr int kKeyEncryptionKeyIdLength = 16;

  /// key_toolkit and kms_connection_config is to get KmsClient from the cache or create
  /// KmsClient if it's not in the cache yet. cache_entry_lifetime_seconds is life time of
  /// KmsClient in the cache. key_material_store is to store "key material" outside
  /// parquet file, NULL if "key material" is stored inside parquet file.
  FileKeyWrapper(KeyToolkit* key_toolkit,
                 const KmsConnectionConfig& kms_connection_config,
                 std::shared_ptr<FileKeyMaterialStore> key_material_store,
                 double cache_entry_lifetime_seconds, bool double_wrapping);

  /// Creates key_metadata field for a given data key, via wrapping the key with the
  /// master key
  std::string GetEncryptionKeyMetadata(const std::string& data_key,
                                       const std::string& master_key_id,
                                       bool is_footer_key);

 private:
  KeyEncryptionKey CreateKeyEncryptionKey(const std::string& master_key_id);

  /// A map of Master Encryption Key ID -> KeyEncryptionKey, for the current token
  std::shared_ptr<::arrow::util::ConcurrentMap<std::string, KeyEncryptionKey>>
      kek_per_master_key_id_;

  std::shared_ptr<KmsClient> kms_client_;
  KmsConnectionConfig kms_connection_config_;
  std::shared_ptr<FileKeyMaterialStore> key_material_store_;
  const double cache_entry_lifetime_seconds_;
  const bool double_wrapping_;
};

}  // namespace encryption
}  // namespace parquet

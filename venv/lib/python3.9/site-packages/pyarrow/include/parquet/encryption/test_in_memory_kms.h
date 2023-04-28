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

#include <unordered_map>

#include "arrow/util/base64.h"

#include "parquet/encryption/kms_client_factory.h"
#include "parquet/encryption/local_wrap_kms_client.h"
#include "parquet/platform.h"

namespace parquet {
namespace encryption {

// This is a mock class, built for testing only. Don't use it as an example of
// LocalWrapKmsClient implementation.
class TestOnlyLocalWrapInMemoryKms : public LocalWrapKmsClient {
 public:
  explicit TestOnlyLocalWrapInMemoryKms(const KmsConnectionConfig& kms_connection_config);

  static void InitializeMasterKeys(
      const std::unordered_map<std::string, std::string>& master_keys_map);

 protected:
  std::string GetMasterKeyFromServer(const std::string& master_key_identifier) override;

 private:
  static std::unordered_map<std::string, std::string> master_key_map_;
};

// This is a mock class, built for testing only. Don't use it as an example of KmsClient
// implementation.
class TestOnlyInServerWrapKms : public KmsClient {
 public:
  static void InitializeMasterKeys(
      const std::unordered_map<std::string, std::string>& master_keys_map);

  std::string WrapKey(const std::string& key_bytes,
                      const std::string& master_key_identifier) override;

  std::string UnwrapKey(const std::string& wrapped_key,
                        const std::string& master_key_identifier) override;

 private:
  std::string GetMasterKeyFromServer(const std::string& master_key_identifier);

  static std::unordered_map<std::string, std::string> master_key_map_;
};

// This is a mock class, built for testing only. Don't use it as an example of
// KmsClientFactory implementation.
class TestOnlyInMemoryKmsClientFactory : public KmsClientFactory {
 public:
  TestOnlyInMemoryKmsClientFactory(
      bool wrap_locally,
      const std::unordered_map<std::string, std::string>& master_keys_map)
      : KmsClientFactory(wrap_locally) {
    TestOnlyLocalWrapInMemoryKms::InitializeMasterKeys(master_keys_map);
    TestOnlyInServerWrapKms::InitializeMasterKeys(master_keys_map);
  }

  std::shared_ptr<KmsClient> CreateKmsClient(
      const KmsConnectionConfig& kms_connection_config) {
    if (wrap_locally_) {
      return std::make_shared<TestOnlyLocalWrapInMemoryKms>(kms_connection_config);
    } else {
      return std::make_shared<TestOnlyInServerWrapKms>();
    }
  }
};

}  // namespace encryption
}  // namespace parquet

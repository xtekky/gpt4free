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

#include "parquet/encryption/kms_client.h"
#include "parquet/platform.h"

namespace parquet {
namespace encryption {

class PARQUET_EXPORT KmsClientFactory {
 public:
  explicit KmsClientFactory(bool wrap_locally = false) : wrap_locally_(wrap_locally) {}

  virtual ~KmsClientFactory() = default;

  virtual std::shared_ptr<KmsClient> CreateKmsClient(
      const KmsConnectionConfig& kms_connection_config) = 0;

 protected:
  bool wrap_locally_;
};

}  // namespace encryption
}  // namespace parquet

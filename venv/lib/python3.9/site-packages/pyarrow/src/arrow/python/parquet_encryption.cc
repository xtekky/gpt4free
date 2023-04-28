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

#include "arrow/python/parquet_encryption.h"
#include "parquet/exception.h"

namespace arrow {
namespace py {
namespace parquet {
namespace encryption {

PyKmsClient::PyKmsClient(PyObject* handler, PyKmsClientVtable vtable)
    : handler_(handler), vtable_(std::move(vtable)) {
  Py_INCREF(handler);
}

PyKmsClient::~PyKmsClient() {}

std::string PyKmsClient::WrapKey(const std::string& key_bytes,
                                 const std::string& master_key_identifier) {
  std::string wrapped;
  auto st = SafeCallIntoPython([&]() -> Status {
    vtable_.wrap_key(handler_.obj(), key_bytes, master_key_identifier, &wrapped);
    return CheckPyError();
  });
  if (!st.ok()) {
    throw ::parquet::ParquetStatusException(st);
  }
  return wrapped;
}

std::string PyKmsClient::UnwrapKey(const std::string& wrapped_key,
                                   const std::string& master_key_identifier) {
  std::string unwrapped;
  auto st = SafeCallIntoPython([&]() -> Status {
    vtable_.unwrap_key(handler_.obj(), wrapped_key, master_key_identifier, &unwrapped);
    return CheckPyError();
  });
  if (!st.ok()) {
    throw ::parquet::ParquetStatusException(st);
  }
  return unwrapped;
}

PyKmsClientFactory::PyKmsClientFactory(PyObject* handler, PyKmsClientFactoryVtable vtable)
    : handler_(handler), vtable_(std::move(vtable)) {
  Py_INCREF(handler);
}

PyKmsClientFactory::~PyKmsClientFactory() {}

std::shared_ptr<::parquet::encryption::KmsClient> PyKmsClientFactory::CreateKmsClient(
    const ::parquet::encryption::KmsConnectionConfig& kms_connection_config) {
  std::shared_ptr<::parquet::encryption::KmsClient> kms_client;
  auto st = SafeCallIntoPython([&]() -> Status {
    vtable_.create_kms_client(handler_.obj(), kms_connection_config, &kms_client);
    return CheckPyError();
  });
  if (!st.ok()) {
    throw ::parquet::ParquetStatusException(st);
  }
  return kms_client;
}

arrow::Result<std::shared_ptr<::parquet::FileEncryptionProperties>>
PyCryptoFactory::SafeGetFileEncryptionProperties(
    const ::parquet::encryption::KmsConnectionConfig& kms_connection_config,
    const ::parquet::encryption::EncryptionConfiguration& encryption_config) {
  PARQUET_CATCH_AND_RETURN(
      this->GetFileEncryptionProperties(kms_connection_config, encryption_config));
}

arrow::Result<std::shared_ptr<::parquet::FileDecryptionProperties>>
PyCryptoFactory::SafeGetFileDecryptionProperties(
    const ::parquet::encryption::KmsConnectionConfig& kms_connection_config,
    const ::parquet::encryption::DecryptionConfiguration& decryption_config) {
  PARQUET_CATCH_AND_RETURN(
      this->GetFileDecryptionProperties(kms_connection_config, decryption_config));
}

}  // namespace encryption
}  // namespace parquet
}  // namespace py
}  // namespace arrow

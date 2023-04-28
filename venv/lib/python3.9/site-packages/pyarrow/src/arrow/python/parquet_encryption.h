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

#include <string>

#include "arrow/python/common.h"
#include "arrow/python/visibility.h"
#include "arrow/util/macros.h"
#include "parquet/encryption/crypto_factory.h"
#include "parquet/encryption/kms_client.h"
#include "parquet/encryption/kms_client_factory.h"

namespace arrow {
namespace py {
namespace parquet {
namespace encryption {

/// \brief A table of function pointers for calling from C++ into
/// Python.
class ARROW_PYTHON_EXPORT PyKmsClientVtable {
 public:
  std::function<void(PyObject*, const std::string& key_bytes,
                     const std::string& master_key_identifier, std::string* out)>
      wrap_key;
  std::function<void(PyObject*, const std::string& wrapped_key,
                     const std::string& master_key_identifier, std::string* out)>
      unwrap_key;
};

/// \brief A helper for KmsClient implementation in Python.
class ARROW_PYTHON_EXPORT PyKmsClient : public ::parquet::encryption::KmsClient {
 public:
  PyKmsClient(PyObject* handler, PyKmsClientVtable vtable);
  ~PyKmsClient() override;

  std::string WrapKey(const std::string& key_bytes,
                      const std::string& master_key_identifier) override;

  std::string UnwrapKey(const std::string& wrapped_key,
                        const std::string& master_key_identifier) override;

 private:
  OwnedRefNoGIL handler_;
  PyKmsClientVtable vtable_;
};

/// \brief A table of function pointers for calling from C++ into
/// Python.
class ARROW_PYTHON_EXPORT PyKmsClientFactoryVtable {
 public:
  std::function<void(
      PyObject*, const ::parquet::encryption::KmsConnectionConfig& kms_connection_config,
      std::shared_ptr<::parquet::encryption::KmsClient>* out)>
      create_kms_client;
};

/// \brief A helper for KmsClientFactory implementation in Python.
class ARROW_PYTHON_EXPORT PyKmsClientFactory
    : public ::parquet::encryption::KmsClientFactory {
 public:
  PyKmsClientFactory(PyObject* handler, PyKmsClientFactoryVtable vtable);
  ~PyKmsClientFactory() override;

  std::shared_ptr<::parquet::encryption::KmsClient> CreateKmsClient(
      const ::parquet::encryption::KmsConnectionConfig& kms_connection_config) override;

 private:
  OwnedRefNoGIL handler_;
  PyKmsClientFactoryVtable vtable_;
};

/// \brief A CryptoFactory that returns Results instead of throwing exceptions.
class ARROW_PYTHON_EXPORT PyCryptoFactory : public ::parquet::encryption::CryptoFactory {
 public:
  arrow::Result<std::shared_ptr<::parquet::FileEncryptionProperties>>
  SafeGetFileEncryptionProperties(
      const ::parquet::encryption::KmsConnectionConfig& kms_connection_config,
      const ::parquet::encryption::EncryptionConfiguration& encryption_config);

  /// The returned FileDecryptionProperties object will use the cache inside this
  /// CryptoFactory object, so please keep this
  /// CryptoFactory object alive along with the returned
  /// FileDecryptionProperties object.
  arrow::Result<std::shared_ptr<::parquet::FileDecryptionProperties>>
  SafeGetFileDecryptionProperties(
      const ::parquet::encryption::KmsConnectionConfig& kms_connection_config,
      const ::parquet::encryption::DecryptionConfiguration& decryption_config);
};

}  // namespace encryption
}  // namespace parquet
}  // namespace py
}  // namespace arrow

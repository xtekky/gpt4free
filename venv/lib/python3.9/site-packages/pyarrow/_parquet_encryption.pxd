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
# cython: language_level = 3

from pyarrow.includes.common cimport *
from pyarrow._parquet cimport (ParquetCipher,
                               CFileEncryptionProperties,
                               CFileDecryptionProperties,
                               FileEncryptionProperties,
                               FileDecryptionProperties,
                               ParquetCipher_AES_GCM_V1,
                               ParquetCipher_AES_GCM_CTR_V1)


cdef extern from "parquet/encryption/kms_client.h" \
        namespace "parquet::encryption" nogil:
    cdef cppclass CKmsClient" parquet::encryption::KmsClient":
        c_string WrapKey(const c_string& key_bytes,
                         const c_string& master_key_identifier) except +
        c_string UnwrapKey(const c_string& wrapped_key,
                           const c_string& master_key_identifier) except +

    cdef cppclass CKeyAccessToken" parquet::encryption::KeyAccessToken":
        CKeyAccessToken(const c_string value)
        void Refresh(const c_string& new_value)
        const c_string& value() const

    cdef cppclass CKmsConnectionConfig \
            " parquet::encryption::KmsConnectionConfig":
        CKmsConnectionConfig()
        c_string kms_instance_id
        c_string kms_instance_url
        shared_ptr[CKeyAccessToken] refreshable_key_access_token
        unordered_map[c_string, c_string] custom_kms_conf

# Callbacks for implementing Python kms clients
# Use typedef to emulate syntax for std::function<void(..)>
ctypedef void CallbackWrapKey(
    object, const c_string&, const c_string&, c_string*)
ctypedef void CallbackUnwrapKey(
    object, const c_string&, const c_string&, c_string*)

cdef extern from "parquet/encryption/kms_client_factory.h" \
        namespace "parquet::encryption" nogil:
    cdef cppclass CKmsClientFactory" parquet::encryption::KmsClientFactory":
        shared_ptr[CKmsClient] CreateKmsClient(
            const CKmsConnectionConfig& kms_connection_config) except +

# Callbacks for implementing Python kms client factories
# Use typedef to emulate syntax for std::function<void(..)>
ctypedef void CallbackCreateKmsClient(
    object,
    const CKmsConnectionConfig&, shared_ptr[CKmsClient]*)

cdef extern from "parquet/encryption/crypto_factory.h" \
        namespace "parquet::encryption" nogil:
    cdef cppclass CEncryptionConfiguration\
            " parquet::encryption::EncryptionConfiguration":
        CEncryptionConfiguration(const c_string& footer_key) except +
        c_string footer_key
        c_string column_keys
        ParquetCipher encryption_algorithm
        c_bool plaintext_footer
        c_bool double_wrapping
        double cache_lifetime_seconds
        c_bool internal_key_material
        int32_t data_key_length_bits

    cdef cppclass CDecryptionConfiguration\
            " parquet::encryption::DecryptionConfiguration":
        CDecryptionConfiguration() except +
        double cache_lifetime_seconds

    cdef cppclass CCryptoFactory" parquet::encryption::CryptoFactory":
        void RegisterKmsClientFactory(
            shared_ptr[CKmsClientFactory] kms_client_factory) except +
        shared_ptr[CFileEncryptionProperties] GetFileEncryptionProperties(
            const CKmsConnectionConfig& kms_connection_config,
            const CEncryptionConfiguration& encryption_config) except +*
        shared_ptr[CFileDecryptionProperties] GetFileDecryptionProperties(
            const CKmsConnectionConfig& kms_connection_config,
            const CDecryptionConfiguration& decryption_config) except +*
        void RemoveCacheEntriesForToken(const c_string& access_token) except +
        void RemoveCacheEntriesForAllTokens() except +

cdef extern from "arrow/python/parquet_encryption.h" \
        namespace "arrow::py::parquet::encryption" nogil:
    cdef cppclass CPyKmsClientVtable \
            " arrow::py::parquet::encryption::PyKmsClientVtable":
        CPyKmsClientVtable()
        function[CallbackWrapKey] wrap_key
        function[CallbackUnwrapKey] unwrap_key

    cdef cppclass CPyKmsClient\
            " arrow::py::parquet::encryption::PyKmsClient"(CKmsClient):
        CPyKmsClient(object handler, CPyKmsClientVtable vtable)

    cdef cppclass CPyKmsClientFactoryVtable\
            " arrow::py::parquet::encryption::PyKmsClientFactoryVtable":
        CPyKmsClientFactoryVtable()
        function[CallbackCreateKmsClient] create_kms_client

    cdef cppclass CPyKmsClientFactory\
            " arrow::py::parquet::encryption::PyKmsClientFactory"(
                CKmsClientFactory):
        CPyKmsClientFactory(object handler, CPyKmsClientFactoryVtable vtable)

    cdef cppclass CPyCryptoFactory\
            " arrow::py::parquet::encryption::PyCryptoFactory"(CCryptoFactory):
        CResult[shared_ptr[CFileEncryptionProperties]] \
            SafeGetFileEncryptionProperties(
            const CKmsConnectionConfig& kms_connection_config,
            const CEncryptionConfiguration& encryption_config)
        CResult[shared_ptr[CFileDecryptionProperties]] \
            SafeGetFileDecryptionProperties(
            const CKmsConnectionConfig& kms_connection_config,
            const CDecryptionConfiguration& decryption_config)

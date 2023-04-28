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

# cython: profile=False
# distutils: language = c++

from datetime import timedelta
import io
import warnings

from libcpp cimport nullptr

from cython.operator cimport dereference as deref
from pyarrow.includes.common cimport *
from pyarrow.includes.libarrow cimport *
from pyarrow.lib cimport _Weakrefable

from pyarrow.lib import (ArrowException,
                         tobytes, frombytes)

cimport cpython as cp


cdef ParquetCipher cipher_from_name(name):
    name = name.upper()
    if name == 'AES_GCM_V1':
        return ParquetCipher_AES_GCM_V1
    elif name == 'AES_GCM_CTR_V1':
        return ParquetCipher_AES_GCM_CTR_V1
    else:
        raise ValueError(f'Invalid cipher name: {name!r}')


cdef cipher_to_name(ParquetCipher cipher):
    if ParquetCipher_AES_GCM_V1 == cipher:
        return 'AES_GCM_V1'
    elif ParquetCipher_AES_GCM_CTR_V1 == cipher:
        return 'AES_GCM_CTR_V1'
    else:
        raise ValueError('Invalid cipher value: {0}'.format(cipher))

cdef class EncryptionConfiguration(_Weakrefable):
    """Configuration of the encryption, such as which columns to encrypt"""
    cdef:
        shared_ptr[CEncryptionConfiguration] configuration

    # Avoid mistakingly creating attributes
    __slots__ = ()

    def __init__(self, footer_key, *, column_keys=None,
                 encryption_algorithm=None,
                 plaintext_footer=None, double_wrapping=None,
                 cache_lifetime=None, internal_key_material=None,
                 data_key_length_bits=None):
        self.configuration.reset(
            new CEncryptionConfiguration(tobytes(footer_key)))
        if column_keys is not None:
            self.column_keys = column_keys
        if encryption_algorithm is not None:
            self.encryption_algorithm = encryption_algorithm
        if plaintext_footer is not None:
            self.plaintext_footer = plaintext_footer
        if double_wrapping is not None:
            self.double_wrapping = double_wrapping
        if cache_lifetime is not None:
            self.cache_lifetime = cache_lifetime
        if internal_key_material is not None:
            self.internal_key_material = internal_key_material
        if data_key_length_bits is not None:
            self.data_key_length_bits = data_key_length_bits

    @property
    def footer_key(self):
        """ID of the master key for footer encryption/signing"""
        return frombytes(self.configuration.get().footer_key)

    @property
    def column_keys(self):
        """
        List of columns to encrypt, with master key IDs.
        """
        column_keys_str = frombytes(self.configuration.get().column_keys)
        # Convert from "masterKeyID:colName,colName;masterKeyID:colName..."
        # (see HIVE-21848) to dictionary of master key ID to column name lists
        column_keys_to_key_list_str = dict(subString.replace(" ", "").split(
            ":") for subString in column_keys_str.split(";"))
        column_keys_dict = {k: v.split(
            ",") for k, v in column_keys_to_key_list_str.items()}
        return column_keys_dict

    @column_keys.setter
    def column_keys(self, dict value):
        if value is not None:
            # convert a dictionary such as
            # '{"key1": ["col1 ", "col2"], "key2": ["col3 ", "col4"]}''
            # to the string defined by the spec
            # 'key1: col1 , col2; key2: col3 , col4'
            column_keys = "; ".join(
                ["{}: {}".format(k, ", ".join(v)) for k, v in value.items()])
            self.configuration.get().column_keys = tobytes(column_keys)

    @property
    def encryption_algorithm(self):
        """Parquet encryption algorithm.
        Can be "AES_GCM_V1" (default), or "AES_GCM_CTR_V1"."""
        return cipher_to_name(self.configuration.get().encryption_algorithm)

    @encryption_algorithm.setter
    def encryption_algorithm(self, value):
        cipher = cipher_from_name(value)
        self.configuration.get().encryption_algorithm = cipher

    @property
    def plaintext_footer(self):
        """Write files with plaintext footer."""
        return self.configuration.get().plaintext_footer

    @plaintext_footer.setter
    def plaintext_footer(self, value):
        self.configuration.get().plaintext_footer = value

    @property
    def double_wrapping(self):
        """Use double wrapping - where data encryption keys (DEKs) are
        encrypted with key encryption keys (KEKs), which in turn are
        encrypted with master keys.
        If set to false, use single wrapping - where DEKs are
        encrypted directly with master keys."""
        return self.configuration.get().double_wrapping

    @double_wrapping.setter
    def double_wrapping(self, value):
        self.configuration.get().double_wrapping = value

    @property
    def cache_lifetime(self):
        """Lifetime of cached entities (key encryption keys,
        local wrapping keys, KMS client objects)."""
        return timedelta(
            seconds=self.configuration.get().cache_lifetime_seconds)

    @cache_lifetime.setter
    def cache_lifetime(self, value):
        if not isinstance(value, timedelta):
            raise TypeError("cache_lifetime should be a timedelta")
        self.configuration.get().cache_lifetime_seconds = value.total_seconds()

    @property
    def internal_key_material(self):
        """Store key material inside Parquet file footers; this mode doesn’t
        produce additional files. If set to false, key material is stored in
        separate files in the same folder, which enables key rotation for
        immutable Parquet files."""
        return self.configuration.get().internal_key_material

    @internal_key_material.setter
    def internal_key_material(self, value):
        self.configuration.get().internal_key_material = value

    @property
    def data_key_length_bits(self):
        """Length of data encryption keys (DEKs), randomly generated by parquet key
        management tools. Can be 128, 192 or 256 bits."""
        return self.configuration.get().data_key_length_bits

    @data_key_length_bits.setter
    def data_key_length_bits(self, value):
        self.configuration.get().data_key_length_bits = value

    cdef inline shared_ptr[CEncryptionConfiguration] unwrap(self) nogil:
        return self.configuration


cdef class DecryptionConfiguration(_Weakrefable):
    """Configuration of the decryption, such as cache timeout."""
    cdef:
        shared_ptr[CDecryptionConfiguration] configuration

    # Avoid mistakingly creating attributes
    __slots__ = ()

    def __init__(self, *, cache_lifetime=None):
        self.configuration.reset(new CDecryptionConfiguration())

    @property
    def cache_lifetime(self):
        """Lifetime of cached entities (key encryption keys,
        local wrapping keys, KMS client objects)."""
        return timedelta(
            seconds=self.configuration.get().cache_lifetime_seconds)

    @cache_lifetime.setter
    def cache_lifetime(self, value):
        self.configuration.get().cache_lifetime_seconds = value.total_seconds()

    cdef inline shared_ptr[CDecryptionConfiguration] unwrap(self) nogil:
        return self.configuration


cdef class KmsConnectionConfig(_Weakrefable):
    """Configuration of the connection to the Key Management Service (KMS)"""
    cdef:
        shared_ptr[CKmsConnectionConfig] configuration

    # Avoid mistakingly creating attributes
    __slots__ = ()

    def __init__(self, *, kms_instance_id=None, kms_instance_url=None,
                 key_access_token=None, custom_kms_conf=None):
        self.configuration.reset(new CKmsConnectionConfig())
        if kms_instance_id is not None:
            self.kms_instance_id = kms_instance_id
        if kms_instance_url is not None:
            self.kms_instance_url = kms_instance_url
        if key_access_token is None:
            self.key_access_token = b'DEFAULT'
        else:
            self.key_access_token = key_access_token
        if custom_kms_conf is not None:
            self.custom_kms_conf = custom_kms_conf

    @property
    def kms_instance_id(self):
        """ID of the KMS instance that will be used for encryption
        (if multiple KMS instances are available)."""
        return frombytes(self.configuration.get().kms_instance_id)

    @kms_instance_id.setter
    def kms_instance_id(self, value):
        self.configuration.get().kms_instance_id = tobytes(value)

    @property
    def kms_instance_url(self):
        """URL of the KMS instance."""
        return frombytes(self.configuration.get().kms_instance_url)

    @kms_instance_url.setter
    def kms_instance_url(self, value):
        self.configuration.get().kms_instance_url = tobytes(value)

    @property
    def key_access_token(self):
        """Authorization token that will be passed to KMS."""
        return frombytes(self.configuration.get()
                         .refreshable_key_access_token.get().value())

    @key_access_token.setter
    def key_access_token(self, value):
        self.refresh_key_access_token(value)

    @property
    def custom_kms_conf(self):
        """A dictionary with KMS-type-specific configuration"""
        custom_kms_conf = {
            frombytes(k): frombytes(v)
            for k, v in self.configuration.get().custom_kms_conf
        }
        return custom_kms_conf

    @custom_kms_conf.setter
    def custom_kms_conf(self, dict value):
        if value is not None:
            for k, v in value.items():
                if isinstance(k, str) and isinstance(v, str):
                    self.configuration.get().custom_kms_conf[tobytes(k)] = \
                        tobytes(v)
                else:
                    raise TypeError("Expected custom_kms_conf to be " +
                                    "a dictionary of strings")

    def refresh_key_access_token(self, value):
        cdef:
            shared_ptr[CKeyAccessToken] c_key_access_token = \
                self.configuration.get().refreshable_key_access_token

        c_key_access_token.get().Refresh(tobytes(value))

    cdef inline shared_ptr[CKmsConnectionConfig] unwrap(self) nogil:
        return self.configuration

    @staticmethod
    cdef wrap(const CKmsConnectionConfig& config):
        result = KmsConnectionConfig()
        result.configuration = make_shared[CKmsConnectionConfig](move(config))
        return result


# Callback definitions for CPyKmsClientVtable
cdef void _cb_wrap_key(
        handler, const c_string& key_bytes,
        const c_string& master_key_identifier, c_string* out) except *:
    mkid_str = frombytes(master_key_identifier)
    wrapped_key = handler.wrap_key(key_bytes, mkid_str)
    out[0] = tobytes(wrapped_key)


cdef void _cb_unwrap_key(
        handler, const c_string& wrapped_key,
        const c_string& master_key_identifier, c_string* out) except *:
    mkid_str = frombytes(master_key_identifier)
    wk_str = frombytes(wrapped_key)
    key = handler.unwrap_key(wk_str, mkid_str)
    out[0] = tobytes(key)


cdef class KmsClient(_Weakrefable):
    """The abstract base class for KmsClient implementations."""
    cdef:
        shared_ptr[CKmsClient] client

    def __init__(self):
        self.init()

    cdef init(self):
        cdef:
            CPyKmsClientVtable vtable = CPyKmsClientVtable()

        vtable.wrap_key = _cb_wrap_key
        vtable.unwrap_key = _cb_unwrap_key

        self.client.reset(new CPyKmsClient(self, vtable))

    def wrap_key(self, key_bytes, master_key_identifier):
        """Wrap a key - encrypt it with the master key."""
        raise NotImplementedError()

    def unwrap_key(self, wrapped_key, master_key_identifier):
        """Unwrap a key - decrypt it with the master key."""
        raise NotImplementedError()

    cdef inline shared_ptr[CKmsClient] unwrap(self) nogil:
        return self.client


# Callback definition for CPyKmsClientFactoryVtable
cdef void _cb_create_kms_client(
        handler,
        const CKmsConnectionConfig& kms_connection_config,
        shared_ptr[CKmsClient]* out) except *:
    connection_config = KmsConnectionConfig.wrap(kms_connection_config)

    result = handler(connection_config)
    if not isinstance(result, KmsClient):
        raise TypeError(
            "callable must return KmsClient instances, but got {}".format(
                type(result)))

    out[0] = (<KmsClient> result).unwrap()


cdef class CryptoFactory(_Weakrefable):
    """ A factory that produces the low-level FileEncryptionProperties and
    FileDecryptionProperties objects, from the high-level parameters."""
    cdef:
        unique_ptr[CPyCryptoFactory] factory

    # Avoid mistakingly creating attributes
    __slots__ = ()

    def __init__(self, kms_client_factory):
        """Create CryptoFactory.

        Parameters
        ----------
        kms_client_factory : a callable that accepts KmsConnectionConfig
            and returns a KmsClient
        """
        self.factory.reset(new CPyCryptoFactory())

        if callable(kms_client_factory):
            self.init(kms_client_factory)
        else:
            raise TypeError("Parameter kms_client_factory must be a callable")

    cdef init(self, callable_client_factory):
        cdef:
            CPyKmsClientFactoryVtable vtable
            shared_ptr[CPyKmsClientFactory] kms_client_factory

        vtable.create_kms_client = _cb_create_kms_client
        kms_client_factory.reset(
            new CPyKmsClientFactory(callable_client_factory, vtable))
        # A KmsClientFactory object must be registered
        # via this method before calling any of
        # file_encryption_properties()/file_decryption_properties() methods.
        self.factory.get().RegisterKmsClientFactory(
            static_pointer_cast[CKmsClientFactory, CPyKmsClientFactory](
                kms_client_factory))

    def file_encryption_properties(self,
                                   KmsConnectionConfig kms_connection_config,
                                   EncryptionConfiguration encryption_config):
        """Create file encryption properties.

        Parameters
        ----------
        kms_connection_config : KmsConnectionConfig
            Configuration of connection to KMS

        encryption_config : EncryptionConfiguration
            Configuration of the encryption, such as which columns to encrypt

        Returns
        -------
        file_encryption_properties : FileEncryptionProperties
            File encryption properties.
        """
        cdef:
            CResult[shared_ptr[CFileEncryptionProperties]] \
                file_encryption_properties_result
        with nogil:
            file_encryption_properties_result = \
                self.factory.get().SafeGetFileEncryptionProperties(
                    deref(kms_connection_config.unwrap().get()),
                    deref(encryption_config.unwrap().get()))
        file_encryption_properties = GetResultValue(
            file_encryption_properties_result)
        return FileEncryptionProperties.wrap(file_encryption_properties)

    def file_decryption_properties(
            self,
            KmsConnectionConfig kms_connection_config,
            DecryptionConfiguration decryption_config=None):
        """Create file decryption properties.

        Parameters
        ----------
        kms_connection_config : KmsConnectionConfig
            Configuration of connection to KMS

        decryption_config : DecryptionConfiguration, default None
            Configuration of the decryption, such as cache timeout.
            Can be None.

        Returns
        -------
        file_decryption_properties : FileDecryptionProperties
            File decryption properties.
        """
        cdef:
            CDecryptionConfiguration c_decryption_config
            CResult[shared_ptr[CFileDecryptionProperties]] \
                c_file_decryption_properties
        if decryption_config is None:
            c_decryption_config = CDecryptionConfiguration()
        else:
            c_decryption_config = deref(decryption_config.unwrap().get())
        with nogil:
            c_file_decryption_properties = \
                self.factory.get().SafeGetFileDecryptionProperties(
                    deref(kms_connection_config.unwrap().get()),
                    c_decryption_config)
        file_decryption_properties = GetResultValue(
            c_file_decryption_properties)
        return FileDecryptionProperties.wrap(file_decryption_properties)

    def remove_cache_entries_for_token(self, access_token):
        self.factory.get().RemoveCacheEntriesForToken(tobytes(access_token))

    def remove_cache_entries_for_all_tokens(self):
        self.factory.get().RemoveCacheEntriesForAllTokens()

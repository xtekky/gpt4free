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

#include <map>
#include <memory>
#include <string>
#include <utility>

#include "parquet/exception.h"
#include "parquet/schema.h"
#include "parquet/types.h"

namespace parquet {

static constexpr ParquetCipher::type kDefaultEncryptionAlgorithm =
    ParquetCipher::AES_GCM_V1;
static constexpr int32_t kMaximalAadMetadataLength = 256;
static constexpr bool kDefaultEncryptedFooter = true;
static constexpr bool kDefaultCheckSignature = true;
static constexpr bool kDefaultAllowPlaintextFiles = false;
static constexpr int32_t kAadFileUniqueLength = 8;

class ColumnDecryptionProperties;
using ColumnPathToDecryptionPropertiesMap =
    std::map<std::string, std::shared_ptr<ColumnDecryptionProperties>>;

class ColumnEncryptionProperties;
using ColumnPathToEncryptionPropertiesMap =
    std::map<std::string, std::shared_ptr<ColumnEncryptionProperties>>;

class PARQUET_EXPORT DecryptionKeyRetriever {
 public:
  virtual std::string GetKey(const std::string& key_metadata) = 0;
  virtual ~DecryptionKeyRetriever() {}
};

/// Simple integer key retriever
class PARQUET_EXPORT IntegerKeyIdRetriever : public DecryptionKeyRetriever {
 public:
  void PutKey(uint32_t key_id, const std::string& key);
  std::string GetKey(const std::string& key_metadata) override;

 private:
  std::map<uint32_t, std::string> key_map_;
};

// Simple string key retriever
class PARQUET_EXPORT StringKeyIdRetriever : public DecryptionKeyRetriever {
 public:
  void PutKey(const std::string& key_id, const std::string& key);
  std::string GetKey(const std::string& key_metadata) override;

 private:
  std::map<std::string, std::string> key_map_;
};

class PARQUET_EXPORT HiddenColumnException : public ParquetException {
 public:
  explicit HiddenColumnException(const std::string& columnPath)
      : ParquetException(columnPath.c_str()) {}
};

class PARQUET_EXPORT KeyAccessDeniedException : public ParquetException {
 public:
  explicit KeyAccessDeniedException(const std::string& columnPath)
      : ParquetException(columnPath.c_str()) {}
};

inline const uint8_t* str2bytes(const std::string& str) {
  if (str.empty()) return NULLPTR;

  char* cbytes = const_cast<char*>(str.c_str());
  return reinterpret_cast<const uint8_t*>(cbytes);
}

class PARQUET_EXPORT ColumnEncryptionProperties {
 public:
  class PARQUET_EXPORT Builder {
   public:
    /// Convenience builder for encrypted columns.
    explicit Builder(const std::string& name) : Builder(name, true) {}

    /// Convenience builder for encrypted columns.
    explicit Builder(const std::shared_ptr<schema::ColumnPath>& path)
        : Builder(path->ToDotString(), true) {}

    /// Set a column-specific key.
    /// If key is not set on an encrypted column, the column will
    /// be encrypted with the footer key.
    /// keyBytes Key length must be either 16, 24 or 32 bytes.
    /// The key is cloned, and will be wiped out (array values set to 0) upon completion
    /// of file writing.
    /// Caller is responsible for wiping out the input key array.
    Builder* key(std::string column_key);

    /// Set a key retrieval metadata.
    /// use either key_metadata() or key_id(), not both
    Builder* key_metadata(const std::string& key_metadata);

    /// A convenience function to set key metadata using a string id.
    /// Set a key retrieval metadata (converted from String).
    /// use either key_metadata() or key_id(), not both
    /// key_id will be converted to metadata (UTF-8 array).
    Builder* key_id(const std::string& key_id);

    std::shared_ptr<ColumnEncryptionProperties> build() {
      return std::shared_ptr<ColumnEncryptionProperties>(
          new ColumnEncryptionProperties(encrypted_, column_path_, key_, key_metadata_));
    }

   private:
    const std::string column_path_;
    bool encrypted_;
    std::string key_;
    std::string key_metadata_;

    Builder(const std::string path, bool encrypted)
        : column_path_(path), encrypted_(encrypted) {}
  };

  std::string column_path() const { return column_path_; }
  bool is_encrypted() const { return encrypted_; }
  bool is_encrypted_with_footer_key() const { return encrypted_with_footer_key_; }
  std::string key() const { return key_; }
  std::string key_metadata() const { return key_metadata_; }

  /// Upon completion of file writing, the encryption key
  /// will be wiped out.
  void WipeOutEncryptionKey() { key_.clear(); }

  bool is_utilized() {
    if (key_.empty())
      return false;  // can re-use column properties without encryption keys
    return utilized_;
  }

  /// ColumnEncryptionProperties object can be used for writing one file only.
  /// Mark ColumnEncryptionProperties as utilized once it is used in
  /// FileEncryptionProperties as the encryption key will be wiped out upon
  /// completion of file writing.
  void set_utilized() { utilized_ = true; }

  std::shared_ptr<ColumnEncryptionProperties> DeepClone() {
    std::string key_copy = key_;
    return std::shared_ptr<ColumnEncryptionProperties>(new ColumnEncryptionProperties(
        encrypted_, column_path_, key_copy, key_metadata_));
  }

  ColumnEncryptionProperties() = default;
  ColumnEncryptionProperties(const ColumnEncryptionProperties& other) = default;
  ColumnEncryptionProperties(ColumnEncryptionProperties&& other) = default;

 private:
  const std::string column_path_;
  bool encrypted_;
  bool encrypted_with_footer_key_;
  std::string key_;
  std::string key_metadata_;
  bool utilized_;
  explicit ColumnEncryptionProperties(bool encrypted, const std::string& column_path,
                                      const std::string& key,
                                      const std::string& key_metadata);
};

class PARQUET_EXPORT ColumnDecryptionProperties {
 public:
  class PARQUET_EXPORT Builder {
   public:
    explicit Builder(const std::string& name) : column_path_(name) {}

    explicit Builder(const std::shared_ptr<schema::ColumnPath>& path)
        : Builder(path->ToDotString()) {}

    /// Set an explicit column key. If applied on a file that contains
    /// key metadata for this column the metadata will be ignored,
    /// the column will be decrypted with this key.
    /// key length must be either 16, 24 or 32 bytes.
    Builder* key(const std::string& key);

    std::shared_ptr<ColumnDecryptionProperties> build();

   private:
    const std::string column_path_;
    std::string key_;
  };

  ColumnDecryptionProperties() = default;
  ColumnDecryptionProperties(const ColumnDecryptionProperties& other) = default;
  ColumnDecryptionProperties(ColumnDecryptionProperties&& other) = default;

  std::string column_path() const { return column_path_; }
  std::string key() const { return key_; }
  bool is_utilized() { return utilized_; }

  /// ColumnDecryptionProperties object can be used for reading one file only.
  /// Mark ColumnDecryptionProperties as utilized once it is used in
  /// FileDecryptionProperties as the encryption key will be wiped out upon
  /// completion of file reading.
  void set_utilized() { utilized_ = true; }

  /// Upon completion of file reading, the encryption key
  /// will be wiped out.
  void WipeOutDecryptionKey();

  std::shared_ptr<ColumnDecryptionProperties> DeepClone();

 private:
  const std::string column_path_;
  std::string key_;
  bool utilized_;

  /// This class is only required for setting explicit column decryption keys -
  /// to override key retriever (or to provide keys when key metadata and/or
  /// key retriever are not available)
  explicit ColumnDecryptionProperties(const std::string& column_path,
                                      const std::string& key);
};

class PARQUET_EXPORT AADPrefixVerifier {
 public:
  /// Verifies identity (AAD Prefix) of individual file,
  /// or of file collection in a data set.
  /// Throws exception if an AAD prefix is wrong.
  /// In a data set, AAD Prefixes should be collected,
  /// and then checked for missing files.
  virtual void Verify(const std::string& aad_prefix) = 0;
  virtual ~AADPrefixVerifier() {}
};

class PARQUET_EXPORT FileDecryptionProperties {
 public:
  class PARQUET_EXPORT Builder {
   public:
    Builder() {
      check_plaintext_footer_integrity_ = kDefaultCheckSignature;
      plaintext_files_allowed_ = kDefaultAllowPlaintextFiles;
    }

    /// Set an explicit footer key. If applied on a file that contains
    /// footer key metadata the metadata will be ignored, the footer
    /// will be decrypted/verified with this key.
    /// If explicit key is not set, footer key will be fetched from
    /// key retriever.
    /// With explicit keys or AAD prefix, new encryption properties object must be
    /// created for each encrypted file.
    /// Explicit encryption keys (footer and column) are cloned.
    /// Upon completion of file reading, the cloned encryption keys in the properties
    /// will be wiped out (array values set to 0).
    /// Caller is responsible for wiping out the input key array.
    /// param footerKey Key length must be either 16, 24 or 32 bytes.
    Builder* footer_key(const std::string footer_key);

    /// Set explicit column keys (decryption properties).
    /// Its also possible to set a key retriever on this property object.
    /// Upon file decryption, availability of explicit keys is checked before
    /// invocation of the retriever callback.
    /// If an explicit key is available for a footer or a column,
    /// its key metadata will be ignored.
    Builder* column_keys(
        const ColumnPathToDecryptionPropertiesMap& column_decryption_properties);

    /// Set a key retriever callback. Its also possible to
    /// set explicit footer or column keys on this file property object.
    /// Upon file decryption, availability of explicit keys is checked before
    /// invocation of the retriever callback.
    /// If an explicit key is available for a footer or a column,
    /// its key metadata will be ignored.
    Builder* key_retriever(const std::shared_ptr<DecryptionKeyRetriever>& key_retriever);

    /// Skip integrity verification of plaintext footers.
    /// If not called, integrity of plaintext footers will be checked in runtime,
    /// and an exception will be thrown in the following situations:
    /// - footer signing key is not available
    /// (not passed, or not found by key retriever)
    /// - footer content and signature don't match
    Builder* disable_footer_signature_verification() {
      check_plaintext_footer_integrity_ = false;
      return this;
    }

    /// Explicitly supply the file AAD prefix.
    /// A must when a prefix is used for file encryption, but not stored in file.
    /// If AAD prefix is stored in file, it will be compared to the explicitly
    /// supplied value and an exception will be thrown if they differ.
    Builder* aad_prefix(const std::string& aad_prefix);

    /// Set callback for verification of AAD Prefixes stored in file.
    Builder* aad_prefix_verifier(std::shared_ptr<AADPrefixVerifier> aad_prefix_verifier);

    /// By default, reading plaintext (unencrypted) files is not
    /// allowed when using a decryptor
    /// - in order to detect files that were not encrypted by mistake.
    /// However, the default behavior can be overridden by calling this method.
    /// The caller should use then a different method to ensure encryption
    /// of files with sensitive data.
    Builder* plaintext_files_allowed() {
      plaintext_files_allowed_ = true;
      return this;
    }

    std::shared_ptr<FileDecryptionProperties> build() {
      return std::shared_ptr<FileDecryptionProperties>(new FileDecryptionProperties(
          footer_key_, key_retriever_, check_plaintext_footer_integrity_, aad_prefix_,
          aad_prefix_verifier_, column_decryption_properties_, plaintext_files_allowed_));
    }

   private:
    std::string footer_key_;
    std::string aad_prefix_;
    std::shared_ptr<AADPrefixVerifier> aad_prefix_verifier_;
    ColumnPathToDecryptionPropertiesMap column_decryption_properties_;

    std::shared_ptr<DecryptionKeyRetriever> key_retriever_;
    bool check_plaintext_footer_integrity_;
    bool plaintext_files_allowed_;
  };

  std::string column_key(const std::string& column_path) const;

  std::string footer_key() const { return footer_key_; }

  std::string aad_prefix() const { return aad_prefix_; }

  const std::shared_ptr<DecryptionKeyRetriever>& key_retriever() const {
    return key_retriever_;
  }

  bool check_plaintext_footer_integrity() const {
    return check_plaintext_footer_integrity_;
  }

  bool plaintext_files_allowed() const { return plaintext_files_allowed_; }

  const std::shared_ptr<AADPrefixVerifier>& aad_prefix_verifier() const {
    return aad_prefix_verifier_;
  }

  /// Upon completion of file reading, the encryption keys in the properties
  /// will be wiped out (array values set to 0).
  void WipeOutDecryptionKeys();

  bool is_utilized();

  /// FileDecryptionProperties object can be used for reading one file only.
  /// Mark FileDecryptionProperties as utilized once it is used to read a file as the
  /// encryption keys will be wiped out upon completion of file reading.
  void set_utilized() { utilized_ = true; }

  /// FileDecryptionProperties object can be used for reading one file only.
  /// (unless this object keeps the keyRetrieval callback only, and no explicit
  /// keys or aadPrefix).
  /// At the end, keys are wiped out in the memory.
  /// This method allows to clone identical properties for another file,
  /// with an option to update the aadPrefix (if newAadPrefix is null,
  /// aadPrefix will be cloned too)
  std::shared_ptr<FileDecryptionProperties> DeepClone(std::string new_aad_prefix = "");

 private:
  std::string footer_key_;
  std::string aad_prefix_;
  std::shared_ptr<AADPrefixVerifier> aad_prefix_verifier_;

  const std::string empty_string_ = "";
  ColumnPathToDecryptionPropertiesMap column_decryption_properties_;

  std::shared_ptr<DecryptionKeyRetriever> key_retriever_;
  bool check_plaintext_footer_integrity_;
  bool plaintext_files_allowed_;
  bool utilized_;

  FileDecryptionProperties(
      const std::string& footer_key,
      std::shared_ptr<DecryptionKeyRetriever> key_retriever,
      bool check_plaintext_footer_integrity, const std::string& aad_prefix,
      std::shared_ptr<AADPrefixVerifier> aad_prefix_verifier,
      const ColumnPathToDecryptionPropertiesMap& column_decryption_properties,
      bool plaintext_files_allowed);
};

class PARQUET_EXPORT FileEncryptionProperties {
 public:
  class PARQUET_EXPORT Builder {
   public:
    explicit Builder(const std::string& footer_key)
        : parquet_cipher_(kDefaultEncryptionAlgorithm),
          encrypted_footer_(kDefaultEncryptedFooter) {
      footer_key_ = footer_key;
      store_aad_prefix_in_file_ = false;
    }

    /// Create files with plaintext footer.
    /// If not called, the files will be created with encrypted footer (default).
    Builder* set_plaintext_footer() {
      encrypted_footer_ = false;
      return this;
    }

    /// Set encryption algorithm.
    /// If not called, files will be encrypted with AES_GCM_V1 (default).
    Builder* algorithm(ParquetCipher::type parquet_cipher) {
      parquet_cipher_ = parquet_cipher;
      return this;
    }

    /// Set a key retrieval metadata (converted from String).
    /// use either footer_key_metadata or footer_key_id, not both.
    Builder* footer_key_id(const std::string& key_id);

    /// Set a key retrieval metadata.
    /// use either footer_key_metadata or footer_key_id, not both.
    Builder* footer_key_metadata(const std::string& footer_key_metadata);

    /// Set the file AAD Prefix.
    Builder* aad_prefix(const std::string& aad_prefix);

    /// Skip storing AAD Prefix in file.
    /// If not called, and if AAD Prefix is set, it will be stored.
    Builder* disable_aad_prefix_storage();

    /// Set the list of encrypted columns and their properties (keys etc).
    /// If not called, all columns will be encrypted with the footer key.
    /// If called, the file columns not in the list will be left unencrypted.
    Builder* encrypted_columns(
        const ColumnPathToEncryptionPropertiesMap& encrypted_columns);

    std::shared_ptr<FileEncryptionProperties> build() {
      return std::shared_ptr<FileEncryptionProperties>(new FileEncryptionProperties(
          parquet_cipher_, footer_key_, footer_key_metadata_, encrypted_footer_,
          aad_prefix_, store_aad_prefix_in_file_, encrypted_columns_));
    }

   private:
    ParquetCipher::type parquet_cipher_;
    bool encrypted_footer_;
    std::string footer_key_;
    std::string footer_key_metadata_;

    std::string aad_prefix_;
    bool store_aad_prefix_in_file_;
    ColumnPathToEncryptionPropertiesMap encrypted_columns_;
  };
  bool encrypted_footer() const { return encrypted_footer_; }

  EncryptionAlgorithm algorithm() const { return algorithm_; }

  std::string footer_key() const { return footer_key_; }

  std::string footer_key_metadata() const { return footer_key_metadata_; }

  std::string file_aad() const { return file_aad_; }

  std::shared_ptr<ColumnEncryptionProperties> column_encryption_properties(
      const std::string& column_path);

  bool is_utilized() const { return utilized_; }

  /// FileEncryptionProperties object can be used for writing one file only.
  /// Mark FileEncryptionProperties as utilized once it is used to write a file as the
  /// encryption keys will be wiped out upon completion of file writing.
  void set_utilized() { utilized_ = true; }

  /// Upon completion of file writing, the encryption keys
  /// will be wiped out (array values set to 0).
  void WipeOutEncryptionKeys();

  /// FileEncryptionProperties object can be used for writing one file only.
  /// (at the end, keys are wiped out in the memory).
  /// This method allows to clone identical properties for another file,
  /// with an option to update the aadPrefix (if newAadPrefix is null,
  /// aadPrefix will be cloned too)
  std::shared_ptr<FileEncryptionProperties> DeepClone(std::string new_aad_prefix = "");

  ColumnPathToEncryptionPropertiesMap encrypted_columns() const {
    return encrypted_columns_;
  }

 private:
  EncryptionAlgorithm algorithm_;
  std::string footer_key_;
  std::string footer_key_metadata_;
  bool encrypted_footer_;
  std::string file_aad_;
  std::string aad_prefix_;
  bool utilized_;
  bool store_aad_prefix_in_file_;
  ColumnPathToEncryptionPropertiesMap encrypted_columns_;

  FileEncryptionProperties(ParquetCipher::type cipher, const std::string& footer_key,
                           const std::string& footer_key_metadata, bool encrypted_footer,
                           const std::string& aad_prefix, bool store_aad_prefix_in_file,
                           const ColumnPathToEncryptionPropertiesMap& encrypted_columns);
};

}  // namespace parquet

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

// This module defines an abstract interface for iterating through pages in a
// Parquet column chunk within a row group. It could be extended in the future
// to iterate through all data pages in all chunks in a file.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include <gtest/gtest.h>

#include "arrow/util/io_util.h"

#include "parquet/encryption/encryption.h"
#include "parquet/test_util.h"

namespace parquet {
class ParquetFileReader;
namespace encryption {
namespace test {

using ::arrow::internal::TemporaryDir;

constexpr int kFixedLength = 10;

const char kFooterEncryptionKey[] = "0123456789012345";  // 128bit/16
const char kColumnEncryptionKey1[] = "1234567890123450";
const char kColumnEncryptionKey2[] = "1234567890123451";
const char kFileName[] = "tester";

// Get the path of file inside parquet test data directory
std::string data_file(const char* file);

// A temporary directory that contains the encrypted files generated in the tests.
extern std::unique_ptr<TemporaryDir> temp_dir;

inline ::arrow::Result<std::unique_ptr<TemporaryDir>> temp_data_dir() {
  return TemporaryDir::Make("parquet-encryption-test-");
}

const char kDoubleFieldName[] = "double_field";
const char kFloatFieldName[] = "float_field";
const char kBooleanFieldName[] = "boolean_field";
const char kInt32FieldName[] = "int32_field";
const char kInt64FieldName[] = "int64_field";
const char kInt96FieldName[] = "int96_field";
const char kByteArrayFieldName[] = "ba_field";
const char kFixedLenByteArrayFieldName[] = "flba_field";

const char kFooterMasterKey[] = "0123456789112345";
const char kFooterMasterKeyId[] = "kf";
const char* const kColumnMasterKeys[] = {"1234567890123450", "1234567890123451",
                                         "1234567890123452", "1234567890123453",
                                         "1234567890123454", "1234567890123455"};
const char* const kColumnMasterKeyIds[] = {"kc1", "kc2", "kc3", "kc4", "kc5", "kc6"};

// The result of this function will be used to set into TestOnlyInMemoryKmsClientFactory
// as the key mapping to look at.
std::unordered_map<std::string, std::string> BuildKeyMap(const char* const* column_ids,
                                                         const char* const* column_keys,
                                                         const char* footer_id,
                                                         const char* footer_key);

// The result of this function will be used to set into EncryptionConfiguration
// as colum keys.
std::string BuildColumnKeyMapping();

// FileEncryptor and FileDecryptor are helper classes to write/read an encrypted parquet
// file corresponding to each pair of FileEncryptionProperties/FileDecryptionProperties.
// FileEncryptor writes the file with fixed data values and FileDecryptor reads the file
// and verify the correctness of data values.
class FileEncryptor {
 public:
  FileEncryptor();

  void EncryptFile(
      std::string file,
      std::shared_ptr<parquet::FileEncryptionProperties> encryption_configurations);

 private:
  std::shared_ptr<schema::GroupNode> SetupEncryptionSchema();

  int num_rowgroups_ = 5;
  int rows_per_rowgroup_ = 50;
  std::shared_ptr<schema::GroupNode> schema_;
};

class FileDecryptor {
 public:
  void DecryptFile(std::string file_name,
                   std::shared_ptr<FileDecryptionProperties> file_decryption_properties);

 private:
  void CheckFile(parquet::ParquetFileReader* file_reader,
                 FileDecryptionProperties* file_decryption_properties);
};

}  // namespace test
}  // namespace encryption
}  // namespace parquet

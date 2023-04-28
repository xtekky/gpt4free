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

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "parquet/platform.h"
#include "parquet/properties.h"
#include "parquet/schema.h"
#include "parquet/types.h"

namespace parquet {

class ColumnDescriptor;
class EncodedStatistics;
class Statistics;
class SchemaDescriptor;

class FileCryptoMetaData;
class InternalFileDecryptor;
class Decryptor;
class Encryptor;
class FooterSigningEncryptor;

namespace schema {

class ColumnPath;

}  // namespace schema

using KeyValueMetadata = ::arrow::KeyValueMetadata;

class PARQUET_EXPORT ApplicationVersion {
 public:
  // Known Versions with Issues
  static const ApplicationVersion& PARQUET_251_FIXED_VERSION();
  static const ApplicationVersion& PARQUET_816_FIXED_VERSION();
  static const ApplicationVersion& PARQUET_CPP_FIXED_STATS_VERSION();
  static const ApplicationVersion& PARQUET_MR_FIXED_STATS_VERSION();
  static const ApplicationVersion& PARQUET_CPP_10353_FIXED_VERSION();

  // Application that wrote the file. e.g. "IMPALA"
  std::string application_;
  // Build name
  std::string build_;

  // Version of the application that wrote the file, expressed as
  // (<major>.<minor>.<patch>). Unmatched parts default to 0.
  // "1.2.3"    => {1, 2, 3}
  // "1.2"      => {1, 2, 0}
  // "1.2-cdh5" => {1, 2, 0}
  struct {
    int major;
    int minor;
    int patch;
    std::string unknown;
    std::string pre_release;
    std::string build_info;
  } version;

  ApplicationVersion() = default;
  explicit ApplicationVersion(const std::string& created_by);
  ApplicationVersion(std::string application, int major, int minor, int patch);

  // Returns true if version is strictly less than other_version
  bool VersionLt(const ApplicationVersion& other_version) const;

  // Returns true if version is strictly equal with other_version
  bool VersionEq(const ApplicationVersion& other_version) const;

  // Checks if the Version has the correct statistics for a given column
  bool HasCorrectStatistics(Type::type primitive, EncodedStatistics& statistics,
                            SortOrder::type sort_order = SortOrder::SIGNED) const;
};

class PARQUET_EXPORT ColumnCryptoMetaData {
 public:
  static std::unique_ptr<ColumnCryptoMetaData> Make(const uint8_t* metadata);
  ~ColumnCryptoMetaData();

  bool Equals(const ColumnCryptoMetaData& other) const;

  std::shared_ptr<schema::ColumnPath> path_in_schema() const;
  bool encrypted_with_footer_key() const;
  const std::string& key_metadata() const;

 private:
  explicit ColumnCryptoMetaData(const uint8_t* metadata);

  class ColumnCryptoMetaDataImpl;
  std::unique_ptr<ColumnCryptoMetaDataImpl> impl_;
};

/// \brief Public struct for Thrift PageEncodingStats in ColumnChunkMetaData
struct PageEncodingStats {
  PageType::type page_type;
  Encoding::type encoding;
  int32_t count;
};

/// \brief Public struct for location to page index in ColumnChunkMetaData.
struct IndexLocation {
  /// File offset of the given index, in bytes
  int64_t offset;
  /// Length of the given index, in bytes
  int32_t length;
};

/// \brief ColumnChunkMetaData is a proxy around format::ColumnChunkMetaData.
class PARQUET_EXPORT ColumnChunkMetaData {
 public:
  // API convenience to get a MetaData accessor

  ARROW_DEPRECATED("Use the ReaderProperties-taking overload")
  static std::unique_ptr<ColumnChunkMetaData> Make(
      const void* metadata, const ColumnDescriptor* descr,
      const ApplicationVersion* writer_version, int16_t row_group_ordinal = -1,
      int16_t column_ordinal = -1,
      std::shared_ptr<InternalFileDecryptor> file_decryptor = NULLPTR);

  static std::unique_ptr<ColumnChunkMetaData> Make(
      const void* metadata, const ColumnDescriptor* descr,
      const ReaderProperties& properties = default_reader_properties(),
      const ApplicationVersion* writer_version = NULLPTR, int16_t row_group_ordinal = -1,
      int16_t column_ordinal = -1,
      std::shared_ptr<InternalFileDecryptor> file_decryptor = NULLPTR);

  ~ColumnChunkMetaData();

  bool Equals(const ColumnChunkMetaData& other) const;

  // column chunk
  int64_t file_offset() const;

  // parameter is only used when a dataset is spread across multiple files
  const std::string& file_path() const;

  // column metadata
  bool is_metadata_set() const;
  Type::type type() const;
  int64_t num_values() const;
  std::shared_ptr<schema::ColumnPath> path_in_schema() const;
  bool is_stats_set() const;
  std::shared_ptr<Statistics> statistics() const;

  Compression::type compression() const;
  // Indicate if the ColumnChunk compression is supported by the current
  // compiled parquet library.
  bool can_decompress() const;

  const std::vector<Encoding::type>& encodings() const;
  const std::vector<PageEncodingStats>& encoding_stats() const;
  bool has_dictionary_page() const;
  int64_t dictionary_page_offset() const;
  int64_t data_page_offset() const;
  bool has_index_page() const;
  int64_t index_page_offset() const;
  int64_t total_compressed_size() const;
  int64_t total_uncompressed_size() const;
  std::unique_ptr<ColumnCryptoMetaData> crypto_metadata() const;
  std::optional<IndexLocation> GetColumnIndexLocation() const;
  std::optional<IndexLocation> GetOffsetIndexLocation() const;

 private:
  explicit ColumnChunkMetaData(
      const void* metadata, const ColumnDescriptor* descr, int16_t row_group_ordinal,
      int16_t column_ordinal, const ReaderProperties& properties,
      const ApplicationVersion* writer_version = NULLPTR,
      std::shared_ptr<InternalFileDecryptor> file_decryptor = NULLPTR);
  // PIMPL Idiom
  class ColumnChunkMetaDataImpl;
  std::unique_ptr<ColumnChunkMetaDataImpl> impl_;
};

/// \brief RowGroupMetaData is a proxy around format::RowGroupMetaData.
class PARQUET_EXPORT RowGroupMetaData {
 public:
  ARROW_DEPRECATED("Use the ReaderProperties-taking overload")
  static std::unique_ptr<RowGroupMetaData> Make(
      const void* metadata, const SchemaDescriptor* schema,
      const ApplicationVersion* writer_version,
      std::shared_ptr<InternalFileDecryptor> file_decryptor = NULLPTR);

  /// \brief Create a RowGroupMetaData from a serialized thrift message.
  static std::unique_ptr<RowGroupMetaData> Make(
      const void* metadata, const SchemaDescriptor* schema,
      const ReaderProperties& properties = default_reader_properties(),
      const ApplicationVersion* writer_version = NULLPTR,
      std::shared_ptr<InternalFileDecryptor> file_decryptor = NULLPTR);

  ~RowGroupMetaData();

  bool Equals(const RowGroupMetaData& other) const;

  /// \brief The number of columns in this row group. The order must match the
  /// parent's column ordering.
  int num_columns() const;

  /// \brief Return the ColumnChunkMetaData of the corresponding column ordinal.
  ///
  /// WARNING, the returned object references memory location in it's parent
  /// (RowGroupMetaData) object. Hence, the parent must outlive the returned
  /// object.
  ///
  /// \param[in] index of the ColumnChunkMetaData to retrieve.
  ///
  /// \throws ParquetException if the index is out of bound.
  std::unique_ptr<ColumnChunkMetaData> ColumnChunk(int index) const;

  /// \brief Number of rows in this row group.
  int64_t num_rows() const;

  /// \brief Total byte size of all the uncompressed column data in this row group.
  int64_t total_byte_size() const;

  /// \brief Total byte size of all the compressed (and potentially encrypted)
  /// column data in this row group.
  ///
  /// This information is optional and may be 0 if omitted.
  int64_t total_compressed_size() const;

  /// \brief Byte offset from beginning of file to first page (data or
  /// dictionary) in this row group
  ///
  /// The file_offset field that this method exposes is optional. This method
  /// will return 0 if that field is not set to a meaningful value.
  int64_t file_offset() const;
  // Return const-pointer to make it clear that this object is not to be copied
  const SchemaDescriptor* schema() const;
  // Indicate if all of the RowGroup's ColumnChunks can be decompressed.
  bool can_decompress() const;

 private:
  explicit RowGroupMetaData(
      const void* metadata, const SchemaDescriptor* schema,
      const ReaderProperties& properties,
      const ApplicationVersion* writer_version = NULLPTR,
      std::shared_ptr<InternalFileDecryptor> file_decryptor = NULLPTR);
  // PIMPL Idiom
  class RowGroupMetaDataImpl;
  std::unique_ptr<RowGroupMetaDataImpl> impl_;
};

class FileMetaDataBuilder;

/// \brief FileMetaData is a proxy around format::FileMetaData.
class PARQUET_EXPORT FileMetaData {
 public:
  ARROW_DEPRECATED("Use the ReaderProperties-taking overload")
  static std::shared_ptr<FileMetaData> Make(
      const void* serialized_metadata, uint32_t* inout_metadata_len,
      std::shared_ptr<InternalFileDecryptor> file_decryptor);

  /// \brief Create a FileMetaData from a serialized thrift message.
  static std::shared_ptr<FileMetaData> Make(
      const void* serialized_metadata, uint32_t* inout_metadata_len,
      const ReaderProperties& properties = default_reader_properties(),
      std::shared_ptr<InternalFileDecryptor> file_decryptor = NULLPTR);

  ~FileMetaData();

  bool Equals(const FileMetaData& other) const;

  /// \brief The number of parquet "leaf" columns.
  ///
  /// Parquet thrift definition requires that nested schema elements are
  /// flattened. This method returns the number of columns in the flattened
  /// version.
  /// For instance, if the schema looks like this :
  /// 0 foo.bar
  ///       foo.bar.baz           0
  ///       foo.bar.baz2          1
  ///   foo.qux                   2
  /// 1 foo2                      3
  /// 2 foo3                      4
  /// This method will return 5, because there are 5 "leaf" fields (so 5
  /// flattened fields)
  int num_columns() const;

  /// \brief The number of flattened schema elements.
  ///
  /// Parquet thrift definition requires that nested schema elements are
  /// flattened. This method returns the total number of elements in the
  /// flattened list.
  int num_schema_elements() const;

  /// \brief The total number of rows.
  int64_t num_rows() const;

  /// \brief The number of row groups in the file.
  int num_row_groups() const;

  /// \brief Return the RowGroupMetaData of the corresponding row group ordinal.
  ///
  /// WARNING, the returned object references memory location in it's parent
  /// (FileMetaData) object. Hence, the parent must outlive the returned object.
  ///
  /// \param[in] index of the RowGroup to retrieve.
  ///
  /// \throws ParquetException if the index is out of bound.
  std::unique_ptr<RowGroupMetaData> RowGroup(int index) const;

  /// \brief Return the "version" of the file
  ///
  /// WARNING: The value returned by this method is unreliable as 1) the Parquet
  /// file metadata stores the version as a single integer and 2) some producers
  /// are known to always write a hardcoded value.  Therefore, you cannot use
  /// this value to know which features are used in the file.
  ParquetVersion::type version() const;

  /// \brief Return the application's user-agent string of the writer.
  const std::string& created_by() const;

  /// \brief Return the application's version of the writer.
  const ApplicationVersion& writer_version() const;

  /// \brief Size of the original thrift encoded metadata footer.
  uint32_t size() const;

  /// \brief Indicate if all of the FileMetadata's RowGroups can be decompressed.
  ///
  /// This will return false if any of the RowGroup's page is compressed with a
  /// compression format which is not compiled in the current parquet library.
  bool can_decompress() const;

  bool is_encryption_algorithm_set() const;
  EncryptionAlgorithm encryption_algorithm() const;
  const std::string& footer_signing_key_metadata() const;

  /// \brief Verify signature of FileMetaData when file is encrypted but footer
  /// is not encrypted (plaintext footer).
  bool VerifySignature(const void* signature);

  void WriteTo(::arrow::io::OutputStream* dst,
               const std::shared_ptr<Encryptor>& encryptor = NULLPTR) const;

  /// \brief Return Thrift-serialized representation of the metadata as a
  /// string
  std::string SerializeToString() const;

  // Return const-pointer to make it clear that this object is not to be copied
  const SchemaDescriptor* schema() const;

  const std::shared_ptr<const KeyValueMetadata>& key_value_metadata() const;

  /// \brief Set a path to all ColumnChunk for all RowGroups.
  ///
  /// Commonly used by systems (Dask, Spark) who generates an metadata-only
  /// parquet file. The path is usually relative to said index file.
  ///
  /// \param[in] path to set.
  void set_file_path(const std::string& path);

  /// \brief Merge row groups from another metadata file into this one.
  ///
  /// The schema of the input FileMetaData must be equal to the
  /// schema of this object.
  ///
  /// This is used by systems who creates an aggregate metadata-only file by
  /// concatenating the row groups of multiple files. This newly created
  /// metadata file acts as an index of all available row groups.
  ///
  /// \param[in] other FileMetaData to merge the row groups from.
  ///
  /// \throws ParquetException if schemas are not equal.
  void AppendRowGroups(const FileMetaData& other);

  /// \brief Return a FileMetaData containing a subset of the row groups in this
  /// FileMetaData.
  std::shared_ptr<FileMetaData> Subset(const std::vector<int>& row_groups) const;

 private:
  friend FileMetaDataBuilder;
  friend class SerializedFile;

  explicit FileMetaData(const void* serialized_metadata, uint32_t* metadata_len,
                        const ReaderProperties& properties,
                        std::shared_ptr<InternalFileDecryptor> file_decryptor = NULLPTR);

  void set_file_decryptor(std::shared_ptr<InternalFileDecryptor> file_decryptor);

  // PIMPL Idiom
  FileMetaData();
  class FileMetaDataImpl;
  std::unique_ptr<FileMetaDataImpl> impl_;
};

class PARQUET_EXPORT FileCryptoMetaData {
 public:
  // API convenience to get a MetaData accessor
  static std::shared_ptr<FileCryptoMetaData> Make(
      const uint8_t* serialized_metadata, uint32_t* metadata_len,
      const ReaderProperties& properties = default_reader_properties());
  ~FileCryptoMetaData();

  EncryptionAlgorithm encryption_algorithm() const;
  const std::string& key_metadata() const;

  void WriteTo(::arrow::io::OutputStream* dst) const;

 private:
  friend FileMetaDataBuilder;
  FileCryptoMetaData(const uint8_t* serialized_metadata, uint32_t* metadata_len,
                     const ReaderProperties& properties);

  // PIMPL Idiom
  FileCryptoMetaData();
  class FileCryptoMetaDataImpl;
  std::unique_ptr<FileCryptoMetaDataImpl> impl_;
};

// Builder API
class PARQUET_EXPORT ColumnChunkMetaDataBuilder {
 public:
  // API convenience to get a MetaData reader
  static std::unique_ptr<ColumnChunkMetaDataBuilder> Make(
      std::shared_ptr<WriterProperties> props, const ColumnDescriptor* column);

  static std::unique_ptr<ColumnChunkMetaDataBuilder> Make(
      std::shared_ptr<WriterProperties> props, const ColumnDescriptor* column,
      void* contents);

  ~ColumnChunkMetaDataBuilder();

  // column chunk
  // Used when a dataset is spread across multiple files
  void set_file_path(const std::string& path);
  // column metadata
  void SetStatistics(const EncodedStatistics& stats);
  // get the column descriptor
  const ColumnDescriptor* descr() const;

  int64_t total_compressed_size() const;
  // commit the metadata

  void Finish(int64_t num_values, int64_t dictionary_page_offset,
              int64_t index_page_offset, int64_t data_page_offset,
              int64_t compressed_size, int64_t uncompressed_size, bool has_dictionary,
              bool dictionary_fallback,
              const std::map<Encoding::type, int32_t>& dict_encoding_stats_,
              const std::map<Encoding::type, int32_t>& data_encoding_stats_,
              const std::shared_ptr<Encryptor>& encryptor = NULLPTR);

  // The metadata contents, suitable for passing to ColumnChunkMetaData::Make
  const void* contents() const;

  // For writing metadata at end of column chunk
  void WriteTo(::arrow::io::OutputStream* sink);

 private:
  explicit ColumnChunkMetaDataBuilder(std::shared_ptr<WriterProperties> props,
                                      const ColumnDescriptor* column);
  explicit ColumnChunkMetaDataBuilder(std::shared_ptr<WriterProperties> props,
                                      const ColumnDescriptor* column, void* contents);
  // PIMPL Idiom
  class ColumnChunkMetaDataBuilderImpl;
  std::unique_ptr<ColumnChunkMetaDataBuilderImpl> impl_;
};

class PARQUET_EXPORT RowGroupMetaDataBuilder {
 public:
  // API convenience to get a MetaData reader
  static std::unique_ptr<RowGroupMetaDataBuilder> Make(
      std::shared_ptr<WriterProperties> props, const SchemaDescriptor* schema_,
      void* contents);

  ~RowGroupMetaDataBuilder();

  ColumnChunkMetaDataBuilder* NextColumnChunk();
  int num_columns();
  int64_t num_rows();
  int current_column() const;

  void set_num_rows(int64_t num_rows);

  // commit the metadata
  void Finish(int64_t total_bytes_written, int16_t row_group_ordinal = -1);

 private:
  explicit RowGroupMetaDataBuilder(std::shared_ptr<WriterProperties> props,
                                   const SchemaDescriptor* schema_, void* contents);
  // PIMPL Idiom
  class RowGroupMetaDataBuilderImpl;
  std::unique_ptr<RowGroupMetaDataBuilderImpl> impl_;
};

class PARQUET_EXPORT FileMetaDataBuilder {
 public:
  // API convenience to get a MetaData reader
  static std::unique_ptr<FileMetaDataBuilder> Make(
      const SchemaDescriptor* schema, std::shared_ptr<WriterProperties> props,
      std::shared_ptr<const KeyValueMetadata> key_value_metadata = NULLPTR);

  ~FileMetaDataBuilder();

  // The prior RowGroupMetaDataBuilder (if any) is destroyed
  RowGroupMetaDataBuilder* AppendRowGroup();

  // Complete the Thrift structure
  std::unique_ptr<FileMetaData> Finish();

  // crypto metadata
  std::unique_ptr<FileCryptoMetaData> GetCryptoMetaData();

 private:
  explicit FileMetaDataBuilder(
      const SchemaDescriptor* schema, std::shared_ptr<WriterProperties> props,
      std::shared_ptr<const KeyValueMetadata> key_value_metadata = NULLPTR);
  // PIMPL Idiom
  class FileMetaDataBuilderImpl;
  std::unique_ptr<FileMetaDataBuilderImpl> impl_;
};

PARQUET_EXPORT std::string ParquetVersionToString(ParquetVersion::type ver);

}  // namespace parquet

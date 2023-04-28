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
#include <memory>
#include <vector>

#include "arrow/adapters/orc/options.h"
#include "arrow/io/interfaces.h"
#include "arrow/memory_pool.h"
#include "arrow/record_batch.h"
#include "arrow/status.h"
#include "arrow/type.h"
#include "arrow/type_fwd.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace adapters {
namespace orc {

/// \brief Information about an ORC stripe
struct StripeInformation {
  /// \brief Offset of the stripe from the start of the file, in bytes
  int64_t offset;
  /// \brief Length of the stripe, in bytes
  int64_t length;
  /// \brief Number of rows in the stripe
  int64_t num_rows;
  /// \brief Index of the first row of the stripe
  int64_t first_row_id;
};

/// \class ORCFileReader
/// \brief Read an Arrow Table or RecordBatch from an ORC file.
class ARROW_EXPORT ORCFileReader {
 public:
  ~ORCFileReader();

  /// \brief Creates a new ORC reader
  ///
  /// \param[in] file the data source
  /// \param[in] pool a MemoryPool to use for buffer allocations
  /// \return the returned reader object
  static Result<std::unique_ptr<ORCFileReader>> Open(
      const std::shared_ptr<io::RandomAccessFile>& file, MemoryPool* pool);

  /// \brief Return the schema read from the ORC file
  ///
  /// \return the returned Schema object
  Result<std::shared_ptr<Schema>> ReadSchema();

  /// \brief Read the file as a Table
  ///
  /// The table will be composed of one record batch per stripe.
  ///
  /// \return the returned Table
  Result<std::shared_ptr<Table>> Read();

  /// \brief Read the file as a Table
  ///
  /// The table will be composed of one record batch per stripe.
  ///
  /// \param[in] schema the Table schema
  /// \return the returned Table
  Result<std::shared_ptr<Table>> Read(const std::shared_ptr<Schema>& schema);

  /// \brief Read the file as a Table
  ///
  /// The table will be composed of one record batch per stripe.
  ///
  /// \param[in] include_indices the selected field indices to read
  /// \return the returned Table
  Result<std::shared_ptr<Table>> Read(const std::vector<int>& include_indices);

  /// \brief Read the file as a Table
  ///
  /// The table will be composed of one record batch per stripe.
  ///
  /// \param[in] include_names the selected field names to read
  /// \return the returned Table
  Result<std::shared_ptr<Table>> Read(const std::vector<std::string>& include_names);

  /// \brief Read the file as a Table
  ///
  /// The table will be composed of one record batch per stripe.
  ///
  /// \param[in] schema the Table schema
  /// \param[in] include_indices the selected field indices to read
  /// \return the returned Table
  Result<std::shared_ptr<Table>> Read(const std::shared_ptr<Schema>& schema,
                                      const std::vector<int>& include_indices);

  /// \brief Read a single stripe as a RecordBatch
  ///
  /// \param[in] stripe the stripe index
  /// \return the returned RecordBatch
  Result<std::shared_ptr<RecordBatch>> ReadStripe(int64_t stripe);

  /// \brief Read a single stripe as a RecordBatch
  ///
  /// \param[in] stripe the stripe index
  /// \param[in] include_indices the selected field indices to read
  /// \return the returned RecordBatch
  Result<std::shared_ptr<RecordBatch>> ReadStripe(
      int64_t stripe, const std::vector<int>& include_indices);

  /// \brief Read a single stripe as a RecordBatch
  ///
  /// \param[in] stripe the stripe index
  /// \param[in] include_names the selected field names to read
  /// \return the returned RecordBatch
  Result<std::shared_ptr<RecordBatch>> ReadStripe(
      int64_t stripe, const std::vector<std::string>& include_names);

  /// \brief Seek to designated row. Invoke NextStripeReader() after seek
  ///        will return stripe reader starting from designated row.
  ///
  /// \param[in] row_number the rows number to seek
  Status Seek(int64_t row_number);

  /// \brief Get a stripe level record batch iterator.
  ///
  /// Each record batch will have up to `batch_size` rows.
  /// NextStripeReader serves as a fine grained alternative to ReadStripe
  /// which may cause OOM issues by loading the whole stripe into memory.
  ///
  /// Note this will only read rows for the current stripe, not the entire
  /// file.
  ///
  /// \param[in] batch_size the maximum number of rows in each record batch
  /// \return the returned stripe reader
  Result<std::shared_ptr<RecordBatchReader>> NextStripeReader(int64_t batch_size);

  /// \brief Get a stripe level record batch iterator.
  ///
  /// Each record batch will have up to `batch_size` rows.
  /// NextStripeReader serves as a fine grained alternative to ReadStripe
  /// which may cause OOM issues by loading the whole stripe into memory.
  ///
  /// Note this will only read rows for the current stripe, not the entire
  /// file.
  ///
  /// \param[in] batch_size the maximum number of rows in each record batch
  /// \param[in] include_indices the selected field indices to read
  /// \return the stripe reader
  Result<std::shared_ptr<RecordBatchReader>> NextStripeReader(
      int64_t batch_size, const std::vector<int>& include_indices);

  /// \brief Get a record batch iterator for the entire file.
  ///
  /// Each record batch will have up to `batch_size` rows.
  ///
  /// \param[in] batch_size the maximum number of rows in each record batch
  /// \param[in] include_names the selected field names to read, if not empty
  /// (otherwise all fields are read)
  /// \return the record batch iterator
  Result<std::shared_ptr<RecordBatchReader>> GetRecordBatchReader(
      int64_t batch_size, const std::vector<std::string>& include_names);

  /// \brief The number of stripes in the file
  int64_t NumberOfStripes();

  /// \brief The number of rows in the file
  int64_t NumberOfRows();

  /// \brief StripeInformation for each stripe.
  StripeInformation GetStripeInformation(int64_t stripe);

  /// \brief Get the format version of the file.
  ///         Currently known values are 0.11 and 0.12.
  ///
  /// \return The FileVersion of the ORC file.
  FileVersion GetFileVersion();

  /// \brief Get the software instance and version that wrote this file.
  ///
  /// \return a user-facing string that specifies the software version
  std::string GetSoftwareVersion();

  /// \brief Get the compression kind of the file.
  ///
  /// \return The kind of compression in the ORC file.
  Result<Compression::type> GetCompression();

  /// \brief Get the buffer size for the compression.
  ///
  /// \return Number of bytes to buffer for the compression codec.
  int64_t GetCompressionSize();

  /// \brief Get the number of rows per an entry in the row index.
  /// \return the number of rows per an entry in the row index or 0 if there
  ///          is no row index.
  int64_t GetRowIndexStride();

  /// \brief Get ID of writer that generated the file.
  ///
  /// \return UNKNOWN_WRITER if the writer ID is undefined
  WriterId GetWriterId();

  /// \brief Get the writer id value when getWriterId() returns an unknown writer.
  ///
  /// \return the integer value of the writer ID.
  int32_t GetWriterIdValue();

  /// \brief Get the version of the writer.
  ///
  /// \return the version of the writer.

  WriterVersion GetWriterVersion();

  /// \brief Get the number of stripe statistics in the file.
  ///
  /// \return the number of stripe statistics
  int64_t GetNumberOfStripeStatistics();

  /// \brief Get the length of the data stripes in the file.
  ///
  /// \return return the number of bytes in stripes
  int64_t GetContentLength();

  /// \brief Get the length of the file stripe statistics.
  ///
  /// \return the number of compressed bytes in the file stripe statistics
  int64_t GetStripeStatisticsLength();

  /// \brief Get the length of the file footer.
  ///
  /// \return the number of compressed bytes in the file footer
  int64_t GetFileFooterLength();

  /// \brief Get the length of the file postscript.
  ///
  /// \return the number of bytes in the file postscript
  int64_t GetFilePostscriptLength();

  /// \brief Get the total length of the file.
  ///
  /// \return the number of bytes in the file
  int64_t GetFileLength();

  /// \brief Get the serialized file tail.
  ///         Usefull if another reader of the same file wants to avoid re-reading
  ///         the file tail. See ReadOptions.SetSerializedFileTail().
  ///
  /// \return a string of bytes with the file tail
  std::string GetSerializedFileTail();

  /// \brief Return the metadata read from the ORC file
  ///
  /// \return A KeyValueMetadata object containing the ORC metadata
  Result<std::shared_ptr<const KeyValueMetadata>> ReadMetadata();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
  ORCFileReader();
};

/// \class ORCFileWriter
/// \brief Write an Arrow Table or RecordBatch to an ORC file.
class ARROW_EXPORT ORCFileWriter {
 public:
  ~ORCFileWriter();
  /// \brief Creates a new ORC writer.
  ///
  /// \param[in] output_stream a pointer to the io::OutputStream to write into
  /// \param[in] write_options the ORC writer options for Arrow
  /// \return the returned writer object
  static Result<std::unique_ptr<ORCFileWriter>> Open(
      io::OutputStream* output_stream,
      const WriteOptions& write_options = WriteOptions());

  /// \brief Write a table. This can be called multiple times.
  ///
  /// Tables passed in subsequent calls must match the schema of the table that was
  /// written first.
  ///
  /// \param[in] table the Arrow table from which data is extracted.
  /// \return Status
  Status Write(const Table& table);

  /// \brief Write a RecordBatch. This can be called multiple times.
  ///
  /// RecordBatches passed in subsequent calls must match the schema of the
  /// RecordBatch that was written first.
  ///
  /// \param[in] record_batch the Arrow RecordBatch from which data is extracted.
  /// \return Status
  Status Write(const RecordBatch& record_batch);

  /// \brief Close an ORC writer (orc::Writer)
  ///
  /// \return Status
  Status Close();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;

 private:
  ORCFileWriter();
};

}  // namespace orc
}  // namespace adapters
}  // namespace arrow

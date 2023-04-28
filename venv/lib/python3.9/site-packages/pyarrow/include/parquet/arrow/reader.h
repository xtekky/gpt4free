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
// N.B. we don't include async_generator.h as it's relatively heavy
#include <functional>
#include <memory>
#include <vector>

#include "parquet/file_reader.h"
#include "parquet/platform.h"
#include "parquet/properties.h"

namespace arrow {

class ChunkedArray;
class KeyValueMetadata;
class RecordBatchReader;
struct Scalar;
class Schema;
class Table;
class RecordBatch;

}  // namespace arrow

namespace parquet {

class FileMetaData;
class SchemaDescriptor;

namespace arrow {

class ColumnChunkReader;
class ColumnReader;
struct SchemaManifest;
class RowGroupReader;

/// \brief Arrow read adapter class for deserializing Parquet files as Arrow row batches.
///
/// This interfaces caters for different use cases and thus provides different
/// interfaces. In its most simplistic form, we cater for a user that wants to
/// read the whole Parquet at once with the `FileReader::ReadTable` method.
///
/// More advanced users that also want to implement parallelism on top of each
/// single Parquet files should do this on the RowGroup level. For this, they can
/// call `FileReader::RowGroup(i)->ReadTable` to receive only the specified
/// RowGroup as a table.
///
/// In the most advanced situation, where a consumer wants to independently read
/// RowGroups in parallel and consume each column individually, they can call
/// `FileReader::RowGroup(i)->Column(j)->Read` and receive an `arrow::Column`
/// instance.
///
/// Finally, one can also get a stream of record batches using
/// `FileReader::GetRecordBatchReader()`. This can internally decode columns
/// in parallel if use_threads was enabled in the ArrowReaderProperties.
///
/// The parquet format supports an optional integer field_id which can be assigned
/// to a field.  Arrow will convert these field IDs to a metadata key named
/// PARQUET:field_id on the appropriate field.
// TODO(wesm): nested data does not always make sense with this user
// interface unless you are only reading a single leaf node from a branch of
// a table. For example:
//
// repeated group data {
//   optional group record {
//     optional int32 val1;
//     optional byte_array val2;
//     optional bool val3;
//   }
//   optional int32 val4;
// }
//
// In the Parquet file, there are 3 leaf nodes:
//
// * data.record.val1
// * data.record.val2
// * data.record.val3
// * data.val4
//
// When materializing this data in an Arrow array, we would have:
//
// data: list<struct<
//   record: struct<
//    val1: int32,
//    val2: string (= list<uint8>),
//    val3: bool,
//   >,
//   val4: int32
// >>
//
// However, in the Parquet format, each leaf node has its own repetition and
// definition levels describing the structure of the intermediate nodes in
// this array structure. Thus, we will need to scan the leaf data for a group
// of leaf nodes part of the same type tree to create a single result Arrow
// nested array structure.
//
// This is additionally complicated "chunky" repeated fields or very large byte
// arrays
class PARQUET_EXPORT FileReader {
 public:
  /// Factory function to create a FileReader from a ParquetFileReader and properties
  static ::arrow::Status Make(::arrow::MemoryPool* pool,
                              std::unique_ptr<ParquetFileReader> reader,
                              const ArrowReaderProperties& properties,
                              std::unique_ptr<FileReader>* out);

  /// Factory function to create a FileReader from a ParquetFileReader
  static ::arrow::Status Make(::arrow::MemoryPool* pool,
                              std::unique_ptr<ParquetFileReader> reader,
                              std::unique_ptr<FileReader>* out);

  // Since the distribution of columns amongst a Parquet file's row groups may
  // be uneven (the number of values in each column chunk can be different), we
  // provide a column-oriented read interface. The ColumnReader hides the
  // details of paging through the file's row groups and yielding
  // fully-materialized arrow::Array instances
  //
  // Returns error status if the column of interest is not flat.
  // The indicated column index is relative to the schema
  virtual ::arrow::Status GetColumn(int i, std::unique_ptr<ColumnReader>* out) = 0;

  /// \brief Return arrow schema for all the columns.
  virtual ::arrow::Status GetSchema(std::shared_ptr<::arrow::Schema>* out) = 0;

  /// \brief Read column as a whole into a chunked array.
  ///
  /// The indicated column index is relative to the schema
  virtual ::arrow::Status ReadColumn(int i,
                                     std::shared_ptr<::arrow::ChunkedArray>* out) = 0;

  // NOTE: Experimental API
  // Reads a specific top level schema field into an Array
  // The index i refers the index of the top level schema field, which may
  // be nested or flat - e.g.
  //
  // 0 foo.bar
  //   foo.bar.baz
  //   foo.qux
  // 1 foo2
  // 2 foo3
  //
  // i=0 will read the entire foo struct, i=1 the foo2 primitive column etc
  ARROW_DEPRECATED("Deprecated in 9.0.0. Use ReadColumn instead.")
  virtual ::arrow::Status ReadSchemaField(
      int i, std::shared_ptr<::arrow::ChunkedArray>* out) = 0;

  /// \brief Return a RecordBatchReader of all row groups and columns.
  virtual ::arrow::Status GetRecordBatchReader(
      std::unique_ptr<::arrow::RecordBatchReader>* out) = 0;

  /// \brief Return a RecordBatchReader of row groups selected from row_group_indices.
  ///
  /// Note that the ordering in row_group_indices matters. FileReaders must outlive
  /// their RecordBatchReaders.
  ///
  /// \returns error Status if row_group_indices contains an invalid index
  virtual ::arrow::Status GetRecordBatchReader(
      const std::vector<int>& row_group_indices,
      std::unique_ptr<::arrow::RecordBatchReader>* out) = 0;

  /// \brief Return a RecordBatchReader of row groups selected from
  /// row_group_indices, whose columns are selected by column_indices.
  ///
  /// Note that the ordering in row_group_indices and column_indices
  /// matter. FileReaders must outlive their RecordBatchReaders.
  ///
  /// \returns error Status if either row_group_indices or column_indices
  ///     contains an invalid index
  virtual ::arrow::Status GetRecordBatchReader(
      const std::vector<int>& row_group_indices, const std::vector<int>& column_indices,
      std::unique_ptr<::arrow::RecordBatchReader>* out) = 0;

  /// \brief Return a RecordBatchReader of row groups selected from
  /// row_group_indices, whose columns are selected by column_indices.
  ///
  /// Note that the ordering in row_group_indices and column_indices
  /// matter. FileReaders must outlive their RecordBatchReaders.
  ///
  /// \param row_group_indices which row groups to read (order determines read order).
  /// \param column_indices which columns to read (order determines output schema).
  /// \param[out] out record batch stream from parquet data.
  ///
  /// \returns error Status if either row_group_indices or column_indices
  ///     contains an invalid index
  ::arrow::Status GetRecordBatchReader(const std::vector<int>& row_group_indices,
                                       const std::vector<int>& column_indices,
                                       std::shared_ptr<::arrow::RecordBatchReader>* out);
  ::arrow::Status GetRecordBatchReader(const std::vector<int>& row_group_indices,
                                       std::shared_ptr<::arrow::RecordBatchReader>* out);
  ::arrow::Status GetRecordBatchReader(std::shared_ptr<::arrow::RecordBatchReader>* out);

  /// \brief Return a generator of record batches.
  ///
  /// The FileReader must outlive the generator, so this requires that you pass in a
  /// shared_ptr.
  ///
  /// \returns error Result if either row_group_indices or column_indices contains an
  ///     invalid index
  virtual ::arrow::Result<
      std::function<::arrow::Future<std::shared_ptr<::arrow::RecordBatch>>()>>
  GetRecordBatchGenerator(std::shared_ptr<FileReader> reader,
                          const std::vector<int> row_group_indices,
                          const std::vector<int> column_indices,
                          ::arrow::internal::Executor* cpu_executor = NULLPTR,
                          int64_t rows_to_readahead = 0) = 0;

  /// Read all columns into a Table
  virtual ::arrow::Status ReadTable(std::shared_ptr<::arrow::Table>* out) = 0;

  /// \brief Read the given columns into a Table
  ///
  /// The indicated column indices are relative to the internal representation
  /// of the parquet table. For instance :
  /// 0 foo.bar
  ///       foo.bar.baz           0
  ///       foo.bar.baz2          1
  ///   foo.qux                   2
  /// 1 foo2                      3
  /// 2 foo3                      4
  ///
  /// i=0 will read foo.bar.baz, i=1 will read only foo.bar.baz2 and so on.
  /// Only leaf fields have indices; foo itself doesn't have an index.
  /// To get the index for a particular leaf field, one can use
  /// manifest().schema_fields to get the top level fields, and then walk the
  /// tree to identify the relevant leaf fields and access its column_index.
  /// To get the total number of leaf fields, use FileMetadata.num_columns().
  virtual ::arrow::Status ReadTable(const std::vector<int>& column_indices,
                                    std::shared_ptr<::arrow::Table>* out) = 0;

  virtual ::arrow::Status ReadRowGroup(int i, const std::vector<int>& column_indices,
                                       std::shared_ptr<::arrow::Table>* out) = 0;

  virtual ::arrow::Status ReadRowGroup(int i, std::shared_ptr<::arrow::Table>* out) = 0;

  virtual ::arrow::Status ReadRowGroups(const std::vector<int>& row_groups,
                                        const std::vector<int>& column_indices,
                                        std::shared_ptr<::arrow::Table>* out) = 0;

  virtual ::arrow::Status ReadRowGroups(const std::vector<int>& row_groups,
                                        std::shared_ptr<::arrow::Table>* out) = 0;

  /// \brief Scan file contents with one thread, return number of rows
  virtual ::arrow::Status ScanContents(std::vector<int> columns,
                                       const int32_t column_batch_size,
                                       int64_t* num_rows) = 0;

  /// \brief Return a reader for the RowGroup, this object must not outlive the
  ///   FileReader.
  virtual std::shared_ptr<RowGroupReader> RowGroup(int row_group_index) = 0;

  /// \brief The number of row groups in the file
  virtual int num_row_groups() const = 0;

  virtual ParquetFileReader* parquet_reader() const = 0;

  /// Set whether to use multiple threads during reads of multiple columns.
  /// By default only one thread is used.
  virtual void set_use_threads(bool use_threads) = 0;

  /// Set number of records to read per batch for the RecordBatchReader.
  virtual void set_batch_size(int64_t batch_size) = 0;

  virtual const ArrowReaderProperties& properties() const = 0;

  virtual const SchemaManifest& manifest() const = 0;

  virtual ~FileReader() = default;
};

class RowGroupReader {
 public:
  virtual ~RowGroupReader() = default;
  virtual std::shared_ptr<ColumnChunkReader> Column(int column_index) = 0;
  virtual ::arrow::Status ReadTable(const std::vector<int>& column_indices,
                                    std::shared_ptr<::arrow::Table>* out) = 0;
  virtual ::arrow::Status ReadTable(std::shared_ptr<::arrow::Table>* out) = 0;

 private:
  struct Iterator;
};

class ColumnChunkReader {
 public:
  virtual ~ColumnChunkReader() = default;
  virtual ::arrow::Status Read(std::shared_ptr<::arrow::ChunkedArray>* out) = 0;
};

// At this point, the column reader is a stream iterator. It only knows how to
// read the next batch of values for a particular column from the file until it
// runs out.
//
// We also do not expose any internal Parquet details, such as row groups. This
// might change in the future.
class PARQUET_EXPORT ColumnReader {
 public:
  virtual ~ColumnReader() = default;

  // Scan the next array of the indicated size. The actual size of the
  // returned array may be less than the passed size depending how much data is
  // available in the file.
  //
  // When all the data in the file has been exhausted, the result is set to
  // nullptr.
  //
  // Returns Status::OK on a successful read, including if you have exhausted
  // the data available in the file.
  virtual ::arrow::Status NextBatch(int64_t batch_size,
                                    std::shared_ptr<::arrow::ChunkedArray>* out) = 0;
};

/// \brief Experimental helper class for bindings (like Python) that struggle
/// either with std::move or C++ exceptions
class PARQUET_EXPORT FileReaderBuilder {
 public:
  FileReaderBuilder();

  /// Create FileReaderBuilder from Arrow file and optional properties / metadata
  ::arrow::Status Open(std::shared_ptr<::arrow::io::RandomAccessFile> file,
                       const ReaderProperties& properties = default_reader_properties(),
                       std::shared_ptr<FileMetaData> metadata = NULLPTR);

  /// Create FileReaderBuilder from file path and optional properties / metadata
  ::arrow::Status OpenFile(const std::string& path, bool memory_map = false,
                           const ReaderProperties& props = default_reader_properties(),
                           std::shared_ptr<FileMetaData> metadata = NULLPTR);

  ParquetFileReader* raw_reader() { return raw_reader_.get(); }

  /// Set Arrow MemoryPool for memory allocation
  FileReaderBuilder* memory_pool(::arrow::MemoryPool* pool);
  /// Set Arrow reader properties
  FileReaderBuilder* properties(const ArrowReaderProperties& arg_properties);
  /// Build FileReader instance
  ::arrow::Status Build(std::unique_ptr<FileReader>* out);
  ::arrow::Result<std::unique_ptr<FileReader>> Build();

 private:
  ::arrow::MemoryPool* pool_;
  ArrowReaderProperties properties_;
  std::unique_ptr<ParquetFileReader> raw_reader_;
};

/// \defgroup parquet-arrow-reader-factories Factory functions for Parquet Arrow readers
///
/// @{

/// \brief Build FileReader from Arrow file and MemoryPool
///
/// Advanced settings are supported through the FileReaderBuilder class.
PARQUET_EXPORT
::arrow::Status OpenFile(std::shared_ptr<::arrow::io::RandomAccessFile>,
                         ::arrow::MemoryPool* allocator,
                         std::unique_ptr<FileReader>* reader);

/// @}

PARQUET_EXPORT
::arrow::Status StatisticsAsScalars(const Statistics& Statistics,
                                    std::shared_ptr<::arrow::Scalar>* min,
                                    std::shared_ptr<::arrow::Scalar>* max);

namespace internal {

PARQUET_EXPORT
::arrow::Status FuzzReader(const uint8_t* data, int64_t size);

}  // namespace internal
}  // namespace arrow
}  // namespace parquet

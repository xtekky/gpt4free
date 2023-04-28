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
#include <string>
#include <vector>

#include "arrow/chunked_array.h"  // IWYU pragma: keep
#include "arrow/record_batch.h"
#include "arrow/status.h"
#include "arrow/type.h"
#include "arrow/type_fwd.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {

class Array;
class ChunkedArray;
class KeyValueMetadata;
class MemoryPool;

/// \class Table
/// \brief Logical table as sequence of chunked arrays
class ARROW_EXPORT Table {
 public:
  virtual ~Table() = default;

  /// \brief Construct a Table from schema and columns
  ///
  /// If columns is zero-length, the table's number of rows is zero
  ///
  /// \param[in] schema The table schema (column types)
  /// \param[in] columns The table's columns as chunked arrays
  /// \param[in] num_rows number of rows in table, -1 (default) to infer from columns
  static std::shared_ptr<Table> Make(std::shared_ptr<Schema> schema,
                                     std::vector<std::shared_ptr<ChunkedArray>> columns,
                                     int64_t num_rows = -1);

  /// \brief Construct a Table from schema and arrays
  ///
  /// \param[in] schema The table schema (column types)
  /// \param[in] arrays The table's columns as arrays
  /// \param[in] num_rows number of rows in table, -1 (default) to infer from columns
  static std::shared_ptr<Table> Make(std::shared_ptr<Schema> schema,
                                     const std::vector<std::shared_ptr<Array>>& arrays,
                                     int64_t num_rows = -1);

  /// \brief Create an empty Table of a given schema
  ///
  /// The output Table will be created with a single empty chunk per column.
  ///
  /// \param[in] schema the schema of the empty Table
  /// \param[in] pool the memory pool to allocate memory from
  /// \return the resulting Table
  static Result<std::shared_ptr<Table>> MakeEmpty(
      std::shared_ptr<Schema> schema, MemoryPool* pool = default_memory_pool());

  /// \brief Construct a Table from a RecordBatchReader.
  ///
  /// \param[in] reader the arrow::Schema for each batch
  static Result<std::shared_ptr<Table>> FromRecordBatchReader(RecordBatchReader* reader);

  /// \brief Construct a Table from RecordBatches, using schema supplied by the first
  /// RecordBatch.
  ///
  /// \param[in] batches a std::vector of record batches
  static Result<std::shared_ptr<Table>> FromRecordBatches(
      const std::vector<std::shared_ptr<RecordBatch>>& batches);

  /// \brief Construct a Table from RecordBatches, using supplied schema. There may be
  /// zero record batches
  ///
  /// \param[in] schema the arrow::Schema for each batch
  /// \param[in] batches a std::vector of record batches
  static Result<std::shared_ptr<Table>> FromRecordBatches(
      std::shared_ptr<Schema> schema,
      const std::vector<std::shared_ptr<RecordBatch>>& batches);

  /// \brief Construct a Table from a chunked StructArray. One column will be produced
  /// for each field of the StructArray.
  ///
  /// \param[in] array a chunked StructArray
  static Result<std::shared_ptr<Table>> FromChunkedStructArray(
      const std::shared_ptr<ChunkedArray>& array);

  /// \brief Return the table schema
  const std::shared_ptr<Schema>& schema() const { return schema_; }

  /// \brief Return a column by index
  virtual std::shared_ptr<ChunkedArray> column(int i) const = 0;

  /// \brief Return vector of all columns for table
  virtual const std::vector<std::shared_ptr<ChunkedArray>>& columns() const = 0;

  /// Return a column's field by index
  std::shared_ptr<Field> field(int i) const { return schema_->field(i); }

  /// \brief Return vector of all fields for table
  std::vector<std::shared_ptr<Field>> fields() const;

  /// \brief Construct a zero-copy slice of the table with the
  /// indicated offset and length
  ///
  /// \param[in] offset the index of the first row in the constructed
  /// slice
  /// \param[in] length the number of rows of the slice. If there are not enough
  /// rows in the table, the length will be adjusted accordingly
  ///
  /// \return a new object wrapped in std::shared_ptr<Table>
  virtual std::shared_ptr<Table> Slice(int64_t offset, int64_t length) const = 0;

  /// \brief Slice from first row at offset until end of the table
  std::shared_ptr<Table> Slice(int64_t offset) const { return Slice(offset, num_rows_); }

  /// \brief Return a column by name
  /// \param[in] name field name
  /// \return an Array or null if no field was found
  std::shared_ptr<ChunkedArray> GetColumnByName(const std::string& name) const {
    auto i = schema_->GetFieldIndex(name);
    return i == -1 ? NULLPTR : column(i);
  }

  /// \brief Remove column from the table, producing a new Table
  virtual Result<std::shared_ptr<Table>> RemoveColumn(int i) const = 0;

  /// \brief Add column to the table, producing a new Table
  virtual Result<std::shared_ptr<Table>> AddColumn(
      int i, std::shared_ptr<Field> field_arg,
      std::shared_ptr<ChunkedArray> column) const = 0;

  /// \brief Replace a column in the table, producing a new Table
  virtual Result<std::shared_ptr<Table>> SetColumn(
      int i, std::shared_ptr<Field> field_arg,
      std::shared_ptr<ChunkedArray> column) const = 0;

  /// \brief Return names of all columns
  std::vector<std::string> ColumnNames() const;

  /// \brief Rename columns with provided names
  Result<std::shared_ptr<Table>> RenameColumns(
      const std::vector<std::string>& names) const;

  /// \brief Return new table with specified columns
  Result<std::shared_ptr<Table>> SelectColumns(const std::vector<int>& indices) const;

  /// \brief Replace schema key-value metadata with new metadata
  /// \since 0.5.0
  ///
  /// \param[in] metadata new KeyValueMetadata
  /// \return new Table
  virtual std::shared_ptr<Table> ReplaceSchemaMetadata(
      const std::shared_ptr<const KeyValueMetadata>& metadata) const = 0;

  /// \brief Flatten the table, producing a new Table.  Any column with a
  /// struct type will be flattened into multiple columns
  ///
  /// \param[in] pool The pool for buffer allocations, if any
  virtual Result<std::shared_ptr<Table>> Flatten(
      MemoryPool* pool = default_memory_pool()) const = 0;

  /// \return PrettyPrint representation suitable for debugging
  std::string ToString() const;

  /// \brief Perform cheap validation checks to determine obvious inconsistencies
  /// within the table's schema and internal data.
  ///
  /// This is O(k*m) where k is the total number of field descendents,
  /// and m is the number of chunks.
  ///
  /// \return Status
  virtual Status Validate() const = 0;

  /// \brief Perform extensive validation checks to determine inconsistencies
  /// within the table's schema and internal data.
  ///
  /// This is O(k*n) where k is the total number of field descendents,
  /// and n is the number of rows.
  ///
  /// \return Status
  virtual Status ValidateFull() const = 0;

  /// \brief Return the number of columns in the table
  int num_columns() const { return schema_->num_fields(); }

  /// \brief Return the number of rows (equal to each column's logical length)
  int64_t num_rows() const { return num_rows_; }

  /// \brief Determine if tables are equal
  ///
  /// Two tables can be equal only if they have equal schemas.
  /// However, they may be equal even if they have different chunkings.
  bool Equals(const Table& other, bool check_metadata = false) const;

  /// \brief Make a new table by combining the chunks this table has.
  ///
  /// All the underlying chunks in the ChunkedArray of each column are
  /// concatenated into zero or one chunk.
  ///
  /// \param[in] pool The pool for buffer allocations
  Result<std::shared_ptr<Table>> CombineChunks(
      MemoryPool* pool = default_memory_pool()) const;

  /// \brief Make a new record batch by combining the chunks this table has.
  ///
  /// All the underlying chunks in the ChunkedArray of each column are
  /// concatenated into a single chunk.
  ///
  /// \param[in] pool The pool for buffer allocations
  Result<std::shared_ptr<RecordBatch>> CombineChunksToBatch(
      MemoryPool* pool = default_memory_pool()) const;

 protected:
  Table();

  std::shared_ptr<Schema> schema_;
  int64_t num_rows_;

 private:
  ARROW_DISALLOW_COPY_AND_ASSIGN(Table);
};

/// \brief Compute a stream of record batches from a (possibly chunked) Table
///
/// The conversion is zero-copy: each record batch is a view over a slice
/// of the table's columns.
class ARROW_EXPORT TableBatchReader : public RecordBatchReader {
 public:
  /// \brief Construct a TableBatchReader for the given table
  explicit TableBatchReader(const Table& table);
  explicit TableBatchReader(std::shared_ptr<Table> table);

  std::shared_ptr<Schema> schema() const override;

  Status ReadNext(std::shared_ptr<RecordBatch>* out) override;

  /// \brief Set the desired maximum chunk size of record batches
  ///
  /// The actual chunk size of each record batch may be smaller, depending
  /// on actual chunking characteristics of each table column.
  void set_chunksize(int64_t chunksize);

 private:
  std::shared_ptr<Table> owned_table_;
  const Table& table_;
  std::vector<ChunkedArray*> column_data_;
  std::vector<int> chunk_numbers_;
  std::vector<int64_t> chunk_offsets_;
  int64_t absolute_row_position_;
  int64_t max_chunksize_;
};

/// \defgroup concat-tables ConcatenateTables function.
///
/// ConcatenateTables function.
/// @{

/// \brief Controls the behavior of ConcatenateTables().
struct ARROW_EXPORT ConcatenateTablesOptions {
  /// If true, the schemas of the tables will be first unified with fields of
  /// the same name being merged, according to `field_merge_options`, then each
  /// table will be promoted to the unified schema before being concatenated.
  /// Otherwise, all tables should have the same schema. Each column in the output table
  /// is the result of concatenating the corresponding columns in all input tables.
  bool unify_schemas = false;

  Field::MergeOptions field_merge_options = Field::MergeOptions::Defaults();

  static ConcatenateTablesOptions Defaults() { return {}; }
};

/// \brief Construct a new table from multiple input tables.
///
/// The new table is assembled from existing column chunks without copying,
/// if schemas are identical. If schemas do not match exactly and
/// unify_schemas is enabled in options (off by default), an attempt is
/// made to unify them, and then column chunks are converted to their
/// respective unified datatype, which will probably incur a copy.
/// :func:`arrow::PromoteTableToSchema` is used to unify schemas.
///
/// Tables are concatenated in order they are provided in and the order of
/// rows within tables will be preserved.
///
/// \param[in] tables a std::vector of Tables to be concatenated
/// \param[in] options specify how to unify schema of input tables
/// \param[in] memory_pool MemoryPool to be used if null-filled arrays need to
/// be created or if existing column chunks need to endure type conversion
/// \return new Table

ARROW_EXPORT
Result<std::shared_ptr<Table>> ConcatenateTables(
    const std::vector<std::shared_ptr<Table>>& tables,
    ConcatenateTablesOptions options = ConcatenateTablesOptions::Defaults(),
    MemoryPool* memory_pool = default_memory_pool());

/// \brief Promotes a table to conform to the given schema.
///
/// If a field in the schema does not have a corresponding column in the
/// table, a column of nulls will be added to the resulting table.
/// If the corresponding column is of type Null, it will be promoted to
/// the type specified by schema, with null values filled.
/// Returns an error:
/// - if the corresponding column's type is not compatible with the
///   schema.
/// - if there is a column in the table that does not exist in the schema.
///
/// \param[in] table the input Table
/// \param[in] schema the target schema to promote to
/// \param[in] pool The memory pool to be used if null-filled arrays need to
/// be created.
ARROW_EXPORT
Result<std::shared_ptr<Table>> PromoteTableToSchema(
    const std::shared_ptr<Table>& table, const std::shared_ptr<Schema>& schema,
    MemoryPool* pool = default_memory_pool());

}  // namespace arrow

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

#include <memory>

#include "arrow/csv/options.h"
#include "arrow/io/interfaces.h"
#include "arrow/ipc/type_fwd.h"
#include "arrow/record_batch.h"
#include "arrow/table.h"

namespace arrow {
namespace csv {

// Functionality for converting Arrow data to Comma separated value text.
// This library supports all primitive types that can be cast to a StringArrays.
// It applies to following formatting rules:
//  - For non-binary types no quotes surround values.  Nulls are represented as the empty
//  string.
//  - For binary types all non-null data is quoted (and quotes within data are escaped
//  with an additional quote).
//    Null values are empty and unquoted.

/// \defgroup csv-write-functions High-level functions for writing CSV files
/// @{

/// \brief Convert table to CSV and write the result to output.
/// Experimental
ARROW_EXPORT Status WriteCSV(const Table& table, const WriteOptions& options,
                             arrow::io::OutputStream* output);
/// \brief Convert batch to CSV and write the result to output.
/// Experimental
ARROW_EXPORT Status WriteCSV(const RecordBatch& batch, const WriteOptions& options,
                             arrow::io::OutputStream* output);
/// \brief Convert batches read through a RecordBatchReader
/// to CSV and write the results to output.
/// Experimental
ARROW_EXPORT Status WriteCSV(const std::shared_ptr<RecordBatchReader>& reader,
                             const WriteOptions& options,
                             arrow::io::OutputStream* output);

/// @}

/// \defgroup csv-writer-factories Functions for creating an incremental CSV writer
/// @{

/// \brief Create a new CSV writer. User is responsible for closing the
/// actual OutputStream.
///
/// \param[in] sink output stream to write to
/// \param[in] schema the schema of the record batches to be written
/// \param[in] options options for serialization
/// \return Result<std::shared_ptr<RecordBatchWriter>>
ARROW_EXPORT
Result<std::shared_ptr<ipc::RecordBatchWriter>> MakeCSVWriter(
    std::shared_ptr<io::OutputStream> sink, const std::shared_ptr<Schema>& schema,
    const WriteOptions& options = WriteOptions::Defaults());

/// \brief Create a new CSV writer.
///
/// \param[in] sink output stream to write to (does not take ownership)
/// \param[in] schema the schema of the record batches to be written
/// \param[in] options options for serialization
/// \return Result<std::shared_ptr<RecordBatchWriter>>
ARROW_EXPORT
Result<std::shared_ptr<ipc::RecordBatchWriter>> MakeCSVWriter(
    io::OutputStream* sink, const std::shared_ptr<Schema>& schema,
    const WriteOptions& options = WriteOptions::Defaults());

/// @}

}  // namespace csv
}  // namespace arrow

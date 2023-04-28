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

#include "arrow/csv/options.h"  // IWYU pragma: keep
#include "arrow/io/interfaces.h"
#include "arrow/record_batch.h"
#include "arrow/result.h"
#include "arrow/type.h"
#include "arrow/type_fwd.h"
#include "arrow/util/future.h"
#include "arrow/util/thread_pool.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace io {
class InputStream;
}  // namespace io

namespace csv {

/// A class that reads an entire CSV file into a Arrow Table
class ARROW_EXPORT TableReader {
 public:
  virtual ~TableReader() = default;

  /// Read the entire CSV file and convert it to a Arrow Table
  virtual Result<std::shared_ptr<Table>> Read() = 0;
  /// Read the entire CSV file and convert it to a Arrow Table
  virtual Future<std::shared_ptr<Table>> ReadAsync() = 0;

  /// Create a TableReader instance
  static Result<std::shared_ptr<TableReader>> Make(io::IOContext io_context,
                                                   std::shared_ptr<io::InputStream> input,
                                                   const ReadOptions&,
                                                   const ParseOptions&,
                                                   const ConvertOptions&);

  ARROW_DEPRECATED(
      "Deprecated in 4.0.0. "
      "Use MemoryPool-less variant (the IOContext holds a pool already)")
  static Result<std::shared_ptr<TableReader>> Make(
      MemoryPool* pool, io::IOContext io_context, std::shared_ptr<io::InputStream> input,
      const ReadOptions&, const ParseOptions&, const ConvertOptions&);
};

/// \brief A class that reads a CSV file incrementally
///
/// Caveats:
/// - For now, this is always single-threaded (regardless of `ReadOptions::use_threads`.
/// - Type inference is done on the first block and types are frozen afterwards;
///   to make sure the right data types are inferred, either set
///   `ReadOptions::block_size` to a large enough value, or use
///   `ConvertOptions::column_types` to set the desired data types explicitly.
class ARROW_EXPORT StreamingReader : public RecordBatchReader {
 public:
  virtual ~StreamingReader() = default;

  virtual Future<std::shared_ptr<RecordBatch>> ReadNextAsync() = 0;

  /// \brief Return the number of bytes which have been read and processed
  ///
  /// The returned number includes CSV bytes which the StreamingReader has
  /// finished processing, but not bytes for which some processing (e.g.
  /// CSV parsing or conversion to Arrow layout) is still ongoing.
  ///
  /// Furthermore, the following rules apply:
  /// - bytes skipped by `ReadOptions.skip_rows` are counted as being read before
  /// any records are returned.
  /// - bytes read while parsing the header are counted as being read before any
  /// records are returned.
  /// - bytes skipped by `ReadOptions.skip_rows_after_names` are counted after the
  /// first batch is returned.
  virtual int64_t bytes_read() const = 0;

  /// Create a StreamingReader instance
  ///
  /// This involves some I/O as the first batch must be loaded during the creation process
  /// so it is returned as a future
  ///
  /// Currently, the StreamingReader is not async-reentrant and does not do any fan-out
  /// parsing (see ARROW-11889)
  static Future<std::shared_ptr<StreamingReader>> MakeAsync(
      io::IOContext io_context, std::shared_ptr<io::InputStream> input,
      arrow::internal::Executor* cpu_executor, const ReadOptions&, const ParseOptions&,
      const ConvertOptions&);

  static Result<std::shared_ptr<StreamingReader>> Make(
      io::IOContext io_context, std::shared_ptr<io::InputStream> input,
      const ReadOptions&, const ParseOptions&, const ConvertOptions&);

  ARROW_DEPRECATED("Deprecated in 4.0.0. Use IOContext-based overload")
  static Result<std::shared_ptr<StreamingReader>> Make(
      MemoryPool* pool, std::shared_ptr<io::InputStream> input,
      const ReadOptions& read_options, const ParseOptions& parse_options,
      const ConvertOptions& convert_options);
};

/// \brief Count the logical rows of data in a CSV file (i.e. the
/// number of rows you would get if you read the file into a table).
ARROW_EXPORT
Future<int64_t> CountRowsAsync(io::IOContext io_context,
                               std::shared_ptr<io::InputStream> input,
                               arrow::internal::Executor* cpu_executor,
                               const ReadOptions&, const ParseOptions&);

}  // namespace csv
}  // namespace arrow

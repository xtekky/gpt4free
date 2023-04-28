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

#include "arrow/io/type_fwd.h"
#include "arrow/json/options.h"
#include "arrow/record_batch.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/macros.h"
#include "arrow/util/type_fwd.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace json {

/// A class that reads an entire JSON file into a Arrow Table
///
/// The file is expected to consist of individual line-separated JSON objects
class ARROW_EXPORT TableReader {
 public:
  virtual ~TableReader() = default;

  /// Read the entire JSON file and convert it to a Arrow Table
  virtual Result<std::shared_ptr<Table>> Read() = 0;

  /// Create a TableReader instance
  static Result<std::shared_ptr<TableReader>> Make(MemoryPool* pool,
                                                   std::shared_ptr<io::InputStream> input,
                                                   const ReadOptions&,
                                                   const ParseOptions&);
};

ARROW_EXPORT Result<std::shared_ptr<RecordBatch>> ParseOne(ParseOptions options,
                                                           std::shared_ptr<Buffer> json);

/// \brief A class that reads a JSON file incrementally
///
/// JSON data is read from a stream in fixed-size blocks (configurable with
/// `ReadOptions::block_size`). Each block is converted to a `RecordBatch`. Yielded
/// batches have a consistent schema but may differ in row count.
///
/// The supplied `ParseOptions` are used to determine a schema, based either on a
/// provided explicit schema or inferred from the first non-empty block.
/// Afterwards, the target schema is frozen. If `UnexpectedFieldBehavior::InferType` is
/// specified, unexpected fields will only be inferred for the first block. Afterwards
/// they'll be treated as errors.
///
/// If `ReadOptions::use_threads` is `true`, each block's parsing/decoding task will be
/// parallelized on the given `cpu_executor` (with readahead corresponding to the
/// executor's capacity). If an executor isn't provided, the global thread pool will be
/// used.
///
/// If `ReadOptions::use_threads` is `false`, computations will be run on the calling
/// thread and `cpu_executor` will be ignored.
class ARROW_EXPORT StreamingReader : public RecordBatchReader {
 public:
  virtual ~StreamingReader() = default;

  /// \brief Read the next `RecordBatch` asynchronously
  /// This function is async-reentrant (but not synchronously reentrant). However, if
  /// threading is disabled, this will block until completion.
  virtual Future<std::shared_ptr<RecordBatch>> ReadNextAsync() = 0;

  /// Get the number of bytes which have been succesfully converted to record batches
  /// and consumed
  [[nodiscard]] virtual int64_t bytes_processed() const = 0;

  /// \brief Create a `StreamingReader` from an `InputStream`
  /// Blocks until the initial batch is loaded
  ///
  /// \param[in] stream JSON source stream
  /// \param[in] read_options Options for reading
  /// \param[in] parse_options Options for chunking, parsing, and conversion
  /// \param[in] io_context Context for IO operations (optional)
  /// \param[in] cpu_executor Executor for computation tasks (optional)
  /// \return The initialized reader
  static Result<std::shared_ptr<StreamingReader>> Make(
      std::shared_ptr<io::InputStream> stream, const ReadOptions& read_options,
      const ParseOptions& parse_options,
      const io::IOContext& io_context = io::default_io_context(),
      ::arrow::internal::Executor* cpu_executor = NULLPTR);

  /// \brief Create a `StreamingReader` from an `InputStream` asynchronously
  /// Returned future completes after loading the first batch
  ///
  /// \param[in] stream JSON source stream
  /// \param[in] read_options Options for reading
  /// \param[in] parse_options Options for chunking, parsing, and conversion
  /// \param[in] io_context Context for IO operations (optional)
  /// \param[in] cpu_executor Executor for computation tasks (optional)
  /// \return Future for the initialized reader
  static Future<std::shared_ptr<StreamingReader>> MakeAsync(
      std::shared_ptr<io::InputStream> stream, const ReadOptions& read_options,
      const ParseOptions& parse_options,
      const io::IOContext& io_context = io::default_io_context(),
      ::arrow::internal::Executor* cpu_executor = NULLPTR);
};

}  // namespace json
}  // namespace arrow

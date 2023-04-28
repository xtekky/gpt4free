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
#include <vector>

#include "arrow/io/caching.h"
#include "arrow/ipc/type_fwd.h"
#include "arrow/status.h"
#include "arrow/type_fwd.h"
#include "arrow/util/compression.h"
#include "arrow/util/visibility.h"

namespace arrow {

class MemoryPool;

namespace ipc {

// ARROW-109: We set this number arbitrarily to help catch user mistakes. For
// deeply nested schemas, it is expected the user will indicate explicitly the
// maximum allowed recursion depth
constexpr int kMaxNestingDepth = 64;

/// \brief Options for writing Arrow IPC messages
struct ARROW_EXPORT IpcWriteOptions {
  /// \brief If true, allow field lengths that don't fit in a signed 32-bit int.
  ///
  /// Some implementations may not be able to parse streams created with this option.
  bool allow_64bit = false;

  /// \brief The maximum permitted schema nesting depth.
  int max_recursion_depth = kMaxNestingDepth;

  /// \brief Write padding after memory buffers up to this multiple of bytes.
  int32_t alignment = 8;

  /// \brief Write the pre-0.15.0 IPC message format
  ///
  /// This legacy format consists of a 4-byte prefix instead of 8-byte.
  bool write_legacy_ipc_format = false;

  /// \brief The memory pool to use for allocations made during IPC writing
  ///
  /// While Arrow IPC is predominantly zero-copy, it may have to allocate
  /// memory in some cases (for example if compression is enabled).
  MemoryPool* memory_pool = default_memory_pool();

  /// \brief Compression codec to use for record batch body buffers
  ///
  /// May only be UNCOMPRESSED, LZ4_FRAME and ZSTD.
  std::shared_ptr<util::Codec> codec;

  /// \brief Use global CPU thread pool to parallelize any computational tasks
  /// like compression
  bool use_threads = true;

  /// \brief Whether to emit dictionary deltas
  ///
  /// If false, a changed dictionary for a given field will emit a full
  /// dictionary replacement.
  /// If true, a changed dictionary will be compared against the previous
  /// version. If possible, a dictionary delta will be emitted, otherwise
  /// a full dictionary replacement.
  ///
  /// Default is false to maximize stream compatibility.
  ///
  /// Also, note that if a changed dictionary is a nested dictionary,
  /// then a delta is never emitted, for compatibility with the read path.
  bool emit_dictionary_deltas = false;

  /// \brief Whether to unify dictionaries for the IPC file format
  ///
  /// The IPC file format doesn't support dictionary replacements.
  /// Therefore, chunks of a column with a dictionary type must have the same
  /// dictionary in each record batch (or an extended dictionary + delta).
  ///
  /// If this option is true, RecordBatchWriter::WriteTable will attempt
  /// to unify dictionaries across each table column.  If this option is
  /// false, incompatible dictionaries across a table column will simply
  /// raise an error.
  ///
  /// Note that enabling this option has a runtime cost. Also, not all types
  /// currently support dictionary unification.
  ///
  /// This option is ignored for IPC streams, which support dictionary replacement
  /// and deltas.
  bool unify_dictionaries = false;

  /// \brief Format version to use for IPC messages and their metadata.
  ///
  /// Presently using V5 version (readable by 1.0.0 and later).
  /// V4 is also available (readable by 0.8.0 and later).
  MetadataVersion metadata_version = MetadataVersion::V5;

  static IpcWriteOptions Defaults();
};

/// \brief Options for reading Arrow IPC messages
struct ARROW_EXPORT IpcReadOptions {
  /// \brief The maximum permitted schema nesting depth.
  int max_recursion_depth = kMaxNestingDepth;

  /// \brief The memory pool to use for allocations made during IPC reading
  ///
  /// While Arrow IPC is predominantly zero-copy, it may have to allocate
  /// memory in some cases (for example if compression is enabled).
  MemoryPool* memory_pool = default_memory_pool();

  /// \brief Top-level schema fields to include when deserializing RecordBatch.
  ///
  /// If empty (the default), return all deserialized fields.
  /// If non-empty, the values are the indices of fields in the top-level schema.
  std::vector<int> included_fields;

  /// \brief Use global CPU thread pool to parallelize any computational tasks
  /// like decompression
  bool use_threads = true;

  /// \brief Whether to convert incoming data to platform-native endianness
  ///
  /// If the endianness of the received schema is not equal to platform-native
  /// endianness, then all buffers with endian-sensitive data will be byte-swapped.
  /// This includes the value buffers of numeric types, temporal types, decimal
  /// types, as well as the offset buffers of variable-sized binary and list-like
  /// types.
  ///
  /// Endianness conversion is achieved by the RecordBatchFileReader,
  /// RecordBatchStreamReader and StreamDecoder classes.
  bool ensure_native_endian = true;

  /// \brief Options to control caching behavior when pre-buffering is requested
  ///
  /// The lazy property will always be reset to true to deliver the expected behavior
  io::CacheOptions pre_buffer_cache_options = io::CacheOptions::LazyDefaults();

  static IpcReadOptions Defaults();
};

namespace internal {

Status CheckCompressionSupported(Compression::type codec);

}  // namespace internal
}  // namespace ipc
}  // namespace arrow

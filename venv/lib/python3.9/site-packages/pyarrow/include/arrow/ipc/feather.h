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

// Public API for the "Feather" file format, originally created at
// http://github.com/wesm/feather

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "arrow/ipc/options.h"
#include "arrow/type_fwd.h"
#include "arrow/util/compression.h"
#include "arrow/util/visibility.h"

namespace arrow {

class Schema;
class Status;
class Table;

namespace io {

class OutputStream;
class RandomAccessFile;

}  // namespace io

namespace ipc {
namespace feather {

static constexpr const int kFeatherV1Version = 2;
static constexpr const int kFeatherV2Version = 3;

// ----------------------------------------------------------------------
// Metadata accessor classes

/// \class Reader
/// \brief An interface for reading columns from Feather files
class ARROW_EXPORT Reader {
 public:
  virtual ~Reader() = default;

  /// \brief Open a Feather file from a RandomAccessFile interface
  ///
  /// \param[in] source a RandomAccessFile instance
  /// \return the table reader
  static Result<std::shared_ptr<Reader>> Open(
      const std::shared_ptr<io::RandomAccessFile>& source);

  /// \brief Open a Feather file from a RandomAccessFile interface
  /// with IPC Read options
  ///
  /// \param[in] source a RandomAccessFile instance
  /// \param[in] options IPC Read options
  /// \return the table reader
  static Result<std::shared_ptr<Reader>> Open(
      const std::shared_ptr<io::RandomAccessFile>& source, const IpcReadOptions& options);

  /// \brief Return the version number of the Feather file
  virtual int version() const = 0;

  virtual std::shared_ptr<Schema> schema() const = 0;

  /// \brief Read all columns from the file as an arrow::Table.
  ///
  /// \param[out] out the returned table
  /// \return Status
  ///
  /// This function is zero-copy if the file source supports zero-copy reads
  virtual Status Read(std::shared_ptr<Table>* out) = 0;

  /// \brief Read only the specified columns from the file as an arrow::Table.
  ///
  /// \param[in] indices the column indices to read
  /// \param[out] out the returned table
  /// \return Status
  ///
  /// This function is zero-copy if the file source supports zero-copy reads
  virtual Status Read(const std::vector<int>& indices, std::shared_ptr<Table>* out) = 0;

  /// \brief Read only the specified columns from the file as an arrow::Table.
  ///
  /// \param[in] names the column names to read
  /// \param[out] out the returned table
  /// \return Status
  ///
  /// This function is zero-copy if the file source supports zero-copy reads
  virtual Status Read(const std::vector<std::string>& names,
                      std::shared_ptr<Table>* out) = 0;
};

struct ARROW_EXPORT WriteProperties {
  static WriteProperties Defaults();

  static WriteProperties DefaultsV1() {
    WriteProperties props = Defaults();
    props.version = kFeatherV1Version;
    return props;
  }

  /// Feather file version number
  ///
  /// version 2: "Feather V1" Apache Arrow <= 0.16.0
  /// version 3: "Feather V2" Apache Arrow > 0.16.0
  int version = kFeatherV2Version;

  // Parameters for Feather V2 only

  /// Number of rows per intra-file chunk. Use smaller chunksize when you need
  /// faster random row access
  int64_t chunksize = 1LL << 16;

  /// Compression type to use. Only UNCOMPRESSED, LZ4_FRAME, and ZSTD are
  /// supported. The default compression returned by Defaults() is LZ4 if the
  /// project is built with support for it, otherwise
  /// UNCOMPRESSED. UNCOMPRESSED is set as the object default here so that if
  /// WriteProperties::Defaults() is not used, the default constructor for
  /// WriteProperties will work regardless of the options used to build the C++
  /// project.
  Compression::type compression = Compression::UNCOMPRESSED;

  /// Compressor-specific compression level
  int compression_level = ::arrow::util::kUseDefaultCompressionLevel;
};

ARROW_EXPORT
Status WriteTable(const Table& table, io::OutputStream* dst,
                  const WriteProperties& properties = WriteProperties::Defaults());

}  // namespace feather
}  // namespace ipc
}  // namespace arrow

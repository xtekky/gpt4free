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

// This API is EXPERIMENTAL.

#pragma once

#include <memory>
#include <string>

#include "arrow/dataset/file_base.h"
#include "arrow/dataset/type_fwd.h"
#include "arrow/dataset/visibility.h"
#include "arrow/io/type_fwd.h"
#include "arrow/ipc/type_fwd.h"
#include "arrow/result.h"

namespace arrow {
namespace dataset {

/// \addtogroup dataset-file-formats
///
/// @{

constexpr char kIpcTypeName[] = "ipc";

/// \brief A FileFormat implementation that reads from and writes to Ipc files
class ARROW_DS_EXPORT IpcFileFormat : public FileFormat {
 public:
  std::string type_name() const override { return kIpcTypeName; }

  IpcFileFormat();

  bool Equals(const FileFormat& other) const override {
    return type_name() == other.type_name();
  }

  Result<bool> IsSupported(const FileSource& source) const override;

  /// \brief Return the schema of the file if possible.
  Result<std::shared_ptr<Schema>> Inspect(const FileSource& source) const override;

  Result<RecordBatchGenerator> ScanBatchesAsync(
      const std::shared_ptr<ScanOptions>& options,
      const std::shared_ptr<FileFragment>& file) const override;

  Future<std::optional<int64_t>> CountRows(
      const std::shared_ptr<FileFragment>& file, compute::Expression predicate,
      const std::shared_ptr<ScanOptions>& options) override;

  Result<std::shared_ptr<FileWriter>> MakeWriter(
      std::shared_ptr<io::OutputStream> destination, std::shared_ptr<Schema> schema,
      std::shared_ptr<FileWriteOptions> options,
      fs::FileLocator destination_locator) const override;

  std::shared_ptr<FileWriteOptions> DefaultWriteOptions() override;
};

/// \brief Per-scan options for IPC fragments
class ARROW_DS_EXPORT IpcFragmentScanOptions : public FragmentScanOptions {
 public:
  std::string type_name() const override { return kIpcTypeName; }

  /// Options passed to the IPC file reader.
  /// included_fields, memory_pool, and use_threads are ignored.
  std::shared_ptr<ipc::IpcReadOptions> options;
  /// If present, the async scanner will enable I/O coalescing.
  /// This is ignored by the sync scanner.
  std::shared_ptr<io::CacheOptions> cache_options;
};

class ARROW_DS_EXPORT IpcFileWriteOptions : public FileWriteOptions {
 public:
  /// Options passed to ipc::MakeFileWriter. use_threads is ignored
  std::shared_ptr<ipc::IpcWriteOptions> options;

  /// custom_metadata written to the file's footer
  std::shared_ptr<const KeyValueMetadata> metadata;

 protected:
  explicit IpcFileWriteOptions(std::shared_ptr<FileFormat> format)
      : FileWriteOptions(std::move(format)) {}

  friend class IpcFileFormat;
};

class ARROW_DS_EXPORT IpcFileWriter : public FileWriter {
 public:
  Status Write(const std::shared_ptr<RecordBatch>& batch) override;

 private:
  IpcFileWriter(std::shared_ptr<io::OutputStream> destination,
                std::shared_ptr<ipc::RecordBatchWriter> writer,
                std::shared_ptr<Schema> schema,
                std::shared_ptr<IpcFileWriteOptions> options,
                fs::FileLocator destination_locator);

  Future<> FinishInternal() override;

  std::shared_ptr<io::OutputStream> destination_;
  std::shared_ptr<ipc::RecordBatchWriter> batch_writer_;

  friend class IpcFileFormat;
};

/// @}

}  // namespace dataset
}  // namespace arrow

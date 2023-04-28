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
#include <string>

#include "arrow/csv/options.h"
#include "arrow/dataset/dataset.h"
#include "arrow/dataset/file_base.h"
#include "arrow/dataset/type_fwd.h"
#include "arrow/dataset/visibility.h"
#include "arrow/ipc/type_fwd.h"
#include "arrow/status.h"
#include "arrow/util/compression.h"

namespace arrow {
namespace dataset {

constexpr char kCsvTypeName[] = "csv";

/// \addtogroup dataset-file-formats
///
/// @{

/// \brief A FileFormat implementation that reads from and writes to Csv files
class ARROW_DS_EXPORT CsvFileFormat : public FileFormat {
 public:
  // TODO(ARROW-18328) Remove this, moved to CsvFragmentScanOptions
  /// Options affecting the parsing of CSV files
  csv::ParseOptions parse_options = csv::ParseOptions::Defaults();

  CsvFileFormat();

  std::string type_name() const override { return kCsvTypeName; }

  bool Equals(const FileFormat& other) const override;

  Result<bool> IsSupported(const FileSource& source) const override;

  /// \brief Return the schema of the file if possible.
  Result<std::shared_ptr<Schema>> Inspect(const FileSource& source) const override;

  Future<std::shared_ptr<FragmentScanner>> BeginScan(
      const FragmentScanRequest& request, const InspectedFragment& inspected_fragment,
      const FragmentScanOptions* format_options,
      compute::ExecContext* exec_context) const override;

  Result<RecordBatchGenerator> ScanBatchesAsync(
      const std::shared_ptr<ScanOptions>& scan_options,
      const std::shared_ptr<FileFragment>& file) const override;

  Future<std::shared_ptr<InspectedFragment>> InspectFragment(
      const FileSource& source, const FragmentScanOptions* format_options,
      compute::ExecContext* exec_context) const override;

  Future<std::optional<int64_t>> CountRows(
      const std::shared_ptr<FileFragment>& file, compute::Expression predicate,
      const std::shared_ptr<ScanOptions>& options) override;

  Result<std::shared_ptr<FileWriter>> MakeWriter(
      std::shared_ptr<io::OutputStream> destination, std::shared_ptr<Schema> schema,
      std::shared_ptr<FileWriteOptions> options,
      fs::FileLocator destination_locator) const override;

  std::shared_ptr<FileWriteOptions> DefaultWriteOptions() override;
};

/// \brief Per-scan options for CSV fragments
struct ARROW_DS_EXPORT CsvFragmentScanOptions : public FragmentScanOptions {
  std::string type_name() const override { return kCsvTypeName; }

  using StreamWrapFunc = std::function<Result<std::shared_ptr<io::InputStream>>(
      std::shared_ptr<io::InputStream>)>;

  /// CSV conversion options
  csv::ConvertOptions convert_options = csv::ConvertOptions::Defaults();

  /// CSV reading options
  ///
  /// Note that use_threads is always ignored.
  csv::ReadOptions read_options = csv::ReadOptions::Defaults();

  /// CSV parse options
  csv::ParseOptions parse_options = csv::ParseOptions::Defaults();

  /// Optional stream wrapping function
  ///
  /// If defined, all open dataset file fragments will be passed
  /// through this function.  One possible use case is to transparently
  /// transcode all input files from a given character set to utf8.
  StreamWrapFunc stream_transform_func{};
};

class ARROW_DS_EXPORT CsvFileWriteOptions : public FileWriteOptions {
 public:
  /// Options passed to csv::MakeCSVWriter.
  std::shared_ptr<csv::WriteOptions> write_options;

 protected:
  explicit CsvFileWriteOptions(std::shared_ptr<FileFormat> format)
      : FileWriteOptions(std::move(format)) {}

  friend class CsvFileFormat;
};

class ARROW_DS_EXPORT CsvFileWriter : public FileWriter {
 public:
  Status Write(const std::shared_ptr<RecordBatch>& batch) override;

 private:
  CsvFileWriter(std::shared_ptr<io::OutputStream> destination,
                std::shared_ptr<ipc::RecordBatchWriter> writer,
                std::shared_ptr<Schema> schema,
                std::shared_ptr<CsvFileWriteOptions> options,
                fs::FileLocator destination_locator);

  Future<> FinishInternal() override;

  std::shared_ptr<io::OutputStream> destination_;
  std::shared_ptr<ipc::RecordBatchWriter> batch_writer_;

  friend class CsvFileFormat;
};

/// @}

}  // namespace dataset
}  // namespace arrow

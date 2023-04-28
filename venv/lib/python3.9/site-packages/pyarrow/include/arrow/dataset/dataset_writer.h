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

#include <string>

#include "arrow/dataset/file_base.h"
#include "arrow/record_batch.h"
#include "arrow/status.h"
#include "arrow/util/async_util.h"
#include "arrow/util/future.h"

namespace arrow {
namespace dataset {
namespace internal {

// This lines up with our other defaults in the scanner and execution plan
constexpr uint64_t kDefaultDatasetWriterMaxRowsQueued = 8 * 1024 * 1024;

/// \brief Utility class that manages a set of writers to different paths
///
/// Writers may be closed and reopened (and a new file created) based on the dataset
/// write options (for example, max_rows_per_file or max_open_files)
///
/// The dataset writer enforces its own back pressure based on the # of rows (as opposed
/// to # of batches which is how it is typically enforced elsewhere) and # of files.
class ARROW_DS_EXPORT DatasetWriter {
 public:
  /// \brief Create a dataset writer
  ///
  /// Will fail if basename_template is invalid or if there is existing data and
  /// existing_data_behavior is kError
  ///
  /// \param write_options options to control how the data should be written
  /// \param max_rows_queued max # of rows allowed to be queued before the dataset_writer
  ///                        will ask for backpressure
  static Result<std::unique_ptr<DatasetWriter>> Make(
      FileSystemDatasetWriteOptions write_options, util::AsyncTaskScheduler* scheduler,
      std::function<void()> pause_callback, std::function<void()> resume_callback,
      std::function<void()> finish_callback,
      uint64_t max_rows_queued = kDefaultDatasetWriterMaxRowsQueued);

  ~DatasetWriter();

  /// \brief Write a batch to the dataset
  /// \param[in] batch The batch to write
  /// \param[in] directory The directory to write to
  ///
  /// Note: The written filename will be {directory}/{filename_factory(i)} where i is a
  /// counter controlled by `max_open_files` and `max_rows_per_file`
  ///
  /// If multiple WriteRecordBatch calls arrive with the same `directory` then the batches
  /// may be written to the same file.
  ///
  /// The returned future will be marked finished when the record batch has been queued
  /// to be written.  If the returned future is unfinished then this indicates the dataset
  /// writer's queue is full and the data provider should pause.
  ///
  /// This method is NOT async reentrant.  The returned future will only be unfinished
  /// if back pressure needs to be applied.  Async reentrancy is not necessary for
  /// concurrent writes to happen.  Calling this method again before the previous future
  /// completes will not just violate max_rows_queued but likely lead to race conditions.
  ///
  /// One thing to note is that the ordering of your data can affect your maximum
  /// potential parallelism.  If this seems odd then consider a dataset where the first
  /// 1000 batches go to the same directory and then the 1001st batch goes to a different
  /// directory.  The only way to get two parallel writes immediately would be to queue
  /// all 1000 pending writes to the first directory.
  void WriteRecordBatch(std::shared_ptr<RecordBatch> batch, const std::string& directory,
                        const std::string& prefix = "");

  /// Finish all pending writes and close any open files
  void Finish();

 protected:
  DatasetWriter(FileSystemDatasetWriteOptions write_options,
                util::AsyncTaskScheduler* scheduler, std::function<void()> pause_callback,
                std::function<void()> resume_callback,
                std::function<void()> finish_callback,
                uint64_t max_rows_queued = kDefaultDatasetWriterMaxRowsQueued);

  class DatasetWriterImpl;
  std::unique_ptr<DatasetWriterImpl> impl_;
};

}  // namespace internal
}  // namespace dataset
}  // namespace arrow

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

/// Logic for automatically determining the structure of multi-file
/// dataset with possible partitioning according to available
/// partitioning

// This API is EXPERIMENTAL.

#pragma once

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "arrow/dataset/partition.h"
#include "arrow/dataset/type_fwd.h"
#include "arrow/dataset/visibility.h"
#include "arrow/filesystem/type_fwd.h"
#include "arrow/result.h"
#include "arrow/util/macros.h"

namespace arrow {
namespace dataset {

/// \defgroup dataset-discovery Discovery API
///
/// @{

struct InspectOptions {
  /// See `fragments` property.
  static constexpr int kInspectAllFragments = -1;

  /// Indicate how many fragments should be inspected to infer the unified dataset
  /// schema. Limiting the number of fragments accessed improves the latency of
  /// the discovery process when dealing with a high number of fragments and/or
  /// high latency file systems.
  ///
  /// The default value of `1` inspects the schema of the first (in no particular
  /// order) fragment only. If the dataset has a uniform schema for all fragments,
  /// this default is the optimal value. In order to inspect all fragments and
  /// robustly unify their potentially varying schemas, set this option to
  /// `kInspectAllFragments`. A value of `0` disables inspection of fragments
  /// altogether so only the partitioning schema will be inspected.
  int fragments = 1;
};

struct FinishOptions {
  /// Finalize the dataset with this given schema. If the schema is not
  /// provided, infer the schema via the Inspect, see the `inspect_options`
  /// property.
  std::shared_ptr<Schema> schema = NULLPTR;

  /// If the schema is not provided, it will be discovered by passing the
  /// following options to `DatasetDiscovery::Inspect`.
  InspectOptions inspect_options{};

  /// Indicate if the given Schema (when specified), should be validated against
  /// the fragments' schemas. `inspect_options` will control how many fragments
  /// are checked.
  bool validate_fragments = false;
};

/// \brief DatasetFactory provides a way to inspect/discover a Dataset's expected
/// schema before materializing said Dataset.
class ARROW_DS_EXPORT DatasetFactory {
 public:
  /// \brief Get the schemas of the Fragments and Partitioning.
  virtual Result<std::vector<std::shared_ptr<Schema>>> InspectSchemas(
      InspectOptions options) = 0;

  /// \brief Get unified schema for the resulting Dataset.
  Result<std::shared_ptr<Schema>> Inspect(InspectOptions options = {});

  /// \brief Create a Dataset
  Result<std::shared_ptr<Dataset>> Finish();
  /// \brief Create a Dataset with the given schema (see \a InspectOptions::schema)
  Result<std::shared_ptr<Dataset>> Finish(std::shared_ptr<Schema> schema);
  /// \brief Create a Dataset with the given options
  virtual Result<std::shared_ptr<Dataset>> Finish(FinishOptions options) = 0;

  /// \brief Optional root partition for the resulting Dataset.
  const compute::Expression& root_partition() const { return root_partition_; }
  /// \brief Set the root partition for the resulting Dataset.
  Status SetRootPartition(compute::Expression partition) {
    root_partition_ = std::move(partition);
    return Status::OK();
  }

  virtual ~DatasetFactory() = default;

 protected:
  DatasetFactory();

  compute::Expression root_partition_;
};

/// @}

/// \brief DatasetFactory provides a way to inspect/discover a Dataset's
/// expected schema before materialization.
/// \ingroup dataset-implementations
class ARROW_DS_EXPORT UnionDatasetFactory : public DatasetFactory {
 public:
  static Result<std::shared_ptr<DatasetFactory>> Make(
      std::vector<std::shared_ptr<DatasetFactory>> factories);

  /// \brief Return the list of child DatasetFactory
  const std::vector<std::shared_ptr<DatasetFactory>>& factories() const {
    return factories_;
  }

  /// \brief Get the schemas of the Datasets.
  ///
  /// Instead of applying options globally, it applies at each child factory.
  /// This will not respect `options.fragments` exactly, but will respect the
  /// spirit of peeking the first fragments or all of them.
  Result<std::vector<std::shared_ptr<Schema>>> InspectSchemas(
      InspectOptions options) override;

  /// \brief Create a Dataset.
  Result<std::shared_ptr<Dataset>> Finish(FinishOptions options) override;

 protected:
  explicit UnionDatasetFactory(std::vector<std::shared_ptr<DatasetFactory>> factories);

  std::vector<std::shared_ptr<DatasetFactory>> factories_;
};

/// \ingroup dataset-filesystem
struct FileSystemFactoryOptions {
  /// Either an explicit Partitioning or a PartitioningFactory to discover one.
  ///
  /// If a factory is provided, it will be used to infer a schema for partition fields
  /// based on file and directory paths then construct a Partitioning. The default
  /// is a Partitioning which will yield no partition information.
  ///
  /// The (explicit or discovered) partitioning will be applied to discovered files
  /// and the resulting partition information embedded in the Dataset.
  PartitioningOrFactory partitioning{Partitioning::Default()};

  /// For the purposes of applying the partitioning, paths will be stripped
  /// of the partition_base_dir. Files not matching the partition_base_dir
  /// prefix will be skipped for partition discovery. The ignored files will still
  /// be part of the Dataset, but will not have partition information.
  ///
  /// Example:
  /// partition_base_dir = "/dataset";
  ///
  /// - "/dataset/US/sales.csv" -> "US/sales.csv" will be given to the partitioning
  ///
  /// - "/home/john/late_sales.csv" -> Will be ignored for partition discovery.
  ///
  /// This is useful for partitioning which parses directory when ordering
  /// is important, e.g. DirectoryPartitioning.
  std::string partition_base_dir;

  /// Invalid files (via selector or explicitly) will be excluded by checking
  /// with the FileFormat::IsSupported method.  This will incur IO for each files
  /// in a serial and single threaded fashion. Disabling this feature will skip the
  /// IO, but unsupported files may be present in the Dataset
  /// (resulting in an error at scan time).
  bool exclude_invalid_files = false;

  /// When discovering from a Selector (and not from an explicit file list), ignore
  /// files and directories matching any of these prefixes.
  ///
  /// Example (with selector = "/dataset/**"):
  /// selector_ignore_prefixes = {"_", ".DS_STORE" };
  ///
  /// - "/dataset/data.csv" -> not ignored
  /// - "/dataset/_metadata" -> ignored
  /// - "/dataset/.DS_STORE" -> ignored
  /// - "/dataset/_hidden/dat" -> ignored
  /// - "/dataset/nested/.DS_STORE" -> ignored
  std::vector<std::string> selector_ignore_prefixes = {
      ".",
      "_",
  };
};

/// \brief FileSystemDatasetFactory creates a Dataset from a vector of
/// fs::FileInfo or a fs::FileSelector.
/// \ingroup dataset-filesystem
class ARROW_DS_EXPORT FileSystemDatasetFactory : public DatasetFactory {
 public:
  /// \brief Build a FileSystemDatasetFactory from an explicit list of
  /// paths.
  ///
  /// \param[in] filesystem passed to FileSystemDataset
  /// \param[in] paths passed to FileSystemDataset
  /// \param[in] format passed to FileSystemDataset
  /// \param[in] options see FileSystemFactoryOptions for more information.
  static Result<std::shared_ptr<DatasetFactory>> Make(
      std::shared_ptr<fs::FileSystem> filesystem, const std::vector<std::string>& paths,
      std::shared_ptr<FileFormat> format, FileSystemFactoryOptions options);

  /// \brief Build a FileSystemDatasetFactory from a fs::FileSelector.
  ///
  /// The selector will expand to a vector of FileInfo. The expansion/crawling
  /// is performed in this function call. Thus, the finalized Dataset is
  /// working with a snapshot of the filesystem.
  //
  /// If options.partition_base_dir is not provided, it will be overwritten
  /// with selector.base_dir.
  ///
  /// \param[in] filesystem passed to FileSystemDataset
  /// \param[in] selector used to crawl and search files
  /// \param[in] format passed to FileSystemDataset
  /// \param[in] options see FileSystemFactoryOptions for more information.
  static Result<std::shared_ptr<DatasetFactory>> Make(
      std::shared_ptr<fs::FileSystem> filesystem, fs::FileSelector selector,
      std::shared_ptr<FileFormat> format, FileSystemFactoryOptions options);

  /// \brief Build a FileSystemDatasetFactory from an uri including filesystem
  /// information.
  ///
  /// \param[in] uri passed to FileSystemDataset
  /// \param[in] format passed to FileSystemDataset
  /// \param[in] options see FileSystemFactoryOptions for more information.
  static Result<std::shared_ptr<DatasetFactory>> Make(std::string uri,
                                                      std::shared_ptr<FileFormat> format,
                                                      FileSystemFactoryOptions options);

  /// \brief Build a FileSystemDatasetFactory from an explicit list of
  /// file information.
  ///
  /// \param[in] filesystem passed to FileSystemDataset
  /// \param[in] files passed to FileSystemDataset
  /// \param[in] format passed to FileSystemDataset
  /// \param[in] options see FileSystemFactoryOptions for more information.
  static Result<std::shared_ptr<DatasetFactory>> Make(
      std::shared_ptr<fs::FileSystem> filesystem, const std::vector<fs::FileInfo>& files,
      std::shared_ptr<FileFormat> format, FileSystemFactoryOptions options);

  Result<std::vector<std::shared_ptr<Schema>>> InspectSchemas(
      InspectOptions options) override;

  Result<std::shared_ptr<Dataset>> Finish(FinishOptions options) override;

 protected:
  FileSystemDatasetFactory(std::vector<fs::FileInfo> files,
                           std::shared_ptr<fs::FileSystem> filesystem,
                           std::shared_ptr<FileFormat> format,
                           FileSystemFactoryOptions options);

  Result<std::shared_ptr<Schema>> PartitionSchema();

  std::vector<fs::FileInfo> files_;
  std::shared_ptr<fs::FileSystem> fs_;
  std::shared_ptr<FileFormat> format_;
  FileSystemFactoryOptions options_;
};

}  // namespace dataset
}  // namespace arrow

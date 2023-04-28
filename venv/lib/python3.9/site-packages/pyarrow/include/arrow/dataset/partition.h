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

#include <functional>
#include <iosfwd>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "arrow/compute/exec/expression.h"
#include "arrow/dataset/type_fwd.h"
#include "arrow/dataset/visibility.h"
#include "arrow/util/compare.h"

namespace arrow {

namespace dataset {

constexpr char kFilenamePartitionSep = '_';

struct ARROW_DS_EXPORT PartitionPathFormat {
  std::string directory, filename;
};

// ----------------------------------------------------------------------
// Partitioning

/// \defgroup dataset-partitioning Partitioning API
///
/// @{

/// \brief Interface for parsing partition expressions from string partition
/// identifiers.
///
/// For example, the identifier "foo=5" might be parsed to an equality expression
/// between the "foo" field and the value 5.
///
/// Some partitionings may store the field names in a metadata
/// store instead of in file paths, for example
/// dataset_root/2009/11/... could be used when the partition fields
/// are "year" and "month"
///
/// Paths are consumed from left to right. Paths must be relative to
/// the root of a partition; path prefixes must be removed before passing
/// the path to a partitioning for parsing.
class ARROW_DS_EXPORT Partitioning : public util::EqualityComparable<Partitioning> {
 public:
  virtual ~Partitioning() = default;

  /// \brief The name identifying the kind of partitioning
  virtual std::string type_name() const = 0;

  //// \brief Return whether the partitionings are equal
  virtual bool Equals(const Partitioning& other) const {
    return schema_->Equals(other.schema_, /*check_metadata=*/false);
  }

  /// \brief If the input batch shares any fields with this partitioning,
  /// produce sub-batches which satisfy mutually exclusive Expressions.
  struct PartitionedBatches {
    RecordBatchVector batches;
    std::vector<compute::Expression> expressions;
  };
  virtual Result<PartitionedBatches> Partition(
      const std::shared_ptr<RecordBatch>& batch) const = 0;

  /// \brief Parse a path into a partition expression
  virtual Result<compute::Expression> Parse(const std::string& path) const = 0;

  virtual Result<PartitionPathFormat> Format(const compute::Expression& expr) const = 0;

  /// \brief A default Partitioning which always yields scalar(true)
  static std::shared_ptr<Partitioning> Default();

  /// \brief The partition schema.
  const std::shared_ptr<Schema>& schema() const { return schema_; }

 protected:
  explicit Partitioning(std::shared_ptr<Schema> schema) : schema_(std::move(schema)) {}

  std::shared_ptr<Schema> schema_;
};

/// \brief The encoding of partition segments.
enum class SegmentEncoding : int8_t {
  /// No encoding.
  None = 0,
  /// Segment values are URL-encoded.
  Uri = 1,
};

ARROW_DS_EXPORT
std::ostream& operator<<(std::ostream& os, SegmentEncoding segment_encoding);

/// \brief Options for key-value based partitioning (hive/directory).
struct ARROW_DS_EXPORT KeyValuePartitioningOptions {
  /// After splitting a path into components, decode the path components
  /// before parsing according to this scheme.
  SegmentEncoding segment_encoding = SegmentEncoding::Uri;
};

/// \brief Options for inferring a partitioning.
struct ARROW_DS_EXPORT PartitioningFactoryOptions {
  /// When inferring a schema for partition fields, yield dictionary encoded types
  /// instead of plain. This can be more efficient when materializing virtual
  /// columns, and Expressions parsed by the finished Partitioning will include
  /// dictionaries of all unique inspected values for each field.
  bool infer_dictionary = false;
  /// Optionally, an expected schema can be provided, in which case inference
  /// will only check discovered fields against the schema and update internal
  /// state (such as dictionaries).
  std::shared_ptr<Schema> schema;
  /// After splitting a path into components, decode the path components
  /// before parsing according to this scheme.
  SegmentEncoding segment_encoding = SegmentEncoding::Uri;

  KeyValuePartitioningOptions AsPartitioningOptions() const;
};

/// \brief Options for inferring a hive-style partitioning.
struct ARROW_DS_EXPORT HivePartitioningFactoryOptions : PartitioningFactoryOptions {
  /// The hive partitioning scheme maps null to a hard coded fallback string.
  std::string null_fallback;

  HivePartitioningOptions AsHivePartitioningOptions() const;
};

/// \brief PartitioningFactory provides creation of a partitioning  when the
/// specific schema must be inferred from available paths (no explicit schema is known).
class ARROW_DS_EXPORT PartitioningFactory {
 public:
  virtual ~PartitioningFactory() = default;

  /// \brief The name identifying the kind of partitioning
  virtual std::string type_name() const = 0;

  /// Get the schema for the resulting Partitioning.
  /// This may reset internal state, for example dictionaries of unique representations.
  virtual Result<std::shared_ptr<Schema>> Inspect(
      const std::vector<std::string>& paths) = 0;

  /// Create a partitioning using the provided schema
  /// (fields may be dropped).
  virtual Result<std::shared_ptr<Partitioning>> Finish(
      const std::shared_ptr<Schema>& schema) const = 0;
};

/// \brief Subclass for the common case of a partitioning which yields an equality
/// expression for each segment
class ARROW_DS_EXPORT KeyValuePartitioning : public Partitioning {
 public:
  /// An unconverted equality expression consisting of a field name and the representation
  /// of a scalar value
  struct Key {
    std::string name;
    std::optional<std::string> value;
  };

  Result<PartitionedBatches> Partition(
      const std::shared_ptr<RecordBatch>& batch) const override;

  Result<compute::Expression> Parse(const std::string& path) const override;

  Result<PartitionPathFormat> Format(const compute::Expression& expr) const override;

  const ArrayVector& dictionaries() const { return dictionaries_; }

  bool Equals(const Partitioning& other) const override;

 protected:
  KeyValuePartitioning(std::shared_ptr<Schema> schema, ArrayVector dictionaries,
                       KeyValuePartitioningOptions options)
      : Partitioning(std::move(schema)),
        dictionaries_(std::move(dictionaries)),
        options_(options) {
    if (dictionaries_.empty()) {
      dictionaries_.resize(schema_->num_fields());
    }
  }

  virtual Result<std::vector<Key>> ParseKeys(const std::string& path) const = 0;

  virtual Result<PartitionPathFormat> FormatValues(const ScalarVector& values) const = 0;

  /// Convert a Key to a full expression.
  Result<compute::Expression> ConvertKey(const Key& key) const;

  Result<std::vector<std::string>> FormatPartitionSegments(
      const ScalarVector& values) const;
  Result<std::vector<Key>> ParsePartitionSegments(
      const std::vector<std::string>& segments) const;

  ArrayVector dictionaries_;
  KeyValuePartitioningOptions options_;
};

/// \brief DirectoryPartitioning parses one segment of a path for each field in its
/// schema. All fields are required, so paths passed to DirectoryPartitioning::Parse
/// must contain segments for each field.
///
/// For example given schema<year:int16, month:int8> the path "/2009/11" would be
/// parsed to ("year"_ == 2009 and "month"_ == 11)
class ARROW_DS_EXPORT DirectoryPartitioning : public KeyValuePartitioning {
 public:
  /// If a field in schema is of dictionary type, the corresponding element of
  /// dictionaries must be contain the dictionary of values for that field.
  explicit DirectoryPartitioning(std::shared_ptr<Schema> schema,
                                 ArrayVector dictionaries = {},
                                 KeyValuePartitioningOptions options = {});

  std::string type_name() const override { return "directory"; }

  bool Equals(const Partitioning& other) const override;

  /// \brief Create a factory for a directory partitioning.
  ///
  /// \param[in] field_names The names for the partition fields. Types will be
  ///     inferred.
  static std::shared_ptr<PartitioningFactory> MakeFactory(
      std::vector<std::string> field_names, PartitioningFactoryOptions = {});

 private:
  Result<std::vector<Key>> ParseKeys(const std::string& path) const override;

  Result<PartitionPathFormat> FormatValues(const ScalarVector& values) const override;
};

/// \brief The default fallback used for null values in a Hive-style partitioning.
static constexpr char kDefaultHiveNullFallback[] = "__HIVE_DEFAULT_PARTITION__";

struct ARROW_DS_EXPORT HivePartitioningOptions : public KeyValuePartitioningOptions {
  std::string null_fallback = kDefaultHiveNullFallback;

  static HivePartitioningOptions DefaultsWithNullFallback(std::string fallback) {
    HivePartitioningOptions options;
    options.null_fallback = std::move(fallback);
    return options;
  }
};

/// \brief Multi-level, directory based partitioning
/// originating from Apache Hive with all data files stored in the
/// leaf directories. Data is partitioned by static values of a
/// particular column in the schema. Partition keys are represented in
/// the form $key=$value in directory names.
/// Field order is ignored, as are missing or unrecognized field names.
///
/// For example given schema<year:int16, month:int8, day:int8> the path
/// "/day=321/ignored=3.4/year=2009" parses to ("year"_ == 2009 and "day"_ == 321)
class ARROW_DS_EXPORT HivePartitioning : public KeyValuePartitioning {
 public:
  /// If a field in schema is of dictionary type, the corresponding element of
  /// dictionaries must be contain the dictionary of values for that field.
  explicit HivePartitioning(std::shared_ptr<Schema> schema, ArrayVector dictionaries = {},
                            std::string null_fallback = kDefaultHiveNullFallback)
      : KeyValuePartitioning(std::move(schema), std::move(dictionaries),
                             KeyValuePartitioningOptions()),
        hive_options_(
            HivePartitioningOptions::DefaultsWithNullFallback(std::move(null_fallback))) {
  }

  explicit HivePartitioning(std::shared_ptr<Schema> schema, ArrayVector dictionaries,
                            HivePartitioningOptions options)
      : KeyValuePartitioning(std::move(schema), std::move(dictionaries), options),
        hive_options_(options) {}

  std::string type_name() const override { return "hive"; }
  std::string null_fallback() const { return hive_options_.null_fallback; }
  const HivePartitioningOptions& options() const { return hive_options_; }

  static Result<std::optional<Key>> ParseKey(const std::string& segment,
                                             const HivePartitioningOptions& options);

  bool Equals(const Partitioning& other) const override;

  /// \brief Create a factory for a hive partitioning.
  static std::shared_ptr<PartitioningFactory> MakeFactory(
      HivePartitioningFactoryOptions = {});

 private:
  const HivePartitioningOptions hive_options_;
  Result<std::vector<Key>> ParseKeys(const std::string& path) const override;

  Result<PartitionPathFormat> FormatValues(const ScalarVector& values) const override;
};

/// \brief Implementation provided by lambda or other callable
class ARROW_DS_EXPORT FunctionPartitioning : public Partitioning {
 public:
  using ParseImpl = std::function<Result<compute::Expression>(const std::string&)>;

  using FormatImpl =
      std::function<Result<PartitionPathFormat>(const compute::Expression&)>;

  FunctionPartitioning(std::shared_ptr<Schema> schema, ParseImpl parse_impl,
                       FormatImpl format_impl = NULLPTR, std::string name = "function")
      : Partitioning(std::move(schema)),
        parse_impl_(std::move(parse_impl)),
        format_impl_(std::move(format_impl)),
        name_(std::move(name)) {}

  std::string type_name() const override { return name_; }

  bool Equals(const Partitioning& other) const override { return false; }

  Result<compute::Expression> Parse(const std::string& path) const override {
    return parse_impl_(path);
  }

  Result<PartitionPathFormat> Format(const compute::Expression& expr) const override {
    if (format_impl_) {
      return format_impl_(expr);
    }
    return Status::NotImplemented("formatting paths from ", type_name(), " Partitioning");
  }

  Result<PartitionedBatches> Partition(
      const std::shared_ptr<RecordBatch>& batch) const override {
    return Status::NotImplemented("partitioning batches from ", type_name(),
                                  " Partitioning");
  }

 private:
  ParseImpl parse_impl_;
  FormatImpl format_impl_;
  std::string name_;
};

class ARROW_DS_EXPORT FilenamePartitioning : public KeyValuePartitioning {
 public:
  /// \brief Construct a FilenamePartitioning from its components.
  ///
  /// If a field in schema is of dictionary type, the corresponding element of
  /// dictionaries must be contain the dictionary of values for that field.
  explicit FilenamePartitioning(std::shared_ptr<Schema> schema,
                                ArrayVector dictionaries = {},
                                KeyValuePartitioningOptions options = {});

  std::string type_name() const override { return "filename"; }

  /// \brief Create a factory for a filename partitioning.
  ///
  /// \param[in] field_names The names for the partition fields. Types will be
  ///     inferred.
  static std::shared_ptr<PartitioningFactory> MakeFactory(
      std::vector<std::string> field_names, PartitioningFactoryOptions = {});

  bool Equals(const Partitioning& other) const override;

 private:
  Result<std::vector<Key>> ParseKeys(const std::string& path) const override;

  Result<PartitionPathFormat> FormatValues(const ScalarVector& values) const override;
};

ARROW_DS_EXPORT std::string StripPrefix(const std::string& path,
                                        const std::string& prefix);

/// \brief Extracts the directory and filename and removes the prefix of a path
///
/// e.g., `StripPrefixAndFilename("/data/year=2019/c.txt", "/data") ->
/// {"year=2019","c.txt"}`
ARROW_DS_EXPORT std::string StripPrefixAndFilename(const std::string& path,
                                                   const std::string& prefix);

/// \brief Vector version of StripPrefixAndFilename.
ARROW_DS_EXPORT std::vector<std::string> StripPrefixAndFilename(
    const std::vector<std::string>& paths, const std::string& prefix);

/// \brief Vector version of StripPrefixAndFilename.
ARROW_DS_EXPORT std::vector<std::string> StripPrefixAndFilename(
    const std::vector<fs::FileInfo>& files, const std::string& prefix);

/// \brief Either a Partitioning or a PartitioningFactory
class ARROW_DS_EXPORT PartitioningOrFactory {
 public:
  explicit PartitioningOrFactory(std::shared_ptr<Partitioning> partitioning)
      : partitioning_(std::move(partitioning)) {}

  explicit PartitioningOrFactory(std::shared_ptr<PartitioningFactory> factory)
      : factory_(std::move(factory)) {}

  PartitioningOrFactory& operator=(std::shared_ptr<Partitioning> partitioning) {
    return *this = PartitioningOrFactory(std::move(partitioning));
  }

  PartitioningOrFactory& operator=(std::shared_ptr<PartitioningFactory> factory) {
    return *this = PartitioningOrFactory(std::move(factory));
  }

  /// \brief The partitioning (if given).
  const std::shared_ptr<Partitioning>& partitioning() const { return partitioning_; }

  /// \brief The partition factory (if given).
  const std::shared_ptr<PartitioningFactory>& factory() const { return factory_; }

  /// \brief Get the partition schema, inferring it with the given factory if needed.
  Result<std::shared_ptr<Schema>> GetOrInferSchema(const std::vector<std::string>& paths);

 private:
  std::shared_ptr<PartitioningFactory> factory_;
  std::shared_ptr<Partitioning> partitioning_;
};

/// @}

}  // namespace dataset
}  // namespace arrow

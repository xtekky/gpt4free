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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "arrow/compute/exec/expression.h"
#include "arrow/compute/exec/options.h"
#include "arrow/compute/type_fwd.h"
#include "arrow/dataset/dataset.h"
#include "arrow/dataset/projector.h"
#include "arrow/dataset/type_fwd.h"
#include "arrow/dataset/visibility.h"
#include "arrow/io/interfaces.h"
#include "arrow/memory_pool.h"
#include "arrow/type_fwd.h"
#include "arrow/util/async_generator.h"
#include "arrow/util/iterator.h"
#include "arrow/util/thread_pool.h"
#include "arrow/util/type_fwd.h"

namespace arrow {

using RecordBatchGenerator = std::function<Future<std::shared_ptr<RecordBatch>>()>;

namespace dataset {

/// \defgroup dataset-scanning Scanning API
///
/// @{

constexpr int64_t kDefaultBatchSize = 1 << 17;  // 128Ki rows
// This will yield 64 batches ~ 8Mi rows
constexpr int32_t kDefaultBatchReadahead = 16;
constexpr int32_t kDefaultFragmentReadahead = 4;
constexpr int32_t kDefaultBytesReadahead = 1 << 25;  // 32MiB

/// Scan-specific options, which can be changed between scans of the same dataset.
struct ARROW_DS_EXPORT ScanOptions {
  /// A row filter (which will be pushed down to partitioning/reading if supported).
  compute::Expression filter = compute::literal(true);
  /// A projection expression (which can add/remove/rename columns).
  compute::Expression projection;

  /// Schema with which batches will be read from fragments. This is also known as the
  /// "reader schema" it will be used (for example) in constructing CSV file readers to
  /// identify column types for parsing. Usually only a subset of its fields (see
  /// MaterializedFields) will be materialized during a scan.
  std::shared_ptr<Schema> dataset_schema;

  /// Schema of projected record batches. This is independent of dataset_schema as its
  /// fields are derived from the projection. For example, let
  ///
  ///   dataset_schema = {"a": int32, "b": int32, "id": utf8}
  ///   projection = project({equal(field_ref("a"), field_ref("b"))}, {"a_plus_b"})
  ///
  /// (no filter specified). In this case, the projected_schema would be
  ///
  ///   {"a_plus_b": int32}
  std::shared_ptr<Schema> projected_schema;

  /// Maximum row count for scanned batches.
  int64_t batch_size = kDefaultBatchSize;

  /// How many batches to read ahead within a fragment.
  ///
  /// Set to 0 to disable batch readahead
  ///
  /// Note: May not be supported by all formats
  /// Note: Will be ignored if use_threads is set to false
  int32_t batch_readahead = kDefaultBatchReadahead;

  /// How many files to read ahead
  ///
  /// Set to 0 to disable fragment readahead
  ///
  /// Note: May not be enforced by all scanners
  /// Note: Will be ignored if use_threads is set to false
  int32_t fragment_readahead = kDefaultFragmentReadahead;

  /// A pool from which materialized and scanned arrays will be allocated.
  MemoryPool* pool = arrow::default_memory_pool();

  /// IOContext for any IO tasks
  ///
  /// Note: The IOContext executor will be ignored if use_threads is set to false
  io::IOContext io_context;

  /// If true the scanner will scan in parallel
  ///
  /// Note: If true, this will use threads from both the cpu_executor and the
  /// io_context.executor
  /// Note: This  must be true in order for any readahead to happen
  bool use_threads = false;

  /// Fragment-specific scan options.
  std::shared_ptr<FragmentScanOptions> fragment_scan_options;

  /// Return a vector of FieldRefs that require materialization.
  ///
  /// This is usually the union of the fields referenced in the projection and the
  /// filter expression. Examples:
  ///
  /// - `SELECT a, b WHERE a < 2 && c > 1` => ["a", "b", "a", "c"]
  /// - `SELECT a + b < 3 WHERE a > 1` => ["a", "b", "a"]
  ///
  /// This is needed for expression where a field may not be directly
  /// used in the final projection but is still required to evaluate the
  /// expression.
  ///
  /// This is used by Fragment implementations to apply the column
  /// sub-selection optimization.
  std::vector<FieldRef> MaterializedFields() const;

  /// Parameters which control when the plan should pause for a slow consumer
  compute::BackpressureOptions backpressure =
      compute::BackpressureOptions::DefaultBackpressure();
};

/// Scan-specific options, which can be changed between scans of the same dataset.
///
/// A dataset consists of one or more individual fragments.  A fragment is anything
/// that is indepedently scannable, often a file.
///
/// Batches from all fragments will be converted to a single schema. This unified
/// schema is referred to as the "dataset schema" and is the output schema for
/// this node.
///
/// Individual fragments may have schemas that are different from the dataset
/// schema.  This is sometimes referred to as the physical or fragment schema.
/// Conversion from the fragment schema to the dataset schema is a process
/// known as evolution.
struct ARROW_DS_EXPORT ScanV2Options : public compute::ExecNodeOptions {
  explicit ScanV2Options(std::shared_ptr<Dataset> dataset)
      : dataset(std::move(dataset)) {}

  /// \brief The dataset to scan
  std::shared_ptr<Dataset> dataset;
  /// \brief A row filter
  ///
  /// The filter expression should be written against the dataset schema.
  /// The filter must be unbound.
  ///
  /// This is an opportunistic pushdown filter.  Filtering capabilities will
  /// vary between formats.  If a format is not capable of applying the filter
  /// then it will ignore it.
  ///
  /// Each fragment will do its best to filter the data based on the information
  /// (partitioning guarantees, statistics) available to it.  If it is able to
  /// apply some filtering then it will indicate what filtering it was able to
  /// apply by attaching a guarantee to the batch.
  ///
  /// For example, if a filter is x < 50 && y > 40 then a batch may be able to
  /// apply a guarantee x < 50.  Post-scan filtering would then only need to
  /// consider y > 40 (for this specific batch).  The next batch may not be able
  /// to attach any guarantee and both clauses would need to be applied to that batch.
  ///
  /// A single guarantee-aware filtering operation should generally be applied to all
  /// resulting batches.  The scan node is not responsible for this.
  ///
  /// Fields that are referenced by the filter should be included in the `columns` vector.
  /// The scan node will not automatically fetch fields referenced by the filter
  /// expression. \see AddFieldsNeededForFilter
  ///
  /// If the filter references fields that are not included in `columns` this may or may
  /// not be an error, depending on the format.
  compute::Expression filter = compute::literal(true);

  /// \brief The columns to scan
  ///
  /// This is not a simple list of top-level column indices but instead a set of paths
  /// allowing for partial selection of columns
  ///
  /// These paths refer to the dataset schema
  ///
  /// For example, consider the following dataset schema:
  ///   schema({
  ///     field("score", int32()),
  ///           "marker", struct_({
  ///              field("color", utf8()),
  ///              field("location", struct_({
  ///                  field("x", float64()),
  ///                  field("y", float64())
  ///              })
  ///          })
  ///   })
  ///
  /// If `columns` is {{0}, {1,1,0}} then the output schema is:
  ///   schema({field("score", int32()), field("x", float64())})
  ///
  /// If `columns` is {{1,1,1}, {1,1}} then the output schema is:
  ///   schema({
  ///       field("y", float64()),
  ///       field("location", struct_({
  ///           field("x", float64()),
  ///           field("y", float64())
  ///       })
  ///   })
  std::vector<FieldPath> columns;

  /// \brief Target number of bytes to read ahead in a fragment
  ///
  /// This limit involves some amount of estimation.  Formats typically only know
  /// batch boundaries in terms of rows (not decoded bytes) and so an estimation
  /// must be done to guess the average row size.  Other formats like CSV and JSON
  /// must make even more generalized guesses.
  ///
  /// This is a best-effort guide.  Some formats may need to read ahead further,
  /// for example, if scanning a parquet file that has batches with 100MiB of data
  /// then the actual readahead will be at least 100MiB
  ///
  /// Set to 0 to disable readhead.  When disabled, the scanner will read the
  /// dataset one batch at a time
  ///
  /// This limit applies across all fragments.  If the limit is 32MiB and the
  /// fragment readahead allows for 20 fragments to be read at once then the
  /// total readahead will still be 32MiB and NOT 20 * 32MiB.
  int32_t target_bytes_readahead = kDefaultBytesReadahead;

  /// \brief Number of fragments to read ahead
  ///
  /// Higher readahead will potentially lead to more efficient I/O but will lead
  /// to the scan operation using more RAM.  The default is fairly conservative
  /// and designed for fast local disks (or slow local spinning disks which cannot
  /// handle much parallelism anyways).  When using a highly parallel remote filesystem
  /// you will likely want to increase these values.
  ///
  /// Set to 0 to disable fragment readahead.  When disabled the dataset will be scanned
  /// one fragment at a time.
  int32_t fragment_readahead = kDefaultFragmentReadahead;
  /// \brief Options specific to the file format
  const FragmentScanOptions* format_options = NULLPTR;

  /// \brief Utility method to get a selection representing all columns in a dataset
  static std::vector<FieldPath> AllColumns(const Schema& dataset_schema);

  /// \brief Utility method to add fields needed for the current filter
  ///
  /// This method adds any fields that are needed by `filter` which are not already
  /// included in the list of columns.  Any new fields added will be added to the end
  /// in no particular order.
  static Status AddFieldsNeededForFilter(ScanV2Options* options);
};

/// \brief Describes a projection
struct ARROW_DS_EXPORT ProjectionDescr {
  /// \brief The projection expression itself
  /// This expression must be a call to make_struct
  compute::Expression expression;
  /// \brief The output schema of the projection.

  /// This can be calculated from the input schema and the expression but it
  /// is cached here for convenience.
  std::shared_ptr<Schema> schema;

  /// \brief Create a ProjectionDescr by binding an expression to the dataset schema
  ///
  /// expression must return a struct type
  static Result<ProjectionDescr> FromStructExpression(
      const compute::Expression& expression, const Schema& dataset_schema);

  /// \brief Create a ProjectionDescr from expressions/names for each field
  static Result<ProjectionDescr> FromExpressions(std::vector<compute::Expression> exprs,
                                                 std::vector<std::string> names,
                                                 const Schema& dataset_schema);

  /// \brief Create a default projection referencing fields in the dataset schema
  static Result<ProjectionDescr> FromNames(std::vector<std::string> names,
                                           const Schema& dataset_schema);

  /// \brief Make a projection that projects every field in the dataset schema
  static Result<ProjectionDescr> Default(const Schema& dataset_schema);
};

/// \brief Utility method to set the projection expression and schema
ARROW_DS_EXPORT void SetProjection(ScanOptions* options, ProjectionDescr projection);

/// \brief Combines a record batch with the fragment that the record batch originated
/// from
///
/// Knowing the source fragment can be useful for debugging & understanding loaded
/// data
struct TaggedRecordBatch {
  std::shared_ptr<RecordBatch> record_batch;
  std::shared_ptr<Fragment> fragment;
};
using TaggedRecordBatchGenerator = std::function<Future<TaggedRecordBatch>()>;
using TaggedRecordBatchIterator = Iterator<TaggedRecordBatch>;

/// \brief Combines a tagged batch with positional information
///
/// This is returned when scanning batches in an unordered fashion.  This information is
/// needed if you ever want to reassemble the batches in order
struct EnumeratedRecordBatch {
  Enumerated<std::shared_ptr<RecordBatch>> record_batch;
  Enumerated<std::shared_ptr<Fragment>> fragment;
};
using EnumeratedRecordBatchGenerator = std::function<Future<EnumeratedRecordBatch>()>;
using EnumeratedRecordBatchIterator = Iterator<EnumeratedRecordBatch>;

/// @}

}  // namespace dataset

template <>
struct IterationTraits<dataset::TaggedRecordBatch> {
  static dataset::TaggedRecordBatch End() {
    return dataset::TaggedRecordBatch{NULLPTR, NULLPTR};
  }
  static bool IsEnd(const dataset::TaggedRecordBatch& val) {
    return val.record_batch == NULLPTR;
  }
};

template <>
struct IterationTraits<dataset::EnumeratedRecordBatch> {
  static dataset::EnumeratedRecordBatch End() {
    return dataset::EnumeratedRecordBatch{
        IterationEnd<Enumerated<std::shared_ptr<RecordBatch>>>(),
        IterationEnd<Enumerated<std::shared_ptr<dataset::Fragment>>>()};
  }
  static bool IsEnd(const dataset::EnumeratedRecordBatch& val) {
    return IsIterationEnd(val.fragment);
  }
};

namespace dataset {

/// \defgroup dataset-scanning Scanning API
///
/// @{

/// \brief A scanner glues together several dataset classes to load in data.
/// The dataset contains a collection of fragments and partitioning rules.
///
/// The fragments identify independently loadable units of data (i.e. each fragment has
/// a potentially unique schema and possibly even format.  It should be possible to read
/// fragments in parallel if desired).
///
/// The fragment's format contains the logic necessary to actually create a task to load
/// the fragment into memory.  That task may or may not support parallel execution of
/// its own.
///
/// The scanner is then responsible for creating scan tasks from every fragment in the
/// dataset and (potentially) sequencing the loaded record batches together.
///
/// The scanner should not buffer the entire dataset in memory (unless asked) instead
/// yielding record batches as soon as they are ready to scan.  Various readahead
/// properties control how much data is allowed to be scanned before pausing to let a
/// slow consumer catchup.
///
/// Today the scanner also handles projection & filtering although that may change in
/// the future.
class ARROW_DS_EXPORT Scanner {
 public:
  virtual ~Scanner() = default;

  /// \brief Apply a visitor to each RecordBatch as it is scanned. If multiple threads
  /// are used (via use_threads), the visitor will be invoked from those threads and is
  /// responsible for any synchronization.
  virtual Status Scan(std::function<Status(TaggedRecordBatch)> visitor) = 0;
  /// \brief Convert a Scanner into a Table.
  ///
  /// Use this convenience utility with care. This will serially materialize the
  /// Scan result in memory before creating the Table.
  virtual Result<std::shared_ptr<Table>> ToTable() = 0;
  /// \brief Scan the dataset into a stream of record batches.  Each batch is tagged
  /// with the fragment it originated from.  The batches will arrive in order.  The
  /// order of fragments is determined by the dataset.
  ///
  /// Note: The scanner will perform some readahead but will avoid materializing too
  /// much in memory (this is goverended by the readahead options and use_threads option).
  /// If the readahead queue fills up then I/O will pause until the calling thread catches
  /// up.
  virtual Result<TaggedRecordBatchIterator> ScanBatches() = 0;
  virtual Result<TaggedRecordBatchGenerator> ScanBatchesAsync() = 0;
  virtual Result<TaggedRecordBatchGenerator> ScanBatchesAsync(
      ::arrow::internal::Executor* cpu_thread_pool) = 0;
  /// \brief Scan the dataset into a stream of record batches.  Unlike ScanBatches this
  /// method may allow record batches to be returned out of order.  This allows for more
  /// efficient scanning: some fragments may be accessed more quickly than others (e.g.
  /// may be cached in RAM or just happen to get scheduled earlier by the I/O)
  ///
  /// To make up for the out-of-order iteration each batch is further tagged with
  /// positional information.
  virtual Result<EnumeratedRecordBatchIterator> ScanBatchesUnordered() = 0;
  virtual Result<EnumeratedRecordBatchGenerator> ScanBatchesUnorderedAsync() = 0;
  virtual Result<EnumeratedRecordBatchGenerator> ScanBatchesUnorderedAsync(
      ::arrow::internal::Executor* cpu_thread_pool) = 0;
  /// \brief A convenience to synchronously load the given rows by index.
  ///
  /// Will only consume as many batches as needed from ScanBatches().
  virtual Result<std::shared_ptr<Table>> TakeRows(const Array& indices) = 0;
  /// \brief Get the first N rows.
  virtual Result<std::shared_ptr<Table>> Head(int64_t num_rows) = 0;
  /// \brief Count rows matching a predicate.
  ///
  /// This method will push down the predicate and compute the result based on fragment
  /// metadata if possible.
  virtual Result<int64_t> CountRows() = 0;
  virtual Future<int64_t> CountRowsAsync() = 0;
  /// \brief Convert the Scanner to a RecordBatchReader so it can be
  /// easily used with APIs that expect a reader.
  virtual Result<std::shared_ptr<RecordBatchReader>> ToRecordBatchReader() = 0;

  /// \brief Get the options for this scan.
  const std::shared_ptr<ScanOptions>& options() const { return scan_options_; }
  /// \brief Get the dataset that this scanner will scan
  virtual const std::shared_ptr<Dataset>& dataset() const = 0;

 protected:
  explicit Scanner(std::shared_ptr<ScanOptions> scan_options)
      : scan_options_(std::move(scan_options)) {}

  Result<EnumeratedRecordBatchIterator> AddPositioningToInOrderScan(
      TaggedRecordBatchIterator scan);

  const std::shared_ptr<ScanOptions> scan_options_;
};

/// \brief ScannerBuilder is a factory class to construct a Scanner. It is used
/// to pass information, notably a potential filter expression and a subset of
/// columns to materialize.
class ARROW_DS_EXPORT ScannerBuilder {
 public:
  explicit ScannerBuilder(std::shared_ptr<Dataset> dataset);

  ScannerBuilder(std::shared_ptr<Dataset> dataset,
                 std::shared_ptr<ScanOptions> scan_options);

  ScannerBuilder(std::shared_ptr<Schema> schema, std::shared_ptr<Fragment> fragment,
                 std::shared_ptr<ScanOptions> scan_options);

  /// \brief Make a scanner from a record batch reader.
  ///
  /// The resulting scanner can be scanned only once. This is intended
  /// to support writing data from streaming sources or other sources
  /// that can be iterated only once.
  static std::shared_ptr<ScannerBuilder> FromRecordBatchReader(
      std::shared_ptr<RecordBatchReader> reader);

  /// \brief Set the subset of columns to materialize.
  ///
  /// Columns which are not referenced may not be read from fragments.
  ///
  /// \param[in] columns list of columns to project. Order and duplicates will
  ///            be preserved.
  ///
  /// \return Failure if any column name does not exists in the dataset's
  ///         Schema.
  Status Project(std::vector<std::string> columns);

  /// \brief Set expressions which will be evaluated to produce the materialized
  /// columns.
  ///
  /// Columns which are not referenced may not be read from fragments.
  ///
  /// \param[in] exprs expressions to evaluate to produce columns.
  /// \param[in] names list of names for the resulting columns.
  ///
  /// \return Failure if any referenced column does not exists in the dataset's
  ///         Schema.
  Status Project(std::vector<compute::Expression> exprs, std::vector<std::string> names);

  /// \brief Set the filter expression to return only rows matching the filter.
  ///
  /// The predicate will be passed down to Sources and corresponding
  /// Fragments to exploit predicate pushdown if possible using
  /// partition information or Fragment internal metadata, e.g. Parquet statistics.
  /// Columns which are not referenced may not be read from fragments.
  ///
  /// \param[in] filter expression to filter rows with.
  ///
  /// \return Failure if any referenced columns does not exist in the dataset's
  ///         Schema.
  Status Filter(const compute::Expression& filter);

  /// \brief Indicate if the Scanner should make use of the available
  ///        ThreadPool found in ScanOptions;
  Status UseThreads(bool use_threads = true);

  /// \brief Set the maximum number of rows per RecordBatch.
  ///
  /// \param[in] batch_size the maximum number of rows.
  /// \returns An error if the number for batch is not greater than 0.
  ///
  /// This option provides a control limiting the memory owned by any RecordBatch.
  Status BatchSize(int64_t batch_size);

  /// \brief Set the number of batches to read ahead within a fragment.
  ///
  /// \param[in] batch_readahead How many batches to read ahead within a fragment
  /// \returns an error if this number is less than 0.
  ///
  /// This option provides a control on the RAM vs I/O tradeoff.
  /// It might not be supported by all file formats, in which case it will
  /// simply be ignored.
  Status BatchReadahead(int32_t batch_readahead);

  /// \brief Set the number of fragments to read ahead
  ///
  /// \param[in] fragment_readahead How many fragments to read ahead
  /// \returns an error if this number is less than 0.
  ///
  /// This option provides a control on the RAM vs I/O tradeoff.
  Status FragmentReadahead(int32_t fragment_readahead);

  /// \brief Set the pool from which materialized and scanned arrays will be allocated.
  Status Pool(MemoryPool* pool);

  /// \brief Set fragment-specific scan options.
  Status FragmentScanOptions(std::shared_ptr<FragmentScanOptions> fragment_scan_options);

  /// \brief Override default backpressure configuration
  Status Backpressure(compute::BackpressureOptions backpressure);

  /// \brief Return the current scan options for the builder.
  Result<std::shared_ptr<ScanOptions>> GetScanOptions();

  /// \brief Return the constructed now-immutable Scanner object
  Result<std::shared_ptr<Scanner>> Finish();

  const std::shared_ptr<Schema>& schema() const;
  const std::shared_ptr<Schema>& projected_schema() const;

 private:
  std::shared_ptr<Dataset> dataset_;
  std::shared_ptr<ScanOptions> scan_options_ = std::make_shared<ScanOptions>();
};

/// \brief Construct a source ExecNode which yields batches from a dataset scan.
///
/// Does not construct associated filter or project nodes.
/// Yielded batches will be augmented with fragment/batch indices to enable stable
/// ordering for simple ExecPlans.
class ARROW_DS_EXPORT ScanNodeOptions : public compute::ExecNodeOptions {
 public:
  explicit ScanNodeOptions(std::shared_ptr<Dataset> dataset,
                           std::shared_ptr<ScanOptions> scan_options,
                           bool require_sequenced_output = false)
      : dataset(std::move(dataset)),
        scan_options(std::move(scan_options)),
        require_sequenced_output(require_sequenced_output) {}

  std::shared_ptr<Dataset> dataset;
  std::shared_ptr<ScanOptions> scan_options;
  bool require_sequenced_output;
};

/// @}

namespace internal {
ARROW_DS_EXPORT void InitializeScanner(arrow::compute::ExecFactoryRegistry* registry);
ARROW_DS_EXPORT void InitializeScannerV2(arrow::compute::ExecFactoryRegistry* registry);
}  // namespace internal
}  // namespace dataset
}  // namespace arrow

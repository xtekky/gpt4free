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

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "arrow/compute/api_aggregate.h"
#include "arrow/compute/api_vector.h"
#include "arrow/compute/exec.h"
#include "arrow/compute/exec/expression.h"
#include "arrow/record_batch.h"
#include "arrow/result.h"
#include "arrow/util/async_generator.h"
#include "arrow/util/async_util.h"
#include "arrow/util/visibility.h"

namespace arrow {

namespace internal {

class Executor;

}  // namespace internal

namespace compute {

using AsyncExecBatchGenerator = AsyncGenerator<std::optional<ExecBatch>>;

/// \addtogroup execnode-options
/// @{
class ARROW_EXPORT ExecNodeOptions {
 public:
  virtual ~ExecNodeOptions() = default;
};

/// \brief Adapt an AsyncGenerator<ExecBatch> as a source node
///
/// plan->exec_context()->executor() will be used to parallelize pushing to
/// outputs, if provided.
class ARROW_EXPORT SourceNodeOptions : public ExecNodeOptions {
 public:
  SourceNodeOptions(std::shared_ptr<Schema> output_schema,
                    std::function<Future<std::optional<ExecBatch>>()> generator)
      : output_schema(std::move(output_schema)), generator(std::move(generator)) {}

  static Result<std::shared_ptr<SourceNodeOptions>> FromTable(const Table& table,
                                                              arrow::internal::Executor*);

  static Result<std::shared_ptr<SourceNodeOptions>> FromRecordBatchReader(
      std::shared_ptr<RecordBatchReader> reader, std::shared_ptr<Schema> schema,
      arrow::internal::Executor*);

  std::shared_ptr<Schema> output_schema;
  std::function<Future<std::optional<ExecBatch>>()> generator;
};

/// \brief An extended Source node which accepts a table
class ARROW_EXPORT TableSourceNodeOptions : public ExecNodeOptions {
 public:
  static constexpr int64_t kDefaultMaxBatchSize = 1 << 20;
  TableSourceNodeOptions(std::shared_ptr<Table> table,
                         int64_t max_batch_size = kDefaultMaxBatchSize)
      : table(table), max_batch_size(max_batch_size) {}

  // arrow table which acts as the data source
  std::shared_ptr<Table> table;
  // Size of batches to emit from this node
  // If the table is larger the node will emit multiple batches from the
  // the table to be processed in parallel.
  int64_t max_batch_size;
};

/// \brief Define a lazy resolved Arrow table.
///
/// The table uniquely identified by the names can typically be resolved at the time when
/// the plan is to be consumed.
///
/// This node is for serialization purposes only and can never be executed.
class ARROW_EXPORT NamedTableNodeOptions : public ExecNodeOptions {
 public:
  NamedTableNodeOptions(std::vector<std::string> names, std::shared_ptr<Schema> schema)
      : names(std::move(names)), schema(schema) {}

  std::vector<std::string> names;
  std::shared_ptr<Schema> schema;
};

/// \brief An extended Source node which accepts a schema
///
/// ItMaker is a maker of an iterator of tabular data.
template <typename ItMaker>
class ARROW_EXPORT SchemaSourceNodeOptions : public ExecNodeOptions {
 public:
  SchemaSourceNodeOptions(std::shared_ptr<Schema> schema, ItMaker it_maker,
                          arrow::internal::Executor* io_executor = NULLPTR)
      : schema(schema), it_maker(std::move(it_maker)), io_executor(io_executor) {}

  /// \brief The schema of the record batches from the iterator
  std::shared_ptr<Schema> schema;

  /// \brief A maker of an iterator which acts as the data source
  ItMaker it_maker;

  /// \brief The executor to use for scanning the iterator
  ///
  /// Defaults to the default I/O executor.
  arrow::internal::Executor* io_executor;
};

class ARROW_EXPORT RecordBatchReaderSourceNodeOptions : public ExecNodeOptions {
 public:
  RecordBatchReaderSourceNodeOptions(std::shared_ptr<RecordBatchReader> reader,
                                     arrow::internal::Executor* io_executor = NULLPTR)
      : reader(std::move(reader)), io_executor(io_executor) {}

  /// \brief The RecordBatchReader which acts as the data source
  std::shared_ptr<RecordBatchReader> reader;

  /// \brief The executor to use for the reader
  ///
  /// Defaults to the default I/O executor.
  arrow::internal::Executor* io_executor;
};

using ArrayVectorIteratorMaker = std::function<Iterator<std::shared_ptr<ArrayVector>>()>;
/// \brief An extended Source node which accepts a schema and array-vectors
class ARROW_EXPORT ArrayVectorSourceNodeOptions
    : public SchemaSourceNodeOptions<ArrayVectorIteratorMaker> {
  using SchemaSourceNodeOptions::SchemaSourceNodeOptions;
};

using ExecBatchIteratorMaker = std::function<Iterator<std::shared_ptr<ExecBatch>>()>;
/// \brief An extended Source node which accepts a schema and exec-batches
class ARROW_EXPORT ExecBatchSourceNodeOptions
    : public SchemaSourceNodeOptions<ExecBatchIteratorMaker> {
  using SchemaSourceNodeOptions::SchemaSourceNodeOptions;
};

using RecordBatchIteratorMaker = std::function<Iterator<std::shared_ptr<RecordBatch>>()>;
/// \brief An extended Source node which accepts a schema and record-batches
class ARROW_EXPORT RecordBatchSourceNodeOptions
    : public SchemaSourceNodeOptions<RecordBatchIteratorMaker> {
  using SchemaSourceNodeOptions::SchemaSourceNodeOptions;
};

/// \brief Make a node which excludes some rows from batches passed through it
///
/// filter_expression will be evaluated against each batch which is pushed to
/// this node. Any rows for which filter_expression does not evaluate to `true` will be
/// excluded in the batch emitted by this node.
class ARROW_EXPORT FilterNodeOptions : public ExecNodeOptions {
 public:
  explicit FilterNodeOptions(Expression filter_expression)
      : filter_expression(std::move(filter_expression)) {}

  Expression filter_expression;
};

/// \brief Make a node which executes expressions on input batches, producing new batches.
///
/// Each expression will be evaluated against each batch which is pushed to
/// this node to produce a corresponding output column.
///
/// If names are not provided, the string representations of exprs will be used.
class ARROW_EXPORT ProjectNodeOptions : public ExecNodeOptions {
 public:
  explicit ProjectNodeOptions(std::vector<Expression> expressions,
                              std::vector<std::string> names = {})
      : expressions(std::move(expressions)), names(std::move(names)) {}

  std::vector<Expression> expressions;
  std::vector<std::string> names;
};

/// \brief Make a node which aggregates input batches, optionally grouped by keys.
///
/// If the keys attribute is a non-empty vector, then each aggregate in `aggregates` is
/// expected to be a HashAggregate function. If the keys attribute is an empty vector,
/// then each aggregate is assumed to be a ScalarAggregate function.
class ARROW_EXPORT AggregateNodeOptions : public ExecNodeOptions {
 public:
  explicit AggregateNodeOptions(std::vector<Aggregate> aggregates,
                                std::vector<FieldRef> keys = {})
      : aggregates(std::move(aggregates)), keys(std::move(keys)) {}

  // aggregations which will be applied to the targetted fields
  std::vector<Aggregate> aggregates;
  // keys by which aggregations will be grouped
  std::vector<FieldRef> keys;
};

constexpr int32_t kDefaultBackpressureHighBytes = 1 << 30;  // 1GiB
constexpr int32_t kDefaultBackpressureLowBytes = 1 << 28;   // 256MiB

class ARROW_EXPORT BackpressureMonitor {
 public:
  virtual ~BackpressureMonitor() = default;
  virtual uint64_t bytes_in_use() = 0;
  virtual bool is_paused() = 0;
};

/// \brief Options to control backpressure behavior
struct ARROW_EXPORT BackpressureOptions {
  /// \brief Create default options that perform no backpressure
  BackpressureOptions() : resume_if_below(0), pause_if_above(0) {}
  /// \brief Create options that will perform backpressure
  ///
  /// \param resume_if_below The producer should resume producing if the backpressure
  ///                        queue has fewer than resume_if_below items.
  /// \param pause_if_above The producer should pause producing if the backpressure
  ///                       queue has more than pause_if_above items
  BackpressureOptions(uint64_t resume_if_below, uint64_t pause_if_above)
      : resume_if_below(resume_if_below), pause_if_above(pause_if_above) {}

  static BackpressureOptions DefaultBackpressure() {
    return BackpressureOptions(kDefaultBackpressureLowBytes,
                               kDefaultBackpressureHighBytes);
  }

  bool should_apply_backpressure() const { return pause_if_above > 0; }

  uint64_t resume_if_below;
  uint64_t pause_if_above;
};

/// \brief Add a sink node which forwards to an AsyncGenerator<ExecBatch>
///
/// Emitted batches will not be ordered.
class ARROW_EXPORT SinkNodeOptions : public ExecNodeOptions {
 public:
  explicit SinkNodeOptions(std::function<Future<std::optional<ExecBatch>>()>* generator,
                           std::shared_ptr<Schema>* schema,
                           BackpressureOptions backpressure = {},
                           BackpressureMonitor** backpressure_monitor = NULLPTR)
      : generator(generator),
        schema(schema),
        backpressure(backpressure),
        backpressure_monitor(backpressure_monitor) {}

  explicit SinkNodeOptions(std::function<Future<std::optional<ExecBatch>>()>* generator,
                           BackpressureOptions backpressure = {},
                           BackpressureMonitor** backpressure_monitor = NULLPTR)
      : generator(generator),
        schema(NULLPTR),
        backpressure(std::move(backpressure)),
        backpressure_monitor(backpressure_monitor) {}

  /// \brief A pointer to a generator of batches.
  ///
  /// This will be set when the node is added to the plan and should be used to consume
  /// data from the plan.  If this function is not called frequently enough then the sink
  /// node will start to accumulate data and may apply backpressure.
  std::function<Future<std::optional<ExecBatch>>()>* generator;
  /// \brief A pointer which will be set to the schema of the generated batches
  ///
  /// This is optional, if nullptr is passed in then it will be ignored.
  /// This will be set when the node is added to the plan, before StartProducing is called
  std::shared_ptr<Schema>* schema;
  /// \brief Options to control when to apply backpressure
  ///
  /// This is optional, the default is to never apply backpressure.  If the plan is not
  /// consumed quickly enough the system may eventually run out of memory.
  BackpressureOptions backpressure;
  /// \brief A pointer to a backpressure monitor
  ///
  /// This will be set when the node is added to the plan.  This can be used to inspect
  /// the amount of data currently queued in the sink node.  This is an optional utility
  /// and backpressure can be applied even if this is not used.
  BackpressureMonitor** backpressure_monitor;
};

/// \brief Control used by a SinkNodeConsumer to pause & resume
///
/// Callers should ensure that they do not call Pause and Resume simultaneously and they
/// should sequence things so that a call to Pause() is always followed by an eventual
/// call to Resume()
class ARROW_EXPORT BackpressureControl {
 public:
  virtual ~BackpressureControl() = default;
  /// \brief Ask the input to pause
  ///
  /// This is best effort, batches may continue to arrive
  /// Must eventually be followed by a call to Resume() or deadlock will occur
  virtual void Pause() = 0;
  /// \brief Ask the input to resume
  virtual void Resume() = 0;
};

class ARROW_EXPORT SinkNodeConsumer {
 public:
  virtual ~SinkNodeConsumer() = default;
  /// \brief Prepare any consumer state
  ///
  /// This will be run once the schema is finalized as the plan is starting and
  /// before any calls to Consume.  A common use is to save off the schema so that
  /// batches can be interpreted.
  /// TODO(ARROW-17837) Move ExecPlan* plan to query context
  virtual Status Init(const std::shared_ptr<Schema>& schema,
                      BackpressureControl* backpressure_control, ExecPlan* plan) = 0;
  /// \brief Consume a batch of data
  virtual Status Consume(ExecBatch batch) = 0;
  /// \brief Signal to the consumer that the last batch has been delivered
  ///
  /// The returned future should only finish when all outstanding tasks have completed
  virtual Future<> Finish() = 0;
};

/// \brief Add a sink node which consumes data within the exec plan run
class ARROW_EXPORT ConsumingSinkNodeOptions : public ExecNodeOptions {
 public:
  explicit ConsumingSinkNodeOptions(std::shared_ptr<SinkNodeConsumer> consumer,
                                    std::vector<std::string> names = {})
      : consumer(std::move(consumer)), names(std::move(names)) {}

  std::shared_ptr<SinkNodeConsumer> consumer;
  /// \brief Names to rename the sink's schema fields to
  ///
  /// If specified then names must be provided for all fields. Currently, only a flat
  /// schema is supported (see ARROW-15901).
  std::vector<std::string> names;
};

/// \brief Make a node which sorts rows passed through it
///
/// All batches pushed to this node will be accumulated, then sorted, by the given
/// fields. Then sorted batches will be forwarded to the generator in sorted order.
class ARROW_EXPORT OrderBySinkNodeOptions : public SinkNodeOptions {
 public:
  explicit OrderBySinkNodeOptions(
      SortOptions sort_options,
      std::function<Future<std::optional<ExecBatch>>()>* generator)
      : SinkNodeOptions(generator), sort_options(std::move(sort_options)) {}

  SortOptions sort_options;
};

/// @}

enum class JoinType {
  LEFT_SEMI,
  RIGHT_SEMI,
  LEFT_ANTI,
  RIGHT_ANTI,
  INNER,
  LEFT_OUTER,
  RIGHT_OUTER,
  FULL_OUTER
};

std::string ToString(JoinType t);

enum class JoinKeyCmp { EQ, IS };

/// \addtogroup execnode-options
/// @{

/// \brief Make a node which implements join operation using hash join strategy.
class ARROW_EXPORT HashJoinNodeOptions : public ExecNodeOptions {
 public:
  static constexpr const char* default_output_suffix_for_left = "";
  static constexpr const char* default_output_suffix_for_right = "";
  HashJoinNodeOptions(
      JoinType in_join_type, std::vector<FieldRef> in_left_keys,
      std::vector<FieldRef> in_right_keys, Expression filter = literal(true),
      std::string output_suffix_for_left = default_output_suffix_for_left,
      std::string output_suffix_for_right = default_output_suffix_for_right,
      bool disable_bloom_filter = false)
      : join_type(in_join_type),
        left_keys(std::move(in_left_keys)),
        right_keys(std::move(in_right_keys)),
        output_all(true),
        output_suffix_for_left(std::move(output_suffix_for_left)),
        output_suffix_for_right(std::move(output_suffix_for_right)),
        filter(std::move(filter)),
        disable_bloom_filter(disable_bloom_filter) {
    this->key_cmp.resize(this->left_keys.size());
    for (size_t i = 0; i < this->left_keys.size(); ++i) {
      this->key_cmp[i] = JoinKeyCmp::EQ;
    }
  }
  HashJoinNodeOptions(std::vector<FieldRef> in_left_keys,
                      std::vector<FieldRef> in_right_keys)
      : left_keys(std::move(in_left_keys)), right_keys(std::move(in_right_keys)) {
    this->join_type = JoinType::INNER;
    this->output_all = true;
    this->output_suffix_for_left = default_output_suffix_for_left;
    this->output_suffix_for_right = default_output_suffix_for_right;
    this->key_cmp.resize(this->left_keys.size());
    for (size_t i = 0; i < this->left_keys.size(); ++i) {
      this->key_cmp[i] = JoinKeyCmp::EQ;
    }
    this->filter = literal(true);
  }
  HashJoinNodeOptions(
      JoinType join_type, std::vector<FieldRef> left_keys,
      std::vector<FieldRef> right_keys, std::vector<FieldRef> left_output,
      std::vector<FieldRef> right_output, Expression filter = literal(true),
      std::string output_suffix_for_left = default_output_suffix_for_left,
      std::string output_suffix_for_right = default_output_suffix_for_right,
      bool disable_bloom_filter = false)
      : join_type(join_type),
        left_keys(std::move(left_keys)),
        right_keys(std::move(right_keys)),
        output_all(false),
        left_output(std::move(left_output)),
        right_output(std::move(right_output)),
        output_suffix_for_left(std::move(output_suffix_for_left)),
        output_suffix_for_right(std::move(output_suffix_for_right)),
        filter(std::move(filter)),
        disable_bloom_filter(disable_bloom_filter) {
    this->key_cmp.resize(this->left_keys.size());
    for (size_t i = 0; i < this->left_keys.size(); ++i) {
      this->key_cmp[i] = JoinKeyCmp::EQ;
    }
  }
  HashJoinNodeOptions(
      JoinType join_type, std::vector<FieldRef> left_keys,
      std::vector<FieldRef> right_keys, std::vector<FieldRef> left_output,
      std::vector<FieldRef> right_output, std::vector<JoinKeyCmp> key_cmp,
      Expression filter = literal(true),
      std::string output_suffix_for_left = default_output_suffix_for_left,
      std::string output_suffix_for_right = default_output_suffix_for_right,
      bool disable_bloom_filter = false)
      : join_type(join_type),
        left_keys(std::move(left_keys)),
        right_keys(std::move(right_keys)),
        output_all(false),
        left_output(std::move(left_output)),
        right_output(std::move(right_output)),
        key_cmp(std::move(key_cmp)),
        output_suffix_for_left(std::move(output_suffix_for_left)),
        output_suffix_for_right(std::move(output_suffix_for_right)),
        filter(std::move(filter)),
        disable_bloom_filter(disable_bloom_filter) {}

  HashJoinNodeOptions() = default;

  // type of join (inner, left, semi...)
  JoinType join_type = JoinType::INNER;
  // key fields from left input
  std::vector<FieldRef> left_keys;
  // key fields from right input
  std::vector<FieldRef> right_keys;
  // if set all valid fields from both left and right input will be output
  // (and field ref vectors for output fields will be ignored)
  bool output_all = false;
  // output fields passed from left input
  std::vector<FieldRef> left_output;
  // output fields passed from right input
  std::vector<FieldRef> right_output;
  // key comparison function (determines whether a null key is equal another null
  // key or not)
  std::vector<JoinKeyCmp> key_cmp;
  // suffix added to names of output fields coming from left input (used to distinguish,
  // if necessary, between fields of the same name in left and right input and can be left
  // empty if there are no name collisions)
  std::string output_suffix_for_left;
  // suffix added to names of output fields coming from right input
  std::string output_suffix_for_right;
  // residual filter which is applied to matching rows.  Rows that do not match
  // the filter are not included.  The filter is applied against the
  // concatenated input schema (left fields then right fields) and can reference
  // fields that are not included in the output.
  Expression filter = literal(true);
  // whether or not to disable Bloom filters in this join
  bool disable_bloom_filter = false;
};

/// \brief Make a node which implements asof join operation
///
/// Note, this API is experimental and will change in the future
///
/// This node takes one left table and any number of right tables, and asof joins them
/// together. Batches produced by each input must be ordered by the "on" key.
/// This node will output one row for each row in the left table.
class ARROW_EXPORT AsofJoinNodeOptions : public ExecNodeOptions {
 public:
  /// \brief Keys for one input table of the AsofJoin operation
  ///
  /// The keys must be consistent across the input tables:
  /// Each "on" key must refer to a field of the same type and units across the tables.
  /// Each "by" key must refer to a list of fields of the same types across the tables.
  struct Keys {
    /// \brief "on" key for the join.
    ///
    /// The input table must be sorted by the "on" key. Must be a single field of a common
    /// type. Inexact match is used on the "on" key. i.e., a row is considered a match iff
    /// left_on - tolerance <= right_on <= left_on.
    /// Currently, the "on" key must be of an integer, date, or timestamp type.
    FieldRef on_key;
    /// \brief "by" key for the join.
    ///
    /// Each input table must have each field of the "by" key.  Exact equality is used for
    /// each field of the "by" key.
    /// Currently, each field of the "by" key must be of an integer, date, timestamp, or
    /// base-binary type.
    std::vector<FieldRef> by_key;
  };

  AsofJoinNodeOptions(std::vector<Keys> input_keys, int64_t tolerance)
      : input_keys(std::move(input_keys)), tolerance(tolerance) {}

  /// \brief AsofJoin keys per input table.
  ///
  /// \see `Keys` for details.
  std::vector<Keys> input_keys;
  /// \brief Tolerance for inexact "on" key matching.  Must be non-negative.
  ///
  /// The tolerance is interpreted in the same units as the "on" key.
  int64_t tolerance;
};

/// \brief Make a node which select top_k/bottom_k rows passed through it
///
/// All batches pushed to this node will be accumulated, then selected, by the given
/// fields. Then sorted batches will be forwarded to the generator in sorted order.
class ARROW_EXPORT SelectKSinkNodeOptions : public SinkNodeOptions {
 public:
  explicit SelectKSinkNodeOptions(
      SelectKOptions select_k_options,
      std::function<Future<std::optional<ExecBatch>>()>* generator)
      : SinkNodeOptions(generator), select_k_options(std::move(select_k_options)) {}

  /// SelectK options
  SelectKOptions select_k_options;
};

/// \brief Adapt a Table as a sink node
///
/// obtains the output of an execution plan to
/// a table pointer.
class ARROW_EXPORT TableSinkNodeOptions : public ExecNodeOptions {
 public:
  explicit TableSinkNodeOptions(std::shared_ptr<Table>* output_table)
      : output_table(output_table) {}

  std::shared_ptr<Table>* output_table;
};

/// @}

}  // namespace compute
}  // namespace arrow

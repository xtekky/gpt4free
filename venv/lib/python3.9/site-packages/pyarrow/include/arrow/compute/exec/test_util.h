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

#include <arrow/testing/gtest_util.h>
#include <arrow/util/vector.h>

#include <functional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "arrow/compute/exec.h"
#include "arrow/compute/exec/exec_plan.h"
#include "arrow/compute/kernel.h"
#include "arrow/testing/visibility.h"
#include "arrow/util/async_generator.h"
#include "arrow/util/pcg_random.h"

namespace arrow {
namespace compute {

using StartProducingFunc = std::function<Status(ExecNode*)>;
using StopProducingFunc = std::function<void(ExecNode*)>;

// Make a dummy node that has no execution behaviour
ARROW_TESTING_EXPORT
ExecNode* MakeDummyNode(ExecPlan* plan, std::string label, std::vector<ExecNode*> inputs,
                        int num_outputs, StartProducingFunc = {}, StopProducingFunc = {});

ARROW_TESTING_EXPORT
ExecBatch ExecBatchFromJSON(const std::vector<TypeHolder>& types, std::string_view json);

/// \brief Shape qualifier for value types. In certain instances
/// (e.g. "map_lookup" kernel), an argument may only be a scalar, where in
/// other kernels arguments can be arrays or scalars
enum class ArgShape { ANY, ARRAY, SCALAR };

ARROW_TESTING_EXPORT
ExecBatch ExecBatchFromJSON(const std::vector<TypeHolder>& types,
                            const std::vector<ArgShape>& shapes, std::string_view json);

struct BatchesWithSchema {
  std::vector<ExecBatch> batches;
  std::shared_ptr<Schema> schema;

  AsyncGenerator<std::optional<ExecBatch>> gen(bool parallel, bool slow) const {
    auto opt_batches = ::arrow::internal::MapVector(
        [](ExecBatch batch) { return std::make_optional(std::move(batch)); }, batches);

    AsyncGenerator<std::optional<ExecBatch>> gen;

    if (parallel) {
      // emulate batches completing initial decode-after-scan on a cpu thread
      gen = MakeBackgroundGenerator(MakeVectorIterator(std::move(opt_batches)),
                                    ::arrow::internal::GetCpuThreadPool())
                .ValueOrDie();

      // ensure that callbacks are not executed immediately on a background thread
      gen =
          MakeTransferredGenerator(std::move(gen), ::arrow::internal::GetCpuThreadPool());
    } else {
      gen = MakeVectorGenerator(std::move(opt_batches));
    }

    if (slow) {
      gen =
          MakeMappedGenerator(std::move(gen), [](const std::optional<ExecBatch>& batch) {
            SleepABit();
            return batch;
          });
    }

    return gen;
  }
};

ARROW_TESTING_EXPORT
Future<> StartAndFinish(ExecPlan* plan);

ARROW_TESTING_EXPORT
Future<std::vector<ExecBatch>> StartAndCollect(
    ExecPlan* plan, AsyncGenerator<std::optional<ExecBatch>> gen);

ARROW_TESTING_EXPORT
BatchesWithSchema MakeBasicBatches();

ARROW_TESTING_EXPORT
BatchesWithSchema MakeNestedBatches();

ARROW_TESTING_EXPORT
BatchesWithSchema MakeRandomBatches(const std::shared_ptr<Schema>& schema,
                                    int num_batches = 10, int batch_size = 4,
                                    int64_t alignment = kDefaultBufferAlignment,
                                    MemoryPool* memory_pool = nullptr);

ARROW_TESTING_EXPORT
BatchesWithSchema MakeBatchesFromString(const std::shared_ptr<Schema>& schema,
                                        const std::vector<std::string_view>& json_strings,
                                        int multiplicity = 1);

ARROW_TESTING_EXPORT
Result<std::vector<std::shared_ptr<ArrayVector>>> ToArrayVectors(
    const BatchesWithSchema& batches_with_schema);

ARROW_TESTING_EXPORT
Result<std::vector<std::shared_ptr<ExecBatch>>> ToExecBatches(
    const BatchesWithSchema& batches);

ARROW_TESTING_EXPORT
Result<std::vector<std::shared_ptr<RecordBatch>>> ToRecordBatches(
    const BatchesWithSchema& batches);

ARROW_TESTING_EXPORT
Result<std::shared_ptr<RecordBatchReader>> ToRecordBatchReader(
    const BatchesWithSchema& batches_with_schema);

ARROW_TESTING_EXPORT
Result<std::vector<std::shared_ptr<ArrayVector>>> ToArrayVectors(
    const BatchesWithSchema& batches_with_schema);

ARROW_TESTING_EXPORT
Result<std::vector<std::shared_ptr<ExecBatch>>> ToExecBatches(
    const BatchesWithSchema& batches);

ARROW_TESTING_EXPORT
Result<std::vector<std::shared_ptr<RecordBatch>>> ToRecordBatches(
    const BatchesWithSchema& batches);

ARROW_TESTING_EXPORT
Result<std::shared_ptr<Table>> SortTableOnAllFields(const std::shared_ptr<Table>& tab);

ARROW_TESTING_EXPORT
void AssertTablesEqualIgnoringOrder(const std::shared_ptr<Table>& exp,
                                    const std::shared_ptr<Table>& act);

ARROW_TESTING_EXPORT
void AssertExecBatchesEqualIgnoringOrder(const std::shared_ptr<Schema>& schema,
                                         const std::vector<ExecBatch>& exp,
                                         const std::vector<ExecBatch>& act);

ARROW_TESTING_EXPORT
bool operator==(const Declaration&, const Declaration&);

ARROW_TESTING_EXPORT
void PrintTo(const Declaration& decl, std::ostream* os);

class Random64Bit {
 public:
  explicit Random64Bit(int32_t seed) : rng_(seed) {}
  uint64_t next() { return dist_(rng_); }
  template <typename T>
  inline T from_range(const T& min_val, const T& max_val) {
    return static_cast<T>(min_val + (next() % (max_val - min_val + 1)));
  }

 private:
  random::pcg32_fast rng_;
  std::uniform_int_distribution<uint64_t> dist_;
};

/// Specify properties of a table to be generated.
struct TableGenerationProperties {
  /// Indicates the amount of time between data points that lie between
  /// the start and end parameters.
  int time_frequency;
  /// The number of additional random columns in the table.
  int num_columns;
  /// The number of unique keys in the table.
  int num_ids;
  /// Specifies the prefix of each randomly generated column.
  std::string column_prefix;
  /// Specifies the minimum value in the randomly generated column(s).
  int min_column_value;
  /// Specifies the maximum value in the randomly generated column(s).
  int max_column_value;
  /// The random seed the random array generator is given to generate the additional
  /// columns.
  int seed;
  /// Specifies the beginning of 'time' recorded in the table, inclusive.
  int start;
  /// Specifies the end of 'time' recorded in the table, inclusive.
  int end;
};

/// The table generated in accordance to the TableGenerationProperties has the following
/// schema: time (int64) id (int32) [properties.column_prefix]{idx} (float64)
/// where idx is in [0, properties.num_columns)
/// Each id has rows corresponding to a singular data point in the time range (start, end,
/// time_frequency). The table is sorted by time.
ARROW_TESTING_EXPORT
Result<std::shared_ptr<Table>> MakeRandomTimeSeriesTable(
    const TableGenerationProperties& properties);

}  // namespace compute
}  // namespace arrow

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
#include <vector>

#include "arrow/compute/exec/accumulation_queue.h"
#include "arrow/compute/exec/bloom_filter.h"
#include "arrow/compute/exec/options.h"
#include "arrow/compute/exec/query_context.h"
#include "arrow/compute/exec/schema_util.h"
#include "arrow/compute/exec/task_util.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type.h"
#include "arrow/util/tracing.h"

namespace arrow {
namespace compute {

using arrow::util::AccumulationQueue;

class HashJoinImpl {
 public:
  using OutputBatchCallback = std::function<void(int64_t, ExecBatch)>;
  using BuildFinishedCallback = std::function<Status(size_t)>;
  using FinishedCallback = std::function<void(int64_t)>;
  using RegisterTaskGroupCallback = std::function<int(
      std::function<Status(size_t, int64_t)>, std::function<Status(size_t)>)>;
  using StartTaskGroupCallback = std::function<Status(int, int64_t)>;
  using AbortContinuationImpl = std::function<void()>;

  virtual ~HashJoinImpl() = default;
  virtual Status Init(QueryContext* ctx, JoinType join_type, size_t num_threads,
                      const HashJoinProjectionMaps* proj_map_left,
                      const HashJoinProjectionMaps* proj_map_right,
                      std::vector<JoinKeyCmp> key_cmp, Expression filter,
                      RegisterTaskGroupCallback register_task_group_callback,
                      StartTaskGroupCallback start_task_group_callback,
                      OutputBatchCallback output_batch_callback,
                      FinishedCallback finished_callback) = 0;

  virtual Status BuildHashTable(size_t thread_index, AccumulationQueue batches,
                                BuildFinishedCallback on_finished) = 0;
  virtual Status ProbeSingleBatch(size_t thread_index, ExecBatch batch) = 0;
  virtual Status ProbingFinished(size_t thread_index) = 0;
  virtual void Abort(TaskScheduler::AbortContinuationImpl pos_abort_callback) = 0;
  virtual std::string ToString() const = 0;

  static Result<std::unique_ptr<HashJoinImpl>> MakeBasic();
  static Result<std::unique_ptr<HashJoinImpl>> MakeSwiss();

 protected:
  util::tracing::Span span_;
};

}  // namespace compute
}  // namespace arrow

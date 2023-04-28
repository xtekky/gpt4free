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
#include <utility>

#include "arrow/status.h"
#include "arrow/type_fwd.h"
#include "arrow/util/cancel.h"
#include "arrow/util/functional.h"
#include "arrow/util/macros.h"
#include "arrow/util/type_fwd.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace internal {

/// \brief A group of related tasks
///
/// A TaskGroup executes tasks with the signature `Status()`.
/// Execution can be serial or parallel, depending on the TaskGroup
/// implementation.  When Finish() returns, it is guaranteed that all
/// tasks have finished, or at least one has errored.
///
/// Once an error has occurred any tasks that are submitted to the task group
/// will not run.  The call to Append will simply return without scheduling the
/// task.
///
/// If the task group is parallel it is possible that multiple tasks could be
/// running at the same time and one of those tasks fails.  This will put the
/// task group in a failure state (so additional tasks cannot be run) however
/// it will not interrupt running tasks.  Finish will not complete
/// until all running tasks have finished, even if one task fails.
///
/// Once a task group has finished new tasks may not be added to it.  If you need to start
/// a new batch of work then you should create a new task group.
class ARROW_EXPORT TaskGroup : public std::enable_shared_from_this<TaskGroup> {
 public:
  /// Add a Status-returning function to execute.  Execution order is
  /// undefined.  The function may be executed immediately or later.
  template <typename Function>
  void Append(Function&& func) {
    return AppendReal(std::forward<Function>(func));
  }

  /// Wait for execution of all tasks (and subgroups) to be finished,
  /// or for at least one task (or subgroup) to error out.
  /// The returned Status propagates the error status of the first failing
  /// task (or subgroup).
  virtual Status Finish() = 0;

  /// Returns a future that will complete the first time all tasks are finished.
  /// This should be called only after all top level tasks
  /// have been added to the task group.
  ///
  /// If you are using a TaskGroup asynchronously there are a few considerations to keep
  /// in mind.  The tasks should not block on I/O, etc (defeats the purpose of using
  /// futures) and should not be doing any nested locking or you run the risk of the tasks
  /// getting stuck in the thread pool waiting for tasks which cannot get scheduled.
  ///
  /// Primarily this call is intended to help migrate existing work written with TaskGroup
  /// in mind to using futures without having to do a complete conversion on the first
  /// pass.
  virtual Future<> FinishAsync() = 0;

  /// The current aggregate error Status.  Non-blocking, useful for stopping early.
  virtual Status current_status() = 0;

  /// Whether some tasks have already failed.  Non-blocking, useful for stopping early.
  virtual bool ok() const = 0;

  /// How many tasks can typically be executed in parallel.
  /// This is only a hint, useful for testing or debugging.
  virtual int parallelism() = 0;

  static std::shared_ptr<TaskGroup> MakeSerial(StopToken = StopToken::Unstoppable());
  static std::shared_ptr<TaskGroup> MakeThreaded(internal::Executor*,
                                                 StopToken = StopToken::Unstoppable());

  virtual ~TaskGroup() = default;

 protected:
  TaskGroup() = default;
  ARROW_DISALLOW_COPY_AND_ASSIGN(TaskGroup);

  virtual void AppendReal(FnOnce<Status()> task) = 0;
};

}  // namespace internal
}  // namespace arrow

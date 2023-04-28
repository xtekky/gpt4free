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

#include <atomic>
#include <functional>
#include <list>
#include <memory>

#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/cancel.h"
#include "arrow/util/functional.h"
#include "arrow/util/future.h"
#include "arrow/util/iterator.h"
#include "arrow/util/mutex.h"
#include "arrow/util/thread_pool.h"

namespace arrow {

using internal::FnOnce;

namespace util {

/// A utility which keeps tracks of, and schedules, asynchronous tasks
///
/// An asynchronous task has a synchronous component and an asynchronous component.
/// The synchronous component typically schedules some kind of work on an external
/// resource (e.g. the I/O thread pool or some kind of kernel-based asynchronous
/// resource like io_uring).  The asynchronous part represents the work
/// done on that external resource.  Executing the synchronous part will be referred
/// to as "submitting the task" since this usually includes submitting the asynchronous
/// portion to the external thread pool.
///
/// By default the scheduler will submit the task (execute the synchronous part) as
/// soon as it is added, assuming the underlying thread pool hasn't terminated or the
/// scheduler hasn't aborted.  In this mode, the scheduler is simply acting as
/// a simple task group.
///
/// A task scheduler starts with an initial task.  That task, and all subsequent tasks
/// are free to add subtasks.  Once all submitted tasks finish the scheduler will
/// finish.  Note, it is not an error to add additional tasks after a scheduler has
/// aborted. These tasks will be ignored and never submitted.  The scheduler returns a
/// future which will complete when all submitted tasks have finished executing.  Once all
/// tasks have been finsihed the scheduler is invalid and should no longer be used.
///
/// Task failure (either the synchronous portion or the asynchronous portion) will cause
/// the scheduler to enter an aborted state.  The first such failure will be reported in
/// the final task future.
class ARROW_EXPORT AsyncTaskScheduler {
 public:
  /// Destructor for AsyncTaskScheduler
  ///
  /// The lifetime of the task scheduled is managed automatically.  The scheduler
  /// will remain valid while any tasks are running (and can always be safely accessed)
  /// within tasks) and will be destroyed as soon as all tasks have finished.
  virtual ~AsyncTaskScheduler() = default;
  /// An interface for a task
  ///
  /// Users may want to override this, for example, to add priority
  /// information for use by a queue.
  class Task {
   public:
    virtual ~Task() = default;
    /// Submit the task
    ///
    /// This will be called by the scheduler at most once when there
    /// is space to run the task.  This is expected to be a fairly quick
    /// function that simply submits the actual task work to an external
    /// resource (e.g. I/O thread pool).
    ///
    /// If this call fails then the scheduler will enter an aborted state.
    virtual Result<Future<>> operator()() = 0;
    /// The cost of the task
    ///
    /// A ThrottledAsyncTaskScheduler can be used to limit the number of concurrent tasks.
    /// A custom cost may be used, for example, if you would like to limit the number of
    /// tasks based on the total expected RAM usage of the tasks (this is done in the
    /// scanner)
    virtual int cost() const { return 1; }
  };

  /// Add a task to the scheduler
  ///
  /// If the scheduler is in an aborted state this call will return false and the task
  /// will never be run.  This is harmless and does not need to be guarded against.
  ///
  /// The return value for this call can usually be ignored.  There is little harm in
  /// attempting to add tasks to an aborted scheduler.  It is only included for callers
  /// that want to avoid future task generation to save effort.
  ///
  /// \param task the task to submit
  ///
  /// \return true if the task was submitted or queued, false if the task was ignored
  virtual bool AddTask(std::unique_ptr<Task> task) = 0;

  /// Adds an async generator to the scheduler
  ///
  /// The async generator will be visited, one item at a time.  Submitting a task
  /// will consist of polling the generator for the next future.  The generator's future
  /// will then represent the task itself.
  ///
  /// This visits the task serially without readahead.  If readahead or parallelism
  /// is desired then it should be added in the generator itself.
  ///
  /// The generator itself will be kept alive until all tasks have been completed.
  /// However, if the scheduler is aborted, the generator will be destroyed as soon as the
  /// next item would be requested.
  ///
  /// \param generator the generator to submit to the scheduler
  /// \param visitor a function which visits each generator future as it completes
  template <typename T>
  bool AddAsyncGenerator(std::function<Future<T>()> generator,
                         std::function<Status(const T&)> visitor);

  template <typename Callable>
  struct SimpleTask : public Task {
    explicit SimpleTask(Callable callable) : callable(std::move(callable)) {}
    Result<Future<>> operator()() override { return callable(); }
    Callable callable;
  };

  /// Add a task with cost 1 to the scheduler
  ///
  /// \see AddTask for details
  template <typename Callable>
  bool AddSimpleTask(Callable callable) {
    return AddTask(std::make_unique<SimpleTask<Callable>>(std::move(callable)));
  }

  /// Construct a scheduler
  ///
  /// \param initial_task The initial task which is responsible for adding
  ///        the first subtasks to the scheduler.
  /// \param abort_callback A callback that will be triggered immediately after a task
  ///        fails while other tasks may still be running.  Nothing needs to be done here,
  ///        when a task fails the scheduler will stop accepting new tasks and eventually
  ///        return the error.  However, this callback can be used to more quickly end
  ///        long running tasks that have already been submitted.  Defaults to doing
  ///        nothing.
  /// \param stop_token An optional stop token that will allow cancellation of the
  ///        scheduler.  This will be checked before each task is submitted and, in the
  ///        event of a cancellation, the scheduler will enter an aborted state. This is
  ///        a graceful cancellation and submitted tasks will still complete.
  /// \return A future that will be completed when the initial task and all subtasks have
  ///         finished.
  static Future<> Make(
      FnOnce<Status(AsyncTaskScheduler*)> initial_task,
      FnOnce<void(const Status&)> abort_callback = [](const Status&) {},
      StopToken stop_token = StopToken::Unstoppable());
};

class ARROW_EXPORT ThrottledAsyncTaskScheduler : public AsyncTaskScheduler {
 public:
  /// An interface for a task queue
  ///
  /// A queue's methods will not be called concurrently
  class Queue {
   public:
    virtual ~Queue() = default;
    /// Push a task to the queue
    ///
    /// \param task the task to enqueue
    virtual void Push(std::unique_ptr<Task> task) = 0;
    /// Pop the next task from the queue
    virtual std::unique_ptr<Task> Pop() = 0;
    /// Peek the next task in the queue
    virtual const Task& Peek() = 0;
    /// Check if the queue is empty
    virtual bool Empty() = 0;
    /// Purge the queue of all items
    virtual void Purge() = 0;
  };

  class Throttle {
   public:
    virtual ~Throttle() = default;
    /// Acquire amt permits
    ///
    /// If nullopt is returned then the permits were immediately
    /// acquired and the caller can proceed.  If a future is returned then the caller
    /// should wait for the future to complete first.  When the returned future completes
    /// the permits have NOT been acquired and the caller must call Acquire again
    ///
    /// \param amt the number of permits to acquire
    virtual std::optional<Future<>> TryAcquire(int amt) = 0;
    /// Release amt permits
    ///
    /// This will possibly complete waiting futures and should probably not be
    /// called while holding locks.
    ///
    /// \param amt the number of permits to release
    virtual void Release(int amt) = 0;

    /// The size of the largest task that can run
    ///
    /// Incoming tasks will have their cost latched to this value to ensure
    /// they can still run (although they will be the only thing allowed to
    /// run at that time).
    virtual int Capacity() = 0;

    /// Pause the throttle
    ///
    /// Any tasks that have been submitted already will continue.  However, no new tasks
    /// will be run until the throttle is resumed.
    virtual void Pause() = 0;
    /// Resume the throttle
    ///
    /// Allows taks to be submitted again.  If there is a max_concurrent_cost limit then
    /// it will still apply.
    virtual void Resume() = 0;
  };

  /// Pause the throttle
  ///
  /// Any tasks that have been submitted already will continue.  However, no new tasks
  /// will be run until the throttle is resumed.
  virtual void Pause() = 0;
  /// Resume the throttle
  ///
  /// Allows taks to be submitted again.  If there is a max_concurrent_cost limit then
  /// it will still apply.
  virtual void Resume() = 0;

  /// Create a throttled view of a scheduler
  ///
  /// Tasks added via this view will be subjected to the throttle and, if the tasks cannot
  /// run immediately, will be placed into a queue.
  ///
  /// Although a shared_ptr is returned it should generally be assumed that the caller
  /// is being given exclusive ownership.  The shared_ptr is used to share the view with
  /// queued and submitted tasks and the lifetime of those is unpredictable.  It is
  /// important the caller keep the returned pointer alive for as long as they plan to add
  /// tasks to the view.
  ///
  /// \param scheduler a scheduler to submit tasks to after throttling
  ///
  /// This can be the root scheduler, another throttled scheduler, or a task group.  These
  /// are all composable.
  ///
  /// \param max_concurrent_cost the maximum amount of cost allowed to run at any one time
  ///
  /// If a task is added that has a cost greater than max_concurrent_cost then its cost
  /// will be reduced to max_concurrent_cost so that it is still possible for the task to
  /// run.
  ///
  /// \param queue the queue to use when tasks cannot be submitted
  ///
  /// By default a FIFO queue will be used.  However, a custom queue can be provided if
  /// some tasks have higher priority than other tasks.
  static std::shared_ptr<ThrottledAsyncTaskScheduler> Make(
      AsyncTaskScheduler* scheduler, int max_concurrent_cost,
      std::unique_ptr<Queue> queue = NULLPTR);

  /// @brief Create a ThrottledAsyncTaskScheduler using a custom throttle
  ///
  /// \see Make
  static std::shared_ptr<ThrottledAsyncTaskScheduler> MakeWithCustomThrottle(
      AsyncTaskScheduler* scheduler, std::unique_ptr<Throttle> throttle,
      std::unique_ptr<Queue> queue = NULLPTR);
};

/// A utility to keep track of a collection of tasks
///
/// Often it is useful to keep track of some state that only needs to stay alive
/// for some small collection of tasks, or to perform some kind of final cleanup
/// when a collection of tasks is finished.
///
/// For example, when scanning, we need to keep the file reader alive while all scan
/// tasks run for a given file, and then we can gracefully close it when we finish the
/// file.
class ARROW_EXPORT AsyncTaskGroup : public AsyncTaskScheduler {
 public:
  /// Destructor for the task group
  ///
  /// The destructor might trigger the finish callback.  If the finish callback fails
  /// then the error will be reported as a task on the scheduler.
  ///
  /// Failure to destroy the async task group will not prevent the scheduler from
  /// finishing.  If the scheduler finishes before the async task group is done then
  /// the finish callback will be run immediately when the async task group finishes.
  ///
  /// If the scheduler has aborted then the finish callback will not run.
  ~AsyncTaskGroup() = default;
  /// Create an async task group
  ///
  /// The finish callback will not run until the task group is destroyed and all
  /// tasks are finished so you will generally want to reset / destroy the returned
  /// unique_ptr at some point.
  ///
  /// \param scheduler The underlying scheduler to submit tasks to
  /// \param finish_callback A callback that will be run only after the task group has
  ///                        been destroyed and all tasks added by the group have
  ///                        finished.
  ///
  /// Note: in error scenarios the finish callback may not run.  However, it will still,
  /// of course, be destroyed.
  static std::unique_ptr<AsyncTaskGroup> Make(AsyncTaskScheduler* scheduler,
                                              FnOnce<Status()> finish_callback);
};

/// Create a task group that is also throttled
///
/// This is a utility factory that creates a throttled view of a scheduler and then
/// wraps that throttled view with a task group that destroys the throttle when finished.
///
/// \see ThrottledAsyncTaskScheduler
/// \see AsyncTaskGroup
/// \param target the underlying scheduler to submit tasks to
/// \param max_concurrent_cost the maximum amount of cost allowed to run at any one time
/// \param queue the queue to use when tasks cannot be submitted
/// \param finish_callback A callback that will be run only after the task group has
///                  been destroyed and all tasks added by the group have finished
ARROW_EXPORT std::unique_ptr<ThrottledAsyncTaskScheduler> MakeThrottledAsyncTaskGroup(
    AsyncTaskScheduler* target, int max_concurrent_cost,
    std::unique_ptr<ThrottledAsyncTaskScheduler::Queue> queue,
    FnOnce<Status()> finish_callback);

// Defined down here to avoid circular dependency between AsyncTaskScheduler and
// AsyncTaskGroup
template <typename T>
bool AsyncTaskScheduler::AddAsyncGenerator(std::function<Future<T>()> generator,
                                           std::function<Status(const T&)> visitor) {
  struct State {
    State(std::function<Future<T>()> generator, std::function<Status(const T&)> visitor,
          std::unique_ptr<AsyncTaskGroup> task_group)
        : generator(std::move(generator)),
          visitor(std::move(visitor)),
          task_group(std::move(task_group)) {}
    std::function<Future<T>()> generator;
    std::function<Status(const T&)> visitor;
    std::unique_ptr<AsyncTaskGroup> task_group;
  };
  struct SubmitTask : public Task {
    explicit SubmitTask(std::unique_ptr<State> state_holder)
        : state_holder(std::move(state_holder)) {}

    struct SubmitTaskCallback {
      SubmitTaskCallback(std::unique_ptr<State> state_holder, Future<> task_completion)
          : state_holder(std::move(state_holder)),
            task_completion(std::move(task_completion)) {}
      void operator()(const Result<T>& maybe_item) {
        if (!maybe_item.ok()) {
          task_completion.MarkFinished(maybe_item.status());
          return;
        }
        const auto& item = *maybe_item;
        if (IsIterationEnd(item)) {
          task_completion.MarkFinished();
          return;
        }
        Status visit_st = state_holder->visitor(item);
        if (!visit_st.ok()) {
          task_completion.MarkFinished(std::move(visit_st));
          return;
        }
        state_holder->task_group->AddTask(
            std::make_unique<SubmitTask>(std::move(state_holder)));
        task_completion.MarkFinished();
      }
      std::unique_ptr<State> state_holder;
      Future<> task_completion;
    };

    Result<Future<>> operator()() {
      Future<> task = Future<>::Make();
      // Consume as many items as we can (those that are already finished)
      // synchronously to avoid recursion / stack overflow.
      while (true) {
        Future<T> next = state_holder->generator();
        if (next.TryAddCallback(
                [&] { return SubmitTaskCallback(std::move(state_holder), task); })) {
          return task;
        }
        ARROW_ASSIGN_OR_RAISE(T item, next.result());
        if (IsIterationEnd(item)) {
          task.MarkFinished();
          return task;
        }
        ARROW_RETURN_NOT_OK(state_holder->visitor(item));
      }
    }
    std::unique_ptr<State> state_holder;
  };
  std::unique_ptr<AsyncTaskGroup> task_group =
      AsyncTaskGroup::Make(this, [] { return Status::OK(); });
  AsyncTaskGroup* task_group_view = task_group.get();
  std::unique_ptr<State> state_holder = std::make_unique<State>(
      std::move(generator), std::move(visitor), std::move(task_group));
  task_group_view->AddTask(std::make_unique<SubmitTask>(std::move(state_holder)));
  return true;
}

}  // namespace util
}  // namespace arrow

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

#include <utility>
#include <vector>

#include "arrow/status.h"
#include "arrow/util/functional.h"
#include "arrow/util/thread_pool.h"
#include "arrow/util/vector.h"

namespace arrow {
namespace internal {

// A parallelizer that takes a `Status(int)` function and calls it with
// arguments between 0 and `num_tasks - 1`, on an arbitrary number of threads.

template <class FUNCTION>
Status ParallelFor(int num_tasks, FUNCTION&& func,
                   Executor* executor = internal::GetCpuThreadPool()) {
  std::vector<Future<>> futures(num_tasks);

  for (int i = 0; i < num_tasks; ++i) {
    ARROW_ASSIGN_OR_RAISE(futures[i], executor->Submit(func, i));
  }
  auto st = Status::OK();
  for (auto& fut : futures) {
    st &= fut.status();
  }
  return st;
}

template <class FUNCTION, typename T,
          typename R = typename internal::call_traits::return_type<FUNCTION>::ValueType>
Future<std::vector<R>> ParallelForAsync(
    std::vector<T> inputs, FUNCTION&& func,
    Executor* executor = internal::GetCpuThreadPool()) {
  std::vector<Future<R>> futures(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    ARROW_ASSIGN_OR_RAISE(futures[i], executor->Submit(func, i, std::move(inputs[i])));
  }
  return All(std::move(futures))
      .Then([](const std::vector<Result<R>>& results) -> Result<std::vector<R>> {
        return UnwrapOrRaise(results);
      });
}

// A parallelizer that takes a `Status(int)` function and calls it with
// arguments between 0 and `num_tasks - 1`, in sequence or in parallel,
// depending on the input boolean.

template <class FUNCTION>
Status OptionalParallelFor(bool use_threads, int num_tasks, FUNCTION&& func,
                           Executor* executor = internal::GetCpuThreadPool()) {
  if (use_threads) {
    return ParallelFor(num_tasks, std::forward<FUNCTION>(func), executor);
  } else {
    for (int i = 0; i < num_tasks; ++i) {
      RETURN_NOT_OK(func(i));
    }
    return Status::OK();
  }
}

// A parallelizer that takes a `Result<R>(int index, T item)` function and
// calls it with each item from the input array, in sequence or in parallel,
// depending on the input boolean.

template <class FUNCTION, typename T,
          typename R = typename internal::call_traits::return_type<FUNCTION>::ValueType>
Future<std::vector<R>> OptionalParallelForAsync(
    bool use_threads, std::vector<T> inputs, FUNCTION&& func,
    Executor* executor = internal::GetCpuThreadPool()) {
  if (use_threads) {
    return ParallelForAsync(std::move(inputs), std::forward<FUNCTION>(func), executor);
  } else {
    std::vector<R> result(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      ARROW_ASSIGN_OR_RAISE(result[i], func(i, inputs[i]));
    }
    return result;
  }
}

}  // namespace internal
}  // namespace arrow

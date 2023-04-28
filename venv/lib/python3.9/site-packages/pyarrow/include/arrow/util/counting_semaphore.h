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

#ifndef ARROW_COUNTING_SEMAPHORE_H
#define ARROW_COUNTING_SEMAPHORE_H

#include <memory>

#include "arrow/status.h"

namespace arrow {
namespace util {

/// \brief Simple mutex-based counting semaphore with timeout
class ARROW_EXPORT CountingSemaphore {
 public:
  /// \brief Create an instance with initial_avail starting permits
  ///
  /// \param[in] initial_avail The semaphore will start with this many permits available
  /// \param[in] timeout_seconds A timeout to be applied to all operations.  Operations
  ///            will return Status::Invalid if this timeout elapses
  explicit CountingSemaphore(uint32_t initial_avail = 0, double timeout_seconds = 10);
  ~CountingSemaphore();
  /// \brief Block until num_permits permits are available
  Status Acquire(uint32_t num_permits);
  /// \brief Make num_permits permits available
  Status Release(uint32_t num_permits);
  /// \brief Wait until num_waiters are waiting on permits
  ///
  /// This method is non-standard but useful in unit tests to ensure sequencing
  Status WaitForWaiters(uint32_t num_waiters);
  /// \brief Immediately time out any waiters
  ///
  /// This method will return Status::OK only if there were no waiters to time out.
  /// Once closed any operation on this instance will return an invalid status.
  Status Close();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace util
}  // namespace arrow

#endif  // ARROW_COUNTING_SEMAPHORE_H

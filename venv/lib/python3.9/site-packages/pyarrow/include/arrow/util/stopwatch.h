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

#include <cassert>
#include <chrono>

namespace arrow {
namespace internal {

class StopWatch {
  // This clock should give us wall clock time
  using ClockType = std::chrono::steady_clock;

 public:
  StopWatch() {}

  void Start() { start_ = ClockType::now(); }

  // Returns time in nanoseconds.
  uint64_t Stop() {
    auto stop = ClockType::now();
    std::chrono::nanoseconds d = stop - start_;
    assert(d.count() >= 0);
    return static_cast<uint64_t>(d.count());
  }

 private:
  std::chrono::time_point<ClockType> start_;
};

}  // namespace internal
}  // namespace arrow

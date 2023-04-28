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

// Contains declarations of time related Arrow builder types.

#pragma once

#include <memory>

#include "arrow/array/builder_base.h"
#include "arrow/array/builder_primitive.h"

namespace arrow {

/// \addtogroup temporal-builders
///
/// @{

// TODO(ARROW-7938): this class is untested

class ARROW_EXPORT DayTimeIntervalBuilder : public NumericBuilder<DayTimeIntervalType> {
 public:
  using DayMilliseconds = DayTimeIntervalType::DayMilliseconds;

  explicit DayTimeIntervalBuilder(MemoryPool* pool = default_memory_pool(),
                                  int64_t alignment = kDefaultBufferAlignment)
      : DayTimeIntervalBuilder(day_time_interval(), pool, alignment) {}

  explicit DayTimeIntervalBuilder(std::shared_ptr<DataType> type,
                                  MemoryPool* pool = default_memory_pool(),
                                  int64_t alignment = kDefaultBufferAlignment)
      : NumericBuilder<DayTimeIntervalType>(type, pool, alignment) {}
};

class ARROW_EXPORT MonthDayNanoIntervalBuilder
    : public NumericBuilder<MonthDayNanoIntervalType> {
 public:
  using MonthDayNanos = MonthDayNanoIntervalType::MonthDayNanos;

  explicit MonthDayNanoIntervalBuilder(MemoryPool* pool = default_memory_pool(),
                                       int64_t alignment = kDefaultBufferAlignment)
      : MonthDayNanoIntervalBuilder(month_day_nano_interval(), pool, alignment) {}

  explicit MonthDayNanoIntervalBuilder(std::shared_ptr<DataType> type,
                                       MemoryPool* pool = default_memory_pool(),
                                       int64_t alignment = kDefaultBufferAlignment)
      : NumericBuilder<MonthDayNanoIntervalType>(type, pool, alignment) {}
};

/// @}

}  // namespace arrow

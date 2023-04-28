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

#include <chrono>
#include <memory>
#include <utility>

#include "arrow/type_fwd.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace util {

enum DivideOrMultiply {
  MULTIPLY,
  DIVIDE,
};

ARROW_EXPORT
std::pair<DivideOrMultiply, int64_t> GetTimestampConversion(TimeUnit::type in_unit,
                                                            TimeUnit::type out_unit);

// Converts a Timestamp value into another Timestamp value.
//
// This function takes care of properly transforming from one unit to another.
//
// \param[in] in the input type. Must be TimestampType.
// \param[in] out the output type. Must be TimestampType.
// \param[in] value the input value.
//
// \return The converted value, or an error.
ARROW_EXPORT Result<int64_t> ConvertTimestampValue(const std::shared_ptr<DataType>& in,
                                                   const std::shared_ptr<DataType>& out,
                                                   int64_t value);

template <typename Visitor, typename... Args>
decltype(std::declval<Visitor>()(std::chrono::seconds{}, std::declval<Args&&>()...))
VisitDuration(TimeUnit::type unit, Visitor&& visitor, Args&&... args) {
  switch (unit) {
    default:
    case TimeUnit::SECOND:
      break;
    case TimeUnit::MILLI:
      return visitor(std::chrono::milliseconds{}, std::forward<Args>(args)...);
    case TimeUnit::MICRO:
      return visitor(std::chrono::microseconds{}, std::forward<Args>(args)...);
    case TimeUnit::NANO:
      return visitor(std::chrono::nanoseconds{}, std::forward<Args>(args)...);
  }
  return visitor(std::chrono::seconds{}, std::forward<Args>(args)...);
}

/// Convert a count of seconds to the corresponding count in a different TimeUnit
struct CastSecondsToUnitImpl {
  template <typename Duration>
  int64_t operator()(Duration, int64_t seconds) {
    auto duration = std::chrono::duration_cast<Duration>(std::chrono::seconds{seconds});
    return static_cast<int64_t>(duration.count());
  }
};

inline int64_t CastSecondsToUnit(TimeUnit::type unit, int64_t seconds) {
  return VisitDuration(unit, CastSecondsToUnitImpl{}, seconds);
}

}  // namespace util
}  // namespace arrow

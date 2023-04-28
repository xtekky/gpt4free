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
#include <string_view>

namespace arrow {
namespace csv {

/// \brief Description of an invalid row
struct InvalidRow {
  /// \brief Number of columns expected in the row
  int32_t expected_columns;
  /// \brief Actual number of columns found in the row
  int32_t actual_columns;
  /// \brief The physical row number if known or -1
  ///
  /// This number is one-based and also accounts for non-data rows (such as
  /// CSV header rows).
  int64_t number;
  /// \brief View of the entire row. Memory will be freed after callback returns
  const std::string_view text;
};

/// \brief Result returned by an InvalidRowHandler
enum class InvalidRowResult {
  // Generate an error describing this row
  Error,
  // Skip over this row
  Skip
};

/// \brief callback for handling a row with an invalid number of columns while parsing
/// \return result indicating if an error should be returned from the parser or the row is
/// skipped
using InvalidRowHandler = std::function<InvalidRowResult(const InvalidRow&)>;

}  // namespace csv
}  // namespace arrow

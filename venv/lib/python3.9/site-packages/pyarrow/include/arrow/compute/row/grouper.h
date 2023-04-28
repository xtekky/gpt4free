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
#include <vector>

#include "arrow/compute/exec.h"
#include "arrow/compute/exec/options.h"
#include "arrow/compute/kernel.h"
#include "arrow/datum.h"
#include "arrow/result.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace compute {

/// Consumes batches of keys and yields batches of the group ids.
class ARROW_EXPORT Grouper {
 public:
  virtual ~Grouper() = default;

  /// Construct a Grouper which receives the specified key types
  static Result<std::unique_ptr<Grouper>> Make(const std::vector<TypeHolder>& key_types,
                                               ExecContext* ctx = default_exec_context());

  /// Consume a batch of keys, producing the corresponding group ids as an integer array.
  /// Currently only uint32 indices will be produced, eventually the bit width will only
  /// be as wide as necessary.
  virtual Result<Datum> Consume(const ExecSpan& batch) = 0;

  /// Get current unique keys. May be called multiple times.
  virtual Result<ExecBatch> GetUniques() = 0;

  /// Get the current number of groups.
  virtual uint32_t num_groups() const = 0;

  /// \brief Assemble lists of indices of identical elements.
  ///
  /// \param[in] ids An unsigned, all-valid integral array which will be
  ///                used as grouping criteria.
  /// \param[in] num_groups An upper bound for the elements of ids
  /// \param[in] ctx Execution context to use during the operation
  /// \return A num_groups-long ListArray where the slot at i contains a
  ///         list of indices where i appears in ids.
  ///
  ///   MakeGroupings([
  ///       2,
  ///       2,
  ///       5,
  ///       5,
  ///       2,
  ///       3
  ///   ], 8) == [
  ///       [],
  ///       [],
  ///       [0, 1, 4],
  ///       [5],
  ///       [],
  ///       [2, 3],
  ///       [],
  ///       []
  ///   ]
  static Result<std::shared_ptr<ListArray>> MakeGroupings(
      const UInt32Array& ids, uint32_t num_groups,
      ExecContext* ctx = default_exec_context());

  /// \brief Produce a ListArray whose slots are selections of `array` which correspond to
  /// the provided groupings.
  ///
  /// For example,
  ///   ApplyGroupings([
  ///       [],
  ///       [],
  ///       [0, 1, 4],
  ///       [5],
  ///       [],
  ///       [2, 3],
  ///       [],
  ///       []
  ///   ], [2, 2, 5, 5, 2, 3]) == [
  ///       [],
  ///       [],
  ///       [2, 2, 2],
  ///       [3],
  ///       [],
  ///       [5, 5],
  ///       [],
  ///       []
  ///   ]
  static Result<std::shared_ptr<ListArray>> ApplyGroupings(
      const ListArray& groupings, const Array& array,
      ExecContext* ctx = default_exec_context());
};

}  // namespace compute
}  // namespace arrow

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

#include <cstdint>
#include <functional>
#include <iosfwd>
#include <memory>

#include "arrow/array/array_base.h"
#include "arrow/array/array_nested.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type.h"
#include "arrow/util/visibility.h"

namespace arrow {

/// \brief Compare two arrays, returning an edit script which expresses the difference
/// between them
///
/// An edit script is an array of struct(insert: bool, run_length: int64_t).
/// Each element of "insert" determines whether an element was inserted into (true)
/// or deleted from (false) base. Each insertion or deletion is followed by a run of
/// elements which are unchanged from base to target; the length of this run is stored
/// in "run_length". (Note that the edit script begins and ends with a run of shared
/// elements but both fields of the struct must have the same length. To accommodate this
/// the first element of "insert" should be ignored.)
///
/// For example for base "hlloo" and target "hello", the edit script would be
/// [
///   {"insert": false, "run_length": 1}, // leading run of length 1 ("h")
///   {"insert": true, "run_length": 3}, // insert("e") then a run of length 3 ("llo")
///   {"insert": false, "run_length": 0} // delete("o") then an empty run
/// ]
///
/// Diffing arrays containing nulls is not currently supported.
///
/// \param[in] base baseline for comparison
/// \param[in] target an array of identical type to base whose elements differ from base's
/// \param[in] pool memory to store the result will be allocated from this memory pool
/// \return an edit script array which can be applied to base to produce target
ARROW_EXPORT
Result<std::shared_ptr<StructArray>> Diff(const Array& base, const Array& target,
                                          MemoryPool* pool = default_memory_pool());

/// \brief visitor interface for easy traversal of an edit script
///
/// visitor will be called for each hunk of insertions and deletions.
ARROW_EXPORT Status VisitEditScript(
    const Array& edits,
    const std::function<Status(int64_t delete_begin, int64_t delete_end,
                               int64_t insert_begin, int64_t insert_end)>& visitor);

/// \brief return a function which will format an edit script in unified
/// diff format to os, given base and target arrays of type
ARROW_EXPORT Result<
    std::function<Status(const Array& edits, const Array& base, const Array& target)>>
MakeUnifiedDiffFormatter(const DataType& type, std::ostream* os);

}  // namespace arrow

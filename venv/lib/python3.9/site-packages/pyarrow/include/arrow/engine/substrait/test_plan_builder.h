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

// These utilities are for internal / unit test use only.
// They allow for the construction of simple Substrait plans
// programmatically without first requiring the construction
// of an ExecPlan

// These utilities have to be here, and not in a test_util.cc
// file (or in a unit test) because only one .so is allowed
// to include each .pb.h file or else protobuf will encounter
// global namespace conflicts.

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "arrow/engine/substrait/visibility.h"
#include "arrow/result.h"
#include "arrow/type_fwd.h"

namespace arrow {
namespace engine {

struct Id;

namespace internal {

/// \brief Create a scan->project->sink plan for tests
///
/// The plan will project one additional column using the function
/// defined by `function_id`, `arguments`, and data_types.  `arguments`
/// and `data_types` should have the same length but only one of each
/// should be defined at each index.
///
/// If `data_types` is defined at an index then the plan will create a
/// direct reference (starting at index 0 and increasing by 1 for each
/// argument of this type).
///
/// If `arguments` is defined at an index then the plan will create an
/// enum argument with that value.
ARROW_ENGINE_EXPORT Result<std::shared_ptr<Buffer>> CreateScanProjectSubstrait(
    Id function_id, const std::shared_ptr<Table>& input_table,
    const std::vector<std::string>& arguments,
    const std::unordered_map<std::string, std::vector<std::string>>& options,
    const std::vector<std::shared_ptr<DataType>>& data_types,
    const DataType& output_type);

/// \brief Create a scan->aggregate->sink plan for tests
///
/// The plan will create an aggregate with one grouping set (defined by
/// key_idxs) and one measure.  The measure will be a unary function
/// defined by `function_id` and a direct reference to `arg_idx`.
ARROW_ENGINE_EXPORT Result<std::shared_ptr<Buffer>> CreateScanAggSubstrait(
    Id function_id, const std::shared_ptr<Table>& input_table,
    const std::vector<int>& key_idxs, int arg_idx, const DataType& output_type);

}  // namespace internal
}  // namespace engine
}  // namespace arrow

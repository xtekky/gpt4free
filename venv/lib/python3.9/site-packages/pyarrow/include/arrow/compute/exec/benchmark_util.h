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

#include <cstdint>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"

#include "arrow/compute/exec.h"
#include "arrow/compute/exec/exec_plan.h"
#include "arrow/compute/exec/test_util.h"

namespace arrow {

namespace compute {

Status BenchmarkNodeOverhead(benchmark::State& state, int32_t num_batches,
                             int32_t batch_size, arrow::compute::BatchesWithSchema data,
                             std::vector<arrow::compute::Declaration>& node_declarations);

Status BenchmarkIsolatedNodeOverhead(benchmark::State& state,
                                     arrow::compute::Expression expr, int32_t num_batches,
                                     int32_t batch_size,
                                     arrow::compute::BatchesWithSchema data,
                                     std::string factory_name,
                                     arrow::compute::ExecNodeOptions& options);

}  // namespace compute
}  // namespace arrow

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

/// \brief MapNode is an ExecNode type class which process a task like filter/project
/// (See SubmitTask method) to each given ExecBatch object, which have one input, one
/// output, and are pure functions on the input
///
/// A simple parallel runner is created with a "map_fn" which is just a function that
/// takes a batch in and returns a batch.  This simple parallel runner also needs an
/// executor (use simple synchronous runner if there is no executor)

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "arrow/compute/exec/exec_plan.h"
#include "arrow/compute/exec/util.h"
#include "arrow/compute/type_fwd.h"
#include "arrow/status.h"
#include "arrow/type_fwd.h"
#include "arrow/util/cancel.h"
#include "arrow/util/type_fwd.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace compute {

class ARROW_EXPORT MapNode : public ExecNode {
 public:
  MapNode(ExecPlan* plan, std::vector<ExecNode*> inputs,
          std::shared_ptr<Schema> output_schema);

  void ErrorReceived(ExecNode* input, Status error) override;

  void InputFinished(ExecNode* input, int total_batches) override;

  Status StartProducing() override;

  void PauseProducing(ExecNode* output, int32_t counter) override;

  void ResumeProducing(ExecNode* output, int32_t counter) override;

  void StopProducing(ExecNode* output) override;

  void StopProducing() override;

 protected:
  void SubmitTask(std::function<Result<ExecBatch>(ExecBatch)> map_fn, ExecBatch batch);

  virtual void Finish(Status finish_st = Status::OK());

 protected:
  // Counter for the number of batches received
  AtomicCounter input_counter_;
};

}  // namespace compute
}  // namespace arrow

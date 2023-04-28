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
#include <memory>
#include <vector>

#include "arrow/compute/exec/options.h"
#include "arrow/record_batch.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type.h"

namespace arrow {
namespace compute {

class OrderByImpl {
 public:
  virtual ~OrderByImpl() = default;

  virtual void InputReceived(const std::shared_ptr<RecordBatch>& batch) = 0;

  virtual Result<Datum> DoFinish() = 0;

  virtual std::string ToString() const = 0;

  static Result<std::unique_ptr<OrderByImpl>> MakeSort(
      ExecContext* ctx, const std::shared_ptr<Schema>& output_schema,
      const SortOptions& options);

  static Result<std::unique_ptr<OrderByImpl>> MakeSelectK(
      ExecContext* ctx, const std::shared_ptr<Schema>& output_schema,
      const SelectKOptions& options);
};

}  // namespace compute
}  // namespace arrow

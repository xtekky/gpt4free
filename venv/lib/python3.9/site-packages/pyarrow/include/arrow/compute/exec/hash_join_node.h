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

#include <vector>

#include "arrow/compute/exec/options.h"
#include "arrow/compute/exec/schema_util.h"
#include "arrow/result.h"
#include "arrow/status.h"

namespace arrow {
namespace compute {

class ARROW_EXPORT HashJoinSchema {
 public:
  Status Init(JoinType join_type, const Schema& left_schema,
              const std::vector<FieldRef>& left_keys, const Schema& right_schema,
              const std::vector<FieldRef>& right_keys, const Expression& filter,
              const std::string& left_field_name_prefix,
              const std::string& right_field_name_prefix);

  Status Init(JoinType join_type, const Schema& left_schema,
              const std::vector<FieldRef>& left_keys,
              const std::vector<FieldRef>& left_output, const Schema& right_schema,
              const std::vector<FieldRef>& right_keys,
              const std::vector<FieldRef>& right_output, const Expression& filter,
              const std::string& left_field_name_prefix,
              const std::string& right_field_name_prefix);

  static Status ValidateSchemas(JoinType join_type, const Schema& left_schema,
                                const std::vector<FieldRef>& left_keys,
                                const std::vector<FieldRef>& left_output,
                                const Schema& right_schema,
                                const std::vector<FieldRef>& right_keys,
                                const std::vector<FieldRef>& right_output,
                                const std::string& left_field_name_prefix,
                                const std::string& right_field_name_prefix);

  bool HasDictionaries() const;

  bool HasLargeBinary() const;

  Result<Expression> BindFilter(Expression filter, const Schema& left_schema,
                                const Schema& right_schema, ExecContext* exec_context);
  std::shared_ptr<Schema> MakeOutputSchema(const std::string& left_field_name_suffix,
                                           const std::string& right_field_name_suffix);

  bool LeftPayloadIsEmpty() { return PayloadIsEmpty(0); }

  bool RightPayloadIsEmpty() { return PayloadIsEmpty(1); }

  static int kMissingField() {
    return SchemaProjectionMaps<HashJoinProjection>::kMissingField;
  }

  SchemaProjectionMaps<HashJoinProjection> proj_maps[2];

 private:
  static bool IsTypeSupported(const DataType& type);

  Status CollectFilterColumns(std::vector<FieldRef>& left_filter,
                              std::vector<FieldRef>& right_filter,
                              const Expression& filter, const Schema& left_schema,
                              const Schema& right_schema);

  Expression RewriteFilterToUseFilterSchema(int right_filter_offset,
                                            const SchemaProjectionMap& left_to_filter,
                                            const SchemaProjectionMap& right_to_filter,
                                            const Expression& filter);

  bool PayloadIsEmpty(int side) {
    ARROW_DCHECK(side == 0 || side == 1);
    return proj_maps[side].num_cols(HashJoinProjection::PAYLOAD) == 0;
  }

  static Result<std::vector<FieldRef>> ComputePayload(const Schema& schema,
                                                      const std::vector<FieldRef>& output,
                                                      const std::vector<FieldRef>& filter,
                                                      const std::vector<FieldRef>& key);
};

}  // namespace compute
}  // namespace arrow

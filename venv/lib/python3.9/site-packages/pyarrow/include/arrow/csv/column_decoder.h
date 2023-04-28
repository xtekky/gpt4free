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
#include <memory>
#include <utility>

#include "arrow/result.h"
#include "arrow/type_fwd.h"
#include "arrow/util/type_fwd.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace csv {

class BlockParser;
struct ConvertOptions;

class ARROW_EXPORT ColumnDecoder {
 public:
  virtual ~ColumnDecoder() = default;

  /// Spawn a task that will try to convert and insert the given CSV block
  virtual Future<std::shared_ptr<Array>> Decode(
      const std::shared_ptr<BlockParser>& parser) = 0;

  /// Construct a strictly-typed ColumnDecoder.
  static Result<std::shared_ptr<ColumnDecoder>> Make(MemoryPool* pool,
                                                     std::shared_ptr<DataType> type,
                                                     int32_t col_index,
                                                     const ConvertOptions& options);

  /// Construct a type-inferring ColumnDecoder.
  /// Inference will run only on the first block, the type will be frozen afterwards.
  static Result<std::shared_ptr<ColumnDecoder>> Make(MemoryPool* pool, int32_t col_index,
                                                     const ConvertOptions& options);

  /// Construct a ColumnDecoder for a column of nulls
  /// (i.e. not present in the CSV file).
  static Result<std::shared_ptr<ColumnDecoder>> MakeNull(MemoryPool* pool,
                                                         std::shared_ptr<DataType> type);

 protected:
  ColumnDecoder() = default;
};

}  // namespace csv
}  // namespace arrow

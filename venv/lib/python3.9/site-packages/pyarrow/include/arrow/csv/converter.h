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

#include "arrow/csv/options.h"
#include "arrow/result.h"
#include "arrow/type_fwd.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace csv {

class BlockParser;

class ARROW_EXPORT Converter {
 public:
  Converter(const std::shared_ptr<DataType>& type, const ConvertOptions& options,
            MemoryPool* pool);
  virtual ~Converter() = default;

  virtual Result<std::shared_ptr<Array>> Convert(const BlockParser& parser,
                                                 int32_t col_index) = 0;

  std::shared_ptr<DataType> type() const { return type_; }

  // Create a Converter for the given data type
  static Result<std::shared_ptr<Converter>> Make(
      const std::shared_ptr<DataType>& type, const ConvertOptions& options,
      MemoryPool* pool = default_memory_pool());

 protected:
  ARROW_DISALLOW_COPY_AND_ASSIGN(Converter);

  virtual Status Initialize() = 0;

  // CAUTION: ConvertOptions can grow large (if it customizes hundreds or
  // thousands of columns), so avoid copying it in each Converter.
  const ConvertOptions& options_;
  MemoryPool* pool_;
  std::shared_ptr<DataType> type_;
};

class ARROW_EXPORT DictionaryConverter : public Converter {
 public:
  DictionaryConverter(const std::shared_ptr<DataType>& value_type,
                      const ConvertOptions& options, MemoryPool* pool);

  // If the dictionary length goes above this value, conversion will fail
  // with Status::IndexError.
  virtual void SetMaxCardinality(int32_t max_length) = 0;

  // Create a Converter for the given dictionary value type.
  // The dictionary index type will always be Int32.
  static Result<std::shared_ptr<DictionaryConverter>> Make(
      const std::shared_ptr<DataType>& value_type, const ConvertOptions& options,
      MemoryPool* pool = default_memory_pool());

 protected:
  std::shared_ptr<DataType> value_type_;
};

}  // namespace csv
}  // namespace arrow

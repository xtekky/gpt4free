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
#include <string>

#include "arrow/status.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {

class Array;
class DataType;
class Field;
class MemoryPool;

namespace json {

/// \brief interface for conversion of Arrays
///
/// Converters are not required to be correct for arbitrary input- only
/// for unconverted arrays emitted by a corresponding parser.
class ARROW_EXPORT Converter {
 public:
  virtual ~Converter() = default;

  /// convert an array
  /// on failure, this converter may be promoted to another converter which
  /// *can* convert the given input.
  virtual Status Convert(const std::shared_ptr<Array>& in,
                         std::shared_ptr<Array>* out) = 0;

  std::shared_ptr<DataType> out_type() const { return out_type_; }

  MemoryPool* pool() { return pool_; }

 protected:
  ARROW_DISALLOW_COPY_AND_ASSIGN(Converter);

  Converter(MemoryPool* pool, const std::shared_ptr<DataType>& out_type)
      : pool_(pool), out_type_(out_type) {}

  MemoryPool* pool_;
  std::shared_ptr<DataType> out_type_;
};

/// \brief produce a single converter to the specified out_type
ARROW_EXPORT Status MakeConverter(const std::shared_ptr<DataType>& out_type,
                                  MemoryPool* pool, std::shared_ptr<Converter>* out);

class ARROW_EXPORT PromotionGraph {
 public:
  virtual ~PromotionGraph() = default;

  /// \brief produce a valid field which will be inferred as null
  virtual std::shared_ptr<Field> Null(const std::string& name) const = 0;

  /// \brief given an unexpected field encountered during parsing, return a type to which
  /// it may be convertible (may return null if none is available)
  virtual std::shared_ptr<DataType> Infer(
      const std::shared_ptr<Field>& unexpected_field) const = 0;

  /// \brief given a type to which conversion failed, return a promoted type to which
  /// conversion may succeed (may return null if none is available)
  virtual std::shared_ptr<DataType> Promote(
      const std::shared_ptr<DataType>& failed,
      const std::shared_ptr<Field>& unexpected_field) const = 0;

 protected:
  ARROW_DISALLOW_COPY_AND_ASSIGN(PromotionGraph);
  PromotionGraph() = default;
};

ARROW_EXPORT const PromotionGraph* GetPromotionGraph();

}  // namespace json
}  // namespace arrow

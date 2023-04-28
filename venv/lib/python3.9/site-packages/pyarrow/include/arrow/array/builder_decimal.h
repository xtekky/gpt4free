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

#include "arrow/array/array_decimal.h"
#include "arrow/array/builder_base.h"
#include "arrow/array/builder_binary.h"
#include "arrow/array/data.h"
#include "arrow/status.h"
#include "arrow/type.h"
#include "arrow/util/visibility.h"

namespace arrow {

/// \addtogroup numeric-builders
///
/// @{

class ARROW_EXPORT Decimal128Builder : public FixedSizeBinaryBuilder {
 public:
  using TypeClass = Decimal128Type;
  using ValueType = Decimal128;

  explicit Decimal128Builder(const std::shared_ptr<DataType>& type,
                             MemoryPool* pool = default_memory_pool(),
                             int64_t alignment = kDefaultBufferAlignment);

  using FixedSizeBinaryBuilder::Append;
  using FixedSizeBinaryBuilder::AppendValues;
  using FixedSizeBinaryBuilder::Reset;

  Status Append(Decimal128 val);
  void UnsafeAppend(Decimal128 val);
  void UnsafeAppend(std::string_view val);

  Status FinishInternal(std::shared_ptr<ArrayData>* out) override;

  /// \cond FALSE
  using ArrayBuilder::Finish;
  /// \endcond

  Status Finish(std::shared_ptr<Decimal128Array>* out) { return FinishTyped(out); }

  std::shared_ptr<DataType> type() const override { return decimal_type_; }

 protected:
  std::shared_ptr<Decimal128Type> decimal_type_;
};

class ARROW_EXPORT Decimal256Builder : public FixedSizeBinaryBuilder {
 public:
  using TypeClass = Decimal256Type;
  using ValueType = Decimal256;

  explicit Decimal256Builder(const std::shared_ptr<DataType>& type,
                             MemoryPool* pool = default_memory_pool(),
                             int64_t alignment = kDefaultBufferAlignment);

  using FixedSizeBinaryBuilder::Append;
  using FixedSizeBinaryBuilder::AppendValues;
  using FixedSizeBinaryBuilder::Reset;

  Status Append(const Decimal256& val);
  void UnsafeAppend(const Decimal256& val);
  void UnsafeAppend(std::string_view val);

  Status FinishInternal(std::shared_ptr<ArrayData>* out) override;

  /// \cond FALSE
  using ArrayBuilder::Finish;
  /// \endcond

  Status Finish(std::shared_ptr<Decimal256Array>* out) { return FinishTyped(out); }

  std::shared_ptr<DataType> type() const override { return decimal_type_; }

 protected:
  std::shared_ptr<Decimal256Type> decimal_type_;
};

using DecimalBuilder = Decimal128Builder;

/// @}

}  // namespace arrow

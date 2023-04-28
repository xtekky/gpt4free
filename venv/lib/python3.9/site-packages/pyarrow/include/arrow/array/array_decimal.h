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
#include <string>

#include "arrow/array/array_binary.h"
#include "arrow/array/data.h"
#include "arrow/type.h"
#include "arrow/util/visibility.h"

namespace arrow {

/// \addtogroup numeric-arrays
///
/// @{

// ----------------------------------------------------------------------
// Decimal128Array

/// Concrete Array class for 128-bit decimal data
class ARROW_EXPORT Decimal128Array : public FixedSizeBinaryArray {
 public:
  using TypeClass = Decimal128Type;

  using FixedSizeBinaryArray::FixedSizeBinaryArray;

  /// \brief Construct Decimal128Array from ArrayData instance
  explicit Decimal128Array(const std::shared_ptr<ArrayData>& data);

  std::string FormatValue(int64_t i) const;
};

// Backward compatibility
using DecimalArray = Decimal128Array;

// ----------------------------------------------------------------------
// Decimal256Array

/// Concrete Array class for 256-bit decimal data
class ARROW_EXPORT Decimal256Array : public FixedSizeBinaryArray {
 public:
  using TypeClass = Decimal256Type;

  using FixedSizeBinaryArray::FixedSizeBinaryArray;

  /// \brief Construct Decimal256Array from ArrayData instance
  explicit Decimal256Array(const std::shared_ptr<ArrayData>& data);

  std::string FormatValue(int64_t i) const;
};

/// @}

}  // namespace arrow

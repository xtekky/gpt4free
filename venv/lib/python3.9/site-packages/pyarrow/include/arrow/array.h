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

// Kitchen-sink public API for arrow::Array data structures. C++ library code
// (especially header files) in Apache Arrow should use more specific headers
// unless it's a file that uses most or all Array types in which case using
// arrow/array.h is fine.

#pragma once

/// \defgroup numeric-arrays Concrete classes for numeric arrays
/// @{
/// @}

/// \defgroup binary-arrays Concrete classes for binary/string arrays
/// @{
/// @}

/// \defgroup nested-arrays Concrete classes for nested arrays
/// @{
/// @}

#include "arrow/array/array_base.h"       // IWYU pragma: keep
#include "arrow/array/array_binary.h"     // IWYU pragma: keep
#include "arrow/array/array_decimal.h"    // IWYU pragma: keep
#include "arrow/array/array_dict.h"       // IWYU pragma: keep
#include "arrow/array/array_nested.h"     // IWYU pragma: keep
#include "arrow/array/array_primitive.h"  // IWYU pragma: keep
#include "arrow/array/data.h"             // IWYU pragma: keep
#include "arrow/array/util.h"             // IWYU pragma: keep

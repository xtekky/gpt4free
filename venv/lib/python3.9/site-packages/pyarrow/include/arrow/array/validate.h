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

#include "arrow/status.h"
#include "arrow/type_fwd.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace internal {

// Internal functions implementing Array::Validate() and friends.

// O(1) array metadata validation

ARROW_EXPORT
Status ValidateArray(const Array& array);

ARROW_EXPORT
Status ValidateArray(const ArrayData& data);

// O(N) array data validation.
// Note that, starting from 7.0.0, "full" routines also validate metadata.
// Before, ValidateArray() needed to be called before ValidateArrayFull()
// to ensure metadata correctness, otherwise invalid memory accesses
// may occur.

ARROW_EXPORT
Status ValidateArrayFull(const Array& array);

ARROW_EXPORT
Status ValidateArrayFull(const ArrayData& data);

ARROW_EXPORT
Status ValidateUTF8(const Array& array);

ARROW_EXPORT
Status ValidateUTF8(const ArrayData& data);

}  // namespace internal
}  // namespace arrow

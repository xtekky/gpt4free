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

#include "arrow/type_fwd.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {

/// \brief Concatenate arrays
///
/// \param[in] arrays a vector of arrays to be concatenated
/// \param[in] pool memory to store the result will be allocated from this memory pool
/// \return the concatenated array
ARROW_EXPORT
Result<std::shared_ptr<Array>> Concatenate(const ArrayVector& arrays,
                                           MemoryPool* pool = default_memory_pool());

}  // namespace arrow

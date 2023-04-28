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
#include <vector>

#include "arrow/result.h"
#include "arrow/type_fwd.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace internal {

/// \brief Generate Bitmap with all position to `value` except for one found
/// at `straggler_pos`.
ARROW_EXPORT
Result<std::shared_ptr<Buffer>> BitmapAllButOne(MemoryPool* pool, int64_t length,
                                                int64_t straggler_pos, bool value = true);

/// \brief Convert vector of bytes to bitmap buffer
ARROW_EXPORT
Result<std::shared_ptr<Buffer>> BytesToBits(const std::vector<uint8_t>&,
                                            MemoryPool* pool = default_memory_pool());

}  // namespace internal
}  // namespace arrow

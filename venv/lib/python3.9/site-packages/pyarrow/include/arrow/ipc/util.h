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

namespace arrow {
namespace ipc {

// Buffers are padded to 64-byte boundaries (for SIMD)
static constexpr int32_t kArrowAlignment = 64;

// Tensors are padded to 64-byte boundaries
static constexpr int32_t kTensorAlignment = 64;

// Align on 8-byte boundaries in IPC
static constexpr int32_t kArrowIpcAlignment = 8;

static constexpr uint8_t kPaddingBytes[kArrowAlignment] = {0};

static inline int64_t PaddedLength(int64_t nbytes, int32_t alignment = kArrowAlignment) {
  return ((nbytes + alignment - 1) / alignment) * alignment;
}

}  // namespace ipc
}  // namespace arrow

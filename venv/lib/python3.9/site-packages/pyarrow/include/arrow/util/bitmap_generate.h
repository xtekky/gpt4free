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

#include "arrow/buffer.h"
#include "arrow/memory_pool.h"
#include "arrow/result.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace internal {

// A std::generate() like function to write sequential bits into a bitmap area.
// Bits preceding the bitmap area are preserved, bits following the bitmap
// area may be clobbered.

template <class Generator>
void GenerateBits(uint8_t* bitmap, int64_t start_offset, int64_t length, Generator&& g) {
  if (length == 0) {
    return;
  }
  uint8_t* cur = bitmap + start_offset / 8;
  uint8_t bit_mask = bit_util::kBitmask[start_offset % 8];
  uint8_t current_byte = *cur & bit_util::kPrecedingBitmask[start_offset % 8];

  for (int64_t index = 0; index < length; ++index) {
    const bool bit = g();
    current_byte = bit ? (current_byte | bit_mask) : current_byte;
    bit_mask = static_cast<uint8_t>(bit_mask << 1);
    if (bit_mask == 0) {
      bit_mask = 1;
      *cur++ = current_byte;
      current_byte = 0;
    }
  }
  if (bit_mask != 1) {
    *cur++ = current_byte;
  }
}

// Like GenerateBits(), but unrolls its main loop for higher performance.

template <class Generator>
void GenerateBitsUnrolled(uint8_t* bitmap, int64_t start_offset, int64_t length,
                          Generator&& g) {
  static_assert(std::is_same<decltype(std::declval<Generator>()()), bool>::value,
                "Functor passed to GenerateBitsUnrolled must return bool");

  if (length == 0) {
    return;
  }
  uint8_t current_byte;
  uint8_t* cur = bitmap + start_offset / 8;
  const uint64_t start_bit_offset = start_offset % 8;
  uint8_t bit_mask = bit_util::kBitmask[start_bit_offset];
  int64_t remaining = length;

  if (bit_mask != 0x01) {
    current_byte = *cur & bit_util::kPrecedingBitmask[start_bit_offset];
    while (bit_mask != 0 && remaining > 0) {
      current_byte |= g() * bit_mask;
      bit_mask = static_cast<uint8_t>(bit_mask << 1);
      --remaining;
    }
    *cur++ = current_byte;
  }

  int64_t remaining_bytes = remaining / 8;
  uint8_t out_results[8];
  while (remaining_bytes-- > 0) {
    for (int i = 0; i < 8; ++i) {
      out_results[i] = g();
    }
    *cur++ = (out_results[0] | out_results[1] << 1 | out_results[2] << 2 |
              out_results[3] << 3 | out_results[4] << 4 | out_results[5] << 5 |
              out_results[6] << 6 | out_results[7] << 7);
  }

  int64_t remaining_bits = remaining % 8;
  if (remaining_bits) {
    current_byte = 0;
    bit_mask = 0x01;
    while (remaining_bits-- > 0) {
      current_byte |= g() * bit_mask;
      bit_mask = static_cast<uint8_t>(bit_mask << 1);
    }
    *cur++ = current_byte;
  }
}

}  // namespace internal
}  // namespace arrow

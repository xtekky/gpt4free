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

#include <algorithm>

#include "arrow/util/bit_util.h"

namespace arrow {
namespace internal {

struct BitmapWordAlignParams {
  int64_t leading_bits;
  int64_t trailing_bits;
  int64_t trailing_bit_offset;
  const uint8_t* aligned_start;
  int64_t aligned_bits;
  int64_t aligned_words;
};

// Compute parameters for accessing a bitmap using aligned word instructions.
// The returned parameters describe:
// - a leading area of size `leading_bits` before the aligned words
// - a word-aligned area of size `aligned_bits`
// - a trailing area of size `trailing_bits` after the aligned words
template <uint64_t ALIGN_IN_BYTES>
inline BitmapWordAlignParams BitmapWordAlign(const uint8_t* data, int64_t bit_offset,
                                             int64_t length) {
  static_assert(bit_util::IsPowerOf2(ALIGN_IN_BYTES),
                "ALIGN_IN_BYTES should be a positive power of two");
  constexpr uint64_t ALIGN_IN_BITS = ALIGN_IN_BYTES * 8;

  BitmapWordAlignParams p;

  // Compute a "bit address" that we can align up to ALIGN_IN_BITS.
  // We don't care about losing the upper bits since we are only interested in the
  // difference between both addresses.
  const uint64_t bit_addr =
      reinterpret_cast<size_t>(data) * 8 + static_cast<uint64_t>(bit_offset);
  const uint64_t aligned_bit_addr = bit_util::RoundUpToPowerOf2(bit_addr, ALIGN_IN_BITS);

  p.leading_bits = std::min<int64_t>(length, aligned_bit_addr - bit_addr);
  p.aligned_words = (length - p.leading_bits) / ALIGN_IN_BITS;
  p.aligned_bits = p.aligned_words * ALIGN_IN_BITS;
  p.trailing_bits = length - p.leading_bits - p.aligned_bits;
  p.trailing_bit_offset = bit_offset + p.leading_bits + p.aligned_bits;

  p.aligned_start = data + (bit_offset + p.leading_bits) / 8;
  return p;
}

}  // namespace internal
}  // namespace arrow

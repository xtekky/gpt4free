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

#include "arrow/util/bit_util.h"
#include "arrow/util/bitmap_reader.h"

namespace arrow {
namespace internal {

// A function that visits each bit in a bitmap and calls a visitor function with a
// boolean representation of that bit. This is intended to be analogous to
// GenerateBits.
template <class Visitor>
void VisitBits(const uint8_t* bitmap, int64_t start_offset, int64_t length,
               Visitor&& visit) {
  BitmapReader reader(bitmap, start_offset, length);
  for (int64_t index = 0; index < length; ++index) {
    visit(reader.IsSet());
    reader.Next();
  }
}

// Like VisitBits(), but unrolls its main loop for better performance.
template <class Visitor>
void VisitBitsUnrolled(const uint8_t* bitmap, int64_t start_offset, int64_t length,
                       Visitor&& visit) {
  if (length == 0) {
    return;
  }

  // Start by visiting any bits preceding the first full byte.
  int64_t num_bits_before_full_bytes =
      bit_util::RoundUpToMultipleOf8(start_offset) - start_offset;
  // Truncate num_bits_before_full_bytes if it is greater than length.
  if (num_bits_before_full_bytes > length) {
    num_bits_before_full_bytes = length;
  }
  // Use the non loop-unrolled VisitBits since we don't want to add branches
  VisitBits<Visitor>(bitmap, start_offset, num_bits_before_full_bytes, visit);

  // Shift the start pointer to the first full byte and compute the
  // number of full bytes to be read.
  const uint8_t* first_full_byte = bitmap + bit_util::CeilDiv(start_offset, 8);
  const int64_t num_full_bytes = (length - num_bits_before_full_bytes) / 8;

  // Iterate over each full byte of the input bitmap and call the visitor in
  // a loop-unrolled manner.
  for (int64_t byte_index = 0; byte_index < num_full_bytes; ++byte_index) {
    // Get the current bit-packed byte value from the bitmap.
    const uint8_t byte = *(first_full_byte + byte_index);

    // Execute the visitor function on each bit of the current byte.
    visit(bit_util::GetBitFromByte(byte, 0));
    visit(bit_util::GetBitFromByte(byte, 1));
    visit(bit_util::GetBitFromByte(byte, 2));
    visit(bit_util::GetBitFromByte(byte, 3));
    visit(bit_util::GetBitFromByte(byte, 4));
    visit(bit_util::GetBitFromByte(byte, 5));
    visit(bit_util::GetBitFromByte(byte, 6));
    visit(bit_util::GetBitFromByte(byte, 7));
  }

  // Write any leftover bits in the last byte.
  const int64_t num_bits_after_full_bytes = (length - num_bits_before_full_bytes) % 8;
  VisitBits<Visitor>(first_full_byte + num_full_bytes, 0, num_bits_after_full_bytes,
                     visit);
}

}  // namespace internal
}  // namespace arrow

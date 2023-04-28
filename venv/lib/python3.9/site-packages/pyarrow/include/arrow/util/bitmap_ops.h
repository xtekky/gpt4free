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

#include "arrow/result.h"
#include "arrow/util/visibility.h"

namespace arrow {

class Buffer;
class MemoryPool;

namespace internal {

// ----------------------------------------------------------------------
// Bitmap utilities

/// Copy a bit range of an existing bitmap
///
/// \param[in] pool memory pool to allocate memory from
/// \param[in] bitmap source data
/// \param[in] offset bit offset into the source data
/// \param[in] length number of bits to copy
///
/// \return Status message
ARROW_EXPORT
Result<std::shared_ptr<Buffer>> CopyBitmap(MemoryPool* pool, const uint8_t* bitmap,
                                           int64_t offset, int64_t length);

/// Copy a bit range of an existing bitmap into an existing bitmap
///
/// \param[in] bitmap source data
/// \param[in] offset bit offset into the source data
/// \param[in] length number of bits to copy
/// \param[in] dest_offset bit offset into the destination
/// \param[out] dest the destination buffer, must have at least space for
/// (offset + length) bits
ARROW_EXPORT
void CopyBitmap(const uint8_t* bitmap, int64_t offset, int64_t length, uint8_t* dest,
                int64_t dest_offset);

/// Invert a bit range of an existing bitmap into an existing bitmap
///
/// \param[in] bitmap source data
/// \param[in] offset bit offset into the source data
/// \param[in] length number of bits to copy
/// \param[in] dest_offset bit offset into the destination
/// \param[out] dest the destination buffer, must have at least space for
/// (offset + length) bits
ARROW_EXPORT
void InvertBitmap(const uint8_t* bitmap, int64_t offset, int64_t length, uint8_t* dest,
                  int64_t dest_offset);

/// Invert a bit range of an existing bitmap
///
/// \param[in] pool memory pool to allocate memory from
/// \param[in] bitmap source data
/// \param[in] offset bit offset into the source data
/// \param[in] length number of bits to copy
///
/// \return Status message
ARROW_EXPORT
Result<std::shared_ptr<Buffer>> InvertBitmap(MemoryPool* pool, const uint8_t* bitmap,
                                             int64_t offset, int64_t length);

/// Reverse a bit range of an existing bitmap into an existing bitmap
///
/// \param[in] bitmap source data
/// \param[in] offset bit offset into the source data
/// \param[in] length number of bits to reverse
/// \param[in] dest_offset bit offset into the destination
/// \param[out] dest the destination buffer, must have at least space for
/// (offset + length) bits
ARROW_EXPORT
void ReverseBitmap(const uint8_t* bitmap, int64_t offset, int64_t length, uint8_t* dest,
                   int64_t dest_offset);

/// Reverse a bit range of an existing bitmap
///
/// \param[in] pool memory pool to allocate memory from
/// \param[in] bitmap source data
/// \param[in] offset bit offset into the source data
/// \param[in] length number of bits to reverse
///
/// \return Status message
ARROW_EXPORT
Result<std::shared_ptr<Buffer>> ReverseBitmap(MemoryPool* pool, const uint8_t* bitmap,
                                              int64_t offset, int64_t length);

/// Compute the number of 1's in the given data array
///
/// \param[in] data a packed LSB-ordered bitmap as a byte array
/// \param[in] bit_offset a bitwise offset into the bitmap
/// \param[in] length the number of bits to inspect in the bitmap relative to
/// the offset
///
/// \return The number of set (1) bits in the range
ARROW_EXPORT
int64_t CountSetBits(const uint8_t* data, int64_t bit_offset, int64_t length);

/// Compute the number of 1's in the result of an "and" (&) of two bitmaps
///
/// \param[in] left_bitmap a packed LSB-ordered bitmap as a byte array
/// \param[in] left_offset a bitwise offset into the left bitmap
/// \param[in] right_bitmap a packed LSB-ordered bitmap as a byte array
/// \param[in] right_offset a bitwise offset into the right bitmap
/// \param[in] length the length of the bitmaps (must be the same)
///
/// \return The number of set (1) bits in the "and" of the two bitmaps
ARROW_EXPORT
int64_t CountAndSetBits(const uint8_t* left_bitmap, int64_t left_offset,
                        const uint8_t* right_bitmap, int64_t right_offset,
                        int64_t length);

ARROW_EXPORT
bool BitmapEquals(const uint8_t* left, int64_t left_offset, const uint8_t* right,
                  int64_t right_offset, int64_t length);

// Same as BitmapEquals, but considers a NULL bitmap pointer the same as an
// all-ones bitmap.
ARROW_EXPORT
bool OptionalBitmapEquals(const uint8_t* left, int64_t left_offset, const uint8_t* right,
                          int64_t right_offset, int64_t length);

ARROW_EXPORT
bool OptionalBitmapEquals(const std::shared_ptr<Buffer>& left, int64_t left_offset,
                          const std::shared_ptr<Buffer>& right, int64_t right_offset,
                          int64_t length);

/// \brief Do a "bitmap and" on right and left buffers starting at
/// their respective bit-offsets for the given bit-length and put
/// the results in out_buffer starting at the given bit-offset.
///
/// out_buffer will be allocated and initialized to zeros using pool before
/// the operation.
ARROW_EXPORT
Result<std::shared_ptr<Buffer>> BitmapAnd(MemoryPool* pool, const uint8_t* left,
                                          int64_t left_offset, const uint8_t* right,
                                          int64_t right_offset, int64_t length,
                                          int64_t out_offset);

/// \brief Do a "bitmap and" on right and left buffers starting at
/// their respective bit-offsets for the given bit-length and put
/// the results in out starting at the given bit-offset.
ARROW_EXPORT
void BitmapAnd(const uint8_t* left, int64_t left_offset, const uint8_t* right,
               int64_t right_offset, int64_t length, int64_t out_offset, uint8_t* out);

/// \brief Do a "bitmap or" for the given bit length on right and left buffers
/// starting at their respective bit-offsets and put the results in out_buffer
/// starting at the given bit-offset.
///
/// out_buffer will be allocated and initialized to zeros using pool before
/// the operation.
ARROW_EXPORT
Result<std::shared_ptr<Buffer>> BitmapOr(MemoryPool* pool, const uint8_t* left,
                                         int64_t left_offset, const uint8_t* right,
                                         int64_t right_offset, int64_t length,
                                         int64_t out_offset);

/// \brief Do a "bitmap or" for the given bit length on right and left buffers
/// starting at their respective bit-offsets and put the results in out
/// starting at the given bit-offset.
ARROW_EXPORT
void BitmapOr(const uint8_t* left, int64_t left_offset, const uint8_t* right,
              int64_t right_offset, int64_t length, int64_t out_offset, uint8_t* out);

/// \brief Do a "bitmap xor" for the given bit-length on right and left
/// buffers starting at their respective bit-offsets and put the results in
/// out_buffer starting at the given bit offset.
///
/// out_buffer will be allocated and initialized to zeros using pool before
/// the operation.
ARROW_EXPORT
Result<std::shared_ptr<Buffer>> BitmapXor(MemoryPool* pool, const uint8_t* left,
                                          int64_t left_offset, const uint8_t* right,
                                          int64_t right_offset, int64_t length,
                                          int64_t out_offset);

/// \brief Do a "bitmap xor" for the given bit-length on right and left
/// buffers starting at their respective bit-offsets and put the results in
/// out starting at the given bit offset.
ARROW_EXPORT
void BitmapXor(const uint8_t* left, int64_t left_offset, const uint8_t* right,
               int64_t right_offset, int64_t length, int64_t out_offset, uint8_t* out);

/// \brief Do a "bitmap and not" on right and left buffers starting at
/// their respective bit-offsets for the given bit-length and put
/// the results in out_buffer starting at the given bit-offset.
///
/// out_buffer will be allocated and initialized to zeros using pool before
/// the operation.
ARROW_EXPORT
Result<std::shared_ptr<Buffer>> BitmapAndNot(MemoryPool* pool, const uint8_t* left,
                                             int64_t left_offset, const uint8_t* right,
                                             int64_t right_offset, int64_t length,
                                             int64_t out_offset);

/// \brief Do a "bitmap and not" on right and left buffers starting at
/// their respective bit-offsets for the given bit-length and put
/// the results in out starting at the given bit-offset.
ARROW_EXPORT
void BitmapAndNot(const uint8_t* left, int64_t left_offset, const uint8_t* right,
                  int64_t right_offset, int64_t length, int64_t out_offset, uint8_t* out);

/// \brief Do a "bitmap or not" on right and left buffers starting at
/// their respective bit-offsets for the given bit-length and put
/// the results in out_buffer starting at the given bit-offset.
///
/// out_buffer will be allocated and initialized to zeros using pool before
/// the operation.
ARROW_EXPORT
Result<std::shared_ptr<Buffer>> BitmapOrNot(MemoryPool* pool, const uint8_t* left,
                                            int64_t left_offset, const uint8_t* right,
                                            int64_t right_offset, int64_t length,
                                            int64_t out_offset);

/// \brief Do a "bitmap or not" on right and left buffers starting at
/// their respective bit-offsets for the given bit-length and put
/// the results in out starting at the given bit-offset.
ARROW_EXPORT
void BitmapOrNot(const uint8_t* left, int64_t left_offset, const uint8_t* right,
                 int64_t right_offset, int64_t length, int64_t out_offset, uint8_t* out);

}  // namespace internal
}  // namespace arrow

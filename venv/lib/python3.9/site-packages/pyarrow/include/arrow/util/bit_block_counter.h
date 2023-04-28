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
#include <cstdint>
#include <limits>
#include <memory>

#include "arrow/buffer.h"
#include "arrow/status.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/endian.h"
#include "arrow/util/macros.h"
#include "arrow/util/ubsan.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace internal {
namespace detail {

inline uint64_t LoadWord(const uint8_t* bytes) {
  return bit_util::ToLittleEndian(util::SafeLoadAs<uint64_t>(bytes));
}

inline uint64_t ShiftWord(uint64_t current, uint64_t next, int64_t shift) {
  if (shift == 0) {
    return current;
  }
  return (current >> shift) | (next << (64 - shift));
}

// These templates are here to help with unit tests

template <typename T>
constexpr T BitNot(T x) {
  return ~x;
}

template <>
constexpr bool BitNot(bool x) {
  return !x;
}

struct BitBlockAnd {
  template <typename T>
  static constexpr T Call(T left, T right) {
    return left & right;
  }
};

struct BitBlockAndNot {
  template <typename T>
  static constexpr T Call(T left, T right) {
    return left & BitNot(right);
  }
};

struct BitBlockOr {
  template <typename T>
  static constexpr T Call(T left, T right) {
    return left | right;
  }
};

struct BitBlockOrNot {
  template <typename T>
  static constexpr T Call(T left, T right) {
    return left | BitNot(right);
  }
};

}  // namespace detail

/// \brief Return value from bit block counters: the total number of bits and
/// the number of set bits.
struct BitBlockCount {
  int16_t length;
  int16_t popcount;

  bool NoneSet() const { return this->popcount == 0; }
  bool AllSet() const { return this->length == this->popcount; }
};

/// \brief A class that scans through a true/false bitmap to compute popcounts
/// 64 or 256 bits at a time. This is used to accelerate processing of
/// mostly-not-null array data.
class ARROW_EXPORT BitBlockCounter {
 public:
  BitBlockCounter(const uint8_t* bitmap, int64_t start_offset, int64_t length)
      : bitmap_(util::MakeNonNull(bitmap) + start_offset / 8),
        bits_remaining_(length),
        offset_(start_offset % 8) {}

  /// \brief The bit size of each word run
  static constexpr int64_t kWordBits = 64;

  /// \brief The bit size of four words run
  static constexpr int64_t kFourWordsBits = kWordBits * 4;

  /// \brief Return the next run of available bits, usually 256. The returned
  /// pair contains the size of run and the number of true values. The last
  /// block will have a length less than 256 if the bitmap length is not a
  /// multiple of 256, and will return 0-length blocks in subsequent
  /// invocations.
  BitBlockCount NextFourWords() {
    using detail::LoadWord;
    using detail::ShiftWord;

    if (!bits_remaining_) {
      return {0, 0};
    }
    int64_t total_popcount = 0;
    if (offset_ == 0) {
      if (bits_remaining_ < kFourWordsBits) {
        return GetBlockSlow(kFourWordsBits);
      }
      total_popcount += bit_util::PopCount(LoadWord(bitmap_));
      total_popcount += bit_util::PopCount(LoadWord(bitmap_ + 8));
      total_popcount += bit_util::PopCount(LoadWord(bitmap_ + 16));
      total_popcount += bit_util::PopCount(LoadWord(bitmap_ + 24));
    } else {
      // When the offset is > 0, we need there to be a word beyond the last
      // aligned word in the bitmap for the bit shifting logic.
      if (bits_remaining_ < 5 * kFourWordsBits - offset_) {
        return GetBlockSlow(kFourWordsBits);
      }
      auto current = LoadWord(bitmap_);
      auto next = LoadWord(bitmap_ + 8);
      total_popcount += bit_util::PopCount(ShiftWord(current, next, offset_));
      current = next;
      next = LoadWord(bitmap_ + 16);
      total_popcount += bit_util::PopCount(ShiftWord(current, next, offset_));
      current = next;
      next = LoadWord(bitmap_ + 24);
      total_popcount += bit_util::PopCount(ShiftWord(current, next, offset_));
      current = next;
      next = LoadWord(bitmap_ + 32);
      total_popcount += bit_util::PopCount(ShiftWord(current, next, offset_));
    }
    bitmap_ += bit_util::BytesForBits(kFourWordsBits);
    bits_remaining_ -= kFourWordsBits;
    return {256, static_cast<int16_t>(total_popcount)};
  }

  /// \brief Return the next run of available bits, usually 64. The returned
  /// pair contains the size of run and the number of true values. The last
  /// block will have a length less than 64 if the bitmap length is not a
  /// multiple of 64, and will return 0-length blocks in subsequent
  /// invocations.
  BitBlockCount NextWord() {
    using detail::LoadWord;
    using detail::ShiftWord;

    if (!bits_remaining_) {
      return {0, 0};
    }
    int64_t popcount = 0;
    if (offset_ == 0) {
      if (bits_remaining_ < kWordBits) {
        return GetBlockSlow(kWordBits);
      }
      popcount = bit_util::PopCount(LoadWord(bitmap_));
    } else {
      // When the offset is > 0, we need there to be a word beyond the last
      // aligned word in the bitmap for the bit shifting logic.
      if (bits_remaining_ < 2 * kWordBits - offset_) {
        return GetBlockSlow(kWordBits);
      }
      popcount = bit_util::PopCount(
          ShiftWord(LoadWord(bitmap_), LoadWord(bitmap_ + 8), offset_));
    }
    bitmap_ += kWordBits / 8;
    bits_remaining_ -= kWordBits;
    return {64, static_cast<int16_t>(popcount)};
  }

 private:
  /// \brief Return block with the requested size when doing word-wise
  /// computation is not possible due to inadequate bits remaining.
  BitBlockCount GetBlockSlow(int64_t block_size) noexcept;

  const uint8_t* bitmap_;
  int64_t bits_remaining_;
  int64_t offset_;
};

/// \brief A tool to iterate through a possibly non-existent validity bitmap,
/// to allow us to write one code path for both the with-nulls and no-nulls
/// cases without giving up a lot of performance.
class ARROW_EXPORT OptionalBitBlockCounter {
 public:
  // validity_bitmap may be NULLPTR
  OptionalBitBlockCounter(const uint8_t* validity_bitmap, int64_t offset, int64_t length);

  // validity_bitmap may be null
  OptionalBitBlockCounter(const std::shared_ptr<Buffer>& validity_bitmap, int64_t offset,
                          int64_t length);

  /// Return block count for next word when the bitmap is available otherwise
  /// return a block with length up to INT16_MAX when there is no validity
  /// bitmap (so all the referenced values are not null).
  BitBlockCount NextBlock() {
    static constexpr int64_t kMaxBlockSize = std::numeric_limits<int16_t>::max();
    if (has_bitmap_) {
      BitBlockCount block = counter_.NextWord();
      position_ += block.length;
      return block;
    } else {
      int16_t block_size =
          static_cast<int16_t>(std::min(kMaxBlockSize, length_ - position_));
      position_ += block_size;
      // All values are non-null
      return {block_size, block_size};
    }
  }

  // Like NextBlock, but returns a word-sized block even when there is no
  // validity bitmap
  BitBlockCount NextWord() {
    static constexpr int64_t kWordSize = 64;
    if (has_bitmap_) {
      BitBlockCount block = counter_.NextWord();
      position_ += block.length;
      return block;
    } else {
      int16_t block_size = static_cast<int16_t>(std::min(kWordSize, length_ - position_));
      position_ += block_size;
      // All values are non-null
      return {block_size, block_size};
    }
  }

 private:
  const bool has_bitmap_;
  int64_t position_;
  int64_t length_;
  BitBlockCounter counter_;
};

/// \brief A class that computes popcounts on the result of bitwise operations
/// between two bitmaps, 64 bits at a time. A 64-bit word is loaded from each
/// bitmap, then the popcount is computed on e.g. the bitwise-and of the two
/// words.
class ARROW_EXPORT BinaryBitBlockCounter {
 public:
  BinaryBitBlockCounter(const uint8_t* left_bitmap, int64_t left_offset,
                        const uint8_t* right_bitmap, int64_t right_offset, int64_t length)
      : left_bitmap_(util::MakeNonNull(left_bitmap) + left_offset / 8),
        left_offset_(left_offset % 8),
        right_bitmap_(util::MakeNonNull(right_bitmap) + right_offset / 8),
        right_offset_(right_offset % 8),
        bits_remaining_(length) {}

  /// \brief Return the popcount of the bitwise-and of the next run of
  /// available bits, up to 64. The returned pair contains the size of run and
  /// the number of true values. The last block will have a length less than 64
  /// if the bitmap length is not a multiple of 64, and will return 0-length
  /// blocks in subsequent invocations.
  BitBlockCount NextAndWord() { return NextWord<detail::BitBlockAnd>(); }

  /// \brief Computes "x & ~y" block for each available run of bits.
  BitBlockCount NextAndNotWord() { return NextWord<detail::BitBlockAndNot>(); }

  /// \brief Computes "x | y" block for each available run of bits.
  BitBlockCount NextOrWord() { return NextWord<detail::BitBlockOr>(); }

  /// \brief Computes "x | ~y" block for each available run of bits.
  BitBlockCount NextOrNotWord() { return NextWord<detail::BitBlockOrNot>(); }

 private:
  template <class Op>
  BitBlockCount NextWord() {
    using detail::LoadWord;
    using detail::ShiftWord;

    if (!bits_remaining_) {
      return {0, 0};
    }
    // When the offset is > 0, we need there to be a word beyond the last aligned
    // word in the bitmap for the bit shifting logic.
    constexpr int64_t kWordBits = BitBlockCounter::kWordBits;
    const int64_t bits_required_to_use_words =
        std::max(left_offset_ == 0 ? 64 : 64 + (64 - left_offset_),
                 right_offset_ == 0 ? 64 : 64 + (64 - right_offset_));
    if (bits_remaining_ < bits_required_to_use_words) {
      const int16_t run_length =
          static_cast<int16_t>(std::min(bits_remaining_, kWordBits));
      int16_t popcount = 0;
      for (int64_t i = 0; i < run_length; ++i) {
        if (Op::Call(bit_util::GetBit(left_bitmap_, left_offset_ + i),
                     bit_util::GetBit(right_bitmap_, right_offset_ + i))) {
          ++popcount;
        }
      }
      // This code path should trigger _at most_ 2 times. In the "two times"
      // case, the first time the run length will be a multiple of 8.
      left_bitmap_ += run_length / 8;
      right_bitmap_ += run_length / 8;
      bits_remaining_ -= run_length;
      return {run_length, popcount};
    }

    int64_t popcount = 0;
    if (left_offset_ == 0 && right_offset_ == 0) {
      popcount =
          bit_util::PopCount(Op::Call(LoadWord(left_bitmap_), LoadWord(right_bitmap_)));
    } else {
      auto left_word =
          ShiftWord(LoadWord(left_bitmap_), LoadWord(left_bitmap_ + 8), left_offset_);
      auto right_word =
          ShiftWord(LoadWord(right_bitmap_), LoadWord(right_bitmap_ + 8), right_offset_);
      popcount = bit_util::PopCount(Op::Call(left_word, right_word));
    }
    left_bitmap_ += kWordBits / 8;
    right_bitmap_ += kWordBits / 8;
    bits_remaining_ -= kWordBits;
    return {64, static_cast<int16_t>(popcount)};
  }

  const uint8_t* left_bitmap_;
  int64_t left_offset_;
  const uint8_t* right_bitmap_;
  int64_t right_offset_;
  int64_t bits_remaining_;
};

class ARROW_EXPORT OptionalBinaryBitBlockCounter {
 public:
  // Any bitmap may be NULLPTR
  OptionalBinaryBitBlockCounter(const uint8_t* left_bitmap, int64_t left_offset,
                                const uint8_t* right_bitmap, int64_t right_offset,
                                int64_t length);

  // Any bitmap may be null
  OptionalBinaryBitBlockCounter(const std::shared_ptr<Buffer>& left_bitmap,
                                int64_t left_offset,
                                const std::shared_ptr<Buffer>& right_bitmap,
                                int64_t right_offset, int64_t length);

  BitBlockCount NextAndBlock() {
    static constexpr int64_t kMaxBlockSize = std::numeric_limits<int16_t>::max();
    switch (has_bitmap_) {
      case HasBitmap::BOTH: {
        BitBlockCount block = binary_counter_.NextAndWord();
        position_ += block.length;
        return block;
      }
      case HasBitmap::ONE: {
        BitBlockCount block = unary_counter_.NextWord();
        position_ += block.length;
        return block;
      }
      case HasBitmap::NONE:
      default: {
        const int16_t block_size =
            static_cast<int16_t>(std::min(kMaxBlockSize, length_ - position_));
        position_ += block_size;
        // All values are non-null
        return {block_size, block_size};
      }
    }
  }

  BitBlockCount NextOrNotBlock() {
    static constexpr int64_t kMaxBlockSize = std::numeric_limits<int16_t>::max();
    switch (has_bitmap_) {
      case HasBitmap::BOTH: {
        BitBlockCount block = binary_counter_.NextOrNotWord();
        position_ += block.length;
        return block;
      }
      case HasBitmap::ONE: {
        BitBlockCount block = unary_counter_.NextWord();
        position_ += block.length;
        return block;
      }
      case HasBitmap::NONE:
      default: {
        const int16_t block_size =
            static_cast<int16_t>(std::min(kMaxBlockSize, length_ - position_));
        position_ += block_size;
        // All values are non-null
        return {block_size, block_size};
      }
    }
  }

 private:
  enum class HasBitmap : int { BOTH, ONE, NONE };

  const HasBitmap has_bitmap_;
  int64_t position_;
  int64_t length_;
  BitBlockCounter unary_counter_;
  BinaryBitBlockCounter binary_counter_;

  static HasBitmap HasBitmapFromBitmaps(bool has_left, bool has_right) {
    switch (static_cast<int>(has_left) + static_cast<int>(has_right)) {
      case 0:
        return HasBitmap::NONE;
      case 1:
        return HasBitmap::ONE;
      default:  // 2
        return HasBitmap::BOTH;
    }
  }
};

// Functional-style bit block visitors.

template <typename VisitNotNull, typename VisitNull>
static Status VisitBitBlocks(const uint8_t* bitmap, int64_t offset, int64_t length,
                             VisitNotNull&& visit_not_null, VisitNull&& visit_null) {
  internal::OptionalBitBlockCounter bit_counter(bitmap, offset, length);
  int64_t position = 0;
  while (position < length) {
    internal::BitBlockCount block = bit_counter.NextBlock();
    if (block.AllSet()) {
      for (int64_t i = 0; i < block.length; ++i, ++position) {
        ARROW_RETURN_NOT_OK(visit_not_null(position));
      }
    } else if (block.NoneSet()) {
      for (int64_t i = 0; i < block.length; ++i, ++position) {
        ARROW_RETURN_NOT_OK(visit_null());
      }
    } else {
      for (int64_t i = 0; i < block.length; ++i, ++position) {
        if (bit_util::GetBit(bitmap, offset + position)) {
          ARROW_RETURN_NOT_OK(visit_not_null(position));
        } else {
          ARROW_RETURN_NOT_OK(visit_null());
        }
      }
    }
  }
  return Status::OK();
}

template <typename VisitNotNull, typename VisitNull>
static void VisitBitBlocksVoid(const uint8_t* bitmap, int64_t offset, int64_t length,
                               VisitNotNull&& visit_not_null, VisitNull&& visit_null) {
  internal::OptionalBitBlockCounter bit_counter(bitmap, offset, length);
  int64_t position = 0;
  while (position < length) {
    internal::BitBlockCount block = bit_counter.NextBlock();
    if (block.AllSet()) {
      for (int64_t i = 0; i < block.length; ++i, ++position) {
        visit_not_null(position);
      }
    } else if (block.NoneSet()) {
      for (int64_t i = 0; i < block.length; ++i, ++position) {
        visit_null();
      }
    } else {
      for (int64_t i = 0; i < block.length; ++i, ++position) {
        if (bit_util::GetBit(bitmap, offset + position)) {
          visit_not_null(position);
        } else {
          visit_null();
        }
      }
    }
  }
}

template <typename VisitNotNull, typename VisitNull>
static Status VisitTwoBitBlocks(const uint8_t* left_bitmap, int64_t left_offset,
                                const uint8_t* right_bitmap, int64_t right_offset,
                                int64_t length, VisitNotNull&& visit_not_null,
                                VisitNull&& visit_null) {
  if (left_bitmap == NULLPTR || right_bitmap == NULLPTR) {
    // At most one bitmap is present
    if (left_bitmap == NULLPTR) {
      return VisitBitBlocks(right_bitmap, right_offset, length,
                            std::forward<VisitNotNull>(visit_not_null),
                            std::forward<VisitNull>(visit_null));
    } else {
      return VisitBitBlocks(left_bitmap, left_offset, length,
                            std::forward<VisitNotNull>(visit_not_null),
                            std::forward<VisitNull>(visit_null));
    }
  }
  BinaryBitBlockCounter bit_counter(left_bitmap, left_offset, right_bitmap, right_offset,
                                    length);
  int64_t position = 0;
  while (position < length) {
    BitBlockCount block = bit_counter.NextAndWord();
    if (block.AllSet()) {
      for (int64_t i = 0; i < block.length; ++i, ++position) {
        ARROW_RETURN_NOT_OK(visit_not_null(position));
      }
    } else if (block.NoneSet()) {
      for (int64_t i = 0; i < block.length; ++i, ++position) {
        ARROW_RETURN_NOT_OK(visit_null());
      }
    } else {
      for (int64_t i = 0; i < block.length; ++i, ++position) {
        if (bit_util::GetBit(left_bitmap, left_offset + position) &&
            bit_util::GetBit(right_bitmap, right_offset + position)) {
          ARROW_RETURN_NOT_OK(visit_not_null(position));
        } else {
          ARROW_RETURN_NOT_OK(visit_null());
        }
      }
    }
  }
  return Status::OK();
}

template <typename VisitNotNull, typename VisitNull>
static void VisitTwoBitBlocksVoid(const uint8_t* left_bitmap, int64_t left_offset,
                                  const uint8_t* right_bitmap, int64_t right_offset,
                                  int64_t length, VisitNotNull&& visit_not_null,
                                  VisitNull&& visit_null) {
  if (left_bitmap == NULLPTR || right_bitmap == NULLPTR) {
    // At most one bitmap is present
    if (left_bitmap == NULLPTR) {
      return VisitBitBlocksVoid(right_bitmap, right_offset, length,
                                std::forward<VisitNotNull>(visit_not_null),
                                std::forward<VisitNull>(visit_null));
    } else {
      return VisitBitBlocksVoid(left_bitmap, left_offset, length,
                                std::forward<VisitNotNull>(visit_not_null),
                                std::forward<VisitNull>(visit_null));
    }
  }
  BinaryBitBlockCounter bit_counter(left_bitmap, left_offset, right_bitmap, right_offset,
                                    length);
  int64_t position = 0;
  while (position < length) {
    BitBlockCount block = bit_counter.NextAndWord();
    if (block.AllSet()) {
      for (int64_t i = 0; i < block.length; ++i, ++position) {
        visit_not_null(position);
      }
    } else if (block.NoneSet()) {
      for (int64_t i = 0; i < block.length; ++i, ++position) {
        visit_null();
      }
    } else {
      for (int64_t i = 0; i < block.length; ++i, ++position) {
        if (bit_util::GetBit(left_bitmap, left_offset + position) &&
            bit_util::GetBit(right_bitmap, right_offset + position)) {
          visit_not_null(position);
        } else {
          visit_null();
        }
      }
    }
  }
}

}  // namespace internal
}  // namespace arrow

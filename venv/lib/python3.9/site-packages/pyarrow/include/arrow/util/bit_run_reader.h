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

#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>

#include "arrow/util/bit_util.h"
#include "arrow/util/bitmap_reader.h"
#include "arrow/util/endian.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace internal {

struct BitRun {
  int64_t length;
  // Whether bits are set at this point.
  bool set;

  std::string ToString() const {
    return std::string("{Length: ") + std::to_string(length) +
           ", set=" + std::to_string(set) + "}";
  }
};

inline bool operator==(const BitRun& lhs, const BitRun& rhs) {
  return lhs.length == rhs.length && lhs.set == rhs.set;
}

inline bool operator!=(const BitRun& lhs, const BitRun& rhs) {
  return lhs.length != rhs.length || lhs.set != rhs.set;
}

class BitRunReaderLinear {
 public:
  BitRunReaderLinear(const uint8_t* bitmap, int64_t start_offset, int64_t length)
      : reader_(bitmap, start_offset, length) {}

  BitRun NextRun() {
    BitRun rl = {/*length=*/0, reader_.IsSet()};
    // Advance while the values are equal and not at the end of list.
    while (reader_.position() < reader_.length() && reader_.IsSet() == rl.set) {
      rl.length++;
      reader_.Next();
    }
    return rl;
  }

 private:
  BitmapReader reader_;
};

#if ARROW_LITTLE_ENDIAN
/// A convenience class for counting the number of contiguous set/unset bits
/// in a bitmap.
class ARROW_EXPORT BitRunReader {
 public:
  /// \brief Constructs new BitRunReader.
  ///
  /// \param[in] bitmap source data
  /// \param[in] start_offset bit offset into the source data
  /// \param[in] length number of bits to copy
  BitRunReader(const uint8_t* bitmap, int64_t start_offset, int64_t length);

  /// Returns a new BitRun containing the number of contiguous
  /// bits with the same value.  length == 0 indicates the
  /// end of the bitmap.
  BitRun NextRun() {
    if (ARROW_PREDICT_FALSE(position_ >= length_)) {
      return {/*length=*/0, false};
    }
    // This implementation relies on a efficient implementations of
    // CountTrailingZeros and assumes that runs are more often then
    // not.  The logic is to incrementally find the next bit change
    // from the current position.  This is done by zeroing all
    // bits in word_ up to position_ and using the TrailingZeroCount
    // to find the index of the next set bit.

    // The runs alternate on each call, so flip the bit.
    current_run_bit_set_ = !current_run_bit_set_;

    int64_t start_position = position_;
    int64_t start_bit_offset = start_position & 63;
    // Invert the word for proper use of CountTrailingZeros and
    // clear bits so CountTrailingZeros can do it magic.
    word_ = ~word_ & ~bit_util::LeastSignificantBitMask(start_bit_offset);

    // Go  forward until the next change from unset to set.
    int64_t new_bits = bit_util::CountTrailingZeros(word_) - start_bit_offset;
    position_ += new_bits;

    if (ARROW_PREDICT_FALSE(bit_util::IsMultipleOf64(position_)) &&
        ARROW_PREDICT_TRUE(position_ < length_)) {
      // Continue extending position while we can advance an entire word.
      // (updates position_ accordingly).
      AdvanceUntilChange();
    }

    return {/*length=*/position_ - start_position, current_run_bit_set_};
  }

 private:
  void AdvanceUntilChange() {
    int64_t new_bits = 0;
    do {
      // Advance the position of the bitmap for loading.
      bitmap_ += sizeof(uint64_t);
      LoadNextWord();
      new_bits = bit_util::CountTrailingZeros(word_);
      // Continue calculating run length.
      position_ += new_bits;
    } while (ARROW_PREDICT_FALSE(bit_util::IsMultipleOf64(position_)) &&
             ARROW_PREDICT_TRUE(position_ < length_) && new_bits > 0);
  }

  void LoadNextWord() { return LoadWord(length_ - position_); }

  // Helper method for Loading the next word.
  void LoadWord(int64_t bits_remaining) {
    word_ = 0;
    // we need at least an extra byte in this case.
    if (ARROW_PREDICT_TRUE(bits_remaining >= 64)) {
      std::memcpy(&word_, bitmap_, 8);
    } else {
      int64_t bytes_to_load = bit_util::BytesForBits(bits_remaining);
      auto word_ptr = reinterpret_cast<uint8_t*>(&word_);
      std::memcpy(word_ptr, bitmap_, bytes_to_load);
      // Ensure stoppage at last bit in bitmap by reversing the next higher
      // order bit.
      bit_util::SetBitTo(word_ptr, bits_remaining,
                         !bit_util::GetBit(word_ptr, bits_remaining - 1));
    }

    // Two cases:
    //   1. For unset, CountTrailingZeros works naturally so we don't
    //   invert the word.
    //   2. Otherwise invert so we can use CountTrailingZeros.
    if (current_run_bit_set_) {
      word_ = ~word_;
    }
  }
  const uint8_t* bitmap_;
  int64_t position_;
  int64_t length_;
  uint64_t word_;
  bool current_run_bit_set_;
};
#else
using BitRunReader = BitRunReaderLinear;
#endif

struct SetBitRun {
  int64_t position;
  int64_t length;

  bool AtEnd() const { return length == 0; }

  std::string ToString() const {
    return std::string("{pos=") + std::to_string(position) +
           ", len=" + std::to_string(length) + "}";
  }

  bool operator==(const SetBitRun& other) const {
    return position == other.position && length == other.length;
  }
  bool operator!=(const SetBitRun& other) const {
    return position != other.position || length != other.length;
  }
};

template <bool Reverse>
class BaseSetBitRunReader {
 public:
  /// \brief Constructs new SetBitRunReader.
  ///
  /// \param[in] bitmap source data
  /// \param[in] start_offset bit offset into the source data
  /// \param[in] length number of bits to copy
  ARROW_NOINLINE
  BaseSetBitRunReader(const uint8_t* bitmap, int64_t start_offset, int64_t length)
      : bitmap_(util::MakeNonNull(bitmap)),
        length_(length),
        remaining_(length_),
        current_word_(0),
        current_num_bits_(0) {
    if (Reverse) {
      bitmap_ += (start_offset + length) / 8;
      const int8_t end_bit_offset = static_cast<int8_t>((start_offset + length) % 8);
      if (length > 0 && end_bit_offset) {
        // Get LSBs from last byte
        ++bitmap_;
        current_num_bits_ =
            std::min(static_cast<int32_t>(length), static_cast<int32_t>(end_bit_offset));
        current_word_ = LoadPartialWord(8 - end_bit_offset, current_num_bits_);
      }
    } else {
      bitmap_ += start_offset / 8;
      const int8_t bit_offset = static_cast<int8_t>(start_offset % 8);
      if (length > 0 && bit_offset) {
        // Get MSBs from first byte
        current_num_bits_ =
            std::min(static_cast<int32_t>(length), static_cast<int32_t>(8 - bit_offset));
        current_word_ = LoadPartialWord(bit_offset, current_num_bits_);
      }
    }
  }

  ARROW_NOINLINE
  SetBitRun NextRun() {
    int64_t pos = 0;
    int64_t len = 0;
    if (current_num_bits_) {
      const auto run = FindCurrentRun();
      assert(remaining_ >= 0);
      if (run.length && current_num_bits_) {
        // The run ends in current_word_
        return AdjustRun(run);
      }
      pos = run.position;
      len = run.length;
    }
    if (!len) {
      // We didn't get any ones in current_word_, so we can skip any zeros
      // in the following words
      SkipNextZeros();
      if (remaining_ == 0) {
        return {0, 0};
      }
      assert(current_num_bits_);
      pos = position();
    } else if (!current_num_bits_) {
      if (ARROW_PREDICT_TRUE(remaining_ >= 64)) {
        current_word_ = LoadFullWord();
        current_num_bits_ = 64;
      } else if (remaining_ > 0) {
        current_word_ = LoadPartialWord(/*bit_offset=*/0, remaining_);
        current_num_bits_ = static_cast<int32_t>(remaining_);
      } else {
        // No bits remaining, perhaps we found a run?
        return AdjustRun({pos, len});
      }
      // If current word starts with a zero, we got a full run
      if (!(current_word_ & kFirstBit)) {
        return AdjustRun({pos, len});
      }
    }
    // Current word should now start with a set bit
    len += CountNextOnes();
    return AdjustRun({pos, len});
  }

 protected:
  int64_t position() const {
    if (Reverse) {
      return remaining_;
    } else {
      return length_ - remaining_;
    }
  }

  SetBitRun AdjustRun(SetBitRun run) {
    if (Reverse) {
      assert(run.position >= run.length);
      run.position -= run.length;
    }
    return run;
  }

  uint64_t LoadFullWord() {
    uint64_t word;
    if (Reverse) {
      bitmap_ -= 8;
    }
    memcpy(&word, bitmap_, 8);
    if (!Reverse) {
      bitmap_ += 8;
    }
    return bit_util::ToLittleEndian(word);
  }

  uint64_t LoadPartialWord(int8_t bit_offset, int64_t num_bits) {
    assert(num_bits > 0);
    uint64_t word = 0;
    const int64_t num_bytes = bit_util::BytesForBits(num_bits);
    if (Reverse) {
      // Read in the most significant bytes of the word
      bitmap_ -= num_bytes;
      memcpy(reinterpret_cast<char*>(&word) + 8 - num_bytes, bitmap_, num_bytes);
      // XXX MostSignificantBitmask
      return (bit_util::ToLittleEndian(word) << bit_offset) &
             ~bit_util::LeastSignificantBitMask(64 - num_bits);
    } else {
      memcpy(&word, bitmap_, num_bytes);
      bitmap_ += num_bytes;
      return (bit_util::ToLittleEndian(word) >> bit_offset) &
             bit_util::LeastSignificantBitMask(num_bits);
    }
  }

  void SkipNextZeros() {
    assert(current_num_bits_ == 0);
    while (ARROW_PREDICT_TRUE(remaining_ >= 64)) {
      current_word_ = LoadFullWord();
      const auto num_zeros = CountFirstZeros(current_word_);
      if (num_zeros < 64) {
        // Run of zeros ends here
        current_word_ = ConsumeBits(current_word_, num_zeros);
        current_num_bits_ = 64 - num_zeros;
        remaining_ -= num_zeros;
        assert(remaining_ >= 0);
        assert(current_num_bits_ >= 0);
        return;
      }
      remaining_ -= 64;
    }
    // Run of zeros continues in last bitmap word
    if (remaining_ > 0) {
      current_word_ = LoadPartialWord(/*bit_offset=*/0, remaining_);
      current_num_bits_ = static_cast<int32_t>(remaining_);
      const auto num_zeros =
          std::min<int32_t>(current_num_bits_, CountFirstZeros(current_word_));
      current_word_ = ConsumeBits(current_word_, num_zeros);
      current_num_bits_ -= num_zeros;
      remaining_ -= num_zeros;
      assert(remaining_ >= 0);
      assert(current_num_bits_ >= 0);
    }
  }

  int64_t CountNextOnes() {
    assert(current_word_ & kFirstBit);

    int64_t len;
    if (~current_word_) {
      const auto num_ones = CountFirstZeros(~current_word_);
      assert(num_ones <= current_num_bits_);
      assert(num_ones <= remaining_);
      remaining_ -= num_ones;
      current_word_ = ConsumeBits(current_word_, num_ones);
      current_num_bits_ -= num_ones;
      if (current_num_bits_) {
        // Run of ones ends here
        return num_ones;
      }
      len = num_ones;
    } else {
      // current_word_ is all ones
      remaining_ -= 64;
      current_num_bits_ = 0;
      len = 64;
    }

    while (ARROW_PREDICT_TRUE(remaining_ >= 64)) {
      current_word_ = LoadFullWord();
      const auto num_ones = CountFirstZeros(~current_word_);
      len += num_ones;
      remaining_ -= num_ones;
      if (num_ones < 64) {
        // Run of ones ends here
        current_word_ = ConsumeBits(current_word_, num_ones);
        current_num_bits_ = 64 - num_ones;
        return len;
      }
    }
    // Run of ones continues in last bitmap word
    if (remaining_ > 0) {
      current_word_ = LoadPartialWord(/*bit_offset=*/0, remaining_);
      current_num_bits_ = static_cast<int32_t>(remaining_);
      const auto num_ones = CountFirstZeros(~current_word_);
      assert(num_ones <= current_num_bits_);
      assert(num_ones <= remaining_);
      current_word_ = ConsumeBits(current_word_, num_ones);
      current_num_bits_ -= num_ones;
      remaining_ -= num_ones;
      len += num_ones;
    }
    return len;
  }

  SetBitRun FindCurrentRun() {
    // Skip any pending zeros
    const auto num_zeros = CountFirstZeros(current_word_);
    if (num_zeros >= current_num_bits_) {
      remaining_ -= current_num_bits_;
      current_word_ = 0;
      current_num_bits_ = 0;
      return {0, 0};
    }
    assert(num_zeros <= remaining_);
    current_word_ = ConsumeBits(current_word_, num_zeros);
    current_num_bits_ -= num_zeros;
    remaining_ -= num_zeros;
    const int64_t pos = position();
    // Count any ones
    const auto num_ones = CountFirstZeros(~current_word_);
    assert(num_ones <= current_num_bits_);
    assert(num_ones <= remaining_);
    current_word_ = ConsumeBits(current_word_, num_ones);
    current_num_bits_ -= num_ones;
    remaining_ -= num_ones;
    return {pos, num_ones};
  }

  inline int CountFirstZeros(uint64_t word);
  inline uint64_t ConsumeBits(uint64_t word, int32_t num_bits);

  const uint8_t* bitmap_;
  const int64_t length_;
  int64_t remaining_;
  uint64_t current_word_;
  int32_t current_num_bits_;

  static constexpr uint64_t kFirstBit = Reverse ? 0x8000000000000000ULL : 1;
};

template <>
inline int BaseSetBitRunReader<false>::CountFirstZeros(uint64_t word) {
  return bit_util::CountTrailingZeros(word);
}

template <>
inline int BaseSetBitRunReader<true>::CountFirstZeros(uint64_t word) {
  return bit_util::CountLeadingZeros(word);
}

template <>
inline uint64_t BaseSetBitRunReader<false>::ConsumeBits(uint64_t word, int32_t num_bits) {
  return word >> num_bits;
}

template <>
inline uint64_t BaseSetBitRunReader<true>::ConsumeBits(uint64_t word, int32_t num_bits) {
  return word << num_bits;
}

using SetBitRunReader = BaseSetBitRunReader</*Reverse=*/false>;
using ReverseSetBitRunReader = BaseSetBitRunReader</*Reverse=*/true>;

// Functional-style bit run visitors.

// XXX: Try to make this function small so the compiler can inline and optimize
// the `visit` function, which is normally a hot loop with vectorizable code.
// - don't inline SetBitRunReader constructor, it doesn't hurt performance
// - un-inline NextRun hurts 'many null' cases a bit, but improves normal cases
template <typename Visit>
inline Status VisitSetBitRuns(const uint8_t* bitmap, int64_t offset, int64_t length,
                              Visit&& visit) {
  if (bitmap == NULLPTR) {
    // Assuming all set (as in a null bitmap)
    return visit(static_cast<int64_t>(0), static_cast<int64_t>(length));
  }
  SetBitRunReader reader(bitmap, offset, length);
  while (true) {
    const auto run = reader.NextRun();
    if (run.length == 0) {
      break;
    }
    ARROW_RETURN_NOT_OK(visit(run.position, run.length));
  }
  return Status::OK();
}

template <typename Visit>
inline void VisitSetBitRunsVoid(const uint8_t* bitmap, int64_t offset, int64_t length,
                                Visit&& visit) {
  if (bitmap == NULLPTR) {
    // Assuming all set (as in a null bitmap)
    visit(static_cast<int64_t>(0), static_cast<int64_t>(length));
    return;
  }
  SetBitRunReader reader(bitmap, offset, length);
  while (true) {
    const auto run = reader.NextRun();
    if (run.length == 0) {
      break;
    }
    visit(run.position, run.length);
  }
}

template <typename Visit>
inline Status VisitSetBitRuns(const std::shared_ptr<Buffer>& bitmap, int64_t offset,
                              int64_t length, Visit&& visit) {
  return VisitSetBitRuns(bitmap ? bitmap->data() : NULLPTR, offset, length,
                         std::forward<Visit>(visit));
}

template <typename Visit>
inline void VisitSetBitRunsVoid(const std::shared_ptr<Buffer>& bitmap, int64_t offset,
                                int64_t length, Visit&& visit) {
  VisitSetBitRunsVoid(bitmap ? bitmap->data() : NULLPTR, offset, length,
                      std::forward<Visit>(visit));
}

}  // namespace internal
}  // namespace arrow

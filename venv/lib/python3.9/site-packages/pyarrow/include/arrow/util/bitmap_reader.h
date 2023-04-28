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

#include "arrow/buffer.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/endian.h"
#include "arrow/util/macros.h"

namespace arrow {
namespace internal {

class BitmapReader {
 public:
  BitmapReader(const uint8_t* bitmap, int64_t start_offset, int64_t length)
      : bitmap_(bitmap), position_(0), length_(length) {
    current_byte_ = 0;
    byte_offset_ = start_offset / 8;
    bit_offset_ = start_offset % 8;
    if (length > 0) {
      current_byte_ = bitmap[byte_offset_];
    }
  }

  bool IsSet() const { return (current_byte_ & (1 << bit_offset_)) != 0; }

  bool IsNotSet() const { return (current_byte_ & (1 << bit_offset_)) == 0; }

  void Next() {
    ++bit_offset_;
    ++position_;
    if (ARROW_PREDICT_FALSE(bit_offset_ == 8)) {
      bit_offset_ = 0;
      ++byte_offset_;
      if (ARROW_PREDICT_TRUE(position_ < length_)) {
        current_byte_ = bitmap_[byte_offset_];
      }
    }
  }

  int64_t position() const { return position_; }

  int64_t length() const { return length_; }

 private:
  const uint8_t* bitmap_;
  int64_t position_;
  int64_t length_;

  uint8_t current_byte_;
  int64_t byte_offset_;
  int64_t bit_offset_;
};

// XXX Cannot name it BitmapWordReader because the name is already used
// in bitmap_ops.cc

class BitmapUInt64Reader {
 public:
  BitmapUInt64Reader(const uint8_t* bitmap, int64_t start_offset, int64_t length)
      : bitmap_(util::MakeNonNull(bitmap) + start_offset / 8),
        num_carry_bits_(8 - start_offset % 8),
        length_(length),
        remaining_length_(length_),
        carry_bits_(0) {
    if (length_ > 0) {
      // Load carry bits from the first byte's MSBs
      if (length_ >= num_carry_bits_) {
        carry_bits_ =
            LoadPartialWord(static_cast<int8_t>(8 - num_carry_bits_), num_carry_bits_);
      } else {
        carry_bits_ = LoadPartialWord(static_cast<int8_t>(8 - num_carry_bits_), length_);
      }
    }
  }

  uint64_t NextWord() {
    if (ARROW_PREDICT_TRUE(remaining_length_ >= 64 + num_carry_bits_)) {
      // We can load a full word
      uint64_t next_word = LoadFullWord();
      // Carry bits come first, then the (64 - num_carry_bits_) LSBs from next_word
      uint64_t word = carry_bits_ | (next_word << num_carry_bits_);
      carry_bits_ = next_word >> (64 - num_carry_bits_);
      remaining_length_ -= 64;
      return word;
    } else if (remaining_length_ > num_carry_bits_) {
      // We can load a partial word
      uint64_t next_word =
          LoadPartialWord(/*bit_offset=*/0, remaining_length_ - num_carry_bits_);
      uint64_t word = carry_bits_ | (next_word << num_carry_bits_);
      carry_bits_ = next_word >> (64 - num_carry_bits_);
      remaining_length_ = std::max<int64_t>(remaining_length_ - 64, 0);
      return word;
    } else {
      remaining_length_ = 0;
      return carry_bits_;
    }
  }

  int64_t position() const { return length_ - remaining_length_; }

  int64_t length() const { return length_; }

 private:
  uint64_t LoadFullWord() {
    uint64_t word;
    memcpy(&word, bitmap_, 8);
    bitmap_ += 8;
    return bit_util::ToLittleEndian(word);
  }

  uint64_t LoadPartialWord(int8_t bit_offset, int64_t num_bits) {
    uint64_t word = 0;
    const int64_t num_bytes = bit_util::BytesForBits(num_bits);
    memcpy(&word, bitmap_, num_bytes);
    bitmap_ += num_bytes;
    return (bit_util::ToLittleEndian(word) >> bit_offset) &
           bit_util::LeastSignificantBitMask(num_bits);
  }

  const uint8_t* bitmap_;
  const int64_t num_carry_bits_;  // in [1, 8]
  const int64_t length_;
  int64_t remaining_length_;
  uint64_t carry_bits_;
};

// BitmapWordReader here is faster than BitmapUInt64Reader (in bitmap_reader.h)
// on sufficiently large inputs.  However, it has a larger prolog / epilog overhead
// and should probably not be used for small bitmaps.

template <typename Word, bool may_have_byte_offset = true>
class BitmapWordReader {
 public:
  BitmapWordReader() = default;
  BitmapWordReader(const uint8_t* bitmap, int64_t offset, int64_t length)
      : offset_(static_cast<int64_t>(may_have_byte_offset) * (offset % 8)),
        bitmap_(bitmap + offset / 8),
        bitmap_end_(bitmap_ + bit_util::BytesForBits(offset_ + length)) {
    // decrement word count by one as we may touch two adjacent words in one iteration
    nwords_ = length / (sizeof(Word) * 8) - 1;
    if (nwords_ < 0) {
      nwords_ = 0;
    }
    trailing_bits_ = static_cast<int>(length - nwords_ * sizeof(Word) * 8);
    trailing_bytes_ = static_cast<int>(bit_util::BytesForBits(trailing_bits_));

    if (nwords_ > 0) {
      current_data.word_ = load<Word>(bitmap_);
    } else if (length > 0) {
      current_data.epi.byte_ = load<uint8_t>(bitmap_);
    }
  }

  Word NextWord() {
    bitmap_ += sizeof(Word);
    const Word next_word = load<Word>(bitmap_);
    Word word = current_data.word_;
    if (may_have_byte_offset && offset_) {
      // combine two adjacent words into one word
      // |<------ next ----->|<---- current ---->|
      // +-------------+-----+-------------+-----+
      // |     ---     |  A  |      B      | --- |
      // +-------------+-----+-------------+-----+
      //                  |         |       offset
      //                  v         v
      //               +-----+-------------+
      //               |  A  |      B      |
      //               +-----+-------------+
      //               |<------ word ----->|
      word >>= offset_;
      word |= next_word << (sizeof(Word) * 8 - offset_);
    }
    current_data.word_ = next_word;
    return word;
  }

  uint8_t NextTrailingByte(int& valid_bits) {
    uint8_t byte;
    assert(trailing_bits_ > 0);

    if (trailing_bits_ <= 8) {
      // last byte
      valid_bits = trailing_bits_;
      trailing_bits_ = 0;
      byte = 0;
      internal::BitmapReader reader(bitmap_, offset_, valid_bits);
      for (int i = 0; i < valid_bits; ++i) {
        byte >>= 1;
        if (reader.IsSet()) {
          byte |= 0x80;
        }
        reader.Next();
      }
      byte >>= (8 - valid_bits);
    } else {
      ++bitmap_;
      const uint8_t next_byte = load<uint8_t>(bitmap_);
      byte = current_data.epi.byte_;
      if (may_have_byte_offset && offset_) {
        byte >>= offset_;
        byte |= next_byte << (8 - offset_);
      }
      current_data.epi.byte_ = next_byte;
      trailing_bits_ -= 8;
      trailing_bytes_--;
      valid_bits = 8;
    }
    return byte;
  }

  int64_t words() const { return nwords_; }
  int trailing_bytes() const { return trailing_bytes_; }

 private:
  int64_t offset_;
  const uint8_t* bitmap_;

  const uint8_t* bitmap_end_;
  int64_t nwords_;
  int trailing_bits_;
  int trailing_bytes_;
  union {
    Word word_;
    struct {
#if ARROW_LITTLE_ENDIAN == 0
      uint8_t padding_bytes_[sizeof(Word) - 1];
#endif
      uint8_t byte_;
    } epi;
  } current_data;

  template <typename DType>
  DType load(const uint8_t* bitmap) {
    assert(bitmap + sizeof(DType) <= bitmap_end_);
    return bit_util::ToLittleEndian(util::SafeLoadAs<DType>(bitmap));
  }
};

/// \brief Index into a possibly non-existent bitmap
struct OptionalBitIndexer {
  const uint8_t* bitmap;
  const int64_t offset;

  explicit OptionalBitIndexer(const uint8_t* buffer = NULLPTR, int64_t offset = 0)
      : bitmap(buffer), offset(offset) {}

  bool operator[](int64_t i) const {
    return bitmap == NULLPTR || bit_util::GetBit(bitmap, offset + i);
  }
};

}  // namespace internal
}  // namespace arrow

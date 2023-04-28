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
#include <cstring>

#include "arrow/util/bit_util.h"
#include "arrow/util/endian.h"
#include "arrow/util/macros.h"

namespace arrow {
namespace internal {

class BitmapWriter {
  // A sequential bitwise writer that preserves surrounding bit values.

 public:
  BitmapWriter(uint8_t* bitmap, int64_t start_offset, int64_t length)
      : bitmap_(bitmap), position_(0), length_(length) {
    byte_offset_ = start_offset / 8;
    bit_mask_ = bit_util::kBitmask[start_offset % 8];
    if (length > 0) {
      current_byte_ = bitmap[byte_offset_];
    } else {
      current_byte_ = 0;
    }
  }

  void Set() { current_byte_ |= bit_mask_; }

  void Clear() { current_byte_ &= bit_mask_ ^ 0xFF; }

  void Next() {
    bit_mask_ = static_cast<uint8_t>(bit_mask_ << 1);
    ++position_;
    if (bit_mask_ == 0) {
      // Finished this byte, need advancing
      bit_mask_ = 0x01;
      bitmap_[byte_offset_++] = current_byte_;
      if (ARROW_PREDICT_TRUE(position_ < length_)) {
        current_byte_ = bitmap_[byte_offset_];
      }
    }
  }

  void Finish() {
    // Store current byte if we didn't went past bitmap storage
    if (length_ > 0 && (bit_mask_ != 0x01 || position_ < length_)) {
      bitmap_[byte_offset_] = current_byte_;
    }
  }

  int64_t position() const { return position_; }

 private:
  uint8_t* bitmap_;
  int64_t position_;
  int64_t length_;

  uint8_t current_byte_;
  uint8_t bit_mask_;
  int64_t byte_offset_;
};

class FirstTimeBitmapWriter {
  // Like BitmapWriter, but any bit values *following* the bits written
  // might be clobbered.  It is hence faster than BitmapWriter, and can
  // also avoid false positives with Valgrind.

 public:
  FirstTimeBitmapWriter(uint8_t* bitmap, int64_t start_offset, int64_t length)
      : bitmap_(bitmap), position_(0), length_(length) {
    current_byte_ = 0;
    byte_offset_ = start_offset / 8;
    bit_mask_ = bit_util::kBitmask[start_offset % 8];
    if (length > 0) {
      current_byte_ =
          bitmap[byte_offset_] & bit_util::kPrecedingBitmask[start_offset % 8];
    } else {
      current_byte_ = 0;
    }
  }

  /// Appends number_of_bits from word to valid_bits and valid_bits_offset.
  ///
  /// \param[in] word The LSB bitmap to append. Any bits past number_of_bits are assumed
  ///            to be unset (i.e. 0).
  /// \param[in] number_of_bits The number of bits to append from word.
  void AppendWord(uint64_t word, int64_t number_of_bits) {
    if (ARROW_PREDICT_FALSE(number_of_bits == 0)) {
      return;
    }

    // Location that the first byte needs to be written to.
    uint8_t* append_position = bitmap_ + byte_offset_;

    // Update state variables except for current_byte_ here.
    position_ += number_of_bits;
    int64_t bit_offset = bit_util::CountTrailingZeros(static_cast<uint32_t>(bit_mask_));
    bit_mask_ = bit_util::kBitmask[(bit_offset + number_of_bits) % 8];
    byte_offset_ += (bit_offset + number_of_bits) / 8;

    if (bit_offset != 0) {
      // We are in the middle of the byte. This code updates the byte and shifts
      // bits appropriately within word so it can be memcpy'd below.
      int64_t bits_to_carry = 8 - bit_offset;
      // Carry over bits from word to current_byte_. We assume any extra bits in word
      // unset so no additional accounting is needed for when number_of_bits <
      // bits_to_carry.
      current_byte_ |= (word & bit_util::kPrecedingBitmask[bits_to_carry]) << bit_offset;
      // Check if everything is transfered into current_byte_.
      if (ARROW_PREDICT_FALSE(number_of_bits < bits_to_carry)) {
        return;
      }
      *append_position = current_byte_;
      append_position++;
      // Move the carry bits off of word.
      word = word >> bits_to_carry;
      number_of_bits -= bits_to_carry;
    }
    word = bit_util::ToLittleEndian(word);
    int64_t bytes_for_word = ::arrow::bit_util::BytesForBits(number_of_bits);
    std::memcpy(append_position, &word, bytes_for_word);
    // At this point, the previous current_byte_ has been written to bitmap_.
    // The new current_byte_ is either the last relevant byte in 'word'
    // or cleared if the new position is byte aligned (i.e. a fresh byte).
    if (bit_mask_ == 0x1) {
      current_byte_ = 0;
    } else {
      current_byte_ = *(append_position + bytes_for_word - 1);
    }
  }

  void Set() { current_byte_ |= bit_mask_; }

  void Clear() {}

  void Next() {
    bit_mask_ = static_cast<uint8_t>(bit_mask_ << 1);
    ++position_;
    if (bit_mask_ == 0) {
      // Finished this byte, need advancing
      bit_mask_ = 0x01;
      bitmap_[byte_offset_++] = current_byte_;
      current_byte_ = 0;
    }
  }

  void Finish() {
    // Store current byte if we didn't went go bitmap storage
    if (length_ > 0 && (bit_mask_ != 0x01 || position_ < length_)) {
      bitmap_[byte_offset_] = current_byte_;
    }
  }

  int64_t position() const { return position_; }

 private:
  uint8_t* bitmap_;
  int64_t position_;
  int64_t length_;

  uint8_t current_byte_;
  uint8_t bit_mask_;
  int64_t byte_offset_;
};

template <typename Word, bool may_have_byte_offset = true>
class BitmapWordWriter {
 public:
  BitmapWordWriter() = default;
  BitmapWordWriter(uint8_t* bitmap, int64_t offset, int64_t length)
      : offset_(static_cast<int64_t>(may_have_byte_offset) * (offset % 8)),
        bitmap_(bitmap + offset / 8),
        bitmap_end_(bitmap_ + bit_util::BytesForBits(offset_ + length)),
        mask_((1U << offset_) - 1) {
    if (offset_) {
      if (length >= static_cast<int>(sizeof(Word) * 8)) {
        current_data.word_ = load<Word>(bitmap_);
      } else if (length > 0) {
        current_data.epi.byte_ = load<uint8_t>(bitmap_);
      }
    }
  }

  void PutNextWord(Word word) {
    if (may_have_byte_offset && offset_) {
      // split one word into two adjacent words, don't touch unused bits
      //               |<------ word ----->|
      //               +-----+-------------+
      //               |  A  |      B      |
      //               +-----+-------------+
      //                  |         |
      //                  v         v       offset
      // +-------------+-----+-------------+-----+
      // |     ---     |  A  |      B      | --- |
      // +-------------+-----+-------------+-----+
      // |<------ next ----->|<---- current ---->|
      word = (word << offset_) | (word >> (sizeof(Word) * 8 - offset_));
      Word next_word = load<Word>(bitmap_ + sizeof(Word));
      current_data.word_ = (current_data.word_ & mask_) | (word & ~mask_);
      next_word = (next_word & ~mask_) | (word & mask_);
      store<Word>(bitmap_, current_data.word_);
      store<Word>(bitmap_ + sizeof(Word), next_word);
      current_data.word_ = next_word;
    } else {
      store<Word>(bitmap_, word);
    }
    bitmap_ += sizeof(Word);
  }

  void PutNextTrailingByte(uint8_t byte, int valid_bits) {
    if (valid_bits == 8) {
      if (may_have_byte_offset && offset_) {
        byte = (byte << offset_) | (byte >> (8 - offset_));
        uint8_t next_byte = load<uint8_t>(bitmap_ + 1);
        current_data.epi.byte_ = (current_data.epi.byte_ & mask_) | (byte & ~mask_);
        next_byte = (next_byte & ~mask_) | (byte & mask_);
        store<uint8_t>(bitmap_, current_data.epi.byte_);
        store<uint8_t>(bitmap_ + 1, next_byte);
        current_data.epi.byte_ = next_byte;
      } else {
        store<uint8_t>(bitmap_, byte);
      }
      ++bitmap_;
    } else {
      assert(valid_bits > 0);
      assert(valid_bits < 8);
      assert(bitmap_ + bit_util::BytesForBits(offset_ + valid_bits) <= bitmap_end_);
      internal::BitmapWriter writer(bitmap_, offset_, valid_bits);
      for (int i = 0; i < valid_bits; ++i) {
        (byte & 0x01) ? writer.Set() : writer.Clear();
        writer.Next();
        byte >>= 1;
      }
      writer.Finish();
    }
  }

 private:
  int64_t offset_;
  uint8_t* bitmap_;

  const uint8_t* bitmap_end_;
  uint64_t mask_;
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

  template <typename DType>
  void store(uint8_t* bitmap, DType data) {
    assert(bitmap + sizeof(DType) <= bitmap_end_);
    util::SafeStore(bitmap, bit_util::FromLittleEndian(data));
  }
};

}  // namespace internal
}  // namespace arrow

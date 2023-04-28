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

// From Apache Impala (incubating) as of 2016-01-29

#pragma once

#include <string.h>

#include <algorithm>
#include <cstdint>

#include "arrow/util/bit_util.h"
#include "arrow/util/bpacking.h"
#include "arrow/util/logging.h"
#include "arrow/util/macros.h"
#include "arrow/util/ubsan.h"

namespace arrow {
namespace bit_util {

/// Utility class to write bit/byte streams.  This class can write data to either be
/// bit packed or byte aligned (and a single stream that has a mix of both).
/// This class does not allocate memory.
class BitWriter {
 public:
  /// buffer: buffer to write bits to.  Buffer should be preallocated with
  /// 'buffer_len' bytes.
  BitWriter(uint8_t* buffer, int buffer_len) : buffer_(buffer), max_bytes_(buffer_len) {
    Clear();
  }

  void Clear() {
    buffered_values_ = 0;
    byte_offset_ = 0;
    bit_offset_ = 0;
  }

  /// The number of current bytes written, including the current byte (i.e. may include a
  /// fraction of a byte). Includes buffered values.
  int bytes_written() const {
    return byte_offset_ + static_cast<int>(bit_util::BytesForBits(bit_offset_));
  }
  uint8_t* buffer() const { return buffer_; }
  int buffer_len() const { return max_bytes_; }

  /// Writes a value to buffered_values_, flushing to buffer_ if necessary.  This is bit
  /// packed.  Returns false if there was not enough space. num_bits must be <= 32.
  bool PutValue(uint64_t v, int num_bits);

  /// Writes v to the next aligned byte using num_bytes. If T is larger than
  /// num_bytes, the extra high-order bytes will be ignored. Returns false if
  /// there was not enough space.
  /// Assume the v is stored in buffer_ as a litte-endian format
  template <typename T>
  bool PutAligned(T v, int num_bytes);

  /// Write a Vlq encoded int to the buffer.  Returns false if there was not enough
  /// room.  The value is written byte aligned.
  /// For more details on vlq:
  /// en.wikipedia.org/wiki/Variable-length_quantity
  bool PutVlqInt(uint32_t v);

  // Writes an int zigzag encoded.
  bool PutZigZagVlqInt(int32_t v);

  /// Write a Vlq encoded int64 to the buffer.  Returns false if there was not enough
  /// room.  The value is written byte aligned.
  /// For more details on vlq:
  /// en.wikipedia.org/wiki/Variable-length_quantity
  bool PutVlqInt(uint64_t v);

  // Writes an int64 zigzag encoded.
  bool PutZigZagVlqInt(int64_t v);

  /// Get a pointer to the next aligned byte and advance the underlying buffer
  /// by num_bytes.
  /// Returns NULL if there was not enough space.
  uint8_t* GetNextBytePtr(int num_bytes = 1);

  /// Flushes all buffered values to the buffer. Call this when done writing to
  /// the buffer.  If 'align' is true, buffered_values_ is reset and any future
  /// writes will be written to the next byte boundary.
  void Flush(bool align = false);

 private:
  uint8_t* buffer_;
  int max_bytes_;

  /// Bit-packed values are initially written to this variable before being memcpy'd to
  /// buffer_. This is faster than writing values byte by byte directly to buffer_.
  uint64_t buffered_values_;

  int byte_offset_;  // Offset in buffer_
  int bit_offset_;   // Offset in buffered_values_
};

/// Utility class to read bit/byte stream.  This class can read bits or bytes
/// that are either byte aligned or not.  It also has utilities to read multiple
/// bytes in one read (e.g. encoded int).
class BitReader {
 public:
  /// 'buffer' is the buffer to read from.  The buffer's length is 'buffer_len'.
  BitReader(const uint8_t* buffer, int buffer_len)
      : buffer_(buffer), max_bytes_(buffer_len), byte_offset_(0), bit_offset_(0) {
    int num_bytes = std::min(8, max_bytes_ - byte_offset_);
    memcpy(&buffered_values_, buffer_ + byte_offset_, num_bytes);
    buffered_values_ = arrow::bit_util::FromLittleEndian(buffered_values_);
  }

  BitReader()
      : buffer_(NULL),
        max_bytes_(0),
        buffered_values_(0),
        byte_offset_(0),
        bit_offset_(0) {}

  void Reset(const uint8_t* buffer, int buffer_len) {
    buffer_ = buffer;
    max_bytes_ = buffer_len;
    byte_offset_ = 0;
    bit_offset_ = 0;
    int num_bytes = std::min(8, max_bytes_ - byte_offset_);
    memcpy(&buffered_values_, buffer_ + byte_offset_, num_bytes);
    buffered_values_ = arrow::bit_util::FromLittleEndian(buffered_values_);
  }

  /// Gets the next value from the buffer.  Returns true if 'v' could be read or false if
  /// there are not enough bytes left.
  template <typename T>
  bool GetValue(int num_bits, T* v);

  /// Get a number of values from the buffer. Return the number of values actually read.
  template <typename T>
  int GetBatch(int num_bits, T* v, int batch_size);

  /// Reads a 'num_bytes'-sized value from the buffer and stores it in 'v'. T
  /// needs to be a little-endian native type and big enough to store
  /// 'num_bytes'. The value is assumed to be byte-aligned so the stream will
  /// be advanced to the start of the next byte before 'v' is read. Returns
  /// false if there are not enough bytes left.
  /// Assume the v was stored in buffer_ as a litte-endian format
  template <typename T>
  bool GetAligned(int num_bytes, T* v);

  /// Advances the stream by a number of bits. Returns true if succeed or false if there
  /// are not enough bits left.
  bool Advance(int64_t num_bits);

  /// Reads a vlq encoded int from the stream.  The encoded int must start at
  /// the beginning of a byte. Return false if there were not enough bytes in
  /// the buffer.
  bool GetVlqInt(uint32_t* v);

  // Reads a zigzag encoded int `into` v.
  bool GetZigZagVlqInt(int32_t* v);

  /// Reads a vlq encoded int64 from the stream.  The encoded int must start at
  /// the beginning of a byte. Return false if there were not enough bytes in
  /// the buffer.
  bool GetVlqInt(uint64_t* v);

  // Reads a zigzag encoded int64 `into` v.
  bool GetZigZagVlqInt(int64_t* v);

  /// Returns the number of bytes left in the stream, not including the current
  /// byte (i.e., there may be an additional fraction of a byte).
  int bytes_left() {
    return max_bytes_ -
           (byte_offset_ + static_cast<int>(bit_util::BytesForBits(bit_offset_)));
  }

  /// Maximum byte length of a vlq encoded int
  static constexpr int kMaxVlqByteLength = 5;

  /// Maximum byte length of a vlq encoded int64
  static constexpr int kMaxVlqByteLengthForInt64 = 10;

 private:
  const uint8_t* buffer_;
  int max_bytes_;

  /// Bytes are memcpy'd from buffer_ and values are read from this variable. This is
  /// faster than reading values byte by byte directly from buffer_.
  uint64_t buffered_values_;

  int byte_offset_;  // Offset in buffer_
  int bit_offset_;   // Offset in buffered_values_
};

inline bool BitWriter::PutValue(uint64_t v, int num_bits) {
  DCHECK_LE(num_bits, 64);
  if (num_bits < 64) {
    DCHECK_EQ(v >> num_bits, 0) << "v = " << v << ", num_bits = " << num_bits;
  }

  if (ARROW_PREDICT_FALSE(byte_offset_ * 8 + bit_offset_ + num_bits > max_bytes_ * 8))
    return false;

  buffered_values_ |= v << bit_offset_;
  bit_offset_ += num_bits;

  if (ARROW_PREDICT_FALSE(bit_offset_ >= 64)) {
    // Flush buffered_values_ and write out bits of v that did not fit
    buffered_values_ = arrow::bit_util::ToLittleEndian(buffered_values_);
    memcpy(buffer_ + byte_offset_, &buffered_values_, 8);
    buffered_values_ = 0;
    byte_offset_ += 8;
    bit_offset_ -= 64;
    buffered_values_ =
        (num_bits - bit_offset_ == 64) ? 0 : (v >> (num_bits - bit_offset_));
  }
  DCHECK_LT(bit_offset_, 64);
  return true;
}

inline void BitWriter::Flush(bool align) {
  int num_bytes = static_cast<int>(bit_util::BytesForBits(bit_offset_));
  DCHECK_LE(byte_offset_ + num_bytes, max_bytes_);
  auto buffered_values = arrow::bit_util::ToLittleEndian(buffered_values_);
  memcpy(buffer_ + byte_offset_, &buffered_values, num_bytes);

  if (align) {
    buffered_values_ = 0;
    byte_offset_ += num_bytes;
    bit_offset_ = 0;
  }
}

inline uint8_t* BitWriter::GetNextBytePtr(int num_bytes) {
  Flush(/* align */ true);
  DCHECK_LE(byte_offset_, max_bytes_);
  if (byte_offset_ + num_bytes > max_bytes_) return NULL;
  uint8_t* ptr = buffer_ + byte_offset_;
  byte_offset_ += num_bytes;
  return ptr;
}

template <typename T>
inline bool BitWriter::PutAligned(T val, int num_bytes) {
  uint8_t* ptr = GetNextBytePtr(num_bytes);
  if (ptr == NULL) return false;
  val = arrow::bit_util::ToLittleEndian(val);
  memcpy(ptr, &val, num_bytes);
  return true;
}

namespace detail {

inline void ResetBufferedValues_(const uint8_t* buffer, int byte_offset,
                                 int bytes_remaining, uint64_t* buffered_values) {
  if (ARROW_PREDICT_TRUE(bytes_remaining >= 8)) {
    memcpy(buffered_values, buffer + byte_offset, 8);
  } else {
    memcpy(buffered_values, buffer + byte_offset, bytes_remaining);
  }
  *buffered_values = arrow::bit_util::FromLittleEndian(*buffered_values);
}

template <typename T>
inline void GetValue_(int num_bits, T* v, int max_bytes, const uint8_t* buffer,
                      int* bit_offset, int* byte_offset, uint64_t* buffered_values) {
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4800)
#endif
  *v = static_cast<T>(bit_util::TrailingBits(*buffered_values, *bit_offset + num_bits) >>
                      *bit_offset);
#ifdef _MSC_VER
#pragma warning(pop)
#endif
  *bit_offset += num_bits;
  if (*bit_offset >= 64) {
    *byte_offset += 8;
    *bit_offset -= 64;

    ResetBufferedValues_(buffer, *byte_offset, max_bytes - *byte_offset, buffered_values);
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4800 4805)
#endif
    // Read bits of v that crossed into new buffered_values_
    if (ARROW_PREDICT_TRUE(num_bits - *bit_offset < static_cast<int>(8 * sizeof(T)))) {
      // if shift exponent(num_bits - *bit_offset) is not less than sizeof(T), *v will not
      // change and the following code may cause a runtime error that the shift exponent
      // is too large
      *v = *v | static_cast<T>(bit_util::TrailingBits(*buffered_values, *bit_offset)
                               << (num_bits - *bit_offset));
    }
#ifdef _MSC_VER
#pragma warning(pop)
#endif
    DCHECK_LE(*bit_offset, 64);
  }
}

}  // namespace detail

template <typename T>
inline bool BitReader::GetValue(int num_bits, T* v) {
  return GetBatch(num_bits, v, 1) == 1;
}

template <typename T>
inline int BitReader::GetBatch(int num_bits, T* v, int batch_size) {
  DCHECK(buffer_ != NULL);
  DCHECK_LE(num_bits, static_cast<int>(sizeof(T) * 8));

  int bit_offset = bit_offset_;
  int byte_offset = byte_offset_;
  uint64_t buffered_values = buffered_values_;
  int max_bytes = max_bytes_;
  const uint8_t* buffer = buffer_;

  const int64_t needed_bits = num_bits * static_cast<int64_t>(batch_size);
  constexpr uint64_t kBitsPerByte = 8;
  const int64_t remaining_bits =
      static_cast<int64_t>(max_bytes - byte_offset) * kBitsPerByte - bit_offset;
  if (remaining_bits < needed_bits) {
    batch_size = static_cast<int>(remaining_bits / num_bits);
  }

  int i = 0;
  if (ARROW_PREDICT_FALSE(bit_offset != 0)) {
    for (; i < batch_size && bit_offset != 0; ++i) {
      detail::GetValue_(num_bits, &v[i], max_bytes, buffer, &bit_offset, &byte_offset,
                        &buffered_values);
    }
  }

  if (sizeof(T) == 4) {
    int num_unpacked =
        internal::unpack32(reinterpret_cast<const uint32_t*>(buffer + byte_offset),
                           reinterpret_cast<uint32_t*>(v + i), batch_size - i, num_bits);
    i += num_unpacked;
    byte_offset += num_unpacked * num_bits / 8;
  } else if (sizeof(T) == 8 && num_bits > 32) {
    // Use unpack64 only if num_bits is larger than 32
    // TODO (ARROW-13677): improve the performance of internal::unpack64
    // and remove the restriction of num_bits
    int num_unpacked =
        internal::unpack64(buffer + byte_offset, reinterpret_cast<uint64_t*>(v + i),
                           batch_size - i, num_bits);
    i += num_unpacked;
    byte_offset += num_unpacked * num_bits / 8;
  } else {
    // TODO: revisit this limit if necessary
    DCHECK_LE(num_bits, 32);
    const int buffer_size = 1024;
    uint32_t unpack_buffer[buffer_size];
    while (i < batch_size) {
      int unpack_size = std::min(buffer_size, batch_size - i);
      int num_unpacked =
          internal::unpack32(reinterpret_cast<const uint32_t*>(buffer + byte_offset),
                             unpack_buffer, unpack_size, num_bits);
      if (num_unpacked == 0) {
        break;
      }
      for (int k = 0; k < num_unpacked; ++k) {
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4800)
#endif
        v[i + k] = static_cast<T>(unpack_buffer[k]);
#ifdef _MSC_VER
#pragma warning(pop)
#endif
      }
      i += num_unpacked;
      byte_offset += num_unpacked * num_bits / 8;
    }
  }

  detail::ResetBufferedValues_(buffer, byte_offset, max_bytes - byte_offset,
                               &buffered_values);

  for (; i < batch_size; ++i) {
    detail::GetValue_(num_bits, &v[i], max_bytes, buffer, &bit_offset, &byte_offset,
                      &buffered_values);
  }

  bit_offset_ = bit_offset;
  byte_offset_ = byte_offset;
  buffered_values_ = buffered_values;

  return batch_size;
}

template <typename T>
inline bool BitReader::GetAligned(int num_bytes, T* v) {
  if (ARROW_PREDICT_FALSE(num_bytes > static_cast<int>(sizeof(T)))) {
    return false;
  }

  int bytes_read = static_cast<int>(bit_util::BytesForBits(bit_offset_));
  if (ARROW_PREDICT_FALSE(byte_offset_ + bytes_read + num_bytes > max_bytes_)) {
    return false;
  }

  // Advance byte_offset to next unread byte and read num_bytes
  byte_offset_ += bytes_read;
  if constexpr (std::is_same_v<T, bool>) {
    // ARROW-18031: if we're trying to get an aligned bool, just check
    // the LSB of the next byte and move on. If we memcpy + FromLittleEndian
    // as usual, we have potential undefined behavior for bools if the value
    // isn't 0 or 1
    *v = *(buffer_ + byte_offset_) & 1;
  } else {
    memcpy(v, buffer_ + byte_offset_, num_bytes);
    *v = arrow::bit_util::FromLittleEndian(*v);
  }
  byte_offset_ += num_bytes;

  bit_offset_ = 0;
  detail::ResetBufferedValues_(buffer_, byte_offset_, max_bytes_ - byte_offset_,
                               &buffered_values_);
  return true;
}

inline bool BitReader::Advance(int64_t num_bits) {
  int64_t bits_required = bit_offset_ + num_bits;
  int64_t bytes_required = bit_util::BytesForBits(bits_required);
  if (ARROW_PREDICT_FALSE(bytes_required > max_bytes_ - byte_offset_)) {
    return false;
  }
  byte_offset_ += static_cast<int>(bits_required >> 3);
  bit_offset_ = static_cast<int>(bits_required & 7);
  detail::ResetBufferedValues_(buffer_, byte_offset_, max_bytes_ - byte_offset_,
                               &buffered_values_);
  return true;
}

inline bool BitWriter::PutVlqInt(uint32_t v) {
  bool result = true;
  while ((v & 0xFFFFFF80UL) != 0UL) {
    result &= PutAligned<uint8_t>(static_cast<uint8_t>((v & 0x7F) | 0x80), 1);
    v >>= 7;
  }
  result &= PutAligned<uint8_t>(static_cast<uint8_t>(v & 0x7F), 1);
  return result;
}

inline bool BitReader::GetVlqInt(uint32_t* v) {
  uint32_t tmp = 0;

  for (int i = 0; i < kMaxVlqByteLength; i++) {
    uint8_t byte = 0;
    if (ARROW_PREDICT_FALSE(!GetAligned<uint8_t>(1, &byte))) {
      return false;
    }
    tmp |= static_cast<uint32_t>(byte & 0x7F) << (7 * i);

    if ((byte & 0x80) == 0) {
      *v = tmp;
      return true;
    }
  }

  return false;
}

inline bool BitWriter::PutZigZagVlqInt(int32_t v) {
  uint32_t u_v = ::arrow::util::SafeCopy<uint32_t>(v);
  u_v = (u_v << 1) ^ static_cast<uint32_t>(v >> 31);
  return PutVlqInt(u_v);
}

inline bool BitReader::GetZigZagVlqInt(int32_t* v) {
  uint32_t u;
  if (!GetVlqInt(&u)) return false;
  u = (u >> 1) ^ (~(u & 1) + 1);
  *v = ::arrow::util::SafeCopy<int32_t>(u);
  return true;
}

inline bool BitWriter::PutVlqInt(uint64_t v) {
  bool result = true;
  while ((v & 0xFFFFFFFFFFFFFF80ULL) != 0ULL) {
    result &= PutAligned<uint8_t>(static_cast<uint8_t>((v & 0x7F) | 0x80), 1);
    v >>= 7;
  }
  result &= PutAligned<uint8_t>(static_cast<uint8_t>(v & 0x7F), 1);
  return result;
}

inline bool BitReader::GetVlqInt(uint64_t* v) {
  uint64_t tmp = 0;

  for (int i = 0; i < kMaxVlqByteLengthForInt64; i++) {
    uint8_t byte = 0;
    if (ARROW_PREDICT_FALSE(!GetAligned<uint8_t>(1, &byte))) {
      return false;
    }
    tmp |= static_cast<uint64_t>(byte & 0x7F) << (7 * i);

    if ((byte & 0x80) == 0) {
      *v = tmp;
      return true;
    }
  }

  return false;
}

inline bool BitWriter::PutZigZagVlqInt(int64_t v) {
  uint64_t u_v = ::arrow::util::SafeCopy<uint64_t>(v);
  u_v = (u_v << 1) ^ static_cast<uint64_t>(v >> 63);
  return PutVlqInt(u_v);
}

inline bool BitReader::GetZigZagVlqInt(int64_t* v) {
  uint64_t u;
  if (!GetVlqInt(&u)) return false;
  u = (u >> 1) ^ (~(u & 1) + 1);
  *v = ::arrow::util::SafeCopy<int64_t>(u);
  return true;
}

}  // namespace bit_util
}  // namespace arrow

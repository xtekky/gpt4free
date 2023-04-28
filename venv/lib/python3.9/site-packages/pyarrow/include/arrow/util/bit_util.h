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

#if defined(_MSC_VER)
#if defined(_M_AMD64) || defined(_M_X64)
#include <intrin.h>  // IWYU pragma: keep
#include <nmmintrin.h>
#endif

#pragma intrinsic(_BitScanReverse)
#pragma intrinsic(_BitScanForward)
#define ARROW_POPCOUNT64 __popcnt64
#define ARROW_POPCOUNT32 __popcnt
#else
#define ARROW_POPCOUNT64 __builtin_popcountll
#define ARROW_POPCOUNT32 __builtin_popcount
#endif

#include <cstdint>
#include <type_traits>

#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace detail {

template <typename Integer>
typename std::make_unsigned<Integer>::type as_unsigned(Integer x) {
  return static_cast<typename std::make_unsigned<Integer>::type>(x);
}

}  // namespace detail

namespace bit_util {

// The number of set bits in a given unsigned byte value, pre-computed
//
// Generated with the following Python code
// output = 'static constexpr uint8_t kBytePopcount[] = {{{0}}};'
// popcounts = [str(bin(i).count('1')) for i in range(0, 256)]
// print(output.format(', '.join(popcounts)))
static constexpr uint8_t kBytePopcount[] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3,
    4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4,
    4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4,
    5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5,
    4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2,
    3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5,
    5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4,
    5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6,
    4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};

static inline uint64_t PopCount(uint64_t bitmap) { return ARROW_POPCOUNT64(bitmap); }
static inline uint32_t PopCount(uint32_t bitmap) { return ARROW_POPCOUNT32(bitmap); }

//
// Bit-related computations on integer values
//

// Returns the ceil of value/divisor
constexpr int64_t CeilDiv(int64_t value, int64_t divisor) {
  return (value == 0) ? 0 : 1 + (value - 1) / divisor;
}

// Return the number of bytes needed to fit the given number of bits
constexpr int64_t BytesForBits(int64_t bits) {
  // This formula avoids integer overflow on very large `bits`
  return (bits >> 3) + ((bits & 7) != 0);
}

constexpr bool IsPowerOf2(int64_t value) {
  return value > 0 && (value & (value - 1)) == 0;
}

constexpr bool IsPowerOf2(uint64_t value) {
  return value > 0 && (value & (value - 1)) == 0;
}

// Returns the smallest power of two that contains v.  If v is already a
// power of two, it is returned as is.
static inline int64_t NextPower2(int64_t n) {
  // Taken from
  // http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n |= n >> 32;
  n++;
  return n;
}

constexpr bool IsMultipleOf64(int64_t n) { return (n & 63) == 0; }

constexpr bool IsMultipleOf8(int64_t n) { return (n & 7) == 0; }

// Returns a mask for the bit_index lower order bits.
// Only valid for bit_index in the range [0, 64).
constexpr uint64_t LeastSignificantBitMask(int64_t bit_index) {
  return (static_cast<uint64_t>(1) << bit_index) - 1;
}

// Returns 'value' rounded up to the nearest multiple of 'factor'
constexpr int64_t RoundUp(int64_t value, int64_t factor) {
  return CeilDiv(value, factor) * factor;
}

// Returns 'value' rounded down to the nearest multiple of 'factor'
constexpr int64_t RoundDown(int64_t value, int64_t factor) {
  return (value / factor) * factor;
}

// Returns 'value' rounded up to the nearest multiple of 'factor' when factor
// is a power of two.
// The result is undefined on overflow, i.e. if `value > 2**64 - factor`,
// since we cannot return the correct result which would be 2**64.
constexpr int64_t RoundUpToPowerOf2(int64_t value, int64_t factor) {
  // DCHECK(value >= 0);
  // DCHECK(IsPowerOf2(factor));
  return (value + (factor - 1)) & ~(factor - 1);
}

constexpr uint64_t RoundUpToPowerOf2(uint64_t value, uint64_t factor) {
  // DCHECK(IsPowerOf2(factor));
  return (value + (factor - 1)) & ~(factor - 1);
}

constexpr int64_t RoundUpToMultipleOf8(int64_t num) { return RoundUpToPowerOf2(num, 8); }

constexpr int64_t RoundUpToMultipleOf64(int64_t num) {
  return RoundUpToPowerOf2(num, 64);
}

// Returns the number of bytes covering a sliced bitmap. Find the length
// rounded to cover full bytes on both extremities.
//
// The following example represents a slice (offset=10, length=9)
//
// 0       8       16     24
// |-------|-------|------|
//           [       ]          (slice)
//         [             ]      (same slice aligned to bytes bounds, length=16)
//
// The covering bytes is the length (in bytes) of this new aligned slice.
constexpr int64_t CoveringBytes(int64_t offset, int64_t length) {
  return (bit_util::RoundUp(length + offset, 8) - bit_util::RoundDown(offset, 8)) / 8;
}

// Returns the 'num_bits' least-significant bits of 'v'.
static inline uint64_t TrailingBits(uint64_t v, int num_bits) {
  if (ARROW_PREDICT_FALSE(num_bits == 0)) return 0;
  if (ARROW_PREDICT_FALSE(num_bits >= 64)) return v;
  int n = 64 - num_bits;
  return (v << n) >> n;
}

/// \brief Count the number of leading zeros in an unsigned integer.
static inline int CountLeadingZeros(uint32_t value) {
#if defined(__clang__) || defined(__GNUC__)
  if (value == 0) return 32;
  return static_cast<int>(__builtin_clz(value));
#elif defined(_MSC_VER)
  unsigned long index;                                               // NOLINT
  if (_BitScanReverse(&index, static_cast<unsigned long>(value))) {  // NOLINT
    return 31 - static_cast<int>(index);
  } else {
    return 32;
  }
#else
  int bitpos = 0;
  while (value != 0) {
    value >>= 1;
    ++bitpos;
  }
  return 32 - bitpos;
#endif
}

static inline int CountLeadingZeros(uint64_t value) {
#if defined(__clang__) || defined(__GNUC__)
  if (value == 0) return 64;
  return static_cast<int>(__builtin_clzll(value));
#elif defined(_MSC_VER)
  unsigned long index;                     // NOLINT
  if (_BitScanReverse64(&index, value)) {  // NOLINT
    return 63 - static_cast<int>(index);
  } else {
    return 64;
  }
#else
  int bitpos = 0;
  while (value != 0) {
    value >>= 1;
    ++bitpos;
  }
  return 64 - bitpos;
#endif
}

static inline int CountTrailingZeros(uint32_t value) {
#if defined(__clang__) || defined(__GNUC__)
  if (value == 0) return 32;
  return static_cast<int>(__builtin_ctzl(value));
#elif defined(_MSC_VER)
  unsigned long index;  // NOLINT
  if (_BitScanForward(&index, value)) {
    return static_cast<int>(index);
  } else {
    return 32;
  }
#else
  int bitpos = 0;
  if (value) {
    while (value & 1 == 0) {
      value >>= 1;
      ++bitpos;
    }
  } else {
    bitpos = 32;
  }
  return bitpos;
#endif
}

static inline int CountTrailingZeros(uint64_t value) {
#if defined(__clang__) || defined(__GNUC__)
  if (value == 0) return 64;
  return static_cast<int>(__builtin_ctzll(value));
#elif defined(_MSC_VER)
  unsigned long index;  // NOLINT
  if (_BitScanForward64(&index, value)) {
    return static_cast<int>(index);
  } else {
    return 64;
  }
#else
  int bitpos = 0;
  if (value) {
    while (value & 1 == 0) {
      value >>= 1;
      ++bitpos;
    }
  } else {
    bitpos = 64;
  }
  return bitpos;
#endif
}

// Returns the minimum number of bits needed to represent an unsigned value
static inline int NumRequiredBits(uint64_t x) { return 64 - CountLeadingZeros(x); }

// Returns ceil(log2(x)).
static inline int Log2(uint64_t x) {
  // DCHECK_GT(x, 0);
  return NumRequiredBits(x - 1);
}

//
// Utilities for reading and writing individual bits by their index
// in a memory area.
//

// Bitmask selecting the k-th bit in a byte
static constexpr uint8_t kBitmask[] = {1, 2, 4, 8, 16, 32, 64, 128};

// the bitwise complement version of kBitmask
static constexpr uint8_t kFlippedBitmask[] = {254, 253, 251, 247, 239, 223, 191, 127};

// Bitmask selecting the (k - 1) preceding bits in a byte
static constexpr uint8_t kPrecedingBitmask[] = {0, 1, 3, 7, 15, 31, 63, 127};
static constexpr uint8_t kPrecedingWrappingBitmask[] = {255, 1, 3, 7, 15, 31, 63, 127};

// the bitwise complement version of kPrecedingBitmask
static constexpr uint8_t kTrailingBitmask[] = {255, 254, 252, 248, 240, 224, 192, 128};

static constexpr bool GetBit(const uint8_t* bits, uint64_t i) {
  return (bits[i >> 3] >> (i & 0x07)) & 1;
}

// Gets the i-th bit from a byte. Should only be used with i <= 7.
static constexpr bool GetBitFromByte(uint8_t byte, uint8_t i) {
  return byte & kBitmask[i];
}

static inline void ClearBit(uint8_t* bits, int64_t i) {
  bits[i / 8] &= kFlippedBitmask[i % 8];
}

static inline void SetBit(uint8_t* bits, int64_t i) { bits[i / 8] |= kBitmask[i % 8]; }

static inline void SetBitTo(uint8_t* bits, int64_t i, bool bit_is_set) {
  // https://graphics.stanford.edu/~seander/bithacks.html
  // "Conditionally set or clear bits without branching"
  // NOTE: this seems to confuse Valgrind as it reads from potentially
  // uninitialized memory
  bits[i / 8] ^= static_cast<uint8_t>(-static_cast<uint8_t>(bit_is_set) ^ bits[i / 8]) &
                 kBitmask[i % 8];
}

/// \brief set or clear a range of bits quickly
ARROW_EXPORT
void SetBitsTo(uint8_t* bits, int64_t start_offset, int64_t length, bool bits_are_set);

/// \brief Sets all bits in the bitmap to true
ARROW_EXPORT
void SetBitmap(uint8_t* data, int64_t offset, int64_t length);

/// \brief Clears all bits in the bitmap (set to false)
ARROW_EXPORT
void ClearBitmap(uint8_t* data, int64_t offset, int64_t length);

/// Returns a mask with lower i bits set to 1. If i >= sizeof(Word)*8, all-ones will be
/// returned
/// ex:
/// ref: https://stackoverflow.com/a/59523400
template <typename Word>
constexpr Word PrecedingWordBitmask(unsigned int const i) {
  return (static_cast<Word>(i < sizeof(Word) * 8) << (i & (sizeof(Word) * 8 - 1))) - 1;
}
static_assert(PrecedingWordBitmask<uint8_t>(0) == 0x00, "");
static_assert(PrecedingWordBitmask<uint8_t>(4) == 0x0f, "");
static_assert(PrecedingWordBitmask<uint8_t>(8) == 0xff, "");
static_assert(PrecedingWordBitmask<uint16_t>(8) == 0x00ff, "");

/// \brief Create a word with low `n` bits from `low` and high `sizeof(Word)-n` bits
/// from `high`.
/// Word ret
/// for (i = 0; i < sizeof(Word)*8; i++){
///     ret[i]= i < n ? low[i]: high[i];
/// }
template <typename Word>
constexpr Word SpliceWord(int n, Word low, Word high) {
  return (high & ~PrecedingWordBitmask<Word>(n)) | (low & PrecedingWordBitmask<Word>(n));
}

/// \brief Pack integers into a bitmap in batches of 8
template <int batch_size>
void PackBits(const uint32_t* values, uint8_t* out) {
  for (int i = 0; i < batch_size / 8; ++i) {
    *out++ = (values[0] | values[1] << 1 | values[2] << 2 | values[3] << 3 |
              values[4] << 4 | values[5] << 5 | values[6] << 6 | values[7] << 7);
    values += 8;
  }
}

}  // namespace bit_util
}  // namespace arrow

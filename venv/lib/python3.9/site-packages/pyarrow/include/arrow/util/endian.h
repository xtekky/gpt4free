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

#ifdef _WIN32
#define ARROW_LITTLE_ENDIAN 1
#else
#if defined(__APPLE__) || defined(__FreeBSD__)
#include <machine/endian.h>  // IWYU pragma: keep
#elif defined(sun) || defined(__sun)
#include <sys/byteorder.h>  // IWYU pragma: keep
#else
#include <endian.h>  // IWYU pragma: keep
#endif
#
#ifndef __BYTE_ORDER__
#error "__BYTE_ORDER__ not defined"
#endif
#
#ifndef __ORDER_LITTLE_ENDIAN__
#error "__ORDER_LITTLE_ENDIAN__ not defined"
#endif
#
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define ARROW_LITTLE_ENDIAN 1
#else
#define ARROW_LITTLE_ENDIAN 0
#endif
#endif

#if defined(_MSC_VER)
#include <intrin.h>  // IWYU pragma: keep
#define ARROW_BYTE_SWAP64 _byteswap_uint64
#define ARROW_BYTE_SWAP32 _byteswap_ulong
#else
#define ARROW_BYTE_SWAP64 __builtin_bswap64
#define ARROW_BYTE_SWAP32 __builtin_bswap32
#endif

#include <algorithm>
#include <array>

#include "arrow/util/type_traits.h"
#include "arrow/util/ubsan.h"

namespace arrow {
namespace bit_util {

//
// Byte-swap 16-bit, 32-bit and 64-bit values
//

// Swap the byte order (i.e. endianness)
static inline int64_t ByteSwap(int64_t value) { return ARROW_BYTE_SWAP64(value); }
static inline uint64_t ByteSwap(uint64_t value) {
  return static_cast<uint64_t>(ARROW_BYTE_SWAP64(value));
}
static inline int32_t ByteSwap(int32_t value) { return ARROW_BYTE_SWAP32(value); }
static inline uint32_t ByteSwap(uint32_t value) {
  return static_cast<uint32_t>(ARROW_BYTE_SWAP32(value));
}
static inline int16_t ByteSwap(int16_t value) {
  constexpr auto m = static_cast<int16_t>(0xff);
  return static_cast<int16_t>(((value >> 8) & m) | ((value & m) << 8));
}
static inline uint16_t ByteSwap(uint16_t value) {
  return static_cast<uint16_t>(ByteSwap(static_cast<int16_t>(value)));
}
static inline uint8_t ByteSwap(uint8_t value) { return value; }
static inline int8_t ByteSwap(int8_t value) { return value; }
static inline double ByteSwap(double value) {
  const uint64_t swapped = ARROW_BYTE_SWAP64(util::SafeCopy<uint64_t>(value));
  return util::SafeCopy<double>(swapped);
}
static inline float ByteSwap(float value) {
  const uint32_t swapped = ARROW_BYTE_SWAP32(util::SafeCopy<uint32_t>(value));
  return util::SafeCopy<float>(swapped);
}

// Write the swapped bytes into dst. Src and dst cannot overlap.
static inline void ByteSwap(void* dst, const void* src, int len) {
  switch (len) {
    case 1:
      *reinterpret_cast<int8_t*>(dst) = *reinterpret_cast<const int8_t*>(src);
      return;
    case 2:
      *reinterpret_cast<int16_t*>(dst) = ByteSwap(*reinterpret_cast<const int16_t*>(src));
      return;
    case 4:
      *reinterpret_cast<int32_t*>(dst) = ByteSwap(*reinterpret_cast<const int32_t*>(src));
      return;
    case 8:
      *reinterpret_cast<int64_t*>(dst) = ByteSwap(*reinterpret_cast<const int64_t*>(src));
      return;
    default:
      break;
  }

  auto d = reinterpret_cast<uint8_t*>(dst);
  auto s = reinterpret_cast<const uint8_t*>(src);
  for (int i = 0; i < len; ++i) {
    d[i] = s[len - i - 1];
  }
}

// Convert to little/big endian format from the machine's native endian format.
#if ARROW_LITTLE_ENDIAN
template <typename T, typename = internal::EnableIfIsOneOf<
                          T, int64_t, uint64_t, int32_t, uint32_t, int16_t, uint16_t,
                          uint8_t, int8_t, float, double, bool>>
static inline T ToBigEndian(T value) {
  return ByteSwap(value);
}

template <typename T, typename = internal::EnableIfIsOneOf<
                          T, int64_t, uint64_t, int32_t, uint32_t, int16_t, uint16_t,
                          uint8_t, int8_t, float, double, bool>>
static inline T ToLittleEndian(T value) {
  return value;
}
#else
template <typename T, typename = internal::EnableIfIsOneOf<
                          T, int64_t, uint64_t, int32_t, uint32_t, int16_t, uint16_t,
                          uint8_t, int8_t, float, double, bool>>
static inline T ToBigEndian(T value) {
  return value;
}

template <typename T, typename = internal::EnableIfIsOneOf<
                          T, int64_t, uint64_t, int32_t, uint32_t, int16_t, uint16_t,
                          uint8_t, int8_t, float, double, bool>>
static inline T ToLittleEndian(T value) {
  return ByteSwap(value);
}
#endif

// Convert from big/little endian format to the machine's native endian format.
#if ARROW_LITTLE_ENDIAN
template <typename T, typename = internal::EnableIfIsOneOf<
                          T, int64_t, uint64_t, int32_t, uint32_t, int16_t, uint16_t,
                          uint8_t, int8_t, float, double, bool>>
static inline T FromBigEndian(T value) {
  return ByteSwap(value);
}

template <typename T, typename = internal::EnableIfIsOneOf<
                          T, int64_t, uint64_t, int32_t, uint32_t, int16_t, uint16_t,
                          uint8_t, int8_t, float, double, bool>>
static inline T FromLittleEndian(T value) {
  return value;
}
#else
template <typename T, typename = internal::EnableIfIsOneOf<
                          T, int64_t, uint64_t, int32_t, uint32_t, int16_t, uint16_t,
                          uint8_t, int8_t, float, double, bool>>
static inline T FromBigEndian(T value) {
  return value;
}

template <typename T, typename = internal::EnableIfIsOneOf<
                          T, int64_t, uint64_t, int32_t, uint32_t, int16_t, uint16_t,
                          uint8_t, int8_t, float, double, bool>>
static inline T FromLittleEndian(T value) {
  return ByteSwap(value);
}
#endif

// Handle endianness in *word* granuality (keep individual array element untouched)
namespace little_endian {

namespace detail {

// Read a native endian array as little endian
template <typename T, size_t N>
struct Reader {
  const std::array<T, N>& native_array;

  explicit Reader(const std::array<T, N>& native_array) : native_array(native_array) {}

  const T& operator[](size_t i) const {
    return native_array[ARROW_LITTLE_ENDIAN ? i : N - 1 - i];
  }
};

// Read/write a native endian array as little endian
template <typename T, size_t N>
struct Writer {
  std::array<T, N>* native_array;

  explicit Writer(std::array<T, N>* native_array) : native_array(native_array) {}

  const T& operator[](size_t i) const {
    return (*native_array)[ARROW_LITTLE_ENDIAN ? i : N - 1 - i];
  }
  T& operator[](size_t i) { return (*native_array)[ARROW_LITTLE_ENDIAN ? i : N - 1 - i]; }
};

}  // namespace detail

// Construct array reader and try to deduce template augments
template <typename T, size_t N>
static inline detail::Reader<T, N> Make(const std::array<T, N>& native_array) {
  return detail::Reader<T, N>(native_array);
}

// Construct array writer and try to deduce template augments
template <typename T, size_t N>
static inline detail::Writer<T, N> Make(std::array<T, N>* native_array) {
  return detail::Writer<T, N>(native_array);
}

// Convert little endian array to native endian
template <typename T, size_t N>
static inline std::array<T, N> ToNative(std::array<T, N> array) {
  if (!ARROW_LITTLE_ENDIAN) {
    std::reverse(array.begin(), array.end());
  }
  return array;
}

// Convert native endian array to little endian
template <typename T, size_t N>
static inline std::array<T, N> FromNative(std::array<T, N> array) {
  return ToNative(array);
}

}  // namespace little_endian

}  // namespace bit_util
}  // namespace arrow

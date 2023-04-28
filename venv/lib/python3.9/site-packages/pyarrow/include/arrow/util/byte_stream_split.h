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

#include "arrow/util/simd.h"
#include "arrow/util/ubsan.h"

#include <stdint.h>
#include <algorithm>

#ifdef ARROW_HAVE_SSE4_2
// Enable the SIMD for ByteStreamSplit Encoder/Decoder
#define ARROW_HAVE_SIMD_SPLIT
#endif  // ARROW_HAVE_SSE4_2

namespace arrow {
namespace util {
namespace internal {

#if defined(ARROW_HAVE_SSE4_2)
template <typename T>
void ByteStreamSplitDecodeSse2(const uint8_t* data, int64_t num_values, int64_t stride,
                               T* out) {
  constexpr size_t kNumStreams = sizeof(T);
  static_assert(kNumStreams == 4U || kNumStreams == 8U, "Invalid number of streams.");
  constexpr size_t kNumStreamsLog2 = (kNumStreams == 8U ? 3U : 2U);

  const int64_t size = num_values * sizeof(T);
  constexpr int64_t kBlockSize = sizeof(__m128i) * kNumStreams;
  const int64_t num_blocks = size / kBlockSize;
  uint8_t* output_data = reinterpret_cast<uint8_t*>(out);

  // First handle suffix.
  // This helps catch if the simd-based processing overflows into the suffix
  // since almost surely a test would fail.
  const int64_t num_processed_elements = (num_blocks * kBlockSize) / kNumStreams;
  for (int64_t i = num_processed_elements; i < num_values; ++i) {
    uint8_t gathered_byte_data[kNumStreams];
    for (size_t b = 0; b < kNumStreams; ++b) {
      const size_t byte_index = b * stride + i;
      gathered_byte_data[b] = data[byte_index];
    }
    out[i] = arrow::util::SafeLoadAs<T>(&gathered_byte_data[0]);
  }

  // The blocks get processed hierarchically using the unpack intrinsics.
  // Example with four streams:
  // Stage 1: AAAA BBBB CCCC DDDD
  // Stage 2: ACAC ACAC BDBD BDBD
  // Stage 3: ABCD ABCD ABCD ABCD
  __m128i stage[kNumStreamsLog2 + 1U][kNumStreams];
  constexpr size_t kNumStreamsHalf = kNumStreams / 2U;

  for (int64_t i = 0; i < num_blocks; ++i) {
    for (size_t j = 0; j < kNumStreams; ++j) {
      stage[0][j] = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(&data[i * sizeof(__m128i) + j * stride]));
    }
    for (size_t step = 0; step < kNumStreamsLog2; ++step) {
      for (size_t j = 0; j < kNumStreamsHalf; ++j) {
        stage[step + 1U][j * 2] =
            _mm_unpacklo_epi8(stage[step][j], stage[step][kNumStreamsHalf + j]);
        stage[step + 1U][j * 2 + 1U] =
            _mm_unpackhi_epi8(stage[step][j], stage[step][kNumStreamsHalf + j]);
      }
    }
    for (size_t j = 0; j < kNumStreams; ++j) {
      _mm_storeu_si128(reinterpret_cast<__m128i*>(
                           &output_data[(i * kNumStreams + j) * sizeof(__m128i)]),
                       stage[kNumStreamsLog2][j]);
    }
  }
}

template <typename T>
void ByteStreamSplitEncodeSse2(const uint8_t* raw_values, const size_t num_values,
                               uint8_t* output_buffer_raw) {
  constexpr size_t kNumStreams = sizeof(T);
  static_assert(kNumStreams == 4U || kNumStreams == 8U, "Invalid number of streams.");
  __m128i stage[3][kNumStreams];
  __m128i final_result[kNumStreams];

  const size_t size = num_values * sizeof(T);
  constexpr size_t kBlockSize = sizeof(__m128i) * kNumStreams;
  const size_t num_blocks = size / kBlockSize;
  const __m128i* raw_values_sse = reinterpret_cast<const __m128i*>(raw_values);
  __m128i* output_buffer_streams[kNumStreams];
  for (size_t i = 0; i < kNumStreams; ++i) {
    output_buffer_streams[i] =
        reinterpret_cast<__m128i*>(&output_buffer_raw[num_values * i]);
  }

  // First handle suffix.
  const size_t num_processed_elements = (num_blocks * kBlockSize) / sizeof(T);
  for (size_t i = num_processed_elements; i < num_values; ++i) {
    for (size_t j = 0U; j < kNumStreams; ++j) {
      const uint8_t byte_in_value = raw_values[i * kNumStreams + j];
      output_buffer_raw[j * num_values + i] = byte_in_value;
    }
  }
  // The current shuffling algorithm diverges for float and double types but the compiler
  // should be able to remove the branch since only one path is taken for each template
  // instantiation.
  // Example run for floats:
  // Step 0, copy:
  //   0: ABCD ABCD ABCD ABCD 1: ABCD ABCD ABCD ABCD ...
  // Step 1: _mm_unpacklo_epi8 and mm_unpackhi_epi8:
  //   0: AABB CCDD AABB CCDD 1: AABB CCDD AABB CCDD ...
  //   0: AAAA BBBB CCCC DDDD 1: AAAA BBBB CCCC DDDD ...
  // Step 3: __mm_unpacklo_epi8 and _mm_unpackhi_epi8:
  //   0: AAAA AAAA BBBB BBBB 1: CCCC CCCC DDDD DDDD ...
  // Step 4: __mm_unpacklo_epi64 and _mm_unpackhi_epi64:
  //   0: AAAA AAAA AAAA AAAA 1: BBBB BBBB BBBB BBBB ...
  for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
    // First copy the data to stage 0.
    for (size_t i = 0; i < kNumStreams; ++i) {
      stage[0][i] = _mm_loadu_si128(&raw_values_sse[block_index * kNumStreams + i]);
    }

    // The shuffling of bytes is performed through the unpack intrinsics.
    // In my measurements this gives better performance then an implementation
    // which uses the shuffle intrinsics.
    for (size_t stage_lvl = 0; stage_lvl < 2U; ++stage_lvl) {
      for (size_t i = 0; i < kNumStreams / 2U; ++i) {
        stage[stage_lvl + 1][i * 2] =
            _mm_unpacklo_epi8(stage[stage_lvl][i * 2], stage[stage_lvl][i * 2 + 1]);
        stage[stage_lvl + 1][i * 2 + 1] =
            _mm_unpackhi_epi8(stage[stage_lvl][i * 2], stage[stage_lvl][i * 2 + 1]);
      }
    }
    if (kNumStreams == 8U) {
      // This is the path for double.
      __m128i tmp[8];
      for (size_t i = 0; i < 4; ++i) {
        tmp[i * 2] = _mm_unpacklo_epi32(stage[2][i], stage[2][i + 4]);
        tmp[i * 2 + 1] = _mm_unpackhi_epi32(stage[2][i], stage[2][i + 4]);
      }

      for (size_t i = 0; i < 4; ++i) {
        final_result[i * 2] = _mm_unpacklo_epi32(tmp[i], tmp[i + 4]);
        final_result[i * 2 + 1] = _mm_unpackhi_epi32(tmp[i], tmp[i + 4]);
      }
    } else {
      // this is the path for float.
      __m128i tmp[4];
      for (size_t i = 0; i < 2; ++i) {
        tmp[i * 2] = _mm_unpacklo_epi8(stage[2][i * 2], stage[2][i * 2 + 1]);
        tmp[i * 2 + 1] = _mm_unpackhi_epi8(stage[2][i * 2], stage[2][i * 2 + 1]);
      }
      for (size_t i = 0; i < 2; ++i) {
        final_result[i * 2] = _mm_unpacklo_epi64(tmp[i], tmp[i + 2]);
        final_result[i * 2 + 1] = _mm_unpackhi_epi64(tmp[i], tmp[i + 2]);
      }
    }
    for (size_t i = 0; i < kNumStreams; ++i) {
      _mm_storeu_si128(&output_buffer_streams[i][block_index], final_result[i]);
    }
  }
}
#endif  // ARROW_HAVE_SSE4_2

#if defined(ARROW_HAVE_AVX2)
template <typename T>
void ByteStreamSplitDecodeAvx2(const uint8_t* data, int64_t num_values, int64_t stride,
                               T* out) {
  constexpr size_t kNumStreams = sizeof(T);
  static_assert(kNumStreams == 4U || kNumStreams == 8U, "Invalid number of streams.");
  constexpr size_t kNumStreamsLog2 = (kNumStreams == 8U ? 3U : 2U);

  const int64_t size = num_values * sizeof(T);
  constexpr int64_t kBlockSize = sizeof(__m256i) * kNumStreams;
  if (size < kBlockSize)  // Back to SSE for small size
    return ByteStreamSplitDecodeSse2(data, num_values, stride, out);
  const int64_t num_blocks = size / kBlockSize;
  uint8_t* output_data = reinterpret_cast<uint8_t*>(out);

  // First handle suffix.
  const int64_t num_processed_elements = (num_blocks * kBlockSize) / kNumStreams;
  for (int64_t i = num_processed_elements; i < num_values; ++i) {
    uint8_t gathered_byte_data[kNumStreams];
    for (size_t b = 0; b < kNumStreams; ++b) {
      const size_t byte_index = b * stride + i;
      gathered_byte_data[b] = data[byte_index];
    }
    out[i] = arrow::util::SafeLoadAs<T>(&gathered_byte_data[0]);
  }

  // Processed hierarchically using unpack intrinsics, then permute intrinsics.
  __m256i stage[kNumStreamsLog2 + 1U][kNumStreams];
  __m256i final_result[kNumStreams];
  constexpr size_t kNumStreamsHalf = kNumStreams / 2U;

  for (int64_t i = 0; i < num_blocks; ++i) {
    for (size_t j = 0; j < kNumStreams; ++j) {
      stage[0][j] = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(&data[i * sizeof(__m256i) + j * stride]));
    }

    for (size_t step = 0; step < kNumStreamsLog2; ++step) {
      for (size_t j = 0; j < kNumStreamsHalf; ++j) {
        stage[step + 1U][j * 2] =
            _mm256_unpacklo_epi8(stage[step][j], stage[step][kNumStreamsHalf + j]);
        stage[step + 1U][j * 2 + 1U] =
            _mm256_unpackhi_epi8(stage[step][j], stage[step][kNumStreamsHalf + j]);
      }
    }

    if (kNumStreams == 8U) {
      // path for double, 128i index:
      //   {0x00, 0x08}, {0x01, 0x09}, {0x02, 0x0A}, {0x03, 0x0B},
      //   {0x04, 0x0C}, {0x05, 0x0D}, {0x06, 0x0E}, {0x07, 0x0F},
      final_result[0] = _mm256_permute2x128_si256(stage[kNumStreamsLog2][0],
                                                  stage[kNumStreamsLog2][1], 0b00100000);
      final_result[1] = _mm256_permute2x128_si256(stage[kNumStreamsLog2][2],
                                                  stage[kNumStreamsLog2][3], 0b00100000);
      final_result[2] = _mm256_permute2x128_si256(stage[kNumStreamsLog2][4],
                                                  stage[kNumStreamsLog2][5], 0b00100000);
      final_result[3] = _mm256_permute2x128_si256(stage[kNumStreamsLog2][6],
                                                  stage[kNumStreamsLog2][7], 0b00100000);
      final_result[4] = _mm256_permute2x128_si256(stage[kNumStreamsLog2][0],
                                                  stage[kNumStreamsLog2][1], 0b00110001);
      final_result[5] = _mm256_permute2x128_si256(stage[kNumStreamsLog2][2],
                                                  stage[kNumStreamsLog2][3], 0b00110001);
      final_result[6] = _mm256_permute2x128_si256(stage[kNumStreamsLog2][4],
                                                  stage[kNumStreamsLog2][5], 0b00110001);
      final_result[7] = _mm256_permute2x128_si256(stage[kNumStreamsLog2][6],
                                                  stage[kNumStreamsLog2][7], 0b00110001);
    } else {
      // path for float, 128i index:
      //   {0x00, 0x04}, {0x01, 0x05}, {0x02, 0x06}, {0x03, 0x07}
      final_result[0] = _mm256_permute2x128_si256(stage[kNumStreamsLog2][0],
                                                  stage[kNumStreamsLog2][1], 0b00100000);
      final_result[1] = _mm256_permute2x128_si256(stage[kNumStreamsLog2][2],
                                                  stage[kNumStreamsLog2][3], 0b00100000);
      final_result[2] = _mm256_permute2x128_si256(stage[kNumStreamsLog2][0],
                                                  stage[kNumStreamsLog2][1], 0b00110001);
      final_result[3] = _mm256_permute2x128_si256(stage[kNumStreamsLog2][2],
                                                  stage[kNumStreamsLog2][3], 0b00110001);
    }

    for (size_t j = 0; j < kNumStreams; ++j) {
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(
                              &output_data[(i * kNumStreams + j) * sizeof(__m256i)]),
                          final_result[j]);
    }
  }
}

template <typename T>
void ByteStreamSplitEncodeAvx2(const uint8_t* raw_values, const size_t num_values,
                               uint8_t* output_buffer_raw) {
  constexpr size_t kNumStreams = sizeof(T);
  static_assert(kNumStreams == 4U || kNumStreams == 8U, "Invalid number of streams.");
  if (kNumStreams == 8U)  // Back to SSE, currently no path for double.
    return ByteStreamSplitEncodeSse2<T>(raw_values, num_values, output_buffer_raw);

  const size_t size = num_values * sizeof(T);
  constexpr size_t kBlockSize = sizeof(__m256i) * kNumStreams;
  if (size < kBlockSize)  // Back to SSE for small size
    return ByteStreamSplitEncodeSse2<T>(raw_values, num_values, output_buffer_raw);
  const size_t num_blocks = size / kBlockSize;
  const __m256i* raw_values_simd = reinterpret_cast<const __m256i*>(raw_values);
  __m256i* output_buffer_streams[kNumStreams];

  for (size_t i = 0; i < kNumStreams; ++i) {
    output_buffer_streams[i] =
        reinterpret_cast<__m256i*>(&output_buffer_raw[num_values * i]);
  }

  // First handle suffix.
  const size_t num_processed_elements = (num_blocks * kBlockSize) / sizeof(T);
  for (size_t i = num_processed_elements; i < num_values; ++i) {
    for (size_t j = 0U; j < kNumStreams; ++j) {
      const uint8_t byte_in_value = raw_values[i * kNumStreams + j];
      output_buffer_raw[j * num_values + i] = byte_in_value;
    }
  }

  // Path for float.
  // 1. Processed hierarchically to 32i blcok using the unpack intrinsics.
  // 2. Pack 128i block using _mm256_permutevar8x32_epi32.
  // 3. Pack final 256i block with _mm256_permute2x128_si256.
  constexpr size_t kNumUnpack = 3U;
  __m256i stage[kNumUnpack + 1][kNumStreams];
  static const __m256i kPermuteMask =
      _mm256_set_epi32(0x07, 0x03, 0x06, 0x02, 0x05, 0x01, 0x04, 0x00);
  __m256i permute[kNumStreams];
  __m256i final_result[kNumStreams];

  for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
    for (size_t i = 0; i < kNumStreams; ++i) {
      stage[0][i] = _mm256_loadu_si256(&raw_values_simd[block_index * kNumStreams + i]);
    }

    for (size_t stage_lvl = 0; stage_lvl < kNumUnpack; ++stage_lvl) {
      for (size_t i = 0; i < kNumStreams / 2U; ++i) {
        stage[stage_lvl + 1][i * 2] =
            _mm256_unpacklo_epi8(stage[stage_lvl][i * 2], stage[stage_lvl][i * 2 + 1]);
        stage[stage_lvl + 1][i * 2 + 1] =
            _mm256_unpackhi_epi8(stage[stage_lvl][i * 2], stage[stage_lvl][i * 2 + 1]);
      }
    }

    for (size_t i = 0; i < kNumStreams; ++i) {
      permute[i] = _mm256_permutevar8x32_epi32(stage[kNumUnpack][i], kPermuteMask);
    }

    final_result[0] = _mm256_permute2x128_si256(permute[0], permute[2], 0b00100000);
    final_result[1] = _mm256_permute2x128_si256(permute[0], permute[2], 0b00110001);
    final_result[2] = _mm256_permute2x128_si256(permute[1], permute[3], 0b00100000);
    final_result[3] = _mm256_permute2x128_si256(permute[1], permute[3], 0b00110001);

    for (size_t i = 0; i < kNumStreams; ++i) {
      _mm256_storeu_si256(&output_buffer_streams[i][block_index], final_result[i]);
    }
  }
}
#endif  // ARROW_HAVE_AVX2

#if defined(ARROW_HAVE_AVX512)
template <typename T>
void ByteStreamSplitDecodeAvx512(const uint8_t* data, int64_t num_values, int64_t stride,
                                 T* out) {
  constexpr size_t kNumStreams = sizeof(T);
  static_assert(kNumStreams == 4U || kNumStreams == 8U, "Invalid number of streams.");
  constexpr size_t kNumStreamsLog2 = (kNumStreams == 8U ? 3U : 2U);

  const int64_t size = num_values * sizeof(T);
  constexpr int64_t kBlockSize = sizeof(__m512i) * kNumStreams;
  if (size < kBlockSize)  // Back to AVX2 for small size
    return ByteStreamSplitDecodeAvx2(data, num_values, stride, out);
  const int64_t num_blocks = size / kBlockSize;
  uint8_t* output_data = reinterpret_cast<uint8_t*>(out);

  // First handle suffix.
  const int64_t num_processed_elements = (num_blocks * kBlockSize) / kNumStreams;
  for (int64_t i = num_processed_elements; i < num_values; ++i) {
    uint8_t gathered_byte_data[kNumStreams];
    for (size_t b = 0; b < kNumStreams; ++b) {
      const size_t byte_index = b * stride + i;
      gathered_byte_data[b] = data[byte_index];
    }
    out[i] = arrow::util::SafeLoadAs<T>(&gathered_byte_data[0]);
  }

  // Processed hierarchically using the unpack, then two shuffles.
  __m512i stage[kNumStreamsLog2 + 1U][kNumStreams];
  __m512i shuffle[kNumStreams];
  __m512i final_result[kNumStreams];
  constexpr size_t kNumStreamsHalf = kNumStreams / 2U;

  for (int64_t i = 0; i < num_blocks; ++i) {
    for (size_t j = 0; j < kNumStreams; ++j) {
      stage[0][j] = _mm512_loadu_si512(
          reinterpret_cast<const __m512i*>(&data[i * sizeof(__m512i) + j * stride]));
    }

    for (size_t step = 0; step < kNumStreamsLog2; ++step) {
      for (size_t j = 0; j < kNumStreamsHalf; ++j) {
        stage[step + 1U][j * 2] =
            _mm512_unpacklo_epi8(stage[step][j], stage[step][kNumStreamsHalf + j]);
        stage[step + 1U][j * 2 + 1U] =
            _mm512_unpackhi_epi8(stage[step][j], stage[step][kNumStreamsHalf + j]);
      }
    }

    if (kNumStreams == 8U) {
      // path for double, 128i index:
      // {0x00, 0x04, 0x08, 0x0C}, {0x10, 0x14, 0x18, 0x1C},
      // {0x01, 0x05, 0x09, 0x0D}, {0x11, 0x15, 0x19, 0x1D},
      // {0x02, 0x06, 0x0A, 0x0E}, {0x12, 0x16, 0x1A, 0x1E},
      // {0x03, 0x07, 0x0B, 0x0F}, {0x13, 0x17, 0x1B, 0x1F},
      shuffle[0] = _mm512_shuffle_i32x4(stage[kNumStreamsLog2][0],
                                        stage[kNumStreamsLog2][1], 0b01000100);
      shuffle[1] = _mm512_shuffle_i32x4(stage[kNumStreamsLog2][2],
                                        stage[kNumStreamsLog2][3], 0b01000100);
      shuffle[2] = _mm512_shuffle_i32x4(stage[kNumStreamsLog2][4],
                                        stage[kNumStreamsLog2][5], 0b01000100);
      shuffle[3] = _mm512_shuffle_i32x4(stage[kNumStreamsLog2][6],
                                        stage[kNumStreamsLog2][7], 0b01000100);
      shuffle[4] = _mm512_shuffle_i32x4(stage[kNumStreamsLog2][0],
                                        stage[kNumStreamsLog2][1], 0b11101110);
      shuffle[5] = _mm512_shuffle_i32x4(stage[kNumStreamsLog2][2],
                                        stage[kNumStreamsLog2][3], 0b11101110);
      shuffle[6] = _mm512_shuffle_i32x4(stage[kNumStreamsLog2][4],
                                        stage[kNumStreamsLog2][5], 0b11101110);
      shuffle[7] = _mm512_shuffle_i32x4(stage[kNumStreamsLog2][6],
                                        stage[kNumStreamsLog2][7], 0b11101110);

      final_result[0] = _mm512_shuffle_i32x4(shuffle[0], shuffle[1], 0b10001000);
      final_result[1] = _mm512_shuffle_i32x4(shuffle[2], shuffle[3], 0b10001000);
      final_result[2] = _mm512_shuffle_i32x4(shuffle[0], shuffle[1], 0b11011101);
      final_result[3] = _mm512_shuffle_i32x4(shuffle[2], shuffle[3], 0b11011101);
      final_result[4] = _mm512_shuffle_i32x4(shuffle[4], shuffle[5], 0b10001000);
      final_result[5] = _mm512_shuffle_i32x4(shuffle[6], shuffle[7], 0b10001000);
      final_result[6] = _mm512_shuffle_i32x4(shuffle[4], shuffle[5], 0b11011101);
      final_result[7] = _mm512_shuffle_i32x4(shuffle[6], shuffle[7], 0b11011101);
    } else {
      // path for float, 128i index:
      // {0x00, 0x04, 0x08, 0x0C}, {0x01, 0x05, 0x09, 0x0D}
      // {0x02, 0x06, 0x0A, 0x0E}, {0x03, 0x07, 0x0B, 0x0F},
      shuffle[0] = _mm512_shuffle_i32x4(stage[kNumStreamsLog2][0],
                                        stage[kNumStreamsLog2][1], 0b01000100);
      shuffle[1] = _mm512_shuffle_i32x4(stage[kNumStreamsLog2][2],
                                        stage[kNumStreamsLog2][3], 0b01000100);
      shuffle[2] = _mm512_shuffle_i32x4(stage[kNumStreamsLog2][0],
                                        stage[kNumStreamsLog2][1], 0b11101110);
      shuffle[3] = _mm512_shuffle_i32x4(stage[kNumStreamsLog2][2],
                                        stage[kNumStreamsLog2][3], 0b11101110);

      final_result[0] = _mm512_shuffle_i32x4(shuffle[0], shuffle[1], 0b10001000);
      final_result[1] = _mm512_shuffle_i32x4(shuffle[0], shuffle[1], 0b11011101);
      final_result[2] = _mm512_shuffle_i32x4(shuffle[2], shuffle[3], 0b10001000);
      final_result[3] = _mm512_shuffle_i32x4(shuffle[2], shuffle[3], 0b11011101);
    }

    for (size_t j = 0; j < kNumStreams; ++j) {
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(
                              &output_data[(i * kNumStreams + j) * sizeof(__m512i)]),
                          final_result[j]);
    }
  }
}

template <typename T>
void ByteStreamSplitEncodeAvx512(const uint8_t* raw_values, const size_t num_values,
                                 uint8_t* output_buffer_raw) {
  constexpr size_t kNumStreams = sizeof(T);
  static_assert(kNumStreams == 4U || kNumStreams == 8U, "Invalid number of streams.");
  const size_t size = num_values * sizeof(T);
  constexpr size_t kBlockSize = sizeof(__m512i) * kNumStreams;
  if (size < kBlockSize)  // Back to AVX2 for small size
    return ByteStreamSplitEncodeAvx2<T>(raw_values, num_values, output_buffer_raw);

  const size_t num_blocks = size / kBlockSize;
  const __m512i* raw_values_simd = reinterpret_cast<const __m512i*>(raw_values);
  __m512i* output_buffer_streams[kNumStreams];
  for (size_t i = 0; i < kNumStreams; ++i) {
    output_buffer_streams[i] =
        reinterpret_cast<__m512i*>(&output_buffer_raw[num_values * i]);
  }

  // First handle suffix.
  const size_t num_processed_elements = (num_blocks * kBlockSize) / sizeof(T);
  for (size_t i = num_processed_elements; i < num_values; ++i) {
    for (size_t j = 0U; j < kNumStreams; ++j) {
      const uint8_t byte_in_value = raw_values[i * kNumStreams + j];
      output_buffer_raw[j * num_values + i] = byte_in_value;
    }
  }

  constexpr size_t KNumUnpack = (kNumStreams == 8U) ? 2U : 3U;
  __m512i final_result[kNumStreams];
  __m512i unpack[KNumUnpack + 1][kNumStreams];
  __m512i permutex[kNumStreams];
  __m512i permutex_mask;
  if (kNumStreams == 8U) {
    // use _mm512_set_epi32, no _mm512_set_epi16 for some old gcc version.
    permutex_mask = _mm512_set_epi32(0x001F0017, 0x000F0007, 0x001E0016, 0x000E0006,
                                     0x001D0015, 0x000D0005, 0x001C0014, 0x000C0004,
                                     0x001B0013, 0x000B0003, 0x001A0012, 0x000A0002,
                                     0x00190011, 0x00090001, 0x00180010, 0x00080000);
  } else {
    permutex_mask = _mm512_set_epi32(0x0F, 0x0B, 0x07, 0x03, 0x0E, 0x0A, 0x06, 0x02, 0x0D,
                                     0x09, 0x05, 0x01, 0x0C, 0x08, 0x04, 0x00);
  }

  for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
    for (size_t i = 0; i < kNumStreams; ++i) {
      unpack[0][i] = _mm512_loadu_si512(&raw_values_simd[block_index * kNumStreams + i]);
    }

    for (size_t unpack_lvl = 0; unpack_lvl < KNumUnpack; ++unpack_lvl) {
      for (size_t i = 0; i < kNumStreams / 2U; ++i) {
        unpack[unpack_lvl + 1][i * 2] = _mm512_unpacklo_epi8(
            unpack[unpack_lvl][i * 2], unpack[unpack_lvl][i * 2 + 1]);
        unpack[unpack_lvl + 1][i * 2 + 1] = _mm512_unpackhi_epi8(
            unpack[unpack_lvl][i * 2], unpack[unpack_lvl][i * 2 + 1]);
      }
    }

    if (kNumStreams == 8U) {
      // path for double
      // 1. unpack to epi16 block
      // 2. permutexvar_epi16 to 128i block
      // 3. shuffle 128i to final 512i target, index:
      //   {0x00, 0x04, 0x08, 0x0C}, {0x10, 0x14, 0x18, 0x1C},
      //   {0x01, 0x05, 0x09, 0x0D}, {0x11, 0x15, 0x19, 0x1D},
      //   {0x02, 0x06, 0x0A, 0x0E}, {0x12, 0x16, 0x1A, 0x1E},
      //   {0x03, 0x07, 0x0B, 0x0F}, {0x13, 0x17, 0x1B, 0x1F},
      for (size_t i = 0; i < kNumStreams; ++i)
        permutex[i] = _mm512_permutexvar_epi16(permutex_mask, unpack[KNumUnpack][i]);

      __m512i shuffle[kNumStreams];
      shuffle[0] = _mm512_shuffle_i32x4(permutex[0], permutex[2], 0b01000100);
      shuffle[1] = _mm512_shuffle_i32x4(permutex[4], permutex[6], 0b01000100);
      shuffle[2] = _mm512_shuffle_i32x4(permutex[0], permutex[2], 0b11101110);
      shuffle[3] = _mm512_shuffle_i32x4(permutex[4], permutex[6], 0b11101110);
      shuffle[4] = _mm512_shuffle_i32x4(permutex[1], permutex[3], 0b01000100);
      shuffle[5] = _mm512_shuffle_i32x4(permutex[5], permutex[7], 0b01000100);
      shuffle[6] = _mm512_shuffle_i32x4(permutex[1], permutex[3], 0b11101110);
      shuffle[7] = _mm512_shuffle_i32x4(permutex[5], permutex[7], 0b11101110);

      final_result[0] = _mm512_shuffle_i32x4(shuffle[0], shuffle[1], 0b10001000);
      final_result[1] = _mm512_shuffle_i32x4(shuffle[0], shuffle[1], 0b11011101);
      final_result[2] = _mm512_shuffle_i32x4(shuffle[2], shuffle[3], 0b10001000);
      final_result[3] = _mm512_shuffle_i32x4(shuffle[2], shuffle[3], 0b11011101);
      final_result[4] = _mm512_shuffle_i32x4(shuffle[4], shuffle[5], 0b10001000);
      final_result[5] = _mm512_shuffle_i32x4(shuffle[4], shuffle[5], 0b11011101);
      final_result[6] = _mm512_shuffle_i32x4(shuffle[6], shuffle[7], 0b10001000);
      final_result[7] = _mm512_shuffle_i32x4(shuffle[6], shuffle[7], 0b11011101);
    } else {
      // Path for float.
      // 1. Processed hierarchically to 32i blcok using the unpack intrinsics.
      // 2. Pack 128i block using _mm256_permutevar8x32_epi32.
      // 3. Pack final 256i block with _mm256_permute2x128_si256.
      for (size_t i = 0; i < kNumStreams; ++i)
        permutex[i] = _mm512_permutexvar_epi32(permutex_mask, unpack[KNumUnpack][i]);

      final_result[0] = _mm512_shuffle_i32x4(permutex[0], permutex[2], 0b01000100);
      final_result[1] = _mm512_shuffle_i32x4(permutex[0], permutex[2], 0b11101110);
      final_result[2] = _mm512_shuffle_i32x4(permutex[1], permutex[3], 0b01000100);
      final_result[3] = _mm512_shuffle_i32x4(permutex[1], permutex[3], 0b11101110);
    }

    for (size_t i = 0; i < kNumStreams; ++i) {
      _mm512_storeu_si512(&output_buffer_streams[i][block_index], final_result[i]);
    }
  }
}
#endif  // ARROW_HAVE_AVX512

#if defined(ARROW_HAVE_SIMD_SPLIT)
template <typename T>
void inline ByteStreamSplitDecodeSimd(const uint8_t* data, int64_t num_values,
                                      int64_t stride, T* out) {
#if defined(ARROW_HAVE_AVX512)
  return ByteStreamSplitDecodeAvx512(data, num_values, stride, out);
#elif defined(ARROW_HAVE_AVX2)
  return ByteStreamSplitDecodeAvx2(data, num_values, stride, out);
#elif defined(ARROW_HAVE_SSE4_2)
  return ByteStreamSplitDecodeSse2(data, num_values, stride, out);
#else
#error "ByteStreamSplitDecodeSimd not implemented"
#endif
}

template <typename T>
void inline ByteStreamSplitEncodeSimd(const uint8_t* raw_values, const size_t num_values,
                                      uint8_t* output_buffer_raw) {
#if defined(ARROW_HAVE_AVX512)
  return ByteStreamSplitEncodeAvx512<T>(raw_values, num_values, output_buffer_raw);
#elif defined(ARROW_HAVE_AVX2)
  return ByteStreamSplitEncodeAvx2<T>(raw_values, num_values, output_buffer_raw);
#elif defined(ARROW_HAVE_SSE4_2)
  return ByteStreamSplitEncodeSse2<T>(raw_values, num_values, output_buffer_raw);
#else
#error "ByteStreamSplitEncodeSimd not implemented"
#endif
}
#endif

template <typename T>
void ByteStreamSplitEncodeScalar(const uint8_t* raw_values, const size_t num_values,
                                 uint8_t* output_buffer_raw) {
  constexpr size_t kNumStreams = sizeof(T);
  for (size_t i = 0U; i < num_values; ++i) {
    for (size_t j = 0U; j < kNumStreams; ++j) {
      const uint8_t byte_in_value = raw_values[i * kNumStreams + j];
      output_buffer_raw[j * num_values + i] = byte_in_value;
    }
  }
}

template <typename T>
void ByteStreamSplitDecodeScalar(const uint8_t* data, int64_t num_values, int64_t stride,
                                 T* out) {
  constexpr size_t kNumStreams = sizeof(T);
  auto output_buffer_raw = reinterpret_cast<uint8_t*>(out);

  for (int64_t i = 0; i < num_values; ++i) {
    for (size_t b = 0; b < kNumStreams; ++b) {
      const size_t byte_index = b * stride + i;
      output_buffer_raw[i * kNumStreams + b] = data[byte_index];
    }
  }
}

template <typename T>
void inline ByteStreamSplitEncode(const uint8_t* raw_values, const size_t num_values,
                                  uint8_t* output_buffer_raw) {
#if defined(ARROW_HAVE_SIMD_SPLIT)
  return ByteStreamSplitEncodeSimd<T>(raw_values, num_values, output_buffer_raw);
#else
  return ByteStreamSplitEncodeScalar<T>(raw_values, num_values, output_buffer_raw);
#endif
}

template <typename T>
void inline ByteStreamSplitDecode(const uint8_t* data, int64_t num_values, int64_t stride,
                                  T* out) {
#if defined(ARROW_HAVE_SIMD_SPLIT)
  return ByteStreamSplitDecodeSimd(data, num_values, stride, out);
#else
  return ByteStreamSplitDecodeScalar(data, num_values, stride, out);
#endif
}

}  // namespace internal
}  // namespace util
}  // namespace arrow

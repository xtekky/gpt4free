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

#if defined(ARROW_HAVE_AVX2)
#include <immintrin.h>
#endif

#include <cstdint>

#include "arrow/compute/exec/util.h"
#include "arrow/compute/light_array.h"

namespace arrow {
namespace compute {

// Forward declarations only needed for making test functions a friend of the classes in
// this file.
//
enum class BloomFilterBuildStrategy;

// Implementations are based on xxh3 32-bit algorithm description from:
// https://github.com/Cyan4973/xxHash/blob/dev/doc/xxhash_spec.md
//
class ARROW_EXPORT Hashing32 {
  friend class TestVectorHash;
  template <typename T>
  friend void TestBloomLargeHashHelper(int64_t, int64_t, const std::vector<uint64_t>&,
                                       int64_t, int, T*);
  friend void TestBloomSmall(BloomFilterBuildStrategy, int64_t, int, bool, bool);

 public:
  static void HashMultiColumn(const std::vector<KeyColumnArray>& cols, LightContext* ctx,
                              uint32_t* out_hash);

  static Status HashBatch(const ExecBatch& key_batch, uint32_t* hashes,
                          std::vector<KeyColumnArray>& column_arrays,
                          int64_t hardware_flags, util::TempVectorStack* temp_stack,
                          int64_t offset, int64_t length);

 private:
  static const uint32_t PRIME32_1 = 0x9E3779B1;
  static const uint32_t PRIME32_2 = 0x85EBCA77;
  static const uint32_t PRIME32_3 = 0xC2B2AE3D;
  static const uint32_t PRIME32_4 = 0x27D4EB2F;
  static const uint32_t PRIME32_5 = 0x165667B1;
  static const uint32_t kCombineConst = 0x9e3779b9UL;
  static const int64_t kStripeSize = 4 * sizeof(uint32_t);

  static void HashFixed(int64_t hardware_flags, bool combine_hashes, uint32_t num_keys,
                        uint64_t length_key, const uint8_t* keys, uint32_t* hashes,
                        uint32_t* temp_hashes_for_combine);

  static void HashVarLen(int64_t hardware_flags, bool combine_hashes, uint32_t num_rows,
                         const uint32_t* offsets, const uint8_t* concatenated_keys,
                         uint32_t* hashes, uint32_t* temp_hashes_for_combine);

  static void HashVarLen(int64_t hardware_flags, bool combine_hashes, uint32_t num_rows,
                         const uint64_t* offsets, const uint8_t* concatenated_keys,
                         uint32_t* hashes, uint32_t* temp_hashes_for_combine);

  static inline uint32_t Avalanche(uint32_t acc) {
    acc ^= (acc >> 15);
    acc *= PRIME32_2;
    acc ^= (acc >> 13);
    acc *= PRIME32_3;
    acc ^= (acc >> 16);
    return acc;
  }
  static inline uint32_t Round(uint32_t acc, uint32_t input);
  static inline uint32_t CombineAccumulators(uint32_t acc1, uint32_t acc2, uint32_t acc3,
                                             uint32_t acc4);
  static inline uint32_t CombineHashesImp(uint32_t previous_hash, uint32_t hash) {
    uint32_t next_hash = previous_hash ^ (hash + kCombineConst + (previous_hash << 6) +
                                          (previous_hash >> 2));
    return next_hash;
  }
  static inline void ProcessFullStripes(uint64_t num_stripes, const uint8_t* key,
                                        uint32_t* out_acc1, uint32_t* out_acc2,
                                        uint32_t* out_acc3, uint32_t* out_acc4);
  static inline void ProcessLastStripe(uint32_t mask1, uint32_t mask2, uint32_t mask3,
                                       uint32_t mask4, const uint8_t* last_stripe,
                                       uint32_t* acc1, uint32_t* acc2, uint32_t* acc3,
                                       uint32_t* acc4);
  static inline void StripeMask(int i, uint32_t* mask1, uint32_t* mask2, uint32_t* mask3,
                                uint32_t* mask4);
  template <bool T_COMBINE_HASHES>
  static void HashFixedLenImp(uint32_t num_rows, uint64_t length, const uint8_t* keys,
                              uint32_t* hashes);
  template <typename T, bool T_COMBINE_HASHES>
  static void HashVarLenImp(uint32_t num_rows, const T* offsets,
                            const uint8_t* concatenated_keys, uint32_t* hashes);
  template <bool T_COMBINE_HASHES>
  static void HashBitImp(int64_t bit_offset, uint32_t num_keys, const uint8_t* keys,
                         uint32_t* hashes);
  static void HashBit(bool combine_hashes, int64_t bit_offset, uint32_t num_keys,
                      const uint8_t* keys, uint32_t* hashes);
  template <bool T_COMBINE_HASHES, typename T>
  static void HashIntImp(uint32_t num_keys, const T* keys, uint32_t* hashes);
  static void HashInt(bool combine_hashes, uint32_t num_keys, uint64_t length_key,
                      const uint8_t* keys, uint32_t* hashes);

#if defined(ARROW_HAVE_AVX2)
  static inline __m256i Avalanche_avx2(__m256i hash);
  static inline __m256i CombineHashesImp_avx2(__m256i previous_hash, __m256i hash);
  template <bool T_COMBINE_HASHES>
  static void AvalancheAll_avx2(uint32_t num_rows, uint32_t* hashes,
                                const uint32_t* hashes_temp_for_combine);
  static inline __m256i Round_avx2(__m256i acc, __m256i input);
  static inline uint64_t CombineAccumulators_avx2(__m256i acc);
  static inline __m256i StripeMask_avx2(int i, int j);
  template <bool two_equal_lengths>
  static inline __m256i ProcessStripes_avx2(int64_t num_stripes_A, int64_t num_stripes_B,
                                            __m256i mask_last_stripe, const uint8_t* keys,
                                            int64_t offset_A, int64_t offset_B);
  template <bool T_COMBINE_HASHES>
  static uint32_t HashFixedLenImp_avx2(uint32_t num_rows, uint64_t length,
                                       const uint8_t* keys, uint32_t* hashes,
                                       uint32_t* hashes_temp_for_combine);
  static uint32_t HashFixedLen_avx2(bool combine_hashes, uint32_t num_rows,
                                    uint64_t length, const uint8_t* keys,
                                    uint32_t* hashes, uint32_t* hashes_temp_for_combine);
  template <typename T, bool T_COMBINE_HASHES>
  static uint32_t HashVarLenImp_avx2(uint32_t num_rows, const T* offsets,
                                     const uint8_t* concatenated_keys, uint32_t* hashes,
                                     uint32_t* hashes_temp_for_combine);
  static uint32_t HashVarLen_avx2(bool combine_hashes, uint32_t num_rows,
                                  const uint32_t* offsets,
                                  const uint8_t* concatenated_keys, uint32_t* hashes,
                                  uint32_t* hashes_temp_for_combine);
  static uint32_t HashVarLen_avx2(bool combine_hashes, uint32_t num_rows,
                                  const uint64_t* offsets,
                                  const uint8_t* concatenated_keys, uint32_t* hashes,
                                  uint32_t* hashes_temp_for_combine);
#endif
};

class ARROW_EXPORT Hashing64 {
  friend class TestVectorHash;
  template <typename T>
  friend void TestBloomLargeHashHelper(int64_t, int64_t, const std::vector<uint64_t>&,
                                       int64_t, int, T*);
  friend void TestBloomSmall(BloomFilterBuildStrategy, int64_t, int, bool, bool);

 public:
  static void HashMultiColumn(const std::vector<KeyColumnArray>& cols, LightContext* ctx,
                              uint64_t* hashes);

  static Status HashBatch(const ExecBatch& key_batch, uint64_t* hashes,
                          std::vector<KeyColumnArray>& column_arrays,
                          int64_t hardware_flags, util::TempVectorStack* temp_stack,
                          int64_t offset, int64_t length);

 private:
  static const uint64_t PRIME64_1 = 0x9E3779B185EBCA87ULL;
  static const uint64_t PRIME64_2 = 0xC2B2AE3D27D4EB4FULL;
  static const uint64_t PRIME64_3 = 0x165667B19E3779F9ULL;
  static const uint64_t PRIME64_4 = 0x85EBCA77C2B2AE63ULL;
  static const uint64_t PRIME64_5 = 0x27D4EB2F165667C5ULL;
  static const uint32_t kCombineConst = 0x9e3779b9UL;
  static const int64_t kStripeSize = 4 * sizeof(uint64_t);

  static void HashFixed(bool combine_hashes, uint32_t num_keys, uint64_t length_key,
                        const uint8_t* keys, uint64_t* hashes);

  static void HashVarLen(bool combine_hashes, uint32_t num_rows, const uint32_t* offsets,
                         const uint8_t* concatenated_keys, uint64_t* hashes);

  static void HashVarLen(bool combine_hashes, uint32_t num_rows, const uint64_t* offsets,
                         const uint8_t* concatenated_keys, uint64_t* hashes);

  static inline uint64_t Avalanche(uint64_t acc);
  static inline uint64_t Round(uint64_t acc, uint64_t input);
  static inline uint64_t CombineAccumulators(uint64_t acc1, uint64_t acc2, uint64_t acc3,
                                             uint64_t acc4);
  static inline uint64_t CombineHashesImp(uint64_t previous_hash, uint64_t hash) {
    uint64_t next_hash = previous_hash ^ (hash + kCombineConst + (previous_hash << 6) +
                                          (previous_hash >> 2));
    return next_hash;
  }
  static inline void ProcessFullStripes(uint64_t num_stripes, const uint8_t* key,
                                        uint64_t* out_acc1, uint64_t* out_acc2,
                                        uint64_t* out_acc3, uint64_t* out_acc4);
  static inline void ProcessLastStripe(uint64_t mask1, uint64_t mask2, uint64_t mask3,
                                       uint64_t mask4, const uint8_t* last_stripe,
                                       uint64_t* acc1, uint64_t* acc2, uint64_t* acc3,
                                       uint64_t* acc4);
  static inline void StripeMask(int i, uint64_t* mask1, uint64_t* mask2, uint64_t* mask3,
                                uint64_t* mask4);
  template <bool T_COMBINE_HASHES>
  static void HashFixedLenImp(uint32_t num_rows, uint64_t length, const uint8_t* keys,
                              uint64_t* hashes);
  template <typename T, bool T_COMBINE_HASHES>
  static void HashVarLenImp(uint32_t num_rows, const T* offsets,
                            const uint8_t* concatenated_keys, uint64_t* hashes);
  template <bool T_COMBINE_HASHES>
  static void HashBitImp(int64_t bit_offset, uint32_t num_keys, const uint8_t* keys,
                         uint64_t* hashes);
  static void HashBit(bool T_COMBINE_HASHES, int64_t bit_offset, uint32_t num_keys,
                      const uint8_t* keys, uint64_t* hashes);
  template <bool T_COMBINE_HASHES, typename T>
  static void HashIntImp(uint32_t num_keys, const T* keys, uint64_t* hashes);
  static void HashInt(bool T_COMBINE_HASHES, uint32_t num_keys, uint64_t length_key,
                      const uint8_t* keys, uint64_t* hashes);
};

}  // namespace compute
}  // namespace arrow

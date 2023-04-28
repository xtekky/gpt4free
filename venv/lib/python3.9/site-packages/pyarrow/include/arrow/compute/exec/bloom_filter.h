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

#include <atomic>
#include <cstdint>
#include <memory>
#include "arrow/compute/exec/partition_util.h"
#include "arrow/compute/exec/util.h"
#include "arrow/memory_pool.h"
#include "arrow/result.h"
#include "arrow/status.h"

namespace arrow {
namespace compute {

// A set of pre-generated bit masks from a 64-bit word.
//
// It is used to map selected bits of hash to a bit mask that will be used in
// a Bloom filter.
//
// These bit masks need to look random and need to have a similar fractions of
// bits set in order for a Bloom filter to have a low false positives rate.
//
struct ARROW_EXPORT BloomFilterMasks {
  // Generate all masks as a single bit vector. Each bit offset in this bit
  // vector corresponds to a single mask.
  // In each consecutive kBitsPerMask bits, there must be between
  // kMinBitsSet and kMaxBitsSet bits set.
  //
  BloomFilterMasks();

  inline uint64_t mask(int bit_offset) {
#if ARROW_LITTLE_ENDIAN
    return (util::SafeLoadAs<uint64_t>(masks_ + bit_offset / 8) >> (bit_offset % 8)) &
           kFullMask;
#else
    return (BYTESWAP(util::SafeLoadAs<uint64_t>(masks_ + bit_offset / 8)) >>
            (bit_offset % 8)) &
           kFullMask;
#endif
  }

  // Masks are 57 bits long because then they can be accessed at an
  // arbitrary bit offset using a single unaligned 64-bit load instruction.
  //
  static constexpr int kBitsPerMask = 57;
  static constexpr uint64_t kFullMask = (1ULL << kBitsPerMask) - 1;

  // Minimum and maximum number of bits set in each mask.
  // This constraint is enforced when generating the bit masks.
  // Values should be close to each other and chosen as to minimize a Bloom
  // filter false positives rate.
  //
  static constexpr int kMinBitsSet = 4;
  static constexpr int kMaxBitsSet = 5;

  // Number of generated masks.
  // Having more masks to choose will improve false positives rate of Bloom
  // filter but will also use more memory, which may lead to more CPU cache
  // misses.
  // The chosen value results in using only a few cache-lines for mask lookups,
  // while providing a good variety of available bit masks.
  //
  static constexpr int kLogNumMasks = 10;
  static constexpr int kNumMasks = 1 << kLogNumMasks;

  // Data of masks. Masks are stored in a single bit vector. Nth mask is
  // kBitsPerMask bits starting at bit offset N.
  //
  static constexpr int kTotalBytes = (kNumMasks + 64) / 8;
  uint8_t masks_[kTotalBytes];
};

// A variant of a blocked Bloom filter implementation.
// A Bloom filter is a data structure that provides approximate membership test
// functionality based only on the hash of the key. Membership test may return
// false positives but not false negatives. Approximation of the result allows
// in general case (for arbitrary data types of keys) to save on both memory and
// lookup cost compared to the accurate membership test.
// The accurate test may sometimes still be cheaper for a specific data types
// and inputs, e.g. integers from a small range.
//
// This blocked Bloom filter is optimized for use in hash joins, to achieve a
// good balance between the size of the filter, the cost of its building and
// querying and the rate of false positives.
//
class ARROW_EXPORT BlockedBloomFilter {
  friend class BloomFilterBuilder_SingleThreaded;
  friend class BloomFilterBuilder_Parallel;

 public:
  BlockedBloomFilter() : log_num_blocks_(0), num_blocks_(0), blocks_(NULLPTR) {}

  inline bool Find(uint64_t hash) const {
    uint64_t m = mask(hash);
    uint64_t b = blocks_[block_id(hash)];
    return (b & m) == m;
  }

  // Uses SIMD if available for smaller Bloom filters.
  // Uses memory prefetching for larger Bloom filters.
  //
  void Find(int64_t hardware_flags, int64_t num_rows, const uint32_t* hashes,
            uint8_t* result_bit_vector, bool enable_prefetch = true) const;
  void Find(int64_t hardware_flags, int64_t num_rows, const uint64_t* hashes,
            uint8_t* result_bit_vector, bool enable_prefetch = true) const;

  int log_num_blocks() const { return log_num_blocks_; }

  int NumHashBitsUsed() const;

  bool IsSameAs(const BlockedBloomFilter* other) const;

  int64_t NumBitsSet() const;

  // Folding of a block Bloom filter after the initial version
  // has been built.
  //
  // One of the parameters for creation of Bloom filter is the number
  // of bits allocated for it. The more bits allocated, the lower the
  // probability of false positives. A good heuristic is to aim for
  // half of the bits set in the constructed Bloom filter. This should
  // result in a good trade off between size (and following cost of
  // memory accesses) and false positives rate.
  //
  // There might have been many duplicate keys in the input provided
  // to Bloom filter builder. In that case the resulting bit vector
  // would be more sparse then originally intended. It is possible to
  // easily correct that and cut in half the size of Bloom filter
  // after it has already been constructed. The process to do that is
  // approximately equal to OR-ing bits from upper and lower half (the
  // way we address these bits when inserting or querying a hash makes
  // such folding in half possible).
  //
  // We will keep folding as long as the fraction of bits set is less
  // than 1/4. The resulting bit vector density should be in the [1/4,
  // 1/2) range.
  //
  void Fold();

 private:
  Status CreateEmpty(int64_t num_rows_to_insert, MemoryPool* pool);

  inline void Insert(uint64_t hash) {
    uint64_t m = mask(hash);
    uint64_t& b = blocks_[block_id(hash)];
    b |= m;
  }

  void Insert(int64_t hardware_flags, int64_t num_rows, const uint32_t* hashes);
  void Insert(int64_t hardware_flags, int64_t num_rows, const uint64_t* hashes);

  inline uint64_t mask(uint64_t hash) const {
    // The lowest bits of hash are used to pick mask index.
    //
    int mask_id = static_cast<int>(hash & (BloomFilterMasks::kNumMasks - 1));
    uint64_t result = masks_.mask(mask_id);

    // The next set of hash bits is used to pick the amount of bit
    // rotation of the mask.
    //
    int rotation = (hash >> BloomFilterMasks::kLogNumMasks) & 63;
    result = ROTL64(result, rotation);

    return result;
  }

  inline int64_t block_id(uint64_t hash) const {
    // The next set of hash bits following the bits used to select a
    // mask is used to pick block id (index of 64-bit word in a bit
    // vector).
    //
    return (hash >> (BloomFilterMasks::kLogNumMasks + 6)) & (num_blocks_ - 1);
  }

  template <typename T>
  inline void InsertImp(int64_t num_rows, const T* hashes);

  template <typename T>
  inline void FindImp(int64_t num_rows, const T* hashes, uint8_t* result_bit_vector,
                      bool enable_prefetch) const;

  void SingleFold(int num_folds);

#if defined(ARROW_HAVE_AVX2)
  inline __m256i mask_avx2(__m256i hash) const;
  inline __m256i block_id_avx2(__m256i hash) const;
  int64_t Insert_avx2(int64_t num_rows, const uint32_t* hashes);
  int64_t Insert_avx2(int64_t num_rows, const uint64_t* hashes);
  template <typename T>
  int64_t InsertImp_avx2(int64_t num_rows, const T* hashes);
  int64_t Find_avx2(int64_t num_rows, const uint32_t* hashes,
                    uint8_t* result_bit_vector) const;
  int64_t Find_avx2(int64_t num_rows, const uint64_t* hashes,
                    uint8_t* result_bit_vector) const;
  template <typename T>
  int64_t FindImp_avx2(int64_t num_rows, const T* hashes,
                       uint8_t* result_bit_vector) const;
#endif

  bool UsePrefetch() const {
    return num_blocks_ * sizeof(uint64_t) > kPrefetchLimitBytes;
  }

  static constexpr int64_t kPrefetchLimitBytes = 256 * 1024;

  static BloomFilterMasks masks_;

  // Total number of bits used by block Bloom filter must be a power
  // of 2.
  //
  int log_num_blocks_;
  int64_t num_blocks_;

  // Buffer allocated to store an array of power of 2 64-bit blocks.
  //
  std::shared_ptr<Buffer> buf_;
  // Pointer to mutable data owned by Buffer
  //
  uint64_t* blocks_;
};

// We have two separate implementations of building a Bloom filter, multi-threaded and
// single-threaded.
//
// Single threaded version is useful in two ways:
// a) It allows to verify parallel implementation in tests (the single threaded one is
// simpler and can be used as the source of truth).
// b) It is preferred for small and medium size Bloom filters, because it skips extra
// synchronization related steps from parallel variant (partitioning and taking locks).
//
enum class BloomFilterBuildStrategy {
  SINGLE_THREADED = 0,
  PARALLEL = 1,
};

class ARROW_EXPORT BloomFilterBuilder {
 public:
  virtual ~BloomFilterBuilder() = default;
  virtual Status Begin(size_t num_threads, int64_t hardware_flags, MemoryPool* pool,
                       int64_t num_rows, int64_t num_batches,
                       BlockedBloomFilter* build_target) = 0;
  virtual int64_t num_tasks() const { return 0; }
  virtual Status PushNextBatch(size_t thread_index, int64_t num_rows,
                               const uint32_t* hashes) = 0;
  virtual Status PushNextBatch(size_t thread_index, int64_t num_rows,
                               const uint64_t* hashes) = 0;
  virtual void CleanUp() {}
  static std::unique_ptr<BloomFilterBuilder> Make(BloomFilterBuildStrategy strategy);
};

class ARROW_EXPORT BloomFilterBuilder_SingleThreaded : public BloomFilterBuilder {
 public:
  Status Begin(size_t num_threads, int64_t hardware_flags, MemoryPool* pool,
               int64_t num_rows, int64_t num_batches,
               BlockedBloomFilter* build_target) override;

  Status PushNextBatch(size_t /*thread_index*/, int64_t num_rows,
                       const uint32_t* hashes) override;

  Status PushNextBatch(size_t /*thread_index*/, int64_t num_rows,
                       const uint64_t* hashes) override;

 private:
  template <typename T>
  void PushNextBatchImp(int64_t num_rows, const T* hashes);

  int64_t hardware_flags_;
  BlockedBloomFilter* build_target_;
};

class ARROW_EXPORT BloomFilterBuilder_Parallel : public BloomFilterBuilder {
 public:
  Status Begin(size_t num_threads, int64_t hardware_flags, MemoryPool* pool,
               int64_t num_rows, int64_t num_batches,
               BlockedBloomFilter* build_target) override;

  Status PushNextBatch(size_t thread_id, int64_t num_rows,
                       const uint32_t* hashes) override;

  Status PushNextBatch(size_t thread_id, int64_t num_rows,
                       const uint64_t* hashes) override;

  void CleanUp() override;

 private:
  template <typename T>
  void PushNextBatchImp(size_t thread_id, int64_t num_rows, const T* hashes);

  int64_t hardware_flags_;
  BlockedBloomFilter* build_target_;
  int log_num_prtns_;
  struct ThreadLocalState {
    std::vector<uint32_t> partitioned_hashes_32;
    std::vector<uint64_t> partitioned_hashes_64;
    std::vector<uint16_t> partition_ranges;
    std::vector<int> unprocessed_partition_ids;
  };
  std::vector<ThreadLocalState> thread_local_states_;
  PartitionLocks prtn_locks_;
};

}  // namespace compute
}  // namespace arrow

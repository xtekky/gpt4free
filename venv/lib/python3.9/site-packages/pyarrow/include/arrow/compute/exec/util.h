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

#include <atomic>
#include <cstdint>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>

#include "arrow/buffer.h"
#include "arrow/compute/exec/expression.h"
#include "arrow/compute/exec/options.h"
#include "arrow/compute/type_fwd.h"
#include "arrow/memory_pool.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/cpu_info.h"
#include "arrow/util/logging.h"
#include "arrow/util/mutex.h"
#include "arrow/util/thread_pool.h"

#if defined(__clang__) || defined(__GNUC__)
#define BYTESWAP(x) __builtin_bswap64(x)
#define ROTL(x, n) (((x) << (n)) | ((x) >> ((-n) & 31)))
#define ROTL64(x, n) (((x) << (n)) | ((x) >> ((-n) & 63)))
#define PREFETCH(ptr) __builtin_prefetch((ptr), 0 /* rw==read */, 3 /* locality */)
#elif defined(_MSC_VER)
#include <intrin.h>
#define BYTESWAP(x) _byteswap_uint64(x)
#define ROTL(x, n) _rotl((x), (n))
#define ROTL64(x, n) _rotl64((x), (n))
#if defined(_M_X64) || defined(_M_I86)
#include <mmintrin.h>  // https://msdn.microsoft.com/fr-fr/library/84szxsww(v=vs.90).aspx
#define PREFETCH(ptr) _mm_prefetch((const char*)(ptr), _MM_HINT_T0)
#else
#define PREFETCH(ptr) (void)(ptr) /* disabled */
#endif
#endif

namespace arrow {
namespace util {

template <typename T>
inline void CheckAlignment(const void* ptr) {
  ARROW_DCHECK(reinterpret_cast<uint64_t>(ptr) % sizeof(T) == 0);
}

// Some platforms typedef int64_t as long int instead of long long int,
// which breaks the _mm256_i64gather_epi64 and _mm256_i32gather_epi64 intrinsics
// which need long long.
// We use the cast to the type below in these intrinsics to make the code
// compile in all cases.
//
using int64_for_gather_t = const long long int;  // NOLINT runtime-int

// All MiniBatch... classes use TempVectorStack for vector allocations and can
// only work with vectors up to 1024 elements.
//
// They should only be allocated on the stack to guarantee the right sequence
// of allocation and deallocation of vectors from TempVectorStack.
//
class MiniBatch {
 public:
  static constexpr int kLogMiniBatchLength = 10;
  static constexpr int kMiniBatchLength = 1 << kLogMiniBatchLength;
};

/// Storage used to allocate temporary vectors of a batch size.
/// Temporary vectors should resemble allocating temporary variables on the stack
/// but in the context of vectorized processing where we need to store a vector of
/// temporaries instead of a single value.
class TempVectorStack {
  template <typename>
  friend class TempVectorHolder;

 public:
  Status Init(MemoryPool* pool, int64_t size) {
    num_vectors_ = 0;
    top_ = 0;
    buffer_size_ = PaddedAllocationSize(size) + kPadding + 2 * sizeof(uint64_t);
    ARROW_ASSIGN_OR_RAISE(auto buffer, AllocateResizableBuffer(size, pool));
    // Ensure later operations don't accidentally read uninitialized memory.
    std::memset(buffer->mutable_data(), 0xFF, size);
    buffer_ = std::move(buffer);
    return Status::OK();
  }

 private:
  int64_t PaddedAllocationSize(int64_t num_bytes) {
    // Round up allocation size to multiple of 8 bytes
    // to avoid returning temp vectors with unaligned address.
    //
    // Also add padding at the end to facilitate loads and stores
    // using SIMD when number of vector elements is not divisible
    // by the number of SIMD lanes.
    //
    return ::arrow::bit_util::RoundUp(num_bytes, sizeof(int64_t)) + kPadding;
  }
  void alloc(uint32_t num_bytes, uint8_t** data, int* id) {
    int64_t old_top = top_;
    top_ += PaddedAllocationSize(num_bytes) + 2 * sizeof(uint64_t);
    // Stack overflow check
    ARROW_DCHECK(top_ <= buffer_size_);
    *data = buffer_->mutable_data() + old_top + sizeof(uint64_t);
    // We set 8 bytes before the beginning of the allocated range and
    // 8 bytes after the end to check for stack overflow (which would
    // result in those known bytes being corrupted).
    reinterpret_cast<uint64_t*>(buffer_->mutable_data() + old_top)[0] = kGuard1;
    reinterpret_cast<uint64_t*>(buffer_->mutable_data() + top_)[-1] = kGuard2;
    *id = num_vectors_++;
  }
  void release(int id, uint32_t num_bytes) {
    ARROW_DCHECK(num_vectors_ == id + 1);
    int64_t size = PaddedAllocationSize(num_bytes) + 2 * sizeof(uint64_t);
    ARROW_DCHECK(reinterpret_cast<const uint64_t*>(buffer_->mutable_data() + top_)[-1] ==
                 kGuard2);
    ARROW_DCHECK(top_ >= size);
    top_ -= size;
    ARROW_DCHECK(reinterpret_cast<const uint64_t*>(buffer_->mutable_data() + top_)[0] ==
                 kGuard1);
    --num_vectors_;
  }
  static constexpr uint64_t kGuard1 = 0x3141592653589793ULL;
  static constexpr uint64_t kGuard2 = 0x0577215664901532ULL;
  static constexpr int64_t kPadding = 64;
  int num_vectors_;
  int64_t top_;
  std::unique_ptr<Buffer> buffer_;
  int64_t buffer_size_;
};

template <typename T>
class TempVectorHolder {
  friend class TempVectorStack;

 public:
  ~TempVectorHolder() { stack_->release(id_, num_elements_ * sizeof(T)); }
  T* mutable_data() { return reinterpret_cast<T*>(data_); }
  TempVectorHolder(TempVectorStack* stack, uint32_t num_elements) {
    stack_ = stack;
    num_elements_ = num_elements;
    stack_->alloc(num_elements * sizeof(T), &data_, &id_);
  }

 private:
  TempVectorStack* stack_;
  uint8_t* data_;
  int id_;
  uint32_t num_elements_;
};

class bit_util {
 public:
  static void bits_to_indexes(int bit_to_search, int64_t hardware_flags,
                              const int num_bits, const uint8_t* bits, int* num_indexes,
                              uint16_t* indexes, int bit_offset = 0);

  static void bits_filter_indexes(int bit_to_search, int64_t hardware_flags,
                                  const int num_bits, const uint8_t* bits,
                                  const uint16_t* input_indexes, int* num_indexes,
                                  uint16_t* indexes, int bit_offset = 0);

  // Input and output indexes may be pointing to the same data (in-place filtering).
  static void bits_split_indexes(int64_t hardware_flags, const int num_bits,
                                 const uint8_t* bits, int* num_indexes_bit0,
                                 uint16_t* indexes_bit0, uint16_t* indexes_bit1,
                                 int bit_offset = 0);

  // Bit 1 is replaced with byte 0xFF.
  static void bits_to_bytes(int64_t hardware_flags, const int num_bits,
                            const uint8_t* bits, uint8_t* bytes, int bit_offset = 0);

  // Return highest bit of each byte.
  static void bytes_to_bits(int64_t hardware_flags, const int num_bits,
                            const uint8_t* bytes, uint8_t* bits, int bit_offset = 0);

  static bool are_all_bytes_zero(int64_t hardware_flags, const uint8_t* bytes,
                                 uint32_t num_bytes);

 private:
  inline static uint64_t SafeLoadUpTo8Bytes(const uint8_t* bytes, int num_bytes);
  inline static void SafeStoreUpTo8Bytes(uint8_t* bytes, int num_bytes, uint64_t value);
  inline static void bits_to_indexes_helper(uint64_t word, uint16_t base_index,
                                            int* num_indexes, uint16_t* indexes);
  inline static void bits_filter_indexes_helper(uint64_t word,
                                                const uint16_t* input_indexes,
                                                int* num_indexes, uint16_t* indexes);
  template <int bit_to_search, bool filter_input_indexes>
  static void bits_to_indexes_internal(int64_t hardware_flags, const int num_bits,
                                       const uint8_t* bits, const uint16_t* input_indexes,
                                       int* num_indexes, uint16_t* indexes,
                                       uint16_t base_index = 0);

#if defined(ARROW_HAVE_AVX2)
  static void bits_to_indexes_avx2(int bit_to_search, const int num_bits,
                                   const uint8_t* bits, int* num_indexes,
                                   uint16_t* indexes, uint16_t base_index = 0);
  static void bits_filter_indexes_avx2(int bit_to_search, const int num_bits,
                                       const uint8_t* bits, const uint16_t* input_indexes,
                                       int* num_indexes, uint16_t* indexes);
  template <int bit_to_search>
  static void bits_to_indexes_imp_avx2(const int num_bits, const uint8_t* bits,
                                       int* num_indexes, uint16_t* indexes,
                                       uint16_t base_index = 0);
  template <int bit_to_search>
  static void bits_filter_indexes_imp_avx2(const int num_bits, const uint8_t* bits,
                                           const uint16_t* input_indexes,
                                           int* num_indexes, uint16_t* indexes);
  static void bits_to_bytes_avx2(const int num_bits, const uint8_t* bits, uint8_t* bytes);
  static void bytes_to_bits_avx2(const int num_bits, const uint8_t* bytes, uint8_t* bits);
  static bool are_all_bytes_zero_avx2(const uint8_t* bytes, uint32_t num_bytes);
#endif
};

}  // namespace util
namespace compute {

ARROW_EXPORT
Status ValidateExecNodeInputs(ExecPlan* plan, const std::vector<ExecNode*>& inputs,
                              int expected_num_inputs, const char* kind_name);

ARROW_EXPORT
Result<std::shared_ptr<Table>> TableFromExecBatches(
    const std::shared_ptr<Schema>& schema, const std::vector<ExecBatch>& exec_batches);

class ARROW_EXPORT AtomicCounter {
 public:
  AtomicCounter() = default;

  int count() const { return count_.load(); }

  std::optional<int> total() const {
    int total = total_.load();
    if (total == -1) return {};
    return total;
  }

  // return true if the counter is complete
  bool Increment() {
    DCHECK_NE(count_.load(), total_.load());
    int count = count_.fetch_add(1) + 1;
    if (count != total_.load()) return false;
    return DoneOnce();
  }

  // return true if the counter is complete
  bool SetTotal(int total) {
    total_.store(total);
    if (count_.load() != total) return false;
    return DoneOnce();
  }

  // return true if the counter has not already been completed
  bool Cancel() { return DoneOnce(); }

  // return true if the counter has finished or been cancelled
  bool Completed() { return complete_.load(); }

 private:
  // ensure there is only one true return from Increment(), SetTotal(), or Cancel()
  bool DoneOnce() {
    bool expected = false;
    return complete_.compare_exchange_strong(expected, true);
  }

  std::atomic<int> count_{0}, total_{-1};
  std::atomic<bool> complete_{false};
};

class ARROW_EXPORT ThreadIndexer {
 public:
  size_t operator()();

  static size_t Capacity();

 private:
  static size_t Check(size_t thread_index);

  util::Mutex mutex_;
  std::unordered_map<std::thread::id, size_t> id_to_index_;
};

// Helper class to calculate the modified number of rows to process using SIMD.
//
// Some array elements at the end will be skipped in order to avoid buffer
// overrun, when doing memory loads and stores using larger word size than a
// single array element.
//
class TailSkipForSIMD {
 public:
  static int64_t FixBitAccess(int num_bytes_accessed_together, int64_t num_rows,
                              int bit_offset) {
    int64_t num_bytes = bit_util::BytesForBits(num_rows + bit_offset);
    int64_t num_bytes_safe =
        std::max(static_cast<int64_t>(0LL), num_bytes - num_bytes_accessed_together + 1);
    int64_t num_rows_safe =
        std::max(static_cast<int64_t>(0LL), 8 * num_bytes_safe - bit_offset);
    return std::min(num_rows_safe, num_rows);
  }
  static int64_t FixBinaryAccess(int num_bytes_accessed_together, int64_t num_rows,
                                 int64_t length) {
    int64_t num_rows_to_skip = bit_util::CeilDiv(length, num_bytes_accessed_together);
    int64_t num_rows_safe =
        std::max(static_cast<int64_t>(0LL), num_rows - num_rows_to_skip);
    return num_rows_safe;
  }
  static int64_t FixVarBinaryAccess(int num_bytes_accessed_together, int64_t num_rows,
                                    const uint32_t* offsets) {
    // Do not process rows that could read past the end of the buffer using N
    // byte loads/stores.
    //
    int64_t num_rows_safe = num_rows;
    while (num_rows_safe > 0 &&
           offsets[num_rows_safe] + num_bytes_accessed_together > offsets[num_rows]) {
      --num_rows_safe;
    }
    return num_rows_safe;
  }
  static int FixSelection(int64_t num_rows_safe, int num_selected,
                          const uint16_t* selection) {
    int num_selected_safe = num_selected;
    while (num_selected_safe > 0 && selection[num_selected_safe - 1] >= num_rows_safe) {
      --num_selected_safe;
    }
    return num_selected_safe;
  }
};

/// \brief A consumer that collects results into an in-memory table
struct ARROW_EXPORT TableSinkNodeConsumer : public SinkNodeConsumer {
 public:
  TableSinkNodeConsumer(std::shared_ptr<Table>* out, MemoryPool* pool)
      : out_(out), pool_(pool) {}
  Status Init(const std::shared_ptr<Schema>& schema,
              BackpressureControl* backpressure_control, ExecPlan* plan) override;
  Status Consume(ExecBatch batch) override;
  Future<> Finish() override;

 private:
  std::shared_ptr<Table>* out_;
  MemoryPool* pool_;
  std::shared_ptr<Schema> schema_;
  std::vector<std::shared_ptr<RecordBatch>> batches_;
  util::Mutex consume_mutex_;
};

class ARROW_EXPORT NullSinkNodeConsumer : public SinkNodeConsumer {
 public:
  Status Init(const std::shared_ptr<Schema>&, BackpressureControl*,
              ExecPlan* plan) override {
    return Status::OK();
  }
  Status Consume(ExecBatch exec_batch) override { return Status::OK(); }
  Future<> Finish() override { return Status::OK(); }

 public:
  static std::shared_ptr<NullSinkNodeConsumer> Make() {
    return std::make_shared<NullSinkNodeConsumer>();
  }
};

/// Modify an Expression with pre-order and post-order visitation.
/// `pre` will be invoked on each Expression. `pre` will visit Calls before their
/// arguments, `post_call` will visit Calls (and no other Expressions) after their
/// arguments. Visitors should return the Identical expression to indicate no change; this
/// will prevent unnecessary construction in the common case where a modification is not
/// possible/necessary/...
///
/// If an argument was modified, `post_call` visits a reconstructed Call with the modified
/// arguments but also receives a pointer to the unmodified Expression as a second
/// argument. If no arguments were modified the unmodified Expression* will be nullptr.
template <typename PreVisit, typename PostVisitCall>
Result<Expression> ModifyExpression(Expression expr, const PreVisit& pre,
                                    const PostVisitCall& post_call) {
  ARROW_ASSIGN_OR_RAISE(expr, Result<Expression>(pre(std::move(expr))));

  auto call = expr.call();
  if (!call) return expr;

  bool at_least_one_modified = false;
  std::vector<Expression> modified_arguments;

  for (size_t i = 0; i < call->arguments.size(); ++i) {
    ARROW_ASSIGN_OR_RAISE(auto modified_argument,
                          ModifyExpression(call->arguments[i], pre, post_call));

    if (Identical(modified_argument, call->arguments[i])) {
      continue;
    }

    if (!at_least_one_modified) {
      modified_arguments = call->arguments;
      at_least_one_modified = true;
    }

    modified_arguments[i] = std::move(modified_argument);
  }

  if (at_least_one_modified) {
    // reconstruct the call expression with the modified arguments
    auto modified_call = *call;
    modified_call.arguments = std::move(modified_arguments);
    return post_call(Expression(std::move(modified_call)), &expr);
  }

  return post_call(std::move(expr), NULLPTR);
}

}  // namespace compute
}  // namespace arrow

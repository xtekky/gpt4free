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
#include <cassert>
#include <cstdint>
#include <functional>
#include <random>
#include "arrow/buffer.h"
#include "arrow/compute/exec/util.h"
#include "arrow/util/pcg_random.h"

namespace arrow {
namespace compute {

class PartitionSort {
 public:
  /// \brief Bucket sort rows on partition ids in O(num_rows) time.
  ///
  /// Include in the output exclusive cummulative sum of bucket sizes.
  /// This corresponds to ranges in the sorted array containing all row ids for
  /// each of the partitions.
  ///
  /// prtn_ranges must be initailized and have at least prtn_ranges + 1 elements
  /// when this method returns prtn_ranges[i] will contains the total number of
  /// elements in partitions 0 through i.  prtn_ranges[0] will be 0.
  ///
  /// prtn_id_impl must be a function that takes in a row id (int) and returns
  /// a partition id (int).  The returned partition id must be between 0 and
  /// num_prtns (exclusive).
  ///
  /// output_pos_impl is a function that takes in a row id (int) and a position (int)
  /// in the bucket sorted output.  The function should insert the row in the
  /// output.
  ///
  /// For example:
  ///
  /// in_arr: [5, 7, 2, 3, 5, 4]
  /// num_prtns: 3
  /// prtn_id_impl: [&in_arr] (int row_id) { return in_arr[row_id] / 3; }
  /// output_pos_impl: [&out_arr] (int row_id, int pos) { out_arr[pos] = row_id; }
  ///
  /// After Execution
  /// out_arr: [2, 5, 3, 5, 4, 7]
  /// prtn_ranges: [0, 1, 5, 6]
  template <class INPUT_PRTN_ID_FN, class OUTPUT_POS_FN>
  static void Eval(int64_t num_rows, int num_prtns, uint16_t* prtn_ranges,
                   INPUT_PRTN_ID_FN prtn_id_impl, OUTPUT_POS_FN output_pos_impl) {
    ARROW_DCHECK(num_rows > 0 && num_rows <= (1 << 15));
    ARROW_DCHECK(num_prtns >= 1 && num_prtns <= (1 << 15));

    memset(prtn_ranges, 0, (num_prtns + 1) * sizeof(uint16_t));

    for (int64_t i = 0; i < num_rows; ++i) {
      int prtn_id = static_cast<int>(prtn_id_impl(i));
      ++prtn_ranges[prtn_id + 1];
    }

    uint16_t sum = 0;
    for (int i = 0; i < num_prtns; ++i) {
      uint16_t sum_next = sum + prtn_ranges[i + 1];
      prtn_ranges[i + 1] = sum;
      sum = sum_next;
    }

    for (int64_t i = 0; i < num_rows; ++i) {
      int prtn_id = static_cast<int>(prtn_id_impl(i));
      int pos = prtn_ranges[prtn_id + 1]++;
      output_pos_impl(i, pos);
    }
  }
};

/// \brief A control for synchronizing threads on a partitionable workload
class PartitionLocks {
 public:
  PartitionLocks();
  ~PartitionLocks();
  /// \brief Initializes the control, must be called before use
  ///
  /// \param num_threads Maximum number of threads that will access the partitions
  /// \param num_prtns Number of partitions to synchronize
  void Init(size_t num_threads, int num_prtns);
  /// \brief Cleans up the control, it should not be used after this call
  void CleanUp();
  /// \brief Acquire a partition to work on one
  ///
  /// \param thread_id The index of the thread trying to acquire the partition lock
  /// \param num_prtns Length of prtns_to_try, must be <= num_prtns used in Init
  /// \param prtns_to_try An array of partitions that still have remaining work
  /// \param limit_retries If false, this method will spinwait forever until success
  /// \param max_retries Max times to attempt checking out work before returning false
  /// \param[out] locked_prtn_id The id of the partition locked
  /// \param[out] locked_prtn_id_pos The index of the partition locked in prtns_to_try
  /// \return True if a partition was locked, false if max_retries was attempted
  ///         without successfully acquiring a lock
  ///
  /// This method is thread safe
  bool AcquirePartitionLock(size_t thread_id, int num_prtns, const int* prtns_to_try,
                            bool limit_retries, int max_retries, int* locked_prtn_id,
                            int* locked_prtn_id_pos);
  /// \brief Release a partition so that other threads can work on it
  void ReleasePartitionLock(int prtn_id);

  // Executes (synchronously and using current thread) the same operation on a set of
  // multiple partitions. Tries to minimize partition locking overhead by randomizing and
  // adjusting order in which partitions are processed.
  //
  // PROCESS_PRTN_FN is a callback which will be executed for each partition after
  // acquiring the lock for that partition. It gets partition id as an argument.
  // IS_PRTN_EMPTY_FN is a callback which filters out (when returning true) partitions
  // with specific ids from processing.
  //
  template <typename IS_PRTN_EMPTY_FN, typename PROCESS_PRTN_FN>
  Status ForEachPartition(size_t thread_id,
                          /*scratch space buffer with space for one element per partition;
                             dirty in and dirty out*/
                          int* temp_unprocessed_prtns, IS_PRTN_EMPTY_FN is_prtn_empty_fn,
                          PROCESS_PRTN_FN process_prtn_fn) {
    int num_unprocessed_partitions = 0;
    for (int i = 0; i < num_prtns_; ++i) {
      bool is_prtn_empty = is_prtn_empty_fn(i);
      if (!is_prtn_empty) {
        temp_unprocessed_prtns[num_unprocessed_partitions++] = i;
      }
    }
    while (num_unprocessed_partitions > 0) {
      int locked_prtn_id;
      int locked_prtn_id_pos;
      AcquirePartitionLock(thread_id, num_unprocessed_partitions, temp_unprocessed_prtns,
                           /*limit_retries=*/false, /*max_retries=*/-1, &locked_prtn_id,
                           &locked_prtn_id_pos);
      {
        class AutoReleaseLock {
         public:
          AutoReleaseLock(PartitionLocks* locks, int prtn_id)
              : locks(locks), prtn_id(prtn_id) {}
          ~AutoReleaseLock() { locks->ReleasePartitionLock(prtn_id); }
          PartitionLocks* locks;
          int prtn_id;
        } auto_release_lock(this, locked_prtn_id);
        ARROW_RETURN_NOT_OK(process_prtn_fn(locked_prtn_id));
      }
      if (locked_prtn_id_pos < num_unprocessed_partitions - 1) {
        temp_unprocessed_prtns[locked_prtn_id_pos] =
            temp_unprocessed_prtns[num_unprocessed_partitions - 1];
      }
      --num_unprocessed_partitions;
    }
    return Status::OK();
  }

 private:
  std::atomic<bool>* lock_ptr(int prtn_id);
  int random_int(size_t thread_id, int num_values);

  struct PartitionLock {
    static constexpr int kCacheLineBytes = 64;
    std::atomic<bool> lock;
    uint8_t padding[kCacheLineBytes];
  };
  int num_prtns_;
  std::unique_ptr<PartitionLock[]> locks_;
  std::unique_ptr<arrow::random::pcg32_fast[]> rngs_;
};

}  // namespace compute
}  // namespace arrow

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
#include "arrow/compute/exec/key_map.h"
#include "arrow/compute/exec/options.h"
#include "arrow/compute/exec/partition_util.h"
#include "arrow/compute/exec/schema_util.h"
#include "arrow/compute/exec/task_util.h"
#include "arrow/compute/kernels/row_encoder.h"
#include "arrow/compute/light_array.h"
#include "arrow/compute/row/encode_internal.h"

namespace arrow {
namespace compute {

class RowArrayAccessor {
 public:
  // Find the index of this varbinary column within the sequence of all
  // varbinary columns encoded in rows.
  //
  static int VarbinaryColumnId(const RowTableMetadata& row_metadata, int column_id);

  // Calculate how many rows to skip from the tail of the
  // sequence of selected rows, such that the total size of skipped rows is at
  // least equal to the size specified by the caller. Skipping of the tail rows
  // is used to allow for faster processing by the caller of remaining rows
  // without checking buffer bounds (useful with SIMD or fixed size memory loads
  // and stores).
  //
  static int NumRowsToSkip(const RowTableImpl& rows, int column_id, int num_rows,
                           const uint32_t* row_ids, int num_tail_bytes_to_skip);

  // The supplied lambda will be called for each row in the given list of rows.
  // The arguments given to it will be:
  // - index of a row (within the set of selected rows),
  // - pointer to the value,
  // - byte length of the value.
  //
  // The information about nulls (validity bitmap) is not used in this call and
  // has to be processed separately.
  //
  template <class PROCESS_VALUE_FN>
  static void Visit(const RowTableImpl& rows, int column_id, int num_rows,
                    const uint32_t* row_ids, PROCESS_VALUE_FN process_value_fn);

  // The supplied lambda will be called for each row in the given list of rows.
  // The arguments given to it will be:
  // - index of a row (within the set of selected rows),
  // - byte 0xFF if the null is set for the row or 0x00 otherwise.
  //
  template <class PROCESS_VALUE_FN>
  static void VisitNulls(const RowTableImpl& rows, int column_id, int num_rows,
                         const uint32_t* row_ids, PROCESS_VALUE_FN process_value_fn);

 private:
#if defined(ARROW_HAVE_AVX2)
  // This is equivalent to Visit method, but processing 8 rows at a time in a
  // loop.
  // Returns the number of processed rows, which may be less than requested (up
  // to 7 rows at the end may be skipped).
  //
  template <class PROCESS_8_VALUES_FN>
  static int Visit_avx2(const RowTableImpl& rows, int column_id, int num_rows,
                        const uint32_t* row_ids, PROCESS_8_VALUES_FN process_8_values_fn);

  // This is equivalent to VisitNulls method, but processing 8 rows at a time in
  // a loop. Returns the number of processed rows, which may be less than
  // requested (up to 7 rows at the end may be skipped).
  //
  template <class PROCESS_8_VALUES_FN>
  static int VisitNulls_avx2(const RowTableImpl& rows, int column_id, int num_rows,
                             const uint32_t* row_ids,
                             PROCESS_8_VALUES_FN process_8_values_fn);
#endif
};

// Write operations (appending batch rows) must not be called by more than one
// thread at the same time.
//
// Read operations (row comparison, column decoding)
// can be called by multiple threads concurrently.
//
struct RowArray {
  RowArray() : is_initialized_(false) {}

  Status InitIfNeeded(MemoryPool* pool, const ExecBatch& batch);
  Status InitIfNeeded(MemoryPool* pool, const RowTableMetadata& row_metadata);

  Status AppendBatchSelection(MemoryPool* pool, const ExecBatch& batch, int begin_row_id,
                              int end_row_id, int num_row_ids, const uint16_t* row_ids,
                              std::vector<KeyColumnArray>& temp_column_arrays);

  // This can only be called for a minibatch.
  //
  void Compare(const ExecBatch& batch, int begin_row_id, int end_row_id, int num_selected,
               const uint16_t* batch_selection_maybe_null, const uint32_t* array_row_ids,
               uint32_t* out_num_not_equal, uint16_t* out_not_equal_selection,
               int64_t hardware_flags, util::TempVectorStack* temp_stack,
               std::vector<KeyColumnArray>& temp_column_arrays,
               uint8_t* out_match_bitvector_maybe_null = NULLPTR);

  // TODO: add AVX2 version
  //
  Status DecodeSelected(ResizableArrayData* target, int column_id, int num_rows_to_append,
                        const uint32_t* row_ids, MemoryPool* pool) const;

  void DebugPrintToFile(const char* filename, bool print_sorted) const;

  int64_t num_rows() const { return is_initialized_ ? rows_.length() : 0; }

  bool is_initialized_;
  RowTableEncoder encoder_;
  RowTableImpl rows_;
  RowTableImpl rows_temp_;
};

// Implements concatenating multiple row arrays into a single one, using
// potentially multiple threads, each processing a single input row array.
//
class RowArrayMerge {
 public:
  // Calculate total number of rows and size in bytes for merged sequence of
  // rows and allocate memory for it.
  //
  // If the rows are of varying length, initialize in the offset array the first
  // entry for the write area for each input row array. Leave all other
  // offsets and buffers uninitialized.
  //
  // All input sources must be initialized, but they can contain zero rows.
  //
  // Output in vector the first target row id for each source (exclusive
  // cummulative sum of number of rows in sources). This output is optional,
  // caller can pass in nullptr to indicate that it is not needed.
  //
  static Status PrepareForMerge(RowArray* target, const std::vector<RowArray*>& sources,
                                std::vector<int64_t>* first_target_row_id,
                                MemoryPool* pool);

  // Copy rows from source array to target array.
  // Both arrays must have the same row metadata.
  // Target array must already have the memory reserved in all internal buffers
  // for the copy of the rows.
  //
  // Copy of the rows will occupy the same amount of space in the target array
  // buffers as in the source array, but in the target array we pick at what row
  // position and offset we start writing.
  //
  // Optionally, the rows may be reordered during copy according to the
  // provided permutation, which represents some sorting order of source rows.
  // Nth element of the permutation array is the source row index for the Nth
  // row written into target array. If permutation is missing (null), then the
  // order of source rows will remain unchanged.
  //
  // In case of varying length rows, we purposefully skip outputting of N+1 (one
  // after last) offset, to allow concurrent copies of rows done to adjacent
  // ranges in the target array. This offset should already contain the right
  // value after calling the method preparing target array for merge (which
  // initializes boundary offsets for target row ranges for each source).
  //
  static void MergeSingle(RowArray* target, const RowArray& source,
                          int64_t first_target_row_id,
                          const int64_t* source_rows_permutation);

 private:
  // Copy rows from source array to a region of the target array.
  // This implementation is for fixed length rows.
  // Null information needs to be handled separately.
  //
  static void CopyFixedLength(RowTableImpl* target, const RowTableImpl& source,
                              int64_t first_target_row_id,
                              const int64_t* source_rows_permutation);

  // Copy rows from source array to a region of the target array.
  // This implementation is for varying length rows.
  // Null information needs to be handled separately.
  //
  static void CopyVaryingLength(RowTableImpl* target, const RowTableImpl& source,
                                int64_t first_target_row_id,
                                int64_t first_target_row_offset,
                                const int64_t* source_rows_permutation);

  // Copy null information from rows from source array to a region of the target
  // array.
  //
  static void CopyNulls(RowTableImpl* target, const RowTableImpl& source,
                        int64_t first_target_row_id,
                        const int64_t* source_rows_permutation);
};

// Implements merging of multiple SwissTables into a single one, using
// potentially multiple threads, each processing a single input source.
//
// Each source should correspond to a range of original hashes.
// A row belongs to a source with index determined by K highest bits of
// original hash. That means that the number of sources must be a power of 2.
//
// We assume that the hash values used and stored inside source tables
// have K highest bits removed from the original hash in order to avoid huge
// number of hash collisions that would occur otherwise.
// These bits will be reinserted back (original hashes will be used) when
// merging into target.
//
class SwissTableMerge {
 public:
  // Calculate total number of blocks for merged table.
  // Allocate buffers sized accordingly and initialize empty target table.
  //
  // All input sources must be initialized, but they can be empty.
  //
  // Output in a vector the first target group id for each source (exclusive
  // cummulative sum of number of groups in sources). This output is optional,
  // caller can pass in nullptr to indicate that it is not needed.
  //
  static Status PrepareForMerge(SwissTable* target,
                                const std::vector<SwissTable*>& sources,
                                std::vector<uint32_t>* first_target_group_id,
                                MemoryPool* pool);

  // Copy all entries from source to a range of blocks (partition) of target.
  //
  // During copy, adjust group ids from source by adding provided base id.
  //
  // Skip entries from source that would cross partition boundaries (range of
  // blocks) when inserted into target. Save their data in output vector for
  // processing later. We postpone inserting these overflow entries in order to
  // allow concurrent processing of all partitions. Overflow entries will be
  // handled by a single-thread afterwards.
  //
  static void MergePartition(SwissTable* target, const SwissTable* source,
                             uint32_t partition_id, int num_partition_bits,
                             uint32_t base_group_id,
                             std::vector<uint32_t>* overflow_group_ids,
                             std::vector<uint32_t>* overflow_hashes);

  // Single-threaded processing of remaining groups, that could not be
  // inserted in partition merge phase
  // (due to entries from one partition spilling over due to full blocks into
  // the next partition).
  //
  static void InsertNewGroups(SwissTable* target, const std::vector<uint32_t>& group_ids,
                              const std::vector<uint32_t>& hashes);

 private:
  // Insert a new group id.
  //
  // Assumes that there are enough slots in the target
  // and there is no need to resize it.
  //
  // Max block id can be provided, in which case the search for an empty slot to
  // insert new entry to will stop after visiting that block.
  //
  // Max block id value greater or equal to the number of blocks guarantees that
  // the search will not be stopped.
  //
  static inline bool InsertNewGroup(SwissTable* target, uint64_t group_id, uint32_t hash,
                                    int64_t max_block_id);
};

struct SwissTableWithKeys {
  struct Input {
    Input(const ExecBatch* in_batch, int in_batch_start_row, int in_batch_end_row,
          util::TempVectorStack* in_temp_stack,
          std::vector<KeyColumnArray>* in_temp_column_arrays);

    Input(const ExecBatch* in_batch, util::TempVectorStack* in_temp_stack,
          std::vector<KeyColumnArray>* in_temp_column_arrays);

    Input(const ExecBatch* in_batch, int in_num_selected, const uint16_t* in_selection,
          util::TempVectorStack* in_temp_stack,
          std::vector<KeyColumnArray>* in_temp_column_arrays,
          std::vector<uint32_t>* in_temp_group_ids);

    Input(const Input& base, int num_rows_to_skip, int num_rows_to_include);

    const ExecBatch* batch;
    // Window of the batch to operate on.
    // The window information is only used if row selection is null.
    //
    int batch_start_row;
    int batch_end_row;
    // Optional selection.
    // Used instead of window of the batch if not null.
    //
    int num_selected;
    const uint16_t* selection_maybe_null;
    // Thread specific scratch buffers for storing temporary data.
    //
    util::TempVectorStack* temp_stack;
    std::vector<KeyColumnArray>* temp_column_arrays;
    std::vector<uint32_t>* temp_group_ids;
  };

  Status Init(int64_t hardware_flags, MemoryPool* pool);

  void InitCallbacks();

  static void Hash(Input* input, uint32_t* hashes, int64_t hardware_flags);

  // If input uses selection, then hashes array must have one element for every
  // row in the whole (unfiltered and not spliced) input exec batch. Otherwise,
  // there must be one element in hashes array for every value in the window of
  // the exec batch specified by input.
  //
  // Output arrays will contain one element for every selected batch row in
  // input (selected either by selection vector if provided or input window
  // otherwise).
  //
  void MapReadOnly(Input* input, const uint32_t* hashes, uint8_t* match_bitvector,
                   uint32_t* key_ids);
  Status MapWithInserts(Input* input, const uint32_t* hashes, uint32_t* key_ids);

  SwissTable* swiss_table() { return &swiss_table_; }
  const SwissTable* swiss_table() const { return &swiss_table_; }
  RowArray* keys() { return &keys_; }
  const RowArray* keys() const { return &keys_; }

 private:
  void EqualCallback(int num_keys, const uint16_t* selection_maybe_null,
                     const uint32_t* group_ids, uint32_t* out_num_keys_mismatch,
                     uint16_t* out_selection_mismatch, void* callback_ctx);
  Status AppendCallback(int num_keys, const uint16_t* selection, void* callback_ctx);
  Status Map(Input* input, bool insert_missing, const uint32_t* hashes,
             uint8_t* match_bitvector_maybe_null, uint32_t* key_ids);

  SwissTable::EqualImpl equal_impl_;
  SwissTable::AppendImpl append_impl_;

  SwissTable swiss_table_;
  RowArray keys_;
};

// Enhances SwissTableWithKeys with the following structures used by hash join:
// - storage of payloads (that unlike keys do not have to be unique)
// - mapping from a key to all inserted payloads corresponding to it (we can
// store multiple rows corresponding to a single key)
// - bit-vectors for keeping track of whether each payload had a match during
// evaluation of join.
//
class SwissTableForJoin {
  friend class SwissTableForJoinBuild;

 public:
  void UpdateHasMatchForKeys(int64_t thread_id, int num_rows, const uint32_t* key_ids);
  void MergeHasMatch();

  const SwissTableWithKeys* keys() const { return &map_; }
  SwissTableWithKeys* keys() { return &map_; }
  const RowArray* payloads() const { return no_payload_columns_ ? NULLPTR : &payloads_; }
  const uint32_t* key_to_payload() const {
    return no_duplicate_keys_ ? NULLPTR : row_offset_for_key_.data();
  }
  const uint8_t* has_match() const {
    return has_match_.empty() ? NULLPTR : has_match_.data();
  }
  int64_t num_keys() const { return map_.keys()->num_rows(); }
  int64_t num_rows() const {
    return no_duplicate_keys_ ? num_keys() : row_offset_for_key_[num_keys()];
  }

  uint32_t payload_id_to_key_id(uint32_t payload_id) const;
  // Input payload ids must form an increasing sequence.
  //
  void payload_ids_to_key_ids(int num_rows, const uint32_t* payload_ids,
                              uint32_t* key_ids) const;

 private:
  uint8_t* local_has_match(int64_t thread_id);

  // Degree of parallelism (number of threads)
  int dop_;

  struct ThreadLocalState {
    std::vector<uint8_t> has_match;
  };
  std::vector<ThreadLocalState> local_states_;
  std::vector<uint8_t> has_match_;

  SwissTableWithKeys map_;

  bool no_duplicate_keys_;
  // Not used if no_duplicate_keys_ is true.
  std::vector<uint32_t> row_offset_for_key_;

  bool no_payload_columns_;
  // Not used if no_payload_columns_ is true.
  RowArray payloads_;
};

// Implements parallel build process for hash table for join from a sequence of
// exec batches with input rows.
//
class SwissTableForJoinBuild {
 public:
  Status Init(SwissTableForJoin* target, int dop, int64_t num_rows,
              bool reject_duplicate_keys, bool no_payload,
              const std::vector<KeyColumnMetadata>& key_types,
              const std::vector<KeyColumnMetadata>& payload_types, MemoryPool* pool,
              int64_t hardware_flags);

  // In the first phase of parallel hash table build, threads pick unprocessed
  // exec batches, partition the rows based on hash, and update all of the
  // partitions with information related to that batch of rows.
  //
  Status PushNextBatch(int64_t thread_id, const ExecBatch& key_batch,
                       const ExecBatch* payload_batch_maybe_null,
                       util::TempVectorStack* temp_stack);

  // Allocate memory and initialize counters required for parallel merging of
  // hash table partitions.
  // Single-threaded.
  //
  Status PreparePrtnMerge();

  // Second phase of parallel hash table build.
  // Each partition can be processed by a different thread.
  // Parallel step.
  //
  void PrtnMerge(int prtn_id);

  // Single-threaded processing of the rows that have been skipped during
  // parallel merging phase, due to hash table search resulting in crossing
  // partition boundaries.
  //
  void FinishPrtnMerge(util::TempVectorStack* temp_stack);

  // The number of partitions is the number of parallel tasks to execute during
  // the final phase of hash table build process.
  //
  int num_prtns() const { return num_prtns_; }

  bool no_payload() const { return no_payload_; }

 private:
  void InitRowArray();
  Status ProcessPartition(int64_t thread_id, const ExecBatch& key_batch,
                          const ExecBatch* payload_batch_maybe_null,
                          util::TempVectorStack* temp_stack, int prtn_id);

  SwissTableForJoin* target_;
  // DOP stands for Degree Of Parallelism - the maximum number of participating
  // threads.
  //
  int dop_;
  // Partition is a unit of parallel work.
  //
  // There must be power of 2 partitions (bits of hash will be used to
  // identify them).
  //
  // Pick number of partitions at least equal to the number of threads (degree
  // of parallelism).
  //
  int log_num_prtns_;
  int num_prtns_;
  int64_t num_rows_;
  // Left-semi and left-anti-semi joins do not need more than one copy of the
  // same key in the hash table.
  // This flag, if set, will result in filtering rows with duplicate keys before
  // inserting them into hash table.
  //
  // Since left-semi and left-anti-semi joins also do not need payload, when
  // this flag is set there also will not be any processing of payload.
  //
  bool reject_duplicate_keys_;
  // This flag, when set, will result in skipping any processing of the payload.
  //
  // The flag for rejecting duplicate keys (which should be set for left-semi
  // and left-anti joins), when set, will force this flag to also be set, but
  // other join flavors may set it to true as well if no payload columns are
  // needed for join output.
  //
  bool no_payload_;
  MemoryPool* pool_;
  int64_t hardware_flags_;

  // One per partition.
  //
  struct PartitionState {
    SwissTableWithKeys keys;
    RowArray payloads;
    std::vector<uint32_t> key_ids;
    std::vector<uint32_t> overflow_key_ids;
    std::vector<uint32_t> overflow_hashes;
  };

  // One per thread.
  //
  // Buffers for storing temporary intermediate results when processing input
  // batches.
  //
  struct ThreadState {
    std::vector<uint32_t> batch_hashes;
    std::vector<uint16_t> batch_prtn_ranges;
    std::vector<uint16_t> batch_prtn_row_ids;
    std::vector<int> temp_prtn_ids;
    std::vector<uint32_t> temp_group_ids;
    std::vector<KeyColumnArray> temp_column_arrays;
  };

  std::vector<PartitionState> prtn_states_;
  std::vector<ThreadState> thread_states_;
  PartitionLocks prtn_locks_;

  std::vector<int64_t> partition_keys_first_row_id_;
  std::vector<int64_t> partition_payloads_first_row_id_;
};

class JoinResultMaterialize {
 public:
  void Init(MemoryPool* pool, const HashJoinProjectionMaps* probe_schemas,
            const HashJoinProjectionMaps* build_schemas);

  void SetBuildSide(const RowArray* build_keys, const RowArray* build_payloads,
                    bool payload_id_same_as_key_id);

  // Input probe side batches should contain all key columns followed by all
  // payload columns.
  //
  Status AppendProbeOnly(const ExecBatch& key_and_payload, int num_rows_to_append,
                         const uint16_t* row_ids, int* num_rows_appended);

  Status AppendBuildOnly(int num_rows_to_append, const uint32_t* key_ids,
                         const uint32_t* payload_ids, int* num_rows_appended);

  Status Append(const ExecBatch& key_and_payload, int num_rows_to_append,
                const uint16_t* row_ids, const uint32_t* key_ids,
                const uint32_t* payload_ids, int* num_rows_appended);

  // Should only be called if num_rows() returns non-zero.
  //
  Status Flush(ExecBatch* out);

  int num_rows() const { return num_rows_; }

  template <class APPEND_ROWS_FN, class OUTPUT_BATCH_FN>
  Status AppendAndOutput(int num_rows_to_append, const APPEND_ROWS_FN& append_rows_fn,
                         const OUTPUT_BATCH_FN& output_batch_fn) {
    int offset = 0;
    for (;;) {
      int num_rows_appended = 0;
      ARROW_RETURN_NOT_OK(append_rows_fn(num_rows_to_append, offset, &num_rows_appended));
      if (num_rows_appended < num_rows_to_append) {
        ExecBatch batch;
        ARROW_RETURN_NOT_OK(Flush(&batch));
        output_batch_fn(batch);
        num_rows_to_append -= num_rows_appended;
        offset += num_rows_appended;
      } else {
        break;
      }
    }
    return Status::OK();
  }

  template <class OUTPUT_BATCH_FN>
  Status AppendProbeOnly(const ExecBatch& key_and_payload, int num_rows_to_append,
                         const uint16_t* row_ids, OUTPUT_BATCH_FN output_batch_fn) {
    return AppendAndOutput(
        num_rows_to_append,
        [&](int num_rows_to_append_left, int offset, int* num_rows_appended) {
          return AppendProbeOnly(key_and_payload, num_rows_to_append_left,
                                 row_ids + offset, num_rows_appended);
        },
        output_batch_fn);
  }

  template <class OUTPUT_BATCH_FN>
  Status AppendBuildOnly(int num_rows_to_append, const uint32_t* key_ids,
                         const uint32_t* payload_ids, OUTPUT_BATCH_FN output_batch_fn) {
    return AppendAndOutput(
        num_rows_to_append,
        [&](int num_rows_to_append_left, int offset, int* num_rows_appended) {
          return AppendBuildOnly(
              num_rows_to_append_left, key_ids ? key_ids + offset : NULLPTR,
              payload_ids ? payload_ids + offset : NULLPTR, num_rows_appended);
        },
        output_batch_fn);
  }

  template <class OUTPUT_BATCH_FN>
  Status Append(const ExecBatch& key_and_payload, int num_rows_to_append,
                const uint16_t* row_ids, const uint32_t* key_ids,
                const uint32_t* payload_ids, OUTPUT_BATCH_FN output_batch_fn) {
    return AppendAndOutput(
        num_rows_to_append,
        [&](int num_rows_to_append_left, int offset, int* num_rows_appended) {
          return Append(key_and_payload, num_rows_to_append_left,
                        row_ids ? row_ids + offset : NULLPTR,
                        key_ids ? key_ids + offset : NULLPTR,
                        payload_ids ? payload_ids + offset : NULLPTR, num_rows_appended);
        },
        output_batch_fn);
  }

  template <class OUTPUT_BATCH_FN>
  Status Flush(OUTPUT_BATCH_FN output_batch_fn) {
    if (num_rows_ > 0) {
      ExecBatch batch({}, num_rows_);
      ARROW_RETURN_NOT_OK(Flush(&batch));
      output_batch_fn(std::move(batch));
    }
    return Status::OK();
  }

  int64_t num_produced_batches() const { return num_produced_batches_; }

 private:
  bool HasProbeOutput() const;
  bool HasBuildKeyOutput() const;
  bool HasBuildPayloadOutput() const;
  bool NeedsKeyId() const;
  bool NeedsPayloadId() const;
  Result<std::shared_ptr<ArrayData>> FlushBuildColumn(
      const std::shared_ptr<DataType>& data_type, const RowArray* row_array,
      int column_id, uint32_t* row_ids);

  MemoryPool* pool_;
  const HashJoinProjectionMaps* probe_schemas_;
  const HashJoinProjectionMaps* build_schemas_;
  const RowArray* build_keys_;
  // Payload array pointer may be left as null, if no payload columns are
  // in the output column set.
  //
  const RowArray* build_payloads_;
  // If true, then ignore updating payload ids and use key ids instead when
  // reading.
  //
  bool payload_id_same_as_key_id_;
  std::vector<int> probe_output_to_key_and_payload_;

  // Number of accumulated rows (since last flush)
  //
  int num_rows_;
  // Accumulated output columns from probe side batches.
  //
  ExecBatchBuilder batch_builder_;
  // Accumulated build side row references.
  //
  std::vector<uint32_t> key_ids_;
  std::vector<uint32_t> payload_ids_;
  // Information about ranges of rows from build side,
  // that in the accumulated materialized results have all fields set to null.
  //
  // Each pair contains index of the first output row in the range and the
  // length of the range. Only rows outside of these ranges have data present in
  // the key_ids_ and payload_ids_ arrays.
  //
  std::vector<std::pair<int, int>> null_ranges_;

  int64_t num_produced_batches_;
};

// When comparing two join key values to check if they are equal, hash join allows to
// chose (even separately for each field within the join key) whether two null values are
// considered to be equal (IS comparison) or not (EQ comparison). For EQ comparison we
// need to filter rows with nulls in keys outside of hash table lookups, since hash table
// implementation always treats two nulls as equal (like IS comparison).
//
// Implements evaluating filter bit vector eliminating rows that do not have
// join matches due to nulls in key columns.
//
class JoinNullFilter {
 public:
  // The batch for which the filter bit vector will be computed
  // needs to start with all key columns but it may contain more columns
  // (payload) following them.
  //
  static void Filter(const ExecBatch& key_batch, int batch_start_row, int num_batch_rows,
                     const std::vector<JoinKeyCmp>& cmp, bool* all_valid,
                     bool and_with_input, uint8_t* out_bit_vector);
};

// A helper class that takes hash table lookup results for a range of rows in
// input batch, that is:
// - bit vector marking whether there was a key match in the hash table
// - key id if there was a match
// - mapping from key id to a range of payload ids associated with that key
// (representing multiple matching rows in a hash table for a single row in an
// input batch), and iterates output batches of limited size containing tuples
// describing all matching pairs of rows:
// - input batch row id (only rows that have matches in the hash table are
// included)
// - key id for a match
// - payload id (different one for each matching row in the hash table)
//
class JoinMatchIterator {
 public:
  void SetLookupResult(int num_batch_rows, int start_batch_row,
                       const uint8_t* batch_has_match, const uint32_t* key_ids,
                       bool no_duplicate_keys, const uint32_t* key_to_payload);
  bool GetNextBatch(int num_rows_max, int* out_num_rows, uint16_t* batch_row_ids,
                    uint32_t* key_ids, uint32_t* payload_ids);

 private:
  int num_batch_rows_;
  int start_batch_row_;
  const uint8_t* batch_has_match_;
  const uint32_t* key_ids_;

  bool no_duplicate_keys_;
  const uint32_t* key_to_payload_;

  // Index of the first not fully processed input row, or number of rows if all
  // have been processed. May be pointing to a row with no matches.
  //
  int current_row_;
  // Index of the first unprocessed match for the input row. May be zero if the
  // row has no matches.
  //
  int current_match_for_row_;
};

// Implements entire processing of a probe side exec batch,
// provided the join hash table is already built and available.
//
class JoinProbeProcessor {
 public:
  using OutputBatchFn = std::function<void(int64_t, ExecBatch)>;

  void Init(int num_key_columns, JoinType join_type, SwissTableForJoin* hash_table,
            std::vector<JoinResultMaterialize*> materialize,
            const std::vector<JoinKeyCmp>* cmp, OutputBatchFn output_batch_fn);
  Status OnNextBatch(int64_t thread_id, const ExecBatch& keypayload_batch,
                     util::TempVectorStack* temp_stack,
                     std::vector<KeyColumnArray>* temp_column_arrays);

  // Must be called by a single-thread having exclusive access to the instance
  // of this class. The caller is responsible for ensuring that.
  //
  Status OnFinished();

 private:
  int num_key_columns_;
  JoinType join_type_;

  SwissTableForJoin* hash_table_;
  // One element per thread
  //
  std::vector<JoinResultMaterialize*> materialize_;
  const std::vector<JoinKeyCmp>* cmp_;
  OutputBatchFn output_batch_fn_;
};

}  // namespace compute
}  // namespace arrow

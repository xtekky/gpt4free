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

#include <memory>
#include <unordered_map>

#include "arrow/compute/exec.h"
#include "arrow/compute/exec/schema_util.h"
#include "arrow/compute/kernels/row_encoder.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type.h"

// This file contains hash join logic related to handling of dictionary encoded key
// columns.
//
// A key column from probe side of the join can be matched against a key column from build
// side of the join, as long as the underlying value types are equal. That means that:
// - both scalars and arrays can be used and even mixed in the same column
// - dictionary column can be matched against non-dictionary column if underlying value
// types are equal
// - dictionary column can be matched against dictionary column with a different index
// type, and potentially using a different dictionary, if underlying value types are equal
//
// We currently require in hash join that for all dictionary encoded columns, the same
// dictionary is used in all input exec batches.
//
// In order to allow matching columns with different dictionaries, different dictionary
// index types, and dictionary key against non-dictionary key, internally comparisons will
// be evaluated after remapping values on both sides of the join to a common
// representation (which will be called "unified representation"). This common
// representation is a column of int32() type (not a dictionary column). It represents an
// index in the unified dictionary computed for the (only) dictionary present on build
// side (an empty dictionary is still created for an empty build side). Null value is
// always represented in this common representation as null int32 value, unified
// dictionary will never contain a null value (so there is no ambiguity of representing
// nulls as either index to a null entry in the dictionary or null index).
//
// Unified dictionary represents values present on build side. There may be values on
// probe side that are not present in it. All such values, that are not null, are mapped
// in the common representation to a special constant kMissingValueId.
//

namespace arrow {
namespace compute {

using internal::RowEncoder;

/// Helper class with operations that are stateless and common to processing of dictionary
/// keys on both build and probe side.
class HashJoinDictUtil {
 public:
  // Null values in unified representation are always represented as null that has
  // corresponding integer set to this constant
  static constexpr int32_t kNullId = 0;
  // Constant representing a value, that is not null, missing on the build side, in
  // unified representation.
  static constexpr int32_t kMissingValueId = -1;

  // Check if data types of corresponding pair of key column on build and probe side are
  // compatible
  static bool KeyDataTypesValid(const std::shared_ptr<DataType>& probe_data_type,
                                const std::shared_ptr<DataType>& build_data_type);

  // Input must be dictionary array or dictionary scalar.
  // A precomputed and provided here lookup table in the form of int32() array will be
  // used to remap input indices to unified representation.
  //
  static Result<std::shared_ptr<ArrayData>> IndexRemapUsingLUT(
      ExecContext* ctx, const Datum& indices, int64_t batch_length,
      const std::shared_ptr<ArrayData>& map_array,
      const std::shared_ptr<DataType>& data_type);

  // Return int32() array that contains indices of input dictionary array or scalar after
  // type casting.
  static Result<std::shared_ptr<ArrayData>> ConvertToInt32(
      const std::shared_ptr<DataType>& from_type, const Datum& input,
      int64_t batch_length, ExecContext* ctx);

  // Return an array that contains elements of input int32() array after casting to a
  // given integer type. This is used for mapping unified representation stored in the
  // hash table on build side back to original input data type of hash join, when
  // outputting hash join results to parent exec node.
  //
  static Result<std::shared_ptr<ArrayData>> ConvertFromInt32(
      const std::shared_ptr<DataType>& to_type, const Datum& input, int64_t batch_length,
      ExecContext* ctx);

  // Return dictionary referenced in either dictionary array or dictionary scalar
  static std::shared_ptr<Array> ExtractDictionary(const Datum& data);
};

/// Implements processing of dictionary arrays/scalars in key columns on the build side of
/// a hash join.
/// Each instance of this class corresponds to a single column and stores and
/// processes only the information related to that column.
/// Const methods are thread-safe, non-const methods are not (the caller must make sure
/// that only one thread at any time will access them).
///
class HashJoinDictBuild {
 public:
  // Returns true if the key column (described in input by its data type) requires any
  // pre- or post-processing related to handling dictionaries.
  //
  static bool KeyNeedsProcessing(const std::shared_ptr<DataType>& build_data_type) {
    return (build_data_type->id() == Type::DICTIONARY);
  }

  // Data type of unified representation
  static std::shared_ptr<DataType> DataTypeAfterRemapping() { return int32(); }

  // Should be called only once in hash join, before processing any build or probe
  // batches.
  //
  // Takes a pointer to the dictionary for a corresponding key column on the build side as
  // an input. If the build side is empty, it still needs to be called, but with
  // dictionary pointer set to null.
  //
  // Currently it is required that all input batches on build side share the same
  // dictionary. For each input batch during its pre-processing, dictionary will be
  // checked and error will be returned if it is different then the one provided in the
  // call to this method.
  //
  // Unifies the dictionary. The order of the values is still preserved.
  // Null and duplicate entries are removed. If the dictionary is already unified, its
  // copy will be produced and stored within this class.
  //
  // Prepares the mapping from ids within original dictionary to the ids in the resulting
  // dictionary. This is used later on to pre-process (map to unified representation) key
  // column on build side.
  //
  // Prepares the reverse mapping (in the form of hash table) from values to the ids in
  // the resulting dictionary. This will be used later on to pre-process (map to unified
  // representation) key column on probe side. Values on probe side that are not present
  // in the original dictionary will be mapped to a special constant kMissingValueId. The
  // exception is made for nulls, which get always mapped to nulls (both when null is
  // represented as a dictionary id pointing to a null and a null dictionary id).
  //
  Status Init(ExecContext* ctx, std::shared_ptr<Array> dictionary,
              std::shared_ptr<DataType> index_type, std::shared_ptr<DataType> value_type);

  // Remap array or scalar values into unified representation (array of int32()).
  // Outputs kMissingValueId if input value is not found in the unified dictionary.
  // Outputs null for null input value (with corresponding data set to kNullId).
  //
  Result<std::shared_ptr<ArrayData>> RemapInputValues(ExecContext* ctx,
                                                      const Datum& values,
                                                      int64_t batch_length) const;

  // Remap dictionary array or dictionary scalar on build side to unified representation.
  // Dictionary referenced in the input must match the dictionary that was
  // given during initialization.
  // The output is a dictionary array that references unified dictionary.
  //
  Result<std::shared_ptr<ArrayData>> RemapInput(
      ExecContext* ctx, const Datum& indices, int64_t batch_length,
      const std::shared_ptr<DataType>& data_type) const;

  // Outputs dictionary array referencing unified dictionary, given an array with 32-bit
  // ids.
  // Used to post-process values looked up in a hash table on build side of the hash join
  // before outputting to the parent exec node.
  //
  Result<std::shared_ptr<ArrayData>> RemapOutput(const ArrayData& indices32Bit,
                                                 ExecContext* ctx) const;

  // Release shared pointers and memory
  void CleanUp();

 private:
  // Data type of dictionary ids for the input dictionary on build side
  std::shared_ptr<DataType> index_type_;
  // Data type of values for the input dictionary on build side
  std::shared_ptr<DataType> value_type_;
  // Mapping from (encoded as string) values to the ids in unified dictionary
  std::unordered_map<std::string, int32_t> hash_table_;
  // Mapping from input dictionary ids to unified dictionary ids
  std::shared_ptr<ArrayData> remapped_ids_;
  // Input dictionary
  std::shared_ptr<Array> dictionary_;
  // Unified dictionary
  std::shared_ptr<ArrayData> unified_dictionary_;
};

/// Implements processing of dictionary arrays/scalars in key columns on the probe side of
/// a hash join.
/// Each instance of this class corresponds to a single column and stores and
/// processes only the information related to that column.
/// It is not thread-safe - every participating thread should use its own instance of
/// this class.
///
class HashJoinDictProbe {
 public:
  static bool KeyNeedsProcessing(const std::shared_ptr<DataType>& probe_data_type,
                                 const std::shared_ptr<DataType>& build_data_type);

  // Data type of the result of remapping input key column.
  //
  // The result of remapping is what is used in hash join for matching keys on build and
  // probe side. The exact data types may be different, as described below, and therefore
  // a common representation is needed for simplifying comparisons of pairs of keys on
  // both sides.
  //
  // We support matching key that is of non-dictionary type with key that is of dictionary
  // type, as long as the underlying value types are equal. We support matching when both
  // keys are of dictionary type, regardless whether underlying dictionary index types are
  // the same or not.
  //
  static std::shared_ptr<DataType> DataTypeAfterRemapping(
      const std::shared_ptr<DataType>& build_data_type);

  // Should only be called if KeyNeedsProcessing method returns true for a pair of
  // corresponding key columns from build and probe side.
  // Converts values in order to match the common representation for
  // both build and probe side used in hash table comparison.
  // Supports arrays and scalars as input.
  // Argument opt_build_side should be null if dictionary key on probe side is matched
  // with non-dictionary key on build side.
  //
  Result<std::shared_ptr<ArrayData>> RemapInput(
      const HashJoinDictBuild* opt_build_side, const Datum& data, int64_t batch_length,
      const std::shared_ptr<DataType>& probe_data_type,
      const std::shared_ptr<DataType>& build_data_type, ExecContext* ctx);

  void CleanUp();

 private:
  // May be null if probe side key is non-dictionary. Otherwise it is used to verify that
  // only a single dictionary is referenced in exec batch on probe side of hash join.
  std::shared_ptr<Array> dictionary_;
  // Mapping from dictionary on probe side of hash join (if it is used) to unified
  // representation.
  std::shared_ptr<ArrayData> remapped_ids_;
  // Encoder of key columns that uses unified representation instead of original data type
  // for key columns that need to use it (have dictionaries on either side of the join).
  internal::RowEncoder encoder_;
};

// Encapsulates dictionary handling logic for build side of hash join.
//
class HashJoinDictBuildMulti {
 public:
  Status Init(const SchemaProjectionMaps<HashJoinProjection>& proj_map,
              const ExecBatch* opt_non_empty_batch, ExecContext* ctx);
  static void InitEncoder(const SchemaProjectionMaps<HashJoinProjection>& proj_map,
                          RowEncoder* encoder, ExecContext* ctx);
  Status EncodeBatch(size_t thread_index,
                     const SchemaProjectionMaps<HashJoinProjection>& proj_map,
                     const ExecBatch& batch, RowEncoder* encoder, ExecContext* ctx) const;
  Status PostDecode(const SchemaProjectionMaps<HashJoinProjection>& proj_map,
                    ExecBatch* decoded_key_batch, ExecContext* ctx);
  const HashJoinDictBuild& get_dict_build(int icol) const { return remap_imp_[icol]; }

 private:
  std::vector<bool> needs_remap_;
  std::vector<HashJoinDictBuild> remap_imp_;
};

// Encapsulates dictionary handling logic for probe side of hash join
//
class HashJoinDictProbeMulti {
 public:
  void Init(size_t num_threads);
  bool BatchRemapNeeded(size_t thread_index,
                        const SchemaProjectionMaps<HashJoinProjection>& proj_map_probe,
                        const SchemaProjectionMaps<HashJoinProjection>& proj_map_build,
                        ExecContext* ctx);
  Status EncodeBatch(size_t thread_index,
                     const SchemaProjectionMaps<HashJoinProjection>& proj_map_probe,
                     const SchemaProjectionMaps<HashJoinProjection>& proj_map_build,
                     const HashJoinDictBuildMulti& dict_build, const ExecBatch& batch,
                     RowEncoder** out_encoder, ExecBatch* opt_out_key_batch,
                     ExecContext* ctx);

 private:
  void InitLocalStateIfNeeded(
      size_t thread_index, const SchemaProjectionMaps<HashJoinProjection>& proj_map_probe,
      const SchemaProjectionMaps<HashJoinProjection>& proj_map_build, ExecContext* ctx);
  static void InitEncoder(const SchemaProjectionMaps<HashJoinProjection>& proj_map_probe,
                          const SchemaProjectionMaps<HashJoinProjection>& proj_map_build,
                          RowEncoder* encoder, ExecContext* ctx);
  struct ThreadLocalState {
    bool is_initialized;
    // Whether any key column needs remapping (because of dictionaries used) before doing
    // join hash table lookups
    bool any_needs_remap;
    // Whether each key column needs remapping before doing join hash table lookups
    std::vector<bool> needs_remap;
    std::vector<HashJoinDictProbe> remap_imp;
    // Encoder of key columns that uses unified representation instead of original data
    // type for key columns that need to use it (have dictionaries on either side of the
    // join).
    RowEncoder post_remap_encoder;
  };
  std::vector<ThreadLocalState> local_states_;
};

}  // namespace compute
}  // namespace arrow

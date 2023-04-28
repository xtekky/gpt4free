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

// Tools for dictionaries in IPC context

#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type_fwd.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace ipc {

namespace internal {

class FieldPosition {
 public:
  FieldPosition() : parent_(NULLPTR), index_(-1), depth_(0) {}

  FieldPosition child(int index) const { return {this, index}; }

  std::vector<int> path() const {
    std::vector<int> path(depth_);
    const FieldPosition* cur = this;
    for (int i = depth_ - 1; i >= 0; --i) {
      path[i] = cur->index_;
      cur = cur->parent_;
    }
    return path;
  }

 protected:
  FieldPosition(const FieldPosition* parent, int index)
      : parent_(parent), index_(index), depth_(parent->depth_ + 1) {}

  const FieldPosition* parent_;
  int index_;
  int depth_;
};

}  // namespace internal

/// \brief Map fields in a schema to dictionary ids
///
/// The mapping is structural, i.e. the field path (as a vector of indices)
/// is associated to the dictionary id.  A dictionary id may be associated
/// to multiple fields.
class ARROW_EXPORT DictionaryFieldMapper {
 public:
  DictionaryFieldMapper();
  explicit DictionaryFieldMapper(const Schema& schema);
  ~DictionaryFieldMapper();

  Status AddSchemaFields(const Schema& schema);
  Status AddField(int64_t id, std::vector<int> field_path);

  Result<int64_t> GetFieldId(std::vector<int> field_path) const;

  int num_fields() const;

  /// \brief Returns number of unique dictionaries, taking into
  /// account that different fields can share the same dictionary.
  int num_dicts() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

using DictionaryVector = std::vector<std::pair<int64_t, std::shared_ptr<Array>>>;

/// \brief Memoization data structure for reading dictionaries from IPC streams
///
/// This structure tracks the following associations:
/// - field position (structural) -> dictionary id
/// - dictionary id -> value type
/// - dictionary id -> dictionary (value) data
///
/// Together, they allow resolving dictionary data when reading an IPC stream,
/// using metadata recorded in the schema message and data recorded in the
/// dictionary batch messages (see ResolveDictionaries).
///
/// This structure isn't useful for writing an IPC stream, where only
/// DictionaryFieldMapper is necessary.
class ARROW_EXPORT DictionaryMemo {
 public:
  DictionaryMemo();
  ~DictionaryMemo();

  DictionaryFieldMapper& fields();
  const DictionaryFieldMapper& fields() const;

  /// \brief Return current dictionary corresponding to a particular
  /// id. Returns KeyError if id not found
  Result<std::shared_ptr<ArrayData>> GetDictionary(int64_t id, MemoryPool* pool) const;

  /// \brief Return dictionary value type corresponding to a
  /// particular dictionary id.
  Result<std::shared_ptr<DataType>> GetDictionaryType(int64_t id) const;

  /// \brief Return true if we have a dictionary for the input id
  bool HasDictionary(int64_t id) const;

  /// \brief Add a dictionary value type to the memo with a particular id.
  /// Returns KeyError if a different type is already registered with the same id.
  Status AddDictionaryType(int64_t id, const std::shared_ptr<DataType>& type);

  /// \brief Add a dictionary to the memo with a particular id. Returns
  /// KeyError if that dictionary already exists
  Status AddDictionary(int64_t id, const std::shared_ptr<ArrayData>& dictionary);

  /// \brief Append a dictionary delta to the memo with a particular id. Returns
  /// KeyError if that dictionary does not exists
  Status AddDictionaryDelta(int64_t id, const std::shared_ptr<ArrayData>& dictionary);

  /// \brief Add a dictionary to the memo if it does not have one with the id,
  /// otherwise, replace the dictionary with the new one.
  ///
  /// Return true if the dictionary was added, false if replaced.
  Result<bool> AddOrReplaceDictionary(int64_t id,
                                      const std::shared_ptr<ArrayData>& dictionary);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// For writing: collect dictionary entries to write to the IPC stream, in order
// (i.e. inner dictionaries before dependent outer dictionaries).
ARROW_EXPORT
Result<DictionaryVector> CollectDictionaries(const RecordBatch& batch,
                                             const DictionaryFieldMapper& mapper);

// For reading: resolve all dictionaries in columns, according to the field
// mapping and dictionary arrays stored in memo.
// Columns may be sparse, i.e. some entries may be left null
// (e.g. if an inclusion mask was used).
ARROW_EXPORT
Status ResolveDictionaries(const ArrayDataVector& columns, const DictionaryMemo& memo,
                           MemoryPool* pool);

namespace internal {

// Like CollectDictionaries above, but uses the memo's DictionaryFieldMapper
// and all collected dictionaries are added to the memo using AddDictionary.
//
// This is used as a shortcut in some roundtripping tests (to avoid emitting
// any actual dictionary batches).
ARROW_EXPORT
Status CollectDictionaries(const RecordBatch& batch, DictionaryMemo* memo);

}  // namespace internal

}  // namespace ipc
}  // namespace arrow

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
#include <memory>

#include "arrow/array/array_base.h"
#include "arrow/array/data.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {

// ----------------------------------------------------------------------
// DictionaryArray

/// \brief Array type for dictionary-encoded data with a
/// data-dependent dictionary
///
/// A dictionary array contains an array of non-negative integers (the
/// "dictionary indices") along with a data type containing a "dictionary"
/// corresponding to the distinct values represented in the data.
///
/// For example, the array
///
///   ["foo", "bar", "foo", "bar", "foo", "bar"]
///
/// with dictionary ["bar", "foo"], would have dictionary array representation
///
///   indices: [1, 0, 1, 0, 1, 0]
///   dictionary: ["bar", "foo"]
///
/// The indices in principle may be any integer type.
class ARROW_EXPORT DictionaryArray : public Array {
 public:
  using TypeClass = DictionaryType;

  explicit DictionaryArray(const std::shared_ptr<ArrayData>& data);

  DictionaryArray(const std::shared_ptr<DataType>& type,
                  const std::shared_ptr<Array>& indices,
                  const std::shared_ptr<Array>& dictionary);

  /// \brief Construct DictionaryArray from dictionary and indices
  /// array and validate
  ///
  /// This function does the validation of the indices and input type. It checks if
  /// all indices are non-negative and smaller than the size of the dictionary.
  ///
  /// \param[in] type a dictionary type
  /// \param[in] dictionary the dictionary with same value type as the
  /// type object
  /// \param[in] indices an array of non-negative integers smaller than the
  /// size of the dictionary
  static Result<std::shared_ptr<Array>> FromArrays(
      const std::shared_ptr<DataType>& type, const std::shared_ptr<Array>& indices,
      const std::shared_ptr<Array>& dictionary);

  static Result<std::shared_ptr<Array>> FromArrays(
      const std::shared_ptr<Array>& indices, const std::shared_ptr<Array>& dictionary) {
    return FromArrays(::arrow::dictionary(indices->type(), dictionary->type()), indices,
                      dictionary);
  }

  /// \brief Transpose this DictionaryArray
  ///
  /// This method constructs a new dictionary array with the given dictionary
  /// type, transposing indices using the transpose map.  The type and the
  /// transpose map are typically computed using DictionaryUnifier.
  ///
  /// \param[in] type the new type object
  /// \param[in] dictionary the new dictionary
  /// \param[in] transpose_map transposition array of this array's indices
  ///   into the target array's indices
  /// \param[in] pool a pool to allocate the array data from
  Result<std::shared_ptr<Array>> Transpose(
      const std::shared_ptr<DataType>& type, const std::shared_ptr<Array>& dictionary,
      const int32_t* transpose_map, MemoryPool* pool = default_memory_pool()) const;

  /// \brief Determine whether dictionary arrays may be compared without unification
  bool CanCompareIndices(const DictionaryArray& other) const;

  /// \brief Return the dictionary for this array, which is stored as
  /// a member of the ArrayData internal structure
  std::shared_ptr<Array> dictionary() const;
  std::shared_ptr<Array> indices() const;

  /// \brief Return the ith value of indices, cast to int64_t. Not recommended
  /// for use in performance-sensitive code. Does not validate whether the
  /// value is null or out-of-bounds.
  int64_t GetValueIndex(int64_t i) const;

  const DictionaryType* dict_type() const { return dict_type_; }

 private:
  void SetData(const std::shared_ptr<ArrayData>& data);
  const DictionaryType* dict_type_;
  std::shared_ptr<Array> indices_;

  // Lazily initialized when invoking dictionary()
  mutable std::shared_ptr<Array> dictionary_;
};

/// \brief Helper class for incremental dictionary unification
class ARROW_EXPORT DictionaryUnifier {
 public:
  virtual ~DictionaryUnifier() = default;

  /// \brief Construct a DictionaryUnifier
  /// \param[in] value_type the data type of the dictionaries
  /// \param[in] pool MemoryPool to use for memory allocations
  static Result<std::unique_ptr<DictionaryUnifier>> Make(
      std::shared_ptr<DataType> value_type, MemoryPool* pool = default_memory_pool());

  /// \brief Unify dictionaries accross array chunks
  ///
  /// The dictionaries in the array chunks will be unified, their indices
  /// accordingly transposed.
  ///
  /// Only dictionaries with a primitive value type are currently supported.
  /// However, dictionaries nested inside a more complex type are correctly unified.
  static Result<std::shared_ptr<ChunkedArray>> UnifyChunkedArray(
      const std::shared_ptr<ChunkedArray>& array,
      MemoryPool* pool = default_memory_pool());

  /// \brief Unify dictionaries accross the chunks of each table column
  ///
  /// The dictionaries in each table column will be unified, their indices
  /// accordingly transposed.
  ///
  /// Only dictionaries with a primitive value type are currently supported.
  /// However, dictionaries nested inside a more complex type are correctly unified.
  static Result<std::shared_ptr<Table>> UnifyTable(
      const Table& table, MemoryPool* pool = default_memory_pool());

  /// \brief Append dictionary to the internal memo
  virtual Status Unify(const Array& dictionary) = 0;

  /// \brief Append dictionary and compute transpose indices
  /// \param[in] dictionary the dictionary values to unify
  /// \param[out] out_transpose a Buffer containing computed transpose indices
  /// as int32_t values equal in length to the passed dictionary. The value in
  /// each slot corresponds to the new index value for each original index
  /// for a DictionaryArray with the old dictionary
  virtual Status Unify(const Array& dictionary,
                       std::shared_ptr<Buffer>* out_transpose) = 0;

  /// \brief Return a result DictionaryType with the smallest possible index
  /// type to accommodate the unified dictionary. The unifier cannot be used
  /// after this is called
  virtual Status GetResult(std::shared_ptr<DataType>* out_type,
                           std::shared_ptr<Array>* out_dict) = 0;

  /// \brief Return a unified dictionary with the given index type.  If
  /// the index type is not large enough then an invalid status will be returned.
  /// The unifier cannot be used after this is called
  virtual Status GetResultWithIndexType(const std::shared_ptr<DataType>& index_type,
                                        std::shared_ptr<Array>* out_dict) = 0;
};

}  // namespace arrow

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

// Array accessor classes for List, LargeList, FixedSizeList, Map, Struct, and
// Union

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "arrow/array/array_base.h"
#include "arrow/array/data.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type.h"
#include "arrow/type_fwd.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {

/// \addtogroup nested-arrays
///
/// @{

// ----------------------------------------------------------------------
// ListArray

template <typename TYPE>
class BaseListArray;

namespace internal {

// Private helper for ListArray::SetData.
// Unfortunately, trying to define BaseListArray::SetData outside of this header
// doesn't play well with MSVC.
template <typename TYPE>
void SetListData(BaseListArray<TYPE>* self, const std::shared_ptr<ArrayData>& data,
                 Type::type expected_type_id = TYPE::type_id);

}  // namespace internal

/// Base class for variable-sized list arrays, regardless of offset size.
template <typename TYPE>
class BaseListArray : public Array {
 public:
  using TypeClass = TYPE;
  using offset_type = typename TypeClass::offset_type;

  const TypeClass* list_type() const { return list_type_; }

  /// \brief Return array object containing the list's values
  ///
  /// Note that this buffer does not account for any slice offset or length.
  std::shared_ptr<Array> values() const { return values_; }

  /// Note that this buffer does not account for any slice offset or length.
  std::shared_ptr<Buffer> value_offsets() const { return data_->buffers[1]; }

  std::shared_ptr<DataType> value_type() const { return list_type_->value_type(); }

  /// Return pointer to raw value offsets accounting for any slice offset
  const offset_type* raw_value_offsets() const {
    return raw_value_offsets_ + data_->offset;
  }

  // The following functions will not perform boundschecking
  offset_type value_offset(int64_t i) const {
    return raw_value_offsets_[i + data_->offset];
  }
  offset_type value_length(int64_t i) const {
    i += data_->offset;
    return raw_value_offsets_[i + 1] - raw_value_offsets_[i];
  }
  std::shared_ptr<Array> value_slice(int64_t i) const {
    return values_->Slice(value_offset(i), value_length(i));
  }

 protected:
  friend void internal::SetListData<TYPE>(BaseListArray<TYPE>* self,
                                          const std::shared_ptr<ArrayData>& data,
                                          Type::type expected_type_id);

  const TypeClass* list_type_ = NULLPTR;
  std::shared_ptr<Array> values_;
  const offset_type* raw_value_offsets_ = NULLPTR;
};

/// Concrete Array class for list data
class ARROW_EXPORT ListArray : public BaseListArray<ListType> {
 public:
  explicit ListArray(std::shared_ptr<ArrayData> data);

  ListArray(std::shared_ptr<DataType> type, int64_t length,
            std::shared_ptr<Buffer> value_offsets, std::shared_ptr<Array> values,
            std::shared_ptr<Buffer> null_bitmap = NULLPTR,
            int64_t null_count = kUnknownNullCount, int64_t offset = 0);

  /// \brief Construct ListArray from array of offsets and child value array
  ///
  /// This function does the bare minimum of validation of the offsets and
  /// input types, and will allocate a new offsets array if necessary (i.e. if
  /// the offsets contain any nulls). If the offsets do not have nulls, they
  /// are assumed to be well-formed
  ///
  /// Offsets of an Array's null bitmap can be present or an explicit
  /// null_bitmap, but not both.
  ///
  /// \param[in] offsets Array containing n + 1 offsets encoding length and
  /// size. Must be of int32 type
  /// \param[in] values Array containing list values
  /// \param[in] pool MemoryPool in case new offsets array needs to be
  /// allocated because of null values
  /// \param[in] null_bitmap Optional validity bitmap
  /// \param[in] null_count Optional null count in null_bitmap
  static Result<std::shared_ptr<ListArray>> FromArrays(
      const Array& offsets, const Array& values, MemoryPool* pool = default_memory_pool(),
      std::shared_ptr<Buffer> null_bitmap = NULLPTR,
      int64_t null_count = kUnknownNullCount);

  static Result<std::shared_ptr<ListArray>> FromArrays(
      std::shared_ptr<DataType> type, const Array& offsets, const Array& values,
      MemoryPool* pool = default_memory_pool(),
      std::shared_ptr<Buffer> null_bitmap = NULLPTR,
      int64_t null_count = kUnknownNullCount);

  /// \brief Return an Array that is a concatenation of the lists in this array.
  ///
  /// Note that it's different from `values()` in that it takes into
  /// consideration of this array's offsets as well as null elements backed
  /// by non-empty lists (they are skipped, thus copying may be needed).
  Result<std::shared_ptr<Array>> Flatten(
      MemoryPool* memory_pool = default_memory_pool()) const;

  /// \brief Return list offsets as an Int32Array
  ///
  /// The returned array will not have a validity bitmap, so you cannot expect
  /// to pass it to ListArray::FromArrays() and get back the same list array
  /// if the original one has nulls.
  std::shared_ptr<Array> offsets() const;

 protected:
  // This constructor defers SetData to a derived array class
  ListArray() = default;

  void SetData(const std::shared_ptr<ArrayData>& data);
};

/// Concrete Array class for large list data (with 64-bit offsets)
class ARROW_EXPORT LargeListArray : public BaseListArray<LargeListType> {
 public:
  explicit LargeListArray(const std::shared_ptr<ArrayData>& data);

  LargeListArray(const std::shared_ptr<DataType>& type, int64_t length,
                 const std::shared_ptr<Buffer>& value_offsets,
                 const std::shared_ptr<Array>& values,
                 const std::shared_ptr<Buffer>& null_bitmap = NULLPTR,
                 int64_t null_count = kUnknownNullCount, int64_t offset = 0);

  /// \brief Construct LargeListArray from array of offsets and child value array
  ///
  /// This function does the bare minimum of validation of the offsets and
  /// input types, and will allocate a new offsets array if necessary (i.e. if
  /// the offsets contain any nulls). If the offsets do not have nulls, they
  /// are assumed to be well-formed
  ///
  /// \param[in] offsets Array containing n + 1 offsets encoding length and
  /// size. Must be of int64 type
  /// \param[in] values Array containing list values
  /// \param[in] pool MemoryPool in case new offsets array needs to be
  /// allocated because of null values
  /// \param[in] null_bitmap Optional validity bitmap
  /// \param[in] null_count Optional null count in null_bitmap
  static Result<std::shared_ptr<LargeListArray>> FromArrays(
      const Array& offsets, const Array& values, MemoryPool* pool = default_memory_pool(),
      std::shared_ptr<Buffer> null_bitmap = NULLPTR,
      int64_t null_count = kUnknownNullCount);

  static Result<std::shared_ptr<LargeListArray>> FromArrays(
      std::shared_ptr<DataType> type, const Array& offsets, const Array& values,
      MemoryPool* pool = default_memory_pool(),
      std::shared_ptr<Buffer> null_bitmap = NULLPTR,
      int64_t null_count = kUnknownNullCount);

  /// \brief Return an Array that is a concatenation of the lists in this array.
  ///
  /// Note that it's different from `values()` in that it takes into
  /// consideration of this array's offsets as well as null elements backed
  /// by non-empty lists (they are skipped, thus copying may be needed).
  Result<std::shared_ptr<Array>> Flatten(
      MemoryPool* memory_pool = default_memory_pool()) const;

  /// \brief Return list offsets as an Int64Array
  std::shared_ptr<Array> offsets() const;

 protected:
  void SetData(const std::shared_ptr<ArrayData>& data);
};

// ----------------------------------------------------------------------
// MapArray

/// Concrete Array class for map data
///
/// NB: "value" in this context refers to a pair of a key and the corresponding item
class ARROW_EXPORT MapArray : public ListArray {
 public:
  using TypeClass = MapType;

  explicit MapArray(const std::shared_ptr<ArrayData>& data);

  MapArray(const std::shared_ptr<DataType>& type, int64_t length,
           const std::shared_ptr<Buffer>& value_offsets,
           const std::shared_ptr<Array>& keys, const std::shared_ptr<Array>& items,
           const std::shared_ptr<Buffer>& null_bitmap = NULLPTR,
           int64_t null_count = kUnknownNullCount, int64_t offset = 0);

  MapArray(const std::shared_ptr<DataType>& type, int64_t length,
           const std::shared_ptr<Buffer>& value_offsets,
           const std::shared_ptr<Array>& values,
           const std::shared_ptr<Buffer>& null_bitmap = NULLPTR,
           int64_t null_count = kUnknownNullCount, int64_t offset = 0);

  /// \brief Construct MapArray from array of offsets and child key, item arrays
  ///
  /// This function does the bare minimum of validation of the offsets and
  /// input types, and will allocate a new offsets array if necessary (i.e. if
  /// the offsets contain any nulls). If the offsets do not have nulls, they
  /// are assumed to be well-formed
  ///
  /// \param[in] offsets Array containing n + 1 offsets encoding length and
  /// size. Must be of int32 type
  /// \param[in] keys Array containing key values
  /// \param[in] items Array containing item values
  /// \param[in] pool MemoryPool in case new offsets array needs to be
  /// allocated because of null values
  static Result<std::shared_ptr<Array>> FromArrays(
      const std::shared_ptr<Array>& offsets, const std::shared_ptr<Array>& keys,
      const std::shared_ptr<Array>& items, MemoryPool* pool = default_memory_pool());

  static Result<std::shared_ptr<Array>> FromArrays(
      std::shared_ptr<DataType> type, const std::shared_ptr<Array>& offsets,
      const std::shared_ptr<Array>& keys, const std::shared_ptr<Array>& items,
      MemoryPool* pool = default_memory_pool());

  const MapType* map_type() const { return map_type_; }

  /// \brief Return array object containing all map keys
  std::shared_ptr<Array> keys() const { return keys_; }

  /// \brief Return array object containing all mapped items
  std::shared_ptr<Array> items() const { return items_; }

  /// Validate child data before constructing the actual MapArray.
  static Status ValidateChildData(
      const std::vector<std::shared_ptr<ArrayData>>& child_data);

 protected:
  void SetData(const std::shared_ptr<ArrayData>& data);

  static Result<std::shared_ptr<Array>> FromArraysInternal(
      std::shared_ptr<DataType> type, const std::shared_ptr<Array>& offsets,
      const std::shared_ptr<Array>& keys, const std::shared_ptr<Array>& items,
      MemoryPool* pool);

 private:
  const MapType* map_type_;
  std::shared_ptr<Array> keys_, items_;
};

// ----------------------------------------------------------------------
// FixedSizeListArray

/// Concrete Array class for fixed size list data
class ARROW_EXPORT FixedSizeListArray : public Array {
 public:
  using TypeClass = FixedSizeListType;
  using offset_type = TypeClass::offset_type;

  explicit FixedSizeListArray(const std::shared_ptr<ArrayData>& data);

  FixedSizeListArray(const std::shared_ptr<DataType>& type, int64_t length,
                     const std::shared_ptr<Array>& values,
                     const std::shared_ptr<Buffer>& null_bitmap = NULLPTR,
                     int64_t null_count = kUnknownNullCount, int64_t offset = 0);

  const FixedSizeListType* list_type() const;

  /// \brief Return array object containing the list's values
  std::shared_ptr<Array> values() const;

  std::shared_ptr<DataType> value_type() const;

  // The following functions will not perform boundschecking
  int64_t value_offset(int64_t i) const {
    i += data_->offset;
    return list_size_ * i;
  }
  int32_t value_length(int64_t i = 0) const {
    ARROW_UNUSED(i);
    return list_size_;
  }
  std::shared_ptr<Array> value_slice(int64_t i) const {
    return values_->Slice(value_offset(i), value_length(i));
  }

  /// \brief Return an Array that is a concatenation of the lists in this array.
  ///
  /// Note that it's different from `values()` in that it takes into
  /// consideration null elements (they are skipped, thus copying may be needed).
  Result<std::shared_ptr<Array>> Flatten(
      MemoryPool* memory_pool = default_memory_pool()) const;

  /// \brief Construct FixedSizeListArray from child value array and value_length
  ///
  /// \param[in] values Array containing list values
  /// \param[in] list_size The fixed length of each list
  /// \return Will have length equal to values.length() / list_size
  static Result<std::shared_ptr<Array>> FromArrays(const std::shared_ptr<Array>& values,
                                                   int32_t list_size);

  /// \brief Construct FixedSizeListArray from child value array and type
  ///
  /// \param[in] values Array containing list values
  /// \param[in] type The fixed sized list type
  /// \return Will have length equal to values.length() / type.list_size()
  static Result<std::shared_ptr<Array>> FromArrays(const std::shared_ptr<Array>& values,
                                                   std::shared_ptr<DataType> type);

 protected:
  void SetData(const std::shared_ptr<ArrayData>& data);
  int32_t list_size_;

 private:
  std::shared_ptr<Array> values_;
};

// ----------------------------------------------------------------------
// Struct

/// Concrete Array class for struct data
class ARROW_EXPORT StructArray : public Array {
 public:
  using TypeClass = StructType;

  explicit StructArray(const std::shared_ptr<ArrayData>& data);

  StructArray(const std::shared_ptr<DataType>& type, int64_t length,
              const std::vector<std::shared_ptr<Array>>& children,
              std::shared_ptr<Buffer> null_bitmap = NULLPTR,
              int64_t null_count = kUnknownNullCount, int64_t offset = 0);

  /// \brief Return a StructArray from child arrays and field names.
  ///
  /// The length and data type are automatically inferred from the arguments.
  /// There should be at least one child array.
  static Result<std::shared_ptr<StructArray>> Make(
      const ArrayVector& children, const std::vector<std::string>& field_names,
      std::shared_ptr<Buffer> null_bitmap = NULLPTR,
      int64_t null_count = kUnknownNullCount, int64_t offset = 0);

  /// \brief Return a StructArray from child arrays and fields.
  ///
  /// The length is automatically inferred from the arguments.
  /// There should be at least one child array.  This method does not
  /// check that field types and child array types are consistent.
  static Result<std::shared_ptr<StructArray>> Make(
      const ArrayVector& children, const FieldVector& fields,
      std::shared_ptr<Buffer> null_bitmap = NULLPTR,
      int64_t null_count = kUnknownNullCount, int64_t offset = 0);

  const StructType* struct_type() const;

  // Return a shared pointer in case the requestor desires to share ownership
  // with this array.  The returned array has its offset, length and null
  // count adjusted.
  const std::shared_ptr<Array>& field(int pos) const;

  const ArrayVector& fields() const;

  /// Returns null if name not found
  std::shared_ptr<Array> GetFieldByName(const std::string& name) const;

  /// \brief Flatten this array as a vector of arrays, one for each field
  ///
  /// \param[in] pool The pool to allocate null bitmaps from, if necessary
  Result<ArrayVector> Flatten(MemoryPool* pool = default_memory_pool()) const;

  /// \brief Get one of the child arrays, combining its null bitmap
  /// with the parent struct array's bitmap.
  ///
  /// \param[in] index Which child array to get
  /// \param[in] pool The pool to allocate null bitmaps from, if necessary
  Result<std::shared_ptr<Array>> GetFlattenedField(
      int index, MemoryPool* pool = default_memory_pool()) const;

 private:
  // For caching boxed child data
  // XXX This is not handled in a thread-safe manner.
  mutable ArrayVector boxed_fields_;
};

// ----------------------------------------------------------------------
// Union

/// Base class for SparseUnionArray and DenseUnionArray
class ARROW_EXPORT UnionArray : public Array {
 public:
  using type_code_t = int8_t;

  /// Note that this buffer does not account for any slice offset
  std::shared_ptr<Buffer> type_codes() const { return data_->buffers[1]; }

  const type_code_t* raw_type_codes() const { return raw_type_codes_ + data_->offset; }

  /// The logical type code of the value at index.
  type_code_t type_code(int64_t i) const { return raw_type_codes_[i + data_->offset]; }

  /// The physical child id containing value at index.
  int child_id(int64_t i) const {
    return union_type_->child_ids()[raw_type_codes_[i + data_->offset]];
  }

  const UnionType* union_type() const { return union_type_; }

  UnionMode::type mode() const { return union_type_->mode(); }

  /// \brief Return the given field as an individual array.
  ///
  /// For sparse unions, the returned array has its offset, length and null
  /// count adjusted.
  std::shared_ptr<Array> field(int pos) const;

 protected:
  void SetData(std::shared_ptr<ArrayData> data);

  const type_code_t* raw_type_codes_;
  const UnionType* union_type_;

  // For caching boxed child data
  mutable std::vector<std::shared_ptr<Array>> boxed_fields_;
};

/// Concrete Array class for sparse union data
class ARROW_EXPORT SparseUnionArray : public UnionArray {
 public:
  using TypeClass = SparseUnionType;

  explicit SparseUnionArray(std::shared_ptr<ArrayData> data);

  SparseUnionArray(std::shared_ptr<DataType> type, int64_t length, ArrayVector children,
                   std::shared_ptr<Buffer> type_ids, int64_t offset = 0);

  /// \brief Construct SparseUnionArray from type_ids and children
  ///
  /// This function does the bare minimum of validation of the input types.
  ///
  /// \param[in] type_ids An array of logical type ids for the union type
  /// \param[in] children Vector of children Arrays containing the data for each type.
  /// \param[in] type_codes Vector of type codes.
  static Result<std::shared_ptr<Array>> Make(const Array& type_ids, ArrayVector children,
                                             std::vector<type_code_t> type_codes) {
    return Make(std::move(type_ids), std::move(children), std::vector<std::string>{},
                std::move(type_codes));
  }

  /// \brief Construct SparseUnionArray with custom field names from type_ids and children
  ///
  /// This function does the bare minimum of validation of the input types.
  ///
  /// \param[in] type_ids An array of logical type ids for the union type
  /// \param[in] children Vector of children Arrays containing the data for each type.
  /// \param[in] field_names Vector of strings containing the name of each field.
  /// \param[in] type_codes Vector of type codes.
  static Result<std::shared_ptr<Array>> Make(const Array& type_ids, ArrayVector children,
                                             std::vector<std::string> field_names = {},
                                             std::vector<type_code_t> type_codes = {});

  const SparseUnionType* union_type() const {
    return internal::checked_cast<const SparseUnionType*>(union_type_);
  }

  /// \brief Get one of the child arrays, adjusting its null bitmap
  /// where the union array type code does not match.
  ///
  /// \param[in] index Which child array to get (i.e. the physical index, not the type
  /// code) \param[in] pool The pool to allocate null bitmaps from, if necessary
  Result<std::shared_ptr<Array>> GetFlattenedField(
      int index, MemoryPool* pool = default_memory_pool()) const;

 protected:
  void SetData(std::shared_ptr<ArrayData> data);
};

/// \brief Concrete Array class for dense union data
///
/// Note that union types do not have a validity bitmap
class ARROW_EXPORT DenseUnionArray : public UnionArray {
 public:
  using TypeClass = DenseUnionType;

  explicit DenseUnionArray(const std::shared_ptr<ArrayData>& data);

  DenseUnionArray(std::shared_ptr<DataType> type, int64_t length, ArrayVector children,
                  std::shared_ptr<Buffer> type_ids,
                  std::shared_ptr<Buffer> value_offsets = NULLPTR, int64_t offset = 0);

  /// \brief Construct DenseUnionArray from type_ids, value_offsets, and children
  ///
  /// This function does the bare minimum of validation of the offsets and
  /// input types.
  ///
  /// \param[in] type_ids An array of logical type ids for the union type
  /// \param[in] value_offsets An array of signed int32 values indicating the
  /// relative offset into the respective child array for the type in a given slot.
  /// The respective offsets for each child value array must be in order / increasing.
  /// \param[in] children Vector of children Arrays containing the data for each type.
  /// \param[in] type_codes Vector of type codes.
  static Result<std::shared_ptr<Array>> Make(const Array& type_ids,
                                             const Array& value_offsets,
                                             ArrayVector children,
                                             std::vector<type_code_t> type_codes) {
    return Make(type_ids, value_offsets, std::move(children), std::vector<std::string>{},
                std::move(type_codes));
  }

  /// \brief Construct DenseUnionArray with custom field names from type_ids,
  /// value_offsets, and children
  ///
  /// This function does the bare minimum of validation of the offsets and
  /// input types.
  ///
  /// \param[in] type_ids An array of logical type ids for the union type
  /// \param[in] value_offsets An array of signed int32 values indicating the
  /// relative offset into the respective child array for the type in a given slot.
  /// The respective offsets for each child value array must be in order / increasing.
  /// \param[in] children Vector of children Arrays containing the data for each type.
  /// \param[in] field_names Vector of strings containing the name of each field.
  /// \param[in] type_codes Vector of type codes.
  static Result<std::shared_ptr<Array>> Make(const Array& type_ids,
                                             const Array& value_offsets,
                                             ArrayVector children,
                                             std::vector<std::string> field_names = {},
                                             std::vector<type_code_t> type_codes = {});

  const DenseUnionType* union_type() const {
    return internal::checked_cast<const DenseUnionType*>(union_type_);
  }

  /// Note that this buffer does not account for any slice offset
  std::shared_ptr<Buffer> value_offsets() const { return data_->buffers[2]; }

  int32_t value_offset(int64_t i) const { return raw_value_offsets_[i + data_->offset]; }

  const int32_t* raw_value_offsets() const { return raw_value_offsets_ + data_->offset; }

 protected:
  const int32_t* raw_value_offsets_;

  void SetData(const std::shared_ptr<ArrayData>& data);
};

/// @}

}  // namespace arrow

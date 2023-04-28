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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <type_traits>

#include "arrow/array/array_base.h"
#include "arrow/array/array_binary.h"
#include "arrow/array/builder_adaptive.h"   // IWYU pragma: export
#include "arrow/array/builder_base.h"       // IWYU pragma: export
#include "arrow/array/builder_primitive.h"  // IWYU pragma: export
#include "arrow/array/data.h"
#include "arrow/array/util.h"
#include "arrow/scalar.h"
#include "arrow/status.h"
#include "arrow/type.h"
#include "arrow/type_traits.h"
#include "arrow/util/bit_block_counter.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/decimal.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {

// ----------------------------------------------------------------------
// Dictionary builder

namespace internal {

template <typename T, typename Enable = void>
struct DictionaryValue {
  using type = typename T::c_type;
  using PhysicalType = T;
};

template <typename T>
struct DictionaryValue<T, enable_if_base_binary<T>> {
  using type = std::string_view;
  using PhysicalType =
      typename std::conditional<std::is_same<typename T::offset_type, int32_t>::value,
                                BinaryType, LargeBinaryType>::type;
};

template <typename T>
struct DictionaryValue<T, enable_if_fixed_size_binary<T>> {
  using type = std::string_view;
  using PhysicalType = BinaryType;
};

class ARROW_EXPORT DictionaryMemoTable {
 public:
  DictionaryMemoTable(MemoryPool* pool, const std::shared_ptr<DataType>& type);
  DictionaryMemoTable(MemoryPool* pool, const std::shared_ptr<Array>& dictionary);
  ~DictionaryMemoTable();

  Status GetArrayData(int64_t start_offset, std::shared_ptr<ArrayData>* out);

  /// \brief Insert new memo values
  Status InsertValues(const Array& values);

  int32_t size() const;

  template <typename T>
  Status GetOrInsert(typename DictionaryValue<T>::type value, int32_t* out) {
    // We want to keep the DictionaryMemoTable implementation private, also we can't
    // use extern template classes because of compiler issues (MinGW?).  Instead,
    // we expose explicit function overrides for each supported physical type.
    const typename DictionaryValue<T>::PhysicalType* physical_type = NULLPTR;
    return GetOrInsert(physical_type, value, out);
  }

 private:
  Status GetOrInsert(const BooleanType*, bool value, int32_t* out);
  Status GetOrInsert(const Int8Type*, int8_t value, int32_t* out);
  Status GetOrInsert(const Int16Type*, int16_t value, int32_t* out);
  Status GetOrInsert(const Int32Type*, int32_t value, int32_t* out);
  Status GetOrInsert(const Int64Type*, int64_t value, int32_t* out);
  Status GetOrInsert(const UInt8Type*, uint8_t value, int32_t* out);
  Status GetOrInsert(const UInt16Type*, uint16_t value, int32_t* out);
  Status GetOrInsert(const UInt32Type*, uint32_t value, int32_t* out);
  Status GetOrInsert(const UInt64Type*, uint64_t value, int32_t* out);
  Status GetOrInsert(const DurationType*, int64_t value, int32_t* out);
  Status GetOrInsert(const TimestampType*, int64_t value, int32_t* out);
  Status GetOrInsert(const Date32Type*, int32_t value, int32_t* out);
  Status GetOrInsert(const Date64Type*, int64_t value, int32_t* out);
  Status GetOrInsert(const Time32Type*, int32_t value, int32_t* out);
  Status GetOrInsert(const Time64Type*, int64_t value, int32_t* out);
  Status GetOrInsert(const MonthDayNanoIntervalType*,
                     MonthDayNanoIntervalType::MonthDayNanos value, int32_t* out);
  Status GetOrInsert(const DayTimeIntervalType*,
                     DayTimeIntervalType::DayMilliseconds value, int32_t* out);
  Status GetOrInsert(const MonthIntervalType*, int32_t value, int32_t* out);
  Status GetOrInsert(const FloatType*, float value, int32_t* out);
  Status GetOrInsert(const DoubleType*, double value, int32_t* out);

  Status GetOrInsert(const BinaryType*, std::string_view value, int32_t* out);
  Status GetOrInsert(const LargeBinaryType*, std::string_view value, int32_t* out);

  class DictionaryMemoTableImpl;
  std::unique_ptr<DictionaryMemoTableImpl> impl_;
};

}  // namespace internal

/// \addtogroup dictionary-builders
///
/// @{

namespace internal {

/// \brief Array builder for created encoded DictionaryArray from
/// dense array
///
/// Unlike other builders, dictionary builder does not completely
/// reset the state on Finish calls.
template <typename BuilderType, typename T>
class DictionaryBuilderBase : public ArrayBuilder {
 public:
  using TypeClass = DictionaryType;
  using Value = typename DictionaryValue<T>::type;

  // WARNING: the type given below is the value type, not the DictionaryType.
  // The DictionaryType is instantiated on the Finish() call.
  template <typename B = BuilderType, typename T1 = T>
  DictionaryBuilderBase(uint8_t start_int_size,
                        enable_if_t<std::is_base_of<AdaptiveIntBuilderBase, B>::value &&
                                        !is_fixed_size_binary_type<T1>::value,
                                    const std::shared_ptr<DataType>&>
                            value_type,
                        MemoryPool* pool = default_memory_pool(),
                        int64_t alignment = kDefaultBufferAlignment)
      : ArrayBuilder(pool, alignment),
        memo_table_(new internal::DictionaryMemoTable(pool, value_type)),
        delta_offset_(0),
        byte_width_(-1),
        indices_builder_(start_int_size, pool, alignment),
        value_type_(value_type) {}

  template <typename T1 = T>
  explicit DictionaryBuilderBase(
      enable_if_t<!is_fixed_size_binary_type<T1>::value, const std::shared_ptr<DataType>&>
          value_type,
      MemoryPool* pool = default_memory_pool(),
      int64_t alignment = kDefaultBufferAlignment)
      : ArrayBuilder(pool, alignment),
        memo_table_(new internal::DictionaryMemoTable(pool, value_type)),
        delta_offset_(0),
        byte_width_(-1),
        indices_builder_(pool, alignment),
        value_type_(value_type) {}

  template <typename T1 = T>
  explicit DictionaryBuilderBase(
      const std::shared_ptr<DataType>& index_type,
      enable_if_t<!is_fixed_size_binary_type<T1>::value, const std::shared_ptr<DataType>&>
          value_type,
      MemoryPool* pool = default_memory_pool(),
      int64_t alignment = kDefaultBufferAlignment)
      : ArrayBuilder(pool, alignment),
        memo_table_(new internal::DictionaryMemoTable(pool, value_type)),
        delta_offset_(0),
        byte_width_(-1),
        indices_builder_(index_type, pool, alignment),
        value_type_(value_type) {}

  template <typename B = BuilderType, typename T1 = T>
  DictionaryBuilderBase(uint8_t start_int_size,
                        enable_if_t<std::is_base_of<AdaptiveIntBuilderBase, B>::value &&
                                        is_fixed_size_binary_type<T1>::value,
                                    const std::shared_ptr<DataType>&>
                            value_type,
                        MemoryPool* pool = default_memory_pool(),
                        int64_t alignment = kDefaultBufferAlignment)
      : ArrayBuilder(pool, alignment),
        memo_table_(new internal::DictionaryMemoTable(pool, value_type)),
        delta_offset_(0),
        byte_width_(static_cast<const T1&>(*value_type).byte_width()),
        indices_builder_(start_int_size, pool, alignment),
        value_type_(value_type) {}

  template <typename T1 = T>
  explicit DictionaryBuilderBase(
      enable_if_fixed_size_binary<T1, const std::shared_ptr<DataType>&> value_type,
      MemoryPool* pool = default_memory_pool(),
      int64_t alignment = kDefaultBufferAlignment)
      : ArrayBuilder(pool, alignment),
        memo_table_(new internal::DictionaryMemoTable(pool, value_type)),
        delta_offset_(0),
        byte_width_(static_cast<const T1&>(*value_type).byte_width()),
        indices_builder_(pool, alignment),
        value_type_(value_type) {}

  template <typename T1 = T>
  explicit DictionaryBuilderBase(
      const std::shared_ptr<DataType>& index_type,
      enable_if_fixed_size_binary<T1, const std::shared_ptr<DataType>&> value_type,
      MemoryPool* pool = default_memory_pool(),
      int64_t alignment = kDefaultBufferAlignment)
      : ArrayBuilder(pool, alignment),
        memo_table_(new internal::DictionaryMemoTable(pool, value_type)),
        delta_offset_(0),
        byte_width_(static_cast<const T1&>(*value_type).byte_width()),
        indices_builder_(index_type, pool, alignment),
        value_type_(value_type) {}

  template <typename T1 = T>
  explicit DictionaryBuilderBase(
      enable_if_parameter_free<T1, MemoryPool*> pool = default_memory_pool())
      : DictionaryBuilderBase<BuilderType, T1>(TypeTraits<T1>::type_singleton(), pool) {}

  // This constructor doesn't check for errors. Use InsertMemoValues instead.
  explicit DictionaryBuilderBase(const std::shared_ptr<Array>& dictionary,
                                 MemoryPool* pool = default_memory_pool(),
                                 int64_t alignment = kDefaultBufferAlignment)
      : ArrayBuilder(pool, alignment),
        memo_table_(new internal::DictionaryMemoTable(pool, dictionary)),
        delta_offset_(0),
        byte_width_(-1),
        indices_builder_(pool, alignment),
        value_type_(dictionary->type()) {}

  ~DictionaryBuilderBase() override = default;

  /// \brief The current number of entries in the dictionary
  int64_t dictionary_length() const { return memo_table_->size(); }

  /// \brief The value byte width (for FixedSizeBinaryType)
  template <typename T1 = T>
  enable_if_fixed_size_binary<T1, int32_t> byte_width() const {
    return byte_width_;
  }

  /// \brief Append a scalar value
  Status Append(Value value) {
    ARROW_RETURN_NOT_OK(Reserve(1));

    int32_t memo_index;
    ARROW_RETURN_NOT_OK(memo_table_->GetOrInsert<T>(value, &memo_index));
    ARROW_RETURN_NOT_OK(indices_builder_.Append(memo_index));
    length_ += 1;

    return Status::OK();
  }

  /// \brief Append a fixed-width string (only for FixedSizeBinaryType)
  template <typename T1 = T>
  enable_if_fixed_size_binary<T1, Status> Append(const uint8_t* value) {
    return Append(std::string_view(reinterpret_cast<const char*>(value), byte_width_));
  }

  /// \brief Append a fixed-width string (only for FixedSizeBinaryType)
  template <typename T1 = T>
  enable_if_fixed_size_binary<T1, Status> Append(const char* value) {
    return Append(std::string_view(value, byte_width_));
  }

  /// \brief Append a string (only for binary types)
  template <typename T1 = T>
  enable_if_binary_like<T1, Status> Append(const uint8_t* value, int32_t length) {
    return Append(reinterpret_cast<const char*>(value), length);
  }

  /// \brief Append a string (only for binary types)
  template <typename T1 = T>
  enable_if_binary_like<T1, Status> Append(const char* value, int32_t length) {
    return Append(std::string_view(value, length));
  }

  /// \brief Append a string (only for string types)
  template <typename T1 = T>
  enable_if_string_like<T1, Status> Append(const char* value, int32_t length) {
    return Append(std::string_view(value, length));
  }

  /// \brief Append a decimal (only for Decimal128Type)
  template <typename T1 = T>
  enable_if_decimal128<T1, Status> Append(const Decimal128& value) {
    uint8_t data[16];
    value.ToBytes(data);
    return Append(data, 16);
  }

  /// \brief Append a decimal (only for Decimal128Type)
  template <typename T1 = T>
  enable_if_decimal256<T1, Status> Append(const Decimal256& value) {
    uint8_t data[32];
    value.ToBytes(data);
    return Append(data, 32);
  }

  /// \brief Append a scalar null value
  Status AppendNull() final {
    length_ += 1;
    null_count_ += 1;

    return indices_builder_.AppendNull();
  }

  Status AppendNulls(int64_t length) final {
    length_ += length;
    null_count_ += length;

    return indices_builder_.AppendNulls(length);
  }

  Status AppendEmptyValue() final {
    length_ += 1;

    return indices_builder_.AppendEmptyValue();
  }

  Status AppendEmptyValues(int64_t length) final {
    length_ += length;

    return indices_builder_.AppendEmptyValues(length);
  }

  Status AppendScalar(const Scalar& scalar, int64_t n_repeats) override {
    if (!scalar.is_valid) return AppendNulls(n_repeats);

    const auto& dict_ty = internal::checked_cast<const DictionaryType&>(*scalar.type);
    const DictionaryScalar& dict_scalar =
        internal::checked_cast<const DictionaryScalar&>(scalar);
    const auto& dict = internal::checked_cast<const typename TypeTraits<T>::ArrayType&>(
        *dict_scalar.value.dictionary);
    ARROW_RETURN_NOT_OK(Reserve(n_repeats));
    switch (dict_ty.index_type()->id()) {
      case Type::UINT8:
        return AppendScalarImpl<UInt8Type>(dict, *dict_scalar.value.index, n_repeats);
      case Type::INT8:
        return AppendScalarImpl<Int8Type>(dict, *dict_scalar.value.index, n_repeats);
      case Type::UINT16:
        return AppendScalarImpl<UInt16Type>(dict, *dict_scalar.value.index, n_repeats);
      case Type::INT16:
        return AppendScalarImpl<Int16Type>(dict, *dict_scalar.value.index, n_repeats);
      case Type::UINT32:
        return AppendScalarImpl<UInt32Type>(dict, *dict_scalar.value.index, n_repeats);
      case Type::INT32:
        return AppendScalarImpl<Int32Type>(dict, *dict_scalar.value.index, n_repeats);
      case Type::UINT64:
        return AppendScalarImpl<UInt64Type>(dict, *dict_scalar.value.index, n_repeats);
      case Type::INT64:
        return AppendScalarImpl<Int64Type>(dict, *dict_scalar.value.index, n_repeats);
      default:
        return Status::TypeError("Invalid index type: ", dict_ty);
    }
    return Status::OK();
  }

  Status AppendScalars(const ScalarVector& scalars) override {
    for (const auto& scalar : scalars) {
      ARROW_RETURN_NOT_OK(AppendScalar(*scalar, /*n_repeats=*/1));
    }
    return Status::OK();
  }

  Status AppendArraySlice(const ArraySpan& array, int64_t offset, int64_t length) final {
    // Visit the indices and insert the unpacked values.
    const auto& dict_ty = internal::checked_cast<const DictionaryType&>(*array.type);
    // See if possible to avoid using ToArrayData here
    const typename TypeTraits<T>::ArrayType dict(array.dictionary().ToArrayData());
    ARROW_RETURN_NOT_OK(Reserve(length));
    switch (dict_ty.index_type()->id()) {
      case Type::UINT8:
        return AppendArraySliceImpl<uint8_t>(dict, array, offset, length);
      case Type::INT8:
        return AppendArraySliceImpl<int8_t>(dict, array, offset, length);
      case Type::UINT16:
        return AppendArraySliceImpl<uint16_t>(dict, array, offset, length);
      case Type::INT16:
        return AppendArraySliceImpl<int16_t>(dict, array, offset, length);
      case Type::UINT32:
        return AppendArraySliceImpl<uint32_t>(dict, array, offset, length);
      case Type::INT32:
        return AppendArraySliceImpl<int32_t>(dict, array, offset, length);
      case Type::UINT64:
        return AppendArraySliceImpl<uint64_t>(dict, array, offset, length);
      case Type::INT64:
        return AppendArraySliceImpl<int64_t>(dict, array, offset, length);
      default:
        return Status::TypeError("Invalid index type: ", dict_ty);
    }
    return Status::OK();
  }

  /// \brief Insert values into the dictionary's memo, but do not append any
  /// indices. Can be used to initialize a new builder with known dictionary
  /// values
  /// \param[in] values dictionary values to add to memo. Type must match
  /// builder type
  Status InsertMemoValues(const Array& values) {
    return memo_table_->InsertValues(values);
  }

  /// \brief Append a whole dense array to the builder
  template <typename T1 = T>
  enable_if_t<!is_fixed_size_binary_type<T1>::value, Status> AppendArray(
      const Array& array) {
    using ArrayType = typename TypeTraits<T>::ArrayType;

#ifndef NDEBUG
    ARROW_RETURN_NOT_OK(ArrayBuilder::CheckArrayType(
        value_type_, array, "Wrong value type of array to be appended"));
#endif

    const auto& concrete_array = static_cast<const ArrayType&>(array);
    for (int64_t i = 0; i < array.length(); i++) {
      if (array.IsNull(i)) {
        ARROW_RETURN_NOT_OK(AppendNull());
      } else {
        ARROW_RETURN_NOT_OK(Append(concrete_array.GetView(i)));
      }
    }
    return Status::OK();
  }

  template <typename T1 = T>
  enable_if_fixed_size_binary<T1, Status> AppendArray(const Array& array) {
#ifndef NDEBUG
    ARROW_RETURN_NOT_OK(ArrayBuilder::CheckArrayType(
        value_type_, array, "Wrong value type of array to be appended"));
#endif

    const auto& concrete_array = static_cast<const FixedSizeBinaryArray&>(array);
    for (int64_t i = 0; i < array.length(); i++) {
      if (array.IsNull(i)) {
        ARROW_RETURN_NOT_OK(AppendNull());
      } else {
        ARROW_RETURN_NOT_OK(Append(concrete_array.GetValue(i)));
      }
    }
    return Status::OK();
  }

  void Reset() override {
    // Perform a partial reset. Call ResetFull to also reset the accumulated
    // dictionary values
    ArrayBuilder::Reset();
    indices_builder_.Reset();
  }

  /// \brief Reset and also clear accumulated dictionary values in memo table
  void ResetFull() {
    Reset();
    memo_table_.reset(new internal::DictionaryMemoTable(pool_, value_type_));
  }

  Status Resize(int64_t capacity) override {
    ARROW_RETURN_NOT_OK(CheckCapacity(capacity));
    capacity = std::max(capacity, kMinBuilderCapacity);
    ARROW_RETURN_NOT_OK(indices_builder_.Resize(capacity));
    capacity_ = indices_builder_.capacity();
    return Status::OK();
  }

  /// \brief Return dictionary indices and a delta dictionary since the last
  /// time that Finish or FinishDelta were called, and reset state of builder
  /// (except the memo table)
  Status FinishDelta(std::shared_ptr<Array>* out_indices,
                     std::shared_ptr<Array>* out_delta) {
    std::shared_ptr<ArrayData> indices_data;
    std::shared_ptr<ArrayData> delta_data;
    ARROW_RETURN_NOT_OK(FinishWithDictOffset(delta_offset_, &indices_data, &delta_data));
    *out_indices = MakeArray(indices_data);
    *out_delta = MakeArray(delta_data);
    return Status::OK();
  }

  /// \cond FALSE
  using ArrayBuilder::Finish;
  /// \endcond

  Status Finish(std::shared_ptr<DictionaryArray>* out) { return FinishTyped(out); }

  std::shared_ptr<DataType> type() const override {
    return ::arrow::dictionary(indices_builder_.type(), value_type_);
  }

 protected:
  template <typename c_type>
  Status AppendArraySliceImpl(const typename TypeTraits<T>::ArrayType& dict,
                              const ArraySpan& array, int64_t offset, int64_t length) {
    const c_type* values = array.GetValues<c_type>(1) + offset;
    return VisitBitBlocks(
        array.buffers[0].data, array.offset + offset, length,
        [&](const int64_t position) {
          const int64_t index = static_cast<int64_t>(values[position]);
          if (dict.IsValid(index)) {
            return Append(dict.GetView(index));
          }
          return AppendNull();
        },
        [&]() { return AppendNull(); });
  }

  template <typename IndexType>
  Status AppendScalarImpl(const typename TypeTraits<T>::ArrayType& dict,
                          const Scalar& index_scalar, int64_t n_repeats) {
    using ScalarType = typename TypeTraits<IndexType>::ScalarType;
    const auto index = internal::checked_cast<const ScalarType&>(index_scalar).value;
    if (index_scalar.is_valid && dict.IsValid(index)) {
      const auto& value = dict.GetView(index);
      for (int64_t i = 0; i < n_repeats; i++) {
        ARROW_RETURN_NOT_OK(Append(value));
      }
      return Status::OK();
    }
    return AppendNulls(n_repeats);
  }

  Status FinishInternal(std::shared_ptr<ArrayData>* out) override {
    std::shared_ptr<ArrayData> dictionary;
    ARROW_RETURN_NOT_OK(FinishWithDictOffset(/*offset=*/0, out, &dictionary));

    // Set type of array data to the right dictionary type
    (*out)->type = type();
    (*out)->dictionary = dictionary;
    return Status::OK();
  }

  Status FinishWithDictOffset(int64_t dict_offset,
                              std::shared_ptr<ArrayData>* out_indices,
                              std::shared_ptr<ArrayData>* out_dictionary) {
    // Finalize indices array
    ARROW_RETURN_NOT_OK(indices_builder_.FinishInternal(out_indices));

    // Generate dictionary array from hash table contents
    ARROW_RETURN_NOT_OK(memo_table_->GetArrayData(dict_offset, out_dictionary));
    delta_offset_ = memo_table_->size();

    // Update internals for further uses of this DictionaryBuilder
    ArrayBuilder::Reset();
    return Status::OK();
  }

  std::unique_ptr<DictionaryMemoTable> memo_table_;

  // The size of the dictionary memo at last invocation of Finish, to use in
  // FinishDelta for computing dictionary deltas
  int32_t delta_offset_;

  // Only used for FixedSizeBinaryType
  int32_t byte_width_;

  BuilderType indices_builder_;
  std::shared_ptr<DataType> value_type_;
};

template <typename BuilderType>
class DictionaryBuilderBase<BuilderType, NullType> : public ArrayBuilder {
 public:
  template <typename B = BuilderType>
  DictionaryBuilderBase(
      enable_if_t<std::is_base_of<AdaptiveIntBuilderBase, B>::value, uint8_t>
          start_int_size,
      const std::shared_ptr<DataType>& value_type,
      MemoryPool* pool = default_memory_pool())
      : ArrayBuilder(pool), indices_builder_(start_int_size, pool) {}

  explicit DictionaryBuilderBase(const std::shared_ptr<DataType>& value_type,
                                 MemoryPool* pool = default_memory_pool())
      : ArrayBuilder(pool), indices_builder_(pool) {}

  explicit DictionaryBuilderBase(const std::shared_ptr<DataType>& index_type,
                                 const std::shared_ptr<DataType>& value_type,
                                 MemoryPool* pool = default_memory_pool())
      : ArrayBuilder(pool), indices_builder_(index_type, pool) {}

  template <typename B = BuilderType>
  explicit DictionaryBuilderBase(
      enable_if_t<std::is_base_of<AdaptiveIntBuilderBase, B>::value, uint8_t>
          start_int_size,
      MemoryPool* pool = default_memory_pool())
      : ArrayBuilder(pool), indices_builder_(start_int_size, pool) {}

  explicit DictionaryBuilderBase(MemoryPool* pool = default_memory_pool())
      : ArrayBuilder(pool), indices_builder_(pool) {}

  explicit DictionaryBuilderBase(const std::shared_ptr<Array>& dictionary,
                                 MemoryPool* pool = default_memory_pool())
      : ArrayBuilder(pool), indices_builder_(pool) {}

  /// \brief Append a scalar null value
  Status AppendNull() final {
    length_ += 1;
    null_count_ += 1;

    return indices_builder_.AppendNull();
  }

  Status AppendNulls(int64_t length) final {
    length_ += length;
    null_count_ += length;

    return indices_builder_.AppendNulls(length);
  }

  Status AppendEmptyValue() final {
    length_ += 1;

    return indices_builder_.AppendEmptyValue();
  }

  Status AppendEmptyValues(int64_t length) final {
    length_ += length;

    return indices_builder_.AppendEmptyValues(length);
  }

  /// \brief Append a whole dense array to the builder
  Status AppendArray(const Array& array) {
#ifndef NDEBUG
    ARROW_RETURN_NOT_OK(ArrayBuilder::CheckArrayType(
        Type::NA, array, "Wrong value type of array to be appended"));
#endif
    for (int64_t i = 0; i < array.length(); i++) {
      ARROW_RETURN_NOT_OK(AppendNull());
    }
    return Status::OK();
  }

  Status Resize(int64_t capacity) override {
    ARROW_RETURN_NOT_OK(CheckCapacity(capacity));
    capacity = std::max(capacity, kMinBuilderCapacity);

    ARROW_RETURN_NOT_OK(indices_builder_.Resize(capacity));
    capacity_ = indices_builder_.capacity();
    return Status::OK();
  }

  Status FinishInternal(std::shared_ptr<ArrayData>* out) override {
    ARROW_RETURN_NOT_OK(indices_builder_.FinishInternal(out));
    (*out)->type = dictionary((*out)->type, null());
    (*out)->dictionary = NullArray(0).data();
    return Status::OK();
  }

  /// \cond FALSE
  using ArrayBuilder::Finish;
  /// \endcond

  Status Finish(std::shared_ptr<DictionaryArray>* out) { return FinishTyped(out); }

  std::shared_ptr<DataType> type() const override {
    return ::arrow::dictionary(indices_builder_.type(), null());
  }

 protected:
  BuilderType indices_builder_;
};

}  // namespace internal

/// \brief A DictionaryArray builder that uses AdaptiveIntBuilder to return the
/// smallest index size that can accommodate the dictionary indices
template <typename T>
class DictionaryBuilder : public internal::DictionaryBuilderBase<AdaptiveIntBuilder, T> {
 public:
  using BASE = internal::DictionaryBuilderBase<AdaptiveIntBuilder, T>;
  using BASE::BASE;

  /// \brief Append dictionary indices directly without modifying memo
  ///
  /// NOTE: Experimental API
  Status AppendIndices(const int64_t* values, int64_t length,
                       const uint8_t* valid_bytes = NULLPTR) {
    int64_t null_count_before = this->indices_builder_.null_count();
    ARROW_RETURN_NOT_OK(this->indices_builder_.AppendValues(values, length, valid_bytes));
    this->capacity_ = this->indices_builder_.capacity();
    this->length_ += length;
    this->null_count_ += this->indices_builder_.null_count() - null_count_before;
    return Status::OK();
  }
};

/// \brief A DictionaryArray builder that always returns int32 dictionary
/// indices so that data cast to dictionary form will have a consistent index
/// type, e.g. for creating a ChunkedArray
template <typename T>
class Dictionary32Builder : public internal::DictionaryBuilderBase<Int32Builder, T> {
 public:
  using BASE = internal::DictionaryBuilderBase<Int32Builder, T>;
  using BASE::BASE;

  /// \brief Append dictionary indices directly without modifying memo
  ///
  /// NOTE: Experimental API
  Status AppendIndices(const int32_t* values, int64_t length,
                       const uint8_t* valid_bytes = NULLPTR) {
    int64_t null_count_before = this->indices_builder_.null_count();
    ARROW_RETURN_NOT_OK(this->indices_builder_.AppendValues(values, length, valid_bytes));
    this->capacity_ = this->indices_builder_.capacity();
    this->length_ += length;
    this->null_count_ += this->indices_builder_.null_count() - null_count_before;
    return Status::OK();
  }
};

// ----------------------------------------------------------------------
// Binary / Unicode builders
// (compatibility aliases; those used to be derived classes with additional
//  Append() overloads, but they have been folded into DictionaryBuilderBase)

using BinaryDictionaryBuilder = DictionaryBuilder<BinaryType>;
using StringDictionaryBuilder = DictionaryBuilder<StringType>;
using BinaryDictionary32Builder = Dictionary32Builder<BinaryType>;
using StringDictionary32Builder = Dictionary32Builder<StringType>;

/// @}

}  // namespace arrow

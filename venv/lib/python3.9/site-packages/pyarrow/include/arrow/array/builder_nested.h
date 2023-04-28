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
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "arrow/array/array_nested.h"
#include "arrow/array/builder_base.h"
#include "arrow/array/data.h"
#include "arrow/buffer.h"
#include "arrow/buffer_builder.h"
#include "arrow/status.h"
#include "arrow/type.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {

/// \addtogroup nested-builders
///
/// @{

// ----------------------------------------------------------------------
// List builder

template <typename TYPE>
class BaseListBuilder : public ArrayBuilder {
 public:
  using TypeClass = TYPE;
  using offset_type = typename TypeClass::offset_type;

  /// Use this constructor to incrementally build the value array along with offsets and
  /// null bitmap.
  BaseListBuilder(MemoryPool* pool, std::shared_ptr<ArrayBuilder> const& value_builder,
                  const std::shared_ptr<DataType>& type,
                  int64_t alignment = kDefaultBufferAlignment)
      : ArrayBuilder(pool, alignment),
        offsets_builder_(pool, alignment),
        value_builder_(value_builder),
        value_field_(type->field(0)->WithType(NULLPTR)) {}

  BaseListBuilder(MemoryPool* pool, std::shared_ptr<ArrayBuilder> const& value_builder,
                  int64_t alignment = kDefaultBufferAlignment)
      : BaseListBuilder(pool, value_builder, list(value_builder->type()), alignment) {}

  Status Resize(int64_t capacity) override {
    if (capacity > maximum_elements()) {
      return Status::CapacityError("List array cannot reserve space for more than ",
                                   maximum_elements(), " got ", capacity);
    }
    ARROW_RETURN_NOT_OK(CheckCapacity(capacity));

    // One more than requested for offsets
    ARROW_RETURN_NOT_OK(offsets_builder_.Resize(capacity + 1));
    return ArrayBuilder::Resize(capacity);
  }

  void Reset() override {
    ArrayBuilder::Reset();
    offsets_builder_.Reset();
    value_builder_->Reset();
  }

  /// \brief Vector append
  ///
  /// If passed, valid_bytes is of equal length to values, and any zero byte
  /// will be considered as a null for that slot
  Status AppendValues(const offset_type* offsets, int64_t length,
                      const uint8_t* valid_bytes = NULLPTR) {
    ARROW_RETURN_NOT_OK(Reserve(length));
    UnsafeAppendToBitmap(valid_bytes, length);
    offsets_builder_.UnsafeAppend(offsets, length);
    return Status::OK();
  }

  /// \brief Start a new variable-length list slot
  ///
  /// This function should be called before beginning to append elements to the
  /// value builder
  Status Append(bool is_valid = true) {
    ARROW_RETURN_NOT_OK(Reserve(1));
    UnsafeAppendToBitmap(is_valid);
    return AppendNextOffset();
  }

  Status AppendNull() final { return Append(false); }

  Status AppendNulls(int64_t length) final {
    ARROW_RETURN_NOT_OK(Reserve(length));
    ARROW_RETURN_NOT_OK(ValidateOverflow(0));
    UnsafeAppendToBitmap(length, false);
    const int64_t num_values = value_builder_->length();
    for (int64_t i = 0; i < length; ++i) {
      offsets_builder_.UnsafeAppend(static_cast<offset_type>(num_values));
    }
    return Status::OK();
  }

  Status AppendEmptyValue() final { return Append(true); }

  Status AppendEmptyValues(int64_t length) final {
    ARROW_RETURN_NOT_OK(Reserve(length));
    ARROW_RETURN_NOT_OK(ValidateOverflow(0));
    UnsafeAppendToBitmap(length, true);
    const int64_t num_values = value_builder_->length();
    for (int64_t i = 0; i < length; ++i) {
      offsets_builder_.UnsafeAppend(static_cast<offset_type>(num_values));
    }
    return Status::OK();
  }

  Status AppendArraySlice(const ArraySpan& array, int64_t offset,
                          int64_t length) override {
    const offset_type* offsets = array.GetValues<offset_type>(1);
    const uint8_t* validity = array.MayHaveNulls() ? array.buffers[0].data : NULLPTR;
    for (int64_t row = offset; row < offset + length; row++) {
      if (!validity || bit_util::GetBit(validity, array.offset + row)) {
        ARROW_RETURN_NOT_OK(Append());
        int64_t slot_length = offsets[row + 1] - offsets[row];
        ARROW_RETURN_NOT_OK(value_builder_->AppendArraySlice(array.child_data[0],
                                                             offsets[row], slot_length));
      } else {
        ARROW_RETURN_NOT_OK(AppendNull());
      }
    }
    return Status::OK();
  }

  Status FinishInternal(std::shared_ptr<ArrayData>* out) override {
    ARROW_RETURN_NOT_OK(AppendNextOffset());

    // Offset padding zeroed by BufferBuilder
    std::shared_ptr<Buffer> offsets, null_bitmap;
    ARROW_RETURN_NOT_OK(offsets_builder_.Finish(&offsets));
    ARROW_RETURN_NOT_OK(null_bitmap_builder_.Finish(&null_bitmap));

    if (value_builder_->length() == 0) {
      // Try to make sure we get a non-null values buffer (ARROW-2744)
      ARROW_RETURN_NOT_OK(value_builder_->Resize(0));
    }

    std::shared_ptr<ArrayData> items;
    ARROW_RETURN_NOT_OK(value_builder_->FinishInternal(&items));

    *out = ArrayData::Make(type(), length_, {null_bitmap, offsets}, {std::move(items)},
                           null_count_);
    Reset();
    return Status::OK();
  }

  Status ValidateOverflow(int64_t new_elements) const {
    auto new_length = value_builder_->length() + new_elements;
    if (ARROW_PREDICT_FALSE(new_length > maximum_elements())) {
      return Status::CapacityError("List array cannot contain more than ",
                                   maximum_elements(), " elements, have ", new_elements);
    } else {
      return Status::OK();
    }
  }

  ArrayBuilder* value_builder() const { return value_builder_.get(); }

  // Cannot make this a static attribute because of linking issues
  static constexpr int64_t maximum_elements() {
    return std::numeric_limits<offset_type>::max() - 1;
  }

  std::shared_ptr<DataType> type() const override {
    return std::make_shared<TYPE>(value_field_->WithType(value_builder_->type()));
  }

 protected:
  TypedBufferBuilder<offset_type> offsets_builder_;
  std::shared_ptr<ArrayBuilder> value_builder_;
  std::shared_ptr<Field> value_field_;

  Status AppendNextOffset() {
    ARROW_RETURN_NOT_OK(ValidateOverflow(0));
    const int64_t num_values = value_builder_->length();
    return offsets_builder_.Append(static_cast<offset_type>(num_values));
  }
};

/// \class ListBuilder
/// \brief Builder class for variable-length list array value types
///
/// To use this class, you must append values to the child array builder and use
/// the Append function to delimit each distinct list value (once the values
/// have been appended to the child array) or use the bulk API to append
/// a sequence of offsets and null values.
///
/// A note on types.  Per arrow/type.h all types in the c++ implementation are
/// logical so even though this class always builds list array, this can
/// represent multiple different logical types.  If no logical type is provided
/// at construction time, the class defaults to List<T> where t is taken from the
/// value_builder/values that the object is constructed with.
class ARROW_EXPORT ListBuilder : public BaseListBuilder<ListType> {
 public:
  using BaseListBuilder::BaseListBuilder;

  /// \cond FALSE
  using ArrayBuilder::Finish;
  /// \endcond

  Status Finish(std::shared_ptr<ListArray>* out) { return FinishTyped(out); }
};

/// \class LargeListBuilder
/// \brief Builder class for large variable-length list array value types
///
/// Like ListBuilder, but to create large list arrays (with 64-bit offsets).
class ARROW_EXPORT LargeListBuilder : public BaseListBuilder<LargeListType> {
 public:
  using BaseListBuilder::BaseListBuilder;

  /// \cond FALSE
  using ArrayBuilder::Finish;
  /// \endcond

  Status Finish(std::shared_ptr<LargeListArray>* out) { return FinishTyped(out); }
};

// ----------------------------------------------------------------------
// Map builder

/// \class MapBuilder
/// \brief Builder class for arrays of variable-size maps
///
/// To use this class, you must append values to the key and item array builders
/// and use the Append function to delimit each distinct map (once the keys and items
/// have been appended) or use the bulk API to append a sequence of offsets and null
/// maps.
///
/// Key uniqueness and ordering are not validated.
class ARROW_EXPORT MapBuilder : public ArrayBuilder {
 public:
  /// Use this constructor to define the built array's type explicitly. If key_builder
  /// or item_builder has indeterminate type, this builder will also.
  MapBuilder(MemoryPool* pool, const std::shared_ptr<ArrayBuilder>& key_builder,
             const std::shared_ptr<ArrayBuilder>& item_builder,
             const std::shared_ptr<DataType>& type);

  /// Use this constructor to infer the built array's type. If key_builder or
  /// item_builder has indeterminate type, this builder will also.
  MapBuilder(MemoryPool* pool, const std::shared_ptr<ArrayBuilder>& key_builder,
             const std::shared_ptr<ArrayBuilder>& item_builder, bool keys_sorted = false);

  MapBuilder(MemoryPool* pool, const std::shared_ptr<ArrayBuilder>& item_builder,
             const std::shared_ptr<DataType>& type);

  Status Resize(int64_t capacity) override;
  void Reset() override;
  Status FinishInternal(std::shared_ptr<ArrayData>* out) override;

  /// \cond FALSE
  using ArrayBuilder::Finish;
  /// \endcond

  Status Finish(std::shared_ptr<MapArray>* out) { return FinishTyped(out); }

  /// \brief Vector append
  ///
  /// If passed, valid_bytes is of equal length to values, and any zero byte
  /// will be considered as a null for that slot
  Status AppendValues(const int32_t* offsets, int64_t length,
                      const uint8_t* valid_bytes = NULLPTR);

  /// \brief Start a new variable-length map slot
  ///
  /// This function should be called before beginning to append elements to the
  /// key and item builders
  Status Append();

  Status AppendNull() final;

  Status AppendNulls(int64_t length) final;

  Status AppendEmptyValue() final;

  Status AppendEmptyValues(int64_t length) final;

  Status AppendArraySlice(const ArraySpan& array, int64_t offset,
                          int64_t length) override {
    const int32_t* offsets = array.GetValues<int32_t>(1);
    const uint8_t* validity = array.MayHaveNulls() ? array.buffers[0].data : NULLPTR;
    for (int64_t row = offset; row < offset + length; row++) {
      if (!validity || bit_util::GetBit(validity, array.offset + row)) {
        ARROW_RETURN_NOT_OK(Append());
        const int64_t slot_length = offsets[row + 1] - offsets[row];
        // Add together the inner StructArray offset to the Map/List offset
        int64_t key_value_offset = array.child_data[0].offset + offsets[row];
        ARROW_RETURN_NOT_OK(key_builder_->AppendArraySlice(
            array.child_data[0].child_data[0], key_value_offset, slot_length));
        ARROW_RETURN_NOT_OK(item_builder_->AppendArraySlice(
            array.child_data[0].child_data[1], key_value_offset, slot_length));
      } else {
        ARROW_RETURN_NOT_OK(AppendNull());
      }
    }
    return Status::OK();
  }

  /// \brief Get builder to append keys.
  ///
  /// Append a key with this builder should be followed by appending
  /// an item or null value with item_builder().
  ArrayBuilder* key_builder() const { return key_builder_.get(); }

  /// \brief Get builder to append items
  ///
  /// Appending an item with this builder should have been preceded
  /// by appending a key with key_builder().
  ArrayBuilder* item_builder() const { return item_builder_.get(); }

  /// \brief Get builder to add Map entries as struct values.
  ///
  /// This is used instead of key_builder()/item_builder() and allows
  /// the Map to be built as a list of struct values.
  ArrayBuilder* value_builder() const { return list_builder_->value_builder(); }

  std::shared_ptr<DataType> type() const override {
    // Key and Item builder may update types, but they don't contain the field names,
    // so we need to reconstruct the type. (See ARROW-13735.)
    return std::make_shared<MapType>(
        field(entries_name_,
              struct_({field(key_name_, key_builder_->type(), false),
                       field(item_name_, item_builder_->type(), item_nullable_)}),
              false),
        keys_sorted_);
  }

  Status ValidateOverflow(int64_t new_elements) {
    return list_builder_->ValidateOverflow(new_elements);
  }

 protected:
  inline Status AdjustStructBuilderLength();

 protected:
  bool keys_sorted_ = false;
  bool item_nullable_ = false;
  std::string entries_name_;
  std::string key_name_;
  std::string item_name_;
  std::shared_ptr<ListBuilder> list_builder_;
  std::shared_ptr<ArrayBuilder> key_builder_;
  std::shared_ptr<ArrayBuilder> item_builder_;
};

// ----------------------------------------------------------------------
// FixedSizeList builder

/// \class FixedSizeListBuilder
/// \brief Builder class for fixed-length list array value types
class ARROW_EXPORT FixedSizeListBuilder : public ArrayBuilder {
 public:
  /// Use this constructor to define the built array's type explicitly. If value_builder
  /// has indeterminate type, this builder will also.
  FixedSizeListBuilder(MemoryPool* pool,
                       std::shared_ptr<ArrayBuilder> const& value_builder,
                       int32_t list_size);

  /// Use this constructor to infer the built array's type. If value_builder has
  /// indeterminate type, this builder will also.
  FixedSizeListBuilder(MemoryPool* pool,
                       std::shared_ptr<ArrayBuilder> const& value_builder,
                       const std::shared_ptr<DataType>& type);

  Status Resize(int64_t capacity) override;
  void Reset() override;
  Status FinishInternal(std::shared_ptr<ArrayData>* out) override;

  /// \cond FALSE
  using ArrayBuilder::Finish;
  /// \endcond

  Status Finish(std::shared_ptr<FixedSizeListArray>* out) { return FinishTyped(out); }

  /// \brief Append a valid fixed length list.
  ///
  /// This function affects only the validity bitmap; the child values must be appended
  /// using the child array builder.
  Status Append();

  /// \brief Vector append
  ///
  /// If passed, valid_bytes wil be read and any zero byte
  /// will cause the corresponding slot to be null
  ///
  /// This function affects only the validity bitmap; the child values must be appended
  /// using the child array builder. This includes appending nulls for null lists.
  /// XXX this restriction is confusing, should this method be omitted?
  Status AppendValues(int64_t length, const uint8_t* valid_bytes = NULLPTR);

  /// \brief Append a null fixed length list.
  ///
  /// The child array builder will have the appropriate number of nulls appended
  /// automatically.
  Status AppendNull() final;

  /// \brief Append length null fixed length lists.
  ///
  /// The child array builder will have the appropriate number of nulls appended
  /// automatically.
  Status AppendNulls(int64_t length) final;

  Status ValidateOverflow(int64_t new_elements);

  Status AppendEmptyValue() final;

  Status AppendEmptyValues(int64_t length) final;

  Status AppendArraySlice(const ArraySpan& array, int64_t offset, int64_t length) final {
    const uint8_t* validity = array.MayHaveNulls() ? array.buffers[0].data : NULLPTR;
    for (int64_t row = offset; row < offset + length; row++) {
      if (!validity || bit_util::GetBit(validity, array.offset + row)) {
        ARROW_RETURN_NOT_OK(value_builder_->AppendArraySlice(
            array.child_data[0], list_size_ * (array.offset + row), list_size_));
        ARROW_RETURN_NOT_OK(Append());
      } else {
        ARROW_RETURN_NOT_OK(AppendNull());
      }
    }
    return Status::OK();
  }

  ArrayBuilder* value_builder() const { return value_builder_.get(); }

  std::shared_ptr<DataType> type() const override {
    return fixed_size_list(value_field_->WithType(value_builder_->type()), list_size_);
  }

  // Cannot make this a static attribute because of linking issues
  static constexpr int64_t maximum_elements() {
    return std::numeric_limits<FixedSizeListType::offset_type>::max() - 1;
  }

 protected:
  std::shared_ptr<Field> value_field_;
  const int32_t list_size_;
  std::shared_ptr<ArrayBuilder> value_builder_;
};

// ----------------------------------------------------------------------
// Struct

// ---------------------------------------------------------------------------------
// StructArray builder
/// Append, Resize and Reserve methods are acting on StructBuilder.
/// Please make sure all these methods of all child-builders' are consistently
/// called to maintain data-structure consistency.
class ARROW_EXPORT StructBuilder : public ArrayBuilder {
 public:
  /// If any of field_builders has indeterminate type, this builder will also
  StructBuilder(const std::shared_ptr<DataType>& type, MemoryPool* pool,
                std::vector<std::shared_ptr<ArrayBuilder>> field_builders);

  Status FinishInternal(std::shared_ptr<ArrayData>* out) override;

  /// \cond FALSE
  using ArrayBuilder::Finish;
  /// \endcond

  Status Finish(std::shared_ptr<StructArray>* out) { return FinishTyped(out); }

  /// Null bitmap is of equal length to every child field, and any zero byte
  /// will be considered as a null for that field, but users must using app-
  /// end methods or advance methods of the child builders' independently to
  /// insert data.
  Status AppendValues(int64_t length, const uint8_t* valid_bytes) {
    ARROW_RETURN_NOT_OK(Reserve(length));
    UnsafeAppendToBitmap(valid_bytes, length);
    return Status::OK();
  }

  /// Append an element to the Struct. All child-builders' Append method must
  /// be called independently to maintain data-structure consistency.
  Status Append(bool is_valid = true) {
    ARROW_RETURN_NOT_OK(Reserve(1));
    UnsafeAppendToBitmap(is_valid);
    return Status::OK();
  }

  /// \brief Append a null value. Automatically appends an empty value to each child
  /// builder.
  Status AppendNull() final {
    for (const auto& field : children_) {
      ARROW_RETURN_NOT_OK(field->AppendEmptyValue());
    }
    return Append(false);
  }

  /// \brief Append multiple null values. Automatically appends empty values to each
  /// child builder.
  Status AppendNulls(int64_t length) final {
    for (const auto& field : children_) {
      ARROW_RETURN_NOT_OK(field->AppendEmptyValues(length));
    }
    ARROW_RETURN_NOT_OK(Reserve(length));
    UnsafeAppendToBitmap(length, false);
    return Status::OK();
  }

  Status AppendEmptyValue() final {
    for (const auto& field : children_) {
      ARROW_RETURN_NOT_OK(field->AppendEmptyValue());
    }
    return Append(true);
  }

  Status AppendEmptyValues(int64_t length) final {
    for (const auto& field : children_) {
      ARROW_RETURN_NOT_OK(field->AppendEmptyValues(length));
    }
    ARROW_RETURN_NOT_OK(Reserve(length));
    UnsafeAppendToBitmap(length, true);
    return Status::OK();
  }

  Status AppendArraySlice(const ArraySpan& array, int64_t offset,
                          int64_t length) override {
    for (int i = 0; static_cast<size_t>(i) < children_.size(); i++) {
      ARROW_RETURN_NOT_OK(children_[i]->AppendArraySlice(array.child_data[i],
                                                         array.offset + offset, length));
    }
    const uint8_t* validity = array.MayHaveNulls() ? array.buffers[0].data : NULLPTR;
    ARROW_RETURN_NOT_OK(Reserve(length));
    UnsafeAppendToBitmap(validity, array.offset + offset, length);
    return Status::OK();
  }

  void Reset() override;

  ArrayBuilder* field_builder(int i) const { return children_[i].get(); }

  int num_fields() const { return static_cast<int>(children_.size()); }

  std::shared_ptr<DataType> type() const override;

 private:
  std::shared_ptr<DataType> type_;
};

/// @}

}  // namespace arrow

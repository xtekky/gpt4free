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
#include <string>
#include <vector>

#include "arrow/array/array_nested.h"
#include "arrow/array/builder_base.h"
#include "arrow/array/data.h"
#include "arrow/buffer_builder.h"
#include "arrow/memory_pool.h"
#include "arrow/status.h"
#include "arrow/type.h"
#include "arrow/util/visibility.h"

namespace arrow {

/// \addtogroup nested-builders
///
/// @{

/// \brief Base class for union array builds.
///
/// Note that while we subclass ArrayBuilder, as union types do not have a
/// validity bitmap, the bitmap builder member of ArrayBuilder is not used.
class ARROW_EXPORT BasicUnionBuilder : public ArrayBuilder {
 public:
  Status FinishInternal(std::shared_ptr<ArrayData>* out) override;

  /// \cond FALSE
  using ArrayBuilder::Finish;
  /// \endcond

  Status Finish(std::shared_ptr<UnionArray>* out) { return FinishTyped(out); }

  /// \brief Make a new child builder available to the UnionArray
  ///
  /// \param[in] new_child the child builder
  /// \param[in] field_name the name of the field in the union array type
  /// if type inference is used
  /// \return child index, which is the "type" argument that needs
  /// to be passed to the "Append" method to add a new element to
  /// the union array.
  int8_t AppendChild(const std::shared_ptr<ArrayBuilder>& new_child,
                     const std::string& field_name = "");

  std::shared_ptr<DataType> type() const override;

  int64_t length() const override { return types_builder_.length(); }

 protected:
  BasicUnionBuilder(MemoryPool* pool, int64_t alignment,
                    const std::vector<std::shared_ptr<ArrayBuilder>>& children,
                    const std::shared_ptr<DataType>& type);

  int8_t NextTypeId();

  std::vector<std::shared_ptr<Field>> child_fields_;
  std::vector<int8_t> type_codes_;
  UnionMode::type mode_;

  std::vector<ArrayBuilder*> type_id_to_children_;
  std::vector<int> type_id_to_child_id_;
  // for all type_id < dense_type_id_, type_id_to_children_[type_id] != nullptr
  int8_t dense_type_id_ = 0;
  TypedBufferBuilder<int8_t> types_builder_;
};

/// \class DenseUnionBuilder
///
/// This API is EXPERIMENTAL.
class ARROW_EXPORT DenseUnionBuilder : public BasicUnionBuilder {
 public:
  /// Use this constructor to initialize the UnionBuilder with no child builders,
  /// allowing type to be inferred. You will need to call AppendChild for each of the
  /// children builders you want to use.
  explicit DenseUnionBuilder(MemoryPool* pool,
                             int64_t alignment = kDefaultBufferAlignment)
      : BasicUnionBuilder(pool, alignment, {}, dense_union(FieldVector{})),
        offsets_builder_(pool, alignment) {}

  /// Use this constructor to specify the type explicitly.
  /// You can still add child builders to the union after using this constructor
  DenseUnionBuilder(MemoryPool* pool,
                    const std::vector<std::shared_ptr<ArrayBuilder>>& children,
                    const std::shared_ptr<DataType>& type,
                    int64_t alignment = kDefaultBufferAlignment)
      : BasicUnionBuilder(pool, alignment, children, type),
        offsets_builder_(pool, alignment) {}

  Status AppendNull() final {
    const int8_t first_child_code = type_codes_[0];
    ArrayBuilder* child_builder = type_id_to_children_[first_child_code];
    ARROW_RETURN_NOT_OK(types_builder_.Append(first_child_code));
    ARROW_RETURN_NOT_OK(
        offsets_builder_.Append(static_cast<int32_t>(child_builder->length())));
    // Append a null arbitrarily to the first child
    return child_builder->AppendNull();
  }

  Status AppendNulls(int64_t length) final {
    const int8_t first_child_code = type_codes_[0];
    ArrayBuilder* child_builder = type_id_to_children_[first_child_code];
    ARROW_RETURN_NOT_OK(types_builder_.Append(length, first_child_code));
    ARROW_RETURN_NOT_OK(
        offsets_builder_.Append(length, static_cast<int32_t>(child_builder->length())));
    // Append just a single null to the first child
    return child_builder->AppendNull();
  }

  Status AppendEmptyValue() final {
    const int8_t first_child_code = type_codes_[0];
    ArrayBuilder* child_builder = type_id_to_children_[first_child_code];
    ARROW_RETURN_NOT_OK(types_builder_.Append(first_child_code));
    ARROW_RETURN_NOT_OK(
        offsets_builder_.Append(static_cast<int32_t>(child_builder->length())));
    // Append an empty value arbitrarily to the first child
    return child_builder->AppendEmptyValue();
  }

  Status AppendEmptyValues(int64_t length) final {
    const int8_t first_child_code = type_codes_[0];
    ArrayBuilder* child_builder = type_id_to_children_[first_child_code];
    ARROW_RETURN_NOT_OK(types_builder_.Append(length, first_child_code));
    ARROW_RETURN_NOT_OK(
        offsets_builder_.Append(length, static_cast<int32_t>(child_builder->length())));
    // Append just a single empty value to the first child
    return child_builder->AppendEmptyValue();
  }

  /// \brief Append an element to the UnionArray. This must be followed
  ///        by an append to the appropriate child builder.
  ///
  /// \param[in] next_type type_id of the child to which the next value will be appended.
  ///
  /// The corresponding child builder must be appended to independently after this method
  /// is called.
  Status Append(int8_t next_type) {
    ARROW_RETURN_NOT_OK(types_builder_.Append(next_type));
    if (type_id_to_children_[next_type]->length() == kListMaximumElements) {
      return Status::CapacityError(
          "a dense UnionArray cannot contain more than 2^31 - 1 elements from a single "
          "child");
    }
    auto offset = static_cast<int32_t>(type_id_to_children_[next_type]->length());
    return offsets_builder_.Append(offset);
  }

  Status AppendArraySlice(const ArraySpan& array, int64_t offset,
                          int64_t length) override;

  Status FinishInternal(std::shared_ptr<ArrayData>* out) override;

 private:
  TypedBufferBuilder<int32_t> offsets_builder_;
};

/// \class SparseUnionBuilder
///
/// This API is EXPERIMENTAL.
class ARROW_EXPORT SparseUnionBuilder : public BasicUnionBuilder {
 public:
  /// Use this constructor to initialize the UnionBuilder with no child builders,
  /// allowing type to be inferred. You will need to call AppendChild for each of the
  /// children builders you want to use.
  explicit SparseUnionBuilder(MemoryPool* pool,
                              int64_t alignment = kDefaultBufferAlignment)
      : BasicUnionBuilder(pool, alignment, {}, sparse_union(FieldVector{})) {}

  /// Use this constructor to specify the type explicitly.
  /// You can still add child builders to the union after using this constructor
  SparseUnionBuilder(MemoryPool* pool,
                     const std::vector<std::shared_ptr<ArrayBuilder>>& children,
                     const std::shared_ptr<DataType>& type,
                     int64_t alignment = kDefaultBufferAlignment)
      : BasicUnionBuilder(pool, alignment, children, type) {}

  /// \brief Append a null value.
  ///
  /// A null is appended to the first child, empty values to the other children.
  Status AppendNull() final {
    const auto first_child_code = type_codes_[0];
    ARROW_RETURN_NOT_OK(types_builder_.Append(first_child_code));
    ARROW_RETURN_NOT_OK(type_id_to_children_[first_child_code]->AppendNull());
    for (int i = 1; i < static_cast<int>(type_codes_.size()); ++i) {
      ARROW_RETURN_NOT_OK(type_id_to_children_[type_codes_[i]]->AppendEmptyValue());
    }
    return Status::OK();
  }

  /// \brief Append multiple null values.
  ///
  /// Nulls are appended to the first child, empty values to the other children.
  Status AppendNulls(int64_t length) final {
    const auto first_child_code = type_codes_[0];
    ARROW_RETURN_NOT_OK(types_builder_.Append(length, first_child_code));
    ARROW_RETURN_NOT_OK(type_id_to_children_[first_child_code]->AppendNulls(length));
    for (int i = 1; i < static_cast<int>(type_codes_.size()); ++i) {
      ARROW_RETURN_NOT_OK(
          type_id_to_children_[type_codes_[i]]->AppendEmptyValues(length));
    }
    return Status::OK();
  }

  Status AppendEmptyValue() final {
    ARROW_RETURN_NOT_OK(types_builder_.Append(type_codes_[0]));
    for (int8_t code : type_codes_) {
      ARROW_RETURN_NOT_OK(type_id_to_children_[code]->AppendEmptyValue());
    }
    return Status::OK();
  }

  Status AppendEmptyValues(int64_t length) final {
    ARROW_RETURN_NOT_OK(types_builder_.Append(length, type_codes_[0]));
    for (int8_t code : type_codes_) {
      ARROW_RETURN_NOT_OK(type_id_to_children_[code]->AppendEmptyValues(length));
    }
    return Status::OK();
  }

  /// \brief Append an element to the UnionArray. This must be followed
  ///        by an append to the appropriate child builder.
  ///
  /// \param[in] next_type type_id of the child to which the next value will be appended.
  ///
  /// The corresponding child builder must be appended to independently after this method
  /// is called, and all other child builders must have null or empty value appended.
  Status Append(int8_t next_type) { return types_builder_.Append(next_type); }

  Status AppendArraySlice(const ArraySpan& array, int64_t offset,
                          int64_t length) override;
};

/// @}

}  // namespace arrow

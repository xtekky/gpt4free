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

// Array accessor classes for Binary, LargeBinart, String, LargeString,
// FixedSizeBinary

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "arrow/array/array_base.h"
#include "arrow/array/data.h"
#include "arrow/buffer.h"
#include "arrow/stl_iterator.h"
#include "arrow/type.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {

/// \addtogroup binary-arrays
///
/// @{

// ----------------------------------------------------------------------
// Binary and String

/// Base class for variable-sized binary arrays, regardless of offset size
/// and logical interpretation.
template <typename TYPE>
class BaseBinaryArray : public FlatArray {
 public:
  using TypeClass = TYPE;
  using offset_type = typename TypeClass::offset_type;
  using IteratorType = stl::ArrayIterator<BaseBinaryArray<TYPE>>;

  /// Return the pointer to the given elements bytes
  // XXX should GetValue(int64_t i) return a string_view?
  const uint8_t* GetValue(int64_t i, offset_type* out_length) const {
    // Account for base offset
    i += data_->offset;
    const offset_type pos = raw_value_offsets_[i];
    *out_length = raw_value_offsets_[i + 1] - pos;
    return raw_data_ + pos;
  }

  /// \brief Get binary value as a string_view
  ///
  /// \param i the value index
  /// \return the view over the selected value
  std::string_view GetView(int64_t i) const {
    // Account for base offset
    i += data_->offset;
    const offset_type pos = raw_value_offsets_[i];
    return std::string_view(reinterpret_cast<const char*>(raw_data_ + pos),
                            raw_value_offsets_[i + 1] - pos);
  }

  std::optional<std::string_view> operator[](int64_t i) const {
    return *IteratorType(*this, i);
  }

  /// \brief Get binary value as a string_view
  /// Provided for consistency with other arrays.
  ///
  /// \param i the value index
  /// \return the view over the selected value
  std::string_view Value(int64_t i) const { return GetView(i); }

  /// \brief Get binary value as a std::string
  ///
  /// \param i the value index
  /// \return the value copied into a std::string
  std::string GetString(int64_t i) const { return std::string(GetView(i)); }

  /// Note that this buffer does not account for any slice offset
  std::shared_ptr<Buffer> value_offsets() const { return data_->buffers[1]; }

  /// Note that this buffer does not account for any slice offset
  std::shared_ptr<Buffer> value_data() const { return data_->buffers[2]; }

  const offset_type* raw_value_offsets() const {
    return raw_value_offsets_ + data_->offset;
  }

  const uint8_t* raw_data() const { return raw_data_; }

  /// \brief Return the data buffer absolute offset of the data for the value
  /// at the passed index.
  ///
  /// Does not perform boundschecking
  offset_type value_offset(int64_t i) const {
    return raw_value_offsets_[i + data_->offset];
  }

  /// \brief Return the length of the data for the value at the passed index.
  ///
  /// Does not perform boundschecking
  offset_type value_length(int64_t i) const {
    i += data_->offset;
    return raw_value_offsets_[i + 1] - raw_value_offsets_[i];
  }

  /// \brief Return the total length of the memory in the data buffer
  /// referenced by this array. If the array has been sliced then this may be
  /// less than the size of the data buffer (data_->buffers[2]).
  offset_type total_values_length() const {
    if (data_->length > 0) {
      return raw_value_offsets_[data_->length + data_->offset] -
             raw_value_offsets_[data_->offset];
    } else {
      return 0;
    }
  }

  IteratorType begin() const { return IteratorType(*this); }

  IteratorType end() const { return IteratorType(*this, length()); }

 protected:
  // For subclasses
  BaseBinaryArray() = default;

  // Protected method for constructors
  void SetData(const std::shared_ptr<ArrayData>& data) {
    this->Array::SetData(data);
    raw_value_offsets_ = data->GetValuesSafe<offset_type>(1, /*offset=*/0);
    raw_data_ = data->GetValuesSafe<uint8_t>(2, /*offset=*/0);
  }

  const offset_type* raw_value_offsets_ = NULLPTR;
  const uint8_t* raw_data_ = NULLPTR;
};

/// Concrete Array class for variable-size binary data
class ARROW_EXPORT BinaryArray : public BaseBinaryArray<BinaryType> {
 public:
  explicit BinaryArray(const std::shared_ptr<ArrayData>& data);

  BinaryArray(int64_t length, const std::shared_ptr<Buffer>& value_offsets,
              const std::shared_ptr<Buffer>& data,
              const std::shared_ptr<Buffer>& null_bitmap = NULLPTR,
              int64_t null_count = kUnknownNullCount, int64_t offset = 0);

 protected:
  // For subclasses such as StringArray
  BinaryArray() : BaseBinaryArray() {}
};

/// Concrete Array class for variable-size string (utf-8) data
class ARROW_EXPORT StringArray : public BinaryArray {
 public:
  using TypeClass = StringType;

  explicit StringArray(const std::shared_ptr<ArrayData>& data);

  StringArray(int64_t length, const std::shared_ptr<Buffer>& value_offsets,
              const std::shared_ptr<Buffer>& data,
              const std::shared_ptr<Buffer>& null_bitmap = NULLPTR,
              int64_t null_count = kUnknownNullCount, int64_t offset = 0);

  /// \brief Validate that this array contains only valid UTF8 entries
  ///
  /// This check is also implied by ValidateFull()
  Status ValidateUTF8() const;
};

/// Concrete Array class for large variable-size binary data
class ARROW_EXPORT LargeBinaryArray : public BaseBinaryArray<LargeBinaryType> {
 public:
  explicit LargeBinaryArray(const std::shared_ptr<ArrayData>& data);

  LargeBinaryArray(int64_t length, const std::shared_ptr<Buffer>& value_offsets,
                   const std::shared_ptr<Buffer>& data,
                   const std::shared_ptr<Buffer>& null_bitmap = NULLPTR,
                   int64_t null_count = kUnknownNullCount, int64_t offset = 0);

 protected:
  // For subclasses such as LargeStringArray
  LargeBinaryArray() : BaseBinaryArray() {}
};

/// Concrete Array class for large variable-size string (utf-8) data
class ARROW_EXPORT LargeStringArray : public LargeBinaryArray {
 public:
  using TypeClass = LargeStringType;

  explicit LargeStringArray(const std::shared_ptr<ArrayData>& data);

  LargeStringArray(int64_t length, const std::shared_ptr<Buffer>& value_offsets,
                   const std::shared_ptr<Buffer>& data,
                   const std::shared_ptr<Buffer>& null_bitmap = NULLPTR,
                   int64_t null_count = kUnknownNullCount, int64_t offset = 0);

  /// \brief Validate that this array contains only valid UTF8 entries
  ///
  /// This check is also implied by ValidateFull()
  Status ValidateUTF8() const;
};

// ----------------------------------------------------------------------
// Fixed width binary

/// Concrete Array class for fixed-size binary data
class ARROW_EXPORT FixedSizeBinaryArray : public PrimitiveArray {
 public:
  using TypeClass = FixedSizeBinaryType;
  using IteratorType = stl::ArrayIterator<FixedSizeBinaryArray>;

  explicit FixedSizeBinaryArray(const std::shared_ptr<ArrayData>& data);

  FixedSizeBinaryArray(const std::shared_ptr<DataType>& type, int64_t length,
                       const std::shared_ptr<Buffer>& data,
                       const std::shared_ptr<Buffer>& null_bitmap = NULLPTR,
                       int64_t null_count = kUnknownNullCount, int64_t offset = 0);

  const uint8_t* GetValue(int64_t i) const;
  const uint8_t* Value(int64_t i) const { return GetValue(i); }

  std::string_view GetView(int64_t i) const {
    return std::string_view(reinterpret_cast<const char*>(GetValue(i)), byte_width());
  }

  std::optional<std::string_view> operator[](int64_t i) const {
    return *IteratorType(*this, i);
  }

  std::string GetString(int64_t i) const { return std::string(GetView(i)); }

  int32_t byte_width() const { return byte_width_; }

  const uint8_t* raw_values() const { return raw_values_ + data_->offset * byte_width_; }

  IteratorType begin() const { return IteratorType(*this); }

  IteratorType end() const { return IteratorType(*this, length()); }

 protected:
  void SetData(const std::shared_ptr<ArrayData>& data) {
    this->PrimitiveArray::SetData(data);
    byte_width_ =
        internal::checked_cast<const FixedSizeBinaryType&>(*type()).byte_width();
  }

  int32_t byte_width_;
};

/// @}

}  // namespace arrow

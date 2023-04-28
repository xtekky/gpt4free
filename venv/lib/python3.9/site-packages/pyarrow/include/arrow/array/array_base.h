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
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

#include "arrow/array/data.h"
#include "arrow/buffer.h"
#include "arrow/compare.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"
#include "arrow/visitor.h"

namespace arrow {

// ----------------------------------------------------------------------
// User array accessor types

/// \brief Array base type
/// Immutable data array with some logical type and some length.
///
/// Any memory is owned by the respective Buffer instance (or its parents).
///
/// The base class is only required to have a null bitmap buffer if the null
/// count is greater than 0
///
/// If known, the null count can be provided in the base Array constructor. If
/// the null count is not known, pass -1 to indicate that the null count is to
/// be computed on the first call to null_count()
class ARROW_EXPORT Array {
 public:
  virtual ~Array() = default;

  /// \brief Return true if value at index is null. Does not boundscheck
  bool IsNull(int64_t i) const {
    return null_bitmap_data_ != NULLPTR
               ? !bit_util::GetBit(null_bitmap_data_, i + data_->offset)
               : data_->null_count == data_->length;
  }

  /// \brief Return true if value at index is valid (not null). Does not
  /// boundscheck
  bool IsValid(int64_t i) const {
    return null_bitmap_data_ != NULLPTR
               ? bit_util::GetBit(null_bitmap_data_, i + data_->offset)
               : data_->null_count != data_->length;
  }

  /// \brief Return a Scalar containing the value of this array at i
  Result<std::shared_ptr<Scalar>> GetScalar(int64_t i) const;

  /// Size in the number of elements this array contains.
  int64_t length() const { return data_->length; }

  /// A relative position into another array's data, to enable zero-copy
  /// slicing. This value defaults to zero
  int64_t offset() const { return data_->offset; }

  /// The number of null entries in the array. If the null count was not known
  /// at time of construction (and set to a negative value), then the null
  /// count will be computed and cached on the first invocation of this
  /// function
  int64_t null_count() const;

  std::shared_ptr<DataType> type() const { return data_->type; }
  Type::type type_id() const { return data_->type->id(); }

  /// Buffer for the validity (null) bitmap, if any. Note that Union types
  /// never have a null bitmap.
  ///
  /// Note that for `null_count == 0` or for null type, this will be null.
  /// This buffer does not account for any slice offset
  const std::shared_ptr<Buffer>& null_bitmap() const { return data_->buffers[0]; }

  /// Raw pointer to the null bitmap.
  ///
  /// Note that for `null_count == 0` or for null type, this will be null.
  /// This buffer does not account for any slice offset
  const uint8_t* null_bitmap_data() const { return null_bitmap_data_; }

  /// Equality comparison with another array
  bool Equals(const Array& arr, const EqualOptions& = EqualOptions::Defaults()) const;
  bool Equals(const std::shared_ptr<Array>& arr,
              const EqualOptions& = EqualOptions::Defaults()) const;

  /// \brief Return the formatted unified diff of arrow::Diff between this
  /// Array and another Array
  std::string Diff(const Array& other) const;

  /// Approximate equality comparison with another array
  ///
  /// epsilon is only used if this is FloatArray or DoubleArray
  bool ApproxEquals(const std::shared_ptr<Array>& arr,
                    const EqualOptions& = EqualOptions::Defaults()) const;
  bool ApproxEquals(const Array& arr,
                    const EqualOptions& = EqualOptions::Defaults()) const;

  /// Compare if the range of slots specified are equal for the given array and
  /// this array.  end_idx exclusive.  This methods does not bounds check.
  bool RangeEquals(int64_t start_idx, int64_t end_idx, int64_t other_start_idx,
                   const Array& other,
                   const EqualOptions& = EqualOptions::Defaults()) const;
  bool RangeEquals(int64_t start_idx, int64_t end_idx, int64_t other_start_idx,
                   const std::shared_ptr<Array>& other,
                   const EqualOptions& = EqualOptions::Defaults()) const;
  bool RangeEquals(const Array& other, int64_t start_idx, int64_t end_idx,
                   int64_t other_start_idx,
                   const EqualOptions& = EqualOptions::Defaults()) const;
  bool RangeEquals(const std::shared_ptr<Array>& other, int64_t start_idx,
                   int64_t end_idx, int64_t other_start_idx,
                   const EqualOptions& = EqualOptions::Defaults()) const;

  /// \brief Apply the ArrayVisitor::Visit() method specialized to the array type
  Status Accept(ArrayVisitor* visitor) const;

  /// Construct a zero-copy view of this array with the given type.
  ///
  /// This method checks if the types are layout-compatible.
  /// Nested types are traversed in depth-first order. Data buffers must have
  /// the same item sizes, even though the logical types may be different.
  /// An error is returned if the types are not layout-compatible.
  Result<std::shared_ptr<Array>> View(const std::shared_ptr<DataType>& type) const;

  /// Construct a zero-copy slice of the array with the indicated offset and
  /// length
  ///
  /// \param[in] offset the position of the first element in the constructed
  /// slice
  /// \param[in] length the length of the slice. If there are not enough
  /// elements in the array, the length will be adjusted accordingly
  ///
  /// \return a new object wrapped in std::shared_ptr<Array>
  std::shared_ptr<Array> Slice(int64_t offset, int64_t length) const;

  /// Slice from offset until end of the array
  std::shared_ptr<Array> Slice(int64_t offset) const;

  /// Input-checking variant of Array::Slice
  Result<std::shared_ptr<Array>> SliceSafe(int64_t offset, int64_t length) const;
  /// Input-checking variant of Array::Slice
  Result<std::shared_ptr<Array>> SliceSafe(int64_t offset) const;

  const std::shared_ptr<ArrayData>& data() const { return data_; }

  int num_fields() const { return static_cast<int>(data_->child_data.size()); }

  /// \return PrettyPrint representation of array suitable for debugging
  std::string ToString() const;

  /// \brief Perform cheap validation checks to determine obvious inconsistencies
  /// within the array's internal data.
  ///
  /// This is O(k) where k is the number of descendents.
  ///
  /// \return Status
  Status Validate() const;

  /// \brief Perform extensive validation checks to determine inconsistencies
  /// within the array's internal data.
  ///
  /// This is potentially O(k*n) where k is the number of descendents and n
  /// is the array length.
  ///
  /// \return Status
  Status ValidateFull() const;

 protected:
  Array() = default;
  ARROW_DEFAULT_MOVE_AND_ASSIGN(Array);

  std::shared_ptr<ArrayData> data_;
  const uint8_t* null_bitmap_data_ = NULLPTR;

  /// Protected method for constructors
  void SetData(const std::shared_ptr<ArrayData>& data) {
    if (data->buffers.size() > 0) {
      null_bitmap_data_ = data->GetValuesSafe<uint8_t>(0, /*offset=*/0);
    } else {
      null_bitmap_data_ = NULLPTR;
    }
    data_ = data;
  }

 private:
  ARROW_DISALLOW_COPY_AND_ASSIGN(Array);

  ARROW_FRIEND_EXPORT friend void PrintTo(const Array& x, std::ostream* os);
};

static inline std::ostream& operator<<(std::ostream& os, const Array& x) {
  os << x.ToString();
  return os;
}

/// Base class for non-nested arrays
class ARROW_EXPORT FlatArray : public Array {
 protected:
  using Array::Array;
};

/// Base class for arrays of fixed-size logical types
class ARROW_EXPORT PrimitiveArray : public FlatArray {
 public:
  PrimitiveArray(const std::shared_ptr<DataType>& type, int64_t length,
                 const std::shared_ptr<Buffer>& data,
                 const std::shared_ptr<Buffer>& null_bitmap = NULLPTR,
                 int64_t null_count = kUnknownNullCount, int64_t offset = 0);

  /// Does not account for any slice offset
  std::shared_ptr<Buffer> values() const { return data_->buffers[1]; }

 protected:
  PrimitiveArray() : raw_values_(NULLPTR) {}

  void SetData(const std::shared_ptr<ArrayData>& data) {
    this->Array::SetData(data);
    raw_values_ = data->GetValuesSafe<uint8_t>(1, /*offset=*/0);
  }

  explicit PrimitiveArray(const std::shared_ptr<ArrayData>& data) { SetData(data); }

  const uint8_t* raw_values_;
};

/// Degenerate null type Array
class ARROW_EXPORT NullArray : public FlatArray {
 public:
  using TypeClass = NullType;

  explicit NullArray(const std::shared_ptr<ArrayData>& data) { SetData(data); }
  explicit NullArray(int64_t length);

 private:
  void SetData(const std::shared_ptr<ArrayData>& data) {
    null_bitmap_data_ = NULLPTR;
    data->null_count = data->length;
    data_ = data;
  }
};

}  // namespace arrow

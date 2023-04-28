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

#include "arrow/buffer.h"
#include "arrow/compare.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type.h"
#include "arrow/type_traits.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {

static inline bool is_tensor_supported(Type::type type_id) {
  switch (type_id) {
    case Type::UINT8:
    case Type::INT8:
    case Type::UINT16:
    case Type::INT16:
    case Type::UINT32:
    case Type::INT32:
    case Type::UINT64:
    case Type::INT64:
    case Type::HALF_FLOAT:
    case Type::FLOAT:
    case Type::DOUBLE:
      return true;
    default:
      break;
  }
  return false;
}

namespace internal {

ARROW_EXPORT
Status ComputeRowMajorStrides(const FixedWidthType& type,
                              const std::vector<int64_t>& shape,
                              std::vector<int64_t>* strides);

ARROW_EXPORT
Status ComputeColumnMajorStrides(const FixedWidthType& type,
                                 const std::vector<int64_t>& shape,
                                 std::vector<int64_t>* strides);

ARROW_EXPORT
bool IsTensorStridesContiguous(const std::shared_ptr<DataType>& type,
                               const std::vector<int64_t>& shape,
                               const std::vector<int64_t>& strides);

ARROW_EXPORT
Status ValidateTensorParameters(const std::shared_ptr<DataType>& type,
                                const std::shared_ptr<Buffer>& data,
                                const std::vector<int64_t>& shape,
                                const std::vector<int64_t>& strides,
                                const std::vector<std::string>& dim_names);

}  // namespace internal

class ARROW_EXPORT Tensor {
 public:
  /// \brief Create a Tensor with full parameters
  ///
  /// This factory function will return Status::Invalid when the parameters are
  /// inconsistent
  ///
  /// \param[in] type The data type of the tensor values
  /// \param[in] data The buffer of the tensor content
  /// \param[in] shape The shape of the tensor
  /// \param[in] strides The strides of the tensor
  ///            (if this is empty, the data assumed to be row-major)
  /// \param[in] dim_names The names of the tensor dimensions
  static inline Result<std::shared_ptr<Tensor>> Make(
      const std::shared_ptr<DataType>& type, const std::shared_ptr<Buffer>& data,
      const std::vector<int64_t>& shape, const std::vector<int64_t>& strides = {},
      const std::vector<std::string>& dim_names = {}) {
    ARROW_RETURN_NOT_OK(
        internal::ValidateTensorParameters(type, data, shape, strides, dim_names));
    return std::make_shared<Tensor>(type, data, shape, strides, dim_names);
  }

  virtual ~Tensor() = default;

  /// Constructor with no dimension names or strides, data assumed to be row-major
  Tensor(const std::shared_ptr<DataType>& type, const std::shared_ptr<Buffer>& data,
         const std::vector<int64_t>& shape);

  /// Constructor with non-negative strides
  Tensor(const std::shared_ptr<DataType>& type, const std::shared_ptr<Buffer>& data,
         const std::vector<int64_t>& shape, const std::vector<int64_t>& strides);

  /// Constructor with non-negative strides and dimension names
  Tensor(const std::shared_ptr<DataType>& type, const std::shared_ptr<Buffer>& data,
         const std::vector<int64_t>& shape, const std::vector<int64_t>& strides,
         const std::vector<std::string>& dim_names);

  std::shared_ptr<DataType> type() const { return type_; }
  std::shared_ptr<Buffer> data() const { return data_; }

  const uint8_t* raw_data() const { return data_->data(); }
  uint8_t* raw_mutable_data() { return data_->mutable_data(); }

  const std::vector<int64_t>& shape() const { return shape_; }
  const std::vector<int64_t>& strides() const { return strides_; }

  int ndim() const { return static_cast<int>(shape_.size()); }

  const std::vector<std::string>& dim_names() const { return dim_names_; }
  const std::string& dim_name(int i) const;

  /// Total number of value cells in the tensor
  int64_t size() const;

  /// Return true if the underlying data buffer is mutable
  bool is_mutable() const { return data_->is_mutable(); }

  /// Either row major or column major
  bool is_contiguous() const;

  /// AKA "C order"
  bool is_row_major() const;

  /// AKA "Fortran order"
  bool is_column_major() const;

  Type::type type_id() const;

  bool Equals(const Tensor& other, const EqualOptions& = EqualOptions::Defaults()) const;

  /// Compute the number of non-zero values in the tensor
  Result<int64_t> CountNonZero() const;

  /// Return the offset of the given index on the given strides
  static int64_t CalculateValueOffset(const std::vector<int64_t>& strides,
                                      const std::vector<int64_t>& index) {
    const int64_t n = static_cast<int64_t>(index.size());
    int64_t offset = 0;
    for (int64_t i = 0; i < n; ++i) {
      offset += index[i] * strides[i];
    }
    return offset;
  }

  int64_t CalculateValueOffset(const std::vector<int64_t>& index) const {
    return Tensor::CalculateValueOffset(strides_, index);
  }

  /// Returns the value at the given index without data-type and bounds checks
  template <typename ValueType>
  const typename ValueType::c_type& Value(const std::vector<int64_t>& index) const {
    using c_type = typename ValueType::c_type;
    const int64_t offset = CalculateValueOffset(index);
    const c_type* ptr = reinterpret_cast<const c_type*>(raw_data() + offset);
    return *ptr;
  }

  Status Validate() const {
    return internal::ValidateTensorParameters(type_, data_, shape_, strides_, dim_names_);
  }

 protected:
  Tensor() {}

  std::shared_ptr<DataType> type_;
  std::shared_ptr<Buffer> data_;
  std::vector<int64_t> shape_;
  std::vector<int64_t> strides_;

  /// These names are optional
  std::vector<std::string> dim_names_;

  template <typename SparseIndexType>
  friend class SparseTensorImpl;

 private:
  ARROW_DISALLOW_COPY_AND_ASSIGN(Tensor);
};

template <typename TYPE>
class NumericTensor : public Tensor {
 public:
  using TypeClass = TYPE;
  using value_type = typename TypeClass::c_type;

  /// \brief Create a NumericTensor with full parameters
  ///
  /// This factory function will return Status::Invalid when the parameters are
  /// inconsistent
  ///
  /// \param[in] data The buffer of the tensor content
  /// \param[in] shape The shape of the tensor
  /// \param[in] strides The strides of the tensor
  ///            (if this is empty, the data assumed to be row-major)
  /// \param[in] dim_names The names of the tensor dimensions
  static Result<std::shared_ptr<NumericTensor<TYPE>>> Make(
      const std::shared_ptr<Buffer>& data, const std::vector<int64_t>& shape,
      const std::vector<int64_t>& strides = {},
      const std::vector<std::string>& dim_names = {}) {
    ARROW_RETURN_NOT_OK(internal::ValidateTensorParameters(
        TypeTraits<TYPE>::type_singleton(), data, shape, strides, dim_names));
    return std::make_shared<NumericTensor<TYPE>>(data, shape, strides, dim_names);
  }

  /// Constructor with non-negative strides and dimension names
  NumericTensor(const std::shared_ptr<Buffer>& data, const std::vector<int64_t>& shape,
                const std::vector<int64_t>& strides,
                const std::vector<std::string>& dim_names)
      : Tensor(TypeTraits<TYPE>::type_singleton(), data, shape, strides, dim_names) {}

  /// Constructor with no dimension names or strides, data assumed to be row-major
  NumericTensor(const std::shared_ptr<Buffer>& data, const std::vector<int64_t>& shape)
      : NumericTensor(data, shape, {}, {}) {}

  /// Constructor with non-negative strides
  NumericTensor(const std::shared_ptr<Buffer>& data, const std::vector<int64_t>& shape,
                const std::vector<int64_t>& strides)
      : NumericTensor(data, shape, strides, {}) {}

  const value_type& Value(const std::vector<int64_t>& index) const {
    return Tensor::Value<TypeClass>(index);
  }
};

}  // namespace arrow

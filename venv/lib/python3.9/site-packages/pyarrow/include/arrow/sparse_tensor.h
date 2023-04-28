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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "arrow/buffer.h"
#include "arrow/compare.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/tensor.h"  // IWYU pragma: export
#include "arrow/type.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {

class MemoryPool;

namespace internal {

ARROW_EXPORT
Status CheckSparseIndexMaximumValue(const std::shared_ptr<DataType>& index_value_type,
                                    const std::vector<int64_t>& shape);

}  // namespace internal

// ----------------------------------------------------------------------
// SparseIndex class

struct SparseTensorFormat {
  /// EXPERIMENTAL: The index format type of SparseTensor
  enum type {
    /// Coordinate list (COO) format.
    COO,
    /// Compressed sparse row (CSR) format.
    CSR,
    /// Compressed sparse column (CSC) format.
    CSC,
    /// Compressed sparse fiber (CSF) format.
    CSF
  };
};

/// \brief EXPERIMENTAL: The base class for the index of a sparse tensor
///
/// SparseIndex describes where the non-zero elements are within a SparseTensor.
///
/// There are several ways to represent this.  The format_id is used to
/// distinguish what kind of representation is used.  Each possible value of
/// format_id must have only one corresponding concrete subclass of SparseIndex.
class ARROW_EXPORT SparseIndex {
 public:
  explicit SparseIndex(SparseTensorFormat::type format_id) : format_id_(format_id) {}

  virtual ~SparseIndex() = default;

  /// \brief Return the identifier of the format type
  SparseTensorFormat::type format_id() const { return format_id_; }

  /// \brief Return the number of non zero values in the sparse tensor related
  /// to this sparse index
  virtual int64_t non_zero_length() const = 0;

  /// \brief Return the string representation of the sparse index
  virtual std::string ToString() const = 0;

  virtual Status ValidateShape(const std::vector<int64_t>& shape) const;

 protected:
  const SparseTensorFormat::type format_id_;
};

namespace internal {
template <typename SparseIndexType>
class SparseIndexBase : public SparseIndex {
 public:
  SparseIndexBase() : SparseIndex(SparseIndexType::format_id) {}
};
}  // namespace internal

// ----------------------------------------------------------------------
// SparseCOOIndex class

/// \brief EXPERIMENTAL: The index data for a COO sparse tensor
///
/// A COO sparse index manages the location of its non-zero values by their
/// coordinates.
class ARROW_EXPORT SparseCOOIndex : public internal::SparseIndexBase<SparseCOOIndex> {
 public:
  static constexpr SparseTensorFormat::type format_id = SparseTensorFormat::COO;

  /// \brief Make SparseCOOIndex from a coords tensor and canonicality
  static Result<std::shared_ptr<SparseCOOIndex>> Make(
      const std::shared_ptr<Tensor>& coords, bool is_canonical);

  /// \brief Make SparseCOOIndex from a coords tensor with canonicality auto-detection
  static Result<std::shared_ptr<SparseCOOIndex>> Make(
      const std::shared_ptr<Tensor>& coords);

  /// \brief Make SparseCOOIndex from raw properties with canonicality auto-detection
  static Result<std::shared_ptr<SparseCOOIndex>> Make(
      const std::shared_ptr<DataType>& indices_type,
      const std::vector<int64_t>& indices_shape,
      const std::vector<int64_t>& indices_strides, std::shared_ptr<Buffer> indices_data);

  /// \brief Make SparseCOOIndex from raw properties
  static Result<std::shared_ptr<SparseCOOIndex>> Make(
      const std::shared_ptr<DataType>& indices_type,
      const std::vector<int64_t>& indices_shape,
      const std::vector<int64_t>& indices_strides, std::shared_ptr<Buffer> indices_data,
      bool is_canonical);

  /// \brief Make SparseCOOIndex from sparse tensor's shape properties and data
  /// with canonicality auto-detection
  ///
  /// The indices_data should be in row-major (C-like) order.  If not,
  /// use the raw properties constructor.
  static Result<std::shared_ptr<SparseCOOIndex>> Make(
      const std::shared_ptr<DataType>& indices_type, const std::vector<int64_t>& shape,
      int64_t non_zero_length, std::shared_ptr<Buffer> indices_data);

  /// \brief Make SparseCOOIndex from sparse tensor's shape properties and data
  ///
  /// The indices_data should be in row-major (C-like) order.  If not,
  /// use the raw properties constructor.
  static Result<std::shared_ptr<SparseCOOIndex>> Make(
      const std::shared_ptr<DataType>& indices_type, const std::vector<int64_t>& shape,
      int64_t non_zero_length, std::shared_ptr<Buffer> indices_data, bool is_canonical);

  /// \brief Construct SparseCOOIndex from column-major NumericTensor
  explicit SparseCOOIndex(const std::shared_ptr<Tensor>& coords, bool is_canonical);

  /// \brief Return a tensor that has the coordinates of the non-zero values
  ///
  /// The returned tensor is a N x D tensor where N is the number of non-zero
  /// values and D is the number of dimensions in the logical data.
  /// The column at index `i` is a D-tuple of coordinates indicating that the
  /// logical value at those coordinates should be found at physical index `i`.
  const std::shared_ptr<Tensor>& indices() const { return coords_; }

  /// \brief Return the number of non zero values in the sparse tensor related
  /// to this sparse index
  int64_t non_zero_length() const override { return coords_->shape()[0]; }

  /// \brief Return whether a sparse tensor index is canonical, or not.
  /// If a sparse tensor index is canonical, it is sorted in the lexicographical order,
  /// and the corresponding sparse tensor doesn't have duplicated entries.
  bool is_canonical() const { return is_canonical_; }

  /// \brief Return a string representation of the sparse index
  std::string ToString() const override;

  /// \brief Return whether the COO indices are equal
  bool Equals(const SparseCOOIndex& other) const {
    return indices()->Equals(*other.indices());
  }

  inline Status ValidateShape(const std::vector<int64_t>& shape) const override {
    ARROW_RETURN_NOT_OK(SparseIndex::ValidateShape(shape));

    if (static_cast<size_t>(coords_->shape()[1]) == shape.size()) {
      return Status::OK();
    }

    return Status::Invalid(
        "shape length is inconsistent with the coords matrix in COO index");
  }

 protected:
  std::shared_ptr<Tensor> coords_;
  bool is_canonical_;
};

namespace internal {

/// EXPERIMENTAL: The axis to be compressed
enum class SparseMatrixCompressedAxis : char {
  /// The value for CSR matrix
  ROW,
  /// The value for CSC matrix
  COLUMN
};

ARROW_EXPORT
Status ValidateSparseCSXIndex(const std::shared_ptr<DataType>& indptr_type,
                              const std::shared_ptr<DataType>& indices_type,
                              const std::vector<int64_t>& indptr_shape,
                              const std::vector<int64_t>& indices_shape,
                              char const* type_name);

ARROW_EXPORT
void CheckSparseCSXIndexValidity(const std::shared_ptr<DataType>& indptr_type,
                                 const std::shared_ptr<DataType>& indices_type,
                                 const std::vector<int64_t>& indptr_shape,
                                 const std::vector<int64_t>& indices_shape,
                                 char const* type_name);

template <typename SparseIndexType, SparseMatrixCompressedAxis COMPRESSED_AXIS>
class SparseCSXIndex : public SparseIndexBase<SparseIndexType> {
 public:
  static constexpr SparseMatrixCompressedAxis kCompressedAxis = COMPRESSED_AXIS;

  /// \brief Make a subclass of SparseCSXIndex from raw properties
  static Result<std::shared_ptr<SparseIndexType>> Make(
      const std::shared_ptr<DataType>& indptr_type,
      const std::shared_ptr<DataType>& indices_type,
      const std::vector<int64_t>& indptr_shape, const std::vector<int64_t>& indices_shape,
      std::shared_ptr<Buffer> indptr_data, std::shared_ptr<Buffer> indices_data) {
    ARROW_RETURN_NOT_OK(ValidateSparseCSXIndex(indptr_type, indices_type, indptr_shape,
                                               indices_shape,
                                               SparseIndexType::kTypeName));
    return std::make_shared<SparseIndexType>(
        std::make_shared<Tensor>(indptr_type, indptr_data, indptr_shape),
        std::make_shared<Tensor>(indices_type, indices_data, indices_shape));
  }

  /// \brief Make a subclass of SparseCSXIndex from raw properties
  static Result<std::shared_ptr<SparseIndexType>> Make(
      const std::shared_ptr<DataType>& indices_type,
      const std::vector<int64_t>& indptr_shape, const std::vector<int64_t>& indices_shape,
      std::shared_ptr<Buffer> indptr_data, std::shared_ptr<Buffer> indices_data) {
    return Make(indices_type, indices_type, indptr_shape, indices_shape, indptr_data,
                indices_data);
  }

  /// \brief Make a subclass of SparseCSXIndex from sparse tensor's shape properties and
  /// data
  static Result<std::shared_ptr<SparseIndexType>> Make(
      const std::shared_ptr<DataType>& indptr_type,
      const std::shared_ptr<DataType>& indices_type, const std::vector<int64_t>& shape,
      int64_t non_zero_length, std::shared_ptr<Buffer> indptr_data,
      std::shared_ptr<Buffer> indices_data) {
    std::vector<int64_t> indptr_shape({shape[0] + 1});
    std::vector<int64_t> indices_shape({non_zero_length});
    return Make(indptr_type, indices_type, indptr_shape, indices_shape, indptr_data,
                indices_data);
  }

  /// \brief Make a subclass of SparseCSXIndex from sparse tensor's shape properties and
  /// data
  static Result<std::shared_ptr<SparseIndexType>> Make(
      const std::shared_ptr<DataType>& indices_type, const std::vector<int64_t>& shape,
      int64_t non_zero_length, std::shared_ptr<Buffer> indptr_data,
      std::shared_ptr<Buffer> indices_data) {
    return Make(indices_type, indices_type, shape, non_zero_length, indptr_data,
                indices_data);
  }

  /// \brief Construct SparseCSXIndex from two index vectors
  explicit SparseCSXIndex(const std::shared_ptr<Tensor>& indptr,
                          const std::shared_ptr<Tensor>& indices)
      : SparseIndexBase<SparseIndexType>(), indptr_(indptr), indices_(indices) {
    CheckSparseCSXIndexValidity(indptr_->type(), indices_->type(), indptr_->shape(),
                                indices_->shape(), SparseIndexType::kTypeName);
  }

  /// \brief Return a 1D tensor of indptr vector
  const std::shared_ptr<Tensor>& indptr() const { return indptr_; }

  /// \brief Return a 1D tensor of indices vector
  const std::shared_ptr<Tensor>& indices() const { return indices_; }

  /// \brief Return the number of non zero values in the sparse tensor related
  /// to this sparse index
  int64_t non_zero_length() const override { return indices_->shape()[0]; }

  /// \brief Return a string representation of the sparse index
  std::string ToString() const override {
    return std::string(SparseIndexType::kTypeName);
  }

  /// \brief Return whether the CSR indices are equal
  bool Equals(const SparseIndexType& other) const {
    return indptr()->Equals(*other.indptr()) && indices()->Equals(*other.indices());
  }

  inline Status ValidateShape(const std::vector<int64_t>& shape) const override {
    ARROW_RETURN_NOT_OK(SparseIndex::ValidateShape(shape));

    if (shape.size() < 2) {
      return Status::Invalid("shape length is too short");
    }

    if (shape.size() > 2) {
      return Status::Invalid("shape length is too long");
    }

    if (indptr_->shape()[0] == shape[static_cast<int64_t>(kCompressedAxis)] + 1) {
      return Status::OK();
    }

    return Status::Invalid("shape length is inconsistent with the ", ToString());
  }

 protected:
  std::shared_ptr<Tensor> indptr_;
  std::shared_ptr<Tensor> indices_;
};

}  // namespace internal

// ----------------------------------------------------------------------
// SparseCSRIndex class

/// \brief EXPERIMENTAL: The index data for a CSR sparse matrix
///
/// A CSR sparse index manages the location of its non-zero values by two
/// vectors.
///
/// The first vector, called indptr, represents the range of the rows; the i-th
/// row spans from indptr[i] to indptr[i+1] in the corresponding value vector.
/// So the length of an indptr vector is the number of rows + 1.
///
/// The other vector, called indices, represents the column indices of the
/// corresponding non-zero values.  So the length of an indices vector is same
/// as the number of non-zero-values.
class ARROW_EXPORT SparseCSRIndex
    : public internal::SparseCSXIndex<SparseCSRIndex,
                                      internal::SparseMatrixCompressedAxis::ROW> {
 public:
  using BaseClass =
      internal::SparseCSXIndex<SparseCSRIndex, internal::SparseMatrixCompressedAxis::ROW>;

  static constexpr SparseTensorFormat::type format_id = SparseTensorFormat::CSR;
  static constexpr char const* kTypeName = "SparseCSRIndex";

  using SparseCSXIndex::kCompressedAxis;
  using SparseCSXIndex::Make;
  using SparseCSXIndex::SparseCSXIndex;
};

// ----------------------------------------------------------------------
// SparseCSCIndex class

/// \brief EXPERIMENTAL: The index data for a CSC sparse matrix
///
/// A CSC sparse index manages the location of its non-zero values by two
/// vectors.
///
/// The first vector, called indptr, represents the range of the column; the i-th
/// column spans from indptr[i] to indptr[i+1] in the corresponding value vector.
/// So the length of an indptr vector is the number of columns + 1.
///
/// The other vector, called indices, represents the row indices of the
/// corresponding non-zero values.  So the length of an indices vector is same
/// as the number of non-zero-values.
class ARROW_EXPORT SparseCSCIndex
    : public internal::SparseCSXIndex<SparseCSCIndex,
                                      internal::SparseMatrixCompressedAxis::COLUMN> {
 public:
  using BaseClass =
      internal::SparseCSXIndex<SparseCSCIndex,
                               internal::SparseMatrixCompressedAxis::COLUMN>;

  static constexpr SparseTensorFormat::type format_id = SparseTensorFormat::CSC;
  static constexpr char const* kTypeName = "SparseCSCIndex";

  using SparseCSXIndex::kCompressedAxis;
  using SparseCSXIndex::Make;
  using SparseCSXIndex::SparseCSXIndex;
};

// ----------------------------------------------------------------------
// SparseCSFIndex class

/// \brief EXPERIMENTAL: The index data for a CSF sparse tensor
///
/// A CSF sparse index manages the location of its non-zero values by set of
/// prefix trees. Each path from a root to leaf forms one tensor non-zero index.
/// CSF is implemented with three vectors.
///
/// Vectors inptr and indices contain N-1 and N buffers respectively, where N is the
/// number of dimensions. Axis_order is a vector of integers of length N. Indptr and
/// indices describe the set of prefix trees. Trees traverse dimensions in order given by
/// axis_order.
class ARROW_EXPORT SparseCSFIndex : public internal::SparseIndexBase<SparseCSFIndex> {
 public:
  static constexpr SparseTensorFormat::type format_id = SparseTensorFormat::CSF;
  static constexpr char const* kTypeName = "SparseCSFIndex";

  /// \brief Make SparseCSFIndex from raw properties
  static Result<std::shared_ptr<SparseCSFIndex>> Make(
      const std::shared_ptr<DataType>& indptr_type,
      const std::shared_ptr<DataType>& indices_type,
      const std::vector<int64_t>& indices_shapes, const std::vector<int64_t>& axis_order,
      const std::vector<std::shared_ptr<Buffer>>& indptr_data,
      const std::vector<std::shared_ptr<Buffer>>& indices_data);

  /// \brief Make SparseCSFIndex from raw properties
  static Result<std::shared_ptr<SparseCSFIndex>> Make(
      const std::shared_ptr<DataType>& indices_type,
      const std::vector<int64_t>& indices_shapes, const std::vector<int64_t>& axis_order,
      const std::vector<std::shared_ptr<Buffer>>& indptr_data,
      const std::vector<std::shared_ptr<Buffer>>& indices_data) {
    return Make(indices_type, indices_type, indices_shapes, axis_order, indptr_data,
                indices_data);
  }

  /// \brief Construct SparseCSFIndex from two index vectors
  explicit SparseCSFIndex(const std::vector<std::shared_ptr<Tensor>>& indptr,
                          const std::vector<std::shared_ptr<Tensor>>& indices,
                          const std::vector<int64_t>& axis_order);

  /// \brief Return a 1D vector of indptr tensors
  const std::vector<std::shared_ptr<Tensor>>& indptr() const { return indptr_; }

  /// \brief Return a 1D vector of indices tensors
  const std::vector<std::shared_ptr<Tensor>>& indices() const { return indices_; }

  /// \brief Return a 1D vector specifying the order of axes
  const std::vector<int64_t>& axis_order() const { return axis_order_; }

  /// \brief Return the number of non zero values in the sparse tensor related
  /// to this sparse index
  int64_t non_zero_length() const override { return indices_.back()->shape()[0]; }

  /// \brief Return a string representation of the sparse index
  std::string ToString() const override;

  /// \brief Return whether the CSF indices are equal
  bool Equals(const SparseCSFIndex& other) const;

 protected:
  std::vector<std::shared_ptr<Tensor>> indptr_;
  std::vector<std::shared_ptr<Tensor>> indices_;
  std::vector<int64_t> axis_order_;
};

// ----------------------------------------------------------------------
// SparseTensor class

/// \brief EXPERIMENTAL: The base class of sparse tensor container
class ARROW_EXPORT SparseTensor {
 public:
  virtual ~SparseTensor() = default;

  SparseTensorFormat::type format_id() const { return sparse_index_->format_id(); }

  /// \brief Return a value type of the sparse tensor
  std::shared_ptr<DataType> type() const { return type_; }

  /// \brief Return a buffer that contains the value vector of the sparse tensor
  std::shared_ptr<Buffer> data() const { return data_; }

  /// \brief Return an immutable raw data pointer
  const uint8_t* raw_data() const { return data_->data(); }

  /// \brief Return a mutable raw data pointer
  uint8_t* raw_mutable_data() const { return data_->mutable_data(); }

  /// \brief Return a shape vector of the sparse tensor
  const std::vector<int64_t>& shape() const { return shape_; }

  /// \brief Return a sparse index of the sparse tensor
  const std::shared_ptr<SparseIndex>& sparse_index() const { return sparse_index_; }

  /// \brief Return a number of dimensions of the sparse tensor
  int ndim() const { return static_cast<int>(shape_.size()); }

  /// \brief Return a vector of dimension names
  const std::vector<std::string>& dim_names() const { return dim_names_; }

  /// \brief Return the name of the i-th dimension
  const std::string& dim_name(int i) const;

  /// \brief Total number of value cells in the sparse tensor
  int64_t size() const;

  /// \brief Return true if the underlying data buffer is mutable
  bool is_mutable() const { return data_->is_mutable(); }

  /// \brief Total number of non-zero cells in the sparse tensor
  int64_t non_zero_length() const {
    return sparse_index_ ? sparse_index_->non_zero_length() : 0;
  }

  /// \brief Return whether sparse tensors are equal
  bool Equals(const SparseTensor& other,
              const EqualOptions& = EqualOptions::Defaults()) const;

  /// \brief Return dense representation of sparse tensor as tensor
  ///
  /// The returned Tensor has row-major order (C-like).
  Result<std::shared_ptr<Tensor>> ToTensor(MemoryPool* pool) const;
  Result<std::shared_ptr<Tensor>> ToTensor() const {
    return ToTensor(default_memory_pool());
  }

 protected:
  // Constructor with all attributes
  SparseTensor(const std::shared_ptr<DataType>& type, const std::shared_ptr<Buffer>& data,
               const std::vector<int64_t>& shape,
               const std::shared_ptr<SparseIndex>& sparse_index,
               const std::vector<std::string>& dim_names);

  std::shared_ptr<DataType> type_;
  std::shared_ptr<Buffer> data_;
  std::vector<int64_t> shape_;
  std::shared_ptr<SparseIndex> sparse_index_;

  // These names are optional
  std::vector<std::string> dim_names_;
};

// ----------------------------------------------------------------------
// SparseTensorImpl class

namespace internal {

ARROW_EXPORT
Status MakeSparseTensorFromTensor(const Tensor& tensor,
                                  SparseTensorFormat::type sparse_format_id,
                                  const std::shared_ptr<DataType>& index_value_type,
                                  MemoryPool* pool,
                                  std::shared_ptr<SparseIndex>* out_sparse_index,
                                  std::shared_ptr<Buffer>* out_data);

}  // namespace internal

/// \brief EXPERIMENTAL: Concrete sparse tensor implementation classes with sparse index
/// type
template <typename SparseIndexType>
class SparseTensorImpl : public SparseTensor {
 public:
  virtual ~SparseTensorImpl() = default;

  /// \brief Construct a sparse tensor from physical data buffer and logical index
  SparseTensorImpl(const std::shared_ptr<SparseIndexType>& sparse_index,
                   const std::shared_ptr<DataType>& type,
                   const std::shared_ptr<Buffer>& data, const std::vector<int64_t>& shape,
                   const std::vector<std::string>& dim_names)
      : SparseTensor(type, data, shape, sparse_index, dim_names) {}

  /// \brief Construct an empty sparse tensor
  SparseTensorImpl(const std::shared_ptr<DataType>& type,
                   const std::vector<int64_t>& shape,
                   const std::vector<std::string>& dim_names = {})
      : SparseTensorImpl(NULLPTR, type, NULLPTR, shape, dim_names) {}

  /// \brief Create a SparseTensor with full parameters
  static inline Result<std::shared_ptr<SparseTensorImpl<SparseIndexType>>> Make(
      const std::shared_ptr<SparseIndexType>& sparse_index,
      const std::shared_ptr<DataType>& type, const std::shared_ptr<Buffer>& data,
      const std::vector<int64_t>& shape, const std::vector<std::string>& dim_names) {
    if (!is_tensor_supported(type->id())) {
      return Status::Invalid(type->ToString(),
                             " is not valid data type for a sparse tensor");
    }
    ARROW_RETURN_NOT_OK(sparse_index->ValidateShape(shape));
    if (dim_names.size() > 0 && dim_names.size() != shape.size()) {
      return Status::Invalid("dim_names length is inconsistent with shape");
    }
    return std::make_shared<SparseTensorImpl<SparseIndexType>>(sparse_index, type, data,
                                                               shape, dim_names);
  }

  /// \brief Create a sparse tensor from a dense tensor
  ///
  /// The dense tensor is re-encoded as a sparse index and a physical
  /// data buffer for the non-zero value.
  static inline Result<std::shared_ptr<SparseTensorImpl<SparseIndexType>>> Make(
      const Tensor& tensor, const std::shared_ptr<DataType>& index_value_type,
      MemoryPool* pool = default_memory_pool()) {
    std::shared_ptr<SparseIndex> sparse_index;
    std::shared_ptr<Buffer> data;
    ARROW_RETURN_NOT_OK(internal::MakeSparseTensorFromTensor(
        tensor, SparseIndexType::format_id, index_value_type, pool, &sparse_index,
        &data));
    return std::make_shared<SparseTensorImpl<SparseIndexType>>(
        internal::checked_pointer_cast<SparseIndexType>(sparse_index), tensor.type(),
        data, tensor.shape(), tensor.dim_names_);
  }

  static inline Result<std::shared_ptr<SparseTensorImpl<SparseIndexType>>> Make(
      const Tensor& tensor, MemoryPool* pool = default_memory_pool()) {
    return Make(tensor, int64(), pool);
  }

 private:
  ARROW_DISALLOW_COPY_AND_ASSIGN(SparseTensorImpl);
};

/// \brief EXPERIMENTAL: Type alias for COO sparse tensor
using SparseCOOTensor = SparseTensorImpl<SparseCOOIndex>;

/// \brief EXPERIMENTAL: Type alias for CSR sparse matrix
using SparseCSRMatrix = SparseTensorImpl<SparseCSRIndex>;

/// \brief EXPERIMENTAL: Type alias for CSC sparse matrix
using SparseCSCMatrix = SparseTensorImpl<SparseCSCIndex>;

/// \brief EXPERIMENTAL: Type alias for CSF sparse matrix
using SparseCSFTensor = SparseTensorImpl<SparseCSFIndex>;

}  // namespace arrow

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

#include "arrow/sparse_tensor.h"  // IWYU pragma: export

#include <memory>

namespace arrow {
namespace internal {

struct SparseTensorConverterMixin {
  static bool IsNonZero(const uint8_t val) { return val != 0; }

  static void AssignIndex(uint8_t* indices, int64_t val, const int elsize);

  static int64_t GetIndexValue(const uint8_t* value_ptr, const int elsize);
};

Status MakeSparseCOOTensorFromTensor(const Tensor& tensor,
                                     const std::shared_ptr<DataType>& index_value_type,
                                     MemoryPool* pool,
                                     std::shared_ptr<SparseIndex>* out_sparse_index,
                                     std::shared_ptr<Buffer>* out_data);

Status MakeSparseCSXMatrixFromTensor(SparseMatrixCompressedAxis axis,
                                     const Tensor& tensor,
                                     const std::shared_ptr<DataType>& index_value_type,
                                     MemoryPool* pool,
                                     std::shared_ptr<SparseIndex>* out_sparse_index,
                                     std::shared_ptr<Buffer>* out_data);

Status MakeSparseCSFTensorFromTensor(const Tensor& tensor,
                                     const std::shared_ptr<DataType>& index_value_type,
                                     MemoryPool* pool,
                                     std::shared_ptr<SparseIndex>* out_sparse_index,
                                     std::shared_ptr<Buffer>* out_data);

Result<std::shared_ptr<Tensor>> MakeTensorFromSparseCOOTensor(
    MemoryPool* pool, const SparseCOOTensor* sparse_tensor);

Result<std::shared_ptr<Tensor>> MakeTensorFromSparseCSRMatrix(
    MemoryPool* pool, const SparseCSRMatrix* sparse_tensor);

Result<std::shared_ptr<Tensor>> MakeTensorFromSparseCSCMatrix(
    MemoryPool* pool, const SparseCSCMatrix* sparse_tensor);

Result<std::shared_ptr<Tensor>> MakeTensorFromSparseCSFTensor(
    MemoryPool* pool, const SparseCSFTensor* sparse_tensor);

}  // namespace internal
}  // namespace arrow

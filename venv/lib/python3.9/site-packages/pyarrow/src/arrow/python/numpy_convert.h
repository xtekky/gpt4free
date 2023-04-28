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

// Functions for converting between pandas's NumPy-based data representation
// and Arrow data structures

#pragma once

#include "arrow/python/platform.h"

#include <memory>
#include <string>
#include <vector>

#include "arrow/buffer.h"
#include "arrow/sparse_tensor.h"
#include "arrow/python/visibility.h"

namespace arrow {

class DataType;
class MemoryPool;
class Status;
class Tensor;

namespace py {

class ARROW_PYTHON_EXPORT NumPyBuffer : public Buffer {
 public:
  explicit NumPyBuffer(PyObject* arr);
  virtual ~NumPyBuffer();

 private:
  PyObject* arr_;
};

ARROW_PYTHON_EXPORT
Status NumPyDtypeToArrow(PyObject* dtype, std::shared_ptr<DataType>* out);
ARROW_PYTHON_EXPORT
Status NumPyDtypeToArrow(PyArray_Descr* descr, std::shared_ptr<DataType>* out);

ARROW_PYTHON_EXPORT Status NdarrayToTensor(MemoryPool* pool, PyObject* ao,
                                           const std::vector<std::string>& dim_names,
                                           std::shared_ptr<Tensor>* out);

ARROW_PYTHON_EXPORT Status TensorToNdarray(const std::shared_ptr<Tensor>& tensor,
                                           PyObject* base, PyObject** out);

ARROW_PYTHON_EXPORT Status
SparseCOOTensorToNdarray(const std::shared_ptr<SparseCOOTensor>& sparse_tensor,
                         PyObject* base, PyObject** out_data, PyObject** out_coords);

Status SparseCSXMatrixToNdarray(const std::shared_ptr<SparseTensor>& sparse_tensor,
                                PyObject* base, PyObject** out_data,
                                PyObject** out_indptr, PyObject** out_indices);

ARROW_PYTHON_EXPORT Status SparseCSRMatrixToNdarray(
    const std::shared_ptr<SparseCSRMatrix>& sparse_tensor, PyObject* base,
    PyObject** out_data, PyObject** out_indptr, PyObject** out_indices);

ARROW_PYTHON_EXPORT Status SparseCSCMatrixToNdarray(
    const std::shared_ptr<SparseCSCMatrix>& sparse_tensor, PyObject* base,
    PyObject** out_data, PyObject** out_indptr, PyObject** out_indices);

ARROW_PYTHON_EXPORT Status SparseCSFTensorToNdarray(
    const std::shared_ptr<SparseCSFTensor>& sparse_tensor, PyObject* base,
    PyObject** out_data, PyObject** out_indptr, PyObject** out_indices);

ARROW_PYTHON_EXPORT Status NdarraysToSparseCOOTensor(
    MemoryPool* pool, PyObject* data_ao, PyObject* coords_ao,
    const std::vector<int64_t>& shape, const std::vector<std::string>& dim_names,
    std::shared_ptr<SparseCOOTensor>* out);

ARROW_PYTHON_EXPORT Status NdarraysToSparseCSRMatrix(
    MemoryPool* pool, PyObject* data_ao, PyObject* indptr_ao, PyObject* indices_ao,
    const std::vector<int64_t>& shape, const std::vector<std::string>& dim_names,
    std::shared_ptr<SparseCSRMatrix>* out);

ARROW_PYTHON_EXPORT Status NdarraysToSparseCSCMatrix(
    MemoryPool* pool, PyObject* data_ao, PyObject* indptr_ao, PyObject* indices_ao,
    const std::vector<int64_t>& shape, const std::vector<std::string>& dim_names,
    std::shared_ptr<SparseCSCMatrix>* out);

ARROW_PYTHON_EXPORT Status NdarraysToSparseCSFTensor(
    MemoryPool* pool, PyObject* data_ao, PyObject* indptr_ao, PyObject* indices_ao,
    const std::vector<int64_t>& shape, const std::vector<int64_t>& axis_order,
    const std::vector<std::string>& dim_names, std::shared_ptr<SparseCSFTensor>* out);

ARROW_PYTHON_EXPORT Status
TensorToSparseCOOTensor(const std::shared_ptr<Tensor>& tensor,
                        std::shared_ptr<SparseCOOTensor>* csparse_tensor);

ARROW_PYTHON_EXPORT Status
TensorToSparseCSRMatrix(const std::shared_ptr<Tensor>& tensor,
                        std::shared_ptr<SparseCSRMatrix>* csparse_tensor);

ARROW_PYTHON_EXPORT Status
TensorToSparseCSCMatrix(const std::shared_ptr<Tensor>& tensor,
                        std::shared_ptr<SparseCSCMatrix>* csparse_tensor);

ARROW_PYTHON_EXPORT Status
TensorToSparseCSFTensor(const std::shared_ptr<Tensor>& tensor,
                        std::shared_ptr<SparseCSFTensor>* csparse_tensor);

}  // namespace py
}  // namespace arrow

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

#include "arrow/python/platform.h"

#include <memory>

#include "arrow/python/visibility.h"

#include "arrow/sparse_tensor.h"

// Work around ARROW-2317 (C linkage warning from Cython)
extern "C++" {

namespace arrow {

class Array;
class Buffer;
class DataType;
class Field;
class RecordBatch;
class Schema;
class Status;
class Table;
class Tensor;

namespace py {

// Returns 0 on success, -1 on error.
ARROW_PYTHON_EXPORT int import_pyarrow();

#define DECLARE_WRAP_FUNCTIONS(FUNC_SUFFIX, TYPE_NAME)                         \
  ARROW_PYTHON_EXPORT bool is_##FUNC_SUFFIX(PyObject*);                        \
  ARROW_PYTHON_EXPORT Result<std::shared_ptr<TYPE_NAME>> unwrap_##FUNC_SUFFIX( \
      PyObject*);                                                              \
  ARROW_PYTHON_EXPORT PyObject* wrap_##FUNC_SUFFIX(const std::shared_ptr<TYPE_NAME>&);

DECLARE_WRAP_FUNCTIONS(buffer, Buffer)

DECLARE_WRAP_FUNCTIONS(data_type, DataType)
DECLARE_WRAP_FUNCTIONS(field, Field)
DECLARE_WRAP_FUNCTIONS(schema, Schema)

DECLARE_WRAP_FUNCTIONS(scalar, Scalar)

DECLARE_WRAP_FUNCTIONS(array, Array)
DECLARE_WRAP_FUNCTIONS(chunked_array, ChunkedArray)

DECLARE_WRAP_FUNCTIONS(sparse_coo_tensor, SparseCOOTensor)
DECLARE_WRAP_FUNCTIONS(sparse_csc_matrix, SparseCSCMatrix)
DECLARE_WRAP_FUNCTIONS(sparse_csf_tensor, SparseCSFTensor)
DECLARE_WRAP_FUNCTIONS(sparse_csr_matrix, SparseCSRMatrix)
DECLARE_WRAP_FUNCTIONS(tensor, Tensor)

DECLARE_WRAP_FUNCTIONS(batch, RecordBatch)
DECLARE_WRAP_FUNCTIONS(table, Table)

#undef DECLARE_WRAP_FUNCTIONS

namespace internal {

ARROW_PYTHON_EXPORT int check_status(const Status& status);

}  // namespace internal
}  // namespace py
}  // namespace arrow

}  // extern "C++"

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

#include "arrow/python/pyarrow.h"

#include <memory>
#include <utility>

#include "arrow/array.h"
#include "arrow/table.h"
#include "arrow/tensor.h"
#include "arrow/type.h"

#include "arrow/python/common.h"
#include "arrow/python/datetime.h"
namespace {
#include "arrow/python/pyarrow_api.h"
}

namespace arrow {
namespace py {

static Status UnwrapError(PyObject* obj, const char* expected_type) {
  return Status::TypeError("Could not unwrap ", expected_type,
                           " from Python object of type '", Py_TYPE(obj)->tp_name, "'");
}

int import_pyarrow() {
#ifdef PYPY_VERSION
  PyDateTime_IMPORT;
#else
  internal::InitDatetime();
#endif
  return ::import_pyarrow__lib();
}

#define DEFINE_WRAP_FUNCTIONS(FUNC_SUFFIX, TYPE_NAME)                                   \
  bool is_##FUNC_SUFFIX(PyObject* obj) { return ::pyarrow_is_##FUNC_SUFFIX(obj) != 0; } \
                                                                                        \
  PyObject* wrap_##FUNC_SUFFIX(const std::shared_ptr<TYPE_NAME>& src) {                 \
    return ::pyarrow_wrap_##FUNC_SUFFIX(src);                                           \
  }                                                                                     \
  Result<std::shared_ptr<TYPE_NAME>> unwrap_##FUNC_SUFFIX(PyObject* obj) {              \
    auto out = ::pyarrow_unwrap_##FUNC_SUFFIX(obj);                                     \
    if (out) {                                                                          \
      return std::move(out);                                                            \
    } else {                                                                            \
      return UnwrapError(obj, #TYPE_NAME);                                              \
    }                                                                                   \
  }

DEFINE_WRAP_FUNCTIONS(buffer, Buffer)

DEFINE_WRAP_FUNCTIONS(data_type, DataType)
DEFINE_WRAP_FUNCTIONS(field, Field)
DEFINE_WRAP_FUNCTIONS(schema, Schema)

DEFINE_WRAP_FUNCTIONS(scalar, Scalar)

DEFINE_WRAP_FUNCTIONS(array, Array)
DEFINE_WRAP_FUNCTIONS(chunked_array, ChunkedArray)

DEFINE_WRAP_FUNCTIONS(sparse_coo_tensor, SparseCOOTensor)
DEFINE_WRAP_FUNCTIONS(sparse_csc_matrix, SparseCSCMatrix)
DEFINE_WRAP_FUNCTIONS(sparse_csf_tensor, SparseCSFTensor)
DEFINE_WRAP_FUNCTIONS(sparse_csr_matrix, SparseCSRMatrix)
DEFINE_WRAP_FUNCTIONS(tensor, Tensor)

DEFINE_WRAP_FUNCTIONS(batch, RecordBatch)
DEFINE_WRAP_FUNCTIONS(table, Table)

#undef DEFINE_WRAP_FUNCTIONS

namespace internal {

int check_status(const Status& status) { return ::pyarrow_internal_check_status(status); }

}  // namespace internal
}  // namespace py
}  // namespace arrow

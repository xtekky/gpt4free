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

#include "arrow/compute/exec.h"
#include "arrow/compute/function.h"
#include "arrow/compute/registry.h"
#include "arrow/python/platform.h"

#include "arrow/python/common.h"
#include "arrow/python/pyarrow.h"
#include "arrow/python/visibility.h"

namespace arrow {

namespace py {

// TODO: TODO(ARROW-16041): UDF Options are not exposed to the Python
// users. This feature will be included when extending to provide advanced
// options for the users.
struct ARROW_PYTHON_EXPORT ScalarUdfOptions {
  std::string func_name;
  compute::Arity arity;
  compute::FunctionDoc func_doc;
  std::vector<std::shared_ptr<DataType>> input_types;
  std::shared_ptr<DataType> output_type;
};

/// \brief A context passed as the first argument of scalar UDF functions.
struct ARROW_PYTHON_EXPORT ScalarUdfContext {
  MemoryPool* pool;
  int64_t batch_length;
};

using ScalarUdfWrapperCallback = std::function<PyObject*(
    PyObject* user_function, const ScalarUdfContext& context, PyObject* inputs)>;

/// \brief register a Scalar user-defined-function from Python
Status ARROW_PYTHON_EXPORT RegisterScalarFunction(PyObject* user_function,
                                                  ScalarUdfWrapperCallback wrapper,
                                                  const ScalarUdfOptions& options);

}  // namespace py

}  // namespace arrow

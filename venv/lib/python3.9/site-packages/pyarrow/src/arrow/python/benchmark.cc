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

#include "arrow/python/benchmark.h"
#include "arrow/python/helpers.h"

namespace arrow {
namespace py {
namespace benchmark {

void Benchmark_PandasObjectIsNull(PyObject* list) {
  if (!PyList_CheckExact(list)) {
    PyErr_SetString(PyExc_TypeError, "expected a list");
    return;
  }
  Py_ssize_t i, n = PyList_GET_SIZE(list);
  for (i = 0; i < n; i++) {
    internal::PandasObjectIsNull(PyList_GET_ITEM(list, i));
  }
}

}  // namespace benchmark
}  // namespace py
}  // namespace arrow

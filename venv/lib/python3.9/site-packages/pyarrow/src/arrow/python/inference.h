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

// Functions for converting between CPython built-in data structures and Arrow
// data structures

#pragma once

#include "arrow/python/platform.h"

#include <memory>

#include "arrow/type.h"
#include "arrow/util/macros.h"
#include "arrow/python/visibility.h"

#include "common.h"

namespace arrow {

class Array;
class Status;

namespace py {

// These functions take a sequence input, not arbitrary iterables

/// \brief Infer Arrow type from a Python sequence
/// \param[in] obj the sequence of values
/// \param[in] mask an optional mask where True values are null. May
/// be nullptr
/// \param[in] pandas_null_sentinels use pandas's null value markers
ARROW_PYTHON_EXPORT
Result<std::shared_ptr<arrow::DataType>> InferArrowType(PyObject* obj, PyObject* mask,
                                                        bool pandas_null_sentinels);

/// Checks whether the passed Python object is a boolean scalar
ARROW_PYTHON_EXPORT
bool IsPyBool(PyObject* obj);

/// Checks whether the passed Python object is an integer scalar
ARROW_PYTHON_EXPORT
bool IsPyInt(PyObject* obj);

/// Checks whether the passed Python object is a float scalar
ARROW_PYTHON_EXPORT
bool IsPyFloat(PyObject* obj);

}  // namespace py
}  // namespace arrow

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

// Converting from pandas memory representation to Arrow data structures

#pragma once

#include "arrow/python/platform.h"

#include <memory>

#include "arrow/compute/api.h"
#include "arrow/python/visibility.h"

namespace arrow {

class Array;
class ChunkedArray;
class DataType;
class MemoryPool;
class Status;

namespace py {

/// Convert NumPy arrays to Arrow. If target data type is not known, pass a
/// type with null
///
/// \param[in] pool Memory pool for any memory allocations
/// \param[in] ao an ndarray with the array data
/// \param[in] mo an ndarray with a null mask (True is null), optional
/// \param[in] from_pandas If true, use pandas's null sentinels to determine
/// whether values are null
/// \param[in] type a specific type to cast to, may be null
/// \param[in] cast_options casting options
/// \param[out] out a ChunkedArray, to accommodate chunked output
ARROW_PYTHON_EXPORT
Status NdarrayToArrow(MemoryPool* pool, PyObject* ao, PyObject* mo, bool from_pandas,
                      const std::shared_ptr<DataType>& type,
                      const compute::CastOptions& cast_options,
                      std::shared_ptr<ChunkedArray>* out);

/// Safely convert NumPy arrays to Arrow. If target data type is not known,
/// pass a type with null.
///
/// \param[in] pool Memory pool for any memory allocations
/// \param[in] ao an ndarray with the array data
/// \param[in] mo an ndarray with a null mask (True is null), optional
/// \param[in] from_pandas If true, use pandas's null sentinels to determine
/// whether values are null
/// \param[in] type a specific type to cast to, may be null
/// \param[out] out a ChunkedArray, to accommodate chunked output
ARROW_PYTHON_EXPORT
Status NdarrayToArrow(MemoryPool* pool, PyObject* ao, PyObject* mo, bool from_pandas,
                      const std::shared_ptr<DataType>& type,
                      std::shared_ptr<ChunkedArray>* out);

}  // namespace py
}  // namespace arrow

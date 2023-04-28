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
#include <unordered_set>

#include "arrow/memory_pool.h"
#include "arrow/python/visibility.h"

namespace arrow {

class Array;
class ChunkedArray;
class Column;
class DataType;
class MemoryPool;
class Status;
class Table;

namespace py {

struct PandasOptions {
  /// arrow::MemoryPool to use for memory allocations
  MemoryPool* pool = default_memory_pool();

  /// If true, we will convert all string columns to categoricals
  bool strings_to_categorical = false;
  bool zero_copy_only = false;
  bool integer_object_nulls = false;
  bool date_as_object = false;
  bool timestamp_as_object = false;
  bool use_threads = false;

  /// Coerce all date and timestamp to datetime64[ns]
  bool coerce_temporal_nanoseconds = false;

  /// Used to maintain backwards compatibility for
  /// timezone bugs (see ARROW-9528).  Should be removed
  /// after Arrow 2.0 release.
  bool ignore_timezone = false;

  /// \brief If true, do not create duplicate PyObject versions of equal
  /// objects. This only applies to immutable objects like strings or datetime
  /// objects
  bool deduplicate_objects = false;

  /// \brief For certain data types, a cast is needed in order to store the
  /// data in a pandas DataFrame or Series (e.g. timestamps are always stored
  /// as nanoseconds in pandas). This option controls whether it is a safe
  /// cast or not.
  bool safe_cast = true;

  /// \brief If true, create one block per column rather than consolidated
  /// blocks (1 per data type). Do zero-copy wrapping when there are no
  /// nulls. pandas currently will consolidate the blocks on its own, causing
  /// increased memory use, so keep this in mind if you are working on a
  /// memory-constrained situation.
  bool split_blocks = false;

  /// \brief If true, allow non-writable zero-copy views to be created for
  /// single column blocks. This option is also used to provide zero copy for
  /// Series data
  bool allow_zero_copy_blocks = false;

  /// \brief If true, attempt to deallocate buffers in passed Arrow object if
  /// it is the only remaining shared_ptr copy of it. See ARROW-3789 for
  /// original context for this feature. Only currently implemented for Table
  /// conversions
  bool self_destruct = false;

  // Used internally for nested arrays.
  bool decode_dictionaries = false;

  // Columns that should be casted to categorical
  std::unordered_set<std::string> categorical_columns;

  // Columns that should be passed through to be converted to
  // ExtensionArray/Block
  std::unordered_set<std::string> extension_columns;
};

ARROW_PYTHON_EXPORT
Status ConvertArrayToPandas(const PandasOptions& options, std::shared_ptr<Array> arr,
                            PyObject* py_ref, PyObject** out);

ARROW_PYTHON_EXPORT
Status ConvertChunkedArrayToPandas(const PandasOptions& options,
                                   std::shared_ptr<ChunkedArray> col, PyObject* py_ref,
                                   PyObject** out);

// Convert a whole table as efficiently as possible to a pandas.DataFrame.
//
// The returned Python object is a list of tuples consisting of the exact 2D
// BlockManager structure of the pandas.DataFrame used as of pandas 0.19.x.
//
// tuple item: (indices: ndarray[int32], block: ndarray[TYPE, ndim=2])
ARROW_PYTHON_EXPORT
Status ConvertTableToPandas(const PandasOptions& options, std::shared_ptr<Table> table,
                            PyObject** out);

}  // namespace py
}  // namespace arrow

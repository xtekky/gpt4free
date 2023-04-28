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

#include "arrow/array.h"
#include "arrow/python/platform.h"

namespace arrow {
namespace py {
namespace internal {
// TODO(ARROW-12976):  See if we can refactor Pandas ObjectWriter logic
// to the .cc file and move this there as well if we can.

// Converts array to a sequency of python objects.
template <typename ArrayType, typename WriteValue, typename Assigner>
inline Status WriteArrayObjects(const ArrayType& arr, WriteValue&& write_func,
                                Assigner out_values) {
  // TODO(ARROW-12976): Use visitor here?
  const bool has_nulls = arr.null_count() > 0;
  for (int64_t i = 0; i < arr.length(); ++i) {
    if (has_nulls && arr.IsNull(i)) {
      Py_INCREF(Py_None);
      *out_values = Py_None;
    } else {
      RETURN_NOT_OK(write_func(arr.GetView(i), out_values));
    }
    ++out_values;
  }
  return Status::OK();
}

}  // namespace internal
}  // namespace py
}  // namespace arrow

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

#include <cassert>
#include <cstdint>
#include <cstring>

#include "arrow/util/bit_run_reader.h"

namespace arrow {
namespace util {
namespace internal {

/// \brief Compress the buffer to spaced, excluding the null entries.
///
/// \param[in] src the source buffer
/// \param[in] num_values the size of source buffer
/// \param[in] valid_bits bitmap data indicating position of valid slots
/// \param[in] valid_bits_offset offset into valid_bits
/// \param[out] output the output buffer spaced
/// \return The size of spaced buffer.
template <typename T>
inline int SpacedCompress(const T* src, int num_values, const uint8_t* valid_bits,
                          int64_t valid_bits_offset, T* output) {
  int num_valid_values = 0;

  arrow::internal::SetBitRunReader reader(valid_bits, valid_bits_offset, num_values);
  while (true) {
    const auto run = reader.NextRun();
    if (run.length == 0) {
      break;
    }
    std::memcpy(output + num_valid_values, src + run.position, run.length * sizeof(T));
    num_valid_values += static_cast<int32_t>(run.length);
  }

  return num_valid_values;
}

/// \brief Relocate values in buffer into positions of non-null values as indicated by
/// a validity bitmap.
///
/// \param[in, out] buffer the in-place buffer
/// \param[in] num_values total size of buffer including null slots
/// \param[in] null_count number of null slots
/// \param[in] valid_bits bitmap data indicating position of valid slots
/// \param[in] valid_bits_offset offset into valid_bits
/// \return The number of values expanded, including nulls.
template <typename T>
inline int SpacedExpand(T* buffer, int num_values, int null_count,
                        const uint8_t* valid_bits, int64_t valid_bits_offset) {
  // Point to end as we add the spacing from the back.
  int idx_decode = num_values - null_count;

  // Depending on the number of nulls, some of the value slots in buffer may
  // be uninitialized, and this will cause valgrind warnings / potentially UB
  std::memset(static_cast<void*>(buffer + idx_decode), 0, null_count * sizeof(T));
  if (idx_decode == 0) {
    // All nulls, nothing more to do
    return num_values;
  }

  arrow::internal::ReverseSetBitRunReader reader(valid_bits, valid_bits_offset,
                                                 num_values);
  while (true) {
    const auto run = reader.NextRun();
    if (run.length == 0) {
      break;
    }
    idx_decode -= static_cast<int32_t>(run.length);
    assert(idx_decode >= 0);
    std::memmove(buffer + run.position, buffer + idx_decode, run.length * sizeof(T));
  }

  // Otherwise caller gave an incorrect null_count
  assert(idx_decode == 0);
  return num_values;
}

}  // namespace internal
}  // namespace util
}  // namespace arrow

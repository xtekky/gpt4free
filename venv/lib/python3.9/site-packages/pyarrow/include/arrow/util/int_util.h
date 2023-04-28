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

#include <cstdint>
#include <type_traits>

#include "arrow/status.h"

#include "arrow/util/visibility.h"

namespace arrow {

class DataType;
struct ArraySpan;
struct Scalar;

namespace internal {

ARROW_EXPORT
uint8_t DetectUIntWidth(const uint64_t* values, int64_t length, uint8_t min_width = 1);

ARROW_EXPORT
uint8_t DetectUIntWidth(const uint64_t* values, const uint8_t* valid_bytes,
                        int64_t length, uint8_t min_width = 1);

ARROW_EXPORT
uint8_t DetectIntWidth(const int64_t* values, int64_t length, uint8_t min_width = 1);

ARROW_EXPORT
uint8_t DetectIntWidth(const int64_t* values, const uint8_t* valid_bytes, int64_t length,
                       uint8_t min_width = 1);

ARROW_EXPORT
void DowncastInts(const int64_t* source, int8_t* dest, int64_t length);

ARROW_EXPORT
void DowncastInts(const int64_t* source, int16_t* dest, int64_t length);

ARROW_EXPORT
void DowncastInts(const int64_t* source, int32_t* dest, int64_t length);

ARROW_EXPORT
void DowncastInts(const int64_t* source, int64_t* dest, int64_t length);

ARROW_EXPORT
void DowncastUInts(const uint64_t* source, uint8_t* dest, int64_t length);

ARROW_EXPORT
void DowncastUInts(const uint64_t* source, uint16_t* dest, int64_t length);

ARROW_EXPORT
void DowncastUInts(const uint64_t* source, uint32_t* dest, int64_t length);

ARROW_EXPORT
void DowncastUInts(const uint64_t* source, uint64_t* dest, int64_t length);

ARROW_EXPORT
void UpcastInts(const int32_t* source, int64_t* dest, int64_t length);

template <typename InputInt, typename OutputInt>
inline typename std::enable_if<(sizeof(InputInt) >= sizeof(OutputInt))>::type CastInts(
    const InputInt* source, OutputInt* dest, int64_t length) {
  DowncastInts(source, dest, length);
}

template <typename InputInt, typename OutputInt>
inline typename std::enable_if<(sizeof(InputInt) < sizeof(OutputInt))>::type CastInts(
    const InputInt* source, OutputInt* dest, int64_t length) {
  UpcastInts(source, dest, length);
}

template <typename InputInt, typename OutputInt>
ARROW_EXPORT void TransposeInts(const InputInt* source, OutputInt* dest, int64_t length,
                                const int32_t* transpose_map);

ARROW_EXPORT
Status TransposeInts(const DataType& src_type, const DataType& dest_type,
                     const uint8_t* src, uint8_t* dest, int64_t src_offset,
                     int64_t dest_offset, int64_t length, const int32_t* transpose_map);

/// \brief Do vectorized boundschecking of integer-type array indices. The
/// indices must be nonnegative and strictly less than the passed upper
/// limit (which is usually the length of an array that is being indexed-into).
ARROW_EXPORT
Status CheckIndexBounds(const ArraySpan& values, uint64_t upper_limit);

/// \brief Boundscheck integer values to determine if they are all between the
/// passed upper and lower limits (inclusive). Upper and lower bounds must be
/// the same type as the data and are not currently casted.
ARROW_EXPORT
Status CheckIntegersInRange(const ArraySpan& values, const Scalar& bound_lower,
                            const Scalar& bound_upper);

/// \brief Use CheckIntegersInRange to determine whether the passed integers
/// can fit safely in the passed integer type. This helps quickly determine if
/// integer narrowing (e.g. int64->int32) is safe to do.
ARROW_EXPORT
Status IntegersCanFit(const ArraySpan& values, const DataType& target_type);

/// \brief Convenience for boundschecking a single Scalar vlue
ARROW_EXPORT
Status IntegersCanFit(const Scalar& value, const DataType& target_type);

/// Upcast an integer to the largest possible width (currently 64 bits)

template <typename Integer>
typename std::enable_if<
    std::is_integral<Integer>::value && std::is_signed<Integer>::value, int64_t>::type
UpcastInt(Integer v) {
  return v;
}

template <typename Integer>
typename std::enable_if<
    std::is_integral<Integer>::value && std::is_unsigned<Integer>::value, uint64_t>::type
UpcastInt(Integer v) {
  return v;
}

}  // namespace internal
}  // namespace arrow

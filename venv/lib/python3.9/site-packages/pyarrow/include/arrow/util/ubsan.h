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

// Contains utilities for making UBSan happy.

#pragma once

#include <cstring>
#include <memory>
#include <type_traits>

#include "arrow/util/macros.h"

namespace arrow {
namespace util {

namespace internal {

constexpr uint8_t kNonNullFiller = 0;

}  // namespace internal

/// \brief Returns maybe_null if not null or a non-null pointer to an arbitrary memory
/// that shouldn't be dereferenced.
///
/// Memset/Memcpy are undefined when a nullptr is passed as an argument use this utility
/// method to wrap locations where this could happen.
///
/// Note: Flatbuffers has UBSan warnings if a zero length vector is passed.
/// https://github.com/google/flatbuffers/pull/5355 is trying to resolve
/// them.
template <typename T>
inline T* MakeNonNull(T* maybe_null = NULLPTR) {
  if (ARROW_PREDICT_TRUE(maybe_null != NULLPTR)) {
    return maybe_null;
  }

  return const_cast<T*>(reinterpret_cast<const T*>(&internal::kNonNullFiller));
}

template <typename T>
inline typename std::enable_if<std::is_trivial<T>::value, T>::type SafeLoadAs(
    const uint8_t* unaligned) {
  typename std::remove_const<T>::type ret;
  std::memcpy(&ret, unaligned, sizeof(T));
  return ret;
}

template <typename T>
inline typename std::enable_if<std::is_trivial<T>::value, T>::type SafeLoad(
    const T* unaligned) {
  typename std::remove_const<T>::type ret;
  std::memcpy(&ret, unaligned, sizeof(T));
  return ret;
}

template <typename U, typename T>
inline typename std::enable_if<std::is_trivial<T>::value && std::is_trivial<U>::value &&
                                   sizeof(T) == sizeof(U),
                               U>::type
SafeCopy(T value) {
  typename std::remove_const<U>::type ret;
  std::memcpy(&ret, &value, sizeof(T));
  return ret;
}

template <typename T>
inline typename std::enable_if<std::is_trivial<T>::value, void>::type SafeStore(
    void* unaligned, T value) {
  std::memcpy(unaligned, &value, sizeof(T));
}

}  // namespace util
}  // namespace arrow

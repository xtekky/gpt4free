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

namespace arrow {
namespace internal {

/// \brief Metafunction to allow checking if a type matches any of another set of types
template <typename...>
struct IsOneOf : std::false_type {};  /// Base case: nothing has matched

template <typename T, typename U, typename... Args>
struct IsOneOf<T, U, Args...> {
  /// Recursive case: T == U or T matches any other types provided (not including U).
  static constexpr bool value = std::is_same<T, U>::value || IsOneOf<T, Args...>::value;
};

/// \brief Shorthand for using IsOneOf + std::enable_if
template <typename T, typename... Args>
using EnableIfIsOneOf = typename std::enable_if<IsOneOf<T, Args...>::value, T>::type;

/// \brief is_null_pointer from C++17
template <typename T>
struct is_null_pointer : std::is_same<std::nullptr_t, typename std::remove_cv<T>::type> {
};

}  // namespace internal
}  // namespace arrow

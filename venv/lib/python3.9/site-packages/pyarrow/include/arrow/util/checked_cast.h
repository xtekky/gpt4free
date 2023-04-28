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

#include <memory>
#include <type_traits>
#include <utility>

namespace arrow {
namespace internal {

template <typename OutputType, typename InputType>
inline OutputType checked_cast(InputType&& value) {
  static_assert(std::is_class<typename std::remove_pointer<
                    typename std::remove_reference<InputType>::type>::type>::value,
                "checked_cast input type must be a class");
  static_assert(std::is_class<typename std::remove_pointer<
                    typename std::remove_reference<OutputType>::type>::type>::value,
                "checked_cast output type must be a class");
#ifdef NDEBUG
  return static_cast<OutputType>(value);
#else
  return dynamic_cast<OutputType>(value);
#endif
}

template <class T, class U>
std::shared_ptr<T> checked_pointer_cast(std::shared_ptr<U> r) noexcept {
#ifdef NDEBUG
  return std::static_pointer_cast<T>(std::move(r));
#else
  return std::dynamic_pointer_cast<T>(std::move(r));
#endif
}

template <class T, class U>
std::unique_ptr<T> checked_pointer_cast(std::unique_ptr<U> r) noexcept {
#ifdef NDEBUG
  return std::unique_ptr<T>(static_cast<T*>(r.release()));
#else
  return std::unique_ptr<T>(dynamic_cast<T*>(r.release()));
#endif
}

}  // namespace internal
}  // namespace arrow

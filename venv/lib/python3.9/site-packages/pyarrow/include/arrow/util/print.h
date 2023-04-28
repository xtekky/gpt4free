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
// under the License. template <typename T>

#pragma once

#include <tuple>

namespace arrow {
namespace internal {

namespace detail {

template <typename OStream, typename Tuple, size_t N>
struct TuplePrinter {
  static void Print(OStream* os, const Tuple& t) {
    TuplePrinter<OStream, Tuple, N - 1>::Print(os, t);
    *os << std::get<N - 1>(t);
  }
};

template <typename OStream, typename Tuple>
struct TuplePrinter<OStream, Tuple, 0> {
  static void Print(OStream* os, const Tuple& t) {}
};

}  // namespace detail

// Print elements from a tuple to a stream, in order.
// Typical use is to pack a bunch of existing values with std::forward_as_tuple()
// before passing it to this function.
template <typename OStream, typename... Args>
void PrintTuple(OStream* os, const std::tuple<Args&...>& tup) {
  detail::TuplePrinter<OStream, std::tuple<Args&...>, sizeof...(Args)>::Print(os, tup);
}

}  // namespace internal
}  // namespace arrow

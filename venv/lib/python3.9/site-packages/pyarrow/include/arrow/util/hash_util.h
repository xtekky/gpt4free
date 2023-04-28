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

namespace arrow {
namespace internal {

// ----------------------------------------------------------------------
// BEGIN Hash utilities from Boost

namespace detail {

#if defined(_MSC_VER)
#define ARROW_HASH_ROTL32(x, r) _rotl(x, r)
#else
#define ARROW_HASH_ROTL32(x, r) (x << r) | (x >> (32 - r))
#endif

template <typename SizeT>
inline void hash_combine_impl(SizeT& seed, SizeT value) {
  seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

inline void hash_combine_impl(uint32_t& h1, uint32_t k1) {
  const uint32_t c1 = 0xcc9e2d51;
  const uint32_t c2 = 0x1b873593;

  k1 *= c1;
  k1 = ARROW_HASH_ROTL32(k1, 15);
  k1 *= c2;

  h1 ^= k1;
  h1 = ARROW_HASH_ROTL32(h1, 13);
  h1 = h1 * 5 + 0xe6546b64;
}

#undef ARROW_HASH_ROTL32

}  // namespace detail

template <class T>
inline void hash_combine(std::size_t& seed, T const& v) {
  std::hash<T> hasher;
  return ::arrow::internal::detail::hash_combine_impl(seed, hasher(v));
}

// END Hash utilities from Boost
// ----------------------------------------------------------------------

}  // namespace internal
}  // namespace arrow

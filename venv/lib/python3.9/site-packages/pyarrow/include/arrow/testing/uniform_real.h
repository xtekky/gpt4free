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

// Random real generation is very slow on Arm if built with clang + libstdc++
// due to software emulated long double arithmetic.
// This file ports some random real libs from llvm libc++ library, which are
// free from long double calculation.
// It improves performance significantly on both Arm (~100x) and x86 (~8x) in
// generating random reals when built with clang + gnu libstdc++.
// Based on: https://github.com/llvm/llvm-project/tree/main/libcxx

#pragma once

#include <limits>

#include <arrow/util/bit_util.h>

namespace arrow {
namespace random {

namespace detail {

// std::generate_canonical, simplified
// https://en.cppreference.com/w/cpp/numeric/random/generate_canonical
template <typename RealType, typename Rng>
RealType generate_canonical(Rng& rng) {
  const size_t b = std::numeric_limits<RealType>::digits;
  const size_t log2R = 63 - ::arrow::bit_util::CountLeadingZeros(
                                static_cast<uint64_t>(Rng::max() - Rng::min()) + 1);
  const size_t k = b / log2R + (b % log2R != 0) + (b == 0);
  const RealType r = static_cast<RealType>(Rng::max() - Rng::min()) + 1;
  RealType base = r;
  RealType sp = static_cast<RealType>(rng() - Rng::min());
  for (size_t i = 1; i < k; ++i, base *= r) {
    sp += (rng() - Rng::min()) * base;
  }
  return sp / base;
}

}  // namespace detail

// std::uniform_real_distribution, simplified
// https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
template <typename RealType = double>
struct uniform_real_distribution {
  const RealType a, b;

  explicit uniform_real_distribution(RealType a = 0, RealType b = 1) : a(a), b(b) {}

  template <typename Rng>
  RealType operator()(Rng& rng) {
    return (b - a) * detail::generate_canonical<RealType>(rng) + a;
  }
};

// std::bernoulli_distribution, simplified
// https://en.cppreference.com/w/cpp/numeric/random/bernoulli_distribution
struct bernoulli_distribution {
  const double p;

  explicit bernoulli_distribution(double p = 0.5) : p(p) {}

  template <class Rng>
  bool operator()(Rng& rng) {
    return detail::generate_canonical<double>(rng) < p;
  }
};

}  // namespace random
}  // namespace arrow

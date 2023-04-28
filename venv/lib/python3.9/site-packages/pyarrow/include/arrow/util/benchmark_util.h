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

#include <algorithm>
#include <cstdint>
#include <string>

#include "benchmark/benchmark.h"

#include "arrow/util/cpu_info.h"

namespace arrow {

// Benchmark changed its parameter type between releases from
// int to int64_t. As it doesn't have version macros, we need
// to apply C++ template magic.

template <typename Func>
struct BenchmarkArgsType;

// Pattern matching that extracts the vector element type of Benchmark::Args()
template <typename Values>
struct BenchmarkArgsType<benchmark::internal::Benchmark* (
    benchmark::internal::Benchmark::*)(const std::vector<Values>&)> {
  using type = Values;
};

using ArgsType =
    typename BenchmarkArgsType<decltype(&benchmark::internal::Benchmark::Args)>::type;

using internal::CpuInfo;

static const CpuInfo* cpu_info = CpuInfo::GetInstance();

static const int64_t kL1Size = cpu_info->CacheSize(CpuInfo::CacheLevel::L1);
static const int64_t kL2Size = cpu_info->CacheSize(CpuInfo::CacheLevel::L2);
static const int64_t kL3Size = cpu_info->CacheSize(CpuInfo::CacheLevel::L3);
static const int64_t kCantFitInL3Size = kL3Size * 4;
static const std::vector<int64_t> kMemorySizes = {kL1Size, kL2Size, kL3Size,
                                                  kCantFitInL3Size};
// 0 is treated as "no nulls"
static const std::vector<ArgsType> kInverseNullProportions = {10000, 100, 10, 2, 1, 0};

struct GenericItemsArgs {
  // number of items processed per iteration
  const int64_t size;

  // proportion of nulls in generated arrays
  double null_proportion;

  explicit GenericItemsArgs(benchmark::State& state)
      : size(state.range(0)), state_(state) {
    if (state.range(1) == 0) {
      this->null_proportion = 0.0;
    } else {
      this->null_proportion = std::min(1., 1. / static_cast<double>(state.range(1)));
    }
  }

  ~GenericItemsArgs() {
    state_.counters["size"] = static_cast<double>(size);
    state_.counters["null_percent"] = null_proportion * 100;
    state_.SetItemsProcessed(state_.iterations() * size);
  }

 private:
  benchmark::State& state_;
};

void BenchmarkSetArgsWithSizes(benchmark::internal::Benchmark* bench,
                               const std::vector<int64_t>& sizes = kMemorySizes) {
  bench->Unit(benchmark::kMicrosecond);

  for (const auto size : sizes) {
    for (const auto inverse_null_proportion : kInverseNullProportions) {
      bench->Args({static_cast<ArgsType>(size), inverse_null_proportion});
    }
  }
}

void BenchmarkSetArgs(benchmark::internal::Benchmark* bench) {
  BenchmarkSetArgsWithSizes(bench, kMemorySizes);
}

void RegressionSetArgs(benchmark::internal::Benchmark* bench) {
  // Regression do not need to account for cache hierarchy, thus optimize for
  // the best case.
  BenchmarkSetArgsWithSizes(bench, {kL1Size});
}

// RAII struct to handle some of the boilerplate in regression benchmarks
struct RegressionArgs {
  // size of memory tested (per iteration) in bytes
  const int64_t size;

  // proportion of nulls in generated arrays
  double null_proportion;

  // If size_is_bytes is true, then it's a number of bytes, otherwise it's the
  // number of items processed (for reporting)
  explicit RegressionArgs(benchmark::State& state, bool size_is_bytes = true)
      : size(state.range(0)), state_(state), size_is_bytes_(size_is_bytes) {
    if (state.range(1) == 0) {
      this->null_proportion = 0.0;
    } else {
      this->null_proportion = std::min(1., 1. / static_cast<double>(state.range(1)));
    }
  }

  ~RegressionArgs() {
    state_.counters["size"] = static_cast<double>(size);
    state_.counters["null_percent"] = null_proportion * 100;
    if (size_is_bytes_) {
      state_.SetBytesProcessed(state_.iterations() * size);
    } else {
      state_.SetItemsProcessed(state_.iterations() * size);
    }
  }

 private:
  benchmark::State& state_;
  bool size_is_bytes_;
};

}  // namespace arrow

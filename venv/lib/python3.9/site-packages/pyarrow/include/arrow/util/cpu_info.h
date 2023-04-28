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

// From Apache Impala (incubating) as of 2016-01-29. Pared down to a minimal
// set of functions needed for Apache Arrow / Apache parquet-cpp

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace internal {

/// CpuInfo is an interface to query for cpu information at runtime.  The caller can
/// ask for the sizes of the caches and what hardware features are supported.
/// On Linux, this information is pulled from a couple of sys files (/proc/cpuinfo and
/// /sys/devices)
class ARROW_EXPORT CpuInfo {
 public:
  ~CpuInfo();

  /// x86 features
  static constexpr int64_t SSSE3 = (1LL << 0);
  static constexpr int64_t SSE4_1 = (1LL << 1);
  static constexpr int64_t SSE4_2 = (1LL << 2);
  static constexpr int64_t POPCNT = (1LL << 3);
  static constexpr int64_t AVX = (1LL << 4);
  static constexpr int64_t AVX2 = (1LL << 5);
  static constexpr int64_t AVX512F = (1LL << 6);
  static constexpr int64_t AVX512CD = (1LL << 7);
  static constexpr int64_t AVX512VL = (1LL << 8);
  static constexpr int64_t AVX512DQ = (1LL << 9);
  static constexpr int64_t AVX512BW = (1LL << 10);
  static constexpr int64_t AVX512 = AVX512F | AVX512CD | AVX512VL | AVX512DQ | AVX512BW;
  static constexpr int64_t BMI1 = (1LL << 11);
  static constexpr int64_t BMI2 = (1LL << 12);

  /// Arm features
  static constexpr int64_t ASIMD = (1LL << 32);

  /// Cache enums for L1 (data), L2 and L3
  enum class CacheLevel { L1 = 0, L2, L3, Last = L3 };

  /// CPU vendors
  enum class Vendor { Unknown, Intel, AMD };

  static const CpuInfo* GetInstance();

  /// Returns all the flags for this cpu
  int64_t hardware_flags() const;

  /// Returns the number of cores (including hyper-threaded) on this machine.
  int num_cores() const;

  /// Returns the vendor of the cpu.
  Vendor vendor() const;

  /// Returns the model name of the cpu (e.g. Intel i7-2600)
  const std::string& model_name() const;

  /// Returns the size of the cache in KB at this cache level
  int64_t CacheSize(CacheLevel level) const;

  /// \brief Returns whether or not the given feature is enabled.
  ///
  /// IsSupported() is true iff IsDetected() is also true and the feature
  /// wasn't disabled by the user (for example by setting the ARROW_USER_SIMD_LEVEL
  /// environment variable).
  bool IsSupported(int64_t flags) const;

  /// Returns whether or not the given feature is available on the CPU.
  bool IsDetected(int64_t flags) const;

  /// Determine if the CPU meets the minimum CPU requirements and if not, issue an error
  /// and terminate.
  void VerifyCpuRequirements() const;

  /// Toggle a hardware feature on and off.  It is not valid to turn on a feature
  /// that the underlying hardware cannot support. This is useful for testing.
  void EnableFeature(int64_t flag, bool enable);

  bool HasEfficientBmi2() const {
    // BMI2 (pext, pdep) is only efficient on Intel X86 processors.
    return vendor() == Vendor::Intel && IsSupported(BMI2);
  }

 private:
  CpuInfo();

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace internal
}  // namespace arrow

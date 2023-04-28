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

#include <utility>
#include <vector>

#include "arrow/status.h"
#include "arrow/util/cpu_info.h"

namespace arrow {
namespace internal {

enum class DispatchLevel : int {
  // These dispatch levels, corresponding to instruction set features,
  // are sorted in increasing order of preference.
  NONE = 0,
  SSE4_2,
  AVX2,
  AVX512,
  NEON,
  MAX
};

/*
  A facility for dynamic dispatch according to available DispatchLevel.

  Typical use:

    static void my_function_default(...);
    static void my_function_avx2(...);

    struct MyDynamicFunction {
      using FunctionType = decltype(&my_function_default);

      static std::vector<std::pair<DispatchLevel, FunctionType>> implementations() {
        return {
          { DispatchLevel::NONE, my_function_default }
    #if defined(ARROW_HAVE_RUNTIME_AVX2)
          , { DispatchLevel::AVX2, my_function_avx2 }
    #endif
        };
      }
    };

    void my_function(...) {
      static DynamicDispatch<MyDynamicFunction> dispatch;
      return dispatch.func(...);
    }
*/
template <typename DynamicFunction>
class DynamicDispatch {
 protected:
  using FunctionType = typename DynamicFunction::FunctionType;
  using Implementation = std::pair<DispatchLevel, FunctionType>;

 public:
  DynamicDispatch() { Resolve(DynamicFunction::implementations()); }

  FunctionType func = {};

 protected:
  // Use the Implementation with the highest DispatchLevel
  void Resolve(const std::vector<Implementation>& implementations) {
    Implementation cur{DispatchLevel::NONE, {}};

    for (const auto& impl : implementations) {
      if (impl.first >= cur.first && IsSupported(impl.first)) {
        // Higher (or same) level than current
        cur = impl;
      }
    }

    if (!cur.second) {
      Status::Invalid("No appropriate implementation found").Abort();
    }
    func = cur.second;
  }

 private:
  bool IsSupported(DispatchLevel level) const {
    static const auto cpu_info = arrow::internal::CpuInfo::GetInstance();

    switch (level) {
      case DispatchLevel::NONE:
        return true;
      case DispatchLevel::SSE4_2:
        return cpu_info->IsSupported(CpuInfo::SSE4_2);
      case DispatchLevel::AVX2:
        return cpu_info->IsSupported(CpuInfo::AVX2);
      case DispatchLevel::AVX512:
        return cpu_info->IsSupported(CpuInfo::AVX512);
      default:
        return false;
    }
  }
};

}  // namespace internal
}  // namespace arrow

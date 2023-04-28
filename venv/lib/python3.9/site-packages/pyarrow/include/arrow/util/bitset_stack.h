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

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "arrow/buffer.h"
#include "arrow/memory_pool.h"
#include "arrow/result.h"
#include "arrow/type_fwd.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/compare.h"
#include "arrow/util/functional.h"
#include "arrow/util/macros.h"
#include "arrow/util/string_builder.h"
#include "arrow/util/type_traits.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace internal {

/// \brief Store a stack of bitsets efficiently. The top bitset may be
/// accessed and its bits may be modified, but it may not be resized.
class BitsetStack {
 public:
  using reference = typename std::vector<bool>::reference;

  /// \brief push a bitset onto the stack
  /// \param size number of bits in the next bitset
  /// \param value initial value for bits in the pushed bitset
  void Push(int size, bool value) {
    offsets_.push_back(bit_count());
    bits_.resize(bit_count() + size, value);
  }

  /// \brief number of bits in the bitset at the top of the stack
  int TopSize() const {
    if (offsets_.size() == 0) return 0;
    return bit_count() - offsets_.back();
  }

  /// \brief pop a bitset off the stack
  void Pop() {
    bits_.resize(offsets_.back());
    offsets_.pop_back();
  }

  /// \brief get the value of a bit in the top bitset
  /// \param i index of the bit to access
  bool operator[](int i) const { return bits_[offsets_.back() + i]; }

  /// \brief get a mutable reference to a bit in the top bitset
  /// \param i index of the bit to access
  reference operator[](int i) { return bits_[offsets_.back() + i]; }

 private:
  int bit_count() const { return static_cast<int>(bits_.size()); }
  std::vector<bool> bits_;
  std::vector<int> offsets_;
};

}  // namespace internal
}  // namespace arrow

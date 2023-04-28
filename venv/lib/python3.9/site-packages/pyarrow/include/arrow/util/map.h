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

#include "arrow/result.h"

namespace arrow {
namespace internal {

/// Helper providing single-lookup conditional insertion into std::map or
/// std::unordered_map. If `key` exists in the container, an iterator to that pair
/// will be returned. If `key` does not exist in the container, `gen(key)` will be
/// invoked and its return value inserted.
template <typename Map, typename Gen>
auto GetOrInsertGenerated(Map* map, typename Map::key_type key, Gen&& gen)
    -> decltype(map->begin()->second = gen(map->begin()->first), map->begin()) {
  decltype(gen(map->begin()->first)) placeholder{};

  auto it_success = map->emplace(std::move(key), std::move(placeholder));
  if (it_success.second) {
    // insertion of placeholder succeeded, overwrite it with gen()
    const auto& inserted_key = it_success.first->first;
    auto* value = &it_success.first->second;
    *value = gen(inserted_key);
  }
  return it_success.first;
}

template <typename Map, typename Gen>
auto GetOrInsertGenerated(Map* map, typename Map::key_type key, Gen&& gen)
    -> Result<decltype(map->begin()->second = gen(map->begin()->first).ValueOrDie(),
                       map->begin())> {
  decltype(gen(map->begin()->first).ValueOrDie()) placeholder{};

  auto it_success = map->emplace(std::move(key), std::move(placeholder));
  if (it_success.second) {
    // insertion of placeholder succeeded, overwrite it with gen()
    const auto& inserted_key = it_success.first->first;
    auto* value = &it_success.first->second;
    ARROW_ASSIGN_OR_RAISE(*value, gen(inserted_key));
  }
  return it_success.first;
}

}  // namespace internal
}  // namespace arrow

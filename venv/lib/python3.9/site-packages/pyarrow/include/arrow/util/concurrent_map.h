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

#include <unordered_map>
#include <utility>

#include "arrow/util/mutex.h"

namespace arrow {
namespace util {

template <typename K, typename V>
class ConcurrentMap {
 public:
  void Insert(const K& key, const V& value) {
    auto lock = mutex_.Lock();
    map_.insert({key, value});
  }

  template <typename ValueFunc>
  V GetOrInsert(const K& key, ValueFunc&& compute_value_func) {
    auto lock = mutex_.Lock();
    auto it = map_.find(key);
    if (it == map_.end()) {
      auto pair = map_.emplace(key, compute_value_func());
      it = pair.first;
    }
    return it->second;
  }

  void Erase(const K& key) {
    auto lock = mutex_.Lock();
    map_.erase(key);
  }

  void Clear() {
    auto lock = mutex_.Lock();
    map_.clear();
  }

  size_t size() const {
    auto lock = mutex_.Lock();
    return map_.size();
  }

 private:
  std::unordered_map<K, V> map_;
  mutable arrow::util::Mutex mutex_;
};

}  // namespace util
}  // namespace arrow

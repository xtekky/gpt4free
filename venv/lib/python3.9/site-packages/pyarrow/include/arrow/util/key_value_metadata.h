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

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {

/// \brief A container for key-value pair type metadata. Not thread-safe
class ARROW_EXPORT KeyValueMetadata {
 public:
  KeyValueMetadata();
  KeyValueMetadata(std::vector<std::string> keys, std::vector<std::string> values);
  explicit KeyValueMetadata(const std::unordered_map<std::string, std::string>& map);

  static std::shared_ptr<KeyValueMetadata> Make(std::vector<std::string> keys,
                                                std::vector<std::string> values);

  void ToUnorderedMap(std::unordered_map<std::string, std::string>* out) const;
  void Append(std::string key, std::string value);

  Result<std::string> Get(const std::string& key) const;
  bool Contains(const std::string& key) const;
  // Note that deleting may invalidate known indices
  Status Delete(const std::string& key);
  Status Delete(int64_t index);
  Status DeleteMany(std::vector<int64_t> indices);
  Status Set(const std::string& key, const std::string& value);

  void reserve(int64_t n);

  int64_t size() const;
  const std::string& key(int64_t i) const;
  const std::string& value(int64_t i) const;
  const std::vector<std::string>& keys() const { return keys_; }
  const std::vector<std::string>& values() const { return values_; }

  std::vector<std::pair<std::string, std::string>> sorted_pairs() const;

  /// \brief Perform linear search for key, returning -1 if not found
  int FindKey(const std::string& key) const;

  std::shared_ptr<KeyValueMetadata> Copy() const;

  /// \brief Return a new KeyValueMetadata by combining the passed metadata
  /// with this KeyValueMetadata. Colliding keys will be overridden by the
  /// passed metadata. Assumes keys in both containers are unique
  std::shared_ptr<KeyValueMetadata> Merge(const KeyValueMetadata& other) const;

  bool Equals(const KeyValueMetadata& other) const;
  std::string ToString() const;

 private:
  std::vector<std::string> keys_;
  std::vector<std::string> values_;

  ARROW_DISALLOW_COPY_AND_ASSIGN(KeyValueMetadata);
};

/// \brief Create a KeyValueMetadata instance
///
/// \param pairs key-value mapping
ARROW_EXPORT std::shared_ptr<KeyValueMetadata> key_value_metadata(
    const std::unordered_map<std::string, std::string>& pairs);

/// \brief Create a KeyValueMetadata instance
///
/// \param keys sequence of metadata keys
/// \param values sequence of corresponding metadata values
ARROW_EXPORT std::shared_ptr<KeyValueMetadata> key_value_metadata(
    std::vector<std::string> keys, std::vector<std::string> values);

}  // namespace arrow

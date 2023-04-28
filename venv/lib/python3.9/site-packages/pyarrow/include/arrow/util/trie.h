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

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "arrow/status.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace internal {

// A non-zero-terminated small string class.
// std::string usually has a small string optimization
// (see review at https://shaharmike.com/cpp/std-string/)
// but this one allows tight control and optimization of memory layout.
template <uint8_t N>
class SmallString {
 public:
  SmallString() : length_(0) {}

  template <typename T>
  SmallString(const T& v) {  // NOLINT implicit constructor
    *this = std::string_view(v);
  }

  SmallString& operator=(const std::string_view s) {
#ifndef NDEBUG
    CheckSize(s.size());
#endif
    length_ = static_cast<uint8_t>(s.size());
    std::memcpy(data_, s.data(), length_);
    return *this;
  }

  SmallString& operator=(const std::string& s) {
    *this = std::string_view(s);
    return *this;
  }

  SmallString& operator=(const char* s) {
    *this = std::string_view(s);
    return *this;
  }

  explicit operator std::string_view() const { return std::string_view(data_, length_); }

  const char* data() const { return data_; }
  size_t length() const { return length_; }
  bool empty() const { return length_ == 0; }
  char operator[](size_t pos) const {
#ifdef NDEBUG
    assert(pos <= length_);
#endif
    return data_[pos];
  }

  SmallString substr(size_t pos) const {
    return SmallString(std::string_view(*this).substr(pos));
  }

  SmallString substr(size_t pos, size_t count) const {
    return SmallString(std::string_view(*this).substr(pos, count));
  }

  template <typename T>
  bool operator==(T&& other) const {
    return std::string_view(*this) == std::string_view(std::forward<T>(other));
  }

  template <typename T>
  bool operator!=(T&& other) const {
    return std::string_view(*this) != std::string_view(std::forward<T>(other));
  }

 protected:
  uint8_t length_;
  char data_[N];

  void CheckSize(size_t n) { assert(n <= N); }
};

template <uint8_t N>
std::ostream& operator<<(std::ostream& os, const SmallString<N>& str) {
  return os << std::string_view(str);
}

// A trie class for byte strings, optimized for small sets of short strings.
// This class is immutable by design, use a TrieBuilder to construct it.
class ARROW_EXPORT Trie {
  using index_type = int16_t;
  using fast_index_type = int_fast16_t;
  static constexpr auto kMaxIndex = std::numeric_limits<index_type>::max();

 public:
  Trie() : size_(0) {}
  Trie(Trie&&) = default;
  Trie& operator=(Trie&&) = default;

  int32_t Find(std::string_view s) const {
    const Node* node = &nodes_[0];
    fast_index_type pos = 0;
    if (s.length() > static_cast<size_t>(kMaxIndex)) {
      return -1;
    }
    fast_index_type remaining = static_cast<fast_index_type>(s.length());

    while (remaining > 0) {
      auto substring_length = node->substring_length();
      if (substring_length > 0) {
        auto substring_data = node->substring_data();
        if (remaining < substring_length) {
          // Input too short
          return -1;
        }
        for (fast_index_type i = 0; i < substring_length; ++i) {
          if (s[pos++] != substring_data[i]) {
            // Mismatching substring
            return -1;
          }
          --remaining;
        }
        if (remaining == 0) {
          // Matched node exactly
          return node->found_index_;
        }
      }
      // Lookup child using next input character
      if (node->child_lookup_ == -1) {
        // Input too long
        return -1;
      }
      auto c = static_cast<uint8_t>(s[pos++]);
      --remaining;
      auto child_index = lookup_table_[node->child_lookup_ * 256 + c];
      if (child_index == -1) {
        // Child not found
        return -1;
      }
      node = &nodes_[child_index];
    }

    // Input exhausted
    if (node->substring_.empty()) {
      // Matched node exactly
      return node->found_index_;
    } else {
      return -1;
    }
  }

  Status Validate() const;

  void Dump() const;

 protected:
  static constexpr size_t kNodeSize = 16;
  static constexpr auto kMaxSubstringLength =
      kNodeSize - 2 * sizeof(index_type) - sizeof(int8_t);

  struct Node {
    // If this node is a valid end of string, index of found string, otherwise -1
    index_type found_index_;
    // Base index for child lookup in lookup_table_ (-1 if no child nodes)
    index_type child_lookup_;
    // The substring for this node.
    SmallString<kMaxSubstringLength> substring_;

    fast_index_type substring_length() const {
      return static_cast<fast_index_type>(substring_.length());
    }
    const char* substring_data() const { return substring_.data(); }
  };

  static_assert(sizeof(Node) == kNodeSize, "Unexpected node size");

  ARROW_DISALLOW_COPY_AND_ASSIGN(Trie);

  void Dump(const Node* node, const std::string& indent) const;

  // Node table: entry 0 is the root node
  std::vector<Node> nodes_;

  // Indexed lookup structure: gives index in node table, or -1 if not found
  std::vector<index_type> lookup_table_;

  // Number of entries
  index_type size_;

  friend class TrieBuilder;
};

class ARROW_EXPORT TrieBuilder {
  using index_type = Trie::index_type;
  using fast_index_type = Trie::fast_index_type;

 public:
  TrieBuilder();
  Status Append(std::string_view s, bool allow_duplicate = false);
  Trie Finish();

 protected:
  // Extend the lookup table by 256 entries, return the index of the new span
  Status ExtendLookupTable(index_type* out_lookup_index);
  // Split the node given by the index at the substring index `split_at`
  Status SplitNode(fast_index_type node_index, fast_index_type split_at);
  // Append an already constructed child node to the parent
  Status AppendChildNode(Trie::Node* parent, uint8_t ch, Trie::Node&& node);
  // Create a matching child node from this parent
  Status CreateChildNode(Trie::Node* parent, uint8_t ch, std::string_view substring);
  Status CreateChildNode(Trie::Node* parent, char ch, std::string_view substring);

  Trie trie_;

  static constexpr auto kMaxIndex = std::numeric_limits<index_type>::max();
};

}  // namespace internal
}  // namespace arrow

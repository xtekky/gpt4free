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
// under the License. template <typename T>

#pragma once

#include <memory>
#include <ostream>
#include <string>
#include <utility>

#include "arrow/util/visibility.h"

namespace arrow {
namespace util {

namespace detail {

class ARROW_EXPORT StringStreamWrapper {
 public:
  StringStreamWrapper();
  ~StringStreamWrapper();

  std::ostream& stream() { return ostream_; }
  std::string str();

 protected:
  std::unique_ptr<std::ostringstream> sstream_;
  std::ostream& ostream_;
};

}  // namespace detail

template <typename Head>
void StringBuilderRecursive(std::ostream& stream, Head&& head) {
  stream << head;
}

template <typename Head, typename... Tail>
void StringBuilderRecursive(std::ostream& stream, Head&& head, Tail&&... tail) {
  StringBuilderRecursive(stream, std::forward<Head>(head));
  StringBuilderRecursive(stream, std::forward<Tail>(tail)...);
}

template <typename... Args>
std::string StringBuilder(Args&&... args) {
  detail::StringStreamWrapper ss;
  StringBuilderRecursive(ss.stream(), std::forward<Args>(args)...);
  return ss.str();
}

/// CRTP helper for declaring string representation. Defines operator<<
template <typename T>
class ToStringOstreamable {
 public:
  ~ToStringOstreamable() {
    static_assert(
        std::is_same<decltype(std::declval<const T>().ToString()), std::string>::value,
        "ToStringOstreamable depends on the method T::ToString() const");
  }

 private:
  const T& cast() const { return static_cast<const T&>(*this); }

  friend inline std::ostream& operator<<(std::ostream& os, const ToStringOstreamable& t) {
    return os << t.cast().ToString();
  }
};

}  // namespace util
}  // namespace arrow

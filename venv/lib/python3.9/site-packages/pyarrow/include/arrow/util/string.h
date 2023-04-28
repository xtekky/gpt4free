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
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#if __has_include(<charconv>)
#include <charconv>
#endif

#include "arrow/result.h"
#include "arrow/util/visibility.h"

namespace arrow {

class Status;

ARROW_EXPORT std::string HexEncode(const uint8_t* data, size_t length);

ARROW_EXPORT std::string Escape(const char* data, size_t length);

ARROW_EXPORT std::string HexEncode(const char* data, size_t length);

ARROW_EXPORT std::string HexEncode(std::string_view str);

ARROW_EXPORT std::string Escape(std::string_view str);

ARROW_EXPORT Status ParseHexValue(const char* data, uint8_t* out);

namespace internal {

/// Like std::string_view::starts_with in C++20
inline bool StartsWith(std::string_view s, std::string_view prefix) {
  return s.length() >= prefix.length() &&
         (s.empty() || s.substr(0, prefix.length()) == prefix);
}

/// Like std::string_view::ends_with in C++20
inline bool EndsWith(std::string_view s, std::string_view suffix) {
  return s.length() >= suffix.length() &&
         (s.empty() || s.substr(s.length() - suffix.length()) == suffix);
}

/// \brief Split a string with a delimiter
ARROW_EXPORT
std::vector<std::string_view> SplitString(std::string_view v, char delim,
                                          int64_t limit = 0);

/// \brief Join strings with a delimiter
ARROW_EXPORT
std::string JoinStrings(const std::vector<std::string_view>& strings,
                        std::string_view delimiter);

/// \brief Join strings with a delimiter
ARROW_EXPORT
std::string JoinStrings(const std::vector<std::string>& strings,
                        std::string_view delimiter);

/// \brief Trim whitespace from left and right sides of string
ARROW_EXPORT
std::string TrimString(std::string value);

ARROW_EXPORT
bool AsciiEqualsCaseInsensitive(std::string_view left, std::string_view right);

ARROW_EXPORT
std::string AsciiToLower(std::string_view value);

ARROW_EXPORT
std::string AsciiToUpper(std::string_view value);

/// \brief Search for the first instance of a token and replace it or return nullopt if
/// the token is not found.
ARROW_EXPORT
std::optional<std::string> Replace(std::string_view s, std::string_view token,
                                   std::string_view replacement);

/// \brief Get boolean value from string
///
/// If "1", "true" (case-insensitive), returns true
/// If "0", "false" (case-insensitive), returns false
/// Otherwise, returns Status::Invalid
ARROW_EXPORT
arrow::Result<bool> ParseBoolean(std::string_view value);

#if __has_include(<charconv>)

namespace detail {
template <typename T, typename = void>
struct can_to_chars : public std::false_type {};

template <typename T>
struct can_to_chars<
    T, std::void_t<decltype(std::to_chars(std::declval<char*>(), std::declval<char*>(),
                                          std::declval<std::remove_reference_t<T>>()))>>
    : public std::true_type {};
}  // namespace detail

/// \brief Whether std::to_chars exists for the current value type.
///
/// This is useful as some C++ libraries do not implement all specified overloads
/// for std::to_chars.
template <typename T>
inline constexpr bool have_to_chars = detail::can_to_chars<T>::value;

/// \brief An ergonomic wrapper around std::to_chars, returning a std::string
///
/// For most inputs, the std::string result will not incur any heap allocation
/// thanks to small string optimization.
///
/// Compared to std::to_string, this function gives locale-agnostic results
/// and might also be faster.
template <typename T, typename... Args>
std::string ToChars(T value, Args&&... args) {
  if constexpr (!have_to_chars<T>) {
    // Some C++ standard libraries do not yet implement std::to_chars for all types,
    // in which case we have to fallback to std::string.
    return std::to_string(value);
  } else {
    // According to various sources, the GNU libstdc++ and Microsoft's C++ STL
    // allow up to 15 bytes of small string optimization, while clang's libc++
    // goes up to 22 bytes. Choose the pessimistic value.
    std::string out(15, 0);
    auto res = std::to_chars(&out.front(), &out.back(), value, args...);
    while (res.ec != std::errc{}) {
      assert(res.ec == std::errc::value_too_large);
      out.resize(out.capacity() * 2);
      res = std::to_chars(&out.front(), &out.back(), value, args...);
    }
    const auto length = res.ptr - out.data();
    assert(length <= static_cast<int64_t>(out.length()));
    out.resize(length);
    return out;
  }
}

#else  // !__has_include(<charconv>)

template <typename T>
inline constexpr bool have_to_chars = false;

template <typename T, typename... Args>
std::string ToChars(T value, Args&&... args) {
  return std::to_string(value);
}

#endif

}  // namespace internal
}  // namespace arrow

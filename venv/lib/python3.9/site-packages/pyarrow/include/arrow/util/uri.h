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
#include <string_view>
#include <utility>
#include <vector>

#include "arrow/type_fwd.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace internal {

/// \brief A parsed URI
class ARROW_EXPORT Uri {
 public:
  Uri();
  ~Uri();
  Uri(Uri&&);
  Uri& operator=(Uri&&);

  // XXX Should we use std::string_view instead?  These functions are
  // not performance-critical.

  /// The URI scheme, such as "http", or the empty string if the URI has no
  /// explicit scheme.
  std::string scheme() const;

  /// Convenience function that returns true if the scheme() is "file"
  bool is_file_scheme() const;

  /// Whether the URI has an explicit host name.  This may return true if
  /// the URI has an empty host (e.g. "file:///tmp/foo"), while it returns
  /// false is the URI has not host component at all (e.g. "file:/tmp/foo").
  bool has_host() const;
  /// The URI host name, such as "localhost", "127.0.0.1" or "::1", or the empty
  /// string is the URI does not have a host component.
  std::string host() const;

  /// The URI port number, as a string such as "80", or the empty string is the URI
  /// does not have a port number component.
  std::string port_text() const;
  /// The URI port parsed as an integer, or -1 if the URI does not have a port
  /// number component.
  int32_t port() const;

  /// The username specified in the URI.
  std::string username() const;
  /// The password specified in the URI.
  std::string password() const;

  /// The URI path component.
  std::string path() const;

  /// The URI query string
  std::string query_string() const;

  /// The URI query items
  ///
  /// Note this API doesn't allow differentiating between an empty value
  /// and a missing value, such in "a&b=1" vs. "a=&b=1".
  Result<std::vector<std::pair<std::string, std::string>>> query_items() const;

  /// Get the string representation of this URI.
  const std::string& ToString() const;

  /// Factory function to parse a URI from its string representation.
  Status Parse(const std::string& uri_string);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Percent-encode the input string, for use e.g. as a URI query parameter.
///
/// This will escape directory separators, making this function unsuitable
/// for encoding URI paths directly. See UriFromAbsolutePath() instead.
ARROW_EXPORT
std::string UriEscape(std::string_view s);

ARROW_EXPORT
std::string UriUnescape(std::string_view s);

/// Encode a host for use within a URI, such as "localhost",
/// "127.0.0.1", or "[::1]".
ARROW_EXPORT
std::string UriEncodeHost(std::string_view host);

/// Whether the string is a syntactically valid URI scheme according to RFC 3986.
ARROW_EXPORT
bool IsValidUriScheme(std::string_view s);

/// Create a file uri from a given absolute path
ARROW_EXPORT
Result<std::string> UriFromAbsolutePath(std::string_view path);

}  // namespace internal
}  // namespace arrow

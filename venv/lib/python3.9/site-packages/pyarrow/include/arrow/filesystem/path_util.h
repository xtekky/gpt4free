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

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "arrow/type_fwd.h"

namespace arrow {
namespace fs {
namespace internal {

constexpr char kSep = '/';

// Computations on abstract paths (not local paths with system-dependent behaviour).
// Abstract paths are typically used in URIs.

// Split an abstract path into its individual components.
ARROW_EXPORT
std::vector<std::string> SplitAbstractPath(const std::string& path, char sep = kSep);

// Return the extension of the file
ARROW_EXPORT
std::string GetAbstractPathExtension(const std::string& s);

// Return the parent directory and basename of an abstract path.  Both values may be
// empty.
ARROW_EXPORT
std::pair<std::string, std::string> GetAbstractPathParent(const std::string& s);

// Validate the components of an abstract path.
ARROW_EXPORT
Status ValidateAbstractPathParts(const std::vector<std::string>& parts);

// Append a non-empty stem to an abstract path.
ARROW_EXPORT
std::string ConcatAbstractPath(const std::string& base, const std::string& stem);

// Make path relative to base, if it starts with base.  Otherwise error out.
ARROW_EXPORT
Result<std::string> MakeAbstractPathRelative(const std::string& base,
                                             const std::string& path);

ARROW_EXPORT
std::string EnsureLeadingSlash(std::string_view s);

ARROW_EXPORT
std::string_view RemoveLeadingSlash(std::string_view s);

ARROW_EXPORT
std::string EnsureTrailingSlash(std::string_view s);

ARROW_EXPORT
std::string_view RemoveTrailingSlash(std::string_view s);

ARROW_EXPORT
Status AssertNoTrailingSlash(std::string_view s);

ARROW_EXPORT
bool HasLeadingSlash(std::string_view s);

ARROW_EXPORT
bool IsAncestorOf(std::string_view ancestor, std::string_view descendant);

ARROW_EXPORT
std::optional<std::string_view> RemoveAncestor(std::string_view ancestor,
                                               std::string_view descendant);

/// Return a vector of ancestors between a base path and a descendant.
/// For example,
///
/// AncestorsFromBasePath("a/b", "a/b/c/d/e") -> ["a/b/c", "a/b/c/d"]
ARROW_EXPORT
std::vector<std::string> AncestorsFromBasePath(std::string_view base_path,
                                               std::string_view descendant);

/// Given a vector of paths of directories which must be created, produce a the minimal
/// subset for passing to CreateDir(recursive=true) by removing redundant parent
/// directories
ARROW_EXPORT
std::vector<std::string> MinimalCreateDirSet(std::vector<std::string> dirs);

// Join the components of an abstract path.
template <class StringIt>
std::string JoinAbstractPath(StringIt it, StringIt end, char sep = kSep) {
  std::string path;
  for (; it != end; ++it) {
    if (it->empty()) continue;

    if (!path.empty()) {
      path += sep;
    }
    path += *it;
  }
  return path;
}

template <class StringRange>
std::string JoinAbstractPath(const StringRange& range, char sep = kSep) {
  return JoinAbstractPath(range.begin(), range.end(), sep);
}

/// Convert slashes to backslashes, on all platforms.  Mostly useful for testing.
ARROW_EXPORT
std::string ToBackslashes(std::string_view s);

/// Ensure a local path is abstract, by converting backslashes to regular slashes
/// on Windows.  Return the path unchanged on other systems.
ARROW_EXPORT
std::string ToSlashes(std::string_view s);

ARROW_EXPORT
bool IsEmptyPath(std::string_view s);

ARROW_EXPORT
bool IsLikelyUri(std::string_view s);

class ARROW_EXPORT Globber {
 public:
  ~Globber();
  explicit Globber(std::string pattern);
  bool Matches(const std::string& path);

 protected:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace internal
}  // namespace fs
}  // namespace arrow

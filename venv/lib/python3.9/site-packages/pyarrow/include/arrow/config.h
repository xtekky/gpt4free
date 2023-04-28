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

#include "arrow/status.h"
#include "arrow/util/config.h"  // IWYU pragma: export
#include "arrow/util/visibility.h"

namespace arrow {

struct BuildInfo {
  /// The packed version number, e.g. 1002003 (decimal) for Arrow 1.2.3
  int version;
  /// The "major" version number, e.g. 1 for Arrow 1.2.3
  int version_major;
  /// The "minor" version number, e.g. 2 for Arrow 1.2.3
  int version_minor;
  /// The "patch" version number, e.g. 3 for Arrow 1.2.3
  int version_patch;
  /// The version string, e.g. "1.2.3"
  std::string version_string;
  std::string so_version;
  std::string full_so_version;

  /// The CMake compiler identifier, e.g. "GNU"
  std::string compiler_id;
  std::string compiler_version;
  std::string compiler_flags;

  /// The git changeset id, if available
  std::string git_id;
  /// The git changeset description, if available
  std::string git_description;
  std::string package_kind;

  /// The uppercase build type, e.g. "DEBUG" or "RELEASE"
  std::string build_type;
};

struct RuntimeInfo {
  /// The enabled SIMD level
  ///
  /// This can be less than `detected_simd_level` if the ARROW_USER_SIMD_LEVEL
  /// environment variable is set to another value.
  std::string simd_level;

  /// The SIMD level available on the OS and CPU
  std::string detected_simd_level;

  /// Whether using the OS-based timezone database
  /// This is set at compile-time.
  bool using_os_timezone_db;

  /// The path to the timezone database; by default None.
  std::optional<std::string> timezone_db_path;
};

/// \brief Get runtime build info.
///
/// The returned values correspond to exact loaded version of the Arrow library,
/// rather than the values frozen at application compile-time through the `ARROW_*`
/// preprocessor definitions.
ARROW_EXPORT
const BuildInfo& GetBuildInfo();

/// \brief Get runtime info.
///
ARROW_EXPORT
RuntimeInfo GetRuntimeInfo();

struct GlobalOptions {
  /// Path to text timezone database. This is only configurable on Windows,
  /// which does not have a compatible OS timezone database.
  std::optional<std::string> timezone_db_path;
};

ARROW_EXPORT
Status Initialize(const GlobalOptions& options) noexcept;

}  // namespace arrow

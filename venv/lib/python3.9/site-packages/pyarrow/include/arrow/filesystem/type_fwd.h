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

namespace arrow {
namespace fs {

/// \brief FileSystem entry type
enum class FileType : int8_t {
  /// Entry is not found
  NotFound,
  /// Entry exists but its type is unknown
  ///
  /// This can designate a special file such as a Unix socket or character
  /// device, or Windows NUL / CON / ...
  Unknown,
  /// Entry is a regular file
  File,
  /// Entry is a directory
  Directory
};

struct FileInfo;

struct FileSelector;

class FileSystem;
class SubTreeFileSystem;
class SlowFileSystem;
class LocalFileSystem;
class S3FileSystem;
class GcsFileSystem;

}  // namespace fs
}  // namespace arrow

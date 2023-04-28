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

#include "arrow/json/type_fwd.h"
#include "arrow/util/visibility.h"

namespace arrow {

class DataType;
class Schema;

namespace json {

enum class UnexpectedFieldBehavior : char {
  /// Unexpected JSON fields are ignored
  Ignore,
  /// Unexpected JSON fields error out
  Error,
  /// Unexpected JSON fields are type-inferred and included in the output
  InferType
};

struct ARROW_EXPORT ParseOptions {
  // Parsing options

  /// Optional explicit schema (disables type inference on those fields)
  std::shared_ptr<Schema> explicit_schema;

  /// Whether objects may be printed across multiple lines (for example pretty-printed)
  ///
  /// If true, parsing may be slower.
  bool newlines_in_values = false;

  /// How JSON fields outside of explicit_schema (if given) are treated
  UnexpectedFieldBehavior unexpected_field_behavior = UnexpectedFieldBehavior::InferType;

  /// Create parsing options with default values
  static ParseOptions Defaults();
};

struct ARROW_EXPORT ReadOptions {
  // Reader options

  /// Whether to use the global CPU thread pool
  bool use_threads = true;
  /// Block size we request from the IO layer; also determines the size of
  /// chunks when use_threads is true
  int32_t block_size = 1 << 20;  // 1 MB

  /// Create read options with default values
  static ReadOptions Defaults();
};

}  // namespace json
}  // namespace arrow

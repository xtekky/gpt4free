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

#include <memory>
#include <string>

#include "arrow/json/options.h"
#include "arrow/status.h"
#include "arrow/util/key_value_metadata.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {

class Array;
class Buffer;
class MemoryPool;
class KeyValueMetadata;
class ResizableBuffer;

namespace json {

struct Kind {
  enum type : uint8_t {
    kNull,
    kBoolean,
    kNumber,
    kString,
    kArray,
    kObject,
    kNumberOrString
  };

  static const std::string& Name(Kind::type);

  static const std::shared_ptr<const KeyValueMetadata>& Tag(Kind::type);

  static Kind::type FromTag(const std::shared_ptr<const KeyValueMetadata>& tag);

  static Status ForType(const DataType& type, Kind::type* kind);
};

constexpr int32_t kMaxParserNumRows = 100000;

/// \class BlockParser
/// \brief A reusable block-based parser for JSON data
///
/// The parser takes a block of newline delimited JSON data and extracts Arrays
/// of unconverted strings which can be fed to a Converter to obtain a usable Array.
///
/// Note that in addition to parse errors (such as malformed JSON) some conversion
/// errors are caught at parse time:
/// - A null value in non-nullable column
/// - Change in the JSON kind of a column. For example, if an explicit schema is provided
///   which stipulates that field "a" is integral, a row of {"a": "not a number"} will
///   result in an error. This also applies to fields outside an explicit schema.
class ARROW_EXPORT BlockParser {
 public:
  virtual ~BlockParser() = default;

  /// \brief Reserve storage for scalars parsed from a block of json
  virtual Status ReserveScalarStorage(int64_t nbytes) = 0;

  /// \brief Parse a block of data
  virtual Status Parse(const std::shared_ptr<Buffer>& json) = 0;

  /// \brief Extract parsed data
  virtual Status Finish(std::shared_ptr<Array>* parsed) = 0;

  /// \brief Return the number of parsed rows
  int32_t num_rows() const { return num_rows_; }

  /// \brief Construct a BlockParser
  ///
  /// \param[in] pool MemoryPool to use when constructing parsed array
  /// \param[in] options ParseOptions to use when parsing JSON
  /// \param[out] out constructed BlockParser
  static Status Make(MemoryPool* pool, const ParseOptions& options,
                     std::unique_ptr<BlockParser>* out);

  static Status Make(const ParseOptions& options, std::unique_ptr<BlockParser>* out);

 protected:
  ARROW_DISALLOW_COPY_AND_ASSIGN(BlockParser);

  explicit BlockParser(MemoryPool* pool) : pool_(pool) {}

  MemoryPool* pool_;
  int32_t num_rows_ = 0;
};

}  // namespace json
}  // namespace arrow

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
#include <string_view>

#include "arrow/status.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {

class Buffer;

class ARROW_EXPORT BoundaryFinder {
 public:
  BoundaryFinder() = default;

  virtual ~BoundaryFinder();

  /// \brief Find the position of the first delimiter inside block
  ///
  /// `partial` is taken to be the beginning of the block, and `block`
  /// its continuation.  Also, `partial` doesn't contain a delimiter.
  ///
  /// The returned `out_pos` is relative to `block`'s start and should point
  /// to the first character after the first delimiter.
  /// `out_pos` will be -1 if no delimiter is found.
  virtual Status FindFirst(std::string_view partial, std::string_view block,
                           int64_t* out_pos) = 0;

  /// \brief Find the position of the last delimiter inside block
  ///
  /// The returned `out_pos` is relative to `block`'s start and should point
  /// to the first character after the last delimiter.
  /// `out_pos` will be -1 if no delimiter is found.
  virtual Status FindLast(std::string_view block, int64_t* out_pos) = 0;

  /// \brief Find the position of the Nth delimiter inside the block
  ///
  /// `partial` is taken to be the beginning of the block, and `block`
  /// its continuation.  Also, `partial` doesn't contain a delimiter.
  ///
  /// The returned `out_pos` is relative to `block`'s start and should point
  /// to the first character after the first delimiter.
  /// `out_pos` will be -1 if no delimiter is found.
  ///
  /// The returned `num_found` is the number of delimiters actually found
  virtual Status FindNth(std::string_view partial, std::string_view block, int64_t count,
                         int64_t* out_pos, int64_t* num_found) = 0;

  static constexpr int64_t kNoDelimiterFound = -1;

 protected:
  ARROW_DISALLOW_COPY_AND_ASSIGN(BoundaryFinder);
};

ARROW_EXPORT
std::shared_ptr<BoundaryFinder> MakeNewlineBoundaryFinder();

/// \brief A reusable block-based chunker for delimited data
///
/// The chunker takes a block of delimited data and helps carve a sub-block
/// which begins and ends on delimiters (suitable for consumption by parsers
/// which can only parse whole objects).
class ARROW_EXPORT Chunker {
 public:
  explicit Chunker(std::shared_ptr<BoundaryFinder> delimiter);
  ~Chunker();

  /// \brief Carve up a chunk in a block of data to contain only whole objects
  ///
  /// Pre-conditions:
  /// - `block` is the start of a valid block of delimited data
  ///   (i.e. starts just after a delimiter)
  ///
  /// Post-conditions:
  /// - block == whole + partial
  /// - `whole` is a valid block of delimited data
  ///   (i.e. starts just after a delimiter and ends with a delimiter)
  /// - `partial` doesn't contain an entire delimited object
  ///   (IOW: `partial` is generally small)
  ///
  /// This method will look for the last delimiter in `block` and may
  /// therefore be costly.
  ///
  /// \param[in] block data to be chunked
  /// \param[out] whole subrange of block containing whole delimited objects
  /// \param[out] partial subrange of block starting with a partial delimited object
  Status Process(std::shared_ptr<Buffer> block, std::shared_ptr<Buffer>* whole,
                 std::shared_ptr<Buffer>* partial);

  /// \brief Carve the completion of a partial object out of a block
  ///
  /// Pre-conditions:
  /// - `partial` is the start of a valid block of delimited data
  ///   (i.e. starts just after a delimiter)
  /// - `block` follows `partial` in file order
  ///
  /// Post-conditions:
  /// - block == completion + rest
  /// - `partial + completion` is a valid block of delimited data
  ///   (i.e. starts just after a delimiter and ends with a delimiter)
  /// - `completion` doesn't contain an entire delimited object
  ///   (IOW: `completion` is generally small)
  ///
  /// This method will look for the first delimiter in `block` and should
  /// therefore be reasonably cheap.
  ///
  /// \param[in] partial incomplete delimited data
  /// \param[in] block delimited data following partial
  /// \param[out] completion subrange of block containing the completion of partial
  /// \param[out] rest subrange of block containing what completion does not cover
  Status ProcessWithPartial(std::shared_ptr<Buffer> partial,
                            std::shared_ptr<Buffer> block,
                            std::shared_ptr<Buffer>* completion,
                            std::shared_ptr<Buffer>* rest);

  /// \brief Like ProcessWithPartial, but for the last block of a file
  ///
  /// This method allows for a final delimited object without a trailing delimiter
  /// (ProcessWithPartial would return an error in that case).
  ///
  /// Pre-conditions:
  /// - `partial` is the start of a valid block of delimited data
  /// - `block` follows `partial` in file order and is the last data block
  ///
  /// Post-conditions:
  /// - block == completion + rest
  /// - `partial + completion` is a valid block of delimited data
  /// - `completion` doesn't contain an entire delimited object
  ///   (IOW: `completion` is generally small)
  ///
  Status ProcessFinal(std::shared_ptr<Buffer> partial, std::shared_ptr<Buffer> block,
                      std::shared_ptr<Buffer>* completion, std::shared_ptr<Buffer>* rest);

  /// \brief Skip count number of rows
  /// Pre-conditions:
  /// - `partial` is the start of a valid block of delimited data
  ///   (i.e. starts just after a delimiter)
  /// - `block` follows `partial` in file order
  ///
  /// Post-conditions:
  /// - `count` is updated to indicate the number of rows that still need to be skipped
  /// - If `count` is > 0 then `rest` is an incomplete block that should be a future
  /// `partial`
  /// - Else `rest` could be one or more valid blocks of delimited data which need to be
  /// parsed
  ///
  /// \param[in] partial incomplete delimited data
  /// \param[in] block delimited data following partial
  /// \param[in] final whether this is the final chunk
  /// \param[in,out] count number of rows that need to be skipped
  /// \param[out] rest subrange of block containing what was not skipped
  Status ProcessSkip(std::shared_ptr<Buffer> partial, std::shared_ptr<Buffer> block,
                     bool final, int64_t* count, std::shared_ptr<Buffer>* rest);

 protected:
  ARROW_DISALLOW_COPY_AND_ASSIGN(Chunker);

  std::shared_ptr<BoundaryFinder> boundary_finder_;
};

}  // namespace arrow

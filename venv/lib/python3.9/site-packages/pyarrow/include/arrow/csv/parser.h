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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "arrow/buffer.h"
#include "arrow/csv/options.h"
#include "arrow/csv/type_fwd.h"
#include "arrow/status.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {

class MemoryPool;

namespace csv {

/// Skip at most num_rows from the given input.  The input pointer is updated
/// and the number of actually skipped rows is returns (may be less than
/// requested if the input is too short).
ARROW_EXPORT
int32_t SkipRows(const uint8_t* data, uint32_t size, int32_t num_rows,
                 const uint8_t** out_data);

class BlockParserImpl;

namespace detail {

struct ParsedValueDesc {
  uint32_t offset : 31;
  bool quoted : 1;
};

class ARROW_EXPORT DataBatch {
 public:
  explicit DataBatch(int32_t num_cols) : num_cols_(num_cols) {}

  /// \brief Return the number of parsed rows (not skipped)
  int32_t num_rows() const { return num_rows_; }
  /// \brief Return the number of parsed columns
  int32_t num_cols() const { return num_cols_; }
  /// \brief Return the total size in bytes of parsed data
  uint32_t num_bytes() const { return parsed_size_; }
  /// \brief Return the number of skipped rows
  int32_t num_skipped_rows() const { return static_cast<int32_t>(skipped_rows_.size()); }

  template <typename Visitor>
  Status VisitColumn(int32_t col_index, int64_t first_row, Visitor&& visit) const {
    using detail::ParsedValueDesc;

    int32_t batch_row = 0;
    for (size_t buf_index = 0; buf_index < values_buffers_.size(); ++buf_index) {
      const auto& values_buffer = values_buffers_[buf_index];
      const auto values = reinterpret_cast<const ParsedValueDesc*>(values_buffer->data());
      const auto max_pos =
          static_cast<int32_t>(values_buffer->size() / sizeof(ParsedValueDesc)) - 1;
      for (int32_t pos = col_index; pos < max_pos; pos += num_cols_, ++batch_row) {
        auto start = values[pos].offset;
        auto stop = values[pos + 1].offset;
        auto quoted = values[pos + 1].quoted;
        Status status = visit(parsed_ + start, stop - start, quoted);
        if (ARROW_PREDICT_FALSE(!status.ok())) {
          return DecorateWithRowNumber(std::move(status), first_row, batch_row);
        }
      }
    }
    return Status::OK();
  }

  template <typename Visitor>
  Status VisitLastRow(Visitor&& visit) const {
    using detail::ParsedValueDesc;

    const auto& values_buffer = values_buffers_.back();
    const auto values = reinterpret_cast<const ParsedValueDesc*>(values_buffer->data());
    const auto start_pos =
        static_cast<int32_t>(values_buffer->size() / sizeof(ParsedValueDesc)) -
        num_cols_ - 1;
    for (int32_t col_index = 0; col_index < num_cols_; ++col_index) {
      auto start = values[start_pos + col_index].offset;
      auto stop = values[start_pos + col_index + 1].offset;
      auto quoted = values[start_pos + col_index + 1].quoted;
      ARROW_RETURN_NOT_OK(visit(parsed_ + start, stop - start, quoted));
    }
    return Status::OK();
  }

 protected:
  Status DecorateWithRowNumber(Status&& status, int64_t first_row,
                               int32_t batch_row) const {
    if (first_row >= 0) {
      // `skipped_rows_` is in ascending order by construction, so use bisection
      // to find out how many rows were skipped before `batch_row`.
      const auto skips_before =
          std::upper_bound(skipped_rows_.begin(), skipped_rows_.end(), batch_row) -
          skipped_rows_.begin();
      status = status.WithMessage("Row #", batch_row + skips_before + first_row, ": ",
                                  status.message());
    }
    // Use return_if so that when extra context is enabled it will be added
    ARROW_RETURN_IF_(true, std::move(status), ARROW_STRINGIFY(status));
    return std::move(status);
  }

  // The number of rows in this batch (not including any skipped ones)
  int32_t num_rows_ = 0;
  // The number of columns
  int32_t num_cols_ = 0;

  // XXX should we ensure the parsed buffer is padded with 8 or 16 excess zero bytes?
  // It may help with null parsing...
  std::vector<std::shared_ptr<Buffer>> values_buffers_;
  std::shared_ptr<Buffer> parsed_buffer_;
  const uint8_t* parsed_ = NULLPTR;
  int32_t parsed_size_ = 0;

  // Record the current num_rows_ each time a row is skipped
  std::vector<int32_t> skipped_rows_;

  friend class ::arrow::csv::BlockParserImpl;
};

}  // namespace detail

constexpr int32_t kMaxParserNumRows = 100000;

/// \class BlockParser
/// \brief A reusable block-based parser for CSV data
///
/// The parser takes a block of CSV data and delimits rows and fields,
/// unquoting and unescaping them on the fly.  Parsed data is own by the
/// parser, so the original buffer can be discarded after Parse() returns.
///
/// If the block is truncated (i.e. not all data can be parsed), it is up
/// to the caller to arrange the next block to start with the trailing data.
/// Also, if the previous block ends with CR (0x0d) and a new block starts
/// with LF (0x0a), the parser will consider the leading newline as an empty
/// line; the caller should therefore strip it.
class ARROW_EXPORT BlockParser {
 public:
  explicit BlockParser(ParseOptions options, int32_t num_cols = -1,
                       int64_t first_row = -1, int32_t max_num_rows = kMaxParserNumRows);
  explicit BlockParser(MemoryPool* pool, ParseOptions options, int32_t num_cols = -1,
                       int64_t first_row = -1, int32_t max_num_rows = kMaxParserNumRows);
  ~BlockParser();

  /// \brief Parse a block of data
  ///
  /// Parse a block of CSV data, ingesting up to max_num_rows rows.
  /// The number of bytes actually parsed is returned in out_size.
  Status Parse(std::string_view data, uint32_t* out_size);

  /// \brief Parse sequential blocks of data
  ///
  /// Only the last block is allowed to be truncated.
  Status Parse(const std::vector<std::string_view>& data, uint32_t* out_size);

  /// \brief Parse the final block of data
  ///
  /// Like Parse(), but called with the final block in a file.
  /// The last row may lack a trailing line separator.
  Status ParseFinal(std::string_view data, uint32_t* out_size);

  /// \brief Parse the final sequential blocks of data
  ///
  /// Only the last block is allowed to be truncated.
  Status ParseFinal(const std::vector<std::string_view>& data, uint32_t* out_size);

  /// \brief Return the number of parsed rows
  int32_t num_rows() const { return parsed_batch().num_rows(); }
  /// \brief Return the number of parsed columns
  int32_t num_cols() const { return parsed_batch().num_cols(); }
  /// \brief Return the total size in bytes of parsed data
  uint32_t num_bytes() const { return parsed_batch().num_bytes(); }

  /// \brief Return the total number of rows including rows which were skipped
  int32_t total_num_rows() const {
    return parsed_batch().num_rows() + parsed_batch().num_skipped_rows();
  }

  /// \brief Return the row number of the first row in the block or -1 if unsupported
  int64_t first_row_num() const;

  /// \brief Visit parsed values in a column
  ///
  /// The signature of the visitor is
  /// Status(const uint8_t* data, uint32_t size, bool quoted)
  template <typename Visitor>
  Status VisitColumn(int32_t col_index, Visitor&& visit) const {
    return parsed_batch().VisitColumn(col_index, first_row_num(),
                                      std::forward<Visitor>(visit));
  }

  template <typename Visitor>
  Status VisitLastRow(Visitor&& visit) const {
    return parsed_batch().VisitLastRow(std::forward<Visitor>(visit));
  }

 protected:
  std::unique_ptr<BlockParserImpl> impl_;

  const detail::DataBatch& parsed_batch() const;
};

}  // namespace csv
}  // namespace arrow

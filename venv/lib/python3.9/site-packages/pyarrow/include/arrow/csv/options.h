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
#include <unordered_map>
#include <vector>

#include "arrow/csv/invalid_row.h"
#include "arrow/csv/type_fwd.h"
#include "arrow/io/interfaces.h"
#include "arrow/status.h"
#include "arrow/util/visibility.h"

namespace arrow {

class DataType;
class TimestampParser;

namespace csv {

// Silly workaround for https://github.com/michaeljones/breathe/issues/453
constexpr char kDefaultEscapeChar = '\\';

struct ARROW_EXPORT ParseOptions {
  // Parsing options

  /// Field delimiter
  char delimiter = ',';
  /// Whether quoting is used
  bool quoting = true;
  /// Quoting character (if `quoting` is true)
  char quote_char = '"';
  /// Whether a quote inside a value is double-quoted
  bool double_quote = true;
  /// Whether escaping is used
  bool escaping = false;
  /// Escaping character (if `escaping` is true)
  char escape_char = kDefaultEscapeChar;
  /// Whether values are allowed to contain CR (0x0d) and LF (0x0a) characters
  bool newlines_in_values = false;
  /// Whether empty lines are ignored.  If false, an empty line represents
  /// a single empty value (assuming a one-column CSV file).
  bool ignore_empty_lines = true;
  /// A handler function for rows which do not have the correct number of columns
  InvalidRowHandler invalid_row_handler;

  /// Create parsing options with default values
  static ParseOptions Defaults();

  /// \brief Test that all set options are valid
  Status Validate() const;
};

struct ARROW_EXPORT ConvertOptions {
  // Conversion options

  /// Whether to check UTF8 validity of string columns
  bool check_utf8 = true;
  /// Optional per-column types (disabling type inference on those columns)
  std::unordered_map<std::string, std::shared_ptr<DataType>> column_types;
  /// Recognized spellings for null values
  std::vector<std::string> null_values;
  /// Recognized spellings for boolean true values
  std::vector<std::string> true_values;
  /// Recognized spellings for boolean false values
  std::vector<std::string> false_values;

  /// Whether string / binary columns can have null values.
  ///
  /// If true, then strings in "null_values" are considered null for string columns.
  /// If false, then all strings are valid string values.
  bool strings_can_be_null = false;

  /// Whether quoted values can be null.
  ///
  /// If true, then strings in "null_values" are also considered null when they
  /// appear quoted in the CSV file. Otherwise, quoted values are never considered null.
  bool quoted_strings_can_be_null = true;

  /// Whether to try to automatically dict-encode string / binary data.
  /// If true, then when type inference detects a string or binary column,
  /// it is dict-encoded up to `auto_dict_max_cardinality` distinct values
  /// (per chunk), after which it switches to regular encoding.
  ///
  /// This setting is ignored for non-inferred columns (those in `column_types`).
  bool auto_dict_encode = false;
  int32_t auto_dict_max_cardinality = 50;

  /// Decimal point character for floating-point and decimal data
  char decimal_point = '.';

  // XXX Should we have a separate FilterOptions?

  /// If non-empty, indicates the names of columns from the CSV file that should
  /// be actually read and converted (in the vector's order).
  /// Columns not in this vector will be ignored.
  std::vector<std::string> include_columns;
  /// If false, columns in `include_columns` but not in the CSV file will error out.
  /// If true, columns in `include_columns` but not in the CSV file will produce
  /// a column of nulls (whose type is selected using `column_types`,
  /// or null by default)
  /// This option is ignored if `include_columns` is empty.
  bool include_missing_columns = false;

  /// User-defined timestamp parsers, using the virtual parser interface in
  /// arrow/util/value_parsing.h. More than one parser can be specified, and
  /// the CSV conversion logic will try parsing values starting from the
  /// beginning of this vector. If no parsers are specified, we use the default
  /// built-in ISO-8601 parser.
  std::vector<std::shared_ptr<TimestampParser>> timestamp_parsers;

  /// Create conversion options with default values, including conventional
  /// values for `null_values`, `true_values` and `false_values`
  static ConvertOptions Defaults();

  /// \brief Test that all set options are valid
  Status Validate() const;
};

struct ARROW_EXPORT ReadOptions {
  // Reader options

  /// Whether to use the global CPU thread pool
  bool use_threads = true;

  /// \brief Block size we request from the IO layer.
  ///
  /// This will determine multi-threading granularity as well as
  /// the size of individual record batches.
  /// Minimum valid value for block size is 1
  int32_t block_size = 1 << 20;  // 1 MB

  /// Number of header rows to skip (not including the row of column names, if any)
  int32_t skip_rows = 0;

  /// Number of rows to skip after the column names are read, if any
  int32_t skip_rows_after_names = 0;

  /// Column names for the target table.
  /// If empty, fall back on autogenerate_column_names.
  std::vector<std::string> column_names;

  /// Whether to autogenerate column names if `column_names` is empty.
  /// If true, column names will be of the form "f0", "f1"...
  /// If false, column names will be read from the first CSV row after `skip_rows`.
  bool autogenerate_column_names = false;

  /// Create read options with default values
  static ReadOptions Defaults();

  /// \brief Test that all set options are valid
  Status Validate() const;
};

/// \brief Quoting style for CSV writing
enum class ARROW_EXPORT QuotingStyle {
  /// Only enclose values in quotes which need them, because their CSV rendering can
  /// contain quotes itself (e.g. strings or binary values)
  Needed,
  /// Enclose all valid values in quotes. Nulls are not quoted. May cause readers to
  /// interpret all values as strings if schema is inferred.
  AllValid,
  /// Do not enclose any values in quotes. Prevents values from containing quotes ("),
  /// cell delimiters (,) or line endings (\\r, \\n), (following RFC4180). If values
  /// contain these characters, an error is caused when attempting to write.
  None
};

struct ARROW_EXPORT WriteOptions {
  /// Whether to write an initial header line with column names
  bool include_header = true;

  /// \brief Maximum number of rows processed at a time
  ///
  /// The CSV writer converts and writes data in batches of N rows.
  /// This number can impact performance.
  int32_t batch_size = 1024;

  /// Field delimiter
  char delimiter = ',';

  /// \brief The string to write for null values. Quotes are not allowed in this string.
  std::string null_string;

  /// \brief IO context for writing.
  io::IOContext io_context;

  /// \brief The end of line character to use for ending rows
  std::string eol = "\n";

  /// \brief Quoting style
  QuotingStyle quoting_style = QuotingStyle::Needed;

  /// Create write options with default values
  static WriteOptions Defaults();

  /// \brief Test that all set options are valid
  Status Validate() const;
};

}  // namespace csv
}  // namespace arrow

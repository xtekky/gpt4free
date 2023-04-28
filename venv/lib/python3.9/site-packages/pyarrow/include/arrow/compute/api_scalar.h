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

// Eager evaluation convenience APIs for invoking common functions, including
// necessary memory allocations

#pragma once

#include <optional>
#include <string>
#include <utility>

#include "arrow/compute/function.h"
#include "arrow/compute/type_fwd.h"
#include "arrow/datum.h"
#include "arrow/result.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace compute {

/// \addtogroup compute-concrete-options
///
/// @{

class ARROW_EXPORT ArithmeticOptions : public FunctionOptions {
 public:
  explicit ArithmeticOptions(bool check_overflow = false);
  static constexpr char const kTypeName[] = "ArithmeticOptions";
  bool check_overflow;
};

class ARROW_EXPORT ElementWiseAggregateOptions : public FunctionOptions {
 public:
  explicit ElementWiseAggregateOptions(bool skip_nulls = true);
  static constexpr char const kTypeName[] = "ElementWiseAggregateOptions";
  static ElementWiseAggregateOptions Defaults() { return ElementWiseAggregateOptions{}; }
  bool skip_nulls;
};

/// Rounding and tie-breaking modes for round compute functions.
/// Additional details and examples are provided in compute.rst.
enum class RoundMode : int8_t {
  /// Round to nearest integer less than or equal in magnitude (aka "floor")
  DOWN,
  /// Round to nearest integer greater than or equal in magnitude (aka "ceil")
  UP,
  /// Get the integral part without fractional digits (aka "trunc")
  TOWARDS_ZERO,
  /// Round negative values with DOWN rule
  /// and positive values with UP rule (aka "away from zero")
  TOWARDS_INFINITY,
  /// Round ties with DOWN rule (also called "round half towards negative infinity")
  HALF_DOWN,
  /// Round ties with UP rule (also called "round half towards positive infinity")
  HALF_UP,
  /// Round ties with TOWARDS_ZERO rule (also called "round half away from infinity")
  HALF_TOWARDS_ZERO,
  /// Round ties with TOWARDS_INFINITY rule (also called "round half away from zero")
  HALF_TOWARDS_INFINITY,
  /// Round ties to nearest even integer
  HALF_TO_EVEN,
  /// Round ties to nearest odd integer
  HALF_TO_ODD,
};

class ARROW_EXPORT RoundOptions : public FunctionOptions {
 public:
  explicit RoundOptions(int64_t ndigits = 0,
                        RoundMode round_mode = RoundMode::HALF_TO_EVEN);
  static constexpr char const kTypeName[] = "RoundOptions";
  static RoundOptions Defaults() { return RoundOptions(); }
  /// Rounding precision (number of digits to round to)
  int64_t ndigits;
  /// Rounding and tie-breaking mode
  RoundMode round_mode;
};

enum class CalendarUnit : int8_t {
  NANOSECOND,
  MICROSECOND,
  MILLISECOND,
  SECOND,
  MINUTE,
  HOUR,
  DAY,
  WEEK,
  MONTH,
  QUARTER,
  YEAR
};

class ARROW_EXPORT RoundTemporalOptions : public FunctionOptions {
 public:
  explicit RoundTemporalOptions(int multiple = 1, CalendarUnit unit = CalendarUnit::DAY,
                                bool week_starts_monday = true,
                                bool ceil_is_strictly_greater = false,
                                bool calendar_based_origin = false);
  static constexpr char const kTypeName[] = "RoundTemporalOptions";
  static RoundTemporalOptions Defaults() { return RoundTemporalOptions(); }

  /// Number of units to round to
  int multiple;
  /// The unit used for rounding of time
  CalendarUnit unit;
  /// What day does the week start with (Monday=true, Sunday=false)
  bool week_starts_monday;
  /// Enable this flag to return a rounded value that is strictly greater than the input.
  /// For example: ceiling 1970-01-01T00:00:00 to 3 hours would yield 1970-01-01T03:00:00
  /// if set to true and 1970-01-01T00:00:00 if set to false.
  /// This applies for ceiling only.
  bool ceil_is_strictly_greater;
  /// By default time is rounded to a multiple of units since 1970-01-01T00:00:00.
  /// By setting calendar_based_origin to true, time will be rounded to a number
  /// of units since the last greater calendar unit.
  /// For example: rounding to a multiple of days since the beginning of the month or
  /// to hours since the beginning of the day.
  /// Exceptions: week and quarter are not used as greater units, therefore days will
  /// will be rounded to the beginning of the month not week. Greater unit of week
  /// is year.
  /// Note that ceiling and rounding might change sorting order of an array near greater
  /// unit change. For example rounding YYYY-mm-dd 23:00:00 to 5 hours will ceil and
  /// round to YYYY-mm-dd+1 01:00:00 and floor to YYYY-mm-dd 20:00:00. On the other hand
  /// YYYY-mm-dd+1 00:00:00 will ceil, round and floor to YYYY-mm-dd+1 00:00:00. This
  /// can break the order of an already ordered array.
  bool calendar_based_origin;
};

class ARROW_EXPORT RoundToMultipleOptions : public FunctionOptions {
 public:
  explicit RoundToMultipleOptions(double multiple = 1.0,
                                  RoundMode round_mode = RoundMode::HALF_TO_EVEN);
  explicit RoundToMultipleOptions(std::shared_ptr<Scalar> multiple,
                                  RoundMode round_mode = RoundMode::HALF_TO_EVEN);
  static constexpr char const kTypeName[] = "RoundToMultipleOptions";
  static RoundToMultipleOptions Defaults() { return RoundToMultipleOptions(); }
  /// Rounding scale (multiple to round to).
  ///
  /// Should be a positive numeric scalar of a type compatible with the
  /// argument to be rounded. The cast kernel is used to convert the rounding
  /// multiple to match the result type.
  std::shared_ptr<Scalar> multiple;
  /// Rounding and tie-breaking mode
  RoundMode round_mode;
};

/// Options for var_args_join.
class ARROW_EXPORT JoinOptions : public FunctionOptions {
 public:
  /// How to handle null values. (A null separator always results in a null output.)
  enum NullHandlingBehavior {
    /// A null in any input results in a null in the output.
    EMIT_NULL,
    /// Nulls in inputs are skipped.
    SKIP,
    /// Nulls in inputs are replaced with the replacement string.
    REPLACE,
  };
  explicit JoinOptions(NullHandlingBehavior null_handling = EMIT_NULL,
                       std::string null_replacement = "");
  static constexpr char const kTypeName[] = "JoinOptions";
  static JoinOptions Defaults() { return JoinOptions(); }
  NullHandlingBehavior null_handling;
  std::string null_replacement;
};

class ARROW_EXPORT MatchSubstringOptions : public FunctionOptions {
 public:
  explicit MatchSubstringOptions(std::string pattern, bool ignore_case = false);
  MatchSubstringOptions();
  static constexpr char const kTypeName[] = "MatchSubstringOptions";

  /// The exact substring (or regex, depending on kernel) to look for inside input values.
  std::string pattern;
  /// Whether to perform a case-insensitive match.
  bool ignore_case;
};

class ARROW_EXPORT SplitOptions : public FunctionOptions {
 public:
  explicit SplitOptions(int64_t max_splits = -1, bool reverse = false);
  static constexpr char const kTypeName[] = "SplitOptions";

  /// Maximum number of splits allowed, or unlimited when -1
  int64_t max_splits;
  /// Start splitting from the end of the string (only relevant when max_splits != -1)
  bool reverse;
};

class ARROW_EXPORT SplitPatternOptions : public FunctionOptions {
 public:
  explicit SplitPatternOptions(std::string pattern, int64_t max_splits = -1,
                               bool reverse = false);
  SplitPatternOptions();
  static constexpr char const kTypeName[] = "SplitPatternOptions";

  /// The exact substring to split on.
  std::string pattern;
  /// Maximum number of splits allowed, or unlimited when -1
  int64_t max_splits;
  /// Start splitting from the end of the string (only relevant when max_splits != -1)
  bool reverse;
};

class ARROW_EXPORT ReplaceSliceOptions : public FunctionOptions {
 public:
  explicit ReplaceSliceOptions(int64_t start, int64_t stop, std::string replacement);
  ReplaceSliceOptions();
  static constexpr char const kTypeName[] = "ReplaceSliceOptions";

  /// Index to start slicing at
  int64_t start;
  /// Index to stop slicing at
  int64_t stop;
  /// String to replace the slice with
  std::string replacement;
};

class ARROW_EXPORT ReplaceSubstringOptions : public FunctionOptions {
 public:
  explicit ReplaceSubstringOptions(std::string pattern, std::string replacement,
                                   int64_t max_replacements = -1);
  ReplaceSubstringOptions();
  static constexpr char const kTypeName[] = "ReplaceSubstringOptions";

  /// Pattern to match, literal, or regular expression depending on which kernel is used
  std::string pattern;
  /// String to replace the pattern with
  std::string replacement;
  /// Max number of substrings to replace (-1 means unbounded)
  int64_t max_replacements;
};

class ARROW_EXPORT ExtractRegexOptions : public FunctionOptions {
 public:
  explicit ExtractRegexOptions(std::string pattern);
  ExtractRegexOptions();
  static constexpr char const kTypeName[] = "ExtractRegexOptions";

  /// Regular expression with named capture fields
  std::string pattern;
};

/// Options for IsIn and IndexIn functions
class ARROW_EXPORT SetLookupOptions : public FunctionOptions {
 public:
  explicit SetLookupOptions(Datum value_set, bool skip_nulls = false);
  SetLookupOptions();
  static constexpr char const kTypeName[] = "SetLookupOptions";

  /// The set of values to look up input values into.
  Datum value_set;
  /// Whether nulls in `value_set` count for lookup.
  ///
  /// If true, any null in `value_set` is ignored and nulls in the input
  /// produce null (IndexIn) or false (IsIn) values in the output.
  /// If false, any null in `value_set` is successfully matched in
  /// the input.
  bool skip_nulls;
};

/// Options for struct_field function
class ARROW_EXPORT StructFieldOptions : public FunctionOptions {
 public:
  explicit StructFieldOptions(std::vector<int> indices);
  explicit StructFieldOptions(std::initializer_list<int>);
  explicit StructFieldOptions(FieldRef field_ref);
  StructFieldOptions();
  static constexpr char const kTypeName[] = "StructFieldOptions";

  /// The FieldRef specifying what to extract from struct or union.
  FieldRef field_ref;
};

class ARROW_EXPORT StrptimeOptions : public FunctionOptions {
 public:
  explicit StrptimeOptions(std::string format, TimeUnit::type unit,
                           bool error_is_null = false);
  StrptimeOptions();
  static constexpr char const kTypeName[] = "StrptimeOptions";

  /// The desired format string.
  std::string format;
  /// The desired time resolution
  TimeUnit::type unit;
  /// Return null on parsing errors if true or raise if false
  bool error_is_null;
};

class ARROW_EXPORT StrftimeOptions : public FunctionOptions {
 public:
  explicit StrftimeOptions(std::string format, std::string locale = "C");
  StrftimeOptions();

  static constexpr char const kTypeName[] = "StrftimeOptions";

  static constexpr const char* kDefaultFormat = "%Y-%m-%dT%H:%M:%S";

  /// The desired format string.
  std::string format;
  /// The desired output locale string.
  std::string locale;
};

class ARROW_EXPORT PadOptions : public FunctionOptions {
 public:
  explicit PadOptions(int64_t width, std::string padding = " ");
  PadOptions();
  static constexpr char const kTypeName[] = "PadOptions";

  /// The desired string length.
  int64_t width;
  /// What to pad the string with. Should be one codepoint (Unicode)/byte (ASCII).
  std::string padding;
};

class ARROW_EXPORT TrimOptions : public FunctionOptions {
 public:
  explicit TrimOptions(std::string characters);
  TrimOptions();
  static constexpr char const kTypeName[] = "TrimOptions";

  /// The individual characters to be trimmed from the string.
  std::string characters;
};

class ARROW_EXPORT SliceOptions : public FunctionOptions {
 public:
  explicit SliceOptions(int64_t start, int64_t stop = std::numeric_limits<int64_t>::max(),
                        int64_t step = 1);
  SliceOptions();
  static constexpr char const kTypeName[] = "SliceOptions";
  int64_t start, stop, step;
};

class ARROW_EXPORT ListSliceOptions : public FunctionOptions {
 public:
  explicit ListSliceOptions(int64_t start, std::optional<int64_t> stop = std::nullopt,
                            int64_t step = 1,
                            std::optional<bool> return_fixed_size_list = std::nullopt);
  ListSliceOptions();
  static constexpr char const kTypeName[] = "ListSliceOptions";
  /// The start of list slicing.
  int64_t start;
  /// Optional stop of list slicing. If not set, then slice to end. (NotImplemented)
  std::optional<int64_t> stop;
  /// Slicing step
  int64_t step;
  // Whether to return a FixedSizeListArray. If true _and_ stop is after
  // a list element's length, nulls will be appended to create the requested slice size.
  // Default of `nullopt` will return whatever type it got in.
  std::optional<bool> return_fixed_size_list;
};

class ARROW_EXPORT NullOptions : public FunctionOptions {
 public:
  explicit NullOptions(bool nan_is_null = false);
  static constexpr char const kTypeName[] = "NullOptions";
  static NullOptions Defaults() { return NullOptions{}; }

  bool nan_is_null;
};

enum CompareOperator : int8_t {
  EQUAL,
  NOT_EQUAL,
  GREATER,
  GREATER_EQUAL,
  LESS,
  LESS_EQUAL,
};

struct ARROW_EXPORT CompareOptions {
  explicit CompareOptions(CompareOperator op) : op(op) {}
  CompareOptions() : CompareOptions(CompareOperator::EQUAL) {}
  enum CompareOperator op;
};

class ARROW_EXPORT MakeStructOptions : public FunctionOptions {
 public:
  MakeStructOptions(std::vector<std::string> n, std::vector<bool> r,
                    std::vector<std::shared_ptr<const KeyValueMetadata>> m);
  explicit MakeStructOptions(std::vector<std::string> n);
  MakeStructOptions();
  static constexpr char const kTypeName[] = "MakeStructOptions";

  /// Names for wrapped columns
  std::vector<std::string> field_names;

  /// Nullability bits for wrapped columns
  std::vector<bool> field_nullability;

  /// Metadata attached to wrapped columns
  std::vector<std::shared_ptr<const KeyValueMetadata>> field_metadata;
};

struct ARROW_EXPORT DayOfWeekOptions : public FunctionOptions {
 public:
  explicit DayOfWeekOptions(bool count_from_zero = true, uint32_t week_start = 1);
  static constexpr char const kTypeName[] = "DayOfWeekOptions";
  static DayOfWeekOptions Defaults() { return DayOfWeekOptions(); }

  /// Number days from 0 if true and from 1 if false
  bool count_from_zero;
  /// What day does the week start with (Monday=1, Sunday=7).
  /// The numbering is unaffected by the count_from_zero parameter.
  uint32_t week_start;
};

/// Used to control timestamp timezone conversion and handling ambiguous/nonexistent
/// times.
struct ARROW_EXPORT AssumeTimezoneOptions : public FunctionOptions {
 public:
  /// \brief How to interpret ambiguous local times that can be interpreted as
  /// multiple instants (normally two) due to DST shifts.
  ///
  /// AMBIGUOUS_EARLIEST emits the earliest instant amongst possible interpretations.
  /// AMBIGUOUS_LATEST emits the latest instant amongst possible interpretations.
  enum Ambiguous { AMBIGUOUS_RAISE, AMBIGUOUS_EARLIEST, AMBIGUOUS_LATEST };

  /// \brief How to handle local times that do not exist due to DST shifts.
  ///
  /// NONEXISTENT_EARLIEST emits the instant "just before" the DST shift instant
  /// in the given timestamp precision (for example, for a nanoseconds precision
  /// timestamp, this is one nanosecond before the DST shift instant).
  /// NONEXISTENT_LATEST emits the DST shift instant.
  enum Nonexistent { NONEXISTENT_RAISE, NONEXISTENT_EARLIEST, NONEXISTENT_LATEST };

  explicit AssumeTimezoneOptions(std::string timezone,
                                 Ambiguous ambiguous = AMBIGUOUS_RAISE,
                                 Nonexistent nonexistent = NONEXISTENT_RAISE);
  AssumeTimezoneOptions();
  static constexpr char const kTypeName[] = "AssumeTimezoneOptions";

  /// Timezone to convert timestamps from
  std::string timezone;

  /// How to interpret ambiguous local times (due to DST shifts)
  Ambiguous ambiguous;
  /// How to interpret non-existent local times (due to DST shifts)
  Nonexistent nonexistent;
};

struct ARROW_EXPORT WeekOptions : public FunctionOptions {
 public:
  explicit WeekOptions(bool week_starts_monday = true, bool count_from_zero = false,
                       bool first_week_is_fully_in_year = false);
  static constexpr char const kTypeName[] = "WeekOptions";
  static WeekOptions Defaults() { return WeekOptions{}; }
  static WeekOptions ISODefaults() {
    return WeekOptions{/*week_starts_monday*/ true,
                       /*count_from_zero=*/false,
                       /*first_week_is_fully_in_year=*/false};
  }
  static WeekOptions USDefaults() {
    return WeekOptions{/*week_starts_monday*/ false,
                       /*count_from_zero=*/false,
                       /*first_week_is_fully_in_year=*/false};
  }

  /// What day does the week start with (Monday=true, Sunday=false)
  bool week_starts_monday;
  /// Dates from current year that fall into last ISO week of the previous year return
  /// 0 if true and 52 or 53 if false.
  bool count_from_zero;
  /// Must the first week be fully in January (true), or is a week that begins on
  /// December 29, 30, or 31 considered to be the first week of the new year (false)?
  bool first_week_is_fully_in_year;
};

struct ARROW_EXPORT Utf8NormalizeOptions : public FunctionOptions {
 public:
  enum Form { NFC, NFKC, NFD, NFKD };

  explicit Utf8NormalizeOptions(Form form = NFC);
  static Utf8NormalizeOptions Defaults() { return Utf8NormalizeOptions(); }
  static constexpr char const kTypeName[] = "Utf8NormalizeOptions";

  /// The Unicode normalization form to apply
  Form form;
};

class ARROW_EXPORT RandomOptions : public FunctionOptions {
 public:
  enum Initializer { SystemRandom, Seed };

  static RandomOptions FromSystemRandom() { return RandomOptions{SystemRandom, 0}; }
  static RandomOptions FromSeed(uint64_t seed) { return RandomOptions{Seed, seed}; }

  RandomOptions(Initializer initializer, uint64_t seed);
  RandomOptions();
  static constexpr char const kTypeName[] = "RandomOptions";
  static RandomOptions Defaults() { return RandomOptions(); }

  /// The type of initialization for random number generation - system or provided seed.
  Initializer initializer;
  /// The seed value used to initialize the random number generation.
  uint64_t seed;
};

/// Options for map_lookup function
class ARROW_EXPORT MapLookupOptions : public FunctionOptions {
 public:
  enum Occurrence {
    /// Return the first matching value
    FIRST,
    /// Return the last matching value
    LAST,
    /// Return all matching values
    ALL
  };

  explicit MapLookupOptions(std::shared_ptr<Scalar> query_key, Occurrence occurrence);
  MapLookupOptions();

  constexpr static char const kTypeName[] = "MapLookupOptions";

  /// The key to lookup in the map
  std::shared_ptr<Scalar> query_key;

  /// Whether to return the first, last, or all matching values
  Occurrence occurrence;
};

/// @}

/// \brief Get the absolute value of a value.
///
/// If argument is null the result will be null.
///
/// \param[in] arg the value transformed
/// \param[in] options arithmetic options (overflow handling), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise absolute value
ARROW_EXPORT
Result<Datum> AbsoluteValue(const Datum& arg,
                            ArithmeticOptions options = ArithmeticOptions(),
                            ExecContext* ctx = NULLPTR);

/// \brief Add two values together. Array values must be the same length. If
/// either addend is null the result will be null.
///
/// \param[in] left the first addend
/// \param[in] right the second addend
/// \param[in] options arithmetic options (overflow handling), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise sum
ARROW_EXPORT
Result<Datum> Add(const Datum& left, const Datum& right,
                  ArithmeticOptions options = ArithmeticOptions(),
                  ExecContext* ctx = NULLPTR);

/// \brief Subtract two values. Array values must be the same length. If the
/// minuend or subtrahend is null the result will be null.
///
/// \param[in] left the value subtracted from (minuend)
/// \param[in] right the value by which the minuend is reduced (subtrahend)
/// \param[in] options arithmetic options (overflow handling), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise difference
ARROW_EXPORT
Result<Datum> Subtract(const Datum& left, const Datum& right,
                       ArithmeticOptions options = ArithmeticOptions(),
                       ExecContext* ctx = NULLPTR);

/// \brief Multiply two values. Array values must be the same length. If either
/// factor is null the result will be null.
///
/// \param[in] left the first factor
/// \param[in] right the second factor
/// \param[in] options arithmetic options (overflow handling), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise product
ARROW_EXPORT
Result<Datum> Multiply(const Datum& left, const Datum& right,
                       ArithmeticOptions options = ArithmeticOptions(),
                       ExecContext* ctx = NULLPTR);

/// \brief Divide two values. Array values must be the same length. If either
/// argument is null the result will be null. For integer types, if there is
/// a zero divisor, an error will be raised.
///
/// \param[in] left the dividend
/// \param[in] right the divisor
/// \param[in] options arithmetic options (enable/disable overflow checking), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise quotient
ARROW_EXPORT
Result<Datum> Divide(const Datum& left, const Datum& right,
                     ArithmeticOptions options = ArithmeticOptions(),
                     ExecContext* ctx = NULLPTR);

/// \brief Negate values.
///
/// If argument is null the result will be null.
///
/// \param[in] arg the value negated
/// \param[in] options arithmetic options (overflow handling), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise negation
ARROW_EXPORT
Result<Datum> Negate(const Datum& arg, ArithmeticOptions options = ArithmeticOptions(),
                     ExecContext* ctx = NULLPTR);

/// \brief Raise the values of base array to the power of the exponent array values.
/// Array values must be the same length. If either base or exponent is null the result
/// will be null.
///
/// \param[in] left the base
/// \param[in] right the exponent
/// \param[in] options arithmetic options (enable/disable overflow checking), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise base value raised to the power of exponent
ARROW_EXPORT
Result<Datum> Power(const Datum& left, const Datum& right,
                    ArithmeticOptions options = ArithmeticOptions(),
                    ExecContext* ctx = NULLPTR);

/// \brief Raise Euler's number to the power of specified exponent, element-wise.
/// If the exponent value is null the result will be null.
///
/// \param[in] arg the exponent
/// \param[in] ctx the function execution context, optional
/// \return the element-wise Euler's number raised to the power of exponent
ARROW_EXPORT
Result<Datum> Exp(const Datum& arg, ExecContext* ctx = NULLPTR);

/// \brief Left shift the left array by the right array. Array values must be the
/// same length. If either operand is null, the result will be null.
///
/// \param[in] left the value to shift
/// \param[in] right the value to shift by
/// \param[in] options arithmetic options (enable/disable overflow checking), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise left value shifted left by the right value
ARROW_EXPORT
Result<Datum> ShiftLeft(const Datum& left, const Datum& right,
                        ArithmeticOptions options = ArithmeticOptions(),
                        ExecContext* ctx = NULLPTR);

/// \brief Right shift the left array by the right array. Array values must be the
/// same length. If either operand is null, the result will be null. Performs a
/// logical shift for unsigned values, and an arithmetic shift for signed values.
///
/// \param[in] left the value to shift
/// \param[in] right the value to shift by
/// \param[in] options arithmetic options (enable/disable overflow checking), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise left value shifted right by the right value
ARROW_EXPORT
Result<Datum> ShiftRight(const Datum& left, const Datum& right,
                         ArithmeticOptions options = ArithmeticOptions(),
                         ExecContext* ctx = NULLPTR);

/// \brief Compute the sine of the array values.
/// \param[in] arg The values to compute the sine for.
/// \param[in] options arithmetic options (enable/disable overflow checking), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise sine of the values
ARROW_EXPORT
Result<Datum> Sin(const Datum& arg, ArithmeticOptions options = ArithmeticOptions(),
                  ExecContext* ctx = NULLPTR);

/// \brief Compute the cosine of the array values.
/// \param[in] arg The values to compute the cosine for.
/// \param[in] options arithmetic options (enable/disable overflow checking), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise cosine of the values
ARROW_EXPORT
Result<Datum> Cos(const Datum& arg, ArithmeticOptions options = ArithmeticOptions(),
                  ExecContext* ctx = NULLPTR);

/// \brief Compute the inverse sine (arcsine) of the array values.
/// \param[in] arg The values to compute the inverse sine for.
/// \param[in] options arithmetic options (enable/disable overflow checking), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise inverse sine of the values
ARROW_EXPORT
Result<Datum> Asin(const Datum& arg, ArithmeticOptions options = ArithmeticOptions(),
                   ExecContext* ctx = NULLPTR);

/// \brief Compute the inverse cosine (arccosine) of the array values.
/// \param[in] arg The values to compute the inverse cosine for.
/// \param[in] options arithmetic options (enable/disable overflow checking), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise inverse cosine of the values
ARROW_EXPORT
Result<Datum> Acos(const Datum& arg, ArithmeticOptions options = ArithmeticOptions(),
                   ExecContext* ctx = NULLPTR);

/// \brief Compute the tangent of the array values.
/// \param[in] arg The values to compute the tangent for.
/// \param[in] options arithmetic options (enable/disable overflow checking), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise tangent of the values
ARROW_EXPORT
Result<Datum> Tan(const Datum& arg, ArithmeticOptions options = ArithmeticOptions(),
                  ExecContext* ctx = NULLPTR);

/// \brief Compute the inverse tangent (arctangent) of the array values.
/// \param[in] arg The values to compute the inverse tangent for.
/// \param[in] ctx the function execution context, optional
/// \return the elementwise inverse tangent of the values
ARROW_EXPORT
Result<Datum> Atan(const Datum& arg, ExecContext* ctx = NULLPTR);

/// \brief Compute the inverse tangent (arctangent) of y/x, using the
/// argument signs to determine the correct quadrant.
/// \param[in] y The y-values to compute the inverse tangent for.
/// \param[in] x The x-values to compute the inverse tangent for.
/// \param[in] ctx the function execution context, optional
/// \return the elementwise inverse tangent of the values
ARROW_EXPORT
Result<Datum> Atan2(const Datum& y, const Datum& x, ExecContext* ctx = NULLPTR);

/// \brief Get the natural log of a value.
///
/// If argument is null the result will be null.
///
/// \param[in] arg The values to compute the logarithm for.
/// \param[in] options arithmetic options (overflow handling), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise natural log
ARROW_EXPORT
Result<Datum> Ln(const Datum& arg, ArithmeticOptions options = ArithmeticOptions(),
                 ExecContext* ctx = NULLPTR);

/// \brief Get the log base 10 of a value.
///
/// If argument is null the result will be null.
///
/// \param[in] arg The values to compute the logarithm for.
/// \param[in] options arithmetic options (overflow handling), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise log base 10
ARROW_EXPORT
Result<Datum> Log10(const Datum& arg, ArithmeticOptions options = ArithmeticOptions(),
                    ExecContext* ctx = NULLPTR);

/// \brief Get the log base 2 of a value.
///
/// If argument is null the result will be null.
///
/// \param[in] arg The values to compute the logarithm for.
/// \param[in] options arithmetic options (overflow handling), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise log base 2
ARROW_EXPORT
Result<Datum> Log2(const Datum& arg, ArithmeticOptions options = ArithmeticOptions(),
                   ExecContext* ctx = NULLPTR);

/// \brief Get the natural log of (1 + value).
///
/// If argument is null the result will be null.
/// This function may be more accurate than Log(1 + value) for values close to zero.
///
/// \param[in] arg The values to compute the logarithm for.
/// \param[in] options arithmetic options (overflow handling), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise natural log
ARROW_EXPORT
Result<Datum> Log1p(const Datum& arg, ArithmeticOptions options = ArithmeticOptions(),
                    ExecContext* ctx = NULLPTR);

/// \brief Get the log of a value to the given base.
///
/// If argument is null the result will be null.
///
/// \param[in] arg The values to compute the logarithm for.
/// \param[in] base The given base.
/// \param[in] options arithmetic options (overflow handling), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise log to the given base
ARROW_EXPORT
Result<Datum> Logb(const Datum& arg, const Datum& base,
                   ArithmeticOptions options = ArithmeticOptions(),
                   ExecContext* ctx = NULLPTR);

/// \brief Get the square-root of a value.
///
/// If argument is null the result will be null.
///
/// \param[in] arg The values to compute the square-root for.
/// \param[in] options arithmetic options (overflow handling), optional
/// \param[in] ctx the function execution context, optional
/// \return the elementwise square-root
ARROW_EXPORT
Result<Datum> Sqrt(const Datum& arg, ArithmeticOptions options = ArithmeticOptions(),
                   ExecContext* ctx = NULLPTR);

/// \brief Round to the nearest integer less than or equal in magnitude to the
/// argument.
///
/// If argument is null the result will be null.
///
/// \param[in] arg the value to round
/// \param[in] ctx the function execution context, optional
/// \return the rounded value
ARROW_EXPORT
Result<Datum> Floor(const Datum& arg, ExecContext* ctx = NULLPTR);

/// \brief Round to the nearest integer greater than or equal in magnitude to the
/// argument.
///
/// If argument is null the result will be null.
///
/// \param[in] arg the value to round
/// \param[in] ctx the function execution context, optional
/// \return the rounded value
ARROW_EXPORT
Result<Datum> Ceil(const Datum& arg, ExecContext* ctx = NULLPTR);

/// \brief Get the integral part without fractional digits.
///
/// If argument is null the result will be null.
///
/// \param[in] arg the value to truncate
/// \param[in] ctx the function execution context, optional
/// \return the truncated value
ARROW_EXPORT
Result<Datum> Trunc(const Datum& arg, ExecContext* ctx = NULLPTR);

/// \brief Find the element-wise maximum of any number of arrays or scalars.
/// Array values must be the same length.
///
/// \param[in] args arrays or scalars to operate on.
/// \param[in] options options for handling nulls, optional
/// \param[in] ctx the function execution context, optional
/// \return the element-wise maximum
ARROW_EXPORT
Result<Datum> MaxElementWise(
    const std::vector<Datum>& args,
    ElementWiseAggregateOptions options = ElementWiseAggregateOptions::Defaults(),
    ExecContext* ctx = NULLPTR);

/// \brief Find the element-wise minimum of any number of arrays or scalars.
/// Array values must be the same length.
///
/// \param[in] args arrays or scalars to operate on.
/// \param[in] options options for handling nulls, optional
/// \param[in] ctx the function execution context, optional
/// \return the element-wise minimum
ARROW_EXPORT
Result<Datum> MinElementWise(
    const std::vector<Datum>& args,
    ElementWiseAggregateOptions options = ElementWiseAggregateOptions::Defaults(),
    ExecContext* ctx = NULLPTR);

/// \brief Get the sign of a value. Array values can be of arbitrary length. If argument
/// is null the result will be null.
///
/// \param[in] arg the value to extract sign from
/// \param[in] ctx the function execution context, optional
/// \return the element-wise sign function
ARROW_EXPORT
Result<Datum> Sign(const Datum& arg, ExecContext* ctx = NULLPTR);

/// \brief Round a value to a given precision.
///
/// If argument is null the result will be null.
///
/// \param[in] arg the value rounded
/// \param[in] options rounding options (rounding mode and number of digits), optional
/// \param[in] ctx the function execution context, optional
/// \return the element-wise rounded value
ARROW_EXPORT
Result<Datum> Round(const Datum& arg, RoundOptions options = RoundOptions::Defaults(),
                    ExecContext* ctx = NULLPTR);

/// \brief Round a value to a given multiple.
///
/// If argument is null the result will be null.
///
/// \param[in] arg the value to round
/// \param[in] options rounding options (rounding mode and multiple), optional
/// \param[in] ctx the function execution context, optional
/// \return the element-wise rounded value
ARROW_EXPORT
Result<Datum> RoundToMultiple(
    const Datum& arg, RoundToMultipleOptions options = RoundToMultipleOptions::Defaults(),
    ExecContext* ctx = NULLPTR);

/// \brief Ceil a temporal value to a given frequency
///
/// If argument is null the result will be null.
///
/// \param[in] arg the temporal value to ceil
/// \param[in] options temporal rounding options, optional
/// \param[in] ctx the function execution context, optional
/// \return the element-wise rounded value
///
/// \since 7.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> CeilTemporal(
    const Datum& arg, RoundTemporalOptions options = RoundTemporalOptions::Defaults(),
    ExecContext* ctx = NULLPTR);

/// \brief Floor a temporal value to a given frequency
///
/// If argument is null the result will be null.
///
/// \param[in] arg the temporal value to floor
/// \param[in] options temporal rounding options, optional
/// \param[in] ctx the function execution context, optional
/// \return the element-wise rounded value
///
/// \since 7.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> FloorTemporal(
    const Datum& arg, RoundTemporalOptions options = RoundTemporalOptions::Defaults(),
    ExecContext* ctx = NULLPTR);

/// \brief Round a temporal value to a given frequency
///
/// If argument is null the result will be null.
///
/// \param[in] arg the temporal value to round
/// \param[in] options temporal rounding options, optional
/// \param[in] ctx the function execution context, optional
/// \return the element-wise rounded value
///
/// \since 7.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> RoundTemporal(
    const Datum& arg, RoundTemporalOptions options = RoundTemporalOptions::Defaults(),
    ExecContext* ctx = NULLPTR);

/// \brief Compare a numeric array with a scalar.
///
/// \param[in] left datum to compare, must be an Array
/// \param[in] right datum to compare, must be a Scalar of the same type than
///            left Datum.
/// \param[in] options compare options
/// \param[in] ctx the function execution context, optional
/// \return resulting datum
///
/// Note on floating point arrays, this uses ieee-754 compare semantics.
///
/// \since 1.0.0
/// \note API not yet finalized
ARROW_DEPRECATED("Deprecated in 5.0.0. Use each compare function directly")
ARROW_EXPORT
Result<Datum> Compare(const Datum& left, const Datum& right, CompareOptions options,
                      ExecContext* ctx = NULLPTR);

/// \brief Invert the values of a boolean datum
/// \param[in] value datum to invert
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 1.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> Invert(const Datum& value, ExecContext* ctx = NULLPTR);

/// \brief Element-wise AND of two boolean datums which always propagates nulls
/// (null and false is null).
///
/// \param[in] left left operand
/// \param[in] right right operand
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 1.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> And(const Datum& left, const Datum& right, ExecContext* ctx = NULLPTR);

/// \brief Element-wise AND of two boolean datums with a Kleene truth table
/// (null and false is false).
///
/// \param[in] left left operand
/// \param[in] right right operand
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 1.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> KleeneAnd(const Datum& left, const Datum& right,
                        ExecContext* ctx = NULLPTR);

/// \brief Element-wise OR of two boolean datums which always propagates nulls
/// (null and true is null).
///
/// \param[in] left left operand
/// \param[in] right right operand
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 1.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> Or(const Datum& left, const Datum& right, ExecContext* ctx = NULLPTR);

/// \brief Element-wise OR of two boolean datums with a Kleene truth table
/// (null or true is true).
///
/// \param[in] left left operand
/// \param[in] right right operand
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 1.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> KleeneOr(const Datum& left, const Datum& right, ExecContext* ctx = NULLPTR);

/// \brief Element-wise XOR of two boolean datums
/// \param[in] left left operand
/// \param[in] right right operand
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 1.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> Xor(const Datum& left, const Datum& right, ExecContext* ctx = NULLPTR);

/// \brief Element-wise AND NOT of two boolean datums which always propagates nulls
/// (null and not true is null).
///
/// \param[in] left left operand
/// \param[in] right right operand
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 3.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> AndNot(const Datum& left, const Datum& right, ExecContext* ctx = NULLPTR);

/// \brief Element-wise AND NOT of two boolean datums with a Kleene truth table
/// (false and not null is false, null and not true is false).
///
/// \param[in] left left operand
/// \param[in] right right operand
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 3.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> KleeneAndNot(const Datum& left, const Datum& right,
                           ExecContext* ctx = NULLPTR);

/// \brief IsIn returns true for each element of `values` that is contained in
/// `value_set`
///
/// Behaviour of nulls is governed by SetLookupOptions::skip_nulls.
///
/// \param[in] values array-like input to look up in value_set
/// \param[in] options SetLookupOptions
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 1.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> IsIn(const Datum& values, const SetLookupOptions& options,
                   ExecContext* ctx = NULLPTR);
ARROW_EXPORT
Result<Datum> IsIn(const Datum& values, const Datum& value_set,
                   ExecContext* ctx = NULLPTR);

/// \brief IndexIn examines each slot in the values against a value_set array.
/// If the value is not found in value_set, null will be output.
/// If found, the index of occurrence within value_set (ignoring duplicates)
/// will be output.
///
/// For example given values = [99, 42, 3, null] and
/// value_set = [3, 3, 99], the output will be = [2, null, 0, null]
///
/// Behaviour of nulls is governed by SetLookupOptions::skip_nulls.
///
/// \param[in] values array-like input
/// \param[in] options SetLookupOptions
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 1.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> IndexIn(const Datum& values, const SetLookupOptions& options,
                      ExecContext* ctx = NULLPTR);
ARROW_EXPORT
Result<Datum> IndexIn(const Datum& values, const Datum& value_set,
                      ExecContext* ctx = NULLPTR);

/// \brief IsValid returns true for each element of `values` that is not null,
/// false otherwise
///
/// \param[in] values input to examine for validity
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 1.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> IsValid(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief IsNull returns true for each element of `values` that is null,
/// false otherwise
///
/// \param[in] values input to examine for nullity
/// \param[in] options NullOptions
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 1.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> IsNull(const Datum& values, NullOptions options = NullOptions::Defaults(),
                     ExecContext* ctx = NULLPTR);

/// \brief IsNan returns true for each element of `values` that is NaN,
/// false otherwise
///
/// \param[in] values input to look for NaN
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 3.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> IsNan(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief IfElse returns elements chosen from `left` or `right`
/// depending on `cond`. `null` values in `cond` will be promoted to the result
///
/// \param[in] cond `Boolean` condition Scalar/ Array
/// \param[in] left Scalar/ Array
/// \param[in] right Scalar/ Array
/// \param[in] ctx the function execution context, optional
///
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> IfElse(const Datum& cond, const Datum& left, const Datum& right,
                     ExecContext* ctx = NULLPTR);

/// \brief CaseWhen behaves like a switch/case or if-else if-else statement: for
/// each row, select the first value for which the corresponding condition is
/// true, or (if given) select the 'else' value, else emit null. Note that a
/// null condition is the same as false.
///
/// \param[in] cond Conditions (Boolean)
/// \param[in] cases Values (any type), along with an optional 'else' value.
/// \param[in] ctx the function execution context, optional
///
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> CaseWhen(const Datum& cond, const std::vector<Datum>& cases,
                       ExecContext* ctx = NULLPTR);

/// \brief Year returns year for each element of `values`
///
/// \param[in] values input to extract year from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> Year(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief IsLeapYear returns if a year is a leap year for each element of `values`
///
/// \param[in] values input to extract leap year indicator from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 8.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> IsLeapYear(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief Month returns month for each element of `values`.
/// Month is encoded as January=1, December=12
///
/// \param[in] values input to extract month from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> Month(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief Day returns day number for each element of `values`
///
/// \param[in] values input to extract day from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> Day(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief YearMonthDay returns a struct containing the Year, Month and Day value for
/// each element of `values`.
///
/// \param[in] values input to extract (year, month, day) struct from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 7.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> YearMonthDay(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief DayOfWeek returns number of the day of the week value for each element of
/// `values`.
///
/// By default week starts on Monday denoted by 0 and ends on Sunday denoted
/// by 6. Start day of the week (Monday=1, Sunday=7) and numbering base (0 or 1) can be
/// set using DayOfWeekOptions
///
/// \param[in] values input to extract number of the day of the week from
/// \param[in] options for setting start of the week and day numbering
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT Result<Datum> DayOfWeek(const Datum& values,
                                     DayOfWeekOptions options = DayOfWeekOptions(),
                                     ExecContext* ctx = NULLPTR);

/// \brief DayOfYear returns number of day of the year for each element of `values`.
/// January 1st maps to day number 1, February 1st to 32, etc.
///
/// \param[in] values input to extract number of day of the year from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT Result<Datum> DayOfYear(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief ISOYear returns ISO year number for each element of `values`.
/// First week of an ISO year has the majority (4 or more) of its days in January.
///
/// \param[in] values input to extract ISO year from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> ISOYear(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief USYear returns US epidemiological year number for each element of `values`.
/// First week of US epidemiological year has the majority (4 or more) of it's
/// days in January. Last week of US epidemiological year has the year's last
/// Wednesday in it. US epidemiological week starts on Sunday.
///
/// \param[in] values input to extract US epidemiological year from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 8.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> USYear(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief ISOWeek returns ISO week of year number for each element of `values`.
/// First ISO week has the majority (4 or more) of its days in January.
/// ISO week starts on Monday. Year can have 52 or 53 weeks.
/// Week numbering can start with 1.
///
/// \param[in] values input to extract ISO week of year from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT Result<Datum> ISOWeek(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief USWeek returns US week of year number for each element of `values`.
/// First US week has the majority (4 or more) of its days in January.
/// US week starts on Sunday. Year can have 52 or 53 weeks.
/// Week numbering starts with 1.
///
/// \param[in] values input to extract US week of year from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 6.0.0
/// \note API not yet finalized
ARROW_EXPORT Result<Datum> USWeek(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief Week returns week of year number for each element of `values`.
/// First ISO week has the majority (4 or more) of its days in January.
/// Year can have 52 or 53 weeks. Week numbering can start with 0 or 1
/// depending on DayOfWeekOptions.count_from_zero.
///
/// \param[in] values input to extract week of year from
/// \param[in] options for setting numbering start
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 6.0.0
/// \note API not yet finalized
ARROW_EXPORT Result<Datum> Week(const Datum& values, WeekOptions options = WeekOptions(),
                                ExecContext* ctx = NULLPTR);

/// \brief ISOCalendar returns a (ISO year, ISO week, ISO day of week) struct for
/// each element of `values`.
/// ISO week starts on Monday denoted by 1 and ends on Sunday denoted by 7.
///
/// \param[in] values input to ISO calendar struct from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT Result<Datum> ISOCalendar(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief Quarter returns the quarter of year number for each element of `values`
/// First quarter maps to 1 and fourth quarter maps to 4.
///
/// \param[in] values input to extract quarter of year from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT Result<Datum> Quarter(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief Hour returns hour value for each element of `values`
///
/// \param[in] values input to extract hour from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> Hour(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief Minute returns minutes value for each element of `values`
///
/// \param[in] values input to extract minutes from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> Minute(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief Second returns seconds value for each element of `values`
///
/// \param[in] values input to extract seconds from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> Second(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief Millisecond returns number of milliseconds since the last full second
/// for each element of `values`
///
/// \param[in] values input to extract milliseconds from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> Millisecond(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief Microsecond returns number of microseconds since the last full millisecond
/// for each element of `values`
///
/// \param[in] values input to extract microseconds from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> Microsecond(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief Nanosecond returns number of nanoseconds since the last full millisecond
/// for each element of `values`
///
/// \param[in] values input to extract nanoseconds from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT
Result<Datum> Nanosecond(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief Subsecond returns the fraction of second elapsed since last full second
/// as a float for each element of `values`
///
/// \param[in] values input to extract subsecond from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 5.0.0
/// \note API not yet finalized
ARROW_EXPORT Result<Datum> Subsecond(const Datum& values, ExecContext* ctx = NULLPTR);

/// \brief Format timestamps according to a format string
///
/// Return formatted time strings according to the format string
/// `StrftimeOptions::format` and to the locale specifier `Strftime::locale`.
///
/// \param[in] values input timestamps
/// \param[in] options for setting format string and locale
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 6.0.0
/// \note API not yet finalized
ARROW_EXPORT Result<Datum> Strftime(const Datum& values, StrftimeOptions options,
                                    ExecContext* ctx = NULLPTR);

/// \brief Parse timestamps according to a format string
///
/// Return parsed timestamps according to the format string
/// `StrptimeOptions::format` at time resolution `Strftime::unit`. Parse errors are
/// raised depending on the `Strftime::error_is_null` setting.
///
/// \param[in] values input strings
/// \param[in] options for setting format string, unit and error_is_null
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 8.0.0
/// \note API not yet finalized
ARROW_EXPORT Result<Datum> Strptime(const Datum& values, StrptimeOptions options,
                                    ExecContext* ctx = NULLPTR);

/// \brief Converts timestamps from local timestamp without a timezone to a timestamp with
/// timezone, interpreting the local timestamp as being in the specified timezone for each
/// element of `values`
///
/// \param[in] values input to convert
/// \param[in] options for setting source timezone, exception and ambiguous timestamp
/// handling.
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 6.0.0
/// \note API not yet finalized
ARROW_EXPORT Result<Datum> AssumeTimezone(const Datum& values,
                                          AssumeTimezoneOptions options,
                                          ExecContext* ctx = NULLPTR);

/// \brief IsDaylightSavings extracts if currently observing daylight savings for each
/// element of `values`
///
/// \param[in] values input to extract daylight savings indicator from
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 8.0.0
/// \note API not yet finalized
ARROW_EXPORT Result<Datum> IsDaylightSavings(const Datum& values,
                                             ExecContext* ctx = NULLPTR);

/// \brief Years Between finds the number of years between two values
///
/// \param[in] left input treated as the start time
/// \param[in] right input treated as the end time
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 8.0.0
/// \note API not yet finalized
ARROW_EXPORT Result<Datum> YearsBetween(const Datum& left, const Datum& right,
                                        ExecContext* ctx = NULLPTR);

/// \brief Quarters Between finds the number of quarters between two values
///
/// \param[in] left input treated as the start time
/// \param[in] right input treated as the end time
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 8.0.0
/// \note API not yet finalized
ARROW_EXPORT Result<Datum> QuartersBetween(const Datum& left, const Datum& right,
                                           ExecContext* ctx = NULLPTR);

/// \brief Months Between finds the number of month between two values
///
/// \param[in] left input treated as the start time
/// \param[in] right input treated as the end time
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 8.0.0
/// \note API not yet finalized
ARROW_EXPORT Result<Datum> MonthsBetween(const Datum& left, const Datum& right,
                                         ExecContext* ctx = NULLPTR);

/// \brief Weeks Between finds the number of weeks between two values
///
/// \param[in] left input treated as the start time
/// \param[in] right input treated as the end time
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 8.0.0
/// \note API not yet finalized
ARROW_EXPORT Result<Datum> WeeksBetween(const Datum& left, const Datum& right,
                                        ExecContext* ctx = NULLPTR);

/// \brief Month Day Nano Between finds the number of months, days, and nonaseconds
/// between two values
///
/// \param[in] left input treated as the start time
/// \param[in] right input treated as the end time
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 8.0.0
/// \note API not yet finalized
ARROW_EXPORT Result<Datum> MonthDayNanoBetween(const Datum& left, const Datum& right,
                                               ExecContext* ctx = NULLPTR);

/// \brief DayTime Between finds the number of days and milliseconds between two values
///
/// \param[in] left input treated as the start time
/// \param[in] right input treated as the end time
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 8.0.0
/// \note API not yet finalized
ARROW_EXPORT Result<Datum> DayTimeBetween(const Datum& left, const Datum& right,
                                          ExecContext* ctx = NULLPTR);

/// \brief Days Between finds the number of days between two values
///
/// \param[in] left input treated as the start time
/// \param[in] right input treated as the end time
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 8.0.0
/// \note API not yet finalized
ARROW_EXPORT Result<Datum> DaysBetween(const Datum& left, const Datum& right,
                                       ExecContext* ctx = NULLPTR);

/// \brief Hours Between finds the number of hours between two values
///
/// \param[in] left input treated as the start time
/// \param[in] right input treated as the end time
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 8.0.0
/// \note API not yet finalized
ARROW_EXPORT Result<Datum> HoursBetween(const Datum& left, const Datum& right,
                                        ExecContext* ctx = NULLPTR);

/// \brief Minutes Between finds the number of minutes between two values
///
/// \param[in] left input treated as the start time
/// \param[in] right input treated as the end time
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 8.0.0
/// \note API not yet finalized
ARROW_EXPORT Result<Datum> MinutesBetween(const Datum& left, const Datum& right,
                                          ExecContext* ctx = NULLPTR);

/// \brief Seconds Between finds the number of hours between two values
///
/// \param[in] left input treated as the start time
/// \param[in] right input treated as the end time
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 8.0.0
/// \note API not yet finalized
ARROW_EXPORT Result<Datum> SecondsBetween(const Datum& left, const Datum& right,
                                          ExecContext* ctx = NULLPTR);

/// \brief Milliseconds Between finds the number of milliseconds between two values
///
/// \param[in] left input treated as the start time
/// \param[in] right input treated as the end time
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 8.0.0
/// \note API not yet finalized
ARROW_EXPORT Result<Datum> MillisecondsBetween(const Datum& left, const Datum& right,
                                               ExecContext* ctx = NULLPTR);

/// \brief Microseconds Between finds the number of microseconds between two values
///
/// \param[in] left input treated as the start time
/// \param[in] right input treated as the end time
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 8.0.0
/// \note API not yet finalized
ARROW_EXPORT Result<Datum> MicrosecondsBetween(const Datum& left, const Datum& right,
                                               ExecContext* ctx = NULLPTR);

/// \brief Nanoseconds Between finds the number of nanoseconds between two values
///
/// \param[in] left input treated as the start time
/// \param[in] right input treated as the end time
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 8.0.0
/// \note API not yet finalized
ARROW_EXPORT Result<Datum> NanosecondsBetween(const Datum& left, const Datum& right,
                                              ExecContext* ctx = NULLPTR);

/// \brief Finds either the FIRST, LAST, or ALL items with a key that matches the given
/// query key in a map.
///
/// Returns an array of items for FIRST and LAST, and an array of list of items for ALL.
///
/// \param[in] map to look in
/// \param[in] options to pass a query key and choose which matching keys to return
/// (FIRST, LAST or ALL)
/// \param[in] ctx the function execution context, optional
/// \return the resulting datum
///
/// \since 8.0.0
/// \note API not yet finalized
ARROW_EXPORT Result<Datum> MapLookup(const Datum& map, MapLookupOptions options,
                                     ExecContext* ctx = NULLPTR);
}  // namespace compute
}  // namespace arrow

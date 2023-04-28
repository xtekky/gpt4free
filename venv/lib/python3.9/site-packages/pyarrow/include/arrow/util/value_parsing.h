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

// This is a private header for string-to-number parsing utilities

#pragma once

#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>

#include "arrow/type.h"
#include "arrow/type_traits.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/config.h"
#include "arrow/util/macros.h"
#include "arrow/util/time.h"
#include "arrow/util/visibility.h"
#include "arrow/vendored/datetime.h"
#include "arrow/vendored/strptime.h"

namespace arrow {

/// \brief A virtual string to timestamp parser
class ARROW_EXPORT TimestampParser {
 public:
  virtual ~TimestampParser() = default;

  virtual bool operator()(const char* s, size_t length, TimeUnit::type out_unit,
                          int64_t* out,
                          bool* out_zone_offset_present = NULLPTR) const = 0;

  virtual const char* kind() const = 0;

  virtual const char* format() const;

  /// \brief Create a TimestampParser that recognizes strptime-like format strings
  static std::shared_ptr<TimestampParser> MakeStrptime(std::string format);

  /// \brief Create a TimestampParser that recognizes (locale-agnostic) ISO8601
  /// timestamps
  static std::shared_ptr<TimestampParser> MakeISO8601();
};

namespace internal {

/// \brief The entry point for conversion from strings.
///
/// Specializations of StringConverter for `ARROW_TYPE` must define:
/// - A default constructible member type `value_type` which will be yielded on a
///   successful parse.
/// - The static member function `Convert`, callable with signature
///   `(const ARROW_TYPE& t, const char* s, size_t length, value_type* out)`.
///   `Convert` returns truthy for successful parses and assigns the parsed values to
///   `*out`. Parameters required for parsing (for example a timestamp's TimeUnit)
///   are acquired from the type parameter `t`.
template <typename ARROW_TYPE, typename Enable = void>
struct StringConverter;

template <typename T>
struct is_parseable {
  template <typename U, typename = typename StringConverter<U>::value_type>
  static std::true_type Test(U*);

  template <typename U>
  static std::false_type Test(...);

  static constexpr bool value = decltype(Test<T>(NULLPTR))::value;
};

template <typename T, typename R = void>
using enable_if_parseable = enable_if_t<is_parseable<T>::value, R>;

template <>
struct StringConverter<BooleanType> {
  using value_type = bool;

  bool Convert(const BooleanType&, const char* s, size_t length, value_type* out) {
    if (length == 1) {
      // "0" or "1"?
      if (s[0] == '0') {
        *out = false;
        return true;
      }
      if (s[0] == '1') {
        *out = true;
        return true;
      }
      return false;
    }
    if (length == 4) {
      // "true"?
      *out = true;
      return ((s[0] == 't' || s[0] == 'T') && (s[1] == 'r' || s[1] == 'R') &&
              (s[2] == 'u' || s[2] == 'U') && (s[3] == 'e' || s[3] == 'E'));
    }
    if (length == 5) {
      // "false"?
      *out = false;
      return ((s[0] == 'f' || s[0] == 'F') && (s[1] == 'a' || s[1] == 'A') &&
              (s[2] == 'l' || s[2] == 'L') && (s[3] == 's' || s[3] == 'S') &&
              (s[4] == 'e' || s[4] == 'E'));
    }
    return false;
  }
};

// Ideas for faster float parsing:
// - http://rapidjson.org/md_doc_internals.html#ParsingDouble
// - https://github.com/google/double-conversion [used here]
// - https://github.com/achan001/dtoa-fast

ARROW_EXPORT
bool StringToFloat(const char* s, size_t length, char decimal_point, float* out);

ARROW_EXPORT
bool StringToFloat(const char* s, size_t length, char decimal_point, double* out);

template <>
struct StringConverter<FloatType> {
  using value_type = float;

  explicit StringConverter(char decimal_point = '.') : decimal_point(decimal_point) {}

  bool Convert(const FloatType&, const char* s, size_t length, value_type* out) {
    return ARROW_PREDICT_TRUE(StringToFloat(s, length, decimal_point, out));
  }

 private:
  const char decimal_point;
};

template <>
struct StringConverter<DoubleType> {
  using value_type = double;

  explicit StringConverter(char decimal_point = '.') : decimal_point(decimal_point) {}

  bool Convert(const DoubleType&, const char* s, size_t length, value_type* out) {
    return ARROW_PREDICT_TRUE(StringToFloat(s, length, decimal_point, out));
  }

 private:
  const char decimal_point;
};

// NOTE: HalfFloatType would require a half<->float conversion library

inline uint8_t ParseDecimalDigit(char c) { return static_cast<uint8_t>(c - '0'); }

#define PARSE_UNSIGNED_ITERATION(C_TYPE)          \
  if (length > 0) {                               \
    uint8_t digit = ParseDecimalDigit(*s++);      \
    result = static_cast<C_TYPE>(result * 10U);   \
    length--;                                     \
    if (ARROW_PREDICT_FALSE(digit > 9U)) {        \
      /* Non-digit */                             \
      return false;                               \
    }                                             \
    result = static_cast<C_TYPE>(result + digit); \
  } else {                                        \
    break;                                        \
  }

#define PARSE_UNSIGNED_ITERATION_LAST(C_TYPE)                                     \
  if (length > 0) {                                                               \
    if (ARROW_PREDICT_FALSE(result > std::numeric_limits<C_TYPE>::max() / 10U)) { \
      /* Overflow */                                                              \
      return false;                                                               \
    }                                                                             \
    uint8_t digit = ParseDecimalDigit(*s++);                                      \
    result = static_cast<C_TYPE>(result * 10U);                                   \
    C_TYPE new_result = static_cast<C_TYPE>(result + digit);                      \
    if (ARROW_PREDICT_FALSE(--length > 0)) {                                      \
      /* Too many digits */                                                       \
      return false;                                                               \
    }                                                                             \
    if (ARROW_PREDICT_FALSE(digit > 9U)) {                                        \
      /* Non-digit */                                                             \
      return false;                                                               \
    }                                                                             \
    if (ARROW_PREDICT_FALSE(new_result < result)) {                               \
      /* Overflow */                                                              \
      return false;                                                               \
    }                                                                             \
    result = new_result;                                                          \
  }

inline bool ParseUnsigned(const char* s, size_t length, uint8_t* out) {
  uint8_t result = 0;

  do {
    PARSE_UNSIGNED_ITERATION(uint8_t);
    PARSE_UNSIGNED_ITERATION(uint8_t);
    PARSE_UNSIGNED_ITERATION_LAST(uint8_t);
  } while (false);
  *out = result;
  return true;
}

inline bool ParseUnsigned(const char* s, size_t length, uint16_t* out) {
  uint16_t result = 0;
  do {
    PARSE_UNSIGNED_ITERATION(uint16_t);
    PARSE_UNSIGNED_ITERATION(uint16_t);
    PARSE_UNSIGNED_ITERATION(uint16_t);
    PARSE_UNSIGNED_ITERATION(uint16_t);
    PARSE_UNSIGNED_ITERATION_LAST(uint16_t);
  } while (false);
  *out = result;
  return true;
}

inline bool ParseUnsigned(const char* s, size_t length, uint32_t* out) {
  uint32_t result = 0;
  do {
    PARSE_UNSIGNED_ITERATION(uint32_t);
    PARSE_UNSIGNED_ITERATION(uint32_t);
    PARSE_UNSIGNED_ITERATION(uint32_t);
    PARSE_UNSIGNED_ITERATION(uint32_t);
    PARSE_UNSIGNED_ITERATION(uint32_t);

    PARSE_UNSIGNED_ITERATION(uint32_t);
    PARSE_UNSIGNED_ITERATION(uint32_t);
    PARSE_UNSIGNED_ITERATION(uint32_t);
    PARSE_UNSIGNED_ITERATION(uint32_t);

    PARSE_UNSIGNED_ITERATION_LAST(uint32_t);
  } while (false);
  *out = result;
  return true;
}

inline bool ParseUnsigned(const char* s, size_t length, uint64_t* out) {
  uint64_t result = 0;
  do {
    PARSE_UNSIGNED_ITERATION(uint64_t);
    PARSE_UNSIGNED_ITERATION(uint64_t);
    PARSE_UNSIGNED_ITERATION(uint64_t);
    PARSE_UNSIGNED_ITERATION(uint64_t);
    PARSE_UNSIGNED_ITERATION(uint64_t);

    PARSE_UNSIGNED_ITERATION(uint64_t);
    PARSE_UNSIGNED_ITERATION(uint64_t);
    PARSE_UNSIGNED_ITERATION(uint64_t);
    PARSE_UNSIGNED_ITERATION(uint64_t);
    PARSE_UNSIGNED_ITERATION(uint64_t);

    PARSE_UNSIGNED_ITERATION(uint64_t);
    PARSE_UNSIGNED_ITERATION(uint64_t);
    PARSE_UNSIGNED_ITERATION(uint64_t);
    PARSE_UNSIGNED_ITERATION(uint64_t);
    PARSE_UNSIGNED_ITERATION(uint64_t);

    PARSE_UNSIGNED_ITERATION(uint64_t);
    PARSE_UNSIGNED_ITERATION(uint64_t);
    PARSE_UNSIGNED_ITERATION(uint64_t);
    PARSE_UNSIGNED_ITERATION(uint64_t);

    PARSE_UNSIGNED_ITERATION_LAST(uint64_t);
  } while (false);
  *out = result;
  return true;
}

#undef PARSE_UNSIGNED_ITERATION
#undef PARSE_UNSIGNED_ITERATION_LAST

template <typename T>
bool ParseHex(const char* s, size_t length, T* out) {
  // lets make sure that the length of the string is not too big
  if (!ARROW_PREDICT_TRUE(sizeof(T) * 2 >= length && length > 0)) {
    return false;
  }
  T result = 0;
  for (size_t i = 0; i < length; i++) {
    result = static_cast<T>(result << 4);
    if (s[i] >= '0' && s[i] <= '9') {
      result = static_cast<T>(result | (s[i] - '0'));
    } else if (s[i] >= 'A' && s[i] <= 'F') {
      result = static_cast<T>(result | (s[i] - 'A' + 10));
    } else if (s[i] >= 'a' && s[i] <= 'f') {
      result = static_cast<T>(result | (s[i] - 'a' + 10));
    } else {
      /* Non-digit */
      return false;
    }
  }
  *out = result;
  return true;
}

template <class ARROW_TYPE>
struct StringToUnsignedIntConverterMixin {
  using value_type = typename ARROW_TYPE::c_type;

  bool Convert(const ARROW_TYPE&, const char* s, size_t length, value_type* out) {
    if (ARROW_PREDICT_FALSE(length == 0)) {
      return false;
    }
    // If it starts with 0x then its hex
    if (length > 2 && s[0] == '0' && ((s[1] == 'x') || (s[1] == 'X'))) {
      length -= 2;
      s += 2;

      return ARROW_PREDICT_TRUE(ParseHex(s, length, out));
    }
    // Skip leading zeros
    while (length > 0 && *s == '0') {
      length--;
      s++;
    }
    return ParseUnsigned(s, length, out);
  }
};

template <>
struct StringConverter<UInt8Type> : public StringToUnsignedIntConverterMixin<UInt8Type> {
  using StringToUnsignedIntConverterMixin<UInt8Type>::StringToUnsignedIntConverterMixin;
};

template <>
struct StringConverter<UInt16Type>
    : public StringToUnsignedIntConverterMixin<UInt16Type> {
  using StringToUnsignedIntConverterMixin<UInt16Type>::StringToUnsignedIntConverterMixin;
};

template <>
struct StringConverter<UInt32Type>
    : public StringToUnsignedIntConverterMixin<UInt32Type> {
  using StringToUnsignedIntConverterMixin<UInt32Type>::StringToUnsignedIntConverterMixin;
};

template <>
struct StringConverter<UInt64Type>
    : public StringToUnsignedIntConverterMixin<UInt64Type> {
  using StringToUnsignedIntConverterMixin<UInt64Type>::StringToUnsignedIntConverterMixin;
};

template <class ARROW_TYPE>
struct StringToSignedIntConverterMixin {
  using value_type = typename ARROW_TYPE::c_type;
  using unsigned_type = typename std::make_unsigned<value_type>::type;

  bool Convert(const ARROW_TYPE&, const char* s, size_t length, value_type* out) {
    static constexpr auto max_positive =
        static_cast<unsigned_type>(std::numeric_limits<value_type>::max());
    // Assuming two's complement
    static constexpr unsigned_type max_negative = max_positive + 1;
    bool negative = false;
    unsigned_type unsigned_value = 0;

    if (ARROW_PREDICT_FALSE(length == 0)) {
      return false;
    }
    // If it starts with 0x then its hex
    if (length > 2 && s[0] == '0' && ((s[1] == 'x') || (s[1] == 'X'))) {
      length -= 2;
      s += 2;

      if (!ARROW_PREDICT_TRUE(ParseHex(s, length, &unsigned_value))) {
        return false;
      }
      *out = static_cast<value_type>(unsigned_value);
      return true;
    }

    if (*s == '-') {
      negative = true;
      s++;
      if (--length == 0) {
        return false;
      }
    }
    // Skip leading zeros
    while (length > 0 && *s == '0') {
      length--;
      s++;
    }
    if (!ARROW_PREDICT_TRUE(ParseUnsigned(s, length, &unsigned_value))) {
      return false;
    }
    if (negative) {
      if (ARROW_PREDICT_FALSE(unsigned_value > max_negative)) {
        return false;
      }
      // To avoid both compiler warnings (with unsigned negation)
      // and undefined behaviour (with signed negation overflow),
      // use the expanded formula for 2's complement negation.
      *out = static_cast<value_type>(~unsigned_value + 1);
    } else {
      if (ARROW_PREDICT_FALSE(unsigned_value > max_positive)) {
        return false;
      }
      *out = static_cast<value_type>(unsigned_value);
    }
    return true;
  }
};

template <>
struct StringConverter<Int8Type> : public StringToSignedIntConverterMixin<Int8Type> {
  using StringToSignedIntConverterMixin<Int8Type>::StringToSignedIntConverterMixin;
};

template <>
struct StringConverter<Int16Type> : public StringToSignedIntConverterMixin<Int16Type> {
  using StringToSignedIntConverterMixin<Int16Type>::StringToSignedIntConverterMixin;
};

template <>
struct StringConverter<Int32Type> : public StringToSignedIntConverterMixin<Int32Type> {
  using StringToSignedIntConverterMixin<Int32Type>::StringToSignedIntConverterMixin;
};

template <>
struct StringConverter<Int64Type> : public StringToSignedIntConverterMixin<Int64Type> {
  using StringToSignedIntConverterMixin<Int64Type>::StringToSignedIntConverterMixin;
};

namespace detail {

// Inline-able ISO-8601 parser

using ts_type = TimestampType::c_type;

template <typename Duration>
static inline bool ParseYYYY_MM_DD(const char* s, Duration* since_epoch) {
  uint16_t year = 0;
  uint8_t month = 0;
  uint8_t day = 0;
  if (ARROW_PREDICT_FALSE(s[4] != '-') || ARROW_PREDICT_FALSE(s[7] != '-')) {
    return false;
  }
  if (ARROW_PREDICT_FALSE(!ParseUnsigned(s + 0, 4, &year))) {
    return false;
  }
  if (ARROW_PREDICT_FALSE(!ParseUnsigned(s + 5, 2, &month))) {
    return false;
  }
  if (ARROW_PREDICT_FALSE(!ParseUnsigned(s + 8, 2, &day))) {
    return false;
  }
  arrow_vendored::date::year_month_day ymd{arrow_vendored::date::year{year},
                                           arrow_vendored::date::month{month},
                                           arrow_vendored::date::day{day}};
  if (ARROW_PREDICT_FALSE(!ymd.ok())) return false;

  *since_epoch = std::chrono::duration_cast<Duration>(
      arrow_vendored::date::sys_days{ymd}.time_since_epoch());
  return true;
}

template <typename Duration>
static inline bool ParseHH(const char* s, Duration* out) {
  uint8_t hours = 0;
  if (ARROW_PREDICT_FALSE(!ParseUnsigned(s + 0, 2, &hours))) {
    return false;
  }
  if (ARROW_PREDICT_FALSE(hours >= 24)) {
    return false;
  }
  *out = std::chrono::duration_cast<Duration>(std::chrono::hours(hours));
  return true;
}

template <typename Duration>
static inline bool ParseHH_MM(const char* s, Duration* out) {
  uint8_t hours = 0;
  uint8_t minutes = 0;
  if (ARROW_PREDICT_FALSE(s[2] != ':')) {
    return false;
  }
  if (ARROW_PREDICT_FALSE(!ParseUnsigned(s + 0, 2, &hours))) {
    return false;
  }
  if (ARROW_PREDICT_FALSE(!ParseUnsigned(s + 3, 2, &minutes))) {
    return false;
  }
  if (ARROW_PREDICT_FALSE(hours >= 24)) {
    return false;
  }
  if (ARROW_PREDICT_FALSE(minutes >= 60)) {
    return false;
  }
  *out = std::chrono::duration_cast<Duration>(std::chrono::hours(hours) +
                                              std::chrono::minutes(minutes));
  return true;
}

template <typename Duration>
static inline bool ParseHHMM(const char* s, Duration* out) {
  uint8_t hours = 0;
  uint8_t minutes = 0;
  if (ARROW_PREDICT_FALSE(!ParseUnsigned(s + 0, 2, &hours))) {
    return false;
  }
  if (ARROW_PREDICT_FALSE(!ParseUnsigned(s + 2, 2, &minutes))) {
    return false;
  }
  if (ARROW_PREDICT_FALSE(hours >= 24)) {
    return false;
  }
  if (ARROW_PREDICT_FALSE(minutes >= 60)) {
    return false;
  }
  *out = std::chrono::duration_cast<Duration>(std::chrono::hours(hours) +
                                              std::chrono::minutes(minutes));
  return true;
}

template <typename Duration>
static inline bool ParseHH_MM_SS(const char* s, Duration* out) {
  uint8_t hours = 0;
  uint8_t minutes = 0;
  uint8_t seconds = 0;
  if (ARROW_PREDICT_FALSE(s[2] != ':') || ARROW_PREDICT_FALSE(s[5] != ':')) {
    return false;
  }
  if (ARROW_PREDICT_FALSE(!ParseUnsigned(s + 0, 2, &hours))) {
    return false;
  }
  if (ARROW_PREDICT_FALSE(!ParseUnsigned(s + 3, 2, &minutes))) {
    return false;
  }
  if (ARROW_PREDICT_FALSE(!ParseUnsigned(s + 6, 2, &seconds))) {
    return false;
  }
  if (ARROW_PREDICT_FALSE(hours >= 24)) {
    return false;
  }
  if (ARROW_PREDICT_FALSE(minutes >= 60)) {
    return false;
  }
  if (ARROW_PREDICT_FALSE(seconds >= 60)) {
    return false;
  }
  *out = std::chrono::duration_cast<Duration>(std::chrono::hours(hours) +
                                              std::chrono::minutes(minutes) +
                                              std::chrono::seconds(seconds));
  return true;
}

static inline bool ParseSubSeconds(const char* s, size_t length, TimeUnit::type unit,
                                   uint32_t* out) {
  // The decimal point has been peeled off at this point

  // Fail if number of decimal places provided exceeds what the unit can hold.
  // Calculate how many trailing decimal places are omitted for the unit
  // e.g. if 4 decimal places are provided and unit is MICRO, 2 are missing
  size_t omitted = 0;
  switch (unit) {
    case TimeUnit::MILLI:
      if (ARROW_PREDICT_FALSE(length > 3)) {
        return false;
      }
      if (length < 3) {
        omitted = 3 - length;
      }
      break;
    case TimeUnit::MICRO:
      if (ARROW_PREDICT_FALSE(length > 6)) {
        return false;
      }
      if (length < 6) {
        omitted = 6 - length;
      }
      break;
    case TimeUnit::NANO:
      if (ARROW_PREDICT_FALSE(length > 9)) {
        return false;
      }
      if (length < 9) {
        omitted = 9 - length;
      }
      break;
    default:
      return false;
  }

  if (ARROW_PREDICT_TRUE(omitted == 0)) {
    return ParseUnsigned(s, length, out);
  } else {
    uint32_t subseconds = 0;
    bool success = ParseUnsigned(s, length, &subseconds);
    if (ARROW_PREDICT_TRUE(success)) {
      switch (omitted) {
        case 1:
          *out = subseconds * 10;
          break;
        case 2:
          *out = subseconds * 100;
          break;
        case 3:
          *out = subseconds * 1000;
          break;
        case 4:
          *out = subseconds * 10000;
          break;
        case 5:
          *out = subseconds * 100000;
          break;
        case 6:
          *out = subseconds * 1000000;
          break;
        case 7:
          *out = subseconds * 10000000;
          break;
        case 8:
          *out = subseconds * 100000000;
          break;
        default:
          // Impossible case
          break;
      }
      return true;
    } else {
      return false;
    }
  }
}

}  // namespace detail

static inline bool ParseTimestampISO8601(const char* s, size_t length,
                                         TimeUnit::type unit, TimestampType::c_type* out,
                                         bool* out_zone_offset_present = NULLPTR) {
  using seconds_type = std::chrono::duration<TimestampType::c_type>;

  // We allow the following zone offset formats:
  // - (none)
  // - Z
  // - [+-]HH(:?MM)?
  //
  // We allow the following formats for all units:
  // - "YYYY-MM-DD"
  // - "YYYY-MM-DD[ T]hhZ?"
  // - "YYYY-MM-DD[ T]hh:mmZ?"
  // - "YYYY-MM-DD[ T]hh:mm:ssZ?"
  //
  // We allow the following formats for unit == MILLI, MICRO, or NANO:
  // - "YYYY-MM-DD[ T]hh:mm:ss.s{1,3}Z?"
  //
  // We allow the following formats for unit == MICRO, or NANO:
  // - "YYYY-MM-DD[ T]hh:mm:ss.s{4,6}Z?"
  //
  // We allow the following formats for unit == NANO:
  // - "YYYY-MM-DD[ T]hh:mm:ss.s{7,9}Z?"
  //
  // UTC is always assumed, and the DataType's timezone is ignored.
  //

  if (ARROW_PREDICT_FALSE(length < 10)) return false;

  seconds_type seconds_since_epoch;
  if (ARROW_PREDICT_FALSE(!detail::ParseYYYY_MM_DD(s, &seconds_since_epoch))) {
    return false;
  }

  if (length == 10) {
    *out = util::CastSecondsToUnit(unit, seconds_since_epoch.count());
    return true;
  }

  if (ARROW_PREDICT_FALSE(s[10] != ' ') && ARROW_PREDICT_FALSE(s[10] != 'T')) {
    return false;
  }

  if (out_zone_offset_present) {
    *out_zone_offset_present = false;
  }

  seconds_type zone_offset(0);
  if (s[length - 1] == 'Z') {
    --length;
    if (out_zone_offset_present) *out_zone_offset_present = true;
  } else if (s[length - 3] == '+' || s[length - 3] == '-') {
    // [+-]HH
    length -= 3;
    if (ARROW_PREDICT_FALSE(!detail::ParseHH(s + length + 1, &zone_offset))) {
      return false;
    }
    if (s[length] == '+') zone_offset *= -1;
    if (out_zone_offset_present) *out_zone_offset_present = true;
  } else if (s[length - 5] == '+' || s[length - 5] == '-') {
    // [+-]HHMM
    length -= 5;
    if (ARROW_PREDICT_FALSE(!detail::ParseHHMM(s + length + 1, &zone_offset))) {
      return false;
    }
    if (s[length] == '+') zone_offset *= -1;
    if (out_zone_offset_present) *out_zone_offset_present = true;
  } else if ((s[length - 6] == '+' || s[length - 6] == '-') && (s[length - 3] == ':')) {
    // [+-]HH:MM
    length -= 6;
    if (ARROW_PREDICT_FALSE(!detail::ParseHH_MM(s + length + 1, &zone_offset))) {
      return false;
    }
    if (s[length] == '+') zone_offset *= -1;
    if (out_zone_offset_present) *out_zone_offset_present = true;
  }

  seconds_type seconds_since_midnight;
  switch (length) {
    case 13:  // YYYY-MM-DD[ T]hh
      if (ARROW_PREDICT_FALSE(!detail::ParseHH(s + 11, &seconds_since_midnight))) {
        return false;
      }
      break;
    case 16:  // YYYY-MM-DD[ T]hh:mm
      if (ARROW_PREDICT_FALSE(!detail::ParseHH_MM(s + 11, &seconds_since_midnight))) {
        return false;
      }
      break;
    case 19:  // YYYY-MM-DD[ T]hh:mm:ss
    case 21:  // YYYY-MM-DD[ T]hh:mm:ss.s
    case 22:  // YYYY-MM-DD[ T]hh:mm:ss.ss
    case 23:  // YYYY-MM-DD[ T]hh:mm:ss.sss
    case 24:  // YYYY-MM-DD[ T]hh:mm:ss.ssss
    case 25:  // YYYY-MM-DD[ T]hh:mm:ss.sssss
    case 26:  // YYYY-MM-DD[ T]hh:mm:ss.ssssss
    case 27:  // YYYY-MM-DD[ T]hh:mm:ss.sssssss
    case 28:  // YYYY-MM-DD[ T]hh:mm:ss.ssssssss
    case 29:  // YYYY-MM-DD[ T]hh:mm:ss.sssssssss
      if (ARROW_PREDICT_FALSE(!detail::ParseHH_MM_SS(s + 11, &seconds_since_midnight))) {
        return false;
      }
      break;
    default:
      return false;
  }

  seconds_since_epoch += seconds_since_midnight;
  seconds_since_epoch += zone_offset;

  if (length <= 19) {
    *out = util::CastSecondsToUnit(unit, seconds_since_epoch.count());
    return true;
  }

  if (ARROW_PREDICT_FALSE(s[19] != '.')) {
    return false;
  }

  uint32_t subseconds = 0;
  if (ARROW_PREDICT_FALSE(
          !detail::ParseSubSeconds(s + 20, length - 20, unit, &subseconds))) {
    return false;
  }

  *out = util::CastSecondsToUnit(unit, seconds_since_epoch.count()) + subseconds;
  return true;
}

#if defined(_WIN32) || defined(ARROW_WITH_MUSL)
static constexpr bool kStrptimeSupportsZone = false;
#else
static constexpr bool kStrptimeSupportsZone = true;
#endif

/// \brief Returns time since the UNIX epoch in the requested unit
static inline bool ParseTimestampStrptime(const char* buf, size_t length,
                                          const char* format, bool ignore_time_in_day,
                                          bool allow_trailing_chars, TimeUnit::type unit,
                                          int64_t* out) {
  // NOTE: strptime() is more than 10x faster than arrow_vendored::date::parse().
  // The buffer may not be nul-terminated
  std::string clean_copy(buf, length);
  struct tm result;
  memset(&result, 0, sizeof(struct tm));
#ifdef _WIN32
  char* ret = arrow_strptime(clean_copy.c_str(), format, &result);
#else
  char* ret = strptime(clean_copy.c_str(), format, &result);
#endif
  if (ret == NULLPTR) {
    return false;
  }
  if (!allow_trailing_chars && static_cast<size_t>(ret - clean_copy.c_str()) != length) {
    return false;
  }
  // ignore the time part
  arrow_vendored::date::sys_seconds secs =
      arrow_vendored::date::sys_days(arrow_vendored::date::year(result.tm_year + 1900) /
                                     (result.tm_mon + 1) / result.tm_mday);
  if (!ignore_time_in_day) {
    secs += (std::chrono::hours(result.tm_hour) + std::chrono::minutes(result.tm_min) +
             std::chrono::seconds(result.tm_sec));
#ifndef _WIN32
    secs -= std::chrono::seconds(result.tm_gmtoff);
#endif
  }
  *out = util::CastSecondsToUnit(unit, secs.time_since_epoch().count());
  return true;
}

template <>
struct StringConverter<TimestampType> {
  using value_type = int64_t;

  bool Convert(const TimestampType& type, const char* s, size_t length, value_type* out) {
    return ParseTimestampISO8601(s, length, type.unit(), out);
  }
};

template <>
struct StringConverter<DurationType>
    : public StringToSignedIntConverterMixin<DurationType> {
  using StringToSignedIntConverterMixin<DurationType>::StringToSignedIntConverterMixin;
};

template <typename DATE_TYPE>
struct StringConverter<DATE_TYPE, enable_if_date<DATE_TYPE>> {
  using value_type = typename DATE_TYPE::c_type;

  using duration_type =
      typename std::conditional<std::is_same<DATE_TYPE, Date32Type>::value,
                                arrow_vendored::date::days,
                                std::chrono::milliseconds>::type;

  bool Convert(const DATE_TYPE& type, const char* s, size_t length, value_type* out) {
    if (ARROW_PREDICT_FALSE(length != 10)) {
      return false;
    }

    duration_type since_epoch;
    if (ARROW_PREDICT_FALSE(!detail::ParseYYYY_MM_DD(s, &since_epoch))) {
      return false;
    }

    *out = static_cast<value_type>(since_epoch.count());
    return true;
  }
};

template <typename TIME_TYPE>
struct StringConverter<TIME_TYPE, enable_if_time<TIME_TYPE>> {
  using value_type = typename TIME_TYPE::c_type;

  // We allow the following formats for all units:
  // - "hh:mm"
  // - "hh:mm:ss"
  //
  // We allow the following formats for unit == MILLI, MICRO, or NANO:
  // - "hh:mm:ss.s{1,3}"
  //
  // We allow the following formats for unit == MICRO, or NANO:
  // - "hh:mm:ss.s{4,6}"
  //
  // We allow the following formats for unit == NANO:
  // - "hh:mm:ss.s{7,9}"

  bool Convert(const TIME_TYPE& type, const char* s, size_t length, value_type* out) {
    const auto unit = type.unit();
    std::chrono::seconds since_midnight;

    if (length == 5) {
      if (ARROW_PREDICT_FALSE(!detail::ParseHH_MM(s, &since_midnight))) {
        return false;
      }
      *out =
          static_cast<value_type>(util::CastSecondsToUnit(unit, since_midnight.count()));
      return true;
    }

    if (ARROW_PREDICT_FALSE(length < 8)) {
      return false;
    }
    if (ARROW_PREDICT_FALSE(!detail::ParseHH_MM_SS(s, &since_midnight))) {
      return false;
    }

    *out = static_cast<value_type>(util::CastSecondsToUnit(unit, since_midnight.count()));

    if (length == 8) {
      return true;
    }

    if (ARROW_PREDICT_FALSE(s[8] != '.')) {
      return false;
    }

    uint32_t subseconds_count = 0;
    if (ARROW_PREDICT_FALSE(
            !detail::ParseSubSeconds(s + 9, length - 9, unit, &subseconds_count))) {
      return false;
    }

    *out += subseconds_count;
    return true;
  }
};

/// \brief Convenience wrappers around internal::StringConverter.
template <typename T>
bool ParseValue(const T& type, const char* s, size_t length,
                typename StringConverter<T>::value_type* out) {
  return StringConverter<T>{}.Convert(type, s, length, out);
}

template <typename T>
enable_if_parameter_free<T, bool> ParseValue(
    const char* s, size_t length, typename StringConverter<T>::value_type* out) {
  static T type;
  return StringConverter<T>{}.Convert(type, s, length, out);
}

}  // namespace internal
}  // namespace arrow

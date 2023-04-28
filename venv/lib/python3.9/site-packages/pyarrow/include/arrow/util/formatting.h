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

// This is a private header for number-to-string formatting utilities

#pragma once

#include <array>
#include <cassert>
#include <chrono>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "arrow/status.h"
#include "arrow/type.h"
#include "arrow/type_traits.h"
#include "arrow/util/double_conversion.h"
#include "arrow/util/macros.h"
#include "arrow/util/string.h"
#include "arrow/util/time.h"
#include "arrow/util/visibility.h"
#include "arrow/vendored/datetime.h"

namespace arrow {
namespace internal {

/// \brief The entry point for conversion to strings.
template <typename ARROW_TYPE, typename Enable = void>
class StringFormatter;

template <typename T>
struct is_formattable {
  template <typename U, typename = typename StringFormatter<U>::value_type>
  static std::true_type Test(U*);

  template <typename U>
  static std::false_type Test(...);

  static constexpr bool value = decltype(Test<T>(NULLPTR))::value;
};

template <typename T, typename R = void>
using enable_if_formattable = enable_if_t<is_formattable<T>::value, R>;

template <typename Appender>
using Return = decltype(std::declval<Appender>()(std::string_view{}));

/////////////////////////////////////////////////////////////////////////
// Boolean formatting

template <>
class StringFormatter<BooleanType> {
 public:
  explicit StringFormatter(const DataType* = NULLPTR) {}

  using value_type = bool;

  template <typename Appender>
  Return<Appender> operator()(bool value, Appender&& append) {
    if (value) {
      const char string[] = "true";
      return append(std::string_view(string));
    } else {
      const char string[] = "false";
      return append(std::string_view(string));
    }
  }
};

/////////////////////////////////////////////////////////////////////////
// Decimals formatting

template <typename ARROW_TYPE>
class DecimalToStringFormatterMixin {
 public:
  explicit DecimalToStringFormatterMixin(const DataType* type)
      : scale_(static_cast<const ARROW_TYPE*>(type)->scale()) {}

  using value_type = typename TypeTraits<ARROW_TYPE>::CType;

  template <typename Appender>
  Return<Appender> operator()(const value_type& value, Appender&& append) {
    return append(value.ToString(scale_));
  }

 private:
  int32_t scale_;
};

template <>
class StringFormatter<Decimal128Type>
    : public DecimalToStringFormatterMixin<Decimal128Type> {
  using DecimalToStringFormatterMixin::DecimalToStringFormatterMixin;
};

template <>
class StringFormatter<Decimal256Type>
    : public DecimalToStringFormatterMixin<Decimal256Type> {
  using DecimalToStringFormatterMixin::DecimalToStringFormatterMixin;
};

/////////////////////////////////////////////////////////////////////////
// Integer formatting

namespace detail {

// A 2x100 direct table mapping integers in [0..99] to their decimal representations.
ARROW_EXPORT extern const char digit_pairs[];

// Based on fmtlib's format_int class:
// Write digits from right to left into a stack allocated buffer
inline void FormatOneChar(char c, char** cursor) { *--*cursor = c; }

template <typename Int>
void FormatOneDigit(Int value, char** cursor) {
  assert(value >= 0 && value <= 9);
  FormatOneChar(static_cast<char>('0' + value), cursor);
}

template <typename Int>
void FormatTwoDigits(Int value, char** cursor) {
  assert(value >= 0 && value <= 99);
  auto digit_pair = &digit_pairs[value * 2];
  FormatOneChar(digit_pair[1], cursor);
  FormatOneChar(digit_pair[0], cursor);
}

template <typename Int>
void FormatAllDigits(Int value, char** cursor) {
  assert(value >= 0);
  while (value >= 100) {
    FormatTwoDigits(value % 100, cursor);
    value /= 100;
  }

  if (value >= 10) {
    FormatTwoDigits(value, cursor);
  } else {
    FormatOneDigit(value, cursor);
  }
}

template <typename Int>
void FormatAllDigitsLeftPadded(Int value, size_t pad, char pad_char, char** cursor) {
  auto end = *cursor - pad;
  FormatAllDigits(value, cursor);
  while (*cursor > end) {
    FormatOneChar(pad_char, cursor);
  }
}

template <size_t BUFFER_SIZE>
std::string_view ViewDigitBuffer(const std::array<char, BUFFER_SIZE>& buffer,
                                 char* cursor) {
  auto buffer_end = buffer.data() + BUFFER_SIZE;
  return {cursor, static_cast<size_t>(buffer_end - cursor)};
}

template <typename Int, typename UInt = typename std::make_unsigned<Int>::type>
constexpr UInt Abs(Int value) {
  return value < 0 ? ~static_cast<UInt>(value) + 1 : static_cast<UInt>(value);
}

template <typename Int>
constexpr size_t Digits10(Int value) {
  return value <= 9 ? 1 : Digits10(value / 10) + 1;
}

}  // namespace detail

template <typename ARROW_TYPE>
class IntToStringFormatterMixin {
 public:
  explicit IntToStringFormatterMixin(const DataType* = NULLPTR) {}

  using value_type = typename ARROW_TYPE::c_type;

  template <typename Appender>
  Return<Appender> operator()(value_type value, Appender&& append) {
    constexpr size_t buffer_size =
        detail::Digits10(std::numeric_limits<value_type>::max()) + 1;

    std::array<char, buffer_size> buffer;
    char* cursor = buffer.data() + buffer_size;
    detail::FormatAllDigits(detail::Abs(value), &cursor);
    if (value < 0) {
      detail::FormatOneChar('-', &cursor);
    }
    return append(detail::ViewDigitBuffer(buffer, cursor));
  }
};

template <>
class StringFormatter<Int8Type> : public IntToStringFormatterMixin<Int8Type> {
  using IntToStringFormatterMixin::IntToStringFormatterMixin;
};

template <>
class StringFormatter<Int16Type> : public IntToStringFormatterMixin<Int16Type> {
  using IntToStringFormatterMixin::IntToStringFormatterMixin;
};

template <>
class StringFormatter<Int32Type> : public IntToStringFormatterMixin<Int32Type> {
  using IntToStringFormatterMixin::IntToStringFormatterMixin;
};

template <>
class StringFormatter<Int64Type> : public IntToStringFormatterMixin<Int64Type> {
  using IntToStringFormatterMixin::IntToStringFormatterMixin;
};

template <>
class StringFormatter<UInt8Type> : public IntToStringFormatterMixin<UInt8Type> {
  using IntToStringFormatterMixin::IntToStringFormatterMixin;
};

template <>
class StringFormatter<UInt16Type> : public IntToStringFormatterMixin<UInt16Type> {
  using IntToStringFormatterMixin::IntToStringFormatterMixin;
};

template <>
class StringFormatter<UInt32Type> : public IntToStringFormatterMixin<UInt32Type> {
  using IntToStringFormatterMixin::IntToStringFormatterMixin;
};

template <>
class StringFormatter<UInt64Type> : public IntToStringFormatterMixin<UInt64Type> {
  using IntToStringFormatterMixin::IntToStringFormatterMixin;
};

/////////////////////////////////////////////////////////////////////////
// Floating-point formatting

class ARROW_EXPORT FloatToStringFormatter {
 public:
  FloatToStringFormatter();
  FloatToStringFormatter(int flags, const char* inf_symbol, const char* nan_symbol,
                         char exp_character, int decimal_in_shortest_low,
                         int decimal_in_shortest_high,
                         int max_leading_padding_zeroes_in_precision_mode,
                         int max_trailing_padding_zeroes_in_precision_mode);
  ~FloatToStringFormatter();

  // Returns the number of characters written
  int FormatFloat(float v, char* out_buffer, int out_size);
  int FormatFloat(double v, char* out_buffer, int out_size);

 protected:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

template <typename ARROW_TYPE>
class FloatToStringFormatterMixin : public FloatToStringFormatter {
 public:
  using value_type = typename ARROW_TYPE::c_type;

  static constexpr int buffer_size = 50;

  explicit FloatToStringFormatterMixin(const DataType* = NULLPTR) {}

  FloatToStringFormatterMixin(int flags, const char* inf_symbol, const char* nan_symbol,
                              char exp_character, int decimal_in_shortest_low,
                              int decimal_in_shortest_high,
                              int max_leading_padding_zeroes_in_precision_mode,
                              int max_trailing_padding_zeroes_in_precision_mode)
      : FloatToStringFormatter(flags, inf_symbol, nan_symbol, exp_character,
                               decimal_in_shortest_low, decimal_in_shortest_high,
                               max_leading_padding_zeroes_in_precision_mode,
                               max_trailing_padding_zeroes_in_precision_mode) {}

  template <typename Appender>
  Return<Appender> operator()(value_type value, Appender&& append) {
    char buffer[buffer_size];
    int size = FormatFloat(value, buffer, buffer_size);
    return append(std::string_view(buffer, size));
  }
};

template <>
class StringFormatter<FloatType> : public FloatToStringFormatterMixin<FloatType> {
 public:
  using FloatToStringFormatterMixin::FloatToStringFormatterMixin;
};

template <>
class StringFormatter<DoubleType> : public FloatToStringFormatterMixin<DoubleType> {
 public:
  using FloatToStringFormatterMixin::FloatToStringFormatterMixin;
};

/////////////////////////////////////////////////////////////////////////
// Temporal formatting

namespace detail {

constexpr size_t BufferSizeYYYY_MM_DD() {
  return 1 + detail::Digits10(99999) + 1 + detail::Digits10(12) + 1 +
         detail::Digits10(31);
}

inline void FormatYYYY_MM_DD(arrow_vendored::date::year_month_day ymd, char** cursor) {
  FormatTwoDigits(static_cast<unsigned>(ymd.day()), cursor);
  FormatOneChar('-', cursor);
  FormatTwoDigits(static_cast<unsigned>(ymd.month()), cursor);
  FormatOneChar('-', cursor);
  auto year = static_cast<int>(ymd.year());
  const auto is_neg_year = year < 0;
  year = std::abs(year);
  assert(year <= 99999);
  FormatTwoDigits(year % 100, cursor);
  year /= 100;
  FormatTwoDigits(year % 100, cursor);
  if (year >= 100) {
    FormatOneDigit(year / 100, cursor);
  }
  if (is_neg_year) {
    FormatOneChar('-', cursor);
  }
}

template <typename Duration>
constexpr size_t BufferSizeHH_MM_SS() {
  return detail::Digits10(23) + 1 + detail::Digits10(59) + 1 + detail::Digits10(59) + 1 +
         detail::Digits10(Duration::period::den) - 1;
}

template <typename Duration>
void FormatHH_MM_SS(arrow_vendored::date::hh_mm_ss<Duration> hms, char** cursor) {
  constexpr size_t subsecond_digits = Digits10(Duration::period::den) - 1;
  if (subsecond_digits != 0) {
    FormatAllDigitsLeftPadded(hms.subseconds().count(), subsecond_digits, '0', cursor);
    FormatOneChar('.', cursor);
  }
  FormatTwoDigits(hms.seconds().count(), cursor);
  FormatOneChar(':', cursor);
  FormatTwoDigits(hms.minutes().count(), cursor);
  FormatOneChar(':', cursor);
  FormatTwoDigits(hms.hours().count(), cursor);
}

// Some out-of-bound datetime values would result in erroneous printing
// because of silent integer wraparound in the `arrow_vendored::date` library.
//
// To avoid such misprinting, we must therefore check the bounds explicitly.
// The bounds correspond to start of year -32767 and end of year 32767,
// respectively (-32768 is an invalid year value in `arrow_vendored::date`).
//
// Note these values are the same as documented for C++20:
// https://en.cppreference.com/w/cpp/chrono/year_month_day/operator_days
template <typename Unit>
bool IsDateTimeInRange(Unit duration) {
  constexpr Unit kMinIncl =
      std::chrono::duration_cast<Unit>(arrow_vendored::date::days{-12687428});
  constexpr Unit kMaxExcl =
      std::chrono::duration_cast<Unit>(arrow_vendored::date::days{11248738});
  return duration >= kMinIncl && duration < kMaxExcl;
}

// IsDateTimeInRange() specialization for nanoseconds: a 64-bit number of
// nanoseconds cannot represent years outside of the [-32767, 32767]
// range, and the {kMinIncl, kMaxExcl} constants above would overflow.
constexpr bool IsDateTimeInRange(std::chrono::nanoseconds duration) { return true; }

template <typename Unit>
bool IsTimeInRange(Unit duration) {
  constexpr Unit kMinIncl = std::chrono::duration_cast<Unit>(std::chrono::seconds{0});
  constexpr Unit kMaxExcl = std::chrono::duration_cast<Unit>(std::chrono::seconds{86400});
  return duration >= kMinIncl && duration < kMaxExcl;
}

template <typename RawValue, typename Appender>
Return<Appender> FormatOutOfRange(RawValue&& raw_value, Appender&& append) {
  // XXX locale-sensitive but good enough for now
  std::string formatted = "<value out of range: " + ToChars(raw_value) + ">";
  return append(std::move(formatted));
}

const auto kEpoch = arrow_vendored::date::sys_days{arrow_vendored::date::jan / 1 / 1970};

}  // namespace detail

template <>
class StringFormatter<DurationType> : public IntToStringFormatterMixin<DurationType> {
  using IntToStringFormatterMixin::IntToStringFormatterMixin;
};

class DateToStringFormatterMixin {
 public:
  explicit DateToStringFormatterMixin(const DataType* = NULLPTR) {}

 protected:
  template <typename Appender>
  Return<Appender> FormatDays(arrow_vendored::date::days since_epoch, Appender&& append) {
    arrow_vendored::date::sys_days timepoint_days{since_epoch};

    constexpr size_t buffer_size = detail::BufferSizeYYYY_MM_DD();

    std::array<char, buffer_size> buffer;
    char* cursor = buffer.data() + buffer_size;

    detail::FormatYYYY_MM_DD(arrow_vendored::date::year_month_day{timepoint_days},
                             &cursor);
    return append(detail::ViewDigitBuffer(buffer, cursor));
  }
};

template <>
class StringFormatter<Date32Type> : public DateToStringFormatterMixin {
 public:
  using value_type = typename Date32Type::c_type;

  using DateToStringFormatterMixin::DateToStringFormatterMixin;

  template <typename Appender>
  Return<Appender> operator()(value_type value, Appender&& append) {
    const auto since_epoch = arrow_vendored::date::days{value};
    if (!ARROW_PREDICT_TRUE(detail::IsDateTimeInRange(since_epoch))) {
      return detail::FormatOutOfRange(value, append);
    }
    return FormatDays(since_epoch, std::forward<Appender>(append));
  }
};

template <>
class StringFormatter<Date64Type> : public DateToStringFormatterMixin {
 public:
  using value_type = typename Date64Type::c_type;

  using DateToStringFormatterMixin::DateToStringFormatterMixin;

  template <typename Appender>
  Return<Appender> operator()(value_type value, Appender&& append) {
    const auto since_epoch = std::chrono::milliseconds{value};
    if (!ARROW_PREDICT_TRUE(detail::IsDateTimeInRange(since_epoch))) {
      return detail::FormatOutOfRange(value, append);
    }
    return FormatDays(std::chrono::duration_cast<arrow_vendored::date::days>(since_epoch),
                      std::forward<Appender>(append));
  }
};

template <>
class StringFormatter<TimestampType> {
 public:
  using value_type = int64_t;

  explicit StringFormatter(const DataType* type)
      : unit_(checked_cast<const TimestampType&>(*type).unit()) {}

  template <typename Duration, typename Appender>
  Return<Appender> operator()(Duration, value_type value, Appender&& append) {
    using arrow_vendored::date::days;

    const Duration since_epoch{value};
    if (!ARROW_PREDICT_TRUE(detail::IsDateTimeInRange(since_epoch))) {
      return detail::FormatOutOfRange(value, append);
    }

    const auto timepoint = detail::kEpoch + since_epoch;
    // Round days towards zero
    // (the naive approach of using arrow_vendored::date::floor() would
    //  result in UB for very large negative timestamps, similarly as
    //  https://github.com/HowardHinnant/date/issues/696)
    auto timepoint_days = std::chrono::time_point_cast<days>(timepoint);
    Duration since_midnight;
    if (timepoint_days <= timepoint) {
      // Year >= 1970
      since_midnight = timepoint - timepoint_days;
    } else {
      // Year < 1970
      since_midnight = days(1) - (timepoint_days - timepoint);
      timepoint_days -= days(1);
    }

    constexpr size_t buffer_size =
        detail::BufferSizeYYYY_MM_DD() + 1 + detail::BufferSizeHH_MM_SS<Duration>();

    std::array<char, buffer_size> buffer;
    char* cursor = buffer.data() + buffer_size;

    detail::FormatHH_MM_SS(arrow_vendored::date::make_time(since_midnight), &cursor);
    detail::FormatOneChar(' ', &cursor);
    detail::FormatYYYY_MM_DD(timepoint_days, &cursor);
    return append(detail::ViewDigitBuffer(buffer, cursor));
  }

  template <typename Appender>
  Return<Appender> operator()(value_type value, Appender&& append) {
    return util::VisitDuration(unit_, *this, value, std::forward<Appender>(append));
  }

 private:
  TimeUnit::type unit_;
};

template <typename T>
class StringFormatter<T, enable_if_time<T>> {
 public:
  using value_type = typename T::c_type;

  explicit StringFormatter(const DataType* type)
      : unit_(checked_cast<const T&>(*type).unit()) {}

  template <typename Duration, typename Appender>
  Return<Appender> operator()(Duration, value_type count, Appender&& append) {
    const Duration since_midnight{count};
    if (!ARROW_PREDICT_TRUE(detail::IsTimeInRange(since_midnight))) {
      return detail::FormatOutOfRange(count, append);
    }

    constexpr size_t buffer_size = detail::BufferSizeHH_MM_SS<Duration>();

    std::array<char, buffer_size> buffer;
    char* cursor = buffer.data() + buffer_size;

    detail::FormatHH_MM_SS(arrow_vendored::date::make_time(since_midnight), &cursor);
    return append(detail::ViewDigitBuffer(buffer, cursor));
  }

  template <typename Appender>
  Return<Appender> operator()(value_type value, Appender&& append) {
    return util::VisitDuration(unit_, *this, value, std::forward<Appender>(append));
  }

 private:
  TimeUnit::type unit_;
};

template <>
class StringFormatter<MonthIntervalType> {
 public:
  using value_type = MonthIntervalType::c_type;

  explicit StringFormatter(const DataType*) {}

  template <typename Appender>
  Return<Appender> operator()(value_type interval, Appender&& append) {
    constexpr size_t buffer_size =
        /*'m'*/ 3 + /*negative signs*/ 1 +
        /*months*/ detail::Digits10(std::numeric_limits<value_type>::max());
    std::array<char, buffer_size> buffer;
    char* cursor = buffer.data() + buffer_size;

    detail::FormatOneChar('M', &cursor);
    detail::FormatAllDigits(detail::Abs(interval), &cursor);
    if (interval < 0) detail::FormatOneChar('-', &cursor);

    return append(detail::ViewDigitBuffer(buffer, cursor));
  }
};

template <>
class StringFormatter<DayTimeIntervalType> {
 public:
  using value_type = DayTimeIntervalType::DayMilliseconds;

  explicit StringFormatter(const DataType*) {}

  template <typename Appender>
  Return<Appender> operator()(value_type interval, Appender&& append) {
    constexpr size_t buffer_size =
        /*d, ms*/ 3 + /*negative signs*/ 2 +
        /*days/milliseconds*/ 2 * detail::Digits10(std::numeric_limits<int32_t>::max());
    std::array<char, buffer_size> buffer;
    char* cursor = buffer.data() + buffer_size;

    detail::FormatOneChar('s', &cursor);
    detail::FormatOneChar('m', &cursor);
    detail::FormatAllDigits(detail::Abs(interval.milliseconds), &cursor);
    if (interval.milliseconds < 0) detail::FormatOneChar('-', &cursor);

    detail::FormatOneChar('d', &cursor);
    detail::FormatAllDigits(detail::Abs(interval.days), &cursor);
    if (interval.days < 0) detail::FormatOneChar('-', &cursor);

    return append(detail::ViewDigitBuffer(buffer, cursor));
  }
};

template <>
class StringFormatter<MonthDayNanoIntervalType> {
 public:
  using value_type = MonthDayNanoIntervalType::MonthDayNanos;

  explicit StringFormatter(const DataType*) {}

  template <typename Appender>
  Return<Appender> operator()(value_type interval, Appender&& append) {
    constexpr size_t buffer_size =
        /*m, d, ns*/ 4 + /*negative signs*/ 3 +
        /*months/days*/ 2 * detail::Digits10(std::numeric_limits<int32_t>::max()) +
        /*nanoseconds*/ detail::Digits10(std::numeric_limits<int64_t>::max());
    std::array<char, buffer_size> buffer;
    char* cursor = buffer.data() + buffer_size;

    detail::FormatOneChar('s', &cursor);
    detail::FormatOneChar('n', &cursor);
    detail::FormatAllDigits(detail::Abs(interval.nanoseconds), &cursor);
    if (interval.nanoseconds < 0) detail::FormatOneChar('-', &cursor);

    detail::FormatOneChar('d', &cursor);
    detail::FormatAllDigits(detail::Abs(interval.days), &cursor);
    if (interval.days < 0) detail::FormatOneChar('-', &cursor);

    detail::FormatOneChar('M', &cursor);
    detail::FormatAllDigits(detail::Abs(interval.months), &cursor);
    if (interval.months < 0) detail::FormatOneChar('-', &cursor);

    return append(detail::ViewDigitBuffer(buffer, cursor));
  }
};

}  // namespace internal
}  // namespace arrow

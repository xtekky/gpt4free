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
#include "datetime.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <regex>
#include <string_view>

#include "arrow/array.h"
#include "arrow/scalar.h"
#include "arrow/status.h"
#include "arrow/type.h"
#include "arrow/util/logging.h"
#include "arrow/util/regex.h"
#include "arrow/util/value_parsing.h"
#include "arrow/python/arrow_to_python_internal.h"
#include "arrow/python/common.h"
#include "arrow/python/helpers.h"
#include "arrow/python/platform.h"

namespace arrow {

using internal::RegexMatch;

namespace py {
namespace internal {

namespace {

bool MatchFixedOffset(const std::string& tz, std::string_view* sign,
                      std::string_view* hour, std::string_view* minute) {
  static const std::regex regex("^([+-])(0[0-9]|1[0-9]|2[0-3]):([0-5][0-9])$");
  if (tz.size() < 5) {
    return false;
  }
  return RegexMatch(regex, tz, {sign, hour, minute});
}

constexpr char* NonConst(const char* st) {
  // Hack for python versions < 3.7 where members of PyStruct members
  // where non-const (C++ doesn't like assigning string literals to these types)
  return const_cast<char*>(st);
}

static PyTypeObject MonthDayNanoTupleType = {};

static PyStructSequence_Field MonthDayNanoField[] = {
    {NonConst("months"), NonConst("The number of months in the interval")},
    {NonConst("days"), NonConst("The number days in the interval")},
    {NonConst("nanoseconds"), NonConst("The number of nanoseconds in the interval")},
    {nullptr, nullptr}};

static PyStructSequence_Desc MonthDayNanoTupleDesc = {
    NonConst("MonthDayNano"),
    NonConst("A calendar interval consisting of months, days and nanoseconds."),
    MonthDayNanoField,
    /*n_in_sequence=*/3};

}  // namespace

#ifndef PYPY_VERSION
PyDateTime_CAPI* datetime_api = nullptr;

void InitDatetime() {
  PyAcquireGIL lock;
  datetime_api =
      reinterpret_cast<PyDateTime_CAPI*>(PyCapsule_Import(PyDateTime_CAPSULE_NAME, 0));
  if (datetime_api == nullptr) {
    Py_FatalError("Could not import datetime C API");
  }
}
#endif

// The following code is adapted from
// https://github.com/numpy/numpy/blob/master/numpy/core/src/multiarray/datetime.c

// Days per month, regular year and leap year
static int64_t _days_per_month_table[2][12] = {
    {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
    {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}};

static bool is_leapyear(int64_t year) {
  return (year & 0x3) == 0 &&  // year % 4 == 0
         ((year % 100) != 0 || (year % 400) == 0);
}

// Calculates the days offset from the 1970 epoch.
static int64_t get_days_from_date(int64_t date_year, int64_t date_month,
                                  int64_t date_day) {
  int64_t i, month;
  int64_t year, days = 0;
  int64_t* month_lengths;

  year = date_year - 1970;
  days = year * 365;

  // Adjust for leap years
  if (days >= 0) {
    // 1968 is the closest leap year before 1970.
    // Exclude the current year, so add 1.
    year += 1;
    // Add one day for each 4 years
    days += year / 4;
    // 1900 is the closest previous year divisible by 100
    year += 68;
    // Subtract one day for each 100 years
    days -= year / 100;
    // 1600 is the closest previous year divisible by 400
    year += 300;
    // Add one day for each 400 years
    days += year / 400;
  } else {
    // 1972 is the closest later year after 1970.
    // Include the current year, so subtract 2.
    year -= 2;
    // Subtract one day for each 4 years
    days += year / 4;
    // 2000 is the closest later year divisible by 100
    year -= 28;
    // Add one day for each 100 years
    days -= year / 100;
    // 2000 is also the closest later year divisible by 400
    // Subtract one day for each 400 years
    days += year / 400;
  }

  month_lengths = _days_per_month_table[is_leapyear(date_year)];
  month = date_month - 1;

  // Add the months
  for (i = 0; i < month; ++i) {
    days += month_lengths[i];
  }

  // Add the days
  days += date_day - 1;

  return days;
}

// Modifies '*days_' to be the day offset within the year,
// and returns the year.
static int64_t days_to_yearsdays(int64_t* days_) {
  const int64_t days_per_400years = (400 * 365 + 100 - 4 + 1);
  // Adjust so it's relative to the year 2000 (divisible by 400)
  int64_t days = (*days_) - (365 * 30 + 7);
  int64_t year;

  // Break down the 400 year cycle to get the year and day within the year
  if (days >= 0) {
    year = 400 * (days / days_per_400years);
    days = days % days_per_400years;
  } else {
    year = 400 * ((days - (days_per_400years - 1)) / days_per_400years);
    days = days % days_per_400years;
    if (days < 0) {
      days += days_per_400years;
    }
  }

  // Work out the year/day within the 400 year cycle
  if (days >= 366) {
    year += 100 * ((days - 1) / (100 * 365 + 25 - 1));
    days = (days - 1) % (100 * 365 + 25 - 1);
    if (days >= 365) {
      year += 4 * ((days + 1) / (4 * 365 + 1));
      days = (days + 1) % (4 * 365 + 1);
      if (days >= 366) {
        year += (days - 1) / 365;
        days = (days - 1) % 365;
      }
    }
  }

  *days_ = days;
  return year + 2000;
}

// Extracts the month and year and day number from a number of days
static void get_date_from_days(int64_t days, int64_t* date_year, int64_t* date_month,
                               int64_t* date_day) {
  int64_t *month_lengths, i;

  *date_year = days_to_yearsdays(&days);
  month_lengths = _days_per_month_table[is_leapyear(*date_year)];

  for (i = 0; i < 12; ++i) {
    if (days < month_lengths[i]) {
      *date_month = i + 1;
      *date_day = days + 1;
      return;
    } else {
      days -= month_lengths[i];
    }
  }

  // Should never get here
  return;
}

// Splitting time quantities, for example splitting total seconds into
// minutes and remaining seconds. After we run
// int64_t remaining = split_time(total, quotient, &next)
// we have
// total = next * quotient + remaining. Handles negative values by propagating
// them: If total is negative, next will be negative and remaining will
// always be non-negative.
static inline int64_t split_time(int64_t total, int64_t quotient, int64_t* next) {
  int64_t r = total % quotient;
  if (r < 0) {
    *next = total / quotient - 1;
    return r + quotient;
  } else {
    *next = total / quotient;
    return r;
  }
}

static inline Status PyTime_convert_int(int64_t val, const TimeUnit::type unit,
                                        int64_t* hour, int64_t* minute, int64_t* second,
                                        int64_t* microsecond) {
  switch (unit) {
    case TimeUnit::NANO:
      if (val % 1000 != 0) {
        return Status::Invalid("Value ", val, " has non-zero nanoseconds");
      }
      val /= 1000;
    // fall through
    case TimeUnit::MICRO:
      *microsecond = split_time(val, 1000000LL, &val);
      *second = split_time(val, 60, &val);
      *minute = split_time(val, 60, hour);
      break;
    case TimeUnit::MILLI:
      *microsecond = split_time(val, 1000, &val) * 1000;
    // fall through
    case TimeUnit::SECOND:
      *second = split_time(val, 60, &val);
      *minute = split_time(val, 60, hour);
      break;
    default:
      break;
  }
  return Status::OK();
}

static inline Status PyDate_convert_int(int64_t val, const DateUnit unit, int64_t* year,
                                        int64_t* month, int64_t* day) {
  switch (unit) {
    case DateUnit::MILLI:
      val /= 86400000LL;  // fall through
    case DateUnit::DAY:
      get_date_from_days(val, year, month, day);
    default:
      break;
  }
  return Status::OK();
}

PyObject* NewMonthDayNanoTupleType() {
  if (MonthDayNanoTupleType.tp_name == nullptr) {
    if (PyStructSequence_InitType2(&MonthDayNanoTupleType, &MonthDayNanoTupleDesc) != 0) {
      Py_FatalError("Could not initialize MonthDayNanoTuple");
    }
  }
  Py_INCREF(&MonthDayNanoTupleType);
  return (PyObject*)&MonthDayNanoTupleType;
}

Status PyTime_from_int(int64_t val, const TimeUnit::type unit, PyObject** out) {
  int64_t hour = 0, minute = 0, second = 0, microsecond = 0;
  RETURN_NOT_OK(PyTime_convert_int(val, unit, &hour, &minute, &second, &microsecond));
  *out = PyTime_FromTime(static_cast<int32_t>(hour), static_cast<int32_t>(minute),
                         static_cast<int32_t>(second), static_cast<int32_t>(microsecond));
  return Status::OK();
}

Status PyDate_from_int(int64_t val, const DateUnit unit, PyObject** out) {
  int64_t year = 0, month = 0, day = 0;
  RETURN_NOT_OK(PyDate_convert_int(val, unit, &year, &month, &day));
  *out = PyDate_FromDate(static_cast<int32_t>(year), static_cast<int32_t>(month),
                         static_cast<int32_t>(day));
  return Status::OK();
}

Status PyDateTime_from_int(int64_t val, const TimeUnit::type unit, PyObject** out) {
  int64_t hour = 0, minute = 0, second = 0, microsecond = 0;
  RETURN_NOT_OK(PyTime_convert_int(val, unit, &hour, &minute, &second, &microsecond));
  int64_t total_days = 0;
  hour = split_time(hour, 24, &total_days);
  int64_t year = 0, month = 0, day = 0;
  get_date_from_days(total_days, &year, &month, &day);
  *out = PyDateTime_FromDateAndTime(
      static_cast<int32_t>(year), static_cast<int32_t>(month), static_cast<int32_t>(day),
      static_cast<int32_t>(hour), static_cast<int32_t>(minute),
      static_cast<int32_t>(second), static_cast<int32_t>(microsecond));
  return Status::OK();
}

int64_t PyDate_to_days(PyDateTime_Date* pydate) {
  return get_days_from_date(PyDateTime_GET_YEAR(pydate), PyDateTime_GET_MONTH(pydate),
                            PyDateTime_GET_DAY(pydate));
}

Result<int64_t> PyDateTime_utcoffset_s(PyObject* obj) {
  // calculate offset from UTC timezone in seconds
  // supports only PyDateTime_DateTime and PyDateTime_Time objects
  OwnedRef pyoffset(PyObject_CallMethod(obj, "utcoffset", NULL));
  RETURN_IF_PYERROR();
  if (pyoffset.obj() != nullptr && pyoffset.obj() != Py_None) {
    auto delta = reinterpret_cast<PyDateTime_Delta*>(pyoffset.obj());
    return internal::PyDelta_to_s(delta);
  } else {
    return 0;
  }
}

Result<std::string> PyTZInfo_utcoffset_hhmm(PyObject* pytzinfo) {
  // attempt to convert timezone offset objects to "+/-{hh}:{mm}" format
  OwnedRef pydelta_object(PyObject_CallMethod(pytzinfo, "utcoffset", "O", Py_None));
  RETURN_IF_PYERROR();

  if (!PyDelta_Check(pydelta_object.obj())) {
    return Status::Invalid(
        "Object returned by tzinfo.utcoffset(None) is not an instance of "
        "datetime.timedelta");
  }
  auto pydelta = reinterpret_cast<PyDateTime_Delta*>(pydelta_object.obj());

  // retrieve the offset as seconds
  auto total_seconds = internal::PyDelta_to_s(pydelta);

  // determine whether the offset is positive or negative
  auto sign = (total_seconds < 0) ? "-" : "+";
  total_seconds = abs(total_seconds);

  // calculate offset components
  int64_t hours, minutes, seconds;
  seconds = split_time(total_seconds, 60, &minutes);
  minutes = split_time(minutes, 60, &hours);
  if (seconds > 0) {
    // check there are no remaining seconds
    return Status::Invalid("Offset must represent whole number of minutes");
  }

  // construct the timezone string
  std::stringstream stream;
  stream << sign << std::setfill('0') << std::setw(2) << hours << ":" << std::setfill('0')
         << std::setw(2) << minutes;
  return stream.str();
}

// Converted from python.  See https://github.com/apache/arrow/pull/7604
// for details.
Result<PyObject*> StringToTzinfo(const std::string& tz) {
  std::string_view sign_str, hour_str, minute_str;
  OwnedRef pytz;
  OwnedRef zoneinfo;
  OwnedRef datetime;

  if (internal::ImportModule("pytz", &pytz).ok()) {
    if (MatchFixedOffset(tz, &sign_str, &hour_str, &minute_str)) {
      int sign = -1;
      if (sign_str == "+") {
        sign = 1;
      }
      OwnedRef fixed_offset;
      RETURN_NOT_OK(internal::ImportFromModule(pytz.obj(), "FixedOffset", &fixed_offset));
      uint32_t minutes, hours;
      if (!::arrow::internal::ParseUnsigned(hour_str.data(), hour_str.size(), &hours) ||
          !::arrow::internal::ParseUnsigned(minute_str.data(), minute_str.size(),
                                            &minutes)) {
        return Status::Invalid("Invalid timezone: ", tz);
      }
      OwnedRef total_minutes(PyLong_FromLong(
          sign * ((static_cast<int>(hours) * 60) + static_cast<int>(minutes))));
      RETURN_IF_PYERROR();
      auto tzinfo =
          PyObject_CallFunctionObjArgs(fixed_offset.obj(), total_minutes.obj(), NULL);
      RETURN_IF_PYERROR();
      return tzinfo;
    }

    OwnedRef timezone;
    RETURN_NOT_OK(internal::ImportFromModule(pytz.obj(), "timezone", &timezone));
    OwnedRef py_tz_string(
        PyUnicode_FromStringAndSize(tz.c_str(), static_cast<Py_ssize_t>(tz.size())));
    auto tzinfo = PyObject_CallFunctionObjArgs(timezone.obj(), py_tz_string.obj(), NULL);
    RETURN_IF_PYERROR();
    return tzinfo;
  }

  // catch fixed offset if pytz is not present
  if (MatchFixedOffset(tz, &sign_str, &hour_str, &minute_str)) {
    RETURN_NOT_OK(internal::ImportModule("datetime", &datetime));
    int sign = -1;
    if (sign_str == "+") {
      sign = 1;
    }

    // import timezone and timedelta module to create a tzinfo object
    OwnedRef class_timezone;
    OwnedRef class_timedelta;
    RETURN_NOT_OK(
        internal::ImportFromModule(datetime.obj(), "timezone", &class_timezone));
    RETURN_NOT_OK(
        internal::ImportFromModule(datetime.obj(), "timedelta", &class_timedelta));

    // check input
    uint32_t minutes, hours;
    if (!::arrow::internal::ParseUnsigned(hour_str.data(), hour_str.size(), &hours) ||
        !::arrow::internal::ParseUnsigned(minute_str.data(), minute_str.size(),
                                          &minutes)) {
      return Status::Invalid("Invalid timezone: ", tz);
    }

    // save offset as a signed integer
    OwnedRef total_minutes(PyLong_FromLong(
        sign * ((static_cast<int>(hours) * 60) + static_cast<int>(minutes))));
    // create zero integers for empty arguments in datetime.timedelta
    OwnedRef zero(PyLong_FromLong(static_cast<int>(0)));

    // call datetime.timedelta to get correct offset object for datetime.timezone
    auto offset =
        PyObject_CallFunctionObjArgs(class_timedelta.obj(), zero.obj(), zero.obj(),
                                     zero.obj(), zero.obj(), total_minutes.obj(), NULL);
    RETURN_IF_PYERROR();
    // call datetime.timezone
    auto tzinfo = PyObject_CallFunctionObjArgs(class_timezone.obj(), offset, NULL);
    RETURN_IF_PYERROR();
    return tzinfo;
  }

  // fallback on zoneinfo if tz is string and pytz is not present
  if (internal::ImportModule("zoneinfo", &zoneinfo).ok()) {
    OwnedRef class_zoneinfo;
    RETURN_NOT_OK(
        internal::ImportFromModule(zoneinfo.obj(), "ZoneInfo", &class_zoneinfo));
    OwnedRef py_tz_string(
        PyUnicode_FromStringAndSize(tz.c_str(), static_cast<Py_ssize_t>(tz.size())));
    auto tzinfo =
        PyObject_CallFunctionObjArgs(class_zoneinfo.obj(), py_tz_string.obj(), NULL);
    RETURN_IF_PYERROR();
    return tzinfo;
  }

  return Status::Invalid(
      "Pytz package or Python>=3.8 for zoneinfo module must be installed.");
}

Result<std::string> TzinfoToString(PyObject* tzinfo) {
  OwnedRef module_pytz;        // import pytz
  OwnedRef module_datetime;    // import datetime
  OwnedRef module_zoneinfo;    // import zoneinfo
  OwnedRef module_dateutil;    // import dateutil
  OwnedRef class_timezone;     // from datetime import timezone
  OwnedRef class_fixedoffset;  // from pytz import _FixedOffset
  OwnedRef class_basetzinfo;   // from pytz import BaseTzInfo
  OwnedRef class_zoneinfo;     // from zoneinfo import ZoneInfo
  OwnedRef class_tzfile;       // from zoneinfo import tzfile

  // import necessary modules
  RETURN_NOT_OK(internal::ImportModule("datetime", &module_datetime));
  // import necessary classes
  RETURN_NOT_OK(
      internal::ImportFromModule(module_datetime.obj(), "timezone", &class_timezone));

  // check that it's a valid tzinfo object
  if (!PyTZInfo_Check(tzinfo)) {
    return Status::TypeError("Not an instance of datetime.tzinfo");
  }

  // if tzinfo is an instance of datetime.timezone return the
  // HH:MM offset string representation
  if (PyObject_IsInstance(tzinfo, class_timezone.obj())) {
    // still recognize datetime.timezone.utc as UTC (instead of +00:00)
    OwnedRef tzname_object(PyObject_CallMethod(tzinfo, "tzname", "O", Py_None));
    RETURN_IF_PYERROR();
    if (PyUnicode_Check(tzname_object.obj())) {
      std::string result;
      RETURN_NOT_OK(internal::PyUnicode_AsStdString(tzname_object.obj(), &result));
      if (result == "UTC") {
        return result;
      }
    }
    return PyTZInfo_utcoffset_hhmm(tzinfo);
  }

  // Try to import pytz if it is available
  if (internal::ImportModule("pytz", &module_pytz).ok()) {
    RETURN_NOT_OK(internal::ImportFromModule(module_pytz.obj(), "_FixedOffset",
                                             &class_fixedoffset));
    RETURN_NOT_OK(
        internal::ImportFromModule(module_pytz.obj(), "BaseTzInfo", &class_basetzinfo));
  }

  // if tzinfo is an instance of pytz._FixedOffset return the
  // HH:MM offset string representation
  if (module_pytz.obj() != nullptr &&
      PyObject_IsInstance(tzinfo, class_fixedoffset.obj())) {
    OwnedRef tzname_object(PyObject_CallMethod(tzinfo, "tzname", "O", Py_None));
    RETURN_IF_PYERROR();
    return PyTZInfo_utcoffset_hhmm(tzinfo);
  }

  // if pytz is installed and tzinfo is and instance of pytz.BaseTzInfo
  if (module_pytz.obj() != nullptr &&
      PyObject_IsInstance(tzinfo, class_basetzinfo.obj())) {
    OwnedRef zone(PyObject_GetAttrString(tzinfo, "zone"));
    RETURN_IF_PYERROR();
    std::string result;
    RETURN_NOT_OK(internal::PyUnicode_AsStdString(zone.obj(), &result));
    return result;
  }

  // Try to import zoneinfo if it is available
  if (internal::ImportModule("zoneinfo", &module_zoneinfo).ok()) {
    RETURN_NOT_OK(
        internal::ImportFromModule(module_zoneinfo.obj(), "ZoneInfo", &class_zoneinfo));
  }

  // if zoneinfo is installed and tzinfo is an instance of zoneinfo.ZoneInfo
  if (module_zoneinfo.obj() != nullptr &&
      PyObject_IsInstance(tzinfo, class_zoneinfo.obj())) {
    OwnedRef key(PyObject_GetAttrString(tzinfo, "key"));
    RETURN_IF_PYERROR();
    std::string result;
    RETURN_NOT_OK(internal::PyUnicode_AsStdString(key.obj(), &result));
    return result;
  }

  // Try to import dateutil if it is available
  if (internal::ImportModule("dateutil.tz", &module_dateutil).ok()) {
    RETURN_NOT_OK(
        internal::ImportFromModule(module_dateutil.obj(), "tzfile", &class_tzfile));
  }

  // if dateutil is installed and tzinfo is an instance of dateutil.tz.tzfile
  if (module_dateutil.obj() != nullptr &&
      PyObject_IsInstance(tzinfo, class_tzfile.obj())) {
    OwnedRef _filename(PyObject_GetAttrString(tzinfo, "_filename"));
    RETURN_IF_PYERROR();
    std::string result;
    RETURN_NOT_OK(internal::PyUnicode_AsStdString(_filename.obj(), &result));
    // _filename returns a full path in general ('/usr/share/zoneinfo/Europe/Paris')
    // or POSIX name on Windows ('Europe/Paris') - we need a substring in first case
    std::size_t pos = result.find("zoneinfo/");
    if (pos != std::string::npos) {
      return result.substr(pos + 9);
    }
    return result;
  }

  // attempt to call tzinfo.tzname(None)
  OwnedRef tzname_object(PyObject_CallMethod(tzinfo, "tzname", "O", Py_None));
  RETURN_IF_PYERROR();
  if (PyUnicode_Check(tzname_object.obj())) {
    std::string result;
    RETURN_NOT_OK(internal::PyUnicode_AsStdString(tzname_object.obj(), &result));
    return result;
  }

  // fall back to HH:MM offset string representation based on tzinfo.utcoffset(None)
  return PyTZInfo_utcoffset_hhmm(tzinfo);
}

PyObject* MonthDayNanoIntervalToNamedTuple(
    const MonthDayNanoIntervalType::MonthDayNanos& interval) {
  OwnedRef tuple(PyStructSequence_New(&MonthDayNanoTupleType));
  if (ARROW_PREDICT_FALSE(tuple.obj() == nullptr)) {
    return nullptr;
  }
  PyStructSequence_SetItem(tuple.obj(), /*pos=*/0, PyLong_FromLong(interval.months));
  PyStructSequence_SetItem(tuple.obj(), /*pos=*/1, PyLong_FromLong(interval.days));
  PyStructSequence_SetItem(tuple.obj(), /*pos=*/2,
                           PyLong_FromLongLong(interval.nanoseconds));
  return tuple.detach();
}

namespace {

// Wrapper around a Python list object that mimics dereference and assignment
// operations.
struct PyListAssigner {
 public:
  explicit PyListAssigner(PyObject* list) : list_(list) { DCHECK(PyList_Check(list_)); }

  PyListAssigner& operator*() { return *this; }

  void operator=(PyObject* obj) {
    if (ARROW_PREDICT_FALSE(PyList_SetItem(list_, current_index_, obj) == -1)) {
      Py_FatalError("list did not have the correct preallocated size.");
    }
  }

  PyListAssigner& operator++() {
    current_index_++;
    return *this;
  }

  PyListAssigner& operator+=(int64_t offset) {
    current_index_ += offset;
    return *this;
  }

 private:
  PyObject* list_;
  int64_t current_index_ = 0;
};

}  // namespace

Result<PyObject*> MonthDayNanoIntervalArrayToPyList(
    const MonthDayNanoIntervalArray& array) {
  OwnedRef out_list(PyList_New(array.length()));
  RETURN_IF_PYERROR();
  PyListAssigner out_objects(out_list.obj());
  auto& interval_array =
      arrow::internal::checked_cast<const MonthDayNanoIntervalArray&>(array);
  RETURN_NOT_OK(internal::WriteArrayObjects(
      interval_array,
      [&](const MonthDayNanoIntervalType::MonthDayNanos& interval, PyListAssigner& out) {
        PyObject* tuple = internal::MonthDayNanoIntervalToNamedTuple(interval);
        if (ARROW_PREDICT_FALSE(tuple == nullptr)) {
          RETURN_IF_PYERROR();
        }

        *out = tuple;
        return Status::OK();
      },
      out_objects));
  return out_list.detach();
}

Result<PyObject*> MonthDayNanoIntervalScalarToPyObject(
    const MonthDayNanoIntervalScalar& scalar) {
  if (scalar.is_valid) {
    return internal::MonthDayNanoIntervalToNamedTuple(scalar.value);
  } else {
    Py_INCREF(Py_None);
    return Py_None;
  }
}

}  // namespace internal
}  // namespace py
}  // namespace arrow

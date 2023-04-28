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
#include <chrono>

#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type.h"
#include "arrow/type_fwd.h"
#include "arrow/util/int_util_overflow.h"
#include "arrow/util/logging.h"
#include "arrow/python/platform.h"
#include "arrow/python/visibility.h"

// By default, PyDateTimeAPI is a *static* variable.  This forces
// PyDateTime_IMPORT to be called in every C/C++ module using the
// C datetime API.  This is error-prone and potentially costly.
// Instead, we redefine PyDateTimeAPI to point to a global variable,
// which is initialized once by calling InitDatetime().
#ifdef PYPY_VERSION
#include "datetime.h"
#else
#define PyDateTimeAPI ::arrow::py::internal::datetime_api
#endif

namespace arrow {
using internal::AddWithOverflow;
using internal::MultiplyWithOverflow;
namespace py {
namespace internal {

#ifndef PYPY_VERSION
extern PyDateTime_CAPI* datetime_api;

ARROW_PYTHON_EXPORT
void InitDatetime();
#endif

// Returns the MonthDayNano namedtuple type (increments the reference count).
ARROW_PYTHON_EXPORT
PyObject* NewMonthDayNanoTupleType();

ARROW_PYTHON_EXPORT
inline int64_t PyTime_to_us(PyObject* pytime) {
  return (PyDateTime_TIME_GET_HOUR(pytime) * 3600000000LL +
          PyDateTime_TIME_GET_MINUTE(pytime) * 60000000LL +
          PyDateTime_TIME_GET_SECOND(pytime) * 1000000LL +
          PyDateTime_TIME_GET_MICROSECOND(pytime));
}

ARROW_PYTHON_EXPORT
inline int64_t PyTime_to_s(PyObject* pytime) { return PyTime_to_us(pytime) / 1000000; }

ARROW_PYTHON_EXPORT
inline int64_t PyTime_to_ms(PyObject* pytime) { return PyTime_to_us(pytime) / 1000; }

ARROW_PYTHON_EXPORT
inline int64_t PyTime_to_ns(PyObject* pytime) { return PyTime_to_us(pytime) * 1000; }

ARROW_PYTHON_EXPORT
Status PyTime_from_int(int64_t val, const TimeUnit::type unit, PyObject** out);

ARROW_PYTHON_EXPORT
Status PyDate_from_int(int64_t val, const DateUnit unit, PyObject** out);

// WARNING: This function returns a naive datetime.
ARROW_PYTHON_EXPORT
Status PyDateTime_from_int(int64_t val, const TimeUnit::type unit, PyObject** out);

// This declaration must be the same as in filesystem/filesystem.h
using TimePoint =
    std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>;

ARROW_PYTHON_EXPORT
int64_t PyDate_to_days(PyDateTime_Date* pydate);

ARROW_PYTHON_EXPORT
inline int64_t PyDate_to_s(PyDateTime_Date* pydate) {
  return PyDate_to_days(pydate) * 86400LL;
}

ARROW_PYTHON_EXPORT
inline int64_t PyDate_to_ms(PyDateTime_Date* pydate) {
  return PyDate_to_days(pydate) * 86400000LL;
}

ARROW_PYTHON_EXPORT
inline int64_t PyDateTime_to_s(PyDateTime_DateTime* pydatetime) {
  return (PyDate_to_s(reinterpret_cast<PyDateTime_Date*>(pydatetime)) +
          PyDateTime_DATE_GET_HOUR(pydatetime) * 3600LL +
          PyDateTime_DATE_GET_MINUTE(pydatetime) * 60LL +
          PyDateTime_DATE_GET_SECOND(pydatetime));
}

ARROW_PYTHON_EXPORT
inline int64_t PyDateTime_to_ms(PyDateTime_DateTime* pydatetime) {
  return (PyDateTime_to_s(pydatetime) * 1000LL +
          PyDateTime_DATE_GET_MICROSECOND(pydatetime) / 1000);
}

ARROW_PYTHON_EXPORT
inline int64_t PyDateTime_to_us(PyDateTime_DateTime* pydatetime) {
  return (PyDateTime_to_s(pydatetime) * 1000000LL +
          PyDateTime_DATE_GET_MICROSECOND(pydatetime));
}

ARROW_PYTHON_EXPORT
inline int64_t PyDateTime_to_ns(PyDateTime_DateTime* pydatetime) {
  return PyDateTime_to_us(pydatetime) * 1000LL;
}

ARROW_PYTHON_EXPORT
inline TimePoint PyDateTime_to_TimePoint(PyDateTime_DateTime* pydatetime) {
  return TimePoint(TimePoint::duration(PyDateTime_to_ns(pydatetime)));
}

ARROW_PYTHON_EXPORT
inline int64_t TimePoint_to_ns(TimePoint val) { return val.time_since_epoch().count(); }

ARROW_PYTHON_EXPORT
inline TimePoint TimePoint_from_s(double val) {
  return TimePoint(TimePoint::duration(static_cast<int64_t>(1e9 * val)));
}

ARROW_PYTHON_EXPORT
inline TimePoint TimePoint_from_ns(int64_t val) {
  return TimePoint(TimePoint::duration(val));
}

ARROW_PYTHON_EXPORT
inline int64_t PyDelta_to_s(PyDateTime_Delta* pytimedelta) {
  return (PyDateTime_DELTA_GET_DAYS(pytimedelta) * 86400LL +
          PyDateTime_DELTA_GET_SECONDS(pytimedelta));
}

ARROW_PYTHON_EXPORT
inline int64_t PyDelta_to_ms(PyDateTime_Delta* pytimedelta) {
  return (PyDelta_to_s(pytimedelta) * 1000LL +
          PyDateTime_DELTA_GET_MICROSECONDS(pytimedelta) / 1000);
}

ARROW_PYTHON_EXPORT
inline Result<int64_t> PyDelta_to_us(PyDateTime_Delta* pytimedelta) {
  int64_t result = PyDelta_to_s(pytimedelta);
  if (MultiplyWithOverflow(result, 1000000LL, &result)) {
    return Status::Invalid("Timedelta too large to fit in 64-bit integer");
  }
  if (AddWithOverflow(result, PyDateTime_DELTA_GET_MICROSECONDS(pytimedelta), &result)) {
    return Status::Invalid("Timedelta too large to fit in 64-bit integer");
  }
  return result;
}

ARROW_PYTHON_EXPORT
inline Result<int64_t> PyDelta_to_ns(PyDateTime_Delta* pytimedelta) {
  ARROW_ASSIGN_OR_RAISE(int64_t result, PyDelta_to_us(pytimedelta));
  if (MultiplyWithOverflow(result, 1000LL, &result)) {
    return Status::Invalid("Timedelta too large to fit in 64-bit integer");
  }
  return result;
}

ARROW_PYTHON_EXPORT
Result<int64_t> PyDateTime_utcoffset_s(PyObject* pydatetime);

/// \brief Convert a time zone name into a time zone object.
///
/// Supported input strings are:
/// * As used in the Olson time zone database (the "tz database" or
///   "tzdata"), such as "America/New_York"
/// * An absolute time zone offset of the form +XX:XX or -XX:XX, such as +07:30
/// GIL must be held when calling this method.
ARROW_PYTHON_EXPORT
Result<PyObject*> StringToTzinfo(const std::string& tz);

/// \brief Convert a time zone object to a string representation.
///
/// The output strings are:
/// * An absolute time zone offset of the form +XX:XX or -XX:XX, such as +07:30
///   if the input object is either an instance of pytz._FixedOffset or
///   datetime.timedelta
/// * The timezone's name if the input object's tzname() method returns with a
///   non-empty timezone name such as "UTC" or "America/New_York"
///
/// GIL must be held when calling this method.
ARROW_PYTHON_EXPORT
Result<std::string> TzinfoToString(PyObject* pytzinfo);

/// \brief Convert MonthDayNano to a python namedtuple.
///
/// Return a named tuple (pyarrow.MonthDayNano) containing attributes
/// "months", "days", "nanoseconds" in the given order
/// with values extracted from the fields on interval.
///
/// GIL must be held when calling this method.
ARROW_PYTHON_EXPORT
PyObject* MonthDayNanoIntervalToNamedTuple(
    const MonthDayNanoIntervalType::MonthDayNanos& interval);

/// \brief Convert the given Array to a PyList object containing
/// pyarrow.MonthDayNano objects.
ARROW_PYTHON_EXPORT
Result<PyObject*> MonthDayNanoIntervalArrayToPyList(
    const MonthDayNanoIntervalArray& array);

/// \brief Convert the Scalar obect to a pyarrow.MonthDayNano (or None if
/// is isn't valid).
ARROW_PYTHON_EXPORT
Result<PyObject*> MonthDayNanoIntervalScalarToPyObject(
    const MonthDayNanoIntervalScalar& scalar);

}  // namespace internal
}  // namespace py
}  // namespace arrow

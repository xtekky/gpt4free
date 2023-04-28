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

#ifdef GANDIVA_IR

// The LLVM IR code doesn't have an NDEBUG mode. And, it shouldn't include references to
// streams or stdc++. So, making the DCHECK calls void in that case.

#define ARROW_IGNORE_EXPR(expr) ((void)(expr))

#define DCHECK(condition) ARROW_IGNORE_EXPR(condition)
#define DCHECK_OK(status) ARROW_IGNORE_EXPR(status)
#define DCHECK_EQ(val1, val2) ARROW_IGNORE_EXPR(val1)
#define DCHECK_NE(val1, val2) ARROW_IGNORE_EXPR(val1)
#define DCHECK_LE(val1, val2) ARROW_IGNORE_EXPR(val1)
#define DCHECK_LT(val1, val2) ARROW_IGNORE_EXPR(val1)
#define DCHECK_GE(val1, val2) ARROW_IGNORE_EXPR(val1)
#define DCHECK_GT(val1, val2) ARROW_IGNORE_EXPR(val1)

#else  // !GANDIVA_IR

#include <memory>
#include <ostream>
#include <string>

#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace util {

enum class ArrowLogLevel : int {
  ARROW_DEBUG = -1,
  ARROW_INFO = 0,
  ARROW_WARNING = 1,
  ARROW_ERROR = 2,
  ARROW_FATAL = 3
};

#define ARROW_LOG_INTERNAL(level) ::arrow::util::ArrowLog(__FILE__, __LINE__, level)
#define ARROW_LOG(level) ARROW_LOG_INTERNAL(::arrow::util::ArrowLogLevel::ARROW_##level)

#define ARROW_IGNORE_EXPR(expr) ((void)(expr))

#define ARROW_CHECK_OR_LOG(condition, level) \
  ARROW_PREDICT_TRUE(condition)              \
  ? ARROW_IGNORE_EXPR(0)                     \
  : ::arrow::util::Voidify() & ARROW_LOG(level) << " Check failed: " #condition " "

#define ARROW_CHECK(condition) ARROW_CHECK_OR_LOG(condition, FATAL)

// If 'to_call' returns a bad status, CHECK immediately with a logged message
// of 'msg' followed by the status.
#define ARROW_CHECK_OK_PREPEND(to_call, msg, level)                 \
  do {                                                              \
    ::arrow::Status _s = (to_call);                                 \
    ARROW_CHECK_OR_LOG(_s.ok(), level)                              \
        << "Operation failed: " << ARROW_STRINGIFY(to_call) << "\n" \
        << (msg) << ": " << _s.ToString();                          \
  } while (false)

// If the status is bad, CHECK immediately, appending the status to the
// logged message.
#define ARROW_CHECK_OK(s) ARROW_CHECK_OK_PREPEND(s, "Bad status", FATAL)

#define ARROW_CHECK_EQ(val1, val2) ARROW_CHECK((val1) == (val2))
#define ARROW_CHECK_NE(val1, val2) ARROW_CHECK((val1) != (val2))
#define ARROW_CHECK_LE(val1, val2) ARROW_CHECK((val1) <= (val2))
#define ARROW_CHECK_LT(val1, val2) ARROW_CHECK((val1) < (val2))
#define ARROW_CHECK_GE(val1, val2) ARROW_CHECK((val1) >= (val2))
#define ARROW_CHECK_GT(val1, val2) ARROW_CHECK((val1) > (val2))

#ifdef NDEBUG
#define ARROW_DFATAL ::arrow::util::ArrowLogLevel::ARROW_WARNING

// CAUTION: DCHECK_OK() always evaluates its argument, but other DCHECK*() macros
// only do so in debug mode.

#define ARROW_DCHECK(condition)               \
  while (false) ARROW_IGNORE_EXPR(condition); \
  while (false) ::arrow::util::detail::NullLog()
#define ARROW_DCHECK_OK(s) \
  ARROW_IGNORE_EXPR(s);    \
  while (false) ::arrow::util::detail::NullLog()
#define ARROW_DCHECK_EQ(val1, val2)      \
  while (false) ARROW_IGNORE_EXPR(val1); \
  while (false) ARROW_IGNORE_EXPR(val2); \
  while (false) ::arrow::util::detail::NullLog()
#define ARROW_DCHECK_NE(val1, val2)      \
  while (false) ARROW_IGNORE_EXPR(val1); \
  while (false) ARROW_IGNORE_EXPR(val2); \
  while (false) ::arrow::util::detail::NullLog()
#define ARROW_DCHECK_LE(val1, val2)      \
  while (false) ARROW_IGNORE_EXPR(val1); \
  while (false) ARROW_IGNORE_EXPR(val2); \
  while (false) ::arrow::util::detail::NullLog()
#define ARROW_DCHECK_LT(val1, val2)      \
  while (false) ARROW_IGNORE_EXPR(val1); \
  while (false) ARROW_IGNORE_EXPR(val2); \
  while (false) ::arrow::util::detail::NullLog()
#define ARROW_DCHECK_GE(val1, val2)      \
  while (false) ARROW_IGNORE_EXPR(val1); \
  while (false) ARROW_IGNORE_EXPR(val2); \
  while (false) ::arrow::util::detail::NullLog()
#define ARROW_DCHECK_GT(val1, val2)      \
  while (false) ARROW_IGNORE_EXPR(val1); \
  while (false) ARROW_IGNORE_EXPR(val2); \
  while (false) ::arrow::util::detail::NullLog()

#else
#define ARROW_DFATAL ::arrow::util::ArrowLogLevel::ARROW_FATAL

#define ARROW_DCHECK ARROW_CHECK
#define ARROW_DCHECK_OK ARROW_CHECK_OK
#define ARROW_DCHECK_EQ ARROW_CHECK_EQ
#define ARROW_DCHECK_NE ARROW_CHECK_NE
#define ARROW_DCHECK_LE ARROW_CHECK_LE
#define ARROW_DCHECK_LT ARROW_CHECK_LT
#define ARROW_DCHECK_GE ARROW_CHECK_GE
#define ARROW_DCHECK_GT ARROW_CHECK_GT

#endif  // NDEBUG

#define DCHECK ARROW_DCHECK
#define DCHECK_OK ARROW_DCHECK_OK
#define DCHECK_EQ ARROW_DCHECK_EQ
#define DCHECK_NE ARROW_DCHECK_NE
#define DCHECK_LE ARROW_DCHECK_LE
#define DCHECK_LT ARROW_DCHECK_LT
#define DCHECK_GE ARROW_DCHECK_GE
#define DCHECK_GT ARROW_DCHECK_GT

// This code is adapted from
// https://github.com/ray-project/ray/blob/master/src/ray/util/logging.h.

// To make the logging lib pluggable with other logging libs and make
// the implementation unawared by the user, ArrowLog is only a declaration
// which hide the implementation into logging.cc file.
// In logging.cc, we can choose different log libs using different macros.

// This is also a null log which does not output anything.
class ARROW_EXPORT ArrowLogBase {
 public:
  virtual ~ArrowLogBase() {}

  virtual bool IsEnabled() const { return false; }

  template <typename T>
  ArrowLogBase& operator<<(const T& t) {
    if (IsEnabled()) {
      Stream() << t;
    }
    return *this;
  }

 protected:
  virtual std::ostream& Stream() = 0;
};

class ARROW_EXPORT ArrowLog : public ArrowLogBase {
 public:
  ArrowLog(const char* file_name, int line_number, ArrowLogLevel severity);
  ~ArrowLog() override;

  /// Return whether or not current logging instance is enabled.
  ///
  /// \return True if logging is enabled and false otherwise.
  bool IsEnabled() const override;

  /// The init function of arrow log for a program which should be called only once.
  ///
  /// \param appName The app name which starts the log.
  /// \param severity_threshold Logging threshold for the program.
  /// \param logDir Logging output file name. If empty, the log won't output to file.
  static void StartArrowLog(const std::string& appName,
                            ArrowLogLevel severity_threshold = ArrowLogLevel::ARROW_INFO,
                            const std::string& logDir = "");

  /// The shutdown function of arrow log, it should be used with StartArrowLog as a pair.
  static void ShutDownArrowLog();

  /// Install the failure signal handler to output call stack when crash.
  /// If glog is not installed, this function won't do anything.
  static void InstallFailureSignalHandler();

  /// Uninstall the signal actions installed by InstallFailureSignalHandler.
  static void UninstallSignalAction();

  /// Return whether or not the log level is enabled in current setting.
  ///
  /// \param log_level The input log level to test.
  /// \return True if input log level is not lower than the threshold.
  static bool IsLevelEnabled(ArrowLogLevel log_level);

 private:
  ARROW_DISALLOW_COPY_AND_ASSIGN(ArrowLog);

  // Hide the implementation of log provider by void *.
  // Otherwise, lib user may define the same macro to use the correct header file.
  void* logging_provider_;
  /// True if log messages should be logged and false if they should be ignored.
  bool is_enabled_;

  static ArrowLogLevel severity_threshold_;

 protected:
  std::ostream& Stream() override;
};

// This class make ARROW_CHECK compilation pass to change the << operator to void.
// This class is copied from glog.
class ARROW_EXPORT Voidify {
 public:
  Voidify() {}
  // This has to be an operator with a precedence lower than << but
  // higher than ?:
  void operator&(ArrowLogBase&) {}
};

namespace detail {

/// @brief A helper for the nil log sink.
///
/// Using this helper is analogous to sending log messages to /dev/null:
/// nothing gets logged.
class NullLog {
 public:
  /// The no-op output operator.
  ///
  /// @param [in] t
  ///   The object to send into the nil sink.
  /// @return Reference to the updated object.
  template <class T>
  NullLog& operator<<(const T& t) {
    return *this;
  }
};

}  // namespace detail
}  // namespace util
}  // namespace arrow

#endif  // GANDIVA_IR

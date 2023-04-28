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

#include <exception>
#include <sstream>
#include <string>
#include <utility>

#include "arrow/type_fwd.h"
#include "arrow/util/string_builder.h"
#include "parquet/platform.h"

// PARQUET-1085
#if !defined(ARROW_UNUSED)
#define ARROW_UNUSED(x) UNUSED(x)
#endif

// Parquet exception to Arrow Status

#define BEGIN_PARQUET_CATCH_EXCEPTIONS try {
#define END_PARQUET_CATCH_EXCEPTIONS                   \
  }                                                    \
  catch (const ::parquet::ParquetStatusException& e) { \
    return e.status();                                 \
  }                                                    \
  catch (const ::parquet::ParquetException& e) {       \
    return ::arrow::Status::IOError(e.what());         \
  }

// clang-format off

#define PARQUET_CATCH_NOT_OK(s)    \
  BEGIN_PARQUET_CATCH_EXCEPTIONS   \
  (s);                             \
  END_PARQUET_CATCH_EXCEPTIONS

// clang-format on

#define PARQUET_CATCH_AND_RETURN(s) \
  BEGIN_PARQUET_CATCH_EXCEPTIONS    \
  return (s);                       \
  END_PARQUET_CATCH_EXCEPTIONS

// Arrow Status to Parquet exception

#define PARQUET_IGNORE_NOT_OK(s)                                \
  do {                                                          \
    ::arrow::Status _s = ::arrow::internal::GenericToStatus(s); \
    ARROW_UNUSED(_s);                                           \
  } while (0)

#define PARQUET_THROW_NOT_OK(s)                                 \
  do {                                                          \
    ::arrow::Status _s = ::arrow::internal::GenericToStatus(s); \
    if (!_s.ok()) {                                             \
      throw ::parquet::ParquetStatusException(std::move(_s));   \
    }                                                           \
  } while (0)

#define PARQUET_ASSIGN_OR_THROW_IMPL(status_name, lhs, rexpr) \
  auto status_name = (rexpr);                                 \
  PARQUET_THROW_NOT_OK(status_name.status());                 \
  lhs = std::move(status_name).ValueOrDie();

#define PARQUET_ASSIGN_OR_THROW(lhs, rexpr)                                              \
  PARQUET_ASSIGN_OR_THROW_IMPL(ARROW_ASSIGN_OR_RAISE_NAME(_error_or_value, __COUNTER__), \
                               lhs, rexpr);

namespace parquet {

class ParquetException : public std::exception {
 public:
  PARQUET_NORETURN static void EofException(const std::string& msg = "") {
    static std::string prefix = "Unexpected end of stream";
    if (msg.empty()) {
      throw ParquetException(prefix);
    }
    throw ParquetException(prefix, ": ", msg);
  }

  PARQUET_NORETURN static void NYI(const std::string& msg = "") {
    throw ParquetException("Not yet implemented: ", msg, ".");
  }

  template <typename... Args>
  explicit ParquetException(Args&&... args)
      : msg_(::arrow::util::StringBuilder(std::forward<Args>(args)...)) {}

  explicit ParquetException(std::string msg) : msg_(std::move(msg)) {}

  explicit ParquetException(const char* msg, const std::exception&) : msg_(msg) {}

  ParquetException(const ParquetException&) = default;
  ParquetException& operator=(const ParquetException&) = default;
  ParquetException(ParquetException&&) = default;
  ParquetException& operator=(ParquetException&&) = default;

  const char* what() const noexcept override { return msg_.c_str(); }

 private:
  std::string msg_;
};

// Support printing a ParquetException.
// This is needed for clang-on-MSVC as there operator<< is not defined for
// std::exception.
PARQUET_EXPORT
std::ostream& operator<<(std::ostream& os, const ParquetException& exception);

class ParquetStatusException : public ParquetException {
 public:
  explicit ParquetStatusException(::arrow::Status status)
      : ParquetException(status.ToString()), status_(std::move(status)) {}

  const ::arrow::Status& status() const { return status_; }

 private:
  ::arrow::Status status_;
};

// This class exists for the purpose of detecting an invalid or corrupted file.
class ParquetInvalidOrCorruptedFileException : public ParquetStatusException {
 public:
  ParquetInvalidOrCorruptedFileException(const ParquetInvalidOrCorruptedFileException&) =
      default;

  template <typename Arg,
            typename std::enable_if<
                !std::is_base_of<ParquetInvalidOrCorruptedFileException, Arg>::value,
                int>::type = 0,
            typename... Args>
  explicit ParquetInvalidOrCorruptedFileException(Arg arg, Args&&... args)
      : ParquetStatusException(::arrow::Status::Invalid(std::forward<Arg>(arg),
                                                        std::forward<Args>(args)...)) {}
};

template <typename StatusReturnBlock>
void ThrowNotOk(StatusReturnBlock&& b) {
  PARQUET_THROW_NOT_OK(b());
}

}  // namespace parquet

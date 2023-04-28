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

#include <utility>

#include <gmock/gmock-matchers.h>

#include "arrow/datum.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/stl_iterator.h"
#include "arrow/testing/future_util.h"
#include "arrow/testing/gtest_util.h"
#include "arrow/util/future.h"
#include "arrow/util/unreachable.h"

namespace arrow {

class PointeesEqualMatcher {
 public:
  template <typename PtrPair>
  operator testing::Matcher<PtrPair>() const {  // NOLINT runtime/explicit
    struct Impl : testing::MatcherInterface<const PtrPair&> {
      void DescribeTo(::std::ostream* os) const override { *os << "pointees are equal"; }

      void DescribeNegationTo(::std::ostream* os) const override {
        *os << "pointees are not equal";
      }

      bool MatchAndExplain(const PtrPair& pair,
                           testing::MatchResultListener* listener) const override {
        const auto& first = *std::get<0>(pair);
        const auto& second = *std::get<1>(pair);
        const bool match = first.Equals(second);
        *listener << "whose pointees " << testing::PrintToString(first) << " and "
                  << testing::PrintToString(second)
                  << (match ? " are equal" : " are not equal");
        return match;
      }
    };

    return testing::Matcher<PtrPair>(new Impl());
  }
};

// A matcher that checks that the values pointed to are Equals().
// Useful in conjunction with other googletest matchers.
inline PointeesEqualMatcher PointeesEqual() { return {}; }

class AnyOfJSONMatcher {
 public:
  AnyOfJSONMatcher(std::shared_ptr<DataType> type, std::string array_json)
      : type_(std::move(type)), array_json_(std::move(array_json)) {}

  template <typename arg_type>
  operator testing::Matcher<arg_type>() const {  // NOLINT runtime/explicit
    struct Impl : testing::MatcherInterface<const arg_type&> {
      Impl(std::shared_ptr<DataType> type, std::string array_json)
          : type_(std::move(type)), array_json_(std::move(array_json)) {
        array = ArrayFromJSON(type_, array_json_);
      }
      void DescribeTo(std::ostream* os) const override {
        *os << "matches at least one scalar from ";
        *os << array->ToString();
      }
      void DescribeNegationTo(::std::ostream* os) const override {
        *os << "matches no scalar from ";
        *os << array->ToString();
      }
      bool MatchAndExplain(
          const arg_type& arg,
          ::testing::MatchResultListener* result_listener) const override {
        for (int64_t i = 0; i < array->length(); ++i) {
          std::shared_ptr<Scalar> scalar;
          auto maybe_scalar = array->GetScalar(i);
          if (maybe_scalar.ok()) {
            scalar = maybe_scalar.ValueOrDie();
          } else {
            *result_listener << "GetScalar() had status "
                             << maybe_scalar.status().ToString() << "at index " << i
                             << " in the input JSON Array";
            return false;
          }

          if (scalar->Equals(arg)) return true;
        }
        *result_listener << "Argument scalar: '" << arg->ToString()
                         << "' matches no scalar from " << array->ToString();
        return false;
      }
      const std::shared_ptr<DataType> type_;
      const std::string array_json_;
      std::shared_ptr<Array> array;
    };

    return testing::Matcher<arg_type>(new Impl(type_, array_json_));
  }

 private:
  const std::shared_ptr<DataType> type_;
  const std::string array_json_;
};

inline AnyOfJSONMatcher AnyOfJSON(std::shared_ptr<DataType> type,
                                  std::string array_json) {
  return {std::move(type), std::move(array_json)};
}

template <typename ResultMatcher>
class FutureMatcher {
 public:
  explicit FutureMatcher(ResultMatcher result_matcher, double wait_seconds)
      : result_matcher_(std::move(result_matcher)), wait_seconds_(wait_seconds) {}

  template <typename Fut,
            typename ValueType = typename std::decay<Fut>::type::ValueType>
  operator testing::Matcher<Fut>() const {  // NOLINT runtime/explicit
    struct Impl : testing::MatcherInterface<const Fut&> {
      explicit Impl(const ResultMatcher& result_matcher, double wait_seconds)
          : result_matcher_(testing::MatcherCast<Result<ValueType>>(result_matcher)),
            wait_seconds_(wait_seconds) {}

      void DescribeTo(::std::ostream* os) const override {
        *os << "value ";
        result_matcher_.DescribeTo(os);
      }

      void DescribeNegationTo(::std::ostream* os) const override {
        *os << "value ";
        result_matcher_.DescribeNegationTo(os);
      }

      bool MatchAndExplain(const Fut& fut,
                           testing::MatchResultListener* listener) const override {
        if (!fut.Wait(wait_seconds_)) {
          *listener << "which didn't finish within " << wait_seconds_ << " seconds";
          return false;
        }
        return result_matcher_.MatchAndExplain(fut.result(), listener);
      }

      const testing::Matcher<Result<ValueType>> result_matcher_;
      const double wait_seconds_;
    };

    return testing::Matcher<Fut>(new Impl(result_matcher_, wait_seconds_));
  }

 private:
  const ResultMatcher result_matcher_;
  const double wait_seconds_;
};

template <typename ValueMatcher>
class ResultMatcher {
 public:
  explicit ResultMatcher(ValueMatcher value_matcher)
      : value_matcher_(std::move(value_matcher)) {}

  template <typename Res,
            typename ValueType = typename std::decay<Res>::type::ValueType>
  operator testing::Matcher<Res>() const {  // NOLINT runtime/explicit
    struct Impl : testing::MatcherInterface<const Res&> {
      explicit Impl(const ValueMatcher& value_matcher)
          : value_matcher_(testing::MatcherCast<ValueType>(value_matcher)) {}

      void DescribeTo(::std::ostream* os) const override {
        *os << "value ";
        value_matcher_.DescribeTo(os);
      }

      void DescribeNegationTo(::std::ostream* os) const override {
        *os << "value ";
        value_matcher_.DescribeNegationTo(os);
      }

      bool MatchAndExplain(const Res& maybe_value,
                           testing::MatchResultListener* listener) const override {
        if (!maybe_value.status().ok()) {
          *listener << "whose error "
                    << testing::PrintToString(maybe_value.status().ToString())
                    << " doesn't match";
          return false;
        }
        const ValueType& value = maybe_value.ValueOrDie();
        testing::StringMatchResultListener value_listener;
        const bool match = value_matcher_.MatchAndExplain(value, &value_listener);
        *listener << "whose value " << testing::PrintToString(value)
                  << (match ? " matches" : " doesn't match");
        testing::internal::PrintIfNotEmpty(value_listener.str(), listener->stream());
        return match;
      }

      const testing::Matcher<ValueType> value_matcher_;
    };

    return testing::Matcher<Res>(new Impl(value_matcher_));
  }

 private:
  const ValueMatcher value_matcher_;
};

class ErrorMatcher {
 public:
  explicit ErrorMatcher(StatusCode code,
                        std::optional<testing::Matcher<std::string>> message_matcher)
      : code_(code), message_matcher_(std::move(message_matcher)) {}

  template <typename Res>
  operator testing::Matcher<Res>() const {  // NOLINT runtime/explicit
    struct Impl : testing::MatcherInterface<const Res&> {
      explicit Impl(StatusCode code,
                    std::optional<testing::Matcher<std::string>> message_matcher)
          : code_(code), message_matcher_(std::move(message_matcher)) {}

      void DescribeTo(::std::ostream* os) const override {
        *os << "raises StatusCode::" << Status::CodeAsString(code_);
        if (message_matcher_) {
          *os << " and message ";
          message_matcher_->DescribeTo(os);
        }
      }

      void DescribeNegationTo(::std::ostream* os) const override {
        *os << "does not raise StatusCode::" << Status::CodeAsString(code_);
        if (message_matcher_) {
          *os << " or message ";
          message_matcher_->DescribeNegationTo(os);
        }
      }

      bool MatchAndExplain(const Res& maybe_value,
                           testing::MatchResultListener* listener) const override {
        const Status& status = internal::GenericToStatus(maybe_value);
        testing::StringMatchResultListener value_listener;

        bool match = status.code() == code_;
        if (message_matcher_) {
          match = match &&
                  message_matcher_->MatchAndExplain(status.message(), &value_listener);
        }

        if (match) {
          *listener << "whose error matches";
        } else if (status.ok()) {
          *listener << "whose non-error doesn't match";
        } else {
          *listener << "whose error doesn't match";
        }

        testing::internal::PrintIfNotEmpty(value_listener.str(), listener->stream());
        return match;
      }

      const StatusCode code_;
      const std::optional<testing::Matcher<std::string>> message_matcher_;
    };

    return testing::Matcher<Res>(new Impl(code_, message_matcher_));
  }

 private:
  const StatusCode code_;
  const std::optional<testing::Matcher<std::string>> message_matcher_;
};

class OkMatcher {
 public:
  template <typename Res>
  operator testing::Matcher<Res>() const {  // NOLINT runtime/explicit
    struct Impl : testing::MatcherInterface<const Res&> {
      void DescribeTo(::std::ostream* os) const override { *os << "is ok"; }

      void DescribeNegationTo(::std::ostream* os) const override { *os << "is not ok"; }

      bool MatchAndExplain(const Res& maybe_value,
                           testing::MatchResultListener* listener) const override {
        const Status& status = internal::GenericToStatus(maybe_value);

        const bool match = status.ok();
        *listener << "whose " << (match ? "non-error matches" : "error doesn't match");
        return match;
      }
    };

    return testing::Matcher<Res>(new Impl());
  }
};

// Returns a matcher that waits on a Future (by default for 16 seconds)
// then applies a matcher to the result.
template <typename ResultMatcher>
FutureMatcher<ResultMatcher> Finishes(
    const ResultMatcher& result_matcher,
    double wait_seconds = kDefaultAssertFinishesWaitSeconds) {
  return FutureMatcher<ResultMatcher>(result_matcher, wait_seconds);
}

// Returns a matcher that matches the value of a successful Result<T>.
template <typename ValueMatcher>
ResultMatcher<ValueMatcher> ResultWith(const ValueMatcher& value_matcher) {
  return ResultMatcher<ValueMatcher>(value_matcher);
}

// Returns a matcher that matches an ok Status or Result<T>.
inline OkMatcher Ok() { return {}; }

// Returns a matcher that matches the StatusCode of a Status or Result<T>.
// Do not use Raises(StatusCode::OK) to match a non error code.
inline ErrorMatcher Raises(StatusCode code) { return ErrorMatcher(code, std::nullopt); }

// Returns a matcher that matches the StatusCode and message of a Status or Result<T>.
template <typename MessageMatcher>
ErrorMatcher Raises(StatusCode code, const MessageMatcher& message_matcher) {
  return ErrorMatcher(code, testing::MatcherCast<std::string>(message_matcher));
}

class DataEqMatcher {
 public:
  // TODO(bkietz) support EqualOptions, ApproxEquals, etc
  // Probably it's better to use something like config-through-key_value_metadata
  // as with the random generators to decouple this from EqualOptions etc.
  explicit DataEqMatcher(Datum expected) : expected_(std::move(expected)) {}

  template <typename Data>
  operator testing::Matcher<Data>() const {  // NOLINT runtime/explicit
    struct Impl : testing::MatcherInterface<const Data&> {
      explicit Impl(Datum expected) : expected_(std::move(expected)) {}

      void DescribeTo(::std::ostream* os) const override {
        *os << "has data ";
        PrintTo(expected_, os);
      }

      void DescribeNegationTo(::std::ostream* os) const override {
        *os << "doesn't have data ";
        PrintTo(expected_, os);
      }

      bool MatchAndExplain(const Data& data,
                           testing::MatchResultListener* listener) const override {
        Datum boxed(data);

        if (boxed.kind() != expected_.kind()) {
          *listener << "whose Datum::kind " << boxed.ToString() << " doesn't match "
                    << expected_.ToString();
          return false;
        }

        if (const auto& boxed_type = boxed.type()) {
          if (*boxed_type != *expected_.type()) {
            *listener << "whose DataType " << boxed_type->ToString() << " doesn't match "
                      << expected_.type()->ToString();
            return false;
          }
        } else if (const auto& boxed_schema = boxed.schema()) {
          if (*boxed_schema != *expected_.schema()) {
            *listener << "whose Schema " << boxed_schema->ToString() << " doesn't match "
                      << expected_.schema()->ToString();
            return false;
          }
        } else {
          Unreachable();
        }

        if (boxed == expected_) {
          *listener << "whose value matches";
          return true;
        }

        if (listener->IsInterested() && boxed.kind() == Datum::ARRAY) {
          *listener << "whose value differs from the expected value by "
                    << boxed.make_array()->Diff(*expected_.make_array());
        } else {
          *listener << "whose value doesn't match";
        }
        return false;
      }

      Datum expected_;
    };

    return testing::Matcher<Data>(new Impl(expected_));
  }

 private:
  Datum expected_;
};

/// Constructs a datum against which arguments are matched
template <typename Data>
DataEqMatcher DataEq(Data&& dat) {
  return DataEqMatcher(Datum(std::forward<Data>(dat)));
}

/// Constructs an array with ArrayFromJSON against which arguments are matched
inline DataEqMatcher DataEqArray(const std::shared_ptr<DataType>& type,
                                 std::string_view json) {
  return DataEq(ArrayFromJSON(type, json));
}

/// Constructs an array from a vector of optionals against which arguments are matched
template <typename T, typename ArrayType = typename TypeTraits<T>::ArrayType,
          typename BuilderType = typename TypeTraits<T>::BuilderType,
          typename ValueType =
              typename ::arrow::stl::detail::DefaultValueAccessor<ArrayType>::ValueType>
DataEqMatcher DataEqArray(T type, const std::vector<std::optional<ValueType>>& values) {
  // FIXME(bkietz) broken until DataType is move constructible
  BuilderType builder(std::make_shared<T>(std::move(type)), default_memory_pool());
  DCHECK_OK(builder.Reserve(static_cast<int64_t>(values.size())));

  // pseudo constexpr:
  static const bool need_safe_append = !is_fixed_width(T::type_id);

  for (auto value : values) {
    if (value) {
      if (need_safe_append) {
        builder.UnsafeAppend(*value);
      } else {
        DCHECK_OK(builder.Append(*value));
      }
    } else {
      builder.UnsafeAppendNull();
    }
  }

  return DataEq(builder.Finish().ValueOrDie());
}

/// Constructs a scalar with ScalarFromJSON against which arguments are matched
inline DataEqMatcher DataEqScalar(const std::shared_ptr<DataType>& type,
                                  std::string_view json) {
  return DataEq(ScalarFromJSON(type, json));
}

/// Constructs a scalar against which arguments are matched
template <typename T, typename ScalarType = typename TypeTraits<T>::ScalarType,
          typename ValueType = typename ScalarType::ValueType>
DataEqMatcher DataEqScalar(T type, std::optional<ValueType> value) {
  ScalarType expected(std::make_shared<T>(std::move(type)));

  if (value) {
    expected.is_valid = true;
    expected.value = std::move(*value);
  }

  return DataEq(std::move(expected));
}

// HasType, HasSchema matchers

}  // namespace arrow

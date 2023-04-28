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

#include "arrow/testing/gtest_util.h"
#include "arrow/util/future.h"

// This macro should be called by futures that are expected to
// complete pretty quickly.  arrow::kDefaultAssertFinishesWaitSeconds is the
// default max wait here.  Anything longer than that and it's a questionable unit test
// anyways.
#define ASSERT_FINISHES_IMPL(fut)                                      \
  do {                                                                 \
    ASSERT_TRUE(fut.Wait(::arrow::kDefaultAssertFinishesWaitSeconds)); \
    if (!fut.is_finished()) {                                          \
      FAIL() << "Future did not finish in a timely fashion";           \
    }                                                                  \
  } while (false)

#define ASSERT_FINISHES_OK(expr)                                              \
  do {                                                                        \
    auto&& _fut = (expr);                                                     \
    ASSERT_TRUE(_fut.Wait(::arrow::kDefaultAssertFinishesWaitSeconds));       \
    if (!_fut.is_finished()) {                                                \
      FAIL() << "Future did not finish in a timely fashion";                  \
    }                                                                         \
    auto& _st = _fut.status();                                                \
    if (!_st.ok()) {                                                          \
      FAIL() << "'" ARROW_STRINGIFY(expr) "' failed with " << _st.ToString(); \
    }                                                                         \
  } while (false)

#define ASSERT_FINISHES_AND_RAISES(ENUM, expr) \
  do {                                         \
    auto&& _fut = (expr);                      \
    ASSERT_FINISHES_IMPL(_fut);                \
    ASSERT_RAISES(ENUM, _fut.status());        \
  } while (false)

#define EXPECT_FINISHES_AND_RAISES_WITH_MESSAGE_THAT(ENUM, matcher, expr) \
  do {                                                                    \
    auto&& fut = (expr);                                                  \
    ASSERT_FINISHES_IMPL(fut);                                            \
    EXPECT_RAISES_WITH_MESSAGE_THAT(ENUM, matcher, fut.status());         \
  } while (false)

#define ASSERT_FINISHES_OK_AND_ASSIGN_IMPL(lhs, rexpr, _future_name) \
  auto _future_name = (rexpr);                                       \
  ASSERT_FINISHES_IMPL(_future_name);                                \
  ASSERT_OK_AND_ASSIGN(lhs, _future_name.result());

#define ASSERT_FINISHES_OK_AND_ASSIGN(lhs, rexpr) \
  ASSERT_FINISHES_OK_AND_ASSIGN_IMPL(lhs, rexpr,  \
                                     ARROW_ASSIGN_OR_RAISE_NAME(_fut, __COUNTER__))

#define ASSERT_FINISHES_OK_AND_EQ(expected, expr)        \
  do {                                                   \
    ASSERT_FINISHES_OK_AND_ASSIGN(auto _actual, (expr)); \
    ASSERT_EQ(expected, _actual);                        \
  } while (0)

#define EXPECT_FINISHES_IMPL(fut)                                      \
  do {                                                                 \
    EXPECT_TRUE(fut.Wait(::arrow::kDefaultAssertFinishesWaitSeconds)); \
    if (!fut.is_finished()) {                                          \
      ADD_FAILURE() << "Future did not finish in a timely fashion";    \
    }                                                                  \
  } while (false)

#define ON_FINISH_ASSIGN_OR_HANDLE_ERROR_IMPL(handle_error, future_name, lhs, rexpr) \
  auto future_name = (rexpr);                                                        \
  EXPECT_FINISHES_IMPL(future_name);                                                 \
  handle_error(future_name.status());                                                \
  EXPECT_OK_AND_ASSIGN(lhs, future_name.result());

#define EXPECT_FINISHES(expr)   \
  do {                          \
    EXPECT_FINISHES_IMPL(expr); \
  } while (0)

#define EXPECT_FINISHES_OK_AND_ASSIGN(lhs, rexpr) \
  ON_FINISH_ASSIGN_OR_HANDLE_ERROR_IMPL(          \
      ARROW_EXPECT_OK, ARROW_ASSIGN_OR_RAISE_NAME(_fut, __COUNTER__), lhs, rexpr);

#define EXPECT_FINISHES_OK_AND_EQ(expected, expr)        \
  do {                                                   \
    EXPECT_FINISHES_OK_AND_ASSIGN(auto _actual, (expr)); \
    EXPECT_EQ(expected, _actual);                        \
  } while (0)

namespace arrow {

constexpr double kDefaultAssertFinishesWaitSeconds = 64;

template <typename T>
void AssertNotFinished(const Future<T>& fut) {
  ASSERT_FALSE(IsFutureFinished(fut.state()));
}

template <typename T>
void AssertFinished(const Future<T>& fut) {
  ASSERT_TRUE(IsFutureFinished(fut.state()));
}

// Assert the future is successful *now*
template <typename T>
void AssertSuccessful(const Future<T>& fut) {
  if (IsFutureFinished(fut.state())) {
    ASSERT_EQ(fut.state(), FutureState::SUCCESS);
    ASSERT_OK(fut.status());
  } else {
    FAIL() << "Expected future to be completed successfully but it was still pending";
  }
}

// Assert the future is failed *now*
template <typename T>
void AssertFailed(const Future<T>& fut) {
  if (IsFutureFinished(fut.state())) {
    ASSERT_EQ(fut.state(), FutureState::FAILURE);
    ASSERT_FALSE(fut.status().ok());
  } else {
    FAIL() << "Expected future to have failed but it was still pending";
  }
}

}  // namespace arrow

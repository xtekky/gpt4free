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

#include <iosfwd>

#include "arrow/testing/gtest_util.h"
#include "arrow/util/iterator.h"

namespace arrow {

struct TestInt {
  TestInt();
  TestInt(int i);  // NOLINT runtime/explicit
  int value;

  bool operator==(const TestInt& other) const;

  friend std::ostream& operator<<(std::ostream& os, const TestInt& v);
};

template <>
struct IterationTraits<TestInt> {
  static TestInt End() { return TestInt(); }
  static bool IsEnd(const TestInt& val) { return val == IterationTraits<TestInt>::End(); }
};

struct TestStr {
  TestStr();
  TestStr(const std::string& s);  // NOLINT runtime/explicit
  TestStr(const char* s);         // NOLINT runtime/explicit
  explicit TestStr(const TestInt& test_int);
  std::string value;

  bool operator==(const TestStr& other) const;

  friend std::ostream& operator<<(std::ostream& os, const TestStr& v);
};

template <>
struct IterationTraits<TestStr> {
  static TestStr End() { return TestStr(); }
  static bool IsEnd(const TestStr& val) { return val == IterationTraits<TestStr>::End(); }
};

std::vector<TestInt> RangeVector(unsigned int max, unsigned int step = 1);

template <typename T>
inline Iterator<T> VectorIt(std::vector<T> v) {
  return MakeVectorIterator<T>(std::move(v));
}

template <typename T>
inline Iterator<T> PossiblySlowVectorIt(std::vector<T> v, bool slow = false) {
  auto iterator = MakeVectorIterator<T>(std::move(v));
  if (slow) {
    return MakeTransformedIterator<T, T>(std::move(iterator),
                                         [](T item) -> Result<TransformFlow<T>> {
                                           SleepABit();
                                           return TransformYield(item);
                                         });
  } else {
    return iterator;
  }
}

template <typename T>
inline void AssertIteratorExhausted(Iterator<T>& it) {
  ASSERT_OK_AND_ASSIGN(T next, it.Next());
  ASSERT_TRUE(IsIterationEnd(next));
}

Transformer<TestInt, TestStr> MakeFilter(std::function<bool(TestInt&)> filter);

}  // namespace arrow

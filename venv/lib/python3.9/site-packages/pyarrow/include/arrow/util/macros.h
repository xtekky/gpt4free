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

#include <cstdint>

#define ARROW_EXPAND(x) x
#define ARROW_STRINGIFY(x) #x
#define ARROW_CONCAT(x, y) x##y

// From Google gutil
#ifndef ARROW_DISALLOW_COPY_AND_ASSIGN
#define ARROW_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;            \
  void operator=(const TypeName&) = delete
#endif

#ifndef ARROW_DEFAULT_MOVE_AND_ASSIGN
#define ARROW_DEFAULT_MOVE_AND_ASSIGN(TypeName) \
  TypeName(TypeName&&) = default;               \
  TypeName& operator=(TypeName&&) = default
#endif

#define ARROW_UNUSED(x) (void)(x)
#define ARROW_ARG_UNUSED(x)
//
// GCC can be told that a certain branch is not likely to be taken (for
// instance, a CHECK failure), and use that information in static analysis.
// Giving it this information can help it optimize for the common case in
// the absence of better information (ie. -fprofile-arcs).
//
#if defined(__GNUC__)
#define ARROW_PREDICT_FALSE(x) (__builtin_expect(!!(x), 0))
#define ARROW_PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))
#define ARROW_NORETURN __attribute__((noreturn))
#define ARROW_NOINLINE __attribute__((noinline))
#define ARROW_PREFETCH(addr) __builtin_prefetch(addr)
#elif defined(_MSC_VER)
#define ARROW_NORETURN __declspec(noreturn)
#define ARROW_NOINLINE __declspec(noinline)
#define ARROW_PREDICT_FALSE(x) (x)
#define ARROW_PREDICT_TRUE(x) (x)
#define ARROW_PREFETCH(addr)
#else
#define ARROW_NORETURN
#define ARROW_PREDICT_FALSE(x) (x)
#define ARROW_PREDICT_TRUE(x) (x)
#define ARROW_PREFETCH(addr)
#endif

#if defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)
#define ARROW_RESTRICT __restrict
#else
#define ARROW_RESTRICT
#endif

// ----------------------------------------------------------------------
// C++/CLI support macros (see ARROW-1134)

#ifndef NULLPTR

#ifdef __cplusplus_cli
#define NULLPTR __nullptr
#else
#define NULLPTR nullptr
#endif

#endif  // ifndef NULLPTR

// ----------------------------------------------------------------------

// clang-format off
// [[deprecated]] is only available in C++14, use this for the time being
// This macro takes an optional deprecation message
#ifdef __COVERITY__
#  define ARROW_DEPRECATED(...)
#else
#  define ARROW_DEPRECATED(...) [[deprecated(__VA_ARGS__)]]
#endif

#ifdef __COVERITY__
#  define ARROW_DEPRECATED_ENUM_VALUE(...)
#else
#  define ARROW_DEPRECATED_ENUM_VALUE(...) [[deprecated(__VA_ARGS__)]]
#endif

// clang-format on

// Macros to disable deprecation warnings

#ifdef __clang__
#define ARROW_SUPPRESS_DEPRECATION_WARNING \
  _Pragma("clang diagnostic push");        \
  _Pragma("clang diagnostic ignored \"-Wdeprecated-declarations\"")
#define ARROW_UNSUPPRESS_DEPRECATION_WARNING _Pragma("clang diagnostic pop")
#elif defined(__GNUC__)
#define ARROW_SUPPRESS_DEPRECATION_WARNING \
  _Pragma("GCC diagnostic push");          \
  _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"")
#define ARROW_UNSUPPRESS_DEPRECATION_WARNING _Pragma("GCC diagnostic pop")
#elif defined(_MSC_VER)
#define ARROW_SUPPRESS_DEPRECATION_WARNING \
  __pragma(warning(push)) __pragma(warning(disable : 4996))
#define ARROW_UNSUPPRESS_DEPRECATION_WARNING __pragma(warning(pop))
#else
#define ARROW_SUPPRESS_DEPRECATION_WARNING
#define ARROW_UNSUPPRESS_DEPRECATION_WARNING
#endif

// ----------------------------------------------------------------------

// macros to disable padding
// these macros are portable across different compilers and platforms
//[https://github.com/google/flatbuffers/blob/master/include/flatbuffers/flatbuffers.h#L1355]
#if !defined(MANUALLY_ALIGNED_STRUCT)
#if defined(_MSC_VER)
#define MANUALLY_ALIGNED_STRUCT(alignment) \
  __pragma(pack(1));                       \
  struct __declspec(align(alignment))
#define STRUCT_END(name, size) \
  __pragma(pack());            \
  static_assert(sizeof(name) == size, "compiler breaks packing rules")
#elif defined(__GNUC__) || defined(__clang__)
#define MANUALLY_ALIGNED_STRUCT(alignment) \
  _Pragma("pack(1)") struct __attribute__((aligned(alignment)))
#define STRUCT_END(name, size) \
  _Pragma("pack()") static_assert(sizeof(name) == size, "compiler breaks packing rules")
#else
#error Unknown compiler, please define structure alignment macros
#endif
#endif  // !defined(MANUALLY_ALIGNED_STRUCT)

// ----------------------------------------------------------------------
// Convenience macro disabling a particular UBSan check in a function

#if defined(__clang__)
#define ARROW_DISABLE_UBSAN(feature) __attribute__((no_sanitize(feature)))
#else
#define ARROW_DISABLE_UBSAN(feature)
#endif

// ----------------------------------------------------------------------
// Machine information

#if INTPTR_MAX == INT64_MAX
#define ARROW_BITNESS 64
#elif INTPTR_MAX == INT32_MAX
#define ARROW_BITNESS 32
#else
#error Unexpected INTPTR_MAX
#endif

// ----------------------------------------------------------------------
// From googletest
// (also in parquet-cpp)

// When you need to test the private or protected members of a class,
// use the FRIEND_TEST macro to declare your tests as friends of the
// class.  For example:
//
// class MyClass {
//  private:
//   void MyMethod();
//   FRIEND_TEST(MyClassTest, MyMethod);
// };
//
// class MyClassTest : public testing::Test {
//   // ...
// };
//
// TEST_F(MyClassTest, MyMethod) {
//   // Can call MyClass::MyMethod() here.
// }

#define FRIEND_TEST(test_case_name, test_name) \
  friend class test_case_name##_##test_name##_Test

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

#include <memory>
#include <tuple>
#include <type_traits>

#include "arrow/result.h"
#include "arrow/util/macros.h"

namespace arrow {
namespace internal {

struct Empty {
  static Result<Empty> ToResult(Status s) {
    if (ARROW_PREDICT_TRUE(s.ok())) {
      return Empty{};
    }
    return s;
  }
};

/// Helper struct for examining lambdas and other callables.
/// TODO(ARROW-12655) support function pointers
struct call_traits {
 public:
  template <typename R, typename... A>
  static std::false_type is_overloaded_impl(R(A...));

  template <typename F>
  static std::false_type is_overloaded_impl(decltype(&F::operator())*);

  template <typename F>
  static std::true_type is_overloaded_impl(...);

  template <typename F, typename R, typename... A>
  static R return_type_impl(R (F::*)(A...));

  template <typename F, typename R, typename... A>
  static R return_type_impl(R (F::*)(A...) const);

  template <std::size_t I, typename F, typename R, typename... A>
  static typename std::tuple_element<I, std::tuple<A...>>::type argument_type_impl(
      R (F::*)(A...));

  template <std::size_t I, typename F, typename R, typename... A>
  static typename std::tuple_element<I, std::tuple<A...>>::type argument_type_impl(
      R (F::*)(A...) const);

  template <std::size_t I, typename F, typename R, typename... A>
  static typename std::tuple_element<I, std::tuple<A...>>::type argument_type_impl(
      R (F::*)(A...) &&);

  template <typename F, typename R, typename... A>
  static std::integral_constant<int, sizeof...(A)> argument_count_impl(R (F::*)(A...));

  template <typename F, typename R, typename... A>
  static std::integral_constant<int, sizeof...(A)> argument_count_impl(R (F::*)(A...)
                                                                           const);

  template <typename F, typename R, typename... A>
  static std::integral_constant<int, sizeof...(A)> argument_count_impl(R (F::*)(A...) &&);

  /// bool constant indicating whether F is a callable with more than one possible
  /// signature. Will be true_type for objects which define multiple operator() or which
  /// define a template operator()
  template <typename F>
  using is_overloaded =
      decltype(is_overloaded_impl<typename std::decay<F>::type>(NULLPTR));

  template <typename F, typename T = void>
  using enable_if_overloaded = typename std::enable_if<is_overloaded<F>::value, T>::type;

  template <typename F, typename T = void>
  using disable_if_overloaded =
      typename std::enable_if<!is_overloaded<F>::value, T>::type;

  /// If F is not overloaded, the argument types of its call operator can be
  /// extracted via call_traits::argument_type<Index, F>
  template <std::size_t I, typename F>
  using argument_type = decltype(argument_type_impl<I>(&std::decay<F>::type::operator()));

  template <typename F>
  using argument_count = decltype(argument_count_impl(&std::decay<F>::type::operator()));

  template <typename F>
  using return_type = decltype(return_type_impl(&std::decay<F>::type::operator()));

  template <typename F, typename T, typename RT = T>
  using enable_if_return =
      typename std::enable_if<std::is_same<return_type<F>, T>::value, RT>;

  template <typename T, typename R = void>
  using enable_if_empty = typename std::enable_if<std::is_same<T, Empty>::value, R>::type;

  template <typename T, typename R = void>
  using enable_if_not_empty =
      typename std::enable_if<!std::is_same<T, Empty>::value, R>::type;
};

/// A type erased callable object which may only be invoked once.
/// It can be constructed from any lambda which matches the provided call signature.
/// Invoking it results in destruction of the lambda, freeing any state/references
/// immediately. Invoking a default constructed FnOnce or one which has already been
/// invoked will segfault.
template <typename Signature>
class FnOnce;

template <typename R, typename... A>
class FnOnce<R(A...)> {
 public:
  FnOnce() = default;

  template <typename Fn,
            typename = typename std::enable_if<std::is_convertible<
                decltype(std::declval<Fn&&>()(std::declval<A>()...)), R>::value>::type>
  FnOnce(Fn fn) : impl_(new FnImpl<Fn>(std::move(fn))) {  // NOLINT runtime/explicit
  }

  explicit operator bool() const { return impl_ != NULLPTR; }

  R operator()(A... a) && {
    auto bye = std::move(impl_);
    return bye->invoke(std::forward<A&&>(a)...);
  }

 private:
  struct Impl {
    virtual ~Impl() = default;
    virtual R invoke(A&&... a) = 0;
  };

  template <typename Fn>
  struct FnImpl : Impl {
    explicit FnImpl(Fn fn) : fn_(std::move(fn)) {}
    R invoke(A&&... a) override { return std::move(fn_)(std::forward<A&&>(a)...); }
    Fn fn_;
  };

  std::unique_ptr<Impl> impl_;
};

}  // namespace internal
}  // namespace arrow

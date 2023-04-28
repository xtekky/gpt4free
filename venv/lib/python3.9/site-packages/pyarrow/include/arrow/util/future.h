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

#include <atomic>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type_fwd.h"
#include "arrow/type_traits.h"
#include "arrow/util/config.h"
#include "arrow/util/functional.h"
#include "arrow/util/macros.h"
#include "arrow/util/tracing.h"
#include "arrow/util/type_fwd.h"
#include "arrow/util/visibility.h"

namespace arrow {

template <typename>
struct EnsureFuture;

namespace detail {

template <typename>
struct is_future : std::false_type {};

template <typename T>
struct is_future<Future<T>> : std::true_type {};

template <typename Signature, typename Enable = void>
struct result_of;

template <typename Fn, typename... A>
struct result_of<Fn(A...),
                 internal::void_t<decltype(std::declval<Fn>()(std::declval<A>()...))>> {
  using type = decltype(std::declval<Fn>()(std::declval<A>()...));
};

template <typename Signature>
using result_of_t = typename result_of<Signature>::type;

// Helper to find the synchronous counterpart for a Future
template <typename T>
struct SyncType {
  using type = Result<T>;
};

template <>
struct SyncType<internal::Empty> {
  using type = Status;
};

template <typename Fn>
using first_arg_is_status =
    std::is_same<typename std::decay<internal::call_traits::argument_type<0, Fn>>::type,
                 Status>;

template <typename Fn, typename Then, typename Else,
          typename Count = internal::call_traits::argument_count<Fn>>
using if_has_no_args = typename std::conditional<Count::value == 0, Then, Else>::type;

/// Creates a callback that can be added to a future to mark a `dest` future finished
template <typename Source, typename Dest, bool SourceEmpty = Source::is_empty,
          bool DestEmpty = Dest::is_empty>
struct MarkNextFinished {};

/// If the source and dest are both empty we can pass on the status
template <typename Source, typename Dest>
struct MarkNextFinished<Source, Dest, true, true> {
  void operator()(const Status& status) && { next.MarkFinished(status); }
  Dest next;
};

/// If the source is not empty but the dest is then we can take the
/// status out of the result
template <typename Source, typename Dest>
struct MarkNextFinished<Source, Dest, false, true> {
  void operator()(const Result<typename Source::ValueType>& res) && {
    next.MarkFinished(internal::Empty::ToResult(res.status()));
  }
  Dest next;
};

/// If neither are empty we pass on the result
template <typename Source, typename Dest>
struct MarkNextFinished<Source, Dest, false, false> {
  void operator()(const Result<typename Source::ValueType>& res) && {
    next.MarkFinished(res);
  }
  Dest next;
};

/// Helper that contains information about how to apply a continuation
struct ContinueFuture {
  template <typename Return>
  struct ForReturnImpl;

  template <typename Return>
  using ForReturn = typename ForReturnImpl<Return>::type;

  template <typename Signature>
  using ForSignature = ForReturn<result_of_t<Signature>>;

  // If the callback returns void then we return Future<> that always finishes OK.
  template <typename ContinueFunc, typename... Args,
            typename ContinueResult = result_of_t<ContinueFunc && (Args && ...)>,
            typename NextFuture = ForReturn<ContinueResult>>
  typename std::enable_if<std::is_void<ContinueResult>::value>::type operator()(
      NextFuture next, ContinueFunc&& f, Args&&... a) const {
    std::forward<ContinueFunc>(f)(std::forward<Args>(a)...);
    next.MarkFinished();
  }

  /// If the callback returns a non-future then we return Future<T>
  /// and mark the future finished with the callback result.  It will get promoted
  /// to Result<T> as part of MarkFinished if it isn't already.
  ///
  /// If the callback returns Status and we return Future<> then also send the callback
  /// result as-is to the destination future.
  template <typename ContinueFunc, typename... Args,
            typename ContinueResult = result_of_t<ContinueFunc && (Args && ...)>,
            typename NextFuture = ForReturn<ContinueResult>>
  typename std::enable_if<
      !std::is_void<ContinueResult>::value && !is_future<ContinueResult>::value &&
      (!NextFuture::is_empty || std::is_same<ContinueResult, Status>::value)>::type
  operator()(NextFuture next, ContinueFunc&& f, Args&&... a) const {
    next.MarkFinished(std::forward<ContinueFunc>(f)(std::forward<Args>(a)...));
  }

  /// If the callback returns a Result and the next future is Future<> then we mark
  /// the future finished with the callback result.
  ///
  /// It may seem odd that the next future is Future<> when the callback returns a
  /// result but this can occur if the OnFailure callback returns a result while the
  /// OnSuccess callback is void/Status (e.g. you would get this calling the one-arg
  /// version of Then with an OnSuccess callback that returns void)
  template <typename ContinueFunc, typename... Args,
            typename ContinueResult = result_of_t<ContinueFunc && (Args && ...)>,
            typename NextFuture = ForReturn<ContinueResult>>
  typename std::enable_if<!std::is_void<ContinueResult>::value &&
                          !is_future<ContinueResult>::value && NextFuture::is_empty &&
                          !std::is_same<ContinueResult, Status>::value>::type
  operator()(NextFuture next, ContinueFunc&& f, Args&&... a) const {
    next.MarkFinished(std::forward<ContinueFunc>(f)(std::forward<Args>(a)...).status());
  }

  /// If the callback returns a Future<T> then we return Future<T>.  We create a new
  /// future and add a callback to the future given to us by the user that forwards the
  /// result to the future we just created
  template <typename ContinueFunc, typename... Args,
            typename ContinueResult = result_of_t<ContinueFunc && (Args && ...)>,
            typename NextFuture = ForReturn<ContinueResult>>
  typename std::enable_if<is_future<ContinueResult>::value>::type operator()(
      NextFuture next, ContinueFunc&& f, Args&&... a) const {
    ContinueResult signal_to_complete_next =
        std::forward<ContinueFunc>(f)(std::forward<Args>(a)...);
    MarkNextFinished<ContinueResult, NextFuture> callback{std::move(next)};
    signal_to_complete_next.AddCallback(std::move(callback));
  }

  /// Helpers to conditionally ignore arguments to ContinueFunc
  template <typename ContinueFunc, typename NextFuture, typename... Args>
  void IgnoringArgsIf(std::true_type, NextFuture&& next, ContinueFunc&& f,
                      Args&&...) const {
    operator()(std::forward<NextFuture>(next), std::forward<ContinueFunc>(f));
  }
  template <typename ContinueFunc, typename NextFuture, typename... Args>
  void IgnoringArgsIf(std::false_type, NextFuture&& next, ContinueFunc&& f,
                      Args&&... a) const {
    operator()(std::forward<NextFuture>(next), std::forward<ContinueFunc>(f),
               std::forward<Args>(a)...);
  }
};

/// Helper struct which tells us what kind of Future gets returned from `Then` based on
/// the return type of the OnSuccess callback
template <>
struct ContinueFuture::ForReturnImpl<void> {
  using type = Future<>;
};

template <>
struct ContinueFuture::ForReturnImpl<Status> {
  using type = Future<>;
};

template <typename R>
struct ContinueFuture::ForReturnImpl {
  using type = Future<R>;
};

template <typename T>
struct ContinueFuture::ForReturnImpl<Result<T>> {
  using type = Future<T>;
};

template <typename T>
struct ContinueFuture::ForReturnImpl<Future<T>> {
  using type = Future<T>;
};

}  // namespace detail

/// A Future's execution or completion status
enum class FutureState : int8_t { PENDING, SUCCESS, FAILURE };

inline bool IsFutureFinished(FutureState state) { return state != FutureState::PENDING; }

/// \brief Describe whether the callback should be scheduled or run synchronously
enum class ShouldSchedule {
  /// Always run the callback synchronously (the default)
  Never = 0,
  /// Schedule a new task only if the future is not finished when the
  /// callback is added
  IfUnfinished = 1,
  /// Always schedule the callback as a new task
  Always = 2,
  /// Schedule a new task only if it would run on an executor other than
  /// the specified executor.
  IfDifferentExecutor = 3,
};

/// \brief Options that control how a continuation is run
struct CallbackOptions {
  /// Describe whether the callback should be run synchronously or scheduled
  ShouldSchedule should_schedule = ShouldSchedule::Never;
  /// If the callback is scheduled then this is the executor it should be scheduled
  /// on.  If this is NULL then should_schedule must be Never
  internal::Executor* executor = NULLPTR;

  static CallbackOptions Defaults() { return {}; }
};

// Untyped private implementation
class ARROW_EXPORT FutureImpl : public std::enable_shared_from_this<FutureImpl> {
 public:
  FutureImpl();
  virtual ~FutureImpl() = default;

  FutureState state() { return state_.load(); }

  static std::unique_ptr<FutureImpl> Make();
  static std::unique_ptr<FutureImpl> MakeFinished(FutureState state);

#ifdef ARROW_WITH_OPENTELEMETRY
  void SetSpan(util::tracing::Span* span) { span_ = span; }
#endif

  // Future API
  void MarkFinished();
  void MarkFailed();
  void Wait();
  bool Wait(double seconds);
  template <typename ValueType>
  Result<ValueType>* CastResult() const {
    return static_cast<Result<ValueType>*>(result_.get());
  }

  using Callback = internal::FnOnce<void(const FutureImpl& impl)>;
  void AddCallback(Callback callback, CallbackOptions opts);
  bool TryAddCallback(const std::function<Callback()>& callback_factory,
                      CallbackOptions opts);

  std::atomic<FutureState> state_{FutureState::PENDING};

  // Type erased storage for arbitrary results
  // XXX small objects could be stored inline instead of boxed in a pointer
  using Storage = std::unique_ptr<void, void (*)(void*)>;
  Storage result_{NULLPTR, NULLPTR};

  struct CallbackRecord {
    Callback callback;
    CallbackOptions options;
  };
  std::vector<CallbackRecord> callbacks_;
#ifdef ARROW_WITH_OPENTELEMETRY
  util::tracing::Span* span_ = NULLPTR;
#endif
};

// ---------------------------------------------------------------------
// Public API

/// \brief EXPERIMENTAL A std::future-like class with more functionality.
///
/// A Future represents the results of a past or future computation.
/// The Future API has two sides: a producer side and a consumer side.
///
/// The producer API allows creating a Future and setting its result or
/// status, possibly after running a computation function.
///
/// The consumer API allows querying a Future's current state, wait for it
/// to complete, and composing futures with callbacks.
template <typename T>
class [[nodiscard]] Future {
 public:
  using ValueType = T;
  using SyncType = typename detail::SyncType<T>::type;
  static constexpr bool is_empty = std::is_same<T, internal::Empty>::value;
  // The default constructor creates an invalid Future.  Use Future::Make()
  // for a valid Future.  This constructor is mostly for the convenience
  // of being able to presize a vector of Futures.
  Future() = default;

#ifdef ARROW_WITH_OPENTELEMETRY
  void SetSpan(util::tracing::Span* span) { impl_->SetSpan(span); }
#endif

  // Consumer API

  bool is_valid() const { return impl_ != NULLPTR; }

  /// \brief Return the Future's current state
  ///
  /// A return value of PENDING is only indicative, as the Future can complete
  /// concurrently.  A return value of FAILURE or SUCCESS is definitive, though.
  FutureState state() const {
    CheckValid();
    return impl_->state();
  }

  /// \brief Whether the Future is finished
  ///
  /// A false return value is only indicative, as the Future can complete
  /// concurrently.  A true return value is definitive, though.
  bool is_finished() const {
    CheckValid();
    return IsFutureFinished(impl_->state());
  }

  /// \brief Wait for the Future to complete and return its Result
  const Result<ValueType>& result() const& {
    Wait();
    return *GetResult();
  }

  /// \brief Returns an rvalue to the result.  This method is potentially unsafe
  ///
  /// The future is not the unique owner of the result, copies of a future will
  /// also point to the same result.  You must make sure that no other copies
  /// of the future exist.  Attempts to add callbacks after you move the result
  /// will result in undefined behavior.
  Result<ValueType>&& MoveResult() {
    Wait();
    return std::move(*GetResult());
  }

  /// \brief Wait for the Future to complete and return its Status
  const Status& status() const { return result().status(); }

  /// \brief Future<T> is convertible to Future<>, which views only the
  /// Status of the original. Marking the returned Future Finished is not supported.
  explicit operator Future<>() const {
    Future<> status_future;
    status_future.impl_ = impl_;
    return status_future;
  }

  /// \brief Wait for the Future to complete
  void Wait() const {
    CheckValid();
    impl_->Wait();
  }

  /// \brief Wait for the Future to complete, or for the timeout to expire
  ///
  /// `true` is returned if the Future completed, `false` if the timeout expired.
  /// Note a `false` value is only indicative, as the Future can complete
  /// concurrently.
  bool Wait(double seconds) const {
    CheckValid();
    return impl_->Wait(seconds);
  }

  // Producer API

  /// \brief Producer API: mark Future finished
  ///
  /// The Future's result is set to `res`.
  void MarkFinished(Result<ValueType> res) { DoMarkFinished(std::move(res)); }

  /// \brief Mark a Future<> completed with the provided Status.
  template <typename E = ValueType, typename = typename std::enable_if<
                                        std::is_same<E, internal::Empty>::value>::type>
  void MarkFinished(Status s = Status::OK()) {
    return DoMarkFinished(E::ToResult(std::move(s)));
  }

  /// \brief Producer API: instantiate a valid Future
  ///
  /// The Future's state is initialized with PENDING.  If you are creating a future with
  /// this method you must ensure that future is eventually completed (with success or
  /// failure).  Creating a future, returning it, and never completing the future can lead
  /// to memory leaks (for example, see Loop).
  static Future Make() {
    Future fut;
    fut.impl_ = FutureImpl::Make();
    return fut;
  }

  /// \brief Producer API: instantiate a finished Future
  static Future<ValueType> MakeFinished(Result<ValueType> res) {
    Future<ValueType> fut;
    fut.InitializeFromResult(std::move(res));
    return fut;
  }

  /// \brief Make a finished Future<> with the provided Status.
  template <typename E = ValueType, typename = typename std::enable_if<
                                        std::is_same<E, internal::Empty>::value>::type>
  static Future<> MakeFinished(Status s = Status::OK()) {
    return MakeFinished(E::ToResult(std::move(s)));
  }

  struct WrapResultyOnComplete {
    template <typename OnComplete>
    struct Callback {
      void operator()(const FutureImpl& impl) && {
        std::move(on_complete)(*impl.CastResult<ValueType>());
      }
      OnComplete on_complete;
    };
  };

  struct WrapStatusyOnComplete {
    template <typename OnComplete>
    struct Callback {
      static_assert(std::is_same<internal::Empty, ValueType>::value,
                    "Only callbacks for Future<> should accept Status and not Result");

      void operator()(const FutureImpl& impl) && {
        std::move(on_complete)(impl.CastResult<ValueType>()->status());
      }
      OnComplete on_complete;
    };
  };

  template <typename OnComplete>
  using WrapOnComplete = typename std::conditional<
      detail::first_arg_is_status<OnComplete>::value, WrapStatusyOnComplete,
      WrapResultyOnComplete>::type::template Callback<OnComplete>;

  /// \brief Consumer API: Register a callback to run when this future completes
  ///
  /// The callback should receive the result of the future (const Result<T>&)
  /// For a void or statusy future this should be (const Status&)
  ///
  /// There is no guarantee to the order in which callbacks will run.  In
  /// particular, callbacks added while the future is being marked complete
  /// may be executed immediately, ahead of, or even the same time as, other
  /// callbacks that have been previously added.
  ///
  /// WARNING: callbacks may hold arbitrary references, including cyclic references.
  /// Since callbacks will only be destroyed after they are invoked, this can lead to
  /// memory leaks if a Future is never marked finished (abandoned):
  ///
  /// {
  ///     auto fut = Future<>::Make();
  ///     fut.AddCallback([fut]() {});
  /// }
  ///
  /// In this example `fut` falls out of scope but is not destroyed because it holds a
  /// cyclic reference to itself through the callback.
  template <typename OnComplete, typename Callback = WrapOnComplete<OnComplete>>
  void AddCallback(OnComplete on_complete,
                   CallbackOptions opts = CallbackOptions::Defaults()) const {
    // We know impl_ will not be dangling when invoking callbacks because at least one
    // thread will be waiting for MarkFinished to return. Thus it's safe to keep a
    // weak reference to impl_ here
    impl_->AddCallback(Callback{std::move(on_complete)}, opts);
  }

  /// \brief Overload of AddCallback that will return false instead of running
  /// synchronously
  ///
  /// This overload will guarantee the callback is never run synchronously.  If the future
  /// is already finished then it will simply return false.  This can be useful to avoid
  /// stack overflow in a situation where you have recursive Futures.  For an example
  /// see the Loop function
  ///
  /// Takes in a callback factory function to allow moving callbacks (the factory function
  /// will only be called if the callback can successfully be added)
  ///
  /// Returns true if a callback was actually added and false if the callback failed
  /// to add because the future was marked complete.
  template <typename CallbackFactory,
            typename OnComplete = detail::result_of_t<CallbackFactory()>,
            typename Callback = WrapOnComplete<OnComplete>>
  bool TryAddCallback(CallbackFactory callback_factory,
                      CallbackOptions opts = CallbackOptions::Defaults()) const {
    return impl_->TryAddCallback([&]() { return Callback{callback_factory()}; }, opts);
  }

  template <typename OnSuccess, typename OnFailure>
  struct ThenOnComplete {
    static constexpr bool has_no_args =
        internal::call_traits::argument_count<OnSuccess>::value == 0;

    using ContinuedFuture = detail::ContinueFuture::ForSignature<
        detail::if_has_no_args<OnSuccess, OnSuccess && (), OnSuccess && (const T&)>>;

    static_assert(
        std::is_same<detail::ContinueFuture::ForSignature<OnFailure && (const Status&)>,
                     ContinuedFuture>::value,
        "OnSuccess and OnFailure must continue with the same future type");

    struct DummyOnSuccess {
      void operator()(const T&);
    };
    using OnSuccessArg = typename std::decay<internal::call_traits::argument_type<
        0, detail::if_has_no_args<OnSuccess, DummyOnSuccess, OnSuccess>>>::type;

    static_assert(
        !std::is_same<OnSuccessArg, typename EnsureResult<OnSuccessArg>::type>::value,
        "OnSuccess' argument should not be a Result");

    void operator()(const Result<T>& result) && {
      detail::ContinueFuture continue_future;
      if (ARROW_PREDICT_TRUE(result.ok())) {
        // move on_failure to a(n immediately destroyed) temporary to free its resources
        ARROW_UNUSED(OnFailure(std::move(on_failure)));
        continue_future.IgnoringArgsIf(
            detail::if_has_no_args<OnSuccess, std::true_type, std::false_type>{},
            std::move(next), std::move(on_success), result.ValueOrDie());
      } else {
        ARROW_UNUSED(OnSuccess(std::move(on_success)));
        continue_future(std::move(next), std::move(on_failure), result.status());
      }
    }

    OnSuccess on_success;
    OnFailure on_failure;
    ContinuedFuture next;
  };

  template <typename OnSuccess>
  struct PassthruOnFailure {
    using ContinuedFuture = detail::ContinueFuture::ForSignature<
        detail::if_has_no_args<OnSuccess, OnSuccess && (), OnSuccess && (const T&)>>;

    Result<typename ContinuedFuture::ValueType> operator()(const Status& s) { return s; }
  };

  /// \brief Consumer API: Register a continuation to run when this future completes
  ///
  /// The continuation will run in the same thread that called MarkFinished (whatever
  /// callback is registered with this function will run before MarkFinished returns).
  /// Avoid long-running callbacks in favor of submitting a task to an Executor and
  /// returning the future.
  ///
  /// Two callbacks are supported:
  /// - OnSuccess, called with the result (const ValueType&) on successful completion.
  ///              for an empty future this will be called with nothing ()
  /// - OnFailure, called with the error (const Status&) on failed completion.
  ///              This callback is optional and defaults to a passthru of any errors.
  ///
  /// Then() returns a Future whose ValueType is derived from the return type of the
  /// callbacks. If a callback returns:
  /// - void, a Future<> will be returned which will completes successfully as soon
  ///   as the callback runs.
  /// - Status, a Future<> will be returned which will complete with the returned Status
  ///   as soon as the callback runs.
  /// - V or Result<V>, a Future<V> will be returned which will complete with the result
  ///   of invoking the callback as soon as the callback runs.
  /// - Future<V>, a Future<V> will be returned which will be marked complete when the
  ///   future returned by the callback completes (and will complete with the same
  ///   result).
  ///
  /// The continued Future type must be the same for both callbacks.
  ///
  /// Note that OnFailure can swallow errors, allowing continued Futures to successfully
  /// complete even if this Future fails.
  ///
  /// If this future is already completed then the callback will be run immediately
  /// and the returned future may already be marked complete.
  ///
  /// See AddCallback for general considerations when writing callbacks.
  template <typename OnSuccess, typename OnFailure = PassthruOnFailure<OnSuccess>,
            typename OnComplete = ThenOnComplete<OnSuccess, OnFailure>,
            typename ContinuedFuture = typename OnComplete::ContinuedFuture>
  ContinuedFuture Then(OnSuccess on_success, OnFailure on_failure = {},
                       CallbackOptions options = CallbackOptions::Defaults()) const {
    auto next = ContinuedFuture::Make();
    AddCallback(OnComplete{std::forward<OnSuccess>(on_success),
                           std::forward<OnFailure>(on_failure), next},
                options);
    return next;
  }

  /// \brief Implicit constructor to create a finished future from a value
  Future(ValueType val) : Future() {  // NOLINT runtime/explicit
    impl_ = FutureImpl::MakeFinished(FutureState::SUCCESS);
    SetResult(std::move(val));
  }

  /// \brief Implicit constructor to create a future from a Result, enabling use
  ///     of macros like ARROW_ASSIGN_OR_RAISE.
  Future(Result<ValueType> res) : Future() {  // NOLINT runtime/explicit
    if (ARROW_PREDICT_TRUE(res.ok())) {
      impl_ = FutureImpl::MakeFinished(FutureState::SUCCESS);
    } else {
      impl_ = FutureImpl::MakeFinished(FutureState::FAILURE);
    }
    SetResult(std::move(res));
  }

  /// \brief Implicit constructor to create a future from a Status, enabling use
  ///     of macros like ARROW_RETURN_NOT_OK.
  Future(Status s)  // NOLINT runtime/explicit
      : Future(Result<ValueType>(std::move(s))) {}

 protected:
  void InitializeFromResult(Result<ValueType> res) {
    if (ARROW_PREDICT_TRUE(res.ok())) {
      impl_ = FutureImpl::MakeFinished(FutureState::SUCCESS);
    } else {
      impl_ = FutureImpl::MakeFinished(FutureState::FAILURE);
    }
    SetResult(std::move(res));
  }

  void Initialize() { impl_ = FutureImpl::Make(); }

  Result<ValueType>* GetResult() const { return impl_->CastResult<ValueType>(); }

  void SetResult(Result<ValueType> res) {
    impl_->result_ = {new Result<ValueType>(std::move(res)),
                      [](void* p) { delete static_cast<Result<ValueType>*>(p); }};
  }

  void DoMarkFinished(Result<ValueType> res) {
    SetResult(std::move(res));

    if (ARROW_PREDICT_TRUE(GetResult()->ok())) {
      impl_->MarkFinished();
    } else {
      impl_->MarkFailed();
    }
  }

  void CheckValid() const {
#ifndef NDEBUG
    if (!is_valid()) {
      Status::Invalid("Invalid Future (default-initialized?)").Abort();
    }
#endif
  }

  explicit Future(std::shared_ptr<FutureImpl> impl) : impl_(std::move(impl)) {}

  std::shared_ptr<FutureImpl> impl_;

  friend struct detail::ContinueFuture;

  template <typename U>
  friend class Future;
  friend class WeakFuture<T>;

  FRIEND_TEST(FutureRefTest, ChainRemoved);
  FRIEND_TEST(FutureRefTest, TailRemoved);
  FRIEND_TEST(FutureRefTest, HeadRemoved);
};

template <typename T>
typename Future<T>::SyncType FutureToSync(const Future<T>& fut) {
  return fut.result();
}

template <>
inline typename Future<internal::Empty>::SyncType FutureToSync<internal::Empty>(
    const Future<internal::Empty>& fut) {
  return fut.status();
}

template <>
inline Future<>::Future(Status s) : Future(internal::Empty::ToResult(std::move(s))) {}

template <typename T>
class WeakFuture {
 public:
  explicit WeakFuture(const Future<T>& future) : impl_(future.impl_) {}

  Future<T> get() { return Future<T>{impl_.lock()}; }

 private:
  std::weak_ptr<FutureImpl> impl_;
};

/// \defgroup future-utilities Functions for working with Futures
/// @{

/// If a Result<Future> holds an error instead of a Future, construct a finished Future
/// holding that error.
template <typename T>
static Future<T> DeferNotOk(Result<Future<T>> maybe_future) {
  if (ARROW_PREDICT_FALSE(!maybe_future.ok())) {
    return Future<T>::MakeFinished(std::move(maybe_future).status());
  }
  return std::move(maybe_future).MoveValueUnsafe();
}

/// \brief Create a Future which completes when all of `futures` complete.
///
/// The future's result is a vector of the results of `futures`.
/// Note that this future will never be marked "failed"; failed results
/// will be stored in the result vector alongside successful results.
template <typename T>
Future<std::vector<Result<T>>> All(std::vector<Future<T>> futures) {
  struct State {
    explicit State(std::vector<Future<T>> f)
        : futures(std::move(f)), n_remaining(futures.size()) {}

    std::vector<Future<T>> futures;
    std::atomic<size_t> n_remaining;
  };

  if (futures.size() == 0) {
    return {std::vector<Result<T>>{}};
  }

  auto state = std::make_shared<State>(std::move(futures));

  auto out = Future<std::vector<Result<T>>>::Make();
  for (const Future<T>& future : state->futures) {
    future.AddCallback([state, out](const Result<T>&) mutable {
      if (state->n_remaining.fetch_sub(1) != 1) return;

      std::vector<Result<T>> results(state->futures.size());
      for (size_t i = 0; i < results.size(); ++i) {
        results[i] = state->futures[i].result();
      }
      out.MarkFinished(std::move(results));
    });
  }
  return out;
}

/// \brief Create a Future which completes when all of `futures` complete.
///
/// The future will be marked complete if all `futures` complete
/// successfully. Otherwise, it will be marked failed with the status of
/// the first failing future.
ARROW_EXPORT
Future<> AllComplete(const std::vector<Future<>>& futures);

/// \brief Create a Future which completes when all of `futures` complete.
///
/// The future will finish with an ok status if all `futures` finish with
/// an ok status. Otherwise, it will be marked failed with the status of
/// one of the failing futures.
///
/// Unlike AllComplete this Future will not complete immediately when a
/// failure occurs.  It will wait until all futures have finished.
ARROW_EXPORT
Future<> AllFinished(const std::vector<Future<>>& futures);

/// @}

struct Continue {
  template <typename T>
  operator std::optional<T>() && {  // NOLINT explicit
    return {};
  }
};

template <typename T = internal::Empty>
std::optional<T> Break(T break_value = {}) {
  return std::optional<T>{std::move(break_value)};
}

template <typename T = internal::Empty>
using ControlFlow = std::optional<T>;

/// \brief Loop through an asynchronous sequence
///
/// \param[in] iterate A generator of Future<ControlFlow<BreakValue>>. On completion
/// of each yielded future the resulting ControlFlow will be examined. A Break will
/// terminate the loop, while a Continue will re-invoke `iterate`.
///
/// \return A future which will complete when a Future returned by iterate completes with
/// a Break
template <typename Iterate,
          typename Control = typename detail::result_of_t<Iterate()>::ValueType,
          typename BreakValueType = typename Control::value_type>
Future<BreakValueType> Loop(Iterate iterate) {
  struct Callback {
    bool CheckForTermination(const Result<Control>& control_res) {
      if (!control_res.ok()) {
        break_fut.MarkFinished(control_res.status());
        return true;
      }
      if (control_res->has_value()) {
        break_fut.MarkFinished(**control_res);
        return true;
      }
      return false;
    }

    void operator()(const Result<Control>& maybe_control) && {
      if (CheckForTermination(maybe_control)) return;

      auto control_fut = iterate();
      while (true) {
        if (control_fut.TryAddCallback([this]() { return *this; })) {
          // Adding a callback succeeded; control_fut was not finished
          // and we must wait to CheckForTermination.
          return;
        }
        // Adding a callback failed; control_fut was finished and we
        // can CheckForTermination immediately. This also avoids recursion and potential
        // stack overflow.
        if (CheckForTermination(control_fut.result())) return;

        control_fut = iterate();
      }
    }

    Iterate iterate;

    // If the future returned by control_fut is never completed then we will be hanging on
    // to break_fut forever even if the listener has given up listening on it.  Instead we
    // rely on the fact that a producer (the caller of Future<>::Make) is always
    // responsible for completing the futures they create.
    // TODO: Could avoid this kind of situation with "future abandonment" similar to mesos
    Future<BreakValueType> break_fut;
  };

  auto break_fut = Future<BreakValueType>::Make();
  auto control_fut = iterate();
  control_fut.AddCallback(Callback{std::move(iterate), break_fut});

  return break_fut;
}

inline Future<> ToFuture(Status status) {
  return Future<>::MakeFinished(std::move(status));
}

template <typename T>
Future<T> ToFuture(T value) {
  return Future<T>::MakeFinished(std::move(value));
}

template <typename T>
Future<T> ToFuture(Result<T> maybe_value) {
  return Future<T>::MakeFinished(std::move(maybe_value));
}

template <typename T>
Future<T> ToFuture(Future<T> fut) {
  return std::move(fut);
}

template <typename T>
struct EnsureFuture {
  using type = decltype(ToFuture(std::declval<T>()));
};

}  // namespace arrow

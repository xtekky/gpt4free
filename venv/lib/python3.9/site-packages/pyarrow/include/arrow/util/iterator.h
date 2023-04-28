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

#include <cassert>
#include <functional>
#include <memory>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/compare.h"
#include "arrow/util/functional.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {

template <typename T>
class Iterator;

template <typename T>
struct IterationTraits {
  /// \brief a reserved value which indicates the end of iteration. By
  /// default this is NULLPTR since most iterators yield pointer types.
  /// Specialize IterationTraits if different end semantics are required.
  ///
  /// Note: This should not be used to determine if a given value is a
  /// terminal value.  Use IsIterationEnd (which uses IsEnd) instead.  This
  /// is only for returning terminal values.
  static T End() { return T(NULLPTR); }

  /// \brief Checks to see if the value is a terminal value.
  /// A method is used here since T is not neccesarily comparable in many
  /// cases even though it has a distinct final value
  static bool IsEnd(const T& val) { return val == End(); }
};

template <typename T>
T IterationEnd() {
  return IterationTraits<T>::End();
}

template <typename T>
bool IsIterationEnd(const T& val) {
  return IterationTraits<T>::IsEnd(val);
}

template <typename T>
struct IterationTraits<std::optional<T>> {
  /// \brief by default when iterating through a sequence of optional,
  /// nullopt indicates the end of iteration.
  /// Specialize IterationTraits if different end semantics are required.
  static std::optional<T> End() { return std::nullopt; }

  /// \brief by default when iterating through a sequence of optional,
  /// nullopt (!has_value()) indicates the end of iteration.
  /// Specialize IterationTraits if different end semantics are required.
  static bool IsEnd(const std::optional<T>& val) { return !val.has_value(); }

  // TODO(bkietz) The range-for loop over Iterator<optional<T>> yields
  // Result<optional<T>> which is unnecessary (since only the unyielded end optional
  // is nullopt. Add IterationTraits::GetRangeElement() to handle this case
};

/// \brief A generic Iterator that can return errors
template <typename T>
class Iterator : public util::EqualityComparable<Iterator<T>> {
 public:
  /// \brief Iterator may be constructed from any type which has a member function
  /// with signature Result<T> Next();
  /// End of iterator is signalled by returning IteratorTraits<T>::End();
  ///
  /// The argument is moved or copied to the heap and kept in a unique_ptr<void>. Only
  /// its destructor and its Next method (which are stored in function pointers) are
  /// referenced after construction.
  ///
  /// This approach is used to dodge MSVC linkage hell (ARROW-6244, ARROW-6558) when using
  /// an abstract template base class: instead of being inlined as usual for a template
  /// function the base's virtual destructor will be exported, leading to multiple
  /// definition errors when linking to any other TU where the base is instantiated.
  template <typename Wrapped>
  explicit Iterator(Wrapped has_next)
      : ptr_(new Wrapped(std::move(has_next)), Delete<Wrapped>), next_(Next<Wrapped>) {}

  Iterator() : ptr_(NULLPTR, [](void*) {}) {}

  /// \brief Return the next element of the sequence, IterationTraits<T>::End() when the
  /// iteration is completed. Calling this on a default constructed Iterator
  /// will result in undefined behavior.
  Result<T> Next() { return next_(ptr_.get()); }

  /// Pass each element of the sequence to a visitor. Will return any error status
  /// returned by the visitor, terminating iteration.
  template <typename Visitor>
  Status Visit(Visitor&& visitor) {
    for (;;) {
      ARROW_ASSIGN_OR_RAISE(auto value, Next());

      if (IsIterationEnd(value)) break;

      ARROW_RETURN_NOT_OK(visitor(std::move(value)));
    }

    return Status::OK();
  }

  /// Iterators will only compare equal if they are both null.
  /// Equality comparability is required to make an Iterator of Iterators
  /// (to check for the end condition).
  bool Equals(const Iterator& other) const { return ptr_ == other.ptr_; }

  explicit operator bool() const { return ptr_ != NULLPTR; }

  class RangeIterator {
   public:
    RangeIterator() : value_(IterationTraits<T>::End()) {}

    explicit RangeIterator(Iterator i)
        : value_(IterationTraits<T>::End()),
          iterator_(std::make_shared<Iterator>(std::move(i))) {
      Next();
    }

    bool operator!=(const RangeIterator& other) const { return value_ != other.value_; }

    RangeIterator& operator++() {
      Next();
      return *this;
    }

    Result<T> operator*() {
      ARROW_RETURN_NOT_OK(value_.status());

      auto value = std::move(value_);
      value_ = IterationTraits<T>::End();
      return value;
    }

   private:
    void Next() {
      if (!value_.ok()) {
        value_ = IterationTraits<T>::End();
        return;
      }
      value_ = iterator_->Next();
    }

    Result<T> value_;
    std::shared_ptr<Iterator> iterator_;
  };

  RangeIterator begin() { return RangeIterator(std::move(*this)); }

  RangeIterator end() { return RangeIterator(); }

  /// \brief Move every element of this iterator into a vector.
  Result<std::vector<T>> ToVector() {
    std::vector<T> out;
    for (auto maybe_element : *this) {
      ARROW_ASSIGN_OR_RAISE(auto element, maybe_element);
      out.push_back(std::move(element));
    }
    // ARROW-8193: On gcc-4.8 without the explicit move it tries to use the
    // copy constructor, which may be deleted on the elements of type T
    return std::move(out);
  }

 private:
  /// Implementation of deleter for ptr_: Casts from void* to the wrapped type and
  /// deletes that.
  template <typename HasNext>
  static void Delete(void* ptr) {
    delete static_cast<HasNext*>(ptr);
  }

  /// Implementation of Next: Casts from void* to the wrapped type and invokes that
  /// type's Next member function.
  template <typename HasNext>
  static Result<T> Next(void* ptr) {
    return static_cast<HasNext*>(ptr)->Next();
  }

  /// ptr_ is a unique_ptr to void with a custom deleter: a function pointer which first
  /// casts from void* to a pointer to the wrapped type then deletes that.
  std::unique_ptr<void, void (*)(void*)> ptr_;

  /// next_ is a function pointer which first casts from void* to a pointer to the wrapped
  /// type then invokes its Next member function.
  Result<T> (*next_)(void*) = NULLPTR;
};

template <typename T>
struct TransformFlow {
  using YieldValueType = T;

  TransformFlow(YieldValueType value, bool ready_for_next)
      : finished_(false),
        ready_for_next_(ready_for_next),
        yield_value_(std::move(value)) {}
  TransformFlow(bool finished, bool ready_for_next)
      : finished_(finished), ready_for_next_(ready_for_next), yield_value_() {}

  bool HasValue() const { return yield_value_.has_value(); }
  bool Finished() const { return finished_; }
  bool ReadyForNext() const { return ready_for_next_; }
  T Value() const { return *yield_value_; }

  bool finished_ = false;
  bool ready_for_next_ = false;
  std::optional<YieldValueType> yield_value_;
};

struct TransformFinish {
  template <typename T>
  operator TransformFlow<T>() && {  // NOLINT explicit
    return TransformFlow<T>(true, true);
  }
};

struct TransformSkip {
  template <typename T>
  operator TransformFlow<T>() && {  // NOLINT explicit
    return TransformFlow<T>(false, true);
  }
};

template <typename T>
TransformFlow<T> TransformYield(T value = {}, bool ready_for_next = true) {
  return TransformFlow<T>(std::move(value), ready_for_next);
}

template <typename T, typename V>
using Transformer = std::function<Result<TransformFlow<V>>(T)>;

template <typename T, typename V>
class TransformIterator {
 public:
  explicit TransformIterator(Iterator<T> it, Transformer<T, V> transformer)
      : it_(std::move(it)),
        transformer_(std::move(transformer)),
        last_value_(),
        finished_() {}

  Result<V> Next() {
    while (!finished_) {
      ARROW_ASSIGN_OR_RAISE(std::optional<V> next, Pump());
      if (next.has_value()) {
        return std::move(*next);
      }
      ARROW_ASSIGN_OR_RAISE(last_value_, it_.Next());
    }
    return IterationTraits<V>::End();
  }

 private:
  // Calls the transform function on the current value.  Can return in several ways
  // * If the next value is requested (e.g. skip) it will return an empty optional
  // * If an invalid status is encountered that will be returned
  // * If finished it will return IterationTraits<V>::End()
  // * If a value is returned by the transformer that will be returned
  Result<std::optional<V>> Pump() {
    if (!finished_ && last_value_.has_value()) {
      auto next_res = transformer_(*last_value_);
      if (!next_res.ok()) {
        finished_ = true;
        return next_res.status();
      }
      auto next = *next_res;
      if (next.ReadyForNext()) {
        if (IsIterationEnd(*last_value_)) {
          finished_ = true;
        }
        last_value_.reset();
      }
      if (next.Finished()) {
        finished_ = true;
      }
      if (next.HasValue()) {
        return next.Value();
      }
    }
    if (finished_) {
      return IterationTraits<V>::End();
    }
    return std::nullopt;
  }

  Iterator<T> it_;
  Transformer<T, V> transformer_;
  std::optional<T> last_value_;
  bool finished_ = false;
};

/// \brief Transforms an iterator according to a transformer, returning a new Iterator.
///
/// The transformer will be called on each element of the source iterator and for each
/// call it can yield a value, skip, or finish the iteration.  When yielding a value the
/// transformer can choose to consume the source item (the default, ready_for_next = true)
/// or to keep it and it will be called again on the same value.
///
/// This is essentially a more generic form of the map operation that can return 0, 1, or
/// many values for each of the source items.
///
/// The transformer will be exposed to the end of the source sequence
/// (IterationTraits::End) in case it needs to return some penultimate item(s).
///
/// Any invalid status returned by the transformer will be returned immediately.
template <typename T, typename V>
Iterator<V> MakeTransformedIterator(Iterator<T> it, Transformer<T, V> op) {
  return Iterator<V>(TransformIterator<T, V>(std::move(it), std::move(op)));
}

template <typename T>
struct IterationTraits<Iterator<T>> {
  // The end condition for an Iterator of Iterators is a default constructed (null)
  // Iterator.
  static Iterator<T> End() { return Iterator<T>(); }
  static bool IsEnd(const Iterator<T>& val) { return !val; }
};

template <typename Fn, typename T>
class FunctionIterator {
 public:
  explicit FunctionIterator(Fn fn) : fn_(std::move(fn)) {}

  Result<T> Next() { return fn_(); }

 private:
  Fn fn_;
};

/// \brief Construct an Iterator which invokes a callable on Next()
template <typename Fn,
          typename Ret = typename internal::call_traits::return_type<Fn>::ValueType>
Iterator<Ret> MakeFunctionIterator(Fn fn) {
  return Iterator<Ret>(FunctionIterator<Fn, Ret>(std::move(fn)));
}

template <typename T>
Iterator<T> MakeEmptyIterator() {
  return MakeFunctionIterator([]() -> Result<T> { return IterationTraits<T>::End(); });
}

template <typename T>
Iterator<T> MakeErrorIterator(Status s) {
  return MakeFunctionIterator([s]() -> Result<T> {
    ARROW_RETURN_NOT_OK(s);
    return IterationTraits<T>::End();
  });
}

/// \brief Simple iterator which yields the elements of a std::vector
template <typename T>
class VectorIterator {
 public:
  explicit VectorIterator(std::vector<T> v) : elements_(std::move(v)) {}

  Result<T> Next() {
    if (i_ == elements_.size()) {
      return IterationTraits<T>::End();
    }
    return std::move(elements_[i_++]);
  }

 private:
  std::vector<T> elements_;
  size_t i_ = 0;
};

template <typename T>
Iterator<T> MakeVectorIterator(std::vector<T> v) {
  return Iterator<T>(VectorIterator<T>(std::move(v)));
}

/// \brief Simple iterator which yields *pointers* to the elements of a std::vector<T>.
/// This is provided to support T where IterationTraits<T>::End is not specialized
template <typename T>
class VectorPointingIterator {
 public:
  explicit VectorPointingIterator(std::vector<T> v) : elements_(std::move(v)) {}

  Result<T*> Next() {
    if (i_ == elements_.size()) {
      return NULLPTR;
    }
    return &elements_[i_++];
  }

 private:
  std::vector<T> elements_;
  size_t i_ = 0;
};

template <typename T>
Iterator<T*> MakeVectorPointingIterator(std::vector<T> v) {
  return Iterator<T*>(VectorPointingIterator<T>(std::move(v)));
}

/// \brief MapIterator takes ownership of an iterator and a function to apply
/// on every element. The mapped function is not allowed to fail.
template <typename Fn, typename I, typename O>
class MapIterator {
 public:
  explicit MapIterator(Fn map, Iterator<I> it)
      : map_(std::move(map)), it_(std::move(it)) {}

  Result<O> Next() {
    ARROW_ASSIGN_OR_RAISE(I i, it_.Next());

    if (IsIterationEnd(i)) {
      return IterationTraits<O>::End();
    }

    return map_(std::move(i));
  }

 private:
  Fn map_;
  Iterator<I> it_;
};

/// \brief MapIterator takes ownership of an iterator and a function to apply
/// on every element. The mapped function is not allowed to fail.
template <typename Fn, typename From = internal::call_traits::argument_type<0, Fn>,
          typename To = internal::call_traits::return_type<Fn>>
Iterator<To> MakeMapIterator(Fn map, Iterator<From> it) {
  return Iterator<To>(MapIterator<Fn, From, To>(std::move(map), std::move(it)));
}

/// \brief Like MapIterator, but where the function can fail.
template <typename Fn, typename From = internal::call_traits::argument_type<0, Fn>,
          typename To = typename internal::call_traits::return_type<Fn>::ValueType>
Iterator<To> MakeMaybeMapIterator(Fn map, Iterator<From> it) {
  return Iterator<To>(MapIterator<Fn, From, To>(std::move(map), std::move(it)));
}

struct FilterIterator {
  enum Action { ACCEPT, REJECT };

  template <typename To>
  static Result<std::pair<To, Action>> Reject() {
    return std::make_pair(IterationTraits<To>::End(), REJECT);
  }

  template <typename To>
  static Result<std::pair<To, Action>> Accept(To out) {
    return std::make_pair(std::move(out), ACCEPT);
  }

  template <typename To>
  static Result<std::pair<To, Action>> MaybeAccept(Result<To> maybe_out) {
    return std::move(maybe_out).Map(Accept<To>);
  }

  template <typename To>
  static Result<std::pair<To, Action>> Error(Status s) {
    return s;
  }

  template <typename Fn, typename From, typename To>
  class Impl {
   public:
    explicit Impl(Fn filter, Iterator<From> it) : filter_(filter), it_(std::move(it)) {}

    Result<To> Next() {
      To out = IterationTraits<To>::End();
      Action action;

      for (;;) {
        ARROW_ASSIGN_OR_RAISE(From i, it_.Next());

        if (IsIterationEnd(i)) {
          return IterationTraits<To>::End();
        }

        ARROW_ASSIGN_OR_RAISE(std::tie(out, action), filter_(std::move(i)));

        if (action == ACCEPT) return out;
      }
    }

   private:
    Fn filter_;
    Iterator<From> it_;
  };
};

/// \brief Like MapIterator, but where the function can fail or reject elements.
template <
    typename Fn, typename From = typename internal::call_traits::argument_type<0, Fn>,
    typename Ret = typename internal::call_traits::return_type<Fn>::ValueType,
    typename To = typename std::tuple_element<0, Ret>::type,
    typename Enable = typename std::enable_if<std::is_same<
        typename std::tuple_element<1, Ret>::type, FilterIterator::Action>::value>::type>
Iterator<To> MakeFilterIterator(Fn filter, Iterator<From> it) {
  return Iterator<To>(
      FilterIterator::Impl<Fn, From, To>(std::move(filter), std::move(it)));
}

/// \brief FlattenIterator takes an iterator generating iterators and yields a
/// unified iterator that flattens/concatenates in a single stream.
template <typename T>
class FlattenIterator {
 public:
  explicit FlattenIterator(Iterator<Iterator<T>> it) : parent_(std::move(it)) {}

  Result<T> Next() {
    if (IsIterationEnd(child_)) {
      // Pop from parent's iterator.
      ARROW_ASSIGN_OR_RAISE(child_, parent_.Next());

      // Check if final iteration reached.
      if (IsIterationEnd(child_)) {
        return IterationTraits<T>::End();
      }

      return Next();
    }

    // Pop from child_ and check for depletion.
    ARROW_ASSIGN_OR_RAISE(T out, child_.Next());
    if (IsIterationEnd(out)) {
      // Reset state such that we pop from parent on the recursive call
      child_ = IterationTraits<Iterator<T>>::End();

      return Next();
    }

    return out;
  }

 private:
  Iterator<Iterator<T>> parent_;
  Iterator<T> child_ = IterationTraits<Iterator<T>>::End();
};

template <typename T>
Iterator<T> MakeFlattenIterator(Iterator<Iterator<T>> it) {
  return Iterator<T>(FlattenIterator<T>(std::move(it)));
}

template <typename Reader>
Iterator<typename Reader::ValueType> MakeIteratorFromReader(
    const std::shared_ptr<Reader>& reader) {
  return MakeFunctionIterator([reader] { return reader->Next(); });
}

}  // namespace arrow

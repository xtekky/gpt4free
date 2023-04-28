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
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

#include "arrow/util/aligned_storage.h"
#include "arrow/util/macros.h"

namespace arrow {
namespace internal {

template <typename T, size_t N, bool NonTrivialDestructor>
struct StaticVectorStorageBase {
  using storage_type = AlignedStorage<T>;

  storage_type static_data_[N];
  size_t size_ = 0;

  void destroy() noexcept {}
};

template <typename T, size_t N>
struct StaticVectorStorageBase<T, N, true> {
  using storage_type = AlignedStorage<T>;

  storage_type static_data_[N];
  size_t size_ = 0;

  ~StaticVectorStorageBase() noexcept { destroy(); }

  void destroy() noexcept { storage_type::destroy_several(static_data_, size_); }
};

template <typename T, size_t N, bool D = !std::is_trivially_destructible<T>::value>
struct StaticVectorStorage : public StaticVectorStorageBase<T, N, D> {
  using Base = StaticVectorStorageBase<T, N, D>;
  using typename Base::storage_type;

  using Base::size_;
  using Base::static_data_;

  StaticVectorStorage() noexcept = default;

  constexpr storage_type* storage_ptr() { return static_data_; }

  constexpr const storage_type* const_storage_ptr() const { return static_data_; }

  // Adjust storage size, but don't initialize any objects
  void bump_size(size_t addend) {
    assert(size_ + addend <= N);
    size_ += addend;
  }

  void ensure_capacity(size_t min_capacity) { assert(min_capacity <= N); }

  // Adjust storage size, but don't destroy any objects
  void reduce_size(size_t reduce_by) {
    assert(reduce_by <= size_);
    size_ -= reduce_by;
  }

  // Move objects from another storage, but don't destroy any objects currently
  // stored in *this.
  // You need to call destroy() first if necessary (e.g. in a
  // move assignment operator).
  void move_construct(StaticVectorStorage&& other) noexcept {
    size_ = other.size_;
    if (size_ != 0) {
      // Use a compile-time memcpy size (N) for trivial types
      storage_type::move_construct_several(other.static_data_, static_data_, size_, N);
    }
  }

  constexpr size_t capacity() const { return N; }

  constexpr size_t max_size() const { return N; }

  void reserve(size_t n) {}

  void clear() {
    storage_type::destroy_several(static_data_, size_);
    size_ = 0;
  }
};

template <typename T, size_t N>
struct SmallVectorStorage {
  using storage_type = AlignedStorage<T>;

  storage_type static_data_[N];
  size_t size_ = 0;
  storage_type* data_ = static_data_;
  size_t dynamic_capacity_ = 0;

  SmallVectorStorage() noexcept = default;

  ~SmallVectorStorage() { destroy(); }

  constexpr storage_type* storage_ptr() { return data_; }

  constexpr const storage_type* const_storage_ptr() const { return data_; }

  void bump_size(size_t addend) {
    const size_t new_size = size_ + addend;
    ensure_capacity(new_size);
    size_ = new_size;
  }

  void ensure_capacity(size_t min_capacity) {
    if (dynamic_capacity_) {
      // Grow dynamic storage if necessary
      if (min_capacity > dynamic_capacity_) {
        size_t new_capacity = std::max(dynamic_capacity_ * 2, min_capacity);
        reallocate_dynamic(new_capacity);
      }
    } else if (min_capacity > N) {
      switch_to_dynamic(min_capacity);
    }
  }

  void reduce_size(size_t reduce_by) {
    assert(reduce_by <= size_);
    size_ -= reduce_by;
  }

  void destroy() noexcept {
    storage_type::destroy_several(data_, size_);
    if (dynamic_capacity_) {
      delete[] data_;
    }
  }

  void move_construct(SmallVectorStorage&& other) noexcept {
    size_ = other.size_;
    dynamic_capacity_ = other.dynamic_capacity_;
    if (dynamic_capacity_) {
      data_ = other.data_;
      other.data_ = other.static_data_;
      other.dynamic_capacity_ = 0;
      other.size_ = 0;
    } else if (size_ != 0) {
      // Use a compile-time memcpy size (N) for trivial types
      storage_type::move_construct_several(other.static_data_, static_data_, size_, N);
    }
  }

  constexpr size_t capacity() const { return dynamic_capacity_ ? dynamic_capacity_ : N; }

  constexpr size_t max_size() const { return std::numeric_limits<size_t>::max(); }

  void reserve(size_t n) {
    if (dynamic_capacity_) {
      if (n > dynamic_capacity_) {
        reallocate_dynamic(n);
      }
    } else if (n > N) {
      switch_to_dynamic(n);
    }
  }

  void clear() {
    storage_type::destroy_several(data_, size_);
    size_ = 0;
  }

 private:
  void switch_to_dynamic(size_t new_capacity) {
    dynamic_capacity_ = new_capacity;
    data_ = new storage_type[new_capacity];
    storage_type::move_construct_several_and_destroy_source(static_data_, data_, size_);
  }

  void reallocate_dynamic(size_t new_capacity) {
    assert(new_capacity >= size_);
    auto new_data = new storage_type[new_capacity];
    storage_type::move_construct_several_and_destroy_source(data_, new_data, size_);
    delete[] data_;
    dynamic_capacity_ = new_capacity;
    data_ = new_data;
  }
};

template <typename T, size_t N, typename Storage>
class StaticVectorImpl {
 private:
  Storage storage_;

  T* data_ptr() { return storage_.storage_ptr()->get(); }

  constexpr const T* const_data_ptr() const {
    return storage_.const_storage_ptr()->get();
  }

 public:
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using iterator = T*;
  using const_iterator = const T*;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  constexpr StaticVectorImpl() noexcept = default;

  // Move and copy constructors
  StaticVectorImpl(StaticVectorImpl&& other) noexcept {
    storage_.move_construct(std::move(other.storage_));
  }

  StaticVectorImpl& operator=(StaticVectorImpl&& other) noexcept {
    if (ARROW_PREDICT_TRUE(&other != this)) {
      // TODO move_assign?
      storage_.destroy();
      storage_.move_construct(std::move(other.storage_));
    }
    return *this;
  }

  StaticVectorImpl(const StaticVectorImpl& other) {
    init_by_copying(other.storage_.size_, other.const_data_ptr());
  }

  StaticVectorImpl& operator=(const StaticVectorImpl& other) noexcept {
    if (ARROW_PREDICT_TRUE(&other != this)) {
      assign_by_copying(other.storage_.size_, other.data());
    }
    return *this;
  }

  // Automatic conversion from std::vector<T>, for convenience
  StaticVectorImpl(const std::vector<T>& other) {  // NOLINT: explicit
    init_by_copying(other.size(), other.data());
  }

  StaticVectorImpl(std::vector<T>&& other) noexcept {  // NOLINT: explicit
    init_by_moving(other.size(), other.data());
  }

  StaticVectorImpl& operator=(const std::vector<T>& other) {
    assign_by_copying(other.size(), other.data());
    return *this;
  }

  StaticVectorImpl& operator=(std::vector<T>&& other) noexcept {
    assign_by_moving(other.size(), other.data());
    return *this;
  }

  // Constructing from count and optional initialization value
  explicit StaticVectorImpl(size_t count) {
    storage_.bump_size(count);
    auto* p = storage_.storage_ptr();
    for (size_t i = 0; i < count; ++i) {
      p[i].construct();
    }
  }

  StaticVectorImpl(size_t count, const T& value) {
    storage_.bump_size(count);
    auto* p = storage_.storage_ptr();
    for (size_t i = 0; i < count; ++i) {
      p[i].construct(value);
    }
  }

  StaticVectorImpl(std::initializer_list<T> values) {
    storage_.bump_size(values.size());
    auto* p = storage_.storage_ptr();
    for (auto&& v : values) {
      // Unfortunately, cannot move initializer values
      p++->construct(v);
    }
  }

  // Size inspection

  constexpr bool empty() const { return storage_.size_ == 0; }

  constexpr size_t size() const { return storage_.size_; }

  constexpr size_t capacity() const { return storage_.capacity(); }

  constexpr size_t max_size() const { return storage_.max_size(); }

  // Data access

  T& operator[](size_t i) { return data_ptr()[i]; }

  constexpr const T& operator[](size_t i) const { return const_data_ptr()[i]; }

  T& front() { return data_ptr()[0]; }

  constexpr const T& front() const { return const_data_ptr()[0]; }

  T& back() { return data_ptr()[storage_.size_ - 1]; }

  constexpr const T& back() const { return const_data_ptr()[storage_.size_ - 1]; }

  T* data() { return data_ptr(); }

  constexpr const T* data() const { return const_data_ptr(); }

  // Iterators

  iterator begin() { return iterator(data_ptr()); }

  constexpr const_iterator begin() const { return const_iterator(const_data_ptr()); }

  constexpr const_iterator cbegin() const { return const_iterator(const_data_ptr()); }

  iterator end() { return iterator(data_ptr() + storage_.size_); }

  constexpr const_iterator end() const {
    return const_iterator(const_data_ptr() + storage_.size_);
  }

  constexpr const_iterator cend() const {
    return const_iterator(const_data_ptr() + storage_.size_);
  }

  reverse_iterator rbegin() { return reverse_iterator(end()); }

  constexpr const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }

  constexpr const_reverse_iterator crbegin() const {
    return const_reverse_iterator(end());
  }

  reverse_iterator rend() { return reverse_iterator(begin()); }

  constexpr const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }

  constexpr const_reverse_iterator crend() const {
    return const_reverse_iterator(begin());
  }

  // Mutations

  void reserve(size_t n) { storage_.reserve(n); }

  void clear() { storage_.clear(); }

  void push_back(const T& value) {
    storage_.bump_size(1);
    storage_.storage_ptr()[storage_.size_ - 1].construct(value);
  }

  void push_back(T&& value) {
    storage_.bump_size(1);
    storage_.storage_ptr()[storage_.size_ - 1].construct(std::move(value));
  }

  template <typename... Args>
  void emplace_back(Args&&... args) {
    storage_.bump_size(1);
    storage_.storage_ptr()[storage_.size_ - 1].construct(std::forward<Args>(args)...);
  }

  template <typename InputIt>
  iterator insert(const_iterator insert_at, InputIt first, InputIt last) {
    const size_t n = storage_.size_;
    const size_t it_size = static_cast<size_t>(last - first);  // XXX might be O(n)?
    const size_t pos = static_cast<size_t>(insert_at - const_data_ptr());
    storage_.bump_size(it_size);
    auto* p = storage_.storage_ptr();
    if (it_size == 0) {
      return p[pos].get();
    }
    const size_t end_pos = pos + it_size;

    // Move [pos; n) to [end_pos; end_pos + n - pos)
    size_t i = n;
    size_t j = end_pos + n - pos;
    while (j > std::max(n, end_pos)) {
      p[--j].move_construct(&p[--i]);
    }
    while (j > end_pos) {
      p[--j].move_assign(&p[--i]);
    }
    assert(j == end_pos);
    // Copy [first; last) to [pos; end_pos)
    j = pos;
    while (j < std::min(n, end_pos)) {
      p[j++].assign(*first++);
    }
    while (j < end_pos) {
      p[j++].construct(*first++);
    }
    assert(first == last);
    return p[pos].get();
  }

  void resize(size_t n) {
    const size_t old_size = storage_.size_;
    if (n > storage_.size_) {
      storage_.bump_size(n - old_size);
      auto* p = storage_.storage_ptr();
      for (size_t i = old_size; i < n; ++i) {
        p[i].construct(T{});
      }
    } else {
      auto* p = storage_.storage_ptr();
      for (size_t i = n; i < old_size; ++i) {
        p[i].destroy();
      }
      storage_.reduce_size(old_size - n);
    }
  }

  void resize(size_t n, const T& value) {
    const size_t old_size = storage_.size_;
    if (n > storage_.size_) {
      storage_.bump_size(n - old_size);
      auto* p = storage_.storage_ptr();
      for (size_t i = old_size; i < n; ++i) {
        p[i].construct(value);
      }
    } else {
      auto* p = storage_.storage_ptr();
      for (size_t i = n; i < old_size; ++i) {
        p[i].destroy();
      }
      storage_.reduce_size(old_size - n);
    }
  }

 private:
  template <typename InputIt>
  void init_by_copying(size_t n, InputIt src) {
    storage_.bump_size(n);
    auto* dest = storage_.storage_ptr();
    for (size_t i = 0; i < n; ++i, ++src) {
      dest[i].construct(*src);
    }
  }

  template <typename InputIt>
  void init_by_moving(size_t n, InputIt src) {
    init_by_copying(n, std::make_move_iterator(src));
  }

  template <typename InputIt>
  void assign_by_copying(size_t n, InputIt src) {
    const size_t old_size = storage_.size_;
    if (n > old_size) {
      storage_.bump_size(n - old_size);
      auto* dest = storage_.storage_ptr();
      for (size_t i = 0; i < old_size; ++i, ++src) {
        dest[i].assign(*src);
      }
      for (size_t i = old_size; i < n; ++i, ++src) {
        dest[i].construct(*src);
      }
    } else {
      auto* dest = storage_.storage_ptr();
      for (size_t i = 0; i < n; ++i, ++src) {
        dest[i].assign(*src);
      }
      for (size_t i = n; i < old_size; ++i) {
        dest[i].destroy();
      }
      storage_.reduce_size(old_size - n);
    }
  }

  template <typename InputIt>
  void assign_by_moving(size_t n, InputIt src) {
    assign_by_copying(n, std::make_move_iterator(src));
  }
};

template <typename T, size_t N>
using StaticVector = StaticVectorImpl<T, N, StaticVectorStorage<T, N>>;

template <typename T, size_t N>
using SmallVector = StaticVectorImpl<T, N, SmallVectorStorage<T, N>>;

}  // namespace internal
}  // namespace arrow

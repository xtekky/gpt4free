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
#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "arrow/memory_pool.h"
#include "arrow/type_fwd.h"
#include "arrow/util/macros.h"

namespace arrow {
namespace stl {

/// \brief A STL allocator delegating allocations to a Arrow MemoryPool
template <class T>
class allocator {
 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  template <class U>
  struct rebind {
    using other = allocator<U>;
  };

  /// \brief Construct an allocator from the default MemoryPool
  allocator() noexcept : pool_(default_memory_pool()) {}
  /// \brief Construct an allocator from the given MemoryPool
  explicit allocator(MemoryPool* pool) noexcept : pool_(pool) {}

  template <class U>
  allocator(const allocator<U>& rhs) noexcept : pool_(rhs.pool()) {}

  ~allocator() { pool_ = NULLPTR; }

  pointer address(reference r) const noexcept { return std::addressof(r); }

  const_pointer address(const_reference r) const noexcept { return std::addressof(r); }

  pointer allocate(size_type n, const void* /*hint*/ = NULLPTR) {
    uint8_t* data;
    Status s = pool_->Allocate(n * sizeof(T), &data);
    if (!s.ok()) throw std::bad_alloc();
    return reinterpret_cast<pointer>(data);
  }

  void deallocate(pointer p, size_type n) {
    pool_->Free(reinterpret_cast<uint8_t*>(p), n * sizeof(T));
  }

  size_type size_max() const noexcept { return size_type(-1) / sizeof(T); }

  template <class U, class... Args>
  void construct(U* p, Args&&... args) {
    new (reinterpret_cast<void*>(p)) U(std::forward<Args>(args)...);
  }

  template <class U>
  void destroy(U* p) {
    p->~U();
  }

  MemoryPool* pool() const noexcept { return pool_; }

 private:
  MemoryPool* pool_;
};

/// \brief A MemoryPool implementation delegating allocations to a STL allocator
///
/// Note that STL allocators don't provide a resizing operation, and therefore
/// any buffer resizes will do a full reallocation and copy.
template <typename Allocator = std::allocator<uint8_t>>
class STLMemoryPool : public MemoryPool {
 public:
  /// \brief Construct a memory pool from the given allocator
  explicit STLMemoryPool(const Allocator& alloc) : alloc_(alloc) {}

  using MemoryPool::Allocate;
  using MemoryPool::Free;
  using MemoryPool::Reallocate;

  Status Allocate(int64_t size, int64_t /*alignment*/, uint8_t** out) override {
    try {
      *out = alloc_.allocate(size);
    } catch (std::bad_alloc& e) {
      return Status::OutOfMemory(e.what());
    }
    stats_.UpdateAllocatedBytes(size);
    return Status::OK();
  }

  Status Reallocate(int64_t old_size, int64_t new_size, int64_t /*alignment*/,
                    uint8_t** ptr) override {
    uint8_t* old_ptr = *ptr;
    try {
      *ptr = alloc_.allocate(new_size);
    } catch (std::bad_alloc& e) {
      return Status::OutOfMemory(e.what());
    }
    memcpy(*ptr, old_ptr, std::min(old_size, new_size));
    alloc_.deallocate(old_ptr, old_size);
    stats_.UpdateAllocatedBytes(new_size - old_size);
    return Status::OK();
  }

  void Free(uint8_t* buffer, int64_t size, int64_t /*alignment*/) override {
    alloc_.deallocate(buffer, size);
    stats_.UpdateAllocatedBytes(-size);
  }

  int64_t bytes_allocated() const override { return stats_.bytes_allocated(); }

  int64_t max_memory() const override { return stats_.max_memory(); }

  std::string backend_name() const override { return "stl"; }

 private:
  Allocator alloc_;
  arrow::internal::MemoryPoolStats stats_;
};

template <class T1, class T2>
bool operator==(const allocator<T1>& lhs, const allocator<T2>& rhs) noexcept {
  return lhs.pool() == rhs.pool();
}

template <class T1, class T2>
bool operator!=(const allocator<T1>& lhs, const allocator<T2>& rhs) noexcept {
  return !(lhs == rhs);
}

}  // namespace stl
}  // namespace arrow

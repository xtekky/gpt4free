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
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type_fwd.h"
#include "arrow/util/visibility.h"

namespace arrow {

namespace internal {

///////////////////////////////////////////////////////////////////////
// Helper tracking memory statistics

class MemoryPoolStats {
 public:
  MemoryPoolStats() : bytes_allocated_(0), max_memory_(0) {}

  int64_t max_memory() const { return max_memory_.load(); }

  int64_t bytes_allocated() const { return bytes_allocated_.load(); }

  inline void UpdateAllocatedBytes(int64_t diff) {
    auto allocated = bytes_allocated_.fetch_add(diff) + diff;
    // "maximum" allocated memory is ill-defined in multi-threaded code,
    // so don't try to be too rigorous here
    if (diff > 0 && allocated > max_memory_) {
      max_memory_ = allocated;
    }
  }

 protected:
  std::atomic<int64_t> bytes_allocated_;
  std::atomic<int64_t> max_memory_;
};

}  // namespace internal

/// Base class for memory allocation on the CPU.
///
/// Besides tracking the number of allocated bytes, the allocator also should
/// take care of the required 64-byte alignment.
class ARROW_EXPORT MemoryPool {
 public:
  virtual ~MemoryPool() = default;

  /// \brief EXPERIMENTAL. Create a new instance of the default MemoryPool
  static std::unique_ptr<MemoryPool> CreateDefault();

  /// Allocate a new memory region of at least size bytes.
  ///
  /// The allocated region shall be 64-byte aligned.
  Status Allocate(int64_t size, uint8_t** out) {
    return Allocate(size, kDefaultBufferAlignment, out);
  }

  /// Allocate a new memory region of at least size bytes aligned to alignment.
  virtual Status Allocate(int64_t size, int64_t alignment, uint8_t** out) = 0;

  /// Resize an already allocated memory section.
  ///
  /// As by default most default allocators on a platform don't support aligned
  /// reallocation, this function can involve a copy of the underlying data.
  virtual Status Reallocate(int64_t old_size, int64_t new_size, int64_t alignment,
                            uint8_t** ptr) = 0;
  Status Reallocate(int64_t old_size, int64_t new_size, uint8_t** ptr) {
    return Reallocate(old_size, new_size, kDefaultBufferAlignment, ptr);
  }

  /// Free an allocated region.
  ///
  /// @param buffer Pointer to the start of the allocated memory region
  /// @param size Allocated size located at buffer. An allocator implementation
  ///   may use this for tracking the amount of allocated bytes as well as for
  ///   faster deallocation if supported by its backend.
  /// @param alignment The alignment of the allocation. Defaults to 64 bytes.
  virtual void Free(uint8_t* buffer, int64_t size, int64_t alignment) = 0;
  void Free(uint8_t* buffer, int64_t size) {
    Free(buffer, size, kDefaultBufferAlignment);
  }

  /// Return unused memory to the OS
  ///
  /// Only applies to allocators that hold onto unused memory.  This will be
  /// best effort, a memory pool may not implement this feature or may be
  /// unable to fulfill the request due to fragmentation.
  virtual void ReleaseUnused() {}

  /// The number of bytes that were allocated and not yet free'd through
  /// this allocator.
  virtual int64_t bytes_allocated() const = 0;

  /// Return peak memory allocation in this memory pool
  ///
  /// \return Maximum bytes allocated. If not known (or not implemented),
  /// returns -1
  virtual int64_t max_memory() const;

  /// The name of the backend used by this MemoryPool (e.g. "system" or "jemalloc").
  virtual std::string backend_name() const = 0;

 protected:
  MemoryPool() = default;
};

class ARROW_EXPORT LoggingMemoryPool : public MemoryPool {
 public:
  explicit LoggingMemoryPool(MemoryPool* pool);
  ~LoggingMemoryPool() override = default;

  using MemoryPool::Allocate;
  using MemoryPool::Free;
  using MemoryPool::Reallocate;

  Status Allocate(int64_t size, int64_t alignment, uint8_t** out) override;
  Status Reallocate(int64_t old_size, int64_t new_size, int64_t alignment,
                    uint8_t** ptr) override;
  void Free(uint8_t* buffer, int64_t size, int64_t alignment) override;

  int64_t bytes_allocated() const override;

  int64_t max_memory() const override;

  std::string backend_name() const override;

 private:
  MemoryPool* pool_;
};

/// Derived class for memory allocation.
///
/// Tracks the number of bytes and maximum memory allocated through its direct
/// calls. Actual allocation is delegated to MemoryPool class.
class ARROW_EXPORT ProxyMemoryPool : public MemoryPool {
 public:
  explicit ProxyMemoryPool(MemoryPool* pool);
  ~ProxyMemoryPool() override;

  using MemoryPool::Allocate;
  using MemoryPool::Free;
  using MemoryPool::Reallocate;

  Status Allocate(int64_t size, int64_t alignment, uint8_t** out) override;
  Status Reallocate(int64_t old_size, int64_t new_size, int64_t alignment,
                    uint8_t** ptr) override;
  void Free(uint8_t* buffer, int64_t size, int64_t alignment) override;

  int64_t bytes_allocated() const override;

  int64_t max_memory() const override;

  std::string backend_name() const override;

 private:
  class ProxyMemoryPoolImpl;
  std::unique_ptr<ProxyMemoryPoolImpl> impl_;
};

/// \brief Return a process-wide memory pool based on the system allocator.
ARROW_EXPORT MemoryPool* system_memory_pool();

/// \brief Return a process-wide memory pool based on jemalloc.
///
/// May return NotImplemented if jemalloc is not available.
ARROW_EXPORT Status jemalloc_memory_pool(MemoryPool** out);

/// \brief Set jemalloc memory page purging behavior for future-created arenas
/// to the indicated number of milliseconds. See dirty_decay_ms and
/// muzzy_decay_ms options in jemalloc for a description of what these do. The
/// default is configured to 1000 (1 second) which releases memory more
/// aggressively to the operating system than the jemalloc default of 10
/// seconds. If you set the value to 0, dirty / muzzy pages will be released
/// immediately rather than with a time decay, but this may reduce application
/// performance.
ARROW_EXPORT
Status jemalloc_set_decay_ms(int ms);

/// \brief Get basic statistics from jemalloc's mallctl.
/// See the MALLCTL NAMESPACE section in jemalloc project documentation for
/// available stats.
ARROW_EXPORT
Result<int64_t> jemalloc_get_stat(const char* name);

/// \brief Reset the counter for peak bytes allocated in the calling thread to zero.
/// This affects subsequent calls to thread.peak.read, but not the values returned by
/// thread.allocated or thread.deallocated.
ARROW_EXPORT
Status jemalloc_peak_reset();

/// \brief Print summary statistics in human-readable form to stderr.
/// See malloc_stats_print documentation in jemalloc project documentation for
/// available opt flags.
ARROW_EXPORT
Status jemalloc_stats_print(const char* opts = "");

/// \brief Print summary statistics in human-readable form using a callback
/// See malloc_stats_print documentation in jemalloc project documentation for
/// available opt flags.
ARROW_EXPORT
Status jemalloc_stats_print(std::function<void(const char*)> write_cb,
                            const char* opts = "");

/// \brief Get summary statistics in human-readable form.
/// See malloc_stats_print documentation in jemalloc project documentation for
/// available opt flags.
ARROW_EXPORT
Result<std::string> jemalloc_stats_string(const char* opts = "");

/// \brief Return a process-wide memory pool based on mimalloc.
///
/// May return NotImplemented if mimalloc is not available.
ARROW_EXPORT Status mimalloc_memory_pool(MemoryPool** out);

/// \brief Return the names of the backends supported by this Arrow build.
ARROW_EXPORT std::vector<std::string> SupportedMemoryBackendNames();

}  // namespace arrow

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

#include "arrow/io/interfaces.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace io {
namespace internal {

template <class LockType>
class SharedLockGuard {
 public:
  explicit SharedLockGuard(LockType* lock) : lock_(lock) { lock_->LockShared(); }

  ~SharedLockGuard() { lock_->UnlockShared(); }

 protected:
  LockType* lock_;
};

template <class LockType>
class ExclusiveLockGuard {
 public:
  explicit ExclusiveLockGuard(LockType* lock) : lock_(lock) { lock_->LockExclusive(); }

  ~ExclusiveLockGuard() { lock_->UnlockExclusive(); }

 protected:
  LockType* lock_;
};

// Debug concurrency checker that marks "shared" and "exclusive" code sections,
// aborting if the concurrency rules get violated.  Does nothing in release mode.
// Note that we intentionally use the same class declaration in debug and
// release builds in order to avoid runtime failures when e.g. loading a
// release-built DLL with a debug-built application, or the reverse.

class ARROW_EXPORT SharedExclusiveChecker {
 public:
  SharedExclusiveChecker();
  void LockShared();
  void UnlockShared();
  void LockExclusive();
  void UnlockExclusive();

  SharedLockGuard<SharedExclusiveChecker> shared_guard() {
    return SharedLockGuard<SharedExclusiveChecker>(this);
  }

  ExclusiveLockGuard<SharedExclusiveChecker> exclusive_guard() {
    return ExclusiveLockGuard<SharedExclusiveChecker>(this);
  }

 protected:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

// Concurrency wrappers for IO classes that check the correctness of
// concurrent calls to various methods.  It is not necessary to wrap all
// IO classes with these, only a few core classes that get used in tests.
//
// We're not using virtual inheritance here as virtual bases have poorly
// understood semantic overhead which we'd be passing on to implementers
// and users of these interfaces.  Instead, we just duplicate the method
// wrappers between those two classes.

template <class Derived>
class ARROW_EXPORT InputStreamConcurrencyWrapper : public InputStream {
 public:
  Status Close() final {
    auto guard = lock_.exclusive_guard();
    return derived()->DoClose();
  }

  Status Abort() final {
    auto guard = lock_.exclusive_guard();
    return derived()->DoAbort();
  }

  Result<int64_t> Tell() const final {
    auto guard = lock_.exclusive_guard();
    return derived()->DoTell();
  }

  Result<int64_t> Read(int64_t nbytes, void* out) final {
    auto guard = lock_.exclusive_guard();
    return derived()->DoRead(nbytes, out);
  }

  Result<std::shared_ptr<Buffer>> Read(int64_t nbytes) final {
    auto guard = lock_.exclusive_guard();
    return derived()->DoRead(nbytes);
  }

  Result<std::string_view> Peek(int64_t nbytes) final {
    auto guard = lock_.exclusive_guard();
    return derived()->DoPeek(nbytes);
  }

  /*
  Methods to implement in derived class:

  Status DoClose();
  Result<int64_t> DoTell() const;
  Result<int64_t> DoRead(int64_t nbytes, void* out);
  Result<std::shared_ptr<Buffer>> DoRead(int64_t nbytes);

  And optionally:

  Status DoAbort() override;
  Result<std::string_view> DoPeek(int64_t nbytes) override;

  These methods should be protected in the derived class and
  InputStreamConcurrencyWrapper declared as a friend with

  friend InputStreamConcurrencyWrapper<derived>;
  */

 protected:
  // Default implementations.  They are virtual because the derived class may
  // have derived classes itself.
  virtual Status DoAbort() { return derived()->DoClose(); }

  virtual Result<std::string_view> DoPeek(int64_t ARROW_ARG_UNUSED(nbytes)) {
    return Status::NotImplemented("Peek not implemented");
  }

  Derived* derived() { return ::arrow::internal::checked_cast<Derived*>(this); }

  const Derived* derived() const {
    return ::arrow::internal::checked_cast<const Derived*>(this);
  }

  mutable SharedExclusiveChecker lock_;
};

template <class Derived>
class ARROW_EXPORT RandomAccessFileConcurrencyWrapper : public RandomAccessFile {
 public:
  Status Close() final {
    auto guard = lock_.exclusive_guard();
    return derived()->DoClose();
  }

  Status Abort() final {
    auto guard = lock_.exclusive_guard();
    return derived()->DoAbort();
  }

  Result<int64_t> Tell() const final {
    auto guard = lock_.exclusive_guard();
    return derived()->DoTell();
  }

  Result<int64_t> Read(int64_t nbytes, void* out) final {
    auto guard = lock_.exclusive_guard();
    return derived()->DoRead(nbytes, out);
  }

  Result<std::shared_ptr<Buffer>> Read(int64_t nbytes) final {
    auto guard = lock_.exclusive_guard();
    return derived()->DoRead(nbytes);
  }

  Result<std::string_view> Peek(int64_t nbytes) final {
    auto guard = lock_.exclusive_guard();
    return derived()->DoPeek(nbytes);
  }

  Status Seek(int64_t position) final {
    auto guard = lock_.exclusive_guard();
    return derived()->DoSeek(position);
  }

  Result<int64_t> GetSize() final {
    auto guard = lock_.shared_guard();
    return derived()->DoGetSize();
  }

  // NOTE: ReadAt doesn't use stream pointer, but it is allowed to update it
  // (it's the case on Windows when using ReadFileEx).
  // So any method that relies on the current position (even if it doesn't
  // update it, such as Peek) cannot run in parallel with ReadAt and has
  // to use the exclusive_guard.

  Result<int64_t> ReadAt(int64_t position, int64_t nbytes, void* out) final {
    auto guard = lock_.shared_guard();
    return derived()->DoReadAt(position, nbytes, out);
  }

  Result<std::shared_ptr<Buffer>> ReadAt(int64_t position, int64_t nbytes) final {
    auto guard = lock_.shared_guard();
    return derived()->DoReadAt(position, nbytes);
  }

  /*
  Methods to implement in derived class:

  Status DoClose();
  Result<int64_t> DoTell() const;
  Result<int64_t> DoRead(int64_t nbytes, void* out);
  Result<std::shared_ptr<Buffer>> DoRead(int64_t nbytes);
  Status DoSeek(int64_t position);
  Result<int64_t> DoGetSize()
  Result<int64_t> DoReadAt(int64_t position, int64_t nbytes, void* out);
  Result<std::shared_ptr<Buffer>> DoReadAt(int64_t position, int64_t nbytes);

  And optionally:

  Status DoAbort() override;
  Result<std::string_view> DoPeek(int64_t nbytes) override;

  These methods should be protected in the derived class and
  RandomAccessFileConcurrencyWrapper declared as a friend with

  friend RandomAccessFileConcurrencyWrapper<derived>;
  */

 protected:
  // Default implementations.  They are virtual because the derived class may
  // have derived classes itself.
  virtual Status DoAbort() { return derived()->DoClose(); }

  virtual Result<std::string_view> DoPeek(int64_t ARROW_ARG_UNUSED(nbytes)) {
    return Status::NotImplemented("Peek not implemented");
  }

  Derived* derived() { return ::arrow::internal::checked_cast<Derived*>(this); }

  const Derived* derived() const {
    return ::arrow::internal::checked_cast<const Derived*>(this);
  }

  mutable SharedExclusiveChecker lock_;
};

}  // namespace internal
}  // namespace io
}  // namespace arrow

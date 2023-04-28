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

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "arrow/status.h"
#include "arrow/type_fwd.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {

class StopToken;

struct StopSourceImpl;

/// EXPERIMENTAL
class ARROW_EXPORT StopSource {
 public:
  StopSource();
  ~StopSource();

  // Consumer API (the side that stops)
  void RequestStop();
  void RequestStop(Status error);
  // Async-signal-safe. TODO Deprecate this?
  void RequestStopFromSignal(int signum);

  StopToken token();

  // For internal use only
  void Reset();

 protected:
  std::shared_ptr<StopSourceImpl> impl_;
};

/// EXPERIMENTAL
class ARROW_EXPORT StopToken {
 public:
  // Public for Cython
  StopToken() {}

  explicit StopToken(std::shared_ptr<StopSourceImpl> impl) : impl_(std::move(impl)) {}

  // A trivial token that never propagates any stop request
  static StopToken Unstoppable() { return StopToken(); }

  /// \brief Check if the stop source has been cancelled.
  ///
  /// Producers should call this method, whenever convenient, to check and
  /// see if they should stop producing early (i.e. have been cancelled).
  /// Failure to call this method often enough will lead to an unresponsive
  /// cancellation.
  ///
  /// This is part of the producer API (the side that gets asked to stop)
  /// This method is thread-safe
  ///
  /// \return An OK status if the stop source has not been cancelled or a
  ///         cancel error if the source has been cancelled.
  Status Poll() const;
  bool IsStopRequested() const;

 protected:
  std::shared_ptr<StopSourceImpl> impl_;
};

/// EXPERIMENTAL: Set a global StopSource that can receive signals
///
/// The only allowed order of calls is the following:
/// - SetSignalStopSource()
/// - any number of pairs of (RegisterCancellingSignalHandler,
///   UnregisterCancellingSignalHandler) calls
/// - ResetSignalStopSource()
///
/// Beware that these settings are process-wide.  Typically, only one
/// thread should call these APIs, even in a multithreaded setting.
ARROW_EXPORT
Result<StopSource*> SetSignalStopSource();

/// EXPERIMENTAL: Reset the global signal-receiving StopSource
///
/// This will invalidate the pointer returned by SetSignalStopSource.
ARROW_EXPORT
void ResetSignalStopSource();

/// EXPERIMENTAL: Register signal handler triggering the signal-receiving StopSource
///
/// Note that those handlers are automatically un-registered in a fork()ed process,
/// therefore the child process will need to call RegisterCancellingSignalHandler()
/// if desired.
ARROW_EXPORT
Status RegisterCancellingSignalHandler(const std::vector<int>& signals);

/// EXPERIMENTAL: Unregister signal handler set up by RegisterCancellingSignalHandler
ARROW_EXPORT
void UnregisterCancellingSignalHandler();

}  // namespace arrow

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
#include <unordered_map>

struct aeEventLoop;

namespace plasma {

// The constants below are defined using hardcoded values taken from ae.h so
// that ae.h does not need to be included in this file.

/// Constant specifying that the timer is done and it will be removed.
constexpr int kEventLoopTimerDone = -1;  // AE_NOMORE

/// A successful status.
constexpr int kEventLoopOk = 0;  // AE_OK

/// Read event on the file descriptor.
constexpr int kEventLoopRead = 1;  // AE_READABLE

/// Write event on the file descriptor.
constexpr int kEventLoopWrite = 2;  // AE_WRITABLE

typedef long long TimerID;  // NOLINT

class EventLoop {
 public:
  // Signature of the handler that will be called when there is a new event
  // on the file descriptor that this handler has been registered for.
  //
  // The arguments are the event flags (read or write).
  using FileCallback = std::function<void(int)>;

  // This handler will be called when a timer times out. The timer id is
  // passed as an argument. The return is the number of milliseconds the timer
  // shall be reset to or kEventLoopTimerDone if the timer shall not be
  // triggered again.
  using TimerCallback = std::function<int(int64_t)>;

  EventLoop();

  ~EventLoop();

  /// Add a new file event handler to the event loop.
  ///
  /// \param fd The file descriptor we are listening to.
  /// \param events The flags for events we are listening to (read or write).
  /// \param callback The callback that will be called when the event happens.
  /// \return Returns true if the event handler was added successfully.
  bool AddFileEvent(int fd, int events, const FileCallback& callback);

  /// Remove a file event handler from the event loop.
  ///
  /// \param fd The file descriptor of the event handler.
  void RemoveFileEvent(int fd);

  /// Register a handler that will be called after a time slice of
  /// "timeout" milliseconds.
  ///
  /// \param timeout The timeout in milliseconds.
  /// \param callback The callback for the timeout.
  /// \return The ID of the newly created timer.
  int64_t AddTimer(int64_t timeout, const TimerCallback& callback);

  /// Remove a timer handler from the event loop.
  ///
  /// \param timer_id The ID of the timer that is to be removed.
  /// \return The ae.c error code. TODO(pcm): needs to be standardized
  int RemoveTimer(int64_t timer_id);

  /// \brief Run the event loop.
  void Start();

  /// \brief Stop the event loop
  void Stop();

  void Shutdown();

 private:
  static void FileEventCallback(aeEventLoop* loop, int fd, void* context, int events);

  static int TimerEventCallback(aeEventLoop* loop, TimerID timer_id, void* context);

  aeEventLoop* loop_;
  std::unordered_map<int, std::unique_ptr<FileCallback>> file_callbacks_;
  std::unordered_map<int64_t, std::unique_ptr<TimerCallback>> timer_callbacks_;
};

}  // namespace plasma

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

// Interfaces for defining middleware for Flight clients. Currently
// experimental.

#pragma once

#include <memory>

#include "arrow/flight/middleware.h"
#include "arrow/flight/visibility.h"  // IWYU pragma: keep
#include "arrow/status.h"

namespace arrow {
namespace flight {

/// \brief Client-side middleware for a call, instantiated per RPC.
///
/// Middleware should be fast and must be infallible: there is no way
/// to reject the call or report errors from the middleware instance.
class ARROW_FLIGHT_EXPORT ClientMiddleware {
 public:
  virtual ~ClientMiddleware() = default;

  /// \brief A callback before headers are sent. Extra headers can be
  /// added, but existing ones cannot be read.
  virtual void SendingHeaders(AddCallHeaders* outgoing_headers) = 0;

  /// \brief A callback when headers are received from the server.
  virtual void ReceivedHeaders(const CallHeaders& incoming_headers) = 0;

  /// \brief A callback after the call has completed.
  virtual void CallCompleted(const Status& status) = 0;
};

/// \brief A factory for new middleware instances.
///
/// If added to a client, this will be called for each RPC (including
/// Handshake) to give the opportunity to intercept the call.
///
/// It is guaranteed that all client middleware methods are called
/// from the same thread that calls the RPC method implementation.
class ARROW_FLIGHT_EXPORT ClientMiddlewareFactory {
 public:
  virtual ~ClientMiddlewareFactory() = default;

  /// \brief A callback for the start of a new call.
  ///
  /// \param info Information about the call.
  /// \param[out] middleware The middleware instance for this call. If
  ///     unset, will not add middleware to this call instance from
  ///     this factory.
  virtual void StartCall(const CallInfo& info,
                         std::unique_ptr<ClientMiddleware>* middleware) = 0;
};

}  // namespace flight
}  // namespace arrow

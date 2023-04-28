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

#include <chrono>
#include <memory>

#include "arrow/flight/transport.h"
#include "arrow/flight/type_fwd.h"
#include "arrow/flight/visibility.h"
#include "arrow/type_fwd.h"

namespace arrow {
namespace ipc {
class Message;
}
namespace flight {
namespace internal {

/// \brief A transport-specific interface for reading/writing Arrow
///   data for a server.
class ARROW_FLIGHT_EXPORT ServerDataStream : public TransportDataStream {
 public:
  /// \brief Attempt to write a non-data message.
  ///
  /// Only implemented for DoPut; mutually exclusive with
  /// WriteData(const FlightPayload&).
  virtual Status WritePutMetadata(const Buffer& payload);
};

/// \brief An implementation of a Flight server for a particular
/// transport.
///
/// This class (the transport implementation) implements the underlying
/// server and handles connections/incoming RPC calls. It should forward RPC
/// calls to the RPC handlers defined on this class, which work in terms of
/// the generic interfaces above. The RPC handlers here then forward calls
/// to the underlying FlightServerBase instance that contains the actual
/// application RPC method handlers.
///
/// Used by FlightServerBase to manage the server lifecycle.
class ARROW_FLIGHT_EXPORT ServerTransport {
 public:
  ServerTransport(FlightServerBase* base, std::shared_ptr<MemoryManager> memory_manager)
      : base_(base), memory_manager_(std::move(memory_manager)) {}
  virtual ~ServerTransport() = default;

  /// \name Server Lifecycle Methods
  /// Transports implement these methods to start/shutdown the underlying
  /// server.
  /// @{
  /// \brief Initialize the server.
  ///
  /// This method should launch the server in a background thread, i.e. it
  /// should not block. Once this returns, the server should be active.
  virtual Status Init(const FlightServerOptions& options,
                      const arrow::internal::Uri& uri) = 0;
  /// \brief Shutdown the server.
  ///
  /// This should wait for active RPCs to finish. Once this returns, the
  /// server is no longer listening.
  virtual Status Shutdown() = 0;
  /// \brief Shutdown the server with a deadline.
  ///
  /// This should wait for active RPCs to finish, or for the deadline to
  /// expire. Once this returns, the server is no longer listening.
  virtual Status Shutdown(const std::chrono::system_clock::time_point& deadline) = 0;
  /// \brief Wait for the server to shutdown (but do not shut down the server).
  ///
  /// Once this returns, the server is no longer listening.
  virtual Status Wait() = 0;
  /// \brief Get the address the server is listening on, else an empty Location.
  virtual Location location() const = 0;
  ///@}

  /// \name RPC Handlers
  /// Implementations of RPC handlers for Flight methods using the common
  /// interfaces here. Transports should call these methods from their
  /// server implementation to handle the actual RPC calls.
  ///@{
  /// \brief Get the FlightServerBase.
  ///
  /// Intended as an escape hatch for now since not all methods have been
  /// factored into a transport-agnostic interface.
  FlightServerBase* base() const { return base_; }
  /// \brief Implement DoGet in terms of a transport-level stream.
  ///
  /// \param[in] context The server context.
  /// \param[in] request The request payload.
  /// \param[in] stream The transport-specific data stream
  ///   implementation. Must implement WriteData(const
  ///   FlightPayload&).
  Status DoGet(const ServerCallContext& context, const Ticket& request,
               ServerDataStream* stream);
  /// \brief Implement DoPut in terms of a transport-level stream.
  ///
  /// \param[in] context The server context.
  /// \param[in] stream The transport-specific data stream
  ///   implementation. Must implement ReadData(FlightData*)
  ///   and WritePutMetadata(const Buffer&).
  Status DoPut(const ServerCallContext& context, ServerDataStream* stream);
  /// \brief Implement DoExchange in terms of a transport-level stream.
  ///
  /// \param[in] context The server context.
  /// \param[in] stream The transport-specific data stream
  ///   implementation. Must implement ReadData(FlightData*)
  ///   and WriteData(const FlightPayload&).
  Status DoExchange(const ServerCallContext& context, ServerDataStream* stream);
  ///@}

 protected:
  FlightServerBase* base_;
  std::shared_ptr<MemoryManager> memory_manager_;
};

}  // namespace internal
}  // namespace flight
}  // namespace arrow

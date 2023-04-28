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

/// \file
/// Internal (but not private) interface for implementing
/// alternate network transports in Flight.
///
/// \warning EXPERIMENTAL. Subject to change.
///
/// To implement a transport, implement ServerTransport and
/// ClientTransport, and register the desired URI schemes with
/// TransportRegistry. Flight takes care of most of the per-RPC
/// details; transports only handle connections and providing a I/O
/// stream implementation (TransportDataStream).
///
/// On the server side:
///
/// 1. Applications subclass FlightServerBase and override RPC handlers.
/// 2. FlightServerBase::Init will look up and create a ServerTransport
///    based on the scheme of the Location given to it.
/// 3. The ServerTransport will start the actual server. (For instance,
///    for gRPC, it creates a gRPC server and registers a gRPC service.)
///    That server will handle connections.
/// 4. The transport should forward incoming calls to the server to the RPC
///    handlers defined on ServerTransport, which implements the actual
///    RPC handler using the interfaces here. Any I/O the RPC handler needs
///    to do is managed by transport-specific implementations of
///    TransportDataStream.
/// 5. ServerTransport calls FlightServerBase for the actual application
///    logic.
///
/// On the client side:
///
/// 1. Applications create a FlightClient with a Location.
/// 2. FlightClient will look up and create a ClientTransport based on
///    the scheme of the Location given to it.
/// 3. When calling a method on FlightClient, FlightClient will delegate to
///    the ClientTransport. There is some indirection, e.g. for DoGet,
///    FlightClient only requests that the ClientTransport start the
///    call and provide it with an I/O stream. The "Flight implementation"
///    itself still lives in FlightClient.

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "arrow/flight/type_fwd.h"
#include "arrow/flight/visibility.h"
#include "arrow/type_fwd.h"

namespace arrow {
namespace ipc {
class Message;
}
namespace flight {
class FlightStatusDetail;
namespace internal {

/// Internal, not user-visible type used for memory-efficient reads
struct FlightData {
  /// Used only for puts, may be null
  std::unique_ptr<FlightDescriptor> descriptor;

  /// Non-length-prefixed Message header as described in format/Message.fbs
  std::shared_ptr<Buffer> metadata;

  /// Application-defined metadata
  std::shared_ptr<Buffer> app_metadata;

  /// Message body
  std::shared_ptr<Buffer> body;

  /// Open IPC message from the metadata and body
  ::arrow::Result<std::unique_ptr<ipc::Message>> OpenMessage();
};

/// \brief A transport-specific interface for reading/writing Arrow data.
///
/// New transports will implement this to read/write IPC payloads to
/// the underlying stream.
class ARROW_FLIGHT_EXPORT TransportDataStream {
 public:
  virtual ~TransportDataStream() = default;
  /// \brief Attempt to read the next FlightData message.
  ///
  /// \return success true if data was populated, false if there was
  ///   an error. For clients, the error can be retrieved from
  ///   Finish(Status).
  virtual bool ReadData(FlightData* data);
  /// \brief Attempt to write a FlightPayload.
  ///
  /// \param[in] payload The data to write.
  /// \return true if the message was accepted by the transport, false
  ///   if not (e.g. due to client/server disconnect), Status if there
  ///   was an error (e.g. with the payload itself).
  virtual arrow::Result<bool> WriteData(const FlightPayload& payload);
  /// \brief Indicate that there are no more writes on this stream.
  ///
  /// This is only a hint for the underlying transport and may not
  /// actually do anything.
  virtual Status WritesDone();
};

/// \brief A transport-specific interface for reading/writing Arrow
///   data for a client.
class ARROW_FLIGHT_EXPORT ClientDataStream : public TransportDataStream {
 public:
  /// \brief Attempt to read a non-data message.
  ///
  /// Only implemented for DoPut; mutually exclusive with
  /// ReadData(FlightData*).
  virtual bool ReadPutMetadata(std::shared_ptr<Buffer>* out);
  /// \brief Attempt to cancel the call.
  ///
  /// This is only a hint and may not take effect immediately. The
  /// client should still finish the call with Finish(Status) as usual.
  virtual void TryCancel() {}
  /// \brief Finish the call, reporting the server-sent status and/or
  ///   any client-side errors as appropriate.
  ///
  /// Implies WritesDone() and DoFinish().
  ///
  /// \param[in] st A client-side status to combine with the
  ///   server-side error. That is, if an error occurs on the
  ///   client-side, call Finish(Status) to finish the server-side
  ///   call, get the server-side status, and merge the statuses
  ///   together so context is not lost.
  Status Finish(Status st);

 protected:
  /// \brief End the call, returning the final server status.
  ///
  /// For implementors: should imply WritesDone() (even if it does not
  /// directly call it).
  ///
  /// Implies WritesDone().
  virtual Status DoFinish() = 0;
};

/// An implementation of a Flight client for a particular transport.
///
/// Transports should override the methods they are capable of
/// supporting. The default method implementations return an error.
class ARROW_FLIGHT_EXPORT ClientTransport {
 public:
  virtual ~ClientTransport() = default;

  /// Initialize the client.
  virtual Status Init(const FlightClientOptions& options, const Location& location,
                      const arrow::internal::Uri& uri) = 0;
  /// Close the client. Once this returns, the client is no longer usable.
  virtual Status Close() = 0;

  virtual Status Authenticate(const FlightCallOptions& options,
                              std::unique_ptr<ClientAuthHandler> auth_handler);
  virtual arrow::Result<std::pair<std::string, std::string>> AuthenticateBasicToken(
      const FlightCallOptions& options, const std::string& username,
      const std::string& password);
  virtual Status DoAction(const FlightCallOptions& options, const Action& action,
                          std::unique_ptr<ResultStream>* results);
  virtual Status ListActions(const FlightCallOptions& options,
                             std::vector<ActionType>* actions);
  virtual Status GetFlightInfo(const FlightCallOptions& options,
                               const FlightDescriptor& descriptor,
                               std::unique_ptr<FlightInfo>* info);
  virtual arrow::Result<std::unique_ptr<SchemaResult>> GetSchema(
      const FlightCallOptions& options, const FlightDescriptor& descriptor);
  virtual Status ListFlights(const FlightCallOptions& options, const Criteria& criteria,
                             std::unique_ptr<FlightListing>* listing);
  virtual Status DoGet(const FlightCallOptions& options, const Ticket& ticket,
                       std::unique_ptr<ClientDataStream>* stream);
  virtual Status DoPut(const FlightCallOptions& options,
                       std::unique_ptr<ClientDataStream>* stream);
  virtual Status DoExchange(const FlightCallOptions& options,
                            std::unique_ptr<ClientDataStream>* stream);
};

/// A registry of transport implementations.
class ARROW_FLIGHT_EXPORT TransportRegistry {
 public:
  using ClientFactory = std::function<arrow::Result<std::unique_ptr<ClientTransport>>()>;
  using ServerFactory = std::function<arrow::Result<std::unique_ptr<ServerTransport>>(
      FlightServerBase*, std::shared_ptr<MemoryManager> memory_manager)>;

  TransportRegistry();
  ~TransportRegistry();

  arrow::Result<std::unique_ptr<ClientTransport>> MakeClient(
      const std::string& scheme) const;
  arrow::Result<std::unique_ptr<ServerTransport>> MakeServer(
      const std::string& scheme, FlightServerBase* base,
      std::shared_ptr<MemoryManager> memory_manager) const;

  Status RegisterClient(const std::string& scheme, ClientFactory factory);
  Status RegisterServer(const std::string& scheme, ServerFactory factory);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

/// \brief Get the registry of transport implementations.
ARROW_FLIGHT_EXPORT
TransportRegistry* GetDefaultTransportRegistry();

//------------------------------------------------------------
// Error propagation helpers

/// \brief Abstract status code as per the Flight specification.
enum class TransportStatusCode {
  kOk = 0,
  kUnknown = 1,
  kInternal = 2,
  kInvalidArgument = 3,
  kTimedOut = 4,
  kNotFound = 5,
  kAlreadyExists = 6,
  kCancelled = 7,
  kUnauthenticated = 8,
  kUnauthorized = 9,
  kUnimplemented = 10,
  kUnavailable = 11,
};

/// \brief Abstract error status.
///
/// Transport implementations may use side channels (e.g. HTTP
/// trailers) to convey additional information to reconstruct the
/// original C++ status for implementations that can use it.
struct ARROW_FLIGHT_EXPORT TransportStatus {
  TransportStatusCode code;
  std::string message;

  /// \brief Convert a C++ status to an abstract transport status.
  static TransportStatus FromStatus(const Status& arrow_status);

  /// \brief Reconstruct a string-encoded TransportStatus.
  static TransportStatus FromCodeStringAndMessage(const std::string& code_str,
                                                  std::string message);

  /// \brief Convert an abstract transport status to a C++ status.
  Status ToStatus() const;
};

/// \brief Convert the string representation of an Arrow status code
///   back to an Arrow status.
ARROW_FLIGHT_EXPORT
Status ReconstructStatus(const std::string& code_str, const Status& current_status,
                         std::optional<std::string> message,
                         std::optional<std::string> detail_message,
                         std::optional<std::string> detail_bin,
                         std::shared_ptr<FlightStatusDetail> detail);

}  // namespace internal
}  // namespace flight
}  // namespace arrow

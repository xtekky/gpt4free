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

#include <signal.h>
#include <utility>

#include "arrow/util/io_util.h"
#include "arrow/util/logging.h"
#include "arrow/python/flight.h"

using arrow::flight::FlightPayload;

namespace arrow {
namespace py {
namespace flight {

const char* kPyServerMiddlewareName = "arrow.py_server_middleware";

PyServerAuthHandler::PyServerAuthHandler(PyObject* handler,
                                         const PyServerAuthHandlerVtable& vtable)
    : vtable_(vtable) {
  Py_INCREF(handler);
  handler_.reset(handler);
}

Status PyServerAuthHandler::Authenticate(arrow::flight::ServerAuthSender* outgoing,
                                         arrow::flight::ServerAuthReader* incoming) {
  return SafeCallIntoPython([=] {
    const Status status = vtable_.authenticate(handler_.obj(), outgoing, incoming);
    RETURN_NOT_OK(CheckPyError());
    return status;
  });
}

Status PyServerAuthHandler::IsValid(const std::string& token,
                                    std::string* peer_identity) {
  return SafeCallIntoPython([=] {
    const Status status = vtable_.is_valid(handler_.obj(), token, peer_identity);
    RETURN_NOT_OK(CheckPyError());
    return status;
  });
}

PyClientAuthHandler::PyClientAuthHandler(PyObject* handler,
                                         const PyClientAuthHandlerVtable& vtable)
    : vtable_(vtable) {
  Py_INCREF(handler);
  handler_.reset(handler);
}

Status PyClientAuthHandler::Authenticate(arrow::flight::ClientAuthSender* outgoing,
                                         arrow::flight::ClientAuthReader* incoming) {
  return SafeCallIntoPython([=] {
    const Status status = vtable_.authenticate(handler_.obj(), outgoing, incoming);
    RETURN_NOT_OK(CheckPyError());
    return status;
  });
}

Status PyClientAuthHandler::GetToken(std::string* token) {
  return SafeCallIntoPython([=] {
    const Status status = vtable_.get_token(handler_.obj(), token);
    RETURN_NOT_OK(CheckPyError());
    return status;
  });
}

PyFlightServer::PyFlightServer(PyObject* server, const PyFlightServerVtable& vtable)
    : vtable_(vtable) {
  Py_INCREF(server);
  server_.reset(server);
}

Status PyFlightServer::ListFlights(
    const arrow::flight::ServerCallContext& context,
    const arrow::flight::Criteria* criteria,
    std::unique_ptr<arrow::flight::FlightListing>* listings) {
  return SafeCallIntoPython([&] {
    const Status status =
        vtable_.list_flights(server_.obj(), context, criteria, listings);
    RETURN_NOT_OK(CheckPyError());
    return status;
  });
}

Status PyFlightServer::GetFlightInfo(const arrow::flight::ServerCallContext& context,
                                     const arrow::flight::FlightDescriptor& request,
                                     std::unique_ptr<arrow::flight::FlightInfo>* info) {
  return SafeCallIntoPython([&] {
    const Status status = vtable_.get_flight_info(server_.obj(), context, request, info);
    RETURN_NOT_OK(CheckPyError());
    return status;
  });
}

Status PyFlightServer::GetSchema(const arrow::flight::ServerCallContext& context,
                                 const arrow::flight::FlightDescriptor& request,
                                 std::unique_ptr<arrow::flight::SchemaResult>* result) {
  return SafeCallIntoPython([&] {
    const Status status = vtable_.get_schema(server_.obj(), context, request, result);
    RETURN_NOT_OK(CheckPyError());
    return status;
  });
}

Status PyFlightServer::DoGet(const arrow::flight::ServerCallContext& context,
                             const arrow::flight::Ticket& request,
                             std::unique_ptr<arrow::flight::FlightDataStream>* stream) {
  return SafeCallIntoPython([&] {
    const Status status = vtable_.do_get(server_.obj(), context, request, stream);
    RETURN_NOT_OK(CheckPyError());
    return status;
  });
}

Status PyFlightServer::DoPut(
    const arrow::flight::ServerCallContext& context,
    std::unique_ptr<arrow::flight::FlightMessageReader> reader,
    std::unique_ptr<arrow::flight::FlightMetadataWriter> writer) {
  return SafeCallIntoPython([&] {
    const Status status =
        vtable_.do_put(server_.obj(), context, std::move(reader), std::move(writer));
    RETURN_NOT_OK(CheckPyError());
    return status;
  });
}

Status PyFlightServer::DoExchange(
    const arrow::flight::ServerCallContext& context,
    std::unique_ptr<arrow::flight::FlightMessageReader> reader,
    std::unique_ptr<arrow::flight::FlightMessageWriter> writer) {
  return SafeCallIntoPython([&] {
    const Status status =
        vtable_.do_exchange(server_.obj(), context, std::move(reader), std::move(writer));
    RETURN_NOT_OK(CheckPyError());
    return status;
  });
}

Status PyFlightServer::DoAction(const arrow::flight::ServerCallContext& context,
                                const arrow::flight::Action& action,
                                std::unique_ptr<arrow::flight::ResultStream>* result) {
  return SafeCallIntoPython([&] {
    const Status status = vtable_.do_action(server_.obj(), context, action, result);
    RETURN_NOT_OK(CheckPyError());
    return status;
  });
}

Status PyFlightServer::ListActions(const arrow::flight::ServerCallContext& context,
                                   std::vector<arrow::flight::ActionType>* actions) {
  return SafeCallIntoPython([&] {
    const Status status = vtable_.list_actions(server_.obj(), context, actions);
    RETURN_NOT_OK(CheckPyError());
    return status;
  });
}

Status PyFlightServer::ServeWithSignals() {
  // Respect the current Python settings, i.e. only interrupt the server if there is
  // an active signal handler for SIGINT and SIGTERM.
  std::vector<int> signals;
  for (const int signum : {SIGINT, SIGTERM}) {
    ARROW_ASSIGN_OR_RAISE(auto handler, ::arrow::internal::GetSignalHandler(signum));
    auto cb = handler.callback();
    if (cb != SIG_DFL && cb != SIG_IGN) {
      signals.push_back(signum);
    }
  }
  RETURN_NOT_OK(SetShutdownOnSignals(signals));

  // Serve until we got told to shutdown or a signal interrupted us
  RETURN_NOT_OK(Serve());
  int signum = GotSignal();
  if (signum != 0) {
    // Issue the signal again with Python's signal handlers restored
    PyAcquireGIL lock;
    raise(signum);
    // XXX Ideally we would loop and serve again if no exception was raised.
    // Unfortunately, gRPC will return immediately if Serve() is called again.
    ARROW_UNUSED(PyErr_CheckSignals());
  }

  return Status::OK();
}

PyFlightResultStream::PyFlightResultStream(PyObject* generator,
                                           PyFlightResultStreamCallback callback)
    : callback_(callback) {
  Py_INCREF(generator);
  generator_.reset(generator);
}

arrow::Result<std::unique_ptr<arrow::flight::Result>> PyFlightResultStream::Next() {
  return SafeCallIntoPython(
      [=]() -> arrow::Result<std::unique_ptr<arrow::flight::Result>> {
        std::unique_ptr<arrow::flight::Result> result;
        const Status status = callback_(generator_.obj(), &result);
        RETURN_NOT_OK(CheckPyError());
        RETURN_NOT_OK(status);
        return result;
      });
}

PyFlightDataStream::PyFlightDataStream(
    PyObject* data_source, std::unique_ptr<arrow::flight::FlightDataStream> stream)
    : stream_(std::move(stream)) {
  Py_INCREF(data_source);
  data_source_.reset(data_source);
}

std::shared_ptr<Schema> PyFlightDataStream::schema() { return stream_->schema(); }

arrow::Result<FlightPayload> PyFlightDataStream::GetSchemaPayload() {
  return stream_->GetSchemaPayload();
}

arrow::Result<FlightPayload> PyFlightDataStream::Next() { return stream_->Next(); }

PyGeneratorFlightDataStream::PyGeneratorFlightDataStream(
    PyObject* generator, std::shared_ptr<arrow::Schema> schema,
    PyGeneratorFlightDataStreamCallback callback, const ipc::IpcWriteOptions& options)
    : schema_(schema), mapper_(*schema_), options_(options), callback_(callback) {
  Py_INCREF(generator);
  generator_.reset(generator);
}

std::shared_ptr<Schema> PyGeneratorFlightDataStream::schema() { return schema_; }

arrow::Result<FlightPayload> PyGeneratorFlightDataStream::GetSchemaPayload() {
  FlightPayload payload;
  RETURN_NOT_OK(ipc::GetSchemaPayload(*schema_, options_, mapper_, &payload.ipc_message));
  return payload;
}

arrow::Result<FlightPayload> PyGeneratorFlightDataStream::Next() {
  return SafeCallIntoPython([=]() -> arrow::Result<FlightPayload> {
    FlightPayload payload;
    const Status status = callback_(generator_.obj(), &payload);
    RETURN_NOT_OK(CheckPyError());
    RETURN_NOT_OK(status);
    return payload;
  });
}

// Flight Server Middleware

PyServerMiddlewareFactory::PyServerMiddlewareFactory(PyObject* factory,
                                                     StartCallCallback start_call)
    : start_call_(start_call) {
  Py_INCREF(factory);
  factory_.reset(factory);
}

Status PyServerMiddlewareFactory::StartCall(
    const arrow::flight::CallInfo& info,
    const arrow::flight::CallHeaders& incoming_headers,
    std::shared_ptr<arrow::flight::ServerMiddleware>* middleware) {
  return SafeCallIntoPython([&] {
    const Status status = start_call_(factory_.obj(), info, incoming_headers, middleware);
    RETURN_NOT_OK(CheckPyError());
    return status;
  });
}

PyServerMiddleware::PyServerMiddleware(PyObject* middleware, Vtable vtable)
    : vtable_(vtable) {
  Py_INCREF(middleware);
  middleware_.reset(middleware);
}

void PyServerMiddleware::SendingHeaders(arrow::flight::AddCallHeaders* outgoing_headers) {
  const Status& status = SafeCallIntoPython([&] {
    const Status status = vtable_.sending_headers(middleware_.obj(), outgoing_headers);
    RETURN_NOT_OK(CheckPyError());
    return status;
  });

  ARROW_WARN_NOT_OK(status, "Python server middleware failed in SendingHeaders");
}

void PyServerMiddleware::CallCompleted(const Status& call_status) {
  const Status& status = SafeCallIntoPython([&] {
    const Status status = vtable_.call_completed(middleware_.obj(), call_status);
    RETURN_NOT_OK(CheckPyError());
    return status;
  });

  ARROW_WARN_NOT_OK(status, "Python server middleware failed in CallCompleted");
}

std::string PyServerMiddleware::name() const { return kPyServerMiddlewareName; }

PyObject* PyServerMiddleware::py_object() const { return middleware_.obj(); }

// Flight Client Middleware

PyClientMiddlewareFactory::PyClientMiddlewareFactory(PyObject* factory,
                                                     StartCallCallback start_call)
    : start_call_(start_call) {
  Py_INCREF(factory);
  factory_.reset(factory);
}

void PyClientMiddlewareFactory::StartCall(
    const arrow::flight::CallInfo& info,
    std::unique_ptr<arrow::flight::ClientMiddleware>* middleware) {
  const Status& status = SafeCallIntoPython([&] {
    const Status status = start_call_(factory_.obj(), info, middleware);
    RETURN_NOT_OK(CheckPyError());
    return status;
  });

  ARROW_WARN_NOT_OK(status, "Python client middleware failed in StartCall");
}

PyClientMiddleware::PyClientMiddleware(PyObject* middleware, Vtable vtable)
    : vtable_(vtable) {
  Py_INCREF(middleware);
  middleware_.reset(middleware);
}

void PyClientMiddleware::SendingHeaders(arrow::flight::AddCallHeaders* outgoing_headers) {
  const Status& status = SafeCallIntoPython([&] {
    const Status status = vtable_.sending_headers(middleware_.obj(), outgoing_headers);
    RETURN_NOT_OK(CheckPyError());
    return status;
  });

  ARROW_WARN_NOT_OK(status, "Python client middleware failed in StartCall");
}

void PyClientMiddleware::ReceivedHeaders(
    const arrow::flight::CallHeaders& incoming_headers) {
  const Status& status = SafeCallIntoPython([&] {
    const Status status = vtable_.received_headers(middleware_.obj(), incoming_headers);
    RETURN_NOT_OK(CheckPyError());
    return status;
  });

  ARROW_WARN_NOT_OK(status, "Python client middleware failed in StartCall");
}

void PyClientMiddleware::CallCompleted(const Status& call_status) {
  const Status& status = SafeCallIntoPython([&] {
    const Status status = vtable_.call_completed(middleware_.obj(), call_status);
    RETURN_NOT_OK(CheckPyError());
    return status;
  });

  ARROW_WARN_NOT_OK(status, "Python client middleware failed in StartCall");
}

Status CreateFlightInfo(const std::shared_ptr<arrow::Schema>& schema,
                        const arrow::flight::FlightDescriptor& descriptor,
                        const std::vector<arrow::flight::FlightEndpoint>& endpoints,
                        int64_t total_records, int64_t total_bytes,
                        std::unique_ptr<arrow::flight::FlightInfo>* out) {
  ARROW_ASSIGN_OR_RAISE(auto result,
                        arrow::flight::FlightInfo::Make(*schema, descriptor, endpoints,
                                                        total_records, total_bytes));
  *out = std::unique_ptr<arrow::flight::FlightInfo>(
      new arrow::flight::FlightInfo(std::move(result)));
  return Status::OK();
}

Status CreateSchemaResult(const std::shared_ptr<arrow::Schema>& schema,
                          std::unique_ptr<arrow::flight::SchemaResult>* out) {
  return arrow::flight::SchemaResult::Make(*schema).Value(out);
}

}  // namespace flight
}  // namespace py
}  // namespace arrow

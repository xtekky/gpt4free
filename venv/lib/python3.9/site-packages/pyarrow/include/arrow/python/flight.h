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
#include <string>
#include <vector>

#include "arrow/flight/api.h"
#include "arrow/ipc/dictionary.h"
#include "arrow/python/common.h"

#if defined(_WIN32) || defined(__CYGWIN__)  // Windows
#if defined(_MSC_VER)
#pragma warning(disable : 4251)
#else
#pragma GCC diagnostic ignored "-Wattributes"
#endif

#ifdef ARROW_PYTHON_STATIC
#define ARROW_PYFLIGHT_EXPORT
#elif defined(ARROW_PYFLIGHT_EXPORTING)
#define ARROW_PYFLIGHT_EXPORT __declspec(dllexport)
#else
#define ARROW_PYFLIGHT_EXPORT __declspec(dllimport)
#endif

#else  // Not Windows
#ifndef ARROW_PYFLIGHT_EXPORT
#define ARROW_PYFLIGHT_EXPORT __attribute__((visibility("default")))
#endif
#endif  // Non-Windows

namespace arrow {

namespace py {

namespace flight {

ARROW_PYFLIGHT_EXPORT
extern const char* kPyServerMiddlewareName;

/// \brief A table of function pointers for calling from C++ into
/// Python.
class ARROW_PYFLIGHT_EXPORT PyFlightServerVtable {
 public:
  std::function<Status(PyObject*, const arrow::flight::ServerCallContext&,
                       const arrow::flight::Criteria*,
                       std::unique_ptr<arrow::flight::FlightListing>*)>
      list_flights;
  std::function<Status(PyObject*, const arrow::flight::ServerCallContext&,
                       const arrow::flight::FlightDescriptor&,
                       std::unique_ptr<arrow::flight::FlightInfo>*)>
      get_flight_info;
  std::function<Status(PyObject*, const arrow::flight::ServerCallContext&,
                       const arrow::flight::FlightDescriptor&,
                       std::unique_ptr<arrow::flight::SchemaResult>*)>
      get_schema;
  std::function<Status(PyObject*, const arrow::flight::ServerCallContext&,
                       const arrow::flight::Ticket&,
                       std::unique_ptr<arrow::flight::FlightDataStream>*)>
      do_get;
  std::function<Status(PyObject*, const arrow::flight::ServerCallContext&,
                       std::unique_ptr<arrow::flight::FlightMessageReader>,
                       std::unique_ptr<arrow::flight::FlightMetadataWriter>)>
      do_put;
  std::function<Status(PyObject*, const arrow::flight::ServerCallContext&,
                       std::unique_ptr<arrow::flight::FlightMessageReader>,
                       std::unique_ptr<arrow::flight::FlightMessageWriter>)>
      do_exchange;
  std::function<Status(PyObject*, const arrow::flight::ServerCallContext&,
                       const arrow::flight::Action&,
                       std::unique_ptr<arrow::flight::ResultStream>*)>
      do_action;
  std::function<Status(PyObject*, const arrow::flight::ServerCallContext&,
                       std::vector<arrow::flight::ActionType>*)>
      list_actions;
};

class ARROW_PYFLIGHT_EXPORT PyServerAuthHandlerVtable {
 public:
  std::function<Status(PyObject*, arrow::flight::ServerAuthSender*,
                       arrow::flight::ServerAuthReader*)>
      authenticate;
  std::function<Status(PyObject*, const std::string&, std::string*)> is_valid;
};

class ARROW_PYFLIGHT_EXPORT PyClientAuthHandlerVtable {
 public:
  std::function<Status(PyObject*, arrow::flight::ClientAuthSender*,
                       arrow::flight::ClientAuthReader*)>
      authenticate;
  std::function<Status(PyObject*, std::string*)> get_token;
};

/// \brief A helper to implement an auth mechanism in Python.
class ARROW_PYFLIGHT_EXPORT PyServerAuthHandler
    : public arrow::flight::ServerAuthHandler {
 public:
  explicit PyServerAuthHandler(PyObject* handler,
                               const PyServerAuthHandlerVtable& vtable);
  Status Authenticate(arrow::flight::ServerAuthSender* outgoing,
                      arrow::flight::ServerAuthReader* incoming) override;
  Status IsValid(const std::string& token, std::string* peer_identity) override;

 private:
  OwnedRefNoGIL handler_;
  PyServerAuthHandlerVtable vtable_;
};

/// \brief A helper to implement an auth mechanism in Python.
class ARROW_PYFLIGHT_EXPORT PyClientAuthHandler
    : public arrow::flight::ClientAuthHandler {
 public:
  explicit PyClientAuthHandler(PyObject* handler,
                               const PyClientAuthHandlerVtable& vtable);
  Status Authenticate(arrow::flight::ClientAuthSender* outgoing,
                      arrow::flight::ClientAuthReader* incoming) override;
  Status GetToken(std::string* token) override;

 private:
  OwnedRefNoGIL handler_;
  PyClientAuthHandlerVtable vtable_;
};

class ARROW_PYFLIGHT_EXPORT PyFlightServer : public arrow::flight::FlightServerBase {
 public:
  explicit PyFlightServer(PyObject* server, const PyFlightServerVtable& vtable);

  // Like Serve(), but set up signals and invoke Python signal handlers
  // if necessary.  This function may return with a Python exception set.
  Status ServeWithSignals();

  Status ListFlights(const arrow::flight::ServerCallContext& context,
                     const arrow::flight::Criteria* criteria,
                     std::unique_ptr<arrow::flight::FlightListing>* listings) override;
  Status GetFlightInfo(const arrow::flight::ServerCallContext& context,
                       const arrow::flight::FlightDescriptor& request,
                       std::unique_ptr<arrow::flight::FlightInfo>* info) override;
  Status GetSchema(const arrow::flight::ServerCallContext& context,
                   const arrow::flight::FlightDescriptor& request,
                   std::unique_ptr<arrow::flight::SchemaResult>* result) override;
  Status DoGet(const arrow::flight::ServerCallContext& context,
               const arrow::flight::Ticket& request,
               std::unique_ptr<arrow::flight::FlightDataStream>* stream) override;
  Status DoPut(const arrow::flight::ServerCallContext& context,
               std::unique_ptr<arrow::flight::FlightMessageReader> reader,
               std::unique_ptr<arrow::flight::FlightMetadataWriter> writer) override;
  Status DoExchange(const arrow::flight::ServerCallContext& context,
                    std::unique_ptr<arrow::flight::FlightMessageReader> reader,
                    std::unique_ptr<arrow::flight::FlightMessageWriter> writer) override;
  Status DoAction(const arrow::flight::ServerCallContext& context,
                  const arrow::flight::Action& action,
                  std::unique_ptr<arrow::flight::ResultStream>* result) override;
  Status ListActions(const arrow::flight::ServerCallContext& context,
                     std::vector<arrow::flight::ActionType>* actions) override;

 private:
  OwnedRefNoGIL server_;
  PyFlightServerVtable vtable_;
};

/// \brief A callback that obtains the next result from a Flight action.
typedef std::function<Status(PyObject*, std::unique_ptr<arrow::flight::Result>*)>
    PyFlightResultStreamCallback;

/// \brief A ResultStream built around a Python callback.
class ARROW_PYFLIGHT_EXPORT PyFlightResultStream : public arrow::flight::ResultStream {
 public:
  /// \brief Construct a FlightResultStream from a Python object and callback.
  /// Must only be called while holding the GIL.
  explicit PyFlightResultStream(PyObject* generator,
                                PyFlightResultStreamCallback callback);
  arrow::Result<std::unique_ptr<arrow::flight::Result>> Next() override;

 private:
  OwnedRefNoGIL generator_;
  PyFlightResultStreamCallback callback_;
};

/// \brief A wrapper around a FlightDataStream that keeps alive a
/// Python object backing it.
class ARROW_PYFLIGHT_EXPORT PyFlightDataStream : public arrow::flight::FlightDataStream {
 public:
  /// \brief Construct a FlightDataStream from a Python object and underlying stream.
  /// Must only be called while holding the GIL.
  explicit PyFlightDataStream(PyObject* data_source,
                              std::unique_ptr<arrow::flight::FlightDataStream> stream);

  std::shared_ptr<Schema> schema() override;
  arrow::Result<arrow::flight::FlightPayload> GetSchemaPayload() override;
  arrow::Result<arrow::flight::FlightPayload> Next() override;

 private:
  OwnedRefNoGIL data_source_;
  std::unique_ptr<arrow::flight::FlightDataStream> stream_;
};

class ARROW_PYFLIGHT_EXPORT PyServerMiddlewareFactory
    : public arrow::flight::ServerMiddlewareFactory {
 public:
  /// \brief A callback to create the middleware instance in Python
  typedef std::function<Status(
      PyObject*, const arrow::flight::CallInfo& info,
      const arrow::flight::CallHeaders& incoming_headers,
      std::shared_ptr<arrow::flight::ServerMiddleware>* middleware)>
      StartCallCallback;

  /// \brief Must only be called while holding the GIL.
  explicit PyServerMiddlewareFactory(PyObject* factory, StartCallCallback start_call);

  Status StartCall(const arrow::flight::CallInfo& info,
                   const arrow::flight::CallHeaders& incoming_headers,
                   std::shared_ptr<arrow::flight::ServerMiddleware>* middleware) override;

 private:
  OwnedRefNoGIL factory_;
  StartCallCallback start_call_;
};

class ARROW_PYFLIGHT_EXPORT PyServerMiddleware : public arrow::flight::ServerMiddleware {
 public:
  typedef std::function<Status(PyObject*,
                               arrow::flight::AddCallHeaders* outgoing_headers)>
      SendingHeadersCallback;
  typedef std::function<Status(PyObject*, const Status& status)> CallCompletedCallback;

  struct Vtable {
    SendingHeadersCallback sending_headers;
    CallCompletedCallback call_completed;
  };

  /// \brief Must only be called while holding the GIL.
  explicit PyServerMiddleware(PyObject* middleware, Vtable vtable);

  void SendingHeaders(arrow::flight::AddCallHeaders* outgoing_headers) override;
  void CallCompleted(const Status& status) override;
  std::string name() const override;
  /// \brief Get the underlying Python object.
  PyObject* py_object() const;

 private:
  OwnedRefNoGIL middleware_;
  Vtable vtable_;
};

class ARROW_PYFLIGHT_EXPORT PyClientMiddlewareFactory
    : public arrow::flight::ClientMiddlewareFactory {
 public:
  /// \brief A callback to create the middleware instance in Python
  typedef std::function<Status(
      PyObject*, const arrow::flight::CallInfo& info,
      std::unique_ptr<arrow::flight::ClientMiddleware>* middleware)>
      StartCallCallback;

  /// \brief Must only be called while holding the GIL.
  explicit PyClientMiddlewareFactory(PyObject* factory, StartCallCallback start_call);

  void StartCall(const arrow::flight::CallInfo& info,
                 std::unique_ptr<arrow::flight::ClientMiddleware>* middleware) override;

 private:
  OwnedRefNoGIL factory_;
  StartCallCallback start_call_;
};

class ARROW_PYFLIGHT_EXPORT PyClientMiddleware : public arrow::flight::ClientMiddleware {
 public:
  typedef std::function<Status(PyObject*,
                               arrow::flight::AddCallHeaders* outgoing_headers)>
      SendingHeadersCallback;
  typedef std::function<Status(PyObject*,
                               const arrow::flight::CallHeaders& incoming_headers)>
      ReceivedHeadersCallback;
  typedef std::function<Status(PyObject*, const Status& status)> CallCompletedCallback;

  struct Vtable {
    SendingHeadersCallback sending_headers;
    ReceivedHeadersCallback received_headers;
    CallCompletedCallback call_completed;
  };

  /// \brief Must only be called while holding the GIL.
  explicit PyClientMiddleware(PyObject* factory, Vtable vtable);

  void SendingHeaders(arrow::flight::AddCallHeaders* outgoing_headers) override;
  void ReceivedHeaders(const arrow::flight::CallHeaders& incoming_headers) override;
  void CallCompleted(const Status& status) override;

 private:
  OwnedRefNoGIL middleware_;
  Vtable vtable_;
};

/// \brief A callback that obtains the next payload from a Flight result stream.
typedef std::function<Status(PyObject*, arrow::flight::FlightPayload*)>
    PyGeneratorFlightDataStreamCallback;

/// \brief A FlightDataStream built around a Python callback.
class ARROW_PYFLIGHT_EXPORT PyGeneratorFlightDataStream
    : public arrow::flight::FlightDataStream {
 public:
  /// \brief Construct a FlightDataStream from a Python object and underlying stream.
  /// Must only be called while holding the GIL.
  explicit PyGeneratorFlightDataStream(PyObject* generator,
                                       std::shared_ptr<arrow::Schema> schema,
                                       PyGeneratorFlightDataStreamCallback callback,
                                       const ipc::IpcWriteOptions& options);
  std::shared_ptr<Schema> schema() override;
  arrow::Result<arrow::flight::FlightPayload> GetSchemaPayload() override;
  arrow::Result<arrow::flight::FlightPayload> Next() override;

 private:
  OwnedRefNoGIL generator_;
  std::shared_ptr<arrow::Schema> schema_;
  ipc::DictionaryFieldMapper mapper_;
  ipc::IpcWriteOptions options_;
  PyGeneratorFlightDataStreamCallback callback_;
};

ARROW_PYFLIGHT_EXPORT
Status CreateFlightInfo(const std::shared_ptr<arrow::Schema>& schema,
                        const arrow::flight::FlightDescriptor& descriptor,
                        const std::vector<arrow::flight::FlightEndpoint>& endpoints,
                        int64_t total_records, int64_t total_bytes,
                        std::unique_ptr<arrow::flight::FlightInfo>* out);

/// \brief Create a SchemaResult from schema.
ARROW_PYFLIGHT_EXPORT
Status CreateSchemaResult(const std::shared_ptr<arrow::Schema>& schema,
                          std::unique_ptr<arrow::flight::SchemaResult>* out);

}  // namespace flight
}  // namespace py
}  // namespace arrow

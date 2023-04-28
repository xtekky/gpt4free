# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# distutils: language = c++

from pyarrow.includes.common cimport *
from pyarrow.includes.libarrow cimport *


cdef extern from "arrow/flight/api.h" namespace "arrow" nogil:
    cdef char* CTracingServerMiddlewareName\
        " arrow::flight::TracingServerMiddleware::kMiddlewareName"

    cdef cppclass CActionType" arrow::flight::ActionType":
        c_string type
        c_string description
        bint operator==(CActionType)
        CResult[c_string] SerializeToString()

        @staticmethod
        CResult[CActionType] Deserialize(const c_string& serialized)

    cdef cppclass CAction" arrow::flight::Action":
        c_string type
        shared_ptr[CBuffer] body
        bint operator==(CAction)
        CResult[c_string] SerializeToString()

        @staticmethod
        CResult[CAction] Deserialize(const c_string& serialized)

    cdef cppclass CFlightResult" arrow::flight::Result":
        CFlightResult()
        CFlightResult(CFlightResult)
        shared_ptr[CBuffer] body
        bint operator==(CFlightResult)
        CResult[c_string] SerializeToString()

        @staticmethod
        CResult[CFlightResult] Deserialize(const c_string& serialized)

    cdef cppclass CBasicAuth" arrow::flight::BasicAuth":
        CBasicAuth()
        CBasicAuth(CBuffer)
        CBasicAuth(CBasicAuth)
        c_string username
        c_string password
        bint operator==(CBasicAuth)
        CResult[c_string] SerializeToString()

        @staticmethod
        CResult[CBasicAuth] Deserialize(const c_string& serialized)

    cdef cppclass CResultStream" arrow::flight::ResultStream":
        CResult[unique_ptr[CFlightResult]] Next()

    cdef cppclass CDescriptorType \
            " arrow::flight::FlightDescriptor::DescriptorType":
        bint operator==(CDescriptorType)

    CDescriptorType CDescriptorTypeUnknown\
        " arrow::flight::FlightDescriptor::UNKNOWN"
    CDescriptorType CDescriptorTypePath\
        " arrow::flight::FlightDescriptor::PATH"
    CDescriptorType CDescriptorTypeCmd\
        " arrow::flight::FlightDescriptor::CMD"

    cdef cppclass CFlightDescriptor" arrow::flight::FlightDescriptor":
        CDescriptorType type
        c_string cmd
        vector[c_string] path
        bint operator==(CFlightDescriptor)
        CResult[c_string] SerializeToString()

        @staticmethod
        CResult[CFlightDescriptor] Deserialize(const c_string& serialized)

    cdef cppclass CTicket" arrow::flight::Ticket":
        CTicket()
        c_string ticket
        bint operator==(CTicket)
        CResult[c_string] SerializeToString()

        @staticmethod
        CResult[CTicket] Deserialize(const c_string& serialized)

    cdef cppclass CCriteria" arrow::flight::Criteria":
        CCriteria()
        c_string expression
        bint operator==(CCriteria)
        CResult[c_string] SerializeToString()

        @staticmethod
        CResult[CCriteria] Deserialize(const c_string& serialized)

    cdef cppclass CLocation" arrow::flight::Location":
        CLocation()
        c_string ToString()
        c_bool Equals(const CLocation& other)

        @staticmethod
        CResult[CLocation] Parse(c_string& uri_string)

        @staticmethod
        CResult[CLocation] ForGrpcTcp(c_string& host, int port)

        @staticmethod
        CResult[CLocation] ForGrpcTls(c_string& host, int port)

        @staticmethod
        CResult[CLocation] ForGrpcUnix(c_string& path)

    cdef cppclass CFlightEndpoint" arrow::flight::FlightEndpoint":
        CFlightEndpoint()

        CTicket ticket
        vector[CLocation] locations

        bint operator==(CFlightEndpoint)
        CResult[c_string] SerializeToString()

        @staticmethod
        CResult[CFlightEndpoint] Deserialize(const c_string& serialized)

    cdef cppclass CFlightInfo" arrow::flight::FlightInfo":
        CFlightInfo(CFlightInfo info)
        int64_t total_records()
        int64_t total_bytes()
        CResult[shared_ptr[CSchema]] GetSchema(CDictionaryMemo* memo)
        CFlightDescriptor& descriptor()
        const vector[CFlightEndpoint]& endpoints()
        CResult[c_string] SerializeToString()

        @staticmethod
        CResult[unique_ptr[CFlightInfo]] Deserialize(
            const c_string& serialized)

    cdef cppclass CSchemaResult" arrow::flight::SchemaResult":
        CSchemaResult()
        CSchemaResult(CSchemaResult result)
        CResult[shared_ptr[CSchema]] GetSchema(CDictionaryMemo* memo)
        bint operator==(CSchemaResult)
        CResult[c_string] SerializeToString()

        @staticmethod
        CResult[CSchemaResult] Deserialize(const c_string& serialized)

    cdef cppclass CFlightListing" arrow::flight::FlightListing":
        CResult[unique_ptr[CFlightInfo]] Next()

    cdef cppclass CSimpleFlightListing" arrow::flight::SimpleFlightListing":
        CSimpleFlightListing(vector[CFlightInfo]&& info)

    cdef cppclass CFlightPayload" arrow::flight::FlightPayload":
        shared_ptr[CBuffer] descriptor
        shared_ptr[CBuffer] app_metadata
        CIpcPayload ipc_message

    cdef cppclass CFlightDataStream" arrow::flight::FlightDataStream":
        shared_ptr[CSchema] schema()
        CResult[CFlightPayload] Next()

    cdef cppclass CFlightStreamChunk" arrow::flight::FlightStreamChunk":
        CFlightStreamChunk()
        shared_ptr[CRecordBatch] data
        shared_ptr[CBuffer] app_metadata

    cdef cppclass CMetadataRecordBatchReader \
            " arrow::flight::MetadataRecordBatchReader":
        CResult[shared_ptr[CSchema]] GetSchema()
        CResult[CFlightStreamChunk] Next()
        CResult[shared_ptr[CTable]] ToTable()

    CResult[shared_ptr[CRecordBatchReader]] MakeRecordBatchReader\
        " arrow::flight::MakeRecordBatchReader"(
            shared_ptr[CMetadataRecordBatchReader])

    cdef cppclass CMetadataRecordBatchWriter \
            " arrow::flight::MetadataRecordBatchWriter"(CRecordBatchWriter):
        CStatus Begin(shared_ptr[CSchema] schema,
                      const CIpcWriteOptions& options)
        CStatus WriteMetadata(shared_ptr[CBuffer] app_metadata)
        CStatus WriteWithMetadata(const CRecordBatch& batch,
                                  shared_ptr[CBuffer] app_metadata)

    cdef cppclass CFlightStreamReader \
            " arrow::flight::FlightStreamReader"(CMetadataRecordBatchReader):
        void Cancel()
        CResult[shared_ptr[CTable]] ToTableWithStopToken" ToTable"\
            (const CStopToken& stop_token)

    cdef cppclass CFlightMessageReader \
            " arrow::flight::FlightMessageReader"(CMetadataRecordBatchReader):
        CFlightDescriptor& descriptor()

    cdef cppclass CFlightMessageWriter \
            " arrow::flight::FlightMessageWriter"(CMetadataRecordBatchWriter):
        pass

    cdef cppclass CFlightStreamWriter \
            " arrow::flight::FlightStreamWriter"(CMetadataRecordBatchWriter):
        CStatus DoneWriting()

    cdef cppclass CRecordBatchStream \
            " arrow::flight::RecordBatchStream"(CFlightDataStream):
        CRecordBatchStream(shared_ptr[CRecordBatchReader]& reader,
                           const CIpcWriteOptions& options)

    cdef cppclass CFlightMetadataReader" arrow::flight::FlightMetadataReader":
        CStatus ReadMetadata(shared_ptr[CBuffer]* out)

    cdef cppclass CFlightMetadataWriter" arrow::flight::FlightMetadataWriter":
        CStatus WriteMetadata(const CBuffer& message)

    cdef cppclass CServerAuthReader" arrow::flight::ServerAuthReader":
        CStatus Read(c_string* token)

    cdef cppclass CServerAuthSender" arrow::flight::ServerAuthSender":
        CStatus Write(c_string& token)

    cdef cppclass CClientAuthReader" arrow::flight::ClientAuthReader":
        CStatus Read(c_string* token)

    cdef cppclass CClientAuthSender" arrow::flight::ClientAuthSender":
        CStatus Write(c_string& token)

    cdef cppclass CServerAuthHandler" arrow::flight::ServerAuthHandler":
        pass

    cdef cppclass CClientAuthHandler" arrow::flight::ClientAuthHandler":
        pass

    cdef cppclass CServerCallContext" arrow::flight::ServerCallContext":
        c_string& peer_identity()
        c_string& peer()
        c_bool is_cancelled()
        CServerMiddleware* GetMiddleware(const c_string& key)

    cdef cppclass CTimeoutDuration" arrow::flight::TimeoutDuration":
        CTimeoutDuration(double)

    cdef cppclass CFlightCallOptions" arrow::flight::FlightCallOptions":
        CFlightCallOptions()
        CTimeoutDuration timeout
        CIpcWriteOptions write_options
        CIpcReadOptions read_options
        vector[pair[c_string, c_string]] headers
        CStopToken stop_token

    cdef cppclass CCertKeyPair" arrow::flight::CertKeyPair":
        CCertKeyPair()
        c_string pem_cert
        c_string pem_key

    cdef cppclass CFlightMethod" arrow::flight::FlightMethod":
        bint operator==(CFlightMethod)

    CFlightMethod CFlightMethodInvalid\
        " arrow::flight::FlightMethod::Invalid"
    CFlightMethod CFlightMethodHandshake\
        " arrow::flight::FlightMethod::Handshake"
    CFlightMethod CFlightMethodListFlights\
        " arrow::flight::FlightMethod::ListFlights"
    CFlightMethod CFlightMethodGetFlightInfo\
        " arrow::flight::FlightMethod::GetFlightInfo"
    CFlightMethod CFlightMethodGetSchema\
        " arrow::flight::FlightMethod::GetSchema"
    CFlightMethod CFlightMethodDoGet\
        " arrow::flight::FlightMethod::DoGet"
    CFlightMethod CFlightMethodDoPut\
        " arrow::flight::FlightMethod::DoPut"
    CFlightMethod CFlightMethodDoAction\
        " arrow::flight::FlightMethod::DoAction"
    CFlightMethod CFlightMethodListActions\
        " arrow::flight::FlightMethod::ListActions"
    CFlightMethod CFlightMethodDoExchange\
        " arrow::flight::FlightMethod::DoExchange"

    cdef cppclass CCallInfo" arrow::flight::CallInfo":
        CFlightMethod method

    # This is really std::unordered_multimap, but Cython has no
    # bindings for it, so treat it as an opaque class and bind the
    # methods we need
    cdef cppclass CCallHeaders" arrow::flight::CallHeaders":
        cppclass const_iterator:
            pair[c_string, c_string] operator*()
            const_iterator operator++()
            bint operator==(const_iterator)
            bint operator!=(const_iterator)
        const_iterator cbegin()
        const_iterator cend()

    cdef cppclass CAddCallHeaders" arrow::flight::AddCallHeaders":
        void AddHeader(const c_string& key, const c_string& value)

    cdef cppclass CServerMiddleware" arrow::flight::ServerMiddleware":
        c_string name()

    cdef cppclass CServerMiddlewareFactory\
            " arrow::flight::ServerMiddlewareFactory":
        pass

    cdef cppclass CClientMiddleware" arrow::flight::ClientMiddleware":
        pass

    cdef cppclass CClientMiddlewareFactory\
            " arrow::flight::ClientMiddlewareFactory":
        pass

    cpdef cppclass CTracingServerMiddlewareTraceKey\
            " arrow::flight::TracingServerMiddleware::TraceKey":
        CTracingServerMiddlewareTraceKey()
        c_string key
        c_string value

    cdef cppclass CTracingServerMiddleware\
            " arrow::flight::TracingServerMiddleware"(CServerMiddleware):
        vector[CTracingServerMiddlewareTraceKey] GetTraceContext()

    cdef shared_ptr[CServerMiddlewareFactory] \
        MakeTracingServerMiddlewareFactory\
        " arrow::flight::MakeTracingServerMiddlewareFactory"()

    cdef cppclass CFlightServerOptions" arrow::flight::FlightServerOptions":
        CFlightServerOptions(const CLocation& location)
        CLocation location
        unique_ptr[CServerAuthHandler] auth_handler
        vector[CCertKeyPair] tls_certificates
        c_bool verify_client
        c_string root_certificates
        vector[pair[c_string, shared_ptr[CServerMiddlewareFactory]]] middleware

    cdef cppclass CFlightClientOptions" arrow::flight::FlightClientOptions":
        c_string tls_root_certs
        c_string cert_chain
        c_string private_key
        c_string override_hostname
        vector[shared_ptr[CClientMiddlewareFactory]] middleware
        int64_t write_size_limit_bytes
        vector[pair[c_string, CIntStringVariant]] generic_options
        c_bool disable_server_verification

        @staticmethod
        CFlightClientOptions Defaults()

    cdef cppclass CDoPutResult" arrow::flight::FlightClient::DoPutResult":
        unique_ptr[CFlightStreamWriter] writer
        unique_ptr[CFlightMetadataReader] reader

    cdef cppclass CDoExchangeResult" arrow::flight::FlightClient::DoExchangeResult":
        unique_ptr[CFlightStreamWriter] writer
        unique_ptr[CFlightStreamReader] reader

    cdef cppclass CFlightClient" arrow::flight::FlightClient":
        @staticmethod
        CResult[unique_ptr[CFlightClient]] Connect(const CLocation& location,
                                                   const CFlightClientOptions& options)

        CStatus Authenticate(CFlightCallOptions& options,
                             unique_ptr[CClientAuthHandler] auth_handler)

        CResult[pair[c_string, c_string]] AuthenticateBasicToken(
            CFlightCallOptions& options,
            const c_string& username,
            const c_string& password)

        CResult[unique_ptr[CResultStream]] DoAction(CFlightCallOptions& options, CAction& action)
        CResult[vector[CActionType]] ListActions(CFlightCallOptions& options)

        CResult[unique_ptr[CFlightListing]] ListFlights(CFlightCallOptions& options, CCriteria criteria)
        CResult[unique_ptr[CFlightInfo]] GetFlightInfo(CFlightCallOptions& options,
                                                       CFlightDescriptor& descriptor)
        CResult[unique_ptr[CSchemaResult]] GetSchema(CFlightCallOptions& options,
                                                     CFlightDescriptor& descriptor)
        CResult[unique_ptr[CFlightStreamReader]] DoGet(CFlightCallOptions& options, CTicket& ticket)
        CResult[CDoPutResult] DoPut(CFlightCallOptions& options,
                                    CFlightDescriptor& descriptor,
                                    shared_ptr[CSchema]& schema)
        CResult[CDoExchangeResult] DoExchange(CFlightCallOptions& options,
                                              CFlightDescriptor& descriptor)
        CStatus Close()

    cdef cppclass CFlightStatusCode" arrow::flight::FlightStatusCode":
        bint operator==(CFlightStatusCode)

    CFlightStatusCode CFlightStatusInternal \
        " arrow::flight::FlightStatusCode::Internal"
    CFlightStatusCode CFlightStatusTimedOut \
        " arrow::flight::FlightStatusCode::TimedOut"
    CFlightStatusCode CFlightStatusCancelled \
        " arrow::flight::FlightStatusCode::Cancelled"
    CFlightStatusCode CFlightStatusUnauthenticated \
        " arrow::flight::FlightStatusCode::Unauthenticated"
    CFlightStatusCode CFlightStatusUnauthorized \
        " arrow::flight::FlightStatusCode::Unauthorized"
    CFlightStatusCode CFlightStatusUnavailable \
        " arrow::flight::FlightStatusCode::Unavailable"
    CFlightStatusCode CFlightStatusFailed \
        " arrow::flight::FlightStatusCode::Failed"

    cdef cppclass FlightStatusDetail" arrow::flight::FlightStatusDetail":
        CFlightStatusCode code()
        c_string extra_info()

        @staticmethod
        shared_ptr[FlightStatusDetail] UnwrapStatus(const CStatus& status)

    cdef cppclass FlightWriteSizeStatusDetail\
            " arrow::flight::FlightWriteSizeStatusDetail":
        int64_t limit()
        int64_t actual()

        @staticmethod
        shared_ptr[FlightWriteSizeStatusDetail] UnwrapStatus(
            const CStatus& status)

    cdef CStatus MakeFlightError" arrow::flight::MakeFlightError" \
        (CFlightStatusCode code, const c_string& message)

    cdef CStatus MakeFlightError" arrow::flight::MakeFlightError" \
        (CFlightStatusCode code,
         const c_string& message,
         const c_string& extra_info)

# Callbacks for implementing Flight servers
# Use typedef to emulate syntax for std::function<void(..)>
ctypedef CStatus cb_list_flights(object, const CServerCallContext&,
                                 const CCriteria*,
                                 unique_ptr[CFlightListing]*)
ctypedef CStatus cb_get_flight_info(object, const CServerCallContext&,
                                    const CFlightDescriptor&,
                                    unique_ptr[CFlightInfo]*)
ctypedef CStatus cb_get_schema(object, const CServerCallContext&,
                               const CFlightDescriptor&,
                               unique_ptr[CSchemaResult]*)
ctypedef CStatus cb_do_put(object, const CServerCallContext&,
                           unique_ptr[CFlightMessageReader],
                           unique_ptr[CFlightMetadataWriter])
ctypedef CStatus cb_do_get(object, const CServerCallContext&,
                           const CTicket&,
                           unique_ptr[CFlightDataStream]*)
ctypedef CStatus cb_do_exchange(object, const CServerCallContext&,
                                unique_ptr[CFlightMessageReader],
                                unique_ptr[CFlightMessageWriter])
ctypedef CStatus cb_do_action(object, const CServerCallContext&,
                              const CAction&,
                              unique_ptr[CResultStream]*)
ctypedef CStatus cb_list_actions(object, const CServerCallContext&,
                                 vector[CActionType]*)
ctypedef CStatus cb_result_next(object, unique_ptr[CFlightResult]*)
ctypedef CStatus cb_data_stream_next(object, CFlightPayload*)
ctypedef CStatus cb_server_authenticate(object, CServerAuthSender*,
                                        CServerAuthReader*)
ctypedef CStatus cb_is_valid(object, const c_string&, c_string*)
ctypedef CStatus cb_client_authenticate(object, CClientAuthSender*,
                                        CClientAuthReader*)
ctypedef CStatus cb_get_token(object, c_string*)

ctypedef CStatus cb_middleware_sending_headers(object, CAddCallHeaders*)
ctypedef CStatus cb_middleware_call_completed(object, const CStatus&)
ctypedef CStatus cb_client_middleware_received_headers(
    object, const CCallHeaders&)
ctypedef CStatus cb_server_middleware_start_call(
    object,
    const CCallInfo&,
    const CCallHeaders&,
    shared_ptr[CServerMiddleware]*)
ctypedef CStatus cb_client_middleware_start_call(
    object,
    const CCallInfo&,
    unique_ptr[CClientMiddleware]*)

cdef extern from "arrow/python/flight.h" namespace "arrow::py::flight" nogil:
    cdef char* CPyServerMiddlewareName\
        " arrow::py::flight::kPyServerMiddlewareName"

    cdef cppclass PyFlightServerVtable:
        PyFlightServerVtable()
        function[cb_list_flights] list_flights
        function[cb_get_flight_info] get_flight_info
        function[cb_get_schema] get_schema
        function[cb_do_put] do_put
        function[cb_do_get] do_get
        function[cb_do_exchange] do_exchange
        function[cb_do_action] do_action
        function[cb_list_actions] list_actions

    cdef cppclass PyServerAuthHandlerVtable:
        PyServerAuthHandlerVtable()
        function[cb_server_authenticate] authenticate
        function[cb_is_valid] is_valid

    cdef cppclass PyClientAuthHandlerVtable:
        PyClientAuthHandlerVtable()
        function[cb_client_authenticate] authenticate
        function[cb_get_token] get_token

    cdef cppclass PyFlightServer:
        PyFlightServer(object server, PyFlightServerVtable vtable)

        CStatus Init(CFlightServerOptions& options)
        int port()
        CStatus ServeWithSignals() except *
        CStatus Shutdown()
        CStatus Wait()

    cdef cppclass PyServerAuthHandler\
            " arrow::py::flight::PyServerAuthHandler"(CServerAuthHandler):
        PyServerAuthHandler(object handler, PyServerAuthHandlerVtable vtable)

    cdef cppclass PyClientAuthHandler\
            " arrow::py::flight::PyClientAuthHandler"(CClientAuthHandler):
        PyClientAuthHandler(object handler, PyClientAuthHandlerVtable vtable)

    cdef cppclass CPyFlightResultStream\
            " arrow::py::flight::PyFlightResultStream"(CResultStream):
        CPyFlightResultStream(object generator,
                              function[cb_result_next] callback)

    cdef cppclass CPyFlightDataStream\
            " arrow::py::flight::PyFlightDataStream"(CFlightDataStream):
        CPyFlightDataStream(object data_source,
                            unique_ptr[CFlightDataStream] stream)

    cdef cppclass CPyGeneratorFlightDataStream\
            " arrow::py::flight::PyGeneratorFlightDataStream"\
            (CFlightDataStream):
        CPyGeneratorFlightDataStream(object generator,
                                     shared_ptr[CSchema] schema,
                                     function[cb_data_stream_next] callback,
                                     const CIpcWriteOptions& options)

    cdef cppclass PyServerMiddlewareVtable\
            " arrow::py::flight::PyServerMiddleware::Vtable":
        PyServerMiddlewareVtable()
        function[cb_middleware_sending_headers] sending_headers
        function[cb_middleware_call_completed] call_completed

    cdef cppclass PyClientMiddlewareVtable\
            " arrow::py::flight::PyClientMiddleware::Vtable":
        PyClientMiddlewareVtable()
        function[cb_middleware_sending_headers] sending_headers
        function[cb_client_middleware_received_headers] received_headers
        function[cb_middleware_call_completed] call_completed

    cdef cppclass CPyServerMiddleware\
            " arrow::py::flight::PyServerMiddleware"(CServerMiddleware):
        CPyServerMiddleware(object middleware, PyServerMiddlewareVtable vtable)
        void* py_object()

    cdef cppclass CPyServerMiddlewareFactory\
            " arrow::py::flight::PyServerMiddlewareFactory"\
            (CServerMiddlewareFactory):
        CPyServerMiddlewareFactory(
            object factory,
            function[cb_server_middleware_start_call] start_call)

    cdef cppclass CPyClientMiddleware\
            " arrow::py::flight::PyClientMiddleware"(CClientMiddleware):
        CPyClientMiddleware(object middleware, PyClientMiddlewareVtable vtable)

    cdef cppclass CPyClientMiddlewareFactory\
            " arrow::py::flight::PyClientMiddlewareFactory"\
            (CClientMiddlewareFactory):
        CPyClientMiddlewareFactory(
            object factory,
            function[cb_client_middleware_start_call] start_call)

    cdef CStatus CreateFlightInfo" arrow::py::flight::CreateFlightInfo"(
        shared_ptr[CSchema] schema,
        CFlightDescriptor& descriptor,
        vector[CFlightEndpoint] endpoints,
        int64_t total_records,
        int64_t total_bytes,
        unique_ptr[CFlightInfo]* out)

    cdef CStatus CreateSchemaResult" arrow::py::flight::CreateSchemaResult"(
        shared_ptr[CSchema] schema,
        unique_ptr[CSchemaResult]* out)


cdef extern from "<variant>" namespace "std" nogil:
    cdef cppclass CIntStringVariant" std::variant<int, std::string>":
        CIntStringVariant()
        CIntStringVariant(int)
        CIntStringVariant(c_string)

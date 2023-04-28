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

namespace arrow {
namespace internal {
class Uri;
}
namespace flight {
struct Action;
struct ActionType;
struct BasicAuth;
class ClientAuthHandler;
class ClientMiddleware;
class ClientMiddlewareFactory;
struct Criteria;
class FlightCallOptions;
struct FlightClientOptions;
struct FlightDescriptor;
struct FlightEndpoint;
class FlightInfo;
class FlightListing;
class FlightMetadataReader;
class FlightMetadataWriter;
struct FlightPayload;
class FlightServerBase;
class FlightServerOptions;
class FlightStreamReader;
class FlightStreamWriter;
struct Location;
struct Result;
class ResultStream;
struct SchemaResult;
class ServerCallContext;
class ServerMiddleware;
class ServerMiddlewareFactory;
struct Ticket;
namespace internal {
class ClientTransport;
struct FlightData;
class ServerTransport;
}  // namespace internal
}  // namespace flight
}  // namespace arrow

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

// Interfaces for defining middleware for Flight clients and
// servers. Currently experimental.

#pragma once

#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "arrow/flight/visibility.h"  // IWYU pragma: keep
#include "arrow/status.h"

namespace arrow {
namespace flight {

/// \brief Headers sent from the client or server.
///
/// Header values are ordered.
using CallHeaders = std::multimap<std::string_view, std::string_view>;

/// \brief A write-only wrapper around headers for an RPC call.
class ARROW_FLIGHT_EXPORT AddCallHeaders {
 public:
  virtual ~AddCallHeaders() = default;

  /// \brief Add a header to be sent to the client.
  ///
  /// \param[in] key The header name. Must be lowercase ASCII; some
  ///   transports may reject invalid header names.
  /// \param[in] value The header value. Some transports may only
  ///   accept binary header values if the header name ends in "-bin".
  virtual void AddHeader(const std::string& key, const std::string& value) = 0;
};

/// \brief An enumeration of the RPC methods Flight implements.
enum class FlightMethod : char {
  Invalid = 0,
  Handshake = 1,
  ListFlights = 2,
  GetFlightInfo = 3,
  GetSchema = 4,
  DoGet = 5,
  DoPut = 6,
  DoAction = 7,
  ListActions = 8,
  DoExchange = 9,
};

/// \brief Get a human-readable name for a Flight method.
ARROW_FLIGHT_EXPORT
std::string ToString(FlightMethod method);

/// \brief Information about an instance of a Flight RPC.
struct ARROW_FLIGHT_EXPORT CallInfo {
 public:
  /// \brief The RPC method of this call.
  FlightMethod method;
};

}  // namespace flight
}  // namespace arrow

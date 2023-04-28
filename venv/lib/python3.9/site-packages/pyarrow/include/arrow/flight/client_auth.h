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

#include <string>

#include "arrow/flight/visibility.h"
#include "arrow/status.h"

namespace arrow {

namespace flight {

/// \brief A reader for messages from the server during an
/// authentication handshake.
class ARROW_FLIGHT_EXPORT ClientAuthReader {
 public:
  virtual ~ClientAuthReader() = default;
  virtual Status Read(std::string* response) = 0;
};

/// \brief A writer for messages to the server during an
/// authentication handshake.
class ARROW_FLIGHT_EXPORT ClientAuthSender {
 public:
  virtual ~ClientAuthSender() = default;
  virtual Status Write(const std::string& token) = 0;
};

/// \brief An authentication implementation for a Flight service.
/// Authentication includes both an initial negotiation and a per-call
/// token validation. Implementations may choose to use either or both
/// mechanisms.
class ARROW_FLIGHT_EXPORT ClientAuthHandler {
 public:
  virtual ~ClientAuthHandler() = default;
  /// \brief Authenticate the client on initial connection. The client
  /// can send messages to/read responses from the server at any time.
  /// \return Status OK if authenticated successfully
  virtual Status Authenticate(ClientAuthSender* outgoing, ClientAuthReader* incoming) = 0;
  /// \brief Get a per-call token.
  /// \param[out] token The token to send to the server.
  virtual Status GetToken(std::string* token) = 0;
};

}  // namespace flight
}  // namespace arrow

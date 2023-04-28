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

/// \brief Server-side APIs to implement authentication for Flight.

#pragma once

#include <string>

#include "arrow/flight/visibility.h"
#include "arrow/status.h"

namespace arrow {

namespace flight {

/// \brief A reader for messages from the client during an
/// authentication handshake.
class ARROW_FLIGHT_EXPORT ServerAuthReader {
 public:
  virtual ~ServerAuthReader() = default;
  virtual Status Read(std::string* token) = 0;
};

/// \brief A writer for messages to the client during an
/// authentication handshake.
class ARROW_FLIGHT_EXPORT ServerAuthSender {
 public:
  virtual ~ServerAuthSender() = default;
  virtual Status Write(const std::string& message) = 0;
};

/// \brief An authentication implementation for a Flight service.
/// Authentication includes both an initial negotiation and a per-call
/// token validation. Implementations may choose to use either or both
/// mechanisms.
/// An implementation may need to track some state, e.g. a mapping of
/// client tokens to authenticated identities.
class ARROW_FLIGHT_EXPORT ServerAuthHandler {
 public:
  virtual ~ServerAuthHandler();
  /// \brief Authenticate the client on initial connection. The server
  /// can send and read responses from the client at any time.
  virtual Status Authenticate(ServerAuthSender* outgoing, ServerAuthReader* incoming) = 0;
  /// \brief Validate a per-call client token.
  /// \param[in] token The client token. May be the empty string if
  /// the client does not provide a token.
  /// \param[out] peer_identity The identity of the peer, if this
  /// authentication method supports it.
  /// \return Status OK if the token is valid, any other status if
  /// validation failed
  virtual Status IsValid(const std::string& token, std::string* peer_identity) = 0;
};

/// \brief An authentication mechanism that does nothing.
class ARROW_FLIGHT_EXPORT NoOpAuthHandler : public ServerAuthHandler {
 public:
  ~NoOpAuthHandler() override;
  Status Authenticate(ServerAuthSender* outgoing, ServerAuthReader* incoming) override;
  Status IsValid(const std::string& token, std::string* peer_identity) override;
};

}  // namespace flight
}  // namespace arrow

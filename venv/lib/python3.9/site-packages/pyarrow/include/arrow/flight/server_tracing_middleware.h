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

// Middleware implementation for propagating OpenTelemetry spans.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "arrow/flight/server_middleware.h"
#include "arrow/flight/visibility.h"
#include "arrow/status.h"

namespace arrow {
namespace flight {

/// \brief Returns a ServerMiddlewareFactory that handles receiving OpenTelemetry spans.
ARROW_FLIGHT_EXPORT std::shared_ptr<ServerMiddlewareFactory>
MakeTracingServerMiddlewareFactory();

/// \brief A server middleware that provides access to the
///   OpenTelemetry context, if present.
///
/// Used to make the OpenTelemetry span available in Python.
class ARROW_FLIGHT_EXPORT TracingServerMiddleware : public ServerMiddleware {
 public:
  ~TracingServerMiddleware();

  static constexpr char const kMiddlewareName[] =
      "arrow::flight::TracingServerMiddleware";

  std::string name() const override { return kMiddlewareName; }
  void SendingHeaders(AddCallHeaders*) override;
  void CallCompleted(const Status&) override;

  struct TraceKey {
    std::string key;
    std::string value;
  };
  /// \brief Get the trace context.
  std::vector<TraceKey> GetTraceContext() const;

 private:
  class Impl;
  friend class TracingServerMiddlewareFactory;

  explicit TracingServerMiddleware(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace flight
}  // namespace arrow

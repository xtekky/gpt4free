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

// Data structure for Flight RPC. API should be considered experimental for now

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "arrow/flight/visibility.h"
#include "arrow/ipc/options.h"
#include "arrow/ipc/writer.h"
#include "arrow/result.h"

namespace arrow {

class Buffer;
class RecordBatch;
class Schema;
class Status;
class Table;

namespace ipc {

class DictionaryMemo;

}  // namespace ipc

namespace internal {

class Uri;

}  // namespace internal

namespace flight {

/// \brief A Flight-specific status code.
enum class FlightStatusCode : int8_t {
  /// An implementation error has occurred.
  Internal,
  /// A request timed out.
  TimedOut,
  /// A request was cancelled.
  Cancelled,
  /// We are not authenticated to the remote service.
  Unauthenticated,
  /// We do not have permission to make this request.
  Unauthorized,
  /// The remote service cannot handle this request at the moment.
  Unavailable,
  /// A request failed for some other reason
  Failed
};

// Silence warning
// "non dll-interface class RecordBatchReader used as base for dll-interface class"
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4275)
#endif

/// \brief Flight-specific error information in a Status.
class ARROW_FLIGHT_EXPORT FlightStatusDetail : public arrow::StatusDetail {
 public:
  explicit FlightStatusDetail(FlightStatusCode code) : code_{code} {}
  explicit FlightStatusDetail(FlightStatusCode code, std::string extra_info)
      : code_{code}, extra_info_(std::move(extra_info)) {}
  const char* type_id() const override;
  std::string ToString() const override;

  /// \brief Get the Flight status code.
  FlightStatusCode code() const;
  /// \brief Get the extra error info
  std::string extra_info() const;
  /// \brief Get the human-readable name of the status code.
  std::string CodeAsString() const;
  /// \brief Set the extra error info
  void set_extra_info(std::string extra_info);

  /// \brief Try to extract a \a FlightStatusDetail from any Arrow
  /// status.
  ///
  /// \return a \a FlightStatusDetail if it could be unwrapped, \a
  /// nullptr otherwise
  static std::shared_ptr<FlightStatusDetail> UnwrapStatus(const arrow::Status& status);

 private:
  FlightStatusCode code_;
  std::string extra_info_;
};

#ifdef _MSC_VER
#pragma warning(pop)
#endif

/// \brief Make an appropriate Arrow status for the given
/// Flight-specific status.
///
/// \param code The Flight status code.
/// \param message The message for the error.
/// \param extra_info Optional extra binary info for the error (eg protobuf)
ARROW_FLIGHT_EXPORT
Status MakeFlightError(FlightStatusCode code, std::string message,
                       std::string extra_info = {});

/// \brief A TLS certificate plus key.
struct ARROW_FLIGHT_EXPORT CertKeyPair {
  /// \brief The certificate in PEM format.
  std::string pem_cert;

  /// \brief The key in PEM format.
  std::string pem_key;
};

/// \brief A type of action that can be performed with the DoAction RPC.
struct ARROW_FLIGHT_EXPORT ActionType {
  /// \brief The name of the action.
  std::string type;

  /// \brief A human-readable description of the action.
  std::string description;

  bool Equals(const ActionType& other) const;

  friend bool operator==(const ActionType& left, const ActionType& right) {
    return left.Equals(right);
  }
  friend bool operator!=(const ActionType& left, const ActionType& right) {
    return !(left == right);
  }

  /// \brief Serialize this message to its wire-format representation.
  arrow::Result<std::string> SerializeToString() const;

  /// \brief Deserialize this message from its wire-format representation.
  static arrow::Result<ActionType> Deserialize(std::string_view serialized);
};

/// \brief Opaque selection criteria for ListFlights RPC
struct ARROW_FLIGHT_EXPORT Criteria {
  /// Opaque criteria expression, dependent on server implementation
  std::string expression;

  bool Equals(const Criteria& other) const;

  friend bool operator==(const Criteria& left, const Criteria& right) {
    return left.Equals(right);
  }
  friend bool operator!=(const Criteria& left, const Criteria& right) {
    return !(left == right);
  }

  /// \brief Serialize this message to its wire-format representation.
  arrow::Result<std::string> SerializeToString() const;

  /// \brief Deserialize this message from its wire-format representation.
  static arrow::Result<Criteria> Deserialize(std::string_view serialized);
};

/// \brief An action to perform with the DoAction RPC
struct ARROW_FLIGHT_EXPORT Action {
  /// The action type
  std::string type;

  /// The action content as a Buffer
  std::shared_ptr<Buffer> body;

  bool Equals(const Action& other) const;

  friend bool operator==(const Action& left, const Action& right) {
    return left.Equals(right);
  }
  friend bool operator!=(const Action& left, const Action& right) {
    return !(left == right);
  }

  /// \brief Serialize this message to its wire-format representation.
  arrow::Result<std::string> SerializeToString() const;

  /// \brief Deserialize this message from its wire-format representation.
  static arrow::Result<Action> Deserialize(std::string_view serialized);
};

/// \brief Opaque result returned after executing an action
struct ARROW_FLIGHT_EXPORT Result {
  std::shared_ptr<Buffer> body;

  bool Equals(const Result& other) const;

  friend bool operator==(const Result& left, const Result& right) {
    return left.Equals(right);
  }
  friend bool operator!=(const Result& left, const Result& right) {
    return !(left == right);
  }

  /// \brief Serialize this message to its wire-format representation.
  arrow::Result<std::string> SerializeToString() const;

  /// \brief Deserialize this message from its wire-format representation.
  static arrow::Result<Result> Deserialize(std::string_view serialized);
};

/// \brief message for simple auth
struct ARROW_FLIGHT_EXPORT BasicAuth {
  std::string username;
  std::string password;

  bool Equals(const BasicAuth& other) const;

  friend bool operator==(const BasicAuth& left, const BasicAuth& right) {
    return left.Equals(right);
  }
  friend bool operator!=(const BasicAuth& left, const BasicAuth& right) {
    return !(left == right);
  }

  /// \brief Deserialize this message from its wire-format representation.
  static arrow::Result<BasicAuth> Deserialize(std::string_view serialized);
  /// \brief Serialize this message to its wire-format representation.
  arrow::Result<std::string> SerializeToString() const;

  ARROW_DEPRECATED("Deprecated in 8.0.0. Use Result-returning overload instead.")
  static Status Deserialize(const std::string& serialized, BasicAuth* out);

  ARROW_DEPRECATED("Deprecated in 8.0.0. Use Result-returning overload instead.")
  static Status Serialize(const BasicAuth& basic_auth, std::string* out);
};

/// \brief A request to retrieve or generate a dataset
struct ARROW_FLIGHT_EXPORT FlightDescriptor {
  enum DescriptorType {
    UNKNOWN = 0,  /// Unused
    PATH = 1,     /// Named path identifying a dataset
    CMD = 2       /// Opaque command to generate a dataset
  };

  /// The descriptor type
  DescriptorType type;

  /// Opaque value used to express a command. Should only be defined when type
  /// is CMD
  std::string cmd;

  /// List of strings identifying a particular dataset. Should only be defined
  /// when type is PATH
  std::vector<std::string> path;

  bool Equals(const FlightDescriptor& other) const;

  /// \brief Get a human-readable form of this descriptor.
  std::string ToString() const;

  /// \brief Get the wire-format representation of this type.
  ///
  /// Useful when interoperating with non-Flight systems (e.g. REST
  /// services) that may want to return Flight types.
  arrow::Result<std::string> SerializeToString() const;

  ARROW_DEPRECATED("Deprecated in 8.0.0. Use Result-returning overload instead.")
  Status SerializeToString(std::string* out) const;

  /// \brief Parse the wire-format representation of this type.
  ///
  /// Useful when interoperating with non-Flight systems (e.g. REST
  /// services) that may want to return Flight types.
  static arrow::Result<FlightDescriptor> Deserialize(std::string_view serialized);

  ARROW_DEPRECATED("Deprecated in 8.0.0. Use Result-returning overload instead.")
  static Status Deserialize(const std::string& serialized, FlightDescriptor* out);

  // Convenience factory functions

  static FlightDescriptor Command(const std::string& c) {
    return FlightDescriptor{CMD, c, {}};
  }

  static FlightDescriptor Path(const std::vector<std::string>& p) {
    return FlightDescriptor{PATH, "", p};
  }

  friend bool operator==(const FlightDescriptor& left, const FlightDescriptor& right) {
    return left.Equals(right);
  }
  friend bool operator!=(const FlightDescriptor& left, const FlightDescriptor& right) {
    return !(left == right);
  }
};

/// \brief Data structure providing an opaque identifier or credential to use
/// when requesting a data stream with the DoGet RPC
struct ARROW_FLIGHT_EXPORT Ticket {
  std::string ticket;

  bool Equals(const Ticket& other) const;

  friend bool operator==(const Ticket& left, const Ticket& right) {
    return left.Equals(right);
  }
  friend bool operator!=(const Ticket& left, const Ticket& right) {
    return !(left == right);
  }

  /// \brief Get the wire-format representation of this type.
  ///
  /// Useful when interoperating with non-Flight systems (e.g. REST
  /// services) that may want to return Flight types.
  arrow::Result<std::string> SerializeToString() const;

  ARROW_DEPRECATED("Deprecated in 8.0.0. Use Result-returning overload instead.")
  Status SerializeToString(std::string* out) const;

  /// \brief Parse the wire-format representation of this type.
  ///
  /// Useful when interoperating with non-Flight systems (e.g. REST
  /// services) that may want to return Flight types.
  static arrow::Result<Ticket> Deserialize(std::string_view serialized);

  ARROW_DEPRECATED("Deprecated in 8.0.0. Use Result-returning overload instead.")
  static Status Deserialize(const std::string& serialized, Ticket* out);
};

class FlightClient;
class FlightServerBase;

ARROW_FLIGHT_EXPORT
extern const char* kSchemeGrpc;
ARROW_FLIGHT_EXPORT
extern const char* kSchemeGrpcTcp;
ARROW_FLIGHT_EXPORT
extern const char* kSchemeGrpcUnix;
ARROW_FLIGHT_EXPORT
extern const char* kSchemeGrpcTls;

/// \brief A host location (a URI)
struct ARROW_FLIGHT_EXPORT Location {
 public:
  /// \brief Initialize a blank location.
  Location();

  /// \brief Initialize a location by parsing a URI string
  static arrow::Result<Location> Parse(const std::string& uri_string);

  ARROW_DEPRECATED("Deprecated in 8.0.0. Use Result-returning overload instead.")
  static Status Parse(const std::string& uri_string, Location* location);

  /// \brief Initialize a location for a non-TLS, gRPC-based Flight
  /// service from a host and port
  /// \param[in] host The hostname to connect to
  /// \param[in] port The port
  /// \return Arrow result with the resulting location
  static arrow::Result<Location> ForGrpcTcp(const std::string& host, const int port);

  ARROW_DEPRECATED("Deprecated in 8.0.0. Use Result-returning overload instead.")
  static Status ForGrpcTcp(const std::string& host, const int port, Location* location);

  /// \brief Initialize a location for a TLS-enabled, gRPC-based Flight
  /// service from a host and port
  /// \param[in] host The hostname to connect to
  /// \param[in] port The port
  /// \return Arrow result with the resulting location
  static arrow::Result<Location> ForGrpcTls(const std::string& host, const int port);

  ARROW_DEPRECATED("Deprecated in 8.0.0. Use Result-returning overload instead.")
  static Status ForGrpcTls(const std::string& host, const int port, Location* location);

  /// \brief Initialize a location for a domain socket-based Flight
  /// service
  /// \param[in] path The path to the domain socket
  /// \return Arrow result with the resulting location
  static arrow::Result<Location> ForGrpcUnix(const std::string& path);

  ARROW_DEPRECATED("Deprecated in 8.0.0. Use Result-returning overload instead.")
  static Status ForGrpcUnix(const std::string& path, Location* location);

  /// \brief Initialize a location based on a URI scheme
  static arrow::Result<Location> ForScheme(const std::string& scheme,
                                           const std::string& host, const int port);

  /// \brief Get a representation of this URI as a string.
  std::string ToString() const;

  /// \brief Get the scheme of this URI.
  std::string scheme() const;

  bool Equals(const Location& other) const;

  friend bool operator==(const Location& left, const Location& right) {
    return left.Equals(right);
  }
  friend bool operator!=(const Location& left, const Location& right) {
    return !(left == right);
  }

 private:
  friend class FlightClient;
  friend class FlightServerBase;
  std::shared_ptr<arrow::internal::Uri> uri_;
};

/// \brief A flight ticket and list of locations where the ticket can be
/// redeemed
struct ARROW_FLIGHT_EXPORT FlightEndpoint {
  /// Opaque ticket identify; use with DoGet RPC
  Ticket ticket;

  /// List of locations where ticket can be redeemed. If the list is empty, the
  /// ticket can only be redeemed on the current service where the ticket was
  /// generated
  std::vector<Location> locations;

  bool Equals(const FlightEndpoint& other) const;

  friend bool operator==(const FlightEndpoint& left, const FlightEndpoint& right) {
    return left.Equals(right);
  }
  friend bool operator!=(const FlightEndpoint& left, const FlightEndpoint& right) {
    return !(left == right);
  }

  /// \brief Serialize this message to its wire-format representation.
  arrow::Result<std::string> SerializeToString() const;

  /// \brief Deserialize this message from its wire-format representation.
  static arrow::Result<FlightEndpoint> Deserialize(std::string_view serialized);
};

/// \brief Staging data structure for messages about to be put on the wire
///
/// This structure corresponds to FlightData in the protocol.
struct ARROW_FLIGHT_EXPORT FlightPayload {
  std::shared_ptr<Buffer> descriptor;
  std::shared_ptr<Buffer> app_metadata;
  ipc::IpcPayload ipc_message;

  /// \brief Check that the payload can be written to the wire.
  Status Validate() const;
};

/// \brief Schema result returned after a schema request RPC
struct ARROW_FLIGHT_EXPORT SchemaResult {
 public:
  SchemaResult() = default;
  explicit SchemaResult(std::string schema) : raw_schema_(std::move(schema)) {}

  /// \brief Factory method to construct a SchemaResult.
  static arrow::Result<std::unique_ptr<SchemaResult>> Make(const Schema& schema);

  /// \brief return schema
  /// \param[in,out] dictionary_memo for dictionary bookkeeping, will
  /// be modified
  /// \return Arrrow result with the reconstructed Schema
  arrow::Result<std::shared_ptr<Schema>> GetSchema(
      ipc::DictionaryMemo* dictionary_memo) const;

  ARROW_DEPRECATED("Deprecated in 8.0.0. Use Result-returning overload instead.")
  Status GetSchema(ipc::DictionaryMemo* dictionary_memo,
                   std::shared_ptr<Schema>* out) const;

  const std::string& serialized_schema() const { return raw_schema_; }

  bool Equals(const SchemaResult& other) const;

  friend bool operator==(const SchemaResult& left, const SchemaResult& right) {
    return left.Equals(right);
  }
  friend bool operator!=(const SchemaResult& left, const SchemaResult& right) {
    return !(left == right);
  }

  /// \brief Serialize this message to its wire-format representation.
  arrow::Result<std::string> SerializeToString() const;

  /// \brief Deserialize this message from its wire-format representation.
  static arrow::Result<SchemaResult> Deserialize(std::string_view serialized);

 private:
  std::string raw_schema_;
};

/// \brief The access coordinates for retireval of a dataset, returned by
/// GetFlightInfo
class ARROW_FLIGHT_EXPORT FlightInfo {
 public:
  struct Data {
    std::string schema;
    FlightDescriptor descriptor;
    std::vector<FlightEndpoint> endpoints;
    int64_t total_records;
    int64_t total_bytes;
  };

  explicit FlightInfo(const Data& data) : data_(data), reconstructed_schema_(false) {}
  explicit FlightInfo(Data&& data)
      : data_(std::move(data)), reconstructed_schema_(false) {}

  /// \brief Factory method to construct a FlightInfo.
  static arrow::Result<FlightInfo> Make(const Schema& schema,
                                        const FlightDescriptor& descriptor,
                                        const std::vector<FlightEndpoint>& endpoints,
                                        int64_t total_records, int64_t total_bytes);

  /// \brief Deserialize the Arrow schema of the dataset. Populate any
  ///   dictionary encoded fields into a DictionaryMemo for
  ///   bookkeeping
  /// \param[in,out] dictionary_memo for dictionary bookkeeping, will
  /// be modified
  /// \return Arrrow result with the reconstructed Schema
  arrow::Result<std::shared_ptr<Schema>> GetSchema(
      ipc::DictionaryMemo* dictionary_memo) const;

  ARROW_DEPRECATED("Deprecated in 8.0.0. Use Result-returning overload instead.")
  Status GetSchema(ipc::DictionaryMemo* dictionary_memo,
                   std::shared_ptr<Schema>* out) const;

  const std::string& serialized_schema() const { return data_.schema; }

  /// The descriptor associated with this flight, may not be set
  const FlightDescriptor& descriptor() const { return data_.descriptor; }

  /// A list of endpoints associated with the flight (dataset). To consume the
  /// whole flight, all endpoints must be consumed
  const std::vector<FlightEndpoint>& endpoints() const { return data_.endpoints; }

  /// The total number of records (rows) in the dataset. If unknown, set to -1
  int64_t total_records() const { return data_.total_records; }

  /// The total number of bytes in the dataset. If unknown, set to -1
  int64_t total_bytes() const { return data_.total_bytes; }

  /// \brief Get the wire-format representation of this type.
  ///
  /// Useful when interoperating with non-Flight systems (e.g. REST
  /// services) that may want to return Flight types.
  arrow::Result<std::string> SerializeToString() const;

  ARROW_DEPRECATED("Deprecated in 8.0.0. Use Result-returning overload instead.")
  Status SerializeToString(std::string* out) const;

  /// \brief Parse the wire-format representation of this type.
  ///
  /// Useful when interoperating with non-Flight systems (e.g. REST
  /// services) that may want to return Flight types.
  static arrow::Result<std::unique_ptr<FlightInfo>> Deserialize(
      std::string_view serialized);

  ARROW_DEPRECATED("Deprecated in 8.0.0. Use Result-returning overload instead.")
  static Status Deserialize(const std::string& serialized,
                            std::unique_ptr<FlightInfo>* out);

 private:
  Data data_;
  mutable std::shared_ptr<Schema> schema_;
  mutable bool reconstructed_schema_;
};

/// \brief An iterator to FlightInfo instances returned by ListFlights.
class ARROW_FLIGHT_EXPORT FlightListing {
 public:
  virtual ~FlightListing() = default;

  /// \brief Retrieve the next FlightInfo from the iterator.
  /// \return Arrow result with a single FlightInfo. Set to \a nullptr if there
  /// are none left.
  virtual arrow::Result<std::unique_ptr<FlightInfo>> Next() = 0;

  ARROW_DEPRECATED("Deprecated in 8.0.0. Use Result-returning overload instead.")
  Status Next(std::unique_ptr<FlightInfo>* info);
};

/// \brief An iterator to Result instances returned by DoAction.
class ARROW_FLIGHT_EXPORT ResultStream {
 public:
  virtual ~ResultStream() = default;

  /// \brief Retrieve the next Result from the iterator.
  /// \return Arrow result with a single Result. Set to \a nullptr if there are none left.
  virtual arrow::Result<std::unique_ptr<Result>> Next() = 0;

  ARROW_DEPRECATED("Deprecated in 8.0.0. Use Result-returning overload instead.")
  Status Next(std::unique_ptr<Result>* info);
};

/// \brief A holder for a RecordBatch with associated Flight metadata.
struct ARROW_FLIGHT_EXPORT FlightStreamChunk {
 public:
  std::shared_ptr<RecordBatch> data;
  std::shared_ptr<Buffer> app_metadata;
};

/// \brief An interface to read Flight data with metadata.
class ARROW_FLIGHT_EXPORT MetadataRecordBatchReader {
 public:
  virtual ~MetadataRecordBatchReader() = default;

  /// \brief Get the schema for this stream.
  virtual arrow::Result<std::shared_ptr<Schema>> GetSchema() = 0;

  /// \brief Get the next message from Flight. If the stream is
  /// finished, then the members of \a FlightStreamChunk will be
  /// nullptr.
  virtual arrow::Result<FlightStreamChunk> Next() = 0;

  ARROW_DEPRECATED("Deprecated in 8.0.0. Use Result-returning overload instead.")
  Status Next(FlightStreamChunk* next);

  /// \brief Consume entire stream as a vector of record batches
  virtual arrow::Result<std::vector<std::shared_ptr<RecordBatch>>> ToRecordBatches();

  ARROW_DEPRECATED("Deprecated in 8.0.0. Use ToRecordBatches instead.")
  Status ReadAll(std::vector<std::shared_ptr<RecordBatch>>* batches);

  /// \brief Consume entire stream as a Table
  virtual arrow::Result<std::shared_ptr<Table>> ToTable();

  ARROW_DEPRECATED("Deprecated in 8.0.0. Use ToTable instead.")
  Status ReadAll(std::shared_ptr<Table>* table);
};

/// \brief Convert a MetadataRecordBatchReader to a regular RecordBatchReader.
ARROW_FLIGHT_EXPORT
arrow::Result<std::shared_ptr<RecordBatchReader>> MakeRecordBatchReader(
    std::shared_ptr<MetadataRecordBatchReader> reader);

/// \brief An interface to write IPC payloads with metadata.
class ARROW_FLIGHT_EXPORT MetadataRecordBatchWriter : public ipc::RecordBatchWriter {
 public:
  virtual ~MetadataRecordBatchWriter() = default;
  /// \brief Begin writing data with the given schema. Only used with \a DoExchange.
  virtual Status Begin(const std::shared_ptr<Schema>& schema,
                       const ipc::IpcWriteOptions& options) = 0;
  virtual Status Begin(const std::shared_ptr<Schema>& schema);
  virtual Status WriteMetadata(std::shared_ptr<Buffer> app_metadata) = 0;
  virtual Status WriteWithMetadata(const RecordBatch& batch,
                                   std::shared_ptr<Buffer> app_metadata) = 0;
};

/// \brief A FlightListing implementation based on a vector of
/// FlightInfo objects.
///
/// This can be iterated once, then it is consumed.
class ARROW_FLIGHT_EXPORT SimpleFlightListing : public FlightListing {
 public:
  explicit SimpleFlightListing(const std::vector<FlightInfo>& flights);
  explicit SimpleFlightListing(std::vector<FlightInfo>&& flights);

  arrow::Result<std::unique_ptr<FlightInfo>> Next() override;

 private:
  int position_;
  std::vector<FlightInfo> flights_;
};

/// \brief A ResultStream implementation based on a vector of
/// Result objects.
///
/// This can be iterated once, then it is consumed.
class ARROW_FLIGHT_EXPORT SimpleResultStream : public ResultStream {
 public:
  explicit SimpleResultStream(std::vector<Result>&& results);
  arrow::Result<std::unique_ptr<Result>> Next() override;

 private:
  std::vector<Result> results_;
  size_t position_;
};

}  // namespace flight
}  // namespace arrow

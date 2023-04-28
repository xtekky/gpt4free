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

// Common test definitions for Flight. Individual transport
// implementations can instantiate these tests.
//
// While Googletest's value-parameterized tests would be a more
// natural way to do this, they cause runtime issues on MinGW/MSVC
// (Googletest thinks the test suite has been defined twice).

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "arrow/flight/server.h"
#include "arrow/flight/types.h"
#include "arrow/util/macros.h"

namespace arrow {
namespace flight {

class ARROW_FLIGHT_EXPORT FlightTest {
 protected:
  virtual std::string transport() const = 0;
  virtual void SetUpTest() {}
  virtual void TearDownTest() {}
};

/// Common tests of startup/shutdown
class ARROW_FLIGHT_EXPORT ConnectivityTest : public FlightTest {
 public:
  // Test methods
  void TestGetPort();
  void TestBuilderHook();
  void TestShutdown();
  void TestShutdownWithDeadline();
  void TestBrokenConnection();
};

#define ARROW_FLIGHT_TEST_CONNECTIVITY(FIXTURE)                                  \
  static_assert(std::is_base_of<ConnectivityTest, FIXTURE>::value,               \
                ARROW_STRINGIFY(FIXTURE) " must inherit from ConnectivityTest"); \
  TEST_F(FIXTURE, GetPort) { TestGetPort(); }                                    \
  TEST_F(FIXTURE, BuilderHook) { TestBuilderHook(); }                            \
  TEST_F(FIXTURE, Shutdown) { TestShutdown(); }                                  \
  TEST_F(FIXTURE, ShutdownWithDeadline) { TestShutdownWithDeadline(); }          \
  TEST_F(FIXTURE, BrokenConnection) { TestBrokenConnection(); }

/// Common tests of data plane methods
class ARROW_FLIGHT_EXPORT DataTest : public FlightTest {
 public:
  void SetUpTest() override;
  void TearDownTest() override;
  Status ConnectClient();

  // Test methods
  void TestDoGetInts();
  void TestDoGetFloats();
  void TestDoGetDicts();
  void TestDoGetLargeBatch();
  void TestFlightDataStreamError();
  void TestOverflowServerBatch();
  void TestOverflowClientBatch();
  void TestDoExchange();
  void TestDoExchangeNoData();
  void TestDoExchangeWriteOnlySchema();
  void TestDoExchangeGet();
  void TestDoExchangePut();
  void TestDoExchangeEcho();
  void TestDoExchangeTotal();
  void TestDoExchangeError();
  void TestDoExchangeConcurrency();
  void TestDoExchangeUndrained();
  void TestIssue5095();

 private:
  void CheckDoGet(
      const FlightDescriptor& descr, const RecordBatchVector& expected_batches,
      std::function<void(const std::vector<FlightEndpoint>&)> check_endpoints);
  void CheckDoGet(const Ticket& ticket, const RecordBatchVector& expected_batches);

  std::unique_ptr<FlightClient> client_;
  std::unique_ptr<FlightServerBase> server_;
};

#define ARROW_FLIGHT_TEST_DATA(FIXTURE)                                               \
  static_assert(std::is_base_of<DataTest, FIXTURE>::value,                            \
                ARROW_STRINGIFY(FIXTURE) " must inherit from DataTest");              \
  TEST_F(FIXTURE, TestDoGetInts) { TestDoGetInts(); }                                 \
  TEST_F(FIXTURE, TestDoGetFloats) { TestDoGetFloats(); }                             \
  TEST_F(FIXTURE, TestDoGetDicts) { TestDoGetDicts(); }                               \
  TEST_F(FIXTURE, TestDoGetLargeBatch) { TestDoGetLargeBatch(); }                     \
  TEST_F(FIXTURE, TestFlightDataStreamError) { TestFlightDataStreamError(); }         \
  TEST_F(FIXTURE, TestOverflowServerBatch) { TestOverflowServerBatch(); }             \
  TEST_F(FIXTURE, TestOverflowClientBatch) { TestOverflowClientBatch(); }             \
  TEST_F(FIXTURE, TestDoExchange) { TestDoExchange(); }                               \
  TEST_F(FIXTURE, TestDoExchangeNoData) { TestDoExchangeNoData(); }                   \
  TEST_F(FIXTURE, TestDoExchangeWriteOnlySchema) { TestDoExchangeWriteOnlySchema(); } \
  TEST_F(FIXTURE, TestDoExchangeGet) { TestDoExchangeGet(); }                         \
  TEST_F(FIXTURE, TestDoExchangePut) { TestDoExchangePut(); }                         \
  TEST_F(FIXTURE, TestDoExchangeEcho) { TestDoExchangeEcho(); }                       \
  TEST_F(FIXTURE, TestDoExchangeTotal) { TestDoExchangeTotal(); }                     \
  TEST_F(FIXTURE, TestDoExchangeError) { TestDoExchangeError(); }                     \
  TEST_F(FIXTURE, TestDoExchangeConcurrency) { TestDoExchangeConcurrency(); }         \
  TEST_F(FIXTURE, TestDoExchangeUndrained) { TestDoExchangeUndrained(); }             \
  TEST_F(FIXTURE, TestIssue5095) { TestIssue5095(); }

/// \brief Specific tests of DoPut.
class ARROW_FLIGHT_EXPORT DoPutTest : public FlightTest {
 public:
  void SetUpTest() override;
  void TearDownTest() override;
  void CheckBatches(const FlightDescriptor& expected_descriptor,
                    const RecordBatchVector& expected_batches);
  void CheckDoPut(const FlightDescriptor& descr, const std::shared_ptr<Schema>& schema,
                  const RecordBatchVector& batches);

  // Test methods
  void TestInts();
  void TestFloats();
  void TestEmptyBatch();
  void TestDicts();
  void TestLargeBatch();
  void TestSizeLimit();
  void TestUndrained();

 private:
  std::unique_ptr<FlightClient> client_;
  std::unique_ptr<FlightServerBase> server_;
};

#define ARROW_FLIGHT_TEST_DO_PUT(FIXTURE)                                 \
  static_assert(std::is_base_of<DoPutTest, FIXTURE>::value,               \
                ARROW_STRINGIFY(FIXTURE) " must inherit from DoPutTest"); \
  TEST_F(FIXTURE, TestInts) { TestInts(); }                               \
  TEST_F(FIXTURE, TestFloats) { TestFloats(); }                           \
  TEST_F(FIXTURE, TestEmptyBatch) { TestEmptyBatch(); }                   \
  TEST_F(FIXTURE, TestDicts) { TestDicts(); }                             \
  TEST_F(FIXTURE, TestLargeBatch) { TestLargeBatch(); }                   \
  TEST_F(FIXTURE, TestSizeLimit) { TestSizeLimit(); }                     \
  TEST_F(FIXTURE, TestUndrained) { TestUndrained(); }

class ARROW_FLIGHT_EXPORT AppMetadataTestServer : public FlightServerBase {
 public:
  virtual ~AppMetadataTestServer() = default;

  Status DoGet(const ServerCallContext& context, const Ticket& request,
               std::unique_ptr<FlightDataStream>* data_stream) override;

  Status DoPut(const ServerCallContext& context,
               std::unique_ptr<FlightMessageReader> reader,
               std::unique_ptr<FlightMetadataWriter> writer) override;
};

/// \brief Tests of app_metadata in data plane methods.
class ARROW_FLIGHT_EXPORT AppMetadataTest : public FlightTest {
 public:
  void SetUpTest() override;
  void TearDownTest() override;

  // Test methods
  void TestDoGet();
  void TestDoGetDictionaries();
  void TestDoPut();
  void TestDoPutDictionaries();
  void TestDoPutReadMetadata();

 private:
  std::unique_ptr<FlightClient> client_;
  std::unique_ptr<FlightServerBase> server_;
};

#define ARROW_FLIGHT_TEST_APP_METADATA(FIXTURE)                                 \
  static_assert(std::is_base_of<AppMetadataTest, FIXTURE>::value,               \
                ARROW_STRINGIFY(FIXTURE) " must inherit from AppMetadataTest"); \
  TEST_F(FIXTURE, TestDoGet) { TestDoGet(); }                                   \
  TEST_F(FIXTURE, TestDoGetDictionaries) { TestDoGetDictionaries(); }           \
  TEST_F(FIXTURE, TestDoPut) { TestDoPut(); }                                   \
  TEST_F(FIXTURE, TestDoPutDictionaries) { TestDoPutDictionaries(); }           \
  TEST_F(FIXTURE, TestDoPutReadMetadata) { TestDoPutReadMetadata(); }

/// \brief Tests of IPC options in data plane methods.
class ARROW_FLIGHT_EXPORT IpcOptionsTest : public FlightTest {
 public:
  void SetUpTest() override;
  void TearDownTest() override;

  // Test methods
  void TestDoGetReadOptions();
  void TestDoPutWriteOptions();
  void TestDoExchangeClientWriteOptions();
  void TestDoExchangeClientWriteOptionsBegin();
  void TestDoExchangeServerWriteOptions();

 private:
  std::unique_ptr<FlightClient> client_;
  std::unique_ptr<FlightServerBase> server_;
};

#define ARROW_FLIGHT_TEST_IPC_OPTIONS(FIXTURE)                                 \
  static_assert(std::is_base_of<IpcOptionsTest, FIXTURE>::value,               \
                ARROW_STRINGIFY(FIXTURE) " must inherit from IpcOptionsTest"); \
  TEST_F(FIXTURE, TestDoGetReadOptions) { TestDoGetReadOptions(); }            \
  TEST_F(FIXTURE, TestDoPutWriteOptions) { TestDoPutWriteOptions(); }          \
  TEST_F(FIXTURE, TestDoExchangeClientWriteOptions) {                          \
    TestDoExchangeClientWriteOptions();                                        \
  }                                                                            \
  TEST_F(FIXTURE, TestDoExchangeClientWriteOptionsBegin) {                     \
    TestDoExchangeClientWriteOptionsBegin();                                   \
  }                                                                            \
  TEST_F(FIXTURE, TestDoExchangeServerWriteOptions) {                          \
    TestDoExchangeServerWriteOptions();                                        \
  }

/// \brief Tests of data plane methods with CUDA memory.
///
/// If not built with ARROW_CUDA, tests are no-ops.
class ARROW_FLIGHT_EXPORT CudaDataTest : public FlightTest {
 public:
  void SetUpTest() override;
  void TearDownTest() override;

  // Test methods
  void TestDoGet();
  void TestDoPut();
  void TestDoExchange();

 private:
  class Impl;
  std::unique_ptr<FlightClient> client_;
  std::unique_ptr<FlightServerBase> server_;
  std::shared_ptr<Impl> impl_;
};

#define ARROW_FLIGHT_TEST_CUDA_DATA(FIXTURE)                                 \
  static_assert(std::is_base_of<CudaDataTest, FIXTURE>::value,               \
                ARROW_STRINGIFY(FIXTURE) " must inherit from CudaDataTest"); \
  TEST_F(FIXTURE, TestDoGet) { TestDoGet(); }                                \
  TEST_F(FIXTURE, TestDoPut) { TestDoPut(); }                                \
  TEST_F(FIXTURE, TestDoExchange) { TestDoExchange(); }

/// \brief Tests of error handling.
class ARROW_FLIGHT_EXPORT ErrorHandlingTest : public FlightTest {
 public:
  void SetUpTest() override;
  void TearDownTest() override;

  // Test methods
  void TestGetFlightInfo();
  void TestDoPut();
  void TestDoExchange();

 private:
  std::unique_ptr<FlightClient> client_;
  std::unique_ptr<FlightServerBase> server_;
};

#define ARROW_FLIGHT_TEST_ERROR_HANDLING(FIXTURE)                                 \
  static_assert(std::is_base_of<ErrorHandlingTest, FIXTURE>::value,               \
                ARROW_STRINGIFY(FIXTURE) " must inherit from ErrorHandlingTest"); \
  TEST_F(FIXTURE, TestGetFlightInfo) { TestGetFlightInfo(); }                     \
  TEST_F(FIXTURE, TestDoPut) { TestDoPut(); }                                     \
  TEST_F(FIXTURE, TestDoExchange) { TestDoExchange(); }

}  // namespace flight
}  // namespace arrow

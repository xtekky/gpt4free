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
#include <utility>

#include <gtest/gtest.h>

#include "arrow/filesystem/s3fs.h"
#include "arrow/status.h"
#include "arrow/testing/gtest_util.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/macros.h"

namespace arrow {
namespace fs {

// A minio test server, managed as a child process

class MinioTestServer {
 public:
  MinioTestServer();
  ~MinioTestServer();

  Status Start();

  Status Stop();

  std::string connect_string() const;

  std::string access_key() const;

  std::string secret_key() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// A Minio "environment" that spawns Minio processes in advances, such as
// to hide process launch latencies during testing.

class MinioTestEnvironment : public ::testing::Environment {
 public:
  MinioTestEnvironment();
  ~MinioTestEnvironment();

  void SetUp() override;

  Result<std::shared_ptr<MinioTestServer>> GetOneServer();

 protected:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// A global test "environment", to ensure that the S3 API is initialized before
// running unit tests.

class S3Environment : public ::testing::Environment {
 public:
  void SetUp() override {
    // Change this to increase logging during tests
    S3GlobalOptions options;
    options.log_level = S3LogLevel::Fatal;
    ASSERT_OK(InitializeS3(options));
  }

  void TearDown() override { ASSERT_OK(FinalizeS3()); }
};

}  // namespace fs
}  // namespace arrow

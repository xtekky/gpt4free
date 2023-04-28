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

#include <cstdint>

#include "arrow/io/interfaces.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace io {

// Output stream that just writes to stdout.
class ARROW_EXPORT StdoutStream : public OutputStream {
 public:
  StdoutStream();
  ~StdoutStream() override {}

  Status Close() override;
  bool closed() const override;

  Result<int64_t> Tell() const override;

  Status Write(const void* data, int64_t nbytes) override;

 private:
  int64_t pos_;
};

// Output stream that just writes to stderr.
class ARROW_EXPORT StderrStream : public OutputStream {
 public:
  StderrStream();
  ~StderrStream() override {}

  Status Close() override;
  bool closed() const override;

  Result<int64_t> Tell() const override;

  Status Write(const void* data, int64_t nbytes) override;

 private:
  int64_t pos_;
};

// Input stream that just reads from stdin.
class ARROW_EXPORT StdinStream : public InputStream {
 public:
  StdinStream();
  ~StdinStream() override {}

  Status Close() override;
  bool closed() const override;

  Result<int64_t> Tell() const override;

  Result<int64_t> Read(int64_t nbytes, void* out) override;

  Result<std::shared_ptr<Buffer>> Read(int64_t nbytes) override;

 private:
  int64_t pos_;
};

}  // namespace io
}  // namespace arrow

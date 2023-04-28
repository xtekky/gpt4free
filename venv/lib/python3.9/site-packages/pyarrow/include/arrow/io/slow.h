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

// Slow stream implementations, mainly for testing and benchmarking

#pragma once

#include <cstdint>
#include <memory>
#include <utility>

#include "arrow/io/interfaces.h"
#include "arrow/util/visibility.h"

namespace arrow {

class Buffer;
class Status;

namespace io {

class ARROW_EXPORT LatencyGenerator {
 public:
  virtual ~LatencyGenerator();

  void Sleep();

  virtual double NextLatency() = 0;

  static std::shared_ptr<LatencyGenerator> Make(double average_latency);
  static std::shared_ptr<LatencyGenerator> Make(double average_latency, int32_t seed);
};

// XXX use ConcurrencyWrapper?  It could increase chances of finding a race.

template <class StreamType>
class SlowInputStreamBase : public StreamType {
 public:
  SlowInputStreamBase(std::shared_ptr<StreamType> stream,
                      std::shared_ptr<LatencyGenerator> latencies)
      : stream_(std::move(stream)), latencies_(std::move(latencies)) {}

  SlowInputStreamBase(std::shared_ptr<StreamType> stream, double average_latency)
      : stream_(std::move(stream)), latencies_(LatencyGenerator::Make(average_latency)) {}

  SlowInputStreamBase(std::shared_ptr<StreamType> stream, double average_latency,
                      int32_t seed)
      : stream_(std::move(stream)),
        latencies_(LatencyGenerator::Make(average_latency, seed)) {}

 protected:
  std::shared_ptr<StreamType> stream_;
  std::shared_ptr<LatencyGenerator> latencies_;
};

/// \brief An InputStream wrapper that makes reads slower.
///
/// Read() calls are made slower by an average latency (in seconds).
/// Actual latencies form a normal distribution closely centered
/// on the average latency.
/// Other calls are forwarded directly.
class ARROW_EXPORT SlowInputStream : public SlowInputStreamBase<InputStream> {
 public:
  ~SlowInputStream() override;

  using SlowInputStreamBase<InputStream>::SlowInputStreamBase;

  Status Close() override;
  Status Abort() override;
  bool closed() const override;

  Result<int64_t> Read(int64_t nbytes, void* out) override;
  Result<std::shared_ptr<Buffer>> Read(int64_t nbytes) override;
  Result<std::string_view> Peek(int64_t nbytes) override;

  Result<int64_t> Tell() const override;
};

/// \brief A RandomAccessFile wrapper that makes reads slower.
///
/// Similar to SlowInputStream, but allows random access and seeking.
class ARROW_EXPORT SlowRandomAccessFile : public SlowInputStreamBase<RandomAccessFile> {
 public:
  ~SlowRandomAccessFile() override;

  using SlowInputStreamBase<RandomAccessFile>::SlowInputStreamBase;

  Status Close() override;
  Status Abort() override;
  bool closed() const override;

  Result<int64_t> Read(int64_t nbytes, void* out) override;
  Result<std::shared_ptr<Buffer>> Read(int64_t nbytes) override;
  Result<int64_t> ReadAt(int64_t position, int64_t nbytes, void* out) override;
  Result<std::shared_ptr<Buffer>> ReadAt(int64_t position, int64_t nbytes) override;
  Result<std::string_view> Peek(int64_t nbytes) override;

  Result<int64_t> GetSize() override;
  Status Seek(int64_t position) override;
  Result<int64_t> Tell() const override;
};

}  // namespace io
}  // namespace arrow

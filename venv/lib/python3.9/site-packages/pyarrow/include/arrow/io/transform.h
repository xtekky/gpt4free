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

// Transform stream implementations

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

#include "arrow/io/interfaces.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace io {

class ARROW_EXPORT TransformInputStream : public InputStream {
 public:
  using TransformFunc =
      std::function<Result<std::shared_ptr<Buffer>>(const std::shared_ptr<Buffer>&)>;

  TransformInputStream(std::shared_ptr<InputStream> wrapped, TransformFunc transform);
  ~TransformInputStream() override;

  Status Close() override;
  Status Abort() override;
  bool closed() const override;

  Result<int64_t> Read(int64_t nbytes, void* out) override;
  Result<std::shared_ptr<Buffer>> Read(int64_t nbytes) override;

  Result<std::shared_ptr<const KeyValueMetadata>> ReadMetadata() override;
  Future<std::shared_ptr<const KeyValueMetadata>> ReadMetadataAsync(
      const IOContext& io_context) override;

  Result<int64_t> Tell() const override;

 protected:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace io
}  // namespace arrow

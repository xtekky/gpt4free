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

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "arrow/buffer.h"
#include "arrow/record_batch.h"
#include "arrow/status.h"
#include "arrow/testing/visibility.h"
#include "arrow/type_fwd.h"
#include "arrow/util/macros.h"

namespace arrow {

template <typename T>
Status CopyBufferFromVector(const std::vector<T>& values, MemoryPool* pool,
                            std::shared_ptr<Buffer>* result) {
  int64_t nbytes = static_cast<int>(values.size()) * sizeof(T);

  ARROW_ASSIGN_OR_RAISE(auto buffer, AllocateBuffer(nbytes, pool));
  auto immutable_data = reinterpret_cast<const uint8_t*>(values.data());
  std::copy(immutable_data, immutable_data + nbytes, buffer->mutable_data());
  memset(buffer->mutable_data() + nbytes, 0,
         static_cast<size_t>(buffer->capacity() - nbytes));

  *result = std::move(buffer);
  return Status::OK();
}

// Sets approximately pct_null of the first n bytes in null_bytes to zero
// and the rest to non-zero (true) values.
ARROW_TESTING_EXPORT void random_null_bytes(int64_t n, double pct_null,
                                            uint8_t* null_bytes);
ARROW_TESTING_EXPORT void random_is_valid(int64_t n, double pct_null,
                                          std::vector<bool>* is_valid,
                                          int random_seed = 0);
ARROW_TESTING_EXPORT void random_bytes(int64_t n, uint32_t seed, uint8_t* out);
ARROW_TESTING_EXPORT std::string random_string(int64_t n, uint32_t seed);
ARROW_TESTING_EXPORT int32_t DecimalSize(int32_t precision);
ARROW_TESTING_EXPORT void random_ascii(int64_t n, uint32_t seed, uint8_t* out);
ARROW_TESTING_EXPORT int64_t CountNulls(const std::vector<uint8_t>& valid_bytes);

ARROW_TESTING_EXPORT Status MakeRandomByteBuffer(int64_t length, MemoryPool* pool,
                                                 std::shared_ptr<ResizableBuffer>* out,
                                                 uint32_t seed = 0);

ARROW_TESTING_EXPORT uint64_t random_seed();

#define DECL_T() typedef typename TestFixture::T T;

#define DECL_TYPE() typedef typename TestFixture::Type Type;

// ----------------------------------------------------------------------
// A RecordBatchReader for serving a sequence of in-memory record batches

class BatchIterator : public RecordBatchReader {
 public:
  BatchIterator(const std::shared_ptr<Schema>& schema,
                const std::vector<std::shared_ptr<RecordBatch>>& batches)
      : schema_(schema), batches_(batches), position_(0) {}

  std::shared_ptr<Schema> schema() const override { return schema_; }

  Status ReadNext(std::shared_ptr<RecordBatch>* out) override {
    if (position_ >= batches_.size()) {
      *out = nullptr;
    } else {
      *out = batches_[position_++];
    }
    return Status::OK();
  }

 private:
  std::shared_ptr<Schema> schema_;
  std::vector<std::shared_ptr<RecordBatch>> batches_;
  size_t position_;
};

static inline std::vector<std::shared_ptr<DataType> (*)(FieldVector, std::vector<int8_t>)>
UnionTypeFactories() {
  return {sparse_union, dense_union};
}

// Return the value of the ARROW_TEST_DATA environment variable or return error
// Status
ARROW_TESTING_EXPORT Status GetTestResourceRoot(std::string*);

// Return the value of the ARROW_TIMEZONE_DATABASE environment variable
ARROW_TESTING_EXPORT std::optional<std::string> GetTestTimezoneDatabaseRoot();

// Set the Timezone database based on the ARROW_TIMEZONE_DATABASE env variable
// This is only relevant on Windows, since other OSs have compatible databases built-in
ARROW_TESTING_EXPORT Status InitTestTimezoneDatabase();

// Get a TCP port number to listen on.  This is a different number every time,
// as reusing the same port across tests can produce spurious bind errors on
// Windows.
ARROW_TESTING_EXPORT int GetListenPort();

// Get a IPv4 "address:port" to listen on.  The address will be a loopback
// address.  Compared to GetListenPort(), this will minimize the risk of
// port conflicts.
ARROW_TESTING_EXPORT std::string GetListenAddress();

ARROW_TESTING_EXPORT
const std::vector<std::shared_ptr<DataType>>& all_dictionary_index_types();

}  // namespace arrow

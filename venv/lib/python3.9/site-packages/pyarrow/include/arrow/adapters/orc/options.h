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

#include <vector>

#include "arrow/io/interfaces.h"
#include "arrow/status.h"
#include "arrow/util/type_fwd.h"
#include "arrow/util/visibility.h"

namespace arrow {

namespace adapters {

namespace orc {

enum class WriterId : int32_t {
  kOrcJava = 0,
  kOrcCpp = 1,
  kPresto = 2,
  kScritchleyGo = 3,
  kTrino = 4,
  kUnknown = INT32_MAX
};

enum class WriterVersion : int32_t {
  kOriginal = 0,
  kHive8732 = 1,
  kHive4243 = 2,
  kHive12055 = 3,
  kHive13083 = 4,
  kOrc101 = 5,
  kOrc135 = 6,
  kOrc517 = 7,
  kOrc203 = 8,
  kOrc14 = 9,
  kMax = INT32_MAX
};

enum class CompressionStrategy : int32_t { kSpeed = 0, kCompression };

class ARROW_EXPORT FileVersion {
 private:
  int32_t major_version_;
  int32_t minor_version_;

 public:
  static const FileVersion& v_0_11();
  static const FileVersion& v_0_12();

  FileVersion(int32_t major, int32_t minor)
      : major_version_(major), minor_version_(minor) {}

  /**
   * Get major version
   */
  int32_t major_version() const { return this->major_version_; }

  /**
   * Get minor version
   */
  int32_t minor_version() const { return this->minor_version_; }

  bool operator==(const FileVersion& right) const {
    return this->major_version() == right.major_version() &&
           this->minor_version() == right.minor_version();
  }

  bool operator!=(const FileVersion& right) const { return !(*this == right); }

  std::string ToString() const;
};

/// Options for the ORC Writer
struct ARROW_EXPORT WriteOptions {
  /// Number of rows the ORC writer writes at a time, default 1024
  int64_t batch_size = 1024;
  /// Which ORC file version to use, default FileVersion(0, 12)
  FileVersion file_version = FileVersion(0, 12);
  /// Size of each ORC stripe in bytes, default 64 MiB
  int64_t stripe_size = 64 * 1024 * 1024;
  /// The compression codec of the ORC file, there is no compression by default
  Compression::type compression = Compression::UNCOMPRESSED;
  /// The size of each compression block in bytes, default 64 KiB
  int64_t compression_block_size = 64 * 1024;
  /// The compression strategy i.e. speed vs size reduction, default
  /// CompressionStrategy::kSpeed
  CompressionStrategy compression_strategy = CompressionStrategy::kSpeed;
  /// The number of rows per an entry in the row index, default 10000
  int64_t row_index_stride = 10000;
  /// The padding tolerance, default 0.0
  double padding_tolerance = 0.0;
  /// The dictionary key size threshold. 0 to disable dictionary encoding.
  /// 1 to always enable dictionary encoding, default 0.0
  double dictionary_key_size_threshold = 0.0;
  /// The array of columns that use the bloom filter, default empty
  std::vector<int64_t> bloom_filter_columns;
  /// The upper limit of the false-positive rate of the bloom filter, default 0.05
  double bloom_filter_fpp = 0.05;
};

}  // namespace orc
}  // namespace adapters
}  // namespace arrow

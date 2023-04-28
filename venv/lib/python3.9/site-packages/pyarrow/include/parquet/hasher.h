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
#include "parquet/types.h"

namespace parquet {
// Abstract class for hash
class Hasher {
 public:
  /// Compute hash for 32 bits value by using its plain encoding result.
  ///
  /// @param value the value to hash.
  /// @return hash result.
  virtual uint64_t Hash(int32_t value) const = 0;

  /// Compute hash for 64 bits value by using its plain encoding result.
  ///
  /// @param value the value to hash.
  /// @return hash result.
  virtual uint64_t Hash(int64_t value) const = 0;

  /// Compute hash for float value by using its plain encoding result.
  ///
  /// @param value the value to hash.
  /// @return hash result.
  virtual uint64_t Hash(float value) const = 0;

  /// Compute hash for double value by using its plain encoding result.
  ///
  /// @param value the value to hash.
  /// @return hash result.
  virtual uint64_t Hash(double value) const = 0;

  /// Compute hash for Int96 value by using its plain encoding result.
  ///
  /// @param value the value to hash.
  /// @return hash result.
  virtual uint64_t Hash(const Int96* value) const = 0;

  /// Compute hash for ByteArray value by using its plain encoding result.
  ///
  /// @param value the value to hash.
  /// @return hash result.
  virtual uint64_t Hash(const ByteArray* value) const = 0;

  /// Compute hash for fixed byte array value by using its plain encoding result.
  ///
  /// @param value the value address.
  /// @param len the value length.
  virtual uint64_t Hash(const FLBA* value, uint32_t len) const = 0;

  virtual ~Hasher() = default;
};

}  // namespace parquet

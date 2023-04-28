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

//-----------------------------------------------------------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.

#pragma once

#include <cstdint>

#include "parquet/hasher.h"
#include "parquet/platform.h"
#include "parquet/types.h"

namespace parquet {

/// Source:
/// https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
/// (Modified to adapt to coding conventions and to inherit the Hasher abstract class)
class PARQUET_EXPORT MurmurHash3 : public Hasher {
 public:
  MurmurHash3() : seed_(DEFAULT_SEED) {}
  uint64_t Hash(int32_t value) const override;
  uint64_t Hash(int64_t value) const override;
  uint64_t Hash(float value) const override;
  uint64_t Hash(double value) const override;
  uint64_t Hash(const Int96* value) const override;
  uint64_t Hash(const ByteArray* value) const override;
  uint64_t Hash(const FLBA* val, uint32_t len) const override;

 private:
  // Default seed for hash which comes from Bloom filter in parquet-mr, it is generated
  // by System.nanoTime() of java.
  static constexpr int DEFAULT_SEED = 1361930890;

  uint32_t seed_;
};

}  // namespace parquet

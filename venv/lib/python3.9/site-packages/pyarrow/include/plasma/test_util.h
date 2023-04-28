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
#include <limits>
#include <random>

#include "plasma/common.h"

namespace plasma {

ObjectID random_object_id() {
  static uint32_t random_seed = 0;
  std::mt19937 gen(random_seed++);
  std::uniform_int_distribution<uint32_t> d(0, std::numeric_limits<uint8_t>::max());
  ObjectID result;
  uint8_t* data = result.mutable_data();
  std::generate(data, data + kUniqueIDSize,
                [&d, &gen] { return static_cast<uint8_t>(d(gen)); });
  return result;
}

#define PLASMA_CHECK_SYSTEM(expr)        \
  do {                                   \
    int status__ = (expr);               \
    EXPECT_TRUE(WIFEXITED(status__));    \
    EXPECT_EQ(WEXITSTATUS(status__), 0); \
  } while (false);

}  // namespace plasma

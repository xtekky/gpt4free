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

namespace arrow {

#define ARROW_GENERATE_FOR_ALL_INTEGER_TYPES(ACTION) \
  ACTION(Int8);                                      \
  ACTION(UInt8);                                     \
  ACTION(Int16);                                     \
  ACTION(UInt16);                                    \
  ACTION(Int32);                                     \
  ACTION(UInt32);                                    \
  ACTION(Int64);                                     \
  ACTION(UInt64)

#define ARROW_GENERATE_FOR_ALL_NUMERIC_TYPES(ACTION) \
  ARROW_GENERATE_FOR_ALL_INTEGER_TYPES(ACTION);      \
  ACTION(HalfFloat);                                 \
  ACTION(Float);                                     \
  ACTION(Double)

#define ARROW_GENERATE_FOR_ALL_TYPES(ACTION)    \
  ACTION(Null);                                 \
  ACTION(Boolean);                              \
  ARROW_GENERATE_FOR_ALL_NUMERIC_TYPES(ACTION); \
  ACTION(String);                               \
  ACTION(Binary);                               \
  ACTION(LargeString);                          \
  ACTION(LargeBinary);                          \
  ACTION(FixedSizeBinary);                      \
  ACTION(Duration);                             \
  ACTION(Date32);                               \
  ACTION(Date64);                               \
  ACTION(Timestamp);                            \
  ACTION(Time32);                               \
  ACTION(Time64);                               \
  ACTION(MonthDayNanoInterval);                 \
  ACTION(MonthInterval);                        \
  ACTION(DayTimeInterval);                      \
  ACTION(Decimal128);                           \
  ACTION(Decimal256);                           \
  ACTION(List);                                 \
  ACTION(LargeList);                            \
  ACTION(Map);                                  \
  ACTION(FixedSizeList);                        \
  ACTION(Struct);                               \
  ACTION(SparseUnion);                          \
  ACTION(DenseUnion);                           \
  ACTION(Dictionary);                           \
  ACTION(Extension)

}  // namespace arrow

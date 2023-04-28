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

// Include this file before including any RapidJSON headers.

#pragma once

#define RAPIDJSON_HAS_STDSTRING 1
#define RAPIDJSON_HAS_CXX11_RVALUE_REFS 1
#define RAPIDJSON_HAS_CXX11_RANGE_FOR 1

// rapidjson will be defined in namespace arrow::rapidjson
#define RAPIDJSON_NAMESPACE arrow::rapidjson
#define RAPIDJSON_NAMESPACE_BEGIN \
  namespace arrow {               \
  namespace rapidjson {
#define RAPIDJSON_NAMESPACE_END \
  }                             \
  }

// enable SIMD whitespace skipping, if available
#if defined(ARROW_HAVE_SSE4_2)
#define RAPIDJSON_SSE2 1
#define RAPIDJSON_SSE42 1
#endif

#if defined(ARROW_HAVE_NEON)
#define RAPIDJSON_NEON 1
#endif

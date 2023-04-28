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

// Automatically generated file; DO NOT EDIT.

#pragma once

#include <cstdint>
#include <cstring>

#include <xsimd/xsimd.hpp>

#include "arrow/util/dispatch.h"
#include "arrow/util/ubsan.h"

namespace arrow {
namespace internal {
namespace {

using ::arrow::util::SafeLoad;

template <DispatchLevel level>
struct UnpackBits128 {

using simd_batch = xsimd::make_sized_batch_t<uint32_t, 4>;

inline static const uint32_t* unpack0_32(const uint32_t* in, uint32_t* out) {
  memset(out, 0x0, 32 * sizeof(*out));
  out += 32;

  return in;
}

inline static const uint32_t* unpack1_32(const uint32_t* in, uint32_t* out) {
  uint32_t mask = 0x1;

  simd_batch masks(mask);
  simd_batch words, shifts;
  simd_batch results;

  // extract 1-bit bundles 0 to 3
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) };
  shifts = simd_batch{ 0, 1, 2, 3 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 1-bit bundles 4 to 7
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) };
  shifts = simd_batch{ 4, 5, 6, 7 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 1-bit bundles 8 to 11
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) };
  shifts = simd_batch{ 8, 9, 10, 11 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 1-bit bundles 12 to 15
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) };
  shifts = simd_batch{ 12, 13, 14, 15 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 1-bit bundles 16 to 19
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) };
  shifts = simd_batch{ 16, 17, 18, 19 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 1-bit bundles 20 to 23
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) };
  shifts = simd_batch{ 20, 21, 22, 23 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 1-bit bundles 24 to 27
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) };
  shifts = simd_batch{ 24, 25, 26, 27 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 1-bit bundles 28 to 31
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) };
  shifts = simd_batch{ 28, 29, 30, 31 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  in += 1;
  return in;
}

inline static const uint32_t* unpack2_32(const uint32_t* in, uint32_t* out) {
  uint32_t mask = 0x3;

  simd_batch masks(mask);
  simd_batch words, shifts;
  simd_batch results;

  // extract 2-bit bundles 0 to 3
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) };
  shifts = simd_batch{ 0, 2, 4, 6 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 2-bit bundles 4 to 7
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) };
  shifts = simd_batch{ 8, 10, 12, 14 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 2-bit bundles 8 to 11
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) };
  shifts = simd_batch{ 16, 18, 20, 22 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 2-bit bundles 12 to 15
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) };
  shifts = simd_batch{ 24, 26, 28, 30 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 2-bit bundles 16 to 19
  words = simd_batch{ SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1) };
  shifts = simd_batch{ 0, 2, 4, 6 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 2-bit bundles 20 to 23
  words = simd_batch{ SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1) };
  shifts = simd_batch{ 8, 10, 12, 14 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 2-bit bundles 24 to 27
  words = simd_batch{ SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1) };
  shifts = simd_batch{ 16, 18, 20, 22 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 2-bit bundles 28 to 31
  words = simd_batch{ SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1) };
  shifts = simd_batch{ 24, 26, 28, 30 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  in += 2;
  return in;
}

inline static const uint32_t* unpack3_32(const uint32_t* in, uint32_t* out) {
  uint32_t mask = 0x7;

  simd_batch masks(mask);
  simd_batch words, shifts;
  simd_batch results;

  // extract 3-bit bundles 0 to 3
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) };
  shifts = simd_batch{ 0, 3, 6, 9 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 3-bit bundles 4 to 7
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) };
  shifts = simd_batch{ 12, 15, 18, 21 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 3-bit bundles 8 to 11
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) >> 30 | SafeLoad<uint32_t>(in + 1) << 2, SafeLoad<uint32_t>(in + 1) };
  shifts = simd_batch{ 24, 27, 0, 1 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 3-bit bundles 12 to 15
  words = simd_batch{ SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1) };
  shifts = simd_batch{ 4, 7, 10, 13 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 3-bit bundles 16 to 19
  words = simd_batch{ SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1) };
  shifts = simd_batch{ 16, 19, 22, 25 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 3-bit bundles 20 to 23
  words = simd_batch{ SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1) >> 31 | SafeLoad<uint32_t>(in + 2) << 1, SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2) };
  shifts = simd_batch{ 28, 0, 2, 5 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 3-bit bundles 24 to 27
  words = simd_batch{ SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2) };
  shifts = simd_batch{ 8, 11, 14, 17 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 3-bit bundles 28 to 31
  words = simd_batch{ SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2) };
  shifts = simd_batch{ 20, 23, 26, 29 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  in += 3;
  return in;
}

inline static const uint32_t* unpack4_32(const uint32_t* in, uint32_t* out) {
  uint32_t mask = 0xf;

  simd_batch masks(mask);
  simd_batch words, shifts;
  simd_batch results;

  // extract 4-bit bundles 0 to 3
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) };
  shifts = simd_batch{ 0, 4, 8, 12 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 4-bit bundles 4 to 7
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) };
  shifts = simd_batch{ 16, 20, 24, 28 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 4-bit bundles 8 to 11
  words = simd_batch{ SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1) };
  shifts = simd_batch{ 0, 4, 8, 12 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 4-bit bundles 12 to 15
  words = simd_batch{ SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1) };
  shifts = simd_batch{ 16, 20, 24, 28 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 4-bit bundles 16 to 19
  words = simd_batch{ SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2) };
  shifts = simd_batch{ 0, 4, 8, 12 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 4-bit bundles 20 to 23
  words = simd_batch{ SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2) };
  shifts = simd_batch{ 16, 20, 24, 28 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 4-bit bundles 24 to 27
  words = simd_batch{ SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3) };
  shifts = simd_batch{ 0, 4, 8, 12 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 4-bit bundles 28 to 31
  words = simd_batch{ SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3) };
  shifts = simd_batch{ 16, 20, 24, 28 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  in += 4;
  return in;
}

inline static const uint32_t* unpack5_32(const uint32_t* in, uint32_t* out) {
  uint32_t mask = 0x1f;

  simd_batch masks(mask);
  simd_batch words, shifts;
  simd_batch results;

  // extract 5-bit bundles 0 to 3
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) };
  shifts = simd_batch{ 0, 5, 10, 15 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 5-bit bundles 4 to 7
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) >> 30 | SafeLoad<uint32_t>(in + 1) << 2, SafeLoad<uint32_t>(in + 1) };
  shifts = simd_batch{ 20, 25, 0, 3 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 5-bit bundles 8 to 11
  words = simd_batch{ SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1) };
  shifts = simd_batch{ 8, 13, 18, 23 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 5-bit bundles 12 to 15
  words = simd_batch{ SafeLoad<uint32_t>(in + 1) >> 28 | SafeLoad<uint32_t>(in + 2) << 4, SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2) };
  shifts = simd_batch{ 0, 1, 6, 11 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 5-bit bundles 16 to 19
  words = simd_batch{ SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2) >> 31 | SafeLoad<uint32_t>(in + 3) << 1 };
  shifts = simd_batch{ 16, 21, 26, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 5-bit bundles 20 to 23
  words = simd_batch{ SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3) };
  shifts = simd_batch{ 4, 9, 14, 19 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 5-bit bundles 24 to 27
  words = simd_batch{ SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3) >> 29 | SafeLoad<uint32_t>(in + 4) << 3, SafeLoad<uint32_t>(in + 4), SafeLoad<uint32_t>(in + 4) };
  shifts = simd_batch{ 24, 0, 2, 7 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 5-bit bundles 28 to 31
  words = simd_batch{ SafeLoad<uint32_t>(in + 4), SafeLoad<uint32_t>(in + 4), SafeLoad<uint32_t>(in + 4), SafeLoad<uint32_t>(in + 4) };
  shifts = simd_batch{ 12, 17, 22, 27 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  in += 5;
  return in;
}

inline static const uint32_t* unpack6_32(const uint32_t* in, uint32_t* out) {
  uint32_t mask = 0x3f;

  simd_batch masks(mask);
  simd_batch words, shifts;
  simd_batch results;

  // extract 6-bit bundles 0 to 3
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) };
  shifts = simd_batch{ 0, 6, 12, 18 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 6-bit bundles 4 to 7
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) >> 30 | SafeLoad<uint32_t>(in + 1) << 2, SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1) };
  shifts = simd_batch{ 24, 0, 4, 10 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 6-bit bundles 8 to 11
  words = simd_batch{ SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1) >> 28 | SafeLoad<uint32_t>(in + 2) << 4, SafeLoad<uint32_t>(in + 2) };
  shifts = simd_batch{ 16, 22, 0, 2 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 6-bit bundles 12 to 15
  words = simd_batch{ SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2) };
  shifts = simd_batch{ 8, 14, 20, 26 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 6-bit bundles 16 to 19
  words = simd_batch{ SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3) };
  shifts = simd_batch{ 0, 6, 12, 18 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 6-bit bundles 20 to 23
  words = simd_batch{ SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3) >> 30 | SafeLoad<uint32_t>(in + 4) << 2, SafeLoad<uint32_t>(in + 4), SafeLoad<uint32_t>(in + 4) };
  shifts = simd_batch{ 24, 0, 4, 10 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 6-bit bundles 24 to 27
  words = simd_batch{ SafeLoad<uint32_t>(in + 4), SafeLoad<uint32_t>(in + 4), SafeLoad<uint32_t>(in + 4) >> 28 | SafeLoad<uint32_t>(in + 5) << 4, SafeLoad<uint32_t>(in + 5) };
  shifts = simd_batch{ 16, 22, 0, 2 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 6-bit bundles 28 to 31
  words = simd_batch{ SafeLoad<uint32_t>(in + 5), SafeLoad<uint32_t>(in + 5), SafeLoad<uint32_t>(in + 5), SafeLoad<uint32_t>(in + 5) };
  shifts = simd_batch{ 8, 14, 20, 26 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  in += 6;
  return in;
}

inline static const uint32_t* unpack7_32(const uint32_t* in, uint32_t* out) {
  uint32_t mask = 0x7f;

  simd_batch masks(mask);
  simd_batch words, shifts;
  simd_batch results;

  // extract 7-bit bundles 0 to 3
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) };
  shifts = simd_batch{ 0, 7, 14, 21 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 7-bit bundles 4 to 7
  words = simd_batch{ SafeLoad<uint32_t>(in + 0) >> 28 | SafeLoad<uint32_t>(in + 1) << 4, SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1) };
  shifts = simd_batch{ 0, 3, 10, 17 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 7-bit bundles 8 to 11
  words = simd_batch{ SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1) >> 31 | SafeLoad<uint32_t>(in + 2) << 1, SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2) };
  shifts = simd_batch{ 24, 0, 6, 13 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 7-bit bundles 12 to 15
  words = simd_batch{ SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2) >> 27 | SafeLoad<uint32_t>(in + 3) << 5, SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3) };
  shifts = simd_batch{ 20, 0, 2, 9 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 7-bit bundles 16 to 19
  words = simd_batch{ SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3) >> 30 | SafeLoad<uint32_t>(in + 4) << 2, SafeLoad<uint32_t>(in + 4) };
  shifts = simd_batch{ 16, 23, 0, 5 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 7-bit bundles 20 to 23
  words = simd_batch{ SafeLoad<uint32_t>(in + 4), SafeLoad<uint32_t>(in + 4), SafeLoad<uint32_t>(in + 4) >> 26 | SafeLoad<uint32_t>(in + 5) << 6, SafeLoad<uint32_t>(in + 5) };
  shifts = simd_batch{ 12, 19, 0, 1 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 7-bit bundles 24 to 27
  words = simd_batch{ SafeLoad<uint32_t>(in + 5), SafeLoad<uint32_t>(in + 5), SafeLoad<uint32_t>(in + 5), SafeLoad<uint32_t>(in + 5) >> 29 | SafeLoad<uint32_t>(in + 6) << 3 };
  shifts = simd_batch{ 8, 15, 22, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 7-bit bundles 28 to 31
  words = simd_batch{ SafeLoad<uint32_t>(in + 6), SafeLoad<uint32_t>(in + 6), SafeLoad<uint32_t>(in + 6), SafeLoad<uint32_t>(in + 6) };
  shifts = simd_batch{ 4, 11, 18, 25 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  in += 7;
  return in;
}

inline static const uint32_t* unpack8_32(const uint32_t* in, uint32_t* out) {
  uint32_t mask = 0xff;

  simd_batch masks(mask);
  simd_batch words, shifts;
  simd_batch results;

  // extract 8-bit bundles 0 to 3
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) };
  shifts = simd_batch{ 0, 8, 16, 24 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 8-bit bundles 4 to 7
  words = simd_batch{ SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1) };
  shifts = simd_batch{ 0, 8, 16, 24 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 8-bit bundles 8 to 11
  words = simd_batch{ SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2) };
  shifts = simd_batch{ 0, 8, 16, 24 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 8-bit bundles 12 to 15
  words = simd_batch{ SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3) };
  shifts = simd_batch{ 0, 8, 16, 24 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 8-bit bundles 16 to 19
  words = simd_batch{ SafeLoad<uint32_t>(in + 4), SafeLoad<uint32_t>(in + 4), SafeLoad<uint32_t>(in + 4), SafeLoad<uint32_t>(in + 4) };
  shifts = simd_batch{ 0, 8, 16, 24 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 8-bit bundles 20 to 23
  words = simd_batch{ SafeLoad<uint32_t>(in + 5), SafeLoad<uint32_t>(in + 5), SafeLoad<uint32_t>(in + 5), SafeLoad<uint32_t>(in + 5) };
  shifts = simd_batch{ 0, 8, 16, 24 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 8-bit bundles 24 to 27
  words = simd_batch{ SafeLoad<uint32_t>(in + 6), SafeLoad<uint32_t>(in + 6), SafeLoad<uint32_t>(in + 6), SafeLoad<uint32_t>(in + 6) };
  shifts = simd_batch{ 0, 8, 16, 24 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 8-bit bundles 28 to 31
  words = simd_batch{ SafeLoad<uint32_t>(in + 7), SafeLoad<uint32_t>(in + 7), SafeLoad<uint32_t>(in + 7), SafeLoad<uint32_t>(in + 7) };
  shifts = simd_batch{ 0, 8, 16, 24 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  in += 8;
  return in;
}

inline static const uint32_t* unpack9_32(const uint32_t* in, uint32_t* out) {
  uint32_t mask = 0x1ff;

  simd_batch masks(mask);
  simd_batch words, shifts;
  simd_batch results;

  // extract 9-bit bundles 0 to 3
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) >> 27 | SafeLoad<uint32_t>(in + 1) << 5 };
  shifts = simd_batch{ 0, 9, 18, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 9-bit bundles 4 to 7
  words = simd_batch{ SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1) >> 31 | SafeLoad<uint32_t>(in + 2) << 1 };
  shifts = simd_batch{ 4, 13, 22, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 9-bit bundles 8 to 11
  words = simd_batch{ SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2) >> 26 | SafeLoad<uint32_t>(in + 3) << 6, SafeLoad<uint32_t>(in + 3) };
  shifts = simd_batch{ 8, 17, 0, 3 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 9-bit bundles 12 to 15
  words = simd_batch{ SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3) >> 30 | SafeLoad<uint32_t>(in + 4) << 2, SafeLoad<uint32_t>(in + 4) };
  shifts = simd_batch{ 12, 21, 0, 7 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 9-bit bundles 16 to 19
  words = simd_batch{ SafeLoad<uint32_t>(in + 4), SafeLoad<uint32_t>(in + 4) >> 25 | SafeLoad<uint32_t>(in + 5) << 7, SafeLoad<uint32_t>(in + 5), SafeLoad<uint32_t>(in + 5) };
  shifts = simd_batch{ 16, 0, 2, 11 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 9-bit bundles 20 to 23
  words = simd_batch{ SafeLoad<uint32_t>(in + 5), SafeLoad<uint32_t>(in + 5) >> 29 | SafeLoad<uint32_t>(in + 6) << 3, SafeLoad<uint32_t>(in + 6), SafeLoad<uint32_t>(in + 6) };
  shifts = simd_batch{ 20, 0, 6, 15 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 9-bit bundles 24 to 27
  words = simd_batch{ SafeLoad<uint32_t>(in + 6) >> 24 | SafeLoad<uint32_t>(in + 7) << 8, SafeLoad<uint32_t>(in + 7), SafeLoad<uint32_t>(in + 7), SafeLoad<uint32_t>(in + 7) };
  shifts = simd_batch{ 0, 1, 10, 19 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 9-bit bundles 28 to 31
  words = simd_batch{ SafeLoad<uint32_t>(in + 7) >> 28 | SafeLoad<uint32_t>(in + 8) << 4, SafeLoad<uint32_t>(in + 8), SafeLoad<uint32_t>(in + 8), SafeLoad<uint32_t>(in + 8) };
  shifts = simd_batch{ 0, 5, 14, 23 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  in += 9;
  return in;
}

inline static const uint32_t* unpack10_32(const uint32_t* in, uint32_t* out) {
  uint32_t mask = 0x3ff;

  simd_batch masks(mask);
  simd_batch words, shifts;
  simd_batch results;

  // extract 10-bit bundles 0 to 3
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) >> 30 | SafeLoad<uint32_t>(in + 1) << 2 };
  shifts = simd_batch{ 0, 10, 20, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 10-bit bundles 4 to 7
  words = simd_batch{ SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1) >> 28 | SafeLoad<uint32_t>(in + 2) << 4, SafeLoad<uint32_t>(in + 2) };
  shifts = simd_batch{ 8, 18, 0, 6 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 10-bit bundles 8 to 11
  words = simd_batch{ SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2) >> 26 | SafeLoad<uint32_t>(in + 3) << 6, SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3) };
  shifts = simd_batch{ 16, 0, 4, 14 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 10-bit bundles 12 to 15
  words = simd_batch{ SafeLoad<uint32_t>(in + 3) >> 24 | SafeLoad<uint32_t>(in + 4) << 8, SafeLoad<uint32_t>(in + 4), SafeLoad<uint32_t>(in + 4), SafeLoad<uint32_t>(in + 4) };
  shifts = simd_batch{ 0, 2, 12, 22 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 10-bit bundles 16 to 19
  words = simd_batch{ SafeLoad<uint32_t>(in + 5), SafeLoad<uint32_t>(in + 5), SafeLoad<uint32_t>(in + 5), SafeLoad<uint32_t>(in + 5) >> 30 | SafeLoad<uint32_t>(in + 6) << 2 };
  shifts = simd_batch{ 0, 10, 20, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 10-bit bundles 20 to 23
  words = simd_batch{ SafeLoad<uint32_t>(in + 6), SafeLoad<uint32_t>(in + 6), SafeLoad<uint32_t>(in + 6) >> 28 | SafeLoad<uint32_t>(in + 7) << 4, SafeLoad<uint32_t>(in + 7) };
  shifts = simd_batch{ 8, 18, 0, 6 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 10-bit bundles 24 to 27
  words = simd_batch{ SafeLoad<uint32_t>(in + 7), SafeLoad<uint32_t>(in + 7) >> 26 | SafeLoad<uint32_t>(in + 8) << 6, SafeLoad<uint32_t>(in + 8), SafeLoad<uint32_t>(in + 8) };
  shifts = simd_batch{ 16, 0, 4, 14 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 10-bit bundles 28 to 31
  words = simd_batch{ SafeLoad<uint32_t>(in + 8) >> 24 | SafeLoad<uint32_t>(in + 9) << 8, SafeLoad<uint32_t>(in + 9), SafeLoad<uint32_t>(in + 9), SafeLoad<uint32_t>(in + 9) };
  shifts = simd_batch{ 0, 2, 12, 22 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  in += 10;
  return in;
}

inline static const uint32_t* unpack11_32(const uint32_t* in, uint32_t* out) {
  uint32_t mask = 0x7ff;

  simd_batch masks(mask);
  simd_batch words, shifts;
  simd_batch results;

  // extract 11-bit bundles 0 to 3
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) >> 22 | SafeLoad<uint32_t>(in + 1) << 10, SafeLoad<uint32_t>(in + 1) };
  shifts = simd_batch{ 0, 11, 0, 1 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 11-bit bundles 4 to 7
  words = simd_batch{ SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1) >> 23 | SafeLoad<uint32_t>(in + 2) << 9, SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2) };
  shifts = simd_batch{ 12, 0, 2, 13 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 11-bit bundles 8 to 11
  words = simd_batch{ SafeLoad<uint32_t>(in + 2) >> 24 | SafeLoad<uint32_t>(in + 3) << 8, SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3) >> 25 | SafeLoad<uint32_t>(in + 4) << 7 };
  shifts = simd_batch{ 0, 3, 14, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 11-bit bundles 12 to 15
  words = simd_batch{ SafeLoad<uint32_t>(in + 4), SafeLoad<uint32_t>(in + 4), SafeLoad<uint32_t>(in + 4) >> 26 | SafeLoad<uint32_t>(in + 5) << 6, SafeLoad<uint32_t>(in + 5) };
  shifts = simd_batch{ 4, 15, 0, 5 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 11-bit bundles 16 to 19
  words = simd_batch{ SafeLoad<uint32_t>(in + 5), SafeLoad<uint32_t>(in + 5) >> 27 | SafeLoad<uint32_t>(in + 6) << 5, SafeLoad<uint32_t>(in + 6), SafeLoad<uint32_t>(in + 6) };
  shifts = simd_batch{ 16, 0, 6, 17 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 11-bit bundles 20 to 23
  words = simd_batch{ SafeLoad<uint32_t>(in + 6) >> 28 | SafeLoad<uint32_t>(in + 7) << 4, SafeLoad<uint32_t>(in + 7), SafeLoad<uint32_t>(in + 7), SafeLoad<uint32_t>(in + 7) >> 29 | SafeLoad<uint32_t>(in + 8) << 3 };
  shifts = simd_batch{ 0, 7, 18, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 11-bit bundles 24 to 27
  words = simd_batch{ SafeLoad<uint32_t>(in + 8), SafeLoad<uint32_t>(in + 8), SafeLoad<uint32_t>(in + 8) >> 30 | SafeLoad<uint32_t>(in + 9) << 2, SafeLoad<uint32_t>(in + 9) };
  shifts = simd_batch{ 8, 19, 0, 9 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 11-bit bundles 28 to 31
  words = simd_batch{ SafeLoad<uint32_t>(in + 9), SafeLoad<uint32_t>(in + 9) >> 31 | SafeLoad<uint32_t>(in + 10) << 1, SafeLoad<uint32_t>(in + 10), SafeLoad<uint32_t>(in + 10) };
  shifts = simd_batch{ 20, 0, 10, 21 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  in += 11;
  return in;
}

inline static const uint32_t* unpack12_32(const uint32_t* in, uint32_t* out) {
  uint32_t mask = 0xfff;

  simd_batch masks(mask);
  simd_batch words, shifts;
  simd_batch results;

  // extract 12-bit bundles 0 to 3
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) >> 24 | SafeLoad<uint32_t>(in + 1) << 8, SafeLoad<uint32_t>(in + 1) };
  shifts = simd_batch{ 0, 12, 0, 4 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 12-bit bundles 4 to 7
  words = simd_batch{ SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1) >> 28 | SafeLoad<uint32_t>(in + 2) << 4, SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2) };
  shifts = simd_batch{ 16, 0, 8, 20 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 12-bit bundles 8 to 11
  words = simd_batch{ SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3) >> 24 | SafeLoad<uint32_t>(in + 4) << 8, SafeLoad<uint32_t>(in + 4) };
  shifts = simd_batch{ 0, 12, 0, 4 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 12-bit bundles 12 to 15
  words = simd_batch{ SafeLoad<uint32_t>(in + 4), SafeLoad<uint32_t>(in + 4) >> 28 | SafeLoad<uint32_t>(in + 5) << 4, SafeLoad<uint32_t>(in + 5), SafeLoad<uint32_t>(in + 5) };
  shifts = simd_batch{ 16, 0, 8, 20 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 12-bit bundles 16 to 19
  words = simd_batch{ SafeLoad<uint32_t>(in + 6), SafeLoad<uint32_t>(in + 6), SafeLoad<uint32_t>(in + 6) >> 24 | SafeLoad<uint32_t>(in + 7) << 8, SafeLoad<uint32_t>(in + 7) };
  shifts = simd_batch{ 0, 12, 0, 4 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 12-bit bundles 20 to 23
  words = simd_batch{ SafeLoad<uint32_t>(in + 7), SafeLoad<uint32_t>(in + 7) >> 28 | SafeLoad<uint32_t>(in + 8) << 4, SafeLoad<uint32_t>(in + 8), SafeLoad<uint32_t>(in + 8) };
  shifts = simd_batch{ 16, 0, 8, 20 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 12-bit bundles 24 to 27
  words = simd_batch{ SafeLoad<uint32_t>(in + 9), SafeLoad<uint32_t>(in + 9), SafeLoad<uint32_t>(in + 9) >> 24 | SafeLoad<uint32_t>(in + 10) << 8, SafeLoad<uint32_t>(in + 10) };
  shifts = simd_batch{ 0, 12, 0, 4 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 12-bit bundles 28 to 31
  words = simd_batch{ SafeLoad<uint32_t>(in + 10), SafeLoad<uint32_t>(in + 10) >> 28 | SafeLoad<uint32_t>(in + 11) << 4, SafeLoad<uint32_t>(in + 11), SafeLoad<uint32_t>(in + 11) };
  shifts = simd_batch{ 16, 0, 8, 20 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  in += 12;
  return in;
}

inline static const uint32_t* unpack13_32(const uint32_t* in, uint32_t* out) {
  uint32_t mask = 0x1fff;

  simd_batch masks(mask);
  simd_batch words, shifts;
  simd_batch results;

  // extract 13-bit bundles 0 to 3
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) >> 26 | SafeLoad<uint32_t>(in + 1) << 6, SafeLoad<uint32_t>(in + 1) };
  shifts = simd_batch{ 0, 13, 0, 7 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 13-bit bundles 4 to 7
  words = simd_batch{ SafeLoad<uint32_t>(in + 1) >> 20 | SafeLoad<uint32_t>(in + 2) << 12, SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2) >> 27 | SafeLoad<uint32_t>(in + 3) << 5 };
  shifts = simd_batch{ 0, 1, 14, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 13-bit bundles 8 to 11
  words = simd_batch{ SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3) >> 21 | SafeLoad<uint32_t>(in + 4) << 11, SafeLoad<uint32_t>(in + 4), SafeLoad<uint32_t>(in + 4) };
  shifts = simd_batch{ 8, 0, 2, 15 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 13-bit bundles 12 to 15
  words = simd_batch{ SafeLoad<uint32_t>(in + 4) >> 28 | SafeLoad<uint32_t>(in + 5) << 4, SafeLoad<uint32_t>(in + 5), SafeLoad<uint32_t>(in + 5) >> 22 | SafeLoad<uint32_t>(in + 6) << 10, SafeLoad<uint32_t>(in + 6) };
  shifts = simd_batch{ 0, 9, 0, 3 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 13-bit bundles 16 to 19
  words = simd_batch{ SafeLoad<uint32_t>(in + 6), SafeLoad<uint32_t>(in + 6) >> 29 | SafeLoad<uint32_t>(in + 7) << 3, SafeLoad<uint32_t>(in + 7), SafeLoad<uint32_t>(in + 7) >> 23 | SafeLoad<uint32_t>(in + 8) << 9 };
  shifts = simd_batch{ 16, 0, 10, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 13-bit bundles 20 to 23
  words = simd_batch{ SafeLoad<uint32_t>(in + 8), SafeLoad<uint32_t>(in + 8), SafeLoad<uint32_t>(in + 8) >> 30 | SafeLoad<uint32_t>(in + 9) << 2, SafeLoad<uint32_t>(in + 9) };
  shifts = simd_batch{ 4, 17, 0, 11 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 13-bit bundles 24 to 27
  words = simd_batch{ SafeLoad<uint32_t>(in + 9) >> 24 | SafeLoad<uint32_t>(in + 10) << 8, SafeLoad<uint32_t>(in + 10), SafeLoad<uint32_t>(in + 10), SafeLoad<uint32_t>(in + 10) >> 31 | SafeLoad<uint32_t>(in + 11) << 1 };
  shifts = simd_batch{ 0, 5, 18, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 13-bit bundles 28 to 31
  words = simd_batch{ SafeLoad<uint32_t>(in + 11), SafeLoad<uint32_t>(in + 11) >> 25 | SafeLoad<uint32_t>(in + 12) << 7, SafeLoad<uint32_t>(in + 12), SafeLoad<uint32_t>(in + 12) };
  shifts = simd_batch{ 12, 0, 6, 19 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  in += 13;
  return in;
}

inline static const uint32_t* unpack14_32(const uint32_t* in, uint32_t* out) {
  uint32_t mask = 0x3fff;

  simd_batch masks(mask);
  simd_batch words, shifts;
  simd_batch results;

  // extract 14-bit bundles 0 to 3
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) >> 28 | SafeLoad<uint32_t>(in + 1) << 4, SafeLoad<uint32_t>(in + 1) };
  shifts = simd_batch{ 0, 14, 0, 10 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 14-bit bundles 4 to 7
  words = simd_batch{ SafeLoad<uint32_t>(in + 1) >> 24 | SafeLoad<uint32_t>(in + 2) << 8, SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2) >> 20 | SafeLoad<uint32_t>(in + 3) << 12, SafeLoad<uint32_t>(in + 3) };
  shifts = simd_batch{ 0, 6, 0, 2 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 14-bit bundles 8 to 11
  words = simd_batch{ SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3) >> 30 | SafeLoad<uint32_t>(in + 4) << 2, SafeLoad<uint32_t>(in + 4), SafeLoad<uint32_t>(in + 4) >> 26 | SafeLoad<uint32_t>(in + 5) << 6 };
  shifts = simd_batch{ 16, 0, 12, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 14-bit bundles 12 to 15
  words = simd_batch{ SafeLoad<uint32_t>(in + 5), SafeLoad<uint32_t>(in + 5) >> 22 | SafeLoad<uint32_t>(in + 6) << 10, SafeLoad<uint32_t>(in + 6), SafeLoad<uint32_t>(in + 6) };
  shifts = simd_batch{ 8, 0, 4, 18 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 14-bit bundles 16 to 19
  words = simd_batch{ SafeLoad<uint32_t>(in + 7), SafeLoad<uint32_t>(in + 7), SafeLoad<uint32_t>(in + 7) >> 28 | SafeLoad<uint32_t>(in + 8) << 4, SafeLoad<uint32_t>(in + 8) };
  shifts = simd_batch{ 0, 14, 0, 10 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 14-bit bundles 20 to 23
  words = simd_batch{ SafeLoad<uint32_t>(in + 8) >> 24 | SafeLoad<uint32_t>(in + 9) << 8, SafeLoad<uint32_t>(in + 9), SafeLoad<uint32_t>(in + 9) >> 20 | SafeLoad<uint32_t>(in + 10) << 12, SafeLoad<uint32_t>(in + 10) };
  shifts = simd_batch{ 0, 6, 0, 2 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 14-bit bundles 24 to 27
  words = simd_batch{ SafeLoad<uint32_t>(in + 10), SafeLoad<uint32_t>(in + 10) >> 30 | SafeLoad<uint32_t>(in + 11) << 2, SafeLoad<uint32_t>(in + 11), SafeLoad<uint32_t>(in + 11) >> 26 | SafeLoad<uint32_t>(in + 12) << 6 };
  shifts = simd_batch{ 16, 0, 12, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 14-bit bundles 28 to 31
  words = simd_batch{ SafeLoad<uint32_t>(in + 12), SafeLoad<uint32_t>(in + 12) >> 22 | SafeLoad<uint32_t>(in + 13) << 10, SafeLoad<uint32_t>(in + 13), SafeLoad<uint32_t>(in + 13) };
  shifts = simd_batch{ 8, 0, 4, 18 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  in += 14;
  return in;
}

inline static const uint32_t* unpack15_32(const uint32_t* in, uint32_t* out) {
  uint32_t mask = 0x7fff;

  simd_batch masks(mask);
  simd_batch words, shifts;
  simd_batch results;

  // extract 15-bit bundles 0 to 3
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) >> 30 | SafeLoad<uint32_t>(in + 1) << 2, SafeLoad<uint32_t>(in + 1) };
  shifts = simd_batch{ 0, 15, 0, 13 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 15-bit bundles 4 to 7
  words = simd_batch{ SafeLoad<uint32_t>(in + 1) >> 28 | SafeLoad<uint32_t>(in + 2) << 4, SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2) >> 26 | SafeLoad<uint32_t>(in + 3) << 6, SafeLoad<uint32_t>(in + 3) };
  shifts = simd_batch{ 0, 11, 0, 9 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 15-bit bundles 8 to 11
  words = simd_batch{ SafeLoad<uint32_t>(in + 3) >> 24 | SafeLoad<uint32_t>(in + 4) << 8, SafeLoad<uint32_t>(in + 4), SafeLoad<uint32_t>(in + 4) >> 22 | SafeLoad<uint32_t>(in + 5) << 10, SafeLoad<uint32_t>(in + 5) };
  shifts = simd_batch{ 0, 7, 0, 5 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 15-bit bundles 12 to 15
  words = simd_batch{ SafeLoad<uint32_t>(in + 5) >> 20 | SafeLoad<uint32_t>(in + 6) << 12, SafeLoad<uint32_t>(in + 6), SafeLoad<uint32_t>(in + 6) >> 18 | SafeLoad<uint32_t>(in + 7) << 14, SafeLoad<uint32_t>(in + 7) };
  shifts = simd_batch{ 0, 3, 0, 1 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 15-bit bundles 16 to 19
  words = simd_batch{ SafeLoad<uint32_t>(in + 7), SafeLoad<uint32_t>(in + 7) >> 31 | SafeLoad<uint32_t>(in + 8) << 1, SafeLoad<uint32_t>(in + 8), SafeLoad<uint32_t>(in + 8) >> 29 | SafeLoad<uint32_t>(in + 9) << 3 };
  shifts = simd_batch{ 16, 0, 14, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 15-bit bundles 20 to 23
  words = simd_batch{ SafeLoad<uint32_t>(in + 9), SafeLoad<uint32_t>(in + 9) >> 27 | SafeLoad<uint32_t>(in + 10) << 5, SafeLoad<uint32_t>(in + 10), SafeLoad<uint32_t>(in + 10) >> 25 | SafeLoad<uint32_t>(in + 11) << 7 };
  shifts = simd_batch{ 12, 0, 10, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 15-bit bundles 24 to 27
  words = simd_batch{ SafeLoad<uint32_t>(in + 11), SafeLoad<uint32_t>(in + 11) >> 23 | SafeLoad<uint32_t>(in + 12) << 9, SafeLoad<uint32_t>(in + 12), SafeLoad<uint32_t>(in + 12) >> 21 | SafeLoad<uint32_t>(in + 13) << 11 };
  shifts = simd_batch{ 8, 0, 6, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 15-bit bundles 28 to 31
  words = simd_batch{ SafeLoad<uint32_t>(in + 13), SafeLoad<uint32_t>(in + 13) >> 19 | SafeLoad<uint32_t>(in + 14) << 13, SafeLoad<uint32_t>(in + 14), SafeLoad<uint32_t>(in + 14) };
  shifts = simd_batch{ 4, 0, 2, 17 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  in += 15;
  return in;
}

inline static const uint32_t* unpack16_32(const uint32_t* in, uint32_t* out) {
  uint32_t mask = 0xffff;

  simd_batch masks(mask);
  simd_batch words, shifts;
  simd_batch results;

  // extract 16-bit bundles 0 to 3
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1) };
  shifts = simd_batch{ 0, 16, 0, 16 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 16-bit bundles 4 to 7
  words = simd_batch{ SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3) };
  shifts = simd_batch{ 0, 16, 0, 16 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 16-bit bundles 8 to 11
  words = simd_batch{ SafeLoad<uint32_t>(in + 4), SafeLoad<uint32_t>(in + 4), SafeLoad<uint32_t>(in + 5), SafeLoad<uint32_t>(in + 5) };
  shifts = simd_batch{ 0, 16, 0, 16 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 16-bit bundles 12 to 15
  words = simd_batch{ SafeLoad<uint32_t>(in + 6), SafeLoad<uint32_t>(in + 6), SafeLoad<uint32_t>(in + 7), SafeLoad<uint32_t>(in + 7) };
  shifts = simd_batch{ 0, 16, 0, 16 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 16-bit bundles 16 to 19
  words = simd_batch{ SafeLoad<uint32_t>(in + 8), SafeLoad<uint32_t>(in + 8), SafeLoad<uint32_t>(in + 9), SafeLoad<uint32_t>(in + 9) };
  shifts = simd_batch{ 0, 16, 0, 16 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 16-bit bundles 20 to 23
  words = simd_batch{ SafeLoad<uint32_t>(in + 10), SafeLoad<uint32_t>(in + 10), SafeLoad<uint32_t>(in + 11), SafeLoad<uint32_t>(in + 11) };
  shifts = simd_batch{ 0, 16, 0, 16 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 16-bit bundles 24 to 27
  words = simd_batch{ SafeLoad<uint32_t>(in + 12), SafeLoad<uint32_t>(in + 12), SafeLoad<uint32_t>(in + 13), SafeLoad<uint32_t>(in + 13) };
  shifts = simd_batch{ 0, 16, 0, 16 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 16-bit bundles 28 to 31
  words = simd_batch{ SafeLoad<uint32_t>(in + 14), SafeLoad<uint32_t>(in + 14), SafeLoad<uint32_t>(in + 15), SafeLoad<uint32_t>(in + 15) };
  shifts = simd_batch{ 0, 16, 0, 16 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  in += 16;
  return in;
}

inline static const uint32_t* unpack17_32(const uint32_t* in, uint32_t* out) {
  uint32_t mask = 0x1ffff;

  simd_batch masks(mask);
  simd_batch words, shifts;
  simd_batch results;

  // extract 17-bit bundles 0 to 3
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) >> 17 | SafeLoad<uint32_t>(in + 1) << 15, SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1) >> 19 | SafeLoad<uint32_t>(in + 2) << 13 };
  shifts = simd_batch{ 0, 0, 2, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 17-bit bundles 4 to 7
  words = simd_batch{ SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2) >> 21 | SafeLoad<uint32_t>(in + 3) << 11, SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3) >> 23 | SafeLoad<uint32_t>(in + 4) << 9 };
  shifts = simd_batch{ 4, 0, 6, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 17-bit bundles 8 to 11
  words = simd_batch{ SafeLoad<uint32_t>(in + 4), SafeLoad<uint32_t>(in + 4) >> 25 | SafeLoad<uint32_t>(in + 5) << 7, SafeLoad<uint32_t>(in + 5), SafeLoad<uint32_t>(in + 5) >> 27 | SafeLoad<uint32_t>(in + 6) << 5 };
  shifts = simd_batch{ 8, 0, 10, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 17-bit bundles 12 to 15
  words = simd_batch{ SafeLoad<uint32_t>(in + 6), SafeLoad<uint32_t>(in + 6) >> 29 | SafeLoad<uint32_t>(in + 7) << 3, SafeLoad<uint32_t>(in + 7), SafeLoad<uint32_t>(in + 7) >> 31 | SafeLoad<uint32_t>(in + 8) << 1 };
  shifts = simd_batch{ 12, 0, 14, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 17-bit bundles 16 to 19
  words = simd_batch{ SafeLoad<uint32_t>(in + 8) >> 16 | SafeLoad<uint32_t>(in + 9) << 16, SafeLoad<uint32_t>(in + 9), SafeLoad<uint32_t>(in + 9) >> 18 | SafeLoad<uint32_t>(in + 10) << 14, SafeLoad<uint32_t>(in + 10) };
  shifts = simd_batch{ 0, 1, 0, 3 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 17-bit bundles 20 to 23
  words = simd_batch{ SafeLoad<uint32_t>(in + 10) >> 20 | SafeLoad<uint32_t>(in + 11) << 12, SafeLoad<uint32_t>(in + 11), SafeLoad<uint32_t>(in + 11) >> 22 | SafeLoad<uint32_t>(in + 12) << 10, SafeLoad<uint32_t>(in + 12) };
  shifts = simd_batch{ 0, 5, 0, 7 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 17-bit bundles 24 to 27
  words = simd_batch{ SafeLoad<uint32_t>(in + 12) >> 24 | SafeLoad<uint32_t>(in + 13) << 8, SafeLoad<uint32_t>(in + 13), SafeLoad<uint32_t>(in + 13) >> 26 | SafeLoad<uint32_t>(in + 14) << 6, SafeLoad<uint32_t>(in + 14) };
  shifts = simd_batch{ 0, 9, 0, 11 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 17-bit bundles 28 to 31
  words = simd_batch{ SafeLoad<uint32_t>(in + 14) >> 28 | SafeLoad<uint32_t>(in + 15) << 4, SafeLoad<uint32_t>(in + 15), SafeLoad<uint32_t>(in + 15) >> 30 | SafeLoad<uint32_t>(in + 16) << 2, SafeLoad<uint32_t>(in + 16) };
  shifts = simd_batch{ 0, 13, 0, 15 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  in += 17;
  return in;
}

inline static const uint32_t* unpack18_32(const uint32_t* in, uint32_t* out) {
  uint32_t mask = 0x3ffff;

  simd_batch masks(mask);
  simd_batch words, shifts;
  simd_batch results;

  // extract 18-bit bundles 0 to 3
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) >> 18 | SafeLoad<uint32_t>(in + 1) << 14, SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1) >> 22 | SafeLoad<uint32_t>(in + 2) << 10 };
  shifts = simd_batch{ 0, 0, 4, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 18-bit bundles 4 to 7
  words = simd_batch{ SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2) >> 26 | SafeLoad<uint32_t>(in + 3) << 6, SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3) >> 30 | SafeLoad<uint32_t>(in + 4) << 2 };
  shifts = simd_batch{ 8, 0, 12, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 18-bit bundles 8 to 11
  words = simd_batch{ SafeLoad<uint32_t>(in + 4) >> 16 | SafeLoad<uint32_t>(in + 5) << 16, SafeLoad<uint32_t>(in + 5), SafeLoad<uint32_t>(in + 5) >> 20 | SafeLoad<uint32_t>(in + 6) << 12, SafeLoad<uint32_t>(in + 6) };
  shifts = simd_batch{ 0, 2, 0, 6 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 18-bit bundles 12 to 15
  words = simd_batch{ SafeLoad<uint32_t>(in + 6) >> 24 | SafeLoad<uint32_t>(in + 7) << 8, SafeLoad<uint32_t>(in + 7), SafeLoad<uint32_t>(in + 7) >> 28 | SafeLoad<uint32_t>(in + 8) << 4, SafeLoad<uint32_t>(in + 8) };
  shifts = simd_batch{ 0, 10, 0, 14 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 18-bit bundles 16 to 19
  words = simd_batch{ SafeLoad<uint32_t>(in + 9), SafeLoad<uint32_t>(in + 9) >> 18 | SafeLoad<uint32_t>(in + 10) << 14, SafeLoad<uint32_t>(in + 10), SafeLoad<uint32_t>(in + 10) >> 22 | SafeLoad<uint32_t>(in + 11) << 10 };
  shifts = simd_batch{ 0, 0, 4, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 18-bit bundles 20 to 23
  words = simd_batch{ SafeLoad<uint32_t>(in + 11), SafeLoad<uint32_t>(in + 11) >> 26 | SafeLoad<uint32_t>(in + 12) << 6, SafeLoad<uint32_t>(in + 12), SafeLoad<uint32_t>(in + 12) >> 30 | SafeLoad<uint32_t>(in + 13) << 2 };
  shifts = simd_batch{ 8, 0, 12, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 18-bit bundles 24 to 27
  words = simd_batch{ SafeLoad<uint32_t>(in + 13) >> 16 | SafeLoad<uint32_t>(in + 14) << 16, SafeLoad<uint32_t>(in + 14), SafeLoad<uint32_t>(in + 14) >> 20 | SafeLoad<uint32_t>(in + 15) << 12, SafeLoad<uint32_t>(in + 15) };
  shifts = simd_batch{ 0, 2, 0, 6 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 18-bit bundles 28 to 31
  words = simd_batch{ SafeLoad<uint32_t>(in + 15) >> 24 | SafeLoad<uint32_t>(in + 16) << 8, SafeLoad<uint32_t>(in + 16), SafeLoad<uint32_t>(in + 16) >> 28 | SafeLoad<uint32_t>(in + 17) << 4, SafeLoad<uint32_t>(in + 17) };
  shifts = simd_batch{ 0, 10, 0, 14 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  in += 18;
  return in;
}

inline static const uint32_t* unpack19_32(const uint32_t* in, uint32_t* out) {
  uint32_t mask = 0x7ffff;

  simd_batch masks(mask);
  simd_batch words, shifts;
  simd_batch results;

  // extract 19-bit bundles 0 to 3
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) >> 19 | SafeLoad<uint32_t>(in + 1) << 13, SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1) >> 25 | SafeLoad<uint32_t>(in + 2) << 7 };
  shifts = simd_batch{ 0, 0, 6, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 19-bit bundles 4 to 7
  words = simd_batch{ SafeLoad<uint32_t>(in + 2), SafeLoad<uint32_t>(in + 2) >> 31 | SafeLoad<uint32_t>(in + 3) << 1, SafeLoad<uint32_t>(in + 3) >> 18 | SafeLoad<uint32_t>(in + 4) << 14, SafeLoad<uint32_t>(in + 4) };
  shifts = simd_batch{ 12, 0, 0, 5 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 19-bit bundles 8 to 11
  words = simd_batch{ SafeLoad<uint32_t>(in + 4) >> 24 | SafeLoad<uint32_t>(in + 5) << 8, SafeLoad<uint32_t>(in + 5), SafeLoad<uint32_t>(in + 5) >> 30 | SafeLoad<uint32_t>(in + 6) << 2, SafeLoad<uint32_t>(in + 6) >> 17 | SafeLoad<uint32_t>(in + 7) << 15 };
  shifts = simd_batch{ 0, 11, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 19-bit bundles 12 to 15
  words = simd_batch{ SafeLoad<uint32_t>(in + 7), SafeLoad<uint32_t>(in + 7) >> 23 | SafeLoad<uint32_t>(in + 8) << 9, SafeLoad<uint32_t>(in + 8), SafeLoad<uint32_t>(in + 8) >> 29 | SafeLoad<uint32_t>(in + 9) << 3 };
  shifts = simd_batch{ 4, 0, 10, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 19-bit bundles 16 to 19
  words = simd_batch{ SafeLoad<uint32_t>(in + 9) >> 16 | SafeLoad<uint32_t>(in + 10) << 16, SafeLoad<uint32_t>(in + 10), SafeLoad<uint32_t>(in + 10) >> 22 | SafeLoad<uint32_t>(in + 11) << 10, SafeLoad<uint32_t>(in + 11) };
  shifts = simd_batch{ 0, 3, 0, 9 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 19-bit bundles 20 to 23
  words = simd_batch{ SafeLoad<uint32_t>(in + 11) >> 28 | SafeLoad<uint32_t>(in + 12) << 4, SafeLoad<uint32_t>(in + 12) >> 15 | SafeLoad<uint32_t>(in + 13) << 17, SafeLoad<uint32_t>(in + 13), SafeLoad<uint32_t>(in + 13) >> 21 | SafeLoad<uint32_t>(in + 14) << 11 };
  shifts = simd_batch{ 0, 0, 2, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 19-bit bundles 24 to 27
  words = simd_batch{ SafeLoad<uint32_t>(in + 14), SafeLoad<uint32_t>(in + 14) >> 27 | SafeLoad<uint32_t>(in + 15) << 5, SafeLoad<uint32_t>(in + 15) >> 14 | SafeLoad<uint32_t>(in + 16) << 18, SafeLoad<uint32_t>(in + 16) };
  shifts = simd_batch{ 8, 0, 0, 1 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 19-bit bundles 28 to 31
  words = simd_batch{ SafeLoad<uint32_t>(in + 16) >> 20 | SafeLoad<uint32_t>(in + 17) << 12, SafeLoad<uint32_t>(in + 17), SafeLoad<uint32_t>(in + 17) >> 26 | SafeLoad<uint32_t>(in + 18) << 6, SafeLoad<uint32_t>(in + 18) };
  shifts = simd_batch{ 0, 7, 0, 13 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  in += 19;
  return in;
}

inline static const uint32_t* unpack20_32(const uint32_t* in, uint32_t* out) {
  uint32_t mask = 0xfffff;

  simd_batch masks(mask);
  simd_batch words, shifts;
  simd_batch results;

  // extract 20-bit bundles 0 to 3
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) >> 20 | SafeLoad<uint32_t>(in + 1) << 12, SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1) >> 28 | SafeLoad<uint32_t>(in + 2) << 4 };
  shifts = simd_batch{ 0, 0, 8, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 20-bit bundles 4 to 7
  words = simd_batch{ SafeLoad<uint32_t>(in + 2) >> 16 | SafeLoad<uint32_t>(in + 3) << 16, SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3) >> 24 | SafeLoad<uint32_t>(in + 4) << 8, SafeLoad<uint32_t>(in + 4) };
  shifts = simd_batch{ 0, 4, 0, 12 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 20-bit bundles 8 to 11
  words = simd_batch{ SafeLoad<uint32_t>(in + 5), SafeLoad<uint32_t>(in + 5) >> 20 | SafeLoad<uint32_t>(in + 6) << 12, SafeLoad<uint32_t>(in + 6), SafeLoad<uint32_t>(in + 6) >> 28 | SafeLoad<uint32_t>(in + 7) << 4 };
  shifts = simd_batch{ 0, 0, 8, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 20-bit bundles 12 to 15
  words = simd_batch{ SafeLoad<uint32_t>(in + 7) >> 16 | SafeLoad<uint32_t>(in + 8) << 16, SafeLoad<uint32_t>(in + 8), SafeLoad<uint32_t>(in + 8) >> 24 | SafeLoad<uint32_t>(in + 9) << 8, SafeLoad<uint32_t>(in + 9) };
  shifts = simd_batch{ 0, 4, 0, 12 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 20-bit bundles 16 to 19
  words = simd_batch{ SafeLoad<uint32_t>(in + 10), SafeLoad<uint32_t>(in + 10) >> 20 | SafeLoad<uint32_t>(in + 11) << 12, SafeLoad<uint32_t>(in + 11), SafeLoad<uint32_t>(in + 11) >> 28 | SafeLoad<uint32_t>(in + 12) << 4 };
  shifts = simd_batch{ 0, 0, 8, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 20-bit bundles 20 to 23
  words = simd_batch{ SafeLoad<uint32_t>(in + 12) >> 16 | SafeLoad<uint32_t>(in + 13) << 16, SafeLoad<uint32_t>(in + 13), SafeLoad<uint32_t>(in + 13) >> 24 | SafeLoad<uint32_t>(in + 14) << 8, SafeLoad<uint32_t>(in + 14) };
  shifts = simd_batch{ 0, 4, 0, 12 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 20-bit bundles 24 to 27
  words = simd_batch{ SafeLoad<uint32_t>(in + 15), SafeLoad<uint32_t>(in + 15) >> 20 | SafeLoad<uint32_t>(in + 16) << 12, SafeLoad<uint32_t>(in + 16), SafeLoad<uint32_t>(in + 16) >> 28 | SafeLoad<uint32_t>(in + 17) << 4 };
  shifts = simd_batch{ 0, 0, 8, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 20-bit bundles 28 to 31
  words = simd_batch{ SafeLoad<uint32_t>(in + 17) >> 16 | SafeLoad<uint32_t>(in + 18) << 16, SafeLoad<uint32_t>(in + 18), SafeLoad<uint32_t>(in + 18) >> 24 | SafeLoad<uint32_t>(in + 19) << 8, SafeLoad<uint32_t>(in + 19) };
  shifts = simd_batch{ 0, 4, 0, 12 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  in += 20;
  return in;
}

inline static const uint32_t* unpack21_32(const uint32_t* in, uint32_t* out) {
  uint32_t mask = 0x1fffff;

  simd_batch masks(mask);
  simd_batch words, shifts;
  simd_batch results;

  // extract 21-bit bundles 0 to 3
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) >> 21 | SafeLoad<uint32_t>(in + 1) << 11, SafeLoad<uint32_t>(in + 1), SafeLoad<uint32_t>(in + 1) >> 31 | SafeLoad<uint32_t>(in + 2) << 1 };
  shifts = simd_batch{ 0, 0, 10, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 21-bit bundles 4 to 7
  words = simd_batch{ SafeLoad<uint32_t>(in + 2) >> 20 | SafeLoad<uint32_t>(in + 3) << 12, SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3) >> 30 | SafeLoad<uint32_t>(in + 4) << 2, SafeLoad<uint32_t>(in + 4) >> 19 | SafeLoad<uint32_t>(in + 5) << 13 };
  shifts = simd_batch{ 0, 9, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 21-bit bundles 8 to 11
  words = simd_batch{ SafeLoad<uint32_t>(in + 5), SafeLoad<uint32_t>(in + 5) >> 29 | SafeLoad<uint32_t>(in + 6) << 3, SafeLoad<uint32_t>(in + 6) >> 18 | SafeLoad<uint32_t>(in + 7) << 14, SafeLoad<uint32_t>(in + 7) };
  shifts = simd_batch{ 8, 0, 0, 7 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 21-bit bundles 12 to 15
  words = simd_batch{ SafeLoad<uint32_t>(in + 7) >> 28 | SafeLoad<uint32_t>(in + 8) << 4, SafeLoad<uint32_t>(in + 8) >> 17 | SafeLoad<uint32_t>(in + 9) << 15, SafeLoad<uint32_t>(in + 9), SafeLoad<uint32_t>(in + 9) >> 27 | SafeLoad<uint32_t>(in + 10) << 5 };
  shifts = simd_batch{ 0, 0, 6, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 21-bit bundles 16 to 19
  words = simd_batch{ SafeLoad<uint32_t>(in + 10) >> 16 | SafeLoad<uint32_t>(in + 11) << 16, SafeLoad<uint32_t>(in + 11), SafeLoad<uint32_t>(in + 11) >> 26 | SafeLoad<uint32_t>(in + 12) << 6, SafeLoad<uint32_t>(in + 12) >> 15 | SafeLoad<uint32_t>(in + 13) << 17 };
  shifts = simd_batch{ 0, 5, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 21-bit bundles 20 to 23
  words = simd_batch{ SafeLoad<uint32_t>(in + 13), SafeLoad<uint32_t>(in + 13) >> 25 | SafeLoad<uint32_t>(in + 14) << 7, SafeLoad<uint32_t>(in + 14) >> 14 | SafeLoad<uint32_t>(in + 15) << 18, SafeLoad<uint32_t>(in + 15) };
  shifts = simd_batch{ 4, 0, 0, 3 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 21-bit bundles 24 to 27
  words = simd_batch{ SafeLoad<uint32_t>(in + 15) >> 24 | SafeLoad<uint32_t>(in + 16) << 8, SafeLoad<uint32_t>(in + 16) >> 13 | SafeLoad<uint32_t>(in + 17) << 19, SafeLoad<uint32_t>(in + 17), SafeLoad<uint32_t>(in + 17) >> 23 | SafeLoad<uint32_t>(in + 18) << 9 };
  shifts = simd_batch{ 0, 0, 2, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 21-bit bundles 28 to 31
  words = simd_batch{ SafeLoad<uint32_t>(in + 18) >> 12 | SafeLoad<uint32_t>(in + 19) << 20, SafeLoad<uint32_t>(in + 19), SafeLoad<uint32_t>(in + 19) >> 22 | SafeLoad<uint32_t>(in + 20) << 10, SafeLoad<uint32_t>(in + 20) };
  shifts = simd_batch{ 0, 1, 0, 11 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  in += 21;
  return in;
}

inline static const uint32_t* unpack22_32(const uint32_t* in, uint32_t* out) {
  uint32_t mask = 0x3fffff;

  simd_batch masks(mask);
  simd_batch words, shifts;
  simd_batch results;

  // extract 22-bit bundles 0 to 3
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) >> 22 | SafeLoad<uint32_t>(in + 1) << 10, SafeLoad<uint32_t>(in + 1) >> 12 | SafeLoad<uint32_t>(in + 2) << 20, SafeLoad<uint32_t>(in + 2) };
  shifts = simd_batch{ 0, 0, 0, 2 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 22-bit bundles 4 to 7
  words = simd_batch{ SafeLoad<uint32_t>(in + 2) >> 24 | SafeLoad<uint32_t>(in + 3) << 8, SafeLoad<uint32_t>(in + 3) >> 14 | SafeLoad<uint32_t>(in + 4) << 18, SafeLoad<uint32_t>(in + 4), SafeLoad<uint32_t>(in + 4) >> 26 | SafeLoad<uint32_t>(in + 5) << 6 };
  shifts = simd_batch{ 0, 0, 4, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 22-bit bundles 8 to 11
  words = simd_batch{ SafeLoad<uint32_t>(in + 5) >> 16 | SafeLoad<uint32_t>(in + 6) << 16, SafeLoad<uint32_t>(in + 6), SafeLoad<uint32_t>(in + 6) >> 28 | SafeLoad<uint32_t>(in + 7) << 4, SafeLoad<uint32_t>(in + 7) >> 18 | SafeLoad<uint32_t>(in + 8) << 14 };
  shifts = simd_batch{ 0, 6, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 22-bit bundles 12 to 15
  words = simd_batch{ SafeLoad<uint32_t>(in + 8), SafeLoad<uint32_t>(in + 8) >> 30 | SafeLoad<uint32_t>(in + 9) << 2, SafeLoad<uint32_t>(in + 9) >> 20 | SafeLoad<uint32_t>(in + 10) << 12, SafeLoad<uint32_t>(in + 10) };
  shifts = simd_batch{ 8, 0, 0, 10 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 22-bit bundles 16 to 19
  words = simd_batch{ SafeLoad<uint32_t>(in + 11), SafeLoad<uint32_t>(in + 11) >> 22 | SafeLoad<uint32_t>(in + 12) << 10, SafeLoad<uint32_t>(in + 12) >> 12 | SafeLoad<uint32_t>(in + 13) << 20, SafeLoad<uint32_t>(in + 13) };
  shifts = simd_batch{ 0, 0, 0, 2 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 22-bit bundles 20 to 23
  words = simd_batch{ SafeLoad<uint32_t>(in + 13) >> 24 | SafeLoad<uint32_t>(in + 14) << 8, SafeLoad<uint32_t>(in + 14) >> 14 | SafeLoad<uint32_t>(in + 15) << 18, SafeLoad<uint32_t>(in + 15), SafeLoad<uint32_t>(in + 15) >> 26 | SafeLoad<uint32_t>(in + 16) << 6 };
  shifts = simd_batch{ 0, 0, 4, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 22-bit bundles 24 to 27
  words = simd_batch{ SafeLoad<uint32_t>(in + 16) >> 16 | SafeLoad<uint32_t>(in + 17) << 16, SafeLoad<uint32_t>(in + 17), SafeLoad<uint32_t>(in + 17) >> 28 | SafeLoad<uint32_t>(in + 18) << 4, SafeLoad<uint32_t>(in + 18) >> 18 | SafeLoad<uint32_t>(in + 19) << 14 };
  shifts = simd_batch{ 0, 6, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 22-bit bundles 28 to 31
  words = simd_batch{ SafeLoad<uint32_t>(in + 19), SafeLoad<uint32_t>(in + 19) >> 30 | SafeLoad<uint32_t>(in + 20) << 2, SafeLoad<uint32_t>(in + 20) >> 20 | SafeLoad<uint32_t>(in + 21) << 12, SafeLoad<uint32_t>(in + 21) };
  shifts = simd_batch{ 8, 0, 0, 10 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  in += 22;
  return in;
}

inline static const uint32_t* unpack23_32(const uint32_t* in, uint32_t* out) {
  uint32_t mask = 0x7fffff;

  simd_batch masks(mask);
  simd_batch words, shifts;
  simd_batch results;

  // extract 23-bit bundles 0 to 3
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) >> 23 | SafeLoad<uint32_t>(in + 1) << 9, SafeLoad<uint32_t>(in + 1) >> 14 | SafeLoad<uint32_t>(in + 2) << 18, SafeLoad<uint32_t>(in + 2) };
  shifts = simd_batch{ 0, 0, 0, 5 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 23-bit bundles 4 to 7
  words = simd_batch{ SafeLoad<uint32_t>(in + 2) >> 28 | SafeLoad<uint32_t>(in + 3) << 4, SafeLoad<uint32_t>(in + 3) >> 19 | SafeLoad<uint32_t>(in + 4) << 13, SafeLoad<uint32_t>(in + 4) >> 10 | SafeLoad<uint32_t>(in + 5) << 22, SafeLoad<uint32_t>(in + 5) };
  shifts = simd_batch{ 0, 0, 0, 1 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 23-bit bundles 8 to 11
  words = simd_batch{ SafeLoad<uint32_t>(in + 5) >> 24 | SafeLoad<uint32_t>(in + 6) << 8, SafeLoad<uint32_t>(in + 6) >> 15 | SafeLoad<uint32_t>(in + 7) << 17, SafeLoad<uint32_t>(in + 7), SafeLoad<uint32_t>(in + 7) >> 29 | SafeLoad<uint32_t>(in + 8) << 3 };
  shifts = simd_batch{ 0, 0, 6, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 23-bit bundles 12 to 15
  words = simd_batch{ SafeLoad<uint32_t>(in + 8) >> 20 | SafeLoad<uint32_t>(in + 9) << 12, SafeLoad<uint32_t>(in + 9) >> 11 | SafeLoad<uint32_t>(in + 10) << 21, SafeLoad<uint32_t>(in + 10), SafeLoad<uint32_t>(in + 10) >> 25 | SafeLoad<uint32_t>(in + 11) << 7 };
  shifts = simd_batch{ 0, 0, 2, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 23-bit bundles 16 to 19
  words = simd_batch{ SafeLoad<uint32_t>(in + 11) >> 16 | SafeLoad<uint32_t>(in + 12) << 16, SafeLoad<uint32_t>(in + 12), SafeLoad<uint32_t>(in + 12) >> 30 | SafeLoad<uint32_t>(in + 13) << 2, SafeLoad<uint32_t>(in + 13) >> 21 | SafeLoad<uint32_t>(in + 14) << 11 };
  shifts = simd_batch{ 0, 7, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 23-bit bundles 20 to 23
  words = simd_batch{ SafeLoad<uint32_t>(in + 14) >> 12 | SafeLoad<uint32_t>(in + 15) << 20, SafeLoad<uint32_t>(in + 15), SafeLoad<uint32_t>(in + 15) >> 26 | SafeLoad<uint32_t>(in + 16) << 6, SafeLoad<uint32_t>(in + 16) >> 17 | SafeLoad<uint32_t>(in + 17) << 15 };
  shifts = simd_batch{ 0, 3, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 23-bit bundles 24 to 27
  words = simd_batch{ SafeLoad<uint32_t>(in + 17), SafeLoad<uint32_t>(in + 17) >> 31 | SafeLoad<uint32_t>(in + 18) << 1, SafeLoad<uint32_t>(in + 18) >> 22 | SafeLoad<uint32_t>(in + 19) << 10, SafeLoad<uint32_t>(in + 19) >> 13 | SafeLoad<uint32_t>(in + 20) << 19 };
  shifts = simd_batch{ 8, 0, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 23-bit bundles 28 to 31
  words = simd_batch{ SafeLoad<uint32_t>(in + 20), SafeLoad<uint32_t>(in + 20) >> 27 | SafeLoad<uint32_t>(in + 21) << 5, SafeLoad<uint32_t>(in + 21) >> 18 | SafeLoad<uint32_t>(in + 22) << 14, SafeLoad<uint32_t>(in + 22) };
  shifts = simd_batch{ 4, 0, 0, 9 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  in += 23;
  return in;
}

inline static const uint32_t* unpack24_32(const uint32_t* in, uint32_t* out) {
  uint32_t mask = 0xffffff;

  simd_batch masks(mask);
  simd_batch words, shifts;
  simd_batch results;

  // extract 24-bit bundles 0 to 3
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) >> 24 | SafeLoad<uint32_t>(in + 1) << 8, SafeLoad<uint32_t>(in + 1) >> 16 | SafeLoad<uint32_t>(in + 2) << 16, SafeLoad<uint32_t>(in + 2) };
  shifts = simd_batch{ 0, 0, 0, 8 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 24-bit bundles 4 to 7
  words = simd_batch{ SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3) >> 24 | SafeLoad<uint32_t>(in + 4) << 8, SafeLoad<uint32_t>(in + 4) >> 16 | SafeLoad<uint32_t>(in + 5) << 16, SafeLoad<uint32_t>(in + 5) };
  shifts = simd_batch{ 0, 0, 0, 8 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 24-bit bundles 8 to 11
  words = simd_batch{ SafeLoad<uint32_t>(in + 6), SafeLoad<uint32_t>(in + 6) >> 24 | SafeLoad<uint32_t>(in + 7) << 8, SafeLoad<uint32_t>(in + 7) >> 16 | SafeLoad<uint32_t>(in + 8) << 16, SafeLoad<uint32_t>(in + 8) };
  shifts = simd_batch{ 0, 0, 0, 8 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 24-bit bundles 12 to 15
  words = simd_batch{ SafeLoad<uint32_t>(in + 9), SafeLoad<uint32_t>(in + 9) >> 24 | SafeLoad<uint32_t>(in + 10) << 8, SafeLoad<uint32_t>(in + 10) >> 16 | SafeLoad<uint32_t>(in + 11) << 16, SafeLoad<uint32_t>(in + 11) };
  shifts = simd_batch{ 0, 0, 0, 8 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 24-bit bundles 16 to 19
  words = simd_batch{ SafeLoad<uint32_t>(in + 12), SafeLoad<uint32_t>(in + 12) >> 24 | SafeLoad<uint32_t>(in + 13) << 8, SafeLoad<uint32_t>(in + 13) >> 16 | SafeLoad<uint32_t>(in + 14) << 16, SafeLoad<uint32_t>(in + 14) };
  shifts = simd_batch{ 0, 0, 0, 8 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 24-bit bundles 20 to 23
  words = simd_batch{ SafeLoad<uint32_t>(in + 15), SafeLoad<uint32_t>(in + 15) >> 24 | SafeLoad<uint32_t>(in + 16) << 8, SafeLoad<uint32_t>(in + 16) >> 16 | SafeLoad<uint32_t>(in + 17) << 16, SafeLoad<uint32_t>(in + 17) };
  shifts = simd_batch{ 0, 0, 0, 8 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 24-bit bundles 24 to 27
  words = simd_batch{ SafeLoad<uint32_t>(in + 18), SafeLoad<uint32_t>(in + 18) >> 24 | SafeLoad<uint32_t>(in + 19) << 8, SafeLoad<uint32_t>(in + 19) >> 16 | SafeLoad<uint32_t>(in + 20) << 16, SafeLoad<uint32_t>(in + 20) };
  shifts = simd_batch{ 0, 0, 0, 8 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 24-bit bundles 28 to 31
  words = simd_batch{ SafeLoad<uint32_t>(in + 21), SafeLoad<uint32_t>(in + 21) >> 24 | SafeLoad<uint32_t>(in + 22) << 8, SafeLoad<uint32_t>(in + 22) >> 16 | SafeLoad<uint32_t>(in + 23) << 16, SafeLoad<uint32_t>(in + 23) };
  shifts = simd_batch{ 0, 0, 0, 8 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  in += 24;
  return in;
}

inline static const uint32_t* unpack25_32(const uint32_t* in, uint32_t* out) {
  uint32_t mask = 0x1ffffff;

  simd_batch masks(mask);
  simd_batch words, shifts;
  simd_batch results;

  // extract 25-bit bundles 0 to 3
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) >> 25 | SafeLoad<uint32_t>(in + 1) << 7, SafeLoad<uint32_t>(in + 1) >> 18 | SafeLoad<uint32_t>(in + 2) << 14, SafeLoad<uint32_t>(in + 2) >> 11 | SafeLoad<uint32_t>(in + 3) << 21 };
  shifts = simd_batch{ 0, 0, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 25-bit bundles 4 to 7
  words = simd_batch{ SafeLoad<uint32_t>(in + 3), SafeLoad<uint32_t>(in + 3) >> 29 | SafeLoad<uint32_t>(in + 4) << 3, SafeLoad<uint32_t>(in + 4) >> 22 | SafeLoad<uint32_t>(in + 5) << 10, SafeLoad<uint32_t>(in + 5) >> 15 | SafeLoad<uint32_t>(in + 6) << 17 };
  shifts = simd_batch{ 4, 0, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 25-bit bundles 8 to 11
  words = simd_batch{ SafeLoad<uint32_t>(in + 6) >> 8 | SafeLoad<uint32_t>(in + 7) << 24, SafeLoad<uint32_t>(in + 7), SafeLoad<uint32_t>(in + 7) >> 26 | SafeLoad<uint32_t>(in + 8) << 6, SafeLoad<uint32_t>(in + 8) >> 19 | SafeLoad<uint32_t>(in + 9) << 13 };
  shifts = simd_batch{ 0, 1, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 25-bit bundles 12 to 15
  words = simd_batch{ SafeLoad<uint32_t>(in + 9) >> 12 | SafeLoad<uint32_t>(in + 10) << 20, SafeLoad<uint32_t>(in + 10), SafeLoad<uint32_t>(in + 10) >> 30 | SafeLoad<uint32_t>(in + 11) << 2, SafeLoad<uint32_t>(in + 11) >> 23 | SafeLoad<uint32_t>(in + 12) << 9 };
  shifts = simd_batch{ 0, 5, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 25-bit bundles 16 to 19
  words = simd_batch{ SafeLoad<uint32_t>(in + 12) >> 16 | SafeLoad<uint32_t>(in + 13) << 16, SafeLoad<uint32_t>(in + 13) >> 9 | SafeLoad<uint32_t>(in + 14) << 23, SafeLoad<uint32_t>(in + 14), SafeLoad<uint32_t>(in + 14) >> 27 | SafeLoad<uint32_t>(in + 15) << 5 };
  shifts = simd_batch{ 0, 0, 2, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 25-bit bundles 20 to 23
  words = simd_batch{ SafeLoad<uint32_t>(in + 15) >> 20 | SafeLoad<uint32_t>(in + 16) << 12, SafeLoad<uint32_t>(in + 16) >> 13 | SafeLoad<uint32_t>(in + 17) << 19, SafeLoad<uint32_t>(in + 17), SafeLoad<uint32_t>(in + 17) >> 31 | SafeLoad<uint32_t>(in + 18) << 1 };
  shifts = simd_batch{ 0, 0, 6, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 25-bit bundles 24 to 27
  words = simd_batch{ SafeLoad<uint32_t>(in + 18) >> 24 | SafeLoad<uint32_t>(in + 19) << 8, SafeLoad<uint32_t>(in + 19) >> 17 | SafeLoad<uint32_t>(in + 20) << 15, SafeLoad<uint32_t>(in + 20) >> 10 | SafeLoad<uint32_t>(in + 21) << 22, SafeLoad<uint32_t>(in + 21) };
  shifts = simd_batch{ 0, 0, 0, 3 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 25-bit bundles 28 to 31
  words = simd_batch{ SafeLoad<uint32_t>(in + 21) >> 28 | SafeLoad<uint32_t>(in + 22) << 4, SafeLoad<uint32_t>(in + 22) >> 21 | SafeLoad<uint32_t>(in + 23) << 11, SafeLoad<uint32_t>(in + 23) >> 14 | SafeLoad<uint32_t>(in + 24) << 18, SafeLoad<uint32_t>(in + 24) };
  shifts = simd_batch{ 0, 0, 0, 7 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  in += 25;
  return in;
}

inline static const uint32_t* unpack26_32(const uint32_t* in, uint32_t* out) {
  uint32_t mask = 0x3ffffff;

  simd_batch masks(mask);
  simd_batch words, shifts;
  simd_batch results;

  // extract 26-bit bundles 0 to 3
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) >> 26 | SafeLoad<uint32_t>(in + 1) << 6, SafeLoad<uint32_t>(in + 1) >> 20 | SafeLoad<uint32_t>(in + 2) << 12, SafeLoad<uint32_t>(in + 2) >> 14 | SafeLoad<uint32_t>(in + 3) << 18 };
  shifts = simd_batch{ 0, 0, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 26-bit bundles 4 to 7
  words = simd_batch{ SafeLoad<uint32_t>(in + 3) >> 8 | SafeLoad<uint32_t>(in + 4) << 24, SafeLoad<uint32_t>(in + 4), SafeLoad<uint32_t>(in + 4) >> 28 | SafeLoad<uint32_t>(in + 5) << 4, SafeLoad<uint32_t>(in + 5) >> 22 | SafeLoad<uint32_t>(in + 6) << 10 };
  shifts = simd_batch{ 0, 2, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 26-bit bundles 8 to 11
  words = simd_batch{ SafeLoad<uint32_t>(in + 6) >> 16 | SafeLoad<uint32_t>(in + 7) << 16, SafeLoad<uint32_t>(in + 7) >> 10 | SafeLoad<uint32_t>(in + 8) << 22, SafeLoad<uint32_t>(in + 8), SafeLoad<uint32_t>(in + 8) >> 30 | SafeLoad<uint32_t>(in + 9) << 2 };
  shifts = simd_batch{ 0, 0, 4, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 26-bit bundles 12 to 15
  words = simd_batch{ SafeLoad<uint32_t>(in + 9) >> 24 | SafeLoad<uint32_t>(in + 10) << 8, SafeLoad<uint32_t>(in + 10) >> 18 | SafeLoad<uint32_t>(in + 11) << 14, SafeLoad<uint32_t>(in + 11) >> 12 | SafeLoad<uint32_t>(in + 12) << 20, SafeLoad<uint32_t>(in + 12) };
  shifts = simd_batch{ 0, 0, 0, 6 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 26-bit bundles 16 to 19
  words = simd_batch{ SafeLoad<uint32_t>(in + 13), SafeLoad<uint32_t>(in + 13) >> 26 | SafeLoad<uint32_t>(in + 14) << 6, SafeLoad<uint32_t>(in + 14) >> 20 | SafeLoad<uint32_t>(in + 15) << 12, SafeLoad<uint32_t>(in + 15) >> 14 | SafeLoad<uint32_t>(in + 16) << 18 };
  shifts = simd_batch{ 0, 0, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 26-bit bundles 20 to 23
  words = simd_batch{ SafeLoad<uint32_t>(in + 16) >> 8 | SafeLoad<uint32_t>(in + 17) << 24, SafeLoad<uint32_t>(in + 17), SafeLoad<uint32_t>(in + 17) >> 28 | SafeLoad<uint32_t>(in + 18) << 4, SafeLoad<uint32_t>(in + 18) >> 22 | SafeLoad<uint32_t>(in + 19) << 10 };
  shifts = simd_batch{ 0, 2, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 26-bit bundles 24 to 27
  words = simd_batch{ SafeLoad<uint32_t>(in + 19) >> 16 | SafeLoad<uint32_t>(in + 20) << 16, SafeLoad<uint32_t>(in + 20) >> 10 | SafeLoad<uint32_t>(in + 21) << 22, SafeLoad<uint32_t>(in + 21), SafeLoad<uint32_t>(in + 21) >> 30 | SafeLoad<uint32_t>(in + 22) << 2 };
  shifts = simd_batch{ 0, 0, 4, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 26-bit bundles 28 to 31
  words = simd_batch{ SafeLoad<uint32_t>(in + 22) >> 24 | SafeLoad<uint32_t>(in + 23) << 8, SafeLoad<uint32_t>(in + 23) >> 18 | SafeLoad<uint32_t>(in + 24) << 14, SafeLoad<uint32_t>(in + 24) >> 12 | SafeLoad<uint32_t>(in + 25) << 20, SafeLoad<uint32_t>(in + 25) };
  shifts = simd_batch{ 0, 0, 0, 6 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  in += 26;
  return in;
}

inline static const uint32_t* unpack27_32(const uint32_t* in, uint32_t* out) {
  uint32_t mask = 0x7ffffff;

  simd_batch masks(mask);
  simd_batch words, shifts;
  simd_batch results;

  // extract 27-bit bundles 0 to 3
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) >> 27 | SafeLoad<uint32_t>(in + 1) << 5, SafeLoad<uint32_t>(in + 1) >> 22 | SafeLoad<uint32_t>(in + 2) << 10, SafeLoad<uint32_t>(in + 2) >> 17 | SafeLoad<uint32_t>(in + 3) << 15 };
  shifts = simd_batch{ 0, 0, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 27-bit bundles 4 to 7
  words = simd_batch{ SafeLoad<uint32_t>(in + 3) >> 12 | SafeLoad<uint32_t>(in + 4) << 20, SafeLoad<uint32_t>(in + 4) >> 7 | SafeLoad<uint32_t>(in + 5) << 25, SafeLoad<uint32_t>(in + 5), SafeLoad<uint32_t>(in + 5) >> 29 | SafeLoad<uint32_t>(in + 6) << 3 };
  shifts = simd_batch{ 0, 0, 2, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 27-bit bundles 8 to 11
  words = simd_batch{ SafeLoad<uint32_t>(in + 6) >> 24 | SafeLoad<uint32_t>(in + 7) << 8, SafeLoad<uint32_t>(in + 7) >> 19 | SafeLoad<uint32_t>(in + 8) << 13, SafeLoad<uint32_t>(in + 8) >> 14 | SafeLoad<uint32_t>(in + 9) << 18, SafeLoad<uint32_t>(in + 9) >> 9 | SafeLoad<uint32_t>(in + 10) << 23 };
  shifts = simd_batch{ 0, 0, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 27-bit bundles 12 to 15
  words = simd_batch{ SafeLoad<uint32_t>(in + 10), SafeLoad<uint32_t>(in + 10) >> 31 | SafeLoad<uint32_t>(in + 11) << 1, SafeLoad<uint32_t>(in + 11) >> 26 | SafeLoad<uint32_t>(in + 12) << 6, SafeLoad<uint32_t>(in + 12) >> 21 | SafeLoad<uint32_t>(in + 13) << 11 };
  shifts = simd_batch{ 4, 0, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 27-bit bundles 16 to 19
  words = simd_batch{ SafeLoad<uint32_t>(in + 13) >> 16 | SafeLoad<uint32_t>(in + 14) << 16, SafeLoad<uint32_t>(in + 14) >> 11 | SafeLoad<uint32_t>(in + 15) << 21, SafeLoad<uint32_t>(in + 15) >> 6 | SafeLoad<uint32_t>(in + 16) << 26, SafeLoad<uint32_t>(in + 16) };
  shifts = simd_batch{ 0, 0, 0, 1 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 27-bit bundles 20 to 23
  words = simd_batch{ SafeLoad<uint32_t>(in + 16) >> 28 | SafeLoad<uint32_t>(in + 17) << 4, SafeLoad<uint32_t>(in + 17) >> 23 | SafeLoad<uint32_t>(in + 18) << 9, SafeLoad<uint32_t>(in + 18) >> 18 | SafeLoad<uint32_t>(in + 19) << 14, SafeLoad<uint32_t>(in + 19) >> 13 | SafeLoad<uint32_t>(in + 20) << 19 };
  shifts = simd_batch{ 0, 0, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 27-bit bundles 24 to 27
  words = simd_batch{ SafeLoad<uint32_t>(in + 20) >> 8 | SafeLoad<uint32_t>(in + 21) << 24, SafeLoad<uint32_t>(in + 21), SafeLoad<uint32_t>(in + 21) >> 30 | SafeLoad<uint32_t>(in + 22) << 2, SafeLoad<uint32_t>(in + 22) >> 25 | SafeLoad<uint32_t>(in + 23) << 7 };
  shifts = simd_batch{ 0, 3, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 27-bit bundles 28 to 31
  words = simd_batch{ SafeLoad<uint32_t>(in + 23) >> 20 | SafeLoad<uint32_t>(in + 24) << 12, SafeLoad<uint32_t>(in + 24) >> 15 | SafeLoad<uint32_t>(in + 25) << 17, SafeLoad<uint32_t>(in + 25) >> 10 | SafeLoad<uint32_t>(in + 26) << 22, SafeLoad<uint32_t>(in + 26) };
  shifts = simd_batch{ 0, 0, 0, 5 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  in += 27;
  return in;
}

inline static const uint32_t* unpack28_32(const uint32_t* in, uint32_t* out) {
  uint32_t mask = 0xfffffff;

  simd_batch masks(mask);
  simd_batch words, shifts;
  simd_batch results;

  // extract 28-bit bundles 0 to 3
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) >> 28 | SafeLoad<uint32_t>(in + 1) << 4, SafeLoad<uint32_t>(in + 1) >> 24 | SafeLoad<uint32_t>(in + 2) << 8, SafeLoad<uint32_t>(in + 2) >> 20 | SafeLoad<uint32_t>(in + 3) << 12 };
  shifts = simd_batch{ 0, 0, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 28-bit bundles 4 to 7
  words = simd_batch{ SafeLoad<uint32_t>(in + 3) >> 16 | SafeLoad<uint32_t>(in + 4) << 16, SafeLoad<uint32_t>(in + 4) >> 12 | SafeLoad<uint32_t>(in + 5) << 20, SafeLoad<uint32_t>(in + 5) >> 8 | SafeLoad<uint32_t>(in + 6) << 24, SafeLoad<uint32_t>(in + 6) };
  shifts = simd_batch{ 0, 0, 0, 4 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 28-bit bundles 8 to 11
  words = simd_batch{ SafeLoad<uint32_t>(in + 7), SafeLoad<uint32_t>(in + 7) >> 28 | SafeLoad<uint32_t>(in + 8) << 4, SafeLoad<uint32_t>(in + 8) >> 24 | SafeLoad<uint32_t>(in + 9) << 8, SafeLoad<uint32_t>(in + 9) >> 20 | SafeLoad<uint32_t>(in + 10) << 12 };
  shifts = simd_batch{ 0, 0, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 28-bit bundles 12 to 15
  words = simd_batch{ SafeLoad<uint32_t>(in + 10) >> 16 | SafeLoad<uint32_t>(in + 11) << 16, SafeLoad<uint32_t>(in + 11) >> 12 | SafeLoad<uint32_t>(in + 12) << 20, SafeLoad<uint32_t>(in + 12) >> 8 | SafeLoad<uint32_t>(in + 13) << 24, SafeLoad<uint32_t>(in + 13) };
  shifts = simd_batch{ 0, 0, 0, 4 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 28-bit bundles 16 to 19
  words = simd_batch{ SafeLoad<uint32_t>(in + 14), SafeLoad<uint32_t>(in + 14) >> 28 | SafeLoad<uint32_t>(in + 15) << 4, SafeLoad<uint32_t>(in + 15) >> 24 | SafeLoad<uint32_t>(in + 16) << 8, SafeLoad<uint32_t>(in + 16) >> 20 | SafeLoad<uint32_t>(in + 17) << 12 };
  shifts = simd_batch{ 0, 0, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 28-bit bundles 20 to 23
  words = simd_batch{ SafeLoad<uint32_t>(in + 17) >> 16 | SafeLoad<uint32_t>(in + 18) << 16, SafeLoad<uint32_t>(in + 18) >> 12 | SafeLoad<uint32_t>(in + 19) << 20, SafeLoad<uint32_t>(in + 19) >> 8 | SafeLoad<uint32_t>(in + 20) << 24, SafeLoad<uint32_t>(in + 20) };
  shifts = simd_batch{ 0, 0, 0, 4 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 28-bit bundles 24 to 27
  words = simd_batch{ SafeLoad<uint32_t>(in + 21), SafeLoad<uint32_t>(in + 21) >> 28 | SafeLoad<uint32_t>(in + 22) << 4, SafeLoad<uint32_t>(in + 22) >> 24 | SafeLoad<uint32_t>(in + 23) << 8, SafeLoad<uint32_t>(in + 23) >> 20 | SafeLoad<uint32_t>(in + 24) << 12 };
  shifts = simd_batch{ 0, 0, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 28-bit bundles 28 to 31
  words = simd_batch{ SafeLoad<uint32_t>(in + 24) >> 16 | SafeLoad<uint32_t>(in + 25) << 16, SafeLoad<uint32_t>(in + 25) >> 12 | SafeLoad<uint32_t>(in + 26) << 20, SafeLoad<uint32_t>(in + 26) >> 8 | SafeLoad<uint32_t>(in + 27) << 24, SafeLoad<uint32_t>(in + 27) };
  shifts = simd_batch{ 0, 0, 0, 4 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  in += 28;
  return in;
}

inline static const uint32_t* unpack29_32(const uint32_t* in, uint32_t* out) {
  uint32_t mask = 0x1fffffff;

  simd_batch masks(mask);
  simd_batch words, shifts;
  simd_batch results;

  // extract 29-bit bundles 0 to 3
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) >> 29 | SafeLoad<uint32_t>(in + 1) << 3, SafeLoad<uint32_t>(in + 1) >> 26 | SafeLoad<uint32_t>(in + 2) << 6, SafeLoad<uint32_t>(in + 2) >> 23 | SafeLoad<uint32_t>(in + 3) << 9 };
  shifts = simd_batch{ 0, 0, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 29-bit bundles 4 to 7
  words = simd_batch{ SafeLoad<uint32_t>(in + 3) >> 20 | SafeLoad<uint32_t>(in + 4) << 12, SafeLoad<uint32_t>(in + 4) >> 17 | SafeLoad<uint32_t>(in + 5) << 15, SafeLoad<uint32_t>(in + 5) >> 14 | SafeLoad<uint32_t>(in + 6) << 18, SafeLoad<uint32_t>(in + 6) >> 11 | SafeLoad<uint32_t>(in + 7) << 21 };
  shifts = simd_batch{ 0, 0, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 29-bit bundles 8 to 11
  words = simd_batch{ SafeLoad<uint32_t>(in + 7) >> 8 | SafeLoad<uint32_t>(in + 8) << 24, SafeLoad<uint32_t>(in + 8) >> 5 | SafeLoad<uint32_t>(in + 9) << 27, SafeLoad<uint32_t>(in + 9), SafeLoad<uint32_t>(in + 9) >> 31 | SafeLoad<uint32_t>(in + 10) << 1 };
  shifts = simd_batch{ 0, 0, 2, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 29-bit bundles 12 to 15
  words = simd_batch{ SafeLoad<uint32_t>(in + 10) >> 28 | SafeLoad<uint32_t>(in + 11) << 4, SafeLoad<uint32_t>(in + 11) >> 25 | SafeLoad<uint32_t>(in + 12) << 7, SafeLoad<uint32_t>(in + 12) >> 22 | SafeLoad<uint32_t>(in + 13) << 10, SafeLoad<uint32_t>(in + 13) >> 19 | SafeLoad<uint32_t>(in + 14) << 13 };
  shifts = simd_batch{ 0, 0, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 29-bit bundles 16 to 19
  words = simd_batch{ SafeLoad<uint32_t>(in + 14) >> 16 | SafeLoad<uint32_t>(in + 15) << 16, SafeLoad<uint32_t>(in + 15) >> 13 | SafeLoad<uint32_t>(in + 16) << 19, SafeLoad<uint32_t>(in + 16) >> 10 | SafeLoad<uint32_t>(in + 17) << 22, SafeLoad<uint32_t>(in + 17) >> 7 | SafeLoad<uint32_t>(in + 18) << 25 };
  shifts = simd_batch{ 0, 0, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 29-bit bundles 20 to 23
  words = simd_batch{ SafeLoad<uint32_t>(in + 18) >> 4 | SafeLoad<uint32_t>(in + 19) << 28, SafeLoad<uint32_t>(in + 19), SafeLoad<uint32_t>(in + 19) >> 30 | SafeLoad<uint32_t>(in + 20) << 2, SafeLoad<uint32_t>(in + 20) >> 27 | SafeLoad<uint32_t>(in + 21) << 5 };
  shifts = simd_batch{ 0, 1, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 29-bit bundles 24 to 27
  words = simd_batch{ SafeLoad<uint32_t>(in + 21) >> 24 | SafeLoad<uint32_t>(in + 22) << 8, SafeLoad<uint32_t>(in + 22) >> 21 | SafeLoad<uint32_t>(in + 23) << 11, SafeLoad<uint32_t>(in + 23) >> 18 | SafeLoad<uint32_t>(in + 24) << 14, SafeLoad<uint32_t>(in + 24) >> 15 | SafeLoad<uint32_t>(in + 25) << 17 };
  shifts = simd_batch{ 0, 0, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 29-bit bundles 28 to 31
  words = simd_batch{ SafeLoad<uint32_t>(in + 25) >> 12 | SafeLoad<uint32_t>(in + 26) << 20, SafeLoad<uint32_t>(in + 26) >> 9 | SafeLoad<uint32_t>(in + 27) << 23, SafeLoad<uint32_t>(in + 27) >> 6 | SafeLoad<uint32_t>(in + 28) << 26, SafeLoad<uint32_t>(in + 28) };
  shifts = simd_batch{ 0, 0, 0, 3 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  in += 29;
  return in;
}

inline static const uint32_t* unpack30_32(const uint32_t* in, uint32_t* out) {
  uint32_t mask = 0x3fffffff;

  simd_batch masks(mask);
  simd_batch words, shifts;
  simd_batch results;

  // extract 30-bit bundles 0 to 3
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) >> 30 | SafeLoad<uint32_t>(in + 1) << 2, SafeLoad<uint32_t>(in + 1) >> 28 | SafeLoad<uint32_t>(in + 2) << 4, SafeLoad<uint32_t>(in + 2) >> 26 | SafeLoad<uint32_t>(in + 3) << 6 };
  shifts = simd_batch{ 0, 0, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 30-bit bundles 4 to 7
  words = simd_batch{ SafeLoad<uint32_t>(in + 3) >> 24 | SafeLoad<uint32_t>(in + 4) << 8, SafeLoad<uint32_t>(in + 4) >> 22 | SafeLoad<uint32_t>(in + 5) << 10, SafeLoad<uint32_t>(in + 5) >> 20 | SafeLoad<uint32_t>(in + 6) << 12, SafeLoad<uint32_t>(in + 6) >> 18 | SafeLoad<uint32_t>(in + 7) << 14 };
  shifts = simd_batch{ 0, 0, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 30-bit bundles 8 to 11
  words = simd_batch{ SafeLoad<uint32_t>(in + 7) >> 16 | SafeLoad<uint32_t>(in + 8) << 16, SafeLoad<uint32_t>(in + 8) >> 14 | SafeLoad<uint32_t>(in + 9) << 18, SafeLoad<uint32_t>(in + 9) >> 12 | SafeLoad<uint32_t>(in + 10) << 20, SafeLoad<uint32_t>(in + 10) >> 10 | SafeLoad<uint32_t>(in + 11) << 22 };
  shifts = simd_batch{ 0, 0, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 30-bit bundles 12 to 15
  words = simd_batch{ SafeLoad<uint32_t>(in + 11) >> 8 | SafeLoad<uint32_t>(in + 12) << 24, SafeLoad<uint32_t>(in + 12) >> 6 | SafeLoad<uint32_t>(in + 13) << 26, SafeLoad<uint32_t>(in + 13) >> 4 | SafeLoad<uint32_t>(in + 14) << 28, SafeLoad<uint32_t>(in + 14) };
  shifts = simd_batch{ 0, 0, 0, 2 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 30-bit bundles 16 to 19
  words = simd_batch{ SafeLoad<uint32_t>(in + 15), SafeLoad<uint32_t>(in + 15) >> 30 | SafeLoad<uint32_t>(in + 16) << 2, SafeLoad<uint32_t>(in + 16) >> 28 | SafeLoad<uint32_t>(in + 17) << 4, SafeLoad<uint32_t>(in + 17) >> 26 | SafeLoad<uint32_t>(in + 18) << 6 };
  shifts = simd_batch{ 0, 0, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 30-bit bundles 20 to 23
  words = simd_batch{ SafeLoad<uint32_t>(in + 18) >> 24 | SafeLoad<uint32_t>(in + 19) << 8, SafeLoad<uint32_t>(in + 19) >> 22 | SafeLoad<uint32_t>(in + 20) << 10, SafeLoad<uint32_t>(in + 20) >> 20 | SafeLoad<uint32_t>(in + 21) << 12, SafeLoad<uint32_t>(in + 21) >> 18 | SafeLoad<uint32_t>(in + 22) << 14 };
  shifts = simd_batch{ 0, 0, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 30-bit bundles 24 to 27
  words = simd_batch{ SafeLoad<uint32_t>(in + 22) >> 16 | SafeLoad<uint32_t>(in + 23) << 16, SafeLoad<uint32_t>(in + 23) >> 14 | SafeLoad<uint32_t>(in + 24) << 18, SafeLoad<uint32_t>(in + 24) >> 12 | SafeLoad<uint32_t>(in + 25) << 20, SafeLoad<uint32_t>(in + 25) >> 10 | SafeLoad<uint32_t>(in + 26) << 22 };
  shifts = simd_batch{ 0, 0, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 30-bit bundles 28 to 31
  words = simd_batch{ SafeLoad<uint32_t>(in + 26) >> 8 | SafeLoad<uint32_t>(in + 27) << 24, SafeLoad<uint32_t>(in + 27) >> 6 | SafeLoad<uint32_t>(in + 28) << 26, SafeLoad<uint32_t>(in + 28) >> 4 | SafeLoad<uint32_t>(in + 29) << 28, SafeLoad<uint32_t>(in + 29) };
  shifts = simd_batch{ 0, 0, 0, 2 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  in += 30;
  return in;
}

inline static const uint32_t* unpack31_32(const uint32_t* in, uint32_t* out) {
  uint32_t mask = 0x7fffffff;

  simd_batch masks(mask);
  simd_batch words, shifts;
  simd_batch results;

  // extract 31-bit bundles 0 to 3
  words = simd_batch{ SafeLoad<uint32_t>(in + 0), SafeLoad<uint32_t>(in + 0) >> 31 | SafeLoad<uint32_t>(in + 1) << 1, SafeLoad<uint32_t>(in + 1) >> 30 | SafeLoad<uint32_t>(in + 2) << 2, SafeLoad<uint32_t>(in + 2) >> 29 | SafeLoad<uint32_t>(in + 3) << 3 };
  shifts = simd_batch{ 0, 0, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 31-bit bundles 4 to 7
  words = simd_batch{ SafeLoad<uint32_t>(in + 3) >> 28 | SafeLoad<uint32_t>(in + 4) << 4, SafeLoad<uint32_t>(in + 4) >> 27 | SafeLoad<uint32_t>(in + 5) << 5, SafeLoad<uint32_t>(in + 5) >> 26 | SafeLoad<uint32_t>(in + 6) << 6, SafeLoad<uint32_t>(in + 6) >> 25 | SafeLoad<uint32_t>(in + 7) << 7 };
  shifts = simd_batch{ 0, 0, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 31-bit bundles 8 to 11
  words = simd_batch{ SafeLoad<uint32_t>(in + 7) >> 24 | SafeLoad<uint32_t>(in + 8) << 8, SafeLoad<uint32_t>(in + 8) >> 23 | SafeLoad<uint32_t>(in + 9) << 9, SafeLoad<uint32_t>(in + 9) >> 22 | SafeLoad<uint32_t>(in + 10) << 10, SafeLoad<uint32_t>(in + 10) >> 21 | SafeLoad<uint32_t>(in + 11) << 11 };
  shifts = simd_batch{ 0, 0, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 31-bit bundles 12 to 15
  words = simd_batch{ SafeLoad<uint32_t>(in + 11) >> 20 | SafeLoad<uint32_t>(in + 12) << 12, SafeLoad<uint32_t>(in + 12) >> 19 | SafeLoad<uint32_t>(in + 13) << 13, SafeLoad<uint32_t>(in + 13) >> 18 | SafeLoad<uint32_t>(in + 14) << 14, SafeLoad<uint32_t>(in + 14) >> 17 | SafeLoad<uint32_t>(in + 15) << 15 };
  shifts = simd_batch{ 0, 0, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 31-bit bundles 16 to 19
  words = simd_batch{ SafeLoad<uint32_t>(in + 15) >> 16 | SafeLoad<uint32_t>(in + 16) << 16, SafeLoad<uint32_t>(in + 16) >> 15 | SafeLoad<uint32_t>(in + 17) << 17, SafeLoad<uint32_t>(in + 17) >> 14 | SafeLoad<uint32_t>(in + 18) << 18, SafeLoad<uint32_t>(in + 18) >> 13 | SafeLoad<uint32_t>(in + 19) << 19 };
  shifts = simd_batch{ 0, 0, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 31-bit bundles 20 to 23
  words = simd_batch{ SafeLoad<uint32_t>(in + 19) >> 12 | SafeLoad<uint32_t>(in + 20) << 20, SafeLoad<uint32_t>(in + 20) >> 11 | SafeLoad<uint32_t>(in + 21) << 21, SafeLoad<uint32_t>(in + 21) >> 10 | SafeLoad<uint32_t>(in + 22) << 22, SafeLoad<uint32_t>(in + 22) >> 9 | SafeLoad<uint32_t>(in + 23) << 23 };
  shifts = simd_batch{ 0, 0, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 31-bit bundles 24 to 27
  words = simd_batch{ SafeLoad<uint32_t>(in + 23) >> 8 | SafeLoad<uint32_t>(in + 24) << 24, SafeLoad<uint32_t>(in + 24) >> 7 | SafeLoad<uint32_t>(in + 25) << 25, SafeLoad<uint32_t>(in + 25) >> 6 | SafeLoad<uint32_t>(in + 26) << 26, SafeLoad<uint32_t>(in + 26) >> 5 | SafeLoad<uint32_t>(in + 27) << 27 };
  shifts = simd_batch{ 0, 0, 0, 0 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  // extract 31-bit bundles 28 to 31
  words = simd_batch{ SafeLoad<uint32_t>(in + 27) >> 4 | SafeLoad<uint32_t>(in + 28) << 28, SafeLoad<uint32_t>(in + 28) >> 3 | SafeLoad<uint32_t>(in + 29) << 29, SafeLoad<uint32_t>(in + 29) >> 2 | SafeLoad<uint32_t>(in + 30) << 30, SafeLoad<uint32_t>(in + 30) };
  shifts = simd_batch{ 0, 0, 0, 1 };
  results = (words >> shifts) & masks;
  results.store_unaligned(out);
  out += 4;

  in += 31;
  return in;
}

inline static const uint32_t* unpack32_32(const uint32_t* in, uint32_t* out) {
  memcpy(out, in, 32 * sizeof(*out));
  in += 32;
  out += 32;

  return in;
}

};  // struct UnpackBits128

}  // namespace
}  // namespace internal
}  // namespace arrow


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

#ifdef _MSC_VER
// MSVC x86_64/arm64

#if defined(_M_AMD64) || defined(_M_X64)
#include <intrin.h>
#endif

#else
// gcc/clang (possibly others)

#if defined(ARROW_HAVE_BMI2)
#include <x86intrin.h>
#endif

#if defined(ARROW_HAVE_AVX2) || defined(ARROW_HAVE_AVX512)
#include <immintrin.h>
#elif defined(ARROW_HAVE_SSE4_2)
#include <nmmintrin.h>
#endif

#ifdef ARROW_HAVE_NEON
#include <arm_neon.h>
#endif

#endif

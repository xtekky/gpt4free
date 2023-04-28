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

#include "arrow/python/platform.h"  // IWYU pragma: export

#include <numpy/numpyconfig.h>  // IWYU pragma: export

// Don't use the deprecated Numpy functions
#ifdef NPY_1_7_API_VERSION
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#else
#define NPY_ARRAY_NOTSWAPPED NPY_NOTSWAPPED
#define NPY_ARRAY_ALIGNED NPY_ALIGNED
#define NPY_ARRAY_WRITEABLE NPY_WRITEABLE
#define NPY_ARRAY_UPDATEIFCOPY NPY_UPDATEIFCOPY
#endif

// This is required to be able to access the NumPy C API properly in C++ files
// other than init.cc.
#define PY_ARRAY_UNIQUE_SYMBOL arrow_ARRAY_API
#ifndef NUMPY_IMPORT_ARRAY
#define NO_IMPORT_ARRAY
#endif

#include <numpy/arrayobject.h>   // IWYU pragma: export
#include <numpy/arrayscalars.h>  // IWYU pragma: export
#include <numpy/ufuncobject.h>   // IWYU pragma: export

// A bit subtle. Numpy has 5 canonical integer types:
// (or, rather, type pairs: signed and unsigned)
//   NPY_BYTE, NPY_SHORT, NPY_INT, NPY_LONG, NPY_LONGLONG
// It also has 4 fixed-width integer aliases.
// When mapping Arrow integer types to these 4 fixed-width aliases,
// we always miss one of the canonical types (even though it may
// have the same width as one of the aliases).
// Which one depends on the platform...
// On a LP64 system, NPY_INT64 maps to NPY_LONG and
// NPY_LONGLONG needs to be handled separately.
// On a LLP64 system, NPY_INT32 maps to NPY_LONG and
// NPY_INT needs to be handled separately.

#if NPY_BITSOF_LONG == 32 && NPY_BITSOF_LONGLONG == 64
#define NPY_INT64_IS_LONG_LONG 1
#else
#define NPY_INT64_IS_LONG_LONG 0
#endif

#if NPY_BITSOF_INT == 32 && NPY_BITSOF_LONG == 64
#define NPY_INT32_IS_INT 1
#else
#define NPY_INT32_IS_INT 0
#endif

namespace arrow {
namespace py {

inline int import_numpy() {
#ifdef NUMPY_IMPORT_ARRAY
  import_array1(-1);
  import_umath1(-1);
#endif

  return 0;
}

// See above about the missing Numpy integer type numbers
inline int fix_numpy_type_num(int type_num) {
#if !NPY_INT32_IS_INT && NPY_BITSOF_INT == 32
  if (type_num == NPY_INT) return NPY_INT32;
  if (type_num == NPY_UINT) return NPY_UINT32;
#endif
#if !NPY_INT64_IS_LONG_LONG && NPY_BITSOF_LONGLONG == 64
  if (type_num == NPY_LONGLONG) return NPY_INT64;
  if (type_num == NPY_ULONGLONG) return NPY_UINT64;
#endif
  return type_num;
}

}  // namespace py
}  // namespace arrow

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

// Functions for converting between pandas's NumPy-based data representation
// and Arrow data structures

#pragma once

// If PY_SSIZE_T_CLEAN is defined, argument parsing functions treat #-specifier
// to mean Py_ssize_t (defining this to suppress deprecation warning)
#define PY_SSIZE_T_CLEAN

#include <Python.h> // IWYU pragma: export
#include <datetime.h>

// Work around C2528 error
#ifdef _MSC_VER
#if _MSC_VER >= 1900
#undef timezone
#endif
#endif


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

#if defined(_WIN32) || defined(__CYGWIN__)
// Windows

#if defined(_MSC_VER)
#pragma warning(disable : 4251)
#else
#pragma GCC diagnostic ignored "-Wattributes"
#endif

#if defined(__cplusplus) && defined(__GNUC__) && !defined(__clang__)
// Use C++ attribute syntax where possible to avoid GCC parser bug
// (https://stackoverflow.com/questions/57993818/gcc-how-to-combine-attribute-dllexport-and-nodiscard-in-a-struct-de)
#define ARROW_DLLEXPORT [[gnu::dllexport]]
#define ARROW_DLLIMPORT [[gnu::dllimport]]
#else
#define ARROW_DLLEXPORT __declspec(dllexport)
#define ARROW_DLLIMPORT __declspec(dllimport)
#endif

#ifdef ARROW_STATIC
#define ARROW_EXPORT
#define ARROW_FRIEND_EXPORT
#define ARROW_TEMPLATE_EXPORT
#elif defined(ARROW_EXPORTING)
#define ARROW_EXPORT ARROW_DLLEXPORT
// For some reason [[gnu::dllexport]] doesn't work well with friend declarations
#define ARROW_FRIEND_EXPORT __declspec(dllexport)
#define ARROW_TEMPLATE_EXPORT ARROW_DLLEXPORT
#else
#define ARROW_EXPORT ARROW_DLLIMPORT
#define ARROW_FRIEND_EXPORT __declspec(dllimport)
#define ARROW_TEMPLATE_EXPORT ARROW_DLLIMPORT
#endif

#define ARROW_NO_EXPORT
#define ARROW_FORCE_INLINE __forceinline

#else

// Non-Windows

#define ARROW_FORCE_INLINE

#if defined(__cplusplus) && (defined(__GNUC__) || defined(__clang__))
#ifndef ARROW_EXPORT
#define ARROW_EXPORT [[gnu::visibility("default")]]
#endif
#ifndef ARROW_NO_EXPORT
#define ARROW_NO_EXPORT [[gnu::visibility("hidden")]]
#endif
#else
// Not C++, or not gcc/clang
#ifndef ARROW_EXPORT
#define ARROW_EXPORT
#endif
#ifndef ARROW_NO_EXPORT
#define ARROW_NO_EXPORT
#endif
#endif

#define ARROW_FRIEND_EXPORT
#define ARROW_TEMPLATE_EXPORT

#endif  // Non-Windows

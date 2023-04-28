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
#include <memory>

#include "arrow/buffer.h"         // IWYU pragma: export
#include "arrow/io/interfaces.h"  // IWYU pragma: export
#include "arrow/status.h"         // IWYU pragma: export
#include "arrow/type_fwd.h"       // IWYU pragma: export
#include "arrow/util/macros.h"    // IWYU pragma: export

#if defined(_WIN32) || defined(__CYGWIN__)

#if defined(_MSC_VER)
#pragma warning(push)
// Disable warning for STL types usage in DLL interface
// https://web.archive.org/web/20130317015847/http://connect.microsoft.com/VisualStudio/feedback/details/696593/vc-10-vs-2010-basic-string-exports
#pragma warning(disable : 4275 4251)
// Disable diamond inheritance warnings
#pragma warning(disable : 4250)
// Disable macro redefinition warnings
#pragma warning(disable : 4005)
// Disable extern before exported template warnings
#pragma warning(disable : 4910)
#else
#pragma GCC diagnostic ignored "-Wattributes"
#endif

#ifdef PARQUET_STATIC
#define PARQUET_EXPORT
#elif defined(PARQUET_EXPORTING)
#define PARQUET_EXPORT __declspec(dllexport)
#else
#define PARQUET_EXPORT __declspec(dllimport)
#endif

#define PARQUET_NO_EXPORT

#else  // Not Windows
#ifndef PARQUET_EXPORT
#define PARQUET_EXPORT __attribute__((visibility("default")))
#endif
#ifndef PARQUET_NO_EXPORT
#define PARQUET_NO_EXPORT __attribute__((visibility("hidden")))
#endif
#endif  // Non-Windows

// This is a complicated topic, some reading on it:
// http://www.codesynthesis.com/~boris/blog/2010/01/18/dll-export-cxx-templates/
#if defined(_MSC_VER) || defined(__clang__)
#define PARQUET_TEMPLATE_CLASS_EXPORT
#define PARQUET_TEMPLATE_EXPORT PARQUET_EXPORT
#else
#define PARQUET_TEMPLATE_CLASS_EXPORT PARQUET_EXPORT
#define PARQUET_TEMPLATE_EXPORT
#endif

#define PARQUET_DISALLOW_COPY_AND_ASSIGN ARROW_DISALLOW_COPY_AND_ASSIGN

#define PARQUET_NORETURN ARROW_NORETURN
#define PARQUET_DEPRECATED ARROW_DEPRECATED

// If ARROW_VALGRIND set when compiling unit tests, also define
// PARQUET_VALGRIND
#ifdef ARROW_VALGRIND
#define PARQUET_VALGRIND
#endif

namespace parquet {

using Buffer = ::arrow::Buffer;
using Codec = ::arrow::util::Codec;
using Compression = ::arrow::Compression;
using MemoryPool = ::arrow::MemoryPool;
using MutableBuffer = ::arrow::MutableBuffer;
using ResizableBuffer = ::arrow::ResizableBuffer;
using ResizableBuffer = ::arrow::ResizableBuffer;
using ArrowInputFile = ::arrow::io::RandomAccessFile;
using ArrowInputStream = ::arrow::io::InputStream;
using ArrowOutputStream = ::arrow::io::OutputStream;

constexpr int64_t kDefaultOutputStreamSize = 1024;

constexpr int16_t kNonPageOrdinal = static_cast<int16_t>(-1);

PARQUET_EXPORT
std::shared_ptr<::arrow::io::BufferOutputStream> CreateOutputStream(
    ::arrow::MemoryPool* pool = ::arrow::default_memory_pool());

PARQUET_EXPORT
std::shared_ptr<ResizableBuffer> AllocateBuffer(
    ::arrow::MemoryPool* pool = ::arrow::default_memory_pool(), int64_t size = 0);

}  // namespace parquet

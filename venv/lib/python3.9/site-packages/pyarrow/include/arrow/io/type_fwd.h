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

#include "arrow/type_fwd.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace io {

struct FileMode {
  enum type { READ, WRITE, READWRITE };
};

struct IOContext;
struct CacheOptions;

/// EXPERIMENTAL: convenience global singleton for default IOContext settings
ARROW_EXPORT
const IOContext& default_io_context();

/// \brief Get the capacity of the global I/O thread pool
///
/// Return the number of worker threads in the thread pool to which
/// Arrow dispatches various I/O-bound tasks.  This is an ideal number,
/// not necessarily the exact number of threads at a given point in time.
///
/// You can change this number using SetIOThreadPoolCapacity().
ARROW_EXPORT int GetIOThreadPoolCapacity();

/// \brief Set the capacity of the global I/O thread pool
///
/// Set the number of worker threads in the thread pool to which
/// Arrow dispatches various I/O-bound tasks.
///
/// The current number is returned by GetIOThreadPoolCapacity().
ARROW_EXPORT Status SetIOThreadPoolCapacity(int threads);

class FileInterface;
class Seekable;
class Writable;
class Readable;
class OutputStream;
class FileOutputStream;
class InputStream;
class ReadableFile;
class RandomAccessFile;
class MemoryMappedFile;
class WritableFile;
class ReadWriteFileInterface;

class LatencyGenerator;

class BufferOutputStream;
class BufferReader;
class CompressedInputStream;
class CompressedOutputStream;
class BufferedInputStream;
class BufferedOutputStream;

}  // namespace io
}  // namespace arrow

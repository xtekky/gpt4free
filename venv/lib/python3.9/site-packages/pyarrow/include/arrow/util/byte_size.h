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

#include "arrow/type_fwd.h"

namespace arrow {

namespace util {

/// \brief The sum of bytes in each buffer referenced by the array
///
/// Note: An array may only reference a portion of a buffer.
///       This method will overestimate in this case and return the
///       byte size of the entire buffer.
/// Note: If a buffer is referenced multiple times then it will
///       only be counted once.
ARROW_EXPORT int64_t TotalBufferSize(const ArrayData& array_data);
/// \brief The sum of bytes in each buffer referenced by the array
/// \see TotalBufferSize(const ArrayData& array_data) for details
ARROW_EXPORT int64_t TotalBufferSize(const Array& array);
/// \brief The sum of bytes in each buffer referenced by the array
/// \see TotalBufferSize(const ArrayData& array_data) for details
ARROW_EXPORT int64_t TotalBufferSize(const ChunkedArray& chunked_array);
/// \brief The sum of bytes in each buffer referenced by the batch
/// \see TotalBufferSize(const ArrayData& array_data) for details
ARROW_EXPORT int64_t TotalBufferSize(const RecordBatch& record_batch);
/// \brief The sum of bytes in each buffer referenced by the table
/// \see TotalBufferSize(const ArrayData& array_data) for details
ARROW_EXPORT int64_t TotalBufferSize(const Table& table);

/// \brief Calculate the buffer ranges referenced by the array
///
/// These ranges will take into account array offsets
///
/// The ranges may contain duplicates
///
/// Dictionary arrays will ignore the offset of their containing array
///
/// The return value will be a struct array corresponding to the schema:
/// schema({field("start", uint64()), field("offset", uint64()), field("length",
/// uint64()))
ARROW_EXPORT Result<std::shared_ptr<Array>> ReferencedRanges(const ArrayData& array_data);

/// \brief Returns the sum of bytes from all buffer ranges referenced
///
/// Unlike TotalBufferSize this method will account for array
/// offsets.
///
/// If buffers are shared between arrays then the shared
/// portion will be counted multiple times.
///
/// Dictionary arrays will always be counted in their entirety
/// even if the array only references a portion of the dictionary.
ARROW_EXPORT Result<int64_t> ReferencedBufferSize(const ArrayData& array_data);
/// \brief Returns the sum of bytes from all buffer ranges referenced
/// \see ReferencedBufferSize(const ArrayData& array_data) for details
ARROW_EXPORT Result<int64_t> ReferencedBufferSize(const Array& array_data);
/// \brief Returns the sum of bytes from all buffer ranges referenced
/// \see ReferencedBufferSize(const ArrayData& array_data) for details
ARROW_EXPORT Result<int64_t> ReferencedBufferSize(const ChunkedArray& array_data);
/// \brief Returns the sum of bytes from all buffer ranges referenced
/// \see ReferencedBufferSize(const ArrayData& array_data) for details
ARROW_EXPORT Result<int64_t> ReferencedBufferSize(const RecordBatch& array_data);
/// \brief Returns the sum of bytes from all buffer ranges referenced
/// \see ReferencedBufferSize(const ArrayData& array_data) for details
ARROW_EXPORT Result<int64_t> ReferencedBufferSize(const Table& array_data);

}  // namespace util

}  // namespace arrow

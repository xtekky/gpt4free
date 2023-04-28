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
#include <vector>

#include "arrow/array.h"
#include "arrow/record_batch.h"
#include "arrow/status.h"
#include "arrow/testing/visibility.h"
#include "arrow/type.h"

namespace arrow {
namespace ipc {
namespace test {

// A typedef used for test parameterization
typedef Status MakeRecordBatch(std::shared_ptr<RecordBatch>* out);

ARROW_TESTING_EXPORT
void CompareArraysDetailed(int index, const Array& result, const Array& expected);

ARROW_TESTING_EXPORT
void CompareBatchColumnsDetailed(const RecordBatch& result, const RecordBatch& expected);

ARROW_TESTING_EXPORT
Status MakeRandomInt32Array(int64_t length, bool include_nulls, MemoryPool* pool,
                            std::shared_ptr<Array>* out, uint32_t seed = 0,
                            int32_t min = 0, int32_t max = 1000);

ARROW_TESTING_EXPORT
Status MakeRandomInt64Array(int64_t length, bool include_nulls, MemoryPool* pool,
                            std::shared_ptr<Array>* out, uint32_t seed = 0);

ARROW_TESTING_EXPORT
Status MakeRandomListArray(const std::shared_ptr<Array>& child_array, int num_lists,
                           bool include_nulls, MemoryPool* pool,
                           std::shared_ptr<Array>* out);

ARROW_TESTING_EXPORT
Status MakeRandomLargeListArray(const std::shared_ptr<Array>& child_array, int num_lists,
                                bool include_nulls, MemoryPool* pool,
                                std::shared_ptr<Array>* out);

ARROW_TESTING_EXPORT
Status MakeRandomBooleanArray(const int length, bool include_nulls,
                              std::shared_ptr<Array>* out);

ARROW_TESTING_EXPORT
Status MakeBooleanBatchSized(const int length, std::shared_ptr<RecordBatch>* out);

ARROW_TESTING_EXPORT
Status MakeBooleanBatch(std::shared_ptr<RecordBatch>* out);

ARROW_TESTING_EXPORT
Status MakeIntBatchSized(int length, std::shared_ptr<RecordBatch>* out,
                         uint32_t seed = 0);

ARROW_TESTING_EXPORT
Status MakeIntRecordBatch(std::shared_ptr<RecordBatch>* out);

ARROW_TESTING_EXPORT
Status MakeFloat3264BatchSized(int length, std::shared_ptr<RecordBatch>* out,
                               uint32_t seed = 0);

ARROW_TESTING_EXPORT
Status MakeFloat3264Batch(std::shared_ptr<RecordBatch>* out);

ARROW_TESTING_EXPORT
Status MakeFloatBatchSized(int length, std::shared_ptr<RecordBatch>* out,
                           uint32_t seed = 0);

ARROW_TESTING_EXPORT
Status MakeFloatBatch(std::shared_ptr<RecordBatch>* out);

ARROW_TESTING_EXPORT
Status MakeRandomStringArray(int64_t length, bool include_nulls, MemoryPool* pool,
                             std::shared_ptr<Array>* out);

ARROW_TESTING_EXPORT
Status MakeStringTypesRecordBatch(std::shared_ptr<RecordBatch>* out,
                                  bool with_nulls = true);

ARROW_TESTING_EXPORT
Status MakeStringTypesRecordBatchWithNulls(std::shared_ptr<RecordBatch>* out);

ARROW_TESTING_EXPORT
Status MakeNullRecordBatch(std::shared_ptr<RecordBatch>* out);

ARROW_TESTING_EXPORT
Status MakeListRecordBatch(std::shared_ptr<RecordBatch>* out);

ARROW_TESTING_EXPORT
Status MakeFixedSizeListRecordBatch(std::shared_ptr<RecordBatch>* out);

ARROW_TESTING_EXPORT
Status MakeZeroLengthRecordBatch(std::shared_ptr<RecordBatch>* out);

ARROW_TESTING_EXPORT
Status MakeNonNullRecordBatch(std::shared_ptr<RecordBatch>* out);

ARROW_TESTING_EXPORT
Status MakeDeeplyNestedList(std::shared_ptr<RecordBatch>* out);

ARROW_TESTING_EXPORT
Status MakeStruct(std::shared_ptr<RecordBatch>* out);

ARROW_TESTING_EXPORT
Status MakeUnion(std::shared_ptr<RecordBatch>* out);

ARROW_TESTING_EXPORT
Status MakeDictionary(std::shared_ptr<RecordBatch>* out);

ARROW_TESTING_EXPORT
Status MakeDictionaryFlat(std::shared_ptr<RecordBatch>* out);

ARROW_TESTING_EXPORT
Status MakeNestedDictionary(std::shared_ptr<RecordBatch>* out);

ARROW_TESTING_EXPORT
Status MakeMap(std::shared_ptr<RecordBatch>* out);

ARROW_TESTING_EXPORT
Status MakeMapOfDictionary(std::shared_ptr<RecordBatch>* out);

ARROW_TESTING_EXPORT
Status MakeDates(std::shared_ptr<RecordBatch>* out);

ARROW_TESTING_EXPORT
Status MakeTimestamps(std::shared_ptr<RecordBatch>* out);

ARROW_TESTING_EXPORT
Status MakeIntervals(std::shared_ptr<RecordBatch>* out);

ARROW_TESTING_EXPORT
Status MakeTimes(std::shared_ptr<RecordBatch>* out);

ARROW_TESTING_EXPORT
Status MakeFWBinary(std::shared_ptr<RecordBatch>* out);

ARROW_TESTING_EXPORT
Status MakeDecimal(std::shared_ptr<RecordBatch>* out);

ARROW_TESTING_EXPORT
Status MakeNull(std::shared_ptr<RecordBatch>* out);

ARROW_TESTING_EXPORT
Status MakeUuid(std::shared_ptr<RecordBatch>* out);

ARROW_TESTING_EXPORT
Status MakeComplex128(std::shared_ptr<RecordBatch>* out);

ARROW_TESTING_EXPORT
Status MakeDictExtension(std::shared_ptr<RecordBatch>* out);

ARROW_TESTING_EXPORT
Status MakeRandomTensor(const std::shared_ptr<DataType>& type,
                        const std::vector<int64_t>& shape, bool row_major_p,
                        std::shared_ptr<Tensor>* out, uint32_t seed = 0);

}  // namespace test
}  // namespace ipc
}  // namespace arrow

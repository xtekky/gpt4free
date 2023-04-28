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

#include <memory>
#include <string>

#include "arrow/c/abi.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type_fwd.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {

/// \defgroup c-data-interface Functions for working with the C data interface.
///
/// @{

/// \brief Export C++ DataType using the C data interface format.
///
/// The root type is considered to have empty name and metadata.
/// If you want the root type to have a name and/or metadata, pass
/// a Field instead.
///
/// \param[in] type DataType object to export
/// \param[out] out C struct where to export the datatype
ARROW_EXPORT
Status ExportType(const DataType& type, struct ArrowSchema* out);

/// \brief Export C++ Field using the C data interface format.
///
/// \param[in] field Field object to export
/// \param[out] out C struct where to export the field
ARROW_EXPORT
Status ExportField(const Field& field, struct ArrowSchema* out);

/// \brief Export C++ Schema using the C data interface format.
///
/// \param[in] schema Schema object to export
/// \param[out] out C struct where to export the field
ARROW_EXPORT
Status ExportSchema(const Schema& schema, struct ArrowSchema* out);

/// \brief Export C++ Array using the C data interface format.
///
/// The resulting ArrowArray struct keeps the array data and buffers alive
/// until its release callback is called by the consumer.
///
/// \param[in] array Array object to export
/// \param[out] out C struct where to export the array
/// \param[out] out_schema optional C struct where to export the array type
ARROW_EXPORT
Status ExportArray(const Array& array, struct ArrowArray* out,
                   struct ArrowSchema* out_schema = NULLPTR);

/// \brief Export C++ RecordBatch using the C data interface format.
///
/// The record batch is exported as if it were a struct array.
/// The resulting ArrowArray struct keeps the record batch data and buffers alive
/// until its release callback is called by the consumer.
///
/// \param[in] batch Record batch to export
/// \param[out] out C struct where to export the record batch
/// \param[out] out_schema optional C struct where to export the record batch schema
ARROW_EXPORT
Status ExportRecordBatch(const RecordBatch& batch, struct ArrowArray* out,
                         struct ArrowSchema* out_schema = NULLPTR);

/// \brief Import C++ DataType from the C data interface.
///
/// The given ArrowSchema struct is released (as per the C data interface
/// specification), even if this function fails.
///
/// \param[in,out] schema C data interface struct representing the data type
/// \return Imported type object
ARROW_EXPORT
Result<std::shared_ptr<DataType>> ImportType(struct ArrowSchema* schema);

/// \brief Import C++ Field from the C data interface.
///
/// The given ArrowSchema struct is released (as per the C data interface
/// specification), even if this function fails.
///
/// \param[in,out] schema C data interface struct representing the field
/// \return Imported field object
ARROW_EXPORT
Result<std::shared_ptr<Field>> ImportField(struct ArrowSchema* schema);

/// \brief Import C++ Schema from the C data interface.
///
/// The given ArrowSchema struct is released (as per the C data interface
/// specification), even if this function fails.
///
/// \param[in,out] schema C data interface struct representing the field
/// \return Imported field object
ARROW_EXPORT
Result<std::shared_ptr<Schema>> ImportSchema(struct ArrowSchema* schema);

/// \brief Import C++ array from the C data interface.
///
/// The ArrowArray struct has its contents moved (as per the C data interface
/// specification) to a private object held alive by the resulting array.
///
/// \param[in,out] array C data interface struct holding the array data
/// \param[in] type type of the imported array
/// \return Imported array object
ARROW_EXPORT
Result<std::shared_ptr<Array>> ImportArray(struct ArrowArray* array,
                                           std::shared_ptr<DataType> type);

/// \brief Import C++ array and its type from the C data interface.
///
/// The ArrowArray struct has its contents moved (as per the C data interface
/// specification) to a private object held alive by the resulting array.
/// The ArrowSchema struct is released, even if this function fails.
///
/// \param[in,out] array C data interface struct holding the array data
/// \param[in,out] type C data interface struct holding the array type
/// \return Imported array object
ARROW_EXPORT
Result<std::shared_ptr<Array>> ImportArray(struct ArrowArray* array,
                                           struct ArrowSchema* type);

/// \brief Import C++ record batch from the C data interface.
///
/// The ArrowArray struct has its contents moved (as per the C data interface
/// specification) to a private object held alive by the resulting record batch.
///
/// \param[in,out] array C data interface struct holding the record batch data
/// \param[in] schema schema of the imported record batch
/// \return Imported record batch object
ARROW_EXPORT
Result<std::shared_ptr<RecordBatch>> ImportRecordBatch(struct ArrowArray* array,
                                                       std::shared_ptr<Schema> schema);

/// \brief Import C++ record batch and its schema from the C data interface.
///
/// The type represented by the ArrowSchema struct must be a struct type array.
/// The ArrowArray struct has its contents moved (as per the C data interface
/// specification) to a private object held alive by the resulting record batch.
/// The ArrowSchema struct is released, even if this function fails.
///
/// \param[in,out] array C data interface struct holding the record batch data
/// \param[in,out] schema C data interface struct holding the record batch schema
/// \return Imported record batch object
ARROW_EXPORT
Result<std::shared_ptr<RecordBatch>> ImportRecordBatch(struct ArrowArray* array,
                                                       struct ArrowSchema* schema);

/// @}

/// \defgroup c-stream-interface Functions for working with the C data interface.
///
/// @{

/// \brief Export C++ RecordBatchReader using the C stream interface.
///
/// The resulting ArrowArrayStream struct keeps the record batch reader alive
/// until its release callback is called by the consumer.
///
/// \param[in] reader RecordBatchReader object to export
/// \param[out] out C struct where to export the stream
ARROW_EXPORT
Status ExportRecordBatchReader(std::shared_ptr<RecordBatchReader> reader,
                               struct ArrowArrayStream* out);

/// \brief Import C++ RecordBatchReader from the C stream interface.
///
/// The ArrowArrayStream struct has its contents moved to a private object
/// held alive by the resulting record batch reader.
///
/// \param[in,out] stream C stream interface struct
/// \return Imported RecordBatchReader object
ARROW_EXPORT
Result<std::shared_ptr<RecordBatchReader>> ImportRecordBatchReader(
    struct ArrowArrayStream* stream);

/// @}

}  // namespace arrow

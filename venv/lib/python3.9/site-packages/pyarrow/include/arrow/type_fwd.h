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

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {

template <typename T>
class Iterator;
template <typename T>
struct IterationTraits;

template <typename T>
class Result;

class Status;

namespace internal {
struct Empty;
}  // namespace internal
template <typename T = internal::Empty>
class Future;

namespace util {
class Codec;
}  // namespace util

class Buffer;
class Device;
class MemoryManager;
class MemoryPool;
class MutableBuffer;
class ResizableBuffer;

using BufferVector = std::vector<std::shared_ptr<Buffer>>;

class DataType;
class Field;
class FieldRef;
class KeyValueMetadata;
enum class Endianness;
class Schema;

using DataTypeVector = std::vector<std::shared_ptr<DataType>>;
using FieldVector = std::vector<std::shared_ptr<Field>>;

class Array;
struct ArrayData;
class ArrayBuilder;
struct Scalar;

using ArrayDataVector = std::vector<std::shared_ptr<ArrayData>>;
using ArrayVector = std::vector<std::shared_ptr<Array>>;
using ScalarVector = std::vector<std::shared_ptr<Scalar>>;

class ChunkedArray;
class RecordBatch;
class RecordBatchReader;
class Table;

struct Datum;
struct TypeHolder;

using ChunkedArrayVector = std::vector<std::shared_ptr<ChunkedArray>>;
using RecordBatchVector = std::vector<std::shared_ptr<RecordBatch>>;
using RecordBatchIterator = Iterator<std::shared_ptr<RecordBatch>>;

class DictionaryType;
class DictionaryArray;
struct DictionaryScalar;

class NullType;
class NullArray;
class NullBuilder;
struct NullScalar;

class FixedWidthType;

class BooleanType;
class BooleanArray;
class BooleanBuilder;
struct BooleanScalar;

class BinaryType;
class BinaryArray;
class BinaryBuilder;
struct BinaryScalar;

class LargeBinaryType;
class LargeBinaryArray;
class LargeBinaryBuilder;
struct LargeBinaryScalar;

class FixedSizeBinaryType;
class FixedSizeBinaryArray;
class FixedSizeBinaryBuilder;
struct FixedSizeBinaryScalar;

class StringType;
class StringArray;
class StringBuilder;
struct StringScalar;

class LargeStringType;
class LargeStringArray;
class LargeStringBuilder;
struct LargeStringScalar;

class ListType;
class ListArray;
class ListBuilder;
struct ListScalar;

class LargeListType;
class LargeListArray;
class LargeListBuilder;
struct LargeListScalar;

class MapType;
class MapArray;
class MapBuilder;
struct MapScalar;

class FixedSizeListType;
class FixedSizeListArray;
class FixedSizeListBuilder;
struct FixedSizeListScalar;

class StructType;
class StructArray;
class StructBuilder;
struct StructScalar;

class Decimal128;
class Decimal256;
class DecimalType;
class Decimal128Type;
class Decimal256Type;
class Decimal128Array;
class Decimal256Array;
class Decimal128Builder;
class Decimal256Builder;
struct Decimal128Scalar;
struct Decimal256Scalar;

struct UnionMode {
  enum type { SPARSE, DENSE };
};

class SparseUnionType;
class SparseUnionArray;
class SparseUnionBuilder;
struct SparseUnionScalar;

class DenseUnionType;
class DenseUnionArray;
class DenseUnionBuilder;
struct DenseUnionScalar;

template <typename TypeClass>
class NumericArray;

template <typename TypeClass>
class NumericBuilder;

template <typename TypeClass>
class NumericTensor;

#define _NUMERIC_TYPE_DECL(KLASS)                     \
  class KLASS##Type;                                  \
  using KLASS##Array = NumericArray<KLASS##Type>;     \
  using KLASS##Builder = NumericBuilder<KLASS##Type>; \
  struct KLASS##Scalar;                               \
  using KLASS##Tensor = NumericTensor<KLASS##Type>;

_NUMERIC_TYPE_DECL(Int8)
_NUMERIC_TYPE_DECL(Int16)
_NUMERIC_TYPE_DECL(Int32)
_NUMERIC_TYPE_DECL(Int64)
_NUMERIC_TYPE_DECL(UInt8)
_NUMERIC_TYPE_DECL(UInt16)
_NUMERIC_TYPE_DECL(UInt32)
_NUMERIC_TYPE_DECL(UInt64)
_NUMERIC_TYPE_DECL(HalfFloat)
_NUMERIC_TYPE_DECL(Float)
_NUMERIC_TYPE_DECL(Double)

#undef _NUMERIC_TYPE_DECL

enum class DateUnit : char { DAY = 0, MILLI = 1 };

class DateType;
class Date32Type;
using Date32Array = NumericArray<Date32Type>;
using Date32Builder = NumericBuilder<Date32Type>;
struct Date32Scalar;

class Date64Type;
using Date64Array = NumericArray<Date64Type>;
using Date64Builder = NumericBuilder<Date64Type>;
struct Date64Scalar;

struct ARROW_EXPORT TimeUnit {
  /// The unit for a time or timestamp DataType
  enum type { SECOND = 0, MILLI = 1, MICRO = 2, NANO = 3 };

  /// Iterate over all valid time units
  static const std::vector<TimeUnit::type>& values();
};

class TimeType;
class Time32Type;
using Time32Array = NumericArray<Time32Type>;
using Time32Builder = NumericBuilder<Time32Type>;
struct Time32Scalar;

class Time64Type;
using Time64Array = NumericArray<Time64Type>;
using Time64Builder = NumericBuilder<Time64Type>;
struct Time64Scalar;

class TimestampType;
using TimestampArray = NumericArray<TimestampType>;
using TimestampBuilder = NumericBuilder<TimestampType>;
struct TimestampScalar;

class MonthIntervalType;
using MonthIntervalArray = NumericArray<MonthIntervalType>;
using MonthIntervalBuilder = NumericBuilder<MonthIntervalType>;
struct MonthIntervalScalar;

class DayTimeIntervalType;
class DayTimeIntervalArray;
class DayTimeIntervalBuilder;
struct DayTimeIntervalScalar;

class MonthDayNanoIntervalType;
class MonthDayNanoIntervalArray;
class MonthDayNanoIntervalBuilder;
struct MonthDayNanoIntervalScalar;

class DurationType;
using DurationArray = NumericArray<DurationType>;
using DurationBuilder = NumericBuilder<DurationType>;
struct DurationScalar;

class ExtensionType;
class ExtensionArray;
struct ExtensionScalar;

class Tensor;
class SparseTensor;

// ----------------------------------------------------------------------

struct Type {
  /// \brief Main data type enumeration
  ///
  /// This enumeration provides a quick way to interrogate the category
  /// of a DataType instance.
  enum type {
    /// A NULL type having no physical storage
    NA = 0,

    /// Boolean as 1 bit, LSB bit-packed ordering
    BOOL,

    /// Unsigned 8-bit little-endian integer
    UINT8,

    /// Signed 8-bit little-endian integer
    INT8,

    /// Unsigned 16-bit little-endian integer
    UINT16,

    /// Signed 16-bit little-endian integer
    INT16,

    /// Unsigned 32-bit little-endian integer
    UINT32,

    /// Signed 32-bit little-endian integer
    INT32,

    /// Unsigned 64-bit little-endian integer
    UINT64,

    /// Signed 64-bit little-endian integer
    INT64,

    /// 2-byte floating point value
    HALF_FLOAT,

    /// 4-byte floating point value
    FLOAT,

    /// 8-byte floating point value
    DOUBLE,

    /// UTF8 variable-length string as List<Char>
    STRING,

    /// Variable-length bytes (no guarantee of UTF8-ness)
    BINARY,

    /// Fixed-size binary. Each value occupies the same number of bytes
    FIXED_SIZE_BINARY,

    /// int32_t days since the UNIX epoch
    DATE32,

    /// int64_t milliseconds since the UNIX epoch
    DATE64,

    /// Exact timestamp encoded with int64 since UNIX epoch
    /// Default unit millisecond
    TIMESTAMP,

    /// Time as signed 32-bit integer, representing either seconds or
    /// milliseconds since midnight
    TIME32,

    /// Time as signed 64-bit integer, representing either microseconds or
    /// nanoseconds since midnight
    TIME64,

    /// YEAR_MONTH interval in SQL style
    INTERVAL_MONTHS,

    /// DAY_TIME interval in SQL style
    INTERVAL_DAY_TIME,

    /// Precision- and scale-based decimal type with 128 bits.
    DECIMAL128,

    /// Defined for backward-compatibility.
    DECIMAL = DECIMAL128,

    /// Precision- and scale-based decimal type with 256 bits.
    DECIMAL256,

    /// A list of some logical data type
    LIST,

    /// Struct of logical types
    STRUCT,

    /// Sparse unions of logical types
    SPARSE_UNION,

    /// Dense unions of logical types
    DENSE_UNION,

    /// Dictionary-encoded type, also called "categorical" or "factor"
    /// in other programming languages. Holds the dictionary value
    /// type but not the dictionary itself, which is part of the
    /// ArrayData struct
    DICTIONARY,

    /// Map, a repeated struct logical type
    MAP,

    /// Custom data type, implemented by user
    EXTENSION,

    /// Fixed size list of some logical type
    FIXED_SIZE_LIST,

    /// Measure of elapsed time in either seconds, milliseconds, microseconds
    /// or nanoseconds.
    DURATION,

    /// Like STRING, but with 64-bit offsets
    LARGE_STRING,

    /// Like BINARY, but with 64-bit offsets
    LARGE_BINARY,

    /// Like LIST, but with 64-bit offsets
    LARGE_LIST,

    /// Calendar interval type with three fields.
    INTERVAL_MONTH_DAY_NANO,

    // Leave this at the end
    MAX_ID
  };
};

/// \brief Get a vector of all type ids
ARROW_EXPORT std::vector<Type::type> AllTypeIds();

/// \defgroup type-factories Factory functions for creating data types
///
/// Factory functions for creating data types
/// @{

/// \brief Return a NullType instance
ARROW_EXPORT const std::shared_ptr<DataType>& null();
/// \brief Return a BooleanType instance
ARROW_EXPORT const std::shared_ptr<DataType>& boolean();
/// \brief Return a Int8Type instance
ARROW_EXPORT const std::shared_ptr<DataType>& int8();
/// \brief Return a Int16Type instance
ARROW_EXPORT const std::shared_ptr<DataType>& int16();
/// \brief Return a Int32Type instance
ARROW_EXPORT const std::shared_ptr<DataType>& int32();
/// \brief Return a Int64Type instance
ARROW_EXPORT const std::shared_ptr<DataType>& int64();
/// \brief Return a UInt8Type instance
ARROW_EXPORT const std::shared_ptr<DataType>& uint8();
/// \brief Return a UInt16Type instance
ARROW_EXPORT const std::shared_ptr<DataType>& uint16();
/// \brief Return a UInt32Type instance
ARROW_EXPORT const std::shared_ptr<DataType>& uint32();
/// \brief Return a UInt64Type instance
ARROW_EXPORT const std::shared_ptr<DataType>& uint64();
/// \brief Return a HalfFloatType instance
ARROW_EXPORT const std::shared_ptr<DataType>& float16();
/// \brief Return a FloatType instance
ARROW_EXPORT const std::shared_ptr<DataType>& float32();
/// \brief Return a DoubleType instance
ARROW_EXPORT const std::shared_ptr<DataType>& float64();
/// \brief Return a StringType instance
ARROW_EXPORT const std::shared_ptr<DataType>& utf8();
/// \brief Return a LargeStringType instance
ARROW_EXPORT const std::shared_ptr<DataType>& large_utf8();
/// \brief Return a BinaryType instance
ARROW_EXPORT const std::shared_ptr<DataType>& binary();
/// \brief Return a LargeBinaryType instance
ARROW_EXPORT const std::shared_ptr<DataType>& large_binary();
/// \brief Return a Date32Type instance
ARROW_EXPORT const std::shared_ptr<DataType>& date32();
/// \brief Return a Date64Type instance
ARROW_EXPORT const std::shared_ptr<DataType>& date64();

/// \brief Create a FixedSizeBinaryType instance.
ARROW_EXPORT
std::shared_ptr<DataType> fixed_size_binary(int32_t byte_width);

/// \brief Create a DecimalType instance depending on the precision
///
/// If the precision is greater than 38, a Decimal256Type is returned,
/// otherwise a Decimal128Type.
ARROW_EXPORT
std::shared_ptr<DataType> decimal(int32_t precision, int32_t scale);

/// \brief Create a Decimal128Type instance
ARROW_EXPORT
std::shared_ptr<DataType> decimal128(int32_t precision, int32_t scale);

/// \brief Create a Decimal256Type instance
ARROW_EXPORT
std::shared_ptr<DataType> decimal256(int32_t precision, int32_t scale);

/// \brief Create a ListType instance from its child Field type
ARROW_EXPORT
std::shared_ptr<DataType> list(const std::shared_ptr<Field>& value_type);

/// \brief Create a ListType instance from its child DataType
ARROW_EXPORT
std::shared_ptr<DataType> list(const std::shared_ptr<DataType>& value_type);

/// \brief Create a LargeListType instance from its child Field type
ARROW_EXPORT
std::shared_ptr<DataType> large_list(const std::shared_ptr<Field>& value_type);

/// \brief Create a LargeListType instance from its child DataType
ARROW_EXPORT
std::shared_ptr<DataType> large_list(const std::shared_ptr<DataType>& value_type);

/// \brief Create a MapType instance from its key and value DataTypes
ARROW_EXPORT
std::shared_ptr<DataType> map(std::shared_ptr<DataType> key_type,
                              std::shared_ptr<DataType> item_type,
                              bool keys_sorted = false);

/// \brief Create a MapType instance from its key DataType and value field.
///
/// The field override is provided to communicate nullability of the value.
ARROW_EXPORT
std::shared_ptr<DataType> map(std::shared_ptr<DataType> key_type,
                              std::shared_ptr<Field> item_field,
                              bool keys_sorted = false);

/// \brief Create a FixedSizeListType instance from its child Field type
ARROW_EXPORT
std::shared_ptr<DataType> fixed_size_list(const std::shared_ptr<Field>& value_type,
                                          int32_t list_size);

/// \brief Create a FixedSizeListType instance from its child DataType
ARROW_EXPORT
std::shared_ptr<DataType> fixed_size_list(const std::shared_ptr<DataType>& value_type,
                                          int32_t list_size);
/// \brief Return a Duration instance (naming use _type to avoid namespace conflict with
/// built in time classes).
ARROW_EXPORT std::shared_ptr<DataType> duration(TimeUnit::type unit);

/// \brief Return a DayTimeIntervalType instance
ARROW_EXPORT std::shared_ptr<DataType> day_time_interval();

/// \brief Return a MonthIntervalType instance
ARROW_EXPORT std::shared_ptr<DataType> month_interval();

/// \brief Return a MonthDayNanoIntervalType instance
ARROW_EXPORT std::shared_ptr<DataType> month_day_nano_interval();

/// \brief Create a TimestampType instance from its unit
ARROW_EXPORT
std::shared_ptr<DataType> timestamp(TimeUnit::type unit);

/// \brief Create a TimestampType instance from its unit and timezone
ARROW_EXPORT
std::shared_ptr<DataType> timestamp(TimeUnit::type unit, const std::string& timezone);

/// \brief Create a 32-bit time type instance
///
/// Unit can be either SECOND or MILLI
ARROW_EXPORT std::shared_ptr<DataType> time32(TimeUnit::type unit);

/// \brief Create a 64-bit time type instance
///
/// Unit can be either MICRO or NANO
ARROW_EXPORT std::shared_ptr<DataType> time64(TimeUnit::type unit);

/// \brief Create a StructType instance
ARROW_EXPORT std::shared_ptr<DataType> struct_(
    const std::vector<std::shared_ptr<Field>>& fields);

/// \brief Create a SparseUnionType instance
ARROW_EXPORT std::shared_ptr<DataType> sparse_union(FieldVector child_fields,
                                                    std::vector<int8_t> type_codes = {});
/// \brief Create a SparseUnionType instance
ARROW_EXPORT std::shared_ptr<DataType> sparse_union(
    const ArrayVector& children, std::vector<std::string> field_names = {},
    std::vector<int8_t> type_codes = {});

/// \brief Create a DenseUnionType instance
ARROW_EXPORT std::shared_ptr<DataType> dense_union(FieldVector child_fields,
                                                   std::vector<int8_t> type_codes = {});
/// \brief Create a DenseUnionType instance
ARROW_EXPORT std::shared_ptr<DataType> dense_union(
    const ArrayVector& children, std::vector<std::string> field_names = {},
    std::vector<int8_t> type_codes = {});

/// \brief Create a DictionaryType instance
/// \param[in] index_type the type of the dictionary indices (must be
/// a signed integer)
/// \param[in] dict_type the type of the values in the variable dictionary
/// \param[in] ordered true if the order of the dictionary values has
/// semantic meaning and should be preserved where possible
ARROW_EXPORT
std::shared_ptr<DataType> dictionary(const std::shared_ptr<DataType>& index_type,
                                     const std::shared_ptr<DataType>& dict_type,
                                     bool ordered = false);

/// @}

/// \defgroup schema-factories Factory functions for fields and schemas
///
/// Factory functions for fields and schemas
/// @{

/// \brief Create a Field instance
///
/// \param name the field name
/// \param type the field value type
/// \param nullable whether the values are nullable, default true
/// \param metadata any custom key-value metadata, default null
ARROW_EXPORT std::shared_ptr<Field> field(
    std::string name, std::shared_ptr<DataType> type, bool nullable = true,
    std::shared_ptr<const KeyValueMetadata> metadata = NULLPTR);

/// \brief Create a Field instance with metadata
///
/// The field will be assumed to be nullable.
///
/// \param name the field name
/// \param type the field value type
/// \param metadata any custom key-value metadata
ARROW_EXPORT std::shared_ptr<Field> field(
    std::string name, std::shared_ptr<DataType> type,
    std::shared_ptr<const KeyValueMetadata> metadata);

/// \brief Create a Schema instance
///
/// \param fields the schema's fields
/// \param metadata any custom key-value metadata, default null
/// \return schema shared_ptr to Schema
ARROW_EXPORT
std::shared_ptr<Schema> schema(
    std::vector<std::shared_ptr<Field>> fields,
    std::shared_ptr<const KeyValueMetadata> metadata = NULLPTR);

/// \brief Create a Schema instance
///
/// \param fields the schema's fields
/// \param endianness the endianness of the data
/// \param metadata any custom key-value metadata, default null
/// \return schema shared_ptr to Schema
ARROW_EXPORT
std::shared_ptr<Schema> schema(
    std::vector<std::shared_ptr<Field>> fields, Endianness endianness,
    std::shared_ptr<const KeyValueMetadata> metadata = NULLPTR);

/// @}

/// Return the process-wide default memory pool.
ARROW_EXPORT MemoryPool* default_memory_pool();

constexpr int64_t kDefaultBufferAlignment = 64;

}  // namespace arrow

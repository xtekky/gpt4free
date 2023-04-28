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
#include <type_traits>
#include <vector>

#include "arrow/type.h"
#include "arrow/util/bit_util.h"

namespace arrow {

//
// Per-type id type lookup
//

template <Type::type id>
struct TypeIdTraits {};

#define TYPE_ID_TRAIT(_id, _typeclass) \
  template <>                          \
  struct TypeIdTraits<Type::_id> {     \
    using Type = _typeclass;           \
  };

TYPE_ID_TRAIT(NA, NullType)
TYPE_ID_TRAIT(BOOL, BooleanType)
TYPE_ID_TRAIT(INT8, Int8Type)
TYPE_ID_TRAIT(INT16, Int16Type)
TYPE_ID_TRAIT(INT32, Int32Type)
TYPE_ID_TRAIT(INT64, Int64Type)
TYPE_ID_TRAIT(UINT8, UInt8Type)
TYPE_ID_TRAIT(UINT16, UInt16Type)
TYPE_ID_TRAIT(UINT32, UInt32Type)
TYPE_ID_TRAIT(UINT64, UInt64Type)
TYPE_ID_TRAIT(HALF_FLOAT, HalfFloatType)
TYPE_ID_TRAIT(FLOAT, FloatType)
TYPE_ID_TRAIT(DOUBLE, DoubleType)
TYPE_ID_TRAIT(STRING, StringType)
TYPE_ID_TRAIT(BINARY, BinaryType)
TYPE_ID_TRAIT(LARGE_STRING, LargeStringType)
TYPE_ID_TRAIT(LARGE_BINARY, LargeBinaryType)
TYPE_ID_TRAIT(FIXED_SIZE_BINARY, FixedSizeBinaryType)
TYPE_ID_TRAIT(DATE32, Date32Type)
TYPE_ID_TRAIT(DATE64, Date64Type)
TYPE_ID_TRAIT(TIME32, Time32Type)
TYPE_ID_TRAIT(TIME64, Time64Type)
TYPE_ID_TRAIT(TIMESTAMP, TimestampType)
TYPE_ID_TRAIT(INTERVAL_DAY_TIME, DayTimeIntervalType)
TYPE_ID_TRAIT(INTERVAL_MONTH_DAY_NANO, MonthDayNanoIntervalType)
TYPE_ID_TRAIT(INTERVAL_MONTHS, MonthIntervalType)
TYPE_ID_TRAIT(DURATION, DurationType)
TYPE_ID_TRAIT(DECIMAL128, Decimal128Type)
TYPE_ID_TRAIT(DECIMAL256, Decimal256Type)
TYPE_ID_TRAIT(STRUCT, StructType)
TYPE_ID_TRAIT(LIST, ListType)
TYPE_ID_TRAIT(LARGE_LIST, LargeListType)
TYPE_ID_TRAIT(FIXED_SIZE_LIST, FixedSizeListType)
TYPE_ID_TRAIT(MAP, MapType)
TYPE_ID_TRAIT(DENSE_UNION, DenseUnionType)
TYPE_ID_TRAIT(SPARSE_UNION, SparseUnionType)
TYPE_ID_TRAIT(DICTIONARY, DictionaryType)
TYPE_ID_TRAIT(EXTENSION, ExtensionType)

#undef TYPE_ID_TRAIT

//
// Per-type type traits
//

/// \addtogroup type-traits
/// \brief Base template for type traits of Arrow data types
/// Type traits provide various information about a type at compile time, such
/// as the associated ArrayType, BuilderType, and ScalarType. Not all types
/// provide all information.
/// \tparam T An Arrow data type
template <typename T>
struct TypeTraits {};

/// \brief Base template for type traits of C++ types
/// \tparam T A standard C++ type
template <typename T>
struct CTypeTraits {};

/// \addtogroup type-traits
/// @{
template <>
struct TypeTraits<NullType> {
  using ArrayType = NullArray;
  using BuilderType = NullBuilder;
  using ScalarType = NullScalar;

  static constexpr int64_t bytes_required(int64_t) { return 0; }
  constexpr static bool is_parameter_free = true;
  static inline std::shared_ptr<DataType> type_singleton() { return null(); }
};

template <>
struct TypeTraits<BooleanType> {
  using ArrayType = BooleanArray;
  using BuilderType = BooleanBuilder;
  using ScalarType = BooleanScalar;
  using CType = bool;

  static constexpr int64_t bytes_required(int64_t elements) {
    return bit_util::BytesForBits(elements);
  }
  constexpr static bool is_parameter_free = true;
  static inline std::shared_ptr<DataType> type_singleton() { return boolean(); }
};
/// @}

/// \addtogroup c-type-traits
template <>
struct CTypeTraits<bool> : public TypeTraits<BooleanType> {
  using ArrowType = BooleanType;
};

#define PRIMITIVE_TYPE_TRAITS_DEF_(CType_, ArrowType_, ArrowArrayType, ArrowBuilderType, \
                                   ArrowScalarType, ArrowTensorType, SingletonFn)        \
  template <>                                                                            \
  struct TypeTraits<ArrowType_> {                                                        \
    using ArrayType = ArrowArrayType;                                                    \
    using BuilderType = ArrowBuilderType;                                                \
    using ScalarType = ArrowScalarType;                                                  \
    using TensorType = ArrowTensorType;                                                  \
    using CType = ArrowType_::c_type;                                                    \
    static constexpr int64_t bytes_required(int64_t elements) {                          \
      return elements * static_cast<int64_t>(sizeof(CType));                             \
    }                                                                                    \
    constexpr static bool is_parameter_free = true;                                      \
    static inline std::shared_ptr<DataType> type_singleton() { return SingletonFn(); }   \
  };                                                                                     \
                                                                                         \
  template <>                                                                            \
  struct CTypeTraits<CType_> : public TypeTraits<ArrowType_> {                           \
    using ArrowType = ArrowType_;                                                        \
  };

#define PRIMITIVE_TYPE_TRAITS_DEF(CType, ArrowShort, SingletonFn)             \
  PRIMITIVE_TYPE_TRAITS_DEF_(                                                 \
      CType, ARROW_CONCAT(ArrowShort, Type), ARROW_CONCAT(ArrowShort, Array), \
      ARROW_CONCAT(ArrowShort, Builder), ARROW_CONCAT(ArrowShort, Scalar),    \
      ARROW_CONCAT(ArrowShort, Tensor), SingletonFn)

PRIMITIVE_TYPE_TRAITS_DEF(uint8_t, UInt8, uint8)
PRIMITIVE_TYPE_TRAITS_DEF(int8_t, Int8, int8)
PRIMITIVE_TYPE_TRAITS_DEF(uint16_t, UInt16, uint16)
PRIMITIVE_TYPE_TRAITS_DEF(int16_t, Int16, int16)
PRIMITIVE_TYPE_TRAITS_DEF(uint32_t, UInt32, uint32)
PRIMITIVE_TYPE_TRAITS_DEF(int32_t, Int32, int32)
PRIMITIVE_TYPE_TRAITS_DEF(uint64_t, UInt64, uint64)
PRIMITIVE_TYPE_TRAITS_DEF(int64_t, Int64, int64)
PRIMITIVE_TYPE_TRAITS_DEF(float, Float, float32)
PRIMITIVE_TYPE_TRAITS_DEF(double, Double, float64)

#undef PRIMITIVE_TYPE_TRAITS_DEF
#undef PRIMITIVE_TYPE_TRAITS_DEF_

/// \addtogroup type-traits
/// @{
template <>
struct TypeTraits<Date64Type> {
  using ArrayType = Date64Array;
  using BuilderType = Date64Builder;
  using ScalarType = Date64Scalar;
  using CType = Date64Type::c_type;

  static constexpr int64_t bytes_required(int64_t elements) {
    return elements * static_cast<int64_t>(sizeof(int64_t));
  }
  constexpr static bool is_parameter_free = true;
  static inline std::shared_ptr<DataType> type_singleton() { return date64(); }
};

template <>
struct TypeTraits<Date32Type> {
  using ArrayType = Date32Array;
  using BuilderType = Date32Builder;
  using ScalarType = Date32Scalar;
  using CType = Date32Type::c_type;

  static constexpr int64_t bytes_required(int64_t elements) {
    return elements * static_cast<int64_t>(sizeof(int32_t));
  }
  constexpr static bool is_parameter_free = true;
  static inline std::shared_ptr<DataType> type_singleton() { return date32(); }
};

template <>
struct TypeTraits<TimestampType> {
  using ArrayType = TimestampArray;
  using BuilderType = TimestampBuilder;
  using ScalarType = TimestampScalar;
  using CType = TimestampType::c_type;

  static constexpr int64_t bytes_required(int64_t elements) {
    return elements * static_cast<int64_t>(sizeof(int64_t));
  }
  constexpr static bool is_parameter_free = false;
};

template <>
struct TypeTraits<DurationType> {
  using ArrayType = DurationArray;
  using BuilderType = DurationBuilder;
  using ScalarType = DurationScalar;
  using CType = DurationType::c_type;

  static constexpr int64_t bytes_required(int64_t elements) {
    return elements * static_cast<int64_t>(sizeof(int64_t));
  }
  constexpr static bool is_parameter_free = false;
};

template <>
struct TypeTraits<DayTimeIntervalType> {
  using ArrayType = DayTimeIntervalArray;
  using BuilderType = DayTimeIntervalBuilder;
  using ScalarType = DayTimeIntervalScalar;
  using CType = DayTimeIntervalType::c_type;

  static constexpr int64_t bytes_required(int64_t elements) {
    return elements * static_cast<int64_t>(sizeof(DayTimeIntervalType::DayMilliseconds));
  }
  constexpr static bool is_parameter_free = true;
  static std::shared_ptr<DataType> type_singleton() { return day_time_interval(); }
};

template <>
struct TypeTraits<MonthDayNanoIntervalType> {
  using ArrayType = MonthDayNanoIntervalArray;
  using BuilderType = MonthDayNanoIntervalBuilder;
  using ScalarType = MonthDayNanoIntervalScalar;
  using CType = MonthDayNanoIntervalType::c_type;

  static constexpr int64_t bytes_required(int64_t elements) {
    return elements *
           static_cast<int64_t>(sizeof(MonthDayNanoIntervalType::MonthDayNanos));
  }
  constexpr static bool is_parameter_free = true;
  static std::shared_ptr<DataType> type_singleton() { return month_day_nano_interval(); }
};

template <>
struct TypeTraits<MonthIntervalType> {
  using ArrayType = MonthIntervalArray;
  using BuilderType = MonthIntervalBuilder;
  using ScalarType = MonthIntervalScalar;
  using CType = MonthIntervalType::c_type;

  static constexpr int64_t bytes_required(int64_t elements) {
    return elements * static_cast<int64_t>(sizeof(int32_t));
  }
  constexpr static bool is_parameter_free = true;
  static std::shared_ptr<DataType> type_singleton() { return month_interval(); }
};

template <>
struct TypeTraits<Time32Type> {
  using ArrayType = Time32Array;
  using BuilderType = Time32Builder;
  using ScalarType = Time32Scalar;
  using CType = Time32Type::c_type;

  static constexpr int64_t bytes_required(int64_t elements) {
    return elements * static_cast<int64_t>(sizeof(int32_t));
  }
  constexpr static bool is_parameter_free = false;
};

template <>
struct TypeTraits<Time64Type> {
  using ArrayType = Time64Array;
  using BuilderType = Time64Builder;
  using ScalarType = Time64Scalar;
  using CType = Time64Type::c_type;

  static constexpr int64_t bytes_required(int64_t elements) {
    return elements * static_cast<int64_t>(sizeof(int64_t));
  }
  constexpr static bool is_parameter_free = false;
};

template <>
struct TypeTraits<HalfFloatType> {
  using ArrayType = HalfFloatArray;
  using BuilderType = HalfFloatBuilder;
  using ScalarType = HalfFloatScalar;
  using TensorType = HalfFloatTensor;

  static constexpr int64_t bytes_required(int64_t elements) {
    return elements * static_cast<int64_t>(sizeof(uint16_t));
  }
  constexpr static bool is_parameter_free = true;
  static inline std::shared_ptr<DataType> type_singleton() { return float16(); }
};

template <>
struct TypeTraits<Decimal128Type> {
  using ArrayType = Decimal128Array;
  using BuilderType = Decimal128Builder;
  using ScalarType = Decimal128Scalar;
  using CType = Decimal128;
  constexpr static bool is_parameter_free = false;
};

template <>
struct TypeTraits<Decimal256Type> {
  using ArrayType = Decimal256Array;
  using BuilderType = Decimal256Builder;
  using ScalarType = Decimal256Scalar;
  using CType = Decimal256;
  constexpr static bool is_parameter_free = false;
};

template <>
struct TypeTraits<BinaryType> {
  using ArrayType = BinaryArray;
  using BuilderType = BinaryBuilder;
  using ScalarType = BinaryScalar;
  using OffsetType = Int32Type;
  constexpr static bool is_parameter_free = true;
  static inline std::shared_ptr<DataType> type_singleton() { return binary(); }
};

template <>
struct TypeTraits<LargeBinaryType> {
  using ArrayType = LargeBinaryArray;
  using BuilderType = LargeBinaryBuilder;
  using ScalarType = LargeBinaryScalar;
  using OffsetType = Int64Type;
  constexpr static bool is_parameter_free = true;
  static inline std::shared_ptr<DataType> type_singleton() { return large_binary(); }
};

template <>
struct TypeTraits<FixedSizeBinaryType> {
  using ArrayType = FixedSizeBinaryArray;
  using BuilderType = FixedSizeBinaryBuilder;
  using ScalarType = FixedSizeBinaryScalar;
  // FixedSizeBinary doesn't have offsets per se, but string length is int32 sized
  using OffsetType = Int32Type;
  constexpr static bool is_parameter_free = false;
};

template <>
struct TypeTraits<StringType> {
  using ArrayType = StringArray;
  using BuilderType = StringBuilder;
  using ScalarType = StringScalar;
  using OffsetType = Int32Type;
  constexpr static bool is_parameter_free = true;
  static inline std::shared_ptr<DataType> type_singleton() { return utf8(); }
};

template <>
struct TypeTraits<LargeStringType> {
  using ArrayType = LargeStringArray;
  using BuilderType = LargeStringBuilder;
  using ScalarType = LargeStringScalar;
  using OffsetType = Int64Type;
  constexpr static bool is_parameter_free = true;
  static inline std::shared_ptr<DataType> type_singleton() { return large_utf8(); }
};

/// @}

/// \addtogroup c-type-traits
/// @{
template <>
struct CTypeTraits<std::string> : public TypeTraits<StringType> {
  using ArrowType = StringType;
};

template <>
struct CTypeTraits<const char*> : public CTypeTraits<std::string> {};

template <size_t N>
struct CTypeTraits<const char (&)[N]> : public CTypeTraits<std::string> {};

template <>
struct CTypeTraits<DayTimeIntervalType::DayMilliseconds>
    : public TypeTraits<DayTimeIntervalType> {
  using ArrowType = DayTimeIntervalType;
};
/// @}

/// \addtogroup type-traits
/// @{
template <>
struct TypeTraits<ListType> {
  using ArrayType = ListArray;
  using BuilderType = ListBuilder;
  using ScalarType = ListScalar;
  using OffsetType = Int32Type;
  using OffsetArrayType = Int32Array;
  using OffsetBuilderType = Int32Builder;
  using OffsetScalarType = Int32Scalar;
  constexpr static bool is_parameter_free = false;
};

template <>
struct TypeTraits<LargeListType> {
  using ArrayType = LargeListArray;
  using BuilderType = LargeListBuilder;
  using ScalarType = LargeListScalar;
  using OffsetType = Int64Type;
  using OffsetArrayType = Int64Array;
  using OffsetBuilderType = Int64Builder;
  using OffsetScalarType = Int64Scalar;
  constexpr static bool is_parameter_free = false;
};

template <>
struct TypeTraits<MapType> {
  using ArrayType = MapArray;
  using BuilderType = MapBuilder;
  using ScalarType = MapScalar;
  using OffsetType = Int32Type;
  using OffsetArrayType = Int32Array;
  using OffsetBuilderType = Int32Builder;
  constexpr static bool is_parameter_free = false;
};

template <>
struct TypeTraits<FixedSizeListType> {
  using ArrayType = FixedSizeListArray;
  using BuilderType = FixedSizeListBuilder;
  using ScalarType = FixedSizeListScalar;
  constexpr static bool is_parameter_free = false;
};
/// @}

/// \addtogroup c-type-traits
template <typename CType>
struct CTypeTraits<std::vector<CType>> : public TypeTraits<ListType> {
  using ArrowType = ListType;

  static inline std::shared_ptr<DataType> type_singleton() {
    return list(CTypeTraits<CType>::type_singleton());
  }
};

/// \addtogroup type-traits
/// @{
template <>
struct TypeTraits<StructType> {
  using ArrayType = StructArray;
  using BuilderType = StructBuilder;
  using ScalarType = StructScalar;
  constexpr static bool is_parameter_free = false;
};

template <>
struct TypeTraits<SparseUnionType> {
  using ArrayType = SparseUnionArray;
  using BuilderType = SparseUnionBuilder;
  using ScalarType = SparseUnionScalar;
  constexpr static bool is_parameter_free = false;
};

template <>
struct TypeTraits<DenseUnionType> {
  using ArrayType = DenseUnionArray;
  using BuilderType = DenseUnionBuilder;
  using ScalarType = DenseUnionScalar;
  constexpr static bool is_parameter_free = false;
};

template <>
struct TypeTraits<DictionaryType> {
  using ArrayType = DictionaryArray;
  using ScalarType = DictionaryScalar;
  constexpr static bool is_parameter_free = false;
};

template <>
struct TypeTraits<ExtensionType> {
  using ArrayType = ExtensionArray;
  using ScalarType = ExtensionScalar;
  constexpr static bool is_parameter_free = false;
};
/// @}

namespace internal {

template <typename... Ts>
struct make_void {
  using type = void;
};

template <typename... Ts>
using void_t = typename make_void<Ts...>::type;

}  // namespace internal

//
// Useful type predicates
//

/// \addtogroup type-predicates
/// @{

// only in C++14
template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

template <typename T>
using is_null_type = std::is_same<NullType, T>;

template <typename T, typename R = void>
using enable_if_null = enable_if_t<is_null_type<T>::value, R>;

template <typename T>
using is_boolean_type = std::is_same<BooleanType, T>;

template <typename T, typename R = void>
using enable_if_boolean = enable_if_t<is_boolean_type<T>::value, R>;

template <typename T>
using is_number_type = std::is_base_of<NumberType, T>;

template <typename T, typename R = void>
using enable_if_number = enable_if_t<is_number_type<T>::value, R>;

template <typename T>
using is_integer_type = std::is_base_of<IntegerType, T>;

template <typename T, typename R = void>
using enable_if_integer = enable_if_t<is_integer_type<T>::value, R>;

template <typename T>
using is_signed_integer_type =
    std::integral_constant<bool, is_integer_type<T>::value &&
                                     std::is_signed<typename T::c_type>::value>;

template <typename T, typename R = void>
using enable_if_signed_integer = enable_if_t<is_signed_integer_type<T>::value, R>;

template <typename T>
using is_unsigned_integer_type =
    std::integral_constant<bool, is_integer_type<T>::value &&
                                     std::is_unsigned<typename T::c_type>::value>;

template <typename T, typename R = void>
using enable_if_unsigned_integer = enable_if_t<is_unsigned_integer_type<T>::value, R>;

// Note this will also include HalfFloatType which is represented by a
// non-floating point primitive (uint16_t).
template <typename T>
using is_floating_type = std::is_base_of<FloatingPointType, T>;

template <typename T, typename R = void>
using enable_if_floating_point = enable_if_t<is_floating_type<T>::value, R>;

// Half floats are special in that they behave physically like an unsigned
// integer.
template <typename T>
using is_half_float_type = std::is_same<HalfFloatType, T>;

template <typename T, typename R = void>
using enable_if_half_float = enable_if_t<is_half_float_type<T>::value, R>;

// Binary Types

// Base binary refers to Binary/LargeBinary/String/LargeString
template <typename T>
using is_base_binary_type = std::is_base_of<BaseBinaryType, T>;

template <typename T, typename R = void>
using enable_if_base_binary = enable_if_t<is_base_binary_type<T>::value, R>;

// Any binary excludes string from Base binary
template <typename T>
using is_binary_type =
    std::integral_constant<bool, std::is_same<BinaryType, T>::value ||
                                     std::is_same<LargeBinaryType, T>::value>;

template <typename T, typename R = void>
using enable_if_binary = enable_if_t<is_binary_type<T>::value, R>;

template <typename T>
using is_string_type =
    std::integral_constant<bool, std::is_same<StringType, T>::value ||
                                     std::is_same<LargeStringType, T>::value>;

template <typename T, typename R = void>
using enable_if_string = enable_if_t<is_string_type<T>::value, R>;

template <typename T>
using is_string_like_type =
    std::integral_constant<bool, is_base_binary_type<T>::value && T::is_utf8>;

template <typename T, typename R = void>
using enable_if_string_like = enable_if_t<is_string_like_type<T>::value, R>;

template <typename T, typename U, typename R = void>
using enable_if_same = enable_if_t<std::is_same<T, U>::value, R>;

// Note that this also includes DecimalType
template <typename T>
using is_fixed_size_binary_type = std::is_base_of<FixedSizeBinaryType, T>;

template <typename T, typename R = void>
using enable_if_fixed_size_binary = enable_if_t<is_fixed_size_binary_type<T>::value, R>;

// This includes primitive, dictionary, and fixed-size-binary types
template <typename T>
using is_fixed_width_type = std::is_base_of<FixedWidthType, T>;

template <typename T, typename R = void>
using enable_if_fixed_width_type = enable_if_t<is_fixed_width_type<T>::value, R>;

template <typename T>
using is_binary_like_type =
    std::integral_constant<bool, (is_base_binary_type<T>::value &&
                                  !is_string_like_type<T>::value) ||
                                     is_fixed_size_binary_type<T>::value>;

template <typename T, typename R = void>
using enable_if_binary_like = enable_if_t<is_binary_like_type<T>::value, R>;

template <typename T>
using is_decimal_type = std::is_base_of<DecimalType, T>;

template <typename T, typename R = void>
using enable_if_decimal = enable_if_t<is_decimal_type<T>::value, R>;

template <typename T>
using is_decimal128_type = std::is_base_of<Decimal128Type, T>;

template <typename T, typename R = void>
using enable_if_decimal128 = enable_if_t<is_decimal128_type<T>::value, R>;

template <typename T>
using is_decimal256_type = std::is_base_of<Decimal256Type, T>;

template <typename T, typename R = void>
using enable_if_decimal256 = enable_if_t<is_decimal256_type<T>::value, R>;

// Nested Types

template <typename T>
using is_nested_type = std::is_base_of<NestedType, T>;

template <typename T, typename R = void>
using enable_if_nested = enable_if_t<is_nested_type<T>::value, R>;

template <typename T, typename R = void>
using enable_if_not_nested = enable_if_t<!is_nested_type<T>::value, R>;

template <typename T>
using is_var_length_list_type =
    std::integral_constant<bool, std::is_base_of<LargeListType, T>::value ||
                                     std::is_base_of<ListType, T>::value>;

template <typename T, typename R = void>
using enable_if_var_size_list = enable_if_t<is_var_length_list_type<T>::value, R>;

// DEPRECATED use is_var_length_list_type.
template <typename T>
using is_base_list_type = is_var_length_list_type<T>;

// DEPRECATED use enable_if_var_size_list
template <typename T, typename R = void>
using enable_if_base_list = enable_if_var_size_list<T, R>;

template <typename T>
using is_fixed_size_list_type = std::is_same<FixedSizeListType, T>;

template <typename T, typename R = void>
using enable_if_fixed_size_list = enable_if_t<is_fixed_size_list_type<T>::value, R>;

template <typename T>
using is_list_type =
    std::integral_constant<bool, std::is_same<T, ListType>::value ||
                                     std::is_same<T, LargeListType>::value ||
                                     std::is_same<T, FixedSizeListType>::value>;

template <typename T, typename R = void>
using enable_if_list_type = enable_if_t<is_list_type<T>::value, R>;

template <typename T>
using is_list_like_type =
    std::integral_constant<bool, is_base_list_type<T>::value ||
                                     is_fixed_size_list_type<T>::value>;

template <typename T, typename R = void>
using enable_if_list_like = enable_if_t<is_list_like_type<T>::value, R>;

template <typename T>
using is_struct_type = std::is_base_of<StructType, T>;

template <typename T, typename R = void>
using enable_if_struct = enable_if_t<is_struct_type<T>::value, R>;

template <typename T>
using is_union_type = std::is_base_of<UnionType, T>;

template <typename T, typename R = void>
using enable_if_union = enable_if_t<is_union_type<T>::value, R>;

// TemporalTypes

template <typename T>
using is_temporal_type = std::is_base_of<TemporalType, T>;

template <typename T, typename R = void>
using enable_if_temporal = enable_if_t<is_temporal_type<T>::value, R>;

template <typename T>
using is_date_type = std::is_base_of<DateType, T>;

template <typename T, typename R = void>
using enable_if_date = enable_if_t<is_date_type<T>::value, R>;

template <typename T>
using is_time_type = std::is_base_of<TimeType, T>;

template <typename T, typename R = void>
using enable_if_time = enable_if_t<is_time_type<T>::value, R>;

template <typename T>
using is_timestamp_type = std::is_base_of<TimestampType, T>;

template <typename T, typename R = void>
using enable_if_timestamp = enable_if_t<is_timestamp_type<T>::value, R>;

template <typename T>
using is_duration_type = std::is_base_of<DurationType, T>;

template <typename T, typename R = void>
using enable_if_duration = enable_if_t<is_duration_type<T>::value, R>;

template <typename T>
using is_interval_type = std::is_base_of<IntervalType, T>;

template <typename T, typename R = void>
using enable_if_interval = enable_if_t<is_interval_type<T>::value, R>;

template <typename T>
using is_dictionary_type = std::is_base_of<DictionaryType, T>;

template <typename T, typename R = void>
using enable_if_dictionary = enable_if_t<is_dictionary_type<T>::value, R>;

template <typename T>
using is_extension_type = std::is_base_of<ExtensionType, T>;

template <typename T, typename R = void>
using enable_if_extension = enable_if_t<is_extension_type<T>::value, R>;

// Attribute differentiation

template <typename T>
using is_primitive_ctype = std::is_base_of<PrimitiveCType, T>;

template <typename T, typename R = void>
using enable_if_primitive_ctype = enable_if_t<is_primitive_ctype<T>::value, R>;

template <typename T>
using has_c_type = std::integral_constant<bool, is_primitive_ctype<T>::value ||
                                                    is_temporal_type<T>::value>;

template <typename T, typename R = void>
using enable_if_has_c_type = enable_if_t<has_c_type<T>::value, R>;

template <typename T>
using has_string_view =
    std::integral_constant<bool, std::is_same<BinaryType, T>::value ||
                                     std::is_same<LargeBinaryType, T>::value ||
                                     std::is_same<StringType, T>::value ||
                                     std::is_same<LargeStringType, T>::value ||
                                     std::is_same<FixedSizeBinaryType, T>::value>;

template <typename T, typename R = void>
using enable_if_has_string_view = enable_if_t<has_string_view<T>::value, R>;

template <typename T>
using is_8bit_int = std::integral_constant<bool, std::is_same<UInt8Type, T>::value ||
                                                     std::is_same<Int8Type, T>::value>;

template <typename T, typename R = void>
using enable_if_8bit_int = enable_if_t<is_8bit_int<T>::value, R>;

template <typename T>
using is_parameter_free_type =
    std::integral_constant<bool, TypeTraits<T>::is_parameter_free>;

template <typename T, typename R = void>
using enable_if_parameter_free = enable_if_t<is_parameter_free_type<T>::value, R>;

// Physical representation quirks

template <typename T>
using is_physical_signed_integer_type =
    std::integral_constant<bool,
                           is_signed_integer_type<T>::value ||
                               (is_temporal_type<T>::value && has_c_type<T>::value &&
                                std::is_integral<typename T::c_type>::value)>;

template <typename T, typename R = void>
using enable_if_physical_signed_integer =
    enable_if_t<is_physical_signed_integer_type<T>::value, R>;

template <typename T>
using is_physical_unsigned_integer_type =
    std::integral_constant<bool, is_unsigned_integer_type<T>::value ||
                                     is_half_float_type<T>::value>;

template <typename T, typename R = void>
using enable_if_physical_unsigned_integer =
    enable_if_t<is_physical_unsigned_integer_type<T>::value, R>;

template <typename T>
using is_physical_integer_type =
    std::integral_constant<bool, is_physical_unsigned_integer_type<T>::value ||
                                     is_physical_signed_integer_type<T>::value>;

template <typename T, typename R = void>
using enable_if_physical_integer = enable_if_t<is_physical_integer_type<T>::value, R>;

// Like is_floating_type but excluding half-floats which don't have a
// float-like c type.
template <typename T>
using is_physical_floating_type =
    std::integral_constant<bool,
                           is_floating_type<T>::value && !is_half_float_type<T>::value>;

template <typename T, typename R = void>
using enable_if_physical_floating_point =
    enable_if_t<is_physical_floating_type<T>::value, R>;

/// @}

/// \addtogroup runtime-type-predicates
/// @{

/// \brief Check for an integer type (signed or unsigned)
///
/// \param[in] type_id the type-id to check
/// \return whether type-id is an integer type one
constexpr bool is_integer(Type::type type_id) {
  switch (type_id) {
    case Type::UINT8:
    case Type::INT8:
    case Type::UINT16:
    case Type::INT16:
    case Type::UINT32:
    case Type::INT32:
    case Type::UINT64:
    case Type::INT64:
      return true;
    default:
      break;
  }
  return false;
}

/// \brief Check for a signed integer type
///
/// \param[in] type_id the type-id to check
/// \return whether type-id is a signed integer type one
constexpr bool is_signed_integer(Type::type type_id) {
  switch (type_id) {
    case Type::INT8:
    case Type::INT16:
    case Type::INT32:
    case Type::INT64:
      return true;
    default:
      break;
  }
  return false;
}

/// \brief Check for an unsigned integer type
///
/// \param[in] type_id the type-id to check
/// \return whether type-id is an unsigned integer type one
constexpr bool is_unsigned_integer(Type::type type_id) {
  switch (type_id) {
    case Type::UINT8:
    case Type::UINT16:
    case Type::UINT32:
    case Type::UINT64:
      return true;
    default:
      break;
  }
  return false;
}

/// \brief Check for a floating point type
///
/// \param[in] type_id the type-id to check
/// \return whether type-id is a floating point type one
constexpr bool is_floating(Type::type type_id) {
  switch (type_id) {
    case Type::HALF_FLOAT:
    case Type::FLOAT:
    case Type::DOUBLE:
      return true;
    default:
      break;
  }
  return false;
}

/// \brief Check for a numeric type
///
/// This predicate doesn't match decimals (see `is_decimal`).
///
/// \param[in] type_id the type-id to check
/// \return whether type-id is a numeric type one
constexpr bool is_numeric(Type::type type_id) {
  switch (type_id) {
    case Type::UINT8:
    case Type::INT8:
    case Type::UINT16:
    case Type::INT16:
    case Type::UINT32:
    case Type::INT32:
    case Type::UINT64:
    case Type::INT64:
    case Type::HALF_FLOAT:
    case Type::FLOAT:
    case Type::DOUBLE:
      return true;
    default:
      break;
  }
  return false;
}

/// \brief Check for a decimal type
///
/// \param[in] type_id the type-id to check
/// \return whether type-id is a decimal type one
constexpr bool is_decimal(Type::type type_id) {
  switch (type_id) {
    case Type::DECIMAL128:
    case Type::DECIMAL256:
      return true;
    default:
      break;
  }
  return false;
}

/// \brief Check for a primitive type
///
/// This predicate doesn't match null, decimals and binary-like types.
///
/// \param[in] type_id the type-id to check
/// \return whether type-id is a primitive type one
constexpr bool is_primitive(Type::type type_id) {
  switch (type_id) {
    case Type::BOOL:
    case Type::UINT8:
    case Type::INT8:
    case Type::UINT16:
    case Type::INT16:
    case Type::UINT32:
    case Type::INT32:
    case Type::UINT64:
    case Type::INT64:
    case Type::HALF_FLOAT:
    case Type::FLOAT:
    case Type::DOUBLE:
    case Type::DATE32:
    case Type::DATE64:
    case Type::TIME32:
    case Type::TIME64:
    case Type::TIMESTAMP:
    case Type::DURATION:
    case Type::INTERVAL_MONTHS:
    case Type::INTERVAL_MONTH_DAY_NANO:
    case Type::INTERVAL_DAY_TIME:
      return true;
    default:
      break;
  }
  return false;
}

/// \brief Check for a base-binary-like type
///
/// This predicate doesn't match fixed-size binary types and will otherwise
/// match all binary- and string-like types regardless of offset width.
///
/// \param[in] type_id the type-id to check
/// \return whether type-id is a base-binary-like type one
constexpr bool is_base_binary_like(Type::type type_id) {
  switch (type_id) {
    case Type::BINARY:
    case Type::LARGE_BINARY:
    case Type::STRING:
    case Type::LARGE_STRING:
      return true;
    default:
      break;
  }
  return false;
}

/// \brief Check for a binary-like type (i.e. with 32-bit offsets)
///
/// \param[in] type_id the type-id to check
/// \return whether type-id is a binary-like type one
constexpr bool is_binary_like(Type::type type_id) {
  switch (type_id) {
    case Type::BINARY:
    case Type::STRING:
      return true;
    default:
      break;
  }
  return false;
}

/// \brief Check for a large-binary-like type (i.e. with 64-bit offsets)
///
/// \param[in] type_id the type-id to check
/// \return whether type-id is a large-binary-like type one
constexpr bool is_large_binary_like(Type::type type_id) {
  switch (type_id) {
    case Type::LARGE_BINARY:
    case Type::LARGE_STRING:
      return true;
    default:
      break;
  }
  return false;
}

/// \brief Check for a binary (non-string) type
///
/// \param[in] type_id the type-id to check
/// \return whether type-id is a binary type one
constexpr bool is_binary(Type::type type_id) {
  switch (type_id) {
    case Type::BINARY:
    case Type::LARGE_BINARY:
      return true;
    default:
      break;
  }
  return false;
}

/// \brief Check for a string type
///
/// \param[in] type_id the type-id to check
/// \return whether type-id is a string type one
constexpr bool is_string(Type::type type_id) {
  switch (type_id) {
    case Type::STRING:
    case Type::LARGE_STRING:
      return true;
    default:
      break;
  }
  return false;
}

/// \brief Check for a temporal type
///
/// \param[in] type_id the type-id to check
/// \return whether type-id is a temporal type one
constexpr bool is_temporal(Type::type type_id) {
  switch (type_id) {
    case Type::DATE32:
    case Type::DATE64:
    case Type::TIME32:
    case Type::TIME64:
    case Type::TIMESTAMP:
      return true;
    default:
      break;
  }
  return false;
}

/// \brief Check for an interval type
///
/// \param[in] type_id the type-id to check
/// \return whether type-id is an interval type one
constexpr bool is_interval(Type::type type_id) {
  switch (type_id) {
    case Type::INTERVAL_MONTHS:
    case Type::INTERVAL_DAY_TIME:
    case Type::INTERVAL_MONTH_DAY_NANO:
      return true;
    default:
      break;
  }
  return false;
}

/// \brief Check for a dictionary type
///
/// \param[in] type_id the type-id to check
/// \return whether type-id is a dictionary type one
constexpr bool is_dictionary(Type::type type_id) { return type_id == Type::DICTIONARY; }

/// \brief Check for a fixed-size-binary type
///
/// This predicate also matches decimals.
/// \param[in] type_id the type-id to check
/// \return whether type-id is a fixed-size-binary type one
constexpr bool is_fixed_size_binary(Type::type type_id) {
  switch (type_id) {
    case Type::DECIMAL128:
    case Type::DECIMAL256:
    case Type::FIXED_SIZE_BINARY:
      return true;
    default:
      break;
  }
  return false;
}

/// \brief Check for a fixed-width type
///
/// \param[in] type_id the type-id to check
/// \return whether type-id is a fixed-width type one
constexpr bool is_fixed_width(Type::type type_id) {
  return is_primitive(type_id) || is_dictionary(type_id) || is_fixed_size_binary(type_id);
}

/// \brief Check for a list-like type
///
/// \param[in] type_id the type-id to check
/// \return whether type-id is a list-like type one
constexpr bool is_list_like(Type::type type_id) {
  switch (type_id) {
    case Type::LIST:
    case Type::LARGE_LIST:
    case Type::FIXED_SIZE_LIST:
    case Type::MAP:
      return true;
    default:
      break;
  }
  return false;
}

/// \brief Check for a nested type
///
/// \param[in] type_id the type-id to check
/// \return whether type-id is a nested type one
constexpr bool is_nested(Type::type type_id) {
  switch (type_id) {
    case Type::LIST:
    case Type::LARGE_LIST:
    case Type::FIXED_SIZE_LIST:
    case Type::MAP:
    case Type::STRUCT:
    case Type::SPARSE_UNION:
    case Type::DENSE_UNION:
      return true;
    default:
      break;
  }
  return false;
}

/// \brief Check for a union type
///
/// \param[in] type_id the type-id to check
/// \return whether type-id is a union type one
constexpr bool is_union(Type::type type_id) {
  switch (type_id) {
    case Type::SPARSE_UNION:
    case Type::DENSE_UNION:
      return true;
    default:
      break;
  }
  return false;
}

/// \brief Return the values bit width of a type
///
/// \param[in] type_id the type-id to check
/// \return the values bit width, or 0 if the type does not have fixed-width values
///
/// For Type::FIXED_SIZE_BINARY, you will instead need to inspect the concrete
/// DataType to get this information.
static inline int bit_width(Type::type type_id) {
  switch (type_id) {
    case Type::BOOL:
      return 1;
    case Type::UINT8:
    case Type::INT8:
      return 8;
    case Type::UINT16:
    case Type::INT16:
      return 16;
    case Type::UINT32:
    case Type::INT32:
    case Type::DATE32:
    case Type::TIME32:
      return 32;
    case Type::UINT64:
    case Type::INT64:
    case Type::DATE64:
    case Type::TIME64:
    case Type::TIMESTAMP:
    case Type::DURATION:
      return 64;

    case Type::HALF_FLOAT:
      return 16;
    case Type::FLOAT:
      return 32;
    case Type::DOUBLE:
      return 64;

    case Type::INTERVAL_MONTHS:
      return 32;
    case Type::INTERVAL_DAY_TIME:
      return 64;
    case Type::INTERVAL_MONTH_DAY_NANO:
      return 128;

    case Type::DECIMAL128:
      return 128;
    case Type::DECIMAL256:
      return 256;

    default:
      break;
  }
  return 0;
}

/// \brief Return the offsets bit width of a type
///
/// \param[in] type_id the type-id to check
/// \return the offsets bit width, or 0 if the type does not have offsets
static inline int offset_bit_width(Type::type type_id) {
  switch (type_id) {
    case Type::STRING:
    case Type::BINARY:
    case Type::LIST:
    case Type::MAP:
    case Type::DENSE_UNION:
      return 32;
    case Type::LARGE_STRING:
    case Type::LARGE_BINARY:
    case Type::LARGE_LIST:
      return 64;
    default:
      break;
  }
  return 0;
}

/// \brief Check for an integer type (signed or unsigned)
///
/// \param[in] type the type to check
/// \return whether type is an integer type
///
/// Convenience for checking using the type's id
static inline bool is_integer(const DataType& type) { return is_integer(type.id()); }

/// \brief Check for a signed integer type
///
/// \param[in] type the type to check
/// \return whether type is a signed integer type
///
/// Convenience for checking using the type's id
static inline bool is_signed_integer(const DataType& type) {
  return is_signed_integer(type.id());
}

/// \brief Check for an unsigned integer type
///
/// \param[in] type the type to check
/// \return whether type is an unsigned integer type
///
/// Convenience for checking using the type's id
static inline bool is_unsigned_integer(const DataType& type) {
  return is_unsigned_integer(type.id());
}

/// \brief Check for a floating point type
///
/// \param[in] type the type to check
/// \return whether type is a floating point type
///
/// Convenience for checking using the type's id
static inline bool is_floating(const DataType& type) { return is_floating(type.id()); }

/// \brief Check for a numeric type (number except boolean type)
///
/// \param[in] type the type to check
/// \return whether type is a numeric type
///
/// Convenience for checking using the type's id
static inline bool is_numeric(const DataType& type) { return is_numeric(type.id()); }

/// \brief Check for a decimal type
///
/// \param[in] type the type to check
/// \return whether type is a decimal type
///
/// Convenience for checking using the type's id
static inline bool is_decimal(const DataType& type) { return is_decimal(type.id()); }

/// \brief Check for a primitive type
///
/// \param[in] type the type to check
/// \return whether type is a primitive type
///
/// Convenience for checking using the type's id
static inline bool is_primitive(const DataType& type) { return is_primitive(type.id()); }

/// \brief Check for a binary or string-like type (except fixed-size binary)
///
/// \param[in] type the type to check
/// \return whether type is a binary or string-like type
///
/// Convenience for checking using the type's id
static inline bool is_base_binary_like(const DataType& type) {
  return is_base_binary_like(type.id());
}

/// \brief Check for a binary-like type
///
/// \param[in] type the type to check
/// \return whether type is a binary-like type
///
/// Convenience for checking using the type's id
static inline bool is_binary_like(const DataType& type) {
  return is_binary_like(type.id());
}

/// \brief Check for a large-binary-like type
///
/// \param[in] type the type to check
/// \return whether type is a large-binary-like type
///
/// Convenience for checking using the type's id
static inline bool is_large_binary_like(const DataType& type) {
  return is_large_binary_like(type.id());
}

/// \brief Check for a binary type
///
/// \param[in] type the type to check
/// \return whether type is a binary type
///
/// Convenience for checking using the type's id
static inline bool is_binary(const DataType& type) { return is_binary(type.id()); }

/// \brief Check for a string type
///
/// \param[in] type the type to check
/// \return whether type is a string type
///
/// Convenience for checking using the type's id
static inline bool is_string(const DataType& type) { return is_string(type.id()); }

/// \brief Check for a temporal type, including time and timestamps for each unit
///
/// \param[in] type the type to check
/// \return whether type is a temporal type
///
/// Convenience for checking using the type's id
static inline bool is_temporal(const DataType& type) { return is_temporal(type.id()); }

/// \brief Check for an interval type
///
/// \param[in] type the type to check
/// \return whether type is a interval type
///
/// Convenience for checking using the type's id
static inline bool is_interval(const DataType& type) { return is_interval(type.id()); }

/// \brief Check for a dictionary type
///
/// \param[in] type the type to check
/// \return whether type is a dictionary type
///
/// Convenience for checking using the type's id
static inline bool is_dictionary(const DataType& type) {
  return is_dictionary(type.id());
}

/// \brief Check for a fixed-size-binary type
///
/// \param[in] type the type to check
/// \return whether type is a fixed-size-binary type
///
/// Convenience for checking using the type's id
static inline bool is_fixed_size_binary(const DataType& type) {
  return is_fixed_size_binary(type.id());
}

/// \brief Check for a fixed-width type
///
/// \param[in] type the type to check
/// \return whether type is a fixed-width type
///
/// Convenience for checking using the type's id
static inline bool is_fixed_width(const DataType& type) {
  return is_fixed_width(type.id());
}

/// \brief Check for a list-like type
///
/// \param[in] type the type to check
/// \return whether type is a list-like type
///
/// Convenience for checking using the type's id
static inline bool is_list_like(const DataType& type) { return is_list_like(type.id()); }

/// \brief Check for a nested type
///
/// \param[in] type the type to check
/// \return whether type is a nested type
///
/// Convenience for checking using the type's id
static inline bool is_nested(const DataType& type) { return is_nested(type.id()); }

/// \brief Check for a union type
///
/// \param[in] type the type to check
/// \return whether type is a union type
///
/// Convenience for checking using the type's id
static inline bool is_union(const DataType& type) { return is_union(type.id()); }

/// @}

}  // namespace arrow

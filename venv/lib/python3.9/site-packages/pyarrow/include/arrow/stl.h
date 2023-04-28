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

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "arrow/array.h"
#include "arrow/array/builder_base.h"
#include "arrow/array/builder_binary.h"
#include "arrow/array/builder_nested.h"
#include "arrow/array/builder_primitive.h"
#include "arrow/chunked_array.h"
#include "arrow/compute/api.h"
#include "arrow/status.h"
#include "arrow/table.h"
#include "arrow/type_fwd.h"
#include "arrow/type_traits.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/macros.h"

namespace arrow {

class Schema;

namespace stl {

namespace internal {

template <typename T, typename = void>
struct is_optional_like : public std::false_type {};

template <typename T, typename = void>
struct is_dereferencable : public std::false_type {};

template <typename T>
struct is_dereferencable<T, arrow::internal::void_t<decltype(*std::declval<T>())>>
    : public std::true_type {};

template <typename T>
struct is_optional_like<
    T, typename std::enable_if<
           std::is_constructible<bool, T>::value && is_dereferencable<T>::value &&
           !std::is_array<typename std::remove_reference<T>::type>::value>::type>
    : public std::true_type {};

template <size_t N, typename Tuple>
using BareTupleElement =
    typename std::decay<typename std::tuple_element<N, Tuple>::type>::type;

}  // namespace internal

template <typename T, typename R = void>
using enable_if_optional_like =
    typename std::enable_if<internal::is_optional_like<T>::value, R>::type;

/// Traits meta class to map standard C/C++ types to equivalent Arrow types.
template <typename T, typename Enable = void>
struct ConversionTraits {};

/// Returns builder type for given standard C/C++ type.
template <typename CType>
using CBuilderType =
    typename TypeTraits<typename ConversionTraits<CType>::ArrowType>::BuilderType;

/// Default implementation of AppendListValues.
///
/// This function can be specialized by user to take advantage of appending
/// contiguous ranges while appending. This default implementation will call
/// ConversionTraits<ValueCType>::AppendRow() for each value in the range.
template <typename ValueCType, typename Range>
inline Status AppendListValues(CBuilderType<ValueCType>& value_builder,
                               Range&& cell_range) {
  for (auto const& value : cell_range) {
    ARROW_RETURN_NOT_OK(ConversionTraits<ValueCType>::AppendRow(value_builder, value));
  }
  return Status::OK();
}

#define ARROW_STL_CONVERSION(CType_, ArrowType_)                                    \
  template <>                                                                       \
  struct ConversionTraits<CType_> : public CTypeTraits<CType_> {                    \
    static Status AppendRow(typename TypeTraits<ArrowType_>::BuilderType& builder,  \
                            CType_ cell) {                                          \
      return builder.Append(cell);                                                  \
    }                                                                               \
    static CType_ GetEntry(const typename TypeTraits<ArrowType_>::ArrayType& array, \
                           size_t j) {                                              \
      return array.Value(j);                                                        \
    }                                                                               \
  };                                                                                \
                                                                                    \
  template <>                                                                       \
  inline Status AppendListValues<CType_, const std::vector<CType_>&>(               \
      typename TypeTraits<ArrowType_>::BuilderType & value_builder,                 \
      const std::vector<CType_>& cell_range) {                                      \
    return value_builder.AppendValues(cell_range);                                  \
  }

ARROW_STL_CONVERSION(bool, BooleanType)
ARROW_STL_CONVERSION(int8_t, Int8Type)
ARROW_STL_CONVERSION(int16_t, Int16Type)
ARROW_STL_CONVERSION(int32_t, Int32Type)
ARROW_STL_CONVERSION(int64_t, Int64Type)
ARROW_STL_CONVERSION(uint8_t, UInt8Type)
ARROW_STL_CONVERSION(uint16_t, UInt16Type)
ARROW_STL_CONVERSION(uint32_t, UInt32Type)
ARROW_STL_CONVERSION(uint64_t, UInt64Type)
ARROW_STL_CONVERSION(float, FloatType)
ARROW_STL_CONVERSION(double, DoubleType)

template <>
struct ConversionTraits<std::string> : public CTypeTraits<std::string> {
  static Status AppendRow(StringBuilder& builder, const std::string& cell) {
    return builder.Append(cell);
  }
  static std::string GetEntry(const StringArray& array, size_t j) {
    return array.GetString(j);
  }
};

/// Append cell range elements as a single value to the list builder.
///
/// Cell range will be added to child builder using AppendListValues<ValueCType>()
/// if provided. AppendListValues<ValueCType>() has a default implementation, but
/// it can be specialized by users.
template <typename ValueCType, typename ListBuilderType, typename Range>
Status AppendCellRange(ListBuilderType& builder, Range&& cell_range) {
  constexpr bool is_list_builder = std::is_same<ListBuilderType, ListBuilder>::value;
  constexpr bool is_large_list_builder =
      std::is_same<ListBuilderType, LargeListBuilder>::value;
  static_assert(
      is_list_builder || is_large_list_builder,
      "Builder type must be either ListBuilder or LargeListBuilder for appending "
      "multiple rows.");

  using ChildBuilderType = CBuilderType<ValueCType>;
  ARROW_RETURN_NOT_OK(builder.Append());
  auto& value_builder =
      ::arrow::internal::checked_cast<ChildBuilderType&>(*builder.value_builder());

  // XXX: Remove appended value before returning if status isn't OK?
  return AppendListValues<ValueCType>(value_builder, std::forward<Range>(cell_range));
}

template <typename ValueCType>
struct ConversionTraits<std::vector<ValueCType>>
    : public CTypeTraits<std::vector<ValueCType>> {
  static Status AppendRow(ListBuilder& builder, const std::vector<ValueCType>& cell) {
    return AppendCellRange<ValueCType>(builder, cell);
  }

  static std::vector<ValueCType> GetEntry(const ListArray& array, size_t j) {
    using ElementArrayType =
        typename TypeTraits<typename ConversionTraits<ValueCType>::ArrowType>::ArrayType;

    const ElementArrayType& value_array =
        ::arrow::internal::checked_cast<const ElementArrayType&>(*array.values());

    std::vector<ValueCType> vec(array.value_length(j));
    for (int64_t i = 0; i < array.value_length(j); i++) {
      vec[i] =
          ConversionTraits<ValueCType>::GetEntry(value_array, array.value_offset(j) + i);
    }
    return vec;
  }
};

template <typename Optional>
struct ConversionTraits<Optional, enable_if_optional_like<Optional>>
    : public CTypeTraits<typename std::decay<decltype(*std::declval<Optional>())>::type> {
  using OptionalInnerType =
      typename std::decay<decltype(*std::declval<Optional>())>::type;
  using typename CTypeTraits<OptionalInnerType>::ArrowType;
  using CTypeTraits<OptionalInnerType>::type_singleton;

  static Status AppendRow(typename TypeTraits<ArrowType>::BuilderType& builder,
                          const Optional& cell) {
    if (cell) {
      return ConversionTraits<OptionalInnerType>::AppendRow(builder, *cell);
    } else {
      return builder.AppendNull();
    }
  }
};

/// Build an arrow::Schema based upon the types defined in a std::tuple-like structure.
///
/// While the type information is available at compile-time, we still need to add the
/// column names at runtime, thus these methods are not constexpr.
template <typename Tuple, std::size_t N = std::tuple_size<Tuple>::value>
struct SchemaFromTuple {
  using Element = internal::BareTupleElement<N - 1, Tuple>;

  // Implementations that take a vector-like object for the column names.

  /// Recursively build a vector of arrow::Field from the defined types.
  ///
  /// In most cases MakeSchema is the better entrypoint for the Schema creation.
  static std::vector<std::shared_ptr<Field>> MakeSchemaRecursion(
      const std::vector<std::string>& names) {
    std::vector<std::shared_ptr<Field>> ret =
        SchemaFromTuple<Tuple, N - 1>::MakeSchemaRecursion(names);
    auto type = ConversionTraits<Element>::type_singleton();
    ret.push_back(field(names[N - 1], type, internal::is_optional_like<Element>::value));
    return ret;
  }

  /// Build a Schema from the types of the tuple-like structure passed in as template
  /// parameter assign the column names at runtime.
  ///
  /// An example usage of this API can look like the following:
  ///
  /// \code{.cpp}
  /// using TupleType = std::tuple<int, std::vector<std::string>>;
  /// std::shared_ptr<Schema> schema =
  ///   SchemaFromTuple<TupleType>::MakeSchema({"int_column", "list_of_strings_column"});
  /// \endcode
  static std::shared_ptr<Schema> MakeSchema(const std::vector<std::string>& names) {
    return std::make_shared<Schema>(MakeSchemaRecursion(names));
  }

  // Implementations that take a tuple-like object for the column names.

  /// Recursively build a vector of arrow::Field from the defined types.
  ///
  /// In most cases MakeSchema is the better entrypoint for the Schema creation.
  template <typename NamesTuple>
  static std::vector<std::shared_ptr<Field>> MakeSchemaRecursionT(
      const NamesTuple& names) {
    using std::get;

    std::vector<std::shared_ptr<Field>> ret =
        SchemaFromTuple<Tuple, N - 1>::MakeSchemaRecursionT(names);
    std::shared_ptr<DataType> type = ConversionTraits<Element>::type_singleton();
    ret.push_back(
        field(get<N - 1>(names), type, internal::is_optional_like<Element>::value));
    return ret;
  }

  /// Build a Schema from the types of the tuple-like structure passed in as template
  /// parameter assign the column names at runtime.
  ///
  /// An example usage of this API can look like the following:
  ///
  /// \code{.cpp}
  /// using TupleType = std::tuple<int, std::vector<std::string>>;
  /// std::shared_ptr<Schema> schema =
  ///   SchemaFromTuple<TupleType>::MakeSchema({"int_column", "list_of_strings_column"});
  /// \endcode
  template <typename NamesTuple>
  static std::shared_ptr<Schema> MakeSchema(const NamesTuple& names) {
    return std::make_shared<Schema>(MakeSchemaRecursionT<NamesTuple>(names));
  }
};

template <typename Tuple>
struct SchemaFromTuple<Tuple, 0> {
  static std::vector<std::shared_ptr<Field>> MakeSchemaRecursion(
      const std::vector<std::string>& names) {
    std::vector<std::shared_ptr<Field>> ret;
    ret.reserve(names.size());
    return ret;
  }

  template <typename NamesTuple>
  static std::vector<std::shared_ptr<Field>> MakeSchemaRecursionT(
      const NamesTuple& names) {
    std::vector<std::shared_ptr<Field>> ret;
    ret.reserve(std::tuple_size<NamesTuple>::value);
    return ret;
  }
};

namespace internal {

template <typename Tuple, std::size_t N = std::tuple_size<Tuple>::value>
struct CreateBuildersRecursive {
  static Status Make(MemoryPool* pool,
                     std::vector<std::unique_ptr<ArrayBuilder>>* builders) {
    using Element = BareTupleElement<N - 1, Tuple>;
    std::shared_ptr<DataType> type = ConversionTraits<Element>::type_singleton();
    ARROW_RETURN_NOT_OK(MakeBuilder(pool, type, &builders->at(N - 1)));

    return CreateBuildersRecursive<Tuple, N - 1>::Make(pool, builders);
  }
};

template <typename Tuple>
struct CreateBuildersRecursive<Tuple, 0> {
  static Status Make(MemoryPool*, std::vector<std::unique_ptr<ArrayBuilder>>*) {
    return Status::OK();
  }
};

template <typename Tuple, std::size_t N = std::tuple_size<Tuple>::value>
struct RowIterator {
  static Status Append(const std::vector<std::unique_ptr<ArrayBuilder>>& builders,
                       const Tuple& row) {
    using std::get;
    using Element = BareTupleElement<N - 1, Tuple>;
    using BuilderType =
        typename TypeTraits<typename ConversionTraits<Element>::ArrowType>::BuilderType;

    BuilderType& builder =
        ::arrow::internal::checked_cast<BuilderType&>(*builders[N - 1]);
    ARROW_RETURN_NOT_OK(ConversionTraits<Element>::AppendRow(builder, get<N - 1>(row)));

    return RowIterator<Tuple, N - 1>::Append(builders, row);
  }
};

template <typename Tuple>
struct RowIterator<Tuple, 0> {
  static Status Append(const std::vector<std::unique_ptr<ArrayBuilder>>& builders,
                       const Tuple& row) {
    return Status::OK();
  }
};

template <typename Tuple, std::size_t N = std::tuple_size<Tuple>::value>
struct EnsureColumnTypes {
  static Status Cast(const Table& table, std::shared_ptr<Table>* table_owner,
                     const compute::CastOptions& cast_options, compute::ExecContext* ctx,
                     std::reference_wrapper<const ::arrow::Table>* result) {
    using Element = BareTupleElement<N - 1, Tuple>;
    std::shared_ptr<DataType> expected_type = ConversionTraits<Element>::type_singleton();

    if (!table.schema()->field(N - 1)->type()->Equals(*expected_type)) {
      ARROW_ASSIGN_OR_RAISE(
          Datum casted,
          compute::Cast(table.column(N - 1), expected_type, cast_options, ctx));
      auto new_field = table.schema()->field(N - 1)->WithType(expected_type);
      ARROW_ASSIGN_OR_RAISE(*table_owner,
                            table.SetColumn(N - 1, new_field, casted.chunked_array()));
      *result = **table_owner;
    }

    return EnsureColumnTypes<Tuple, N - 1>::Cast(result->get(), table_owner, cast_options,
                                                 ctx, result);
  }
};

template <typename Tuple>
struct EnsureColumnTypes<Tuple, 0> {
  static Status Cast(const Table& table, std::shared_ptr<Table>* table_owner,
                     const compute::CastOptions& cast_options, compute::ExecContext* ctx,
                     std::reference_wrapper<const ::arrow::Table>* result) {
    return Status::OK();
  }
};

template <typename Range, typename Tuple, std::size_t N = std::tuple_size<Tuple>::value>
struct TupleSetter {
  static void Fill(const Table& table, Range* rows) {
    using std::get;
    using Element = typename std::tuple_element<N - 1, Tuple>::type;
    using ArrayType =
        typename TypeTraits<typename ConversionTraits<Element>::ArrowType>::ArrayType;

    auto iter = rows->begin();
    const ChunkedArray& chunked_array = *table.column(N - 1);
    for (int i = 0; i < chunked_array.num_chunks(); i++) {
      const ArrayType& array =
          ::arrow::internal::checked_cast<const ArrayType&>(*chunked_array.chunk(i));
      for (int64_t j = 0; j < array.length(); j++) {
        get<N - 1>(*iter++) = ConversionTraits<Element>::GetEntry(array, j);
      }
    }

    return TupleSetter<Range, Tuple, N - 1>::Fill(table, rows);
  }
};

template <typename Range, typename Tuple>
struct TupleSetter<Range, Tuple, 0> {
  static void Fill(const Table& table, Range* rows) {}
};

}  // namespace internal

template <typename Range>
Status TableFromTupleRange(MemoryPool* pool, Range&& rows,
                           const std::vector<std::string>& names,
                           std::shared_ptr<Table>* table) {
  using row_type = typename std::iterator_traits<decltype(std::begin(rows))>::value_type;
  constexpr std::size_t n_columns = std::tuple_size<row_type>::value;

  std::shared_ptr<Schema> schema = SchemaFromTuple<row_type>::MakeSchema(names);

  std::vector<std::unique_ptr<ArrayBuilder>> builders(n_columns);
  ARROW_RETURN_NOT_OK(internal::CreateBuildersRecursive<row_type>::Make(pool, &builders));

  for (auto const& row : rows) {
    ARROW_RETURN_NOT_OK(internal::RowIterator<row_type>::Append(builders, row));
  }

  std::vector<std::shared_ptr<Array>> arrays;
  for (auto const& builder : builders) {
    std::shared_ptr<Array> array;
    ARROW_RETURN_NOT_OK(builder->Finish(&array));
    arrays.emplace_back(array);
  }

  *table = Table::Make(std::move(schema), std::move(arrays));

  return Status::OK();
}

template <typename Range>
Status TupleRangeFromTable(const Table& table, const compute::CastOptions& cast_options,
                           compute::ExecContext* ctx, Range* rows) {
  using row_type = typename std::decay<decltype(*std::begin(*rows))>::type;
  constexpr std::size_t n_columns = std::tuple_size<row_type>::value;

  if (table.schema()->num_fields() != n_columns) {
    return Status::Invalid(
        "Number of columns in the table does not match the width of the target: ",
        table.schema()->num_fields(), " != ", n_columns);
  }

  if (std::size(*rows) != static_cast<size_t>(table.num_rows())) {
    return Status::Invalid(
        "Number of rows in the table does not match the size of the target: ",
        table.num_rows(), " != ", std::size(*rows));
  }

  // Check that all columns have the correct type, otherwise cast them.
  std::shared_ptr<Table> table_owner;
  std::reference_wrapper<const ::arrow::Table> current_table(table);

  ARROW_RETURN_NOT_OK(internal::EnsureColumnTypes<row_type>::Cast(
      table, &table_owner, cast_options, ctx, &current_table));

  internal::TupleSetter<Range, row_type>::Fill(current_table.get(), rows);

  return Status::OK();
}

}  // namespace stl
}  // namespace arrow

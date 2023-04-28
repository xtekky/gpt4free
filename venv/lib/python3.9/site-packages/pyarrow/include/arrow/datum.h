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
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "arrow/array/data.h"
#include "arrow/scalar.h"
#include "arrow/type.h"
#include "arrow/type_traits.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {

class Array;
class ChunkedArray;
class RecordBatch;
class Table;

/// \class Datum
/// \brief Variant type for various Arrow C++ data structures
struct ARROW_EXPORT Datum {
  enum Kind { NONE, SCALAR, ARRAY, CHUNKED_ARRAY, RECORD_BATCH, TABLE };

  struct Empty {};

  // Datums variants may have a length. This special value indicate that the
  // current variant does not have a length.
  static constexpr int64_t kUnknownLength = -1;

  std::variant<Empty, std::shared_ptr<Scalar>, std::shared_ptr<ArrayData>,
               std::shared_ptr<ChunkedArray>, std::shared_ptr<RecordBatch>,
               std::shared_ptr<Table>>
      value;

  /// \brief Empty datum, to be populated elsewhere
  Datum() = default;

  Datum(const Datum& other) = default;
  Datum& operator=(const Datum& other) = default;
  Datum(Datum&& other) = default;
  Datum& operator=(Datum&& other) = default;

  Datum(std::shared_ptr<Scalar> value)  // NOLINT implicit conversion
      : value(std::move(value)) {}

  Datum(std::shared_ptr<ArrayData> value)  // NOLINT implicit conversion
      : value(std::move(value)) {}

  Datum(ArrayData arg)  // NOLINT implicit conversion
      : value(std::make_shared<ArrayData>(std::move(arg))) {}

  Datum(const Array& value);                   // NOLINT implicit conversion
  Datum(const std::shared_ptr<Array>& value);  // NOLINT implicit conversion
  Datum(std::shared_ptr<ChunkedArray> value);  // NOLINT implicit conversion
  Datum(std::shared_ptr<RecordBatch> value);   // NOLINT implicit conversion
  Datum(std::shared_ptr<Table> value);         // NOLINT implicit conversion

  // Explicit constructors from const-refs. Can be expensive, prefer the
  // shared_ptr constructors
  explicit Datum(const ChunkedArray& value);
  explicit Datum(const RecordBatch& value);
  explicit Datum(const Table& value);

  // Cast from subtypes of Array or Scalar to Datum
  template <typename T, bool IsArray = std::is_base_of_v<Array, T>,
            bool IsScalar = std::is_base_of_v<Scalar, T>,
            typename = enable_if_t<IsArray || IsScalar>>
  Datum(std::shared_ptr<T> value)  // NOLINT implicit conversion
      : Datum(std::shared_ptr<typename std::conditional<IsArray, Array, Scalar>::type>(
            std::move(value))) {}

  // Cast from subtypes of Array or Scalar to Datum
  template <typename T, typename TV = typename std::remove_reference_t<T>,
            bool IsArray = std::is_base_of_v<Array, T>,
            bool IsScalar = std::is_base_of_v<Scalar, T>,
            typename = enable_if_t<IsArray || IsScalar>>
  Datum(T&& value)  // NOLINT implicit conversion
      : Datum(std::make_shared<TV>(std::forward<T>(value))) {}

  // Many Scalars are copyable, let that happen
  template <typename T, typename = enable_if_t<std::is_base_of_v<Scalar, T>>>
  Datum(const T& value)  // NOLINT implicit conversion
      : Datum(std::make_shared<T>(value)) {}

  // Convenience constructors
  explicit Datum(bool value);
  explicit Datum(int8_t value);
  explicit Datum(uint8_t value);
  explicit Datum(int16_t value);
  explicit Datum(uint16_t value);
  explicit Datum(int32_t value);
  explicit Datum(uint32_t value);
  explicit Datum(int64_t value);
  explicit Datum(uint64_t value);
  explicit Datum(float value);
  explicit Datum(double value);
  explicit Datum(std::string value);
  explicit Datum(const char* value);

  // Forward to convenience constructors for a DurationScalar from std::chrono::duration
  template <template <typename, typename> class StdDuration, typename Rep,
            typename Period,
            typename = decltype(DurationScalar{StdDuration<Rep, Period>{}})>
  explicit Datum(StdDuration<Rep, Period> d) : Datum{DurationScalar(d)} {}

  Datum::Kind kind() const {
    switch (this->value.index()) {
      case 0:
        return Datum::NONE;
      case 1:
        return Datum::SCALAR;
      case 2:
        return Datum::ARRAY;
      case 3:
        return Datum::CHUNKED_ARRAY;
      case 4:
        return Datum::RECORD_BATCH;
      case 5:
        return Datum::TABLE;
      default:
        return Datum::NONE;
    }
  }

  const std::shared_ptr<ArrayData>& array() const {
    return std::get<std::shared_ptr<ArrayData>>(this->value);
  }

  /// \brief The sum of bytes in each buffer referenced by the datum
  /// Note: Scalars report a size of 0
  /// \see arrow::util::TotalBufferSize for caveats
  int64_t TotalBufferSize() const;

  ArrayData* mutable_array() const { return this->array().get(); }

  std::shared_ptr<Array> make_array() const;

  const std::shared_ptr<ChunkedArray>& chunked_array() const {
    return std::get<std::shared_ptr<ChunkedArray>>(this->value);
  }

  const std::shared_ptr<RecordBatch>& record_batch() const {
    return std::get<std::shared_ptr<RecordBatch>>(this->value);
  }

  const std::shared_ptr<Table>& table() const {
    return std::get<std::shared_ptr<Table>>(this->value);
  }

  const std::shared_ptr<Scalar>& scalar() const {
    return std::get<std::shared_ptr<Scalar>>(this->value);
  }

  template <typename ExactType>
  std::shared_ptr<ExactType> array_as() const {
    return internal::checked_pointer_cast<ExactType>(this->make_array());
  }

  template <typename ExactType>
  const ExactType& scalar_as() const {
    return internal::checked_cast<const ExactType&>(*this->scalar());
  }

  bool is_array() const { return this->kind() == Datum::ARRAY; }

  bool is_chunked_array() const { return this->kind() == Datum::CHUNKED_ARRAY; }

  bool is_arraylike() const {
    return this->kind() == Datum::ARRAY || this->kind() == Datum::CHUNKED_ARRAY;
  }

  bool is_scalar() const { return this->kind() == Datum::SCALAR; }

  /// \brief True if Datum contains a scalar or array-like data
  bool is_value() const { return this->is_arraylike() || this->is_scalar(); }

  int64_t null_count() const;

  /// \brief The value type of the variant, if any
  ///
  /// \return nullptr if no type
  const std::shared_ptr<DataType>& type() const;

  /// \brief The schema of the variant, if any
  ///
  /// \return nullptr if no schema
  const std::shared_ptr<Schema>& schema() const;

  /// \brief The value length of the variant, if any
  ///
  /// \return kUnknownLength if no type
  int64_t length() const;

  /// \brief The array chunks of the variant, if any
  ///
  /// \return empty if not arraylike
  ArrayVector chunks() const;

  bool Equals(const Datum& other) const;

  bool operator==(const Datum& other) const { return Equals(other); }
  bool operator!=(const Datum& other) const { return !Equals(other); }

  std::string ToString() const;
};

ARROW_EXPORT void PrintTo(const Datum&, std::ostream*);

ARROW_EXPORT std::string ToString(Datum::Kind kind);

}  // namespace arrow

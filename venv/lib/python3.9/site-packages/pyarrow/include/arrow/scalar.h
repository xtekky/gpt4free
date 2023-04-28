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

// Object model for scalar (non-Array) values. Not intended for use with large
// amounts of data

#pragma once

#include <iosfwd>
#include <memory>
#include <ratio>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "arrow/compare.h"
#include "arrow/extension_type.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type.h"
#include "arrow/type_fwd.h"
#include "arrow/type_traits.h"
#include "arrow/util/compare.h"
#include "arrow/util/decimal.h"
#include "arrow/util/visibility.h"
#include "arrow/visit_type_inline.h"

namespace arrow {

class Array;

/// \brief Base class for scalar values
///
/// A Scalar represents a single value with a specific DataType.
/// Scalars are useful for passing single value inputs to compute functions,
/// or for representing individual array elements (with a non-trivial
/// wrapping cost, though).
struct ARROW_EXPORT Scalar : public std::enable_shared_from_this<Scalar>,
                             public util::EqualityComparable<Scalar> {
  virtual ~Scalar() = default;

  /// \brief The type of the scalar value
  std::shared_ptr<DataType> type;

  /// \brief Whether the value is valid (not null) or not
  bool is_valid = false;

  using util::EqualityComparable<Scalar>::operator==;
  using util::EqualityComparable<Scalar>::Equals;
  bool Equals(const Scalar& other,
              const EqualOptions& options = EqualOptions::Defaults()) const;

  bool ApproxEquals(const Scalar& other,
                    const EqualOptions& options = EqualOptions::Defaults()) const;

  struct ARROW_EXPORT Hash {
    size_t operator()(const Scalar& scalar) const { return scalar.hash(); }

    size_t operator()(const std::shared_ptr<Scalar>& scalar) const {
      return scalar->hash();
    }
  };

  size_t hash() const;

  std::string ToString() const;

  /// \brief Perform cheap validation checks
  ///
  /// This is O(k) where k is the number of descendents.
  ///
  /// \return Status
  Status Validate() const;

  /// \brief Perform extensive data validation checks
  ///
  /// This is potentially O(k*n) where k is the number of descendents and n
  /// is the length of descendents (if list scalars are involved).
  ///
  /// \return Status
  Status ValidateFull() const;

  static Result<std::shared_ptr<Scalar>> Parse(const std::shared_ptr<DataType>& type,
                                               std::string_view repr);

  // TODO(bkietz) add compute::CastOptions
  Result<std::shared_ptr<Scalar>> CastTo(std::shared_ptr<DataType> to) const;

  /// \brief Apply the ScalarVisitor::Visit() method specialized to the scalar type
  Status Accept(ScalarVisitor* visitor) const;

  /// \brief EXPERIMENTAL Enable obtaining shared_ptr<Scalar> from a const
  /// Scalar& context.
  std::shared_ptr<Scalar> GetSharedPtr() const {
    return const_cast<Scalar*>(this)->shared_from_this();
  }

 protected:
  Scalar(std::shared_ptr<DataType> type, bool is_valid)
      : type(std::move(type)), is_valid(is_valid) {}
};

ARROW_EXPORT void PrintTo(const Scalar& scalar, std::ostream* os);

/// \defgroup concrete-scalar-classes Concrete Scalar subclasses
///
/// @{

/// \brief A scalar value for NullType. Never valid
struct ARROW_EXPORT NullScalar : public Scalar {
 public:
  using TypeClass = NullType;

  NullScalar() : Scalar{null(), false} {}
};

/// @}

namespace internal {

struct ARROW_EXPORT PrimitiveScalarBase : public Scalar {
  explicit PrimitiveScalarBase(std::shared_ptr<DataType> type)
      : Scalar(std::move(type), false) {}

  using Scalar::Scalar;
  /// \brief Get a mutable pointer to the value of this scalar. May be null.
  virtual void* mutable_data() = 0;
  /// \brief Get an immutable view of the value of this scalar as bytes.
  virtual std::string_view view() const = 0;
};

template <typename T, typename CType = typename T::c_type>
struct ARROW_EXPORT PrimitiveScalar : public PrimitiveScalarBase {
  using PrimitiveScalarBase::PrimitiveScalarBase;
  using TypeClass = T;
  using ValueType = CType;

  // Non-null constructor.
  PrimitiveScalar(ValueType value, std::shared_ptr<DataType> type)
      : PrimitiveScalarBase(std::move(type), true), value(value) {}

  explicit PrimitiveScalar(std::shared_ptr<DataType> type)
      : PrimitiveScalarBase(std::move(type), false) {}

  ValueType value{};

  void* mutable_data() override { return &value; }
  std::string_view view() const override {
    return std::string_view(reinterpret_cast<const char*>(&value), sizeof(ValueType));
  };
};

}  // namespace internal

/// \addtogroup concrete-scalar-classes Concrete Scalar subclasses
///
/// @{

struct ARROW_EXPORT BooleanScalar : public internal::PrimitiveScalar<BooleanType, bool> {
  using Base = internal::PrimitiveScalar<BooleanType, bool>;
  using Base::Base;

  explicit BooleanScalar(bool value) : Base(value, boolean()) {}

  BooleanScalar() : Base(boolean()) {}
};

template <typename T>
struct NumericScalar : public internal::PrimitiveScalar<T> {
  using Base = typename internal::PrimitiveScalar<T>;
  using Base::Base;
  using TypeClass = typename Base::TypeClass;
  using ValueType = typename Base::ValueType;

  explicit NumericScalar(ValueType value)
      : Base(value, TypeTraits<T>::type_singleton()) {}

  NumericScalar() : Base(TypeTraits<T>::type_singleton()) {}
};

struct ARROW_EXPORT Int8Scalar : public NumericScalar<Int8Type> {
  using NumericScalar<Int8Type>::NumericScalar;
};

struct ARROW_EXPORT Int16Scalar : public NumericScalar<Int16Type> {
  using NumericScalar<Int16Type>::NumericScalar;
};

struct ARROW_EXPORT Int32Scalar : public NumericScalar<Int32Type> {
  using NumericScalar<Int32Type>::NumericScalar;
};

struct ARROW_EXPORT Int64Scalar : public NumericScalar<Int64Type> {
  using NumericScalar<Int64Type>::NumericScalar;
};

struct ARROW_EXPORT UInt8Scalar : public NumericScalar<UInt8Type> {
  using NumericScalar<UInt8Type>::NumericScalar;
};

struct ARROW_EXPORT UInt16Scalar : public NumericScalar<UInt16Type> {
  using NumericScalar<UInt16Type>::NumericScalar;
};

struct ARROW_EXPORT UInt32Scalar : public NumericScalar<UInt32Type> {
  using NumericScalar<UInt32Type>::NumericScalar;
};

struct ARROW_EXPORT UInt64Scalar : public NumericScalar<UInt64Type> {
  using NumericScalar<UInt64Type>::NumericScalar;
};

struct ARROW_EXPORT HalfFloatScalar : public NumericScalar<HalfFloatType> {
  using NumericScalar<HalfFloatType>::NumericScalar;
};

struct ARROW_EXPORT FloatScalar : public NumericScalar<FloatType> {
  using NumericScalar<FloatType>::NumericScalar;
};

struct ARROW_EXPORT DoubleScalar : public NumericScalar<DoubleType> {
  using NumericScalar<DoubleType>::NumericScalar;
};

struct ARROW_EXPORT BaseBinaryScalar : public internal::PrimitiveScalarBase {
  using internal::PrimitiveScalarBase::PrimitiveScalarBase;
  using ValueType = std::shared_ptr<Buffer>;

  std::shared_ptr<Buffer> value;

  void* mutable_data() override {
    return value ? reinterpret_cast<void*>(value->mutable_data()) : NULLPTR;
  }
  std::string_view view() const override {
    return value ? std::string_view(*value) : std::string_view();
  }

 protected:
  BaseBinaryScalar(std::shared_ptr<Buffer> value, std::shared_ptr<DataType> type)
      : internal::PrimitiveScalarBase{std::move(type), true}, value(std::move(value)) {}
};

struct ARROW_EXPORT BinaryScalar : public BaseBinaryScalar {
  using BaseBinaryScalar::BaseBinaryScalar;
  using TypeClass = BinaryType;

  BinaryScalar(std::shared_ptr<Buffer> value, std::shared_ptr<DataType> type)
      : BaseBinaryScalar(std::move(value), std::move(type)) {}

  explicit BinaryScalar(std::shared_ptr<Buffer> value)
      : BinaryScalar(std::move(value), binary()) {}

  explicit BinaryScalar(std::string s);

  BinaryScalar() : BinaryScalar(binary()) {}
};

struct ARROW_EXPORT StringScalar : public BinaryScalar {
  using BinaryScalar::BinaryScalar;
  using TypeClass = StringType;

  explicit StringScalar(std::shared_ptr<Buffer> value)
      : StringScalar(std::move(value), utf8()) {}

  explicit StringScalar(std::string s);

  StringScalar() : StringScalar(utf8()) {}
};

struct ARROW_EXPORT LargeBinaryScalar : public BaseBinaryScalar {
  using BaseBinaryScalar::BaseBinaryScalar;
  using TypeClass = LargeBinaryType;

  LargeBinaryScalar(std::shared_ptr<Buffer> value, std::shared_ptr<DataType> type)
      : BaseBinaryScalar(std::move(value), std::move(type)) {}

  explicit LargeBinaryScalar(std::shared_ptr<Buffer> value)
      : LargeBinaryScalar(std::move(value), large_binary()) {}

  explicit LargeBinaryScalar(std::string s);

  LargeBinaryScalar() : LargeBinaryScalar(large_binary()) {}
};

struct ARROW_EXPORT LargeStringScalar : public LargeBinaryScalar {
  using LargeBinaryScalar::LargeBinaryScalar;
  using TypeClass = LargeStringType;

  explicit LargeStringScalar(std::shared_ptr<Buffer> value)
      : LargeStringScalar(std::move(value), large_utf8()) {}

  explicit LargeStringScalar(std::string s);

  LargeStringScalar() : LargeStringScalar(large_utf8()) {}
};

struct ARROW_EXPORT FixedSizeBinaryScalar : public BinaryScalar {
  using TypeClass = FixedSizeBinaryType;

  FixedSizeBinaryScalar(std::shared_ptr<Buffer> value, std::shared_ptr<DataType> type,
                        bool is_valid = true);

  explicit FixedSizeBinaryScalar(const std::shared_ptr<Buffer>& value,
                                 bool is_valid = true);

  explicit FixedSizeBinaryScalar(std::string s, bool is_valid = true);
};

template <typename T>
struct TemporalScalar : internal::PrimitiveScalar<T> {
  using internal::PrimitiveScalar<T>::PrimitiveScalar;
  using ValueType = typename internal::PrimitiveScalar<T>::ValueType;

  TemporalScalar(ValueType value, std::shared_ptr<DataType> type)
      : internal::PrimitiveScalar<T>(std::move(value), type) {}
};

template <typename T>
struct DateScalar : public TemporalScalar<T> {
  using TemporalScalar<T>::TemporalScalar;
  using ValueType = typename TemporalScalar<T>::ValueType;

  explicit DateScalar(ValueType value)
      : TemporalScalar<T>(std::move(value), TypeTraits<T>::type_singleton()) {}
  DateScalar() : TemporalScalar<T>(TypeTraits<T>::type_singleton()) {}
};

struct ARROW_EXPORT Date32Scalar : public DateScalar<Date32Type> {
  using DateScalar<Date32Type>::DateScalar;
};

struct ARROW_EXPORT Date64Scalar : public DateScalar<Date64Type> {
  using DateScalar<Date64Type>::DateScalar;
};

template <typename T>
struct ARROW_EXPORT TimeScalar : public TemporalScalar<T> {
  using TemporalScalar<T>::TemporalScalar;

  TimeScalar(typename TemporalScalar<T>::ValueType value, TimeUnit::type unit)
      : TimeScalar(std::move(value), std::make_shared<T>(unit)) {}
};

struct ARROW_EXPORT Time32Scalar : public TimeScalar<Time32Type> {
  using TimeScalar<Time32Type>::TimeScalar;
};

struct ARROW_EXPORT Time64Scalar : public TimeScalar<Time64Type> {
  using TimeScalar<Time64Type>::TimeScalar;
};

struct ARROW_EXPORT TimestampScalar : public TemporalScalar<TimestampType> {
  using TemporalScalar<TimestampType>::TemporalScalar;

  TimestampScalar(typename TemporalScalar<TimestampType>::ValueType value,
                  TimeUnit::type unit, std::string tz = "")
      : TimestampScalar(std::move(value), timestamp(unit, std::move(tz))) {}

  static Result<TimestampScalar> FromISO8601(std::string_view iso8601,
                                             TimeUnit::type unit);
};

template <typename T>
struct IntervalScalar : public TemporalScalar<T> {
  using TemporalScalar<T>::TemporalScalar;
  using ValueType = typename TemporalScalar<T>::ValueType;

  explicit IntervalScalar(ValueType value)
      : TemporalScalar<T>(value, TypeTraits<T>::type_singleton()) {}
  IntervalScalar() : TemporalScalar<T>(TypeTraits<T>::type_singleton()) {}
};

struct ARROW_EXPORT MonthIntervalScalar : public IntervalScalar<MonthIntervalType> {
  using IntervalScalar<MonthIntervalType>::IntervalScalar;
};

struct ARROW_EXPORT DayTimeIntervalScalar : public IntervalScalar<DayTimeIntervalType> {
  using IntervalScalar<DayTimeIntervalType>::IntervalScalar;
};

struct ARROW_EXPORT MonthDayNanoIntervalScalar
    : public IntervalScalar<MonthDayNanoIntervalType> {
  using IntervalScalar<MonthDayNanoIntervalType>::IntervalScalar;
};

struct ARROW_EXPORT DurationScalar : public TemporalScalar<DurationType> {
  using TemporalScalar<DurationType>::TemporalScalar;

  DurationScalar(typename TemporalScalar<DurationType>::ValueType value,
                 TimeUnit::type unit)
      : DurationScalar(std::move(value), duration(unit)) {}

  // Convenience constructors for a DurationScalar from std::chrono::nanoseconds
  template <template <typename, typename> class StdDuration, typename Rep>
  explicit DurationScalar(StdDuration<Rep, std::nano> d)
      : DurationScalar{DurationScalar(d.count(), duration(TimeUnit::NANO))} {}

  // Convenience constructors for a DurationScalar from std::chrono::microseconds
  template <template <typename, typename> class StdDuration, typename Rep>
  explicit DurationScalar(StdDuration<Rep, std::micro> d)
      : DurationScalar{DurationScalar(d.count(), duration(TimeUnit::MICRO))} {}

  // Convenience constructors for a DurationScalar from std::chrono::milliseconds
  template <template <typename, typename> class StdDuration, typename Rep>
  explicit DurationScalar(StdDuration<Rep, std::milli> d)
      : DurationScalar{DurationScalar(d.count(), duration(TimeUnit::MILLI))} {}

  // Convenience constructors for a DurationScalar from std::chrono::seconds
  // or from units which are whole numbers of seconds
  template <template <typename, typename> class StdDuration, typename Rep, intmax_t Num>
  explicit DurationScalar(StdDuration<Rep, std::ratio<Num, 1>> d)
      : DurationScalar{DurationScalar(d.count() * Num, duration(TimeUnit::SECOND))} {}
};

template <typename TYPE_CLASS, typename VALUE_TYPE>
struct ARROW_EXPORT DecimalScalar : public internal::PrimitiveScalarBase {
  using internal::PrimitiveScalarBase::PrimitiveScalarBase;
  using TypeClass = TYPE_CLASS;
  using ValueType = VALUE_TYPE;

  DecimalScalar(ValueType value, std::shared_ptr<DataType> type)
      : internal::PrimitiveScalarBase(std::move(type), true), value(value) {}

  void* mutable_data() override {
    return reinterpret_cast<void*>(value.mutable_native_endian_bytes());
  }

  std::string_view view() const override {
    return std::string_view(reinterpret_cast<const char*>(value.native_endian_bytes()),
                            ValueType::kByteWidth);
  }

  ValueType value;
};

struct ARROW_EXPORT Decimal128Scalar : public DecimalScalar<Decimal128Type, Decimal128> {
  using DecimalScalar::DecimalScalar;
};

struct ARROW_EXPORT Decimal256Scalar : public DecimalScalar<Decimal256Type, Decimal256> {
  using DecimalScalar::DecimalScalar;
};

struct ARROW_EXPORT BaseListScalar : public Scalar {
  using Scalar::Scalar;
  using ValueType = std::shared_ptr<Array>;

  BaseListScalar(std::shared_ptr<Array> value, std::shared_ptr<DataType> type,
                 bool is_valid = true);

  std::shared_ptr<Array> value;
};

struct ARROW_EXPORT ListScalar : public BaseListScalar {
  using TypeClass = ListType;
  using BaseListScalar::BaseListScalar;

  explicit ListScalar(std::shared_ptr<Array> value, bool is_valid = true);
};

struct ARROW_EXPORT LargeListScalar : public BaseListScalar {
  using TypeClass = LargeListType;
  using BaseListScalar::BaseListScalar;

  explicit LargeListScalar(std::shared_ptr<Array> value, bool is_valid = true);
};

struct ARROW_EXPORT MapScalar : public BaseListScalar {
  using TypeClass = MapType;
  using BaseListScalar::BaseListScalar;

  explicit MapScalar(std::shared_ptr<Array> value, bool is_valid = true);
};

struct ARROW_EXPORT FixedSizeListScalar : public BaseListScalar {
  using TypeClass = FixedSizeListType;

  FixedSizeListScalar(std::shared_ptr<Array> value, std::shared_ptr<DataType> type,
                      bool is_valid = true);

  explicit FixedSizeListScalar(std::shared_ptr<Array> value, bool is_valid = true);
};

struct ARROW_EXPORT StructScalar : public Scalar {
  using TypeClass = StructType;
  using ValueType = std::vector<std::shared_ptr<Scalar>>;

  ScalarVector value;

  Result<std::shared_ptr<Scalar>> field(FieldRef ref) const;

  StructScalar(ValueType value, std::shared_ptr<DataType> type, bool is_valid = true)
      : Scalar(std::move(type), is_valid), value(std::move(value)) {}

  static Result<std::shared_ptr<StructScalar>> Make(ValueType value,
                                                    std::vector<std::string> field_names);
};

struct ARROW_EXPORT UnionScalar : public Scalar {
  int8_t type_code;

  virtual const std::shared_ptr<Scalar>& child_value() const = 0;

 protected:
  UnionScalar(std::shared_ptr<DataType> type, int8_t type_code, bool is_valid)
      : Scalar(std::move(type), is_valid), type_code(type_code) {}
};

struct ARROW_EXPORT SparseUnionScalar : public UnionScalar {
  using TypeClass = SparseUnionType;

  // Even though only one of the union values is relevant for this scalar, we
  // nonetheless construct a vector of scalars, one per union value, to have
  // enough data to reconstruct a valid ArraySpan of length 1 from this scalar
  using ValueType = std::vector<std::shared_ptr<Scalar>>;
  ValueType value;

  // The value index corresponding to the active type code
  int child_id;

  SparseUnionScalar(ValueType value, int8_t type_code, std::shared_ptr<DataType> type);

  const std::shared_ptr<Scalar>& child_value() const override {
    return this->value[this->child_id];
  }

  /// \brief Construct a SparseUnionScalar from a single value, versus having
  /// to construct a vector of scalars
  static std::shared_ptr<Scalar> FromValue(std::shared_ptr<Scalar> value, int field_index,
                                           std::shared_ptr<DataType> type);
};

struct ARROW_EXPORT DenseUnionScalar : public UnionScalar {
  using TypeClass = DenseUnionType;

  // For DenseUnionScalar, we can make a valid ArraySpan of length 1 from this
  // scalar
  using ValueType = std::shared_ptr<Scalar>;
  ValueType value;

  const std::shared_ptr<Scalar>& child_value() const override { return this->value; }

  DenseUnionScalar(ValueType value, int8_t type_code, std::shared_ptr<DataType> type)
      : UnionScalar(std::move(type), type_code, value->is_valid),
        value(std::move(value)) {}
};

/// \brief A Scalar value for DictionaryType
///
/// `is_valid` denotes the validity of the `index`, regardless of
/// the corresponding value in the `dictionary`.
struct ARROW_EXPORT DictionaryScalar : public internal::PrimitiveScalarBase {
  using TypeClass = DictionaryType;
  struct ValueType {
    std::shared_ptr<Scalar> index;
    std::shared_ptr<Array> dictionary;
  } value;

  explicit DictionaryScalar(std::shared_ptr<DataType> type);

  DictionaryScalar(ValueType value, std::shared_ptr<DataType> type, bool is_valid = true)
      : internal::PrimitiveScalarBase(std::move(type), is_valid),
        value(std::move(value)) {}

  static std::shared_ptr<DictionaryScalar> Make(std::shared_ptr<Scalar> index,
                                                std::shared_ptr<Array> dict);

  Result<std::shared_ptr<Scalar>> GetEncodedValue() const;

  void* mutable_data() override {
    return internal::checked_cast<internal::PrimitiveScalarBase&>(*value.index)
        .mutable_data();
  }
  std::string_view view() const override {
    return internal::checked_cast<const internal::PrimitiveScalarBase&>(*value.index)
        .view();
  }
};

/// \brief A Scalar value for ExtensionType
///
/// The value is the underlying storage scalar.
/// `is_valid` must only be true if `value` is non-null and `value->is_valid` is true
struct ARROW_EXPORT ExtensionScalar : public Scalar {
  using TypeClass = ExtensionType;
  using ValueType = std::shared_ptr<Scalar>;

  ExtensionScalar(std::shared_ptr<Scalar> storage, std::shared_ptr<DataType> type,
                  bool is_valid = true)
      : Scalar(std::move(type), is_valid), value(std::move(storage)) {}

  template <typename Storage,
            typename = enable_if_t<std::is_base_of<Scalar, Storage>::value>>
  ExtensionScalar(Storage&& storage, std::shared_ptr<DataType> type, bool is_valid = true)
      : ExtensionScalar(std::make_shared<Storage>(std::move(storage)), std::move(type),
                        is_valid) {}

  std::shared_ptr<Scalar> value;
};

/// @}

namespace internal {

inline Status CheckBufferLength(...) { return Status::OK(); }

ARROW_EXPORT Status CheckBufferLength(const FixedSizeBinaryType* t,
                                      const std::shared_ptr<Buffer>* b);

}  // namespace internal

template <typename ValueRef>
struct MakeScalarImpl;

/// \defgroup scalar-factories Scalar factory functions
///
/// @{

/// \brief Scalar factory for null scalars
ARROW_EXPORT
std::shared_ptr<Scalar> MakeNullScalar(std::shared_ptr<DataType> type);

/// \brief Scalar factory for non-null scalars
template <typename Value>
Result<std::shared_ptr<Scalar>> MakeScalar(std::shared_ptr<DataType> type,
                                           Value&& value) {
  return MakeScalarImpl<Value&&>{type, std::forward<Value>(value), NULLPTR}.Finish();
}

/// \brief Type-inferring scalar factory for non-null scalars
///
/// Construct a Scalar instance with a DataType determined by the input C++ type.
/// (for example Int8Scalar for a int8_t input).
/// Only non-parametric primitive types and String are supported.
template <typename Value, typename Traits = CTypeTraits<typename std::decay<Value>::type>,
          typename ScalarType = typename Traits::ScalarType,
          typename Enable = decltype(ScalarType(std::declval<Value>(),
                                                Traits::type_singleton()))>
std::shared_ptr<Scalar> MakeScalar(Value value) {
  return std::make_shared<ScalarType>(std::move(value), Traits::type_singleton());
}

inline std::shared_ptr<Scalar> MakeScalar(std::string value) {
  return std::make_shared<StringScalar>(std::move(value));
}

/// @}

template <typename ValueRef>
struct MakeScalarImpl {
  template <typename T, typename ScalarType = typename TypeTraits<T>::ScalarType,
            typename ValueType = typename ScalarType::ValueType,
            typename Enable = typename std::enable_if<
                std::is_constructible<ScalarType, ValueType,
                                      std::shared_ptr<DataType>>::value &&
                std::is_convertible<ValueRef, ValueType>::value>::type>
  Status Visit(const T& t) {
    ARROW_RETURN_NOT_OK(internal::CheckBufferLength(&t, &value_));
    // `static_cast<ValueRef>` makes a rvalue if ValueRef is `ValueType&&`
    out_ = std::make_shared<ScalarType>(
        static_cast<ValueType>(static_cast<ValueRef>(value_)), std::move(type_));
    return Status::OK();
  }

  Status Visit(const ExtensionType& t) {
    ARROW_ASSIGN_OR_RAISE(auto storage,
                          MakeScalar(t.storage_type(), static_cast<ValueRef>(value_)));
    out_ = std::make_shared<ExtensionScalar>(std::move(storage), type_);
    return Status::OK();
  }

  // Enable constructing string/binary scalars (but not decimal, etc) from std::string
  template <typename T>
  enable_if_t<
      std::is_same<typename std::remove_reference<ValueRef>::type, std::string>::value &&
          (is_base_binary_type<T>::value || std::is_same<T, FixedSizeBinaryType>::value),
      Status>
  Visit(const T& t) {
    using ScalarType = typename TypeTraits<T>::ScalarType;
    out_ = std::make_shared<ScalarType>(Buffer::FromString(std::move(value_)),
                                        std::move(type_));
    return Status::OK();
  }

  Status Visit(const DataType& t) {
    return Status::NotImplemented("constructing scalars of type ", t,
                                  " from unboxed values");
  }

  Result<std::shared_ptr<Scalar>> Finish() && {
    ARROW_RETURN_NOT_OK(VisitTypeInline(*type_, this));
    return std::move(out_);
  }

  std::shared_ptr<DataType> type_;
  ValueRef value_;
  std::shared_ptr<Scalar> out_;
};

}  // namespace arrow

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

#include <atomic>
#include <climits>
#include <cstdint>
#include <iosfwd>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "arrow/result.h"
#include "arrow/type_fwd.h"  // IWYU pragma: export
#include "arrow/util/checked_cast.h"
#include "arrow/util/endian.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"
#include "arrow/visitor.h"  // IWYU pragma: keep

namespace arrow {
namespace detail {

/// \defgroup numeric-datatypes Datatypes for numeric data
/// @{
/// @}

/// \defgroup binary-datatypes Datatypes for binary/string data
/// @{
/// @}

/// \defgroup temporal-datatypes Datatypes for temporal data
/// @{
/// @}

/// \defgroup nested-datatypes Datatypes for nested data
/// @{
/// @}

class ARROW_EXPORT Fingerprintable {
 public:
  virtual ~Fingerprintable();

  const std::string& fingerprint() const {
    auto p = fingerprint_.load();
    if (ARROW_PREDICT_TRUE(p != NULLPTR)) {
      return *p;
    }
    return LoadFingerprintSlow();
  }

  const std::string& metadata_fingerprint() const {
    auto p = metadata_fingerprint_.load();
    if (ARROW_PREDICT_TRUE(p != NULLPTR)) {
      return *p;
    }
    return LoadMetadataFingerprintSlow();
  }

 protected:
  const std::string& LoadFingerprintSlow() const;
  const std::string& LoadMetadataFingerprintSlow() const;

  virtual std::string ComputeFingerprint() const = 0;
  virtual std::string ComputeMetadataFingerprint() const = 0;

  mutable std::atomic<std::string*> fingerprint_{NULLPTR};
  mutable std::atomic<std::string*> metadata_fingerprint_{NULLPTR};
};

}  // namespace detail

/// EXPERIMENTAL: Layout specification for a data type
struct ARROW_EXPORT DataTypeLayout {
  enum BufferKind { FIXED_WIDTH, VARIABLE_WIDTH, BITMAP, ALWAYS_NULL };

  /// Layout specification for a single data type buffer
  struct BufferSpec {
    BufferKind kind;
    int64_t byte_width;  // For FIXED_WIDTH

    bool operator==(const BufferSpec& other) const {
      return kind == other.kind &&
             (kind != FIXED_WIDTH || byte_width == other.byte_width);
    }
    bool operator!=(const BufferSpec& other) const { return !(*this == other); }
  };

  static BufferSpec FixedWidth(int64_t w) { return BufferSpec{FIXED_WIDTH, w}; }
  static BufferSpec VariableWidth() { return BufferSpec{VARIABLE_WIDTH, -1}; }
  static BufferSpec Bitmap() { return BufferSpec{BITMAP, -1}; }
  static BufferSpec AlwaysNull() { return BufferSpec{ALWAYS_NULL, -1}; }

  /// A vector of buffer layout specifications, one for each expected buffer
  std::vector<BufferSpec> buffers;
  /// Whether this type expects an associated dictionary array.
  bool has_dictionary = false;

  explicit DataTypeLayout(std::vector<BufferSpec> v) : buffers(std::move(v)) {}
};

/// \brief Base class for all data types
///
/// Data types in this library are all *logical*. They can be expressed as
/// either a primitive physical type (bytes or bits of some fixed size), a
/// nested type consisting of other data types, or another data type (e.g. a
/// timestamp encoded as an int64).
///
/// Simple datatypes may be entirely described by their Type::type id, but
/// complex datatypes are usually parametric.
class ARROW_EXPORT DataType : public std::enable_shared_from_this<DataType>,
                              public detail::Fingerprintable,
                              public util::EqualityComparable<DataType> {
 public:
  explicit DataType(Type::type id) : detail::Fingerprintable(), id_(id) {}
  ~DataType() override;

  /// \brief Return whether the types are equal
  ///
  /// Types that are logically convertible from one to another (e.g. List<UInt8>
  /// and Binary) are NOT equal.
  bool Equals(const DataType& other, bool check_metadata = false) const;

  /// \brief Return whether the types are equal
  bool Equals(const std::shared_ptr<DataType>& other, bool check_metadata = false) const;

  /// \brief Return the child field at index i.
  const std::shared_ptr<Field>& field(int i) const { return children_[i]; }

  /// \brief Return the children fields associated with this type.
  const std::vector<std::shared_ptr<Field>>& fields() const { return children_; }

  /// \brief Return the number of children fields associated with this type.
  int num_fields() const { return static_cast<int>(children_.size()); }

  /// \brief Apply the TypeVisitor::Visit() method specialized to the data type
  Status Accept(TypeVisitor* visitor) const;

  /// \brief A string representation of the type, including any children
  virtual std::string ToString() const = 0;

  /// \brief Return hash value (excluding metadata in child fields)
  size_t Hash() const;

  /// \brief A string name of the type, omitting any child fields
  ///
  /// \since 0.7.0
  virtual std::string name() const = 0;

  /// \brief Return the data type layout.  Children are not included.
  ///
  /// \note Experimental API
  virtual DataTypeLayout layout() const = 0;

  /// \brief Return the type category
  Type::type id() const { return id_; }

  /// \brief Return the type category of the storage type
  virtual Type::type storage_id() const { return id_; }

  /// \brief Returns the type's fixed byte width, if any. Returns -1
  /// for non-fixed-width types, and should only be used for
  /// subclasses of FixedWidthType
  virtual int32_t byte_width() const {
    int32_t num_bits = this->bit_width();
    return num_bits > 0 ? num_bits / 8 : -1;
  }

  /// \brief Returns the type's fixed bit width, if any. Returns -1
  /// for non-fixed-width types, and should only be used for
  /// subclasses of FixedWidthType
  virtual int bit_width() const { return -1; }

  // \brief EXPERIMENTAL: Enable retrieving shared_ptr<DataType> from a const
  // context.
  std::shared_ptr<DataType> GetSharedPtr() const {
    return const_cast<DataType*>(this)->shared_from_this();
  }

 protected:
  // Dummy version that returns a null string (indicating not implemented).
  // Subclasses should override for fast equality checks.
  std::string ComputeFingerprint() const override;

  // Generic versions that works for all regular types, nested or not.
  std::string ComputeMetadataFingerprint() const override;

  Type::type id_;
  std::vector<std::shared_ptr<Field>> children_;

 private:
  ARROW_DISALLOW_COPY_AND_ASSIGN(DataType);
};

/// \brief EXPERIMENTAL: Container for a type pointer which can hold a
/// dynamically created shared_ptr<DataType> if it needs to.
struct ARROW_EXPORT TypeHolder {
  const DataType* type = NULLPTR;
  std::shared_ptr<DataType> owned_type;

  TypeHolder() = default;
  TypeHolder(const TypeHolder& other) = default;
  TypeHolder& operator=(const TypeHolder& other) = default;
  TypeHolder(TypeHolder&& other) = default;
  TypeHolder& operator=(TypeHolder&& other) = default;

  TypeHolder(std::shared_ptr<DataType> owned_type)  // NOLINT implicit construction
      : type(owned_type.get()), owned_type(std::move(owned_type)) {}

  TypeHolder(const DataType* type)  // NOLINT implicit construction
      : type(type) {}

  Type::type id() const { return this->type->id(); }

  std::shared_ptr<DataType> GetSharedPtr() const {
    return this->type != NULLPTR ? this->type->GetSharedPtr() : NULLPTR;
  }

  const DataType& operator*() const { return *this->type; }

  operator bool() const { return this->type != NULLPTR; }

  bool operator==(const TypeHolder& other) const {
    if (type == other.type) return true;
    if (type == NULLPTR || other.type == NULLPTR) return false;
    return type->Equals(*other.type);
  }

  bool operator==(decltype(NULLPTR)) const { return this->type == NULLPTR; }

  bool operator==(const DataType& other) const {
    if (this->type == NULLPTR) return false;
    return other.Equals(*this->type);
  }

  bool operator!=(const DataType& other) const { return !(*this == other); }

  bool operator==(const std::shared_ptr<DataType>& other) const {
    return *this == *other;
  }

  bool operator!=(const TypeHolder& other) const { return !(*this == other); }

  std::string ToString() const {
    return this->type ? this->type->ToString() : "<NULLPTR>";
  }

  static std::string ToString(const std::vector<TypeHolder>&);
};

ARROW_EXPORT
std::ostream& operator<<(std::ostream& os, const DataType& type);

ARROW_EXPORT
std::ostream& operator<<(std::ostream& os, const TypeHolder& type);

/// \brief Return the compatible physical data type
///
/// Some types may have distinct logical meanings but the exact same physical
/// representation.  For example, TimestampType has Int64Type as a physical
/// type (defined as TimestampType::PhysicalType).
///
/// The return value is as follows:
/// - if a `PhysicalType` alias exists in the concrete type class, return
///   an instance of `PhysicalType`.
/// - otherwise, return the input type itself.
std::shared_ptr<DataType> GetPhysicalType(const std::shared_ptr<DataType>& type);

/// \brief Base class for all fixed-width data types
class ARROW_EXPORT FixedWidthType : public DataType {
 public:
  using DataType::DataType;
};

/// \brief Base class for all data types representing primitive values
class ARROW_EXPORT PrimitiveCType : public FixedWidthType {
 public:
  using FixedWidthType::FixedWidthType;
};

/// \brief Base class for all numeric data types
class ARROW_EXPORT NumberType : public PrimitiveCType {
 public:
  using PrimitiveCType::PrimitiveCType;
};

/// \brief Base class for all integral data types
class ARROW_EXPORT IntegerType : public NumberType {
 public:
  using NumberType::NumberType;
  virtual bool is_signed() const = 0;
};

/// \brief Base class for all floating-point data types
class ARROW_EXPORT FloatingPointType : public NumberType {
 public:
  using NumberType::NumberType;
  enum Precision { HALF, SINGLE, DOUBLE };
  virtual Precision precision() const = 0;
};

/// \brief Base class for all parametric data types
class ParametricType {};

class ARROW_EXPORT NestedType : public DataType, public ParametricType {
 public:
  using DataType::DataType;
};

/// \brief The combination of a field name and data type, with optional metadata
///
/// Fields are used to describe the individual constituents of a
/// nested DataType or a Schema.
///
/// A field's metadata is represented by a KeyValueMetadata instance,
/// which holds arbitrary key-value pairs.
class ARROW_EXPORT Field : public detail::Fingerprintable,
                           public util::EqualityComparable<Field> {
 public:
  Field(std::string name, std::shared_ptr<DataType> type, bool nullable = true,
        std::shared_ptr<const KeyValueMetadata> metadata = NULLPTR)
      : detail::Fingerprintable(),
        name_(std::move(name)),
        type_(std::move(type)),
        nullable_(nullable),
        metadata_(std::move(metadata)) {}

  ~Field() override;

  /// \brief Return the field's attached metadata
  std::shared_ptr<const KeyValueMetadata> metadata() const { return metadata_; }

  /// \brief Return whether the field has non-empty metadata
  bool HasMetadata() const;

  /// \brief Return a copy of this field with the given metadata attached to it
  std::shared_ptr<Field> WithMetadata(
      const std::shared_ptr<const KeyValueMetadata>& metadata) const;

  /// \brief EXPERIMENTAL: Return a copy of this field with the given metadata
  /// merged with existing metadata (any colliding keys will be overridden by
  /// the passed metadata)
  std::shared_ptr<Field> WithMergedMetadata(
      const std::shared_ptr<const KeyValueMetadata>& metadata) const;

  /// \brief Return a copy of this field without any metadata attached to it
  std::shared_ptr<Field> RemoveMetadata() const;

  /// \brief Return a copy of this field with the replaced type.
  std::shared_ptr<Field> WithType(const std::shared_ptr<DataType>& type) const;

  /// \brief Return a copy of this field with the replaced name.
  std::shared_ptr<Field> WithName(const std::string& name) const;

  /// \brief Return a copy of this field with the replaced nullability.
  std::shared_ptr<Field> WithNullable(bool nullable) const;

  /// \brief Options that control the behavior of `MergeWith`.
  /// Options are to be added to allow type conversions, including integer
  /// widening, promotion from integer to float, or conversion to or from boolean.
  struct MergeOptions {
    /// If true, a Field of NullType can be unified with a Field of another type.
    /// The unified field will be of the other type and become nullable.
    /// Nullability will be promoted to the looser option (nullable if one is not
    /// nullable).
    bool promote_nullability = true;

    static MergeOptions Defaults() { return MergeOptions(); }
  };

  /// \brief Merge the current field with a field of the same name.
  ///
  /// The two fields must be compatible, i.e:
  ///   - have the same name
  ///   - have the same type, or of compatible types according to `options`.
  ///
  /// The metadata of the current field is preserved; the metadata of the other
  /// field is discarded.
  Result<std::shared_ptr<Field>> MergeWith(
      const Field& other, MergeOptions options = MergeOptions::Defaults()) const;
  Result<std::shared_ptr<Field>> MergeWith(
      const std::shared_ptr<Field>& other,
      MergeOptions options = MergeOptions::Defaults()) const;

  std::vector<std::shared_ptr<Field>> Flatten() const;

  /// \brief Indicate if fields are equals.
  ///
  /// \param[in] other field to check equality with.
  /// \param[in] check_metadata controls if it should check for metadata
  ///            equality.
  ///
  /// \return true if fields are equal, false otherwise.
  bool Equals(const Field& other, bool check_metadata = false) const;
  bool Equals(const std::shared_ptr<Field>& other, bool check_metadata = false) const;

  /// \brief Indicate if fields are compatibles.
  ///
  /// See the criteria of MergeWith.
  ///
  /// \return true if fields are compatible, false otherwise.
  bool IsCompatibleWith(const Field& other) const;
  bool IsCompatibleWith(const std::shared_ptr<Field>& other) const;

  /// \brief Return a string representation ot the field
  /// \param[in] show_metadata when true, if KeyValueMetadata is non-empty,
  /// print keys and values in the output
  std::string ToString(bool show_metadata = false) const;

  /// \brief Return the field name
  const std::string& name() const { return name_; }
  /// \brief Return the field data type
  const std::shared_ptr<DataType>& type() const { return type_; }
  /// \brief Return whether the field is nullable
  bool nullable() const { return nullable_; }

  std::shared_ptr<Field> Copy() const;

 private:
  std::string ComputeFingerprint() const override;
  std::string ComputeMetadataFingerprint() const override;

  // Field name
  std::string name_;

  // The field's data type
  std::shared_ptr<DataType> type_;

  // Fields can be nullable
  bool nullable_;

  // The field's metadata, if any
  std::shared_ptr<const KeyValueMetadata> metadata_;

  ARROW_DISALLOW_COPY_AND_ASSIGN(Field);
};

ARROW_EXPORT void PrintTo(const Field& field, std::ostream* os);

namespace detail {

template <typename DERIVED, typename BASE, Type::type TYPE_ID, typename C_TYPE>
class ARROW_EXPORT CTypeImpl : public BASE {
 public:
  static constexpr Type::type type_id = TYPE_ID;
  using c_type = C_TYPE;
  using PhysicalType = DERIVED;

  CTypeImpl() : BASE(TYPE_ID) {}

  int bit_width() const override { return static_cast<int>(sizeof(C_TYPE) * CHAR_BIT); }

  DataTypeLayout layout() const override {
    return DataTypeLayout(
        {DataTypeLayout::Bitmap(), DataTypeLayout::FixedWidth(sizeof(C_TYPE))});
  }

  std::string name() const override { return DERIVED::type_name(); }

  std::string ToString() const override { return this->name(); }
};

template <typename DERIVED, typename BASE, Type::type TYPE_ID, typename C_TYPE>
constexpr Type::type CTypeImpl<DERIVED, BASE, TYPE_ID, C_TYPE>::type_id;

template <typename DERIVED, Type::type TYPE_ID, typename C_TYPE>
class IntegerTypeImpl : public detail::CTypeImpl<DERIVED, IntegerType, TYPE_ID, C_TYPE> {
  bool is_signed() const override { return std::is_signed<C_TYPE>::value; }
};

}  // namespace detail

/// Concrete type class for always-null data
class ARROW_EXPORT NullType : public DataType {
 public:
  static constexpr Type::type type_id = Type::NA;

  static constexpr const char* type_name() { return "null"; }

  NullType() : DataType(Type::NA) {}

  std::string ToString() const override;

  DataTypeLayout layout() const override {
    return DataTypeLayout({DataTypeLayout::AlwaysNull()});
  }

  std::string name() const override { return "null"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// Concrete type class for boolean data
class ARROW_EXPORT BooleanType
    : public detail::CTypeImpl<BooleanType, PrimitiveCType, Type::BOOL, bool> {
 public:
  static constexpr const char* type_name() { return "bool"; }

  // BooleanType within arrow use a single bit instead of the C 8-bits layout.
  int bit_width() const final { return 1; }

  DataTypeLayout layout() const override {
    return DataTypeLayout({DataTypeLayout::Bitmap(), DataTypeLayout::Bitmap()});
  }

 protected:
  std::string ComputeFingerprint() const override;
};

/// \addtogroup numeric-datatypes
///
/// @{

/// Concrete type class for unsigned 8-bit integer data
class ARROW_EXPORT UInt8Type
    : public detail::IntegerTypeImpl<UInt8Type, Type::UINT8, uint8_t> {
 public:
  static constexpr const char* type_name() { return "uint8"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// Concrete type class for signed 8-bit integer data
class ARROW_EXPORT Int8Type
    : public detail::IntegerTypeImpl<Int8Type, Type::INT8, int8_t> {
 public:
  static constexpr const char* type_name() { return "int8"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// Concrete type class for unsigned 16-bit integer data
class ARROW_EXPORT UInt16Type
    : public detail::IntegerTypeImpl<UInt16Type, Type::UINT16, uint16_t> {
 public:
  static constexpr const char* type_name() { return "uint16"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// Concrete type class for signed 16-bit integer data
class ARROW_EXPORT Int16Type
    : public detail::IntegerTypeImpl<Int16Type, Type::INT16, int16_t> {
 public:
  static constexpr const char* type_name() { return "int16"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// Concrete type class for unsigned 32-bit integer data
class ARROW_EXPORT UInt32Type
    : public detail::IntegerTypeImpl<UInt32Type, Type::UINT32, uint32_t> {
 public:
  static constexpr const char* type_name() { return "uint32"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// Concrete type class for signed 32-bit integer data
class ARROW_EXPORT Int32Type
    : public detail::IntegerTypeImpl<Int32Type, Type::INT32, int32_t> {
 public:
  static constexpr const char* type_name() { return "int32"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// Concrete type class for unsigned 64-bit integer data
class ARROW_EXPORT UInt64Type
    : public detail::IntegerTypeImpl<UInt64Type, Type::UINT64, uint64_t> {
 public:
  static constexpr const char* type_name() { return "uint64"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// Concrete type class for signed 64-bit integer data
class ARROW_EXPORT Int64Type
    : public detail::IntegerTypeImpl<Int64Type, Type::INT64, int64_t> {
 public:
  static constexpr const char* type_name() { return "int64"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// Concrete type class for 16-bit floating-point data
class ARROW_EXPORT HalfFloatType
    : public detail::CTypeImpl<HalfFloatType, FloatingPointType, Type::HALF_FLOAT,
                               uint16_t> {
 public:
  Precision precision() const override;
  static constexpr const char* type_name() { return "halffloat"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// Concrete type class for 32-bit floating-point data (C "float")
class ARROW_EXPORT FloatType
    : public detail::CTypeImpl<FloatType, FloatingPointType, Type::FLOAT, float> {
 public:
  Precision precision() const override;
  static constexpr const char* type_name() { return "float"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// Concrete type class for 64-bit floating-point data (C "double")
class ARROW_EXPORT DoubleType
    : public detail::CTypeImpl<DoubleType, FloatingPointType, Type::DOUBLE, double> {
 public:
  Precision precision() const override;
  static constexpr const char* type_name() { return "double"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// @}

/// \brief Base class for all variable-size binary data types
class ARROW_EXPORT BaseBinaryType : public DataType {
 public:
  using DataType::DataType;
};

constexpr int64_t kBinaryMemoryLimit = std::numeric_limits<int32_t>::max() - 1;

/// \addtogroup binary-datatypes
///
/// @{

/// \brief Concrete type class for variable-size binary data
class ARROW_EXPORT BinaryType : public BaseBinaryType {
 public:
  static constexpr Type::type type_id = Type::BINARY;
  static constexpr bool is_utf8 = false;
  using offset_type = int32_t;
  using PhysicalType = BinaryType;

  static constexpr const char* type_name() { return "binary"; }

  BinaryType() : BinaryType(Type::BINARY) {}

  DataTypeLayout layout() const override {
    return DataTypeLayout({DataTypeLayout::Bitmap(),
                           DataTypeLayout::FixedWidth(sizeof(offset_type)),
                           DataTypeLayout::VariableWidth()});
  }

  std::string ToString() const override;
  std::string name() const override { return "binary"; }

 protected:
  std::string ComputeFingerprint() const override;

  // Allow subclasses like StringType to change the logical type.
  explicit BinaryType(Type::type logical_type) : BaseBinaryType(logical_type) {}
};

/// \brief Concrete type class for large variable-size binary data
class ARROW_EXPORT LargeBinaryType : public BaseBinaryType {
 public:
  static constexpr Type::type type_id = Type::LARGE_BINARY;
  static constexpr bool is_utf8 = false;
  using offset_type = int64_t;
  using PhysicalType = LargeBinaryType;

  static constexpr const char* type_name() { return "large_binary"; }

  LargeBinaryType() : LargeBinaryType(Type::LARGE_BINARY) {}

  DataTypeLayout layout() const override {
    return DataTypeLayout({DataTypeLayout::Bitmap(),
                           DataTypeLayout::FixedWidth(sizeof(offset_type)),
                           DataTypeLayout::VariableWidth()});
  }

  std::string ToString() const override;
  std::string name() const override { return "large_binary"; }

 protected:
  std::string ComputeFingerprint() const override;

  // Allow subclasses like LargeStringType to change the logical type.
  explicit LargeBinaryType(Type::type logical_type) : BaseBinaryType(logical_type) {}
};

/// \brief Concrete type class for variable-size string data, utf8-encoded
class ARROW_EXPORT StringType : public BinaryType {
 public:
  static constexpr Type::type type_id = Type::STRING;
  static constexpr bool is_utf8 = true;
  using PhysicalType = BinaryType;

  static constexpr const char* type_name() { return "utf8"; }

  StringType() : BinaryType(Type::STRING) {}

  std::string ToString() const override;
  std::string name() const override { return "utf8"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// \brief Concrete type class for large variable-size string data, utf8-encoded
class ARROW_EXPORT LargeStringType : public LargeBinaryType {
 public:
  static constexpr Type::type type_id = Type::LARGE_STRING;
  static constexpr bool is_utf8 = true;
  using PhysicalType = LargeBinaryType;

  static constexpr const char* type_name() { return "large_utf8"; }

  LargeStringType() : LargeBinaryType(Type::LARGE_STRING) {}

  std::string ToString() const override;
  std::string name() const override { return "large_utf8"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// \brief Concrete type class for fixed-size binary data
class ARROW_EXPORT FixedSizeBinaryType : public FixedWidthType, public ParametricType {
 public:
  static constexpr Type::type type_id = Type::FIXED_SIZE_BINARY;
  static constexpr bool is_utf8 = false;

  static constexpr const char* type_name() { return "fixed_size_binary"; }

  explicit FixedSizeBinaryType(int32_t byte_width)
      : FixedWidthType(Type::FIXED_SIZE_BINARY), byte_width_(byte_width) {}
  explicit FixedSizeBinaryType(int32_t byte_width, Type::type override_type_id)
      : FixedWidthType(override_type_id), byte_width_(byte_width) {}

  std::string ToString() const override;
  std::string name() const override { return "fixed_size_binary"; }

  DataTypeLayout layout() const override {
    return DataTypeLayout(
        {DataTypeLayout::Bitmap(), DataTypeLayout::FixedWidth(byte_width())});
  }

  int byte_width() const override { return byte_width_; }

  int bit_width() const override;

  // Validating constructor
  static Result<std::shared_ptr<DataType>> Make(int32_t byte_width);

 protected:
  std::string ComputeFingerprint() const override;

  int32_t byte_width_;
};

/// @}

/// \addtogroup numeric-datatypes
///
/// @{

/// \brief Base type class for (fixed-size) decimal data
class ARROW_EXPORT DecimalType : public FixedSizeBinaryType {
 public:
  explicit DecimalType(Type::type type_id, int32_t byte_width, int32_t precision,
                       int32_t scale)
      : FixedSizeBinaryType(byte_width, type_id), precision_(precision), scale_(scale) {}

  /// Constructs concrete decimal types
  static Result<std::shared_ptr<DataType>> Make(Type::type type_id, int32_t precision,
                                                int32_t scale);

  int32_t precision() const { return precision_; }
  int32_t scale() const { return scale_; }

  /// \brief Returns the number of bytes needed for precision.
  ///
  /// precision must be >= 1
  static int32_t DecimalSize(int32_t precision);

 protected:
  std::string ComputeFingerprint() const override;

  int32_t precision_;
  int32_t scale_;
};

/// \brief Concrete type class for 128-bit decimal data
///
/// Arrow decimals are fixed-point decimal numbers encoded as a scaled
/// integer.  The precision is the number of significant digits that the
/// decimal type can represent; the scale is the number of digits after
/// the decimal point (note the scale can be negative).
///
/// As an example, `Decimal128Type(7, 3)` can exactly represent the numbers
/// 1234.567 and -1234.567 (encoded internally as the 128-bit integers
/// 1234567 and -1234567, respectively), but neither 12345.67 nor 123.4567.
///
/// Decimal128Type has a maximum precision of 38 significant digits
/// (also available as Decimal128Type::kMaxPrecision).
/// If higher precision is needed, consider using Decimal256Type.
class ARROW_EXPORT Decimal128Type : public DecimalType {
 public:
  static constexpr Type::type type_id = Type::DECIMAL128;

  static constexpr const char* type_name() { return "decimal128"; }

  /// Decimal128Type constructor that aborts on invalid input.
  explicit Decimal128Type(int32_t precision, int32_t scale);

  /// Decimal128Type constructor that returns an error on invalid input.
  static Result<std::shared_ptr<DataType>> Make(int32_t precision, int32_t scale);

  std::string ToString() const override;
  std::string name() const override { return "decimal128"; }

  static constexpr int32_t kMinPrecision = 1;
  static constexpr int32_t kMaxPrecision = 38;
  static constexpr int32_t kByteWidth = 16;
};

/// \brief Concrete type class for 256-bit decimal data
///
/// Arrow decimals are fixed-point decimal numbers encoded as a scaled
/// integer.  The precision is the number of significant digits that the
/// decimal type can represent; the scale is the number of digits after
/// the decimal point (note the scale can be negative).
///
/// Decimal256Type has a maximum precision of 76 significant digits.
/// (also available as Decimal256Type::kMaxPrecision).
///
/// For most use cases, the maximum precision offered by Decimal128Type
/// is sufficient, and it will result in a more compact and more efficient
/// encoding.
class ARROW_EXPORT Decimal256Type : public DecimalType {
 public:
  static constexpr Type::type type_id = Type::DECIMAL256;

  static constexpr const char* type_name() { return "decimal256"; }

  /// Decimal256Type constructor that aborts on invalid input.
  explicit Decimal256Type(int32_t precision, int32_t scale);

  /// Decimal256Type constructor that returns an error on invalid input.
  static Result<std::shared_ptr<DataType>> Make(int32_t precision, int32_t scale);

  std::string ToString() const override;
  std::string name() const override { return "decimal256"; }

  static constexpr int32_t kMinPrecision = 1;
  static constexpr int32_t kMaxPrecision = 76;
  static constexpr int32_t kByteWidth = 32;
};

/// @}

/// \addtogroup nested-datatypes
///
/// @{

/// \brief Base class for all variable-size list data types
class ARROW_EXPORT BaseListType : public NestedType {
 public:
  using NestedType::NestedType;
  const std::shared_ptr<Field>& value_field() const { return children_[0]; }

  std::shared_ptr<DataType> value_type() const { return children_[0]->type(); }
};

/// \brief Concrete type class for list data
///
/// List data is nested data where each value is a variable number of
/// child items.  Lists can be recursively nested, for example
/// list(list(int32)).
class ARROW_EXPORT ListType : public BaseListType {
 public:
  static constexpr Type::type type_id = Type::LIST;
  using offset_type = int32_t;

  static constexpr const char* type_name() { return "list"; }

  // List can contain any other logical value type
  explicit ListType(const std::shared_ptr<DataType>& value_type)
      : ListType(std::make_shared<Field>("item", value_type)) {}

  explicit ListType(const std::shared_ptr<Field>& value_field) : BaseListType(type_id) {
    children_ = {value_field};
  }

  DataTypeLayout layout() const override {
    return DataTypeLayout(
        {DataTypeLayout::Bitmap(), DataTypeLayout::FixedWidth(sizeof(offset_type))});
  }

  std::string ToString() const override;

  std::string name() const override { return "list"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// \brief Concrete type class for large list data
///
/// LargeListType is like ListType but with 64-bit rather than 32-bit offsets.
class ARROW_EXPORT LargeListType : public BaseListType {
 public:
  static constexpr Type::type type_id = Type::LARGE_LIST;
  using offset_type = int64_t;

  static constexpr const char* type_name() { return "large_list"; }

  // List can contain any other logical value type
  explicit LargeListType(const std::shared_ptr<DataType>& value_type)
      : LargeListType(std::make_shared<Field>("item", value_type)) {}

  explicit LargeListType(const std::shared_ptr<Field>& value_field)
      : BaseListType(type_id) {
    children_ = {value_field};
  }

  DataTypeLayout layout() const override {
    return DataTypeLayout(
        {DataTypeLayout::Bitmap(), DataTypeLayout::FixedWidth(sizeof(offset_type))});
  }

  std::string ToString() const override;

  std::string name() const override { return "large_list"; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// \brief Concrete type class for map data
///
/// Map data is nested data where each value is a variable number of
/// key-item pairs.  Its physical representation is the same as
/// a list of `{key, item}` structs.
///
/// Maps can be recursively nested, for example map(utf8, map(utf8, int32)).
class ARROW_EXPORT MapType : public ListType {
 public:
  static constexpr Type::type type_id = Type::MAP;

  static constexpr const char* type_name() { return "map"; }

  MapType(std::shared_ptr<DataType> key_type, std::shared_ptr<DataType> item_type,
          bool keys_sorted = false);

  MapType(std::shared_ptr<DataType> key_type, std::shared_ptr<Field> item_field,
          bool keys_sorted = false);

  MapType(std::shared_ptr<Field> key_field, std::shared_ptr<Field> item_field,
          bool keys_sorted = false);

  explicit MapType(std::shared_ptr<Field> value_field, bool keys_sorted = false);

  // Validating constructor
  static Result<std::shared_ptr<DataType>> Make(std::shared_ptr<Field> value_field,
                                                bool keys_sorted = false);

  std::shared_ptr<Field> key_field() const { return value_type()->field(0); }
  std::shared_ptr<DataType> key_type() const { return key_field()->type(); }

  std::shared_ptr<Field> item_field() const { return value_type()->field(1); }
  std::shared_ptr<DataType> item_type() const { return item_field()->type(); }

  std::string ToString() const override;

  std::string name() const override { return "map"; }

  bool keys_sorted() const { return keys_sorted_; }

 private:
  std::string ComputeFingerprint() const override;

  bool keys_sorted_;
};

/// \brief Concrete type class for fixed size list data
class ARROW_EXPORT FixedSizeListType : public BaseListType {
 public:
  static constexpr Type::type type_id = Type::FIXED_SIZE_LIST;
  // While the individual item size is 32-bit, the overall data size
  // (item size * list length) may not fit in a 32-bit int.
  using offset_type = int64_t;

  static constexpr const char* type_name() { return "fixed_size_list"; }

  // List can contain any other logical value type
  FixedSizeListType(const std::shared_ptr<DataType>& value_type, int32_t list_size)
      : FixedSizeListType(std::make_shared<Field>("item", value_type), list_size) {}

  FixedSizeListType(const std::shared_ptr<Field>& value_field, int32_t list_size)
      : BaseListType(type_id), list_size_(list_size) {
    children_ = {value_field};
  }

  DataTypeLayout layout() const override {
    return DataTypeLayout({DataTypeLayout::Bitmap()});
  }

  std::string ToString() const override;

  std::string name() const override { return "fixed_size_list"; }

  int32_t list_size() const { return list_size_; }

 protected:
  std::string ComputeFingerprint() const override;

  int32_t list_size_;
};

/// \brief Concrete type class for struct data
class ARROW_EXPORT StructType : public NestedType {
 public:
  static constexpr Type::type type_id = Type::STRUCT;

  static constexpr const char* type_name() { return "struct"; }

  explicit StructType(const std::vector<std::shared_ptr<Field>>& fields);

  ~StructType() override;

  DataTypeLayout layout() const override {
    return DataTypeLayout({DataTypeLayout::Bitmap()});
  }

  std::string ToString() const override;
  std::string name() const override { return "struct"; }

  /// Returns null if name not found
  std::shared_ptr<Field> GetFieldByName(const std::string& name) const;

  /// Return all fields having this name
  std::vector<std::shared_ptr<Field>> GetAllFieldsByName(const std::string& name) const;

  /// Returns -1 if name not found or if there are multiple fields having the
  /// same name
  int GetFieldIndex(const std::string& name) const;

  /// \brief Return the indices of all fields having this name in sorted order
  std::vector<int> GetAllFieldIndices(const std::string& name) const;

  /// \brief Create a new StructType with field added at given index
  Result<std::shared_ptr<StructType>> AddField(int i,
                                               const std::shared_ptr<Field>& field) const;
  /// \brief Create a new StructType by removing the field at given index
  Result<std::shared_ptr<StructType>> RemoveField(int i) const;
  /// \brief Create a new StructType by changing the field at given index
  Result<std::shared_ptr<StructType>> SetField(int i,
                                               const std::shared_ptr<Field>& field) const;

 private:
  std::string ComputeFingerprint() const override;

  class Impl;
  std::unique_ptr<Impl> impl_;
};

/// \brief Base type class for union data
class ARROW_EXPORT UnionType : public NestedType {
 public:
  static constexpr int8_t kMaxTypeCode = 127;
  static constexpr int kInvalidChildId = -1;

  static Result<std::shared_ptr<DataType>> Make(
      const std::vector<std::shared_ptr<Field>>& fields,
      const std::vector<int8_t>& type_codes, UnionMode::type mode = UnionMode::SPARSE) {
    if (mode == UnionMode::SPARSE) {
      return sparse_union(fields, type_codes);
    } else {
      return dense_union(fields, type_codes);
    }
  }

  DataTypeLayout layout() const override;

  std::string ToString() const override;

  /// The array of logical type ids.
  ///
  /// For example, the first type in the union might be denoted by the id 5
  /// (instead of 0).
  const std::vector<int8_t>& type_codes() const { return type_codes_; }

  /// An array mapping logical type ids to physical child ids.
  const std::vector<int>& child_ids() const { return child_ids_; }

  uint8_t max_type_code() const;

  UnionMode::type mode() const;

 protected:
  UnionType(std::vector<std::shared_ptr<Field>> fields, std::vector<int8_t> type_codes,
            Type::type id);

  static Status ValidateParameters(const std::vector<std::shared_ptr<Field>>& fields,
                                   const std::vector<int8_t>& type_codes,
                                   UnionMode::type mode);

 private:
  std::string ComputeFingerprint() const override;

  std::vector<int8_t> type_codes_;
  std::vector<int> child_ids_;
};

/// \brief Concrete type class for sparse union data
///
/// A sparse union is a nested type where each logical value is taken from
/// a single child.  A buffer of 8-bit type ids indicates which child
/// a given logical value is to be taken from.
///
/// In a sparse union, each child array should have the same length as the
/// union array, regardless of the actual number of union values that
/// refer to it.
///
/// Note that, unlike most other types, unions don't have a top-level validity bitmap.
class ARROW_EXPORT SparseUnionType : public UnionType {
 public:
  static constexpr Type::type type_id = Type::SPARSE_UNION;

  static constexpr const char* type_name() { return "sparse_union"; }

  SparseUnionType(std::vector<std::shared_ptr<Field>> fields,
                  std::vector<int8_t> type_codes);

  // A constructor variant that validates input parameters
  static Result<std::shared_ptr<DataType>> Make(
      std::vector<std::shared_ptr<Field>> fields, std::vector<int8_t> type_codes);

  std::string name() const override { return "sparse_union"; }
};

/// \brief Concrete type class for dense union data
///
/// A dense union is a nested type where each logical value is taken from
/// a single child, at a specific offset.  A buffer of 8-bit type ids
/// indicates which child a given logical value is to be taken from,
/// and a buffer of 32-bit offsets indicates at which physical position
/// in the given child array the logical value is to be taken from.
///
/// Unlike a sparse union, a dense union allows encoding only the child array
/// values which are actually referred to by the union array.  This is
/// counterbalanced by the additional footprint of the offsets buffer, and
/// the additional indirection cost when looking up values.
///
/// Note that, unlike most other types, unions don't have a top-level validity bitmap.
class ARROW_EXPORT DenseUnionType : public UnionType {
 public:
  static constexpr Type::type type_id = Type::DENSE_UNION;

  static constexpr const char* type_name() { return "dense_union"; }

  DenseUnionType(std::vector<std::shared_ptr<Field>> fields,
                 std::vector<int8_t> type_codes);

  // A constructor variant that validates input parameters
  static Result<std::shared_ptr<DataType>> Make(
      std::vector<std::shared_ptr<Field>> fields, std::vector<int8_t> type_codes);

  std::string name() const override { return "dense_union"; }
};

/// @}

// ----------------------------------------------------------------------
// Date and time types

/// \addtogroup temporal-datatypes
///
/// @{

/// \brief Base type for all date and time types
class ARROW_EXPORT TemporalType : public FixedWidthType {
 public:
  using FixedWidthType::FixedWidthType;

  DataTypeLayout layout() const override {
    return DataTypeLayout(
        {DataTypeLayout::Bitmap(), DataTypeLayout::FixedWidth(bit_width() / 8)});
  }
};

/// \brief Base type class for date data
class ARROW_EXPORT DateType : public TemporalType {
 public:
  virtual DateUnit unit() const = 0;

 protected:
  explicit DateType(Type::type type_id);
};

/// Concrete type class for 32-bit date data (as number of days since UNIX epoch)
class ARROW_EXPORT Date32Type : public DateType {
 public:
  static constexpr Type::type type_id = Type::DATE32;
  static constexpr DateUnit UNIT = DateUnit::DAY;
  using c_type = int32_t;
  using PhysicalType = Int32Type;

  static constexpr const char* type_name() { return "date32"; }

  Date32Type();

  int bit_width() const override { return static_cast<int>(sizeof(c_type) * CHAR_BIT); }

  std::string ToString() const override;

  std::string name() const override { return "date32"; }
  DateUnit unit() const override { return UNIT; }

 protected:
  std::string ComputeFingerprint() const override;
};

/// Concrete type class for 64-bit date data (as number of milliseconds since UNIX epoch)
class ARROW_EXPORT Date64Type : public DateType {
 public:
  static constexpr Type::type type_id = Type::DATE64;
  static constexpr DateUnit UNIT = DateUnit::MILLI;
  using c_type = int64_t;
  using PhysicalType = Int64Type;

  static constexpr const char* type_name() { return "date64"; }

  Date64Type();

  int bit_width() const override { return static_cast<int>(sizeof(c_type) * CHAR_BIT); }

  std::string ToString() const override;

  std::string name() const override { return "date64"; }
  DateUnit unit() const override { return UNIT; }

 protected:
  std::string ComputeFingerprint() const override;
};

ARROW_EXPORT
std::ostream& operator<<(std::ostream& os, TimeUnit::type unit);

/// Base type class for time data
class ARROW_EXPORT TimeType : public TemporalType, public ParametricType {
 public:
  TimeUnit::type unit() const { return unit_; }

 protected:
  TimeType(Type::type type_id, TimeUnit::type unit);
  std::string ComputeFingerprint() const override;

  TimeUnit::type unit_;
};

/// Concrete type class for 32-bit time data (as number of seconds or milliseconds
/// since midnight)
class ARROW_EXPORT Time32Type : public TimeType {
 public:
  static constexpr Type::type type_id = Type::TIME32;
  using c_type = int32_t;
  using PhysicalType = Int32Type;

  static constexpr const char* type_name() { return "time32"; }

  int bit_width() const override { return static_cast<int>(sizeof(c_type) * CHAR_BIT); }

  explicit Time32Type(TimeUnit::type unit = TimeUnit::MILLI);

  std::string ToString() const override;

  std::string name() const override { return "time32"; }
};

/// Concrete type class for 64-bit time data (as number of microseconds or nanoseconds
/// since midnight)
class ARROW_EXPORT Time64Type : public TimeType {
 public:
  static constexpr Type::type type_id = Type::TIME64;
  using c_type = int64_t;
  using PhysicalType = Int64Type;

  static constexpr const char* type_name() { return "time64"; }

  int bit_width() const override { return static_cast<int>(sizeof(c_type) * CHAR_BIT); }

  explicit Time64Type(TimeUnit::type unit = TimeUnit::NANO);

  std::string ToString() const override;

  std::string name() const override { return "time64"; }
};

/// \brief Concrete type class for datetime data (as number of seconds, milliseconds,
/// microseconds or nanoseconds since UNIX epoch)
///
/// If supplied, the timezone string should take either the form (i) "Area/Location",
/// with values drawn from the names in the IANA Time Zone Database (such as
/// "Europe/Zurich"); or (ii) "(+|-)HH:MM" indicating an absolute offset from GMT
/// (such as "-08:00").  To indicate a native UTC timestamp, one of the strings "UTC",
/// "Etc/UTC" or "+00:00" should be used.
///
/// If any non-empty string is supplied as the timezone for a TimestampType, then the
/// Arrow field containing that timestamp type (and by extension the column associated
/// with such a field) is considered "timezone-aware".  The integer arrays that comprise
/// a timezone-aware column must contain UTC normalized datetime values, regardless of
/// the contents of their timezone string.  More precisely, (i) the producer of a
/// timezone-aware column must populate its constituent arrays with valid UTC values
/// (performing offset conversions from non-UTC values if necessary); and (ii) the
/// consumer of a timezone-aware column may assume that the column's values are directly
/// comparable (that is, with no offset adjustment required) to the values of any other
/// timezone-aware column or to any other valid UTC datetime value (provided all values
/// are expressed in the same units).
///
/// If a TimestampType is constructed without a timezone (or, equivalently, if the
/// timezone supplied is an empty string) then the resulting Arrow field (column) is
/// considered "timezone-naive".  The producer of a timezone-naive column may populate
/// its constituent integer arrays with datetime values from any timezone; the consumer
/// of a timezone-naive column should make no assumptions about the interoperability or
/// comparability of the values of such a column with those of any other timestamp
/// column or datetime value.
///
/// If a timezone-aware field contains a recognized timezone, its values may be
/// localized to that locale upon display; the values of timezone-naive fields must
/// always be displayed "as is", with no localization performed on them.
class ARROW_EXPORT TimestampType : public TemporalType, public ParametricType {
 public:
  using Unit = TimeUnit;

  static constexpr Type::type type_id = Type::TIMESTAMP;
  using c_type = int64_t;
  using PhysicalType = Int64Type;

  static constexpr const char* type_name() { return "timestamp"; }

  int bit_width() const override { return static_cast<int>(sizeof(int64_t) * CHAR_BIT); }

  explicit TimestampType(TimeUnit::type unit = TimeUnit::MILLI)
      : TemporalType(Type::TIMESTAMP), unit_(unit) {}

  explicit TimestampType(TimeUnit::type unit, const std::string& timezone)
      : TemporalType(Type::TIMESTAMP), unit_(unit), timezone_(timezone) {}

  std::string ToString() const override;
  std::string name() const override { return "timestamp"; }

  TimeUnit::type unit() const { return unit_; }
  const std::string& timezone() const { return timezone_; }

 protected:
  std::string ComputeFingerprint() const override;

 private:
  TimeUnit::type unit_;
  std::string timezone_;
};

// Base class for the different kinds of calendar intervals.
class ARROW_EXPORT IntervalType : public TemporalType, public ParametricType {
 public:
  enum type { MONTHS, DAY_TIME, MONTH_DAY_NANO };

  virtual type interval_type() const = 0;

 protected:
  explicit IntervalType(Type::type subtype) : TemporalType(subtype) {}
  std::string ComputeFingerprint() const override;
};

/// \brief Represents a number of months.
///
/// Type representing a number of months.  Corresponds to YearMonth type
/// in Schema.fbs (years are defined as 12 months).
class ARROW_EXPORT MonthIntervalType : public IntervalType {
 public:
  static constexpr Type::type type_id = Type::INTERVAL_MONTHS;
  using c_type = int32_t;
  using PhysicalType = Int32Type;

  static constexpr const char* type_name() { return "month_interval"; }

  IntervalType::type interval_type() const override { return IntervalType::MONTHS; }

  int bit_width() const override { return static_cast<int>(sizeof(c_type) * CHAR_BIT); }

  MonthIntervalType() : IntervalType(type_id) {}

  std::string ToString() const override { return name(); }
  std::string name() const override { return "month_interval"; }
};

/// \brief Represents a number of days and milliseconds (fraction of day).
class ARROW_EXPORT DayTimeIntervalType : public IntervalType {
 public:
  struct DayMilliseconds {
    int32_t days = 0;
    int32_t milliseconds = 0;
    constexpr DayMilliseconds() = default;
    constexpr DayMilliseconds(int32_t days, int32_t milliseconds)
        : days(days), milliseconds(milliseconds) {}
    bool operator==(DayMilliseconds other) const {
      return this->days == other.days && this->milliseconds == other.milliseconds;
    }
    bool operator!=(DayMilliseconds other) const { return !(*this == other); }
    bool operator<(DayMilliseconds other) const {
      return this->days < other.days || this->milliseconds < other.milliseconds;
    }
  };
  using c_type = DayMilliseconds;
  using PhysicalType = DayTimeIntervalType;

  static_assert(sizeof(DayMilliseconds) == 8,
                "DayMilliseconds struct assumed to be of size 8 bytes");
  static constexpr Type::type type_id = Type::INTERVAL_DAY_TIME;

  static constexpr const char* type_name() { return "day_time_interval"; }

  IntervalType::type interval_type() const override { return IntervalType::DAY_TIME; }

  DayTimeIntervalType() : IntervalType(type_id) {}

  int bit_width() const override { return static_cast<int>(sizeof(c_type) * CHAR_BIT); }

  std::string ToString() const override { return name(); }
  std::string name() const override { return "day_time_interval"; }
};

ARROW_EXPORT
std::ostream& operator<<(std::ostream& os, DayTimeIntervalType::DayMilliseconds interval);

/// \brief Represents a number of months, days and nanoseconds between
/// two dates.
///
/// All fields are independent from one another.
class ARROW_EXPORT MonthDayNanoIntervalType : public IntervalType {
 public:
  struct MonthDayNanos {
    int32_t months;
    int32_t days;
    int64_t nanoseconds;
    bool operator==(MonthDayNanos other) const {
      return this->months == other.months && this->days == other.days &&
             this->nanoseconds == other.nanoseconds;
    }
    bool operator!=(MonthDayNanos other) const { return !(*this == other); }
  };
  using c_type = MonthDayNanos;
  using PhysicalType = MonthDayNanoIntervalType;

  static_assert(sizeof(MonthDayNanos) == 16,
                "MonthDayNanos struct assumed to be of size 16 bytes");
  static constexpr Type::type type_id = Type::INTERVAL_MONTH_DAY_NANO;

  static constexpr const char* type_name() { return "month_day_nano_interval"; }

  IntervalType::type interval_type() const override {
    return IntervalType::MONTH_DAY_NANO;
  }

  MonthDayNanoIntervalType() : IntervalType(type_id) {}

  int bit_width() const override { return static_cast<int>(sizeof(c_type) * CHAR_BIT); }

  std::string ToString() const override { return name(); }
  std::string name() const override { return "month_day_nano_interval"; }
};

ARROW_EXPORT
std::ostream& operator<<(std::ostream& os,
                         MonthDayNanoIntervalType::MonthDayNanos interval);

/// \brief Represents an elapsed time without any relation to a calendar artifact.
class ARROW_EXPORT DurationType : public TemporalType, public ParametricType {
 public:
  using Unit = TimeUnit;

  static constexpr Type::type type_id = Type::DURATION;
  using c_type = int64_t;
  using PhysicalType = Int64Type;

  static constexpr const char* type_name() { return "duration"; }

  int bit_width() const override { return static_cast<int>(sizeof(int64_t) * CHAR_BIT); }

  explicit DurationType(TimeUnit::type unit = TimeUnit::MILLI)
      : TemporalType(Type::DURATION), unit_(unit) {}

  std::string ToString() const override;
  std::string name() const override { return "duration"; }

  TimeUnit::type unit() const { return unit_; }

 protected:
  std::string ComputeFingerprint() const override;

 private:
  TimeUnit::type unit_;
};

/// @}

// ----------------------------------------------------------------------
// Dictionary type (for representing categorical or dictionary-encoded
// in memory)

/// \brief Dictionary-encoded value type with data-dependent
/// dictionary. Indices are represented by any integer types.
class ARROW_EXPORT DictionaryType : public FixedWidthType {
 public:
  static constexpr Type::type type_id = Type::DICTIONARY;

  static constexpr const char* type_name() { return "dictionary"; }

  DictionaryType(const std::shared_ptr<DataType>& index_type,
                 const std::shared_ptr<DataType>& value_type, bool ordered = false);

  // A constructor variant that validates its input parameters
  static Result<std::shared_ptr<DataType>> Make(
      const std::shared_ptr<DataType>& index_type,
      const std::shared_ptr<DataType>& value_type, bool ordered = false);

  std::string ToString() const override;
  std::string name() const override { return "dictionary"; }

  int bit_width() const override;

  DataTypeLayout layout() const override;

  const std::shared_ptr<DataType>& index_type() const { return index_type_; }
  const std::shared_ptr<DataType>& value_type() const { return value_type_; }

  bool ordered() const { return ordered_; }

 protected:
  static Status ValidateParameters(const DataType& index_type,
                                   const DataType& value_type);

  std::string ComputeFingerprint() const override;

  // Must be an integer type (not currently checked)
  std::shared_ptr<DataType> index_type_;
  std::shared_ptr<DataType> value_type_;
  bool ordered_;
};

// ----------------------------------------------------------------------
// FieldRef

/// \class FieldPath
///
/// Represents a path to a nested field using indices of child fields.
/// For example, given indices {5, 9, 3} the field would be retrieved with
/// schema->field(5)->type()->field(9)->type()->field(3)
///
/// Attempting to retrieve a child field using a FieldPath which is not valid for
/// a given schema will raise an error. Invalid FieldPaths include:
/// - an index is out of range
/// - the path is empty (note: a default constructed FieldPath will be empty)
///
/// FieldPaths provide a number of accessors for drilling down to potentially nested
/// children. They are overloaded for convenience to support Schema (returns a field),
/// DataType (returns a child field), Field (returns a child field of this field's type)
/// Array (returns a child array), RecordBatch (returns a column).
class ARROW_EXPORT FieldPath {
 public:
  FieldPath() = default;

  FieldPath(std::vector<int> indices)  // NOLINT runtime/explicit
      : indices_(std::move(indices)) {}

  FieldPath(std::initializer_list<int> indices)  // NOLINT runtime/explicit
      : indices_(std::move(indices)) {}

  std::string ToString() const;

  size_t hash() const;
  struct Hash {
    size_t operator()(const FieldPath& path) const { return path.hash(); }
  };

  bool empty() const { return indices_.empty(); }
  bool operator==(const FieldPath& other) const { return indices() == other.indices(); }
  bool operator!=(const FieldPath& other) const { return indices() != other.indices(); }

  const std::vector<int>& indices() const { return indices_; }
  int operator[](size_t i) const { return indices_[i]; }
  std::vector<int>::const_iterator begin() const { return indices_.begin(); }
  std::vector<int>::const_iterator end() const { return indices_.end(); }

  /// \brief Retrieve the referenced child Field from a Schema, Field, or DataType
  Result<std::shared_ptr<Field>> Get(const Schema& schema) const;
  Result<std::shared_ptr<Field>> Get(const Field& field) const;
  Result<std::shared_ptr<Field>> Get(const DataType& type) const;
  Result<std::shared_ptr<Field>> Get(const FieldVector& fields) const;

  static Result<std::shared_ptr<Schema>> GetAll(const Schema& schema,
                                                const std::vector<FieldPath>& paths);

  /// \brief Retrieve the referenced column from a RecordBatch or Table
  Result<std::shared_ptr<Array>> Get(const RecordBatch& batch) const;

  /// \brief Retrieve the referenced child from an Array or ArrayData
  Result<std::shared_ptr<Array>> Get(const Array& array) const;
  Result<std::shared_ptr<ArrayData>> Get(const ArrayData& data) const;

 private:
  std::vector<int> indices_;
};

/// \class FieldRef
/// \brief Descriptor of a (potentially nested) field within a schema.
///
/// Unlike FieldPath (which exclusively uses indices of child fields), FieldRef may
/// reference a field by name. It is intended to replace parameters like `int field_index`
/// and `const std::string& field_name`; it can be implicitly constructed from either a
/// field index or a name.
///
/// Nested fields can be referenced as well. Given
///     schema({field("a", struct_({field("n", null())})), field("b", int32())})
///
/// the following all indicate the nested field named "n":
///     FieldRef ref1(0, 0);
///     FieldRef ref2("a", 0);
///     FieldRef ref3("a", "n");
///     FieldRef ref4(0, "n");
///     ARROW_ASSIGN_OR_RAISE(FieldRef ref5,
///                           FieldRef::FromDotPath(".a[0]"));
///
/// FieldPaths matching a FieldRef are retrieved using the member function FindAll.
/// Multiple matches are possible because field names may be duplicated within a schema.
/// For example:
///     Schema a_is_ambiguous({field("a", int32()), field("a", float32())});
///     auto matches = FieldRef("a").FindAll(a_is_ambiguous);
///     assert(matches.size() == 2);
///     assert(matches[0].Get(a_is_ambiguous)->Equals(a_is_ambiguous.field(0)));
///     assert(matches[1].Get(a_is_ambiguous)->Equals(a_is_ambiguous.field(1)));
///
/// Convenience accessors are available which raise a helpful error if the field is not
/// found or ambiguous, and for immediately calling FieldPath::Get to retrieve any
/// matching children:
///     auto maybe_match = FieldRef("struct", "field_i32").FindOneOrNone(schema);
///     auto maybe_column = FieldRef("struct", "field_i32").GetOne(some_table);
class ARROW_EXPORT FieldRef : public util::EqualityComparable<FieldRef> {
 public:
  FieldRef() = default;

  /// Construct a FieldRef using a string of indices. The reference will be retrieved as:
  /// schema.fields[self.indices[0]].type.fields[self.indices[1]] ...
  ///
  /// Empty indices are not valid.
  FieldRef(FieldPath indices);  // NOLINT runtime/explicit

  /// Construct a by-name FieldRef. Multiple fields may match a by-name FieldRef:
  /// [f for f in schema.fields where f.name == self.name]
  FieldRef(std::string name) : impl_(std::move(name)) {}    // NOLINT runtime/explicit
  FieldRef(const char* name) : impl_(std::string(name)) {}  // NOLINT runtime/explicit

  /// Equivalent to a single index string of indices.
  FieldRef(int index) : impl_(FieldPath({index})) {}  // NOLINT runtime/explicit

  /// Construct a nested FieldRef.
  FieldRef(std::vector<FieldRef> refs) {  // NOLINT runtime/explicit
    Flatten(std::move(refs));
  }

  /// Convenience constructor for nested FieldRefs: each argument will be used to
  /// construct a FieldRef
  template <typename A0, typename A1, typename... A>
  FieldRef(A0&& a0, A1&& a1, A&&... a) {
    Flatten({// cpplint thinks the following are constructor decls
             FieldRef(std::forward<A0>(a0)),     // NOLINT runtime/explicit
             FieldRef(std::forward<A1>(a1)),     // NOLINT runtime/explicit
             FieldRef(std::forward<A>(a))...});  // NOLINT runtime/explicit
  }

  /// Parse a dot path into a FieldRef.
  ///
  /// dot_path = '.' name
  ///          | '[' digit+ ']'
  ///          | dot_path+
  ///
  /// Examples:
  ///   ".alpha" => FieldRef("alpha")
  ///   "[2]" => FieldRef(2)
  ///   ".beta[3]" => FieldRef("beta", 3)
  ///   "[5].gamma.delta[7]" => FieldRef(5, "gamma", "delta", 7)
  ///   ".hello world" => FieldRef("hello world")
  ///   R"(.\[y\]\\tho\.\)" => FieldRef(R"([y]\tho.\)")
  ///
  /// Note: When parsing a name, a '\' preceding any other character will be dropped from
  /// the resulting name. Therefore if a name must contain the characters '.', '\', or '['
  /// those must be escaped with a preceding '\'.
  static Result<FieldRef> FromDotPath(const std::string& dot_path);
  std::string ToDotPath() const;

  bool Equals(const FieldRef& other) const { return impl_ == other.impl_; }

  std::string ToString() const;

  size_t hash() const;
  struct Hash {
    size_t operator()(const FieldRef& ref) const { return ref.hash(); }
  };

  explicit operator bool() const { return Equals(FieldPath{}); }
  bool operator!() const { return !Equals(FieldPath{}); }

  bool IsFieldPath() const { return std::holds_alternative<FieldPath>(impl_); }
  bool IsName() const { return std::holds_alternative<std::string>(impl_); }
  bool IsNested() const {
    if (IsName()) return false;
    if (IsFieldPath()) return std::get<FieldPath>(impl_).indices().size() > 1;
    return true;
  }

  const FieldPath* field_path() const {
    return IsFieldPath() ? &std::get<FieldPath>(impl_) : NULLPTR;
  }
  const std::string* name() const {
    return IsName() ? &std::get<std::string>(impl_) : NULLPTR;
  }
  const std::vector<FieldRef>* nested_refs() const {
    return std::holds_alternative<std::vector<FieldRef>>(impl_)
               ? &std::get<std::vector<FieldRef>>(impl_)
               : NULLPTR;
  }

  /// \brief Retrieve FieldPath of every child field which matches this FieldRef.
  std::vector<FieldPath> FindAll(const Schema& schema) const;
  std::vector<FieldPath> FindAll(const Field& field) const;
  std::vector<FieldPath> FindAll(const DataType& type) const;
  std::vector<FieldPath> FindAll(const FieldVector& fields) const;

  /// \brief Convenience function which applies FindAll to arg's type or schema.
  std::vector<FieldPath> FindAll(const ArrayData& array) const;
  std::vector<FieldPath> FindAll(const Array& array) const;
  std::vector<FieldPath> FindAll(const RecordBatch& batch) const;

  /// \brief Convenience function: raise an error if matches is empty.
  template <typename T>
  Status CheckNonEmpty(const std::vector<FieldPath>& matches, const T& root) const {
    if (matches.empty()) {
      return Status::Invalid("No match for ", ToString(), " in ", root.ToString());
    }
    return Status::OK();
  }

  /// \brief Convenience function: raise an error if matches contains multiple FieldPaths.
  template <typename T>
  Status CheckNonMultiple(const std::vector<FieldPath>& matches, const T& root) const {
    if (matches.size() > 1) {
      return Status::Invalid("Multiple matches for ", ToString(), " in ",
                             root.ToString());
    }
    return Status::OK();
  }

  /// \brief Retrieve FieldPath of a single child field which matches this
  /// FieldRef. Emit an error if none or multiple match.
  template <typename T>
  Result<FieldPath> FindOne(const T& root) const {
    auto matches = FindAll(root);
    ARROW_RETURN_NOT_OK(CheckNonEmpty(matches, root));
    ARROW_RETURN_NOT_OK(CheckNonMultiple(matches, root));
    return std::move(matches[0]);
  }

  /// \brief Retrieve FieldPath of a single child field which matches this
  /// FieldRef. Emit an error if multiple match. An empty (invalid) FieldPath
  /// will be returned if none match.
  template <typename T>
  Result<FieldPath> FindOneOrNone(const T& root) const {
    auto matches = FindAll(root);
    ARROW_RETURN_NOT_OK(CheckNonMultiple(matches, root));
    if (matches.empty()) {
      return FieldPath();
    }
    return std::move(matches[0]);
  }

  template <typename T>
  using GetType = decltype(std::declval<FieldPath>().Get(std::declval<T>()).ValueOrDie());

  /// \brief Get all children matching this FieldRef.
  template <typename T>
  std::vector<GetType<T>> GetAll(const T& root) const {
    std::vector<GetType<T>> out;
    for (const auto& match : FindAll(root)) {
      out.push_back(match.Get(root).ValueOrDie());
    }
    return out;
  }

  /// \brief Get the single child matching this FieldRef.
  /// Emit an error if none or multiple match.
  template <typename T>
  Result<GetType<T>> GetOne(const T& root) const {
    ARROW_ASSIGN_OR_RAISE(auto match, FindOne(root));
    return match.Get(root).ValueOrDie();
  }

  /// \brief Get the single child matching this FieldRef.
  /// Return nullptr if none match, emit an error if multiple match.
  template <typename T>
  Result<GetType<T>> GetOneOrNone(const T& root) const {
    ARROW_ASSIGN_OR_RAISE(auto match, FindOneOrNone(root));
    if (match.empty()) {
      return static_cast<GetType<T>>(NULLPTR);
    }
    return match.Get(root).ValueOrDie();
  }

 private:
  void Flatten(std::vector<FieldRef> children);

  std::variant<FieldPath, std::string, std::vector<FieldRef>> impl_;
};

ARROW_EXPORT void PrintTo(const FieldRef& ref, std::ostream* os);

ARROW_EXPORT
std::ostream& operator<<(std::ostream& os, const FieldRef&);

// ----------------------------------------------------------------------
// Schema

enum class Endianness {
  Little = 0,
  Big = 1,
#if ARROW_LITTLE_ENDIAN
  Native = Little
#else
  Native = Big
#endif
};

/// \class Schema
/// \brief Sequence of arrow::Field objects describing the columns of a record
/// batch or table data structure
class ARROW_EXPORT Schema : public detail::Fingerprintable,
                            public util::EqualityComparable<Schema>,
                            public util::ToStringOstreamable<Schema> {
 public:
  explicit Schema(FieldVector fields, Endianness endianness,
                  std::shared_ptr<const KeyValueMetadata> metadata = NULLPTR);

  explicit Schema(FieldVector fields,
                  std::shared_ptr<const KeyValueMetadata> metadata = NULLPTR);

  Schema(const Schema&);

  ~Schema() override;

  /// Returns true if all of the schema fields are equal
  bool Equals(const Schema& other, bool check_metadata = false) const;
  bool Equals(const std::shared_ptr<Schema>& other, bool check_metadata = false) const;

  /// \brief Set endianness in the schema
  ///
  /// \return new Schema
  std::shared_ptr<Schema> WithEndianness(Endianness endianness) const;

  /// \brief Return endianness in the schema
  Endianness endianness() const;

  /// \brief Indicate if endianness is equal to platform-native endianness
  bool is_native_endian() const;

  /// \brief Return the number of fields (columns) in the schema
  int num_fields() const;

  /// Return the ith schema element. Does not boundscheck
  const std::shared_ptr<Field>& field(int i) const;

  const FieldVector& fields() const;

  std::vector<std::string> field_names() const;

  /// Returns null if name not found
  std::shared_ptr<Field> GetFieldByName(const std::string& name) const;

  /// \brief Return the indices of all fields having this name in sorted order
  FieldVector GetAllFieldsByName(const std::string& name) const;

  /// Returns -1 if name not found
  int GetFieldIndex(const std::string& name) const;

  /// Return the indices of all fields having this name
  std::vector<int> GetAllFieldIndices(const std::string& name) const;

  /// Indicate if fields named `names` can be found unambiguously in the schema.
  Status CanReferenceFieldsByNames(const std::vector<std::string>& names) const;

  /// \brief The custom key-value metadata, if any
  ///
  /// \return metadata may be null
  const std::shared_ptr<const KeyValueMetadata>& metadata() const;

  /// \brief Render a string representation of the schema suitable for debugging
  /// \param[in] show_metadata when true, if KeyValueMetadata is non-empty,
  /// print keys and values in the output
  std::string ToString(bool show_metadata = false) const;

  Result<std::shared_ptr<Schema>> AddField(int i,
                                           const std::shared_ptr<Field>& field) const;
  Result<std::shared_ptr<Schema>> RemoveField(int i) const;
  Result<std::shared_ptr<Schema>> SetField(int i,
                                           const std::shared_ptr<Field>& field) const;

  /// \brief Replace key-value metadata with new metadata
  ///
  /// \param[in] metadata new KeyValueMetadata
  /// \return new Schema
  std::shared_ptr<Schema> WithMetadata(
      const std::shared_ptr<const KeyValueMetadata>& metadata) const;

  /// \brief Return copy of Schema without the KeyValueMetadata
  std::shared_ptr<Schema> RemoveMetadata() const;

  /// \brief Indicate that the Schema has non-empty KevValueMetadata
  bool HasMetadata() const;

  /// \brief Indicate that the Schema has distinct field names.
  bool HasDistinctFieldNames() const;

 protected:
  std::string ComputeFingerprint() const override;
  std::string ComputeMetadataFingerprint() const override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

ARROW_EXPORT void PrintTo(const Schema& s, std::ostream* os);

ARROW_EXPORT
std::string EndiannessToString(Endianness endianness);

// ----------------------------------------------------------------------

/// \brief Convenience class to incrementally construct/merge schemas.
///
/// This class amortizes the cost of validating field name conflicts by
/// maintaining the mapping. The caller also controls the conflict resolution
/// scheme.
class ARROW_EXPORT SchemaBuilder {
 public:
  // Indicate how field conflict(s) should be resolved when building a schema. A
  // conflict arise when a field is added to the builder and one or more field(s)
  // with the same name already exists.
  enum ConflictPolicy {
    // Ignore the conflict and append the field. This is the default behavior of the
    // Schema constructor and the `arrow::schema` factory function.
    CONFLICT_APPEND = 0,
    // Keep the existing field and ignore the newer one.
    CONFLICT_IGNORE,
    // Replace the existing field with the newer one.
    CONFLICT_REPLACE,
    // Merge the fields. The merging behavior can be controlled by `Field::MergeOptions`
    // specified at construction time. Also see documentation of `Field::MergeWith`.
    CONFLICT_MERGE,
    // Refuse the new field and error out.
    CONFLICT_ERROR
  };

  /// \brief Construct an empty SchemaBuilder
  /// `field_merge_options` is only effective when `conflict_policy` == `CONFLICT_MERGE`.
  SchemaBuilder(
      ConflictPolicy conflict_policy = CONFLICT_APPEND,
      Field::MergeOptions field_merge_options = Field::MergeOptions::Defaults());
  /// \brief Construct a SchemaBuilder from a list of fields
  /// `field_merge_options` is only effective when `conflict_policy` == `CONFLICT_MERGE`.
  SchemaBuilder(
      std::vector<std::shared_ptr<Field>> fields,
      ConflictPolicy conflict_policy = CONFLICT_APPEND,
      Field::MergeOptions field_merge_options = Field::MergeOptions::Defaults());
  /// \brief Construct a SchemaBuilder from a schema, preserving the metadata
  /// `field_merge_options` is only effective when `conflict_policy` == `CONFLICT_MERGE`.
  SchemaBuilder(
      const std::shared_ptr<Schema>& schema,
      ConflictPolicy conflict_policy = CONFLICT_APPEND,
      Field::MergeOptions field_merge_options = Field::MergeOptions::Defaults());

  /// \brief Return the conflict resolution method.
  ConflictPolicy policy() const;

  /// \brief Set the conflict resolution method.
  void SetPolicy(ConflictPolicy resolution);

  /// \brief Add a field to the constructed schema.
  ///
  /// \param[in] field to add to the constructed Schema.
  /// \return A failure if encountered.
  Status AddField(const std::shared_ptr<Field>& field);

  /// \brief Add multiple fields to the constructed schema.
  ///
  /// \param[in] fields to add to the constructed Schema.
  /// \return The first failure encountered, if any.
  Status AddFields(const std::vector<std::shared_ptr<Field>>& fields);

  /// \brief Add fields of a Schema to the constructed Schema.
  ///
  /// \param[in] schema to take fields to add to the constructed Schema.
  /// \return The first failure encountered, if any.
  Status AddSchema(const std::shared_ptr<Schema>& schema);

  /// \brief Add fields of multiple Schemas to the constructed Schema.
  ///
  /// \param[in] schemas to take fields to add to the constructed Schema.
  /// \return The first failure encountered, if any.
  Status AddSchemas(const std::vector<std::shared_ptr<Schema>>& schemas);

  Status AddMetadata(const KeyValueMetadata& metadata);

  /// \brief Return the constructed Schema.
  ///
  /// The builder internal state is not affected by invoking this method, i.e.
  /// a single builder can yield multiple incrementally constructed schemas.
  ///
  /// \return the constructed schema.
  Result<std::shared_ptr<Schema>> Finish() const;

  /// \brief Merge schemas in a unified schema according to policy.
  static Result<std::shared_ptr<Schema>> Merge(
      const std::vector<std::shared_ptr<Schema>>& schemas,
      ConflictPolicy policy = CONFLICT_MERGE);

  /// \brief Indicate if schemas are compatible to merge according to policy.
  static Status AreCompatible(const std::vector<std::shared_ptr<Schema>>& schemas,
                              ConflictPolicy policy = CONFLICT_MERGE);

  /// \brief Reset internal state with an empty schema (and metadata).
  void Reset();

  ~SchemaBuilder();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;

  Status AppendField(const std::shared_ptr<Field>& field);
};

/// \brief Unifies schemas by merging fields by name.
///
/// The behavior of field merging can be controlled via `Field::MergeOptions`.
///
/// The resulting schema will contain the union of fields from all schemas.
/// Fields with the same name will be merged. See `Field::MergeOptions`.
/// - They are expected to be mergeable under provided `field_merge_options`.
/// - The unified field will inherit the metadata from the schema where
///   that field is first defined.
/// - The first N fields in the schema will be ordered the same as the
///   N fields in the first schema.
/// The resulting schema will inherit its metadata from the first input schema.
/// Returns an error if:
/// - Any input schema contains fields with duplicate names.
/// - Fields of the same name are not mergeable.
ARROW_EXPORT
Result<std::shared_ptr<Schema>> UnifySchemas(
    const std::vector<std::shared_ptr<Schema>>& schemas,
    Field::MergeOptions field_merge_options = Field::MergeOptions::Defaults());

namespace internal {

static inline bool HasValidityBitmap(Type::type id) {
  switch (id) {
    case Type::NA:
    case Type::DENSE_UNION:
    case Type::SPARSE_UNION:
      return false;
    default:
      return true;
  }
}

ARROW_EXPORT
std::string ToString(Type::type id);

ARROW_EXPORT
std::string ToTypeName(Type::type id);

ARROW_EXPORT
std::string ToString(TimeUnit::type unit);

}  // namespace internal

// Helpers to get instances of data types based on general categories

/// \brief Signed integer types
ARROW_EXPORT
const std::vector<std::shared_ptr<DataType>>& SignedIntTypes();
/// \brief Unsigned integer types
ARROW_EXPORT
const std::vector<std::shared_ptr<DataType>>& UnsignedIntTypes();
/// \brief Signed and unsigned integer types
ARROW_EXPORT
const std::vector<std::shared_ptr<DataType>>& IntTypes();
/// \brief Floating point types
ARROW_EXPORT
const std::vector<std::shared_ptr<DataType>>& FloatingPointTypes();
/// \brief Number types without boolean - integer and floating point types
ARROW_EXPORT
const std::vector<std::shared_ptr<DataType>>& NumericTypes();
/// \brief Binary and string-like types (except fixed-size binary)
ARROW_EXPORT
const std::vector<std::shared_ptr<DataType>>& BaseBinaryTypes();
/// \brief Binary and large-binary types
ARROW_EXPORT
const std::vector<std::shared_ptr<DataType>>& BinaryTypes();
/// \brief String and large-string types
ARROW_EXPORT
const std::vector<std::shared_ptr<DataType>>& StringTypes();
/// \brief Temporal types including date, time and timestamps for each unit
ARROW_EXPORT
const std::vector<std::shared_ptr<DataType>>& TemporalTypes();
/// \brief Interval types
ARROW_EXPORT
const std::vector<std::shared_ptr<DataType>>& IntervalTypes();
/// \brief Duration types for each unit
ARROW_EXPORT
const std::vector<std::shared_ptr<DataType>>& DurationTypes();
/// \brief Numeric, base binary, date, boolean and null types
ARROW_EXPORT
const std::vector<std::shared_ptr<DataType>>& PrimitiveTypes();

}  // namespace arrow

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

#include "arrow/status.h"
#include "arrow/type_fwd.h"
#include "arrow/util/visibility.h"

namespace arrow {

/// \brief Abstract array visitor class
///
/// Subclass this to create a visitor that can be used with the Array::Accept()
/// method.
class ARROW_EXPORT ArrayVisitor {
 public:
  virtual ~ArrayVisitor() = default;

  virtual Status Visit(const NullArray& array);
  virtual Status Visit(const BooleanArray& array);
  virtual Status Visit(const Int8Array& array);
  virtual Status Visit(const Int16Array& array);
  virtual Status Visit(const Int32Array& array);
  virtual Status Visit(const Int64Array& array);
  virtual Status Visit(const UInt8Array& array);
  virtual Status Visit(const UInt16Array& array);
  virtual Status Visit(const UInt32Array& array);
  virtual Status Visit(const UInt64Array& array);
  virtual Status Visit(const HalfFloatArray& array);
  virtual Status Visit(const FloatArray& array);
  virtual Status Visit(const DoubleArray& array);
  virtual Status Visit(const StringArray& array);
  virtual Status Visit(const BinaryArray& array);
  virtual Status Visit(const LargeStringArray& array);
  virtual Status Visit(const LargeBinaryArray& array);
  virtual Status Visit(const FixedSizeBinaryArray& array);
  virtual Status Visit(const Date32Array& array);
  virtual Status Visit(const Date64Array& array);
  virtual Status Visit(const Time32Array& array);
  virtual Status Visit(const Time64Array& array);
  virtual Status Visit(const TimestampArray& array);
  virtual Status Visit(const DayTimeIntervalArray& array);
  virtual Status Visit(const MonthDayNanoIntervalArray& array);
  virtual Status Visit(const MonthIntervalArray& array);
  virtual Status Visit(const DurationArray& array);
  virtual Status Visit(const Decimal128Array& array);
  virtual Status Visit(const Decimal256Array& array);
  virtual Status Visit(const ListArray& array);
  virtual Status Visit(const LargeListArray& array);
  virtual Status Visit(const MapArray& array);
  virtual Status Visit(const FixedSizeListArray& array);
  virtual Status Visit(const StructArray& array);
  virtual Status Visit(const SparseUnionArray& array);
  virtual Status Visit(const DenseUnionArray& array);
  virtual Status Visit(const DictionaryArray& array);
  virtual Status Visit(const ExtensionArray& array);
};

/// \brief Abstract type visitor class
///
/// Subclass this to create a visitor that can be used with the DataType::Accept()
/// method.
class ARROW_EXPORT TypeVisitor {
 public:
  virtual ~TypeVisitor() = default;

  virtual Status Visit(const NullType& type);
  virtual Status Visit(const BooleanType& type);
  virtual Status Visit(const Int8Type& type);
  virtual Status Visit(const Int16Type& type);
  virtual Status Visit(const Int32Type& type);
  virtual Status Visit(const Int64Type& type);
  virtual Status Visit(const UInt8Type& type);
  virtual Status Visit(const UInt16Type& type);
  virtual Status Visit(const UInt32Type& type);
  virtual Status Visit(const UInt64Type& type);
  virtual Status Visit(const HalfFloatType& type);
  virtual Status Visit(const FloatType& type);
  virtual Status Visit(const DoubleType& type);
  virtual Status Visit(const StringType& type);
  virtual Status Visit(const BinaryType& type);
  virtual Status Visit(const LargeStringType& type);
  virtual Status Visit(const LargeBinaryType& type);
  virtual Status Visit(const FixedSizeBinaryType& type);
  virtual Status Visit(const Date64Type& type);
  virtual Status Visit(const Date32Type& type);
  virtual Status Visit(const Time32Type& type);
  virtual Status Visit(const Time64Type& type);
  virtual Status Visit(const TimestampType& type);
  virtual Status Visit(const MonthDayNanoIntervalType& type);
  virtual Status Visit(const MonthIntervalType& type);
  virtual Status Visit(const DayTimeIntervalType& type);
  virtual Status Visit(const DurationType& type);
  virtual Status Visit(const Decimal128Type& type);
  virtual Status Visit(const Decimal256Type& type);
  virtual Status Visit(const ListType& type);
  virtual Status Visit(const LargeListType& type);
  virtual Status Visit(const MapType& type);
  virtual Status Visit(const FixedSizeListType& type);
  virtual Status Visit(const StructType& type);
  virtual Status Visit(const SparseUnionType& type);
  virtual Status Visit(const DenseUnionType& type);
  virtual Status Visit(const DictionaryType& type);
  virtual Status Visit(const ExtensionType& type);
};

/// \brief Abstract scalar visitor class
///
/// Subclass this to create a visitor that can be used with the Scalar::Accept()
/// method.
class ARROW_EXPORT ScalarVisitor {
 public:
  virtual ~ScalarVisitor() = default;

  virtual Status Visit(const NullScalar& scalar);
  virtual Status Visit(const BooleanScalar& scalar);
  virtual Status Visit(const Int8Scalar& scalar);
  virtual Status Visit(const Int16Scalar& scalar);
  virtual Status Visit(const Int32Scalar& scalar);
  virtual Status Visit(const Int64Scalar& scalar);
  virtual Status Visit(const UInt8Scalar& scalar);
  virtual Status Visit(const UInt16Scalar& scalar);
  virtual Status Visit(const UInt32Scalar& scalar);
  virtual Status Visit(const UInt64Scalar& scalar);
  virtual Status Visit(const HalfFloatScalar& scalar);
  virtual Status Visit(const FloatScalar& scalar);
  virtual Status Visit(const DoubleScalar& scalar);
  virtual Status Visit(const StringScalar& scalar);
  virtual Status Visit(const BinaryScalar& scalar);
  virtual Status Visit(const LargeStringScalar& scalar);
  virtual Status Visit(const LargeBinaryScalar& scalar);
  virtual Status Visit(const FixedSizeBinaryScalar& scalar);
  virtual Status Visit(const Date64Scalar& scalar);
  virtual Status Visit(const Date32Scalar& scalar);
  virtual Status Visit(const Time32Scalar& scalar);
  virtual Status Visit(const Time64Scalar& scalar);
  virtual Status Visit(const TimestampScalar& scalar);
  virtual Status Visit(const DayTimeIntervalScalar& scalar);
  virtual Status Visit(const MonthDayNanoIntervalScalar& type);
  virtual Status Visit(const MonthIntervalScalar& scalar);
  virtual Status Visit(const DurationScalar& scalar);
  virtual Status Visit(const Decimal128Scalar& scalar);
  virtual Status Visit(const Decimal256Scalar& scalar);
  virtual Status Visit(const ListScalar& scalar);
  virtual Status Visit(const LargeListScalar& scalar);
  virtual Status Visit(const MapScalar& scalar);
  virtual Status Visit(const FixedSizeListScalar& scalar);
  virtual Status Visit(const StructScalar& scalar);
  virtual Status Visit(const DictionaryScalar& scalar);
  virtual Status Visit(const SparseUnionScalar& scalar);
  virtual Status Visit(const DenseUnionScalar& scalar);
  virtual Status Visit(const ExtensionScalar& scalar);
};

}  // namespace arrow

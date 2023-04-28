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

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>

#include "arrow/util/endian.h"
#include "arrow/util/macros.h"
#include "arrow/util/type_traits.h"
#include "arrow/util/visibility.h"

namespace arrow {

enum class DecimalStatus {
  kSuccess,
  kDivideByZero,
  kOverflow,
  kRescaleDataLoss,
};

template <typename Derived, int BIT_WIDTH, int NWORDS = BIT_WIDTH / 64>
class ARROW_EXPORT GenericBasicDecimal {
 protected:
  struct LittleEndianArrayTag {};

#if ARROW_LITTLE_ENDIAN
  static constexpr int kHighWordIndex = NWORDS - 1;
#else
  static constexpr int kHighWordIndex = 0;
#endif

 public:
  static constexpr int kBitWidth = BIT_WIDTH;
  static constexpr int kByteWidth = kBitWidth / 8;

  // A constructor tag to introduce a little-endian encoded array
  static constexpr LittleEndianArrayTag LittleEndianArray{};

  using WordArray = std::array<uint64_t, NWORDS>;

  /// \brief Empty constructor creates a decimal with a value of 0.
  constexpr GenericBasicDecimal() noexcept : array_({0}) {}

  /// \brief Create a decimal from the two's complement representation.
  ///
  /// Input array is assumed to be in native endianness.
  constexpr GenericBasicDecimal(
      const WordArray& array) noexcept  // NOLINT(runtime/explicit)
      : array_(array) {}

  /// \brief Create a decimal from the two's complement representation.
  ///
  /// Input array is assumed to be in little endianness, with native endian elements.
  GenericBasicDecimal(LittleEndianArrayTag, const WordArray& array) noexcept
      : GenericBasicDecimal(bit_util::little_endian::ToNative(array)) {}

  /// \brief Create a decimal from an array of bytes.
  ///
  /// Bytes are assumed to be in native-endian byte order.
  explicit GenericBasicDecimal(const uint8_t* bytes) {
    memcpy(array_.data(), bytes, sizeof(array_));
  }

  /// \brief Get the bits of the two's complement representation of the number.
  ///
  /// The elements are in native endian order. The bits within each uint64_t element
  /// are in native endian order. For example, on a little endian machine,
  /// BasicDecimal128(123).native_endian_array() = {123, 0};
  /// but on a big endian machine,
  /// BasicDecimal128(123).native_endian_array() = {0, 123};
  constexpr const WordArray& native_endian_array() const { return array_; }

  /// \brief Get the bits of the two's complement representation of the number.
  ///
  /// The elements are in little endian order. However, the bits within each
  /// uint64_t element are in native endian order.
  /// For example, BasicDecimal128(123).little_endian_array() = {123, 0};
  WordArray little_endian_array() const {
    return bit_util::little_endian::FromNative(array_);
  }

  const uint8_t* native_endian_bytes() const {
    return reinterpret_cast<const uint8_t*>(array_.data());
  }

  uint8_t* mutable_native_endian_bytes() {
    return reinterpret_cast<uint8_t*>(array_.data());
  }

  /// \brief Return the raw bytes of the value in native-endian byte order.
  std::array<uint8_t, kByteWidth> ToBytes() const {
    std::array<uint8_t, kByteWidth> out{{0}};
    memcpy(out.data(), array_.data(), kByteWidth);
    return out;
  }

  /// \brief Copy the raw bytes of the value in native-endian byte order.
  void ToBytes(uint8_t* out) const { memcpy(out, array_.data(), kByteWidth); }

  /// Return 1 if positive or zero, -1 if strictly negative.
  int64_t Sign() const {
    return 1 | (static_cast<int64_t>(array_[kHighWordIndex]) >> 63);
  }

  bool IsNegative() const { return static_cast<int64_t>(array_[kHighWordIndex]) < 0; }

 protected:
  WordArray array_;
};

/// Represents a signed 128-bit integer in two's complement.
///
/// This class is also compiled into LLVM IR - so, it should not have cpp references like
/// streams and boost.
class ARROW_EXPORT BasicDecimal128 : public GenericBasicDecimal<BasicDecimal128, 128> {
 public:
  static constexpr int kMaxPrecision = 38;
  static constexpr int kMaxScale = 38;

  using GenericBasicDecimal::GenericBasicDecimal;

  constexpr BasicDecimal128() noexcept : GenericBasicDecimal() {}

  /// \brief Create a BasicDecimal128 from the two's complement representation.
#if ARROW_LITTLE_ENDIAN
  constexpr BasicDecimal128(int64_t high, uint64_t low) noexcept
      : BasicDecimal128(WordArray{low, static_cast<uint64_t>(high)}) {}
#else
  constexpr BasicDecimal128(int64_t high, uint64_t low) noexcept
      : BasicDecimal128(WordArray{static_cast<uint64_t>(high), low}) {}
#endif

  /// \brief Convert any integer value into a BasicDecimal128.
  template <typename T,
            typename = typename std::enable_if<
                std::is_integral<T>::value && (sizeof(T) <= sizeof(uint64_t)), T>::type>
  constexpr BasicDecimal128(T value) noexcept  // NOLINT(runtime/explicit)
      : BasicDecimal128(value >= T{0} ? 0 : -1, static_cast<uint64_t>(value)) {  // NOLINT
  }

  /// \brief Negate the current value (in-place)
  BasicDecimal128& Negate();

  /// \brief Absolute value (in-place)
  BasicDecimal128& Abs();

  /// \brief Absolute value
  static BasicDecimal128 Abs(const BasicDecimal128& left);

  /// \brief Add a number to this one. The result is truncated to 128 bits.
  BasicDecimal128& operator+=(const BasicDecimal128& right);

  /// \brief Subtract a number from this one. The result is truncated to 128 bits.
  BasicDecimal128& operator-=(const BasicDecimal128& right);

  /// \brief Multiply this number by another number. The result is truncated to 128 bits.
  BasicDecimal128& operator*=(const BasicDecimal128& right);

  /// Divide this number by right and return the result.
  ///
  /// This operation is not destructive.
  /// The answer rounds to zero. Signs work like:
  ///   21 /  5 ->  4,  1
  ///  -21 /  5 -> -4, -1
  ///   21 / -5 -> -4,  1
  ///  -21 / -5 ->  4, -1
  /// \param[in] divisor the number to divide by
  /// \param[out] result the quotient
  /// \param[out] remainder the remainder after the division
  DecimalStatus Divide(const BasicDecimal128& divisor, BasicDecimal128* result,
                       BasicDecimal128* remainder) const;

  /// \brief In-place division.
  BasicDecimal128& operator/=(const BasicDecimal128& right);

  /// \brief Bitwise "or" between two BasicDecimal128.
  BasicDecimal128& operator|=(const BasicDecimal128& right);

  /// \brief Bitwise "and" between two BasicDecimal128.
  BasicDecimal128& operator&=(const BasicDecimal128& right);

  /// \brief Shift left by the given number of bits.
  BasicDecimal128& operator<<=(uint32_t bits);

  BasicDecimal128 operator<<(uint32_t bits) const {
    auto res = *this;
    res <<= bits;
    return res;
  }

  /// \brief Shift right by the given number of bits. Negative values will
  BasicDecimal128& operator>>=(uint32_t bits);

  BasicDecimal128 operator>>(uint32_t bits) const {
    auto res = *this;
    res >>= bits;
    return res;
  }

  /// \brief Get the high bits of the two's complement representation of the number.
  constexpr int64_t high_bits() const {
#if ARROW_LITTLE_ENDIAN
    return static_cast<int64_t>(array_[1]);
#else
    return static_cast<int64_t>(array_[0]);
#endif
  }

  /// \brief Get the low bits of the two's complement representation of the number.
  constexpr uint64_t low_bits() const {
#if ARROW_LITTLE_ENDIAN
    return array_[0];
#else
    return array_[1];
#endif
  }

  /// \brief separate the integer and fractional parts for the given scale.
  void GetWholeAndFraction(int32_t scale, BasicDecimal128* whole,
                           BasicDecimal128* fraction) const;

  /// \brief Scale multiplier for given scale value.
  static const BasicDecimal128& GetScaleMultiplier(int32_t scale);
  /// \brief Half-scale multiplier for given scale value.
  static const BasicDecimal128& GetHalfScaleMultiplier(int32_t scale);

  /// \brief Convert BasicDecimal128 from one scale to another
  DecimalStatus Rescale(int32_t original_scale, int32_t new_scale,
                        BasicDecimal128* out) const;

  /// \brief Scale up.
  BasicDecimal128 IncreaseScaleBy(int32_t increase_by) const;

  /// \brief Scale down.
  /// - If 'round' is true, the right-most digits are dropped and the result value is
  ///   rounded up (+1 for +ve, -1 for -ve) based on the value of the dropped digits
  ///   (>= 10^reduce_by / 2).
  /// - If 'round' is false, the right-most digits are simply dropped.
  BasicDecimal128 ReduceScaleBy(int32_t reduce_by, bool round = true) const;

  /// \brief Whether this number fits in the given precision
  ///
  /// Return true if the number of significant digits is less or equal to `precision`.
  bool FitsInPrecision(int32_t precision) const;

  /// \brief count the number of leading binary zeroes.
  int32_t CountLeadingBinaryZeros() const;

  /// \brief Get the maximum valid unscaled decimal value.
  static const BasicDecimal128& GetMaxValue();

  /// \brief Get the maximum valid unscaled decimal value for the given precision.
  static BasicDecimal128 GetMaxValue(int32_t precision);

  /// \brief Get the maximum decimal value (is not a valid value).
  static constexpr BasicDecimal128 GetMaxSentinel() {
    return BasicDecimal128(/*high=*/std::numeric_limits<int64_t>::max(),
                           /*low=*/std::numeric_limits<uint64_t>::max());
  }
  /// \brief Get the minimum decimal value (is not a valid value).
  static constexpr BasicDecimal128 GetMinSentinel() {
    return BasicDecimal128(/*high=*/std::numeric_limits<int64_t>::min(),
                           /*low=*/std::numeric_limits<uint64_t>::min());
  }
};

ARROW_EXPORT bool operator==(const BasicDecimal128& left, const BasicDecimal128& right);
ARROW_EXPORT bool operator!=(const BasicDecimal128& left, const BasicDecimal128& right);
ARROW_EXPORT bool operator<(const BasicDecimal128& left, const BasicDecimal128& right);
ARROW_EXPORT bool operator<=(const BasicDecimal128& left, const BasicDecimal128& right);
ARROW_EXPORT bool operator>(const BasicDecimal128& left, const BasicDecimal128& right);
ARROW_EXPORT bool operator>=(const BasicDecimal128& left, const BasicDecimal128& right);

ARROW_EXPORT BasicDecimal128 operator-(const BasicDecimal128& operand);
ARROW_EXPORT BasicDecimal128 operator~(const BasicDecimal128& operand);
ARROW_EXPORT BasicDecimal128 operator+(const BasicDecimal128& left,
                                       const BasicDecimal128& right);
ARROW_EXPORT BasicDecimal128 operator-(const BasicDecimal128& left,
                                       const BasicDecimal128& right);
ARROW_EXPORT BasicDecimal128 operator*(const BasicDecimal128& left,
                                       const BasicDecimal128& right);
ARROW_EXPORT BasicDecimal128 operator/(const BasicDecimal128& left,
                                       const BasicDecimal128& right);
ARROW_EXPORT BasicDecimal128 operator%(const BasicDecimal128& left,
                                       const BasicDecimal128& right);

class ARROW_EXPORT BasicDecimal256 : public GenericBasicDecimal<BasicDecimal256, 256> {
 private:
  // Due to a bug in clang, we have to declare the extend method prior to its
  // usage.
  template <typename T>
  static constexpr uint64_t extend(T low_bits) noexcept {
    return low_bits >= T() ? uint64_t{0} : ~uint64_t{0};
  }

 public:
  using GenericBasicDecimal::GenericBasicDecimal;

  static constexpr int kMaxPrecision = 76;
  static constexpr int kMaxScale = 76;

  constexpr BasicDecimal256() noexcept : GenericBasicDecimal() {}

  /// \brief Convert any integer value into a BasicDecimal256.
  template <typename T,
            typename = typename std::enable_if<
                std::is_integral<T>::value && (sizeof(T) <= sizeof(uint64_t)), T>::type>
  constexpr BasicDecimal256(T value) noexcept  // NOLINT(runtime/explicit)
      : BasicDecimal256(bit_util::little_endian::ToNative<uint64_t, 4>(
            {static_cast<uint64_t>(value), extend(value), extend(value),
             extend(value)})) {}

  explicit BasicDecimal256(const BasicDecimal128& value) noexcept
      : BasicDecimal256(bit_util::little_endian::ToNative<uint64_t, 4>(
            {value.low_bits(), static_cast<uint64_t>(value.high_bits()),
             extend(value.high_bits()), extend(value.high_bits())})) {}

  /// \brief Negate the current value (in-place)
  BasicDecimal256& Negate();

  /// \brief Absolute value (in-place)
  BasicDecimal256& Abs();

  /// \brief Absolute value
  static BasicDecimal256 Abs(const BasicDecimal256& left);

  /// \brief Add a number to this one. The result is truncated to 256 bits.
  BasicDecimal256& operator+=(const BasicDecimal256& right);

  /// \brief Subtract a number from this one. The result is truncated to 256 bits.
  BasicDecimal256& operator-=(const BasicDecimal256& right);

  /// \brief Get the lowest bits of the two's complement representation of the number.
  uint64_t low_bits() const { return bit_util::little_endian::Make(array_)[0]; }

  /// \brief Scale multiplier for given scale value.
  static const BasicDecimal256& GetScaleMultiplier(int32_t scale);
  /// \brief Half-scale multiplier for given scale value.
  static const BasicDecimal256& GetHalfScaleMultiplier(int32_t scale);

  /// \brief Convert BasicDecimal256 from one scale to another
  DecimalStatus Rescale(int32_t original_scale, int32_t new_scale,
                        BasicDecimal256* out) const;

  /// \brief Scale up.
  BasicDecimal256 IncreaseScaleBy(int32_t increase_by) const;

  /// \brief Scale down.
  /// - If 'round' is true, the right-most digits are dropped and the result value is
  ///   rounded up (+1 for positive, -1 for negative) based on the value of the
  ///   dropped digits (>= 10^reduce_by / 2).
  /// - If 'round' is false, the right-most digits are simply dropped.
  BasicDecimal256 ReduceScaleBy(int32_t reduce_by, bool round = true) const;

  /// \brief Whether this number fits in the given precision
  ///
  /// Return true if the number of significant digits is less or equal to `precision`.
  bool FitsInPrecision(int32_t precision) const;

  /// \brief Multiply this number by another number. The result is truncated to 256 bits.
  BasicDecimal256& operator*=(const BasicDecimal256& right);

  /// Divide this number by right and return the result.
  ///
  /// This operation is not destructive.
  /// The answer rounds to zero. Signs work like:
  ///   21 /  5 ->  4,  1
  ///  -21 /  5 -> -4, -1
  ///   21 / -5 -> -4,  1
  ///  -21 / -5 ->  4, -1
  /// \param[in] divisor the number to divide by
  /// \param[out] result the quotient
  /// \param[out] remainder the remainder after the division
  DecimalStatus Divide(const BasicDecimal256& divisor, BasicDecimal256* result,
                       BasicDecimal256* remainder) const;

  /// \brief Shift left by the given number of bits.
  BasicDecimal256& operator<<=(uint32_t bits);

  BasicDecimal256 operator<<(uint32_t bits) const {
    auto res = *this;
    res <<= bits;
    return res;
  }

  /// \brief In-place division.
  BasicDecimal256& operator/=(const BasicDecimal256& right);

  /// \brief Get the maximum valid unscaled decimal value for the given precision.
  static BasicDecimal256 GetMaxValue(int32_t precision);

  /// \brief Get the maximum decimal value (is not a valid value).
  static constexpr BasicDecimal256 GetMaxSentinel() {
#if ARROW_LITTLE_ENDIAN
    return BasicDecimal256({std::numeric_limits<uint64_t>::max(),
                            std::numeric_limits<uint64_t>::max(),
                            std::numeric_limits<uint64_t>::max(),
                            static_cast<uint64_t>(std::numeric_limits<int64_t>::max())});
#else
    return BasicDecimal256({static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                            std::numeric_limits<uint64_t>::max(),
                            std::numeric_limits<uint64_t>::max(),
                            std::numeric_limits<uint64_t>::max()});
#endif
  }
  /// \brief Get the minimum decimal value (is not a valid value).
  static constexpr BasicDecimal256 GetMinSentinel() {
#if ARROW_LITTLE_ENDIAN
    return BasicDecimal256(
        {0, 0, 0, static_cast<uint64_t>(std::numeric_limits<int64_t>::min())});
#else
    return BasicDecimal256(
        {static_cast<uint64_t>(std::numeric_limits<int64_t>::min()), 0, 0, 0});
#endif
  }
};

ARROW_EXPORT inline bool operator==(const BasicDecimal256& left,
                                    const BasicDecimal256& right) {
  return left.native_endian_array() == right.native_endian_array();
}

ARROW_EXPORT inline bool operator!=(const BasicDecimal256& left,
                                    const BasicDecimal256& right) {
  return left.native_endian_array() != right.native_endian_array();
}

ARROW_EXPORT bool operator<(const BasicDecimal256& left, const BasicDecimal256& right);

ARROW_EXPORT inline bool operator<=(const BasicDecimal256& left,
                                    const BasicDecimal256& right) {
  return !operator<(right, left);
}

ARROW_EXPORT inline bool operator>(const BasicDecimal256& left,
                                   const BasicDecimal256& right) {
  return operator<(right, left);
}

ARROW_EXPORT inline bool operator>=(const BasicDecimal256& left,
                                    const BasicDecimal256& right) {
  return !operator<(left, right);
}

ARROW_EXPORT BasicDecimal256 operator-(const BasicDecimal256& operand);
ARROW_EXPORT BasicDecimal256 operator~(const BasicDecimal256& operand);
ARROW_EXPORT BasicDecimal256 operator+(const BasicDecimal256& left,
                                       const BasicDecimal256& right);
ARROW_EXPORT BasicDecimal256 operator*(const BasicDecimal256& left,
                                       const BasicDecimal256& right);
ARROW_EXPORT BasicDecimal256 operator/(const BasicDecimal256& left,
                                       const BasicDecimal256& right);

}  // namespace arrow

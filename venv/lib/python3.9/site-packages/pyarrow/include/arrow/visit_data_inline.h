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

#include <string_view>

#include "arrow/array.h"
#include "arrow/status.h"
#include "arrow/type.h"
#include "arrow/type_traits.h"
#include "arrow/util/bit_block_counter.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/functional.h"

namespace arrow {
namespace internal {

template <typename T, typename Enable = void>
struct ArraySpanInlineVisitor {};

// Numeric and primitive C-compatible types
template <typename T>
struct ArraySpanInlineVisitor<T, enable_if_has_c_type<T>> {
  using c_type = typename T::c_type;

  template <typename ValidFunc, typename NullFunc>
  static Status VisitStatus(const ArraySpan& arr, ValidFunc&& valid_func,
                            NullFunc&& null_func) {
    const c_type* data = arr.GetValues<c_type>(1);
    auto visit_valid = [&](int64_t i) { return valid_func(data[i]); };
    return VisitBitBlocks(arr.buffers[0].data, arr.offset, arr.length,
                          std::move(visit_valid), std::forward<NullFunc>(null_func));
  }

  template <typename ValidFunc, typename NullFunc>
  static void VisitVoid(const ArraySpan& arr, ValidFunc&& valid_func,
                        NullFunc&& null_func) {
    using c_type = typename T::c_type;
    const c_type* data = arr.GetValues<c_type>(1);
    auto visit_valid = [&](int64_t i) { valid_func(data[i]); };
    VisitBitBlocksVoid(arr.buffers[0].data, arr.offset, arr.length,
                       std::move(visit_valid), std::forward<NullFunc>(null_func));
  }
};

// Boolean
template <>
struct ArraySpanInlineVisitor<BooleanType> {
  using c_type = bool;

  template <typename ValidFunc, typename NullFunc>
  static Status VisitStatus(const ArraySpan& arr, ValidFunc&& valid_func,
                            NullFunc&& null_func) {
    int64_t offset = arr.offset;
    const uint8_t* data = arr.buffers[1].data;
    return VisitBitBlocks(
        arr.buffers[0].data, offset, arr.length,
        [&](int64_t i) { return valid_func(bit_util::GetBit(data, offset + i)); },
        std::forward<NullFunc>(null_func));
  }

  template <typename ValidFunc, typename NullFunc>
  static void VisitVoid(const ArraySpan& arr, ValidFunc&& valid_func,
                        NullFunc&& null_func) {
    int64_t offset = arr.offset;
    const uint8_t* data = arr.buffers[1].data;
    VisitBitBlocksVoid(
        arr.buffers[0].data, offset, arr.length,
        [&](int64_t i) { valid_func(bit_util::GetBit(data, offset + i)); },
        std::forward<NullFunc>(null_func));
  }
};

// Binary, String...
template <typename T>
struct ArraySpanInlineVisitor<T, enable_if_base_binary<T>> {
  using c_type = std::string_view;

  template <typename ValidFunc, typename NullFunc>
  static Status VisitStatus(const ArraySpan& arr, ValidFunc&& valid_func,
                            NullFunc&& null_func) {
    using offset_type = typename T::offset_type;
    constexpr char empty_value = 0;

    if (arr.length == 0) {
      return Status::OK();
    }
    const offset_type* offsets = arr.GetValues<offset_type>(1);
    const char* data;
    if (arr.buffers[2].data == NULLPTR) {
      data = &empty_value;
    } else {
      // Do not apply the array offset to the values array; the value_offsets
      // index the non-sliced values array.
      data = arr.GetValues<char>(2, /*absolute_offset=*/0);
    }
    offset_type cur_offset = *offsets++;
    return VisitBitBlocks(
        arr.buffers[0].data, arr.offset, arr.length,
        [&](int64_t i) {
          ARROW_UNUSED(i);
          auto value = std::string_view(data + cur_offset, *offsets - cur_offset);
          cur_offset = *offsets++;
          return valid_func(value);
        },
        [&]() {
          cur_offset = *offsets++;
          return null_func();
        });
  }

  template <typename ValidFunc, typename NullFunc>
  static void VisitVoid(const ArraySpan& arr, ValidFunc&& valid_func,
                        NullFunc&& null_func) {
    using offset_type = typename T::offset_type;
    constexpr uint8_t empty_value = 0;

    if (arr.length == 0) {
      return;
    }
    const offset_type* offsets = arr.GetValues<offset_type>(1);
    const uint8_t* data;
    if (arr.buffers[2].data == NULLPTR) {
      data = &empty_value;
    } else {
      // Do not apply the array offset to the values array; the value_offsets
      // index the non-sliced values array.
      data = arr.GetValues<uint8_t>(2, /*absolute_offset=*/0);
    }

    VisitBitBlocksVoid(
        arr.buffers[0].data, arr.offset, arr.length,
        [&](int64_t i) {
          auto value = std::string_view(reinterpret_cast<const char*>(data + offsets[i]),
                                        offsets[i + 1] - offsets[i]);
          valid_func(value);
        },
        std::forward<NullFunc>(null_func));
  }
};

// FixedSizeBinary, Decimal128
template <typename T>
struct ArraySpanInlineVisitor<T, enable_if_fixed_size_binary<T>> {
  using c_type = std::string_view;

  template <typename ValidFunc, typename NullFunc>
  static Status VisitStatus(const ArraySpan& arr, ValidFunc&& valid_func,
                            NullFunc&& null_func) {
    const int32_t byte_width = arr.type->byte_width();
    const char* data = arr.GetValues<char>(1,
                                           /*absolute_offset=*/arr.offset * byte_width);
    return VisitBitBlocks(
        arr.buffers[0].data, arr.offset, arr.length,
        [&](int64_t i) {
          auto value = std::string_view(data, byte_width);
          data += byte_width;
          return valid_func(value);
        },
        [&]() {
          data += byte_width;
          return null_func();
        });
  }

  template <typename ValidFunc, typename NullFunc>
  static void VisitVoid(const ArraySpan& arr, ValidFunc&& valid_func,
                        NullFunc&& null_func) {
    const int32_t byte_width = arr.type->byte_width();
    const char* data = arr.GetValues<char>(1,
                                           /*absolute_offset=*/arr.offset * byte_width);
    VisitBitBlocksVoid(
        arr.buffers[0].data, arr.offset, arr.length,
        [&](int64_t i) {
          valid_func(std::string_view(data, byte_width));
          data += byte_width;
        },
        [&]() {
          data += byte_width;
          null_func();
        });
  }
};

}  // namespace internal

template <typename T, typename ValidFunc, typename NullFunc>
typename internal::call_traits::enable_if_return<ValidFunc, Status>::type
VisitArraySpanInline(const ArraySpan& arr, ValidFunc&& valid_func, NullFunc&& null_func) {
  return internal::ArraySpanInlineVisitor<T>::VisitStatus(
      arr, std::forward<ValidFunc>(valid_func), std::forward<NullFunc>(null_func));
}

template <typename T, typename ValidFunc, typename NullFunc>
typename internal::call_traits::enable_if_return<ValidFunc, void>::type
VisitArraySpanInline(const ArraySpan& arr, ValidFunc&& valid_func, NullFunc&& null_func) {
  return internal::ArraySpanInlineVisitor<T>::VisitVoid(
      arr, std::forward<ValidFunc>(valid_func), std::forward<NullFunc>(null_func));
}

// Visit an array's data values, in order, without overhead.
//
// The Visit method's `visitor` argument should be an object with two public methods:
// - Status VisitNull()
// - Status VisitValue(<scalar>)
//
// The scalar value's type depends on the array data type:
// - the type's `c_type`, if any
// - for boolean arrays, a `bool`
// - for binary, string and fixed-size binary arrays, a `std::string_view`

template <typename T>
struct ArraySpanVisitor {
  using InlineVisitorType = internal::ArraySpanInlineVisitor<T>;
  using c_type = typename InlineVisitorType::c_type;

  template <typename Visitor>
  static Status Visit(const ArraySpan& arr, Visitor* visitor) {
    return InlineVisitorType::VisitStatus(
        arr, [visitor](c_type v) { return visitor->VisitValue(v); },
        [visitor]() { return visitor->VisitNull(); });
  }
};

// Visit a null bitmap, in order, without overhead.
//
// The given `ValidFunc` should be a callable with either of these signatures:
// - void()
// - Status()
//
// The `NullFunc` should have the same return type as `ValidFunc`.

template <typename ValidFunc, typename NullFunc>
typename internal::call_traits::enable_if_return<ValidFunc, Status>::type
VisitNullBitmapInline(const uint8_t* valid_bits, int64_t valid_bits_offset,
                      int64_t num_values, int64_t null_count, ValidFunc&& valid_func,
                      NullFunc&& null_func) {
  ARROW_UNUSED(null_count);
  internal::OptionalBitBlockCounter bit_counter(valid_bits, valid_bits_offset,
                                                num_values);
  int64_t position = 0;
  int64_t offset_position = valid_bits_offset;
  while (position < num_values) {
    internal::BitBlockCount block = bit_counter.NextBlock();
    if (block.AllSet()) {
      for (int64_t i = 0; i < block.length; ++i) {
        ARROW_RETURN_NOT_OK(valid_func());
      }
    } else if (block.NoneSet()) {
      for (int64_t i = 0; i < block.length; ++i) {
        ARROW_RETURN_NOT_OK(null_func());
      }
    } else {
      for (int64_t i = 0; i < block.length; ++i) {
        ARROW_RETURN_NOT_OK(bit_util::GetBit(valid_bits, offset_position + i)
                                ? valid_func()
                                : null_func());
      }
    }
    position += block.length;
    offset_position += block.length;
  }
  return Status::OK();
}

template <typename ValidFunc, typename NullFunc>
typename internal::call_traits::enable_if_return<ValidFunc, void>::type
VisitNullBitmapInline(const uint8_t* valid_bits, int64_t valid_bits_offset,
                      int64_t num_values, int64_t null_count, ValidFunc&& valid_func,
                      NullFunc&& null_func) {
  ARROW_UNUSED(null_count);
  internal::OptionalBitBlockCounter bit_counter(valid_bits, valid_bits_offset,
                                                num_values);
  int64_t position = 0;
  int64_t offset_position = valid_bits_offset;
  while (position < num_values) {
    internal::BitBlockCount block = bit_counter.NextBlock();
    if (block.AllSet()) {
      for (int64_t i = 0; i < block.length; ++i) {
        valid_func();
      }
    } else if (block.NoneSet()) {
      for (int64_t i = 0; i < block.length; ++i) {
        null_func();
      }
    } else {
      for (int64_t i = 0; i < block.length; ++i) {
        bit_util::GetBit(valid_bits, offset_position + i) ? valid_func() : null_func();
      }
    }
    position += block.length;
    offset_position += block.length;
  }
}

}  // namespace arrow

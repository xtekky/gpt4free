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
#include <type_traits>
#include <vector>

#include "arrow/array.h"
#include "arrow/array/builder_binary.h"
#include "arrow/array/builder_primitive.h"
#include "arrow/array/builder_time.h"
#include "arrow/buffer.h"
#include "arrow/testing/gtest_util.h"
#include "arrow/type_fwd.h"
#include "arrow/util/bit_util.h"
#include "arrow/visit_type_inline.h"

namespace arrow {

// ArrayFromVector: construct an Array from vectors of C values

template <typename TYPE, typename C_TYPE = typename TYPE::c_type>
void ArrayFromVector(const std::shared_ptr<DataType>& type,
                     const std::vector<bool>& is_valid, const std::vector<C_TYPE>& values,
                     std::shared_ptr<Array>* out) {
  auto type_id = TYPE::type_id;
  ASSERT_EQ(type_id, type->id())
      << "template parameter and concrete DataType instance don't agree";

  std::unique_ptr<ArrayBuilder> builder_ptr;
  ASSERT_OK(MakeBuilder(default_memory_pool(), type, &builder_ptr));
  // Get the concrete builder class to access its Append() specializations
  auto& builder = dynamic_cast<typename TypeTraits<TYPE>::BuilderType&>(*builder_ptr);

  for (size_t i = 0; i < values.size(); ++i) {
    if (is_valid[i]) {
      ASSERT_OK(builder.Append(values[i]));
    } else {
      ASSERT_OK(builder.AppendNull());
    }
  }
  ASSERT_OK(builder.Finish(out));
}

template <typename TYPE, typename C_TYPE = typename TYPE::c_type>
void ArrayFromVector(const std::shared_ptr<DataType>& type,
                     const std::vector<C_TYPE>& values, std::shared_ptr<Array>* out) {
  auto type_id = TYPE::type_id;
  ASSERT_EQ(type_id, type->id())
      << "template parameter and concrete DataType instance don't agree";

  std::unique_ptr<ArrayBuilder> builder_ptr;
  ASSERT_OK(MakeBuilder(default_memory_pool(), type, &builder_ptr));
  // Get the concrete builder class to access its Append() specializations
  auto& builder = dynamic_cast<typename TypeTraits<TYPE>::BuilderType&>(*builder_ptr);

  for (size_t i = 0; i < values.size(); ++i) {
    ASSERT_OK(builder.Append(values[i]));
  }
  ASSERT_OK(builder.Finish(out));
}

// Overloads without a DataType argument, for parameterless types

template <typename TYPE, typename C_TYPE = typename TYPE::c_type>
void ArrayFromVector(const std::vector<bool>& is_valid, const std::vector<C_TYPE>& values,
                     std::shared_ptr<Array>* out) {
  auto type = TypeTraits<TYPE>::type_singleton();
  ArrayFromVector<TYPE, C_TYPE>(type, is_valid, values, out);
}

template <typename TYPE, typename C_TYPE = typename TYPE::c_type>
void ArrayFromVector(const std::vector<C_TYPE>& values, std::shared_ptr<Array>* out) {
  auto type = TypeTraits<TYPE>::type_singleton();
  ArrayFromVector<TYPE, C_TYPE>(type, values, out);
}

// ChunkedArrayFromVector: construct a ChunkedArray from vectors of C values

template <typename TYPE, typename C_TYPE = typename TYPE::c_type>
void ChunkedArrayFromVector(const std::shared_ptr<DataType>& type,
                            const std::vector<std::vector<bool>>& is_valid,
                            const std::vector<std::vector<C_TYPE>>& values,
                            std::shared_ptr<ChunkedArray>* out) {
  ArrayVector chunks;
  ASSERT_EQ(is_valid.size(), values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    std::shared_ptr<Array> array;
    ArrayFromVector<TYPE, C_TYPE>(type, is_valid[i], values[i], &array);
    chunks.push_back(array);
  }
  *out = std::make_shared<ChunkedArray>(chunks);
}

template <typename TYPE, typename C_TYPE = typename TYPE::c_type>
void ChunkedArrayFromVector(const std::shared_ptr<DataType>& type,
                            const std::vector<std::vector<C_TYPE>>& values,
                            std::shared_ptr<ChunkedArray>* out) {
  ArrayVector chunks;
  for (size_t i = 0; i < values.size(); ++i) {
    std::shared_ptr<Array> array;
    ArrayFromVector<TYPE, C_TYPE>(type, values[i], &array);
    chunks.push_back(array);
  }
  *out = std::make_shared<ChunkedArray>(chunks);
}

// Overloads without a DataType argument, for parameterless types

template <typename TYPE, typename C_TYPE = typename TYPE::c_type>
void ChunkedArrayFromVector(const std::vector<std::vector<bool>>& is_valid,
                            const std::vector<std::vector<C_TYPE>>& values,
                            std::shared_ptr<ChunkedArray>* out) {
  auto type = TypeTraits<TYPE>::type_singleton();
  ChunkedArrayFromVector<TYPE, C_TYPE>(type, is_valid, values, out);
}

template <typename TYPE, typename C_TYPE = typename TYPE::c_type>
void ChunkedArrayFromVector(const std::vector<std::vector<C_TYPE>>& values,
                            std::shared_ptr<ChunkedArray>* out) {
  auto type = TypeTraits<TYPE>::type_singleton();
  ChunkedArrayFromVector<TYPE, C_TYPE>(type, values, out);
}

template <typename BuilderType>
void FinishAndCheckPadding(BuilderType* builder, std::shared_ptr<Array>* out) {
  ASSERT_OK_AND_ASSIGN(*out, builder->Finish());
  AssertZeroPadded(**out);
  TestInitialized(**out);
}

template <class T, class Builder>
Status MakeArray(const std::vector<uint8_t>& valid_bytes, const std::vector<T>& values,
                 int64_t size, Builder* builder, std::shared_ptr<Array>* out) {
  // Append the first 1000
  for (int64_t i = 0; i < size; ++i) {
    if (valid_bytes[i] > 0) {
      RETURN_NOT_OK(builder->Append(values[i]));
    } else {
      RETURN_NOT_OK(builder->AppendNull());
    }
  }
  return builder->Finish(out);
}

template <typename Fn>
struct VisitBuilder {
  template <typename T, typename BuilderType = typename TypeTraits<T>::BuilderType,
            // need to let SFINAE drop this Visit when it would result in
            // [](NullBuilder*){}(double_builder)
            typename = decltype(std::declval<Fn>()(std::declval<BuilderType*>()))>
  Status Visit(const T&, ArrayBuilder* builder, Fn&& fn) {
    fn(internal::checked_cast<BuilderType*>(builder));
    return Status::OK();
  }

  Status Visit(const DataType& t, ArrayBuilder* builder, Fn&& fn) {
    return Status::NotImplemented("visiting builders of type ", t);
  }
};

template <typename Fn>
Result<std::shared_ptr<Array>> ArrayFromBuilderVisitor(
    const std::shared_ptr<DataType>& type, int64_t initial_capacity,
    int64_t visitor_repetitions, Fn&& fn) {
  std::unique_ptr<ArrayBuilder> builder;
  RETURN_NOT_OK(MakeBuilder(default_memory_pool(), type, &builder));

  if (initial_capacity != 0) {
    RETURN_NOT_OK(builder->Resize(initial_capacity));
  }

  VisitBuilder<Fn> visitor;
  for (int64_t i = 0; i < visitor_repetitions; ++i) {
    RETURN_NOT_OK(
        VisitTypeInline(*builder->type(), &visitor, builder.get(), std::forward<Fn>(fn)));
  }

  std::shared_ptr<Array> out;
  RETURN_NOT_OK(builder->Finish(&out));
  return std::move(out);
}

template <typename Fn>
Result<std::shared_ptr<Array>> ArrayFromBuilderVisitor(
    const std::shared_ptr<DataType>& type, int64_t length, Fn&& fn) {
  return ArrayFromBuilderVisitor(type, length, length, std::forward<Fn>(fn));
}

template <typename T>
static inline Status GetBitmapFromVector(const std::vector<T>& is_valid,
                                         std::shared_ptr<Buffer>* result) {
  size_t length = is_valid.size();

  ARROW_ASSIGN_OR_RAISE(auto buffer, AllocateEmptyBitmap(length));

  uint8_t* bitmap = buffer->mutable_data();
  for (size_t i = 0; i < static_cast<size_t>(length); ++i) {
    if (is_valid[i]) {
      bit_util::SetBit(bitmap, i);
    }
  }

  *result = buffer;
  return Status::OK();
}

template <typename T>
inline void BitmapFromVector(const std::vector<T>& is_valid,
                             std::shared_ptr<Buffer>* out) {
  ASSERT_OK(GetBitmapFromVector(is_valid, out));
}

}  // namespace arrow

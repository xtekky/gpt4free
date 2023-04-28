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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "arrow/array.h"
#include "arrow/chunked_array.h"
#include "arrow/status.h"
#include "arrow/type.h"
#include "arrow/type_traits.h"
#include "arrow/util/checked_cast.h"
#include "arrow/visit_type_inline.h"

namespace arrow {
namespace internal {

template <typename BaseConverter, template <typename...> class ConverterTrait>
static Result<std::unique_ptr<BaseConverter>> MakeConverter(
    std::shared_ptr<DataType> type, typename BaseConverter::OptionsType options,
    MemoryPool* pool);

template <typename Input, typename Options>
class Converter {
 public:
  using Self = Converter<Input, Options>;
  using InputType = Input;
  using OptionsType = Options;

  virtual ~Converter() = default;

  Status Construct(std::shared_ptr<DataType> type, OptionsType options,
                   MemoryPool* pool) {
    type_ = std::move(type);
    options_ = std::move(options);
    return Init(pool);
  }

  virtual Status Append(InputType value) { return Status::NotImplemented("Append"); }

  virtual Status Extend(InputType values, int64_t size, int64_t offset = 0) {
    return Status::NotImplemented("Extend");
  }

  virtual Status ExtendMasked(InputType values, InputType mask, int64_t size,
                              int64_t offset = 0) {
    return Status::NotImplemented("ExtendMasked");
  }

  const std::shared_ptr<ArrayBuilder>& builder() const { return builder_; }

  const std::shared_ptr<DataType>& type() const { return type_; }

  OptionsType options() const { return options_; }

  bool may_overflow() const { return may_overflow_; }

  bool rewind_on_overflow() const { return rewind_on_overflow_; }

  virtual Status Reserve(int64_t additional_capacity) {
    return builder_->Reserve(additional_capacity);
  }

  Status AppendNull() { return builder_->AppendNull(); }

  virtual Result<std::shared_ptr<Array>> ToArray() { return builder_->Finish(); }

  virtual Result<std::shared_ptr<Array>> ToArray(int64_t length) {
    ARROW_ASSIGN_OR_RAISE(auto arr, this->ToArray());
    return arr->Slice(0, length);
  }

  virtual Result<std::shared_ptr<ChunkedArray>> ToChunkedArray() {
    ARROW_ASSIGN_OR_RAISE(auto array, ToArray());
    std::vector<std::shared_ptr<Array>> chunks = {std::move(array)};
    return std::make_shared<ChunkedArray>(chunks);
  }

 protected:
  virtual Status Init(MemoryPool* pool) { return Status::OK(); }

  std::shared_ptr<DataType> type_;
  std::shared_ptr<ArrayBuilder> builder_;
  OptionsType options_;
  bool may_overflow_ = false;
  bool rewind_on_overflow_ = false;
};

template <typename ArrowType, typename BaseConverter>
class PrimitiveConverter : public BaseConverter {
 public:
  using BuilderType = typename TypeTraits<ArrowType>::BuilderType;

 protected:
  Status Init(MemoryPool* pool) override {
    this->builder_ = std::make_shared<BuilderType>(this->type_, pool);
    // Narrow variable-sized binary types may overflow
    this->may_overflow_ = is_binary_like(this->type_->id());
    primitive_type_ = checked_cast<const ArrowType*>(this->type_.get());
    primitive_builder_ = checked_cast<BuilderType*>(this->builder_.get());
    return Status::OK();
  }

  const ArrowType* primitive_type_;
  BuilderType* primitive_builder_;
};

template <typename ArrowType, typename BaseConverter,
          template <typename...> class ConverterTrait>
class ListConverter : public BaseConverter {
 public:
  using BuilderType = typename TypeTraits<ArrowType>::BuilderType;
  using ConverterType = typename ConverterTrait<ArrowType>::type;

 protected:
  Status Init(MemoryPool* pool) override {
    list_type_ = checked_cast<const ArrowType*>(this->type_.get());
    ARROW_ASSIGN_OR_RAISE(value_converter_,
                          (MakeConverter<BaseConverter, ConverterTrait>(
                              list_type_->value_type(), this->options_, pool)));
    this->builder_ =
        std::make_shared<BuilderType>(pool, value_converter_->builder(), this->type_);
    list_builder_ = checked_cast<BuilderType*>(this->builder_.get());
    // Narrow list types may overflow
    this->may_overflow_ = this->rewind_on_overflow_ =
        sizeof(typename ArrowType::offset_type) < sizeof(int64_t);
    return Status::OK();
  }

  const ArrowType* list_type_;
  BuilderType* list_builder_;
  std::unique_ptr<BaseConverter> value_converter_;
};

template <typename BaseConverter, template <typename...> class ConverterTrait>
class StructConverter : public BaseConverter {
 public:
  using ConverterType = typename ConverterTrait<StructType>::type;

  Status Reserve(int64_t additional_capacity) override {
    ARROW_RETURN_NOT_OK(this->builder_->Reserve(additional_capacity));
    for (const auto& child : children_) {
      ARROW_RETURN_NOT_OK(child->Reserve(additional_capacity));
    }
    return Status::OK();
  }

 protected:
  Status Init(MemoryPool* pool) override {
    std::unique_ptr<BaseConverter> child_converter;
    std::vector<std::shared_ptr<ArrayBuilder>> child_builders;

    struct_type_ = checked_cast<const StructType*>(this->type_.get());
    for (const auto& field : struct_type_->fields()) {
      ARROW_ASSIGN_OR_RAISE(child_converter,
                            (MakeConverter<BaseConverter, ConverterTrait>(
                                field->type(), this->options_, pool)));
      this->may_overflow_ |= child_converter->may_overflow();
      this->rewind_on_overflow_ = this->may_overflow_;
      child_builders.push_back(child_converter->builder());
      children_.push_back(std::move(child_converter));
    }

    this->builder_ =
        std::make_shared<StructBuilder>(this->type_, pool, std::move(child_builders));
    struct_builder_ = checked_cast<StructBuilder*>(this->builder_.get());

    return Status::OK();
  }

  const StructType* struct_type_;
  StructBuilder* struct_builder_;
  std::vector<std::unique_ptr<BaseConverter>> children_;
};

template <typename ValueType, typename BaseConverter>
class DictionaryConverter : public BaseConverter {
 public:
  using BuilderType = DictionaryBuilder<ValueType>;

 protected:
  Status Init(MemoryPool* pool) override {
    std::unique_ptr<ArrayBuilder> builder;
    ARROW_RETURN_NOT_OK(MakeDictionaryBuilder(pool, this->type_, NULLPTR, &builder));
    this->builder_ = std::move(builder);
    this->may_overflow_ = false;
    dict_type_ = checked_cast<const DictionaryType*>(this->type_.get());
    value_type_ = checked_cast<const ValueType*>(dict_type_->value_type().get());
    value_builder_ = checked_cast<BuilderType*>(this->builder_.get());
    return Status::OK();
  }

  const DictionaryType* dict_type_;
  const ValueType* value_type_;
  BuilderType* value_builder_;
};

template <typename BaseConverter, template <typename...> class ConverterTrait>
struct MakeConverterImpl {
  template <typename T, typename ConverterType = typename ConverterTrait<T>::type>
  Status Visit(const T&) {
    out.reset(new ConverterType());
    return out->Construct(std::move(type), std::move(options), pool);
  }

  Status Visit(const DictionaryType& t) {
    switch (t.value_type()->id()) {
#define DICTIONARY_CASE(TYPE)                                                       \
  case TYPE::type_id:                                                               \
    out = std::make_unique<                                                         \
        typename ConverterTrait<DictionaryType>::template dictionary_type<TYPE>>(); \
    break;
      DICTIONARY_CASE(BooleanType);
      DICTIONARY_CASE(Int8Type);
      DICTIONARY_CASE(Int16Type);
      DICTIONARY_CASE(Int32Type);
      DICTIONARY_CASE(Int64Type);
      DICTIONARY_CASE(UInt8Type);
      DICTIONARY_CASE(UInt16Type);
      DICTIONARY_CASE(UInt32Type);
      DICTIONARY_CASE(UInt64Type);
      DICTIONARY_CASE(FloatType);
      DICTIONARY_CASE(DoubleType);
      DICTIONARY_CASE(BinaryType);
      DICTIONARY_CASE(StringType);
      DICTIONARY_CASE(FixedSizeBinaryType);
#undef DICTIONARY_CASE
      default:
        return Status::NotImplemented("DictionaryArray converter for type ", t.ToString(),
                                      " not implemented");
    }
    return out->Construct(std::move(type), std::move(options), pool);
  }

  Status Visit(const DataType& t) { return Status::NotImplemented(t.name()); }

  std::shared_ptr<DataType> type;
  typename BaseConverter::OptionsType options;
  MemoryPool* pool;
  std::unique_ptr<BaseConverter> out;
};

template <typename BaseConverter, template <typename...> class ConverterTrait>
static Result<std::unique_ptr<BaseConverter>> MakeConverter(
    std::shared_ptr<DataType> type, typename BaseConverter::OptionsType options,
    MemoryPool* pool) {
  MakeConverterImpl<BaseConverter, ConverterTrait> visitor{
      std::move(type), std::move(options), pool, NULLPTR};
  ARROW_RETURN_NOT_OK(VisitTypeInline(*visitor.type, &visitor));
  return std::move(visitor.out);
}

template <typename Converter>
class Chunker {
 public:
  using InputType = typename Converter::InputType;

  explicit Chunker(std::unique_ptr<Converter> converter)
      : converter_(std::move(converter)) {}

  Status Reserve(int64_t additional_capacity) {
    ARROW_RETURN_NOT_OK(converter_->Reserve(additional_capacity));
    reserved_ += additional_capacity;
    return Status::OK();
  }

  Status AppendNull() {
    auto status = converter_->AppendNull();
    if (ARROW_PREDICT_FALSE(status.IsCapacityError())) {
      if (converter_->builder()->length() == 0) {
        // Builder length == 0 means the individual element is too large to append.
        // In this case, no need to try again.
        return status;
      }
      ARROW_RETURN_NOT_OK(FinishChunk());
      return converter_->AppendNull();
    }
    ++length_;
    return status;
  }

  Status Append(InputType value) {
    auto status = converter_->Append(value);
    if (ARROW_PREDICT_FALSE(status.IsCapacityError())) {
      if (converter_->builder()->length() == 0) {
        return status;
      }
      ARROW_RETURN_NOT_OK(FinishChunk());
      return Append(value);
    }
    ++length_;
    return status;
  }

  Status Extend(InputType values, int64_t size, int64_t offset = 0) {
    while (offset < size) {
      auto length_before = converter_->builder()->length();
      auto status = converter_->Extend(values, size, offset);
      auto length_after = converter_->builder()->length();
      auto num_converted = length_after - length_before;

      offset += num_converted;
      length_ += num_converted;

      if (status.IsCapacityError()) {
        if (converter_->builder()->length() == 0) {
          // Builder length == 0 means the individual element is too large to append.
          // In this case, no need to try again.
          return status;
        } else if (converter_->rewind_on_overflow()) {
          // The list-like and binary-like conversion paths may raise  a capacity error,
          // we need to handle them differently. While the binary-like converters check
          // the capacity before append/extend the list-like converters just check after
          // append/extend. Thus depending on the implementation semantics we may need
          // to rewind (slice) the output chunk by one.
          length_ -= 1;
          offset -= 1;
        }
        ARROW_RETURN_NOT_OK(FinishChunk());
      } else if (!status.ok()) {
        return status;
      }
    }
    return Status::OK();
  }

  Status ExtendMasked(InputType values, InputType mask, int64_t size,
                      int64_t offset = 0) {
    while (offset < size) {
      auto length_before = converter_->builder()->length();
      auto status = converter_->ExtendMasked(values, mask, size, offset);
      auto length_after = converter_->builder()->length();
      auto num_converted = length_after - length_before;

      offset += num_converted;
      length_ += num_converted;

      if (status.IsCapacityError()) {
        if (converter_->builder()->length() == 0) {
          // Builder length == 0 means the individual element is too large to append.
          // In this case, no need to try again.
          return status;
        } else if (converter_->rewind_on_overflow()) {
          // The list-like and binary-like conversion paths may raise  a capacity error,
          // we need to handle them differently. While the binary-like converters check
          // the capacity before append/extend the list-like converters just check after
          // append/extend. Thus depending on the implementation semantics we may need
          // to rewind (slice) the output chunk by one.
          length_ -= 1;
          offset -= 1;
        }
        ARROW_RETURN_NOT_OK(FinishChunk());
      } else if (!status.ok()) {
        return status;
      }
    }
    return Status::OK();
  }

  Status FinishChunk() {
    ARROW_ASSIGN_OR_RAISE(auto chunk, converter_->ToArray(length_));
    chunks_.push_back(chunk);
    // Reserve space for the remaining items.
    // Besides being an optimization, it is also required if the converter's
    // implementation relies on unsafe builder methods in converter->Append().
    auto remaining = reserved_ - length_;
    Reset();
    return Reserve(remaining);
  }

  Result<std::shared_ptr<ChunkedArray>> ToChunkedArray() {
    ARROW_RETURN_NOT_OK(FinishChunk());
    return std::make_shared<ChunkedArray>(chunks_);
  }

 protected:
  void Reset() {
    converter_->builder()->Reset();
    length_ = 0;
    reserved_ = 0;
  }

  int64_t length_ = 0;
  int64_t reserved_ = 0;
  std::unique_ptr<Converter> converter_;
  std::vector<std::shared_ptr<Array>> chunks_;
};

template <typename T>
static Result<std::unique_ptr<Chunker<T>>> MakeChunker(std::unique_ptr<T> converter) {
  return std::make_unique<Chunker<T>>(std::move(converter));
}

}  // namespace internal
}  // namespace arrow

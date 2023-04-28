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

// This module defines an abstract interface for iterating through pages in a
// Parquet column chunk within a row group. It could be extended in the future
// to iterate through all data pages in all chunks in a file.

#pragma once

#include <algorithm>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "arrow/io/memory.h"
#include "arrow/testing/util.h"

#include "parquet/column_page.h"
#include "parquet/column_reader.h"
#include "parquet/column_writer.h"
#include "parquet/encoding.h"
#include "parquet/platform.h"

namespace parquet {

static constexpr int FLBA_LENGTH = 12;

inline bool operator==(const FixedLenByteArray& a, const FixedLenByteArray& b) {
  return 0 == memcmp(a.ptr, b.ptr, FLBA_LENGTH);
}

namespace test {

typedef ::testing::Types<BooleanType, Int32Type, Int64Type, Int96Type, FloatType,
                         DoubleType, ByteArrayType, FLBAType>
    ParquetTypes;

class ParquetTestException : public parquet::ParquetException {
  using ParquetException::ParquetException;
};

const char* get_data_dir();
std::string get_bad_data_dir();

std::string get_data_file(const std::string& filename, bool is_good = true);

template <typename T>
static inline void assert_vector_equal(const std::vector<T>& left,
                                       const std::vector<T>& right) {
  ASSERT_EQ(left.size(), right.size());

  for (size_t i = 0; i < left.size(); ++i) {
    ASSERT_EQ(left[i], right[i]) << i;
  }
}

template <typename T>
static inline bool vector_equal(const std::vector<T>& left, const std::vector<T>& right) {
  if (left.size() != right.size()) {
    return false;
  }

  for (size_t i = 0; i < left.size(); ++i) {
    if (left[i] != right[i]) {
      std::cerr << "index " << i << " left was " << left[i] << " right was " << right[i]
                << std::endl;
      return false;
    }
  }

  return true;
}

template <typename T>
static std::vector<T> slice(const std::vector<T>& values, int start, int end) {
  if (end < start) {
    return std::vector<T>(0);
  }

  std::vector<T> out(end - start);
  for (int i = start; i < end; ++i) {
    out[i - start] = values[i];
  }
  return out;
}

void random_bytes(int n, uint32_t seed, std::vector<uint8_t>* out);
void random_bools(int n, double p, uint32_t seed, bool* out);

template <typename T>
inline void random_numbers(int n, uint32_t seed, T min_value, T max_value, T* out) {
  std::default_random_engine gen(seed);
  std::uniform_int_distribution<T> d(min_value, max_value);
  for (int i = 0; i < n; ++i) {
    out[i] = d(gen);
  }
}

template <>
inline void random_numbers(int n, uint32_t seed, float min_value, float max_value,
                           float* out) {
  std::default_random_engine gen(seed);
  std::uniform_real_distribution<float> d(min_value, max_value);
  for (int i = 0; i < n; ++i) {
    out[i] = d(gen);
  }
}

template <>
inline void random_numbers(int n, uint32_t seed, double min_value, double max_value,
                           double* out) {
  std::default_random_engine gen(seed);
  std::uniform_real_distribution<double> d(min_value, max_value);
  for (int i = 0; i < n; ++i) {
    out[i] = d(gen);
  }
}

void random_Int96_numbers(int n, uint32_t seed, int32_t min_value, int32_t max_value,
                          Int96* out);

void random_fixed_byte_array(int n, uint32_t seed, uint8_t* buf, int len, FLBA* out);

void random_byte_array(int n, uint32_t seed, uint8_t* buf, ByteArray* out, int min_size,
                       int max_size);

void random_byte_array(int n, uint32_t seed, uint8_t* buf, ByteArray* out, int max_size);

template <typename Type, typename Sequence>
std::shared_ptr<Buffer> EncodeValues(Encoding::type encoding, bool use_dictionary,
                                     const Sequence& values, int length,
                                     const ColumnDescriptor* descr) {
  auto encoder = MakeTypedEncoder<Type>(encoding, use_dictionary, descr);
  encoder->Put(values, length);
  return encoder->FlushValues();
}

template <typename T>
static void InitValues(int num_values, uint32_t seed, std::vector<T>& values,
                       std::vector<uint8_t>& buffer) {
  random_numbers(num_values, seed, std::numeric_limits<T>::min(),
                 std::numeric_limits<T>::max(), values.data());
}

template <typename T>
static void InitValues(int num_values, std::vector<T>& values,
                       std::vector<uint8_t>& buffer) {
  InitValues(num_values, 0, values, buffer);
}

template <typename T>
static void InitDictValues(int num_values, int num_dicts, std::vector<T>& values,
                           std::vector<uint8_t>& buffer) {
  int repeat_factor = num_values / num_dicts;
  InitValues<T>(num_dicts, values, buffer);
  // add some repeated values
  for (int j = 1; j < repeat_factor; ++j) {
    for (int i = 0; i < num_dicts; ++i) {
      std::memcpy(&values[num_dicts * j + i], &values[i], sizeof(T));
    }
  }
  // computed only dict_per_page * repeat_factor - 1 values < num_values
  // compute remaining
  for (int i = num_dicts * repeat_factor; i < num_values; ++i) {
    std::memcpy(&values[i], &values[i - num_dicts * repeat_factor], sizeof(T));
  }
}

template <>
inline void InitDictValues<bool>(int num_values, int num_dicts, std::vector<bool>& values,
                                 std::vector<uint8_t>& buffer) {
  // No op for bool
}

class MockPageReader : public PageReader {
 public:
  explicit MockPageReader(const std::vector<std::shared_ptr<Page>>& pages)
      : pages_(pages), page_index_(0) {}

  std::shared_ptr<Page> NextPage() override {
    if (page_index_ == static_cast<int>(pages_.size())) {
      // EOS to consumer
      return std::shared_ptr<Page>(nullptr);
    }
    return pages_[page_index_++];
  }

  // No-op
  void set_max_page_header_size(uint32_t size) override {}

 private:
  std::vector<std::shared_ptr<Page>> pages_;
  int page_index_;
};

// TODO(wesm): this is only used for testing for now. Refactor to form part of
// primary file write path
template <typename Type>
class DataPageBuilder {
 public:
  using c_type = typename Type::c_type;

  // This class writes data and metadata to the passed inputs
  explicit DataPageBuilder(ArrowOutputStream* sink)
      : sink_(sink),
        num_values_(0),
        encoding_(Encoding::PLAIN),
        definition_level_encoding_(Encoding::RLE),
        repetition_level_encoding_(Encoding::RLE),
        have_def_levels_(false),
        have_rep_levels_(false),
        have_values_(false) {}

  void AppendDefLevels(const std::vector<int16_t>& levels, int16_t max_level,
                       Encoding::type encoding = Encoding::RLE) {
    AppendLevels(levels, max_level, encoding);

    num_values_ = std::max(static_cast<int32_t>(levels.size()), num_values_);
    definition_level_encoding_ = encoding;
    have_def_levels_ = true;
  }

  void AppendRepLevels(const std::vector<int16_t>& levels, int16_t max_level,
                       Encoding::type encoding = Encoding::RLE) {
    AppendLevels(levels, max_level, encoding);

    num_values_ = std::max(static_cast<int32_t>(levels.size()), num_values_);
    repetition_level_encoding_ = encoding;
    have_rep_levels_ = true;
  }

  void AppendValues(const ColumnDescriptor* d, const std::vector<c_type>& values,
                    Encoding::type encoding = Encoding::PLAIN) {
    std::shared_ptr<Buffer> values_sink = EncodeValues<Type>(
        encoding, false, values.data(), static_cast<int>(values.size()), d);
    PARQUET_THROW_NOT_OK(sink_->Write(values_sink->data(), values_sink->size()));

    num_values_ = std::max(static_cast<int32_t>(values.size()), num_values_);
    encoding_ = encoding;
    have_values_ = true;
  }

  int32_t num_values() const { return num_values_; }

  Encoding::type encoding() const { return encoding_; }

  Encoding::type rep_level_encoding() const { return repetition_level_encoding_; }

  Encoding::type def_level_encoding() const { return definition_level_encoding_; }

 private:
  ArrowOutputStream* sink_;

  int32_t num_values_;
  Encoding::type encoding_;
  Encoding::type definition_level_encoding_;
  Encoding::type repetition_level_encoding_;

  bool have_def_levels_;
  bool have_rep_levels_;
  bool have_values_;

  // Used internally for both repetition and definition levels
  void AppendLevels(const std::vector<int16_t>& levels, int16_t max_level,
                    Encoding::type encoding) {
    if (encoding != Encoding::RLE) {
      ParquetException::NYI("only rle encoding currently implemented");
    }

    std::vector<uint8_t> encode_buffer(LevelEncoder::MaxBufferSize(
        Encoding::RLE, max_level, static_cast<int>(levels.size())));

    // We encode into separate memory from the output stream because the
    // RLE-encoded bytes have to be preceded in the stream by their absolute
    // size.
    LevelEncoder encoder;
    encoder.Init(encoding, max_level, static_cast<int>(levels.size()),
                 encode_buffer.data(), static_cast<int>(encode_buffer.size()));

    encoder.Encode(static_cast<int>(levels.size()), levels.data());

    int32_t rle_bytes = encoder.len();
    PARQUET_THROW_NOT_OK(
        sink_->Write(reinterpret_cast<const uint8_t*>(&rle_bytes), sizeof(int32_t)));
    PARQUET_THROW_NOT_OK(sink_->Write(encode_buffer.data(), rle_bytes));
  }
};

template <>
inline void DataPageBuilder<BooleanType>::AppendValues(const ColumnDescriptor* d,
                                                       const std::vector<bool>& values,
                                                       Encoding::type encoding) {
  if (encoding != Encoding::PLAIN) {
    ParquetException::NYI("only plain encoding currently implemented");
  }

  auto encoder = MakeTypedEncoder<BooleanType>(Encoding::PLAIN, false, d);
  dynamic_cast<BooleanEncoder*>(encoder.get())
      ->Put(values, static_cast<int>(values.size()));
  std::shared_ptr<Buffer> buffer = encoder->FlushValues();
  PARQUET_THROW_NOT_OK(sink_->Write(buffer->data(), buffer->size()));

  num_values_ = std::max(static_cast<int32_t>(values.size()), num_values_);
  encoding_ = encoding;
  have_values_ = true;
}

template <typename Type>
static std::shared_ptr<DataPageV1> MakeDataPage(
    const ColumnDescriptor* d, const std::vector<typename Type::c_type>& values,
    int num_vals, Encoding::type encoding, const uint8_t* indices, int indices_size,
    const std::vector<int16_t>& def_levels, int16_t max_def_level,
    const std::vector<int16_t>& rep_levels, int16_t max_rep_level) {
  int num_values = 0;

  auto page_stream = CreateOutputStream();
  test::DataPageBuilder<Type> page_builder(page_stream.get());

  if (!rep_levels.empty()) {
    page_builder.AppendRepLevels(rep_levels, max_rep_level);
  }
  if (!def_levels.empty()) {
    page_builder.AppendDefLevels(def_levels, max_def_level);
  }

  if (encoding == Encoding::PLAIN) {
    page_builder.AppendValues(d, values, encoding);
    num_values = std::max(page_builder.num_values(), num_vals);
  } else {  // DICTIONARY PAGES
    PARQUET_THROW_NOT_OK(page_stream->Write(indices, indices_size));
    num_values = std::max(page_builder.num_values(), num_vals);
  }

  PARQUET_ASSIGN_OR_THROW(auto buffer, page_stream->Finish());

  return std::make_shared<DataPageV1>(buffer, num_values, encoding,
                                      page_builder.def_level_encoding(),
                                      page_builder.rep_level_encoding(), buffer->size());
}

template <typename TYPE>
class DictionaryPageBuilder {
 public:
  typedef typename TYPE::c_type TC;
  static constexpr int TN = TYPE::type_num;
  using SpecializedEncoder = typename EncodingTraits<TYPE>::Encoder;

  // This class writes data and metadata to the passed inputs
  explicit DictionaryPageBuilder(const ColumnDescriptor* d)
      : num_dict_values_(0), have_values_(false) {
    auto encoder = MakeTypedEncoder<TYPE>(Encoding::PLAIN, true, d);
    dict_traits_ = dynamic_cast<DictEncoder<TYPE>*>(encoder.get());
    encoder_.reset(dynamic_cast<SpecializedEncoder*>(encoder.release()));
  }

  ~DictionaryPageBuilder() {}

  std::shared_ptr<Buffer> AppendValues(const std::vector<TC>& values) {
    int num_values = static_cast<int>(values.size());
    // Dictionary encoding
    encoder_->Put(values.data(), num_values);
    num_dict_values_ = dict_traits_->num_entries();
    have_values_ = true;
    return encoder_->FlushValues();
  }

  std::shared_ptr<Buffer> WriteDict() {
    std::shared_ptr<Buffer> dict_buffer =
        AllocateBuffer(::arrow::default_memory_pool(), dict_traits_->dict_encoded_size());
    dict_traits_->WriteDict(dict_buffer->mutable_data());
    return dict_buffer;
  }

  int32_t num_values() const { return num_dict_values_; }

 private:
  DictEncoder<TYPE>* dict_traits_;
  std::unique_ptr<SpecializedEncoder> encoder_;
  int32_t num_dict_values_;
  bool have_values_;
};

template <>
inline DictionaryPageBuilder<BooleanType>::DictionaryPageBuilder(
    const ColumnDescriptor* d) {
  ParquetException::NYI("only plain encoding currently implemented for boolean");
}

template <>
inline std::shared_ptr<Buffer> DictionaryPageBuilder<BooleanType>::WriteDict() {
  ParquetException::NYI("only plain encoding currently implemented for boolean");
  return nullptr;
}

template <>
inline std::shared_ptr<Buffer> DictionaryPageBuilder<BooleanType>::AppendValues(
    const std::vector<TC>& values) {
  ParquetException::NYI("only plain encoding currently implemented for boolean");
  return nullptr;
}

template <typename Type>
inline static std::shared_ptr<DictionaryPage> MakeDictPage(
    const ColumnDescriptor* d, const std::vector<typename Type::c_type>& values,
    const std::vector<int>& values_per_page, Encoding::type encoding,
    std::vector<std::shared_ptr<Buffer>>& rle_indices) {
  test::DictionaryPageBuilder<Type> page_builder(d);
  int num_pages = static_cast<int>(values_per_page.size());
  int value_start = 0;

  for (int i = 0; i < num_pages; i++) {
    rle_indices.push_back(page_builder.AppendValues(
        slice(values, value_start, value_start + values_per_page[i])));
    value_start += values_per_page[i];
  }

  auto buffer = page_builder.WriteDict();

  return std::make_shared<DictionaryPage>(buffer, page_builder.num_values(),
                                          Encoding::PLAIN);
}

// Given def/rep levels and values create multiple dict pages
template <typename Type>
inline static void PaginateDict(const ColumnDescriptor* d,
                                const std::vector<typename Type::c_type>& values,
                                const std::vector<int16_t>& def_levels,
                                int16_t max_def_level,
                                const std::vector<int16_t>& rep_levels,
                                int16_t max_rep_level, int num_levels_per_page,
                                const std::vector<int>& values_per_page,
                                std::vector<std::shared_ptr<Page>>& pages,
                                Encoding::type encoding = Encoding::RLE_DICTIONARY) {
  int num_pages = static_cast<int>(values_per_page.size());
  std::vector<std::shared_ptr<Buffer>> rle_indices;
  std::shared_ptr<DictionaryPage> dict_page =
      MakeDictPage<Type>(d, values, values_per_page, encoding, rle_indices);
  pages.push_back(dict_page);
  int def_level_start = 0;
  int def_level_end = 0;
  int rep_level_start = 0;
  int rep_level_end = 0;
  for (int i = 0; i < num_pages; i++) {
    if (max_def_level > 0) {
      def_level_start = i * num_levels_per_page;
      def_level_end = (i + 1) * num_levels_per_page;
    }
    if (max_rep_level > 0) {
      rep_level_start = i * num_levels_per_page;
      rep_level_end = (i + 1) * num_levels_per_page;
    }
    std::shared_ptr<DataPageV1> data_page = MakeDataPage<Int32Type>(
        d, {}, values_per_page[i], encoding, rle_indices[i]->data(),
        static_cast<int>(rle_indices[i]->size()),
        slice(def_levels, def_level_start, def_level_end), max_def_level,
        slice(rep_levels, rep_level_start, rep_level_end), max_rep_level);
    pages.push_back(data_page);
  }
}

// Given def/rep levels and values create multiple plain pages
template <typename Type>
static inline void PaginatePlain(const ColumnDescriptor* d,
                                 const std::vector<typename Type::c_type>& values,
                                 const std::vector<int16_t>& def_levels,
                                 int16_t max_def_level,
                                 const std::vector<int16_t>& rep_levels,
                                 int16_t max_rep_level, int num_levels_per_page,
                                 const std::vector<int>& values_per_page,
                                 std::vector<std::shared_ptr<Page>>& pages,
                                 Encoding::type encoding = Encoding::PLAIN) {
  int num_pages = static_cast<int>(values_per_page.size());
  int def_level_start = 0;
  int def_level_end = 0;
  int rep_level_start = 0;
  int rep_level_end = 0;
  int value_start = 0;
  for (int i = 0; i < num_pages; i++) {
    if (max_def_level > 0) {
      def_level_start = i * num_levels_per_page;
      def_level_end = (i + 1) * num_levels_per_page;
    }
    if (max_rep_level > 0) {
      rep_level_start = i * num_levels_per_page;
      rep_level_end = (i + 1) * num_levels_per_page;
    }
    std::shared_ptr<DataPage> page = MakeDataPage<Type>(
        d, slice(values, value_start, value_start + values_per_page[i]),
        values_per_page[i], encoding, nullptr, 0,
        slice(def_levels, def_level_start, def_level_end), max_def_level,
        slice(rep_levels, rep_level_start, rep_level_end), max_rep_level);
    pages.push_back(page);
    value_start += values_per_page[i];
  }
}

// Generates pages from randomly generated data
template <typename Type>
static inline int MakePages(const ColumnDescriptor* d, int num_pages, int levels_per_page,
                            std::vector<int16_t>& def_levels,
                            std::vector<int16_t>& rep_levels,
                            std::vector<typename Type::c_type>& values,
                            std::vector<uint8_t>& buffer,
                            std::vector<std::shared_ptr<Page>>& pages,
                            Encoding::type encoding = Encoding::PLAIN) {
  int num_levels = levels_per_page * num_pages;
  int num_values = 0;
  uint32_t seed = 0;
  int16_t zero = 0;
  int16_t max_def_level = d->max_definition_level();
  int16_t max_rep_level = d->max_repetition_level();
  std::vector<int> values_per_page(num_pages, levels_per_page);
  // Create definition levels
  if (max_def_level > 0) {
    def_levels.resize(num_levels);
    random_numbers(num_levels, seed, zero, max_def_level, def_levels.data());
    for (int p = 0; p < num_pages; p++) {
      int num_values_per_page = 0;
      for (int i = 0; i < levels_per_page; i++) {
        if (def_levels[i + p * levels_per_page] == max_def_level) {
          num_values_per_page++;
          num_values++;
        }
      }
      values_per_page[p] = num_values_per_page;
    }
  } else {
    num_values = num_levels;
  }
  // Create repetition levels
  if (max_rep_level > 0) {
    rep_levels.resize(num_levels);
    random_numbers(num_levels, seed, zero, max_rep_level, rep_levels.data());
  }
  // Create values
  values.resize(num_values);
  if (encoding == Encoding::PLAIN) {
    InitValues<typename Type::c_type>(num_values, values, buffer);
    PaginatePlain<Type>(d, values, def_levels, max_def_level, rep_levels, max_rep_level,
                        levels_per_page, values_per_page, pages);
  } else if (encoding == Encoding::RLE_DICTIONARY ||
             encoding == Encoding::PLAIN_DICTIONARY) {
    // Calls InitValues and repeats the data
    InitDictValues<typename Type::c_type>(num_values, levels_per_page, values, buffer);
    PaginateDict<Type>(d, values, def_levels, max_def_level, rep_levels, max_rep_level,
                       levels_per_page, values_per_page, pages);
  }

  return num_values;
}

// ----------------------------------------------------------------------
// Test data generation

template <>
void inline InitValues<bool>(int num_values, uint32_t seed, std::vector<bool>& values,
                             std::vector<uint8_t>& buffer) {
  values = {};
  if (seed == 0) {
    seed = static_cast<uint32_t>(::arrow::random_seed());
  }
  ::arrow::random_is_valid(num_values, 0.5, &values, static_cast<int>(seed));
}

template <>
inline void InitValues<ByteArray>(int num_values, uint32_t seed,
                                  std::vector<ByteArray>& values,
                                  std::vector<uint8_t>& buffer) {
  int max_byte_array_len = 12;
  int num_bytes = static_cast<int>(max_byte_array_len + sizeof(uint32_t));
  size_t nbytes = num_values * num_bytes;
  buffer.resize(nbytes);
  random_byte_array(num_values, seed, buffer.data(), values.data(), max_byte_array_len);
}

inline void InitWideByteArrayValues(int num_values, std::vector<ByteArray>& values,
                                    std::vector<uint8_t>& buffer, int min_len,
                                    int max_len) {
  int num_bytes = static_cast<int>(max_len + sizeof(uint32_t));
  size_t nbytes = num_values * num_bytes;
  buffer.resize(nbytes);
  random_byte_array(num_values, 0, buffer.data(), values.data(), min_len, max_len);
}

template <>
inline void InitValues<FLBA>(int num_values, uint32_t seed, std::vector<FLBA>& values,
                             std::vector<uint8_t>& buffer) {
  size_t nbytes = num_values * FLBA_LENGTH;
  buffer.resize(nbytes);
  random_fixed_byte_array(num_values, seed, buffer.data(), FLBA_LENGTH, values.data());
}

template <>
inline void InitValues<Int96>(int num_values, uint32_t seed, std::vector<Int96>& values,
                              std::vector<uint8_t>& buffer) {
  random_Int96_numbers(num_values, seed, std::numeric_limits<int32_t>::min(),
                       std::numeric_limits<int32_t>::max(), values.data());
}

inline std::string TestColumnName(int i) {
  std::stringstream col_name;
  col_name << "column_" << i;
  return col_name.str();
}

// This class lives here because of its dependency on the InitValues specializations.
template <typename TestType>
class PrimitiveTypedTest : public ::testing::Test {
 public:
  using c_type = typename TestType::c_type;

  void SetUpSchema(Repetition::type repetition, int num_columns = 1) {
    std::vector<schema::NodePtr> fields;

    for (int i = 0; i < num_columns; ++i) {
      std::string name = TestColumnName(i);
      fields.push_back(schema::PrimitiveNode::Make(name, repetition, TestType::type_num,
                                                   ConvertedType::NONE, FLBA_LENGTH));
    }
    node_ = schema::GroupNode::Make("schema", Repetition::REQUIRED, fields);
    schema_.Init(node_);
  }

  void GenerateData(int64_t num_values, uint32_t seed = 0);
  void SetupValuesOut(int64_t num_values);
  void SyncValuesOut();

 protected:
  schema::NodePtr node_;
  SchemaDescriptor schema_;

  // Input buffers
  std::vector<c_type> values_;

  std::vector<int16_t> def_levels_;

  std::vector<uint8_t> buffer_;
  // Pointer to the values, needed as we cannot use std::vector<bool>::data()
  c_type* values_ptr_;
  std::vector<uint8_t> bool_buffer_;

  // Output buffers
  std::vector<c_type> values_out_;
  std::vector<uint8_t> bool_buffer_out_;
  c_type* values_out_ptr_;
};

template <typename TestType>
inline void PrimitiveTypedTest<TestType>::SyncValuesOut() {}

template <>
inline void PrimitiveTypedTest<BooleanType>::SyncValuesOut() {
  std::vector<uint8_t>::const_iterator source_iterator = bool_buffer_out_.begin();
  std::vector<c_type>::iterator destination_iterator = values_out_.begin();
  while (source_iterator != bool_buffer_out_.end()) {
    *destination_iterator++ = *source_iterator++ != 0;
  }
}

template <typename TestType>
inline void PrimitiveTypedTest<TestType>::SetupValuesOut(int64_t num_values) {
  values_out_.clear();
  values_out_.resize(num_values);
  values_out_ptr_ = values_out_.data();
}

template <>
inline void PrimitiveTypedTest<BooleanType>::SetupValuesOut(int64_t num_values) {
  values_out_.clear();
  values_out_.resize(num_values);

  bool_buffer_out_.clear();
  bool_buffer_out_.resize(num_values);
  // Write once to all values so we can copy it without getting Valgrind errors
  // about uninitialised values.
  std::fill(bool_buffer_out_.begin(), bool_buffer_out_.end(), true);
  values_out_ptr_ = reinterpret_cast<bool*>(bool_buffer_out_.data());
}

template <typename TestType>
inline void PrimitiveTypedTest<TestType>::GenerateData(int64_t num_values,
                                                       uint32_t seed) {
  def_levels_.resize(num_values);
  values_.resize(num_values);

  InitValues<c_type>(static_cast<int>(num_values), seed, values_, buffer_);
  values_ptr_ = values_.data();

  std::fill(def_levels_.begin(), def_levels_.end(), 1);
}

template <>
inline void PrimitiveTypedTest<BooleanType>::GenerateData(int64_t num_values,
                                                          uint32_t seed) {
  def_levels_.resize(num_values);
  values_.resize(num_values);

  InitValues<c_type>(static_cast<int>(num_values), seed, values_, buffer_);
  bool_buffer_.resize(num_values);
  std::copy(values_.begin(), values_.end(), bool_buffer_.begin());
  values_ptr_ = reinterpret_cast<bool*>(bool_buffer_.data());

  std::fill(def_levels_.begin(), def_levels_.end(), 1);
}

}  // namespace test
}  // namespace parquet

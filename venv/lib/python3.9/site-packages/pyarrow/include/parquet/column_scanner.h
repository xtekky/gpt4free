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

#include <stdio.h>

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "parquet/column_reader.h"
#include "parquet/exception.h"
#include "parquet/platform.h"
#include "parquet/schema.h"
#include "parquet/types.h"

namespace parquet {

static constexpr int64_t DEFAULT_SCANNER_BATCH_SIZE = 128;

class PARQUET_EXPORT Scanner {
 public:
  explicit Scanner(std::shared_ptr<ColumnReader> reader,
                   int64_t batch_size = DEFAULT_SCANNER_BATCH_SIZE,
                   ::arrow::MemoryPool* pool = ::arrow::default_memory_pool())
      : batch_size_(batch_size),
        level_offset_(0),
        levels_buffered_(0),
        value_buffer_(AllocateBuffer(pool)),
        value_offset_(0),
        values_buffered_(0),
        reader_(std::move(reader)) {
    def_levels_.resize(descr()->max_definition_level() > 0 ? batch_size_ : 0);
    rep_levels_.resize(descr()->max_repetition_level() > 0 ? batch_size_ : 0);
  }

  virtual ~Scanner() {}

  static std::shared_ptr<Scanner> Make(
      std::shared_ptr<ColumnReader> col_reader,
      int64_t batch_size = DEFAULT_SCANNER_BATCH_SIZE,
      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool());

  virtual void PrintNext(std::ostream& out, int width, bool with_levels = false) = 0;

  bool HasNext() { return level_offset_ < levels_buffered_ || reader_->HasNext(); }

  const ColumnDescriptor* descr() const { return reader_->descr(); }

  int64_t batch_size() const { return batch_size_; }

  void SetBatchSize(int64_t batch_size) { batch_size_ = batch_size; }

 protected:
  int64_t batch_size_;

  std::vector<int16_t> def_levels_;
  std::vector<int16_t> rep_levels_;
  int level_offset_;
  int levels_buffered_;

  std::shared_ptr<ResizableBuffer> value_buffer_;
  int value_offset_;
  int64_t values_buffered_;
  std::shared_ptr<ColumnReader> reader_;
};

template <typename DType>
class PARQUET_TEMPLATE_CLASS_EXPORT TypedScanner : public Scanner {
 public:
  typedef typename DType::c_type T;

  explicit TypedScanner(std::shared_ptr<ColumnReader> reader,
                        int64_t batch_size = DEFAULT_SCANNER_BATCH_SIZE,
                        ::arrow::MemoryPool* pool = ::arrow::default_memory_pool())
      : Scanner(std::move(reader), batch_size, pool) {
    typed_reader_ = static_cast<TypedColumnReader<DType>*>(reader_.get());
    int value_byte_size = type_traits<DType::type_num>::value_byte_size;
    PARQUET_THROW_NOT_OK(value_buffer_->Resize(batch_size_ * value_byte_size));
    values_ = reinterpret_cast<T*>(value_buffer_->mutable_data());
  }

  virtual ~TypedScanner() {}

  bool NextLevels(int16_t* def_level, int16_t* rep_level) {
    if (level_offset_ == levels_buffered_) {
      levels_buffered_ = static_cast<int>(
          typed_reader_->ReadBatch(static_cast<int>(batch_size_), def_levels_.data(),
                                   rep_levels_.data(), values_, &values_buffered_));

      value_offset_ = 0;
      level_offset_ = 0;
      if (!levels_buffered_) {
        return false;
      }
    }
    *def_level = descr()->max_definition_level() > 0 ? def_levels_[level_offset_] : 0;
    *rep_level = descr()->max_repetition_level() > 0 ? rep_levels_[level_offset_] : 0;
    level_offset_++;
    return true;
  }

  bool Next(T* val, int16_t* def_level, int16_t* rep_level, bool* is_null) {
    if (level_offset_ == levels_buffered_) {
      if (!HasNext()) {
        // Out of data pages
        return false;
      }
    }

    NextLevels(def_level, rep_level);
    *is_null = *def_level < descr()->max_definition_level();

    if (*is_null) {
      return true;
    }

    if (value_offset_ == values_buffered_) {
      throw ParquetException("Value was non-null, but has not been buffered");
    }
    *val = values_[value_offset_++];
    return true;
  }

  // Returns true if there is a next value
  bool NextValue(T* val, bool* is_null) {
    if (level_offset_ == levels_buffered_) {
      if (!HasNext()) {
        // Out of data pages
        return false;
      }
    }

    // Out of values
    int16_t def_level = -1;
    int16_t rep_level = -1;
    NextLevels(&def_level, &rep_level);
    *is_null = def_level < descr()->max_definition_level();

    if (*is_null) {
      return true;
    }

    if (value_offset_ == values_buffered_) {
      throw ParquetException("Value was non-null, but has not been buffered");
    }
    *val = values_[value_offset_++];
    return true;
  }

  virtual void PrintNext(std::ostream& out, int width, bool with_levels = false) {
    T val{};
    int16_t def_level = -1;
    int16_t rep_level = -1;
    bool is_null = false;
    char buffer[80];

    if (!Next(&val, &def_level, &rep_level, &is_null)) {
      throw ParquetException("No more values buffered");
    }

    if (with_levels) {
      out << "  D:" << def_level << " R:" << rep_level << " ";
      if (!is_null) {
        out << "V:";
      }
    }

    if (is_null) {
      std::string null_fmt = format_fwf<ByteArrayType>(width);
      snprintf(buffer, sizeof(buffer), null_fmt.c_str(), "NULL");
    } else {
      FormatValue(&val, buffer, sizeof(buffer), width);
    }
    out << buffer;
  }

 private:
  // The ownership of this object is expressed through the reader_ variable in the base
  TypedColumnReader<DType>* typed_reader_;

  inline void FormatValue(void* val, char* buffer, int bufsize, int width);

  T* values_;
};

template <typename DType>
inline void TypedScanner<DType>::FormatValue(void* val, char* buffer, int bufsize,
                                             int width) {
  std::string fmt = format_fwf<DType>(width);
  snprintf(buffer, bufsize, fmt.c_str(), *reinterpret_cast<T*>(val));
}

template <>
inline void TypedScanner<Int96Type>::FormatValue(void* val, char* buffer, int bufsize,
                                                 int width) {
  std::string fmt = format_fwf<Int96Type>(width);
  std::string result = Int96ToString(*reinterpret_cast<Int96*>(val));
  snprintf(buffer, bufsize, fmt.c_str(), result.c_str());
}

template <>
inline void TypedScanner<ByteArrayType>::FormatValue(void* val, char* buffer, int bufsize,
                                                     int width) {
  std::string fmt = format_fwf<ByteArrayType>(width);
  std::string result = ByteArrayToString(*reinterpret_cast<ByteArray*>(val));
  snprintf(buffer, bufsize, fmt.c_str(), result.c_str());
}

template <>
inline void TypedScanner<FLBAType>::FormatValue(void* val, char* buffer, int bufsize,
                                                int width) {
  std::string fmt = format_fwf<FLBAType>(width);
  std::string result = FixedLenByteArrayToString(
      *reinterpret_cast<FixedLenByteArray*>(val), descr()->type_length());
  snprintf(buffer, bufsize, fmt.c_str(), result.c_str());
}

typedef TypedScanner<BooleanType> BoolScanner;
typedef TypedScanner<Int32Type> Int32Scanner;
typedef TypedScanner<Int64Type> Int64Scanner;
typedef TypedScanner<Int96Type> Int96Scanner;
typedef TypedScanner<FloatType> FloatScanner;
typedef TypedScanner<DoubleType> DoubleScanner;
typedef TypedScanner<ByteArrayType> ByteArrayScanner;
typedef TypedScanner<FLBAType> FixedLenByteArrayScanner;

template <typename RType>
int64_t ScanAll(int32_t batch_size, int16_t* def_levels, int16_t* rep_levels,
                uint8_t* values, int64_t* values_buffered,
                parquet::ColumnReader* reader) {
  typedef typename RType::T Type;
  auto typed_reader = static_cast<RType*>(reader);
  auto vals = reinterpret_cast<Type*>(&values[0]);
  return typed_reader->ReadBatch(batch_size, def_levels, rep_levels, vals,
                                 values_buffered);
}

int64_t PARQUET_EXPORT ScanAllValues(int32_t batch_size, int16_t* def_levels,
                                     int16_t* rep_levels, uint8_t* values,
                                     int64_t* values_buffered,
                                     parquet::ColumnReader* reader);

}  // namespace parquet

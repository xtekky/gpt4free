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
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "parquet/column_writer.h"
#include "parquet/file_writer.h"

namespace parquet {

/// \brief A class for writing Parquet files using an output stream type API.
///
/// The values given must be of the correct type i.e. the type must
/// match the file schema exactly otherwise a ParquetException will be
/// thrown.
///
/// The user must explicitly indicate the end of the row using the
/// EndRow() function or EndRow output manipulator.
///
/// A maximum row group size can be configured, the default size is
/// 512MB.  Alternatively the row group size can be set to zero and the
/// user can create new row groups by calling the EndRowGroup()
/// function or using the EndRowGroup output manipulator.
///
/// Required and optional fields are supported:
/// - Required fields are written using operator<<(T)
/// - Optional fields are written using
///   operator<<(std::optional<T>).
///
/// Note that operator<<(T) can be used to write optional fields.
///
/// Similarly, operator<<(std::optional<T>) can be used to
/// write required fields.  However if the optional parameter does not
/// have a value (i.e. it is nullopt) then a ParquetException will be
/// raised.
///
/// Currently there is no support for repeated fields.
///
class PARQUET_EXPORT StreamWriter {
 public:
  template <typename T>
  using optional = ::std::optional<T>;

  // N.B. Default constructed objects are not usable.  This
  //      constructor is provided so that the object may be move
  //      assigned afterwards.
  StreamWriter() = default;

  explicit StreamWriter(std::unique_ptr<ParquetFileWriter> writer);

  ~StreamWriter() = default;

  static void SetDefaultMaxRowGroupSize(int64_t max_size);

  void SetMaxRowGroupSize(int64_t max_size);

  int current_column() const { return column_index_; }

  int64_t current_row() const { return current_row_; }

  int num_columns() const;

  // Moving is possible.
  StreamWriter(StreamWriter&&) = default;
  StreamWriter& operator=(StreamWriter&&) = default;

  // Copying is not allowed.
  StreamWriter(const StreamWriter&) = delete;
  StreamWriter& operator=(const StreamWriter&) = delete;

  /// \brief Output operators for required fields.
  /// These can also be used for optional fields when a value must be set.
  StreamWriter& operator<<(bool v);

  StreamWriter& operator<<(int8_t v);

  StreamWriter& operator<<(uint8_t v);

  StreamWriter& operator<<(int16_t v);

  StreamWriter& operator<<(uint16_t v);

  StreamWriter& operator<<(int32_t v);

  StreamWriter& operator<<(uint32_t v);

  StreamWriter& operator<<(int64_t v);

  StreamWriter& operator<<(uint64_t v);

  StreamWriter& operator<<(const std::chrono::milliseconds& v);

  StreamWriter& operator<<(const std::chrono::microseconds& v);

  StreamWriter& operator<<(float v);

  StreamWriter& operator<<(double v);

  StreamWriter& operator<<(char v);

  /// \brief Helper class to write fixed length strings.
  /// This is useful as the standard string view (such as
  /// std::string_view) is for variable length data.
  struct PARQUET_EXPORT FixedStringView {
    FixedStringView() = default;

    explicit FixedStringView(const char* data_ptr);

    FixedStringView(const char* data_ptr, std::size_t data_len);

    const char* data{NULLPTR};
    std::size_t size{0};
  };

  /// \brief Output operators for fixed length strings.
  template <int N>
  StreamWriter& operator<<(const char (&v)[N]) {
    return WriteFixedLength(v, N);
  }
  template <std::size_t N>
  StreamWriter& operator<<(const std::array<char, N>& v) {
    return WriteFixedLength(v.data(), N);
  }
  StreamWriter& operator<<(FixedStringView v);

  /// \brief Output operators for variable length strings.
  StreamWriter& operator<<(const char* v);
  StreamWriter& operator<<(const std::string& v);
  StreamWriter& operator<<(::std::string_view v);

  /// \brief Output operator for optional fields.
  template <typename T>
  StreamWriter& operator<<(const optional<T>& v) {
    if (v) {
      return operator<<(*v);
    }
    SkipOptionalColumn();
    return *this;
  }

  /// \brief Skip the next N columns of optional data.  If there are
  /// less than N columns remaining then the excess columns are
  /// ignored.
  /// \throws ParquetException if there is an attempt to skip any
  /// required column.
  /// \return Number of columns actually skipped.
  int64_t SkipColumns(int num_columns_to_skip);

  /// \brief Terminate the current row and advance to next one.
  /// \throws ParquetException if all columns in the row were not
  /// written or skipped.
  void EndRow();

  /// \brief Terminate the current row group and create new one.
  void EndRowGroup();

 protected:
  template <typename WriterType, typename T>
  StreamWriter& Write(const T v) {
    auto writer = static_cast<WriterType*>(row_group_writer_->column(column_index_++));

    writer->WriteBatch(kBatchSizeOne, &kDefLevelOne, &kRepLevelZero, &v);

    if (max_row_group_size_ > 0) {
      row_group_size_ += writer->EstimatedBufferedValueBytes();
    }
    return *this;
  }

  StreamWriter& WriteVariableLength(const char* data_ptr, std::size_t data_len);

  StreamWriter& WriteFixedLength(const char* data_ptr, std::size_t data_len);

  void CheckColumn(Type::type physical_type, ConvertedType::type converted_type,
                   int length = -1);

  /// \brief Skip the next column which must be optional.
  /// \throws ParquetException if the next column does not exist or is
  /// not optional.
  void SkipOptionalColumn();

  void WriteNullValue(ColumnWriter* writer);

 private:
  using node_ptr_type = std::shared_ptr<schema::PrimitiveNode>;

  struct null_deleter {
    void operator()(void*) {}
  };

  int32_t column_index_{0};
  int64_t current_row_{0};
  int64_t row_group_size_{0};
  int64_t max_row_group_size_{default_row_group_size_};

  std::unique_ptr<ParquetFileWriter> file_writer_;
  std::unique_ptr<RowGroupWriter, null_deleter> row_group_writer_;
  std::vector<node_ptr_type> nodes_;

  static constexpr int16_t kDefLevelZero = 0;
  static constexpr int16_t kDefLevelOne = 1;
  static constexpr int16_t kRepLevelZero = 0;
  static constexpr int64_t kBatchSizeOne = 1;

  static int64_t default_row_group_size_;
};

struct PARQUET_EXPORT EndRowType {};
constexpr EndRowType EndRow = {};

struct PARQUET_EXPORT EndRowGroupType {};
constexpr EndRowGroupType EndRowGroup = {};

PARQUET_EXPORT
StreamWriter& operator<<(StreamWriter&, EndRowType);

PARQUET_EXPORT
StreamWriter& operator<<(StreamWriter&, EndRowGroupType);

}  // namespace parquet

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
#include <cstring>
#include <memory>

#include "parquet/exception.h"
#include "parquet/platform.h"
#include "parquet/types.h"

namespace arrow {

class Array;

namespace bit_util {
class BitWriter;
}  // namespace bit_util

namespace util {
class RleEncoder;
}  // namespace util

}  // namespace arrow

namespace parquet {

struct ArrowWriteContext;
class ColumnDescriptor;
class DataPage;
class DictionaryPage;
class ColumnChunkMetaDataBuilder;
class Encryptor;
class WriterProperties;

class PARQUET_EXPORT LevelEncoder {
 public:
  LevelEncoder();
  ~LevelEncoder();

  static int MaxBufferSize(Encoding::type encoding, int16_t max_level,
                           int num_buffered_values);

  // Initialize the LevelEncoder.
  void Init(Encoding::type encoding, int16_t max_level, int num_buffered_values,
            uint8_t* data, int data_size);

  // Encodes a batch of levels from an array and returns the number of levels encoded
  int Encode(int batch_size, const int16_t* levels);

  int32_t len() {
    if (encoding_ != Encoding::RLE) {
      throw ParquetException("Only implemented for RLE encoding");
    }
    return rle_length_;
  }

 private:
  int bit_width_;
  int rle_length_;
  Encoding::type encoding_;
  std::unique_ptr<::arrow::util::RleEncoder> rle_encoder_;
  std::unique_ptr<::arrow::bit_util::BitWriter> bit_packed_encoder_;
};

class PARQUET_EXPORT PageWriter {
 public:
  virtual ~PageWriter() {}

  static std::unique_ptr<PageWriter> Open(
      std::shared_ptr<ArrowOutputStream> sink, Compression::type codec,
      int compression_level, ColumnChunkMetaDataBuilder* metadata,
      int16_t row_group_ordinal = -1, int16_t column_chunk_ordinal = -1,
      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool(),
      bool buffered_row_group = false,
      std::shared_ptr<Encryptor> header_encryptor = NULLPTR,
      std::shared_ptr<Encryptor> data_encryptor = NULLPTR);

  // The Column Writer decides if dictionary encoding is used if set and
  // if the dictionary encoding has fallen back to default encoding on reaching dictionary
  // page limit
  virtual void Close(bool has_dictionary, bool fallback) = 0;

  // Return the number of uncompressed bytes written (including header size)
  virtual int64_t WriteDataPage(const DataPage& page) = 0;

  // Return the number of uncompressed bytes written (including header size)
  virtual int64_t WriteDictionaryPage(const DictionaryPage& page) = 0;

  virtual bool has_compressor() = 0;

  virtual void Compress(const Buffer& src_buffer, ResizableBuffer* dest_buffer) = 0;
};

static constexpr int WRITE_BATCH_SIZE = 1000;
class PARQUET_EXPORT ColumnWriter {
 public:
  virtual ~ColumnWriter() = default;

  static std::shared_ptr<ColumnWriter> Make(ColumnChunkMetaDataBuilder*,
                                            std::unique_ptr<PageWriter>,
                                            const WriterProperties* properties);

  /// \brief Closes the ColumnWriter, commits any buffered values to pages.
  /// \return Total size of the column in bytes
  virtual int64_t Close() = 0;

  /// \brief The physical Parquet type of the column
  virtual Type::type type() const = 0;

  /// \brief The schema for the column
  virtual const ColumnDescriptor* descr() const = 0;

  /// \brief The number of rows written so far
  virtual int64_t rows_written() const = 0;

  /// \brief The total size of the compressed pages + page headers. Some values
  /// might be still buffered and not written to a page yet
  virtual int64_t total_compressed_bytes() const = 0;

  /// \brief The total number of bytes written as serialized data and
  /// dictionary pages to the ColumnChunk so far
  virtual int64_t total_bytes_written() const = 0;

  /// \brief The file-level writer properties
  virtual const WriterProperties* properties() = 0;

  /// \brief Write Apache Arrow columnar data directly to ColumnWriter. Returns
  /// error status if the array data type is not compatible with the concrete
  /// writer type.
  ///
  /// leaf_array is always a primitive (possibly dictionary encoded type).
  /// Leaf_field_nullable indicates whether the leaf array is considered nullable
  /// according to its schema in a Table or its parent array.
  virtual ::arrow::Status WriteArrow(const int16_t* def_levels, const int16_t* rep_levels,
                                     int64_t num_levels, const ::arrow::Array& leaf_array,
                                     ArrowWriteContext* ctx,
                                     bool leaf_field_nullable) = 0;
};

// API to write values to a single column. This is the main client facing API.
template <typename DType>
class TypedColumnWriter : public ColumnWriter {
 public:
  using T = typename DType::c_type;

  // Write a batch of repetition levels, definition levels, and values to the
  // column.
  // `num_values` is the number of logical leaf values.
  // `def_levels` (resp. `rep_levels`) can be null if the column's max definition level
  // (resp. max repetition level) is 0.
  // If not null, each of `def_levels` and `rep_levels` must have at least
  // `num_values`.
  //
  // The number of physical values written (taken from `values`) is returned.
  // It can be smaller than `num_values` is there are some undefined values.
  virtual int64_t WriteBatch(int64_t num_values, const int16_t* def_levels,
                             const int16_t* rep_levels, const T* values) = 0;

  /// Write a batch of repetition levels, definition levels, and values to the
  /// column.
  ///
  /// In comparison to WriteBatch the length of repetition and definition levels
  /// is the same as of the number of values read for max_definition_level == 1.
  /// In the case of max_definition_level > 1, the repetition and definition
  /// levels are larger than the values but the values include the null entries
  /// with definition_level == (max_definition_level - 1). Thus we have to differentiate
  /// in the parameters of this function if the input has the length of num_values or the
  /// _number of rows in the lowest nesting level_.
  ///
  /// In the case that the most inner node in the Parquet is required, the _number of rows
  /// in the lowest nesting level_ is equal to the number of non-null values. If the
  /// inner-most schema node is optional, the _number of rows in the lowest nesting level_
  /// also includes all values with definition_level == (max_definition_level - 1).
  ///
  /// @param num_values number of levels to write.
  /// @param def_levels The Parquet definition levels, length is num_values
  /// @param rep_levels The Parquet repetition levels, length is num_values
  /// @param valid_bits Bitmap that indicates if the row is null on the lowest nesting
  ///   level. The length is number of rows in the lowest nesting level.
  /// @param valid_bits_offset The offset in bits of the valid_bits where the
  ///   first relevant bit resides.
  /// @param values The values in the lowest nested level including
  ///   spacing for nulls on the lowest levels; input has the length
  ///   of the number of rows on the lowest nesting level.
  virtual void WriteBatchSpaced(int64_t num_values, const int16_t* def_levels,
                                const int16_t* rep_levels, const uint8_t* valid_bits,
                                int64_t valid_bits_offset, const T* values) = 0;

  // Estimated size of the values that are not written to a page yet
  virtual int64_t EstimatedBufferedValueBytes() const = 0;
};

using BoolWriter = TypedColumnWriter<BooleanType>;
using Int32Writer = TypedColumnWriter<Int32Type>;
using Int64Writer = TypedColumnWriter<Int64Type>;
using Int96Writer = TypedColumnWriter<Int96Type>;
using FloatWriter = TypedColumnWriter<FloatType>;
using DoubleWriter = TypedColumnWriter<DoubleType>;
using ByteArrayWriter = TypedColumnWriter<ByteArrayType>;
using FixedLenByteArrayWriter = TypedColumnWriter<FLBAType>;

namespace internal {

/**
 * Timestamp conversion constants
 */
constexpr int64_t kJulianEpochOffsetDays = INT64_C(2440588);

template <int64_t UnitPerDay, int64_t NanosecondsPerUnit>
inline void ArrowTimestampToImpalaTimestamp(const int64_t time, Int96* impala_timestamp) {
  int64_t julian_days = (time / UnitPerDay) + kJulianEpochOffsetDays;
  (*impala_timestamp).value[2] = (uint32_t)julian_days;

  int64_t last_day_units = time % UnitPerDay;
  auto last_day_nanos = last_day_units * NanosecondsPerUnit;
  // impala_timestamp will be unaligned every other entry so do memcpy instead
  // of assign and reinterpret cast to avoid undefined behavior.
  std::memcpy(impala_timestamp, &last_day_nanos, sizeof(int64_t));
}

constexpr int64_t kSecondsInNanos = INT64_C(1000000000);

inline void SecondsToImpalaTimestamp(const int64_t seconds, Int96* impala_timestamp) {
  ArrowTimestampToImpalaTimestamp<kSecondsPerDay, kSecondsInNanos>(seconds,
                                                                   impala_timestamp);
}

constexpr int64_t kMillisecondsInNanos = kSecondsInNanos / INT64_C(1000);

inline void MillisecondsToImpalaTimestamp(const int64_t milliseconds,
                                          Int96* impala_timestamp) {
  ArrowTimestampToImpalaTimestamp<kMillisecondsPerDay, kMillisecondsInNanos>(
      milliseconds, impala_timestamp);
}

constexpr int64_t kMicrosecondsInNanos = kMillisecondsInNanos / INT64_C(1000);

inline void MicrosecondsToImpalaTimestamp(const int64_t microseconds,
                                          Int96* impala_timestamp) {
  ArrowTimestampToImpalaTimestamp<kMicrosecondsPerDay, kMicrosecondsInNanos>(
      microseconds, impala_timestamp);
}

constexpr int64_t kNanosecondsInNanos = INT64_C(1);

inline void NanosecondsToImpalaTimestamp(const int64_t nanoseconds,
                                         Int96* impala_timestamp) {
  ArrowTimestampToImpalaTimestamp<kNanosecondsPerDay, kNanosecondsInNanos>(
      nanoseconds, impala_timestamp);
}

}  // namespace internal
}  // namespace parquet

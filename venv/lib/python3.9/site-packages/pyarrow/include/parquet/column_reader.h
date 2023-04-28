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
#include <utility>
#include <vector>

#include "parquet/exception.h"
#include "parquet/level_conversion.h"
#include "parquet/metadata.h"
#include "parquet/platform.h"
#include "parquet/properties.h"
#include "parquet/schema.h"
#include "parquet/types.h"

namespace arrow {

class Array;
class ChunkedArray;

namespace bit_util {
class BitReader;
}  // namespace bit_util

namespace util {
class RleDecoder;
}  // namespace util

}  // namespace arrow

namespace parquet {

class Decryptor;
class Page;

// 16 MB is the default maximum page header size
static constexpr uint32_t kDefaultMaxPageHeaderSize = 16 * 1024 * 1024;

// 16 KB is the default expected page header size
static constexpr uint32_t kDefaultPageHeaderSize = 16 * 1024;

// \brief DataPageStats stores encoded statistics and number of values/rows for
// a page.
struct PARQUET_EXPORT DataPageStats {
  DataPageStats(const EncodedStatistics* encoded_statistics, int32_t num_values,
                std::optional<int32_t> num_rows)
      : encoded_statistics(encoded_statistics),
        num_values(num_values),
        num_rows(num_rows) {}

  // Encoded statistics extracted from the page header.
  // Nullptr if there are no statistics in the page header.
  const EncodedStatistics* encoded_statistics;
  // Number of values stored in the page. Filled for both V1 and V2 data pages.
  // For repeated fields, this can be greater than number of rows. For
  // non-repeated fields, this will be the same as the number of rows.
  int32_t num_values;
  // Number of rows stored in the page. std::nullopt if not available.
  std::optional<int32_t> num_rows;
};

class PARQUET_EXPORT LevelDecoder {
 public:
  LevelDecoder();
  ~LevelDecoder();

  // Initialize the LevelDecoder state with new data
  // and return the number of bytes consumed
  int SetData(Encoding::type encoding, int16_t max_level, int num_buffered_values,
              const uint8_t* data, int32_t data_size);

  void SetDataV2(int32_t num_bytes, int16_t max_level, int num_buffered_values,
                 const uint8_t* data);

  // Decodes a batch of levels into an array and returns the number of levels decoded
  int Decode(int batch_size, int16_t* levels);

 private:
  int bit_width_;
  int num_values_remaining_;
  Encoding::type encoding_;
  std::unique_ptr<::arrow::util::RleDecoder> rle_decoder_;
  std::unique_ptr<::arrow::bit_util::BitReader> bit_packed_decoder_;
  int16_t max_level_;
};

struct CryptoContext {
  CryptoContext(bool start_with_dictionary_page, int16_t rg_ordinal, int16_t col_ordinal,
                std::shared_ptr<Decryptor> meta, std::shared_ptr<Decryptor> data)
      : start_decrypt_with_dictionary_page(start_with_dictionary_page),
        row_group_ordinal(rg_ordinal),
        column_ordinal(col_ordinal),
        meta_decryptor(std::move(meta)),
        data_decryptor(std::move(data)) {}
  CryptoContext() {}

  bool start_decrypt_with_dictionary_page = false;
  int16_t row_group_ordinal = -1;
  int16_t column_ordinal = -1;
  std::shared_ptr<Decryptor> meta_decryptor;
  std::shared_ptr<Decryptor> data_decryptor;
};

// Abstract page iterator interface. This way, we can feed column pages to the
// ColumnReader through whatever mechanism we choose
class PARQUET_EXPORT PageReader {
  using DataPageFilter = std::function<bool(const DataPageStats&)>;

 public:
  virtual ~PageReader() = default;

  static std::unique_ptr<PageReader> Open(
      std::shared_ptr<ArrowInputStream> stream, int64_t total_num_values,
      Compression::type codec, bool always_compressed = false,
      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool(),
      const CryptoContext* ctx = NULLPTR);
  static std::unique_ptr<PageReader> Open(std::shared_ptr<ArrowInputStream> stream,
                                          int64_t total_num_values,
                                          Compression::type codec,
                                          const ReaderProperties& properties,
                                          bool always_compressed = false,
                                          const CryptoContext* ctx = NULLPTR);

  // If data_page_filter is present (not null), NextPage() will call the
  // callback function exactly once per page in the order the pages appear in
  // the column. If the callback function returns true the page will be
  // skipped. The callback will be called only if the page type is DATA_PAGE or
  // DATA_PAGE_V2. Dictionary pages will not be skipped.
  // Caller is responsible for checking that statistics are correct using
  // ApplicationVersion::HasCorrectStatistics().
  // \note API EXPERIMENTAL
  void set_data_page_filter(DataPageFilter data_page_filter) {
    data_page_filter_ = std::move(data_page_filter);
  }

  // @returns: shared_ptr<Page>(nullptr) on EOS, std::shared_ptr<Page>
  // containing new Page otherwise
  virtual std::shared_ptr<Page> NextPage() = 0;

  virtual void set_max_page_header_size(uint32_t size) = 0;

 protected:
  // Callback that decides if we should skip a page or not.
  DataPageFilter data_page_filter_;
};

class PARQUET_EXPORT ColumnReader {
 public:
  virtual ~ColumnReader() = default;

  static std::shared_ptr<ColumnReader> Make(
      const ColumnDescriptor* descr, std::unique_ptr<PageReader> pager,
      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool());

  // Returns true if there are still values in this column.
  virtual bool HasNext() = 0;

  virtual Type::type type() const = 0;

  virtual const ColumnDescriptor* descr() const = 0;

  // Get the encoding that can be exposed by this reader. If it returns
  // dictionary encoding, then ReadBatchWithDictionary can be used to read data.
  //
  // \note API EXPERIMENTAL
  virtual ExposedEncoding GetExposedEncoding() = 0;

 protected:
  friend class RowGroupReader;
  // Set the encoding that can be exposed by this reader.
  //
  // \note API EXPERIMENTAL
  virtual void SetExposedEncoding(ExposedEncoding encoding) = 0;
};

// API to read values from a single column. This is a main client facing API.
template <typename DType>
class TypedColumnReader : public ColumnReader {
 public:
  typedef typename DType::c_type T;

  // Read a batch of repetition levels, definition levels, and values from the
  // column.
  //
  // Since null values are not stored in the values, the number of values read
  // may be less than the number of repetition and definition levels. With
  // nested data this is almost certainly true.
  //
  // Set def_levels or rep_levels to nullptr if you want to skip reading them.
  // This is only safe if you know through some other source that there are no
  // undefined values.
  //
  // To fully exhaust a row group, you must read batches until the number of
  // values read reaches the number of stored values according to the metadata.
  //
  // This API is the same for both V1 and V2 of the DataPage
  //
  // @returns: actual number of levels read (see values_read for number of values read)
  virtual int64_t ReadBatch(int64_t batch_size, int16_t* def_levels, int16_t* rep_levels,
                            T* values, int64_t* values_read) = 0;

  /// Read a batch of repetition levels, definition levels, and values from the
  /// column and leave spaces for null entries on the lowest level in the values
  /// buffer.
  ///
  /// In comparison to ReadBatch the length of repetition and definition levels
  /// is the same as of the number of values read for max_definition_level == 1.
  /// In the case of max_definition_level > 1, the repetition and definition
  /// levels are larger than the values but the values include the null entries
  /// with definition_level == (max_definition_level - 1).
  ///
  /// To fully exhaust a row group, you must read batches until the number of
  /// values read reaches the number of stored values according to the metadata.
  ///
  /// @param batch_size the number of levels to read
  /// @param[out] def_levels The Parquet definition levels, output has
  ///   the length levels_read.
  /// @param[out] rep_levels The Parquet repetition levels, output has
  ///   the length levels_read.
  /// @param[out] values The values in the lowest nested level including
  ///   spacing for nulls on the lowest levels; output has the length
  ///   values_read.
  /// @param[out] valid_bits Memory allocated for a bitmap that indicates if
  ///   the row is null or on the maximum definition level. For performance
  ///   reasons the underlying buffer should be able to store 1 bit more than
  ///   required. If this requires an additional byte, this byte is only read
  ///   but never written to.
  /// @param valid_bits_offset The offset in bits of the valid_bits where the
  ///   first relevant bit resides.
  /// @param[out] levels_read The number of repetition/definition levels that were read.
  /// @param[out] values_read The number of values read, this includes all
  ///   non-null entries as well as all null-entries on the lowest level
  ///   (i.e. definition_level == max_definition_level - 1)
  /// @param[out] null_count The number of nulls on the lowest levels.
  ///   (i.e. (values_read - null_count) is total number of non-null entries)
  ///
  /// \deprecated Since 4.0.0
  ARROW_DEPRECATED("Doesn't handle nesting correctly and unused outside of unit tests.")
  virtual int64_t ReadBatchSpaced(int64_t batch_size, int16_t* def_levels,
                                  int16_t* rep_levels, T* values, uint8_t* valid_bits,
                                  int64_t valid_bits_offset, int64_t* levels_read,
                                  int64_t* values_read, int64_t* null_count) = 0;

  // Skip reading values. This method will work for both repeated and
  // non-repeated fields. Note that this method is skipping values and not
  // records. This distinction is important for repeated fields, meaning that
  // we are not skipping over the values to the next record. For example,
  // consider the following two consecutive records containing one repeated field:
  // {[1, 2, 3]}, {[4, 5]}. If we Skip(2), our next read value will be 3, which
  // is inside the first record.
  // Returns the number of values skipped.
  virtual int64_t Skip(int64_t num_values_to_skip) = 0;

  // Read a batch of repetition levels, definition levels, and indices from the
  // column. And read the dictionary if a dictionary page is encountered during
  // reading pages. This API is similar to ReadBatch(), with ability to read
  // dictionary and indices. It is only valid to call this method  when the reader can
  // expose dictionary encoding. (i.e., the reader's GetExposedEncoding() returns
  // DICTIONARY).
  //
  // The dictionary is read along with the data page. When there's no data page,
  // the dictionary won't be returned.
  //
  // @param batch_size The batch size to read
  // @param[out] def_levels The Parquet definition levels.
  // @param[out] rep_levels The Parquet repetition levels.
  // @param[out] indices The dictionary indices.
  // @param[out] indices_read The number of indices read.
  // @param[out] dict The pointer to dictionary values. It will return nullptr if
  // there's no data page. Each column chunk only has one dictionary page. The dictionary
  // is owned by the reader, so the caller is responsible for copying the dictionary
  // values before the reader gets destroyed.
  // @param[out] dict_len The dictionary length. It will return 0 if there's no data
  // page.
  // @returns: actual number of levels read (see indices_read for number of
  // indices read
  //
  // \note API EXPERIMENTAL
  virtual int64_t ReadBatchWithDictionary(int64_t batch_size, int16_t* def_levels,
                                          int16_t* rep_levels, int32_t* indices,
                                          int64_t* indices_read, const T** dict,
                                          int32_t* dict_len) = 0;
};

namespace internal {

/// \brief Stateful column reader that delimits semantic records for both flat
/// and nested columns
///
/// \note API EXPERIMENTAL
/// \since 1.3.0
class PARQUET_EXPORT RecordReader {
 public:
  static std::shared_ptr<RecordReader> Make(
      const ColumnDescriptor* descr, LevelInfo leaf_info,
      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool(),
      const bool read_dictionary = false);

  virtual ~RecordReader() = default;

  /// \brief Attempt to read indicated number of records from column chunk
  /// Note that for repeated fields, a record may have more than one value
  /// and all of them are read.
  /// \return number of records read
  virtual int64_t ReadRecords(int64_t num_records) = 0;

  /// \brief Attempt to skip indicated number of records from column chunk.
  /// Note that for repeated fields, a record may have more than one value
  /// and all of them are skipped.
  /// \return number of records skipped
  virtual int64_t SkipRecords(int64_t num_records) = 0;

  /// \brief Pre-allocate space for data. Results in better flat read performance
  virtual void Reserve(int64_t num_values) = 0;

  /// \brief Clear consumed values and repetition/definition levels as the
  /// result of calling ReadRecords
  virtual void Reset() = 0;

  /// \brief Transfer filled values buffer to caller. A new one will be
  /// allocated in subsequent ReadRecords calls
  virtual std::shared_ptr<ResizableBuffer> ReleaseValues() = 0;

  /// \brief Transfer filled validity bitmap buffer to caller. A new one will
  /// be allocated in subsequent ReadRecords calls
  virtual std::shared_ptr<ResizableBuffer> ReleaseIsValid() = 0;

  /// \brief Return true if the record reader has more internal data yet to
  /// process
  virtual bool HasMoreData() const = 0;

  /// \brief Advance record reader to the next row group. Must be set before
  /// any records could be read/skipped.
  /// \param[in] reader obtained from RowGroupReader::GetColumnPageReader
  virtual void SetPageReader(std::unique_ptr<PageReader> reader) = 0;

  virtual void DebugPrintState() = 0;

  /// \brief Decoded definition levels
  int16_t* def_levels() const {
    return reinterpret_cast<int16_t*>(def_levels_->mutable_data());
  }

  /// \brief Decoded repetition levels
  int16_t* rep_levels() const {
    return reinterpret_cast<int16_t*>(rep_levels_->mutable_data());
  }

  /// \brief Decoded values, including nulls, if any
  uint8_t* values() const { return values_->mutable_data(); }

  /// \brief Number of values written including nulls (if any)
  /// There is no read-ahead/buffering for values.
  int64_t values_written() const { return values_written_; }

  /// \brief Number of definition / repetition levels (from those that have
  /// been decoded) that have been consumed inside the reader.
  int64_t levels_position() const { return levels_position_; }

  /// \brief Number of definition / repetition levels that have been written
  /// internally in the reader. This may be larger than values_written() because
  /// for repeated fields we need to look at the levels in advance to figure out
  /// the record boundaries.
  int64_t levels_written() const { return levels_written_; }

  /// \brief Number of nulls in the leaf that we have read so far.
  int64_t null_count() const { return null_count_; }

  /// \brief True if the leaf values are nullable
  bool nullable_values() const { return nullable_values_; }

  /// \brief True if reading directly as Arrow dictionary-encoded
  bool read_dictionary() const { return read_dictionary_; }

 protected:
  /// \brief Indicates if we can have nullable values.
  bool nullable_values_;

  bool at_record_start_;
  int64_t records_read_;

  /// \brief Stores values. These values are populated based on each ReadRecords
  /// call. No extra values are buffered for the next call. SkipRecords will not
  /// add any value to this buffer.
  std::shared_ptr<::arrow::ResizableBuffer> values_;
  /// \brief False for BYTE_ARRAY, in which case we don't allocate the values
  /// buffer and we directly read into builder classes.
  bool uses_values_;

  /// \brief Values that we have read into 'values_' + 'null_count_'.
  int64_t values_written_;
  int64_t values_capacity_;
  int64_t null_count_;

  /// \brief Each bit corresponds to one element in 'values_' and specifies if it
  /// is null or not null.
  std::shared_ptr<::arrow::ResizableBuffer> valid_bits_;

  /// \brief Buffer for definition levels. May contain more levels than
  /// is actually read. This is because we read levels ahead to
  /// figure out record boundaries for repeated fields.
  /// For flat required fields, 'def_levels_' and 'rep_levels_' are not
  ///  populated. For non-repeated fields 'rep_levels_' is not populated.
  /// 'def_levels_' and 'rep_levels_' must be of the same size if present.
  std::shared_ptr<::arrow::ResizableBuffer> def_levels_;
  /// \brief Buffer for repetition levels. Only populated for repeated
  /// fields.
  std::shared_ptr<::arrow::ResizableBuffer> rep_levels_;

  /// \brief Number of definition / repetition levels that have been written
  /// internally in the reader. This may be larger than values_written() since
  /// for repeated fields we need to look at the levels in advance to figure out
  /// the record boundaries.
  int64_t levels_written_;
  /// \brief Position of the next level that should be consumed.
  int64_t levels_position_;
  int64_t levels_capacity_;

  bool read_dictionary_ = false;
};

class BinaryRecordReader : virtual public RecordReader {
 public:
  virtual std::vector<std::shared_ptr<::arrow::Array>> GetBuilderChunks() = 0;
};

/// \brief Read records directly to dictionary-encoded Arrow form (int32
/// indices). Only valid for BYTE_ARRAY columns
class DictionaryRecordReader : virtual public RecordReader {
 public:
  virtual std::shared_ptr<::arrow::ChunkedArray> GetResult() = 0;
};

}  // namespace internal

using BoolReader = TypedColumnReader<BooleanType>;
using Int32Reader = TypedColumnReader<Int32Type>;
using Int64Reader = TypedColumnReader<Int64Type>;
using Int96Reader = TypedColumnReader<Int96Type>;
using FloatReader = TypedColumnReader<FloatType>;
using DoubleReader = TypedColumnReader<DoubleType>;
using ByteArrayReader = TypedColumnReader<ByteArrayType>;
using FixedLenByteArrayReader = TypedColumnReader<FLBAType>;

}  // namespace parquet

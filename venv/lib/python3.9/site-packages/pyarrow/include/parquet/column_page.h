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

#include <cstdint>
#include <memory>
#include <string>

#include "parquet/statistics.h"
#include "parquet/types.h"

namespace parquet {

// TODO: Parallel processing is not yet safe because of memory-ownership
// semantics (the PageReader may or may not own the memory referenced by a
// page)
//
// TODO(wesm): In the future Parquet implementations may store the crc code
// in format::PageHeader. parquet-mr currently does not, so we also skip it
// here, both on the read and write path
class Page {
 public:
  Page(const std::shared_ptr<Buffer>& buffer, PageType::type type)
      : buffer_(buffer), type_(type) {}

  PageType::type type() const { return type_; }

  std::shared_ptr<Buffer> buffer() const { return buffer_; }

  // @returns: a pointer to the page's data
  const uint8_t* data() const { return buffer_->data(); }

  // @returns: the total size in bytes of the page's data buffer
  int32_t size() const { return static_cast<int32_t>(buffer_->size()); }

 private:
  std::shared_ptr<Buffer> buffer_;
  PageType::type type_;
};

/// \brief Base type for DataPageV1 and DataPageV2 including common attributes
class DataPage : public Page {
 public:
  int32_t num_values() const { return num_values_; }
  Encoding::type encoding() const { return encoding_; }
  int64_t uncompressed_size() const { return uncompressed_size_; }
  const EncodedStatistics& statistics() const { return statistics_; }

  virtual ~DataPage() = default;

 protected:
  DataPage(PageType::type type, const std::shared_ptr<Buffer>& buffer, int32_t num_values,
           Encoding::type encoding, int64_t uncompressed_size,
           const EncodedStatistics& statistics = EncodedStatistics())
      : Page(buffer, type),
        num_values_(num_values),
        encoding_(encoding),
        uncompressed_size_(uncompressed_size),
        statistics_(statistics) {}

  int32_t num_values_;
  Encoding::type encoding_;
  int64_t uncompressed_size_;
  EncodedStatistics statistics_;
};

class DataPageV1 : public DataPage {
 public:
  DataPageV1(const std::shared_ptr<Buffer>& buffer, int32_t num_values,
             Encoding::type encoding, Encoding::type definition_level_encoding,
             Encoding::type repetition_level_encoding, int64_t uncompressed_size,
             const EncodedStatistics& statistics = EncodedStatistics())
      : DataPage(PageType::DATA_PAGE, buffer, num_values, encoding, uncompressed_size,
                 statistics),
        definition_level_encoding_(definition_level_encoding),
        repetition_level_encoding_(repetition_level_encoding) {}

  Encoding::type repetition_level_encoding() const { return repetition_level_encoding_; }

  Encoding::type definition_level_encoding() const { return definition_level_encoding_; }

 private:
  Encoding::type definition_level_encoding_;
  Encoding::type repetition_level_encoding_;
};

class DataPageV2 : public DataPage {
 public:
  DataPageV2(const std::shared_ptr<Buffer>& buffer, int32_t num_values, int32_t num_nulls,
             int32_t num_rows, Encoding::type encoding,
             int32_t definition_levels_byte_length, int32_t repetition_levels_byte_length,
             int64_t uncompressed_size, bool is_compressed = false,
             const EncodedStatistics& statistics = EncodedStatistics())
      : DataPage(PageType::DATA_PAGE_V2, buffer, num_values, encoding, uncompressed_size,
                 statistics),
        num_nulls_(num_nulls),
        num_rows_(num_rows),
        definition_levels_byte_length_(definition_levels_byte_length),
        repetition_levels_byte_length_(repetition_levels_byte_length),
        is_compressed_(is_compressed) {}

  int32_t num_nulls() const { return num_nulls_; }

  int32_t num_rows() const { return num_rows_; }

  int32_t definition_levels_byte_length() const { return definition_levels_byte_length_; }

  int32_t repetition_levels_byte_length() const { return repetition_levels_byte_length_; }

  bool is_compressed() const { return is_compressed_; }

 private:
  int32_t num_nulls_;
  int32_t num_rows_;
  int32_t definition_levels_byte_length_;
  int32_t repetition_levels_byte_length_;
  bool is_compressed_;
};

class DictionaryPage : public Page {
 public:
  DictionaryPage(const std::shared_ptr<Buffer>& buffer, int32_t num_values,
                 Encoding::type encoding, bool is_sorted = false)
      : Page(buffer, PageType::DICTIONARY_PAGE),
        num_values_(num_values),
        encoding_(encoding),
        is_sorted_(is_sorted) {}

  int32_t num_values() const { return num_values_; }

  Encoding::type encoding() const { return encoding_; }

  bool is_sorted() const { return is_sorted_; }

 private:
  int32_t num_values_;
  Encoding::type encoding_;
  bool is_sorted_;
};

}  // namespace parquet

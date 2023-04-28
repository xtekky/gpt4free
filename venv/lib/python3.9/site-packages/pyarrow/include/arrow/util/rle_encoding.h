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

// Imported from Apache Impala (incubating) on 2016-01-29 and modified for use
// in parquet-cpp, Arrow

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "arrow/util/bit_block_counter.h"
#include "arrow/util/bit_run_reader.h"
#include "arrow/util/bit_stream_utils.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/macros.h"

namespace arrow {
namespace util {

/// Utility classes to do run length encoding (RLE) for fixed bit width values.  If runs
/// are sufficiently long, RLE is used, otherwise, the values are just bit-packed
/// (literal encoding).
/// For both types of runs, there is a byte-aligned indicator which encodes the length
/// of the run and the type of the run.
/// This encoding has the benefit that when there aren't any long enough runs, values
/// are always decoded at fixed (can be precomputed) bit offsets OR both the value and
/// the run length are byte aligned. This allows for very efficient decoding
/// implementations.
/// The encoding is:
///    encoded-block := run*
///    run := literal-run | repeated-run
///    literal-run := literal-indicator < literal bytes >
///    repeated-run := repeated-indicator < repeated value. padded to byte boundary >
///    literal-indicator := varint_encode( number_of_groups << 1 | 1)
///    repeated-indicator := varint_encode( number_of_repetitions << 1 )
//
/// Each run is preceded by a varint. The varint's least significant bit is
/// used to indicate whether the run is a literal run or a repeated run. The rest
/// of the varint is used to determine the length of the run (eg how many times the
/// value repeats).
//
/// In the case of literal runs, the run length is always a multiple of 8 (i.e. encode
/// in groups of 8), so that no matter the bit-width of the value, the sequence will end
/// on a byte boundary without padding.
/// Given that we know it is a multiple of 8, we store the number of 8-groups rather than
/// the actual number of encoded ints. (This means that the total number of encoded values
/// can not be determined from the encoded data, since the number of values in the last
/// group may not be a multiple of 8). For the last group of literal runs, we pad
/// the group to 8 with zeros. This allows for 8 at a time decoding on the read side
/// without the need for additional checks.
//
/// There is a break-even point when it is more storage efficient to do run length
/// encoding.  For 1 bit-width values, that point is 8 values.  They require 2 bytes
/// for both the repeated encoding or the literal encoding.  This value can always
/// be computed based on the bit-width.
/// TODO: think about how to use this for strings.  The bit packing isn't quite the same.
//
/// Examples with bit-width 1 (eg encoding booleans):
/// ----------------------------------------
/// 100 1s followed by 100 0s:
/// <varint(100 << 1)> <1, padded to 1 byte> <varint(100 << 1)> <0, padded to 1 byte>
///  - (total 4 bytes)
//
/// alternating 1s and 0s (200 total):
/// 200 ints = 25 groups of 8
/// <varint((25 << 1) | 1)> <25 bytes of values, bitpacked>
/// (total 26 bytes, 1 byte overhead)
//

/// Decoder class for RLE encoded data.
class RleDecoder {
 public:
  /// Create a decoder object. buffer/buffer_len is the decoded data.
  /// bit_width is the width of each value (before encoding).
  RleDecoder(const uint8_t* buffer, int buffer_len, int bit_width)
      : bit_reader_(buffer, buffer_len),
        bit_width_(bit_width),
        current_value_(0),
        repeat_count_(0),
        literal_count_(0) {
    DCHECK_GE(bit_width_, 0);
    DCHECK_LE(bit_width_, 64);
  }

  RleDecoder() : bit_width_(-1) {}

  void Reset(const uint8_t* buffer, int buffer_len, int bit_width) {
    DCHECK_GE(bit_width, 0);
    DCHECK_LE(bit_width, 64);
    bit_reader_.Reset(buffer, buffer_len);
    bit_width_ = bit_width;
    current_value_ = 0;
    repeat_count_ = 0;
    literal_count_ = 0;
  }

  /// Gets the next value.  Returns false if there are no more.
  template <typename T>
  bool Get(T* val);

  /// Gets a batch of values.  Returns the number of decoded elements.
  template <typename T>
  int GetBatch(T* values, int batch_size);

  /// Like GetBatch but add spacing for null entries
  template <typename T>
  int GetBatchSpaced(int batch_size, int null_count, const uint8_t* valid_bits,
                     int64_t valid_bits_offset, T* out);

  /// Like GetBatch but the values are then decoded using the provided dictionary
  template <typename T>
  int GetBatchWithDict(const T* dictionary, int32_t dictionary_length, T* values,
                       int batch_size);

  /// Like GetBatchWithDict but add spacing for null entries
  ///
  /// Null entries will be zero-initialized in `values` to avoid leaking
  /// private data.
  template <typename T>
  int GetBatchWithDictSpaced(const T* dictionary, int32_t dictionary_length, T* values,
                             int batch_size, int null_count, const uint8_t* valid_bits,
                             int64_t valid_bits_offset);

 protected:
  ::arrow::bit_util::BitReader bit_reader_;
  /// Number of bits needed to encode the value. Must be between 0 and 64.
  int bit_width_;
  uint64_t current_value_;
  int32_t repeat_count_;
  int32_t literal_count_;

 private:
  /// Fills literal_count_ and repeat_count_ with next values. Returns false if there
  /// are no more.
  template <typename T>
  bool NextCounts();

  /// Utility methods for retrieving spaced values.
  template <typename T, typename RunType, typename Converter>
  int GetSpaced(Converter converter, int batch_size, int null_count,
                const uint8_t* valid_bits, int64_t valid_bits_offset, T* out);
};

/// Class to incrementally build the rle data.   This class does not allocate any memory.
/// The encoding has two modes: encoding repeated runs and literal runs.
/// If the run is sufficiently short, it is more efficient to encode as a literal run.
/// This class does so by buffering 8 values at a time.  If they are not all the same
/// they are added to the literal run.  If they are the same, they are added to the
/// repeated run.  When we switch modes, the previous run is flushed out.
class RleEncoder {
 public:
  /// buffer/buffer_len: preallocated output buffer.
  /// bit_width: max number of bits for value.
  /// TODO: consider adding a min_repeated_run_length so the caller can control
  /// when values should be encoded as repeated runs.  Currently this is derived
  /// based on the bit_width, which can determine a storage optimal choice.
  /// TODO: allow 0 bit_width (and have dict encoder use it)
  RleEncoder(uint8_t* buffer, int buffer_len, int bit_width)
      : bit_width_(bit_width), bit_writer_(buffer, buffer_len) {
    DCHECK_GE(bit_width_, 0);
    DCHECK_LE(bit_width_, 64);
    max_run_byte_size_ = MinBufferSize(bit_width);
    DCHECK_GE(buffer_len, max_run_byte_size_) << "Input buffer not big enough.";
    Clear();
  }

  /// Returns the minimum buffer size needed to use the encoder for 'bit_width'
  /// This is the maximum length of a single run for 'bit_width'.
  /// It is not valid to pass a buffer less than this length.
  static int MinBufferSize(int bit_width) {
    /// 1 indicator byte and MAX_VALUES_PER_LITERAL_RUN 'bit_width' values.
    int max_literal_run_size = 1 + static_cast<int>(::arrow::bit_util::BytesForBits(
                                       MAX_VALUES_PER_LITERAL_RUN * bit_width));
    /// Up to kMaxVlqByteLength indicator and a single 'bit_width' value.
    int max_repeated_run_size =
        ::arrow::bit_util::BitReader::kMaxVlqByteLength +
        static_cast<int>(::arrow::bit_util::BytesForBits(bit_width));
    return std::max(max_literal_run_size, max_repeated_run_size);
  }

  /// Returns the maximum byte size it could take to encode 'num_values'.
  static int MaxBufferSize(int bit_width, int num_values) {
    // For a bit_width > 1, the worst case is the repetition of "literal run of length 8
    // and then a repeated run of length 8".
    // 8 values per smallest run, 8 bits per byte
    int bytes_per_run = bit_width;
    int num_runs = static_cast<int>(::arrow::bit_util::CeilDiv(num_values, 8));
    int literal_max_size = num_runs + num_runs * bytes_per_run;

    // In the very worst case scenario, the data is a concatenation of repeated
    // runs of 8 values. Repeated run has a 1 byte varint followed by the
    // bit-packed repeated value
    int min_repeated_run_size =
        1 + static_cast<int>(::arrow::bit_util::BytesForBits(bit_width));
    int repeated_max_size = static_cast<int>(::arrow::bit_util::CeilDiv(num_values, 8)) *
                            min_repeated_run_size;

    return std::max(literal_max_size, repeated_max_size);
  }

  /// Encode value.  Returns true if the value fits in buffer, false otherwise.
  /// This value must be representable with bit_width_ bits.
  bool Put(uint64_t value);

  /// Flushes any pending values to the underlying buffer.
  /// Returns the total number of bytes written
  int Flush();

  /// Resets all the state in the encoder.
  void Clear();

  /// Returns pointer to underlying buffer
  uint8_t* buffer() { return bit_writer_.buffer(); }
  int32_t len() { return bit_writer_.bytes_written(); }

 private:
  /// Flushes any buffered values.  If this is part of a repeated run, this is largely
  /// a no-op.
  /// If it is part of a literal run, this will call FlushLiteralRun, which writes
  /// out the buffered literal values.
  /// If 'done' is true, the current run would be written even if it would normally
  /// have been buffered more.  This should only be called at the end, when the
  /// encoder has received all values even if it would normally continue to be
  /// buffered.
  void FlushBufferedValues(bool done);

  /// Flushes literal values to the underlying buffer.  If update_indicator_byte,
  /// then the current literal run is complete and the indicator byte is updated.
  void FlushLiteralRun(bool update_indicator_byte);

  /// Flushes a repeated run to the underlying buffer.
  void FlushRepeatedRun();

  /// Checks and sets buffer_full_. This must be called after flushing a run to
  /// make sure there are enough bytes remaining to encode the next run.
  void CheckBufferFull();

  /// The maximum number of values in a single literal run
  /// (number of groups encodable by a 1-byte indicator * 8)
  static const int MAX_VALUES_PER_LITERAL_RUN = (1 << 6) * 8;

  /// Number of bits needed to encode the value. Must be between 0 and 64.
  const int bit_width_;

  /// Underlying buffer.
  ::arrow::bit_util::BitWriter bit_writer_;

  /// If true, the buffer is full and subsequent Put()'s will fail.
  bool buffer_full_;

  /// The maximum byte size a single run can take.
  int max_run_byte_size_;

  /// We need to buffer at most 8 values for literals.  This happens when the
  /// bit_width is 1 (so 8 values fit in one byte).
  /// TODO: generalize this to other bit widths
  int64_t buffered_values_[8];

  /// Number of values in buffered_values_
  int num_buffered_values_;

  /// The current (also last) value that was written and the count of how
  /// many times in a row that value has been seen.  This is maintained even
  /// if we are in a literal run.  If the repeat_count_ get high enough, we switch
  /// to encoding repeated runs.
  uint64_t current_value_;
  int repeat_count_;

  /// Number of literals in the current run.  This does not include the literals
  /// that might be in buffered_values_.  Only after we've got a group big enough
  /// can we decide if they should part of the literal_count_ or repeat_count_
  int literal_count_;

  /// Pointer to a byte in the underlying buffer that stores the indicator byte.
  /// This is reserved as soon as we need a literal run but the value is written
  /// when the literal run is complete.
  uint8_t* literal_indicator_byte_;
};

template <typename T>
inline bool RleDecoder::Get(T* val) {
  return GetBatch(val, 1) == 1;
}

template <typename T>
inline int RleDecoder::GetBatch(T* values, int batch_size) {
  DCHECK_GE(bit_width_, 0);
  int values_read = 0;

  auto* out = values;

  while (values_read < batch_size) {
    int remaining = batch_size - values_read;

    if (repeat_count_ > 0) {  // Repeated value case.
      int repeat_batch = std::min(remaining, repeat_count_);
      std::fill(out, out + repeat_batch, static_cast<T>(current_value_));

      repeat_count_ -= repeat_batch;
      values_read += repeat_batch;
      out += repeat_batch;
    } else if (literal_count_ > 0) {
      int literal_batch = std::min(remaining, literal_count_);
      int actual_read = bit_reader_.GetBatch(bit_width_, out, literal_batch);
      if (actual_read != literal_batch) {
        return values_read;
      }

      literal_count_ -= literal_batch;
      values_read += literal_batch;
      out += literal_batch;
    } else {
      if (!NextCounts<T>()) return values_read;
    }
  }

  return values_read;
}

template <typename T, typename RunType, typename Converter>
inline int RleDecoder::GetSpaced(Converter converter, int batch_size, int null_count,
                                 const uint8_t* valid_bits, int64_t valid_bits_offset,
                                 T* out) {
  if (ARROW_PREDICT_FALSE(null_count == batch_size)) {
    converter.FillZero(out, out + batch_size);
    return batch_size;
  }

  DCHECK_GE(bit_width_, 0);
  int values_read = 0;
  int values_remaining = batch_size - null_count;

  // Assume no bits to start.
  arrow::internal::BitRunReader bit_reader(valid_bits, valid_bits_offset,
                                           /*length=*/batch_size);
  arrow::internal::BitRun valid_run = bit_reader.NextRun();
  while (values_read < batch_size) {
    if (ARROW_PREDICT_FALSE(valid_run.length == 0)) {
      valid_run = bit_reader.NextRun();
    }

    DCHECK_GT(batch_size, 0);
    DCHECK_GT(valid_run.length, 0);

    if (valid_run.set) {
      if ((repeat_count_ == 0) && (literal_count_ == 0)) {
        if (!NextCounts<RunType>()) return values_read;
        DCHECK((repeat_count_ > 0) ^ (literal_count_ > 0));
      }

      if (repeat_count_ > 0) {
        int repeat_batch = 0;
        // Consume the entire repeat counts incrementing repeat_batch to
        // be the total of nulls + values consumed, we only need to
        // get the total count because we can fill in the same value for
        // nulls and non-nulls. This proves to be a big efficiency win.
        while (repeat_count_ > 0 && (values_read + repeat_batch) < batch_size) {
          DCHECK_GT(valid_run.length, 0);
          if (valid_run.set) {
            int update_size = std::min(static_cast<int>(valid_run.length), repeat_count_);
            repeat_count_ -= update_size;
            repeat_batch += update_size;
            valid_run.length -= update_size;
            values_remaining -= update_size;
          } else {
            // We can consume all nulls here because we would do so on
            //  the next loop anyways.
            repeat_batch += static_cast<int>(valid_run.length);
            valid_run.length = 0;
          }
          if (valid_run.length == 0) {
            valid_run = bit_reader.NextRun();
          }
        }
        RunType current_value = static_cast<RunType>(current_value_);
        if (ARROW_PREDICT_FALSE(!converter.IsValid(current_value))) {
          return values_read;
        }
        converter.Fill(out, out + repeat_batch, current_value);
        out += repeat_batch;
        values_read += repeat_batch;
      } else if (literal_count_ > 0) {
        int literal_batch = std::min(values_remaining, literal_count_);
        DCHECK_GT(literal_batch, 0);

        // Decode the literals
        constexpr int kBufferSize = 1024;
        RunType indices[kBufferSize];
        literal_batch = std::min(literal_batch, kBufferSize);
        int actual_read = bit_reader_.GetBatch(bit_width_, indices, literal_batch);
        if (ARROW_PREDICT_FALSE(actual_read != literal_batch)) {
          return values_read;
        }
        if (!converter.IsValid(indices, /*length=*/actual_read)) {
          return values_read;
        }
        int skipped = 0;
        int literals_read = 0;
        while (literals_read < literal_batch) {
          if (valid_run.set) {
            int update_size = std::min(literal_batch - literals_read,
                                       static_cast<int>(valid_run.length));
            converter.Copy(out, indices + literals_read, update_size);
            literals_read += update_size;
            out += update_size;
            valid_run.length -= update_size;
          } else {
            converter.FillZero(out, out + valid_run.length);
            out += valid_run.length;
            skipped += static_cast<int>(valid_run.length);
            valid_run.length = 0;
          }
          if (valid_run.length == 0) {
            valid_run = bit_reader.NextRun();
          }
        }
        literal_count_ -= literal_batch;
        values_remaining -= literal_batch;
        values_read += literal_batch + skipped;
      }
    } else {
      converter.FillZero(out, out + valid_run.length);
      out += valid_run.length;
      values_read += static_cast<int>(valid_run.length);
      valid_run.length = 0;
    }
  }
  DCHECK_EQ(valid_run.length, 0);
  DCHECK_EQ(values_remaining, 0);
  return values_read;
}

// Converter for GetSpaced that handles runs that get returned
// directly as output.
template <typename T>
struct PlainRleConverter {
  T kZero = {};
  inline bool IsValid(const T& values) const { return true; }
  inline bool IsValid(const T* values, int32_t length) const { return true; }
  inline void Fill(T* begin, T* end, const T& run_value) const {
    std::fill(begin, end, run_value);
  }
  inline void FillZero(T* begin, T* end) { std::fill(begin, end, kZero); }
  inline void Copy(T* out, const T* values, int length) const {
    std::memcpy(out, values, length * sizeof(T));
  }
};

template <typename T>
inline int RleDecoder::GetBatchSpaced(int batch_size, int null_count,
                                      const uint8_t* valid_bits,
                                      int64_t valid_bits_offset, T* out) {
  if (null_count == 0) {
    return GetBatch<T>(out, batch_size);
  }

  PlainRleConverter<T> converter;
  arrow::internal::BitBlockCounter block_counter(valid_bits, valid_bits_offset,
                                                 batch_size);

  int total_processed = 0;
  int processed = 0;
  arrow::internal::BitBlockCount block;

  do {
    block = block_counter.NextFourWords();
    if (block.length == 0) {
      break;
    }
    if (block.AllSet()) {
      processed = GetBatch<T>(out, block.length);
    } else if (block.NoneSet()) {
      converter.FillZero(out, out + block.length);
      processed = block.length;
    } else {
      processed = GetSpaced<T, /*RunType=*/T, PlainRleConverter<T>>(
          converter, block.length, block.length - block.popcount, valid_bits,
          valid_bits_offset, out);
    }
    total_processed += processed;
    out += block.length;
    valid_bits_offset += block.length;
  } while (processed == block.length);
  return total_processed;
}

static inline bool IndexInRange(int32_t idx, int32_t dictionary_length) {
  return idx >= 0 && idx < dictionary_length;
}

// Converter for GetSpaced that handles runs of returned dictionary
// indices.
template <typename T>
struct DictionaryConverter {
  T kZero = {};
  const T* dictionary;
  int32_t dictionary_length;

  inline bool IsValid(int32_t value) { return IndexInRange(value, dictionary_length); }

  inline bool IsValid(const int32_t* values, int32_t length) const {
    using IndexType = int32_t;
    IndexType min_index = std::numeric_limits<IndexType>::max();
    IndexType max_index = std::numeric_limits<IndexType>::min();
    for (int x = 0; x < length; x++) {
      min_index = std::min(values[x], min_index);
      max_index = std::max(values[x], max_index);
    }

    return IndexInRange(min_index, dictionary_length) &&
           IndexInRange(max_index, dictionary_length);
  }
  inline void Fill(T* begin, T* end, const int32_t& run_value) const {
    std::fill(begin, end, dictionary[run_value]);
  }
  inline void FillZero(T* begin, T* end) { std::fill(begin, end, kZero); }

  inline void Copy(T* out, const int32_t* values, int length) const {
    for (int x = 0; x < length; x++) {
      out[x] = dictionary[values[x]];
    }
  }
};

template <typename T>
inline int RleDecoder::GetBatchWithDict(const T* dictionary, int32_t dictionary_length,
                                        T* values, int batch_size) {
  // Per https://github.com/apache/parquet-format/blob/master/Encodings.md,
  // the maximum dictionary index width in Parquet is 32 bits.
  using IndexType = int32_t;
  DictionaryConverter<T> converter;
  converter.dictionary = dictionary;
  converter.dictionary_length = dictionary_length;

  DCHECK_GE(bit_width_, 0);
  int values_read = 0;

  auto* out = values;

  while (values_read < batch_size) {
    int remaining = batch_size - values_read;

    if (repeat_count_ > 0) {
      auto idx = static_cast<IndexType>(current_value_);
      if (ARROW_PREDICT_FALSE(!IndexInRange(idx, dictionary_length))) {
        return values_read;
      }
      T val = dictionary[idx];

      int repeat_batch = std::min(remaining, repeat_count_);
      std::fill(out, out + repeat_batch, val);

      /* Upkeep counters */
      repeat_count_ -= repeat_batch;
      values_read += repeat_batch;
      out += repeat_batch;
    } else if (literal_count_ > 0) {
      constexpr int kBufferSize = 1024;
      IndexType indices[kBufferSize];

      int literal_batch = std::min(remaining, literal_count_);
      literal_batch = std::min(literal_batch, kBufferSize);

      int actual_read = bit_reader_.GetBatch(bit_width_, indices, literal_batch);
      if (ARROW_PREDICT_FALSE(actual_read != literal_batch)) {
        return values_read;
      }
      if (ARROW_PREDICT_FALSE(!converter.IsValid(indices, /*length=*/literal_batch))) {
        return values_read;
      }
      converter.Copy(out, indices, literal_batch);

      /* Upkeep counters */
      literal_count_ -= literal_batch;
      values_read += literal_batch;
      out += literal_batch;
    } else {
      if (!NextCounts<IndexType>()) return values_read;
    }
  }

  return values_read;
}

template <typename T>
inline int RleDecoder::GetBatchWithDictSpaced(const T* dictionary,
                                              int32_t dictionary_length, T* out,
                                              int batch_size, int null_count,
                                              const uint8_t* valid_bits,
                                              int64_t valid_bits_offset) {
  if (null_count == 0) {
    return GetBatchWithDict<T>(dictionary, dictionary_length, out, batch_size);
  }
  arrow::internal::BitBlockCounter block_counter(valid_bits, valid_bits_offset,
                                                 batch_size);
  using IndexType = int32_t;
  DictionaryConverter<T> converter;
  converter.dictionary = dictionary;
  converter.dictionary_length = dictionary_length;

  int total_processed = 0;
  int processed = 0;
  arrow::internal::BitBlockCount block;
  do {
    block = block_counter.NextFourWords();
    if (block.length == 0) {
      break;
    }
    if (block.AllSet()) {
      processed = GetBatchWithDict<T>(dictionary, dictionary_length, out, block.length);
    } else if (block.NoneSet()) {
      converter.FillZero(out, out + block.length);
      processed = block.length;
    } else {
      processed = GetSpaced<T, /*RunType=*/IndexType, DictionaryConverter<T>>(
          converter, block.length, block.length - block.popcount, valid_bits,
          valid_bits_offset, out);
    }
    total_processed += processed;
    out += block.length;
    valid_bits_offset += block.length;
  } while (processed == block.length);
  return total_processed;
}

template <typename T>
bool RleDecoder::NextCounts() {
  // Read the next run's indicator int, it could be a literal or repeated run.
  // The int is encoded as a vlq-encoded value.
  uint32_t indicator_value = 0;
  if (!bit_reader_.GetVlqInt(&indicator_value)) return false;

  // lsb indicates if it is a literal run or repeated run
  bool is_literal = indicator_value & 1;
  uint32_t count = indicator_value >> 1;
  if (is_literal) {
    if (ARROW_PREDICT_FALSE(count == 0 || count > static_cast<uint32_t>(INT32_MAX) / 8)) {
      return false;
    }
    literal_count_ = count * 8;
  } else {
    if (ARROW_PREDICT_FALSE(count == 0 || count > static_cast<uint32_t>(INT32_MAX))) {
      return false;
    }
    repeat_count_ = count;
    T value = {};
    if (!bit_reader_.GetAligned<T>(
            static_cast<int>(::arrow::bit_util::CeilDiv(bit_width_, 8)), &value)) {
      return false;
    }
    current_value_ = static_cast<uint64_t>(value);
  }
  return true;
}

/// This function buffers input values 8 at a time.  After seeing all 8 values,
/// it decides whether they should be encoded as a literal or repeated run.
inline bool RleEncoder::Put(uint64_t value) {
  DCHECK(bit_width_ == 64 || value < (1ULL << bit_width_));
  if (ARROW_PREDICT_FALSE(buffer_full_)) return false;

  if (ARROW_PREDICT_TRUE(current_value_ == value)) {
    ++repeat_count_;
    if (repeat_count_ > 8) {
      // This is just a continuation of the current run, no need to buffer the
      // values.
      // Note that this is the fast path for long repeated runs.
      return true;
    }
  } else {
    if (repeat_count_ >= 8) {
      // We had a run that was long enough but it has ended.  Flush the
      // current repeated run.
      DCHECK_EQ(literal_count_, 0);
      FlushRepeatedRun();
    }
    repeat_count_ = 1;
    current_value_ = value;
  }

  buffered_values_[num_buffered_values_] = value;
  if (++num_buffered_values_ == 8) {
    DCHECK_EQ(literal_count_ % 8, 0);
    FlushBufferedValues(false);
  }
  return true;
}

inline void RleEncoder::FlushLiteralRun(bool update_indicator_byte) {
  if (literal_indicator_byte_ == NULL) {
    // The literal indicator byte has not been reserved yet, get one now.
    literal_indicator_byte_ = bit_writer_.GetNextBytePtr();
    DCHECK(literal_indicator_byte_ != NULL);
  }

  // Write all the buffered values as bit packed literals
  for (int i = 0; i < num_buffered_values_; ++i) {
    bool success = bit_writer_.PutValue(buffered_values_[i], bit_width_);
    DCHECK(success) << "There is a bug in using CheckBufferFull()";
  }
  num_buffered_values_ = 0;

  if (update_indicator_byte) {
    // At this point we need to write the indicator byte for the literal run.
    // We only reserve one byte, to allow for streaming writes of literal values.
    // The logic makes sure we flush literal runs often enough to not overrun
    // the 1 byte.
    DCHECK_EQ(literal_count_ % 8, 0);
    int num_groups = literal_count_ / 8;
    int32_t indicator_value = (num_groups << 1) | 1;
    DCHECK_EQ(indicator_value & 0xFFFFFF00, 0);
    *literal_indicator_byte_ = static_cast<uint8_t>(indicator_value);
    literal_indicator_byte_ = NULL;
    literal_count_ = 0;
    CheckBufferFull();
  }
}

inline void RleEncoder::FlushRepeatedRun() {
  DCHECK_GT(repeat_count_, 0);
  bool result = true;
  // The lsb of 0 indicates this is a repeated run
  int32_t indicator_value = repeat_count_ << 1 | 0;
  result &= bit_writer_.PutVlqInt(static_cast<uint32_t>(indicator_value));
  result &= bit_writer_.PutAligned(
      current_value_, static_cast<int>(::arrow::bit_util::CeilDiv(bit_width_, 8)));
  DCHECK(result);
  num_buffered_values_ = 0;
  repeat_count_ = 0;
  CheckBufferFull();
}

/// Flush the values that have been buffered.  At this point we decide whether
/// we need to switch between the run types or continue the current one.
inline void RleEncoder::FlushBufferedValues(bool done) {
  if (repeat_count_ >= 8) {
    // Clear the buffered values.  They are part of the repeated run now and we
    // don't want to flush them out as literals.
    num_buffered_values_ = 0;
    if (literal_count_ != 0) {
      // There was a current literal run.  All the values in it have been flushed
      // but we still need to update the indicator byte.
      DCHECK_EQ(literal_count_ % 8, 0);
      DCHECK_EQ(repeat_count_, 8);
      FlushLiteralRun(true);
    }
    DCHECK_EQ(literal_count_, 0);
    return;
  }

  literal_count_ += num_buffered_values_;
  DCHECK_EQ(literal_count_ % 8, 0);
  int num_groups = literal_count_ / 8;
  if (num_groups + 1 >= (1 << 6)) {
    // We need to start a new literal run because the indicator byte we've reserved
    // cannot store more values.
    DCHECK(literal_indicator_byte_ != NULL);
    FlushLiteralRun(true);
  } else {
    FlushLiteralRun(done);
  }
  repeat_count_ = 0;
}

inline int RleEncoder::Flush() {
  if (literal_count_ > 0 || repeat_count_ > 0 || num_buffered_values_ > 0) {
    bool all_repeat = literal_count_ == 0 && (repeat_count_ == num_buffered_values_ ||
                                              num_buffered_values_ == 0);
    // There is something pending, figure out if it's a repeated or literal run
    if (repeat_count_ > 0 && all_repeat) {
      FlushRepeatedRun();
    } else {
      DCHECK_EQ(literal_count_ % 8, 0);
      // Buffer the last group of literals to 8 by padding with 0s.
      for (; num_buffered_values_ != 0 && num_buffered_values_ < 8;
           ++num_buffered_values_) {
        buffered_values_[num_buffered_values_] = 0;
      }
      literal_count_ += num_buffered_values_;
      FlushLiteralRun(true);
      repeat_count_ = 0;
    }
  }
  bit_writer_.Flush();
  DCHECK_EQ(num_buffered_values_, 0);
  DCHECK_EQ(literal_count_, 0);
  DCHECK_EQ(repeat_count_, 0);

  return bit_writer_.bytes_written();
}

inline void RleEncoder::CheckBufferFull() {
  int bytes_written = bit_writer_.bytes_written();
  if (bytes_written + max_run_byte_size_ > bit_writer_.buffer_len()) {
    buffer_full_ = true;
  }
}

inline void RleEncoder::Clear() {
  buffer_full_ = false;
  current_value_ = 0;
  repeat_count_ = 0;
  num_buffered_values_ = 0;
  literal_count_ = 0;
  literal_indicator_byte_ = NULL;
  bit_writer_.Clear();
}

}  // namespace util
}  // namespace arrow

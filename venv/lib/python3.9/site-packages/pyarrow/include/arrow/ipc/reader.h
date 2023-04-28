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

// Read Arrow files and streams

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "arrow/io/caching.h"
#include "arrow/io/type_fwd.h"
#include "arrow/ipc/message.h"
#include "arrow/ipc/options.h"
#include "arrow/record_batch.h"
#include "arrow/result.h"
#include "arrow/type_fwd.h"
#include "arrow/util/async_generator.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace ipc {

class DictionaryMemo;
struct IpcPayload;

using RecordBatchReader = ::arrow::RecordBatchReader;

struct ReadStats {
  /// Number of IPC messages read.
  int64_t num_messages = 0;
  /// Number of record batches read.
  int64_t num_record_batches = 0;
  /// Number of dictionary batches read.
  ///
  /// Note: num_dictionary_batches >= num_dictionary_deltas + num_replaced_dictionaries
  int64_t num_dictionary_batches = 0;

  /// Number of dictionary deltas read.
  int64_t num_dictionary_deltas = 0;
  /// Number of replaced dictionaries (i.e. where a dictionary batch replaces
  /// an existing dictionary with an unrelated new dictionary).
  int64_t num_replaced_dictionaries = 0;
};

/// \brief Synchronous batch stream reader that reads from io::InputStream
///
/// This class reads the schema (plus any dictionaries) as the first messages
/// in the stream, followed by record batches. For more granular zero-copy
/// reads see the ReadRecordBatch functions
class ARROW_EXPORT RecordBatchStreamReader : public RecordBatchReader {
 public:
  /// Create batch reader from generic MessageReader.
  /// This will take ownership of the given MessageReader.
  ///
  /// \param[in] message_reader a MessageReader implementation
  /// \param[in] options any IPC reading options (optional)
  /// \return the created batch reader
  static Result<std::shared_ptr<RecordBatchStreamReader>> Open(
      std::unique_ptr<MessageReader> message_reader,
      const IpcReadOptions& options = IpcReadOptions::Defaults());

  /// \brief Record batch stream reader from InputStream
  ///
  /// \param[in] stream an input stream instance. Must stay alive throughout
  /// lifetime of stream reader
  /// \param[in] options any IPC reading options (optional)
  /// \return the created batch reader
  static Result<std::shared_ptr<RecordBatchStreamReader>> Open(
      io::InputStream* stream,
      const IpcReadOptions& options = IpcReadOptions::Defaults());

  /// \brief Open stream and retain ownership of stream object
  /// \param[in] stream the input stream
  /// \param[in] options any IPC reading options (optional)
  /// \return the created batch reader
  static Result<std::shared_ptr<RecordBatchStreamReader>> Open(
      const std::shared_ptr<io::InputStream>& stream,
      const IpcReadOptions& options = IpcReadOptions::Defaults());

  /// \brief Return current read statistics
  virtual ReadStats stats() const = 0;
};

/// \brief Reads the record batch file format
class ARROW_EXPORT RecordBatchFileReader
    : public std::enable_shared_from_this<RecordBatchFileReader> {
 public:
  virtual ~RecordBatchFileReader() = default;

  /// \brief Open a RecordBatchFileReader
  ///
  /// Open a file-like object that is assumed to be self-contained; i.e., the
  /// end of the file interface is the end of the Arrow file. Note that there
  /// can be any amount of data preceding the Arrow-formatted data, because we
  /// need only locate the end of the Arrow file stream to discover the metadata
  /// and then proceed to read the data into memory.
  static Result<std::shared_ptr<RecordBatchFileReader>> Open(
      io::RandomAccessFile* file,
      const IpcReadOptions& options = IpcReadOptions::Defaults());

  /// \brief Open a RecordBatchFileReader
  /// If the file is embedded within some larger file or memory region, you can
  /// pass the absolute memory offset to the end of the file (which contains the
  /// metadata footer). The metadata must have been written with memory offsets
  /// relative to the start of the containing file
  ///
  /// \param[in] file the data source
  /// \param[in] footer_offset the position of the end of the Arrow file
  /// \param[in] options options for IPC reading
  /// \return the returned reader
  static Result<std::shared_ptr<RecordBatchFileReader>> Open(
      io::RandomAccessFile* file, int64_t footer_offset,
      const IpcReadOptions& options = IpcReadOptions::Defaults());

  /// \brief Version of Open that retains ownership of file
  ///
  /// \param[in] file the data source
  /// \param[in] options options for IPC reading
  /// \return the returned reader
  static Result<std::shared_ptr<RecordBatchFileReader>> Open(
      const std::shared_ptr<io::RandomAccessFile>& file,
      const IpcReadOptions& options = IpcReadOptions::Defaults());

  /// \brief Version of Open that retains ownership of file
  ///
  /// \param[in] file the data source
  /// \param[in] footer_offset the position of the end of the Arrow file
  /// \param[in] options options for IPC reading
  /// \return the returned reader
  static Result<std::shared_ptr<RecordBatchFileReader>> Open(
      const std::shared_ptr<io::RandomAccessFile>& file, int64_t footer_offset,
      const IpcReadOptions& options = IpcReadOptions::Defaults());

  /// \brief Open a file asynchronously (owns the file).
  static Future<std::shared_ptr<RecordBatchFileReader>> OpenAsync(
      const std::shared_ptr<io::RandomAccessFile>& file,
      const IpcReadOptions& options = IpcReadOptions::Defaults());

  /// \brief Open a file asynchronously (borrows the file).
  static Future<std::shared_ptr<RecordBatchFileReader>> OpenAsync(
      io::RandomAccessFile* file,
      const IpcReadOptions& options = IpcReadOptions::Defaults());

  /// \brief Open a file asynchronously (owns the file).
  static Future<std::shared_ptr<RecordBatchFileReader>> OpenAsync(
      const std::shared_ptr<io::RandomAccessFile>& file, int64_t footer_offset,
      const IpcReadOptions& options = IpcReadOptions::Defaults());

  /// \brief Open a file asynchronously (borrows the file).
  static Future<std::shared_ptr<RecordBatchFileReader>> OpenAsync(
      io::RandomAccessFile* file, int64_t footer_offset,
      const IpcReadOptions& options = IpcReadOptions::Defaults());

  /// \brief The schema read from the file
  virtual std::shared_ptr<Schema> schema() const = 0;

  /// \brief Returns the number of record batches in the file
  virtual int num_record_batches() const = 0;

  /// \brief Return the metadata version from the file metadata
  virtual MetadataVersion version() const = 0;

  /// \brief Return the contents of the custom_metadata field from the file's
  /// Footer
  virtual std::shared_ptr<const KeyValueMetadata> metadata() const = 0;

  /// \brief Read a particular record batch from the file. Does not copy memory
  /// if the input source supports zero-copy.
  ///
  /// \param[in] i the index of the record batch to return
  /// \return the read batch
  virtual Result<std::shared_ptr<RecordBatch>> ReadRecordBatch(int i) = 0;

  /// \brief Read a particular record batch along with its custom metadada from the file.
  /// Does not copy memory if the input source supports zero-copy.
  ///
  /// \param[in] i the index of the record batch to return
  /// \return a struct containing the read batch and its custom metadata
  virtual Result<RecordBatchWithMetadata> ReadRecordBatchWithCustomMetadata(int i) = 0;

  /// \brief Return current read statistics
  virtual ReadStats stats() const = 0;

  /// \brief Computes the total number of rows in the file.
  virtual Result<int64_t> CountRows() = 0;

  /// \brief Begin loading metadata for the desired batches into memory.
  ///
  /// This method will also begin loading all dictionaries messages into memory.
  ///
  /// For a regular file this will immediately begin disk I/O in the background on a
  /// thread on the IOContext's thread pool.  If the file is memory mapped this will
  /// ensure the memory needed for the metadata is paged from disk into memory
  ///
  /// \param indices Indices of the batches to prefetch
  ///                If empty then all batches will be prefetched.
  virtual Status PreBufferMetadata(const std::vector<int>& indices) = 0;

  /// \brief Get a reentrant generator of record batches.
  ///
  /// \param[in] coalesce If true, enable I/O coalescing.
  /// \param[in] io_context The IOContext to use (controls which thread pool
  ///     is used for I/O).
  /// \param[in] cache_options Options for coalescing (if enabled).
  /// \param[in] executor Optionally, an executor to use for decoding record
  ///     batches. This is generally only a benefit for very wide and/or
  ///     compressed batches.
  virtual Result<AsyncGenerator<std::shared_ptr<RecordBatch>>> GetRecordBatchGenerator(
      const bool coalesce = false,
      const io::IOContext& io_context = io::default_io_context(),
      const io::CacheOptions cache_options = io::CacheOptions::LazyDefaults(),
      arrow::internal::Executor* executor = NULLPTR) = 0;
};

/// \brief A general listener class to receive events.
///
/// You must implement callback methods for interested events.
///
/// This API is EXPERIMENTAL.
///
/// \since 0.17.0
class ARROW_EXPORT Listener {
 public:
  virtual ~Listener() = default;

  /// \brief Called when end-of-stream is received.
  ///
  /// The default implementation just returns arrow::Status::OK().
  ///
  /// \return Status
  ///
  /// \see StreamDecoder
  virtual Status OnEOS();

  /// \brief Called when a record batch is decoded.
  ///
  /// The default implementation just returns
  /// arrow::Status::NotImplemented().
  ///
  /// \param[in] record_batch a record batch decoded
  /// \return Status
  ///
  /// \see StreamDecoder
  virtual Status OnRecordBatchDecoded(std::shared_ptr<RecordBatch> record_batch);

  /// \brief Called when a schema is decoded.
  ///
  /// The default implementation just returns arrow::Status::OK().
  ///
  /// \param[in] schema a schema decoded
  /// \return Status
  ///
  /// \see StreamDecoder
  virtual Status OnSchemaDecoded(std::shared_ptr<Schema> schema);
};

/// \brief Collect schema and record batches decoded by StreamDecoder.
///
/// This API is EXPERIMENTAL.
///
/// \since 0.17.0
class ARROW_EXPORT CollectListener : public Listener {
 public:
  CollectListener() : schema_(), record_batches_() {}
  virtual ~CollectListener() = default;

  Status OnSchemaDecoded(std::shared_ptr<Schema> schema) override {
    schema_ = std::move(schema);
    return Status::OK();
  }

  Status OnRecordBatchDecoded(std::shared_ptr<RecordBatch> record_batch) override {
    record_batches_.push_back(std::move(record_batch));
    return Status::OK();
  }

  /// \return the decoded schema
  std::shared_ptr<Schema> schema() const { return schema_; }

  /// \return the all decoded record batches
  std::vector<std::shared_ptr<RecordBatch>> record_batches() const {
    return record_batches_;
  }

 private:
  std::shared_ptr<Schema> schema_;
  std::vector<std::shared_ptr<RecordBatch>> record_batches_;
};

/// \brief Push style stream decoder that receives data from user.
///
/// This class decodes the Apache Arrow IPC streaming format data.
///
/// This API is EXPERIMENTAL.
///
/// \see https://arrow.apache.org/docs/format/Columnar.html#ipc-streaming-format
///
/// \since 0.17.0
class ARROW_EXPORT StreamDecoder {
 public:
  /// \brief Construct a stream decoder.
  ///
  /// \param[in] listener a Listener that must implement
  /// Listener::OnRecordBatchDecoded() to receive decoded record batches
  /// \param[in] options any IPC reading options (optional)
  StreamDecoder(std::shared_ptr<Listener> listener,
                IpcReadOptions options = IpcReadOptions::Defaults());

  virtual ~StreamDecoder();

  /// \brief Feed data to the decoder as a raw data.
  ///
  /// If the decoder can read one or more record batches by the data,
  /// the decoder calls listener->OnRecordBatchDecoded() with a
  /// decoded record batch multiple times.
  ///
  /// \param[in] data a raw data to be processed. This data isn't
  /// copied. The passed memory must be kept alive through record
  /// batch processing.
  /// \param[in] size raw data size.
  /// \return Status
  Status Consume(const uint8_t* data, int64_t size);

  /// \brief Feed data to the decoder as a Buffer.
  ///
  /// If the decoder can read one or more record batches by the
  /// Buffer, the decoder calls listener->RecordBatchReceived() with a
  /// decoded record batch multiple times.
  ///
  /// \param[in] buffer a Buffer to be processed.
  /// \return Status
  Status Consume(std::shared_ptr<Buffer> buffer);

  /// \return the shared schema of the record batches in the stream
  std::shared_ptr<Schema> schema() const;

  /// \brief Return the number of bytes needed to advance the state of
  /// the decoder.
  ///
  /// This method is provided for users who want to optimize performance.
  /// Normal users don't need to use this method.
  ///
  /// Here is an example usage for normal users:
  ///
  /// ~~~{.cpp}
  /// decoder.Consume(buffer1);
  /// decoder.Consume(buffer2);
  /// decoder.Consume(buffer3);
  /// ~~~
  ///
  /// Decoder has internal buffer. If consumed data isn't enough to
  /// advance the state of the decoder, consumed data is buffered to
  /// the internal buffer. It causes performance overhead.
  ///
  /// If you pass next_required_size() size data to each Consume()
  /// call, the decoder doesn't use its internal buffer. It improves
  /// performance.
  ///
  /// Here is an example usage to avoid using internal buffer:
  ///
  /// ~~~{.cpp}
  /// buffer1 = get_data(decoder.next_required_size());
  /// decoder.Consume(buffer1);
  /// buffer2 = get_data(decoder.next_required_size());
  /// decoder.Consume(buffer2);
  /// ~~~
  ///
  /// Users can use this method to avoid creating small chunks. Record
  /// batch data must be contiguous data. If users pass small chunks
  /// to the decoder, the decoder needs concatenate small chunks
  /// internally. It causes performance overhead.
  ///
  /// Here is an example usage to reduce small chunks:
  ///
  /// ~~~{.cpp}
  /// buffer = AllocateResizableBuffer();
  /// while ((small_chunk = get_data(&small_chunk_size))) {
  ///   auto current_buffer_size = buffer->size();
  ///   buffer->Resize(current_buffer_size + small_chunk_size);
  ///   memcpy(buffer->mutable_data() + current_buffer_size,
  ///          small_chunk,
  ///          small_chunk_size);
  ///   if (buffer->size() < decoder.next_required_size()) {
  ///     continue;
  ///   }
  ///   std::shared_ptr<arrow::Buffer> chunk(buffer.release());
  ///   decoder.Consume(chunk);
  ///   buffer = AllocateResizableBuffer();
  /// }
  /// if (buffer->size() > 0) {
  ///   std::shared_ptr<arrow::Buffer> chunk(buffer.release());
  ///   decoder.Consume(chunk);
  /// }
  /// ~~~
  ///
  /// \return the number of bytes needed to advance the state of the
  /// decoder
  int64_t next_required_size() const;

  /// \brief Return current read statistics
  ReadStats stats() const;

 private:
  class StreamDecoderImpl;
  std::unique_ptr<StreamDecoderImpl> impl_;

  ARROW_DISALLOW_COPY_AND_ASSIGN(StreamDecoder);
};

// Generic read functions; does not copy data if the input supports zero copy reads

/// \brief Read Schema from stream serialized as a single IPC message
/// and populate any dictionary-encoded fields into a DictionaryMemo
///
/// \param[in] stream an InputStream
/// \param[in] dictionary_memo for recording dictionary-encoded fields
/// \return the output Schema
///
/// If record batches follow the schema, it is better to use
/// RecordBatchStreamReader
ARROW_EXPORT
Result<std::shared_ptr<Schema>> ReadSchema(io::InputStream* stream,
                                           DictionaryMemo* dictionary_memo);

/// \brief Read Schema from encapsulated Message
///
/// \param[in] message the message containing the Schema IPC metadata
/// \param[in] dictionary_memo DictionaryMemo for recording dictionary-encoded
/// fields. Can be nullptr if you are sure there are no
/// dictionary-encoded fields
/// \return the resulting Schema
ARROW_EXPORT
Result<std::shared_ptr<Schema>> ReadSchema(const Message& message,
                                           DictionaryMemo* dictionary_memo);

/// Read record batch as encapsulated IPC message with metadata size prefix and
/// header
///
/// \param[in] schema the record batch schema
/// \param[in] dictionary_memo DictionaryMemo which has any
/// dictionaries. Can be nullptr if you are sure there are no
/// dictionary-encoded fields
/// \param[in] options IPC options for reading
/// \param[in] stream the file where the batch is located
/// \return the read record batch
ARROW_EXPORT
Result<std::shared_ptr<RecordBatch>> ReadRecordBatch(
    const std::shared_ptr<Schema>& schema, const DictionaryMemo* dictionary_memo,
    const IpcReadOptions& options, io::InputStream* stream);

/// \brief Read record batch from message
///
/// \param[in] message a Message containing the record batch metadata
/// \param[in] schema the record batch schema
/// \param[in] dictionary_memo DictionaryMemo which has any
/// dictionaries. Can be nullptr if you are sure there are no
/// dictionary-encoded fields
/// \param[in] options IPC options for reading
/// \return the read record batch
ARROW_EXPORT
Result<std::shared_ptr<RecordBatch>> ReadRecordBatch(
    const Message& message, const std::shared_ptr<Schema>& schema,
    const DictionaryMemo* dictionary_memo, const IpcReadOptions& options);

/// Read record batch from file given metadata and schema
///
/// \param[in] metadata a Message containing the record batch metadata
/// \param[in] schema the record batch schema
/// \param[in] dictionary_memo DictionaryMemo which has any
/// dictionaries. Can be nullptr if you are sure there are no
/// dictionary-encoded fields
/// \param[in] file a random access file
/// \param[in] options options for deserialization
/// \return the read record batch
ARROW_EXPORT
Result<std::shared_ptr<RecordBatch>> ReadRecordBatch(
    const Buffer& metadata, const std::shared_ptr<Schema>& schema,
    const DictionaryMemo* dictionary_memo, const IpcReadOptions& options,
    io::RandomAccessFile* file);

/// \brief Read arrow::Tensor as encapsulated IPC message in file
///
/// \param[in] file an InputStream pointed at the start of the message
/// \return the read tensor
ARROW_EXPORT
Result<std::shared_ptr<Tensor>> ReadTensor(io::InputStream* file);

/// \brief EXPERIMENTAL: Read arrow::Tensor from IPC message
///
/// \param[in] message a Message containing the tensor metadata and body
/// \return the read tensor
ARROW_EXPORT
Result<std::shared_ptr<Tensor>> ReadTensor(const Message& message);

/// \brief EXPERIMENTAL: Read arrow::SparseTensor as encapsulated IPC message in file
///
/// \param[in] file an InputStream pointed at the start of the message
/// \return the read sparse tensor
ARROW_EXPORT
Result<std::shared_ptr<SparseTensor>> ReadSparseTensor(io::InputStream* file);

/// \brief EXPERIMENTAL: Read arrow::SparseTensor from IPC message
///
/// \param[in] message a Message containing the tensor metadata and body
/// \return the read sparse tensor
ARROW_EXPORT
Result<std::shared_ptr<SparseTensor>> ReadSparseTensor(const Message& message);

namespace internal {

// These internal APIs may change without warning or deprecation

/// \brief EXPERIMENTAL: Read arrow::SparseTensorFormat::type from a metadata
/// \param[in] metadata a Buffer containing the sparse tensor metadata
/// \return the count of the body buffers
ARROW_EXPORT
Result<size_t> ReadSparseTensorBodyBufferCount(const Buffer& metadata);

/// \brief EXPERIMENTAL: Read arrow::SparseTensor from an IpcPayload
/// \param[in] payload a IpcPayload contains a serialized SparseTensor
/// \return the read sparse tensor
ARROW_EXPORT
Result<std::shared_ptr<SparseTensor>> ReadSparseTensorPayload(const IpcPayload& payload);

// For fuzzing targets
ARROW_EXPORT
Status FuzzIpcStream(const uint8_t* data, int64_t size);
ARROW_EXPORT
Status FuzzIpcTensorStream(const uint8_t* data, int64_t size);
ARROW_EXPORT
Status FuzzIpcFile(const uint8_t* data, int64_t size);

}  // namespace internal

}  // namespace ipc
}  // namespace arrow

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from collections import namedtuple
import warnings


cpdef enum MetadataVersion:
    V1 = <char> CMetadataVersion_V1
    V2 = <char> CMetadataVersion_V2
    V3 = <char> CMetadataVersion_V3
    V4 = <char> CMetadataVersion_V4
    V5 = <char> CMetadataVersion_V5


cdef object _wrap_metadata_version(CMetadataVersion version):
    return MetadataVersion(<char> version)


cdef CMetadataVersion _unwrap_metadata_version(
        MetadataVersion version) except *:
    if version == MetadataVersion.V1:
        return CMetadataVersion_V1
    elif version == MetadataVersion.V2:
        return CMetadataVersion_V2
    elif version == MetadataVersion.V3:
        return CMetadataVersion_V3
    elif version == MetadataVersion.V4:
        return CMetadataVersion_V4
    elif version == MetadataVersion.V5:
        return CMetadataVersion_V5
    raise ValueError("Not a metadata version: " + repr(version))


_WriteStats = namedtuple(
    'WriteStats',
    ('num_messages', 'num_record_batches', 'num_dictionary_batches',
     'num_dictionary_deltas', 'num_replaced_dictionaries'))


class WriteStats(_WriteStats):
    """IPC write statistics

    Parameters
    ----------
    num_messages : int
        Number of messages.
    num_record_batches : int
        Number of record batches.
    num_dictionary_batches : int
        Number of dictionary batches.
    num_dictionary_deltas : int
        Delta of dictionaries.
    num_replaced_dictionaries : int
        Number of replaced dictionaries.
    """
    __slots__ = ()


@staticmethod
cdef _wrap_write_stats(CIpcWriteStats c):
    return WriteStats(c.num_messages, c.num_record_batches,
                      c.num_dictionary_batches, c.num_dictionary_deltas,
                      c.num_replaced_dictionaries)


_ReadStats = namedtuple(
    'ReadStats',
    ('num_messages', 'num_record_batches', 'num_dictionary_batches',
     'num_dictionary_deltas', 'num_replaced_dictionaries'))


class ReadStats(_ReadStats):
    """IPC read statistics

    Parameters
    ----------
    num_messages : int
        Number of messages.
    num_record_batches : int
        Number of record batches.
    num_dictionary_batches : int
        Number of dictionary batches.
    num_dictionary_deltas : int
        Delta of dictionaries.
    num_replaced_dictionaries : int
        Number of replaced dictionaries.
    """
    __slots__ = ()


@staticmethod
cdef _wrap_read_stats(CIpcReadStats c):
    return ReadStats(c.num_messages, c.num_record_batches,
                     c.num_dictionary_batches, c.num_dictionary_deltas,
                     c.num_replaced_dictionaries)


cdef class IpcReadOptions(_Weakrefable):
    """
    Serialization options for reading IPC format.

    Parameters
    ----------
    ensure_native_endian : bool, default True
        Whether to convert incoming data to platform-native endianness.
    use_threads : bool
        Whether to use the global CPU thread pool to parallelize any
        computational tasks like decompression
    included_fields : list
        If empty (the default), return all deserialized fields.
        If non-empty, the values are the indices of fields to read on
        the top-level schema
    """
    __slots__ = ()

    # cdef block is in lib.pxd

    def __init__(self, *, bint ensure_native_endian=True,
                 bint use_threads=True, list included_fields=None):
        self.c_options = CIpcReadOptions.Defaults()
        self.ensure_native_endian = ensure_native_endian
        self.use_threads = use_threads
        if included_fields is not None:
            self.included_fields = included_fields

    @property
    def ensure_native_endian(self):
        return self.c_options.ensure_native_endian

    @ensure_native_endian.setter
    def ensure_native_endian(self, bint value):
        self.c_options.ensure_native_endian = value

    @property
    def use_threads(self):
        return self.c_options.use_threads

    @use_threads.setter
    def use_threads(self, bint value):
        self.c_options.use_threads = value

    @property
    def included_fields(self):
        return self.c_options.included_fields

    @included_fields.setter
    def included_fields(self, list value not None):
        self.c_options.included_fields = value


cdef class IpcWriteOptions(_Weakrefable):
    """
    Serialization options for the IPC format.

    Parameters
    ----------
    metadata_version : MetadataVersion, default MetadataVersion.V5
        The metadata version to write.  V5 is the current and latest,
        V4 is the pre-1.0 metadata version (with incompatible Union layout).
    allow_64bit : bool, default False
        If true, allow field lengths that don't fit in a signed 32-bit int.
    use_legacy_format : bool, default False
        Whether to use the pre-Arrow 0.15 IPC format.
    compression : str, Codec, or None
        compression codec to use for record batch buffers.
        If None then batch buffers will be uncompressed.
        Must be "lz4", "zstd" or None.
        To specify a compression_level use `pyarrow.Codec`
    use_threads : bool
        Whether to use the global CPU thread pool to parallelize any
        computational tasks like compression.
    emit_dictionary_deltas : bool
        Whether to emit dictionary deltas.  Default is false for maximum
        stream compatibility.
    unify_dictionaries : bool
        If true then calls to write_table will attempt to unify dictionaries
        across all batches in the table.  This can help avoid the need for
        replacement dictionaries (which the file format does not support)
        but requires computing the unified dictionary and then remapping
        the indices arrays.

        This parameter is ignored when writing to the IPC stream format as
        the IPC stream format can support replacement dictionaries.
    """
    __slots__ = ()

    # cdef block is in lib.pxd

    def __init__(self, *, metadata_version=MetadataVersion.V5,
                 bint allow_64bit=False, use_legacy_format=False,
                 compression=None, bint use_threads=True,
                 bint emit_dictionary_deltas=False,
                 bint unify_dictionaries=False):
        self.c_options = CIpcWriteOptions.Defaults()
        self.allow_64bit = allow_64bit
        self.use_legacy_format = use_legacy_format
        self.metadata_version = metadata_version
        if compression is not None:
            self.compression = compression
        self.use_threads = use_threads
        self.emit_dictionary_deltas = emit_dictionary_deltas
        self.unify_dictionaries = unify_dictionaries

    @property
    def allow_64bit(self):
        return self.c_options.allow_64bit

    @allow_64bit.setter
    def allow_64bit(self, bint value):
        self.c_options.allow_64bit = value

    @property
    def use_legacy_format(self):
        return self.c_options.write_legacy_ipc_format

    @use_legacy_format.setter
    def use_legacy_format(self, bint value):
        self.c_options.write_legacy_ipc_format = value

    @property
    def metadata_version(self):
        return _wrap_metadata_version(self.c_options.metadata_version)

    @metadata_version.setter
    def metadata_version(self, value):
        self.c_options.metadata_version = _unwrap_metadata_version(value)

    @property
    def compression(self):
        if self.c_options.codec == nullptr:
            return None
        else:
            return frombytes(self.c_options.codec.get().name())

    @compression.setter
    def compression(self, value):
        if value is None:
            self.c_options.codec.reset()
        elif isinstance(value, str):
            codec_type = _ensure_compression(value)
            if codec_type != CCompressionType_ZSTD and codec_type != CCompressionType_LZ4_FRAME:
                raise ValueError("Compression type must be lz4, zstd or None")
            self.c_options.codec = shared_ptr[CCodec](GetResultValue(
                CCodec.Create(codec_type)).release())
        elif isinstance(value, Codec):
            if value.name != "lz4" and value.name != "zstd":
                raise ValueError("Compression type must be lz4, zstd or None")
            self.c_options.codec = (<Codec>value).wrapped
        else:
            raise TypeError(
                "Property `compression` must be None, str, or pyarrow.Codec")

    @property
    def use_threads(self):
        return self.c_options.use_threads

    @use_threads.setter
    def use_threads(self, bint value):
        self.c_options.use_threads = value

    @property
    def emit_dictionary_deltas(self):
        return self.c_options.emit_dictionary_deltas

    @emit_dictionary_deltas.setter
    def emit_dictionary_deltas(self, bint value):
        self.c_options.emit_dictionary_deltas = value

    @property
    def unify_dictionaries(self):
        return self.c_options.unify_dictionaries

    @unify_dictionaries.setter
    def unify_dictionaries(self, bint value):
        self.c_options.unify_dictionaries = value


cdef class Message(_Weakrefable):
    """
    Container for an Arrow IPC message with metadata and optional body
    """

    def __cinit__(self):
        pass

    def __init__(self):
        raise TypeError("Do not call {}'s constructor directly, use "
                        "`pyarrow.ipc.read_message` function instead."
                        .format(self.__class__.__name__))

    @property
    def type(self):
        return frombytes(FormatMessageType(self.message.get().type()))

    @property
    def metadata(self):
        return pyarrow_wrap_buffer(self.message.get().metadata())

    @property
    def metadata_version(self):
        return _wrap_metadata_version(self.message.get().metadata_version())

    @property
    def body(self):
        cdef shared_ptr[CBuffer] body = self.message.get().body()
        if body.get() == NULL:
            return None
        else:
            return pyarrow_wrap_buffer(body)

    def equals(self, Message other):
        """
        Returns True if the message contents (metadata and body) are identical

        Parameters
        ----------
        other : Message

        Returns
        -------
        are_equal : bool
        """
        cdef c_bool result
        with nogil:
            result = self.message.get().Equals(deref(other.message.get()))
        return result

    def serialize_to(self, NativeFile sink, alignment=8, memory_pool=None):
        """
        Write message to generic OutputStream

        Parameters
        ----------
        sink : NativeFile
        alignment : int, default 8
            Byte alignment for metadata and body
        memory_pool : MemoryPool, default None
            Uses default memory pool if not specified
        """
        cdef:
            int64_t output_length = 0
            COutputStream* out
            CIpcWriteOptions options

        options.alignment = alignment
        out = sink.get_output_stream().get()
        with nogil:
            check_status(self.message.get()
                         .SerializeTo(out, options, &output_length))

    def serialize(self, alignment=8, memory_pool=None):
        """
        Write message as encapsulated IPC message

        Parameters
        ----------
        alignment : int, default 8
            Byte alignment for metadata and body
        memory_pool : MemoryPool, default None
            Uses default memory pool if not specified

        Returns
        -------
        serialized : Buffer
        """
        stream = BufferOutputStream(memory_pool)
        self.serialize_to(stream, alignment=alignment, memory_pool=memory_pool)
        return stream.getvalue()

    def __repr__(self):
        if self.message == nullptr:
            return """pyarrow.Message(uninitialized)"""

        metadata_len = self.metadata.size
        body = self.body
        body_len = 0 if body is None else body.size

        return """pyarrow.Message
type: {0}
metadata length: {1}
body length: {2}""".format(self.type, metadata_len, body_len)


cdef class MessageReader(_Weakrefable):
    """
    Interface for reading Message objects from some source (like an
    InputStream)
    """
    cdef:
        unique_ptr[CMessageReader] reader

    def __cinit__(self):
        pass

    def __init__(self):
        raise TypeError("Do not call {}'s constructor directly, use "
                        "`pyarrow.ipc.MessageReader.open_stream` function "
                        "instead.".format(self.__class__.__name__))

    @staticmethod
    def open_stream(source):
        """
        Open stream from source, if you want to use memory map use
        MemoryMappedFile as source.

        Parameters
        ----------
        source : bytes/buffer-like, pyarrow.NativeFile, or file-like Python object
            A readable source, like an InputStream
        """
        cdef:
            MessageReader result = MessageReader.__new__(MessageReader)
            shared_ptr[CInputStream] in_stream
            unique_ptr[CMessageReader] reader

        _get_input_stream(source, &in_stream)
        with nogil:
            reader = CMessageReader.Open(in_stream)
            result.reader.reset(reader.release())

        return result

    def __iter__(self):
        while True:
            yield self.read_next_message()

    def read_next_message(self):
        """
        Read next Message from the stream.

        Raises
        ------
        StopIteration
            At end of stream
        """
        cdef Message result = Message.__new__(Message)

        with nogil:
            result.message = move(GetResultValue(self.reader.get()
                                                 .ReadNextMessage()))

        if result.message.get() == NULL:
            raise StopIteration

        return result

# ----------------------------------------------------------------------
# File and stream readers and writers

cdef class _CRecordBatchWriter(_Weakrefable):
    """The base RecordBatchWriter wrapper.

    Provides common implementations of convenience methods. Should not
    be instantiated directly by user code.
    """

    # cdef block is in lib.pxd

    def write(self, table_or_batch):
        """
        Write RecordBatch or Table to stream.

        Parameters
        ----------
        table_or_batch : {RecordBatch, Table}
        """
        if isinstance(table_or_batch, RecordBatch):
            self.write_batch(table_or_batch)
        elif isinstance(table_or_batch, Table):
            self.write_table(table_or_batch)
        else:
            raise ValueError(type(table_or_batch))

    def write_batch(self, RecordBatch batch, custom_metadata=None):
        """
        Write RecordBatch to stream.

        Parameters
        ----------
        batch : RecordBatch
        custom_metadata : mapping or KeyValueMetadata
            Keys and values must be string-like / coercible to bytes
        """
        metadata = ensure_metadata(custom_metadata, allow_none=True)
        c_meta = pyarrow_unwrap_metadata(metadata)

        with nogil:
            check_status(self.writer.get()
                         .WriteRecordBatch(deref(batch.batch), c_meta))

    def write_table(self, Table table, max_chunksize=None):
        """
        Write Table to stream in (contiguous) RecordBatch objects.

        Parameters
        ----------
        table : Table
        max_chunksize : int, default None
            Maximum size for RecordBatch chunks. Individual chunks may be
            smaller depending on the chunk layout of individual columns.
        """
        cdef:
            # max_chunksize must be > 0 to have any impact
            int64_t c_max_chunksize = -1

        if max_chunksize is not None:
            c_max_chunksize = max_chunksize

        with nogil:
            check_status(self.writer.get().WriteTable(table.table[0],
                                                      c_max_chunksize))

    def close(self):
        """
        Close stream and write end-of-stream 0 marker.
        """
        with nogil:
            check_status(self.writer.get().Close())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def stats(self):
        """
        Current IPC write statistics.
        """
        if not self.writer:
            raise ValueError("Operation on closed writer")
        return _wrap_write_stats(self.writer.get().stats())


cdef class _RecordBatchStreamWriter(_CRecordBatchWriter):
    cdef:
        CIpcWriteOptions options
        bint closed

    def __cinit__(self):
        pass

    def __dealloc__(self):
        pass

    @property
    def _use_legacy_format(self):
        # For testing (see test_ipc.py)
        return self.options.write_legacy_ipc_format

    @property
    def _metadata_version(self):
        # For testing (see test_ipc.py)
        return _wrap_metadata_version(self.options.metadata_version)

    def _open(self, sink, Schema schema not None,
              IpcWriteOptions options=IpcWriteOptions()):
        cdef:
            shared_ptr[COutputStream] c_sink

        self.options = options.c_options
        get_writer(sink, &c_sink)
        with nogil:
            self.writer = GetResultValue(
                MakeStreamWriter(c_sink, schema.sp_schema,
                                 self.options))


cdef _get_input_stream(object source, shared_ptr[CInputStream]* out):
    try:
        source = as_buffer(source)
    except TypeError:
        # Non-buffer-like
        pass

    get_input_stream(source, True, out)


class _ReadPandasMixin:

    def read_pandas(self, **options):
        """
        Read contents of stream to a pandas.DataFrame.

        Read all record batches as a pyarrow.Table then convert it to a
        pandas.DataFrame using Table.to_pandas.

        Parameters
        ----------
        **options
            Arguments to forward to :meth:`Table.to_pandas`.

        Returns
        -------
        df : pandas.DataFrame
        """
        table = self.read_all()
        return table.to_pandas(**options)


cdef class RecordBatchReader(_Weakrefable):
    """Base class for reading stream of record batches.

    Record batch readers function as iterators of record batches that also
    provide the schema (without the need to get any batches).

    Warnings
    --------
    Do not call this class's constructor directly, use one of the
    ``RecordBatchReader.from_*`` functions instead.

    Notes
    -----
    To import and export using the Arrow C stream interface, use the
    ``_import_from_c`` and ``_export_from_c`` methods. However, keep in mind this
    interface is intended for expert users.

    Examples
    --------
    >>> import pyarrow as pa
    >>> schema = pa.schema([('x', pa.int64())])
    >>> def iter_record_batches():
    ...     for i in range(2):
    ...         yield pa.RecordBatch.from_arrays([pa.array([1, 2, 3])], schema=schema)
    >>> reader = pa.RecordBatchReader.from_batches(schema, iter_record_batches())
    >>> print(reader.schema)
    x: int64
    >>> for batch in reader:
    ...     print(batch)
    pyarrow.RecordBatch
    x: int64
    pyarrow.RecordBatch
    x: int64
    """

    # cdef block is in lib.pxd

    def __iter__(self):
        while True:
            try:
                yield self.read_next_batch()
            except StopIteration:
                return

    @property
    def schema(self):
        """
        Shared schema of the record batches in the stream.

        Returns
        -------
        Schema
        """
        cdef shared_ptr[CSchema] c_schema

        with nogil:
            c_schema = self.reader.get().schema()

        return pyarrow_wrap_schema(c_schema)

    def read_next_batch(self):
        """
        Read next RecordBatch from the stream.

        Raises
        ------
        StopIteration:
            At end of stream.

        Returns
        -------
        RecordBatch
        """
        cdef shared_ptr[CRecordBatch] batch

        with nogil:
            check_status(self.reader.get().ReadNext(&batch))

        if batch.get() == NULL:
            raise StopIteration

        return pyarrow_wrap_batch(batch)

    def read_next_batch_with_custom_metadata(self):
        """
        Read next RecordBatch from the stream along with its custom metadata.

        Raises
        ------
        StopIteration:
            At end of stream.

        Returns
        -------
        batch : RecordBatch
        custom_metadata : KeyValueMetadata
        """
        cdef:
            CRecordBatchWithMetadata batch_with_metadata

        with nogil:
            batch_with_metadata = GetResultValue(self.reader.get().ReadNext())

        if batch_with_metadata.batch.get() == NULL:
            raise StopIteration

        return _wrap_record_batch_with_metadata(batch_with_metadata)

    def iter_batches_with_custom_metadata(self):
        """
        Iterate over record batches from the stream along with their custom
        metadata.

        Yields
        ------
        RecordBatchWithMetadata
        """
        while True:
            try:
                yield self.read_next_batch_with_custom_metadata()
            except StopIteration:
                return

    def read_all(self):
        """
        Read all record batches as a pyarrow.Table.

        Returns
        -------
        Table
        """
        cdef shared_ptr[CTable] table
        with nogil:
            check_status(self.reader.get().ToTable().Value(&table))
        return pyarrow_wrap_table(table)

    read_pandas = _ReadPandasMixin.read_pandas

    def close(self):
        """
        Release any resources associated with the reader.
        """
        with nogil:
            check_status(self.reader.get().Close())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _export_to_c(self, out_ptr):
        """
        Export to a C ArrowArrayStream struct, given its pointer.

        Parameters
        ----------
        out_ptr: int
            The raw pointer to a C ArrowArrayStream struct.

        Be careful: if you don't pass the ArrowArrayStream struct to a
        consumer, array memory will leak.  This is a low-level function
        intended for expert users.
        """
        cdef:
            void* c_ptr = _as_c_pointer(out_ptr)
        with nogil:
            check_status(ExportRecordBatchReader(
                self.reader, <ArrowArrayStream*> c_ptr))

    @staticmethod
    def _import_from_c(in_ptr):
        """
        Import RecordBatchReader from a C ArrowArrayStream struct,
        given its pointer.

        Parameters
        ----------
        in_ptr: int
            The raw pointer to a C ArrowArrayStream struct.

        This is a low-level function intended for expert users.
        """
        cdef:
            void* c_ptr = _as_c_pointer(in_ptr)
            shared_ptr[CRecordBatchReader] c_reader
            RecordBatchReader self

        with nogil:
            c_reader = GetResultValue(ImportRecordBatchReader(
                <ArrowArrayStream*> c_ptr))

        self = RecordBatchReader.__new__(RecordBatchReader)
        self.reader = c_reader
        return self

    @staticmethod
    def from_batches(Schema schema not None, batches):
        """
        Create RecordBatchReader from an iterable of batches.

        Parameters
        ----------
        schema : Schema
            The shared schema of the record batches
        batches : Iterable[RecordBatch]
            The batches that this reader will return.

        Returns
        -------
        reader : RecordBatchReader
        """
        cdef:
            shared_ptr[CSchema] c_schema
            shared_ptr[CRecordBatchReader] c_reader
            RecordBatchReader self

        c_schema = pyarrow_unwrap_schema(schema)
        c_reader = GetResultValue(CPyRecordBatchReader.Make(
            c_schema, batches))

        self = RecordBatchReader.__new__(RecordBatchReader)
        self.reader = c_reader
        return self


cdef class _RecordBatchStreamReader(RecordBatchReader):
    cdef:
        shared_ptr[CInputStream] in_stream
        CIpcReadOptions options
        CRecordBatchStreamReader* stream_reader

    def __cinit__(self):
        pass

    def _open(self, source, IpcReadOptions options=IpcReadOptions(),
              MemoryPool memory_pool=None):
        self.options = options.c_options
        self.options.memory_pool = maybe_unbox_memory_pool(memory_pool)
        _get_input_stream(source, &self.in_stream)
        with nogil:
            self.reader = GetResultValue(CRecordBatchStreamReader.Open(
                self.in_stream, self.options))
            self.stream_reader = <CRecordBatchStreamReader*> self.reader.get()

    @property
    def stats(self):
        """
        Current IPC read statistics.
        """
        if not self.reader:
            raise ValueError("Operation on closed reader")
        return _wrap_read_stats(self.stream_reader.stats())


cdef class _RecordBatchFileWriter(_RecordBatchStreamWriter):

    def _open(self, sink, Schema schema not None,
              IpcWriteOptions options=IpcWriteOptions()):
        cdef:
            shared_ptr[COutputStream] c_sink

        self.options = options.c_options
        get_writer(sink, &c_sink)
        with nogil:
            self.writer = GetResultValue(
                MakeFileWriter(c_sink, schema.sp_schema, self.options))

_RecordBatchWithMetadata = namedtuple(
    'RecordBatchWithMetadata',
    ('batch', 'custom_metadata'))


class RecordBatchWithMetadata(_RecordBatchWithMetadata):
    """RecordBatch with its custom metadata

    Parameters
    ----------
    batch : RecordBatch
    custom_metadata : KeyValueMetadata
    """
    __slots__ = ()


@staticmethod
cdef _wrap_record_batch_with_metadata(CRecordBatchWithMetadata c):
    return RecordBatchWithMetadata(pyarrow_wrap_batch(c.batch),
                                   pyarrow_wrap_metadata(c.custom_metadata))


cdef class _RecordBatchFileReader(_Weakrefable):
    cdef:
        shared_ptr[CRecordBatchFileReader] reader
        shared_ptr[CRandomAccessFile] file
        CIpcReadOptions options

    cdef readonly:
        Schema schema

    def __cinit__(self):
        pass

    def _open(self, source, footer_offset=None,
              IpcReadOptions options=IpcReadOptions(),
              MemoryPool memory_pool=None):
        self.options = options.c_options
        self.options.memory_pool = maybe_unbox_memory_pool(memory_pool)
        try:
            source = as_buffer(source)
        except TypeError:
            pass

        get_reader(source, False, &self.file)

        cdef int64_t offset = 0
        if footer_offset is not None:
            offset = footer_offset

        with nogil:
            if offset != 0:
                self.reader = GetResultValue(
                    CRecordBatchFileReader.Open2(self.file.get(), offset,
                                                 self.options))

            else:
                self.reader = GetResultValue(
                    CRecordBatchFileReader.Open(self.file.get(),
                                                self.options))

        self.schema = pyarrow_wrap_schema(self.reader.get().schema())

    @property
    def num_record_batches(self):
        """
        The number of record batches in the IPC file.
        """
        return self.reader.get().num_record_batches()

    def get_batch(self, int i):
        """
        Read the record batch with the given index.

        Parameters
        ----------
        i : int
            The index of the record batch in the IPC file.

        Returns
        -------
        batch : RecordBatch
        """
        cdef shared_ptr[CRecordBatch] batch

        if i < 0 or i >= self.num_record_batches:
            raise ValueError('Batch number {0} out of range'.format(i))

        with nogil:
            batch = GetResultValue(self.reader.get().ReadRecordBatch(i))

        return pyarrow_wrap_batch(batch)

    # TODO(wesm): ARROW-503: Function was renamed. Remove after a period of
    # time has passed
    get_record_batch = get_batch

    def get_batch_with_custom_metadata(self, int i):
        """
        Read the record batch with the given index along with 
        its custom metadata

        Parameters
        ----------
        i : int
            The index of the record batch in the IPC file.

        Returns
        -------
        batch : RecordBatch
        custom_metadata : KeyValueMetadata
        """
        cdef:
            CRecordBatchWithMetadata batch_with_metadata

        if i < 0 or i >= self.num_record_batches:
            raise ValueError('Batch number {0} out of range'.format(i))

        with nogil:
            batch_with_metadata = GetResultValue(
                self.reader.get().ReadRecordBatchWithCustomMetadata(i))

        return _wrap_record_batch_with_metadata(batch_with_metadata)

    def read_all(self):
        """
        Read all record batches as a pyarrow.Table
        """
        cdef:
            vector[shared_ptr[CRecordBatch]] batches
            shared_ptr[CTable] table
            int i, nbatches

        nbatches = self.num_record_batches

        batches.resize(nbatches)
        with nogil:
            for i in range(nbatches):
                batches[i] = GetResultValue(self.reader.get()
                                            .ReadRecordBatch(i))
            table = GetResultValue(
                CTable.FromRecordBatches(self.schema.sp_schema, move(batches)))

        return pyarrow_wrap_table(table)

    read_pandas = _ReadPandasMixin.read_pandas

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @property
    def stats(self):
        """
        Current IPC read statistics.
        """
        if not self.reader:
            raise ValueError("Operation on closed reader")
        return _wrap_read_stats(self.reader.get().stats())


def get_tensor_size(Tensor tensor):
    """
    Return total size of serialized Tensor including metadata and padding.

    Parameters
    ----------
    tensor : Tensor
        The tensor for which we want to known the size.
    """
    cdef int64_t size
    with nogil:
        check_status(GetTensorSize(deref(tensor.tp), &size))
    return size


def get_record_batch_size(RecordBatch batch):
    """
    Return total size of serialized RecordBatch including metadata and padding.

    Parameters
    ----------
    batch : RecordBatch
        The recordbatch for which we want to know the size.
    """
    cdef int64_t size
    with nogil:
        check_status(GetRecordBatchSize(deref(batch.batch), &size))
    return size


def write_tensor(Tensor tensor, NativeFile dest):
    """
    Write pyarrow.Tensor to pyarrow.NativeFile object its current position.

    Parameters
    ----------
    tensor : pyarrow.Tensor
    dest : pyarrow.NativeFile

    Returns
    -------
    bytes_written : int
        Total number of bytes written to the file
    """
    cdef:
        int32_t metadata_length
        int64_t body_length

    handle = dest.get_output_stream()

    with nogil:
        check_status(
            WriteTensor(deref(tensor.tp), handle.get(),
                        &metadata_length, &body_length))

    return metadata_length + body_length


cdef NativeFile as_native_file(source):
    if not isinstance(source, NativeFile):
        if hasattr(source, 'read'):
            source = PythonFile(source)
        else:
            source = BufferReader(source)

    if not isinstance(source, NativeFile):
        raise ValueError('Unable to read message from object with type: {0}'
                         .format(type(source)))
    return source


def read_tensor(source):
    """Read pyarrow.Tensor from pyarrow.NativeFile object from current
    position. If the file source supports zero copy (e.g. a memory map), then
    this operation does not allocate any memory. This function not assume that
    the stream is aligned

    Parameters
    ----------
    source : pyarrow.NativeFile

    Returns
    -------
    tensor : Tensor

    """
    cdef:
        shared_ptr[CTensor] sp_tensor
        CInputStream* c_stream
        NativeFile nf = as_native_file(source)

    c_stream = nf.get_input_stream().get()
    with nogil:
        sp_tensor = GetResultValue(ReadTensor(c_stream))
    return pyarrow_wrap_tensor(sp_tensor)


def read_message(source):
    """
    Read length-prefixed message from file or buffer-like object

    Parameters
    ----------
    source : pyarrow.NativeFile, file-like object, or buffer-like object

    Returns
    -------
    message : Message
    """
    cdef:
        Message result = Message.__new__(Message)
        CInputStream* c_stream

    cdef NativeFile nf = as_native_file(source)
    c_stream = nf.get_input_stream().get()

    with nogil:
        result.message = move(
            GetResultValue(ReadMessage(c_stream, c_default_memory_pool())))

    if result.message == nullptr:
        raise EOFError("End of Arrow stream")

    return result


def read_schema(obj, DictionaryMemo dictionary_memo=None):
    """
    Read Schema from message or buffer

    Parameters
    ----------
    obj : buffer or Message
    dictionary_memo : DictionaryMemo, optional
        Needed to be able to reconstruct dictionary-encoded fields
        with read_record_batch

    Returns
    -------
    schema : Schema
    """
    cdef:
        shared_ptr[CSchema] result
        shared_ptr[CRandomAccessFile] cpp_file
        Message message
        CDictionaryMemo temp_memo
        CDictionaryMemo* arg_dict_memo

    if dictionary_memo is not None:
        arg_dict_memo = dictionary_memo.memo
    else:
        arg_dict_memo = &temp_memo

    if isinstance(obj, Message):
        message = obj
        with nogil:
            result = GetResultValue(ReadSchema(
                deref(message.message.get()), arg_dict_memo))
    else:
        get_reader(obj, False, &cpp_file)
        with nogil:
            result = GetResultValue(ReadSchema(cpp_file.get(), arg_dict_memo))

    return pyarrow_wrap_schema(result)


def read_record_batch(obj, Schema schema,
                      DictionaryMemo dictionary_memo=None):
    """
    Read RecordBatch from message, given a known schema. If reading data from a
    complete IPC stream, use ipc.open_stream instead

    Parameters
    ----------
    obj : Message or Buffer-like
    schema : Schema
    dictionary_memo : DictionaryMemo, optional
        If message contains dictionaries, must pass a populated
        DictionaryMemo

    Returns
    -------
    batch : RecordBatch
    """
    cdef:
        shared_ptr[CRecordBatch] result
        Message message
        CDictionaryMemo temp_memo
        CDictionaryMemo* arg_dict_memo

    if isinstance(obj, Message):
        message = obj
    else:
        message = read_message(obj)

    if dictionary_memo is not None:
        arg_dict_memo = dictionary_memo.memo
    else:
        arg_dict_memo = &temp_memo

    with nogil:
        result = GetResultValue(
            ReadRecordBatch(deref(message.message.get()),
                            schema.sp_schema,
                            arg_dict_memo,
                            CIpcReadOptions.Defaults()))

    return pyarrow_wrap_batch(result)

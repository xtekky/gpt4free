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

# Arrow file and stream reader/writer classes, and other messaging tools

import os

import pyarrow as pa

from pyarrow.lib import (IpcReadOptions, IpcWriteOptions, ReadStats, WriteStats,  # noqa
                         Message, MessageReader,
                         RecordBatchReader, _ReadPandasMixin,
                         MetadataVersion,
                         read_message, read_record_batch, read_schema,
                         read_tensor, write_tensor,
                         get_record_batch_size, get_tensor_size)
import pyarrow.lib as lib


class RecordBatchStreamReader(lib._RecordBatchStreamReader):
    """
    Reader for the Arrow streaming binary format.

    Parameters
    ----------
    source : bytes/buffer-like, pyarrow.NativeFile, or file-like Python object
        Either an in-memory buffer, or a readable file object.
        If you want to use memory map use MemoryMappedFile as source.
    options : pyarrow.ipc.IpcReadOptions
        Options for IPC deserialization.
        If None, default values will be used.
    memory_pool : MemoryPool, default None
        If None, default memory pool is used.
    """

    def __init__(self, source, *, options=None, memory_pool=None):
        options = _ensure_default_ipc_read_options(options)
        self._open(source, options=options, memory_pool=memory_pool)


_ipc_writer_class_doc = """\
Parameters
----------
sink : str, pyarrow.NativeFile, or file-like Python object
    Either a file path, or a writable file object.
schema : pyarrow.Schema
    The Arrow schema for data to be written to the file.
use_legacy_format : bool, default None
    Deprecated in favor of setting options. Cannot be provided with
    options.

    If None, False will be used unless this default is overridden by
    setting the environment variable ARROW_PRE_0_15_IPC_FORMAT=1
options : pyarrow.ipc.IpcWriteOptions
    Options for IPC serialization.

    If None, default values will be used: the legacy format will not
    be used unless overridden by setting the environment variable
    ARROW_PRE_0_15_IPC_FORMAT=1, and the V5 metadata version will be
    used unless overridden by setting the environment variable
    ARROW_PRE_1_0_METADATA_VERSION=1."""


class RecordBatchStreamWriter(lib._RecordBatchStreamWriter):
    __doc__ = """Writer for the Arrow streaming binary format

{}""".format(_ipc_writer_class_doc)

    def __init__(self, sink, schema, *, use_legacy_format=None, options=None):
        options = _get_legacy_format_default(use_legacy_format, options)
        self._open(sink, schema, options=options)


class RecordBatchFileReader(lib._RecordBatchFileReader):
    """
    Class for reading Arrow record batch data from the Arrow binary file format

    Parameters
    ----------
    source : bytes/buffer-like, pyarrow.NativeFile, or file-like Python object
        Either an in-memory buffer, or a readable file object.
        If you want to use memory map use MemoryMappedFile as source.
    footer_offset : int, default None
        If the file is embedded in some larger file, this is the byte offset to
        the very end of the file data
    options : pyarrow.ipc.IpcReadOptions
        Options for IPC serialization.
        If None, default values will be used.
    memory_pool : MemoryPool, default None
        If None, default memory pool is used.
    """

    def __init__(self, source, footer_offset=None, *, options=None,
                 memory_pool=None):
        options = _ensure_default_ipc_read_options(options)
        self._open(source, footer_offset=footer_offset,
                   options=options, memory_pool=memory_pool)


class RecordBatchFileWriter(lib._RecordBatchFileWriter):

    __doc__ = """Writer to create the Arrow binary file format

{}""".format(_ipc_writer_class_doc)

    def __init__(self, sink, schema, *, use_legacy_format=None, options=None):
        options = _get_legacy_format_default(use_legacy_format, options)
        self._open(sink, schema, options=options)


def _get_legacy_format_default(use_legacy_format, options):
    if use_legacy_format is not None and options is not None:
        raise ValueError(
            "Can provide at most one of options and use_legacy_format")
    elif options:
        if not isinstance(options, IpcWriteOptions):
            raise TypeError("expected IpcWriteOptions, got {}"
                            .format(type(options)))
        return options

    metadata_version = MetadataVersion.V5
    if use_legacy_format is None:
        use_legacy_format = \
            bool(int(os.environ.get('ARROW_PRE_0_15_IPC_FORMAT', '0')))
    if bool(int(os.environ.get('ARROW_PRE_1_0_METADATA_VERSION', '0'))):
        metadata_version = MetadataVersion.V4
    return IpcWriteOptions(use_legacy_format=use_legacy_format,
                           metadata_version=metadata_version)


def _ensure_default_ipc_read_options(options):
    if options and not isinstance(options, IpcReadOptions):
        raise TypeError(
            "expected IpcReadOptions, got {}".format(type(options))
        )
    return options or IpcReadOptions()


def new_stream(sink, schema, *, use_legacy_format=None, options=None):
    return RecordBatchStreamWriter(sink, schema,
                                   use_legacy_format=use_legacy_format,
                                   options=options)


new_stream.__doc__ = """\
Create an Arrow columnar IPC stream writer instance

{}

Returns
-------
writer : RecordBatchStreamWriter
    A writer for the given sink
""".format(_ipc_writer_class_doc)


def open_stream(source, *, options=None, memory_pool=None):
    """
    Create reader for Arrow streaming format.

    Parameters
    ----------
    source : bytes/buffer-like, pyarrow.NativeFile, or file-like Python object
        Either an in-memory buffer, or a readable file object.
    options : pyarrow.ipc.IpcReadOptions
        Options for IPC serialization.
        If None, default values will be used.
    memory_pool : MemoryPool, default None
        If None, default memory pool is used.

    Returns
    -------
    reader : RecordBatchStreamReader
        A reader for the given source
    """
    return RecordBatchStreamReader(source, options=options,
                                   memory_pool=memory_pool)


def new_file(sink, schema, *, use_legacy_format=None, options=None):
    return RecordBatchFileWriter(sink, schema,
                                 use_legacy_format=use_legacy_format,
                                 options=options)


new_file.__doc__ = """\
Create an Arrow columnar IPC file writer instance

{}

Returns
-------
writer : RecordBatchFileWriter
    A writer for the given sink
""".format(_ipc_writer_class_doc)


def open_file(source, footer_offset=None, *, options=None, memory_pool=None):
    """
    Create reader for Arrow file format.

    Parameters
    ----------
    source : bytes/buffer-like, pyarrow.NativeFile, or file-like Python object
        Either an in-memory buffer, or a readable file object.
    footer_offset : int, default None
        If the file is embedded in some larger file, this is the byte offset to
        the very end of the file data.
    options : pyarrow.ipc.IpcReadOptions
        Options for IPC serialization.
        If None, default values will be used.
    memory_pool : MemoryPool, default None
        If None, default memory pool is used.

    Returns
    -------
    reader : RecordBatchFileReader
        A reader for the given source
    """
    return RecordBatchFileReader(
        source, footer_offset=footer_offset,
        options=options, memory_pool=memory_pool)


def serialize_pandas(df, *, nthreads=None, preserve_index=None):
    """
    Serialize a pandas DataFrame into a buffer protocol compatible object.

    Parameters
    ----------
    df : pandas.DataFrame
    nthreads : int, default None
        Number of threads to use for conversion to Arrow, default all CPUs.
    preserve_index : bool, default None
        The default of None will store the index as a column, except for
        RangeIndex which is stored as metadata only. If True, always
        preserve the pandas index data as a column. If False, no index
        information is saved and the result will have a default RangeIndex.

    Returns
    -------
    buf : buffer
        An object compatible with the buffer protocol.
    """
    batch = pa.RecordBatch.from_pandas(df, nthreads=nthreads,
                                       preserve_index=preserve_index)
    sink = pa.BufferOutputStream()
    with pa.RecordBatchStreamWriter(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue()


def deserialize_pandas(buf, *, use_threads=True):
    """Deserialize a buffer protocol compatible object into a pandas DataFrame.

    Parameters
    ----------
    buf : buffer
        An object compatible with the buffer protocol.
    use_threads : bool, default True
        Whether to parallelize the conversion using multiple threads.

    Returns
    -------
    df : pandas.DataFrame
        The buffer deserialized as pandas DataFrame
    """
    buffer_reader = pa.BufferReader(buf)
    with pa.RecordBatchStreamReader(buffer_reader) as reader:
        table = reader.read_all()
    return table.to_pandas(use_threads=use_threads)

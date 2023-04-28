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

# Cython wrappers for IO interfaces defined in arrow::io and messaging in
# arrow::ipc

from libc.stdlib cimport malloc, free

import codecs
import re
import sys
import threading
import time
import warnings
from io import BufferedIOBase, IOBase, TextIOBase, UnsupportedOperation
from queue import Queue, Empty as QueueEmpty

from pyarrow.util import _is_path_like, _stringify_path


# 64K
DEFAULT_BUFFER_SIZE = 2 ** 16


cdef extern from "Python.h":
    # To let us get a PyObject* and avoid Cython auto-ref-counting
    PyObject* PyBytes_FromStringAndSizeNative" PyBytes_FromStringAndSize"(
        char *v, Py_ssize_t len) except NULL

    # Workaround https://github.com/cython/cython/issues/4707
    bytearray PyByteArray_FromStringAndSize(char *string, Py_ssize_t len)


def io_thread_count():
    """
    Return the number of threads to use for I/O operations.

    Many operations, such as scanning a dataset, will implicitly make
    use of this pool. The number of threads is set to a fixed value at
    startup. It can be modified at runtime by calling
    :func:`set_io_thread_count()`.

    See Also
    --------
    set_io_thread_count : Modify the size of this pool.
    cpu_count : The analogous function for the CPU thread pool.
    """
    return GetIOThreadPoolCapacity()


def set_io_thread_count(int count):
    """
    Set the number of threads to use for I/O operations.

    Many operations, such as scanning a dataset, will implicitly make
    use of this pool.

    Parameters
    ----------
    count : int
        The max number of threads that may be used for I/O.
        Must be positive.

    See Also
    --------
    io_thread_count : Get the size of this pool.
    set_cpu_count : The analogous function for the CPU thread pool.
    """
    if count < 1:
        raise ValueError("IO thread count must be strictly positive")
    check_status(SetIOThreadPoolCapacity(count))


cdef class NativeFile(_Weakrefable):
    """
    The base class for all Arrow streams.

    Streams are either readable, writable, or both.
    They optionally support seeking.

    While this class exposes methods to read or write data from Python, the
    primary intent of using a Arrow stream is to pass it to other Arrow
    facilities that will make use of it, such as Arrow IPC routines.

    Be aware that there are subtle differences with regular Python files,
    e.g. destroying a writable Arrow stream without closing it explicitly
    will not flush any pending data.
    """

    # Default chunk size for chunked reads.
    # Use a large enough value for networked filesystems.
    _default_chunk_size = 256 * 1024

    def __cinit__(self):
        self.own_file = False
        self.is_readable = False
        self.is_writable = False
        self.is_seekable = False

    def __dealloc__(self):
        if self.own_file:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.close()

    def __repr__(self):
        name = f"pyarrow.{self.__class__.__name__}"
        return (f"<{name} "
                f"closed={self.closed} "
                f"own_file={self.own_file} "
                f"is_seekable={self.is_seekable} "
                f"is_writable={self.is_writable} "
                f"is_readable={self.is_readable}>")

    @property
    def mode(self):
        """
        The file mode. Currently instances of NativeFile may support:

        * rb: binary read
        * wb: binary write
        * rb+: binary read and write
        """
        # Emulate built-in file modes
        if self.is_readable and self.is_writable:
            return 'rb+'
        elif self.is_readable:
            return 'rb'
        elif self.is_writable:
            return 'wb'
        else:
            raise ValueError('File object is malformed, has no mode')

    def readable(self):
        self._assert_open()
        return self.is_readable

    def writable(self):
        self._assert_open()
        return self.is_writable

    def seekable(self):
        self._assert_open()
        return self.is_seekable

    def isatty(self):
        self._assert_open()
        return False

    def fileno(self):
        """
        NOT IMPLEMENTED
        """
        raise UnsupportedOperation()

    @property
    def closed(self):
        if self.is_readable:
            return self.input_stream.get().closed()
        elif self.is_writable:
            return self.output_stream.get().closed()
        else:
            return True

    def close(self):
        if not self.closed:
            with nogil:
                if self.is_readable:
                    check_status(self.input_stream.get().Close())
                else:
                    check_status(self.output_stream.get().Close())

    cdef set_random_access_file(self, shared_ptr[CRandomAccessFile] handle):
        self.input_stream = <shared_ptr[CInputStream]> handle
        self.random_access = handle
        self.is_seekable = True

    cdef set_input_stream(self, shared_ptr[CInputStream] handle):
        self.input_stream = handle
        self.random_access.reset()
        self.is_seekable = False

    cdef set_output_stream(self, shared_ptr[COutputStream] handle):
        self.output_stream = handle

    cdef shared_ptr[CRandomAccessFile] get_random_access_file(self) except *:
        self._assert_readable()
        self._assert_seekable()
        return self.random_access

    cdef shared_ptr[CInputStream] get_input_stream(self) except *:
        self._assert_readable()
        return self.input_stream

    cdef shared_ptr[COutputStream] get_output_stream(self) except *:
        self._assert_writable()
        return self.output_stream

    def _assert_open(self):
        if self.closed:
            raise ValueError("I/O operation on closed file")

    def _assert_readable(self):
        self._assert_open()
        if not self.is_readable:
            # XXX UnsupportedOperation
            raise IOError("only valid on readable files")

    def _assert_writable(self):
        self._assert_open()
        if not self.is_writable:
            raise IOError("only valid on writable files")

    def _assert_seekable(self):
        self._assert_open()
        if not self.is_seekable:
            raise IOError("only valid on seekable files")

    def size(self):
        """
        Return file size
        """
        cdef int64_t size

        handle = self.get_random_access_file()
        with nogil:
            size = GetResultValue(handle.get().GetSize())

        return size

    def metadata(self):
        """
        Return file metadata
        """
        cdef:
            shared_ptr[const CKeyValueMetadata] c_metadata

        handle = self.get_input_stream()
        with nogil:
            c_metadata = GetResultValue(handle.get().ReadMetadata())

        metadata = {}
        if c_metadata.get() != nullptr:
            for i in range(c_metadata.get().size()):
                metadata[frombytes(c_metadata.get().key(i))] = \
                    c_metadata.get().value(i)
        return metadata

    def tell(self):
        """
        Return current stream position
        """
        cdef int64_t position

        if self.is_readable:
            rd_handle = self.get_random_access_file()
            with nogil:
                position = GetResultValue(rd_handle.get().Tell())
        else:
            wr_handle = self.get_output_stream()
            with nogil:
                position = GetResultValue(wr_handle.get().Tell())

        return position

    def seek(self, int64_t position, int whence=0):
        """
        Change current file stream position

        Parameters
        ----------
        position : int
            Byte offset, interpreted relative to value of whence argument
        whence : int, default 0
            Point of reference for seek offset

        Notes
        -----
        Values of whence:
        * 0 -- start of stream (the default); offset should be zero or positive
        * 1 -- current stream position; offset may be negative
        * 2 -- end of stream; offset is usually negative

        Returns
        -------
        int
            The new absolute stream position.
        """
        cdef int64_t offset
        handle = self.get_random_access_file()

        with nogil:
            if whence == 0:
                offset = position
            elif whence == 1:
                offset = GetResultValue(handle.get().Tell())
                offset = offset + position
            elif whence == 2:
                offset = GetResultValue(handle.get().GetSize())
                offset = offset + position
            else:
                with gil:
                    raise ValueError("Invalid value of whence: {0}"
                                     .format(whence))
            check_status(handle.get().Seek(offset))

        return self.tell()

    def flush(self):
        """
        Flush the stream, if applicable.

        An error is raised if stream is not writable.
        """
        self._assert_open()
        # For IOBase compatibility, flush() on an input stream is a no-op
        if self.is_writable:
            handle = self.get_output_stream()
            with nogil:
                check_status(handle.get().Flush())

    def write(self, data):
        """
        Write data to the file.

        Parameters
        ----------
        data : bytes-like object or exporter of buffer protocol

        Returns
        -------
        int
            nbytes: number of bytes written
        """
        self._assert_writable()
        handle = self.get_output_stream()

        cdef shared_ptr[CBuffer] buf = as_c_buffer(data)

        with nogil:
            check_status(handle.get().WriteBuffer(buf))
        return buf.get().size()

    def read(self, nbytes=None):
        """
        Read and return up to n bytes.

        If *nbytes* is None, then the entire remaining file contents are read.

        Parameters
        ----------
        nbytes : int, default None

        Returns
        -------
        data : bytes
        """
        cdef:
            int64_t c_nbytes
            int64_t bytes_read = 0
            PyObject* obj

        if nbytes is None:
            if not self.is_seekable:
                # Cannot get file size => read chunkwise
                bs = self._default_chunk_size
                chunks = []
                while True:
                    chunk = self.read(bs)
                    if not chunk:
                        break
                    chunks.append(chunk)
                return b"".join(chunks)

            c_nbytes = self.size() - self.tell()
        else:
            c_nbytes = nbytes

        handle = self.get_input_stream()

        # Allocate empty write space
        obj = PyBytes_FromStringAndSizeNative(NULL, c_nbytes)

        cdef uint8_t* buf = <uint8_t*> cp.PyBytes_AS_STRING(<object> obj)
        with nogil:
            bytes_read = GetResultValue(handle.get().Read(c_nbytes, buf))

        if bytes_read < c_nbytes:
            cp._PyBytes_Resize(&obj, <Py_ssize_t> bytes_read)

        return PyObject_to_object(obj)

    def get_stream(self, file_offset, nbytes):
        """
        Return an input stream that reads a file segment independent of the
        state of the file.

        Allows reading portions of a random access file as an input stream
        without interfering with each other.

        Parameters
        ----------
        file_offset : int
        nbytes : int

        Returns
        -------
        stream : NativeFile
        """
        cdef:
            shared_ptr[CInputStream] data
            int64_t c_file_offset
            int64_t c_nbytes

        c_file_offset = file_offset
        c_nbytes = nbytes

        handle = self.get_random_access_file()

        data = GetResultValue(
            CRandomAccessFile.GetStream(handle, c_file_offset, c_nbytes))

        stream = NativeFile()
        stream.set_input_stream(data)
        stream.is_readable = True

        return stream

    def read_at(self, nbytes, offset):
        """
        Read indicated number of bytes at offset from the file

        Parameters
        ----------
        nbytes : int
        offset : int

        Returns
        -------
        data : bytes
        """
        cdef:
            int64_t c_nbytes
            int64_t c_offset
            int64_t bytes_read = 0
            PyObject* obj

        c_nbytes = nbytes

        c_offset = offset

        handle = self.get_random_access_file()

        # Allocate empty write space
        obj = PyBytes_FromStringAndSizeNative(NULL, c_nbytes)

        cdef uint8_t* buf = <uint8_t*> cp.PyBytes_AS_STRING(<object> obj)
        with nogil:
            bytes_read = GetResultValue(handle.get().
                                        ReadAt(c_offset, c_nbytes, buf))

        if bytes_read < c_nbytes:
            cp._PyBytes_Resize(&obj, <Py_ssize_t> bytes_read)

        return PyObject_to_object(obj)

    def read1(self, nbytes=None):
        """Read and return up to n bytes.

        Unlike read(), if *nbytes* is None then a chunk is read, not the
        entire file.

        Parameters
        ----------
        nbytes : int, default None
            The maximum number of bytes to read.

        Returns
        -------
        data : bytes
        """
        if nbytes is None:
            # The expectation when passing `nbytes=None` is not to read the
            # entire file but to issue a single underlying read call up to
            # a reasonable size (the use case being to read a bufferable
            # amount of bytes, such as with io.TextIOWrapper).
            nbytes = self._default_chunk_size
        return self.read(nbytes)

    def readall(self):
        return self.read()

    def readinto(self, b):
        """
        Read into the supplied buffer

        Parameters
        ----------
        b : buffer-like object
            A writable buffer object (such as a bytearray).

        Returns
        -------
        written : int
            number of bytes written
        """

        cdef:
            int64_t bytes_read
            uint8_t* buf
            Buffer py_buf
            int64_t buf_len

        handle = self.get_input_stream()

        py_buf = py_buffer(b)
        buf_len = py_buf.size
        buf = py_buf.buffer.get().mutable_data()

        with nogil:
            bytes_read = GetResultValue(handle.get().Read(buf_len, buf))

        return bytes_read

    def readline(self, size=None):
        """NOT IMPLEMENTED. Read and return a line of bytes from the file.

        If size is specified, read at most size bytes.

        Line terminator is always b"\\n".

        Parameters
        ----------
        size : int
            maximum number of bytes read
        """
        raise UnsupportedOperation()

    def readlines(self, hint=None):
        """NOT IMPLEMENTED. Read lines of the file

        Parameters
        ----------
        hint : int
            maximum number of bytes read until we stop
        """
        raise UnsupportedOperation()

    def __iter__(self):
        self._assert_readable()
        return self

    def __next__(self):
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    def read_buffer(self, nbytes=None):
        cdef:
            int64_t c_nbytes
            int64_t bytes_read = 0
            shared_ptr[CBuffer] output

        handle = self.get_input_stream()

        if nbytes is None:
            if not self.is_seekable:
                # Cannot get file size => read chunkwise
                return py_buffer(self.read())
            c_nbytes = self.size() - self.tell()
        else:
            c_nbytes = nbytes

        with nogil:
            output = GetResultValue(handle.get().ReadBuffer(c_nbytes))

        return pyarrow_wrap_buffer(output)

    def truncate(self):
        """
        NOT IMPLEMENTED
        """
        raise UnsupportedOperation()

    def writelines(self, lines):
        self._assert_writable()

        for line in lines:
            self.write(line)

    def download(self, stream_or_path, buffer_size=None):
        """
        Read this file completely to a local path or destination stream.

        This method first seeks to the beginning of the file.

        Parameters
        ----------
        stream_or_path : str or file-like object
            If a string, a local file path to write to; otherwise,
            should be a writable stream.
        buffer_size : int, optional
            The buffer size to use for data transfers.
        """
        cdef:
            int64_t bytes_read = 0
            uint8_t* buf

        handle = self.get_input_stream()

        buffer_size = buffer_size or DEFAULT_BUFFER_SIZE

        write_queue = Queue(50)

        if not hasattr(stream_or_path, 'read'):
            stream = open(stream_or_path, 'wb')

            def cleanup():
                stream.close()
        else:
            stream = stream_or_path

            def cleanup():
                pass

        done = False
        exc_info = None

        def bg_write():
            try:
                while not done or write_queue.qsize() > 0:
                    try:
                        buf = write_queue.get(timeout=0.01)
                    except QueueEmpty:
                        continue
                    stream.write(buf)
            except Exception as e:
                exc_info = sys.exc_info()
            finally:
                cleanup()

        self.seek(0)

        writer_thread = threading.Thread(target=bg_write)

        # This isn't ideal -- PyBytes_FromStringAndSize copies the data from
        # the passed buffer, so it's hard for us to avoid doubling the memory
        buf = <uint8_t*> malloc(buffer_size)
        if buf == NULL:
            raise MemoryError("Failed to allocate {0} bytes"
                              .format(buffer_size))

        writer_thread.start()

        cdef int64_t total_bytes = 0
        cdef int32_t c_buffer_size = buffer_size

        try:
            while True:
                with nogil:
                    bytes_read = GetResultValue(
                        handle.get().Read(c_buffer_size, buf))

                total_bytes += bytes_read

                # EOF
                if bytes_read == 0:
                    break

                pybuf = cp.PyBytes_FromStringAndSize(<const char*>buf,
                                                     bytes_read)

                if writer_thread.is_alive():
                    while write_queue.full():
                        time.sleep(0.01)
                else:
                    break

                write_queue.put_nowait(pybuf)
        finally:
            free(buf)
            done = True

        writer_thread.join()
        if exc_info is not None:
            raise exc_info[0], exc_info[1], exc_info[2]

    def upload(self, stream, buffer_size=None):
        """
        Write from a source stream to this file.

        Parameters
        ----------
        stream : file-like object
            Source stream to pipe to this file.
        buffer_size : int, optional
            The buffer size to use for data transfers.
        """
        write_queue = Queue(50)
        self._assert_writable()

        buffer_size = buffer_size or DEFAULT_BUFFER_SIZE

        done = False
        exc_info = None

        def bg_write():
            try:
                while not done or write_queue.qsize() > 0:
                    try:
                        buf = write_queue.get(timeout=0.01)
                    except QueueEmpty:
                        continue

                    self.write(buf)

            except Exception as e:
                exc_info = sys.exc_info()

        writer_thread = threading.Thread(target=bg_write)
        writer_thread.start()

        try:
            while True:
                buf = stream.read(buffer_size)
                if not buf:
                    break

                if writer_thread.is_alive():
                    while write_queue.full():
                        time.sleep(0.01)
                else:
                    break

                write_queue.put_nowait(buf)
        finally:
            done = True

        writer_thread.join()
        if exc_info is not None:
            raise exc_info[0], exc_info[1], exc_info[2]

BufferedIOBase.register(NativeFile)

# ----------------------------------------------------------------------
# Python file-like objects


cdef class PythonFile(NativeFile):
    """
    A stream backed by a Python file object.

    This class allows using Python file objects with arbitrary Arrow
    functions, including functions written in another language than Python.

    As a downside, there is a non-zero redirection cost in translating
    Arrow stream calls to Python method calls.  Furthermore, Python's
    Global Interpreter Lock may limit parallelism in some situations.

    Examples
    --------
    >>> import io
    >>> import pyarrow as pa
    >>> pa.PythonFile(io.BytesIO())
    <pyarrow.PythonFile closed=False own_file=False is_seekable=False is_writable=True is_readable=False>
    """
    cdef:
        object handle

    def __cinit__(self, handle, mode=None):
        self.handle = handle

        if mode is None:
            try:
                inferred_mode = handle.mode
            except AttributeError:
                # Not all file-like objects have a mode attribute
                # (e.g. BytesIO)
                try:
                    inferred_mode = 'w' if handle.writable() else 'r'
                except AttributeError:
                    raise ValueError("could not infer open mode for file-like "
                                     "object %r, please pass it explicitly"
                                     % (handle,))
        else:
            inferred_mode = mode

        if inferred_mode.startswith('w'):
            kind = 'w'
        elif inferred_mode.startswith('r'):
            kind = 'r'
        else:
            raise ValueError('Invalid file mode: {0}'.format(mode))

        # If mode was given, check it matches the given file
        if mode is not None:
            if isinstance(handle, IOBase):
                # Python 3 IO object
                if kind == 'r':
                    if not handle.readable():
                        raise TypeError("readable file expected")
                else:
                    if not handle.writable():
                        raise TypeError("writable file expected")
            # (other duck-typed file-like objects are possible)

        # If possible, check the file is a binary file
        if isinstance(handle, TextIOBase):
            raise TypeError("binary file expected, got text file")

        if kind == 'r':
            self.set_random_access_file(
                shared_ptr[CRandomAccessFile](new PyReadableFile(handle)))
            self.is_readable = True
        else:
            self.set_output_stream(
                shared_ptr[COutputStream](new PyOutputStream(handle)))
            self.is_writable = True

    def truncate(self, pos=None):
        self.handle.truncate(pos)

    def readline(self, size=None):
        return self.handle.readline(size)

    def readlines(self, hint=None):
        return self.handle.readlines(hint)


cdef class MemoryMappedFile(NativeFile):
    """
    A stream that represents a memory-mapped file.

    Supports 'r', 'r+', 'w' modes.
    """
    cdef:
        shared_ptr[CMemoryMappedFile] handle
        object path

    @staticmethod
    def create(path, size):
        """
        Create a MemoryMappedFile

        Parameters
        ----------
        path : str
            Where to create the file.
        size : int
            Size of the memory mapped file.
        """
        cdef:
            shared_ptr[CMemoryMappedFile] handle
            c_string c_path = encode_file_path(path)
            int64_t c_size = size

        with nogil:
            handle = GetResultValue(CMemoryMappedFile.Create(c_path, c_size))

        cdef MemoryMappedFile result = MemoryMappedFile()
        result.path = path
        result.is_readable = True
        result.is_writable = True
        result.set_output_stream(<shared_ptr[COutputStream]> handle)
        result.set_random_access_file(<shared_ptr[CRandomAccessFile]> handle)
        result.handle = handle

        return result

    def _open(self, path, mode='r'):
        self.path = path

        cdef:
            FileMode c_mode
            shared_ptr[CMemoryMappedFile] handle
            c_string c_path = encode_file_path(path)

        if mode in ('r', 'rb'):
            c_mode = FileMode_READ
            self.is_readable = True
        elif mode in ('w', 'wb'):
            c_mode = FileMode_WRITE
            self.is_writable = True
        elif mode in ('r+', 'r+b', 'rb+'):
            c_mode = FileMode_READWRITE
            self.is_readable = True
            self.is_writable = True
        else:
            raise ValueError('Invalid file mode: {0}'.format(mode))

        with nogil:
            handle = GetResultValue(CMemoryMappedFile.Open(c_path, c_mode))

        self.set_output_stream(<shared_ptr[COutputStream]> handle)
        self.set_random_access_file(<shared_ptr[CRandomAccessFile]> handle)
        self.handle = handle

    def resize(self, new_size):
        """
        Resize the map and underlying file.

        Parameters
        ----------
        new_size : new size in bytes
        """
        check_status(self.handle.get().Resize(new_size))

    def fileno(self):
        self._assert_open()
        return self.handle.get().file_descriptor()


def memory_map(path, mode='r'):
    """
    Open memory map at file path. Size of the memory map cannot change.

    Parameters
    ----------
    path : str
    mode : {'r', 'r+', 'w'}, default 'r'
        Whether the file is opened for reading ('r'), writing ('w')
        or both ('r+').

    Returns
    -------
    mmap : MemoryMappedFile
    """
    _check_is_file(path)

    cdef MemoryMappedFile mmap = MemoryMappedFile()
    mmap._open(path, mode)
    return mmap


cdef _check_is_file(path):
    if os.path.isdir(path):
        raise IOError("Expected file path, but {0} is a directory"
                      .format(path))


def create_memory_map(path, size):
    """
    Create a file of the given size and memory-map it.

    Parameters
    ----------
    path : str
        The file path to create, on the local filesystem.
    size : int
        The file size to create.

    Returns
    -------
    mmap : MemoryMappedFile
    """
    return MemoryMappedFile.create(path, size)


cdef class OSFile(NativeFile):
    """
    A stream backed by a regular file descriptor.
    """
    cdef:
        object path

    def __cinit__(self, path, mode='r', MemoryPool memory_pool=None):
        _check_is_file(path)
        self.path = path

        cdef:
            FileMode c_mode
            shared_ptr[Readable] handle
            c_string c_path = encode_file_path(path)

        if mode in ('r', 'rb'):
            self._open_readable(c_path, maybe_unbox_memory_pool(memory_pool))
        elif mode in ('w', 'wb'):
            self._open_writable(c_path)
        else:
            raise ValueError('Invalid file mode: {0}'.format(mode))

    cdef _open_readable(self, c_string path, CMemoryPool* pool):
        cdef shared_ptr[ReadableFile] handle

        with nogil:
            handle = GetResultValue(ReadableFile.Open(path, pool))

        self.is_readable = True
        self.set_random_access_file(<shared_ptr[CRandomAccessFile]> handle)

    cdef _open_writable(self, c_string path):
        with nogil:
            self.output_stream = GetResultValue(FileOutputStream.Open(path))
        self.is_writable = True

    def fileno(self):
        self._assert_open()
        return self.handle.file_descriptor()


cdef class FixedSizeBufferWriter(NativeFile):
    """
    A stream writing to a Arrow buffer.
    """

    def __cinit__(self, Buffer buffer):
        self.output_stream.reset(new CFixedSizeBufferWriter(buffer.buffer))
        self.is_writable = True

    def set_memcopy_threads(self, int num_threads):
        cdef CFixedSizeBufferWriter* writer = \
            <CFixedSizeBufferWriter*> self.output_stream.get()
        writer.set_memcopy_threads(num_threads)

    def set_memcopy_blocksize(self, int64_t blocksize):
        cdef CFixedSizeBufferWriter* writer = \
            <CFixedSizeBufferWriter*> self.output_stream.get()
        writer.set_memcopy_blocksize(blocksize)

    def set_memcopy_threshold(self, int64_t threshold):
        cdef CFixedSizeBufferWriter* writer = \
            <CFixedSizeBufferWriter*> self.output_stream.get()
        writer.set_memcopy_threshold(threshold)


# ----------------------------------------------------------------------
# Arrow buffers


cdef class Buffer(_Weakrefable):
    """
    The base class for all Arrow buffers.

    A buffer represents a contiguous memory area.  Many buffers will own
    their memory, though not all of them do.
    """

    def __cinit__(self):
        pass

    def __init__(self):
        raise TypeError("Do not call Buffer's constructor directly, use "
                        "`pyarrow.py_buffer` function instead.")

    cdef void init(self, const shared_ptr[CBuffer]& buffer):
        self.buffer = buffer
        self.shape[0] = self.size
        self.strides[0] = <Py_ssize_t>(1)

    def __len__(self):
        return self.size

    def __repr__(self):
        name = f"pyarrow.{self.__class__.__name__}"
        return (f"<{name} "
                f"address={hex(self.address)} "
                f"size={self.size} "
                f"is_cpu={self.is_cpu} "
                f"is_mutable={self.is_mutable}>")

    @property
    def size(self):
        """
        The buffer size in bytes.
        """
        return self.buffer.get().size()

    @property
    def address(self):
        """
        The buffer's address, as an integer.

        The returned address may point to CPU or device memory.
        Use `is_cpu()` to disambiguate.
        """
        return self.buffer.get().address()

    def hex(self):
        """
        Compute hexadecimal representation of the buffer.

        Returns
        -------
        : bytes
        """
        return self.buffer.get().ToHexString()

    @property
    def is_mutable(self):
        """
        Whether the buffer is mutable.
        """
        return self.buffer.get().is_mutable()

    @property
    def is_cpu(self):
        """
        Whether the buffer is CPU-accessible.
        """
        return self.buffer.get().is_cpu()

    @property
    def parent(self):
        cdef shared_ptr[CBuffer] parent_buf = self.buffer.get().parent()

        if parent_buf.get() == NULL:
            return None
        else:
            return pyarrow_wrap_buffer(parent_buf)

    def __getitem__(self, key):
        if isinstance(key, slice):
            if (key.step or 1) != 1:
                raise IndexError('only slices with step 1 supported')
            return _normalize_slice(self, key)

        return self.getitem(_normalize_index(key, self.size))

    cdef getitem(self, int64_t i):
        return self.buffer.get().data()[i]

    def slice(self, offset=0, length=None):
        """
        Slice this buffer.  Memory is not copied.

        You can also use the Python slice notation ``buffer[start:stop]``.

        Parameters
        ----------
        offset : int, default 0
            Offset from start of buffer to slice.
        length : int, default None
            Length of slice (default is until end of Buffer starting from
            offset).

        Returns
        -------
        sliced : Buffer
            A logical view over this buffer.
        """
        cdef shared_ptr[CBuffer] result

        if offset < 0:
            raise IndexError('Offset must be non-negative')

        if length is None:
            result = GetResultValue(SliceBufferSafe(self.buffer, offset))
        else:
            result = GetResultValue(SliceBufferSafe(self.buffer, offset,
                                                    length))
        return pyarrow_wrap_buffer(result)

    def equals(self, Buffer other):
        """
        Determine if two buffers contain exactly the same data.

        Parameters
        ----------
        other : Buffer

        Returns
        -------
        are_equal : bool
            True if buffer contents and size are equal
        """
        cdef c_bool result = False
        with nogil:
            result = self.buffer.get().Equals(deref(other.buffer.get()))
        return result

    def __eq__(self, other):
        if isinstance(other, Buffer):
            return self.equals(other)
        else:
            return self.equals(py_buffer(other))

    def __reduce_ex__(self, protocol):
        if protocol >= 5:
            bufobj = builtin_pickle.PickleBuffer(self)
        elif self.buffer.get().is_mutable():
            # Need to pass a bytearray to recreate a mutable buffer when
            # unpickling.
            bufobj = PyByteArray_FromStringAndSize(
                <const char*>self.buffer.get().data(),
                self.buffer.get().size())
        else:
            bufobj = self.to_pybytes()
        return py_buffer, (bufobj,)

    def to_pybytes(self):
        """
        Return this buffer as a Python bytes object. Memory is copied.
        """
        return cp.PyBytes_FromStringAndSize(
            <const char*>self.buffer.get().data(),
            self.buffer.get().size())

    def __getbuffer__(self, cp.Py_buffer* buffer, int flags):
        if self.buffer.get().is_mutable():
            buffer.readonly = 0
        else:
            if flags & cp.PyBUF_WRITABLE:
                raise BufferError("Writable buffer requested but Arrow "
                                  "buffer was not mutable")
            buffer.readonly = 1
        buffer.buf = <char *>self.buffer.get().data()
        buffer.len = self.size
        if buffer.buf == NULL:
            # ARROW-16048: Ensure we don't export a NULL address.
            assert buffer.len == 0
            buffer.buf = cp.PyBytes_AS_STRING(b"")
        buffer.format = 'b'
        buffer.internal = NULL
        buffer.itemsize = 1
        buffer.ndim = 1
        buffer.obj = self
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL

    def __getsegcount__(self, Py_ssize_t *len_out):
        if len_out != NULL:
            len_out[0] = <Py_ssize_t>self.size
        return 1

    def __getreadbuffer__(self, Py_ssize_t idx, void **p):
        if idx != 0:
            raise SystemError("accessing non-existent buffer segment")
        if p != NULL:
            p[0] = <void*> self.buffer.get().data()
        return self.size

    def __getwritebuffer__(self, Py_ssize_t idx, void **p):
        if not self.buffer.get().is_mutable():
            raise SystemError("trying to write an immutable buffer")
        if idx != 0:
            raise SystemError("accessing non-existent buffer segment")
        if p != NULL:
            p[0] = <void*> self.buffer.get().data()
        return self.size


cdef class ResizableBuffer(Buffer):
    """
    A base class for buffers that can be resized.
    """

    cdef void init_rz(self, const shared_ptr[CResizableBuffer]& buffer):
        self.init(<shared_ptr[CBuffer]> buffer)

    def resize(self, int64_t new_size, shrink_to_fit=False):
        """
        Resize buffer to indicated size.

        Parameters
        ----------
        new_size : int
            New size of buffer (padding may be added internally).
        shrink_to_fit : bool, default False
            If this is true, the buffer is shrunk when new_size is less
            than the current size.
            If this is false, the buffer is never shrunk.
        """
        cdef c_bool c_shrink_to_fit = shrink_to_fit
        with nogil:
            check_status((<CResizableBuffer*> self.buffer.get())
                         .Resize(new_size, c_shrink_to_fit))


cdef shared_ptr[CResizableBuffer] _allocate_buffer(CMemoryPool* pool) except *:
    with nogil:
        return to_shared(GetResultValue(AllocateResizableBuffer(0, pool)))


def allocate_buffer(int64_t size, MemoryPool memory_pool=None,
                    resizable=False):
    """
    Allocate a mutable buffer.

    Parameters
    ----------
    size : int
        Number of bytes to allocate (plus internal padding)
    memory_pool : MemoryPool, optional
        The pool to allocate memory from.
        If not given, the default memory pool is used.
    resizable : bool, default False
        If true, the returned buffer is resizable.

    Returns
    -------
    buffer : Buffer or ResizableBuffer
    """
    cdef:
        CMemoryPool* cpool = maybe_unbox_memory_pool(memory_pool)
        shared_ptr[CResizableBuffer] c_rz_buffer
        shared_ptr[CBuffer] c_buffer

    if resizable:
        with nogil:
            c_rz_buffer = to_shared(GetResultValue(
                AllocateResizableBuffer(size, cpool)))
        return pyarrow_wrap_resizable_buffer(c_rz_buffer)
    else:
        with nogil:
            c_buffer = to_shared(GetResultValue(AllocateBuffer(size, cpool)))
        return pyarrow_wrap_buffer(c_buffer)


cdef class BufferOutputStream(NativeFile):

    cdef:
        shared_ptr[CResizableBuffer] buffer

    def __cinit__(self, MemoryPool memory_pool=None):
        self.buffer = _allocate_buffer(maybe_unbox_memory_pool(memory_pool))
        self.output_stream.reset(new CBufferOutputStream(
            <shared_ptr[CResizableBuffer]> self.buffer))
        self.is_writable = True

    def getvalue(self):
        """
        Finalize output stream and return result as pyarrow.Buffer.

        Returns
        -------
        value : Buffer
        """
        with nogil:
            check_status(self.output_stream.get().Close())
        return pyarrow_wrap_buffer(<shared_ptr[CBuffer]> self.buffer)


cdef class MockOutputStream(NativeFile):

    def __cinit__(self):
        self.output_stream.reset(new CMockOutputStream())
        self.is_writable = True

    def size(self):
        handle = <CMockOutputStream*> self.output_stream.get()
        return handle.GetExtentBytesWritten()


cdef class BufferReader(NativeFile):
    """
    Zero-copy reader from objects convertible to Arrow buffer.

    Parameters
    ----------
    obj : Python bytes or pyarrow.Buffer
    """
    cdef:
        Buffer buffer

    # XXX Needed to make numpydoc happy
    def __init__(self, obj):
        pass

    def __cinit__(self, object obj):
        self.buffer = as_buffer(obj)
        self.set_random_access_file(shared_ptr[CRandomAccessFile](
            new CBufferReader(self.buffer.buffer)))
        self.is_readable = True


cdef class CompressedInputStream(NativeFile):
    """
    An input stream wrapper which decompresses data on the fly.

    Parameters
    ----------
    stream : string, path, pyarrow.NativeFile, or file-like object
        Input stream object to wrap with the compression.
    compression : str
        The compression type ("bz2", "brotli", "gzip", "lz4" or "zstd").
    """

    def __init__(self, object stream, str compression not None):
        cdef:
            NativeFile nf
            Codec codec = Codec(compression)
            shared_ptr[CInputStream] c_reader
            shared_ptr[CCompressedInputStream] compressed_stream
        nf = get_native_file(stream, False)
        c_reader = nf.get_input_stream()
        compressed_stream = GetResultValue(
            CCompressedInputStream.Make(codec.unwrap(), c_reader)
        )
        self.set_input_stream(<shared_ptr[CInputStream]> compressed_stream)
        self.is_readable = True


cdef class CompressedOutputStream(NativeFile):
    """
    An output stream wrapper which compresses data on the fly.

    Parameters
    ----------
    stream : string, path, pyarrow.NativeFile, or file-like object
        Input stream object to wrap with the compression.
    compression : str
        The compression type ("bz2", "brotli", "gzip", "lz4" or "zstd").
    """

    def __init__(self, object stream, str compression not None):
        cdef:
            Codec codec = Codec(compression)
            shared_ptr[COutputStream] c_writer
            shared_ptr[CCompressedOutputStream] compressed_stream
        get_writer(stream, &c_writer)
        compressed_stream = GetResultValue(
            CCompressedOutputStream.Make(codec.unwrap(), c_writer)
        )
        self.set_output_stream(<shared_ptr[COutputStream]> compressed_stream)
        self.is_writable = True


ctypedef CBufferedInputStream* _CBufferedInputStreamPtr
ctypedef CBufferedOutputStream* _CBufferedOutputStreamPtr
ctypedef CRandomAccessFile* _RandomAccessFilePtr


cdef class BufferedInputStream(NativeFile):
    """
    An input stream that performs buffered reads from
    an unbuffered input stream, which can mitigate the overhead
    of many small reads in some cases.

    Parameters
    ----------
    stream : NativeFile
        The input stream to wrap with the buffer
    buffer_size : int
        Size of the temporary read buffer.
    memory_pool : MemoryPool
        The memory pool used to allocate the buffer.
    """

    def __init__(self, NativeFile stream, int buffer_size,
                 MemoryPool memory_pool=None):
        cdef shared_ptr[CBufferedInputStream] buffered_stream

        if buffer_size <= 0:
            raise ValueError('Buffer size must be larger than zero')
        buffered_stream = GetResultValue(CBufferedInputStream.Create(
            buffer_size, maybe_unbox_memory_pool(memory_pool),
            stream.get_input_stream()))

        self.set_input_stream(<shared_ptr[CInputStream]> buffered_stream)
        self.is_readable = True

    def detach(self):
        """
        Release the raw InputStream.
        Further operations on this stream are invalid.

        Returns
        -------
        raw : NativeFile
            The underlying raw input stream
        """
        cdef:
            shared_ptr[CInputStream] c_raw
            _CBufferedInputStreamPtr buffered
            NativeFile raw

        buffered = dynamic_cast[_CBufferedInputStreamPtr](
            self.input_stream.get())
        assert buffered != nullptr

        with nogil:
            c_raw = GetResultValue(buffered.Detach())

        raw = NativeFile()
        raw.is_readable = True
        # Find out whether the raw stream is a RandomAccessFile
        # or a mere InputStream.  This helps us support seek() etc.
        # selectively.
        if dynamic_cast[_RandomAccessFilePtr](c_raw.get()) != nullptr:
            raw.set_random_access_file(
                static_pointer_cast[CRandomAccessFile, CInputStream](c_raw))
        else:
            raw.set_input_stream(c_raw)
        return raw


cdef class BufferedOutputStream(NativeFile):
    """
    An output stream that performs buffered reads from
    an unbuffered output stream, which can mitigate the overhead
    of many small writes in some cases.

    Parameters
    ----------
    stream : NativeFile
        The writable output stream to wrap with the buffer
    buffer_size : int
        Size of the buffer that should be added.
    memory_pool : MemoryPool
        The memory pool used to allocate the buffer.
    """

    def __init__(self, NativeFile stream, int buffer_size,
                 MemoryPool memory_pool=None):
        cdef shared_ptr[CBufferedOutputStream] buffered_stream

        if buffer_size <= 0:
            raise ValueError('Buffer size must be larger than zero')
        buffered_stream = GetResultValue(CBufferedOutputStream.Create(
            buffer_size, maybe_unbox_memory_pool(memory_pool),
            stream.get_output_stream()))

        self.set_output_stream(<shared_ptr[COutputStream]> buffered_stream)
        self.is_writable = True

    def detach(self):
        """
        Flush any buffered writes and release the raw OutputStream.
        Further operations on this stream are invalid.

        Returns
        -------
        raw : NativeFile
            The underlying raw output stream.
        """
        cdef:
            shared_ptr[COutputStream] c_raw
            _CBufferedOutputStreamPtr buffered
            NativeFile raw

        buffered = dynamic_cast[_CBufferedOutputStreamPtr](
            self.output_stream.get())
        assert buffered != nullptr

        with nogil:
            c_raw = GetResultValue(buffered.Detach())

        raw = NativeFile()
        raw.is_writable = True
        raw.set_output_stream(c_raw)
        return raw


cdef void _cb_transform(transform_func, const shared_ptr[CBuffer]& src,
                        shared_ptr[CBuffer]* dest) except *:
    py_dest = transform_func(pyarrow_wrap_buffer(src))
    dest[0] = pyarrow_unwrap_buffer(py_buffer(py_dest))


cdef class TransformInputStream(NativeFile):
    """
    Transform an input stream.

    Parameters
    ----------
    stream : NativeFile
        The stream to transform.
    transform_func : callable
        The transformation to apply.
    """

    def __init__(self, NativeFile stream, transform_func):
        self.set_input_stream(TransformInputStream.make_native(
            stream.get_input_stream(), transform_func))
        self.is_readable = True

    @staticmethod
    cdef shared_ptr[CInputStream] make_native(
            shared_ptr[CInputStream] stream, transform_func) except *:
        cdef:
            shared_ptr[CInputStream] transform_stream
            CTransformInputStreamVTable vtable

        vtable.transform = _cb_transform
        return MakeTransformInputStream(stream, move(vtable),
                                        transform_func)


class Transcoder:

    def __init__(self, decoder, encoder):
        self._decoder = decoder
        self._encoder = encoder

    def __call__(self, buf):
        final = len(buf) == 0
        return self._encoder.encode(self._decoder.decode(buf, final), final)


cdef shared_ptr[function[StreamWrapFunc]] make_streamwrap_func(
        src_encoding, dest_encoding) except *:
    """
    Create a function that will add a transcoding transformation to a stream.
    Data from that stream will be decoded according to ``src_encoding`` and
    then re-encoded according to ``dest_encoding``.
    The created function can be used to wrap streams.

    Parameters
    ----------
    src_encoding : str
        The codec to use when reading data.
    dest_encoding : str
        The codec to use for emitted data.
    """
    cdef:
        shared_ptr[function[StreamWrapFunc]] empty_func
        CTransformInputStreamVTable vtable

    vtable.transform = _cb_transform
    src_codec = codecs.lookup(src_encoding)
    dest_codec = codecs.lookup(dest_encoding)
    return MakeStreamTransformFunc(move(vtable),
                                   Transcoder(src_codec.incrementaldecoder(),
                                   dest_codec.incrementalencoder()))


def transcoding_input_stream(stream, src_encoding, dest_encoding):
    """
    Add a transcoding transformation to the stream.
    Incoming data will be decoded according to ``src_encoding`` and
    then re-encoded according to ``dest_encoding``.

    Parameters
    ----------
    stream : NativeFile
        The stream to which the transformation should be applied.
    src_encoding : str
        The codec to use when reading data.
    dest_encoding : str
        The codec to use for emitted data.
    """
    src_codec = codecs.lookup(src_encoding)
    dest_codec = codecs.lookup(dest_encoding)
    if src_codec.name == dest_codec.name:
        # Avoid losing performance on no-op transcoding
        # (encoding errors won't be detected)
        return stream
    return TransformInputStream(stream,
                                Transcoder(src_codec.incrementaldecoder(),
                                           dest_codec.incrementalencoder()))


cdef shared_ptr[CInputStream] native_transcoding_input_stream(
        shared_ptr[CInputStream] stream, src_encoding,
        dest_encoding) except *:
    src_codec = codecs.lookup(src_encoding)
    dest_codec = codecs.lookup(dest_encoding)
    if src_codec.name == dest_codec.name:
        # Avoid losing performance on no-op transcoding
        # (encoding errors won't be detected)
        return stream
    return TransformInputStream.make_native(
        stream, Transcoder(src_codec.incrementaldecoder(),
                           dest_codec.incrementalencoder()))


def py_buffer(object obj):
    """
    Construct an Arrow buffer from a Python bytes-like or buffer-like object

    Parameters
    ----------
    obj : object
        the object from which the buffer should be constructed.
    """
    cdef shared_ptr[CBuffer] buf
    buf = GetResultValue(PyBuffer.FromPyObject(obj))
    return pyarrow_wrap_buffer(buf)


def foreign_buffer(address, size, base=None):
    """
    Construct an Arrow buffer with the given *address* and *size*.

    The buffer will be optionally backed by the Python *base* object, if given.
    The *base* object will be kept alive as long as this buffer is alive,
    including across language boundaries (for example if the buffer is
    referenced by C++ code).

    Parameters
    ----------
    address : int
        The starting address of the buffer. The address can
        refer to both device or host memory but it must be
        accessible from device after mapping it with
        `get_device_address` method.
    size : int
        The size of device buffer in bytes.
    base : {None, object}
        Object that owns the referenced memory.
    """
    cdef:
        intptr_t c_addr = address
        int64_t c_size = size
        shared_ptr[CBuffer] buf

    check_status(PyForeignBuffer.Make(<uint8_t*> c_addr, c_size,
                                      base, &buf))
    return pyarrow_wrap_buffer(buf)


def as_buffer(object o):
    if isinstance(o, Buffer):
        return o
    return py_buffer(o)


cdef shared_ptr[CBuffer] as_c_buffer(object o) except *:
    cdef shared_ptr[CBuffer] buf
    if isinstance(o, Buffer):
        buf = (<Buffer> o).buffer
        if buf == nullptr:
            raise ValueError("got null buffer")
    else:
        buf = GetResultValue(PyBuffer.FromPyObject(o))
    return buf


cdef NativeFile get_native_file(object source, c_bool use_memory_map):
    try:
        source_path = _stringify_path(source)
    except TypeError:
        if isinstance(source, Buffer):
            source = BufferReader(source)
        elif not isinstance(source, NativeFile) and hasattr(source, 'read'):
            # Optimistically hope this is file-like
            source = PythonFile(source, mode='r')
    else:
        if use_memory_map:
            source = memory_map(source_path, mode='r')
        else:
            source = OSFile(source_path, mode='r')

    return source


cdef get_reader(object source, c_bool use_memory_map,
                shared_ptr[CRandomAccessFile]* reader):
    cdef NativeFile nf

    nf = get_native_file(source, use_memory_map)
    reader[0] = nf.get_random_access_file()


cdef get_input_stream(object source, c_bool use_memory_map,
                      shared_ptr[CInputStream]* out):
    """
    Like get_reader(), but can automatically decompress, and returns
    an InputStream.
    """
    cdef:
        NativeFile nf
        Codec codec
        shared_ptr[CInputStream] input_stream

    try:
        codec = Codec.detect(source)
    except TypeError:
        codec = None

    nf = get_native_file(source, use_memory_map)
    input_stream = nf.get_input_stream()

    # codec is None if compression can't be detected
    if codec is not None:
        input_stream = <shared_ptr[CInputStream]> GetResultValue(
            CCompressedInputStream.Make(codec.unwrap(), input_stream)
        )

    out[0] = input_stream


cdef get_writer(object source, shared_ptr[COutputStream]* writer):
    cdef NativeFile nf

    try:
        source_path = _stringify_path(source)
    except TypeError:
        if not isinstance(source, NativeFile) and hasattr(source, 'write'):
            # Optimistically hope this is file-like
            source = PythonFile(source, mode='w')
    else:
        source = OSFile(source_path, mode='w')

    if isinstance(source, NativeFile):
        nf = source
        writer[0] = nf.get_output_stream()
    else:
        raise TypeError('Unable to read from object of type: {0}'
                        .format(type(source)))


# ---------------------------------------------------------------------


def _detect_compression(path):
    if isinstance(path, str):
        if path.endswith('.bz2'):
            return 'bz2'
        elif path.endswith('.gz'):
            return 'gzip'
        elif path.endswith('.lz4'):
            return 'lz4'
        elif path.endswith('.zst'):
            return 'zstd'


cdef CCompressionType _ensure_compression(str name) except *:
    uppercase = name.upper()
    if uppercase == 'BZ2':
        return CCompressionType_BZ2
    elif uppercase == 'GZIP':
        return CCompressionType_GZIP
    elif uppercase == 'BROTLI':
        return CCompressionType_BROTLI
    elif uppercase == 'LZ4' or uppercase == 'LZ4_FRAME':
        return CCompressionType_LZ4_FRAME
    elif uppercase == 'LZ4_RAW':
        return CCompressionType_LZ4
    elif uppercase == 'SNAPPY':
        return CCompressionType_SNAPPY
    elif uppercase == 'ZSTD':
        return CCompressionType_ZSTD
    else:
        raise ValueError('Invalid value for compression: {!r}'.format(name))


cdef class Codec(_Weakrefable):
    """
    Compression codec.

    Parameters
    ----------
    compression : str
        Type of compression codec to initialize, valid values are: 'gzip',
        'bz2', 'brotli', 'lz4' (or 'lz4_frame'), 'lz4_raw', 'zstd' and
        'snappy'.
    compression_level : int, None
        Optional parameter specifying how aggressively to compress.  The
        possible ranges and effect of this parameter depend on the specific
        codec chosen.  Higher values compress more but typically use more
        resources (CPU/RAM).  Some codecs support negative values.

        gzip
            The compression_level maps to the memlevel parameter of
            deflateInit2.  Higher levels use more RAM but are faster
            and should have higher compression ratios.

        bz2
            The compression level maps to the blockSize100k parameter of
            the BZ2_bzCompressInit function.  Higher levels use more RAM
            but are faster and should have higher compression ratios.

        brotli
            The compression level maps to the BROTLI_PARAM_QUALITY
            parameter.  Higher values are slower and should have higher
            compression ratios.

        lz4/lz4_frame/lz4_raw
            The compression level parameter is not supported and must
            be None

        zstd
            The compression level maps to the compressionLevel parameter
            of ZSTD_initCStream.  Negative values are supported.  Higher
            values are slower and should have higher compression ratios.

        snappy
            The compression level parameter is not supported and must
            be None


    Raises
    ------
    ValueError
        If invalid compression value is passed.

    Examples
    --------
    >>> import pyarrow as pa
    >>> pa.Codec.is_available('gzip')
    True
    >>> codec = pa.Codec('gzip')
    >>> codec.name
    'gzip'
    >>> codec.compression_level
    9
    """

    def __init__(self, str compression not None, compression_level=None):
        cdef CCompressionType typ = _ensure_compression(compression)
        if compression_level is not None:
            self.wrapped = shared_ptr[CCodec](move(GetResultValue(
                CCodec.CreateWithLevel(typ, compression_level))))
        else:
            self.wrapped = shared_ptr[CCodec](move(GetResultValue(
                CCodec.Create(typ))))

    cdef inline CCodec* unwrap(self) nogil:
        return self.wrapped.get()

    @staticmethod
    def detect(path):
        """
        Detect and instantiate compression codec based on file extension.

        Parameters
        ----------
        path : str, path-like
            File-path to detect compression from.

        Raises
        ------
        TypeError
            If the passed value is not path-like.
        ValueError
            If the compression can't be detected from the path.

        Returns
        -------
        Codec
        """
        return Codec(_detect_compression(_stringify_path(path)))

    @staticmethod
    def is_available(str compression not None):
        """
        Returns whether the compression support has been built and enabled.

        Parameters
        ----------
        compression : str
             Type of compression codec,
             refer to Codec docstring for a list of supported ones.

        Returns
        -------
        bool
        """
        cdef CCompressionType typ = _ensure_compression(compression)
        return CCodec.IsAvailable(typ)

    @staticmethod
    def supports_compression_level(str compression not None):
        """
        Returns true if the compression level parameter is supported
        for the given codec.

        Parameters
        ----------
        compression : str
            Type of compression codec,
            refer to Codec docstring for a list of supported ones.
        """
        cdef CCompressionType typ = _ensure_compression(compression)
        return CCodec.SupportsCompressionLevel(typ)

    @staticmethod
    def default_compression_level(str compression not None):
        """
        Returns the compression level that Arrow will use for the codec if
        None is specified.

        Parameters
        ----------
        compression : str
            Type of compression codec,
            refer to Codec docstring for a list of supported ones.
        """
        cdef CCompressionType typ = _ensure_compression(compression)
        return GetResultValue(CCodec.DefaultCompressionLevel(typ))

    @staticmethod
    def minimum_compression_level(str compression not None):
        """
        Returns the smallest valid value for the compression level

        Parameters
        ----------
        compression : str
            Type of compression codec,
            refer to Codec docstring for a list of supported ones.
        """
        cdef CCompressionType typ = _ensure_compression(compression)
        return GetResultValue(CCodec.MinimumCompressionLevel(typ))

    @staticmethod
    def maximum_compression_level(str compression not None):
        """
        Returns the largest valid value for the compression level

        Parameters
        ----------
        compression : str
            Type of compression codec,
            refer to Codec docstring for a list of supported ones.
        """
        cdef CCompressionType typ = _ensure_compression(compression)
        return GetResultValue(CCodec.MaximumCompressionLevel(typ))

    @property
    def name(self):
        """Returns the name of the codec"""
        return frombytes(self.unwrap().name())

    @property
    def compression_level(self):
        """Returns the compression level parameter of the codec"""
        if self.name == 'snappy':
            return None
        return self.unwrap().compression_level()

    def compress(self, object buf, asbytes=False, memory_pool=None):
        """
        Compress data from buffer-like object.

        Parameters
        ----------
        buf : pyarrow.Buffer, bytes, or other object supporting buffer protocol
        asbytes : bool, default False
            Return result as Python bytes object, otherwise Buffer
        memory_pool : MemoryPool, default None
            Memory pool to use for buffer allocations, if any

        Returns
        -------
        compressed : pyarrow.Buffer or bytes (if asbytes=True)
        """
        cdef:
            shared_ptr[CBuffer] owned_buf
            CBuffer* c_buf
            PyObject* pyobj
            ResizableBuffer out_buf
            int64_t max_output_size
            int64_t output_length
            uint8_t* output_buffer = NULL

        owned_buf = as_c_buffer(buf)
        c_buf = owned_buf.get()

        max_output_size = self.wrapped.get().MaxCompressedLen(
            c_buf.size(), c_buf.data()
        )

        if asbytes:
            pyobj = PyBytes_FromStringAndSizeNative(NULL, max_output_size)
            output_buffer = <uint8_t*> cp.PyBytes_AS_STRING(<object> pyobj)
        else:
            out_buf = allocate_buffer(
                max_output_size, memory_pool=memory_pool, resizable=True
            )
            output_buffer = out_buf.buffer.get().mutable_data()

        with nogil:
            output_length = GetResultValue(
                self.unwrap().Compress(
                    c_buf.size(),
                    c_buf.data(),
                    max_output_size,
                    output_buffer
                )
            )

        if asbytes:
            cp._PyBytes_Resize(&pyobj, <Py_ssize_t> output_length)
            return PyObject_to_object(pyobj)
        else:
            out_buf.resize(output_length)
            return out_buf

    def decompress(self, object buf, decompressed_size=None, asbytes=False,
                   memory_pool=None):
        """
        Decompress data from buffer-like object.

        Parameters
        ----------
        buf : pyarrow.Buffer, bytes, or memoryview-compatible object
        decompressed_size : int, default None
            Size of the decompressed result
        asbytes : boolean, default False
            Return result as Python bytes object, otherwise Buffer
        memory_pool : MemoryPool, default None
            Memory pool to use for buffer allocations, if any.

        Returns
        -------
        uncompressed : pyarrow.Buffer or bytes (if asbytes=True)
        """
        cdef:
            shared_ptr[CBuffer] owned_buf
            CBuffer* c_buf
            Buffer out_buf
            int64_t output_size
            uint8_t* output_buffer = NULL

        owned_buf = as_c_buffer(buf)
        c_buf = owned_buf.get()

        if decompressed_size is None:
            raise ValueError(
                "Must pass decompressed_size"
            )

        output_size = decompressed_size

        if asbytes:
            pybuf = cp.PyBytes_FromStringAndSize(NULL, output_size)
            output_buffer = <uint8_t*> cp.PyBytes_AS_STRING(pybuf)
        else:
            out_buf = allocate_buffer(output_size, memory_pool=memory_pool)
            output_buffer = out_buf.buffer.get().mutable_data()

        with nogil:
            GetResultValue(
                self.unwrap().Decompress(
                    c_buf.size(),
                    c_buf.data(),
                    output_size,
                    output_buffer
                )
            )

        return pybuf if asbytes else out_buf

    def __repr__(self):
        name = f"pyarrow.{self.__class__.__name__}"
        return (f"<{name} "
                f"name={self.name} "
                f"compression_level={self.compression_level}>")


def compress(object buf, codec='lz4', asbytes=False, memory_pool=None):
    """
    Compress data from buffer-like object.

    Parameters
    ----------
    buf : pyarrow.Buffer, bytes, or other object supporting buffer protocol
    codec : str, default 'lz4'
        Compression codec.
        Supported types: {'brotli, 'gzip', 'lz4', 'lz4_raw', 'snappy', 'zstd'}
    asbytes : bool, default False
        Return result as Python bytes object, otherwise Buffer.
    memory_pool : MemoryPool, default None
        Memory pool to use for buffer allocations, if any.

    Returns
    -------
    compressed : pyarrow.Buffer or bytes (if asbytes=True)
    """
    cdef Codec coder = Codec(codec)
    return coder.compress(buf, asbytes=asbytes, memory_pool=memory_pool)


def decompress(object buf, decompressed_size=None, codec='lz4',
               asbytes=False, memory_pool=None):
    """
    Decompress data from buffer-like object.

    Parameters
    ----------
    buf : pyarrow.Buffer, bytes, or memoryview-compatible object
        Input object to decompress data from.
    decompressed_size : int, default None
        Size of the decompressed result
    codec : str, default 'lz4'
        Compression codec.
        Supported types: {'brotli, 'gzip', 'lz4', 'lz4_raw', 'snappy', 'zstd'}
    asbytes : bool, default False
        Return result as Python bytes object, otherwise Buffer.
    memory_pool : MemoryPool, default None
        Memory pool to use for buffer allocations, if any.

    Returns
    -------
    uncompressed : pyarrow.Buffer or bytes (if asbytes=True)
    """
    cdef Codec decoder = Codec(codec)
    return decoder.decompress(buf, asbytes=asbytes, memory_pool=memory_pool,
                              decompressed_size=decompressed_size)


def input_stream(source, compression='detect', buffer_size=None):
    """
    Create an Arrow input stream.

    Parameters
    ----------
    source : str, Path, buffer, or file-like object
        The source to open for reading.
    compression : str optional, default 'detect'
        The compression algorithm to use for on-the-fly decompression.
        If "detect" and source is a file path, then compression will be
        chosen based on the file extension.
        If None, no compression will be applied.
        Otherwise, a well-known algorithm name must be supplied (e.g. "gzip").
    buffer_size : int, default None
        If None or 0, no buffering will happen. Otherwise the size of the
        temporary read buffer.
    """
    cdef NativeFile stream

    try:
        source_path = _stringify_path(source)
    except TypeError:
        source_path = None

    if isinstance(source, NativeFile):
        stream = source
    elif source_path is not None:
        stream = OSFile(source_path, 'r')
    elif isinstance(source, (Buffer, memoryview)):
        stream = BufferReader(as_buffer(source))
    elif (hasattr(source, 'read') and
          hasattr(source, 'close') and
          hasattr(source, 'closed')):
        stream = PythonFile(source, 'r')
    else:
        raise TypeError("pa.input_stream() called with instance of '{}'"
                        .format(source.__class__))

    if compression == 'detect':
        # detect for OSFile too
        compression = _detect_compression(source_path)

    if buffer_size is not None and buffer_size != 0:
        stream = BufferedInputStream(stream, buffer_size)

    if compression is not None:
        stream = CompressedInputStream(stream, compression)

    return stream


def output_stream(source, compression='detect', buffer_size=None):
    """
    Create an Arrow output stream.

    Parameters
    ----------
    source : str, Path, buffer, file-like object
        The source to open for writing.
    compression : str optional, default 'detect'
        The compression algorithm to use for on-the-fly compression.
        If "detect" and source is a file path, then compression will be
        chosen based on the file extension.
        If None, no compression will be applied.
        Otherwise, a well-known algorithm name must be supplied (e.g. "gzip").
    buffer_size : int, default None
        If None or 0, no buffering will happen. Otherwise the size of the
        temporary write buffer.
    """
    cdef NativeFile stream

    try:
        source_path = _stringify_path(source)
    except TypeError:
        source_path = None

    if isinstance(source, NativeFile):
        stream = source
    elif source_path is not None:
        stream = OSFile(source_path, 'w')
    elif isinstance(source, (Buffer, memoryview)):
        stream = FixedSizeBufferWriter(as_buffer(source))
    elif (hasattr(source, 'write') and
          hasattr(source, 'close') and
          hasattr(source, 'closed')):
        stream = PythonFile(source, 'w')
    else:
        raise TypeError("pa.output_stream() called with instance of '{}'"
                        .format(source.__class__))

    if compression == 'detect':
        compression = _detect_compression(source_path)

    if buffer_size is not None and buffer_size != 0:
        stream = BufferedOutputStream(stream, buffer_size)

    if compression is not None:
        stream = CompressedOutputStream(stream, compression)

    return stream

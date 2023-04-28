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

# cython: language_level = 3

from cpython.datetime cimport datetime, PyDateTime_DateTime

from pyarrow.includes.common cimport *
from pyarrow.includes.libarrow_python cimport PyDateTime_to_TimePoint
from pyarrow.lib import _detect_compression, frombytes, tobytes
from pyarrow.lib cimport *
from pyarrow.util import _stringify_path

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import os
import pathlib
import sys


cdef _init_ca_paths():
    cdef CFileSystemGlobalOptions options

    import ssl
    paths = ssl.get_default_verify_paths()
    if paths.cafile:
        options.tls_ca_file_path = os.fsencode(paths.cafile)
    if paths.capath:
        options.tls_ca_dir_path = os.fsencode(paths.capath)
    check_status(CFileSystemsInitialize(options))


if sys.platform == 'linux':
    # ARROW-9261: On Linux, we may need to fixup the paths to TLS CA certs
    # (especially in manylinux packages) since the values hardcoded at
    # compile-time in libcurl may be wrong.
    _init_ca_paths()


cdef inline c_string _path_as_bytes(path) except *:
    # handle only abstract paths, not bound to any filesystem like pathlib is,
    # so we only accept plain strings
    if not isinstance(path, (bytes, str)):
        raise TypeError('Path must be a string')
    # tobytes always uses utf-8, which is more or less ok, at least on Windows
    # since the C++ side then decodes from utf-8. On Unix, os.fsencode may be
    # better.
    return tobytes(path)


cdef object _wrap_file_type(CFileType ty):
    return FileType(<int8_t> ty)


cdef CFileType _unwrap_file_type(FileType ty) except *:
    if ty == FileType.Unknown:
        return CFileType_Unknown
    elif ty == FileType.NotFound:
        return CFileType_NotFound
    elif ty == FileType.File:
        return CFileType_File
    elif ty == FileType.Directory:
        return CFileType_Directory
    assert 0


def _file_type_to_string(ty):
    # Python 3.11 changed str(IntEnum) to return the string representation
    # of the integer value: https://github.com/python/cpython/issues/94763
    return f"{ty.__class__.__name__}.{ty._name_}"


cdef class FileInfo(_Weakrefable):
    """
    FileSystem entry info.

    Parameters
    ----------
    path : str
        The full path to the filesystem entry.
    type : FileType
        The type of the filesystem entry.
    mtime : datetime or float, default None
        If given, the modification time of the filesystem entry.
        If a float is given, it is the number of seconds since the
        Unix epoch.
    mtime_ns : int, default None
        If given, the modification time of the filesystem entry,
        in nanoseconds since the Unix epoch.
        `mtime` and `mtime_ns` are mutually exclusive.
    size : int, default None
        If given, the filesystem entry size in bytes.  This should only
        be given if `type` is `FileType.File`.

    Examples
    --------
    Generate a file:

    >>> from pyarrow import fs
    >>> local = fs.LocalFileSystem()
    >>> path_fs = local_path + '/pyarrow-fs-example.dat'
    >>> with local.open_output_stream(path_fs) as stream:
    ...     stream.write(b'data')
    4

    Get FileInfo object using ``get_file_info()``:

    >>> file_info = local.get_file_info(path_fs)
    >>> file_info
    <FileInfo for '.../pyarrow-fs-example.dat': type=FileType.File, size=4>

    Inspect FileInfo attributes:

    >>> file_info.type
    <FileType.File: 2>

    >>> file_info.is_file
    True

    >>> file_info.path
    '/.../pyarrow-fs-example.dat'

    >>> file_info.base_name
    'pyarrow-fs-example.dat'

    >>> file_info.size
    4

    >>> file_info.extension
    'dat'

    >>> file_info.mtime # doctest: +SKIP
    datetime.datetime(2022, 6, 29, 7, 56, 10, 873922, tzinfo=datetime.timezone.utc)

    >>> file_info.mtime_ns # doctest: +SKIP
    1656489370873922073
    """

    def __init__(self, path, FileType type=FileType.Unknown, *,
                 mtime=None, mtime_ns=None, size=None):
        self.info.set_path(tobytes(path))
        self.info.set_type(_unwrap_file_type(type))
        if mtime is not None:
            if mtime_ns is not None:
                raise TypeError("Only one of mtime and mtime_ns "
                                "can be given")
            if isinstance(mtime, datetime):
                self.info.set_mtime(PyDateTime_to_TimePoint(
                    <PyDateTime_DateTime*> mtime))
            else:
                self.info.set_mtime(TimePoint_from_s(mtime))
        elif mtime_ns is not None:
            self.info.set_mtime(TimePoint_from_ns(mtime_ns))
        if size is not None:
            self.info.set_size(size)

    @staticmethod
    cdef wrap(CFileInfo info):
        cdef FileInfo self = FileInfo.__new__(FileInfo)
        self.info = move(info)
        return self

    cdef inline CFileInfo unwrap(self) nogil:
        return self.info

    @staticmethod
    cdef CFileInfo unwrap_safe(obj):
        if not isinstance(obj, FileInfo):
            raise TypeError("Expected FileInfo instance, got {0}"
                            .format(type(obj)))
        return (<FileInfo> obj).unwrap()

    def __repr__(self):
        def getvalue(attr):
            try:
                return getattr(self, attr)
            except ValueError:
                return ''

        s = (f'<FileInfo for {self.path!r}: '
             f'type={_file_type_to_string(self.type)}')
        if self.is_file:
            s += f', size={self.size}'
        s += '>'
        return s

    @property
    def type(self):
        """
        Type of the file.

        The returned enum values can be the following:

        - FileType.NotFound: target does not exist
        - FileType.Unknown: target exists but its type is unknown (could be a
          special file such as a Unix socket or character device, or
          Windows NUL / CON / ...)
        - FileType.File: target is a regular file
        - FileType.Directory: target is a regular directory

        Returns
        -------
        type : FileType
        """
        return _wrap_file_type(self.info.type())

    @property
    def is_file(self):
        """
        """
        return self.type == FileType.File

    @property
    def path(self):
        """
        The full file path in the filesystem.

        Examples
        --------
        >>> file_info = local.get_file_info(path)
        >>> file_info.path
        '/.../pyarrow-fs-example.dat'
        """
        return frombytes(self.info.path())

    @property
    def base_name(self):
        """
        The file base name.

        Component after the last directory separator.

        Examples
        --------
        >>> file_info = local.get_file_info(path)
        >>> file_info.base_name
        'pyarrow-fs-example.dat'
        """
        return frombytes(self.info.base_name())

    @property
    def size(self):
        """
        The size in bytes, if available.

        Only regular files are guaranteed to have a size.

        Returns
        -------
        size : int or None
        """
        cdef int64_t size
        size = self.info.size()
        return (size if size != -1 else None)

    @property
    def extension(self):
        """
        The file extension.

        Examples
        --------
        >>> file_info = local.get_file_info(path)
        >>> file_info.extension
        'dat'
        """
        return frombytes(self.info.extension())

    @property
    def mtime(self):
        """
        The time of last modification, if available.

        Returns
        -------
        mtime : datetime.datetime or None

        Examples
        --------
        >>> file_info = local.get_file_info(path)
        >>> file_info.mtime # doctest: +SKIP
        datetime.datetime(2022, 6, 29, 7, 56, 10, 873922, tzinfo=datetime.timezone.utc)
        """
        cdef int64_t nanoseconds
        nanoseconds = TimePoint_to_ns(self.info.mtime())
        return (datetime.fromtimestamp(nanoseconds / 1.0e9, timezone.utc)
                if nanoseconds != -1 else None)

    @property
    def mtime_ns(self):
        """
        The time of last modification, if available, expressed in nanoseconds
        since the Unix epoch.

        Returns
        -------
        mtime_ns : int or None

        Examples
        --------
        >>> file_info = local.get_file_info(path)
        >>> file_info.mtime_ns # doctest: +SKIP
        1656489370873922073
        """
        cdef int64_t nanoseconds
        nanoseconds = TimePoint_to_ns(self.info.mtime())
        return (nanoseconds if nanoseconds != -1 else None)


cdef class FileSelector(_Weakrefable):
    """
    File and directory selector.

    It contains a set of options that describes how to search for files and
    directories.

    Parameters
    ----------
    base_dir : str
        The directory in which to select files. Relative paths also work, use
        '.' for the current directory and '..' for the parent.
    allow_not_found : bool, default False
        The behavior if `base_dir` doesn't exist in the filesystem.
        If false, an error is returned.
        If true, an empty selection is returned.
    recursive : bool, default False
        Whether to recurse into subdirectories.

    Examples
    --------
    List the contents of a directory and subdirectories:

    >>> selector_1 = fs.FileSelector(local_path, recursive=True)
    >>> local.get_file_info(selector_1) # doctest: +SKIP
    [<FileInfo for 'tmp/alphabet/example.dat': type=FileType.File, size=4>,
    <FileInfo for 'tmp/alphabet/subdir': type=FileType.Directory>,
    <FileInfo for 'tmp/alphabet/subdir/example_copy.dat': type=FileType.File, size=4>]

    List only the contents of the base directory:

    >>> selector_2 = fs.FileSelector(local_path)
    >>> local.get_file_info(selector_2) # doctest: +SKIP
    [<FileInfo for 'tmp/alphabet/example.dat': type=FileType.File, size=4>,
    <FileInfo for 'tmp/alphabet/subdir': type=FileType.Directory>]

    Return empty selection if the directory doesn't exist:

    >>> selector_not_found = fs.FileSelector(local_path + '/missing',
    ...                                      recursive=True,
    ...                                      allow_not_found=True)
    >>> local.get_file_info(selector_not_found)
    []
    """

    def __init__(self, base_dir, bint allow_not_found=False,
                 bint recursive=False):
        self.base_dir = base_dir
        self.recursive = recursive
        self.allow_not_found = allow_not_found

    @staticmethod
    cdef FileSelector wrap(CFileSelector wrapped):
        cdef FileSelector self = FileSelector.__new__(FileSelector)
        self.selector = move(wrapped)
        return self

    cdef inline CFileSelector unwrap(self) nogil:
        return self.selector

    @property
    def base_dir(self):
        return frombytes(self.selector.base_dir)

    @base_dir.setter
    def base_dir(self, base_dir):
        self.selector.base_dir = _path_as_bytes(base_dir)

    @property
    def allow_not_found(self):
        return self.selector.allow_not_found

    @allow_not_found.setter
    def allow_not_found(self, bint allow_not_found):
        self.selector.allow_not_found = allow_not_found

    @property
    def recursive(self):
        return self.selector.recursive

    @recursive.setter
    def recursive(self, bint recursive):
        self.selector.recursive = recursive

    def __repr__(self):
        return ("<FileSelector base_dir={0.base_dir!r} "
                "recursive={0.recursive}>".format(self))


cdef class FileSystem(_Weakrefable):
    """
    Abstract file system API.
    """

    def __init__(self):
        raise TypeError("FileSystem is an abstract class, instantiate one of "
                        "the subclasses instead: LocalFileSystem or "
                        "SubTreeFileSystem")

    @staticmethod
    def from_uri(uri):
        """
        Create a new FileSystem from URI or Path.

        Recognized URI schemes are "file", "mock", "s3fs", "gs", "gcs", "hdfs" and "viewfs".
        In addition, the argument can be a pathlib.Path object, or a string
        describing an absolute local path.

        Parameters
        ----------
        uri : string
            URI-based path, for example: file:///some/local/path.

        Returns
        -------
        tuple of (FileSystem, str path)
            With (filesystem, path) tuple where path is the abstract path
            inside the FileSystem instance.

        Examples
        --------
        Create a new FileSystem subclass from a URI:

        >>> uri = 'file:///{}/pyarrow-fs-example.dat'.format(local_path)
        >>> local_new, path_new = fs.FileSystem.from_uri(uri)
        >>> local_new
        <pyarrow._fs.LocalFileSystem object at ...
        >>> path_new
        '/.../pyarrow-fs-example.dat'

        Or from a s3 bucket:

        >>> fs.FileSystem.from_uri("s3://usgs-landsat/collection02/")
        (<pyarrow._s3fs.S3FileSystem object at ...>, 'usgs-landsat/collection02')
        """
        cdef:
            c_string c_path
            c_string c_uri
            CResult[shared_ptr[CFileSystem]] result

        if isinstance(uri, pathlib.Path):
            # Make absolute
            uri = uri.resolve().absolute()
        c_uri = tobytes(_stringify_path(uri))
        with nogil:
            result = CFileSystemFromUriOrPath(c_uri, &c_path)
        return FileSystem.wrap(GetResultValue(result)), frombytes(c_path)

    cdef init(self, const shared_ptr[CFileSystem]& wrapped):
        self.wrapped = wrapped
        self.fs = wrapped.get()

    @staticmethod
    cdef wrap(const shared_ptr[CFileSystem]& sp):
        cdef FileSystem self

        typ = frombytes(sp.get().type_name())
        if typ == 'local':
            self = LocalFileSystem.__new__(LocalFileSystem)
        elif typ == 'mock':
            self = _MockFileSystem.__new__(_MockFileSystem)
        elif typ == 'subtree':
            self = SubTreeFileSystem.__new__(SubTreeFileSystem)
        elif typ == 's3':
            from pyarrow._s3fs import S3FileSystem
            self = S3FileSystem.__new__(S3FileSystem)
        elif typ == 'gcs':
            from pyarrow._gcsfs import GcsFileSystem
            self = GcsFileSystem.__new__(GcsFileSystem)
        elif typ == 'hdfs':
            from pyarrow._hdfs import HadoopFileSystem
            self = HadoopFileSystem.__new__(HadoopFileSystem)
        elif typ.startswith('py::'):
            self = PyFileSystem.__new__(PyFileSystem)
        else:
            raise TypeError('Cannot wrap FileSystem pointer')

        self.init(sp)
        return self

    cdef inline shared_ptr[CFileSystem] unwrap(self) nogil:
        return self.wrapped

    def equals(self, FileSystem other):
        return self.fs.Equals(other.unwrap())

    def __eq__(self, other):
        try:
            return self.equals(other)
        except TypeError:
            return NotImplemented

    @property
    def type_name(self):
        """
        The filesystem's type name.
        """
        return frombytes(self.fs.type_name())

    def get_file_info(self, paths_or_selector):
        """
        Get info for the given files.

        Any symlink is automatically dereferenced, recursively. A non-existing
        or unreachable file returns a FileStat object and has a FileType of
        value NotFound. An exception indicates a truly exceptional condition
        (low-level I/O error, etc.).

        Parameters
        ----------
        paths_or_selector : FileSelector, path-like or list of path-likes
            Either a selector object, a path-like object or a list of
            path-like objects. The selector's base directory will not be
            part of the results, even if it exists. If it doesn't exist,
            use `allow_not_found`.

        Returns
        -------
        FileInfo or list of FileInfo
            Single FileInfo object is returned for a single path, otherwise
            a list of FileInfo objects is returned.

        Examples
        --------
        >>> local
        <pyarrow._fs.LocalFileSystem object at ...>
        >>> local.get_file_info("/{}/pyarrow-fs-example.dat".format(local_path))
        <FileInfo for '/.../pyarrow-fs-example.dat': type=FileType.File, size=4>
        """
        cdef:
            CFileInfo info
            c_string path
            vector[CFileInfo] infos
            vector[c_string] paths
            CFileSelector selector

        if isinstance(paths_or_selector, FileSelector):
            with nogil:
                selector = (<FileSelector>paths_or_selector).selector
                infos = GetResultValue(self.fs.GetFileInfo(selector))
        elif isinstance(paths_or_selector, (list, tuple)):
            paths = [_path_as_bytes(s) for s in paths_or_selector]
            with nogil:
                infos = GetResultValue(self.fs.GetFileInfo(paths))
        elif isinstance(paths_or_selector, (bytes, str)):
            path =_path_as_bytes(paths_or_selector)
            with nogil:
                info = GetResultValue(self.fs.GetFileInfo(path))
            return FileInfo.wrap(info)
        else:
            raise TypeError('Must pass either path(s) or a FileSelector')

        return [FileInfo.wrap(info) for info in infos]

    def create_dir(self, path, *, bint recursive=True):
        """
        Create a directory and subdirectories.

        This function succeeds if the directory already exists.

        Parameters
        ----------
        path : str
            The path of the new directory.
        recursive : bool, default True
            Create nested directories as well.
        """
        cdef c_string directory = _path_as_bytes(path)
        with nogil:
            check_status(self.fs.CreateDir(directory, recursive=recursive))

    def delete_dir(self, path):
        """
        Delete a directory and its contents, recursively.

        Parameters
        ----------
        path : str
            The path of the directory to be deleted.
        """
        cdef c_string directory = _path_as_bytes(path)
        with nogil:
            check_status(self.fs.DeleteDir(directory))

    def delete_dir_contents(self, path, *,
                            bint accept_root_dir=False,
                            bint missing_dir_ok=False):
        """
        Delete a directory's contents, recursively.

        Like delete_dir, but doesn't delete the directory itself.

        Parameters
        ----------
        path : str
            The path of the directory to be deleted.
        accept_root_dir : boolean, default False
            Allow deleting the root directory's contents
            (if path is empty or "/")
        missing_dir_ok : boolean, default False
            If False then an error is raised if path does
            not exist
        """
        cdef c_string directory = _path_as_bytes(path)
        if accept_root_dir and directory.strip(b"/") == b"":
            with nogil:
                check_status(self.fs.DeleteRootDirContents())
        else:
            with nogil:
                check_status(self.fs.DeleteDirContents(directory,
                             missing_dir_ok))

    def move(self, src, dest):
        """
        Move / rename a file or directory.

        If the destination exists:
        - if it is a non-empty directory, an error is returned
        - otherwise, if it has the same type as the source, it is replaced
        - otherwise, behavior is unspecified (implementation-dependent).

        Parameters
        ----------
        src : str
            The path of the file or the directory to be moved.
        dest : str
            The destination path where the file or directory is moved to.

        Examples
        --------
        Create a new folder with a file:

        >>> local.create_dir('/tmp/other_dir')
        >>> local.copy_file(path,'/tmp/move_example.dat')

        Move the file:

        >>> local.move('/tmp/move_example.dat',
        ...            '/tmp/other_dir/move_example_2.dat')

        Inspect the file info:

        >>> local.get_file_info('/tmp/other_dir/move_example_2.dat')
        <FileInfo for '/tmp/other_dir/move_example_2.dat': type=FileType.File, size=4>
        >>> local.get_file_info('/tmp/move_example.dat')
        <FileInfo for '/tmp/move_example.dat': type=FileType.NotFound>

        Delete the folder:
        >>> local.delete_dir('/tmp/other_dir')
        """
        cdef:
            c_string source = _path_as_bytes(src)
            c_string destination = _path_as_bytes(dest)
        with nogil:
            check_status(self.fs.Move(source, destination))

    def copy_file(self, src, dest):
        """
        Copy a file.

        If the destination exists and is a directory, an error is returned.
        Otherwise, it is replaced.

        Parameters
        ----------
        src : str
            The path of the file to be copied from.
        dest : str
            The destination path where the file is copied to.

        Examples
        --------
        >>> local.copy_file(path,
        ...                 local_path + '/pyarrow-fs-example_copy.dat')

        Inspect the file info:

        >>> local.get_file_info(local_path + '/pyarrow-fs-example_copy.dat')
        <FileInfo for '/.../pyarrow-fs-example_copy.dat': type=FileType.File, size=4>
        >>> local.get_file_info(path)
        <FileInfo for '/.../pyarrow-fs-example.dat': type=FileType.File, size=4>
        """
        cdef:
            c_string source = _path_as_bytes(src)
            c_string destination = _path_as_bytes(dest)
        with nogil:
            check_status(self.fs.CopyFile(source, destination))

    def delete_file(self, path):
        """
        Delete a file.

        Parameters
        ----------
        path : str
            The path of the file to be deleted.
        """
        cdef c_string file = _path_as_bytes(path)
        with nogil:
            check_status(self.fs.DeleteFile(file))

    def _wrap_input_stream(self, stream, path, compression, buffer_size):
        if buffer_size is not None and buffer_size != 0:
            stream = BufferedInputStream(stream, buffer_size)
        if compression == 'detect':
            compression = _detect_compression(path)
        if compression is not None:
            stream = CompressedInputStream(stream, compression)
        return stream

    def _wrap_output_stream(self, stream, path, compression, buffer_size):
        if buffer_size is not None and buffer_size != 0:
            stream = BufferedOutputStream(stream, buffer_size)
        if compression == 'detect':
            compression = _detect_compression(path)
        if compression is not None:
            stream = CompressedOutputStream(stream, compression)
        return stream

    def open_input_file(self, path):
        """
        Open an input file for random access reading.

        Parameters
        ----------
        path : str
            The source to open for reading.

        Returns
        -------
        stream : NativeFile

        Examples
        --------
        Print the data from the file with `open_input_file()`:

        >>> with local.open_input_file(path) as f:
        ...     print(f.readall())
        b'data'
        """
        cdef:
            c_string pathstr = _path_as_bytes(path)
            NativeFile stream = NativeFile()
            shared_ptr[CRandomAccessFile] in_handle

        with nogil:
            in_handle = GetResultValue(self.fs.OpenInputFile(pathstr))

        stream.set_random_access_file(in_handle)
        stream.is_readable = True
        return stream

    def open_input_stream(self, path, compression='detect', buffer_size=None):
        """
        Open an input stream for sequential reading.

        Parameters
        ----------
        path : str
            The source to open for reading.
        compression : str optional, default 'detect'
            The compression algorithm to use for on-the-fly decompression.
            If "detect" and source is a file path, then compression will be
            chosen based on the file extension.
            If None, no compression will be applied. Otherwise, a well-known
            algorithm name must be supplied (e.g. "gzip").
        buffer_size : int optional, default None
            If None or 0, no buffering will happen. Otherwise the size of the
            temporary read buffer.

        Returns
        -------
        stream : NativeFile

        Examples
        --------
        Print the data from the file with `open_input_stream()`:

        >>> with local.open_input_stream(path) as f:
        ...     print(f.readall())
        b'data'
        """
        cdef:
            c_string pathstr = _path_as_bytes(path)
            NativeFile stream = NativeFile()
            shared_ptr[CInputStream] in_handle

        with nogil:
            in_handle = GetResultValue(self.fs.OpenInputStream(pathstr))

        stream.set_input_stream(in_handle)
        stream.is_readable = True

        return self._wrap_input_stream(
            stream, path=path, compression=compression, buffer_size=buffer_size
        )

    def open_output_stream(self, path, compression='detect',
                           buffer_size=None, metadata=None):
        """
        Open an output stream for sequential writing.

        If the target already exists, existing data is truncated.

        Parameters
        ----------
        path : str
            The source to open for writing.
        compression : str optional, default 'detect'
            The compression algorithm to use for on-the-fly compression.
            If "detect" and source is a file path, then compression will be
            chosen based on the file extension.
            If None, no compression will be applied. Otherwise, a well-known
            algorithm name must be supplied (e.g. "gzip").
        buffer_size : int optional, default None
            If None or 0, no buffering will happen. Otherwise the size of the
            temporary write buffer.
        metadata : dict optional, default None
            If not None, a mapping of string keys to string values.
            Some filesystems support storing metadata along the file
            (such as "Content-Type").
            Unsupported metadata keys will be ignored.

        Returns
        -------
        stream : NativeFile

        Examples
        --------
        >>> local = fs.LocalFileSystem()
        >>> with local.open_output_stream(path) as stream:
        ...     stream.write(b'data')
        4
        """
        cdef:
            c_string pathstr = _path_as_bytes(path)
            NativeFile stream = NativeFile()
            shared_ptr[COutputStream] out_handle
            shared_ptr[const CKeyValueMetadata] c_metadata

        if metadata is not None:
            c_metadata = pyarrow_unwrap_metadata(KeyValueMetadata(metadata))

        with nogil:
            out_handle = GetResultValue(
                self.fs.OpenOutputStream(pathstr, c_metadata))

        stream.set_output_stream(out_handle)
        stream.is_writable = True

        return self._wrap_output_stream(
            stream, path=path, compression=compression, buffer_size=buffer_size
        )

    def open_append_stream(self, path, compression='detect',
                           buffer_size=None, metadata=None):
        """
        Open an output stream for appending.

        If the target doesn't exist, a new empty file is created.

        .. note::
            Some filesystem implementations do not support efficient
            appending to an existing file, in which case this method will
            raise NotImplementedError.
            Consider writing to multiple files (using e.g. the dataset layer)
            instead.

        Parameters
        ----------
        path : str
            The source to open for writing.
        compression : str optional, default 'detect'
            The compression algorithm to use for on-the-fly compression.
            If "detect" and source is a file path, then compression will be
            chosen based on the file extension.
            If None, no compression will be applied. Otherwise, a well-known
            algorithm name must be supplied (e.g. "gzip").
        buffer_size : int optional, default None
            If None or 0, no buffering will happen. Otherwise the size of the
            temporary write buffer.
        metadata : dict optional, default None
            If not None, a mapping of string keys to string values.
            Some filesystems support storing metadata along the file
            (such as "Content-Type").
            Unsupported metadata keys will be ignored.

        Returns
        -------
        stream : NativeFile

        Examples
        --------
        Append new data to a FileSystem subclass with nonempty file:

        >>> with local.open_append_stream(path) as f:
        ...     f.write(b'+newly added')
        12

        Print out the content fo the file:

        >>> with local.open_input_file(path) as f:
        ...     print(f.readall())
        b'data+newly added'
        """
        cdef:
            c_string pathstr = _path_as_bytes(path)
            NativeFile stream = NativeFile()
            shared_ptr[COutputStream] out_handle
            shared_ptr[const CKeyValueMetadata] c_metadata

        if metadata is not None:
            c_metadata = pyarrow_unwrap_metadata(KeyValueMetadata(metadata))

        with nogil:
            out_handle = GetResultValue(
                self.fs.OpenAppendStream(pathstr, c_metadata))

        stream.set_output_stream(out_handle)
        stream.is_writable = True

        return self._wrap_output_stream(
            stream, path=path, compression=compression, buffer_size=buffer_size
        )

    def normalize_path(self, path):
        """
        Normalize filesystem path.

        Parameters
        ----------
        path : str
            The path to normalize

        Returns
        -------
        normalized_path : str
            The normalized path
        """
        cdef:
            c_string c_path = _path_as_bytes(path)
            c_string c_path_normalized

        c_path_normalized = GetResultValue(self.fs.NormalizePath(c_path))
        return frombytes(c_path_normalized)


cdef class LocalFileSystem(FileSystem):
    """
    A FileSystem implementation accessing files on the local machine.

    Details such as symlinks are abstracted away (symlinks are always followed,
    except when deleting an entry).

    Parameters
    ----------
    use_mmap : bool, default False
        Whether open_input_stream and open_input_file should return
        a mmap'ed file or a regular file.

    Examples
    --------
    Create a FileSystem object with LocalFileSystem constructor:

    >>> from pyarrow import fs
    >>> local = fs.LocalFileSystem()
    >>> local
    <pyarrow._fs.LocalFileSystem object at ...>

    and write data on to the file:

    >>> with local.open_output_stream('/tmp/local_fs.dat') as stream:
    ...     stream.write(b'data')
    4
    >>> with local.open_input_stream('/tmp/local_fs.dat') as stream:
    ...     print(stream.readall())
    b'data'

    Create a FileSystem object inferred from a URI of the saved file:

    >>> local_new, path = fs.LocalFileSystem().from_uri('/tmp/local_fs.dat')
    >>> local_new
    <pyarrow._fs.LocalFileSystem object at ...
    >>> path
    '/tmp/local_fs.dat'

    Check if FileSystems `local` and `local_new` are equal:

    >>> local.equals(local_new)
    True

    Compare two different FileSystems:

    >>> local2 = fs.LocalFileSystem(use_mmap=True)
    >>> local.equals(local2)
    False

    Copy a file and print out the data:

    >>> local.copy_file('/tmp/local_fs.dat', '/tmp/local_fs-copy.dat')
    >>> with local.open_input_stream('/tmp/local_fs-copy.dat') as stream:
    ...     print(stream.readall())
    ...
    b'data'

    Open an output stream for appending, add text and print the new data:

    >>> with local.open_append_stream('/tmp/local_fs-copy.dat') as f:
    ...     f.write(b'+newly added')
    12

    >>> with local.open_input_stream('/tmp/local_fs-copy.dat') as f:
    ...     print(f.readall())
    b'data+newly added'

    Create a directory, copy a file into it and then delete the whole directory:

    >>> local.create_dir('/tmp/new_folder')
    >>> local.copy_file('/tmp/local_fs.dat', '/tmp/new_folder/local_fs.dat')
    >>> local.get_file_info('/tmp/new_folder')
    <FileInfo for '/tmp/new_folder': type=FileType.Directory>
    >>> local.delete_dir('/tmp/new_folder')
    >>> local.get_file_info('/tmp/new_folder')
    <FileInfo for '/tmp/new_folder': type=FileType.NotFound>

    Create a directory, copy a file into it and then delete
    the content of the directory:

    >>> local.create_dir('/tmp/new_folder')
    >>> local.copy_file('/tmp/local_fs.dat', '/tmp/new_folder/local_fs.dat')
    >>> local.get_file_info('/tmp/new_folder/local_fs.dat')
    <FileInfo for '/tmp/new_folder/local_fs.dat': type=FileType.File, size=4>
    >>> local.delete_dir_contents('/tmp/new_folder')
    >>> local.get_file_info('/tmp/new_folder')
    <FileInfo for '/tmp/new_folder': type=FileType.Directory>
    >>> local.get_file_info('/tmp/new_folder/local_fs.dat')
    <FileInfo for '/tmp/new_folder/local_fs.dat': type=FileType.NotFound>

    Create a directory, copy a file into it and then delete
    the file from the directory:

    >>> local.create_dir('/tmp/new_folder')
    >>> local.copy_file('/tmp/local_fs.dat', '/tmp/new_folder/local_fs.dat')
    >>> local.delete_file('/tmp/new_folder/local_fs.dat')
    >>> local.get_file_info('/tmp/new_folder/local_fs.dat')
    <FileInfo for '/tmp/new_folder/local_fs.dat': type=FileType.NotFound>
    >>> local.get_file_info('/tmp/new_folder')
    <FileInfo for '/tmp/new_folder': type=FileType.Directory>

    Move the file:

    >>> local.move('/tmp/local_fs-copy.dat', '/tmp/new_folder/local_fs-copy.dat')
    >>> local.get_file_info('/tmp/new_folder/local_fs-copy.dat')
    <FileInfo for '/tmp/new_folder/local_fs-copy.dat': type=FileType.File, size=16>
    >>> local.get_file_info('/tmp/local_fs-copy.dat')
    <FileInfo for '/tmp/local_fs-copy.dat': type=FileType.NotFound>

    To finish delete the file left:
    >>> local.delete_file('/tmp/local_fs.dat')
    """

    def __init__(self, *, use_mmap=False):
        cdef:
            CLocalFileSystemOptions opts
            shared_ptr[CLocalFileSystem] fs

        opts = CLocalFileSystemOptions.Defaults()
        opts.use_mmap = use_mmap

        fs = make_shared[CLocalFileSystem](opts)
        self.init(<shared_ptr[CFileSystem]> fs)

    cdef init(self, const shared_ptr[CFileSystem]& c_fs):
        FileSystem.init(self, c_fs)
        self.localfs = <CLocalFileSystem*> c_fs.get()

    @classmethod
    def _reconstruct(cls, kwargs):
        # __reduce__ doesn't allow passing named arguments directly to the
        # reconstructor, hence this wrapper.
        return cls(**kwargs)

    def __reduce__(self):
        cdef CLocalFileSystemOptions opts = self.localfs.options()
        return LocalFileSystem._reconstruct, (dict(
            use_mmap=opts.use_mmap),)


cdef class SubTreeFileSystem(FileSystem):
    """
    Delegates to another implementation after prepending a fixed base path.

    This is useful to expose a logical view of a subtree of a filesystem,
    for example a directory in a LocalFileSystem.

    Note, that this makes no security guarantee. For example, symlinks may
    allow to "escape" the subtree and access other parts of the underlying
    filesystem.

    Parameters
    ----------
    base_path : str
        The root of the subtree.
    base_fs : FileSystem
        FileSystem object the operations delegated to.

    Examples
    --------
    Create a LocalFileSystem instance:

    >>> from pyarrow import fs
    >>> local = fs.LocalFileSystem()
    >>> with local.open_output_stream('/tmp/local_fs.dat') as stream:
    ...     stream.write(b'data')
    4

    Create a directory and a SubTreeFileSystem instance:

    >>> local.create_dir('/tmp/sub_tree')
    >>> subtree = fs.SubTreeFileSystem('/tmp/sub_tree', local)

    Write data into the existing file:

    >>> with subtree.open_append_stream('sub_tree_fs.dat') as f:
    ...     f.write(b'+newly added')
    12

    Print out the attributes:

    >>> subtree.base_fs
    <pyarrow._fs.LocalFileSystem object at ...>
    >>> subtree.base_path
    '/tmp/sub_tree/'

    Get info for the given directory or given file:

    >>> subtree.get_file_info('')
    <FileInfo for '': type=FileType.Directory>
    >>> subtree.get_file_info('sub_tree_fs.dat')
    <FileInfo for 'sub_tree_fs.dat': type=FileType.File, size=12>

    Delete the file and directory:

    >>> subtree.delete_file('sub_tree_fs.dat')
    >>> local.delete_dir('/tmp/sub_tree')
    >>> local.delete_file('/tmp/local_fs.dat')

    For usage of the methods see examples for :func:`~pyarrow.fs.LocalFileSystem`.
    """

    def __init__(self, base_path, FileSystem base_fs):
        cdef:
            c_string pathstr
            shared_ptr[CSubTreeFileSystem] wrapped

        pathstr = _path_as_bytes(base_path)
        wrapped = make_shared[CSubTreeFileSystem](pathstr, base_fs.wrapped)

        self.init(<shared_ptr[CFileSystem]> wrapped)

    cdef init(self, const shared_ptr[CFileSystem]& wrapped):
        FileSystem.init(self, wrapped)
        self.subtreefs = <CSubTreeFileSystem*> wrapped.get()

    def __repr__(self):
        return ("SubTreeFileSystem(base_path={}, base_fs={}"
                .format(self.base_path, self.base_fs))

    def __reduce__(self):
        return SubTreeFileSystem, (
            frombytes(self.subtreefs.base_path()),
            FileSystem.wrap(self.subtreefs.base_fs())
        )

    @property
    def base_path(self):
        return frombytes(self.subtreefs.base_path())

    @property
    def base_fs(self):
        return FileSystem.wrap(self.subtreefs.base_fs())


cdef class _MockFileSystem(FileSystem):

    def __init__(self, datetime current_time=None):
        cdef shared_ptr[CMockFileSystem] wrapped

        current_time = current_time or datetime.now()
        wrapped = make_shared[CMockFileSystem](
            PyDateTime_to_TimePoint(<PyDateTime_DateTime*> current_time)
        )

        self.init(<shared_ptr[CFileSystem]> wrapped)

    cdef init(self, const shared_ptr[CFileSystem]& wrapped):
        FileSystem.init(self, wrapped)
        self.mockfs = <CMockFileSystem*> wrapped.get()


cdef class PyFileSystem(FileSystem):
    """
    A FileSystem with behavior implemented in Python.

    Parameters
    ----------
    handler : FileSystemHandler
        The handler object implementing custom filesystem behavior.

    Examples
    --------
    Create an fsspec-based filesystem object for GitHub:

    >>> from fsspec.implementations import github
    >>> gfs = github.GithubFileSystem('apache', 'arrow') # doctest: +SKIP

    Get a PyArrow FileSystem object:

    >>> from pyarrow.fs import PyFileSystem, FSSpecHandler
    >>> pa_fs = PyFileSystem(FSSpecHandler(gfs)) # doctest: +SKIP

    Use :func:`~pyarrow.fs.FileSystem` functionality ``get_file_info()``:

    >>> pa_fs.get_file_info('README.md') # doctest: +SKIP
    <FileInfo for 'README.md': type=FileType.File, size=...>
    """

    def __init__(self, handler):
        cdef:
            CPyFileSystemVtable vtable
            shared_ptr[CPyFileSystem] wrapped

        if not isinstance(handler, FileSystemHandler):
            raise TypeError("Expected a FileSystemHandler instance, got {0}"
                            .format(type(handler)))

        vtable.get_type_name = _cb_get_type_name
        vtable.equals = _cb_equals
        vtable.get_file_info = _cb_get_file_info
        vtable.get_file_info_vector = _cb_get_file_info_vector
        vtable.get_file_info_selector = _cb_get_file_info_selector
        vtable.create_dir = _cb_create_dir
        vtable.delete_dir = _cb_delete_dir
        vtable.delete_dir_contents = _cb_delete_dir_contents
        vtable.delete_root_dir_contents = _cb_delete_root_dir_contents
        vtable.delete_file = _cb_delete_file
        vtable.move = _cb_move
        vtable.copy_file = _cb_copy_file
        vtable.open_input_stream = _cb_open_input_stream
        vtable.open_input_file = _cb_open_input_file
        vtable.open_output_stream = _cb_open_output_stream
        vtable.open_append_stream = _cb_open_append_stream
        vtable.normalize_path = _cb_normalize_path

        wrapped = CPyFileSystem.Make(handler, move(vtable))
        self.init(<shared_ptr[CFileSystem]> wrapped)

    cdef init(self, const shared_ptr[CFileSystem]& wrapped):
        FileSystem.init(self, wrapped)
        self.pyfs = <CPyFileSystem*> wrapped.get()

    @property
    def handler(self):
        """
        The filesystem's underlying handler.

        Returns
        -------
        handler : FileSystemHandler
        """
        return <object> self.pyfs.handler()

    def __reduce__(self):
        return PyFileSystem, (self.handler,)


class FileSystemHandler(ABC):
    """
    An abstract class exposing methods to implement PyFileSystem's behavior.
    """

    @abstractmethod
    def get_type_name(self):
        """
        Implement PyFileSystem.type_name.
        """

    @abstractmethod
    def get_file_info(self, paths):
        """
        Implement PyFileSystem.get_file_info(paths).

        Parameters
        ----------
        paths : list of str
            paths for which we want to retrieve the info.
        """

    @abstractmethod
    def get_file_info_selector(self, selector):
        """
        Implement PyFileSystem.get_file_info(selector).

        Parameters
        ----------
        selector : FileSelector
            selector for which we want to retrieve the info.
        """

    @abstractmethod
    def create_dir(self, path, recursive):
        """
        Implement PyFileSystem.create_dir(...).

        Parameters
        ----------
        path : str
            path of the directory.
        recursive : bool
            if the parent directories should be created too.
        """

    @abstractmethod
    def delete_dir(self, path):
        """
        Implement PyFileSystem.delete_dir(...).

        Parameters
        ----------
        path : str
            path of the directory.
        """

    @abstractmethod
    def delete_dir_contents(self, path, missing_dir_ok=False):
        """
        Implement PyFileSystem.delete_dir_contents(...).

        Parameters
        ----------
        path : str
            path of the directory.
        missing_dir_ok : bool
            if False an error should be raised if path does not exist
        """

    @abstractmethod
    def delete_root_dir_contents(self):
        """
        Implement PyFileSystem.delete_dir_contents("/", accept_root_dir=True).
        """

    @abstractmethod
    def delete_file(self, path):
        """
        Implement PyFileSystem.delete_file(...).

        Parameters
        ----------
        path : str
            path of the file.
        """

    @abstractmethod
    def move(self, src, dest):
        """
        Implement PyFileSystem.move(...).

        Parameters
        ----------
        src : str
            path of what should be moved.
        dest : str
            path of where it should be moved to.
        """

    @abstractmethod
    def copy_file(self, src, dest):
        """
        Implement PyFileSystem.copy_file(...).

        Parameters
        ----------
        src : str
            path of what should be copied.
        dest : str
            path of where it should be copied to.
        """

    @abstractmethod
    def open_input_stream(self, path):
        """
        Implement PyFileSystem.open_input_stream(...).

        Parameters
        ----------
        path : str
            path of what should be opened.
        """

    @abstractmethod
    def open_input_file(self, path):
        """
        Implement PyFileSystem.open_input_file(...).

        Parameters
        ----------
        path : str
            path of what should be opened.
        """

    @abstractmethod
    def open_output_stream(self, path, metadata):
        """
        Implement PyFileSystem.open_output_stream(...).

        Parameters
        ----------
        path : str
            path of what should be opened.
        metadata :  mapping
            Mapping of string keys to string values.
            Some filesystems support storing metadata along the file
            (such as "Content-Type").
        """

    @abstractmethod
    def open_append_stream(self, path, metadata):
        """
        Implement PyFileSystem.open_append_stream(...).

        Parameters
        ----------
        path : str
            path of what should be opened.
        metadata :  mapping
            Mapping of string keys to string values.
            Some filesystems support storing metadata along the file
            (such as "Content-Type").
        """

    @abstractmethod
    def normalize_path(self, path):
        """
        Implement PyFileSystem.normalize_path(...).

        Parameters
        ----------
        path : str
            path of what should be normalized.
        """

# Callback definitions for CPyFileSystemVtable


cdef void _cb_get_type_name(handler, c_string* out) except *:
    out[0] = tobytes("py::" + handler.get_type_name())

cdef c_bool _cb_equals(handler, const CFileSystem& c_other) except False:
    if c_other.type_name().startswith(b"py::"):
        return <object> (<const CPyFileSystem&> c_other).handler() == handler

    return False

cdef void _cb_get_file_info(handler, const c_string& path,
                            CFileInfo* out) except *:
    infos = handler.get_file_info([frombytes(path)])
    if not isinstance(infos, list) or len(infos) != 1:
        raise TypeError("get_file_info should have returned a 1-element list")
    out[0] = FileInfo.unwrap_safe(infos[0])

cdef void _cb_get_file_info_vector(handler, const vector[c_string]& paths,
                                   vector[CFileInfo]* out) except *:
    py_paths = [frombytes(paths[i]) for i in range(len(paths))]
    infos = handler.get_file_info(py_paths)
    if not isinstance(infos, list):
        raise TypeError("get_file_info should have returned a list")
    out[0].clear()
    out[0].reserve(len(infos))
    for info in infos:
        out[0].push_back(FileInfo.unwrap_safe(info))

cdef void _cb_get_file_info_selector(handler, const CFileSelector& selector,
                                     vector[CFileInfo]* out) except *:
    infos = handler.get_file_info_selector(FileSelector.wrap(selector))
    if not isinstance(infos, list):
        raise TypeError("get_file_info_selector should have returned a list")
    out[0].clear()
    out[0].reserve(len(infos))
    for info in infos:
        out[0].push_back(FileInfo.unwrap_safe(info))

cdef void _cb_create_dir(handler, const c_string& path,
                         c_bool recursive) except *:
    handler.create_dir(frombytes(path), recursive)

cdef void _cb_delete_dir(handler, const c_string& path) except *:
    handler.delete_dir(frombytes(path))

cdef void _cb_delete_dir_contents(handler, const c_string& path,
                                  c_bool missing_dir_ok) except *:
    handler.delete_dir_contents(frombytes(path), missing_dir_ok)

cdef void _cb_delete_root_dir_contents(handler) except *:
    handler.delete_root_dir_contents()

cdef void _cb_delete_file(handler, const c_string& path) except *:
    handler.delete_file(frombytes(path))

cdef void _cb_move(handler, const c_string& src,
                   const c_string& dest) except *:
    handler.move(frombytes(src), frombytes(dest))

cdef void _cb_copy_file(handler, const c_string& src,
                        const c_string& dest) except *:
    handler.copy_file(frombytes(src), frombytes(dest))

cdef void _cb_open_input_stream(handler, const c_string& path,
                                shared_ptr[CInputStream]* out) except *:
    stream = handler.open_input_stream(frombytes(path))
    if not isinstance(stream, NativeFile):
        raise TypeError("open_input_stream should have returned "
                        "a PyArrow file")
    out[0] = (<NativeFile> stream).get_input_stream()

cdef void _cb_open_input_file(handler, const c_string& path,
                              shared_ptr[CRandomAccessFile]* out) except *:
    stream = handler.open_input_file(frombytes(path))
    if not isinstance(stream, NativeFile):
        raise TypeError("open_input_file should have returned "
                        "a PyArrow file")
    out[0] = (<NativeFile> stream).get_random_access_file()

cdef void _cb_open_output_stream(
        handler, const c_string& path,
        const shared_ptr[const CKeyValueMetadata]& metadata,
        shared_ptr[COutputStream]* out) except *:
    stream = handler.open_output_stream(
        frombytes(path), pyarrow_wrap_metadata(metadata))
    if not isinstance(stream, NativeFile):
        raise TypeError("open_output_stream should have returned "
                        "a PyArrow file")
    out[0] = (<NativeFile> stream).get_output_stream()

cdef void _cb_open_append_stream(
        handler, const c_string& path,
        const shared_ptr[const CKeyValueMetadata]& metadata,
        shared_ptr[COutputStream]* out) except *:
    stream = handler.open_append_stream(
        frombytes(path), pyarrow_wrap_metadata(metadata))
    if not isinstance(stream, NativeFile):
        raise TypeError("open_append_stream should have returned "
                        "a PyArrow file")
    out[0] = (<NativeFile> stream).get_output_stream()

cdef void _cb_normalize_path(handler, const c_string& path,
                             c_string* out) except *:
    out[0] = tobytes(handler.normalize_path(frombytes(path)))


def _copy_files(FileSystem source_fs, str source_path,
                FileSystem destination_fs, str destination_path,
                int64_t chunk_size, c_bool use_threads):
    # low-level helper exposed through pyarrow/fs.py::copy_files
    cdef:
        CFileLocator c_source
        vector[CFileLocator] c_sources
        CFileLocator c_destination
        vector[CFileLocator] c_destinations
        FileSystem fs
        CStatus c_status
        shared_ptr[CFileSystem] c_fs

    c_source.filesystem = source_fs.unwrap()
    c_source.path = tobytes(source_path)
    c_sources.push_back(c_source)

    c_destination.filesystem = destination_fs.unwrap()
    c_destination.path = tobytes(destination_path)
    c_destinations.push_back(c_destination)

    with nogil:
        check_status(CCopyFiles(
            c_sources, c_destinations,
            c_default_io_context(), chunk_size, use_threads,
        ))


def _copy_files_selector(FileSystem source_fs, FileSelector source_sel,
                         FileSystem destination_fs, str destination_base_dir,
                         int64_t chunk_size, c_bool use_threads):
    # low-level helper exposed through pyarrow/fs.py::copy_files
    cdef c_string c_destination_base_dir = tobytes(destination_base_dir)

    with nogil:
        check_status(CCopyFilesWithSelector(
            source_fs.unwrap(), source_sel.unwrap(),
            destination_fs.unwrap(), c_destination_base_dir,
            c_default_io_context(), chunk_size, use_threads,
        ))

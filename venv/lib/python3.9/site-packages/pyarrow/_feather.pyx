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

# ---------------------------------------------------------------------
# Implement Feather file format

# cython: profile=False
# distutils: language = c++
# cython: language_level=3

from cython.operator cimport dereference as deref
from pyarrow.includes.common cimport *
from pyarrow.includes.libarrow cimport *
from pyarrow.includes.libarrow_feather cimport *
from pyarrow.lib cimport (check_status, Table, _Weakrefable,
                          get_writer, get_reader, pyarrow_wrap_table)
from pyarrow.lib import tobytes


class FeatherError(Exception):
    pass


def write_feather(Table table, object dest, compression=None,
                  compression_level=None, chunksize=None, version=2):
    cdef shared_ptr[COutputStream] sink
    get_writer(dest, &sink)

    cdef CFeatherProperties properties
    if version == 2:
        properties.version = kFeatherV2Version
    else:
        properties.version = kFeatherV1Version

    if compression == 'zstd':
        properties.compression = CCompressionType_ZSTD
    elif compression == 'lz4':
        properties.compression = CCompressionType_LZ4_FRAME
    else:
        properties.compression = CCompressionType_UNCOMPRESSED

    if chunksize is not None:
        properties.chunksize = chunksize

    if compression_level is not None:
        properties.compression_level = compression_level

    with nogil:
        check_status(WriteFeather(deref(table.table), sink.get(),
                                  properties))


cdef class FeatherReader(_Weakrefable):
    cdef:
        shared_ptr[CFeatherReader] reader

    def __cinit__(self, source, c_bool use_memory_map, c_bool use_threads):
        cdef:
            shared_ptr[CRandomAccessFile] reader
            CIpcReadOptions options = CIpcReadOptions.Defaults()
        options.use_threads = use_threads

        get_reader(source, use_memory_map, &reader)
        with nogil:
            self.reader = GetResultValue(CFeatherReader.Open(reader, options))

    @property
    def version(self):
        return self.reader.get().version()

    def read(self):
        cdef shared_ptr[CTable] sp_table
        with nogil:
            check_status(self.reader.get()
                         .Read(&sp_table))

        return pyarrow_wrap_table(sp_table)

    def read_indices(self, indices):
        cdef:
            shared_ptr[CTable] sp_table
            vector[int] c_indices

        for index in indices:
            c_indices.push_back(index)
        with nogil:
            check_status(self.reader.get()
                         .Read(c_indices, &sp_table))

        return pyarrow_wrap_table(sp_table)

    def read_names(self, names):
        cdef:
            shared_ptr[CTable] sp_table
            vector[c_string] c_names

        for name in names:
            c_names.push_back(tobytes(name))
        with nogil:
            check_status(self.reader.get()
                         .Read(c_names, &sp_table))

        return pyarrow_wrap_table(sp_table)

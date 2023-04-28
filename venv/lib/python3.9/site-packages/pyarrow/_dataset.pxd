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

"""Dataset is currently unstable. APIs subject to change without notice."""

from pyarrow.includes.common cimport *
from pyarrow.includes.libarrow_dataset cimport *
from pyarrow.lib cimport *
from pyarrow._fs cimport FileSystem


cdef CFileSource _make_file_source(object file, FileSystem filesystem=*)


cdef class DatasetFactory(_Weakrefable):

    cdef:
        shared_ptr[CDatasetFactory] wrapped
        CDatasetFactory* factory

    cdef init(self, const shared_ptr[CDatasetFactory]& sp)

    @staticmethod
    cdef wrap(const shared_ptr[CDatasetFactory]& sp)

    cdef inline shared_ptr[CDatasetFactory] unwrap(self) nogil


cdef class Dataset(_Weakrefable):

    cdef:
        shared_ptr[CDataset] wrapped
        CDataset* dataset
        public dict _scan_options

    cdef void init(self, const shared_ptr[CDataset]& sp)

    @staticmethod
    cdef wrap(const shared_ptr[CDataset]& sp)

    cdef shared_ptr[CDataset] unwrap(self) nogil


cdef class Scanner(_Weakrefable):
    cdef:
        shared_ptr[CScanner] wrapped
        CScanner* scanner

    cdef void init(self, const shared_ptr[CScanner]& sp)

    @staticmethod
    cdef wrap(const shared_ptr[CScanner]& sp)

    cdef shared_ptr[CScanner] unwrap(self)

    @staticmethod
    cdef shared_ptr[CScanOptions] _make_scan_options(Dataset dataset, dict py_scanoptions) except *


cdef class FragmentScanOptions(_Weakrefable):

    cdef:
        shared_ptr[CFragmentScanOptions] wrapped

    cdef void init(self, const shared_ptr[CFragmentScanOptions]& sp)

    @staticmethod
    cdef wrap(const shared_ptr[CFragmentScanOptions]& sp)


cdef class FileFormat(_Weakrefable):

    cdef:
        shared_ptr[CFileFormat] wrapped
        CFileFormat* format

    cdef void init(self, const shared_ptr[CFileFormat]& sp)

    @staticmethod
    cdef wrap(const shared_ptr[CFileFormat]& sp)

    cdef inline shared_ptr[CFileFormat] unwrap(self)

    cdef _set_default_fragment_scan_options(self, FragmentScanOptions options)

    # Return a WrittenFile after a file was written.
    # May be overridden by subclasses, e.g. to add metadata.
    cdef WrittenFile _finish_write(self, path, base_dir,
                                   CFileWriter* file_writer)


cdef class FileWriteOptions(_Weakrefable):

    cdef:
        shared_ptr[CFileWriteOptions] wrapped
        CFileWriteOptions* c_options

    cdef void init(self, const shared_ptr[CFileWriteOptions]& sp)

    @staticmethod
    cdef wrap(const shared_ptr[CFileWriteOptions]& sp)

    cdef inline shared_ptr[CFileWriteOptions] unwrap(self)


cdef class Fragment(_Weakrefable):

    cdef:
        shared_ptr[CFragment] wrapped
        CFragment* fragment

    cdef void init(self, const shared_ptr[CFragment]& sp)

    @staticmethod
    cdef wrap(const shared_ptr[CFragment]& sp)

    cdef inline shared_ptr[CFragment] unwrap(self)


cdef class FileFragment(Fragment):

    cdef:
        CFileFragment* file_fragment

    cdef void init(self, const shared_ptr[CFragment]& sp)


cdef class Partitioning(_Weakrefable):

    cdef:
        shared_ptr[CPartitioning] wrapped
        CPartitioning* partitioning

    cdef init(self, const shared_ptr[CPartitioning]& sp)

    @staticmethod
    cdef wrap(const shared_ptr[CPartitioning]& sp)

    cdef inline shared_ptr[CPartitioning] unwrap(self)


cdef class PartitioningFactory(_Weakrefable):

    cdef:
        shared_ptr[CPartitioningFactory] wrapped
        CPartitioningFactory* factory

    cdef init(self, const shared_ptr[CPartitioningFactory]& sp)

    @staticmethod
    cdef wrap(const shared_ptr[CPartitioningFactory]& sp)

    cdef inline shared_ptr[CPartitioningFactory] unwrap(self)


cdef class WrittenFile(_Weakrefable):

    # The full path to the created file
    cdef public str path
    # Optional Parquet metadata
    # This metadata will have the file path attribute set to the path of
    # the written file.
    cdef public object metadata
    # The size of the file in bytes
    cdef public int64_t size

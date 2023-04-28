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

from pyarrow.includes.libarrow cimport GetBuildInfo

from collections import namedtuple


VersionInfo = namedtuple('VersionInfo', ('major', 'minor', 'patch'))

BuildInfo = namedtuple(
    'BuildInfo',
    ('version', 'version_info', 'so_version', 'full_so_version',
     'compiler_id', 'compiler_version', 'compiler_flags',
     'git_id', 'git_description', 'package_kind', 'build_type'))

RuntimeInfo = namedtuple('RuntimeInfo',
                         ('simd_level', 'detected_simd_level'))

cdef _build_info():
    cdef:
        const CBuildInfo* c_info

    c_info = &GetBuildInfo()

    return BuildInfo(version=frombytes(c_info.version_string),
                     version_info=VersionInfo(c_info.version_major,
                                              c_info.version_minor,
                                              c_info.version_patch),
                     so_version=frombytes(c_info.so_version),
                     full_so_version=frombytes(c_info.full_so_version),
                     compiler_id=frombytes(c_info.compiler_id),
                     compiler_version=frombytes(c_info.compiler_version),
                     compiler_flags=frombytes(c_info.compiler_flags),
                     git_id=frombytes(c_info.git_id),
                     git_description=frombytes(c_info.git_description),
                     package_kind=frombytes(c_info.package_kind),
                     build_type=frombytes(c_info.build_type).lower(),
                     )


cpp_build_info = _build_info()
cpp_version = cpp_build_info.version
cpp_version_info = cpp_build_info.version_info


def runtime_info():
    """
    Get runtime information.

    Returns
    -------
    info : pyarrow.RuntimeInfo
    """
    cdef:
        CRuntimeInfo c_info

    c_info = GetRuntimeInfo()

    return RuntimeInfo(
        simd_level=frombytes(c_info.simd_level),
        detected_simd_level=frombytes(c_info.detected_simd_level))

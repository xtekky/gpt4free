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

# cython: profile=False, binding=True
# distutils: language = c++

from pyarrow.includes.common cimport *
from pyarrow.includes.libarrow cimport *
from pyarrow.lib cimport check_status

from pyarrow.lib import frombytes


cdef class CppTestCase:
    """
    A simple wrapper for a C++ test case.
    """
    cdef:
        CTestCase c_case

    @staticmethod
    cdef wrap(CTestCase c_case):
        cdef:
            CppTestCase obj
        obj = CppTestCase.__new__(CppTestCase)
        obj.c_case = c_case
        return obj

    @property
    def name(self):
        return frombytes(self.c_case.name)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.name!r}>"

    def __call__(self):
        check_status(self.c_case.func())


def get_cpp_tests():
    """
    Get a list of C++ test cases.
    """
    cases = []
    c_cases = GetCppTestCases()
    for c_case in c_cases:
        cases.append(CppTestCase.wrap(c_case))
    return cases

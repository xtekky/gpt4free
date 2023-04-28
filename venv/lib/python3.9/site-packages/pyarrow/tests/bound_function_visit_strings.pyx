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

# distutils: language=c++
# cython: language_level = 3

import pyarrow as pa
from pyarrow.lib cimport *
from pyarrow.lib import frombytes, tobytes

# basic test to roundtrip through a BoundFunction

ctypedef CStatus visit_string_cb(const c_string&)

cdef extern from * namespace "arrow::py" nogil:
    """
    #include <functional>
    #include <string>
    #include <vector>

    #include "arrow/status.h"

    namespace arrow {
    namespace py {

    Status VisitStrings(const std::vector<std::string>& strs,
                        std::function<Status(const std::string&)> cb) {
      for (const std::string& str : strs) {
        RETURN_NOT_OK(cb(str));
      }
      return Status::OK();
    }

    }  // namespace py
    }  // namespace arrow
    """
    cdef CStatus CVisitStrings" arrow::py::VisitStrings"(
        vector[c_string], function[visit_string_cb])


cdef void _visit_strings_impl(py_cb, const c_string& s) except *:
    py_cb(frombytes(s))


def _visit_strings(strings, cb):
    cdef:
        function[visit_string_cb] c_cb
        vector[c_string] c_strings

    c_cb = BindFunction[visit_string_cb](&_visit_strings_impl, cb)
    for s in strings:
        c_strings.push_back(tobytes(s))

    check_status(CVisitStrings(c_strings, c_cb))

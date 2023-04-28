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

# distutils: language = c++

from libcpp.vector cimport vector as std_vector

from pyarrow.includes.common cimport *
from pyarrow.includes.libarrow cimport *

ctypedef CResult[CDeclaration] CNamedTableProvider(const std_vector[c_string]&)

cdef extern from "arrow/engine/substrait/options.h" namespace "arrow::engine" nogil:
    cdef enum ConversionStrictness \
            "arrow::engine::ConversionStrictness":
        EXACT_ROUNDTRIP \
            "arrow::engine::ConversionStrictness::EXACT_ROUNDTRIP"
        PRESERVE_STRUCTURE \
            "arrow::engine::ConversionStrictness::PRESERVE_STRUCTURE"
        BEST_EFFORT \
            "arrow::engine::ConversionStrictness::BEST_EFFORT"

    cdef cppclass CConversionOptions \
            "arrow::engine::ConversionOptions":
        ConversionStrictness conversion_strictness
        function[CNamedTableProvider] named_table_provider

cdef extern from "arrow/engine/substrait/extension_set.h" \
        namespace "arrow::engine" nogil:

    cdef cppclass ExtensionIdRegistry:
        std_vector[c_string] GetSupportedSubstraitFunctions()

    ExtensionIdRegistry* default_extension_id_registry()


cdef extern from "arrow/engine/substrait/util.h" namespace "arrow::engine" nogil:
    CResult[shared_ptr[CRecordBatchReader]] ExecuteSerializedPlan(
        const CBuffer& substrait_buffer, const ExtensionIdRegistry* registry,
        CFunctionRegistry* func_registry, const CConversionOptions& conversion_options,
        c_bool use_threads)

    CResult[shared_ptr[CBuffer]] SerializeJsonPlan(const c_string& substrait_json)

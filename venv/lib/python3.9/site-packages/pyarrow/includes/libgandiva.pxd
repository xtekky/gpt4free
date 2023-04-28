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

from libcpp.string cimport string as c_string
from libcpp.unordered_set cimport unordered_set as c_unordered_set
from libc.stdint cimport int64_t, int32_t, uint8_t, uintptr_t

from pyarrow.includes.common cimport *
from pyarrow.includes.libarrow cimport *

cdef extern from "gandiva/node.h" namespace "gandiva" nogil:

    cdef cppclass CNode" gandiva::Node":
        c_string ToString()
        shared_ptr[CDataType] return_type()

    cdef cppclass CGandivaExpression" gandiva::Expression":
        c_string ToString()
        shared_ptr[CNode] root()
        shared_ptr[CField] result()

    ctypedef vector[shared_ptr[CNode]] CNodeVector" gandiva::NodeVector"

    ctypedef vector[shared_ptr[CGandivaExpression]] \
        CExpressionVector" gandiva::ExpressionVector"

cdef extern from "gandiva/selection_vector.h" namespace "gandiva" nogil:

    cdef cppclass CSelectionVector" gandiva::SelectionVector":

        shared_ptr[CArray] ToArray()

    enum CSelectionVector_Mode" gandiva::SelectionVector::Mode":
        CSelectionVector_Mode_NONE" gandiva::SelectionVector::Mode::MODE_NONE"
        CSelectionVector_Mode_UINT16" \
                gandiva::SelectionVector::Mode::MODE_UINT16"
        CSelectionVector_Mode_UINT32" \
                gandiva::SelectionVector::Mode::MODE_UINT32"
        CSelectionVector_Mode_UINT64" \
                gandiva::SelectionVector::Mode::MODE_UINT64"

    cdef CStatus SelectionVector_MakeInt16\
        "gandiva::SelectionVector::MakeInt16"(
            int64_t max_slots, CMemoryPool* pool,
            shared_ptr[CSelectionVector]* selection_vector)

    cdef CStatus SelectionVector_MakeInt32\
        "gandiva::SelectionVector::MakeInt32"(
            int64_t max_slots, CMemoryPool* pool,
            shared_ptr[CSelectionVector]* selection_vector)

    cdef CStatus SelectionVector_MakeInt64\
        "gandiva::SelectionVector::MakeInt64"(
            int64_t max_slots, CMemoryPool* pool,
            shared_ptr[CSelectionVector]* selection_vector)

cdef inline CSelectionVector_Mode _ensure_selection_mode(str name) except *:
    uppercase = name.upper()
    if uppercase == 'NONE':
        return CSelectionVector_Mode_NONE
    elif uppercase == 'UINT16':
        return CSelectionVector_Mode_UINT16
    elif uppercase == 'UINT32':
        return CSelectionVector_Mode_UINT32
    elif uppercase == 'UINT64':
        return CSelectionVector_Mode_UINT64
    else:
        raise ValueError('Invalid value for Selection Mode: {!r}'.format(name))

cdef inline str _selection_mode_name(CSelectionVector_Mode ctype):
    if ctype == CSelectionVector_Mode_NONE:
        return 'NONE'
    elif ctype == CSelectionVector_Mode_UINT16:
        return 'UINT16'
    elif ctype == CSelectionVector_Mode_UINT32:
        return 'UINT32'
    elif ctype == CSelectionVector_Mode_UINT64:
        return 'UINT64'
    else:
        raise RuntimeError('Unexpected CSelectionVector_Mode value')

cdef extern from "gandiva/condition.h" namespace "gandiva" nogil:

    cdef cppclass CCondition" gandiva::Condition":
        c_string ToString()
        shared_ptr[CNode] root()
        shared_ptr[CField] result()

cdef extern from "gandiva/arrow.h" namespace "gandiva" nogil:

    ctypedef vector[shared_ptr[CArray]] CArrayVector" gandiva::ArrayVector"


cdef extern from "gandiva/tree_expr_builder.h" namespace "gandiva" nogil:

    cdef shared_ptr[CNode] TreeExprBuilder_MakeBoolLiteral \
        "gandiva::TreeExprBuilder::MakeLiteral"(c_bool value)

    cdef shared_ptr[CNode] TreeExprBuilder_MakeUInt8Literal \
        "gandiva::TreeExprBuilder::MakeLiteral"(uint8_t value)

    cdef shared_ptr[CNode] TreeExprBuilder_MakeUInt16Literal \
        "gandiva::TreeExprBuilder::MakeLiteral"(uint16_t value)

    cdef shared_ptr[CNode] TreeExprBuilder_MakeUInt32Literal \
        "gandiva::TreeExprBuilder::MakeLiteral"(uint32_t value)

    cdef shared_ptr[CNode] TreeExprBuilder_MakeUInt64Literal \
        "gandiva::TreeExprBuilder::MakeLiteral"(uint64_t value)

    cdef shared_ptr[CNode] TreeExprBuilder_MakeInt8Literal \
        "gandiva::TreeExprBuilder::MakeLiteral"(int8_t value)

    cdef shared_ptr[CNode] TreeExprBuilder_MakeInt16Literal \
        "gandiva::TreeExprBuilder::MakeLiteral"(int16_t value)

    cdef shared_ptr[CNode] TreeExprBuilder_MakeInt32Literal \
        "gandiva::TreeExprBuilder::MakeLiteral"(int32_t value)

    cdef shared_ptr[CNode] TreeExprBuilder_MakeInt64Literal \
        "gandiva::TreeExprBuilder::MakeLiteral"(int64_t value)

    cdef shared_ptr[CNode] TreeExprBuilder_MakeFloatLiteral \
        "gandiva::TreeExprBuilder::MakeLiteral"(float value)

    cdef shared_ptr[CNode] TreeExprBuilder_MakeDoubleLiteral \
        "gandiva::TreeExprBuilder::MakeLiteral"(double value)

    cdef shared_ptr[CNode] TreeExprBuilder_MakeStringLiteral \
        "gandiva::TreeExprBuilder::MakeStringLiteral"(const c_string& value)

    cdef shared_ptr[CNode] TreeExprBuilder_MakeBinaryLiteral \
        "gandiva::TreeExprBuilder::MakeBinaryLiteral"(const c_string& value)

    cdef shared_ptr[CGandivaExpression] TreeExprBuilder_MakeExpression\
        "gandiva::TreeExprBuilder::MakeExpression"(
            shared_ptr[CNode] root_node, shared_ptr[CField] result_field)

    cdef shared_ptr[CNode] TreeExprBuilder_MakeFunction \
        "gandiva::TreeExprBuilder::MakeFunction"(
            const c_string& name, const CNodeVector& children,
            shared_ptr[CDataType] return_type)

    cdef shared_ptr[CNode] TreeExprBuilder_MakeField \
        "gandiva::TreeExprBuilder::MakeField"(shared_ptr[CField] field)

    cdef shared_ptr[CNode] TreeExprBuilder_MakeIf \
        "gandiva::TreeExprBuilder::MakeIf"(
            shared_ptr[CNode] condition, shared_ptr[CNode] this_node,
            shared_ptr[CNode] else_node, shared_ptr[CDataType] return_type)

    cdef shared_ptr[CNode] TreeExprBuilder_MakeAnd \
        "gandiva::TreeExprBuilder::MakeAnd"(const CNodeVector& children)

    cdef shared_ptr[CNode] TreeExprBuilder_MakeOr \
        "gandiva::TreeExprBuilder::MakeOr"(const CNodeVector& children)

    cdef shared_ptr[CCondition] TreeExprBuilder_MakeCondition \
        "gandiva::TreeExprBuilder::MakeCondition"(
            shared_ptr[CNode] condition)

    cdef shared_ptr[CNode] TreeExprBuilder_MakeInExpressionInt32 \
        "gandiva::TreeExprBuilder::MakeInExpressionInt32"(
            shared_ptr[CNode] node, const c_unordered_set[int32_t]& values)

    cdef shared_ptr[CNode] TreeExprBuilder_MakeInExpressionInt64 \
        "gandiva::TreeExprBuilder::MakeInExpressionInt64"(
            shared_ptr[CNode] node, const c_unordered_set[int64_t]& values)

    cdef shared_ptr[CNode] TreeExprBuilder_MakeInExpressionTime32 \
        "gandiva::TreeExprBuilder::MakeInExpressionTime32"(
            shared_ptr[CNode] node, const c_unordered_set[int32_t]& values)

    cdef shared_ptr[CNode] TreeExprBuilder_MakeInExpressionTime64 \
        "gandiva::TreeExprBuilder::MakeInExpressionTime64"(
            shared_ptr[CNode] node, const c_unordered_set[int64_t]& values)

    cdef shared_ptr[CNode] TreeExprBuilder_MakeInExpressionDate32 \
        "gandiva::TreeExprBuilder::MakeInExpressionDate32"(
            shared_ptr[CNode] node, const c_unordered_set[int32_t]& values)

    cdef shared_ptr[CNode] TreeExprBuilder_MakeInExpressionDate64 \
        "gandiva::TreeExprBuilder::MakeInExpressionDate64"(
            shared_ptr[CNode] node, const c_unordered_set[int64_t]& values)

    cdef shared_ptr[CNode] TreeExprBuilder_MakeInExpressionTimeStamp \
        "gandiva::TreeExprBuilder::MakeInExpressionTimeStamp"(
            shared_ptr[CNode] node, const c_unordered_set[int64_t]& values)

    cdef shared_ptr[CNode] TreeExprBuilder_MakeInExpressionString \
        "gandiva::TreeExprBuilder::MakeInExpressionString"(
            shared_ptr[CNode] node, const c_unordered_set[c_string]& values)

    cdef shared_ptr[CNode] TreeExprBuilder_MakeInExpressionBinary \
        "gandiva::TreeExprBuilder::MakeInExpressionBinary"(
            shared_ptr[CNode] node, const c_unordered_set[c_string]& values)

cdef extern from "gandiva/projector.h" namespace "gandiva" nogil:

    cdef cppclass CProjector" gandiva::Projector":

        CStatus Evaluate(
            const CRecordBatch& batch, CMemoryPool* pool,
            const CArrayVector* output)

        CStatus Evaluate(
            const CRecordBatch& batch,
            const CSelectionVector* selection,
            CMemoryPool* pool,
            const CArrayVector* output)

        c_string DumpIR()

    cdef CStatus Projector_Make \
        "gandiva::Projector::Make"(
            shared_ptr[CSchema] schema, const CExpressionVector& children,
            shared_ptr[CProjector]* projector)

    cdef CStatus Projector_Make \
        "gandiva::Projector::Make"(
            shared_ptr[CSchema] schema, const CExpressionVector& children,
            CSelectionVector_Mode mode,
            shared_ptr[CConfiguration] configuration,
            shared_ptr[CProjector]* projector)

cdef extern from "gandiva/filter.h" namespace "gandiva" nogil:

    cdef cppclass CFilter" gandiva::Filter":

        CStatus Evaluate(
            const CRecordBatch& batch,
            shared_ptr[CSelectionVector] out_selection)

        c_string DumpIR()

    cdef CStatus Filter_Make \
        "gandiva::Filter::Make"(
            shared_ptr[CSchema] schema, shared_ptr[CCondition] condition,
            shared_ptr[CFilter]* filter)

cdef extern from "gandiva/function_signature.h" namespace "gandiva" nogil:

    cdef cppclass CFunctionSignature" gandiva::FunctionSignature":

        CFunctionSignature(const c_string& base_name,
                           vector[shared_ptr[CDataType]] param_types,
                           shared_ptr[CDataType] ret_type)

        shared_ptr[CDataType] ret_type() const

        const c_string& base_name() const

        vector[shared_ptr[CDataType]] param_types() const

        c_string ToString() const

cdef extern from "gandiva/expression_registry.h" namespace "gandiva" nogil:

    cdef vector[shared_ptr[CFunctionSignature]] \
        GetRegisteredFunctionSignatures()

cdef extern from "gandiva/configuration.h" namespace "gandiva" nogil:

    cdef cppclass CConfiguration" gandiva::Configuration":
        pass

    cdef cppclass CConfigurationBuilder \
            " gandiva::ConfigurationBuilder":
        @staticmethod
        shared_ptr[CConfiguration] DefaultConfiguration()

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

# cython: profile=False
# distutils: language = c++
# cython: language_level = 3

from libcpp cimport bool as c_bool, nullptr
from libcpp.memory cimport shared_ptr, unique_ptr, make_shared
from libcpp.string cimport string as c_string
from libcpp.vector cimport vector as c_vector
from libcpp.unordered_set cimport unordered_set as c_unordered_set
from libc.stdint cimport int64_t, int32_t, uint8_t, uintptr_t

from pyarrow.includes.libarrow cimport *
from pyarrow.lib cimport (Array, DataType, Field, MemoryPool, RecordBatch,
                          Schema, check_status, pyarrow_wrap_array,
                          pyarrow_wrap_data_type, ensure_type, _Weakrefable,
                          pyarrow_wrap_field)
from pyarrow.lib import frombytes

from pyarrow.includes.libgandiva cimport (
    CCondition, CGandivaExpression,
    CNode, CProjector, CFilter,
    CSelectionVector,
    CSelectionVector_Mode,
    _ensure_selection_mode,
    CConfiguration,
    CConfigurationBuilder,
    TreeExprBuilder_MakeExpression,
    TreeExprBuilder_MakeFunction,
    TreeExprBuilder_MakeBoolLiteral,
    TreeExprBuilder_MakeUInt8Literal,
    TreeExprBuilder_MakeUInt16Literal,
    TreeExprBuilder_MakeUInt32Literal,
    TreeExprBuilder_MakeUInt64Literal,
    TreeExprBuilder_MakeInt8Literal,
    TreeExprBuilder_MakeInt16Literal,
    TreeExprBuilder_MakeInt32Literal,
    TreeExprBuilder_MakeInt64Literal,
    TreeExprBuilder_MakeFloatLiteral,
    TreeExprBuilder_MakeDoubleLiteral,
    TreeExprBuilder_MakeStringLiteral,
    TreeExprBuilder_MakeBinaryLiteral,
    TreeExprBuilder_MakeField,
    TreeExprBuilder_MakeIf,
    TreeExprBuilder_MakeAnd,
    TreeExprBuilder_MakeOr,
    TreeExprBuilder_MakeCondition,
    TreeExprBuilder_MakeInExpressionInt32,
    TreeExprBuilder_MakeInExpressionInt64,
    TreeExprBuilder_MakeInExpressionTime32,
    TreeExprBuilder_MakeInExpressionTime64,
    TreeExprBuilder_MakeInExpressionDate32,
    TreeExprBuilder_MakeInExpressionDate64,
    TreeExprBuilder_MakeInExpressionTimeStamp,
    TreeExprBuilder_MakeInExpressionString,
    TreeExprBuilder_MakeInExpressionBinary,
    SelectionVector_MakeInt16,
    SelectionVector_MakeInt32,
    SelectionVector_MakeInt64,
    Projector_Make,
    Filter_Make,
    CFunctionSignature,
    GetRegisteredFunctionSignatures)


cdef class Node(_Weakrefable):
    cdef:
        shared_ptr[CNode] node

    def __init__(self):
        raise TypeError("Do not call {}'s constructor directly, use the "
                        "TreeExprBuilder API directly"
                        .format(self.__class__.__name__))

    @staticmethod
    cdef create(shared_ptr[CNode] node):
        cdef Node self = Node.__new__(Node)
        self.node = node
        return self

    def __str__(self):
        return self.node.get().ToString().decode()

    def __repr__(self):
        type_format = object.__repr__(self)
        return '{0}\n{1}'.format(type_format, str(self))

    def return_type(self):
        return pyarrow_wrap_data_type(self.node.get().return_type())


cdef class Expression(_Weakrefable):
    cdef:
        shared_ptr[CGandivaExpression] expression

    cdef void init(self, shared_ptr[CGandivaExpression] expression):
        self.expression = expression

    def __str__(self):
        return self.expression.get().ToString().decode()

    def __repr__(self):
        type_format = object.__repr__(self)
        return '{0}\n{1}'.format(type_format, str(self))

    def root(self):
        return Node.create(self.expression.get().root())

    def result(self):
        return pyarrow_wrap_field(self.expression.get().result())


cdef class Condition(_Weakrefable):
    cdef:
        shared_ptr[CCondition] condition

    def __init__(self):
        raise TypeError("Do not call {}'s constructor directly, use the "
                        "TreeExprBuilder API instead"
                        .format(self.__class__.__name__))

    @staticmethod
    cdef create(shared_ptr[CCondition] condition):
        cdef Condition self = Condition.__new__(Condition)
        self.condition = condition
        return self

    def __str__(self):
        return self.condition.get().ToString().decode()

    def __repr__(self):
        type_format = object.__repr__(self)
        return '{0}\n{1}'.format(type_format, str(self))

    def root(self):
        return Node.create(self.condition.get().root())

    def result(self):
        return pyarrow_wrap_field(self.condition.get().result())


cdef class SelectionVector(_Weakrefable):
    cdef:
        shared_ptr[CSelectionVector] selection_vector

    def __init__(self):
        raise TypeError("Do not call {}'s constructor directly."
                        .format(self.__class__.__name__))

    @staticmethod
    cdef create(shared_ptr[CSelectionVector] selection_vector):
        cdef SelectionVector self = SelectionVector.__new__(SelectionVector)
        self.selection_vector = selection_vector
        return self

    def to_array(self):
        cdef shared_ptr[CArray] result = self.selection_vector.get().ToArray()
        return pyarrow_wrap_array(result)


cdef class Projector(_Weakrefable):
    cdef:
        shared_ptr[CProjector] projector
        MemoryPool pool

    def __init__(self):
        raise TypeError("Do not call {}'s constructor directly, use "
                        "make_projector instead"
                        .format(self.__class__.__name__))

    @staticmethod
    cdef create(shared_ptr[CProjector] projector, MemoryPool pool):
        cdef Projector self = Projector.__new__(Projector)
        self.projector = projector
        self.pool = pool
        return self

    @property
    def llvm_ir(self):
        return self.projector.get().DumpIR().decode()

    def evaluate(self, RecordBatch batch, SelectionVector selection=None):
        cdef vector[shared_ptr[CArray]] results
        if selection is None:
            check_status(self.projector.get().Evaluate(
                batch.sp_batch.get()[0], self.pool.pool, &results))
        else:
            check_status(
                self.projector.get().Evaluate(
                    batch.sp_batch.get()[0], selection.selection_vector.get(),
                    self.pool.pool, &results))
        cdef shared_ptr[CArray] result
        arrays = []
        for result in results:
            arrays.append(pyarrow_wrap_array(result))
        return arrays


cdef class Filter(_Weakrefable):
    cdef:
        shared_ptr[CFilter] filter

    def __init__(self):
        raise TypeError("Do not call {}'s constructor directly, use "
                        "make_filter instead"
                        .format(self.__class__.__name__))

    @staticmethod
    cdef create(shared_ptr[CFilter] filter):
        cdef Filter self = Filter.__new__(Filter)
        self.filter = filter
        return self

    @property
    def llvm_ir(self):
        return self.filter.get().DumpIR().decode()

    def evaluate(self, RecordBatch batch, MemoryPool pool, dtype='int32'):
        cdef:
            DataType type = ensure_type(dtype)
            shared_ptr[CSelectionVector] selection

        if type.id == _Type_INT16:
            check_status(SelectionVector_MakeInt16(
                batch.num_rows, pool.pool, &selection))
        elif type.id == _Type_INT32:
            check_status(SelectionVector_MakeInt32(
                batch.num_rows, pool.pool, &selection))
        elif type.id == _Type_INT64:
            check_status(SelectionVector_MakeInt64(
                batch.num_rows, pool.pool, &selection))
        else:
            raise ValueError("'dtype' of the selection vector should be "
                             "one of 'int16', 'int32' and 'int64'.")

        check_status(self.filter.get().Evaluate(
            batch.sp_batch.get()[0], selection))
        return SelectionVector.create(selection)


cdef class TreeExprBuilder(_Weakrefable):

    def make_literal(self, value, dtype):
        cdef:
            DataType type = ensure_type(dtype)
            shared_ptr[CNode] r

        if type.id == _Type_BOOL:
            r = TreeExprBuilder_MakeBoolLiteral(value)
        elif type.id == _Type_UINT8:
            r = TreeExprBuilder_MakeUInt8Literal(value)
        elif type.id == _Type_UINT16:
            r = TreeExprBuilder_MakeUInt16Literal(value)
        elif type.id == _Type_UINT32:
            r = TreeExprBuilder_MakeUInt32Literal(value)
        elif type.id == _Type_UINT64:
            r = TreeExprBuilder_MakeUInt64Literal(value)
        elif type.id == _Type_INT8:
            r = TreeExprBuilder_MakeInt8Literal(value)
        elif type.id == _Type_INT16:
            r = TreeExprBuilder_MakeInt16Literal(value)
        elif type.id == _Type_INT32:
            r = TreeExprBuilder_MakeInt32Literal(value)
        elif type.id == _Type_INT64:
            r = TreeExprBuilder_MakeInt64Literal(value)
        elif type.id == _Type_FLOAT:
            r = TreeExprBuilder_MakeFloatLiteral(value)
        elif type.id == _Type_DOUBLE:
            r = TreeExprBuilder_MakeDoubleLiteral(value)
        elif type.id == _Type_STRING:
            r = TreeExprBuilder_MakeStringLiteral(value.encode('UTF-8'))
        elif type.id == _Type_BINARY:
            r = TreeExprBuilder_MakeBinaryLiteral(value)
        else:
            raise TypeError("Didn't recognize dtype " + str(dtype))

        return Node.create(r)

    def make_expression(self, Node root_node not None,
                        Field return_field not None):
        cdef shared_ptr[CGandivaExpression] r = TreeExprBuilder_MakeExpression(
            root_node.node, return_field.sp_field)
        cdef Expression expression = Expression()
        expression.init(r)
        return expression

    def make_function(self, name, children, DataType return_type):
        cdef c_vector[shared_ptr[CNode]] c_children
        cdef Node child
        for child in children:
            if child is None:
                raise TypeError("Child nodes must not be None")
            c_children.push_back(child.node)
        cdef shared_ptr[CNode] r = TreeExprBuilder_MakeFunction(
            name.encode(), c_children, return_type.sp_type)
        return Node.create(r)

    def make_field(self, Field field not None):
        cdef shared_ptr[CNode] r = TreeExprBuilder_MakeField(field.sp_field)
        return Node.create(r)

    def make_if(self, Node condition not None, Node this_node not None,
                Node else_node not None, DataType return_type not None):
        cdef shared_ptr[CNode] r = TreeExprBuilder_MakeIf(
            condition.node, this_node.node, else_node.node,
            return_type.sp_type)
        return Node.create(r)

    def make_and(self, children):
        cdef c_vector[shared_ptr[CNode]] c_children
        cdef Node child
        for child in children:
            if child is None:
                raise TypeError("Child nodes must not be None")
            c_children.push_back(child.node)
        cdef shared_ptr[CNode] r = TreeExprBuilder_MakeAnd(c_children)
        return Node.create(r)

    def make_or(self, children):
        cdef c_vector[shared_ptr[CNode]] c_children
        cdef Node child
        for child in children:
            if child is None:
                raise TypeError("Child nodes must not be None")
            c_children.push_back(child.node)
        cdef shared_ptr[CNode] r = TreeExprBuilder_MakeOr(c_children)
        return Node.create(r)

    def _make_in_expression_int32(self, Node node not None, values):
        cdef shared_ptr[CNode] r
        cdef c_unordered_set[int32_t] c_values
        cdef int32_t v
        for v in values:
            c_values.insert(v)
        r = TreeExprBuilder_MakeInExpressionInt32(node.node, c_values)
        return Node.create(r)

    def _make_in_expression_int64(self, Node node not None, values):
        cdef shared_ptr[CNode] r
        cdef c_unordered_set[int64_t] c_values
        cdef int64_t v
        for v in values:
            c_values.insert(v)
        r = TreeExprBuilder_MakeInExpressionInt64(node.node, c_values)
        return Node.create(r)

    def _make_in_expression_time32(self, Node node not None, values):
        cdef shared_ptr[CNode] r
        cdef c_unordered_set[int32_t] c_values
        cdef int32_t v
        for v in values:
            c_values.insert(v)
        r = TreeExprBuilder_MakeInExpressionTime32(node.node, c_values)
        return Node.create(r)

    def _make_in_expression_time64(self, Node node not None, values):
        cdef shared_ptr[CNode] r
        cdef c_unordered_set[int64_t] c_values
        cdef int64_t v
        for v in values:
            c_values.insert(v)
        r = TreeExprBuilder_MakeInExpressionTime64(node.node, c_values)
        return Node.create(r)

    def _make_in_expression_date32(self, Node node not None, values):
        cdef shared_ptr[CNode] r
        cdef c_unordered_set[int32_t] c_values
        cdef int32_t v
        for v in values:
            c_values.insert(v)
        r = TreeExprBuilder_MakeInExpressionDate32(node.node, c_values)
        return Node.create(r)

    def _make_in_expression_date64(self, Node node not None, values):
        cdef shared_ptr[CNode] r
        cdef c_unordered_set[int64_t] c_values
        cdef int64_t v
        for v in values:
            c_values.insert(v)
        r = TreeExprBuilder_MakeInExpressionDate64(node.node, c_values)
        return Node.create(r)

    def _make_in_expression_timestamp(self, Node node not None, values):
        cdef shared_ptr[CNode] r
        cdef c_unordered_set[int64_t] c_values
        cdef int64_t v
        for v in values:
            c_values.insert(v)
        r = TreeExprBuilder_MakeInExpressionTimeStamp(node.node, c_values)
        return Node.create(r)

    def _make_in_expression_binary(self, Node node not None, values):
        cdef shared_ptr[CNode] r
        cdef c_unordered_set[c_string] c_values
        cdef c_string v
        for v in values:
            c_values.insert(v)
        r = TreeExprBuilder_MakeInExpressionString(node.node, c_values)
        return Node.create(r)

    def _make_in_expression_string(self, Node node not None, values):
        cdef shared_ptr[CNode] r
        cdef c_unordered_set[c_string] c_values
        cdef c_string _v
        for v in values:
            _v = v.encode('UTF-8')
            c_values.insert(_v)
        r = TreeExprBuilder_MakeInExpressionString(node.node, c_values)
        return Node.create(r)

    def make_in_expression(self, Node node not None, values, dtype):
        cdef DataType type = ensure_type(dtype)

        if type.id == _Type_INT32:
            return self._make_in_expression_int32(node, values)
        elif type.id == _Type_INT64:
            return self._make_in_expression_int64(node, values)
        elif type.id == _Type_TIME32:
            return self._make_in_expression_time32(node, values)
        elif type.id == _Type_TIME64:
            return self._make_in_expression_time64(node, values)
        elif type.id == _Type_TIMESTAMP:
            return self._make_in_expression_timestamp(node, values)
        elif type.id == _Type_DATE32:
            return self._make_in_expression_date32(node, values)
        elif type.id == _Type_DATE64:
            return self._make_in_expression_date64(node, values)
        elif type.id == _Type_BINARY:
            return self._make_in_expression_binary(node, values)
        elif type.id == _Type_STRING:
            return self._make_in_expression_string(node, values)
        else:
            raise TypeError("Data type " + str(dtype) + " not supported.")

    def make_condition(self, Node condition not None):
        cdef shared_ptr[CCondition] r = TreeExprBuilder_MakeCondition(
            condition.node)
        return Condition.create(r)


cpdef make_projector(Schema schema, children, MemoryPool pool,
                     str selection_mode="NONE"):
    """
    Construct a projection using expressions.

    A projector is built for a specific schema and vector of expressions.
    Once the projector is built, it can be used to evaluate many row batches.

    Parameters
    ----------
    schema : pyarrow.Schema
        Schema for the record batches, and the expressions.
    children : list[pyarrow.gandiva.Expression]
        List of projectable expression objects.
    pool : pyarrow.MemoryPool
        Memory pool used to allocate output arrays.
    selection_mode : str, default "NONE"
        Possible values are NONE, UINT16, UINT32, UINT64.

    Returns
    -------
    Projector instance
    """
    cdef:
        Expression child
        c_vector[shared_ptr[CGandivaExpression]] c_children
        shared_ptr[CProjector] result

    for child in children:
        if child is None:
            raise TypeError("Expressions must not be None")
        c_children.push_back(child.expression)

    check_status(
        Projector_Make(schema.sp_schema, c_children,
                       _ensure_selection_mode(selection_mode),
                       CConfigurationBuilder.DefaultConfiguration(),
                       &result))
    return Projector.create(result, pool)


cpdef make_filter(Schema schema, Condition condition):
    """
    Construct a filter based on a condition.

    A filter is built for a specific schema and condition. Once the filter is
    built, it can be used to evaluate many row batches.

    Parameters
    ----------
    schema : pyarrow.Schema
        Schema for the record batches, and the condition.
    condition : pyarrow.gandiva.Condition
        Filter condition.

    Returns
    -------
    Filter instance
    """
    cdef shared_ptr[CFilter] result
    if condition is None:
        raise TypeError("Condition must not be None")
    check_status(
        Filter_Make(schema.sp_schema, condition.condition, &result))
    return Filter.create(result)


cdef class FunctionSignature(_Weakrefable):
    """
    Signature of a Gandiva function including name, parameter types
    and return type.
    """

    cdef:
        shared_ptr[CFunctionSignature] signature

    def __init__(self):
        raise TypeError("Do not call {}'s constructor directly."
                        .format(self.__class__.__name__))

    @staticmethod
    cdef create(shared_ptr[CFunctionSignature] signature):
        cdef FunctionSignature self = FunctionSignature.__new__(
            FunctionSignature)
        self.signature = signature
        return self

    def return_type(self):
        return pyarrow_wrap_data_type(self.signature.get().ret_type())

    def param_types(self):
        result = []
        cdef vector[shared_ptr[CDataType]] types = \
            self.signature.get().param_types()
        for t in types:
            result.append(pyarrow_wrap_data_type(t))
        return result

    def name(self):
        return self.signature.get().base_name().decode()

    def __repr__(self):
        signature = self.signature.get().ToString().decode()
        return "FunctionSignature(" + signature + ")"


def get_registered_function_signatures():
    """
    Return the function in Gandiva's ExpressionRegistry.

    Returns
    -------
    registry: a list of registered function signatures
    """
    results = []

    cdef vector[shared_ptr[CFunctionSignature]] signatures = \
        GetRegisteredFunctionSignatures()

    for signature in signatures:
        results.append(FunctionSignature.create(signature))

    return results

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

import datetime
import pytest

import pyarrow as pa


@pytest.mark.gandiva
def test_tree_exp_builder():
    import pyarrow.gandiva as gandiva

    builder = gandiva.TreeExprBuilder()

    field_a = pa.field('a', pa.int32())
    field_b = pa.field('b', pa.int32())

    schema = pa.schema([field_a, field_b])

    field_result = pa.field('res', pa.int32())

    node_a = builder.make_field(field_a)
    node_b = builder.make_field(field_b)

    assert node_a.return_type() == field_a.type

    condition = builder.make_function("greater_than", [node_a, node_b],
                                      pa.bool_())
    if_node = builder.make_if(condition, node_a, node_b, pa.int32())

    expr = builder.make_expression(if_node, field_result)

    assert expr.result().type == pa.int32()

    projector = gandiva.make_projector(
        schema, [expr], pa.default_memory_pool())

    # Gandiva generates compute kernel function named `@expr_X`
    assert projector.llvm_ir.find("@expr_") != -1

    a = pa.array([10, 12, -20, 5], type=pa.int32())
    b = pa.array([5, 15, 15, 17], type=pa.int32())
    e = pa.array([10, 15, 15, 17], type=pa.int32())
    input_batch = pa.RecordBatch.from_arrays([a, b], names=['a', 'b'])

    r, = projector.evaluate(input_batch)
    assert r.equals(e)


@pytest.mark.gandiva
def test_table():
    import pyarrow.gandiva as gandiva

    table = pa.Table.from_arrays([pa.array([1.0, 2.0]), pa.array([3.0, 4.0])],
                                 ['a', 'b'])

    builder = gandiva.TreeExprBuilder()
    node_a = builder.make_field(table.schema.field("a"))
    node_b = builder.make_field(table.schema.field("b"))

    sum = builder.make_function("add", [node_a, node_b], pa.float64())

    field_result = pa.field("c", pa.float64())
    expr = builder.make_expression(sum, field_result)

    projector = gandiva.make_projector(
        table.schema, [expr], pa.default_memory_pool())

    # TODO: Add .evaluate function which can take Tables instead of
    # RecordBatches
    r, = projector.evaluate(table.to_batches()[0])

    e = pa.array([4.0, 6.0])
    assert r.equals(e)


@pytest.mark.gandiva
def test_filter():
    import pyarrow.gandiva as gandiva

    table = pa.Table.from_arrays([pa.array([1.0 * i for i in range(10000)])],
                                 ['a'])

    builder = gandiva.TreeExprBuilder()
    node_a = builder.make_field(table.schema.field("a"))
    thousand = builder.make_literal(1000.0, pa.float64())
    cond = builder.make_function("less_than", [node_a, thousand], pa.bool_())
    condition = builder.make_condition(cond)

    assert condition.result().type == pa.bool_()

    filter = gandiva.make_filter(table.schema, condition)
    # Gandiva generates compute kernel function named `@expr_X`
    assert filter.llvm_ir.find("@expr_") != -1

    result = filter.evaluate(table.to_batches()[0], pa.default_memory_pool())
    assert result.to_array().equals(pa.array(range(1000), type=pa.uint32()))


@pytest.mark.gandiva
def test_in_expr():
    import pyarrow.gandiva as gandiva

    arr = pa.array(["ga", "an", "nd", "di", "iv", "va"])
    table = pa.Table.from_arrays([arr], ["a"])

    # string
    builder = gandiva.TreeExprBuilder()
    node_a = builder.make_field(table.schema.field("a"))
    cond = builder.make_in_expression(node_a, ["an", "nd"], pa.string())
    condition = builder.make_condition(cond)
    filter = gandiva.make_filter(table.schema, condition)
    result = filter.evaluate(table.to_batches()[0], pa.default_memory_pool())
    assert result.to_array().equals(pa.array([1, 2], type=pa.uint32()))

    # int32
    arr = pa.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 4])
    table = pa.Table.from_arrays([arr.cast(pa.int32())], ["a"])
    node_a = builder.make_field(table.schema.field("a"))
    cond = builder.make_in_expression(node_a, [1, 5], pa.int32())
    condition = builder.make_condition(cond)
    filter = gandiva.make_filter(table.schema, condition)
    result = filter.evaluate(table.to_batches()[0], pa.default_memory_pool())
    assert result.to_array().equals(pa.array([1, 3, 4, 8], type=pa.uint32()))

    # int64
    arr = pa.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 4])
    table = pa.Table.from_arrays([arr], ["a"])
    node_a = builder.make_field(table.schema.field("a"))
    cond = builder.make_in_expression(node_a, [1, 5], pa.int64())
    condition = builder.make_condition(cond)
    filter = gandiva.make_filter(table.schema, condition)
    result = filter.evaluate(table.to_batches()[0], pa.default_memory_pool())
    assert result.to_array().equals(pa.array([1, 3, 4, 8], type=pa.uint32()))


@pytest.mark.skip(reason="Gandiva C++ did not have *real* binary, "
                         "time and date support.")
def test_in_expr_todo():
    import pyarrow.gandiva as gandiva
    # TODO: Implement reasonable support for timestamp, time & date.
    # Current exceptions:
    # pyarrow.lib.ArrowException: ExpressionValidationError:
    # Evaluation expression for IN clause returns XXXX values are of typeXXXX

    # binary
    arr = pa.array([b"ga", b"an", b"nd", b"di", b"iv", b"va"])
    table = pa.Table.from_arrays([arr], ["a"])

    builder = gandiva.TreeExprBuilder()
    node_a = builder.make_field(table.schema.field("a"))
    cond = builder.make_in_expression(node_a, [b'an', b'nd'], pa.binary())
    condition = builder.make_condition(cond)

    filter = gandiva.make_filter(table.schema, condition)
    result = filter.evaluate(table.to_batches()[0], pa.default_memory_pool())
    assert result.to_array().equals(pa.array([1, 2], type=pa.uint32()))

    # timestamp
    datetime_1 = datetime.datetime.utcfromtimestamp(1542238951.621877)
    datetime_2 = datetime.datetime.utcfromtimestamp(1542238911.621877)
    datetime_3 = datetime.datetime.utcfromtimestamp(1542238051.621877)

    arr = pa.array([datetime_1, datetime_2, datetime_3])
    table = pa.Table.from_arrays([arr], ["a"])

    builder = gandiva.TreeExprBuilder()
    node_a = builder.make_field(table.schema.field("a"))
    cond = builder.make_in_expression(node_a, [datetime_2], pa.timestamp('ms'))
    condition = builder.make_condition(cond)

    filter = gandiva.make_filter(table.schema, condition)
    result = filter.evaluate(table.to_batches()[0], pa.default_memory_pool())
    assert list(result.to_array()) == [1]

    # time
    time_1 = datetime_1.time()
    time_2 = datetime_2.time()
    time_3 = datetime_3.time()

    arr = pa.array([time_1, time_2, time_3])
    table = pa.Table.from_arrays([arr], ["a"])

    builder = gandiva.TreeExprBuilder()
    node_a = builder.make_field(table.schema.field("a"))
    cond = builder.make_in_expression(node_a, [time_2], pa.time64('ms'))
    condition = builder.make_condition(cond)

    filter = gandiva.make_filter(table.schema, condition)
    result = filter.evaluate(table.to_batches()[0], pa.default_memory_pool())
    assert list(result.to_array()) == [1]

    # date
    date_1 = datetime_1.date()
    date_2 = datetime_2.date()
    date_3 = datetime_3.date()

    arr = pa.array([date_1, date_2, date_3])
    table = pa.Table.from_arrays([arr], ["a"])

    builder = gandiva.TreeExprBuilder()
    node_a = builder.make_field(table.schema.field("a"))
    cond = builder.make_in_expression(node_a, [date_2], pa.date32())
    condition = builder.make_condition(cond)

    filter = gandiva.make_filter(table.schema, condition)
    result = filter.evaluate(table.to_batches()[0], pa.default_memory_pool())
    assert list(result.to_array()) == [1]


@pytest.mark.gandiva
def test_boolean():
    import pyarrow.gandiva as gandiva

    table = pa.Table.from_arrays([
        pa.array([1., 31., 46., 3., 57., 44., 22.]),
        pa.array([5., 45., 36., 73., 83., 23., 76.])],
        ['a', 'b'])

    builder = gandiva.TreeExprBuilder()
    node_a = builder.make_field(table.schema.field("a"))
    node_b = builder.make_field(table.schema.field("b"))
    fifty = builder.make_literal(50.0, pa.float64())
    eleven = builder.make_literal(11.0, pa.float64())

    cond_1 = builder.make_function("less_than", [node_a, fifty], pa.bool_())
    cond_2 = builder.make_function("greater_than", [node_a, node_b],
                                   pa.bool_())
    cond_3 = builder.make_function("less_than", [node_b, eleven], pa.bool_())
    cond = builder.make_or([builder.make_and([cond_1, cond_2]), cond_3])
    condition = builder.make_condition(cond)

    filter = gandiva.make_filter(table.schema, condition)
    result = filter.evaluate(table.to_batches()[0], pa.default_memory_pool())
    assert result.to_array().equals(pa.array([0, 2, 5], type=pa.uint32()))


@pytest.mark.gandiva
def test_literals():
    import pyarrow.gandiva as gandiva

    builder = gandiva.TreeExprBuilder()

    builder.make_literal(True, pa.bool_())
    builder.make_literal(0, pa.uint8())
    builder.make_literal(1, pa.uint16())
    builder.make_literal(2, pa.uint32())
    builder.make_literal(3, pa.uint64())
    builder.make_literal(4, pa.int8())
    builder.make_literal(5, pa.int16())
    builder.make_literal(6, pa.int32())
    builder.make_literal(7, pa.int64())
    builder.make_literal(8.0, pa.float32())
    builder.make_literal(9.0, pa.float64())
    builder.make_literal("hello", pa.string())
    builder.make_literal(b"world", pa.binary())

    builder.make_literal(True, "bool")
    builder.make_literal(0, "uint8")
    builder.make_literal(1, "uint16")
    builder.make_literal(2, "uint32")
    builder.make_literal(3, "uint64")
    builder.make_literal(4, "int8")
    builder.make_literal(5, "int16")
    builder.make_literal(6, "int32")
    builder.make_literal(7, "int64")
    builder.make_literal(8.0, "float32")
    builder.make_literal(9.0, "float64")
    builder.make_literal("hello", "string")
    builder.make_literal(b"world", "binary")

    with pytest.raises(TypeError):
        builder.make_literal("hello", pa.int64())
    with pytest.raises(TypeError):
        builder.make_literal(True, None)


@pytest.mark.gandiva
def test_regex():
    import pyarrow.gandiva as gandiva

    elements = ["park", "sparkle", "bright spark and fire", "spark"]
    data = pa.array(elements, type=pa.string())
    table = pa.Table.from_arrays([data], names=['a'])

    builder = gandiva.TreeExprBuilder()
    node_a = builder.make_field(table.schema.field("a"))
    regex = builder.make_literal("%spark%", pa.string())
    like = builder.make_function("like", [node_a, regex], pa.bool_())

    field_result = pa.field("b", pa.bool_())
    expr = builder.make_expression(like, field_result)

    projector = gandiva.make_projector(
        table.schema, [expr], pa.default_memory_pool())

    r, = projector.evaluate(table.to_batches()[0])
    b = pa.array([False, True, True, True], type=pa.bool_())
    assert r.equals(b)


@pytest.mark.gandiva
def test_get_registered_function_signatures():
    import pyarrow.gandiva as gandiva
    signatures = gandiva.get_registered_function_signatures()

    assert type(signatures[0].return_type()) is pa.DataType
    assert type(signatures[0].param_types()) is list
    assert hasattr(signatures[0], "name")


@pytest.mark.gandiva
def test_filter_project():
    import pyarrow.gandiva as gandiva
    mpool = pa.default_memory_pool()
    # Create a table with some sample data
    array0 = pa.array([10, 12, -20, 5, 21, 29], pa.int32())
    array1 = pa.array([5, 15, 15, 17, 12, 3], pa.int32())
    array2 = pa.array([1, 25, 11, 30, -21, None], pa.int32())

    table = pa.Table.from_arrays([array0, array1, array2], ['a', 'b', 'c'])

    field_result = pa.field("res", pa.int32())

    builder = gandiva.TreeExprBuilder()
    node_a = builder.make_field(table.schema.field("a"))
    node_b = builder.make_field(table.schema.field("b"))
    node_c = builder.make_field(table.schema.field("c"))

    greater_than_function = builder.make_function("greater_than",
                                                  [node_a, node_b], pa.bool_())
    filter_condition = builder.make_condition(
        greater_than_function)

    project_condition = builder.make_function("less_than",
                                              [node_b, node_c], pa.bool_())
    if_node = builder.make_if(project_condition,
                              node_b, node_c, pa.int32())
    expr = builder.make_expression(if_node, field_result)

    # Build a filter for the expressions.
    filter = gandiva.make_filter(table.schema, filter_condition)

    # Build a projector for the expressions.
    projector = gandiva.make_projector(
        table.schema, [expr], mpool, "UINT32")

    # Evaluate filter
    selection_vector = filter.evaluate(table.to_batches()[0], mpool)

    # Evaluate project
    r, = projector.evaluate(
        table.to_batches()[0], selection_vector)

    exp = pa.array([1, -21, None], pa.int32())
    assert r.equals(exp)


@pytest.mark.gandiva
def test_to_string():
    import pyarrow.gandiva as gandiva
    builder = gandiva.TreeExprBuilder()

    assert str(builder.make_literal(2.0, pa.float64())
               ).startswith('(const double) 2 raw(')
    assert str(builder.make_literal(2, pa.int64())) == '(const int64) 2'
    assert str(builder.make_field(pa.field('x', pa.float64()))) == '(double) x'
    assert str(builder.make_field(pa.field('y', pa.string()))) == '(string) y'

    field_z = builder.make_field(pa.field('z', pa.bool_()))
    func_node = builder.make_function('not', [field_z], pa.bool_())
    assert str(func_node) == 'bool not((bool) z)'

    field_y = builder.make_field(pa.field('y', pa.bool_()))
    and_node = builder.make_and([func_node, field_y])
    assert str(and_node) == 'bool not((bool) z) && (bool) y'


@pytest.mark.gandiva
def test_rejects_none():
    import pyarrow.gandiva as gandiva

    builder = gandiva.TreeExprBuilder()

    field_x = pa.field('x', pa.int32())
    schema = pa.schema([field_x])
    literal_true = builder.make_literal(True, pa.bool_())

    with pytest.raises(TypeError):
        builder.make_field(None)

    with pytest.raises(TypeError):
        builder.make_if(literal_true, None, None, None)

    with pytest.raises(TypeError):
        builder.make_and([literal_true, None])

    with pytest.raises(TypeError):
        builder.make_or([None, literal_true])

    with pytest.raises(TypeError):
        builder.make_in_expression(None, [1, 2, 3], pa.int32())

    with pytest.raises(TypeError):
        builder.make_expression(None, field_x)

    with pytest.raises(TypeError):
        builder.make_condition(None)

    with pytest.raises(TypeError):
        builder.make_function('less_than', [literal_true, None], pa.bool_())

    with pytest.raises(TypeError):
        gandiva.make_projector(schema, [None])

    with pytest.raises(TypeError):
        gandiva.make_filter(schema, None)

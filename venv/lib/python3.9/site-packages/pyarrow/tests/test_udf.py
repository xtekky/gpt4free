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


import pytest

import pyarrow as pa
from pyarrow import compute as pc

# UDFs are all tested with a dataset scan
pytestmark = pytest.mark.dataset


try:
    import pyarrow.dataset as ds
except ImportError:
    ds = None


def mock_udf_context(batch_length=10):
    from pyarrow._compute import _get_scalar_udf_context
    return _get_scalar_udf_context(pa.default_memory_pool(), batch_length)


class MyError(RuntimeError):
    pass


@pytest.fixture(scope="session")
def unary_func_fixture():
    """
    Register a unary scalar function.
    """
    def unary_function(ctx, x):
        return pc.call_function("add", [x, 1],
                                memory_pool=ctx.memory_pool)
    func_name = "y=x+1"
    unary_doc = {"summary": "add function",
                 "description": "test add function"}
    pc.register_scalar_function(unary_function,
                                func_name,
                                unary_doc,
                                {"array": pa.int64()},
                                pa.int64())
    return unary_function, func_name


@pytest.fixture(scope="session")
def binary_func_fixture():
    """
    Register a binary scalar function.
    """
    def binary_function(ctx, m, x):
        return pc.call_function("multiply", [m, x],
                                memory_pool=ctx.memory_pool)
    func_name = "y=mx"
    binary_doc = {"summary": "y=mx",
                  "description": "find y from y = mx"}
    pc.register_scalar_function(binary_function,
                                func_name,
                                binary_doc,
                                {"m": pa.int64(),
                                 "x": pa.int64(),
                                 },
                                pa.int64())
    return binary_function, func_name


@pytest.fixture(scope="session")
def ternary_func_fixture():
    """
    Register a ternary scalar function.
    """
    def ternary_function(ctx, m, x, c):
        mx = pc.call_function("multiply", [m, x],
                              memory_pool=ctx.memory_pool)
        return pc.call_function("add", [mx, c],
                                memory_pool=ctx.memory_pool)
    ternary_doc = {"summary": "y=mx+c",
                   "description": "find y from y = mx + c"}
    func_name = "y=mx+c"
    pc.register_scalar_function(ternary_function,
                                func_name,
                                ternary_doc,
                                {
                                    "array1": pa.int64(),
                                    "array2": pa.int64(),
                                    "array3": pa.int64(),
                                },
                                pa.int64())
    return ternary_function, func_name


@pytest.fixture(scope="session")
def varargs_func_fixture():
    """
    Register a varargs scalar function with at least two arguments.
    """
    def varargs_function(ctx, first, *values):
        acc = first
        for val in values:
            acc = pc.call_function("add", [acc, val],
                                   memory_pool=ctx.memory_pool)
        return acc
    func_name = "z=ax+by+c"
    varargs_doc = {"summary": "z=ax+by+c",
                   "description": "find z from z = ax + by + c"
                   }
    pc.register_scalar_function(varargs_function,
                                func_name,
                                varargs_doc,
                                {
                                    "array1": pa.int64(),
                                    "array2": pa.int64(),
                                },
                                pa.int64())
    return varargs_function, func_name


@pytest.fixture(scope="session")
def nullary_func_fixture():
    """
    Register a nullary scalar function.
    """
    def nullary_func(context):
        return pa.array([42] * context.batch_length, type=pa.int64(),
                        memory_pool=context.memory_pool)

    func_doc = {
        "summary": "random function",
        "description": "generates a random value"
    }
    func_name = "test_nullary_func"
    pc.register_scalar_function(nullary_func,
                                func_name,
                                func_doc,
                                {},
                                pa.int64())

    return nullary_func, func_name


@pytest.fixture(scope="session")
def wrong_output_type_func_fixture():
    """
    Register a scalar function which returns something that is neither
    a Arrow scalar or array.
    """
    def wrong_output_type(ctx):
        return 42

    func_name = "test_wrong_output_type"
    in_types = {}
    out_type = pa.int64()
    doc = {
        "summary": "return wrong output type",
        "description": ""
    }
    pc.register_scalar_function(wrong_output_type, func_name, doc,
                                in_types, out_type)
    return wrong_output_type, func_name


@pytest.fixture(scope="session")
def wrong_output_datatype_func_fixture():
    """
    Register a scalar function whose actual output DataType doesn't
    match the declared output DataType.
    """
    def wrong_output_datatype(ctx, array):
        return pc.call_function("add", [array, 1])
    func_name = "test_wrong_output_datatype"
    in_types = {"array": pa.int64()}
    # The actual output DataType will be int64.
    out_type = pa.int16()
    doc = {
        "summary": "return wrong output datatype",
        "description": ""
    }
    pc.register_scalar_function(wrong_output_datatype, func_name, doc,
                                in_types, out_type)
    return wrong_output_datatype, func_name


@pytest.fixture(scope="session")
def wrong_signature_func_fixture():
    """
    Register a scalar function with the wrong signature.
    """
    # Missing the context argument
    def wrong_signature():
        return pa.scalar(1, type=pa.int64())

    func_name = "test_wrong_signature"
    in_types = {}
    out_type = pa.int64()
    doc = {
        "summary": "UDF with wrong signature",
        "description": ""
    }
    pc.register_scalar_function(wrong_signature, func_name, doc,
                                in_types, out_type)
    return wrong_signature, func_name


@pytest.fixture(scope="session")
def raising_func_fixture():
    """
    Register a scalar function which raises a custom exception.
    """
    def raising_func(ctx):
        raise MyError("error raised by scalar UDF")
    func_name = "test_raise"
    doc = {
        "summary": "raising function",
        "description": ""
    }
    pc.register_scalar_function(raising_func, func_name, doc,
                                {}, pa.int64())
    return raising_func, func_name


def check_scalar_function(func_fixture,
                          inputs, *,
                          run_in_dataset=True,
                          batch_length=None):
    function, name = func_fixture
    if batch_length is None:
        all_scalar = True
        for arg in inputs:
            if isinstance(arg, pa.Array):
                all_scalar = False
                batch_length = len(arg)
        if all_scalar:
            batch_length = 1

    expected_output = function(mock_udf_context(batch_length), *inputs)
    func = pc.get_function(name)
    assert func.name == name

    result = pc.call_function(name, inputs, length=batch_length)
    assert result == expected_output
    # At the moment there is an issue when handling nullary functions.
    # See: ARROW-15286 and ARROW-16290.
    if run_in_dataset:
        field_names = [f'field{index}' for index, in_arr in inputs]
        table = pa.Table.from_arrays(inputs, field_names)
        dataset = ds.dataset(table)
        func_args = [ds.field(field_name) for field_name in field_names]
        result_table = dataset.to_table(
            columns={'result': ds.field('')._call(name, func_args)})
        assert result_table.column(0).chunks[0] == expected_output


def test_scalar_udf_array_unary(unary_func_fixture):
    check_scalar_function(unary_func_fixture,
                          [
                              pa.array([10, 20], pa.int64())
                          ]
                          )


def test_scalar_udf_array_binary(binary_func_fixture):
    check_scalar_function(binary_func_fixture,
                          [
                              pa.array([10, 20], pa.int64()),
                              pa.array([2, 4], pa.int64())
                          ]
                          )


def test_scalar_udf_array_ternary(ternary_func_fixture):
    check_scalar_function(ternary_func_fixture,
                          [
                              pa.array([10, 20], pa.int64()),
                              pa.array([2, 4], pa.int64()),
                              pa.array([5, 10], pa.int64())
                          ]
                          )


def test_scalar_udf_array_varargs(varargs_func_fixture):
    check_scalar_function(varargs_func_fixture,
                          [
                              pa.array([2, 3], pa.int64()),
                              pa.array([10, 20], pa.int64()),
                              pa.array([3, 7], pa.int64()),
                              pa.array([20, 30], pa.int64()),
                              pa.array([5, 10], pa.int64())
                          ]
                          )


def test_registration_errors():
    # validate function name
    doc = {
        "summary": "test udf input",
        "description": "parameters are validated"
    }
    in_types = {"scalar": pa.int64()}
    out_type = pa.int64()

    def test_reg_function(context):
        return pa.array([10])

    with pytest.raises(TypeError):
        pc.register_scalar_function(test_reg_function,
                                    None, doc, in_types,
                                    out_type)

    # validate function
    with pytest.raises(TypeError, match="func must be a callable"):
        pc.register_scalar_function(None, "test_none_function", doc, in_types,
                                    out_type)

    # validate output type
    expected_expr = "DataType expected, got <class 'NoneType'>"
    with pytest.raises(TypeError, match=expected_expr):
        pc.register_scalar_function(test_reg_function,
                                    "test_output_function", doc, in_types,
                                    None)

    # validate input type
    expected_expr = "in_types must be a dictionary of DataType"
    with pytest.raises(TypeError, match=expected_expr):
        pc.register_scalar_function(test_reg_function,
                                    "test_input_function", doc, None,
                                    out_type)

    # register an already registered function
    # first registration
    pc.register_scalar_function(test_reg_function,
                                "test_reg_function", doc, {},
                                out_type)
    # second registration
    expected_expr = "Already have a function registered with name:" \
        + " test_reg_function"
    with pytest.raises(KeyError, match=expected_expr):
        pc.register_scalar_function(test_reg_function,
                                    "test_reg_function", doc, {},
                                    out_type)


def test_varargs_function_validation(varargs_func_fixture):
    _, func_name = varargs_func_fixture

    error_msg = r"VarArgs function 'z=ax\+by\+c' needs at least 2 arguments"

    with pytest.raises(ValueError, match=error_msg):
        pc.call_function(func_name, [42])


def test_function_doc_validation():
    # validate arity
    in_types = {"scalar": pa.int64()}
    out_type = pa.int64()

    # doc with no summary
    func_doc = {
        "description": "desc"
    }

    def add_const(ctx, scalar):
        return pc.call_function("add", [scalar, 1])

    with pytest.raises(ValueError,
                       match="Function doc must contain a summary"):
        pc.register_scalar_function(add_const, "test_no_summary",
                                    func_doc, in_types,
                                    out_type)

    # doc with no decription
    func_doc = {
        "summary": "test summary"
    }

    with pytest.raises(ValueError,
                       match="Function doc must contain a description"):
        pc.register_scalar_function(add_const, "test_no_desc",
                                    func_doc, in_types,
                                    out_type)


def test_nullary_function(nullary_func_fixture):
    # XXX the Python compute layer API doesn't let us override batch_length,
    # so only test with the default value of 1.
    check_scalar_function(nullary_func_fixture, [], run_in_dataset=False,
                          batch_length=1)


def test_wrong_output_type(wrong_output_type_func_fixture):
    _, func_name = wrong_output_type_func_fixture

    with pytest.raises(TypeError,
                       match="Unexpected output type: int"):
        pc.call_function(func_name, [], length=1)


def test_wrong_output_datatype(wrong_output_datatype_func_fixture):
    _, func_name = wrong_output_datatype_func_fixture

    expected_expr = ("Expected output datatype int16, "
                     "but function returned datatype int64")

    with pytest.raises(TypeError, match=expected_expr):
        pc.call_function(func_name, [pa.array([20, 30])])


def test_wrong_signature(wrong_signature_func_fixture):
    _, func_name = wrong_signature_func_fixture

    expected_expr = (r"wrong_signature\(\) takes 0 positional arguments "
                     "but 1 was given")

    with pytest.raises(TypeError, match=expected_expr):
        pc.call_function(func_name, [], length=1)


def test_wrong_datatype_declaration():
    def identity(ctx, val):
        return val

    func_name = "test_wrong_datatype_declaration"
    in_types = {"array": pa.int64()}
    out_type = {}
    doc = {
        "summary": "test output value",
        "description": "test output"
    }
    with pytest.raises(TypeError,
                       match="DataType expected, got <class 'dict'>"):
        pc.register_scalar_function(identity, func_name,
                                    doc, in_types, out_type)


def test_wrong_input_type_declaration():
    def identity(ctx, val):
        return val

    func_name = "test_wrong_input_type_declaration"
    in_types = {"array": None}
    out_type = pa.int64()
    doc = {
        "summary": "test invalid input type",
        "description": "invalid input function"
    }
    with pytest.raises(TypeError,
                       match="DataType expected, got <class 'NoneType'>"):
        pc.register_scalar_function(identity, func_name, doc,
                                    in_types, out_type)


def test_udf_context(unary_func_fixture):
    # Check the memory_pool argument is properly propagated
    proxy_pool = pa.proxy_memory_pool(pa.default_memory_pool())
    _, func_name = unary_func_fixture

    res = pc.call_function(func_name,
                           [pa.array([1] * 1000, type=pa.int64())],
                           memory_pool=proxy_pool)
    assert res == pa.array([2] * 1000, type=pa.int64())
    assert proxy_pool.bytes_allocated() == 1000 * 8
    # Destroying Python array should destroy underlying C++ memory
    res = None
    assert proxy_pool.bytes_allocated() == 0


def test_raising_func(raising_func_fixture):
    _, func_name = raising_func_fixture
    with pytest.raises(MyError, match="error raised by scalar UDF"):
        pc.call_function(func_name, [], length=1)


def test_scalar_input(unary_func_fixture):
    function, func_name = unary_func_fixture
    res = pc.call_function(func_name, [pa.scalar(10)])
    assert res == pa.scalar(11)


def test_input_lifetime(unary_func_fixture):
    function, func_name = unary_func_fixture

    proxy_pool = pa.proxy_memory_pool(pa.default_memory_pool())
    assert proxy_pool.bytes_allocated() == 0

    v = pa.array([1] * 1000, type=pa.int64(), memory_pool=proxy_pool)
    assert proxy_pool.bytes_allocated() == 1000 * 8
    pc.call_function(func_name, [v])
    assert proxy_pool.bytes_allocated() == 1000 * 8
    # Calling a UDF should not have kept `v` alive longer than required
    v = None
    assert proxy_pool.bytes_allocated() == 0

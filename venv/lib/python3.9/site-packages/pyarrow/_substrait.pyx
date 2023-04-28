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
from cython.operator cimport dereference as deref
from libcpp.vector cimport vector as std_vector

from pyarrow import Buffer, py_buffer
from pyarrow.lib import frombytes, tobytes
from pyarrow.lib cimport *
from pyarrow.includes.libarrow cimport *
from pyarrow.includes.libarrow_substrait cimport *


cdef CDeclaration _create_named_table_provider(dict named_args, const std_vector[c_string]& names):
    cdef:
        c_string c_name
        shared_ptr[CTable] c_in_table
        shared_ptr[CTableSourceNodeOptions] c_tablesourceopts
        shared_ptr[CExecNodeOptions] c_input_node_opts
        vector[CDeclaration.Input] no_c_inputs

    py_names = []
    for i in range(names.size()):
        c_name = names[i]
        py_names.append(frombytes(c_name))

    py_table = named_args["provider"](py_names)
    c_in_table = pyarrow_unwrap_table(py_table)
    c_tablesourceopts = make_shared[CTableSourceNodeOptions](c_in_table)
    c_input_node_opts = static_pointer_cast[CExecNodeOptions, CTableSourceNodeOptions](
        c_tablesourceopts)
    return CDeclaration(tobytes("table_source"),
                        no_c_inputs, c_input_node_opts)


def run_query(plan, *, table_provider=None, use_threads=True):
    """
    Execute a Substrait plan and read the results as a RecordBatchReader.

    Parameters
    ----------
    plan : Union[Buffer, bytes]
        The serialized Substrait plan to execute.
    table_provider : object (optional)
        A function to resolve any NamedTable relation to a table.
        The function will receive a single argument which will be a list
        of strings representing the table name and should return a pyarrow.Table.
    use_threads : bool, default True
        If True then multiple threads will be used to run the query.  If False then
        all CPU intensive work will be done on the calling thread.

    Returns
    -------
    RecordBatchReader
        A reader containing the result of the executed query

    Examples
    --------
    >>> import pyarrow as pa
    >>> from pyarrow.lib import tobytes
    >>> import pyarrow.substrait as substrait
    >>> test_table_1 = pa.Table.from_pydict({"x": [1, 2, 3]})
    >>> test_table_2 = pa.Table.from_pydict({"x": [4, 5, 6]})
    >>> def table_provider(names):
    ...     if not names:
    ...        raise Exception("No names provided")
    ...     elif names[0] == "t1":
    ...        return test_table_1
    ...     elif names[1] == "t2":
    ...        return test_table_2
    ...     else:
    ...        raise Exception("Unrecognized table name")
    ... 
    >>> substrait_query = '''
    ...         {
    ...             "relations": [
    ...             {"rel": {
    ...                 "read": {
    ...                 "base_schema": {
    ...                     "struct": {
    ...                     "types": [
    ...                                 {"i64": {}}
    ...                             ]
    ...                     },
    ...                     "names": [
    ...                             "x"
    ...                             ]
    ...                 },
    ...                 "namedTable": {
    ...                         "names": ["t1"]
    ...                 }
    ...                 }
    ...             }}
    ...             ]
    ...         }
    ... '''
    >>> buf = pa._substrait._parse_json_plan(tobytes(substrait_query))
    >>> reader = pa.substrait.run_query(buf, table_provider)
    >>> reader.read_all()
    pyarrow.Table
    x: int64
    ----
    x: [[1,2,3]]
    """

    cdef:
        CResult[shared_ptr[CRecordBatchReader]] c_res_reader
        shared_ptr[CRecordBatchReader] c_reader
        RecordBatchReader reader
        c_string c_str_plan
        shared_ptr[CBuffer] c_buf_plan
        function[CNamedTableProvider] c_named_table_provider
        CConversionOptions c_conversion_options
        c_bool c_use_threads

    c_use_threads = use_threads
    if isinstance(plan, bytes):
        c_buf_plan = pyarrow_unwrap_buffer(py_buffer(plan))
    elif isinstance(plan, Buffer):
        c_buf_plan = pyarrow_unwrap_buffer(plan)
    else:
        raise TypeError(
            f"Expected 'pyarrow.Buffer' or bytes, got '{type(plan)}'")

    if table_provider is not None:
        named_table_args = {
            "provider": table_provider
        }
        c_conversion_options.named_table_provider = BindFunction[CNamedTableProvider](
            &_create_named_table_provider, named_table_args)

    with nogil:
        c_res_reader = ExecuteSerializedPlan(
            deref(c_buf_plan), default_extension_id_registry(),
            GetFunctionRegistry(), c_conversion_options, c_use_threads)

    c_reader = GetResultValue(c_res_reader)

    reader = RecordBatchReader.__new__(RecordBatchReader)
    reader.reader = c_reader
    return reader


def _parse_json_plan(plan):
    """
    Parse a JSON plan into equivalent serialized Protobuf.

    Parameters
    ----------
    plan: bytes
        Substrait plan in JSON.

    Returns
    -------
    Buffer
        A buffer containing the serialized Protobuf plan.
    """

    cdef:
        CResult[shared_ptr[CBuffer]] c_res_buffer
        c_string c_str_plan
        shared_ptr[CBuffer] c_buf_plan

    c_str_plan = plan
    c_res_buffer = SerializeJsonPlan(c_str_plan)
    with nogil:
        c_buf_plan = GetResultValue(c_res_buffer)
    return pyarrow_wrap_buffer(c_buf_plan)


def get_supported_functions():
    """
    Get a list of Substrait functions that the underlying
    engine currently supports.

    Returns
    -------
    list[str]
        A list of function ids encoded as '{uri}#{name}'
    """

    cdef:
        ExtensionIdRegistry* c_id_registry
        std_vector[c_string] c_ids

    c_id_registry = default_extension_id_registry()
    c_ids = c_id_registry.GetSupportedSubstraitFunctions()

    functions_list = []
    for c_id in c_ids:
        functions_list.append(frombytes(c_id))
    return functions_list

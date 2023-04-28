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

import os
import pathlib

import pytest

import pyarrow as pa
from pyarrow.lib import tobytes
from pyarrow.lib import ArrowInvalid

try:
    import pyarrow.substrait as substrait
except ImportError:
    substrait = None

# Marks all of the tests in this module
# Ignore these with pytest ... -m 'not substrait'
pytestmark = [pytest.mark.dataset, pytest.mark.substrait]


def _write_dummy_data_to_disk(tmpdir, file_name, table):
    path = os.path.join(str(tmpdir), file_name)
    with pa.ipc.RecordBatchFileWriter(path, schema=table.schema) as writer:
        writer.write_table(table)
    return path


@pytest.mark.parametrize("use_threads", [True, False])
def test_run_serialized_query(tmpdir, use_threads):
    substrait_query = """
    {
        "version": { "major": 9999 },
        "relations": [
        {"rel": {
            "read": {
            "base_schema": {
                "struct": {
                "types": [
                            {"i64": {}}
                        ]
                },
                "names": [
                        "foo"
                        ]
            },
            "local_files": {
                "items": [
                {
                    "uri_file": "FILENAME_PLACEHOLDER",
                    "arrow": {}
                }
                ]
            }
            }
        }}
        ]
    }
    """

    file_name = "read_data.arrow"
    table = pa.table([[1, 2, 3, 4, 5]], names=['foo'])
    path = _write_dummy_data_to_disk(tmpdir, file_name, table)
    query = tobytes(substrait_query.replace(
        "FILENAME_PLACEHOLDER", pathlib.Path(path).as_uri()))

    buf = pa._substrait._parse_json_plan(query)

    reader = substrait.run_query(buf, use_threads=use_threads)
    res_tb = reader.read_all()

    assert table.select(["foo"]) == res_tb.select(["foo"])


@pytest.mark.parametrize("query", (pa.py_buffer(b'buffer'), b"bytes", 1))
def test_run_query_input_types(tmpdir, query):

    # Passing unsupported type, like int, will not segfault.
    if not isinstance(query, (pa.Buffer, bytes)):
        msg = f"Expected 'pyarrow.Buffer' or bytes, got '{type(query)}'"
        with pytest.raises(TypeError, match=msg):
            substrait.run_query(query)
        return

    # Otherwise error for invalid query
    msg = "ParseFromZeroCopyStream failed for substrait.Plan"
    with pytest.raises(OSError, match=msg):
        substrait.run_query(query)


def test_invalid_plan():
    query = """
    {
        "relations": [
        ]
    }
    """
    buf = pa._substrait._parse_json_plan(tobytes(query))
    exec_message = "No RelRoot in plan"
    with pytest.raises(ArrowInvalid, match=exec_message):
        substrait.run_query(buf)


@pytest.mark.parametrize("use_threads", [True, False])
def test_binary_conversion_with_json_options(tmpdir, use_threads):
    substrait_query = """
    {
        "version": { "major": 9999 },
        "relations": [
        {"rel": {
            "read": {
            "base_schema": {
                "struct": {
                "types": [
                            {"i64": {}}
                        ]
                },
                "names": [
                        "bar"
                        ]
            },
            "local_files": {
                "items": [
                {
                    "uri_file": "FILENAME_PLACEHOLDER",
                    "arrow": {},
                    "metadata" : {
                      "created_by" : {},
                    }
                }
                ]
            }
            }
        }}
        ]
    }
    """

    file_name = "binary_json_data.arrow"
    table = pa.table([[1, 2, 3, 4, 5]], names=['bar'])
    path = _write_dummy_data_to_disk(tmpdir, file_name, table)
    query = tobytes(substrait_query.replace(
        "FILENAME_PLACEHOLDER", pathlib.Path(path).as_uri()))
    buf = pa._substrait._parse_json_plan(tobytes(query))

    reader = substrait.run_query(buf, use_threads=use_threads)
    res_tb = reader.read_all()

    assert table.select(["bar"]) == res_tb.select(["bar"])


# Substrait has not finalized what the URI should be for standard functions
# In the meantime, lets just check the suffix
def has_function(fns, ext_file, fn_name):
    suffix = f'{ext_file}#{fn_name}'
    for fn in fns:
        if fn.endswith(suffix):
            return True
    return False


def test_get_supported_functions():
    supported_functions = pa._substrait.get_supported_functions()
    # It probably doesn't make sense to exhaustively verfiy this list but
    # we can check a sample aggregate and a sample non-aggregate entry
    assert has_function(supported_functions,
                        'functions_arithmetic.yaml', 'add')
    assert has_function(supported_functions,
                        'functions_arithmetic.yaml', 'sum')


@pytest.mark.parametrize("use_threads", [True, False])
def test_named_table(use_threads):
    test_table_1 = pa.Table.from_pydict({"x": [1, 2, 3]})
    test_table_2 = pa.Table.from_pydict({"x": [4, 5, 6]})

    def table_provider(names):
        if not names:
            raise Exception("No names provided")
        elif names[0] == "t1":
            return test_table_1
        elif names[1] == "t2":
            return test_table_2
        else:
            raise Exception("Unrecognized table name")

    substrait_query = """
    {
        "version": { "major": 9999 },
        "relations": [
        {"rel": {
            "read": {
            "base_schema": {
                "struct": {
                "types": [
                            {"i64": {}}
                        ]
                },
                "names": [
                        "x"
                        ]
            },
            "namedTable": {
                    "names": ["t1"]
            }
            }
        }}
        ]
    }
    """

    buf = pa._substrait._parse_json_plan(tobytes(substrait_query))
    reader = pa.substrait.run_query(
        buf, table_provider=table_provider, use_threads=use_threads)
    res_tb = reader.read_all()
    assert res_tb == test_table_1


def test_named_table_invalid_table_name():
    test_table_1 = pa.Table.from_pydict({"x": [1, 2, 3]})

    def table_provider(names):
        if not names:
            raise Exception("No names provided")
        elif names[0] == "t1":
            return test_table_1
        else:
            raise Exception("Unrecognized table name")

    substrait_query = """
    {
        "version": { "major": 9999 },
        "relations": [
        {"rel": {
            "read": {
            "base_schema": {
                "struct": {
                "types": [
                            {"i64": {}}
                        ]
                },
                "names": [
                        "x"
                        ]
            },
            "namedTable": {
                    "names": ["t3"]
            }
            }
        }}
        ]
    }
    """

    buf = pa._substrait._parse_json_plan(tobytes(substrait_query))
    exec_message = "Invalid NamedTable Source"
    with pytest.raises(ArrowInvalid, match=exec_message):
        substrait.run_query(buf, table_provider=table_provider)


def test_named_table_empty_names():
    test_table_1 = pa.Table.from_pydict({"x": [1, 2, 3]})

    def table_provider(names):
        if not names:
            raise Exception("No names provided")
        elif names[0] == "t1":
            return test_table_1
        else:
            raise Exception("Unrecognized table name")

    substrait_query = """
    {
        "version": { "major": 9999 },
        "relations": [
        {"rel": {
            "read": {
            "base_schema": {
                "struct": {
                "types": [
                            {"i64": {}}
                        ]
                },
                "names": [
                        "x"
                        ]
            },
            "namedTable": {
                    "names": []
            }
            }
        }}
        ]
    }
    """
    query = tobytes(substrait_query)
    buf = pa._substrait._parse_json_plan(tobytes(query))
    exec_message = "names for NamedTable not provided"
    with pytest.raises(ArrowInvalid, match=exec_message):
        substrait.run_query(buf, table_provider=table_provider)

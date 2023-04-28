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

from collections import OrderedDict
from decimal import Decimal
import io
import itertools
import json
import pickle
import string
import unittest

import numpy as np
import pytest

import pyarrow as pa
from pyarrow.json import read_json, ReadOptions, ParseOptions


def generate_col_names():
    # 'a', 'b'... 'z', then 'aa', 'ab'...
    letters = string.ascii_lowercase
    yield from letters
    for first in letters:
        for second in letters:
            yield first + second


def make_random_json(num_cols=2, num_rows=10, linesep='\r\n'):
    arr = np.random.RandomState(42).randint(0, 1000, size=(num_cols, num_rows))
    col_names = list(itertools.islice(generate_col_names(), num_cols))
    lines = []
    for row in arr.T:
        json_obj = OrderedDict([(k, int(v)) for (k, v) in zip(col_names, row)])
        lines.append(json.dumps(json_obj))
    data = linesep.join(lines).encode()
    columns = [pa.array(col, type=pa.int64()) for col in arr]
    expected = pa.Table.from_arrays(columns, col_names)
    return data, expected


def check_options_class_pickling(cls, **attr_values):
    opts = cls(**attr_values)
    new_opts = pickle.loads(pickle.dumps(opts,
                                         protocol=pickle.HIGHEST_PROTOCOL))
    for name, value in attr_values.items():
        assert getattr(new_opts, name) == value


def test_read_options():
    cls = ReadOptions
    opts = cls()

    assert opts.block_size > 0
    opts.block_size = 12345
    assert opts.block_size == 12345

    assert opts.use_threads is True
    opts.use_threads = False
    assert opts.use_threads is False

    opts = cls(block_size=1234, use_threads=False)
    assert opts.block_size == 1234
    assert opts.use_threads is False

    check_options_class_pickling(cls, block_size=1234,
                                 use_threads=False)


def test_parse_options():
    cls = ParseOptions
    opts = cls()
    assert opts.newlines_in_values is False
    assert opts.explicit_schema is None

    opts.newlines_in_values = True
    assert opts.newlines_in_values is True

    schema = pa.schema([pa.field('foo', pa.int32())])
    opts.explicit_schema = schema
    assert opts.explicit_schema == schema

    assert opts.unexpected_field_behavior == "infer"
    for value in ["ignore", "error", "infer"]:
        opts.unexpected_field_behavior = value
        assert opts.unexpected_field_behavior == value

    with pytest.raises(ValueError):
        opts.unexpected_field_behavior = "invalid-value"

    check_options_class_pickling(cls, explicit_schema=schema,
                                 newlines_in_values=False,
                                 unexpected_field_behavior="ignore")


class BaseTestJSONRead:

    def read_bytes(self, b, **kwargs):
        return self.read_json(pa.py_buffer(b), **kwargs)

    def check_names(self, table, names):
        assert table.num_columns == len(names)
        assert [c.name for c in table.columns] == names

    def test_file_object(self):
        data = b'{"a": 1, "b": 2}\n'
        expected_data = {'a': [1], 'b': [2]}
        bio = io.BytesIO(data)
        table = self.read_json(bio)
        assert table.to_pydict() == expected_data
        # Text files not allowed
        sio = io.StringIO(data.decode())
        with pytest.raises(TypeError):
            self.read_json(sio)

    def test_block_sizes(self):
        rows = b'{"a": 1}\n{"a": 2}\n{"a": 3}'
        read_options = ReadOptions()
        parse_options = ParseOptions()

        for data in [rows, rows + b'\n']:
            for newlines_in_values in [False, True]:
                parse_options.newlines_in_values = newlines_in_values
                read_options.block_size = 4
                with pytest.raises(ValueError,
                                   match="try to increase block size"):
                    self.read_bytes(data, read_options=read_options,
                                    parse_options=parse_options)

                # Validate reader behavior with various block sizes.
                # There used to be bugs in this area.
                for block_size in range(9, 20):
                    read_options.block_size = block_size
                    table = self.read_bytes(data, read_options=read_options,
                                            parse_options=parse_options)
                    assert table.to_pydict() == {'a': [1, 2, 3]}

    def test_no_newline_at_end(self):
        rows = b'{"a": 1,"b": 2, "c": 3}\n{"a": 4,"b": 5, "c": 6}'
        table = self.read_bytes(rows)
        assert table.to_pydict() == {
            'a': [1, 4],
            'b': [2, 5],
            'c': [3, 6],
        }

    def test_simple_ints(self):
        # Infer integer columns
        rows = b'{"a": 1,"b": 2, "c": 3}\n{"a": 4,"b": 5, "c": 6}\n'
        table = self.read_bytes(rows)
        schema = pa.schema([('a', pa.int64()),
                            ('b', pa.int64()),
                            ('c', pa.int64())])
        assert table.schema == schema
        assert table.to_pydict() == {
            'a': [1, 4],
            'b': [2, 5],
            'c': [3, 6],
        }

    def test_simple_varied(self):
        # Infer various kinds of data
        rows = (b'{"a": 1,"b": 2, "c": "3", "d": false}\n'
                b'{"a": 4.0, "b": -5, "c": "foo", "d": true}\n')
        table = self.read_bytes(rows)
        schema = pa.schema([('a', pa.float64()),
                            ('b', pa.int64()),
                            ('c', pa.string()),
                            ('d', pa.bool_())])
        assert table.schema == schema
        assert table.to_pydict() == {
            'a': [1.0, 4.0],
            'b': [2, -5],
            'c': ["3", "foo"],
            'd': [False, True],
        }

    def test_simple_nulls(self):
        # Infer various kinds of data, with nulls
        rows = (b'{"a": 1, "b": 2, "c": null, "d": null, "e": null}\n'
                b'{"a": null, "b": -5, "c": "foo", "d": null, "e": true}\n'
                b'{"a": 4.5, "b": null, "c": "nan", "d": null,"e": false}\n')
        table = self.read_bytes(rows)
        schema = pa.schema([('a', pa.float64()),
                            ('b', pa.int64()),
                            ('c', pa.string()),
                            ('d', pa.null()),
                            ('e', pa.bool_())])
        assert table.schema == schema
        assert table.to_pydict() == {
            'a': [1.0, None, 4.5],
            'b': [2, -5, None],
            'c': [None, "foo", "nan"],
            'd': [None, None, None],
            'e': [None, True, False],
        }

    def test_empty_lists(self):
        # ARROW-10955: Infer list(null)
        rows = b'{"a": []}'
        table = self.read_bytes(rows)
        schema = pa.schema([('a', pa.list_(pa.null()))])
        assert table.schema == schema
        assert table.to_pydict() == {'a': [[]]}

    def test_empty_rows(self):
        rows = b'{}\n{}\n'
        table = self.read_bytes(rows)
        schema = pa.schema([])
        assert table.schema == schema
        assert table.num_columns == 0
        assert table.num_rows == 2

    def test_reconcile_accross_blocks(self):
        # ARROW-12065: reconciling inferred types across blocks
        first_row = b'{                               }\n'
        read_options = ReadOptions(block_size=len(first_row))
        for next_rows, expected_pylist in [
            (b'{"a": 0}', [None, 0]),
            (b'{"a": []}', [None, []]),
            (b'{"a": []}\n{"a": [[1]]}', [None, [], [[1]]]),
            (b'{"a": {}}', [None, {}]),
            (b'{"a": {}}\n{"a": {"b": {"c": 1}}}',
             [None, {"b": None}, {"b": {"c": 1}}]),
        ]:
            table = self.read_bytes(first_row + next_rows,
                                    read_options=read_options)
            expected = {"a": expected_pylist}
            assert table.to_pydict() == expected
            # Check that the issue was exercised
            assert table.column("a").num_chunks > 1

    def test_explicit_schema_decimal(self):
        rows = (b'{"a": 1}\n'
                b'{"a": 1.45}\n'
                b'{"a": -23.456}\n'
                b'{}\n')
        expected = {
            'a': [Decimal("1"), Decimal("1.45"), Decimal("-23.456"), None],
        }
        for type_factory in (pa.decimal128, pa.decimal256):
            schema = pa.schema([('a', type_factory(9, 4))])
            opts = ParseOptions(explicit_schema=schema)
            table = self.read_bytes(rows, parse_options=opts)
            assert table.schema == schema
            assert table.to_pydict() == expected

    def test_explicit_schema_with_unexpected_behaviour(self):
        # infer by default
        rows = (b'{"foo": "bar", "num": 0}\n'
                b'{"foo": "baz", "num": 1}\n')
        schema = pa.schema([
            ('foo', pa.binary())
        ])

        opts = ParseOptions(explicit_schema=schema)
        table = self.read_bytes(rows, parse_options=opts)
        assert table.schema == pa.schema([
            ('foo', pa.binary()),
            ('num', pa.int64())
        ])
        assert table.to_pydict() == {
            'foo': [b'bar', b'baz'],
            'num': [0, 1],
        }

        # ignore the unexpected fields
        opts = ParseOptions(explicit_schema=schema,
                            unexpected_field_behavior="ignore")
        table = self.read_bytes(rows, parse_options=opts)
        assert table.schema == pa.schema([
            ('foo', pa.binary()),
        ])
        assert table.to_pydict() == {
            'foo': [b'bar', b'baz'],
        }

        # raise error
        opts = ParseOptions(explicit_schema=schema,
                            unexpected_field_behavior="error")
        with pytest.raises(pa.ArrowInvalid,
                           match="JSON parse error: unexpected field"):
            self.read_bytes(rows, parse_options=opts)

    def test_small_random_json(self):
        data, expected = make_random_json(num_cols=2, num_rows=10)
        table = self.read_bytes(data)
        assert table.schema == expected.schema
        assert table.equals(expected)
        assert table.to_pydict() == expected.to_pydict()

    def test_stress_block_sizes(self):
        # Test a number of small block sizes to stress block stitching
        data_base, expected = make_random_json(num_cols=2, num_rows=100)
        read_options = ReadOptions()
        parse_options = ParseOptions()

        for data in [data_base, data_base.rstrip(b'\r\n')]:
            for newlines_in_values in [False, True]:
                parse_options.newlines_in_values = newlines_in_values
                for block_size in [22, 23, 37]:
                    read_options.block_size = block_size
                    table = self.read_bytes(data, read_options=read_options,
                                            parse_options=parse_options)
                    assert table.schema == expected.schema
                    if not table.equals(expected):
                        # Better error output
                        assert table.to_pydict() == expected.to_pydict()


class TestSerialJSONRead(BaseTestJSONRead, unittest.TestCase):

    def read_json(self, *args, **kwargs):
        read_options = kwargs.setdefault('read_options', ReadOptions())
        read_options.use_threads = False
        table = read_json(*args, **kwargs)
        table.validate(full=True)
        return table


class TestParallelJSONRead(BaseTestJSONRead, unittest.TestCase):

    def read_json(self, *args, **kwargs):
        read_options = kwargs.setdefault('read_options', ReadOptions())
        read_options.use_threads = True
        table = read_json(*args, **kwargs)
        table.validate(full=True)
        return table

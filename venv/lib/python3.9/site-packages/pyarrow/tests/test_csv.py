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

import abc
import bz2
from datetime import date, datetime
from decimal import Decimal
import gc
import gzip
import io
import itertools
import os
import pickle
import select
import shutil
import signal
import string
import tempfile
import threading
import time
import unittest
import weakref

import pytest

import numpy as np

import pyarrow as pa
from pyarrow.csv import (
    open_csv, read_csv, ReadOptions, ParseOptions, ConvertOptions, ISO8601,
    write_csv, WriteOptions, CSVWriter, InvalidRow)
from pyarrow.tests import util


def generate_col_names():
    # 'a', 'b'... 'z', then 'aa', 'ab'...
    letters = string.ascii_lowercase
    yield from letters
    for first in letters:
        for second in letters:
            yield first + second


def make_random_csv(num_cols=2, num_rows=10, linesep='\r\n', write_names=True):
    arr = np.random.RandomState(42).randint(0, 1000, size=(num_cols, num_rows))
    csv = io.StringIO()
    col_names = list(itertools.islice(generate_col_names(), num_cols))
    if write_names:
        csv.write(",".join(col_names))
        csv.write(linesep)
    for row in arr.T:
        csv.write(",".join(map(str, row)))
        csv.write(linesep)
    csv = csv.getvalue().encode()
    columns = [pa.array(a, type=pa.int64()) for a in arr]
    expected = pa.Table.from_arrays(columns, col_names)
    return csv, expected


def make_empty_csv(column_names):
    csv = io.StringIO()
    csv.write(",".join(column_names))
    csv.write("\n")
    return csv.getvalue().encode()


def check_options_class(cls, **attr_values):
    """
    Check setting and getting attributes of an *Options class.
    """
    opts = cls()

    for name, values in attr_values.items():
        assert getattr(opts, name) == values[0], \
            "incorrect default value for " + name
        for v in values:
            setattr(opts, name, v)
            assert getattr(opts, name) == v, "failed setting value"

    with pytest.raises(AttributeError):
        opts.zzz_non_existent = True

    # Check constructor named arguments
    non_defaults = {name: values[1] for name, values in attr_values.items()}
    opts = cls(**non_defaults)
    for name, value in non_defaults.items():
        assert getattr(opts, name) == value


# The various options classes need to be picklable for dataset
def check_options_class_pickling(cls, **attr_values):
    opts = cls(**attr_values)
    new_opts = pickle.loads(pickle.dumps(opts,
                                         protocol=pickle.HIGHEST_PROTOCOL))
    for name, value in attr_values.items():
        assert getattr(new_opts, name) == value


class InvalidRowHandler:
    def __init__(self, result):
        self.result = result
        self.rows = []

    def __call__(self, row):
        self.rows.append(row)
        return self.result

    def __eq__(self, other):
        return (isinstance(other, InvalidRowHandler) and
                other.result == self.result)

    def __ne__(self, other):
        return (not isinstance(other, InvalidRowHandler) or
                other.result != self.result)


def test_read_options():
    cls = ReadOptions
    opts = cls()

    check_options_class(cls, use_threads=[True, False],
                        skip_rows=[0, 3],
                        column_names=[[], ["ab", "cd"]],
                        autogenerate_column_names=[False, True],
                        encoding=['utf8', 'utf16'],
                        skip_rows_after_names=[0, 27])

    check_options_class_pickling(cls, use_threads=True,
                                 skip_rows=3,
                                 column_names=["ab", "cd"],
                                 autogenerate_column_names=False,
                                 encoding='utf16',
                                 skip_rows_after_names=27)

    assert opts.block_size > 0
    opts.block_size = 12345
    assert opts.block_size == 12345

    opts = cls(block_size=1234)
    assert opts.block_size == 1234

    opts.validate()

    match = "ReadOptions: block_size must be at least 1: 0"
    with pytest.raises(pa.ArrowInvalid, match=match):
        opts = cls()
        opts.block_size = 0
        opts.validate()

    match = "ReadOptions: skip_rows cannot be negative: -1"
    with pytest.raises(pa.ArrowInvalid, match=match):
        opts = cls()
        opts.skip_rows = -1
        opts.validate()

    match = "ReadOptions: skip_rows_after_names cannot be negative: -1"
    with pytest.raises(pa.ArrowInvalid, match=match):
        opts = cls()
        opts.skip_rows_after_names = -1
        opts.validate()

    match = "ReadOptions: autogenerate_column_names cannot be true when" \
            " column_names are provided"
    with pytest.raises(pa.ArrowInvalid, match=match):
        opts = cls()
        opts.autogenerate_column_names = True
        opts.column_names = ('a', 'b')
        opts.validate()


def test_parse_options():
    cls = ParseOptions
    skip_handler = InvalidRowHandler('skip')

    check_options_class(cls, delimiter=[',', 'x'],
                        escape_char=[False, 'y'],
                        quote_char=['"', 'z', False],
                        double_quote=[True, False],
                        newlines_in_values=[False, True],
                        ignore_empty_lines=[True, False],
                        invalid_row_handler=[None, skip_handler])

    check_options_class_pickling(cls, delimiter='x',
                                 escape_char='y',
                                 quote_char=False,
                                 double_quote=False,
                                 newlines_in_values=True,
                                 ignore_empty_lines=False,
                                 invalid_row_handler=skip_handler)

    cls().validate()
    opts = cls()
    opts.delimiter = "\t"
    opts.validate()

    match = "ParseOptions: delimiter cannot be \\\\r or \\\\n"
    with pytest.raises(pa.ArrowInvalid, match=match):
        opts = cls()
        opts.delimiter = "\n"
        opts.validate()

    with pytest.raises(pa.ArrowInvalid, match=match):
        opts = cls()
        opts.delimiter = "\r"
        opts.validate()

    match = "ParseOptions: quote_char cannot be \\\\r or \\\\n"
    with pytest.raises(pa.ArrowInvalid, match=match):
        opts = cls()
        opts.quote_char = "\n"
        opts.validate()

    with pytest.raises(pa.ArrowInvalid, match=match):
        opts = cls()
        opts.quote_char = "\r"
        opts.validate()

    match = "ParseOptions: escape_char cannot be \\\\r or \\\\n"
    with pytest.raises(pa.ArrowInvalid, match=match):
        opts = cls()
        opts.escape_char = "\n"
        opts.validate()

    with pytest.raises(pa.ArrowInvalid, match=match):
        opts = cls()
        opts.escape_char = "\r"
        opts.validate()


def test_convert_options():
    cls = ConvertOptions
    opts = cls()

    check_options_class(
        cls, check_utf8=[True, False],
        strings_can_be_null=[False, True],
        quoted_strings_can_be_null=[True, False],
        decimal_point=['.', ','],
        include_columns=[[], ['def', 'abc']],
        include_missing_columns=[False, True],
        auto_dict_encode=[False, True],
        timestamp_parsers=[[], [ISO8601, '%y-%m']])

    check_options_class_pickling(
        cls, check_utf8=False,
        strings_can_be_null=True,
        quoted_strings_can_be_null=False,
        decimal_point=',',
        include_columns=['def', 'abc'],
        include_missing_columns=False,
        auto_dict_encode=True,
        timestamp_parsers=[ISO8601, '%y-%m'])

    with pytest.raises(ValueError):
        opts.decimal_point = '..'

    assert opts.auto_dict_max_cardinality > 0
    opts.auto_dict_max_cardinality = 99999
    assert opts.auto_dict_max_cardinality == 99999

    assert opts.column_types == {}
    # Pass column_types as mapping
    opts.column_types = {'b': pa.int16(), 'c': pa.float32()}
    assert opts.column_types == {'b': pa.int16(), 'c': pa.float32()}
    opts.column_types = {'v': 'int16', 'w': 'null'}
    assert opts.column_types == {'v': pa.int16(), 'w': pa.null()}
    # Pass column_types as schema
    schema = pa.schema([('a', pa.int32()), ('b', pa.string())])
    opts.column_types = schema
    assert opts.column_types == {'a': pa.int32(), 'b': pa.string()}
    # Pass column_types as sequence
    opts.column_types = [('x', pa.binary())]
    assert opts.column_types == {'x': pa.binary()}

    with pytest.raises(TypeError, match='DataType expected'):
        opts.column_types = {'a': None}
    with pytest.raises(TypeError):
        opts.column_types = 0

    assert isinstance(opts.null_values, list)
    assert '' in opts.null_values
    assert 'N/A' in opts.null_values
    opts.null_values = ['xxx', 'yyy']
    assert opts.null_values == ['xxx', 'yyy']

    assert isinstance(opts.true_values, list)
    opts.true_values = ['xxx', 'yyy']
    assert opts.true_values == ['xxx', 'yyy']

    assert isinstance(opts.false_values, list)
    opts.false_values = ['xxx', 'yyy']
    assert opts.false_values == ['xxx', 'yyy']

    assert opts.timestamp_parsers == []
    opts.timestamp_parsers = [ISO8601]
    assert opts.timestamp_parsers == [ISO8601]

    opts = cls(column_types={'a': pa.null()},
               null_values=['N', 'nn'], true_values=['T', 'tt'],
               false_values=['F', 'ff'], auto_dict_max_cardinality=999,
               timestamp_parsers=[ISO8601, '%Y-%m-%d'])
    assert opts.column_types == {'a': pa.null()}
    assert opts.null_values == ['N', 'nn']
    assert opts.false_values == ['F', 'ff']
    assert opts.true_values == ['T', 'tt']
    assert opts.auto_dict_max_cardinality == 999
    assert opts.timestamp_parsers == [ISO8601, '%Y-%m-%d']


def test_write_options():
    cls = WriteOptions
    opts = cls()

    check_options_class(
        cls, include_header=[True, False], delimiter=[',', '\t', '|'],
        quoting_style=['needed', 'none', 'all_valid'])

    assert opts.batch_size > 0
    opts.batch_size = 12345
    assert opts.batch_size == 12345

    opts = cls(batch_size=9876)
    assert opts.batch_size == 9876

    opts.validate()

    match = "WriteOptions: batch_size must be at least 1: 0"
    with pytest.raises(pa.ArrowInvalid, match=match):
        opts = cls()
        opts.batch_size = 0
        opts.validate()


class BaseTestCSV(abc.ABC):
    """Common tests which are shared by streaming and non streaming readers"""

    @abc.abstractmethod
    def read_bytes(self, b, **kwargs):
        """
        :param b: bytes to be parsed
        :param kwargs: arguments passed on to open the csv file
        :return: b parsed as a single RecordBatch
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def use_threads(self):
        """Whether this test is multi-threaded"""
        raise NotImplementedError

    @staticmethod
    def check_names(table, names):
        assert table.num_columns == len(names)
        assert table.column_names == names

    def test_header_skip_rows(self):
        rows = b"ab,cd\nef,gh\nij,kl\nmn,op\n"

        opts = ReadOptions()
        opts.skip_rows = 1
        table = self.read_bytes(rows, read_options=opts)
        self.check_names(table, ["ef", "gh"])
        assert table.to_pydict() == {
            "ef": ["ij", "mn"],
            "gh": ["kl", "op"],
        }

        opts.skip_rows = 3
        table = self.read_bytes(rows, read_options=opts)
        self.check_names(table, ["mn", "op"])
        assert table.to_pydict() == {
            "mn": [],
            "op": [],
        }

        opts.skip_rows = 4
        with pytest.raises(pa.ArrowInvalid):
            # Not enough rows
            table = self.read_bytes(rows, read_options=opts)

        # Can skip rows with a different number of columns
        rows = b"abcd\n,,,,,\nij,kl\nmn,op\n"
        opts.skip_rows = 2
        table = self.read_bytes(rows, read_options=opts)
        self.check_names(table, ["ij", "kl"])
        assert table.to_pydict() == {
            "ij": ["mn"],
            "kl": ["op"],
        }

        # Can skip all rows exactly when columns are given
        opts.skip_rows = 4
        opts.column_names = ['ij', 'kl']
        table = self.read_bytes(rows, read_options=opts)
        self.check_names(table, ["ij", "kl"])
        assert table.to_pydict() == {
            "ij": [],
            "kl": [],
        }

    def test_skip_rows_after_names(self):
        rows = b"ab,cd\nef,gh\nij,kl\nmn,op\n"

        opts = ReadOptions()
        opts.skip_rows_after_names = 1
        table = self.read_bytes(rows, read_options=opts)
        self.check_names(table, ["ab", "cd"])
        assert table.to_pydict() == {
            "ab": ["ij", "mn"],
            "cd": ["kl", "op"],
        }

        # Can skip exact number of rows
        opts.skip_rows_after_names = 3
        table = self.read_bytes(rows, read_options=opts)
        self.check_names(table, ["ab", "cd"])
        assert table.to_pydict() == {
            "ab": [],
            "cd": [],
        }

        # Can skip beyond all rows
        opts.skip_rows_after_names = 4
        table = self.read_bytes(rows, read_options=opts)
        self.check_names(table, ["ab", "cd"])
        assert table.to_pydict() == {
            "ab": [],
            "cd": [],
        }

        # Can skip rows with a different number of columns
        rows = b"abcd\n,,,,,\nij,kl\nmn,op\n"
        opts.skip_rows_after_names = 2
        opts.column_names = ["f0", "f1"]
        table = self.read_bytes(rows, read_options=opts)
        self.check_names(table, ["f0", "f1"])
        assert table.to_pydict() == {
            "f0": ["ij", "mn"],
            "f1": ["kl", "op"],
        }
        opts = ReadOptions()

        # Can skip rows with new lines in the value
        rows = b'ab,cd\n"e\nf","g\n\nh"\n"ij","k\nl"\nmn,op'
        opts.skip_rows_after_names = 2
        parse_opts = ParseOptions()
        parse_opts.newlines_in_values = True
        table = self.read_bytes(rows, read_options=opts,
                                parse_options=parse_opts)
        self.check_names(table, ["ab", "cd"])
        assert table.to_pydict() == {
            "ab": ["mn"],
            "cd": ["op"],
        }

        # Can skip rows when block ends in middle of quoted value
        opts.skip_rows_after_names = 2
        opts.block_size = 26
        table = self.read_bytes(rows, read_options=opts,
                                parse_options=parse_opts)
        self.check_names(table, ["ab", "cd"])
        assert table.to_pydict() == {
            "ab": ["mn"],
            "cd": ["op"],
        }
        opts = ReadOptions()

        # Can skip rows that are beyond the first block without lexer
        rows, expected = make_random_csv(num_cols=5, num_rows=1000)
        opts.skip_rows_after_names = 900
        opts.block_size = len(rows) / 11
        table = self.read_bytes(rows, read_options=opts)
        assert table.schema == expected.schema
        assert table.num_rows == 100
        table_dict = table.to_pydict()
        for name, values in expected.to_pydict().items():
            assert values[900:] == table_dict[name]

        # Can skip rows that are beyond the first block with lexer
        table = self.read_bytes(rows, read_options=opts,
                                parse_options=parse_opts)
        assert table.schema == expected.schema
        assert table.num_rows == 100
        table_dict = table.to_pydict()
        for name, values in expected.to_pydict().items():
            assert values[900:] == table_dict[name]

        # Skip rows and skip rows after names
        rows, expected = make_random_csv(num_cols=5, num_rows=200,
                                         write_names=False)
        opts = ReadOptions()
        opts.skip_rows = 37
        opts.skip_rows_after_names = 41
        opts.column_names = expected.schema.names
        table = self.read_bytes(rows, read_options=opts,
                                parse_options=parse_opts)
        assert table.schema == expected.schema
        assert (table.num_rows ==
                expected.num_rows - opts.skip_rows -
                opts.skip_rows_after_names)
        table_dict = table.to_pydict()
        for name, values in expected.to_pydict().items():
            assert (values[opts.skip_rows + opts.skip_rows_after_names:] ==
                    table_dict[name])

    def test_row_number_offset_in_errors(self):
        # Row numbers are only correctly counted in serial reads
        def format_msg(msg_format, row, *args):
            if self.use_threads:
                row_info = ""
            else:
                row_info = "Row #{}: ".format(row)
            return msg_format.format(row_info, *args)

        csv, _ = make_random_csv(4, 100, write_names=True)

        read_options = ReadOptions()
        read_options.block_size = len(csv) / 3
        convert_options = ConvertOptions()
        convert_options.column_types = {"a": pa.int32()}

        # Test without skip_rows and column names in the csv
        csv_bad_columns = csv + b"1,2\r\n"
        message_columns = format_msg("{}Expected 4 columns, got 2", 102)
        with pytest.raises(pa.ArrowInvalid, match=message_columns):
            self.read_bytes(csv_bad_columns,
                            read_options=read_options,
                            convert_options=convert_options)

        csv_bad_type = csv + b"a,b,c,d\r\n"
        message_value = format_msg(
            "In CSV column #0: {}"
            "CSV conversion error to int32: invalid value 'a'",
            102, csv)
        with pytest.raises(pa.ArrowInvalid, match=message_value):
            self.read_bytes(csv_bad_type,
                            read_options=read_options,
                            convert_options=convert_options)

        long_row = (b"this is a long row" * 15) + b",3\r\n"
        csv_bad_columns_long = csv + long_row
        message_long = format_msg("{}Expected 4 columns, got 2: {} ...", 102,
                                  long_row[0:96].decode("utf-8"))
        with pytest.raises(pa.ArrowInvalid, match=message_long):
            self.read_bytes(csv_bad_columns_long,
                            read_options=read_options,
                            convert_options=convert_options)

        # Test skipping rows after the names
        read_options.skip_rows_after_names = 47

        with pytest.raises(pa.ArrowInvalid, match=message_columns):
            self.read_bytes(csv_bad_columns,
                            read_options=read_options,
                            convert_options=convert_options)

        with pytest.raises(pa.ArrowInvalid, match=message_value):
            self.read_bytes(csv_bad_type,
                            read_options=read_options,
                            convert_options=convert_options)

        with pytest.raises(pa.ArrowInvalid, match=message_long):
            self.read_bytes(csv_bad_columns_long,
                            read_options=read_options,
                            convert_options=convert_options)

        read_options.skip_rows_after_names = 0

        # Test without skip_rows and column names not in the csv
        csv, _ = make_random_csv(4, 100, write_names=False)
        read_options.column_names = ["a", "b", "c", "d"]
        csv_bad_columns = csv + b"1,2\r\n"
        message_columns = format_msg("{}Expected 4 columns, got 2", 101)
        with pytest.raises(pa.ArrowInvalid, match=message_columns):
            self.read_bytes(csv_bad_columns,
                            read_options=read_options,
                            convert_options=convert_options)

        csv_bad_columns_long = csv + long_row
        message_long = format_msg("{}Expected 4 columns, got 2: {} ...", 101,
                                  long_row[0:96].decode("utf-8"))
        with pytest.raises(pa.ArrowInvalid, match=message_long):
            self.read_bytes(csv_bad_columns_long,
                            read_options=read_options,
                            convert_options=convert_options)

        csv_bad_type = csv + b"a,b,c,d\r\n"
        message_value = format_msg(
            "In CSV column #0: {}"
            "CSV conversion error to int32: invalid value 'a'",
            101)
        message_value = message_value.format(len(csv))
        with pytest.raises(pa.ArrowInvalid, match=message_value):
            self.read_bytes(csv_bad_type,
                            read_options=read_options,
                            convert_options=convert_options)

        # Test with skip_rows and column names not in the csv
        read_options.skip_rows = 23
        with pytest.raises(pa.ArrowInvalid, match=message_columns):
            self.read_bytes(csv_bad_columns,
                            read_options=read_options,
                            convert_options=convert_options)

        with pytest.raises(pa.ArrowInvalid, match=message_value):
            self.read_bytes(csv_bad_type,
                            read_options=read_options,
                            convert_options=convert_options)

    def test_invalid_row_handler(self):
        rows = b"a,b\nc\nd,e\nf,g,h\ni,j\n"
        parse_opts = ParseOptions()
        with pytest.raises(
                ValueError,
                match="Expected 2 columns, got 1: c"):
            self.read_bytes(rows, parse_options=parse_opts)

        # Skip requested
        parse_opts.invalid_row_handler = InvalidRowHandler('skip')
        table = self.read_bytes(rows, parse_options=parse_opts)
        assert table.to_pydict() == {
            'a': ["d", "i"],
            'b': ["e", "j"],
        }

        def row_num(x):
            return None if self.use_threads else x
        expected_rows = [
            InvalidRow(2, 1, row_num(2), "c"),
            InvalidRow(2, 3, row_num(4), "f,g,h"),
        ]
        assert parse_opts.invalid_row_handler.rows == expected_rows

        # Error requested
        parse_opts.invalid_row_handler = InvalidRowHandler('error')
        with pytest.raises(
                ValueError,
                match="Expected 2 columns, got 1: c"):
            self.read_bytes(rows, parse_options=parse_opts)
        expected_rows = [InvalidRow(2, 1, row_num(2), "c")]
        assert parse_opts.invalid_row_handler.rows == expected_rows

        # Test ser/de
        parse_opts.invalid_row_handler = InvalidRowHandler('skip')
        parse_opts = pickle.loads(pickle.dumps(parse_opts))

        table = self.read_bytes(rows, parse_options=parse_opts)
        assert table.to_pydict() == {
            'a': ["d", "i"],
            'b': ["e", "j"],
        }


class BaseCSVTableRead(BaseTestCSV):

    def read_csv(self, csv, *args, validate_full=True, **kwargs):
        """
        Reads the CSV file into memory using pyarrow's read_csv
        csv The CSV bytes
        args Positional arguments to be forwarded to pyarrow's read_csv
        validate_full Whether or not to fully validate the resulting table
        kwargs Keyword arguments to be forwarded to pyarrow's read_csv
        """
        assert isinstance(self.use_threads, bool)  # sanity check
        read_options = kwargs.setdefault('read_options', ReadOptions())
        read_options.use_threads = self.use_threads
        table = read_csv(csv, *args, **kwargs)
        table.validate(full=validate_full)
        return table

    def read_bytes(self, b, **kwargs):
        return self.read_csv(pa.py_buffer(b), **kwargs)

    def test_file_object(self):
        data = b"a,b\n1,2\n"
        expected_data = {'a': [1], 'b': [2]}
        bio = io.BytesIO(data)
        table = self.read_csv(bio)
        assert table.to_pydict() == expected_data
        # Text files not allowed
        sio = io.StringIO(data.decode())
        with pytest.raises(TypeError):
            self.read_csv(sio)

    def test_header(self):
        rows = b"abc,def,gh\n"
        table = self.read_bytes(rows)
        assert isinstance(table, pa.Table)
        self.check_names(table, ["abc", "def", "gh"])
        assert table.num_rows == 0

    def test_bom(self):
        rows = b"\xef\xbb\xbfa,b\n1,2\n"
        expected_data = {'a': [1], 'b': [2]}
        table = self.read_bytes(rows)
        assert table.to_pydict() == expected_data

    def test_one_chunk(self):
        # ARROW-7661: lack of newline at end of file should not produce
        # an additional chunk.
        rows = [b"a,b", b"1,2", b"3,4", b"56,78"]
        for line_ending in [b'\n', b'\r', b'\r\n']:
            for file_ending in [b'', line_ending]:
                data = line_ending.join(rows) + file_ending
                table = self.read_bytes(data)
                assert len(table.to_batches()) == 1
                assert table.to_pydict() == {
                    "a": [1, 3, 56],
                    "b": [2, 4, 78],
                }

    def test_header_column_names(self):
        rows = b"ab,cd\nef,gh\nij,kl\nmn,op\n"

        opts = ReadOptions()
        opts.column_names = ["x", "y"]
        table = self.read_bytes(rows, read_options=opts)
        self.check_names(table, ["x", "y"])
        assert table.to_pydict() == {
            "x": ["ab", "ef", "ij", "mn"],
            "y": ["cd", "gh", "kl", "op"],
        }

        opts.skip_rows = 3
        table = self.read_bytes(rows, read_options=opts)
        self.check_names(table, ["x", "y"])
        assert table.to_pydict() == {
            "x": ["mn"],
            "y": ["op"],
        }

        opts.skip_rows = 4
        table = self.read_bytes(rows, read_options=opts)
        self.check_names(table, ["x", "y"])
        assert table.to_pydict() == {
            "x": [],
            "y": [],
        }

        opts.skip_rows = 5
        with pytest.raises(pa.ArrowInvalid):
            # Not enough rows
            table = self.read_bytes(rows, read_options=opts)

        # Unexpected number of columns
        opts.skip_rows = 0
        opts.column_names = ["x", "y", "z"]
        with pytest.raises(pa.ArrowInvalid,
                           match="Expected 3 columns, got 2"):
            table = self.read_bytes(rows, read_options=opts)

        # Can skip rows with a different number of columns
        rows = b"abcd\n,,,,,\nij,kl\nmn,op\n"
        opts.skip_rows = 2
        opts.column_names = ["x", "y"]
        table = self.read_bytes(rows, read_options=opts)
        self.check_names(table, ["x", "y"])
        assert table.to_pydict() == {
            "x": ["ij", "mn"],
            "y": ["kl", "op"],
        }

    def test_header_autogenerate_column_names(self):
        rows = b"ab,cd\nef,gh\nij,kl\nmn,op\n"

        opts = ReadOptions()
        opts.autogenerate_column_names = True
        table = self.read_bytes(rows, read_options=opts)
        self.check_names(table, ["f0", "f1"])
        assert table.to_pydict() == {
            "f0": ["ab", "ef", "ij", "mn"],
            "f1": ["cd", "gh", "kl", "op"],
        }

        opts.skip_rows = 3
        table = self.read_bytes(rows, read_options=opts)
        self.check_names(table, ["f0", "f1"])
        assert table.to_pydict() == {
            "f0": ["mn"],
            "f1": ["op"],
        }

        # Not enough rows, impossible to infer number of columns
        opts.skip_rows = 4
        with pytest.raises(pa.ArrowInvalid):
            table = self.read_bytes(rows, read_options=opts)

    def test_include_columns(self):
        rows = b"ab,cd\nef,gh\nij,kl\nmn,op\n"

        convert_options = ConvertOptions()
        convert_options.include_columns = ['ab']
        table = self.read_bytes(rows, convert_options=convert_options)
        self.check_names(table, ["ab"])
        assert table.to_pydict() == {
            "ab": ["ef", "ij", "mn"],
        }

        # Order of include_columns is respected, regardless of CSV order
        convert_options.include_columns = ['cd', 'ab']
        table = self.read_bytes(rows, convert_options=convert_options)
        schema = pa.schema([('cd', pa.string()),
                            ('ab', pa.string())])
        assert table.schema == schema
        assert table.to_pydict() == {
            "cd": ["gh", "kl", "op"],
            "ab": ["ef", "ij", "mn"],
        }

        # Include a column not in the CSV file => raises by default
        convert_options.include_columns = ['xx', 'ab', 'yy']
        with pytest.raises(KeyError,
                           match="Column 'xx' in include_columns "
                                 "does not exist in CSV file"):
            self.read_bytes(rows, convert_options=convert_options)

    def test_include_missing_columns(self):
        rows = b"ab,cd\nef,gh\nij,kl\nmn,op\n"

        read_options = ReadOptions()
        convert_options = ConvertOptions()
        convert_options.include_columns = ['xx', 'ab', 'yy']
        convert_options.include_missing_columns = True
        table = self.read_bytes(rows, read_options=read_options,
                                convert_options=convert_options)
        schema = pa.schema([('xx', pa.null()),
                            ('ab', pa.string()),
                            ('yy', pa.null())])
        assert table.schema == schema
        assert table.to_pydict() == {
            "xx": [None, None, None],
            "ab": ["ef", "ij", "mn"],
            "yy": [None, None, None],
        }

        # Combining with `column_names`
        read_options.column_names = ["xx", "yy"]
        convert_options.include_columns = ["yy", "cd"]
        table = self.read_bytes(rows, read_options=read_options,
                                convert_options=convert_options)
        schema = pa.schema([('yy', pa.string()),
                            ('cd', pa.null())])
        assert table.schema == schema
        assert table.to_pydict() == {
            "yy": ["cd", "gh", "kl", "op"],
            "cd": [None, None, None, None],
        }

        # And with `column_types` as well
        convert_options.column_types = {"yy": pa.binary(),
                                        "cd": pa.int32()}
        table = self.read_bytes(rows, read_options=read_options,
                                convert_options=convert_options)
        schema = pa.schema([('yy', pa.binary()),
                            ('cd', pa.int32())])
        assert table.schema == schema
        assert table.to_pydict() == {
            "yy": [b"cd", b"gh", b"kl", b"op"],
            "cd": [None, None, None, None],
        }

    def test_simple_ints(self):
        # Infer integer columns
        rows = b"a,b,c\n1,2,3\n4,5,6\n"
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
        rows = b"a,b,c,d\n1,2,3,0\n4.0,-5,foo,True\n"
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
        rows = (b"a,b,c,d,e,f\n"
                b"1,2,,,3,N/A\n"
                b"nan,-5,foo,,nan,TRUE\n"
                b"4.5,#N/A,nan,,\xff,false\n")
        table = self.read_bytes(rows)
        schema = pa.schema([('a', pa.float64()),
                            ('b', pa.int64()),
                            ('c', pa.string()),
                            ('d', pa.null()),
                            ('e', pa.binary()),
                            ('f', pa.bool_())])
        assert table.schema == schema
        assert table.to_pydict() == {
            'a': [1.0, None, 4.5],
            'b': [2, -5, None],
            'c': ["", "foo", "nan"],
            'd': [None, None, None],
            'e': [b"3", b"nan", b"\xff"],
            'f': [None, True, False],
        }

    def test_decimal_point(self):
        # Infer floats with a custom decimal point
        parse_options = ParseOptions(delimiter=';')
        rows = b"a;b\n1.25;2,5\nNA;-3\n-4;NA"

        table = self.read_bytes(rows, parse_options=parse_options)
        schema = pa.schema([('a', pa.float64()),
                            ('b', pa.string())])
        assert table.schema == schema
        assert table.to_pydict() == {
            'a': [1.25, None, -4.0],
            'b': ["2,5", "-3", "NA"],
        }

        convert_options = ConvertOptions(decimal_point=',')
        table = self.read_bytes(rows, parse_options=parse_options,
                                convert_options=convert_options)
        schema = pa.schema([('a', pa.string()),
                            ('b', pa.float64())])
        assert table.schema == schema
        assert table.to_pydict() == {
            'a': ["1.25", "NA", "-4"],
            'b': [2.5, -3.0, None],
        }

    def test_simple_timestamps(self):
        # Infer a timestamp column
        rows = (b"a,b,c\n"
                b"1970,1970-01-01 00:00:00,1970-01-01 00:00:00.123\n"
                b"1989,1989-07-14 01:00:00,1989-07-14 01:00:00.123456\n")
        table = self.read_bytes(rows)
        schema = pa.schema([('a', pa.int64()),
                            ('b', pa.timestamp('s')),
                            ('c', pa.timestamp('ns'))])
        assert table.schema == schema
        assert table.to_pydict() == {
            'a': [1970, 1989],
            'b': [datetime(1970, 1, 1), datetime(1989, 7, 14, 1)],
            'c': [datetime(1970, 1, 1, 0, 0, 0, 123000),
                  datetime(1989, 7, 14, 1, 0, 0, 123456)],
        }

    def test_timestamp_parsers(self):
        # Infer timestamps with custom parsers
        rows = b"a,b\n1970/01/01,1980-01-01 00\n1970/01/02,1980-01-02 00\n"
        opts = ConvertOptions()

        table = self.read_bytes(rows, convert_options=opts)
        schema = pa.schema([('a', pa.string()),
                            ('b', pa.timestamp('s'))])
        assert table.schema == schema
        assert table.to_pydict() == {
            'a': ['1970/01/01', '1970/01/02'],
            'b': [datetime(1980, 1, 1), datetime(1980, 1, 2)],
        }

        opts.timestamp_parsers = ['%Y/%m/%d']
        table = self.read_bytes(rows, convert_options=opts)
        schema = pa.schema([('a', pa.timestamp('s')),
                            ('b', pa.string())])
        assert table.schema == schema
        assert table.to_pydict() == {
            'a': [datetime(1970, 1, 1), datetime(1970, 1, 2)],
            'b': ['1980-01-01 00', '1980-01-02 00'],
        }

        opts.timestamp_parsers = ['%Y/%m/%d', ISO8601]
        table = self.read_bytes(rows, convert_options=opts)
        schema = pa.schema([('a', pa.timestamp('s')),
                            ('b', pa.timestamp('s'))])
        assert table.schema == schema
        assert table.to_pydict() == {
            'a': [datetime(1970, 1, 1), datetime(1970, 1, 2)],
            'b': [datetime(1980, 1, 1), datetime(1980, 1, 2)],
        }

    def test_dates(self):
        # Dates are inferred as date32 by default
        rows = b"a,b\n1970-01-01,1970-01-02\n1971-01-01,1971-01-02\n"
        table = self.read_bytes(rows)
        schema = pa.schema([('a', pa.date32()),
                            ('b', pa.date32())])
        assert table.schema == schema
        assert table.to_pydict() == {
            'a': [date(1970, 1, 1), date(1971, 1, 1)],
            'b': [date(1970, 1, 2), date(1971, 1, 2)],
        }

        # Can ask for date types explicitly
        opts = ConvertOptions()
        opts.column_types = {'a': pa.date32(), 'b': pa.date64()}
        table = self.read_bytes(rows, convert_options=opts)
        schema = pa.schema([('a', pa.date32()),
                            ('b', pa.date64())])
        assert table.schema == schema
        assert table.to_pydict() == {
            'a': [date(1970, 1, 1), date(1971, 1, 1)],
            'b': [date(1970, 1, 2), date(1971, 1, 2)],
        }

        # Can ask for timestamp types explicitly
        opts = ConvertOptions()
        opts.column_types = {'a': pa.timestamp('s'), 'b': pa.timestamp('ms')}
        table = self.read_bytes(rows, convert_options=opts)
        schema = pa.schema([('a', pa.timestamp('s')),
                            ('b', pa.timestamp('ms'))])
        assert table.schema == schema
        assert table.to_pydict() == {
            'a': [datetime(1970, 1, 1), datetime(1971, 1, 1)],
            'b': [datetime(1970, 1, 2), datetime(1971, 1, 2)],
        }

    def test_times(self):
        # Times are inferred as time32[s] by default
        from datetime import time

        rows = b"a,b\n12:34:56,12:34:56.789\n23:59:59,23:59:59.999\n"
        table = self.read_bytes(rows)
        # Column 'b' has subseconds, so cannot be inferred as time32[s]
        schema = pa.schema([('a', pa.time32('s')),
                            ('b', pa.string())])
        assert table.schema == schema
        assert table.to_pydict() == {
            'a': [time(12, 34, 56), time(23, 59, 59)],
            'b': ["12:34:56.789", "23:59:59.999"],
        }

        # Can ask for time types explicitly
        opts = ConvertOptions()
        opts.column_types = {'a': pa.time64('us'), 'b': pa.time32('ms')}
        table = self.read_bytes(rows, convert_options=opts)
        schema = pa.schema([('a', pa.time64('us')),
                            ('b', pa.time32('ms'))])
        assert table.schema == schema
        assert table.to_pydict() == {
            'a': [time(12, 34, 56), time(23, 59, 59)],
            'b': [time(12, 34, 56, 789000), time(23, 59, 59, 999000)],
        }

    def test_auto_dict_encode(self):
        opts = ConvertOptions(auto_dict_encode=True)
        rows = "a,b\nab,1\ncdé,2\ncdé,3\nab,4".encode()
        table = self.read_bytes(rows, convert_options=opts)
        schema = pa.schema([('a', pa.dictionary(pa.int32(), pa.string())),
                            ('b', pa.int64())])
        expected = {
            'a': ["ab", "cdé", "cdé", "ab"],
            'b': [1, 2, 3, 4],
        }
        assert table.schema == schema
        assert table.to_pydict() == expected

        opts.auto_dict_max_cardinality = 2
        table = self.read_bytes(rows, convert_options=opts)
        assert table.schema == schema
        assert table.to_pydict() == expected

        # Cardinality above max => plain-encoded
        opts.auto_dict_max_cardinality = 1
        table = self.read_bytes(rows, convert_options=opts)
        assert table.schema == pa.schema([('a', pa.string()),
                                          ('b', pa.int64())])
        assert table.to_pydict() == expected

        # With invalid UTF8, not checked
        opts.auto_dict_max_cardinality = 50
        opts.check_utf8 = False
        rows = b"a,b\nab,1\ncd\xff,2\nab,3"
        table = self.read_bytes(rows, convert_options=opts,
                                validate_full=False)
        assert table.schema == schema
        dict_values = table['a'].chunk(0).dictionary
        assert len(dict_values) == 2
        assert dict_values[0].as_py() == "ab"
        assert dict_values[1].as_buffer() == b"cd\xff"

        # With invalid UTF8, checked
        opts.check_utf8 = True
        table = self.read_bytes(rows, convert_options=opts)
        schema = pa.schema([('a', pa.dictionary(pa.int32(), pa.binary())),
                            ('b', pa.int64())])
        expected = {
            'a': [b"ab", b"cd\xff", b"ab"],
            'b': [1, 2, 3],
        }
        assert table.schema == schema
        assert table.to_pydict() == expected

    def test_custom_nulls(self):
        # Infer nulls with custom values
        opts = ConvertOptions(null_values=['Xxx', 'Zzz'])
        rows = b"""a,b,c,d\nZzz,"Xxx",1,2\nXxx,#N/A,,Zzz\n"""
        table = self.read_bytes(rows, convert_options=opts)
        schema = pa.schema([('a', pa.null()),
                            ('b', pa.string()),
                            ('c', pa.string()),
                            ('d', pa.int64())])
        assert table.schema == schema
        assert table.to_pydict() == {
            'a': [None, None],
            'b': ["Xxx", "#N/A"],
            'c': ["1", ""],
            'd': [2, None],
        }

        opts = ConvertOptions(null_values=['Xxx', 'Zzz'],
                              strings_can_be_null=True)
        table = self.read_bytes(rows, convert_options=opts)
        assert table.to_pydict() == {
            'a': [None, None],
            'b': [None, "#N/A"],
            'c': ["1", ""],
            'd': [2, None],
        }
        opts.quoted_strings_can_be_null = False
        table = self.read_bytes(rows, convert_options=opts)
        assert table.to_pydict() == {
            'a': [None, None],
            'b': ["Xxx", "#N/A"],
            'c': ["1", ""],
            'd': [2, None],
        }

        opts = ConvertOptions(null_values=[])
        rows = b"a,b\n#N/A,\n"
        table = self.read_bytes(rows, convert_options=opts)
        schema = pa.schema([('a', pa.string()),
                            ('b', pa.string())])
        assert table.schema == schema
        assert table.to_pydict() == {
            'a': ["#N/A"],
            'b': [""],
        }

    def test_custom_bools(self):
        # Infer booleans with custom values
        opts = ConvertOptions(true_values=['T', 'yes'],
                              false_values=['F', 'no'])
        rows = (b"a,b,c\n"
                b"True,T,t\n"
                b"False,F,f\n"
                b"True,yes,yes\n"
                b"False,no,no\n"
                b"N/A,N/A,N/A\n")
        table = self.read_bytes(rows, convert_options=opts)
        schema = pa.schema([('a', pa.string()),
                            ('b', pa.bool_()),
                            ('c', pa.string())])
        assert table.schema == schema
        assert table.to_pydict() == {
            'a': ["True", "False", "True", "False", "N/A"],
            'b': [True, False, True, False, None],
            'c': ["t", "f", "yes", "no", "N/A"],
        }

    def test_column_types(self):
        # Ask for specific column types in ConvertOptions
        opts = ConvertOptions(column_types={'b': 'float32',
                                            'c': 'string',
                                            'd': 'boolean',
                                            'e': pa.decimal128(11, 2),
                                            'zz': 'null'})
        rows = b"a,b,c,d,e\n1,2,3,true,1.0\n4,-5,6,false,0\n"
        table = self.read_bytes(rows, convert_options=opts)
        schema = pa.schema([('a', pa.int64()),
                            ('b', pa.float32()),
                            ('c', pa.string()),
                            ('d', pa.bool_()),
                            ('e', pa.decimal128(11, 2))])
        expected = {
            'a': [1, 4],
            'b': [2.0, -5.0],
            'c': ["3", "6"],
            'd': [True, False],
            'e': [Decimal("1.00"), Decimal("0.00")]
        }
        assert table.schema == schema
        assert table.to_pydict() == expected
        # Pass column_types as schema
        opts = ConvertOptions(
            column_types=pa.schema([('b', pa.float32()),
                                    ('c', pa.string()),
                                    ('d', pa.bool_()),
                                    ('e', pa.decimal128(11, 2)),
                                    ('zz', pa.bool_())]))
        table = self.read_bytes(rows, convert_options=opts)
        assert table.schema == schema
        assert table.to_pydict() == expected
        # One of the columns in column_types fails converting
        rows = b"a,b,c,d,e\n1,XXX,3,true,5\n4,-5,6,false,7\n"
        with pytest.raises(pa.ArrowInvalid) as exc:
            self.read_bytes(rows, convert_options=opts)
        err = str(exc.value)
        assert "In CSV column #1: " in err
        assert "CSV conversion error to float: invalid value 'XXX'" in err

    def test_column_types_dict(self):
        # Ask for dict-encoded column types in ConvertOptions
        column_types = [
            ('a', pa.dictionary(pa.int32(), pa.utf8())),
            ('b', pa.dictionary(pa.int32(), pa.int64())),
            ('c', pa.dictionary(pa.int32(), pa.decimal128(11, 2))),
            ('d', pa.dictionary(pa.int32(), pa.large_utf8()))]

        opts = ConvertOptions(column_types=dict(column_types))
        rows = (b"a,b,c,d\n"
                b"abc,123456,1.0,zz\n"
                b"defg,123456,0.5,xx\n"
                b"abc,N/A,1.0,xx\n")
        table = self.read_bytes(rows, convert_options=opts)

        schema = pa.schema(column_types)
        expected = {
            'a': ["abc", "defg", "abc"],
            'b': [123456, 123456, None],
            'c': [Decimal("1.00"), Decimal("0.50"), Decimal("1.00")],
            'd': ["zz", "xx", "xx"],
        }
        assert table.schema == schema
        assert table.to_pydict() == expected

        # Unsupported index type
        column_types[0] = ('a', pa.dictionary(pa.int8(), pa.utf8()))

        opts = ConvertOptions(column_types=dict(column_types))
        with pytest.raises(NotImplementedError):
            table = self.read_bytes(rows, convert_options=opts)

    def test_column_types_with_column_names(self):
        # When both `column_names` and `column_types` are given, names
        # in `column_types` should refer to names in `column_names`
        rows = b"a,b\nc,d\ne,f\n"
        read_options = ReadOptions(column_names=['x', 'y'])
        convert_options = ConvertOptions(column_types={'x': pa.binary()})
        table = self.read_bytes(rows, read_options=read_options,
                                convert_options=convert_options)
        schema = pa.schema([('x', pa.binary()),
                            ('y', pa.string())])
        assert table.schema == schema
        assert table.to_pydict() == {
            'x': [b'a', b'c', b'e'],
            'y': ['b', 'd', 'f'],
        }

    def test_no_ending_newline(self):
        # No \n after last line
        rows = b"a,b,c\n1,2,3\n4,5,6"
        table = self.read_bytes(rows)
        assert table.to_pydict() == {
            'a': [1, 4],
            'b': [2, 5],
            'c': [3, 6],
        }

    def test_trivial(self):
        # A bit pointless, but at least it shouldn't crash
        rows = b",\n\n"
        table = self.read_bytes(rows)
        assert table.to_pydict() == {'': []}

    def test_empty_lines(self):
        rows = b"a,b\n\r1,2\r\n\r\n3,4\r\n"
        table = self.read_bytes(rows)
        assert table.to_pydict() == {
            'a': [1, 3],
            'b': [2, 4],
        }
        parse_options = ParseOptions(ignore_empty_lines=False)
        table = self.read_bytes(rows, parse_options=parse_options)
        assert table.to_pydict() == {
            'a': [None, 1, None, 3],
            'b': [None, 2, None, 4],
        }
        read_options = ReadOptions(skip_rows=2)
        table = self.read_bytes(rows, parse_options=parse_options,
                                read_options=read_options)
        assert table.to_pydict() == {
            '1': [None, 3],
            '2': [None, 4],
        }

    def test_invalid_csv(self):
        # Various CSV errors
        rows = b"a,b,c\n1,2\n4,5,6\n"
        with pytest.raises(pa.ArrowInvalid, match="Expected 3 columns, got 2"):
            self.read_bytes(rows)
        rows = b"a,b,c\n1,2,3\n4"
        with pytest.raises(pa.ArrowInvalid, match="Expected 3 columns, got 1"):
            self.read_bytes(rows)
        for rows in [b"", b"\n", b"\r\n", b"\r", b"\n\n"]:
            with pytest.raises(pa.ArrowInvalid, match="Empty CSV file"):
                self.read_bytes(rows)

    def test_options_delimiter(self):
        rows = b"a;b,c\nde,fg;eh\n"
        table = self.read_bytes(rows)
        assert table.to_pydict() == {
            'a;b': ['de'],
            'c': ['fg;eh'],
        }
        opts = ParseOptions(delimiter=';')
        table = self.read_bytes(rows, parse_options=opts)
        assert table.to_pydict() == {
            'a': ['de,fg'],
            'b,c': ['eh'],
        }

    def test_small_random_csv(self):
        csv, expected = make_random_csv(num_cols=2, num_rows=10)
        table = self.read_bytes(csv)
        assert table.schema == expected.schema
        assert table.equals(expected)
        assert table.to_pydict() == expected.to_pydict()

    def test_stress_block_sizes(self):
        # Test a number of small block sizes to stress block stitching
        csv_base, expected = make_random_csv(num_cols=2, num_rows=500)
        block_sizes = [11, 12, 13, 17, 37, 111]
        csvs = [csv_base, csv_base.rstrip(b'\r\n')]
        for csv in csvs:
            for block_size in block_sizes:
                read_options = ReadOptions(block_size=block_size)
                table = self.read_bytes(csv, read_options=read_options)
                assert table.schema == expected.schema
                if not table.equals(expected):
                    # Better error output
                    assert table.to_pydict() == expected.to_pydict()

    def test_stress_convert_options_blowup(self):
        # ARROW-6481: A convert_options with a very large number of columns
        # should not blow memory and CPU time.
        try:
            clock = time.thread_time
        except AttributeError:
            clock = time.time
        num_columns = 10000
        col_names = ["K{}".format(i) for i in range(num_columns)]
        csv = make_empty_csv(col_names)
        t1 = clock()
        convert_options = ConvertOptions(
            column_types={k: pa.string() for k in col_names[::2]})
        table = self.read_bytes(csv, convert_options=convert_options)
        dt = clock() - t1
        # Check that processing time didn't blow up.
        # This is a conservative check (it takes less than 300 ms
        # in debug mode on my local machine).
        assert dt <= 10.0
        # Check result
        assert table.num_columns == num_columns
        assert table.num_rows == 0
        assert table.column_names == col_names

    def test_cancellation(self):
        if (threading.current_thread().ident !=
                threading.main_thread().ident):
            pytest.skip("test only works from main Python thread")
        # Skips test if not available
        raise_signal = util.get_raise_signal()
        signum = signal.SIGINT

        def signal_from_thread():
            # Give our workload a chance to start up
            time.sleep(0.2)
            raise_signal(signum)

        # We start with a small CSV reading workload and increase its size
        # until it's large enough to get an interruption during it, even in
        # release mode on fast machines.
        last_duration = 0.0
        workload_size = 100_000
        attempts = 0

        while last_duration < 5.0 and attempts < 10:
            print("workload size:", workload_size)
            large_csv = b"a,b,c\n" + b"1,2,3\n" * workload_size
            exc_info = None

            try:
                # We use a signal fd to reliably ensure that the signal
                # has been delivered to Python, regardless of how exactly
                # it was caught.
                with util.signal_wakeup_fd() as sigfd:
                    try:
                        t = threading.Thread(target=signal_from_thread)
                        t.start()
                        t1 = time.time()
                        try:
                            self.read_bytes(large_csv)
                        except KeyboardInterrupt as e:
                            exc_info = e
                            last_duration = time.time() - t1
                    finally:
                        # Wait for signal to arrive if it didn't already,
                        # to avoid getting a KeyboardInterrupt after the
                        # `except` block below.
                        select.select([sigfd], [], [sigfd], 10.0)

            except KeyboardInterrupt:
                # KeyboardInterrupt didn't interrupt `read_bytes` above.
                pass

            if exc_info is not None:
                # We managed to get `self.read_bytes` interrupted, see if it
                # was actually interrupted inside Arrow C++ or in the Python
                # scaffolding.
                if exc_info.__context__ is not None:
                    # Interrupted inside Arrow C++, we're satisfied now
                    break

            # Increase workload size to get a better chance
            workload_size = workload_size * 3

        if exc_info is None:
            pytest.fail("Failed to get an interruption during CSV reading")

        # Interruption should have arrived timely
        assert last_duration <= 1.0
        e = exc_info.__context__
        assert isinstance(e, pa.ArrowCancelled)
        assert e.signum == signum

    def test_cancellation_disabled(self):
        # ARROW-12622: reader would segfault when the cancelling signal
        # handler was not enabled (e.g. if disabled, or if not on the
        # main thread)
        t = threading.Thread(
            target=lambda: self.read_bytes(b"f64\n0.1"))
        t.start()
        t.join()


class TestSerialCSVTableRead(BaseCSVTableRead):
    @property
    def use_threads(self):
        return False


class TestThreadedCSVTableRead(BaseCSVTableRead):
    @property
    def use_threads(self):
        return True


class BaseStreamingCSVRead(BaseTestCSV):

    def open_csv(self, csv, *args, **kwargs):
        """
        Reads the CSV file into memory using pyarrow's open_csv
        csv The CSV bytes
        args Positional arguments to be forwarded to pyarrow's open_csv
        kwargs Keyword arguments to be forwarded to pyarrow's open_csv
        """
        read_options = kwargs.setdefault('read_options', ReadOptions())
        read_options.use_threads = self.use_threads
        return open_csv(csv, *args, **kwargs)

    def open_bytes(self, b, **kwargs):
        return self.open_csv(pa.py_buffer(b), **kwargs)

    def check_reader(self, reader, expected_schema, expected_data):
        assert reader.schema == expected_schema
        batches = list(reader)
        assert len(batches) == len(expected_data)
        for batch, expected_batch in zip(batches, expected_data):
            batch.validate(full=True)
            assert batch.schema == expected_schema
            assert batch.to_pydict() == expected_batch

    def read_bytes(self, b, **kwargs):
        return self.open_bytes(b, **kwargs).read_all()

    def test_file_object(self):
        data = b"a,b\n1,2\n3,4\n"
        expected_data = {'a': [1, 3], 'b': [2, 4]}
        bio = io.BytesIO(data)
        reader = self.open_csv(bio)
        expected_schema = pa.schema([('a', pa.int64()),
                                     ('b', pa.int64())])
        self.check_reader(reader, expected_schema, [expected_data])

    def test_header(self):
        rows = b"abc,def,gh\n"
        reader = self.open_bytes(rows)
        expected_schema = pa.schema([('abc', pa.null()),
                                     ('def', pa.null()),
                                     ('gh', pa.null())])
        self.check_reader(reader, expected_schema, [])

    def test_inference(self):
        # Inference is done on first block
        rows = b"a,b\n123,456\nabc,de\xff\ngh,ij\n"
        expected_schema = pa.schema([('a', pa.string()),
                                     ('b', pa.binary())])

        read_options = ReadOptions()
        read_options.block_size = len(rows)
        reader = self.open_bytes(rows, read_options=read_options)
        self.check_reader(reader, expected_schema,
                          [{'a': ['123', 'abc', 'gh'],
                            'b': [b'456', b'de\xff', b'ij']}])

        read_options.block_size = len(rows) - 1
        reader = self.open_bytes(rows, read_options=read_options)
        self.check_reader(reader, expected_schema,
                          [{'a': ['123', 'abc'],
                            'b': [b'456', b'de\xff']},
                           {'a': ['gh'],
                            'b': [b'ij']}])

    def test_inference_failure(self):
        # Inference on first block, then conversion failure on second block
        rows = b"a,b\n123,456\nabc,de\xff\ngh,ij\n"
        read_options = ReadOptions()
        read_options.block_size = len(rows) - 7
        reader = self.open_bytes(rows, read_options=read_options)
        expected_schema = pa.schema([('a', pa.int64()),
                                     ('b', pa.int64())])
        assert reader.schema == expected_schema
        assert reader.read_next_batch().to_pydict() == {
            'a': [123], 'b': [456]
        }
        # Second block
        with pytest.raises(ValueError,
                           match="CSV conversion error to int64"):
            reader.read_next_batch()
        # EOF
        with pytest.raises(StopIteration):
            reader.read_next_batch()

    def test_invalid_csv(self):
        # CSV errors on first block
        rows = b"a,b\n1,2,3\n4,5\n6,7\n"
        read_options = ReadOptions()
        read_options.block_size = 10
        with pytest.raises(pa.ArrowInvalid,
                           match="Expected 2 columns, got 3"):
            reader = self.open_bytes(
                rows, read_options=read_options)

        # CSV errors on second block
        rows = b"a,b\n1,2\n3,4,5\n6,7\n"
        read_options.block_size = 8
        reader = self.open_bytes(rows, read_options=read_options)
        assert reader.read_next_batch().to_pydict() == {'a': [1], 'b': [2]}
        with pytest.raises(pa.ArrowInvalid,
                           match="Expected 2 columns, got 3"):
            reader.read_next_batch()
        # Cannot continue after a parse error
        with pytest.raises(StopIteration):
            reader.read_next_batch()

    def test_options_delimiter(self):
        rows = b"a;b,c\nde,fg;eh\n"
        reader = self.open_bytes(rows)
        expected_schema = pa.schema([('a;b', pa.string()),
                                     ('c', pa.string())])
        self.check_reader(reader, expected_schema,
                          [{'a;b': ['de'],
                            'c': ['fg;eh']}])

        opts = ParseOptions(delimiter=';')
        reader = self.open_bytes(rows, parse_options=opts)
        expected_schema = pa.schema([('a', pa.string()),
                                     ('b,c', pa.string())])
        self.check_reader(reader, expected_schema,
                          [{'a': ['de,fg'],
                            'b,c': ['eh']}])

    def test_no_ending_newline(self):
        # No \n after last line
        rows = b"a,b,c\n1,2,3\n4,5,6"
        reader = self.open_bytes(rows)
        expected_schema = pa.schema([('a', pa.int64()),
                                     ('b', pa.int64()),
                                     ('c', pa.int64())])
        self.check_reader(reader, expected_schema,
                          [{'a': [1, 4],
                            'b': [2, 5],
                            'c': [3, 6]}])

    def test_empty_file(self):
        with pytest.raises(ValueError, match="Empty CSV file"):
            self.open_bytes(b"")

    def test_column_options(self):
        # With column_names
        rows = b"1,2,3\n4,5,6"
        read_options = ReadOptions()
        read_options.column_names = ['d', 'e', 'f']
        reader = self.open_bytes(rows, read_options=read_options)
        expected_schema = pa.schema([('d', pa.int64()),
                                     ('e', pa.int64()),
                                     ('f', pa.int64())])
        self.check_reader(reader, expected_schema,
                          [{'d': [1, 4],
                            'e': [2, 5],
                            'f': [3, 6]}])

        # With include_columns
        convert_options = ConvertOptions()
        convert_options.include_columns = ['f', 'e']
        reader = self.open_bytes(rows, read_options=read_options,
                                 convert_options=convert_options)
        expected_schema = pa.schema([('f', pa.int64()),
                                     ('e', pa.int64())])
        self.check_reader(reader, expected_schema,
                          [{'e': [2, 5],
                            'f': [3, 6]}])

        # With column_types
        convert_options.column_types = {'e': pa.string()}
        reader = self.open_bytes(rows, read_options=read_options,
                                 convert_options=convert_options)
        expected_schema = pa.schema([('f', pa.int64()),
                                     ('e', pa.string())])
        self.check_reader(reader, expected_schema,
                          [{'e': ["2", "5"],
                            'f': [3, 6]}])

        # Missing columns in include_columns
        convert_options.include_columns = ['g', 'f', 'e']
        with pytest.raises(
                KeyError,
                match="Column 'g' in include_columns does not exist"):
            reader = self.open_bytes(rows, read_options=read_options,
                                     convert_options=convert_options)

        convert_options.include_missing_columns = True
        reader = self.open_bytes(rows, read_options=read_options,
                                 convert_options=convert_options)
        expected_schema = pa.schema([('g', pa.null()),
                                     ('f', pa.int64()),
                                     ('e', pa.string())])
        self.check_reader(reader, expected_schema,
                          [{'g': [None, None],
                            'e': ["2", "5"],
                            'f': [3, 6]}])

        convert_options.column_types = {'e': pa.string(), 'g': pa.float64()}
        reader = self.open_bytes(rows, read_options=read_options,
                                 convert_options=convert_options)
        expected_schema = pa.schema([('g', pa.float64()),
                                     ('f', pa.int64()),
                                     ('e', pa.string())])
        self.check_reader(reader, expected_schema,
                          [{'g': [None, None],
                            'e': ["2", "5"],
                            'f': [3, 6]}])

    def test_encoding(self):
        # latin-1 (invalid utf-8)
        rows = b"a,b\nun,\xe9l\xe9phant"
        read_options = ReadOptions()
        reader = self.open_bytes(rows, read_options=read_options)
        expected_schema = pa.schema([('a', pa.string()),
                                     ('b', pa.binary())])
        self.check_reader(reader, expected_schema,
                          [{'a': ["un"],
                            'b': [b"\xe9l\xe9phant"]}])

        read_options.encoding = 'latin1'
        reader = self.open_bytes(rows, read_options=read_options)
        expected_schema = pa.schema([('a', pa.string()),
                                     ('b', pa.string())])
        self.check_reader(reader, expected_schema,
                          [{'a': ["un"],
                            'b': ["éléphant"]}])

        # utf-16
        rows = (b'\xff\xfea\x00,\x00b\x00\n\x00u\x00n\x00,'
                b'\x00\xe9\x00l\x00\xe9\x00p\x00h\x00a\x00n\x00t\x00')
        read_options.encoding = 'utf16'
        reader = self.open_bytes(rows, read_options=read_options)
        expected_schema = pa.schema([('a', pa.string()),
                                     ('b', pa.string())])
        self.check_reader(reader, expected_schema,
                          [{'a': ["un"],
                            'b': ["éléphant"]}])

    def test_small_random_csv(self):
        csv, expected = make_random_csv(num_cols=2, num_rows=10)
        reader = self.open_bytes(csv)
        table = reader.read_all()
        assert table.schema == expected.schema
        assert table.equals(expected)
        assert table.to_pydict() == expected.to_pydict()

    def test_stress_block_sizes(self):
        # Test a number of small block sizes to stress block stitching
        csv_base, expected = make_random_csv(num_cols=2, num_rows=500)
        block_sizes = [19, 21, 23, 26, 37, 111]
        csvs = [csv_base, csv_base.rstrip(b'\r\n')]
        for csv in csvs:
            for block_size in block_sizes:
                # Need at least two lines for type inference
                assert csv[:block_size].count(b'\n') >= 2
                read_options = ReadOptions(block_size=block_size)
                reader = self.open_bytes(
                    csv, read_options=read_options)
                table = reader.read_all()
                assert table.schema == expected.schema
                if not table.equals(expected):
                    # Better error output
                    assert table.to_pydict() == expected.to_pydict()

    def test_batch_lifetime(self):
        gc.collect()
        old_allocated = pa.total_allocated_bytes()

        # Memory occupation should not grow with CSV file size
        def check_one_batch(reader, expected):
            batch = reader.read_next_batch()
            assert batch.to_pydict() == expected

        rows = b"10,11\n12,13\n14,15\n16,17\n"
        read_options = ReadOptions()
        read_options.column_names = ['a', 'b']
        read_options.block_size = 6
        reader = self.open_bytes(rows, read_options=read_options)
        check_one_batch(reader, {'a': [10], 'b': [11]})
        allocated_after_first_batch = pa.total_allocated_bytes()
        check_one_batch(reader, {'a': [12], 'b': [13]})
        assert pa.total_allocated_bytes() <= allocated_after_first_batch
        check_one_batch(reader, {'a': [14], 'b': [15]})
        assert pa.total_allocated_bytes() <= allocated_after_first_batch
        check_one_batch(reader, {'a': [16], 'b': [17]})
        assert pa.total_allocated_bytes() <= allocated_after_first_batch
        with pytest.raises(StopIteration):
            reader.read_next_batch()
        assert pa.total_allocated_bytes() == old_allocated
        reader = None
        assert pa.total_allocated_bytes() == old_allocated

    def test_header_skip_rows(self):
        super().test_header_skip_rows()

        rows = b"ab,cd\nef,gh\nij,kl\nmn,op\n"

        # Skipping all rows immediately results in end of iteration
        opts = ReadOptions()
        opts.skip_rows = 4
        opts.column_names = ['ab', 'cd']
        reader = self.open_bytes(rows, read_options=opts)
        with pytest.raises(StopIteration):
            assert reader.read_next_batch()

    def test_skip_rows_after_names(self):
        super().test_skip_rows_after_names()

        rows = b"ab,cd\nef,gh\nij,kl\nmn,op\n"

        # Skipping all rows immediately results in end of iteration
        opts = ReadOptions()
        opts.skip_rows_after_names = 3
        reader = self.open_bytes(rows, read_options=opts)
        with pytest.raises(StopIteration):
            assert reader.read_next_batch()

        # Skipping beyond all rows immediately results in end of iteration
        opts.skip_rows_after_names = 99999
        reader = self.open_bytes(rows, read_options=opts)
        with pytest.raises(StopIteration):
            assert reader.read_next_batch()


class TestSerialStreamingCSVRead(BaseStreamingCSVRead, unittest.TestCase):
    @property
    def use_threads(self):
        return False


class TestThreadedStreamingCSVRead(BaseStreamingCSVRead, unittest.TestCase):
    @property
    def use_threads(self):
        return True


class BaseTestCompressedCSVRead:

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='arrow-csv-test-')

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def read_csv(self, csv_path):
        try:
            return read_csv(csv_path)
        except pa.ArrowNotImplementedError as e:
            pytest.skip(str(e))

    def test_random_csv(self):
        csv, expected = make_random_csv(num_cols=2, num_rows=100)
        csv_path = os.path.join(self.tmpdir, self.csv_filename)
        self.write_file(csv_path, csv)
        table = self.read_csv(csv_path)
        table.validate(full=True)
        assert table.schema == expected.schema
        assert table.equals(expected)
        assert table.to_pydict() == expected.to_pydict()


class TestGZipCSVRead(BaseTestCompressedCSVRead, unittest.TestCase):
    csv_filename = "compressed.csv.gz"

    def write_file(self, path, contents):
        with gzip.open(path, 'wb', 3) as f:
            f.write(contents)

    def test_concatenated(self):
        # ARROW-5974
        csv_path = os.path.join(self.tmpdir, self.csv_filename)
        with gzip.open(csv_path, 'wb', 3) as f:
            f.write(b"ab,cd\nef,gh\n")
        with gzip.open(csv_path, 'ab', 3) as f:
            f.write(b"ij,kl\nmn,op\n")
        table = self.read_csv(csv_path)
        assert table.to_pydict() == {
            'ab': ['ef', 'ij', 'mn'],
            'cd': ['gh', 'kl', 'op'],
        }


class TestBZ2CSVRead(BaseTestCompressedCSVRead, unittest.TestCase):
    csv_filename = "compressed.csv.bz2"

    def write_file(self, path, contents):
        with bz2.BZ2File(path, 'w') as f:
            f.write(contents)


def test_read_csv_does_not_close_passed_file_handles():
    # ARROW-4823
    buf = io.BytesIO(b"a,b,c\n1,2,3\n4,5,6")
    read_csv(buf)
    assert not buf.closed


def test_write_read_round_trip():
    t = pa.Table.from_arrays([[1, 2, 3], ["a", "b", "c"]], ["c1", "c2"])
    record_batch = t.to_batches(max_chunksize=4)[0]
    for data in [t, record_batch]:
        # Test with header
        buf = io.BytesIO()
        write_csv(data, buf, WriteOptions(include_header=True))
        buf.seek(0)
        assert t == read_csv(buf)

        # Test without header
        buf = io.BytesIO()
        write_csv(data, buf, WriteOptions(include_header=False))
        buf.seek(0)

        read_options = ReadOptions(column_names=t.column_names)
        assert t == read_csv(buf, read_options=read_options)

    # Test with writer
    for read_options, parse_options, write_options in [
        (None, None, WriteOptions(include_header=True)),
        (ReadOptions(column_names=t.column_names), None,
         WriteOptions(include_header=False)),
        (None, ParseOptions(delimiter='|'),
         WriteOptions(include_header=True, delimiter='|')),
        (ReadOptions(column_names=t.column_names),
         ParseOptions(delimiter='\t'),
         WriteOptions(include_header=False, delimiter='\t')),
    ]:
        buf = io.BytesIO()
        with CSVWriter(buf, t.schema, write_options=write_options) as writer:
            writer.write_table(t)
        buf.seek(0)
        assert t == read_csv(buf, read_options=read_options,
                             parse_options=parse_options)
        buf = io.BytesIO()
        with CSVWriter(buf, t.schema, write_options=write_options) as writer:
            for batch in t.to_batches(max_chunksize=1):
                writer.write_batch(batch)
        buf.seek(0)
        assert t == read_csv(buf, read_options=read_options,
                             parse_options=parse_options)


def test_write_quoting_style():
    t = pa.Table.from_arrays([[1, 2, None], ["a", None, "c"]], ["c1", "c2"])
    buf = io.BytesIO()
    for write_options, res in [
        (WriteOptions(quoting_style='none'), b'"c1","c2"\n1,a\n2,\n,c\n'),
        (WriteOptions(), b'"c1","c2"\n1,"a"\n2,\n,"c"\n'),
        (WriteOptions(quoting_style='all_valid'),
         b'"c1","c2"\n"1","a"\n"2",\n,"c"\n'),
    ]:
        with CSVWriter(buf, t.schema, write_options=write_options) as writer:
            writer.write_table(t)
        assert buf.getvalue() == res
        buf.seek(0)

    # Test writing special characters with different quoting styles
    t = pa.Table.from_arrays([[",", "\""]], ["c1"])
    buf = io.BytesIO()
    for write_options, res in [
        (WriteOptions(quoting_style='needed'), b'"c1"\n","\n""""\n'),
        (WriteOptions(quoting_style='none'), pa.lib.ArrowInvalid),
    ]:
        with CSVWriter(buf, t.schema, write_options=write_options) as writer:
            try:
                writer.write_table(t)
            except Exception as e:
                # This will trigger when we try to write a comma (,)
                # without quotes, which is invalid
                assert type(e) == res
                break
        assert buf.getvalue() == res
        buf.seek(0)


def test_read_csv_reference_cycle():
    # ARROW-13187
    def inner():
        buf = io.BytesIO(b"a,b,c\n1,2,3\n4,5,6")
        table = read_csv(buf)
        return weakref.ref(table)

    with util.disabled_gc():
        wr = inner()
        assert wr() is None


@pytest.mark.parametrize("type_factory", (
    lambda: pa.decimal128(20, 1),
    lambda: pa.decimal128(38, 15),
    lambda: pa.decimal256(20, 1),
    lambda: pa.decimal256(76, 10),
))
def test_write_csv_decimal(tmpdir, type_factory):
    type = type_factory()
    table = pa.table({"col": pa.array([1, 2]).cast(type)})

    write_csv(table, tmpdir / "out.csv")
    out = read_csv(tmpdir / "out.csv")

    assert out.column('col').cast(type) == table.column('col')

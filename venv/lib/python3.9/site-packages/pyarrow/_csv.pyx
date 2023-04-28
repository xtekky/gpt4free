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

from cython.operator cimport dereference as deref

import codecs
from collections import namedtuple
from collections.abc import Mapping

from pyarrow.includes.common cimport *
from pyarrow.includes.libarrow cimport *
from pyarrow.includes.libarrow_python cimport (MakeInvalidRowHandler,
                                               PyInvalidRowCallback)
from pyarrow.lib cimport (check_status, Field, MemoryPool, Schema,
                          RecordBatchReader, ensure_type,
                          maybe_unbox_memory_pool, get_input_stream,
                          get_writer, native_transcoding_input_stream,
                          pyarrow_unwrap_batch, pyarrow_unwrap_schema,
                          pyarrow_unwrap_table, pyarrow_wrap_schema,
                          pyarrow_wrap_table, pyarrow_wrap_data_type,
                          pyarrow_unwrap_data_type, Table, RecordBatch,
                          StopToken, _CRecordBatchWriter)
from pyarrow.lib import frombytes, tobytes, SignalStopHandler
from pyarrow.util import _stringify_path


cdef unsigned char _single_char(s) except 0:
    val = ord(s)
    if val == 0 or val > 127:
        raise ValueError("Expecting an ASCII character")
    return <unsigned char> val


_InvalidRow = namedtuple(
    "_InvalidRow", ("expected_columns", "actual_columns", "number", "text"),
    module=__name__)


class InvalidRow(_InvalidRow):
    """
    Description of an invalid row in a CSV file.

    Parameters
    ----------
    expected_columns : int
        The expected number of columns in the row.
    actual_columns : int
        The actual number of columns in the row.
    number : int or None
        The physical row number if known, otherwise None.
    text : str
        The contents of the row.
    """
    __slots__ = ()


cdef CInvalidRowResult _handle_invalid_row(
        handler, const CCSVInvalidRow& c_row) except CInvalidRowResult_Error:
    # A negative row number means undetermined (because of parallel reading)
    row_number = c_row.number if c_row.number >= 0 else None
    row = InvalidRow(c_row.expected_columns, c_row.actual_columns,
                     row_number, frombytes(<c_string> c_row.text))
    result = handler(row)
    if result == 'error':
        return CInvalidRowResult_Error
    elif result == 'skip':
        return CInvalidRowResult_Skip
    else:
        raise ValueError("Invalid return value for invalid row handler: "
                         f"expected 'error' or 'skip', got {result!r}")


cdef class ReadOptions(_Weakrefable):
    """
    Options for reading CSV files.

    Parameters
    ----------
    use_threads : bool, optional (default True)
        Whether to use multiple threads to accelerate reading
    block_size : int, optional
        How much bytes to process at a time from the input stream.
        This will determine multi-threading granularity as well as
        the size of individual record batches or table chunks.
        Minimum valid value for block size is 1
    skip_rows : int, optional (default 0)
        The number of rows to skip before the column names (if any)
        and the CSV data.
    skip_rows_after_names : int, optional (default 0)
        The number of rows to skip after the column names.
        This number can be larger than the number of rows in one
        block, and empty rows are counted.
        The order of application is as follows:
        - `skip_rows` is applied (if non-zero);
        - column names aread (unless `column_names` is set);
        - `skip_rows_after_names` is applied (if non-zero).
    column_names : list, optional
        The column names of the target table.  If empty, fall back on
        `autogenerate_column_names`.
    autogenerate_column_names : bool, optional (default False)
        Whether to autogenerate column names if `column_names` is empty.
        If true, column names will be of the form "f0", "f1"...
        If false, column names will be read from the first CSV row
        after `skip_rows`.
    encoding : str, optional (default 'utf8')
        The character encoding of the CSV data.  Columns that cannot
        decode using this encoding can still be read as Binary.

    Examples
    --------

    Defining an example data:

    >>> import io
    >>> s = "1,2,3\\nFlamingo,2,2022-03-01\\nHorse,4,2022-03-02\\nBrittle stars,5,2022-03-03\\nCentipede,100,2022-03-04"
    >>> print(s)
    1,2,3
    Flamingo,2,2022-03-01
    Horse,4,2022-03-02
    Brittle stars,5,2022-03-03
    Centipede,100,2022-03-04

    Ignore the first numbered row and substitute it with defined
    or autogenerated column names:

    >>> from pyarrow import csv
    >>> read_options = csv.ReadOptions(
    ...                column_names=["animals", "n_legs", "entry"],
    ...                skip_rows=1)
    >>> csv.read_csv(io.BytesIO(s.encode()), read_options=read_options)
    pyarrow.Table
    animals: string
    n_legs: int64
    entry: date32[day]
    ----
    animals: [["Flamingo","Horse","Brittle stars","Centipede"]]
    n_legs: [[2,4,5,100]]
    entry: [[2022-03-01,2022-03-02,2022-03-03,2022-03-04]]

    >>> read_options = csv.ReadOptions(autogenerate_column_names=True,
    ...                                skip_rows=1)
    >>> csv.read_csv(io.BytesIO(s.encode()), read_options=read_options)
    pyarrow.Table
    f0: string
    f1: int64
    f2: date32[day]
    ----
    f0: [["Flamingo","Horse","Brittle stars","Centipede"]]
    f1: [[2,4,5,100]]
    f2: [[2022-03-01,2022-03-02,2022-03-03,2022-03-04]]

    Remove the first 2 rows of the data:

    >>> read_options = csv.ReadOptions(skip_rows_after_names=2)
    >>> csv.read_csv(io.BytesIO(s.encode()), read_options=read_options)
    pyarrow.Table
    1: string
    2: int64
    3: date32[day]
    ----
    1: [["Brittle stars","Centipede"]]
    2: [[5,100]]
    3: [[2022-03-03,2022-03-04]]
    """

    # Avoid mistakingly creating attributes
    __slots__ = ()

    # __init__() is not called when unpickling, initialize storage here
    def __cinit__(self, *argw, **kwargs):
        self.options.reset(new CCSVReadOptions(CCSVReadOptions.Defaults()))

    def __init__(self, *, use_threads=None, block_size=None, skip_rows=None,
                 skip_rows_after_names=None, column_names=None,
                 autogenerate_column_names=None, encoding='utf8'):
        if use_threads is not None:
            self.use_threads = use_threads
        if block_size is not None:
            self.block_size = block_size
        if skip_rows is not None:
            self.skip_rows = skip_rows
        if skip_rows_after_names is not None:
            self.skip_rows_after_names = skip_rows_after_names
        if column_names is not None:
            self.column_names = column_names
        if autogenerate_column_names is not None:
            self.autogenerate_column_names= autogenerate_column_names
        # Python-specific option
        self.encoding = encoding

    @property
    def use_threads(self):
        """
        Whether to use multiple threads to accelerate reading.
        """
        return deref(self.options).use_threads

    @use_threads.setter
    def use_threads(self, value):
        deref(self.options).use_threads = value

    @property
    def block_size(self):
        """
        How much bytes to process at a time from the input stream.
        This will determine multi-threading granularity as well as
        the size of individual record batches or table chunks.
        """
        return deref(self.options).block_size

    @block_size.setter
    def block_size(self, value):
        deref(self.options).block_size = value

    @property
    def skip_rows(self):
        """
        The number of rows to skip before the column names (if any)
        and the CSV data.
        See `skip_rows_after_names` for interaction description
        """
        return deref(self.options).skip_rows

    @skip_rows.setter
    def skip_rows(self, value):
        deref(self.options).skip_rows = value

    @property
    def skip_rows_after_names(self):
        """
        The number of rows to skip after the column names.
        This number can be larger than the number of rows in one
        block, and empty rows are counted.
        The order of application is as follows:
        - `skip_rows` is applied (if non-zero);
        - column names aread (unless `column_names` is set);
        - `skip_rows_after_names` is applied (if non-zero).
        """
        return deref(self.options).skip_rows_after_names

    @skip_rows_after_names.setter
    def skip_rows_after_names(self, value):
        deref(self.options).skip_rows_after_names = value

    @property
    def column_names(self):
        """
        The column names of the target table.  If empty, fall back on
        `autogenerate_column_names`.
        """
        return [frombytes(s) for s in deref(self.options).column_names]

    @column_names.setter
    def column_names(self, value):
        deref(self.options).column_names.clear()
        for item in value:
            deref(self.options).column_names.push_back(tobytes(item))

    @property
    def autogenerate_column_names(self):
        """
        Whether to autogenerate column names if `column_names` is empty.
        If true, column names will be of the form "f0", "f1"...
        If false, column names will be read from the first CSV row
        after `skip_rows`.
        """
        return deref(self.options).autogenerate_column_names

    @autogenerate_column_names.setter
    def autogenerate_column_names(self, value):
        deref(self.options).autogenerate_column_names = value

    def validate(self):
        check_status(deref(self.options).Validate())

    def equals(self, ReadOptions other):
        return (
            self.use_threads == other.use_threads and
            self.block_size == other.block_size and
            self.skip_rows == other.skip_rows and
            self.skip_rows_after_names == other.skip_rows_after_names and
            self.column_names == other.column_names and
            self.autogenerate_column_names ==
            other.autogenerate_column_names and
            self.encoding == other.encoding
        )

    @staticmethod
    cdef ReadOptions wrap(CCSVReadOptions options):
        out = ReadOptions()
        out.options.reset(new CCSVReadOptions(move(options)))
        out.encoding = 'utf8'  # No way to know this
        return out

    def __getstate__(self):
        return (self.use_threads, self.block_size, self.skip_rows,
                self.column_names, self.autogenerate_column_names,
                self.encoding, self.skip_rows_after_names)

    def __setstate__(self, state):
        (self.use_threads, self.block_size, self.skip_rows,
         self.column_names, self.autogenerate_column_names,
         self.encoding, self.skip_rows_after_names) = state

    def __eq__(self, other):
        try:
            return self.equals(other)
        except TypeError:
            return False


cdef class ParseOptions(_Weakrefable):
    """
    Options for parsing CSV files.

    Parameters
    ----------
    delimiter : 1-character string, optional (default ',')
        The character delimiting individual cells in the CSV data.
    quote_char : 1-character string or False, optional (default '"')
        The character used optionally for quoting CSV values
        (False if quoting is not allowed).
    double_quote : bool, optional (default True)
        Whether two quotes in a quoted CSV value denote a single quote
        in the data.
    escape_char : 1-character string or False, optional (default False)
        The character used optionally for escaping special characters
        (False if escaping is not allowed).
    newlines_in_values : bool, optional (default False)
        Whether newline characters are allowed in CSV values.
        Setting this to True reduces the performance of multi-threaded
        CSV reading.
    ignore_empty_lines : bool, optional (default True)
        Whether empty lines are ignored in CSV input.
        If False, an empty line is interpreted as containing a single empty
        value (assuming a one-column CSV file).
    invalid_row_handler : callable, optional (default None)
        If not None, this object is called for each CSV row that fails
        parsing (because of a mismatching number of columns).
        It should accept a single InvalidRow argument and return either
        "skip" or "error" depending on the desired outcome.

    Examples
    --------

    Defining an example file from bytes object:

    >>> import io
    >>> s = "animals;n_legs;entry\\nFlamingo;2;2022-03-01\\n# Comment here:\\nHorse;4;2022-03-02\\nBrittle stars;5;2022-03-03\\nCentipede;100;2022-03-04"
    >>> print(s)
    animals;n_legs;entry
    Flamingo;2;2022-03-01
    # Comment here:
    Horse;4;2022-03-02
    Brittle stars;5;2022-03-03
    Centipede;100;2022-03-04
    >>> source = io.BytesIO(s.encode())

    Read the data from a file skipping rows with comments
    and defining the delimiter:

    >>> from pyarrow import csv
    >>> def skip_comment(row):
    ...     if row.text.startswith("# "):
    ...         return 'skip'
    ...     else:
    ...         return 'error'
    ...
    >>> parse_options = csv.ParseOptions(delimiter=";", invalid_row_handler=skip_comment)
    >>> csv.read_csv(source, parse_options=parse_options)
    pyarrow.Table
    animals: string
    n_legs: int64
    entry: date32[day]
    ----
    animals: [["Flamingo","Horse","Brittle stars","Centipede"]]
    n_legs: [[2,4,5,100]]
    entry: [[2022-03-01,2022-03-02,2022-03-03,2022-03-04]]
    """
    __slots__ = ()

    def __cinit__(self, *argw, **kwargs):
        self._invalid_row_handler = None
        self.options.reset(new CCSVParseOptions(CCSVParseOptions.Defaults()))

    def __init__(self, *, delimiter=None, quote_char=None, double_quote=None,
                 escape_char=None, newlines_in_values=None,
                 ignore_empty_lines=None, invalid_row_handler=None):
        if delimiter is not None:
            self.delimiter = delimiter
        if quote_char is not None:
            self.quote_char = quote_char
        if double_quote is not None:
            self.double_quote = double_quote
        if escape_char is not None:
            self.escape_char = escape_char
        if newlines_in_values is not None:
            self.newlines_in_values = newlines_in_values
        if ignore_empty_lines is not None:
            self.ignore_empty_lines = ignore_empty_lines
        if invalid_row_handler is not None:
            self.invalid_row_handler = invalid_row_handler

    @property
    def delimiter(self):
        """
        The character delimiting individual cells in the CSV data.
        """
        return chr(deref(self.options).delimiter)

    @delimiter.setter
    def delimiter(self, value):
        deref(self.options).delimiter = _single_char(value)

    @property
    def quote_char(self):
        """
        The character used optionally for quoting CSV values
        (False if quoting is not allowed).
        """
        if deref(self.options).quoting:
            return chr(deref(self.options).quote_char)
        else:
            return False

    @quote_char.setter
    def quote_char(self, value):
        if value is False:
            deref(self.options).quoting = False
        else:
            deref(self.options).quote_char = _single_char(value)
            deref(self.options).quoting = True

    @property
    def double_quote(self):
        """
        Whether two quotes in a quoted CSV value denote a single quote
        in the data.
        """
        return deref(self.options).double_quote

    @double_quote.setter
    def double_quote(self, value):
        deref(self.options).double_quote = value

    @property
    def escape_char(self):
        """
        The character used optionally for escaping special characters
        (False if escaping is not allowed).
        """
        if deref(self.options).escaping:
            return chr(deref(self.options).escape_char)
        else:
            return False

    @escape_char.setter
    def escape_char(self, value):
        if value is False:
            deref(self.options).escaping = False
        else:
            deref(self.options).escape_char = _single_char(value)
            deref(self.options).escaping = True

    @property
    def newlines_in_values(self):
        """
        Whether newline characters are allowed in CSV values.
        Setting this to True reduces the performance of multi-threaded
        CSV reading.
        """
        return deref(self.options).newlines_in_values

    @newlines_in_values.setter
    def newlines_in_values(self, value):
        deref(self.options).newlines_in_values = value

    @property
    def ignore_empty_lines(self):
        """
        Whether empty lines are ignored in CSV input.
        If False, an empty line is interpreted as containing a single empty
        value (assuming a one-column CSV file).
        """
        return deref(self.options).ignore_empty_lines

    @property
    def invalid_row_handler(self):
        """
        Optional handler for invalid rows.

        If not None, this object is called for each CSV row that fails
        parsing (because of a mismatching number of columns).
        It should accept a single InvalidRow argument and return either
        "skip" or "error" depending on the desired outcome.
        """
        return self._invalid_row_handler

    @invalid_row_handler.setter
    def invalid_row_handler(self, value):
        if value is not None and not callable(value):
            raise TypeError("Expected callable or None, "
                            f"got instance of {type(value)!r}")
        self._invalid_row_handler = value
        deref(self.options).invalid_row_handler = MakeInvalidRowHandler(
            <function[PyInvalidRowCallback]> &_handle_invalid_row, value)

    @ignore_empty_lines.setter
    def ignore_empty_lines(self, value):
        deref(self.options).ignore_empty_lines = value

    def validate(self):
        check_status(deref(self.options).Validate())

    def equals(self, ParseOptions other):
        return (
            self.delimiter == other.delimiter and
            self.quote_char == other.quote_char and
            self.double_quote == other.double_quote and
            self.escape_char == other.escape_char and
            self.newlines_in_values == other.newlines_in_values and
            self.ignore_empty_lines == other.ignore_empty_lines and
            self._invalid_row_handler == other._invalid_row_handler
        )

    @staticmethod
    cdef ParseOptions wrap(CCSVParseOptions options):
        out = ParseOptions()
        out.options.reset(new CCSVParseOptions(move(options)))
        return out

    def __getstate__(self):
        return (self.delimiter, self.quote_char, self.double_quote,
                self.escape_char, self.newlines_in_values,
                self.ignore_empty_lines, self.invalid_row_handler)

    def __setstate__(self, state):
        (self.delimiter, self.quote_char, self.double_quote,
         self.escape_char, self.newlines_in_values,
         self.ignore_empty_lines, self.invalid_row_handler) = state

    def __eq__(self, other):
        try:
            return self.equals(other)
        except TypeError:
            return False


cdef class _ISO8601(_Weakrefable):
    """
    A special object indicating ISO-8601 parsing.
    """
    __slots__ = ()

    def __str__(self):
        return 'ISO8601'

    def __eq__(self, other):
        return isinstance(other, _ISO8601)


ISO8601 = _ISO8601()


cdef class ConvertOptions(_Weakrefable):
    """
    Options for converting CSV data.

    Parameters
    ----------
    check_utf8 : bool, optional (default True)
        Whether to check UTF8 validity of string columns.
    column_types : pyarrow.Schema or dict, optional
        Explicitly map column names to column types. Passing this argument
        disables type inference on the defined columns.
    null_values : list, optional
        A sequence of strings that denote nulls in the data
        (defaults are appropriate in most cases). Note that by default,
        string columns are not checked for null values. To enable
        null checking for those, specify ``strings_can_be_null=True``.
    true_values : list, optional
        A sequence of strings that denote true booleans in the data
        (defaults are appropriate in most cases).
    false_values : list, optional
        A sequence of strings that denote false booleans in the data
        (defaults are appropriate in most cases).
    decimal_point : 1-character string, optional (default '.')
        The character used as decimal point in floating-point and decimal
        data.
    strings_can_be_null : bool, optional (default False)
        Whether string / binary columns can have null values.
        If true, then strings in null_values are considered null for
        string columns.
        If false, then all strings are valid string values.
    quoted_strings_can_be_null : bool, optional (default True)
        Whether quoted values can be null.
        If true, then strings in "null_values" are also considered null
        when they appear quoted in the CSV file. Otherwise, quoted values
        are never considered null.
    include_columns : list, optional
        The names of columns to include in the Table.
        If empty, the Table will include all columns from the CSV file.
        If not empty, only these columns will be included, in this order.
    include_missing_columns : bool, optional (default False)
        If false, columns in `include_columns` but not in the CSV file will
        error out.
        If true, columns in `include_columns` but not in the CSV file will
        produce a column of nulls (whose type is selected using
        `column_types`, or null by default).
        This option is ignored if `include_columns` is empty.
    auto_dict_encode : bool, optional (default False)
        Whether to try to automatically dict-encode string / binary data.
        If true, then when type inference detects a string or binary column,
        it it dict-encoded up to `auto_dict_max_cardinality` distinct values
        (per chunk), after which it switches to regular encoding.
        This setting is ignored for non-inferred columns (those in
        `column_types`).
    auto_dict_max_cardinality : int, optional
        The maximum dictionary cardinality for `auto_dict_encode`.
        This value is per chunk.
    timestamp_parsers : list, optional
        A sequence of strptime()-compatible format strings, tried in order
        when attempting to infer or convert timestamp values (the special
        value ISO8601() can also be given).  By default, a fast built-in
        ISO-8601 parser is used.

    Examples
    --------

    Defining an example data:

    >>> import io
    >>> s = "animals,n_legs,entry,fast\\nFlamingo,2,01/03/2022,Yes\\nHorse,4,02/03/2022,Yes\\nBrittle stars,5,03/03/2022,No\\nCentipede,100,04/03/2022,No\\n,6,05/03/2022,"
    >>> print(s)
    animals,n_legs,entry,fast
    Flamingo,2,01/03/2022,Yes
    Horse,4,02/03/2022,Yes
    Brittle stars,5,03/03/2022,No
    Centipede,100,04/03/2022,No
    ,6,05/03/2022,

    Change the type of a column:

    >>> import pyarrow as pa
    >>> from pyarrow import csv
    >>> convert_options = csv.ConvertOptions(column_types={"n_legs": pa.float64()})
    >>> csv.read_csv(io.BytesIO(s.encode()), convert_options=convert_options)
    pyarrow.Table
    animals: string
    n_legs: double
    entry: string
    fast: string
    ----
    animals: [["Flamingo","Horse","Brittle stars","Centipede",""]]
    n_legs: [[2,4,5,100,6]]
    entry: [["01/03/2022","02/03/2022","03/03/2022","04/03/2022","05/03/2022"]]
    fast: [["Yes","Yes","No","No",""]]

    Define a date parsing format to get a timestamp type column
    (in case dates are not in ISO format and not converted by default):

    >>> convert_options = csv.ConvertOptions(
    ...                   timestamp_parsers=["%m/%d/%Y", "%m-%d-%Y"])
    >>> csv.read_csv(io.BytesIO(s.encode()), convert_options=convert_options)
    pyarrow.Table
    animals: string
    n_legs: int64
    entry: timestamp[s]
    fast: string
    ----
    animals: [["Flamingo","Horse","Brittle stars","Centipede",""]]
    n_legs: [[2,4,5,100,6]]
    entry: [[2022-01-03 00:00:00,2022-02-03 00:00:00,2022-03-03 00:00:00,2022-04-03 00:00:00,2022-05-03 00:00:00]]
    fast: [["Yes","Yes","No","No",""]]

    Specify a subset of columns to be read:

    >>> convert_options = csv.ConvertOptions(
    ...                   include_columns=["animals", "n_legs"])
    >>> csv.read_csv(io.BytesIO(s.encode()), convert_options=convert_options)
    pyarrow.Table
    animals: string
    n_legs: int64
    ----
    animals: [["Flamingo","Horse","Brittle stars","Centipede",""]]
    n_legs: [[2,4,5,100,6]]

    List additional column to be included as a null typed column:

    >>> convert_options = csv.ConvertOptions(
    ...                   include_columns=["animals", "n_legs", "location"],
    ...                   include_missing_columns=True)
    >>> csv.read_csv(io.BytesIO(s.encode()), convert_options=convert_options)
    pyarrow.Table
    animals: string
    n_legs: int64
    location: null
    ----
    animals: [["Flamingo","Horse","Brittle stars","Centipede",""]]
    n_legs: [[2,4,5,100,6]]
    location: [5 nulls]

    Define columns as dictionary type (by default only the
    string/binary columns are dictionary encoded):

    >>> convert_options = csv.ConvertOptions(
    ...                   timestamp_parsers=["%m/%d/%Y", "%m-%d-%Y"],
    ...                   auto_dict_encode=True)
    >>> csv.read_csv(io.BytesIO(s.encode()), convert_options=convert_options)
    pyarrow.Table
    animals: dictionary<values=string, indices=int32, ordered=0>
    n_legs: int64
    entry: timestamp[s]
    fast: dictionary<values=string, indices=int32, ordered=0>
    ----
    animals: [  -- dictionary:
    ["Flamingo","Horse","Brittle stars","Centipede",""]  -- indices:
    [0,1,2,3,4]]
    n_legs: [[2,4,5,100,6]]
    entry: [[2022-01-03 00:00:00,2022-02-03 00:00:00,2022-03-03 00:00:00,2022-04-03 00:00:00,2022-05-03 00:00:00]]
    fast: [  -- dictionary:
    ["Yes","No",""]  -- indices:
    [0,0,1,1,2]]

    Set upper limit for the number of categories. If the categories
    is more than the limit, the conversion to dictionary will not
    happen:

    >>> convert_options = csv.ConvertOptions(
    ...                   include_columns=["animals"],
    ...                   auto_dict_encode=True,
    ...                   auto_dict_max_cardinality=2)
    >>> csv.read_csv(io.BytesIO(s.encode()), convert_options=convert_options)
    pyarrow.Table
    animals: string
    ----
    animals: [["Flamingo","Horse","Brittle stars","Centipede",""]]

    Set empty strings to missing values:

    >>> convert_options = csv.ConvertOptions(include_columns=["animals", "n_legs"],
    ...                   strings_can_be_null=True)
    >>> csv.read_csv(io.BytesIO(s.encode()), convert_options=convert_options)
    pyarrow.Table
    animals: string
    n_legs: int64
    ----
    animals: [["Flamingo","Horse","Brittle stars","Centipede",null]]
    n_legs: [[2,4,5,100,6]]

    Define values to be True and False when converting a column
    into a bool type:

    >>> convert_options = csv.ConvertOptions(
    ...                   include_columns=["fast"],
    ...                   false_values=["No"],
    ...                   true_values=["Yes"])
    >>> csv.read_csv(io.BytesIO(s.encode()), convert_options=convert_options)
    pyarrow.Table
    fast: bool
    ----
    fast: [[true,true,false,false,null]]
    """

    # Avoid mistakingly creating attributes
    __slots__ = ()

    def __cinit__(self, *argw, **kwargs):
        self.options.reset(
            new CCSVConvertOptions(CCSVConvertOptions.Defaults()))

    def __init__(self, *, check_utf8=None, column_types=None, null_values=None,
                 true_values=None, false_values=None, decimal_point=None,
                 strings_can_be_null=None, quoted_strings_can_be_null=None,
                 include_columns=None, include_missing_columns=None,
                 auto_dict_encode=None, auto_dict_max_cardinality=None,
                 timestamp_parsers=None):
        if check_utf8 is not None:
            self.check_utf8 = check_utf8
        if column_types is not None:
            self.column_types = column_types
        if null_values is not None:
            self.null_values = null_values
        if true_values is not None:
            self.true_values = true_values
        if false_values is not None:
            self.false_values = false_values
        if decimal_point is not None:
            self.decimal_point = decimal_point
        if strings_can_be_null is not None:
            self.strings_can_be_null = strings_can_be_null
        if quoted_strings_can_be_null is not None:
            self.quoted_strings_can_be_null = quoted_strings_can_be_null
        if include_columns is not None:
            self.include_columns = include_columns
        if include_missing_columns is not None:
            self.include_missing_columns = include_missing_columns
        if auto_dict_encode is not None:
            self.auto_dict_encode = auto_dict_encode
        if auto_dict_max_cardinality is not None:
            self.auto_dict_max_cardinality = auto_dict_max_cardinality
        if timestamp_parsers is not None:
            self.timestamp_parsers = timestamp_parsers

    @property
    def check_utf8(self):
        """
        Whether to check UTF8 validity of string columns.
        """
        return deref(self.options).check_utf8

    @check_utf8.setter
    def check_utf8(self, value):
        deref(self.options).check_utf8 = value

    @property
    def strings_can_be_null(self):
        """
        Whether string / binary columns can have null values.
        """
        return deref(self.options).strings_can_be_null

    @strings_can_be_null.setter
    def strings_can_be_null(self, value):
        deref(self.options).strings_can_be_null = value

    @property
    def quoted_strings_can_be_null(self):
        """
        Whether quoted values can be null.
        """
        return deref(self.options).quoted_strings_can_be_null

    @quoted_strings_can_be_null.setter
    def quoted_strings_can_be_null(self, value):
        deref(self.options).quoted_strings_can_be_null = value

    @property
    def column_types(self):
        """
        Explicitly map column names to column types.
        """
        d = {frombytes(item.first): pyarrow_wrap_data_type(item.second)
             for item in deref(self.options).column_types}
        return d

    @column_types.setter
    def column_types(self, value):
        cdef:
            shared_ptr[CDataType] typ

        if isinstance(value, Mapping):
            value = value.items()

        deref(self.options).column_types.clear()
        for item in value:
            if isinstance(item, Field):
                k = item.name
                v = item.type
            else:
                k, v = item
            typ = pyarrow_unwrap_data_type(ensure_type(v))
            assert typ != NULL
            deref(self.options).column_types[tobytes(k)] = typ

    @property
    def null_values(self):
        """
        A sequence of strings that denote nulls in the data.
        """
        return [frombytes(x) for x in deref(self.options).null_values]

    @null_values.setter
    def null_values(self, value):
        deref(self.options).null_values = [tobytes(x) for x in value]

    @property
    def true_values(self):
        """
        A sequence of strings that denote true booleans in the data.
        """
        return [frombytes(x) for x in deref(self.options).true_values]

    @true_values.setter
    def true_values(self, value):
        deref(self.options).true_values = [tobytes(x) for x in value]

    @property
    def false_values(self):
        """
        A sequence of strings that denote false booleans in the data.
        """
        return [frombytes(x) for x in deref(self.options).false_values]

    @false_values.setter
    def false_values(self, value):
        deref(self.options).false_values = [tobytes(x) for x in value]

    @property
    def decimal_point(self):
        """
        The character used as decimal point in floating-point and decimal
        data.
        """
        return chr(deref(self.options).decimal_point)

    @decimal_point.setter
    def decimal_point(self, value):
        deref(self.options).decimal_point = _single_char(value)

    @property
    def auto_dict_encode(self):
        """
        Whether to try to automatically dict-encode string / binary data.
        """
        return deref(self.options).auto_dict_encode

    @auto_dict_encode.setter
    def auto_dict_encode(self, value):
        deref(self.options).auto_dict_encode = value

    @property
    def auto_dict_max_cardinality(self):
        """
        The maximum dictionary cardinality for `auto_dict_encode`.

        This value is per chunk.
        """
        return deref(self.options).auto_dict_max_cardinality

    @auto_dict_max_cardinality.setter
    def auto_dict_max_cardinality(self, value):
        deref(self.options).auto_dict_max_cardinality = value

    @property
    def include_columns(self):
        """
        The names of columns to include in the Table.

        If empty, the Table will include all columns from the CSV file.
        If not empty, only these columns will be included, in this order.
        """
        return [frombytes(s) for s in deref(self.options).include_columns]

    @include_columns.setter
    def include_columns(self, value):
        deref(self.options).include_columns.clear()
        for item in value:
            deref(self.options).include_columns.push_back(tobytes(item))

    @property
    def include_missing_columns(self):
        """
        If false, columns in `include_columns` but not in the CSV file will
        error out.
        If true, columns in `include_columns` but not in the CSV file will
        produce a null column (whose type is selected using `column_types`,
        or null by default).
        This option is ignored if `include_columns` is empty.
        """
        return deref(self.options).include_missing_columns

    @include_missing_columns.setter
    def include_missing_columns(self, value):
        deref(self.options).include_missing_columns = value

    @property
    def timestamp_parsers(self):
        """
        A sequence of strptime()-compatible format strings, tried in order
        when attempting to infer or convert timestamp values (the special
        value ISO8601() can also be given).  By default, a fast built-in
        ISO-8601 parser is used.
        """
        cdef:
            shared_ptr[CTimestampParser] c_parser
            c_string kind

        parsers = []
        for c_parser in deref(self.options).timestamp_parsers:
            kind = deref(c_parser).kind()
            if kind == b'strptime':
                parsers.append(frombytes(deref(c_parser).format()))
            else:
                assert kind == b'iso8601'
                parsers.append(ISO8601)

        return parsers

    @timestamp_parsers.setter
    def timestamp_parsers(self, value):
        cdef:
            vector[shared_ptr[CTimestampParser]] c_parsers

        for v in value:
            if isinstance(v, str):
                c_parsers.push_back(CTimestampParser.MakeStrptime(tobytes(v)))
            elif v == ISO8601:
                c_parsers.push_back(CTimestampParser.MakeISO8601())
            else:
                raise TypeError("Expected list of str or ISO8601 objects")

        deref(self.options).timestamp_parsers = move(c_parsers)

    @staticmethod
    cdef ConvertOptions wrap(CCSVConvertOptions options):
        out = ConvertOptions()
        out.options.reset(new CCSVConvertOptions(move(options)))
        return out

    def validate(self):
        check_status(deref(self.options).Validate())

    def equals(self, ConvertOptions other):
        return (
            self.check_utf8 == other.check_utf8 and
            self.column_types == other.column_types and
            self.null_values == other.null_values and
            self.true_values == other.true_values and
            self.false_values == other.false_values and
            self.decimal_point == other.decimal_point and
            self.timestamp_parsers == other.timestamp_parsers and
            self.strings_can_be_null == other.strings_can_be_null and
            self.quoted_strings_can_be_null ==
            other.quoted_strings_can_be_null and
            self.auto_dict_encode == other.auto_dict_encode and
            self.auto_dict_max_cardinality ==
            other.auto_dict_max_cardinality and
            self.include_columns == other.include_columns and
            self.include_missing_columns == other.include_missing_columns
        )

    def __getstate__(self):
        return (self.check_utf8, self.column_types, self.null_values,
                self.true_values, self.false_values, self.decimal_point,
                self.timestamp_parsers, self.strings_can_be_null,
                self.quoted_strings_can_be_null, self.auto_dict_encode,
                self.auto_dict_max_cardinality, self.include_columns,
                self.include_missing_columns)

    def __setstate__(self, state):
        (self.check_utf8, self.column_types, self.null_values,
         self.true_values, self.false_values, self.decimal_point,
         self.timestamp_parsers, self.strings_can_be_null,
         self.quoted_strings_can_be_null, self.auto_dict_encode,
         self.auto_dict_max_cardinality, self.include_columns,
         self.include_missing_columns) = state

    def __eq__(self, other):
        try:
            return self.equals(other)
        except TypeError:
            return False


cdef _get_reader(input_file, ReadOptions read_options,
                 shared_ptr[CInputStream]* out):
    use_memory_map = False
    get_input_stream(input_file, use_memory_map, out)
    if read_options is not None:
        out[0] = native_transcoding_input_stream(out[0],
                                                 read_options.encoding,
                                                 'utf8')


cdef _get_read_options(ReadOptions read_options, CCSVReadOptions* out):
    if read_options is None:
        out[0] = CCSVReadOptions.Defaults()
    else:
        out[0] = deref(read_options.options)


cdef _get_parse_options(ParseOptions parse_options, CCSVParseOptions* out):
    if parse_options is None:
        out[0] = CCSVParseOptions.Defaults()
    else:
        out[0] = deref(parse_options.options)


cdef _get_convert_options(ConvertOptions convert_options,
                          CCSVConvertOptions* out):
    if convert_options is None:
        out[0] = CCSVConvertOptions.Defaults()
    else:
        out[0] = deref(convert_options.options)


cdef class CSVStreamingReader(RecordBatchReader):
    """An object that reads record batches incrementally from a CSV file.

    Should not be instantiated directly by user code.
    """
    cdef readonly:
        Schema schema

    def __init__(self):
        raise TypeError("Do not call {}'s constructor directly, "
                        "use pyarrow.csv.open_csv() instead."
                        .format(self.__class__.__name__))

    # Note about cancellation: we cannot create a SignalStopHandler
    # by default here, as several CSVStreamingReader instances may be
    # created (including by the same thread).  Handling cancellation
    # would require having the user pass the SignalStopHandler.
    # (in addition to solving ARROW-11853)

    cdef _open(self, shared_ptr[CInputStream] stream,
               CCSVReadOptions c_read_options,
               CCSVParseOptions c_parse_options,
               CCSVConvertOptions c_convert_options,
               MemoryPool memory_pool):
        cdef:
            shared_ptr[CSchema] c_schema
            CIOContext io_context

        io_context = CIOContext(maybe_unbox_memory_pool(memory_pool))

        with nogil:
            self.reader = <shared_ptr[CRecordBatchReader]> GetResultValue(
                CCSVStreamingReader.Make(
                    io_context, stream,
                    move(c_read_options), move(c_parse_options),
                    move(c_convert_options)))
            c_schema = self.reader.get().schema()

        self.schema = pyarrow_wrap_schema(c_schema)


def read_csv(input_file, read_options=None, parse_options=None,
             convert_options=None, MemoryPool memory_pool=None):
    """
    Read a Table from a stream of CSV data.

    Parameters
    ----------
    input_file : string, path or file-like object
        The location of CSV data.  If a string or path, and if it ends
        with a recognized compressed file extension (e.g. ".gz" or ".bz2"),
        the data is automatically decompressed when reading.
    read_options : pyarrow.csv.ReadOptions, optional
        Options for the CSV reader (see pyarrow.csv.ReadOptions constructor
        for defaults)
    parse_options : pyarrow.csv.ParseOptions, optional
        Options for the CSV parser
        (see pyarrow.csv.ParseOptions constructor for defaults)
    convert_options : pyarrow.csv.ConvertOptions, optional
        Options for converting CSV data
        (see pyarrow.csv.ConvertOptions constructor for defaults)
    memory_pool : MemoryPool, optional
        Pool to allocate Table memory from

    Returns
    -------
    :class:`pyarrow.Table`
        Contents of the CSV file as a in-memory table.

    Examples
    --------

    Defining an example file from bytes object:

    >>> import io
    >>> s = "animals,n_legs,entry\\nFlamingo,2,2022-03-01\\nHorse,4,2022-03-02\\nBrittle stars,5,2022-03-03\\nCentipede,100,2022-03-04"
    >>> print(s)
    animals,n_legs,entry
    Flamingo,2,2022-03-01
    Horse,4,2022-03-02
    Brittle stars,5,2022-03-03
    Centipede,100,2022-03-04
    >>> source = io.BytesIO(s.encode())

    Reading from the file

    >>> from pyarrow import csv
    >>> csv.read_csv(source)
    pyarrow.Table
    animals: string
    n_legs: int64
    entry: date32[day]
    ----
    animals: [["Flamingo","Horse","Brittle stars","Centipede"]]
    n_legs: [[2,4,5,100]]
    entry: [[2022-03-01,2022-03-02,2022-03-03,2022-03-04]]
    """
    cdef:
        shared_ptr[CInputStream] stream
        CCSVReadOptions c_read_options
        CCSVParseOptions c_parse_options
        CCSVConvertOptions c_convert_options
        CIOContext io_context
        shared_ptr[CCSVReader] reader
        shared_ptr[CTable] table

    _get_reader(input_file, read_options, &stream)
    _get_read_options(read_options, &c_read_options)
    _get_parse_options(parse_options, &c_parse_options)
    _get_convert_options(convert_options, &c_convert_options)

    with SignalStopHandler() as stop_handler:
        io_context = CIOContext(
            maybe_unbox_memory_pool(memory_pool),
            (<StopToken> stop_handler.stop_token).stop_token)
        reader = GetResultValue(CCSVReader.Make(
            io_context, stream,
            c_read_options, c_parse_options, c_convert_options))

        with nogil:
            table = GetResultValue(reader.get().Read())

    return pyarrow_wrap_table(table)


def open_csv(input_file, read_options=None, parse_options=None,
             convert_options=None, MemoryPool memory_pool=None):
    """
    Open a streaming reader of CSV data.

    Reading using this function is always single-threaded.

    Parameters
    ----------
    input_file : string, path or file-like object
        The location of CSV data.  If a string or path, and if it ends
        with a recognized compressed file extension (e.g. ".gz" or ".bz2"),
        the data is automatically decompressed when reading.
    read_options : pyarrow.csv.ReadOptions, optional
        Options for the CSV reader (see pyarrow.csv.ReadOptions constructor
        for defaults)
    parse_options : pyarrow.csv.ParseOptions, optional
        Options for the CSV parser
        (see pyarrow.csv.ParseOptions constructor for defaults)
    convert_options : pyarrow.csv.ConvertOptions, optional
        Options for converting CSV data
        (see pyarrow.csv.ConvertOptions constructor for defaults)
    memory_pool : MemoryPool, optional
        Pool to allocate Table memory from

    Returns
    -------
    :class:`pyarrow.csv.CSVStreamingReader`
    """
    cdef:
        shared_ptr[CInputStream] stream
        CCSVReadOptions c_read_options
        CCSVParseOptions c_parse_options
        CCSVConvertOptions c_convert_options
        CSVStreamingReader reader

    _get_reader(input_file, read_options, &stream)
    _get_read_options(read_options, &c_read_options)
    _get_parse_options(parse_options, &c_parse_options)
    _get_convert_options(convert_options, &c_convert_options)

    reader = CSVStreamingReader.__new__(CSVStreamingReader)
    reader._open(stream, move(c_read_options), move(c_parse_options),
                 move(c_convert_options), memory_pool)
    return reader


def _raise_invalid_function_option(value, description, *,
                                   exception_class=ValueError):
    raise exception_class(f"\"{value}\" is not a valid {description}")


cdef CQuotingStyle unwrap_quoting_style(quoting_style) except *:
    if quoting_style == "needed":
        return CQuotingStyle_Needed
    elif quoting_style == "all_valid":
        return CQuotingStyle_AllValid
    elif quoting_style == "none":
        return CQuotingStyle_None
    _raise_invalid_function_option(quoting_style, "quoting style")


cdef wrap_quoting_style(quoting_style):
    if quoting_style == CQuotingStyle_Needed:
        return 'needed'
    elif quoting_style == CQuotingStyle_AllValid:
        return 'all_valid'
    elif quoting_style == CQuotingStyle_None:
        return 'none'


cdef class WriteOptions(_Weakrefable):
    """
    Options for writing CSV files.

    Parameters
    ----------
    include_header : bool, optional (default True)
        Whether to write an initial header line with column names
    batch_size : int, optional (default 1024)
        How many rows to process together when converting and writing
        CSV data
    delimiter : 1-character string, optional (default ",")
        The character delimiting individual cells in the CSV data.
    quoting_style : str, optional (default "needed")
        Whether to quote values, and if so, which quoting style to use.
        The following values are accepted:

        - "needed" (default): only enclose values in quotes when needed.
        - "all_valid": enclose all valid values in quotes; nulls are not quoted.
        - "none": do not enclose any values in quotes; values containing
          special characters (such as quotes, cell delimiters or line endings)
          will raise an error.
    """

    # Avoid mistakingly creating attributes
    __slots__ = ()

    def __init__(self, *, include_header=None, batch_size=None,
                 delimiter=None, quoting_style=None):
        self.options.reset(new CCSVWriteOptions(CCSVWriteOptions.Defaults()))
        if include_header is not None:
            self.include_header = include_header
        if batch_size is not None:
            self.batch_size = batch_size
        if delimiter is not None:
            self.delimiter = delimiter
        if quoting_style is not None:
            self.quoting_style = quoting_style

    @property
    def include_header(self):
        """
        Whether to write an initial header line with column names.
        """
        return deref(self.options).include_header

    @include_header.setter
    def include_header(self, value):
        deref(self.options).include_header = value

    @property
    def batch_size(self):
        """
        How many rows to process together when converting and writing
        CSV data.
        """
        return deref(self.options).batch_size

    @batch_size.setter
    def batch_size(self, value):
        deref(self.options).batch_size = value

    @property
    def delimiter(self):
        """
        The character delimiting individual cells in the CSV data.
        """
        return chr(deref(self.options).delimiter)

    @delimiter.setter
    def delimiter(self, value):
        deref(self.options).delimiter = _single_char(value)

    @property
    def quoting_style(self):
        """
        Whether to quote values, and if so, which quoting style to use.
        The following values are accepted:

        - "needed" (default): only enclose values in quotes when needed.
        - "all_valid": enclose all valid values in quotes; nulls are not quoted.
        - "none": do not enclose any values in quotes; values containing
          special characters (such as quotes, cell delimiters or line endings)
          will raise an error.
        """
        return wrap_quoting_style(deref(self.options).quoting_style)

    @quoting_style.setter
    def quoting_style(self, value):
        deref(self.options).quoting_style = unwrap_quoting_style(value)

    @staticmethod
    cdef WriteOptions wrap(CCSVWriteOptions options):
        out = WriteOptions()
        out.options.reset(new CCSVWriteOptions(move(options)))
        return out

    def validate(self):
        check_status(self.options.get().Validate())


cdef _get_write_options(WriteOptions write_options, CCSVWriteOptions* out):
    if write_options is None:
        out[0] = CCSVWriteOptions.Defaults()
    else:
        out[0] = deref(write_options.options)


def write_csv(data, output_file, write_options=None,
              MemoryPool memory_pool=None):
    """
    Write record batch or table to a CSV file.

    Parameters
    ----------
    data : pyarrow.RecordBatch or pyarrow.Table
        The data to write.
    output_file : string, path, pyarrow.NativeFile, or file-like object
        The location where to write the CSV data.
    write_options : pyarrow.csv.WriteOptions
        Options to configure writing the CSV data.
    memory_pool : MemoryPool, optional
        Pool for temporary allocations.

    Examples
    --------

    >>> import pyarrow as pa
    >>> from pyarrow import csv

    >>> legs = pa.array([2, 4, 5, 100])
    >>> animals = pa.array(["Flamingo", "Horse", "Brittle stars", "Centipede"])
    >>> entry_date = pa.array(["01/03/2022", "02/03/2022",
    ...                        "03/03/2022", "04/03/2022"])
    >>> table = pa.table([animals, legs, entry_date],
    ...                  names=["animals", "n_legs", "entry"])

    >>> csv.write_csv(table, "animals.csv")

    >>> write_options = csv.WriteOptions(include_header=False)
    >>> csv.write_csv(table, "animals.csv", write_options=write_options)

    >>> write_options = csv.WriteOptions(delimiter=";")
    >>> csv.write_csv(table, "animals.csv", write_options=write_options)
    """
    cdef:
        shared_ptr[COutputStream] stream
        CCSVWriteOptions c_write_options
        CMemoryPool* c_memory_pool
        CRecordBatch* batch
        CTable* table
    _get_write_options(write_options, &c_write_options)

    get_writer(output_file, &stream)
    c_memory_pool = maybe_unbox_memory_pool(memory_pool)
    c_write_options.io_context = CIOContext(c_memory_pool)
    if isinstance(data, RecordBatch):
        batch = pyarrow_unwrap_batch(data).get()
        with nogil:
            check_status(WriteCSV(deref(batch), c_write_options, stream.get()))
    elif isinstance(data, Table):
        table = pyarrow_unwrap_table(data).get()
        with nogil:
            check_status(WriteCSV(deref(table), c_write_options, stream.get()))
    else:
        raise TypeError(f"Expected Table or RecordBatch, got '{type(data)}'")


cdef class CSVWriter(_CRecordBatchWriter):
    """
    Writer to create a CSV file.

    Parameters
    ----------
    sink : str, path, pyarrow.OutputStream or file-like object
        The location where to write the CSV data.
    schema : pyarrow.Schema
        The schema of the data to be written.
    write_options : pyarrow.csv.WriteOptions
        Options to configure writing the CSV data.
    memory_pool : MemoryPool, optional
        Pool for temporary allocations.
    """

    def __init__(self, sink, Schema schema, *,
                 WriteOptions write_options=None, MemoryPool memory_pool=None):
        cdef:
            shared_ptr[COutputStream] c_stream
            shared_ptr[CSchema] c_schema = pyarrow_unwrap_schema(schema)
            CCSVWriteOptions c_write_options
            CMemoryPool* c_memory_pool = maybe_unbox_memory_pool(memory_pool)
        _get_write_options(write_options, &c_write_options)
        c_write_options.io_context = CIOContext(c_memory_pool)
        get_writer(sink, &c_stream)
        with nogil:
            self.writer = GetResultValue(MakeCSVWriter(
                c_stream, c_schema, c_write_options))

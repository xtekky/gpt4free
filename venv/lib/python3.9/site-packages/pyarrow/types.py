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

# Tools for dealing with Arrow type metadata in Python


from pyarrow.lib import (is_boolean_value,  # noqa
                         is_integer_value,
                         is_float_value)

import pyarrow.lib as lib


_SIGNED_INTEGER_TYPES = {lib.Type_INT8, lib.Type_INT16, lib.Type_INT32,
                         lib.Type_INT64}
_UNSIGNED_INTEGER_TYPES = {lib.Type_UINT8, lib.Type_UINT16, lib.Type_UINT32,
                           lib.Type_UINT64}
_INTEGER_TYPES = _SIGNED_INTEGER_TYPES | _UNSIGNED_INTEGER_TYPES
_FLOATING_TYPES = {lib.Type_HALF_FLOAT, lib.Type_FLOAT, lib.Type_DOUBLE}
_DECIMAL_TYPES = {lib.Type_DECIMAL128, lib.Type_DECIMAL256}
_DATE_TYPES = {lib.Type_DATE32, lib.Type_DATE64}
_TIME_TYPES = {lib.Type_TIME32, lib.Type_TIME64}
_INTERVAL_TYPES = {lib.Type_INTERVAL_MONTH_DAY_NANO}
_TEMPORAL_TYPES = ({lib.Type_TIMESTAMP,
                    lib.Type_DURATION} | _TIME_TYPES | _DATE_TYPES |
                   _INTERVAL_TYPES)
_UNION_TYPES = {lib.Type_SPARSE_UNION, lib.Type_DENSE_UNION}
_NESTED_TYPES = {lib.Type_LIST, lib.Type_LARGE_LIST, lib.Type_STRUCT,
                 lib.Type_MAP} | _UNION_TYPES


def is_null(t):
    """
    Return True if value is an instance of a null type.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_NA


def is_boolean(t):
    """
    Return True if value is an instance of a boolean type.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_BOOL


def is_integer(t):
    """
    Return True if value is an instance of any integer type.

    Parameters
    ----------
    t : DataType
    """
    return t.id in _INTEGER_TYPES


def is_signed_integer(t):
    """
    Return True if value is an instance of any signed integer type.

    Parameters
    ----------
    t : DataType
    """
    return t.id in _SIGNED_INTEGER_TYPES


def is_unsigned_integer(t):
    """
    Return True if value is an instance of any unsigned integer type.

    Parameters
    ----------
    t : DataType
    """
    return t.id in _UNSIGNED_INTEGER_TYPES


def is_int8(t):
    """
    Return True if value is an instance of an int8 type.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_INT8


def is_int16(t):
    """
    Return True if value is an instance of an int16 type.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_INT16


def is_int32(t):
    """
    Return True if value is an instance of an int32 type.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_INT32


def is_int64(t):
    """
    Return True if value is an instance of an int64 type.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_INT64


def is_uint8(t):
    """
    Return True if value is an instance of an uint8 type.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_UINT8


def is_uint16(t):
    """
    Return True if value is an instance of an uint16 type.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_UINT16


def is_uint32(t):
    """
    Return True if value is an instance of an uint32 type.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_UINT32


def is_uint64(t):
    """
    Return True if value is an instance of an uint64 type.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_UINT64


def is_floating(t):
    """
    Return True if value is an instance of a floating point numeric type.

    Parameters
    ----------
    t : DataType
    """
    return t.id in _FLOATING_TYPES


def is_float16(t):
    """
    Return True if value is an instance of a float16 (half-precision) type.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_HALF_FLOAT


def is_float32(t):
    """
    Return True if value is an instance of a float32 (single precision) type.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_FLOAT


def is_float64(t):
    """
    Return True if value is an instance of a float64 (double precision) type.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_DOUBLE


def is_list(t):
    """
    Return True if value is an instance of a list type.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_LIST


def is_large_list(t):
    """
    Return True if value is an instance of a large list type.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_LARGE_LIST


def is_fixed_size_list(t):
    """
    Return True if value is an instance of a fixed size list type.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_FIXED_SIZE_LIST


def is_struct(t):
    """
    Return True if value is an instance of a struct type.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_STRUCT


def is_union(t):
    """
    Return True if value is an instance of a union type.

    Parameters
    ----------
    t : DataType
    """
    return t.id in _UNION_TYPES


def is_nested(t):
    """
    Return True if value is an instance of a nested type.

    Parameters
    ----------
    t : DataType
    """
    return t.id in _NESTED_TYPES


def is_temporal(t):
    """
    Return True if value is an instance of date, time, timestamp or duration.

    Parameters
    ----------
    t : DataType
    """
    return t.id in _TEMPORAL_TYPES


def is_timestamp(t):
    """
    Return True if value is an instance of a timestamp type.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_TIMESTAMP


def is_duration(t):
    """
    Return True if value is an instance of a duration type.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_DURATION


def is_time(t):
    """
    Return True if value is an instance of a time type.

    Parameters
    ----------
    t : DataType
    """
    return t.id in _TIME_TYPES


def is_time32(t):
    """
    Return True if value is an instance of a time32 type.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_TIME32


def is_time64(t):
    """
    Return True if value is an instance of a time64 type.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_TIME64


def is_binary(t):
    """
    Return True if value is an instance of a variable-length binary type.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_BINARY


def is_large_binary(t):
    """
    Return True if value is an instance of a large variable-length
    binary type.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_LARGE_BINARY


def is_unicode(t):
    """
    Alias for is_string.

    Parameters
    ----------
    t : DataType
    """
    return is_string(t)


def is_string(t):
    """
    Return True if value is an instance of string (utf8 unicode) type.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_STRING


def is_large_unicode(t):
    """
    Alias for is_large_string.

    Parameters
    ----------
    t : DataType
    """
    return is_large_string(t)


def is_large_string(t):
    """
    Return True if value is an instance of large string (utf8 unicode) type.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_LARGE_STRING


def is_fixed_size_binary(t):
    """
    Return True if value is an instance of a fixed size binary type.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_FIXED_SIZE_BINARY


def is_date(t):
    """
    Return True if value is an instance of a date type.

    Parameters
    ----------
    t : DataType
    """
    return t.id in _DATE_TYPES


def is_date32(t):
    """
    Return True if value is an instance of a date32 (days) type.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_DATE32


def is_date64(t):
    """
    Return True if value is an instance of a date64 (milliseconds) type.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_DATE64


def is_map(t):
    """
    Return True if value is an instance of a map logical type.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_MAP


def is_decimal(t):
    """
    Return True if value is an instance of a decimal type.

    Parameters
    ----------
    t : DataType
    """
    return t.id in _DECIMAL_TYPES


def is_decimal128(t):
    """
    Return True if value is an instance of a decimal type.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_DECIMAL128


def is_decimal256(t):
    """
    Return True if value is an instance of a decimal type.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_DECIMAL256


def is_dictionary(t):
    """
    Return True if value is an instance of a dictionary-encoded type.

    Parameters
    ----------
    t : DataType
    """
    return t.id == lib.Type_DICTIONARY


def is_interval(t):
    """
    Return True if the value is an instance of an interval type.

    Parameters
    ----------
    t : DateType
    """
    return t.id == lib.Type_INTERVAL_MONTH_DAY_NANO


def is_primitive(t):
    """
    Return True if the value is an instance of a primitive type.

    Parameters
    ----------
    t : DataType
    """
    return lib._is_primitive(t.id)

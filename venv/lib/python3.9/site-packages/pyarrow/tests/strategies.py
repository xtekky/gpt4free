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
import sys

import pytest
import hypothesis as h
import hypothesis.strategies as st
import hypothesis.extra.numpy as npst
try:
    import hypothesis.extra.pytz as tzst
except ImportError:
    tzst = None
try:
    import zoneinfo
except ImportError:
    zoneinfo = None
if sys.platform == 'win32':
    try:
        import tzdata  # noqa:F401
    except ImportError:
        zoneinfo = None
import numpy as np

import pyarrow as pa


# TODO(kszucs): alphanum_text, surrogate_text
custom_text = st.text(
    alphabet=st.characters(
        min_codepoint=0x41,
        max_codepoint=0x7E
    )
)

null_type = st.just(pa.null())
bool_type = st.just(pa.bool_())

binary_type = st.just(pa.binary())
string_type = st.just(pa.string())
large_binary_type = st.just(pa.large_binary())
large_string_type = st.just(pa.large_string())
fixed_size_binary_type = st.builds(
    pa.binary,
    st.integers(min_value=0, max_value=16)
)
binary_like_types = st.one_of(
    binary_type,
    string_type,
    large_binary_type,
    large_string_type,
    fixed_size_binary_type
)

signed_integer_types = st.sampled_from([
    pa.int8(),
    pa.int16(),
    pa.int32(),
    pa.int64()
])
unsigned_integer_types = st.sampled_from([
    pa.uint8(),
    pa.uint16(),
    pa.uint32(),
    pa.uint64()
])
integer_types = st.one_of(signed_integer_types, unsigned_integer_types)

floating_types = st.sampled_from([
    pa.float16(),
    pa.float32(),
    pa.float64()
])
decimal128_type = st.builds(
    pa.decimal128,
    precision=st.integers(min_value=1, max_value=38),
    scale=st.integers(min_value=1, max_value=38)
)
decimal256_type = st.builds(
    pa.decimal256,
    precision=st.integers(min_value=1, max_value=76),
    scale=st.integers(min_value=1, max_value=76)
)
numeric_types = st.one_of(integer_types, floating_types,
                          decimal128_type, decimal256_type)

date_types = st.sampled_from([
    pa.date32(),
    pa.date64()
])
time_types = st.sampled_from([
    pa.time32('s'),
    pa.time32('ms'),
    pa.time64('us'),
    pa.time64('ns')
])

if tzst and zoneinfo:
    timezones = st.one_of(st.none(), tzst.timezones(), st.timezones())
elif tzst:
    timezones = st.one_of(st.none(), tzst.timezones())
elif zoneinfo:
    timezones = st.one_of(st.none(), st.timezones())
else:
    timezones = st.none()
timestamp_types = st.builds(
    pa.timestamp,
    unit=st.sampled_from(['s', 'ms', 'us', 'ns']),
    tz=timezones
)
duration_types = st.builds(
    pa.duration,
    st.sampled_from(['s', 'ms', 'us', 'ns'])
)
interval_types = st.just(pa.month_day_nano_interval())
temporal_types = st.one_of(
    date_types,
    time_types,
    timestamp_types,
    duration_types,
    interval_types
)

primitive_types = st.one_of(
    null_type,
    bool_type,
    numeric_types,
    temporal_types,
    binary_like_types
)

metadata = st.dictionaries(st.text(), st.text())


@st.composite
def fields(draw, type_strategy=primitive_types):
    name = draw(custom_text)
    typ = draw(type_strategy)
    if pa.types.is_null(typ):
        nullable = True
    else:
        nullable = draw(st.booleans())
    meta = draw(metadata)
    return pa.field(name, type=typ, nullable=nullable, metadata=meta)


def list_types(item_strategy=primitive_types):
    return (
        st.builds(pa.list_, item_strategy) |
        st.builds(pa.large_list, item_strategy) |
        st.builds(
            pa.list_,
            item_strategy,
            st.integers(min_value=0, max_value=16)
        )
    )


@st.composite
def struct_types(draw, item_strategy=primitive_types):
    fields_strategy = st.lists(fields(item_strategy))
    fields_rendered = draw(fields_strategy)
    field_names = [field.name for field in fields_rendered]
    # check that field names are unique, see ARROW-9997
    h.assume(len(set(field_names)) == len(field_names))
    return pa.struct(fields_rendered)


def dictionary_types(key_strategy=None, value_strategy=None):
    key_strategy = key_strategy or signed_integer_types
    value_strategy = value_strategy or st.one_of(
        bool_type,
        integer_types,
        st.sampled_from([pa.float32(), pa.float64()]),
        binary_type,
        string_type,
        fixed_size_binary_type,
    )
    return st.builds(pa.dictionary, key_strategy, value_strategy)


@st.composite
def map_types(draw, key_strategy=primitive_types,
              item_strategy=primitive_types):
    key_type = draw(key_strategy)
    h.assume(not pa.types.is_null(key_type))
    value_type = draw(item_strategy)
    return pa.map_(key_type, value_type)


# union type
# extension type


def schemas(type_strategy=primitive_types, max_fields=None):
    children = st.lists(fields(type_strategy), max_size=max_fields)
    return st.builds(pa.schema, children)


all_types = st.deferred(
    lambda: (
        primitive_types |
        list_types() |
        struct_types() |
        dictionary_types() |
        map_types() |
        list_types(all_types) |
        struct_types(all_types)
    )
)
all_fields = fields(all_types)
all_schemas = schemas(all_types)


_default_array_sizes = st.integers(min_value=0, max_value=20)


@st.composite
def _pylist(draw, value_type, size, nullable=True):
    arr = draw(arrays(value_type, size=size, nullable=False))
    return arr.to_pylist()


@st.composite
def _pymap(draw, key_type, value_type, size, nullable=True):
    length = draw(size)
    keys = draw(_pylist(key_type, size=length, nullable=False))
    values = draw(_pylist(value_type, size=length, nullable=nullable))
    return list(zip(keys, values))


@st.composite
def arrays(draw, type, size=None, nullable=True):
    if isinstance(type, st.SearchStrategy):
        ty = draw(type)
    elif isinstance(type, pa.DataType):
        ty = type
    else:
        raise TypeError('Type must be a pyarrow DataType')

    if isinstance(size, st.SearchStrategy):
        size = draw(size)
    elif size is None:
        size = draw(_default_array_sizes)
    elif not isinstance(size, int):
        raise TypeError('Size must be an integer')

    if pa.types.is_null(ty):
        h.assume(nullable)
        value = st.none()
    elif pa.types.is_boolean(ty):
        value = st.booleans()
    elif pa.types.is_integer(ty):
        values = draw(npst.arrays(ty.to_pandas_dtype(), shape=(size,)))
        return pa.array(values, type=ty)
    elif pa.types.is_floating(ty):
        values = draw(npst.arrays(ty.to_pandas_dtype(), shape=(size,)))
        # Workaround ARROW-4952: no easy way to assert array equality
        # in a NaN-tolerant way.
        values[np.isnan(values)] = -42.0
        return pa.array(values, type=ty)
    elif pa.types.is_decimal(ty):
        # TODO(kszucs): properly limit the precision
        # value = st.decimals(places=type.scale, allow_infinity=False)
        h.reject()
    elif pa.types.is_time(ty):
        value = st.times()
    elif pa.types.is_date(ty):
        value = st.dates()
    elif pa.types.is_timestamp(ty):
        if zoneinfo is None:
            pytest.skip('no module named zoneinfo (or tzdata on Windows)')
        if ty.tz is None:
            pytest.skip('requires timezone not None')
        min_int64 = -(2**63)
        max_int64 = 2**63 - 1
        min_datetime = datetime.datetime.fromtimestamp(
            min_int64 // 10**9) + datetime.timedelta(hours=12)
        max_datetime = datetime.datetime.fromtimestamp(
            max_int64 // 10**9) - datetime.timedelta(hours=12)
        try:
            offset = ty.tz.split(":")
            offset_hours = int(offset[0])
            offset_min = int(offset[1])
            tz = datetime.timedelta(hours=offset_hours, minutes=offset_min)
        except ValueError:
            tz = zoneinfo.ZoneInfo(ty.tz)
        value = st.datetimes(timezones=st.just(tz), min_value=min_datetime,
                             max_value=max_datetime)
    elif pa.types.is_duration(ty):
        value = st.timedeltas()
    elif pa.types.is_interval(ty):
        value = st.timedeltas()
    elif pa.types.is_binary(ty) or pa.types.is_large_binary(ty):
        value = st.binary()
    elif pa.types.is_string(ty) or pa.types.is_large_string(ty):
        value = st.text()
    elif pa.types.is_fixed_size_binary(ty):
        value = st.binary(min_size=ty.byte_width, max_size=ty.byte_width)
    elif pa.types.is_list(ty):
        value = _pylist(ty.value_type, size=size, nullable=nullable)
    elif pa.types.is_large_list(ty):
        value = _pylist(ty.value_type, size=size, nullable=nullable)
    elif pa.types.is_fixed_size_list(ty):
        value = _pylist(ty.value_type, size=ty.list_size, nullable=nullable)
    elif pa.types.is_dictionary(ty):
        values = _pylist(ty.value_type, size=size, nullable=nullable)
        return pa.array(draw(values), type=ty)
    elif pa.types.is_map(ty):
        value = _pymap(ty.key_type, ty.item_type, size=_default_array_sizes,
                       nullable=nullable)
    elif pa.types.is_struct(ty):
        h.assume(len(ty) > 0)
        fields, child_arrays = [], []
        for field in ty:
            fields.append(field)
            child_arrays.append(draw(arrays(field.type, size=size)))
        return pa.StructArray.from_arrays(child_arrays, fields=fields)
    else:
        raise NotImplementedError(ty)

    if nullable:
        value = st.one_of(st.none(), value)
    values = st.lists(value, min_size=size, max_size=size)

    return pa.array(draw(values), type=ty)


@st.composite
def chunked_arrays(draw, type, min_chunks=0, max_chunks=None, chunk_size=None):
    if isinstance(type, st.SearchStrategy):
        type = draw(type)

    # TODO(kszucs): remove it, field metadata is not kept
    h.assume(not pa.types.is_struct(type))

    chunk = arrays(type, size=chunk_size)
    chunks = st.lists(chunk, min_size=min_chunks, max_size=max_chunks)

    return pa.chunked_array(draw(chunks), type=type)


@st.composite
def record_batches(draw, type, rows=None, max_fields=None):
    if isinstance(rows, st.SearchStrategy):
        rows = draw(rows)
    elif rows is None:
        rows = draw(_default_array_sizes)
    elif not isinstance(rows, int):
        raise TypeError('Rows must be an integer')

    schema = draw(schemas(type, max_fields=max_fields))
    children = [draw(arrays(field.type, size=rows)) for field in schema]
    # TODO(kszucs): the names and schema arguments are not consistent with
    #               Table.from_array's arguments
    return pa.RecordBatch.from_arrays(children, names=schema)


@st.composite
def tables(draw, type, rows=None, max_fields=None):
    if isinstance(rows, st.SearchStrategy):
        rows = draw(rows)
    elif rows is None:
        rows = draw(_default_array_sizes)
    elif not isinstance(rows, int):
        raise TypeError('Rows must be an integer')

    schema = draw(schemas(type, max_fields=max_fields))
    children = [draw(arrays(field.type, size=rows)) for field in schema]
    return pa.Table.from_arrays(children, schema=schema)


all_arrays = arrays(all_types)
all_chunked_arrays = chunked_arrays(all_types)
all_record_batches = record_batches(all_types)
all_tables = tables(all_types)


# Define the same rules as above for pandas tests by excluding certain types
# from the generation because of known issues.

pandas_compatible_primitive_types = st.one_of(
    null_type,
    bool_type,
    integer_types,
    st.sampled_from([pa.float32(), pa.float64()]),
    decimal128_type,
    date_types,
    time_types,
    # Need to exclude timestamp and duration types otherwise hypothesis
    # discovers ARROW-10210
    # timestamp_types,
    # duration_types
    interval_types,
    binary_type,
    string_type,
    large_binary_type,
    large_string_type,
)

# Need to exclude floating point types otherwise hypothesis discovers
# ARROW-10211
pandas_compatible_dictionary_value_types = st.one_of(
    bool_type,
    integer_types,
    binary_type,
    string_type,
    fixed_size_binary_type,
)


def pandas_compatible_list_types(
    item_strategy=pandas_compatible_primitive_types
):
    # Need to exclude fixed size list type otherwise hypothesis discovers
    # ARROW-10194
    return (
        st.builds(pa.list_, item_strategy) |
        st.builds(pa.large_list, item_strategy)
    )


pandas_compatible_types = st.deferred(
    lambda: st.one_of(
        pandas_compatible_primitive_types,
        pandas_compatible_list_types(pandas_compatible_primitive_types),
        struct_types(pandas_compatible_primitive_types),
        dictionary_types(
            value_strategy=pandas_compatible_dictionary_value_types
        ),
        pandas_compatible_list_types(pandas_compatible_types),
        struct_types(pandas_compatible_types)
    )
)

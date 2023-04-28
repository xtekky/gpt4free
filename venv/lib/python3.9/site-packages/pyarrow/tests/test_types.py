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
from collections.abc import Iterator
from functools import partial
import datetime
import sys

import pickle
import pytest
import hypothesis as h
import hypothesis.strategies as st
try:
    import hypothesis.extra.pytz as tzst
except ImportError:
    tzst = None
import weakref

import numpy as np
import pyarrow as pa
import pyarrow.types as types
import pyarrow.tests.strategies as past


def get_many_types():
    # returning them from a function is required because of pa.dictionary
    # type holds a pyarrow array and test_array.py::test_toal_bytes_allocated
    # checks that the default memory pool has zero allocated bytes
    return (
        pa.null(),
        pa.bool_(),
        pa.int32(),
        pa.time32('s'),
        pa.time64('us'),
        pa.date32(),
        pa.timestamp('us'),
        pa.timestamp('us', tz='UTC'),
        pa.timestamp('us', tz='Europe/Paris'),
        pa.duration('s'),
        pa.float16(),
        pa.float32(),
        pa.float64(),
        pa.decimal128(19, 4),
        pa.decimal256(76, 38),
        pa.string(),
        pa.binary(),
        pa.binary(10),
        pa.large_string(),
        pa.large_binary(),
        pa.list_(pa.int32()),
        pa.list_(pa.int32(), 2),
        pa.large_list(pa.uint16()),
        pa.map_(pa.string(), pa.int32()),
        pa.map_(pa.field('key', pa.int32(), nullable=False),
                pa.field('value', pa.int32())),
        pa.struct([pa.field('a', pa.int32()),
                   pa.field('b', pa.int8()),
                   pa.field('c', pa.string())]),
        pa.struct([pa.field('a', pa.int32(), nullable=False),
                   pa.field('b', pa.int8(), nullable=False),
                   pa.field('c', pa.string())]),
        pa.union([pa.field('a', pa.binary(10)),
                  pa.field('b', pa.string())], mode=pa.lib.UnionMode_DENSE),
        pa.union([pa.field('a', pa.binary(10)),
                  pa.field('b', pa.string())], mode=pa.lib.UnionMode_DENSE,
                 type_codes=[4, 8]),
        pa.union([pa.field('a', pa.binary(10)),
                  pa.field('b', pa.string())], mode=pa.lib.UnionMode_SPARSE),
        pa.union([pa.field('a', pa.binary(10), nullable=False),
                  pa.field('b', pa.string())], mode=pa.lib.UnionMode_SPARSE),
        pa.dictionary(pa.int32(), pa.string())
    )


def test_is_boolean():
    assert types.is_boolean(pa.bool_())
    assert not types.is_boolean(pa.int8())


def test_is_integer():
    signed_ints = [pa.int8(), pa.int16(), pa.int32(), pa.int64()]
    unsigned_ints = [pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64()]

    for t in signed_ints + unsigned_ints:
        assert types.is_integer(t)

    for t in signed_ints:
        assert types.is_signed_integer(t)
        assert not types.is_unsigned_integer(t)

    for t in unsigned_ints:
        assert types.is_unsigned_integer(t)
        assert not types.is_signed_integer(t)

    assert not types.is_integer(pa.float32())
    assert not types.is_signed_integer(pa.float32())


def test_is_floating():
    for t in [pa.float16(), pa.float32(), pa.float64()]:
        assert types.is_floating(t)

    assert not types.is_floating(pa.int32())


def test_is_null():
    assert types.is_null(pa.null())
    assert not types.is_null(pa.list_(pa.int32()))


def test_null_field_may_not_be_non_nullable():
    # ARROW-7273
    with pytest.raises(ValueError):
        pa.field('f0', pa.null(), nullable=False)


def test_is_decimal():
    decimal128 = pa.decimal128(19, 4)
    decimal256 = pa.decimal256(76, 38)
    int32 = pa.int32()

    assert types.is_decimal(decimal128)
    assert types.is_decimal(decimal256)
    assert not types.is_decimal(int32)

    assert types.is_decimal128(decimal128)
    assert not types.is_decimal128(decimal256)
    assert not types.is_decimal128(int32)

    assert not types.is_decimal256(decimal128)
    assert types.is_decimal256(decimal256)
    assert not types.is_decimal256(int32)


def test_is_list():
    a = pa.list_(pa.int32())
    b = pa.large_list(pa.int32())
    c = pa.list_(pa.int32(), 3)

    assert types.is_list(a)
    assert not types.is_large_list(a)
    assert not types.is_fixed_size_list(a)
    assert types.is_large_list(b)
    assert not types.is_list(b)
    assert not types.is_fixed_size_list(b)
    assert types.is_fixed_size_list(c)
    assert not types.is_list(c)
    assert not types.is_large_list(c)

    assert not types.is_list(pa.int32())


def test_is_map():
    m = pa.map_(pa.utf8(), pa.int32())

    assert types.is_map(m)
    assert not types.is_map(pa.int32())

    fields = pa.map_(pa.field('key_name', pa.utf8(), nullable=False),
                     pa.field('value_name', pa.int32()))
    assert types.is_map(fields)

    entries_type = pa.struct([pa.field('key', pa.int8()),
                              pa.field('value', pa.int8())])
    list_type = pa.list_(entries_type)
    assert not types.is_map(list_type)


def test_is_dictionary():
    assert types.is_dictionary(pa.dictionary(pa.int32(), pa.string()))
    assert not types.is_dictionary(pa.int32())


def test_is_nested_or_struct():
    struct_ex = pa.struct([pa.field('a', pa.int32()),
                           pa.field('b', pa.int8()),
                           pa.field('c', pa.string())])

    assert types.is_struct(struct_ex)
    assert not types.is_struct(pa.list_(pa.int32()))

    assert types.is_nested(struct_ex)
    assert types.is_nested(pa.list_(pa.int32()))
    assert types.is_nested(pa.large_list(pa.int32()))
    assert not types.is_nested(pa.int32())


def test_is_union():
    for mode in [pa.lib.UnionMode_SPARSE, pa.lib.UnionMode_DENSE]:
        assert types.is_union(pa.union([pa.field('a', pa.int32()),
                                        pa.field('b', pa.int8()),
                                        pa.field('c', pa.string())],
                                       mode=mode))
    assert not types.is_union(pa.list_(pa.int32()))


# TODO(wesm): is_map, once implemented


def test_is_binary_string():
    assert types.is_binary(pa.binary())
    assert not types.is_binary(pa.string())
    assert not types.is_binary(pa.large_binary())
    assert not types.is_binary(pa.large_string())

    assert types.is_string(pa.string())
    assert types.is_unicode(pa.string())
    assert not types.is_string(pa.binary())
    assert not types.is_string(pa.large_string())
    assert not types.is_string(pa.large_binary())

    assert types.is_large_binary(pa.large_binary())
    assert not types.is_large_binary(pa.large_string())
    assert not types.is_large_binary(pa.binary())
    assert not types.is_large_binary(pa.string())

    assert types.is_large_string(pa.large_string())
    assert not types.is_large_string(pa.large_binary())
    assert not types.is_large_string(pa.string())
    assert not types.is_large_string(pa.binary())

    assert types.is_fixed_size_binary(pa.binary(5))
    assert not types.is_fixed_size_binary(pa.binary())


def test_is_temporal_date_time_timestamp():
    date_types = [pa.date32(), pa.date64()]
    time_types = [pa.time32('s'), pa.time64('ns')]
    timestamp_types = [pa.timestamp('ms')]
    duration_types = [pa.duration('ms')]
    interval_types = [pa.month_day_nano_interval()]

    for case in (date_types + time_types + timestamp_types + duration_types +
                 interval_types):
        assert types.is_temporal(case)

    for case in date_types:
        assert types.is_date(case)
        assert not types.is_time(case)
        assert not types.is_timestamp(case)
        assert not types.is_duration(case)
        assert not types.is_interval(case)

    for case in time_types:
        assert types.is_time(case)
        assert not types.is_date(case)
        assert not types.is_timestamp(case)
        assert not types.is_duration(case)
        assert not types.is_interval(case)

    for case in timestamp_types:
        assert types.is_timestamp(case)
        assert not types.is_date(case)
        assert not types.is_time(case)
        assert not types.is_duration(case)
        assert not types.is_interval(case)

    for case in duration_types:
        assert types.is_duration(case)
        assert not types.is_date(case)
        assert not types.is_time(case)
        assert not types.is_timestamp(case)
        assert not types.is_interval(case)

    for case in interval_types:
        assert types.is_interval(case)
        assert not types.is_date(case)
        assert not types.is_time(case)
        assert not types.is_timestamp(case)

    assert not types.is_temporal(pa.int32())


def test_is_primitive():
    assert types.is_primitive(pa.int32())
    assert not types.is_primitive(pa.list_(pa.int32()))


@pytest.mark.parametrize(('tz', 'expected'), [
    (datetime.timezone.utc, 'UTC'),
    (datetime.timezone(datetime.timedelta(hours=1, minutes=30)), '+01:30')
])
def test_tzinfo_to_string(tz, expected):
    assert pa.lib.tzinfo_to_string(tz) == expected


def test_pytz_tzinfo_to_string():
    pytz = pytest.importorskip("pytz")

    tz = [pytz.utc, pytz.timezone('Europe/Paris')]
    expected = ['UTC', 'Europe/Paris']
    assert [pa.lib.tzinfo_to_string(i) for i in tz] == expected

    # StaticTzInfo.tzname returns with '-09' so we need to infer the timezone's
    # name from the tzinfo.zone attribute
    tz = [pytz.timezone('Etc/GMT-9'), pytz.FixedOffset(180)]
    expected = ['Etc/GMT-9', '+03:00']
    assert [pa.lib.tzinfo_to_string(i) for i in tz] == expected


def test_dateutil_tzinfo_to_string():
    pytest.importorskip("dateutil")
    import dateutil.tz

    tz = dateutil.tz.UTC
    assert pa.lib.tzinfo_to_string(tz) == 'UTC'
    tz = dateutil.tz.gettz('Europe/Paris')
    assert pa.lib.tzinfo_to_string(tz) == 'Europe/Paris'


def test_zoneinfo_tzinfo_to_string():
    zoneinfo = pytest.importorskip('zoneinfo')
    if sys.platform == 'win32':
        # zoneinfo requires an additional dependency On Windows
        # tzdata provides IANA time zone data
        pytest.importorskip('tzdata')

    tz = zoneinfo.ZoneInfo('UTC')
    assert pa.lib.tzinfo_to_string(tz) == 'UTC'
    tz = zoneinfo.ZoneInfo('Europe/Paris')
    assert pa.lib.tzinfo_to_string(tz) == 'Europe/Paris'


def test_tzinfo_to_string_errors():
    msg = "Not an instance of datetime.tzinfo"
    with pytest.raises(TypeError):
        pa.lib.tzinfo_to_string("Europe/Budapest")

    if sys.version_info >= (3, 8):
        # before 3.8 it was only possible to create timezone objects with whole
        # number of minutes
        tz = datetime.timezone(datetime.timedelta(hours=1, seconds=30))
        msg = "Offset must represent whole number of minutes"
        with pytest.raises(ValueError, match=msg):
            pa.lib.tzinfo_to_string(tz)


if tzst:
    timezones = tzst.timezones()
else:
    timezones = st.none()


@h.given(timezones)
def test_pytz_timezone_roundtrip(tz):
    if tz is None:
        pytest.skip('requires timezone not None')
    timezone_string = pa.lib.tzinfo_to_string(tz)
    timezone_tzinfo = pa.lib.string_to_tzinfo(timezone_string)
    assert timezone_tzinfo == tz


def test_convert_custom_tzinfo_objects_to_string():
    class CorrectTimezone1(datetime.tzinfo):
        """
        Conversion is using utcoffset()
        """

        def tzname(self, dt):
            return None

        def utcoffset(self, dt):
            return datetime.timedelta(hours=-3, minutes=30)

    class CorrectTimezone2(datetime.tzinfo):
        """
        Conversion is using tzname()
        """

        def tzname(self, dt):
            return "+03:00"

        def utcoffset(self, dt):
            return datetime.timedelta(hours=3)

    class BuggyTimezone1(datetime.tzinfo):
        """
        Unable to infer name or offset
        """

        def tzname(self, dt):
            return None

        def utcoffset(self, dt):
            return None

    class BuggyTimezone2(datetime.tzinfo):
        """
        Wrong offset type
        """

        def tzname(self, dt):
            return None

        def utcoffset(self, dt):
            return "one hour"

    class BuggyTimezone3(datetime.tzinfo):
        """
        Wrong timezone name type
        """

        def tzname(self, dt):
            return 240

        def utcoffset(self, dt):
            return None

    assert pa.lib.tzinfo_to_string(CorrectTimezone1()) == "-02:30"
    assert pa.lib.tzinfo_to_string(CorrectTimezone2()) == "+03:00"

    msg = (r"Object returned by tzinfo.utcoffset\(None\) is not an instance "
           r"of datetime.timedelta")
    for wrong in [BuggyTimezone1(), BuggyTimezone2(), BuggyTimezone3()]:
        with pytest.raises(ValueError, match=msg):
            pa.lib.tzinfo_to_string(wrong)


def test_string_to_tzinfo():
    string = ['UTC', 'Europe/Paris', '+03:00', '+01:30', '-02:00']
    try:
        import pytz
        expected = [pytz.utc, pytz.timezone('Europe/Paris'),
                    pytz.FixedOffset(180), pytz.FixedOffset(90),
                    pytz.FixedOffset(-120)]
        result = [pa.lib.string_to_tzinfo(i) for i in string]
        assert result == expected

    except ImportError:
        try:
            import zoneinfo
            expected = [zoneinfo.ZoneInfo(key='UTC'),
                        zoneinfo.ZoneInfo(key='Europe/Paris'),
                        datetime.timezone(datetime.timedelta(hours=3)),
                        datetime.timezone(
                            datetime.timedelta(hours=1, minutes=30)),
                        datetime.timezone(-datetime.timedelta(hours=2))]
            result = [pa.lib.string_to_tzinfo(i) for i in string]
            assert result == expected

        except ImportError:
            pytest.skip('requires pytz or zoneinfo to be installed')


def test_timezone_string_roundtrip_pytz():
    pytz = pytest.importorskip("pytz")

    tz = [pytz.FixedOffset(90), pytz.FixedOffset(-90),
          pytz.utc, pytz.timezone('America/New_York')]
    name = ['+01:30', '-01:30', 'UTC', 'America/New_York']

    assert [pa.lib.tzinfo_to_string(i) for i in tz] == name
    assert [pa.lib.string_to_tzinfo(i)for i in name] == tz


def test_timestamp():
    for unit in ('s', 'ms', 'us', 'ns'):
        for tz in (None, 'UTC', 'Europe/Paris'):
            ty = pa.timestamp(unit, tz=tz)
            assert ty.unit == unit
            assert ty.tz == tz

    for invalid_unit in ('m', 'arbit', 'rary'):
        with pytest.raises(ValueError, match='Invalid time unit'):
            pa.timestamp(invalid_unit)


def test_time32_units():
    for valid_unit in ('s', 'ms'):
        ty = pa.time32(valid_unit)
        assert ty.unit == valid_unit

    for invalid_unit in ('m', 'us', 'ns'):
        error_msg = 'Invalid time unit for time32: {!r}'.format(invalid_unit)
        with pytest.raises(ValueError, match=error_msg):
            pa.time32(invalid_unit)


def test_time64_units():
    for valid_unit in ('us', 'ns'):
        ty = pa.time64(valid_unit)
        assert ty.unit == valid_unit

    for invalid_unit in ('m', 's', 'ms'):
        error_msg = 'Invalid time unit for time64: {!r}'.format(invalid_unit)
        with pytest.raises(ValueError, match=error_msg):
            pa.time64(invalid_unit)


def test_duration():
    for unit in ('s', 'ms', 'us', 'ns'):
        ty = pa.duration(unit)
        assert ty.unit == unit

    for invalid_unit in ('m', 'arbit', 'rary'):
        with pytest.raises(ValueError, match='Invalid time unit'):
            pa.duration(invalid_unit)


def test_list_type():
    ty = pa.list_(pa.int64())
    assert isinstance(ty, pa.ListType)
    assert ty.value_type == pa.int64()
    assert ty.value_field == pa.field("item", pa.int64(), nullable=True)

    # nullability matters in comparison
    ty_non_nullable = pa.list_(pa.field("item", pa.int64(), nullable=False))
    assert ty != ty_non_nullable

    # field names don't matter by default
    ty_named = pa.list_(pa.field("element", pa.int64()))
    assert ty == ty_named
    assert not ty.equals(ty_named, check_metadata=True)

    # metadata doesn't matter by default
    ty_metadata = pa.list_(
        pa.field("item", pa.int64(), metadata={"hello": "world"}))
    assert ty == ty_metadata
    assert not ty.equals(ty_metadata, check_metadata=True)

    with pytest.raises(TypeError):
        pa.list_(None)


def test_large_list_type():
    ty = pa.large_list(pa.utf8())
    assert isinstance(ty, pa.LargeListType)
    assert ty.value_type == pa.utf8()
    assert ty.value_field == pa.field("item", pa.utf8(), nullable=True)

    with pytest.raises(TypeError):
        pa.large_list(None)


def test_map_type():
    ty = pa.map_(pa.utf8(), pa.int32())
    assert isinstance(ty, pa.MapType)
    assert ty.key_type == pa.utf8()
    assert ty.key_field == pa.field("key", pa.utf8(), nullable=False)
    assert ty.item_type == pa.int32()
    assert ty.item_field == pa.field("value", pa.int32(), nullable=True)

    # nullability matters in comparison
    ty_non_nullable = pa.map_(pa.utf8(), pa.field(
        "value", pa.int32(), nullable=False))
    assert ty != ty_non_nullable

    # field names don't matter by default
    ty_named = pa.map_(pa.field("x", pa.utf8(), nullable=False),
                       pa.field("y", pa.int32()))
    assert ty == ty_named
    assert not ty.equals(ty_named, check_metadata=True)

    # metadata doesn't matter by default
    ty_metadata = pa.map_(pa.utf8(), pa.field(
        "value", pa.int32(), metadata={"hello": "world"}))
    assert ty == ty_metadata
    assert not ty.equals(ty_metadata, check_metadata=True)

    with pytest.raises(TypeError):
        pa.map_(None)
    with pytest.raises(TypeError):
        pa.map_(pa.int32(), None)
    with pytest.raises(TypeError):
        pa.map_(pa.field("name", pa.string(), nullable=True), pa.int64())


def test_fixed_size_list_type():
    ty = pa.list_(pa.float64(), 2)
    assert isinstance(ty, pa.FixedSizeListType)
    assert ty.value_type == pa.float64()
    assert ty.value_field == pa.field("item", pa.float64(), nullable=True)
    assert ty.list_size == 2

    with pytest.raises(ValueError):
        pa.list_(pa.float64(), -2)


def test_struct_type():
    fields = [
        # Duplicate field name on purpose
        pa.field('a', pa.int64()),
        pa.field('a', pa.int32()),
        pa.field('b', pa.int32())
    ]
    ty = pa.struct(fields)

    assert len(ty) == ty.num_fields == 3
    assert list(ty) == fields
    assert ty[0].name == 'a'
    assert ty[2].type == pa.int32()
    with pytest.raises(IndexError):
        assert ty[3]

    assert ty['b'] == ty[2]

    assert ty['b'] == ty.field('b')

    assert ty[2] == ty.field(2)

    # Not found
    with pytest.raises(KeyError):
        ty['c']

    with pytest.raises(KeyError):
        ty.field('c')

    # Neither integer nor string
    with pytest.raises(TypeError):
        ty[None]

    with pytest.raises(TypeError):
        ty.field(None)

    for a, b in zip(ty, fields):
        a == b

    # Construct from list of tuples
    ty = pa.struct([('a', pa.int64()),
                    ('a', pa.int32()),
                    ('b', pa.int32())])
    assert list(ty) == fields
    for a, b in zip(ty, fields):
        a == b

    # Construct from mapping
    fields = [pa.field('a', pa.int64()),
              pa.field('b', pa.int32())]
    ty = pa.struct(OrderedDict([('a', pa.int64()),
                                ('b', pa.int32())]))
    assert list(ty) == fields
    for a, b in zip(ty, fields):
        a == b

    # Invalid args
    with pytest.raises(TypeError):
        pa.struct([('a', None)])


def test_struct_duplicate_field_names():
    fields = [
        pa.field('a', pa.int64()),
        pa.field('b', pa.int32()),
        pa.field('a', pa.int32())
    ]
    ty = pa.struct(fields)

    # Duplicate
    with pytest.warns(UserWarning):
        with pytest.raises(KeyError):
            ty['a']

    # StructType::GetFieldIndex
    assert ty.get_field_index('a') == -1

    # StructType::GetAllFieldIndices
    assert ty.get_all_field_indices('a') == [0, 2]


def test_union_type():
    def check_fields(ty, fields):
        assert ty.num_fields == len(fields)
        assert [ty[i] for i in range(ty.num_fields)] == fields
        assert [ty.field(i) for i in range(ty.num_fields)] == fields

    fields = [pa.field('x', pa.list_(pa.int32())),
              pa.field('y', pa.binary())]
    type_codes = [5, 9]

    sparse_factories = [
        partial(pa.union, mode='sparse'),
        partial(pa.union, mode=pa.lib.UnionMode_SPARSE),
        pa.sparse_union,
    ]

    dense_factories = [
        partial(pa.union, mode='dense'),
        partial(pa.union, mode=pa.lib.UnionMode_DENSE),
        pa.dense_union,
    ]

    for factory in sparse_factories:
        ty = factory(fields)
        assert isinstance(ty, pa.SparseUnionType)
        assert ty.mode == 'sparse'
        check_fields(ty, fields)
        assert ty.type_codes == [0, 1]
        ty = factory(fields, type_codes=type_codes)
        assert ty.mode == 'sparse'
        check_fields(ty, fields)
        assert ty.type_codes == type_codes
        # Invalid number of type codes
        with pytest.raises(ValueError):
            factory(fields, type_codes=type_codes[1:])

    for factory in dense_factories:
        ty = factory(fields)
        assert isinstance(ty, pa.DenseUnionType)
        assert ty.mode == 'dense'
        check_fields(ty, fields)
        assert ty.type_codes == [0, 1]
        ty = factory(fields, type_codes=type_codes)
        assert ty.mode == 'dense'
        check_fields(ty, fields)
        assert ty.type_codes == type_codes
        # Invalid number of type codes
        with pytest.raises(ValueError):
            factory(fields, type_codes=type_codes[1:])

    for mode in ('unknown', 2):
        with pytest.raises(ValueError, match='Invalid union mode'):
            pa.union(fields, mode=mode)


def test_dictionary_type():
    ty0 = pa.dictionary(pa.int32(), pa.string())
    assert ty0.index_type == pa.int32()
    assert ty0.value_type == pa.string()
    assert ty0.ordered is False

    ty1 = pa.dictionary(pa.int8(), pa.float64(), ordered=True)
    assert ty1.index_type == pa.int8()
    assert ty1.value_type == pa.float64()
    assert ty1.ordered is True

    # construct from non-arrow objects
    ty2 = pa.dictionary('int8', 'string')
    assert ty2.index_type == pa.int8()
    assert ty2.value_type == pa.string()
    assert ty2.ordered is False

    # allow unsigned integers for index type
    ty3 = pa.dictionary(pa.uint32(), pa.string())
    assert ty3.index_type == pa.uint32()
    assert ty3.value_type == pa.string()
    assert ty3.ordered is False

    # invalid index type raises
    with pytest.raises(TypeError):
        pa.dictionary(pa.string(), pa.int64())


def test_dictionary_ordered_equals():
    # Python side checking of ARROW-6345
    d1 = pa.dictionary('int32', 'binary', ordered=True)
    d2 = pa.dictionary('int32', 'binary', ordered=False)
    d3 = pa.dictionary('int8', 'binary', ordered=True)
    d4 = pa.dictionary('int32', 'binary', ordered=True)

    assert not d1.equals(d2)
    assert not d1.equals(d3)
    assert d1.equals(d4)


def test_types_hashable():
    many_types = get_many_types()
    in_dict = {}
    for i, type_ in enumerate(many_types):
        assert hash(type_) == hash(type_)
        in_dict[type_] = i
    assert len(in_dict) == len(many_types)
    for i, type_ in enumerate(many_types):
        assert in_dict[type_] == i


def test_types_picklable():
    for ty in get_many_types():
        data = pickle.dumps(ty)
        assert pickle.loads(data) == ty


def test_types_weakref():
    for ty in get_many_types():
        wr = weakref.ref(ty)
        assert wr() is not None
        # Note that ty may be a singleton and therefore outlive this loop

    wr = weakref.ref(pa.int32())
    assert wr() is not None  # singleton
    wr = weakref.ref(pa.list_(pa.int32()))
    assert wr() is None  # not a singleton


def test_fields_hashable():
    in_dict = {}
    fields = [pa.field('a', pa.int32()),
              pa.field('a', pa.int64()),
              pa.field('a', pa.int64(), nullable=False),
              pa.field('b', pa.int32()),
              pa.field('b', pa.int32(), nullable=False)]
    for i, field in enumerate(fields):
        in_dict[field] = i
    assert len(in_dict) == len(fields)
    for i, field in enumerate(fields):
        assert in_dict[field] == i


def test_fields_weakrefable():
    field = pa.field('a', pa.int32())
    wr = weakref.ref(field)
    assert wr() is not None
    del field
    assert wr() is None


@pytest.mark.parametrize('t,check_func', [
    (pa.date32(), types.is_date32),
    (pa.date64(), types.is_date64),
    (pa.time32('s'), types.is_time32),
    (pa.time64('ns'), types.is_time64),
    (pa.int8(), types.is_int8),
    (pa.int16(), types.is_int16),
    (pa.int32(), types.is_int32),
    (pa.int64(), types.is_int64),
    (pa.uint8(), types.is_uint8),
    (pa.uint16(), types.is_uint16),
    (pa.uint32(), types.is_uint32),
    (pa.uint64(), types.is_uint64),
    (pa.float16(), types.is_float16),
    (pa.float32(), types.is_float32),
    (pa.float64(), types.is_float64)
])
def test_exact_primitive_types(t, check_func):
    assert check_func(t)


def test_type_id():
    # enum values are not exposed publicly
    for ty in get_many_types():
        assert isinstance(ty.id, int)


def test_bit_width():
    for ty, expected in [(pa.bool_(), 1),
                         (pa.int8(), 8),
                         (pa.uint32(), 32),
                         (pa.float16(), 16),
                         (pa.decimal128(19, 4), 128),
                         (pa.decimal256(76, 38), 256),
                         (pa.binary(42), 42 * 8)]:
        assert ty.bit_width == expected
    for ty in [pa.binary(), pa.string(), pa.list_(pa.int16())]:
        with pytest.raises(ValueError, match="fixed width"):
            ty.bit_width


def test_fixed_size_binary_byte_width():
    ty = pa.binary(5)
    assert ty.byte_width == 5


def test_decimal_properties():
    ty = pa.decimal128(19, 4)
    assert ty.byte_width == 16
    assert ty.precision == 19
    assert ty.scale == 4
    ty = pa.decimal256(76, 38)
    assert ty.byte_width == 32
    assert ty.precision == 76
    assert ty.scale == 38


def test_decimal_overflow():
    pa.decimal128(1, 0)
    pa.decimal128(38, 0)
    for i in (0, -1, 39):
        with pytest.raises(ValueError):
            pa.decimal128(i, 0)

    pa.decimal256(1, 0)
    pa.decimal256(76, 0)
    for i in (0, -1, 77):
        with pytest.raises(ValueError):
            pa.decimal256(i, 0)


def test_timedelta_overflow():
    # microsecond resolution, overflow
    d = datetime.timedelta(days=-106751992, seconds=71945, microseconds=224192)
    with pytest.raises(pa.ArrowInvalid):
        pa.scalar(d)

    # microsecond resolution, overflow
    d = datetime.timedelta(days=106751991, seconds=14454, microseconds=775808)
    with pytest.raises(pa.ArrowInvalid):
        pa.scalar(d)

    # nanosecond resolution, overflow
    d = datetime.timedelta(days=-106752, seconds=763, microseconds=145224)
    with pytest.raises(pa.ArrowInvalid):
        pa.scalar(d, type=pa.duration('ns'))

    # microsecond resolution, not overflow
    pa.scalar(d, type=pa.duration('us')).as_py() == d

    # second/millisecond resolution, not overflow
    for d in [datetime.timedelta.min, datetime.timedelta.max]:
        pa.scalar(d, type=pa.duration('ms')).as_py() == d
        pa.scalar(d, type=pa.duration('s')).as_py() == d


def test_type_equality_operators():
    many_types = get_many_types()
    non_pyarrow = ('foo', 16, {'s', 'e', 't'})

    for index, ty in enumerate(many_types):
        # could use two parametrization levels,
        # but that'd bloat pytest's output
        for i, other in enumerate(many_types + non_pyarrow):
            if i == index:
                assert ty == other
            else:
                assert ty != other


def test_key_value_metadata():
    m = pa.KeyValueMetadata({'a': 'A', 'b': 'B'})
    assert len(m) == 2
    assert m['a'] == b'A'
    assert m[b'a'] == b'A'
    assert m['b'] == b'B'
    assert 'a' in m
    assert b'a' in m
    assert 'c' not in m

    m1 = pa.KeyValueMetadata({'a': 'A', 'b': 'B'})
    m2 = pa.KeyValueMetadata(a='A', b='B')
    m3 = pa.KeyValueMetadata([('a', 'A'), ('b', 'B')])

    assert m1 != 2
    assert m1 == m2
    assert m2 == m3
    assert m1 == {'a': 'A', 'b': 'B'}
    assert m1 != {'a': 'A', 'b': 'C'}

    with pytest.raises(TypeError):
        pa.KeyValueMetadata({'a': 1})
    with pytest.raises(TypeError):
        pa.KeyValueMetadata({1: 'a'})
    with pytest.raises(TypeError):
        pa.KeyValueMetadata(a=1)

    expected = [(b'a', b'A'), (b'b', b'B')]
    result = [(k, v) for k, v in m3.items()]
    assert result == expected
    assert list(m3.items()) == expected
    assert list(m3.keys()) == [b'a', b'b']
    assert list(m3.values()) == [b'A', b'B']
    assert len(m3) == 2

    # test duplicate key support
    md = pa.KeyValueMetadata([
        ('a', 'alpha'),
        ('b', 'beta'),
        ('a', 'Alpha'),
        ('a', 'ALPHA'),
    ])

    expected = [
        (b'a', b'alpha'),
        (b'b', b'beta'),
        (b'a', b'Alpha'),
        (b'a', b'ALPHA')
    ]
    assert len(md) == 4
    assert isinstance(md.keys(), Iterator)
    assert isinstance(md.values(), Iterator)
    assert isinstance(md.items(), Iterator)
    assert list(md.items()) == expected
    assert list(md.keys()) == [k for k, _ in expected]
    assert list(md.values()) == [v for _, v in expected]

    # first occurrence
    assert md['a'] == b'alpha'
    assert md['b'] == b'beta'
    assert md.get_all('a') == [b'alpha', b'Alpha', b'ALPHA']
    assert md.get_all('b') == [b'beta']
    assert md.get_all('unkown') == []

    with pytest.raises(KeyError):
        md = pa.KeyValueMetadata([
            ('a', 'alpha'),
            ('b', 'beta'),
            ('a', 'Alpha'),
            ('a', 'ALPHA'),
        ], b='BETA')


def test_key_value_metadata_duplicates():
    meta = pa.KeyValueMetadata({'a': '1', 'b': '2'})

    with pytest.raises(KeyError):
        pa.KeyValueMetadata(meta, a='3')


def test_field_basic():
    t = pa.string()
    f = pa.field('foo', t)

    assert f.name == 'foo'
    assert f.nullable
    assert f.type is t
    assert repr(f) == "pyarrow.Field<foo: string>"

    f = pa.field('foo', t, False)
    assert not f.nullable

    with pytest.raises(TypeError):
        pa.field('foo', None)


def test_field_equals():
    meta1 = {b'foo': b'bar'}
    meta2 = {b'bizz': b'bazz'}

    f1 = pa.field('a', pa.int8(), nullable=True)
    f2 = pa.field('a', pa.int8(), nullable=True)
    f3 = pa.field('a', pa.int8(), nullable=False)
    f4 = pa.field('a', pa.int16(), nullable=False)
    f5 = pa.field('b', pa.int16(), nullable=False)
    f6 = pa.field('a', pa.int8(), nullable=True, metadata=meta1)
    f7 = pa.field('a', pa.int8(), nullable=True, metadata=meta1)
    f8 = pa.field('a', pa.int8(), nullable=True, metadata=meta2)

    assert f1.equals(f2)
    assert f6.equals(f7)
    assert not f1.equals(f3)
    assert not f1.equals(f4)
    assert not f3.equals(f4)
    assert not f4.equals(f5)

    # No metadata in f1, but metadata in f6
    assert f1.equals(f6)
    assert not f1.equals(f6, check_metadata=True)

    # Different metadata
    assert f6.equals(f7)
    assert f7.equals(f8)
    assert not f7.equals(f8, check_metadata=True)


def test_field_equality_operators():
    f1 = pa.field('a', pa.int8(), nullable=True)
    f2 = pa.field('a', pa.int8(), nullable=True)
    f3 = pa.field('b', pa.int8(), nullable=True)
    f4 = pa.field('b', pa.int8(), nullable=False)

    assert f1 == f2
    assert f1 != f3
    assert f3 != f4
    assert f1 != 'foo'


def test_field_metadata():
    f1 = pa.field('a', pa.int8())
    f2 = pa.field('a', pa.int8(), metadata={})
    f3 = pa.field('a', pa.int8(), metadata={b'bizz': b'bazz'})

    assert f1.metadata is None
    assert f2.metadata == {}
    assert f3.metadata[b'bizz'] == b'bazz'


def test_field_add_remove_metadata():
    import collections

    f0 = pa.field('foo', pa.int32())

    assert f0.metadata is None

    metadata = {b'foo': b'bar', b'pandas': b'badger'}
    metadata2 = collections.OrderedDict([
        (b'a', b'alpha'),
        (b'b', b'beta')
    ])

    f1 = f0.with_metadata(metadata)
    assert f1.metadata == metadata

    f2 = f0.with_metadata(metadata2)
    assert f2.metadata == metadata2

    with pytest.raises(TypeError):
        f0.with_metadata([1, 2, 3])

    f3 = f1.remove_metadata()
    assert f3.metadata is None

    # idempotent
    f4 = f3.remove_metadata()
    assert f4.metadata is None

    f5 = pa.field('foo', pa.int32(), True, metadata)
    f6 = f0.with_metadata(metadata)
    assert f5.equals(f6)


def test_field_modified_copies():
    f0 = pa.field('foo', pa.int32(), True)
    f0_ = pa.field('foo', pa.int32(), True)
    assert f0.equals(f0_)

    f1 = pa.field('foo', pa.int64(), True)
    f1_ = f0.with_type(pa.int64())
    assert f1.equals(f1_)
    # Original instance is unmodified
    assert f0.equals(f0_)

    f2 = pa.field('foo', pa.int32(), False)
    f2_ = f0.with_nullable(False)
    assert f2.equals(f2_)
    # Original instance is unmodified
    assert f0.equals(f0_)

    f3 = pa.field('bar', pa.int32(), True)
    f3_ = f0.with_name('bar')
    assert f3.equals(f3_)
    # Original instance is unmodified
    assert f0.equals(f0_)


def test_is_integer_value():
    assert pa.types.is_integer_value(1)
    assert pa.types.is_integer_value(np.int64(1))
    assert not pa.types.is_integer_value('1')


def test_is_float_value():
    assert not pa.types.is_float_value(1)
    assert pa.types.is_float_value(1.)
    assert pa.types.is_float_value(np.float64(1))
    assert not pa.types.is_float_value('1.0')


def test_is_boolean_value():
    assert not pa.types.is_boolean_value(1)
    assert pa.types.is_boolean_value(True)
    assert pa.types.is_boolean_value(False)
    assert pa.types.is_boolean_value(np.bool_(True))
    assert pa.types.is_boolean_value(np.bool_(False))


@h.given(
    past.all_types |
    past.all_fields |
    past.all_schemas
)
@h.example(
    pa.field(name='', type=pa.null(), metadata={'0': '', '': ''})
)
def test_pickling(field):
    data = pickle.dumps(field)
    assert pickle.loads(data) == field


@h.given(
    st.lists(past.all_types) |
    st.lists(past.all_fields) |
    st.lists(past.all_schemas)
)
def test_hashing(items):
    h.assume(
        # well, this is still O(n^2), but makes the input unique
        all(not a.equals(b) for i, a in enumerate(items) for b in items[:i])
    )

    container = {}
    for i, item in enumerate(items):
        assert hash(item) == hash(item)
        container[item] = i

    assert len(container) == len(items)

    for i, item in enumerate(items):
        assert container[item] == i


def test_types_come_back_with_specific_type():
    for arrow_type in get_many_types():
        schema = pa.schema([pa.field("field_name", arrow_type)])
        type_back = schema.field("field_name").type
        assert type(type_back) is type(arrow_type)

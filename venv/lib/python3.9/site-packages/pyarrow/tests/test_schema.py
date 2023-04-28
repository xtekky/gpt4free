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
import pickle
import sys
import weakref

import pytest
import numpy as np
import pyarrow as pa

import pyarrow.tests.util as test_util


def test_schema_constructor_errors():
    msg = ("Do not call Schema's constructor directly, use `pyarrow.schema` "
           "instead")
    with pytest.raises(TypeError, match=msg):
        pa.Schema()


def test_type_integers():
    dtypes = ['int8', 'int16', 'int32', 'int64',
              'uint8', 'uint16', 'uint32', 'uint64']

    for name in dtypes:
        factory = getattr(pa, name)
        t = factory()
        assert str(t) == name


def test_type_to_pandas_dtype():
    M8_ns = np.dtype('datetime64[ns]')
    cases = [
        (pa.null(), np.object_),
        (pa.bool_(), np.bool_),
        (pa.int8(), np.int8),
        (pa.int16(), np.int16),
        (pa.int32(), np.int32),
        (pa.int64(), np.int64),
        (pa.uint8(), np.uint8),
        (pa.uint16(), np.uint16),
        (pa.uint32(), np.uint32),
        (pa.uint64(), np.uint64),
        (pa.float16(), np.float16),
        (pa.float32(), np.float32),
        (pa.float64(), np.float64),
        (pa.date32(), M8_ns),
        (pa.date64(), M8_ns),
        (pa.timestamp('ms'), M8_ns),
        (pa.binary(), np.object_),
        (pa.binary(12), np.object_),
        (pa.string(), np.object_),
        (pa.list_(pa.int8()), np.object_),
        # (pa.list_(pa.int8(), 2), np.object_),  # TODO needs pandas conversion
        (pa.map_(pa.int64(), pa.float64()), np.object_),
    ]
    for arrow_type, numpy_type in cases:
        assert arrow_type.to_pandas_dtype() == numpy_type


@pytest.mark.pandas
def test_type_to_pandas_dtype_check_import():
    # ARROW-7980
    test_util.invoke_script('arrow_7980.py')


def test_type_list():
    value_type = pa.int32()
    list_type = pa.list_(value_type)
    assert str(list_type) == 'list<item: int32>'

    field = pa.field('my_item', pa.string())
    l2 = pa.list_(field)
    assert str(l2) == 'list<my_item: string>'


def test_type_comparisons():
    val = pa.int32()
    assert val == pa.int32()
    assert val == 'int32'
    assert val != 5


def test_type_for_alias():
    cases = [
        ('i1', pa.int8()),
        ('int8', pa.int8()),
        ('i2', pa.int16()),
        ('int16', pa.int16()),
        ('i4', pa.int32()),
        ('int32', pa.int32()),
        ('i8', pa.int64()),
        ('int64', pa.int64()),
        ('u1', pa.uint8()),
        ('uint8', pa.uint8()),
        ('u2', pa.uint16()),
        ('uint16', pa.uint16()),
        ('u4', pa.uint32()),
        ('uint32', pa.uint32()),
        ('u8', pa.uint64()),
        ('uint64', pa.uint64()),
        ('f4', pa.float32()),
        ('float32', pa.float32()),
        ('f8', pa.float64()),
        ('float64', pa.float64()),
        ('date32', pa.date32()),
        ('date64', pa.date64()),
        ('string', pa.string()),
        ('str', pa.string()),
        ('binary', pa.binary()),
        ('time32[s]', pa.time32('s')),
        ('time32[ms]', pa.time32('ms')),
        ('time64[us]', pa.time64('us')),
        ('time64[ns]', pa.time64('ns')),
        ('timestamp[s]', pa.timestamp('s')),
        ('timestamp[ms]', pa.timestamp('ms')),
        ('timestamp[us]', pa.timestamp('us')),
        ('timestamp[ns]', pa.timestamp('ns')),
        ('duration[s]', pa.duration('s')),
        ('duration[ms]', pa.duration('ms')),
        ('duration[us]', pa.duration('us')),
        ('duration[ns]', pa.duration('ns')),
        ('month_day_nano_interval', pa.month_day_nano_interval()),
    ]

    for val, expected in cases:
        assert pa.type_for_alias(val) == expected


def test_type_string():
    t = pa.string()
    assert str(t) == 'string'


def test_type_timestamp_with_tz():
    tz = 'America/Los_Angeles'
    t = pa.timestamp('ns', tz=tz)
    assert t.unit == 'ns'
    assert t.tz == tz


def test_time_types():
    t1 = pa.time32('s')
    t2 = pa.time32('ms')
    t3 = pa.time64('us')
    t4 = pa.time64('ns')

    assert t1.unit == 's'
    assert t2.unit == 'ms'
    assert t3.unit == 'us'
    assert t4.unit == 'ns'

    assert str(t1) == 'time32[s]'
    assert str(t4) == 'time64[ns]'

    with pytest.raises(ValueError):
        pa.time32('us')

    with pytest.raises(ValueError):
        pa.time64('s')


def test_from_numpy_dtype():
    cases = [
        (np.dtype('bool'), pa.bool_()),
        (np.dtype('int8'), pa.int8()),
        (np.dtype('int16'), pa.int16()),
        (np.dtype('int32'), pa.int32()),
        (np.dtype('int64'), pa.int64()),
        (np.dtype('uint8'), pa.uint8()),
        (np.dtype('uint16'), pa.uint16()),
        (np.dtype('uint32'), pa.uint32()),
        (np.dtype('float16'), pa.float16()),
        (np.dtype('float32'), pa.float32()),
        (np.dtype('float64'), pa.float64()),
        (np.dtype('U'), pa.string()),
        (np.dtype('S'), pa.binary()),
        (np.dtype('datetime64[s]'), pa.timestamp('s')),
        (np.dtype('datetime64[ms]'), pa.timestamp('ms')),
        (np.dtype('datetime64[us]'), pa.timestamp('us')),
        (np.dtype('datetime64[ns]'), pa.timestamp('ns')),
        (np.dtype('timedelta64[s]'), pa.duration('s')),
        (np.dtype('timedelta64[ms]'), pa.duration('ms')),
        (np.dtype('timedelta64[us]'), pa.duration('us')),
        (np.dtype('timedelta64[ns]'), pa.duration('ns')),
    ]

    for dt, pt in cases:
        result = pa.from_numpy_dtype(dt)
        assert result == pt

    # Things convertible to numpy dtypes work
    assert pa.from_numpy_dtype('U') == pa.string()
    assert pa.from_numpy_dtype(np.str_) == pa.string()
    assert pa.from_numpy_dtype('int32') == pa.int32()
    assert pa.from_numpy_dtype(bool) == pa.bool_()

    with pytest.raises(NotImplementedError):
        pa.from_numpy_dtype(np.dtype('O'))

    with pytest.raises(TypeError):
        pa.from_numpy_dtype('not_convertible_to_dtype')


def test_schema():
    fields = [
        pa.field('foo', pa.int32()),
        pa.field('bar', pa.string()),
        pa.field('baz', pa.list_(pa.int8()))
    ]
    sch = pa.schema(fields)

    assert sch.names == ['foo', 'bar', 'baz']
    assert sch.types == [pa.int32(), pa.string(), pa.list_(pa.int8())]

    assert len(sch) == 3
    assert sch[0].name == 'foo'
    assert sch[0].type == fields[0].type
    assert sch.field('foo').name == 'foo'
    assert sch.field('foo').type == fields[0].type

    assert repr(sch) == """\
foo: int32
bar: string
baz: list<item: int8>
  child 0, item: int8"""

    with pytest.raises(TypeError):
        pa.schema([None])


def test_schema_weakref():
    fields = [
        pa.field('foo', pa.int32()),
        pa.field('bar', pa.string()),
        pa.field('baz', pa.list_(pa.int8()))
    ]
    schema = pa.schema(fields)
    wr = weakref.ref(schema)
    assert wr() is not None
    del schema
    assert wr() is None


def test_schema_to_string_with_metadata():
    lorem = """\
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla accumsan vel
turpis et mollis. Aliquam tincidunt arcu id tortor blandit blandit. Donec
eget leo quis lectus scelerisque varius. Class aptent taciti sociosqu ad
litora torquent per conubia nostra, per inceptos himenaeos. Praesent
faucibus, diam eu volutpat iaculis, tellus est porta ligula, a efficitur
turpis nulla facilisis quam. Aliquam vitae lorem erat. Proin a dolor ac libero
dignissim mollis vitae eu mauris. Quisque posuere tellus vitae massa
pellentesque sagittis. Aenean feugiat, diam ac dignissim fermentum, lorem
sapien commodo massa, vel volutpat orci nisi eu justo. Nulla non blandit
sapien. Quisque pretium vestibulum urna eu vehicula."""
    # ARROW-7063
    my_schema = pa.schema([pa.field("foo", "int32", False,
                                    metadata={"key1": "value1"}),
                           pa.field("bar", "string", True,
                                    metadata={"key3": "value3"})],
                          metadata={"lorem": lorem})

    assert my_schema.to_string() == """\
foo: int32 not null
  -- field metadata --
  key1: 'value1'
bar: string
  -- field metadata --
  key3: 'value3'
-- schema metadata --
lorem: '""" + lorem[:65] + "' + " + str(len(lorem) - 65)

    # Metadata that exactly fits
    result = pa.schema([('f0', 'int32')],
                       metadata={'key': 'value' + 'x' * 62}).to_string()
    assert result == """\
f0: int32
-- schema metadata --
key: 'valuexxxxxxxxxxxxxxxxxxxxxxxxxxxxx\
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'"""

    assert my_schema.to_string(truncate_metadata=False) == """\
foo: int32 not null
  -- field metadata --
  key1: 'value1'
bar: string
  -- field metadata --
  key3: 'value3'
-- schema metadata --
lorem: '{}'""".format(lorem)

    assert my_schema.to_string(truncate_metadata=False,
                               show_field_metadata=False) == """\
foo: int32 not null
bar: string
-- schema metadata --
lorem: '{}'""".format(lorem)

    assert my_schema.to_string(truncate_metadata=False,
                               show_schema_metadata=False) == """\
foo: int32 not null
  -- field metadata --
  key1: 'value1'
bar: string
  -- field metadata --
  key3: 'value3'"""

    assert my_schema.to_string(truncate_metadata=False,
                               show_field_metadata=False,
                               show_schema_metadata=False) == """\
foo: int32 not null
bar: string"""


def test_schema_from_tuples():
    fields = [
        ('foo', pa.int32()),
        ('bar', pa.string()),
        ('baz', pa.list_(pa.int8())),
    ]
    sch = pa.schema(fields)
    assert sch.names == ['foo', 'bar', 'baz']
    assert sch.types == [pa.int32(), pa.string(), pa.list_(pa.int8())]
    assert len(sch) == 3
    assert repr(sch) == """\
foo: int32
bar: string
baz: list<item: int8>
  child 0, item: int8"""

    with pytest.raises(TypeError):
        pa.schema([('foo', None)])


def test_schema_from_mapping():
    fields = OrderedDict([
        ('foo', pa.int32()),
        ('bar', pa.string()),
        ('baz', pa.list_(pa.int8())),
    ])
    sch = pa.schema(fields)
    assert sch.names == ['foo', 'bar', 'baz']
    assert sch.types == [pa.int32(), pa.string(), pa.list_(pa.int8())]
    assert len(sch) == 3
    assert repr(sch) == """\
foo: int32
bar: string
baz: list<item: int8>
  child 0, item: int8"""

    fields = OrderedDict([('foo', None)])
    with pytest.raises(TypeError):
        pa.schema(fields)


def test_schema_duplicate_fields():
    fields = [
        pa.field('foo', pa.int32()),
        pa.field('bar', pa.string()),
        pa.field('foo', pa.list_(pa.int8())),
    ]
    sch = pa.schema(fields)
    assert sch.names == ['foo', 'bar', 'foo']
    assert sch.types == [pa.int32(), pa.string(), pa.list_(pa.int8())]
    assert len(sch) == 3
    assert repr(sch) == """\
foo: int32
bar: string
foo: list<item: int8>
  child 0, item: int8"""

    assert sch[0].name == 'foo'
    assert sch[0].type == fields[0].type
    with pytest.warns(FutureWarning):
        assert sch.field_by_name('bar') == fields[1]
    with pytest.warns(FutureWarning):
        assert sch.field_by_name('xxx') is None
    with pytest.warns((UserWarning, FutureWarning)):
        assert sch.field_by_name('foo') is None

    # Schema::GetFieldIndex
    assert sch.get_field_index('foo') == -1

    # Schema::GetAllFieldIndices
    assert sch.get_all_field_indices('foo') == [0, 2]


def test_field_flatten():
    f0 = pa.field('foo', pa.int32()).with_metadata({b'foo': b'bar'})
    assert f0.flatten() == [f0]

    f1 = pa.field('bar', pa.float64(), nullable=False)
    ff = pa.field('ff', pa.struct([f0, f1]), nullable=False)
    assert ff.flatten() == [
        pa.field('ff.foo', pa.int32()).with_metadata({b'foo': b'bar'}),
        pa.field('ff.bar', pa.float64(), nullable=False)]  # XXX

    # Nullable parent makes flattened child nullable
    ff = pa.field('ff', pa.struct([f0, f1]))
    assert ff.flatten() == [
        pa.field('ff.foo', pa.int32()).with_metadata({b'foo': b'bar'}),
        pa.field('ff.bar', pa.float64())]

    fff = pa.field('fff', pa.struct([ff]))
    assert fff.flatten() == [pa.field('fff.ff', pa.struct([f0, f1]))]


def test_schema_add_remove_metadata():
    fields = [
        pa.field('foo', pa.int32()),
        pa.field('bar', pa.string()),
        pa.field('baz', pa.list_(pa.int8()))
    ]

    s1 = pa.schema(fields)

    assert s1.metadata is None

    metadata = {b'foo': b'bar', b'pandas': b'badger'}

    s2 = s1.with_metadata(metadata)
    assert s2.metadata == metadata

    s3 = s2.remove_metadata()
    assert s3.metadata is None

    # idempotent
    s4 = s3.remove_metadata()
    assert s4.metadata is None


def test_schema_equals():
    fields = [
        pa.field('foo', pa.int32()),
        pa.field('bar', pa.string()),
        pa.field('baz', pa.list_(pa.int8()))
    ]
    metadata = {b'foo': b'bar', b'pandas': b'badger'}

    sch1 = pa.schema(fields)
    sch2 = pa.schema(fields)
    sch3 = pa.schema(fields, metadata=metadata)
    sch4 = pa.schema(fields, metadata=metadata)

    assert sch1.equals(sch2, check_metadata=True)
    assert sch3.equals(sch4, check_metadata=True)
    assert sch1.equals(sch3)
    assert not sch1.equals(sch3, check_metadata=True)
    assert not sch1.equals(sch3, check_metadata=True)

    del fields[-1]
    sch3 = pa.schema(fields)
    assert not sch1.equals(sch3)


def test_schema_equals_propagates_check_metadata():
    # ARROW-4088
    schema1 = pa.schema([
        pa.field('foo', pa.int32()),
        pa.field('bar', pa.string())
    ])
    schema2 = pa.schema([
        pa.field('foo', pa.int32()),
        pa.field('bar', pa.string(), metadata={'a': 'alpha'}),
    ])
    assert not schema1.equals(schema2, check_metadata=True)
    assert schema1.equals(schema2)


def test_schema_equals_invalid_type():
    # ARROW-5873
    schema = pa.schema([pa.field("a", pa.int64())])

    for val in [None, 'string', pa.array([1, 2])]:
        with pytest.raises(TypeError):
            schema.equals(val)


def test_schema_equality_operators():
    fields = [
        pa.field('foo', pa.int32()),
        pa.field('bar', pa.string()),
        pa.field('baz', pa.list_(pa.int8()))
    ]
    metadata = {b'foo': b'bar', b'pandas': b'badger'}

    sch1 = pa.schema(fields)
    sch2 = pa.schema(fields)
    sch3 = pa.schema(fields, metadata=metadata)
    sch4 = pa.schema(fields, metadata=metadata)

    assert sch1 == sch2
    assert sch3 == sch4

    # __eq__ and __ne__ do not check metadata
    assert sch1 == sch3
    assert not sch1 != sch3

    assert sch2 == sch4

    # comparison with other types doesn't raise
    assert sch1 != []
    assert sch3 != 'foo'


def test_schema_get_fields():
    fields = [
        pa.field('foo', pa.int32()),
        pa.field('bar', pa.string()),
        pa.field('baz', pa.list_(pa.int8()))
    ]

    schema = pa.schema(fields)

    assert schema.field('foo').name == 'foo'
    assert schema.field(0).name == 'foo'
    assert schema.field(-1).name == 'baz'

    with pytest.raises(KeyError):
        schema.field('other')
    with pytest.raises(TypeError):
        schema.field(0.0)
    with pytest.raises(IndexError):
        schema.field(4)


def test_schema_negative_indexing():
    fields = [
        pa.field('foo', pa.int32()),
        pa.field('bar', pa.string()),
        pa.field('baz', pa.list_(pa.int8()))
    ]

    schema = pa.schema(fields)

    assert schema[-1].equals(schema[2])
    assert schema[-2].equals(schema[1])
    assert schema[-3].equals(schema[0])

    with pytest.raises(IndexError):
        schema[-4]

    with pytest.raises(IndexError):
        schema[3]


def test_schema_repr_with_dictionaries():
    fields = [
        pa.field('one', pa.dictionary(pa.int16(), pa.string())),
        pa.field('two', pa.int32())
    ]
    sch = pa.schema(fields)

    expected = (
        """\
one: dictionary<values=string, indices=int16, ordered=0>
two: int32""")

    assert repr(sch) == expected


def test_type_schema_pickling():
    cases = [
        pa.int8(),
        pa.string(),
        pa.binary(),
        pa.binary(10),
        pa.list_(pa.string()),
        pa.map_(pa.string(), pa.int8()),
        pa.struct([
            pa.field('a', 'int8'),
            pa.field('b', 'string')
        ]),
        pa.union([
            pa.field('a', pa.int8()),
            pa.field('b', pa.int16())
        ], pa.lib.UnionMode_SPARSE),
        pa.union([
            pa.field('a', pa.int8()),
            pa.field('b', pa.int16())
        ], pa.lib.UnionMode_DENSE),
        pa.time32('s'),
        pa.time64('us'),
        pa.date32(),
        pa.date64(),
        pa.timestamp('ms'),
        pa.timestamp('ns'),
        pa.decimal128(12, 2),
        pa.decimal256(76, 38),
        pa.field('a', 'string', metadata={b'foo': b'bar'}),
        pa.list_(pa.field("element", pa.int64())),
        pa.large_list(pa.field("element", pa.int64())),
        pa.map_(pa.field("key", pa.string(), nullable=False),
                pa.field("value", pa.int8()))
    ]

    for val in cases:
        roundtripped = pickle.loads(pickle.dumps(val))
        assert val == roundtripped

    fields = []
    for i, f in enumerate(cases):
        if isinstance(f, pa.Field):
            fields.append(f)
        else:
            fields.append(pa.field('_f{}'.format(i), f))

    schema = pa.schema(fields, metadata={b'foo': b'bar'})
    roundtripped = pickle.loads(pickle.dumps(schema))
    assert schema == roundtripped


def test_empty_table():
    schema1 = pa.schema([
        pa.field('f0', pa.int64()),
        pa.field('f1', pa.dictionary(pa.int32(), pa.string())),
        pa.field('f2', pa.list_(pa.list_(pa.int64()))),
    ])
    # test it preserves field nullability
    schema2 = pa.schema([
        pa.field('a', pa.int64(), nullable=False),
        pa.field('b', pa.int64())
    ])

    for schema in [schema1, schema2]:
        table = schema.empty_table()
        assert isinstance(table, pa.Table)
        assert table.num_rows == 0
        assert table.schema == schema


@pytest.mark.pandas
def test_schema_from_pandas():
    import pandas as pd
    inputs = [
        list(range(10)),
        pd.Categorical(list(range(10))),
        ['foo', 'bar', None, 'baz', 'qux'],
        np.array([
            '2007-07-13T01:23:34.123456789',
            '2006-01-13T12:34:56.432539784',
            '2010-08-13T05:46:57.437699912'
        ], dtype='datetime64[ns]'),
        pd.array([1, 2, None], dtype=pd.Int32Dtype()),
    ]
    for data in inputs:
        df = pd.DataFrame({'a': data}, index=data)
        schema = pa.Schema.from_pandas(df)
        expected = pa.Table.from_pandas(df).schema
        assert schema == expected


def test_schema_sizeof():
    schema = pa.schema([
        pa.field('foo', pa.int32()),
        pa.field('bar', pa.string()),
    ])

    assert sys.getsizeof(schema) > 30

    schema2 = schema.with_metadata({"key": "some metadata"})
    assert sys.getsizeof(schema2) > sys.getsizeof(schema)
    schema3 = schema.with_metadata({"key": "some more metadata"})
    assert sys.getsizeof(schema3) > sys.getsizeof(schema2)


def test_schema_merge():
    a = pa.schema([
        pa.field('foo', pa.int32()),
        pa.field('bar', pa.string()),
        pa.field('baz', pa.list_(pa.int8()))
    ])
    b = pa.schema([
        pa.field('foo', pa.int32()),
        pa.field('qux', pa.bool_())
    ])
    c = pa.schema([
        pa.field('quux', pa.dictionary(pa.int32(), pa.string()))
    ])
    d = pa.schema([
        pa.field('foo', pa.int64()),
        pa.field('qux', pa.bool_())
    ])

    result = pa.unify_schemas([a, b, c])
    expected = pa.schema([
        pa.field('foo', pa.int32()),
        pa.field('bar', pa.string()),
        pa.field('baz', pa.list_(pa.int8())),
        pa.field('qux', pa.bool_()),
        pa.field('quux', pa.dictionary(pa.int32(), pa.string()))
    ])
    assert result.equals(expected)

    with pytest.raises(pa.ArrowInvalid):
        pa.unify_schemas([b, d])

    # ARROW-14002: Try with tuple instead of list
    result = pa.unify_schemas((a, b, c))
    assert result.equals(expected)

    # raise proper error when passing a non-Schema value
    with pytest.raises(TypeError):
        pa.unify_schemas([a, 1])


def test_undecodable_metadata():
    # ARROW-10214: undecodable metadata shouldn't fail repr()
    data1 = b'abcdef\xff\x00'
    data2 = b'ghijkl\xff\x00'
    schema = pa.schema(
        [pa.field('ints', pa.int16(), metadata={'key': data1})],
        metadata={'key': data2})
    assert 'abcdef' in str(schema)
    assert 'ghijkl' in str(schema)

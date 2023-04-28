# -*- coding: utf-8 -*-
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

import gc

import pyarrow as pa
try:
    from pyarrow.cffi import ffi
except ImportError:
    ffi = None

import pytest

try:
    import pandas as pd
    import pandas.testing as tm
except ImportError:
    pd = tm = None


needs_cffi = pytest.mark.skipif(ffi is None,
                                reason="test needs cffi package installed")


assert_schema_released = pytest.raises(
    ValueError, match="Cannot import released ArrowSchema")

assert_array_released = pytest.raises(
    ValueError, match="Cannot import released ArrowArray")

assert_stream_released = pytest.raises(
    ValueError, match="Cannot import released ArrowArrayStream")


class ParamExtType(pa.PyExtensionType):

    def __init__(self, width):
        self._width = width
        pa.PyExtensionType.__init__(self, pa.binary(width))

    @property
    def width(self):
        return self._width

    def __reduce__(self):
        return ParamExtType, (self.width,)


def make_schema():
    return pa.schema([('ints', pa.list_(pa.int32()))],
                     metadata={b'key1': b'value1'})


def make_extension_schema():
    return pa.schema([('ext', ParamExtType(3))],
                     metadata={b'key1': b'value1'})


def make_batch():
    return pa.record_batch([[[1], [2, 42]]], make_schema())


def make_extension_batch():
    schema = make_extension_schema()
    ext_col = schema[0].type.wrap_array(pa.array([b"foo", b"bar"],
                                                 type=pa.binary(3)))
    return pa.record_batch([ext_col], schema)


def make_batches():
    schema = make_schema()
    return [
        pa.record_batch([[[1], [2, 42]]], schema),
        pa.record_batch([[None, [], [5, 6]]], schema),
    ]


def make_serialized(schema, batches):
    with pa.BufferOutputStream() as sink:
        with pa.ipc.new_stream(sink, schema) as out:
            for batch in batches:
                out.write(batch)
        return sink.getvalue()


@needs_cffi
def test_export_import_type():
    c_schema = ffi.new("struct ArrowSchema*")
    ptr_schema = int(ffi.cast("uintptr_t", c_schema))

    gc.collect()  # Make sure no Arrow data dangles in a ref cycle
    old_allocated = pa.total_allocated_bytes()

    typ = pa.list_(pa.int32())
    typ._export_to_c(ptr_schema)
    assert pa.total_allocated_bytes() > old_allocated
    # Delete and recreate C++ object from exported pointer
    del typ
    assert pa.total_allocated_bytes() > old_allocated
    typ_new = pa.DataType._import_from_c(ptr_schema)
    assert typ_new == pa.list_(pa.int32())
    assert pa.total_allocated_bytes() == old_allocated
    # Now released
    with assert_schema_released:
        pa.DataType._import_from_c(ptr_schema)

    # Invalid format string
    pa.int32()._export_to_c(ptr_schema)
    bad_format = ffi.new("char[]", b"zzz")
    c_schema.format = bad_format
    with pytest.raises(ValueError,
                       match="Invalid or unsupported format string"):
        pa.DataType._import_from_c(ptr_schema)
    # Now released
    with assert_schema_released:
        pa.DataType._import_from_c(ptr_schema)


@needs_cffi
def test_export_import_field():
    c_schema = ffi.new("struct ArrowSchema*")
    ptr_schema = int(ffi.cast("uintptr_t", c_schema))

    gc.collect()  # Make sure no Arrow data dangles in a ref cycle
    old_allocated = pa.total_allocated_bytes()

    field = pa.field("test", pa.list_(pa.int32()), nullable=True)
    field._export_to_c(ptr_schema)
    assert pa.total_allocated_bytes() > old_allocated
    # Delete and recreate C++ object from exported pointer
    del field
    assert pa.total_allocated_bytes() > old_allocated

    field_new = pa.Field._import_from_c(ptr_schema)
    assert field_new == pa.field("test", pa.list_(pa.int32()), nullable=True)
    assert pa.total_allocated_bytes() == old_allocated

    # Now released
    with assert_schema_released:
        pa.Field._import_from_c(ptr_schema)


@needs_cffi
def test_export_import_array():
    c_schema = ffi.new("struct ArrowSchema*")
    ptr_schema = int(ffi.cast("uintptr_t", c_schema))
    c_array = ffi.new("struct ArrowArray*")
    ptr_array = int(ffi.cast("uintptr_t", c_array))

    gc.collect()  # Make sure no Arrow data dangles in a ref cycle
    old_allocated = pa.total_allocated_bytes()

    # Type is known up front
    typ = pa.list_(pa.int32())
    arr = pa.array([[1], [2, 42]], type=typ)
    py_value = arr.to_pylist()
    arr._export_to_c(ptr_array)
    assert pa.total_allocated_bytes() > old_allocated
    # Delete recreate C++ object from exported pointer
    del arr
    arr_new = pa.Array._import_from_c(ptr_array, typ)
    assert arr_new.to_pylist() == py_value
    assert arr_new.type == pa.list_(pa.int32())
    assert pa.total_allocated_bytes() > old_allocated
    del arr_new, typ
    assert pa.total_allocated_bytes() == old_allocated
    # Now released
    with assert_array_released:
        pa.Array._import_from_c(ptr_array, pa.list_(pa.int32()))

    # Type is exported and imported at the same time
    arr = pa.array([[1], [2, 42]], type=pa.list_(pa.int32()))
    py_value = arr.to_pylist()
    arr._export_to_c(ptr_array, ptr_schema)
    # Delete and recreate C++ objects from exported pointers
    del arr
    arr_new = pa.Array._import_from_c(ptr_array, ptr_schema)
    assert arr_new.to_pylist() == py_value
    assert arr_new.type == pa.list_(pa.int32())
    assert pa.total_allocated_bytes() > old_allocated
    del arr_new
    assert pa.total_allocated_bytes() == old_allocated
    # Now released
    with assert_schema_released:
        pa.Array._import_from_c(ptr_array, ptr_schema)


def check_export_import_schema(schema_factory):
    c_schema = ffi.new("struct ArrowSchema*")
    ptr_schema = int(ffi.cast("uintptr_t", c_schema))

    gc.collect()  # Make sure no Arrow data dangles in a ref cycle
    old_allocated = pa.total_allocated_bytes()

    schema_factory()._export_to_c(ptr_schema)
    assert pa.total_allocated_bytes() > old_allocated
    # Delete and recreate C++ object from exported pointer
    schema_new = pa.Schema._import_from_c(ptr_schema)
    assert schema_new == schema_factory()
    assert pa.total_allocated_bytes() == old_allocated
    del schema_new
    assert pa.total_allocated_bytes() == old_allocated
    # Now released
    with assert_schema_released:
        pa.Schema._import_from_c(ptr_schema)

    # Not a struct type
    pa.int32()._export_to_c(ptr_schema)
    with pytest.raises(ValueError,
                       match="ArrowSchema describes non-struct type"):
        pa.Schema._import_from_c(ptr_schema)
    # Now released
    with assert_schema_released:
        pa.Schema._import_from_c(ptr_schema)


@needs_cffi
def test_export_import_schema():
    check_export_import_schema(make_schema)


@needs_cffi
def test_export_import_schema_with_extension():
    check_export_import_schema(make_extension_schema)


@needs_cffi
def test_export_import_schema_float_pointer():
    # Previous versions of the R Arrow library used to pass pointer
    # values as a double.
    c_schema = ffi.new("struct ArrowSchema*")
    ptr_schema = int(ffi.cast("uintptr_t", c_schema))

    match = "Passing a pointer value as a float is unsafe"
    with pytest.warns(UserWarning, match=match):
        make_schema()._export_to_c(float(ptr_schema))
    with pytest.warns(UserWarning, match=match):
        schema_new = pa.Schema._import_from_c(float(ptr_schema))
    assert schema_new == make_schema()


def check_export_import_batch(batch_factory):
    c_schema = ffi.new("struct ArrowSchema*")
    ptr_schema = int(ffi.cast("uintptr_t", c_schema))
    c_array = ffi.new("struct ArrowArray*")
    ptr_array = int(ffi.cast("uintptr_t", c_array))

    gc.collect()  # Make sure no Arrow data dangles in a ref cycle
    old_allocated = pa.total_allocated_bytes()

    # Schema is known up front
    batch = batch_factory()
    schema = batch.schema
    py_value = batch.to_pydict()
    batch._export_to_c(ptr_array)
    assert pa.total_allocated_bytes() > old_allocated
    # Delete and recreate C++ object from exported pointer
    del batch
    batch_new = pa.RecordBatch._import_from_c(ptr_array, schema)
    assert batch_new.to_pydict() == py_value
    assert batch_new.schema == schema
    assert pa.total_allocated_bytes() > old_allocated
    del batch_new, schema
    assert pa.total_allocated_bytes() == old_allocated
    # Now released
    with assert_array_released:
        pa.RecordBatch._import_from_c(ptr_array, make_schema())

    # Type is exported and imported at the same time
    batch = batch_factory()
    py_value = batch.to_pydict()
    batch._export_to_c(ptr_array, ptr_schema)
    # Delete and recreate C++ objects from exported pointers
    del batch
    batch_new = pa.RecordBatch._import_from_c(ptr_array, ptr_schema)
    assert batch_new.to_pydict() == py_value
    assert batch_new.schema == batch_factory().schema
    assert pa.total_allocated_bytes() > old_allocated
    del batch_new
    assert pa.total_allocated_bytes() == old_allocated
    # Now released
    with assert_schema_released:
        pa.RecordBatch._import_from_c(ptr_array, ptr_schema)

    # Not a struct type
    pa.int32()._export_to_c(ptr_schema)
    batch_factory()._export_to_c(ptr_array)
    with pytest.raises(ValueError,
                       match="ArrowSchema describes non-struct type"):
        pa.RecordBatch._import_from_c(ptr_array, ptr_schema)
    # Now released
    with assert_schema_released:
        pa.RecordBatch._import_from_c(ptr_array, ptr_schema)


@needs_cffi
def test_export_import_batch():
    check_export_import_batch(make_batch)


@needs_cffi
def test_export_import_batch_with_extension():
    check_export_import_batch(make_extension_batch)


def _export_import_batch_reader(ptr_stream, reader_factory):
    # Prepare input
    batches = make_batches()
    schema = batches[0].schema

    reader = reader_factory(schema, batches)
    reader._export_to_c(ptr_stream)
    # Delete and recreate C++ object from exported pointer
    del reader, batches

    reader_new = pa.RecordBatchReader._import_from_c(ptr_stream)
    assert reader_new.schema == schema
    got_batches = list(reader_new)
    del reader_new
    assert got_batches == make_batches()

    # Test read_pandas()
    if pd is not None:
        batches = make_batches()
        schema = batches[0].schema
        expected_df = pa.Table.from_batches(batches).to_pandas()

        reader = reader_factory(schema, batches)
        reader._export_to_c(ptr_stream)
        del reader, batches

        reader_new = pa.RecordBatchReader._import_from_c(ptr_stream)
        got_df = reader_new.read_pandas()
        del reader_new
        tm.assert_frame_equal(expected_df, got_df)


def make_ipc_stream_reader(schema, batches):
    return pa.ipc.open_stream(make_serialized(schema, batches))


def make_py_record_batch_reader(schema, batches):
    return pa.RecordBatchReader.from_batches(schema, batches)


@needs_cffi
@pytest.mark.parametrize('reader_factory',
                         [make_ipc_stream_reader,
                          make_py_record_batch_reader])
def test_export_import_batch_reader(reader_factory):
    c_stream = ffi.new("struct ArrowArrayStream*")
    ptr_stream = int(ffi.cast("uintptr_t", c_stream))

    gc.collect()  # Make sure no Arrow data dangles in a ref cycle
    old_allocated = pa.total_allocated_bytes()

    _export_import_batch_reader(ptr_stream, reader_factory)

    assert pa.total_allocated_bytes() == old_allocated

    # Now released
    with assert_stream_released:
        pa.RecordBatchReader._import_from_c(ptr_stream)


@needs_cffi
def test_imported_batch_reader_error():
    c_stream = ffi.new("struct ArrowArrayStream*")
    ptr_stream = int(ffi.cast("uintptr_t", c_stream))

    schema = pa.schema([('foo', pa.int32())])
    batches = [pa.record_batch([[1, 2, 3]], schema=schema),
               pa.record_batch([[4, 5, 6]], schema=schema)]
    buf = make_serialized(schema, batches)

    # Open a corrupt/incomplete stream and export it
    reader = pa.ipc.open_stream(buf[:-16])
    reader._export_to_c(ptr_stream)
    del reader

    reader_new = pa.RecordBatchReader._import_from_c(ptr_stream)
    batch = reader_new.read_next_batch()
    assert batch == batches[0]
    with pytest.raises(OSError,
                       match="Expected to be able to read 16 bytes "
                             "for message body, got 8"):
        reader_new.read_next_batch()

    # Again, but call read_all()
    reader = pa.ipc.open_stream(buf[:-16])
    reader._export_to_c(ptr_stream)
    del reader

    reader_new = pa.RecordBatchReader._import_from_c(ptr_stream)
    with pytest.raises(OSError,
                       match="Expected to be able to read 16 bytes "
                             "for message body, got 8"):
        reader_new.read_all()

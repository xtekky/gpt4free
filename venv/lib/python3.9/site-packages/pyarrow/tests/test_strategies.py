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

import hypothesis as h

import pyarrow as pa
import pyarrow.tests.strategies as past


@h.given(past.all_types)
def test_types(ty):
    assert isinstance(ty, pa.lib.DataType)


@h.given(past.all_fields)
def test_fields(field):
    assert isinstance(field, pa.lib.Field)


@h.given(past.all_schemas)
def test_schemas(schema):
    assert isinstance(schema, pa.lib.Schema)


@h.given(past.all_arrays)
def test_arrays(array):
    assert isinstance(array, pa.lib.Array)


@h.given(past.arrays(past.primitive_types, nullable=False))
def test_array_nullability(array):
    assert array.null_count == 0


@h.given(past.all_chunked_arrays)
def test_chunked_arrays(chunked_array):
    assert isinstance(chunked_array, pa.lib.ChunkedArray)


@h.given(past.all_record_batches)
def test_record_batches(record_bath):
    assert isinstance(record_bath, pa.lib.RecordBatch)


@h.given(past.all_tables)
def test_tables(table):
    assert isinstance(table, pa.lib.Table)

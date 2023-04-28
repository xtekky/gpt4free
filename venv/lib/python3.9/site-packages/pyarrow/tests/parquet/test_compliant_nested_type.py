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

import pytest

import pyarrow as pa
from pyarrow.tests.parquet.common import parametrize_legacy_dataset

try:
    import pyarrow.parquet as pq
    from pyarrow.tests.parquet.common import (_read_table,
                                              _check_roundtrip)
except ImportError:
    pq = None

try:
    import pandas as pd
    import pandas.testing as tm

    from pyarrow.tests.parquet.common import _roundtrip_pandas_dataframe
except ImportError:
    pd = tm = None


# Marks all of the tests in this module
# Ignore these with pytest ... -m 'not parquet'
pytestmark = pytest.mark.parquet


# Tests for ARROW-11497
_test_data_simple = [
    {'items': [1, 2]},
    {'items': [0]},
]

_test_data_complex = [
    {'items': [{'name': 'elem1', 'value': '1'},
               {'name': 'elem2', 'value': '2'}]},
    {'items': [{'name': 'elem1', 'value': '0'}]},
]

parametrize_test_data = pytest.mark.parametrize(
    "test_data", [_test_data_simple, _test_data_complex])


@pytest.mark.pandas
@parametrize_legacy_dataset
@parametrize_test_data
def test_write_compliant_nested_type_enable(tempdir,
                                            use_legacy_dataset, test_data):
    # prepare dataframe for testing
    df = pd.DataFrame(data=test_data)
    # verify that we can read/write pandas df with new flag
    _roundtrip_pandas_dataframe(df,
                                write_kwargs={
                                    'use_compliant_nested_type': True},
                                use_legacy_dataset=use_legacy_dataset)

    # Write to a parquet file with compliant nested type
    table = pa.Table.from_pandas(df, preserve_index=False)
    path = str(tempdir / 'data.parquet')
    with pq.ParquetWriter(path, table.schema,
                          use_compliant_nested_type=True,
                          version='2.6') as writer:
        writer.write_table(table)
    # Read back as a table
    new_table = _read_table(path)
    # Validate that "items" columns compliant to Parquet nested format
    # Should be like this: list<element: struct<name: string, value: string>>
    assert isinstance(new_table.schema.types[0], pa.ListType)
    assert new_table.schema.types[0].value_field.name == 'element'

    # Verify that the new table can be read/written correctly
    _check_roundtrip(new_table,
                     use_legacy_dataset=use_legacy_dataset,
                     use_compliant_nested_type=True)


@pytest.mark.pandas
@parametrize_legacy_dataset
@parametrize_test_data
def test_write_compliant_nested_type_disable(tempdir,
                                             use_legacy_dataset, test_data):
    # prepare dataframe for testing
    df = pd.DataFrame(data=test_data)
    # verify that we can read/write with new flag disabled (default behaviour)
    _roundtrip_pandas_dataframe(df, write_kwargs={},
                                use_legacy_dataset=use_legacy_dataset)

    # Write to a parquet file while disabling compliant nested type
    table = pa.Table.from_pandas(df, preserve_index=False)
    path = str(tempdir / 'data.parquet')
    with pq.ParquetWriter(path, table.schema, version='2.6') as writer:
        writer.write_table(table)
    new_table = _read_table(path)

    # Validate that "items" columns is not compliant to Parquet nested format
    # Should be like this: list<item: struct<name: string, value: string>>
    assert isinstance(new_table.schema.types[0], pa.ListType)
    assert new_table.schema.types[0].value_field.name == 'item'

    # Verify that the new table can be read/written correctly
    _check_roundtrip(new_table,
                     use_legacy_dataset=use_legacy_dataset,
                     use_compliant_nested_type=False)

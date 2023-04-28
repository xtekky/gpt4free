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

import io

import numpy as np
import pytest

import pyarrow as pa
from pyarrow.tests import util

legacy_filter_mark = pytest.mark.filterwarnings(
    "ignore:Passing 'use_legacy:FutureWarning"
)

parametrize_legacy_dataset = pytest.mark.parametrize(
    "use_legacy_dataset",
    [pytest.param(True, marks=legacy_filter_mark),
     pytest.param(False, marks=pytest.mark.dataset)]
)
parametrize_legacy_dataset_not_supported = pytest.mark.parametrize(
    "use_legacy_dataset",
    [pytest.param(True, marks=legacy_filter_mark),
     pytest.param(False, marks=pytest.mark.skip)]
)
parametrize_legacy_dataset_fixed = pytest.mark.parametrize(
    "use_legacy_dataset",
    [pytest.param(True, marks=[pytest.mark.xfail, legacy_filter_mark]),
     pytest.param(False, marks=pytest.mark.dataset)]
)


def _write_table(table, path, **kwargs):
    # So we see the ImportError somewhere
    import pyarrow.parquet as pq
    from pyarrow.pandas_compat import _pandas_api

    if _pandas_api.is_data_frame(table):
        table = pa.Table.from_pandas(table)

    pq.write_table(table, path, **kwargs)
    return table


def _read_table(*args, **kwargs):
    import pyarrow.parquet as pq

    table = pq.read_table(*args, **kwargs)
    table.validate(full=True)
    return table


def _roundtrip_table(table, read_table_kwargs=None,
                     write_table_kwargs=None, use_legacy_dataset=False):
    read_table_kwargs = read_table_kwargs or {}
    write_table_kwargs = write_table_kwargs or {}

    writer = pa.BufferOutputStream()
    _write_table(table, writer, **write_table_kwargs)
    reader = pa.BufferReader(writer.getvalue())
    return _read_table(reader, use_legacy_dataset=use_legacy_dataset,
                       **read_table_kwargs)


def _check_roundtrip(table, expected=None, read_table_kwargs=None,
                     use_legacy_dataset=False, **write_table_kwargs):
    if expected is None:
        expected = table

    read_table_kwargs = read_table_kwargs or {}

    # intentionally check twice
    result = _roundtrip_table(table, read_table_kwargs=read_table_kwargs,
                              write_table_kwargs=write_table_kwargs,
                              use_legacy_dataset=use_legacy_dataset)
    assert result.equals(expected)
    result = _roundtrip_table(result, read_table_kwargs=read_table_kwargs,
                              write_table_kwargs=write_table_kwargs,
                              use_legacy_dataset=use_legacy_dataset)
    assert result.equals(expected)


def _roundtrip_pandas_dataframe(df, write_kwargs, use_legacy_dataset=False):
    table = pa.Table.from_pandas(df)
    result = _roundtrip_table(
        table, write_table_kwargs=write_kwargs,
        use_legacy_dataset=use_legacy_dataset)
    return result.to_pandas()


def _random_integers(size, dtype):
    # We do not generate integers outside the int64 range
    platform_int_info = np.iinfo('int_')
    iinfo = np.iinfo(dtype)
    return np.random.randint(max(iinfo.min, platform_int_info.min),
                             min(iinfo.max, platform_int_info.max),
                             size=size).astype(dtype)


def _test_dataframe(size=10000, seed=0):
    import pandas as pd

    np.random.seed(seed)
    df = pd.DataFrame({
        'uint8': _random_integers(size, np.uint8),
        'uint16': _random_integers(size, np.uint16),
        'uint32': _random_integers(size, np.uint32),
        'uint64': _random_integers(size, np.uint64),
        'int8': _random_integers(size, np.int8),
        'int16': _random_integers(size, np.int16),
        'int32': _random_integers(size, np.int32),
        'int64': _random_integers(size, np.int64),
        'float32': np.random.randn(size).astype(np.float32),
        'float64': np.arange(size, dtype=np.float64),
        'bool': np.random.randn(size) > 0,
        'strings': [util.rands(10) for i in range(size)],
        'all_none': [None] * size,
        'all_none_category': [None] * size
    })

    # TODO(PARQUET-1015)
    # df['all_none_category'] = df['all_none_category'].astype('category')
    return df


def make_sample_file(table_or_df):
    import pyarrow.parquet as pq

    if isinstance(table_or_df, pa.Table):
        a_table = table_or_df
    else:
        a_table = pa.Table.from_pandas(table_or_df)

    buf = io.BytesIO()
    _write_table(a_table, buf, compression='SNAPPY', version='2.6',
                 coerce_timestamps='ms')

    buf.seek(0)
    return pq.ParquetFile(buf)


def alltypes_sample(size=10000, seed=0, categorical=False):
    import pandas as pd

    np.random.seed(seed)
    arrays = {
        'uint8': np.arange(size, dtype=np.uint8),
        'uint16': np.arange(size, dtype=np.uint16),
        'uint32': np.arange(size, dtype=np.uint32),
        'uint64': np.arange(size, dtype=np.uint64),
        'int8': np.arange(size, dtype=np.int16),
        'int16': np.arange(size, dtype=np.int16),
        'int32': np.arange(size, dtype=np.int32),
        'int64': np.arange(size, dtype=np.int64),
        'float32': np.arange(size, dtype=np.float32),
        'float64': np.arange(size, dtype=np.float64),
        'bool': np.random.randn(size) > 0,
        # TODO(wesm): Test other timestamp resolutions now that arrow supports
        # them
        'datetime': np.arange("2016-01-01T00:00:00.001", size,
                              dtype='datetime64[ms]').astype('datetime64[ns]'),
        'timedelta': np.arange(0, size, dtype="timedelta64[ns]"),
        'str': pd.Series([str(x) for x in range(size)]),
        'empty_str': [''] * size,
        'str_with_nulls': [None] + [str(x) for x in range(size - 2)] + [None],
        'null': [None] * size,
        'null_list': [None] * 2 + [[None] * (x % 4) for x in range(size - 2)],
    }
    if categorical:
        arrays['str_category'] = arrays['str'].astype('category')
    return pd.DataFrame(arrays)

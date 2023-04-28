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
import io
import warnings

import numpy as np
import pytest

import pyarrow as pa
from pyarrow.tests.parquet.common import (
    _check_roundtrip, parametrize_legacy_dataset)

try:
    import pyarrow.parquet as pq
    from pyarrow.tests.parquet.common import _read_table, _write_table
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


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_pandas_parquet_datetime_tz(use_legacy_dataset):
    s = pd.Series([datetime.datetime(2017, 9, 6)])
    s = s.dt.tz_localize('utc')

    s.index = s

    # Both a column and an index to hit both use cases
    df = pd.DataFrame({'tz_aware': s,
                       'tz_eastern': s.dt.tz_convert('US/Eastern')},
                      index=s)

    f = io.BytesIO()

    arrow_table = pa.Table.from_pandas(df)

    _write_table(arrow_table, f, coerce_timestamps='ms')
    f.seek(0)

    table_read = pq.read_pandas(f, use_legacy_dataset=use_legacy_dataset)

    df_read = table_read.to_pandas()
    tm.assert_frame_equal(df, df_read)


@pytest.mark.pandas
@parametrize_legacy_dataset
def test_datetime_timezone_tzinfo(use_legacy_dataset):
    value = datetime.datetime(2018, 1, 1, 1, 23, 45,
                              tzinfo=datetime.timezone.utc)
    df = pd.DataFrame({'foo': [value]})

    _roundtrip_pandas_dataframe(
        df, write_kwargs={}, use_legacy_dataset=use_legacy_dataset)


@pytest.mark.pandas
def test_coerce_timestamps(tempdir):
    from collections import OrderedDict

    # ARROW-622
    arrays = OrderedDict()
    fields = [pa.field('datetime64',
                       pa.list_(pa.timestamp('ms')))]
    arrays['datetime64'] = [
        np.array(['2007-07-13T01:23:34.123456789',
                  None,
                  '2010-08-13T05:46:57.437699912'],
                 dtype='datetime64[ms]'),
        None,
        None,
        np.array(['2007-07-13T02',
                  None,
                  '2010-08-13T05:46:57.437699912'],
                 dtype='datetime64[ms]'),
    ]

    df = pd.DataFrame(arrays)
    schema = pa.schema(fields)

    filename = tempdir / 'pandas_roundtrip.parquet'
    arrow_table = pa.Table.from_pandas(df, schema=schema)

    _write_table(arrow_table, filename, version='2.6', coerce_timestamps='us')
    table_read = _read_table(filename)
    df_read = table_read.to_pandas()

    df_expected = df.copy()
    for i, x in enumerate(df_expected['datetime64']):
        if isinstance(x, np.ndarray):
            df_expected['datetime64'][i] = x.astype('M8[us]')

    tm.assert_frame_equal(df_expected, df_read)

    with pytest.raises(ValueError):
        _write_table(arrow_table, filename, version='2.6',
                     coerce_timestamps='unknown')


@pytest.mark.pandas
def test_coerce_timestamps_truncated(tempdir):
    """
    ARROW-2555: Test that we can truncate timestamps when coercing if
    explicitly allowed.
    """
    dt_us = datetime.datetime(year=2017, month=1, day=1, hour=1, minute=1,
                              second=1, microsecond=1)
    dt_ms = datetime.datetime(year=2017, month=1, day=1, hour=1, minute=1,
                              second=1)

    fields_us = [pa.field('datetime64', pa.timestamp('us'))]
    arrays_us = {'datetime64': [dt_us, dt_ms]}

    df_us = pd.DataFrame(arrays_us)
    schema_us = pa.schema(fields_us)

    filename = tempdir / 'pandas_truncated.parquet'
    table_us = pa.Table.from_pandas(df_us, schema=schema_us)

    _write_table(table_us, filename, version='2.6', coerce_timestamps='ms',
                 allow_truncated_timestamps=True)
    table_ms = _read_table(filename)
    df_ms = table_ms.to_pandas()

    arrays_expected = {'datetime64': [dt_ms, dt_ms]}
    df_expected = pd.DataFrame(arrays_expected)
    tm.assert_frame_equal(df_expected, df_ms)


@pytest.mark.pandas
def test_date_time_types(tempdir):
    t1 = pa.date32()
    data1 = np.array([17259, 17260, 17261], dtype='int32')
    a1 = pa.array(data1, type=t1)

    t2 = pa.date64()
    data2 = data1.astype('int64') * 86400000
    a2 = pa.array(data2, type=t2)

    t3 = pa.timestamp('us')
    start = pd.Timestamp('2001-01-01').value / 1000
    data3 = np.array([start, start + 1, start + 2], dtype='int64')
    a3 = pa.array(data3, type=t3)

    t4 = pa.time32('ms')
    data4 = np.arange(3, dtype='i4')
    a4 = pa.array(data4, type=t4)

    t5 = pa.time64('us')
    a5 = pa.array(data4.astype('int64'), type=t5)

    t6 = pa.time32('s')
    a6 = pa.array(data4, type=t6)

    ex_t6 = pa.time32('ms')
    ex_a6 = pa.array(data4 * 1000, type=ex_t6)

    t7 = pa.timestamp('ns')
    start = pd.Timestamp('2001-01-01').value
    data7 = np.array([start, start + 1000, start + 2000],
                     dtype='int64')
    a7 = pa.array(data7, type=t7)

    table = pa.Table.from_arrays([a1, a2, a3, a4, a5, a6, a7],
                                 ['date32', 'date64', 'timestamp[us]',
                                  'time32[s]', 'time64[us]',
                                  'time32_from64[s]',
                                  'timestamp[ns]'])

    # date64 as date32
    # time32[s] to time32[ms]
    expected = pa.Table.from_arrays([a1, a1, a3, a4, a5, ex_a6, a7],
                                    ['date32', 'date64', 'timestamp[us]',
                                     'time32[s]', 'time64[us]',
                                     'time32_from64[s]',
                                     'timestamp[ns]'])

    _check_roundtrip(table, expected=expected, version='2.6')

    t0 = pa.timestamp('ms')
    data0 = np.arange(4, dtype='int64')
    a0 = pa.array(data0, type=t0)

    t1 = pa.timestamp('us')
    data1 = np.arange(4, dtype='int64')
    a1 = pa.array(data1, type=t1)

    t2 = pa.timestamp('ns')
    data2 = np.arange(4, dtype='int64')
    a2 = pa.array(data2, type=t2)

    table = pa.Table.from_arrays([a0, a1, a2],
                                 ['ts[ms]', 'ts[us]', 'ts[ns]'])
    expected = pa.Table.from_arrays([a0, a1, a2],
                                    ['ts[ms]', 'ts[us]', 'ts[ns]'])

    # int64 for all timestamps supported by default
    filename = tempdir / 'int64_timestamps.parquet'
    _write_table(table, filename, version='2.6')
    parquet_schema = pq.ParquetFile(filename).schema
    for i in range(3):
        assert parquet_schema.column(i).physical_type == 'INT64'
    read_table = _read_table(filename)
    assert read_table.equals(expected)

    t0_ns = pa.timestamp('ns')
    data0_ns = np.array(data0 * 1000000, dtype='int64')
    a0_ns = pa.array(data0_ns, type=t0_ns)

    t1_ns = pa.timestamp('ns')
    data1_ns = np.array(data1 * 1000, dtype='int64')
    a1_ns = pa.array(data1_ns, type=t1_ns)

    expected = pa.Table.from_arrays([a0_ns, a1_ns, a2],
                                    ['ts[ms]', 'ts[us]', 'ts[ns]'])

    # int96 nanosecond timestamps produced upon request
    filename = tempdir / 'explicit_int96_timestamps.parquet'
    _write_table(table, filename, version='2.6',
                 use_deprecated_int96_timestamps=True)
    parquet_schema = pq.ParquetFile(filename).schema
    for i in range(3):
        assert parquet_schema.column(i).physical_type == 'INT96'
    read_table = _read_table(filename)
    assert read_table.equals(expected)

    # int96 nanosecond timestamps implied by flavor 'spark'
    filename = tempdir / 'spark_int96_timestamps.parquet'
    _write_table(table, filename, version='2.6',
                 flavor='spark')
    parquet_schema = pq.ParquetFile(filename).schema
    for i in range(3):
        assert parquet_schema.column(i).physical_type == 'INT96'
    read_table = _read_table(filename)
    assert read_table.equals(expected)


@pytest.mark.pandas
@pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
def test_coerce_int96_timestamp_unit(unit):
    i_s = pd.Timestamp('2010-01-01').value / 1000000000  # := 1262304000

    d_s = np.arange(i_s, i_s + 10, 1, dtype='int64')
    d_ms = d_s * 1000
    d_us = d_ms * 1000
    d_ns = d_us * 1000

    a_s = pa.array(d_s, type=pa.timestamp('s'))
    a_ms = pa.array(d_ms, type=pa.timestamp('ms'))
    a_us = pa.array(d_us, type=pa.timestamp('us'))
    a_ns = pa.array(d_ns, type=pa.timestamp('ns'))

    arrays = {"s": a_s, "ms": a_ms, "us": a_us, "ns": a_ns}
    names = ['ts_s', 'ts_ms', 'ts_us', 'ts_ns']
    table = pa.Table.from_arrays([a_s, a_ms, a_us, a_ns], names)

    # For either Parquet version, coercing to nanoseconds is allowed
    # if Int96 storage is used
    expected = pa.Table.from_arrays([arrays.get(unit)]*4, names)
    read_table_kwargs = {"coerce_int96_timestamp_unit": unit}
    _check_roundtrip(table, expected,
                     read_table_kwargs=read_table_kwargs,
                     use_deprecated_int96_timestamps=True)
    _check_roundtrip(table, expected, version='2.6',
                     read_table_kwargs=read_table_kwargs,
                     use_deprecated_int96_timestamps=True)


@pytest.mark.pandas
@pytest.mark.parametrize('pq_reader_method', ['ParquetFile', 'read_table'])
def test_coerce_int96_timestamp_overflow(pq_reader_method, tempdir):

    def get_table(pq_reader_method, filename, **kwargs):
        if pq_reader_method == "ParquetFile":
            return pq.ParquetFile(filename, **kwargs).read()
        elif pq_reader_method == "read_table":
            return pq.read_table(filename, **kwargs)

    # Recreating the initial JIRA issue referenced in ARROW-12096
    oob_dts = [
        datetime.datetime(1000, 1, 1),
        datetime.datetime(2000, 1, 1),
        datetime.datetime(3000, 1, 1)
    ]
    df = pd.DataFrame({"a": oob_dts})
    table = pa.table(df)

    filename = tempdir / "test_round_trip_overflow.parquet"
    pq.write_table(table, filename, use_deprecated_int96_timestamps=True,
                   version="1.0")

    # with the default resolution of ns, we get wrong values for INT96
    # that are out of bounds for nanosecond range
    tab_error = get_table(pq_reader_method, filename)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                                "Discarding nonzero nanoseconds in conversion",
                                UserWarning)
        assert tab_error["a"].to_pylist() != oob_dts

    # avoid this overflow by specifying the resolution to use for INT96 values
    tab_correct = get_table(
        pq_reader_method, filename, coerce_int96_timestamp_unit="s"
    )
    df_correct = tab_correct.to_pandas(timestamp_as_object=True)
    tm.assert_frame_equal(df, df_correct)


def test_timestamp_restore_timezone():
    # ARROW-5888, restore timezone from serialized metadata
    ty = pa.timestamp('ms', tz='America/New_York')
    arr = pa.array([1, 2, 3], type=ty)
    t = pa.table([arr], names=['f0'])
    _check_roundtrip(t)


def test_timestamp_restore_timezone_nanosecond():
    # ARROW-9634, also restore timezone for nanosecond data that get stored
    # as microseconds in the parquet file
    ty = pa.timestamp('ns', tz='America/New_York')
    arr = pa.array([1000, 2000, 3000], type=ty)
    table = pa.table([arr], names=['f0'])
    ty_us = pa.timestamp('us', tz='America/New_York')
    expected = pa.table([arr.cast(ty_us)], names=['f0'])
    _check_roundtrip(table, expected=expected)


@pytest.mark.pandas
def test_list_of_datetime_time_roundtrip():
    # ARROW-4135
    times = pd.to_datetime(['09:00', '09:30', '10:00', '10:30', '11:00',
                            '11:30', '12:00'])
    df = pd.DataFrame({'time': [times.time]})
    _roundtrip_pandas_dataframe(df, write_kwargs={})


@pytest.mark.pandas
def test_parquet_version_timestamp_differences():
    i_s = pd.Timestamp('2010-01-01').value / 1000000000  # := 1262304000

    d_s = np.arange(i_s, i_s + 10, 1, dtype='int64')
    d_ms = d_s * 1000
    d_us = d_ms * 1000
    d_ns = d_us * 1000

    a_s = pa.array(d_s, type=pa.timestamp('s'))
    a_ms = pa.array(d_ms, type=pa.timestamp('ms'))
    a_us = pa.array(d_us, type=pa.timestamp('us'))
    a_ns = pa.array(d_ns, type=pa.timestamp('ns'))

    names = ['ts:s', 'ts:ms', 'ts:us', 'ts:ns']
    table = pa.Table.from_arrays([a_s, a_ms, a_us, a_ns], names)

    # Using Parquet version 1.0, seconds should be coerced to milliseconds
    # and nanoseconds should be coerced to microseconds by default
    expected = pa.Table.from_arrays([a_ms, a_ms, a_us, a_us], names)
    _check_roundtrip(table, expected)

    # Using Parquet version 2.0, seconds should be coerced to milliseconds
    # and nanoseconds should be retained by default
    expected = pa.Table.from_arrays([a_ms, a_ms, a_us, a_ns], names)
    _check_roundtrip(table, expected, version='2.6')

    # Using Parquet version 1.0, coercing to milliseconds or microseconds
    # is allowed
    expected = pa.Table.from_arrays([a_ms, a_ms, a_ms, a_ms], names)
    _check_roundtrip(table, expected, coerce_timestamps='ms')

    # Using Parquet version 2.0, coercing to milliseconds or microseconds
    # is allowed
    expected = pa.Table.from_arrays([a_us, a_us, a_us, a_us], names)
    _check_roundtrip(table, expected, version='2.6', coerce_timestamps='us')

    # TODO: after pyarrow allows coerce_timestamps='ns', tests like the
    # following should pass ...

    # Using Parquet version 1.0, coercing to nanoseconds is not allowed
    # expected = None
    # with pytest.raises(NotImplementedError):
    #     _roundtrip_table(table, coerce_timestamps='ns')

    # Using Parquet version 2.0, coercing to nanoseconds is allowed
    # expected = pa.Table.from_arrays([a_ns, a_ns, a_ns, a_ns], names)
    # _check_roundtrip(table, expected, version='2.6', coerce_timestamps='ns')

    # For either Parquet version, coercing to nanoseconds is allowed
    # if Int96 storage is used
    expected = pa.Table.from_arrays([a_ns, a_ns, a_ns, a_ns], names)
    _check_roundtrip(table, expected,
                     use_deprecated_int96_timestamps=True)
    _check_roundtrip(table, expected, version='2.6',
                     use_deprecated_int96_timestamps=True)


@pytest.mark.pandas
def test_noncoerced_nanoseconds_written_without_exception(tempdir):
    # ARROW-1957: the Parquet version 2.0 writer preserves Arrow
    # nanosecond timestamps by default
    n = 9
    df = pd.DataFrame({'x': range(n)},
                      index=pd.date_range('2017-01-01', freq='1n', periods=n))
    tb = pa.Table.from_pandas(df)

    filename = tempdir / 'written.parquet'
    try:
        pq.write_table(tb, filename, version='2.6')
    except Exception:
        pass
    assert filename.exists()

    recovered_table = pq.read_table(filename)
    assert tb.equals(recovered_table)

    # Loss of data through coercion (without explicit override) still an error
    filename = tempdir / 'not_written.parquet'
    with pytest.raises(ValueError):
        pq.write_table(tb, filename, coerce_timestamps='ms', version='2.6')


def test_duration_type():
    # ARROW-6780
    arrays = [pa.array([0, 1, 2, 3], type=pa.duration(unit))
              for unit in ["s", "ms", "us", "ns"]]
    table = pa.Table.from_arrays(arrays, ["d[s]", "d[ms]", "d[us]", "d[ns]"])

    _check_roundtrip(table)

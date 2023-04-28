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
from datetime import date, time

import numpy as np
import pandas as pd
import pyarrow as pa


def dataframe_with_arrays(include_index=False):
    """
    Dataframe with numpy arrays columns of every possible primitive type.

    Returns
    -------
    df: pandas.DataFrame
    schema: pyarrow.Schema
        Arrow schema definition that is in line with the constructed df.
    """
    dtypes = [('i1', pa.int8()), ('i2', pa.int16()),
              ('i4', pa.int32()), ('i8', pa.int64()),
              ('u1', pa.uint8()), ('u2', pa.uint16()),
              ('u4', pa.uint32()), ('u8', pa.uint64()),
              ('f4', pa.float32()), ('f8', pa.float64())]

    arrays = OrderedDict()
    fields = []
    for dtype, arrow_dtype in dtypes:
        fields.append(pa.field(dtype, pa.list_(arrow_dtype)))
        arrays[dtype] = [
            np.arange(10, dtype=dtype),
            np.arange(5, dtype=dtype),
            None,
            np.arange(1, dtype=dtype)
        ]

    fields.append(pa.field('str', pa.list_(pa.string())))
    arrays['str'] = [
        np.array(["1", "ä"], dtype="object"),
        None,
        np.array(["1"], dtype="object"),
        np.array(["1", "2", "3"], dtype="object")
    ]

    fields.append(pa.field('datetime64', pa.list_(pa.timestamp('ms'))))
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

    if include_index:
        fields.append(pa.field('__index_level_0__', pa.int64()))
    df = pd.DataFrame(arrays)
    schema = pa.schema(fields)

    return df, schema


def dataframe_with_lists(include_index=False, parquet_compatible=False):
    """
    Dataframe with list columns of every possible primitive type.

    Returns
    -------
    df: pandas.DataFrame
    schema: pyarrow.Schema
        Arrow schema definition that is in line with the constructed df.
    parquet_compatible: bool
        Exclude types not supported by parquet
    """
    arrays = OrderedDict()
    fields = []

    fields.append(pa.field('int64', pa.list_(pa.int64())))
    arrays['int64'] = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4],
        None,
        [],
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 2,
                 dtype=np.int64)[::2]
    ]
    fields.append(pa.field('double', pa.list_(pa.float64())))
    arrays['double'] = [
        [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.],
        [0., 1., 2., 3., 4.],
        None,
        [],
        np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.] * 2)[::2],
    ]
    fields.append(pa.field('bytes_list', pa.list_(pa.binary())))
    arrays['bytes_list'] = [
        [b"1", b"f"],
        None,
        [b"1"],
        [b"1", b"2", b"3"],
        [],
    ]
    fields.append(pa.field('str_list', pa.list_(pa.string())))
    arrays['str_list'] = [
        ["1", "ä"],
        None,
        ["1"],
        ["1", "2", "3"],
        [],
    ]

    date_data = [
        [],
        [date(2018, 1, 1), date(2032, 12, 30)],
        [date(2000, 6, 7)],
        None,
        [date(1969, 6, 9), date(1972, 7, 3)]
    ]
    time_data = [
        [time(23, 11, 11), time(1, 2, 3), time(23, 59, 59)],
        [],
        [time(22, 5, 59)],
        None,
        [time(0, 0, 0), time(18, 0, 2), time(12, 7, 3)]
    ]

    temporal_pairs = [
        (pa.date32(), date_data),
        (pa.date64(), date_data),
        (pa.time32('s'), time_data),
        (pa.time32('ms'), time_data),
        (pa.time64('us'), time_data)
    ]
    if not parquet_compatible:
        temporal_pairs += [
            (pa.time64('ns'), time_data),
        ]

    for value_type, data in temporal_pairs:
        field_name = '{}_list'.format(value_type)
        field_type = pa.list_(value_type)
        field = pa.field(field_name, field_type)
        fields.append(field)
        arrays[field_name] = data

    if include_index:
        fields.append(pa.field('__index_level_0__', pa.int64()))

    df = pd.DataFrame(arrays)
    schema = pa.schema(fields)

    return df, schema

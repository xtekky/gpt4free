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

import numpy as np
import pyarrow as pa

import pyarrow.tests.util as test_util

try:
    import pandas as pd
except ImportError:
    pass


@pytest.mark.memory_leak
@pytest.mark.pandas
def test_deserialize_pandas_arrow_7956():
    df = pd.DataFrame({'a': np.arange(10000),
                       'b': [test_util.rands(5) for _ in range(10000)]})

    def action():
        df_bytes = pa.ipc.serialize_pandas(df).to_pybytes()
        buf = pa.py_buffer(df_bytes)
        pa.ipc.deserialize_pandas(buf)

    # Abort at 128MB threshold
    test_util.memory_leak_check(action, threshold=1 << 27, iterations=100)

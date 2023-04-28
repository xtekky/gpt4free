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

# This file is called from a test in test_pandas.py.

from concurrent.futures import ThreadPoolExecutor
import faulthandler
import sys

import pyarrow as pa

num_threads = 60
timeout = 10  # seconds


def thread_func(i):
    pa.array([i]).to_pandas()


def main():
    # In case of import deadlock, crash after a finite timeout
    faulthandler.dump_traceback_later(timeout, exit=True)
    with ThreadPoolExecutor(num_threads) as pool:
        assert "pandas" not in sys.modules  # pandas is imported lazily
        list(pool.map(thread_func, range(num_threads)))
        assert "pandas" in sys.modules


if __name__ == "__main__":
    main()

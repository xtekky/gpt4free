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
import signal
import sys
import weakref

import pytest

from pyarrow import util
from pyarrow.tests.util import disabled_gc


def exhibit_signal_refcycle():
    # Put an object in the frame locals and return a weakref to it.
    # If `signal.getsignal` has a bug where it creates a reference cycle
    # keeping alive the current execution frames, `obj` will not be
    # destroyed immediately when this function returns.
    obj = set()
    signal.getsignal(signal.SIGINT)
    return weakref.ref(obj)


def test_signal_refcycle():
    # Test possible workaround for https://bugs.python.org/issue42248
    with disabled_gc():
        wr = exhibit_signal_refcycle()
        if wr() is None:
            pytest.skip(
                "Python version does not have the bug we're testing for")

    gc.collect()
    with disabled_gc():
        wr = exhibit_signal_refcycle()
        assert wr() is not None
        util._break_traceback_cycle_from_frame(sys._getframe(0))
        assert wr() is None

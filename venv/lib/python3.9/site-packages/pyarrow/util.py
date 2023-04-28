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

# Miscellaneous utility code

import os
import contextlib
import functools
import gc
import socket
import sys
import types
import warnings


_DEPR_MSG = (
    "pyarrow.{} is deprecated as of {}, please use pyarrow.{} instead."
)


def implements(f):
    def decorator(g):
        g.__doc__ = f.__doc__
        return g
    return decorator


def _deprecate_api(old_name, new_name, api, next_version, type=FutureWarning):
    msg = _DEPR_MSG.format(old_name, next_version, new_name)

    def wrapper(*args, **kwargs):
        warnings.warn(msg, type)
        return api(*args, **kwargs)
    return wrapper


def _deprecate_class(old_name, new_class, next_version,
                     instancecheck=True):
    """
    Raise warning if a deprecated class is used in an isinstance check.
    """
    class _DeprecatedMeta(type):
        def __instancecheck__(self, other):
            warnings.warn(
                _DEPR_MSG.format(old_name, next_version, new_class.__name__),
                FutureWarning,
                stacklevel=2
            )
            return isinstance(other, new_class)

    return _DeprecatedMeta(old_name, (new_class,), {})


def _is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def _is_path_like(path):
    return isinstance(path, str) or hasattr(path, '__fspath__')


def _stringify_path(path):
    """
    Convert *path* to a string or unicode path if possible.
    """
    if isinstance(path, str):
        return os.path.expanduser(path)

    # checking whether path implements the filesystem protocol
    try:
        return os.path.expanduser(path.__fspath__())
    except AttributeError:
        pass

    raise TypeError("not a path-like object")


def product(seq):
    """
    Return a product of sequence items.
    """
    return functools.reduce(lambda a, b: a*b, seq, 1)


def get_contiguous_span(shape, strides, itemsize):
    """
    Return a contiguous span of N-D array data.

    Parameters
    ----------
    shape : tuple
    strides : tuple
    itemsize : int
      Specify array shape data

    Returns
    -------
    start, end : int
      The span end points.
    """
    if not strides:
        start = 0
        end = itemsize * product(shape)
    else:
        start = 0
        end = itemsize
        for i, dim in enumerate(shape):
            if dim == 0:
                start = end = 0
                break
            stride = strides[i]
            if stride > 0:
                end += stride * (dim - 1)
            elif stride < 0:
                start += stride * (dim - 1)
        if end - start != itemsize * product(shape):
            raise ValueError('array data is non-contiguous')
    return start, end


def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with contextlib.closing(sock) as sock:
        sock.bind(('', 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.getsockname()[1]


def guid():
    from uuid import uuid4
    return uuid4().hex


def _break_traceback_cycle_from_frame(frame):
    # Clear local variables in all inner frames, so as to break the
    # reference cycle.
    this_frame = sys._getframe(0)
    refs = gc.get_referrers(frame)
    while refs:
        for frame in refs:
            if frame is not this_frame and isinstance(frame, types.FrameType):
                break
        else:
            # No frame found in referrers (finished?)
            break
        refs = None
        # Clear the frame locals, to try and break the cycle (it is
        # somewhere along the chain of execution frames).
        frame.clear()
        # To visit the inner frame, we need to find it among the
        # referrers of this frame (while `frame.f_back` would let
        # us visit the outer frame).
        refs = gc.get_referrers(frame)
    refs = frame = this_frame = None

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


def encode_file_path(path):
    if isinstance(path, str):
        # POSIX systems can handle utf-8. UTF8 is converted to utf16-le in
        # libarrow
        encoded_path = path.encode('utf-8')
    else:
        encoded_path = path

    # Windows file system requires utf-16le for file names; Arrow C++ libraries
    # will convert utf8 to utf16
    return encoded_path


# Starting with Python 3.7, dicts are guaranteed to be insertion-ordered.
ordered_dict = dict


try:
    import pickle5 as builtin_pickle
except ImportError:
    import pickle as builtin_pickle


try:
    import cloudpickle as pickle
except ImportError:
    pickle = builtin_pickle


def tobytes(o):
    """
    Encode a unicode or bytes string to bytes.

    Parameters
    ----------
    o : str or bytes
        Input string.
    """
    if isinstance(o, str):
        return o.encode('utf8')
    else:
        return o


def frombytes(o, *, safe=False):
    """
    Decode the given bytestring to unicode.

    Parameters
    ----------
    o : bytes-like
        Input object.
    safe : bool, default False
        If true, raise on encoding errors.
    """
    if safe:
        return o.decode('utf8', errors='replace')
    else:
        return o.decode('utf8')

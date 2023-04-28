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

# pandas lazy-loading API shim that reduces API call and import overhead

import warnings


cdef class _PandasAPIShim(object):
    """
    Lazy pandas importer that isolates usages of pandas APIs and avoids
    importing pandas until it's actually needed
    """
    cdef:
        bint _tried_importing_pandas
        bint _have_pandas

    cdef readonly:
        object _loose_version, _version
        object _pd, _types_api, _compat_module
        object _data_frame, _index, _series, _categorical_type
        object _datetimetz_type, _extension_array, _extension_dtype
        object _array_like_types, _is_extension_array_dtype
        bint has_sparse
        bint _pd024

    def __init__(self):
        self._tried_importing_pandas = False
        self._have_pandas = 0

    cdef _import_pandas(self, bint raise_):
        try:
            import pandas as pd
            import pyarrow.pandas_compat as pdcompat
        except ImportError:
            self._have_pandas = False
            if raise_:
                raise
            else:
                return

        from pyarrow.vendored.version import Version

        self._pd = pd
        self._version = pd.__version__
        self._loose_version = Version(pd.__version__)

        if self._loose_version < Version('1.0.0'):
            self._have_pandas = False
            if raise_:
                raise ImportError(
                    "pyarrow requires pandas 1.0.0 or above, pandas {} is "
                    "installed".format(self._version)
                )
            else:
                warnings.warn(
                    "pyarrow requires pandas 1.0.0 or above, pandas {} is "
                    "installed. Therefore, pandas-specific integration is not "
                    "used.".format(self._version), stacklevel=2)
                return

        self._compat_module = pdcompat
        self._data_frame = pd.DataFrame
        self._index = pd.Index
        self._categorical_type = pd.Categorical
        self._series = pd.Series
        self._extension_array = pd.api.extensions.ExtensionArray
        self._array_like_types = (
            self._series, self._index, self._categorical_type,
            self._extension_array)
        self._extension_dtype = pd.api.extensions.ExtensionDtype
        self._is_extension_array_dtype = (
            pd.api.types.is_extension_array_dtype)
        self._types_api = pd.api.types
        self._datetimetz_type = pd.api.types.DatetimeTZDtype
        self._have_pandas = True
        self.has_sparse = False

    cdef inline _check_import(self, bint raise_=True):
        if self._tried_importing_pandas:
            if not self._have_pandas and raise_:
                self._import_pandas(raise_)
            return

        self._tried_importing_pandas = True
        self._import_pandas(raise_)

    def series(self, *args, **kwargs):
        self._check_import()
        return self._series(*args, **kwargs)

    def data_frame(self, *args, **kwargs):
        self._check_import()
        return self._data_frame(*args, **kwargs)

    cdef inline bint _have_pandas_internal(self):
        if not self._tried_importing_pandas:
            self._check_import(raise_=False)
        return self._have_pandas

    @property
    def have_pandas(self):
        return self._have_pandas_internal()

    @property
    def compat(self):
        self._check_import()
        return self._compat_module

    @property
    def pd(self):
        self._check_import()
        return self._pd

    cpdef infer_dtype(self, obj):
        self._check_import()
        try:
            return self._types_api.infer_dtype(obj, skipna=False)
        except AttributeError:
            return self._pd.lib.infer_dtype(obj)

    cpdef pandas_dtype(self, dtype):
        self._check_import()
        try:
            return self._types_api.pandas_dtype(dtype)
        except AttributeError:
            return None

    @property
    def loose_version(self):
        self._check_import()
        return self._loose_version

    @property
    def version(self):
        self._check_import()
        return self._version

    @property
    def categorical_type(self):
        self._check_import()
        return self._categorical_type

    @property
    def datetimetz_type(self):
        self._check_import()
        return self._datetimetz_type

    @property
    def extension_dtype(self):
        self._check_import()
        return self._extension_dtype

    cpdef is_array_like(self, obj):
        self._check_import()
        return isinstance(obj, self._array_like_types)

    cpdef is_categorical(self, obj):
        if self._have_pandas_internal():
            return isinstance(obj, self._categorical_type)
        else:
            return False

    cpdef is_datetimetz(self, obj):
        if self._have_pandas_internal():
            return isinstance(obj, self._datetimetz_type)
        else:
            return False

    cpdef is_extension_array_dtype(self, obj):
        self._check_import()
        if self._is_extension_array_dtype:
            return self._is_extension_array_dtype(obj)
        else:
            return False

    cpdef is_sparse(self, obj):
        if self._have_pandas_internal():
            return self._types_api.is_sparse(obj)
        else:
            return False

    cpdef is_data_frame(self, obj):
        if self._have_pandas_internal():
            return isinstance(obj, self._data_frame)
        else:
            return False

    cpdef is_series(self, obj):
        if self._have_pandas_internal():
            return isinstance(obj, self._series)
        else:
            return False

    cpdef is_index(self, obj):
        if self._have_pandas_internal():
            return isinstance(obj, self._index)
        else:
            return False

    cpdef get_values(self, obj):
        """
        Get the underlying array values of a pandas Series or Index in the
        format (np.ndarray or pandas ExtensionArray) as we need them.

        Assumes obj is a pandas Series or Index.
        """
        self._check_import()
        if isinstance(obj.dtype, (self.pd.api.types.IntervalDtype,
                                  self.pd.api.types.PeriodDtype)):
            return obj.array
        return obj.values

    def assert_frame_equal(self, *args, **kwargs):
        self._check_import()
        return self._pd.util.testing.assert_frame_equal

    def get_rangeindex_attribute(self, level, name):
        # public start/stop/step attributes added in pandas 0.25.0
        self._check_import()
        if hasattr(level, name):
            return getattr(level, name)
        return getattr(level, '_' + name)


cdef _PandasAPIShim pandas_api = _PandasAPIShim()
_pandas_api = pandas_api

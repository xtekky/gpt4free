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


import ast
from collections.abc import Sequence
from concurrent import futures
# import threading submodule upfront to avoid partially initialized
# module bug (ARROW-11983)
import concurrent.futures.thread  # noqa
from copy import deepcopy
from itertools import zip_longest
import json
import operator
import re
import warnings

import numpy as np

import pyarrow as pa
from pyarrow.lib import _pandas_api, builtin_pickle, frombytes  # noqa


_logical_type_map = {}


def get_logical_type_map():
    global _logical_type_map

    if not _logical_type_map:
        _logical_type_map.update({
            pa.lib.Type_NA: 'empty',
            pa.lib.Type_BOOL: 'bool',
            pa.lib.Type_INT8: 'int8',
            pa.lib.Type_INT16: 'int16',
            pa.lib.Type_INT32: 'int32',
            pa.lib.Type_INT64: 'int64',
            pa.lib.Type_UINT8: 'uint8',
            pa.lib.Type_UINT16: 'uint16',
            pa.lib.Type_UINT32: 'uint32',
            pa.lib.Type_UINT64: 'uint64',
            pa.lib.Type_HALF_FLOAT: 'float16',
            pa.lib.Type_FLOAT: 'float32',
            pa.lib.Type_DOUBLE: 'float64',
            pa.lib.Type_DATE32: 'date',
            pa.lib.Type_DATE64: 'date',
            pa.lib.Type_TIME32: 'time',
            pa.lib.Type_TIME64: 'time',
            pa.lib.Type_BINARY: 'bytes',
            pa.lib.Type_FIXED_SIZE_BINARY: 'bytes',
            pa.lib.Type_STRING: 'unicode',
        })
    return _logical_type_map


def get_logical_type(arrow_type):
    logical_type_map = get_logical_type_map()

    try:
        return logical_type_map[arrow_type.id]
    except KeyError:
        if isinstance(arrow_type, pa.lib.DictionaryType):
            return 'categorical'
        elif isinstance(arrow_type, pa.lib.ListType):
            return 'list[{}]'.format(get_logical_type(arrow_type.value_type))
        elif isinstance(arrow_type, pa.lib.TimestampType):
            return 'datetimetz' if arrow_type.tz is not None else 'datetime'
        elif isinstance(arrow_type, pa.lib.Decimal128Type):
            return 'decimal'
        return 'object'


_numpy_logical_type_map = {
    np.bool_: 'bool',
    np.int8: 'int8',
    np.int16: 'int16',
    np.int32: 'int32',
    np.int64: 'int64',
    np.uint8: 'uint8',
    np.uint16: 'uint16',
    np.uint32: 'uint32',
    np.uint64: 'uint64',
    np.float32: 'float32',
    np.float64: 'float64',
    'datetime64[D]': 'date',
    np.unicode_: 'string',
    np.bytes_: 'bytes',
}


def get_logical_type_from_numpy(pandas_collection):
    try:
        return _numpy_logical_type_map[pandas_collection.dtype.type]
    except KeyError:
        if hasattr(pandas_collection.dtype, 'tz'):
            return 'datetimetz'
        # See https://github.com/pandas-dev/pandas/issues/24739
        if str(pandas_collection.dtype) == 'datetime64[ns]':
            return 'datetime64[ns]'
        result = _pandas_api.infer_dtype(pandas_collection)
        if result == 'string':
            return 'unicode'
        return result


def get_extension_dtype_info(column):
    dtype = column.dtype
    if str(dtype) == 'category':
        cats = getattr(column, 'cat', column)
        assert cats is not None
        metadata = {
            'num_categories': len(cats.categories),
            'ordered': cats.ordered,
        }
        physical_dtype = str(cats.codes.dtype)
    elif hasattr(dtype, 'tz'):
        metadata = {'timezone': pa.lib.tzinfo_to_string(dtype.tz)}
        physical_dtype = 'datetime64[ns]'
    else:
        metadata = None
        physical_dtype = str(dtype)
    return physical_dtype, metadata


def get_column_metadata(column, name, arrow_type, field_name):
    """Construct the metadata for a given column

    Parameters
    ----------
    column : pandas.Series or pandas.Index
    name : str
    arrow_type : pyarrow.DataType
    field_name : str
        Equivalent to `name` when `column` is a `Series`, otherwise if `column`
        is a pandas Index then `field_name` will not be the same as `name`.
        This is the name of the field in the arrow Table's schema.

    Returns
    -------
    dict
    """
    logical_type = get_logical_type(arrow_type)

    string_dtype, extra_metadata = get_extension_dtype_info(column)
    if logical_type == 'decimal':
        extra_metadata = {
            'precision': arrow_type.precision,
            'scale': arrow_type.scale,
        }
        string_dtype = 'object'

    if name is not None and not isinstance(name, str):
        raise TypeError(
            'Column name must be a string. Got column {} of type {}'.format(
                name, type(name).__name__
            )
        )

    assert field_name is None or isinstance(field_name, str), \
        str(type(field_name))
    return {
        'name': name,
        'field_name': 'None' if field_name is None else field_name,
        'pandas_type': logical_type,
        'numpy_type': string_dtype,
        'metadata': extra_metadata,
    }


def construct_metadata(columns_to_convert, df, column_names, index_levels,
                       index_descriptors, preserve_index, types):
    """Returns a dictionary containing enough metadata to reconstruct a pandas
    DataFrame as an Arrow Table, including index columns.

    Parameters
    ----------
    columns_to_convert : list[pd.Series]
    df : pandas.DataFrame
    index_levels : List[pd.Index]
    index_descriptors : List[Dict]
    preserve_index : bool
    types : List[pyarrow.DataType]

    Returns
    -------
    dict
    """
    num_serialized_index_levels = len([descr for descr in index_descriptors
                                       if not isinstance(descr, dict)])
    # Use ntypes instead of Python shorthand notation [:-len(x)] as [:-0]
    # behaves differently to what we want.
    ntypes = len(types)
    df_types = types[:ntypes - num_serialized_index_levels]
    index_types = types[ntypes - num_serialized_index_levels:]

    column_metadata = []
    for col, sanitized_name, arrow_type in zip(columns_to_convert,
                                               column_names, df_types):
        metadata = get_column_metadata(col, name=sanitized_name,
                                       arrow_type=arrow_type,
                                       field_name=sanitized_name)
        column_metadata.append(metadata)

    index_column_metadata = []
    if preserve_index is not False:
        non_str_index_names = []
        for level, arrow_type, descriptor in zip(index_levels, index_types,
                                                 index_descriptors):
            if isinstance(descriptor, dict):
                # The index is represented in a non-serialized fashion,
                # e.g. RangeIndex
                continue

            if level.name is not None and not isinstance(level.name, str):
                non_str_index_names.append(level.name)

            metadata = get_column_metadata(
                level,
                name=_column_name_to_strings(level.name),
                arrow_type=arrow_type,
                field_name=descriptor,
            )
            index_column_metadata.append(metadata)

        if len(non_str_index_names) > 0:
            warnings.warn(
                f"The DataFrame has non-str index name `{non_str_index_names}`"
                " which will be converted to string"
                " and not roundtrip correctly.",
                UserWarning, stacklevel=4)

        column_indexes = []

        levels = getattr(df.columns, 'levels', [df.columns])
        names = getattr(df.columns, 'names', [df.columns.name])
        for level, name in zip(levels, names):
            metadata = _get_simple_index_descriptor(level, name)
            column_indexes.append(metadata)
    else:
        index_descriptors = index_column_metadata = column_indexes = []

    return {
        b'pandas': json.dumps({
            'index_columns': index_descriptors,
            'column_indexes': column_indexes,
            'columns': column_metadata + index_column_metadata,
            'creator': {
                'library': 'pyarrow',
                'version': pa.__version__
            },
            'pandas_version': _pandas_api.version
        }).encode('utf8')
    }


def _get_simple_index_descriptor(level, name):
    string_dtype, extra_metadata = get_extension_dtype_info(level)
    pandas_type = get_logical_type_from_numpy(level)
    if 'mixed' in pandas_type:
        warnings.warn(
            "The DataFrame has column names of mixed type. They will be "
            "converted to strings and not roundtrip correctly.",
            UserWarning, stacklevel=4)
    if pandas_type == 'unicode':
        assert not extra_metadata
        extra_metadata = {'encoding': 'UTF-8'}
    return {
        'name': name,
        'field_name': name,
        'pandas_type': pandas_type,
        'numpy_type': string_dtype,
        'metadata': extra_metadata,
    }


def _column_name_to_strings(name):
    """Convert a column name (or level) to either a string or a recursive
    collection of strings.

    Parameters
    ----------
    name : str or tuple

    Returns
    -------
    value : str or tuple

    Examples
    --------
    >>> name = 'foo'
    >>> _column_name_to_strings(name)
    'foo'
    >>> name = ('foo', 'bar')
    >>> _column_name_to_strings(name)
    "('foo', 'bar')"
    >>> import pandas as pd
    >>> name = (1, pd.Timestamp('2017-02-01 00:00:00'))
    >>> _column_name_to_strings(name)
    "('1', '2017-02-01 00:00:00')"
    """
    if isinstance(name, str):
        return name
    elif isinstance(name, bytes):
        # XXX: should we assume that bytes in Python 3 are UTF-8?
        return name.decode('utf8')
    elif isinstance(name, tuple):
        return str(tuple(map(_column_name_to_strings, name)))
    elif isinstance(name, Sequence):
        raise TypeError("Unsupported type for MultiIndex level")
    elif name is None:
        return None
    return str(name)


def _index_level_name(index, i, column_names):
    """Return the name of an index level or a default name if `index.name` is
    None or is already a column name.

    Parameters
    ----------
    index : pandas.Index
    i : int

    Returns
    -------
    name : str
    """
    if index.name is not None and index.name not in column_names:
        return _column_name_to_strings(index.name)
    else:
        return '__index_level_{:d}__'.format(i)


def _get_columns_to_convert(df, schema, preserve_index, columns):
    columns = _resolve_columns_of_interest(df, schema, columns)

    if not df.columns.is_unique:
        raise ValueError(
            'Duplicate column names found: {}'.format(list(df.columns))
        )

    if schema is not None:
        return _get_columns_to_convert_given_schema(df, schema, preserve_index)

    column_names = []

    index_levels = (
        _get_index_level_values(df.index) if preserve_index is not False
        else []
    )

    columns_to_convert = []
    convert_fields = []

    for name in columns:
        col = df[name]
        name = _column_name_to_strings(name)

        if _pandas_api.is_sparse(col):
            raise TypeError(
                "Sparse pandas data (column {}) not supported.".format(name))

        columns_to_convert.append(col)
        convert_fields.append(None)
        column_names.append(name)

    index_descriptors = []
    index_column_names = []
    for i, index_level in enumerate(index_levels):
        name = _index_level_name(index_level, i, column_names)
        if (isinstance(index_level, _pandas_api.pd.RangeIndex) and
                preserve_index is None):
            descr = _get_range_index_descriptor(index_level)
        else:
            columns_to_convert.append(index_level)
            convert_fields.append(None)
            descr = name
            index_column_names.append(name)
        index_descriptors.append(descr)

    all_names = column_names + index_column_names

    # all_names : all of the columns in the resulting table including the data
    # columns and serialized index columns
    # column_names : the names of the data columns
    # index_column_names : the names of the serialized index columns
    # index_descriptors : descriptions of each index to be used for
    # reconstruction
    # index_levels : the extracted index level values
    # columns_to_convert : assembled raw data (both data columns and indexes)
    # to be converted to Arrow format
    # columns_fields : specified column to use for coercion / casting
    # during serialization, if a Schema was provided
    return (all_names, column_names, index_column_names, index_descriptors,
            index_levels, columns_to_convert, convert_fields)


def _get_columns_to_convert_given_schema(df, schema, preserve_index):
    """
    Specialized version of _get_columns_to_convert in case a Schema is
    specified.
    In that case, the Schema is used as the single point of truth for the
    table structure (types, which columns are included, order of columns, ...).
    """
    column_names = []
    columns_to_convert = []
    convert_fields = []
    index_descriptors = []
    index_column_names = []
    index_levels = []

    for name in schema.names:
        try:
            col = df[name]
            is_index = False
        except KeyError:
            try:
                col = _get_index_level(df, name)
            except (KeyError, IndexError):
                # name not found as index level
                raise KeyError(
                    "name '{}' present in the specified schema is not found "
                    "in the columns or index".format(name))
            if preserve_index is False:
                raise ValueError(
                    "name '{}' present in the specified schema corresponds "
                    "to the index, but 'preserve_index=False' was "
                    "specified".format(name))
            elif (preserve_index is None and
                    isinstance(col, _pandas_api.pd.RangeIndex)):
                raise ValueError(
                    "name '{}' is present in the schema, but it is a "
                    "RangeIndex which will not be converted as a column "
                    "in the Table, but saved as metadata-only not in "
                    "columns. Specify 'preserve_index=True' to force it "
                    "being added as a column, or remove it from the "
                    "specified schema".format(name))
            is_index = True

        name = _column_name_to_strings(name)

        if _pandas_api.is_sparse(col):
            raise TypeError(
                "Sparse pandas data (column {}) not supported.".format(name))

        field = schema.field(name)
        columns_to_convert.append(col)
        convert_fields.append(field)
        column_names.append(name)

        if is_index:
            index_column_names.append(name)
            index_descriptors.append(name)
            index_levels.append(col)

    all_names = column_names + index_column_names

    return (all_names, column_names, index_column_names, index_descriptors,
            index_levels, columns_to_convert, convert_fields)


def _get_index_level(df, name):
    """
    Get the index level of a DataFrame given 'name' (column name in an arrow
    Schema).
    """
    key = name
    if name not in df.index.names and _is_generated_index_name(name):
        # we know we have an autogenerated name => extract number and get
        # the index level positionally
        key = int(name[len("__index_level_"):-2])
    return df.index.get_level_values(key)


def _level_name(name):
    # preserve type when default serializable, otherwise str it
    try:
        json.dumps(name)
        return name
    except TypeError:
        return str(name)


def _get_range_index_descriptor(level):
    # public start/stop/step attributes added in pandas 0.25.0
    return {
        'kind': 'range',
        'name': _level_name(level.name),
        'start': _pandas_api.get_rangeindex_attribute(level, 'start'),
        'stop': _pandas_api.get_rangeindex_attribute(level, 'stop'),
        'step': _pandas_api.get_rangeindex_attribute(level, 'step')
    }


def _get_index_level_values(index):
    n = len(getattr(index, 'levels', [index]))
    return [index.get_level_values(i) for i in range(n)]


def _resolve_columns_of_interest(df, schema, columns):
    if schema is not None and columns is not None:
        raise ValueError('Schema and columns arguments are mutually '
                         'exclusive, pass only one of them')
    elif schema is not None:
        columns = schema.names
    elif columns is not None:
        columns = [c for c in columns if c in df.columns]
    else:
        columns = df.columns

    return columns


def dataframe_to_types(df, preserve_index, columns=None):
    (all_names,
     column_names,
     _,
     index_descriptors,
     index_columns,
     columns_to_convert,
     _) = _get_columns_to_convert(df, None, preserve_index, columns)

    types = []
    # If pandas knows type, skip conversion
    for c in columns_to_convert:
        values = c.values
        if _pandas_api.is_categorical(values):
            type_ = pa.array(c, from_pandas=True).type
        elif _pandas_api.is_extension_array_dtype(values):
            empty = c.head(0) if isinstance(
                c, _pandas_api.pd.Series) else c[:0]
            type_ = pa.array(empty, from_pandas=True).type
        else:
            values, type_ = get_datetimetz_type(values, c.dtype, None)
            type_ = pa.lib._ndarray_to_arrow_type(values, type_)
            if type_ is None:
                type_ = pa.array(c, from_pandas=True).type
        types.append(type_)

    metadata = construct_metadata(
        columns_to_convert, df, column_names, index_columns,
        index_descriptors, preserve_index, types
    )

    return all_names, types, metadata


def dataframe_to_arrays(df, schema, preserve_index, nthreads=1, columns=None,
                        safe=True):
    (all_names,
     column_names,
     index_column_names,
     index_descriptors,
     index_columns,
     columns_to_convert,
     convert_fields) = _get_columns_to_convert(df, schema, preserve_index,
                                               columns)

    # NOTE(wesm): If nthreads=None, then we use a heuristic to decide whether
    # using a thread pool is worth it. Currently the heuristic is whether the
    # nrows > 100 * ncols and ncols > 1.
    if nthreads is None:
        nrows, ncols = len(df), len(df.columns)
        if nrows > ncols * 100 and ncols > 1:
            nthreads = pa.cpu_count()
        else:
            nthreads = 1

    def convert_column(col, field):
        if field is None:
            field_nullable = True
            type_ = None
        else:
            field_nullable = field.nullable
            type_ = field.type

        try:
            result = pa.array(col, type=type_, from_pandas=True, safe=safe)
        except (pa.ArrowInvalid,
                pa.ArrowNotImplementedError,
                pa.ArrowTypeError) as e:
            e.args += ("Conversion failed for column {!s} with type {!s}"
                       .format(col.name, col.dtype),)
            raise e
        if not field_nullable and result.null_count > 0:
            raise ValueError("Field {} was non-nullable but pandas column "
                             "had {} null values".format(str(field),
                                                         result.null_count))
        return result

    def _can_definitely_zero_copy(arr):
        return (isinstance(arr, np.ndarray) and
                arr.flags.contiguous and
                issubclass(arr.dtype.type, np.integer))

    if nthreads == 1:
        arrays = [convert_column(c, f)
                  for c, f in zip(columns_to_convert, convert_fields)]
    else:
        arrays = []
        with futures.ThreadPoolExecutor(nthreads) as executor:
            for c, f in zip(columns_to_convert, convert_fields):
                if _can_definitely_zero_copy(c.values):
                    arrays.append(convert_column(c, f))
                else:
                    arrays.append(executor.submit(convert_column, c, f))

        for i, maybe_fut in enumerate(arrays):
            if isinstance(maybe_fut, futures.Future):
                arrays[i] = maybe_fut.result()

    types = [x.type for x in arrays]

    if schema is None:
        fields = []
        for name, type_ in zip(all_names, types):
            name = name if name is not None else 'None'
            fields.append(pa.field(name, type_))
        schema = pa.schema(fields)

    pandas_metadata = construct_metadata(
        columns_to_convert, df, column_names, index_columns,
        index_descriptors, preserve_index, types
    )
    metadata = deepcopy(schema.metadata) if schema.metadata else dict()
    metadata.update(pandas_metadata)
    schema = schema.with_metadata(metadata)

    # If dataframe is empty but with RangeIndex ->
    # remember the length of the indexes
    n_rows = None
    if len(arrays) == 0:
        try:
            kind = index_descriptors[0]["kind"]
            if kind == "range":
                start = index_descriptors[0]["start"]
                stop = index_descriptors[0]["stop"]
                step = index_descriptors[0]["step"]
                n_rows = len(range(start, stop, step))
        except IndexError:
            pass

    return arrays, schema, n_rows


def get_datetimetz_type(values, dtype, type_):
    if values.dtype.type != np.datetime64:
        return values, type_

    if _pandas_api.is_datetimetz(dtype) and type_ is None:
        # If no user type passed, construct a tz-aware timestamp type
        tz = dtype.tz
        unit = dtype.unit
        type_ = pa.timestamp(unit, tz)
    elif type_ is None:
        # Trust the NumPy dtype
        type_ = pa.from_numpy_dtype(values.dtype)

    return values, type_

# ----------------------------------------------------------------------
# Converting pandas.DataFrame to a dict containing only NumPy arrays or other
# objects friendly to pyarrow.serialize


def dataframe_to_serialized_dict(frame):
    block_manager = frame._data

    blocks = []
    axes = [ax for ax in block_manager.axes]

    for block in block_manager.blocks:
        values = block.values
        block_data = {}

        if _pandas_api.is_datetimetz(values.dtype):
            block_data['timezone'] = pa.lib.tzinfo_to_string(values.tz)
            if hasattr(values, 'values'):
                values = values.values
        elif _pandas_api.is_categorical(values):
            block_data.update(dictionary=values.categories,
                              ordered=values.ordered)
            values = values.codes
        block_data.update(
            placement=block.mgr_locs.as_array,
            block=values
        )

        # If we are dealing with an object array, pickle it instead.
        if values.dtype == np.dtype(object):
            block_data['object'] = None
            block_data['block'] = builtin_pickle.dumps(
                values, protocol=builtin_pickle.HIGHEST_PROTOCOL)

        blocks.append(block_data)

    return {
        'blocks': blocks,
        'axes': axes
    }


def serialized_dict_to_dataframe(data):
    import pandas.core.internals as _int
    reconstructed_blocks = [_reconstruct_block(block)
                            for block in data['blocks']]

    block_mgr = _int.BlockManager(reconstructed_blocks, data['axes'])
    return _pandas_api.data_frame(block_mgr)


def _reconstruct_block(item, columns=None, extension_columns=None):
    """
    Construct a pandas Block from the `item` dictionary coming from pyarrow's
    serialization or returned by arrow::python::ConvertTableToPandas.

    This function takes care of converting dictionary types to pandas
    categorical, Timestamp-with-timezones to the proper pandas Block, and
    conversion to pandas ExtensionBlock

    Parameters
    ----------
    item : dict
        For basic types, this is a dictionary in the form of
        {'block': np.ndarray of values, 'placement': pandas block placement}.
        Additional keys are present for other types (dictionary, timezone,
        object).
    columns :
        Column names of the table being constructed, used for extension types
    extension_columns : dict
        Dictionary of {column_name: pandas_dtype} that includes all columns
        and corresponding dtypes that will be converted to a pandas
        ExtensionBlock.

    Returns
    -------
    pandas Block

    """
    import pandas.core.internals as _int

    block_arr = item.get('block', None)
    placement = item['placement']
    if 'dictionary' in item:
        cat = _pandas_api.categorical_type.from_codes(
            block_arr, categories=item['dictionary'],
            ordered=item['ordered'])
        block = _int.make_block(cat, placement=placement)
    elif 'timezone' in item:
        dtype = make_datetimetz(item['timezone'])
        block = _int.make_block(block_arr, placement=placement,
                                klass=_int.DatetimeTZBlock,
                                dtype=dtype)
    elif 'object' in item:
        block = _int.make_block(builtin_pickle.loads(block_arr),
                                placement=placement)
    elif 'py_array' in item:
        # create ExtensionBlock
        arr = item['py_array']
        assert len(placement) == 1
        name = columns[placement[0]]
        pandas_dtype = extension_columns[name]
        if not hasattr(pandas_dtype, '__from_arrow__'):
            raise ValueError("This column does not support to be converted "
                             "to a pandas ExtensionArray")
        pd_ext_arr = pandas_dtype.__from_arrow__(arr)
        block = _int.make_block(pd_ext_arr, placement=placement)
    else:
        block = _int.make_block(block_arr, placement=placement)

    return block


def make_datetimetz(tz):
    tz = pa.lib.string_to_tzinfo(tz)
    return _pandas_api.datetimetz_type('ns', tz=tz)


# ----------------------------------------------------------------------
# Converting pyarrow.Table efficiently to pandas.DataFrame


def table_to_blockmanager(options, table, categories=None,
                          ignore_metadata=False, types_mapper=None):
    from pandas.core.internals import BlockManager

    all_columns = []
    column_indexes = []
    pandas_metadata = table.schema.pandas_metadata

    if not ignore_metadata and pandas_metadata is not None:
        all_columns = pandas_metadata['columns']
        column_indexes = pandas_metadata.get('column_indexes', [])
        index_descriptors = pandas_metadata['index_columns']
        table = _add_any_metadata(table, pandas_metadata)
        table, index = _reconstruct_index(table, index_descriptors,
                                          all_columns)
        ext_columns_dtypes = _get_extension_dtypes(
            table, all_columns, types_mapper)
    else:
        index = _pandas_api.pd.RangeIndex(table.num_rows)
        ext_columns_dtypes = _get_extension_dtypes(table, [], types_mapper)

    _check_data_column_metadata_consistency(all_columns)
    columns = _deserialize_column_index(table, all_columns, column_indexes)
    blocks = _table_to_blocks(options, table, categories, ext_columns_dtypes)

    axes = [columns, index]
    return BlockManager(blocks, axes)


# Set of the string repr of all numpy dtypes that can be stored in a pandas
# dataframe (complex not included since not supported by Arrow)
_pandas_supported_numpy_types = {
    str(np.dtype(typ))
    for typ in (np.sctypes['int'] + np.sctypes['uint'] + np.sctypes['float'] +
                ['object', 'bool'])
}


def _get_extension_dtypes(table, columns_metadata, types_mapper=None):
    """
    Based on the stored column pandas metadata and the extension types
    in the arrow schema, infer which columns should be converted to a
    pandas extension dtype.

    The 'numpy_type' field in the column metadata stores the string
    representation of the original pandas dtype (and, despite its name,
    not the 'pandas_type' field).
    Based on this string representation, a pandas/numpy dtype is constructed
    and then we can check if this dtype supports conversion from arrow.

    """
    ext_columns = {}

    # older pandas version that does not yet support extension dtypes
    if _pandas_api.extension_dtype is None:
        return ext_columns

    # infer the extension columns from the pandas metadata
    for col_meta in columns_metadata:
        try:
            name = col_meta['field_name']
        except KeyError:
            name = col_meta['name']
        dtype = col_meta['numpy_type']

        if dtype not in _pandas_supported_numpy_types:
            # pandas_dtype is expensive, so avoid doing this for types
            # that are certainly numpy dtypes
            pandas_dtype = _pandas_api.pandas_dtype(dtype)
            if isinstance(pandas_dtype, _pandas_api.extension_dtype):
                if hasattr(pandas_dtype, "__from_arrow__"):
                    ext_columns[name] = pandas_dtype

    # infer from extension type in the schema
    for field in table.schema:
        typ = field.type
        if isinstance(typ, pa.BaseExtensionType):
            try:
                pandas_dtype = typ.to_pandas_dtype()
            except NotImplementedError:
                pass
            else:
                ext_columns[field.name] = pandas_dtype

    # use the specified mapping of built-in arrow types to pandas dtypes
    if types_mapper:
        for field in table.schema:
            typ = field.type
            pandas_dtype = types_mapper(typ)
            if pandas_dtype is not None:
                ext_columns[field.name] = pandas_dtype

    return ext_columns


def _check_data_column_metadata_consistency(all_columns):
    # It can never be the case in a released version of pyarrow that
    # c['name'] is None *and* 'field_name' is not a key in the column metadata,
    # because the change to allow c['name'] to be None and the change to add
    # 'field_name' are in the same release (0.8.0)
    assert all(
        (c['name'] is None and 'field_name' in c) or c['name'] is not None
        for c in all_columns
    )


def _deserialize_column_index(block_table, all_columns, column_indexes):
    column_strings = [frombytes(x) if isinstance(x, bytes) else x
                      for x in block_table.column_names]
    if all_columns:
        columns_name_dict = {
            c.get('field_name', _column_name_to_strings(c['name'])): c['name']
            for c in all_columns
        }
        columns_values = [
            columns_name_dict.get(name, name) for name in column_strings
        ]
    else:
        columns_values = column_strings

    # If we're passed multiple column indexes then evaluate with
    # ast.literal_eval, since the column index values show up as a list of
    # tuples
    to_pair = ast.literal_eval if len(column_indexes) > 1 else lambda x: (x,)

    # Create the column index

    # Construct the base index
    if not columns_values:
        columns = _pandas_api.pd.Index(columns_values)
    else:
        columns = _pandas_api.pd.MultiIndex.from_tuples(
            list(map(to_pair, columns_values)),
            names=[col_index['name'] for col_index in column_indexes] or None,
        )

    # if we're reconstructing the index
    if len(column_indexes) > 0:
        columns = _reconstruct_columns_from_metadata(columns, column_indexes)

    # ARROW-1751: flatten a single level column MultiIndex for pandas 0.21.0
    columns = _flatten_single_level_multiindex(columns)

    return columns


def _reconstruct_index(table, index_descriptors, all_columns):
    # 0. 'field_name' is the name of the column in the arrow Table
    # 1. 'name' is the user-facing name of the column, that is, it came from
    #    pandas
    # 2. 'field_name' and 'name' differ for index columns
    # 3. We fall back on c['name'] for backwards compatibility
    field_name_to_metadata = {
        c.get('field_name', c['name']): c
        for c in all_columns
    }

    # Build up a list of index columns and names while removing those columns
    # from the original table
    index_arrays = []
    index_names = []
    result_table = table
    for descr in index_descriptors:
        if isinstance(descr, str):
            result_table, index_level, index_name = _extract_index_level(
                table, result_table, descr, field_name_to_metadata)
            if index_level is None:
                # ARROW-1883: the serialized index column was not found
                continue
        elif descr['kind'] == 'range':
            index_name = descr['name']
            index_level = _pandas_api.pd.RangeIndex(descr['start'],
                                                    descr['stop'],
                                                    step=descr['step'],
                                                    name=index_name)
            if len(index_level) != len(table):
                # Possibly the result of munged metadata
                continue
        else:
            raise ValueError("Unrecognized index kind: {}"
                             .format(descr['kind']))
        index_arrays.append(index_level)
        index_names.append(index_name)

    pd = _pandas_api.pd

    # Reconstruct the row index
    if len(index_arrays) > 1:
        index = pd.MultiIndex.from_arrays(index_arrays, names=index_names)
    elif len(index_arrays) == 1:
        index = index_arrays[0]
        if not isinstance(index, pd.Index):
            # Box anything that wasn't boxed above
            index = pd.Index(index, name=index_names[0])
    else:
        index = pd.RangeIndex(table.num_rows)

    return result_table, index


def _extract_index_level(table, result_table, field_name,
                         field_name_to_metadata):
    logical_name = field_name_to_metadata[field_name]['name']
    index_name = _backwards_compatible_index_name(field_name, logical_name)
    i = table.schema.get_field_index(field_name)

    if i == -1:
        # The serialized index column was removed by the user
        return result_table, None, None

    pd = _pandas_api.pd

    col = table.column(i)
    values = col.to_pandas().values

    if hasattr(values, 'flags') and not values.flags.writeable:
        # ARROW-1054: in pandas 0.19.2, factorize will reject
        # non-writeable arrays when calling MultiIndex.from_arrays
        values = values.copy()

    if isinstance(col.type, pa.lib.TimestampType) and col.type.tz is not None:
        index_level = make_tz_aware(pd.Series(values), col.type.tz)
    else:
        index_level = pd.Series(values, dtype=values.dtype)
    result_table = result_table.remove_column(
        result_table.schema.get_field_index(field_name)
    )
    return result_table, index_level, index_name


def _backwards_compatible_index_name(raw_name, logical_name):
    """Compute the name of an index column that is compatible with older
    versions of :mod:`pyarrow`.

    Parameters
    ----------
    raw_name : str
    logical_name : str

    Returns
    -------
    result : str

    Notes
    -----
    * Part of :func:`~pyarrow.pandas_compat.table_to_blockmanager`
    """
    # Part of table_to_blockmanager
    if raw_name == logical_name and _is_generated_index_name(raw_name):
        return None
    else:
        return logical_name


def _is_generated_index_name(name):
    pattern = r'^__index_level_\d+__$'
    return re.match(pattern, name) is not None


_pandas_logical_type_map = {
    'date': 'datetime64[D]',
    'datetime': 'datetime64[ns]',
    'datetimetz': 'datetime64[ns]',
    'unicode': np.unicode_,
    'bytes': np.bytes_,
    'string': np.str_,
    'integer': np.int64,
    'floating': np.float64,
    'empty': np.object_,
}


def _pandas_type_to_numpy_type(pandas_type):
    """Get the numpy dtype that corresponds to a pandas type.

    Parameters
    ----------
    pandas_type : str
        The result of a call to pandas.lib.infer_dtype.

    Returns
    -------
    dtype : np.dtype
        The dtype that corresponds to `pandas_type`.
    """
    try:
        return _pandas_logical_type_map[pandas_type]
    except KeyError:
        if 'mixed' in pandas_type:
            # catching 'mixed', 'mixed-integer' and 'mixed-integer-float'
            return np.object_
        return np.dtype(pandas_type)


def _get_multiindex_codes(mi):
    if isinstance(mi, _pandas_api.pd.MultiIndex):
        return mi.codes
    else:
        return None


def _reconstruct_columns_from_metadata(columns, column_indexes):
    """Construct a pandas MultiIndex from `columns` and column index metadata
    in `column_indexes`.

    Parameters
    ----------
    columns : List[pd.Index]
        The columns coming from a pyarrow.Table
    column_indexes : List[Dict[str, str]]
        The column index metadata deserialized from the JSON schema metadata
        in a :class:`~pyarrow.Table`.

    Returns
    -------
    result : MultiIndex
        The index reconstructed using `column_indexes` metadata with levels of
        the correct type.

    Notes
    -----
    * Part of :func:`~pyarrow.pandas_compat.table_to_blockmanager`
    """
    pd = _pandas_api.pd
    # Get levels and labels, and provide sane defaults if the index has a
    # single level to avoid if/else spaghetti.
    levels = getattr(columns, 'levels', None) or [columns]
    labels = _get_multiindex_codes(columns) or [
        pd.RangeIndex(len(level)) for level in levels
    ]

    # Convert each level to the dtype provided in the metadata
    levels_dtypes = [
        (level, col_index.get('pandas_type', str(level.dtype)),
         col_index.get('numpy_type', None))
        for level, col_index in zip_longest(
            levels, column_indexes, fillvalue={}
        )
    ]

    new_levels = []
    encoder = operator.methodcaller('encode', 'UTF-8')

    for level, pandas_dtype, numpy_dtype in levels_dtypes:
        dtype = _pandas_type_to_numpy_type(pandas_dtype)
        # Since our metadata is UTF-8 encoded, Python turns things that were
        # bytes into unicode strings when json.loads-ing them. We need to
        # convert them back to bytes to preserve metadata.
        if dtype == np.bytes_:
            level = level.map(encoder)
        # ARROW-13756: if index is timezone aware DataTimeIndex
        if pandas_dtype == "datetimetz":
            tz = pa.lib.string_to_tzinfo(
                column_indexes[0]['metadata']['timezone'])
            dt = level.astype(numpy_dtype)
            level = dt.tz_localize('utc').tz_convert(tz)
        elif level.dtype != dtype:
            level = level.astype(dtype)
        # ARROW-9096: if original DataFrame was upcast we keep that
        if level.dtype != numpy_dtype and pandas_dtype != "datetimetz":
            level = level.astype(numpy_dtype)

        new_levels.append(level)

    return pd.MultiIndex(new_levels, labels, names=columns.names)


def _table_to_blocks(options, block_table, categories, extension_columns):
    # Part of table_to_blockmanager

    # Convert an arrow table to Block from the internal pandas API
    columns = block_table.column_names
    result = pa.lib.table_to_blocks(options, block_table, categories,
                                    list(extension_columns.keys()))
    return [_reconstruct_block(item, columns, extension_columns)
            for item in result]


def _flatten_single_level_multiindex(index):
    pd = _pandas_api.pd
    if isinstance(index, pd.MultiIndex) and index.nlevels == 1:
        levels, = index.levels
        labels, = _get_multiindex_codes(index)
        # ARROW-9096: use levels.dtype to match cast with original DataFrame
        dtype = levels.dtype

        # Cheaply check that we do not somehow have duplicate column names
        if not index.is_unique:
            raise ValueError('Found non-unique column index')

        return pd.Index(
            [levels[_label] if _label != -1 else None for _label in labels],
            dtype=dtype,
            name=index.names[0]
        )
    return index


def _add_any_metadata(table, pandas_metadata):
    modified_columns = {}
    modified_fields = {}

    schema = table.schema

    index_columns = pandas_metadata['index_columns']
    # only take index columns into account if they are an actual table column
    index_columns = [idx_col for idx_col in index_columns
                     if isinstance(idx_col, str)]
    n_index_levels = len(index_columns)
    n_columns = len(pandas_metadata['columns']) - n_index_levels

    # Add time zones
    for i, col_meta in enumerate(pandas_metadata['columns']):

        raw_name = col_meta.get('field_name')
        if not raw_name:
            # deal with metadata written with arrow < 0.8 or fastparquet
            raw_name = col_meta['name']
            if i >= n_columns:
                # index columns
                raw_name = index_columns[i - n_columns]
            if raw_name is None:
                raw_name = 'None'

        idx = schema.get_field_index(raw_name)
        if idx != -1:
            if col_meta['pandas_type'] == 'datetimetz':
                col = table[idx]
                if not isinstance(col.type, pa.lib.TimestampType):
                    continue
                metadata = col_meta['metadata']
                if not metadata:
                    continue
                metadata_tz = metadata.get('timezone')
                if metadata_tz and metadata_tz != col.type.tz:
                    converted = col.to_pandas()
                    tz_aware_type = pa.timestamp('ns', tz=metadata_tz)
                    with_metadata = pa.Array.from_pandas(converted,
                                                         type=tz_aware_type)

                    modified_fields[idx] = pa.field(schema[idx].name,
                                                    tz_aware_type)
                    modified_columns[idx] = with_metadata

    if len(modified_columns) > 0:
        columns = []
        fields = []
        for i in range(len(table.schema)):
            if i in modified_columns:
                columns.append(modified_columns[i])
                fields.append(modified_fields[i])
            else:
                columns.append(table[i])
                fields.append(table.schema[i])
        return pa.Table.from_arrays(columns, schema=pa.schema(fields))
    else:
        return table


# ----------------------------------------------------------------------
# Helper functions used in lib


def make_tz_aware(series, tz):
    """
    Make a datetime64 Series timezone-aware for the given tz
    """
    tz = pa.lib.string_to_tzinfo(tz)
    series = (series.dt.tz_localize('utc')
                    .dt.tz_convert(tz))
    return series

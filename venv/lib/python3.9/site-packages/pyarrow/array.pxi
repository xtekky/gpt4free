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

import os
import warnings


cdef _sequence_to_array(object sequence, object mask, object size,
                        DataType type, CMemoryPool* pool, c_bool from_pandas):
    cdef:
        int64_t c_size
        PyConversionOptions options
        shared_ptr[CChunkedArray] chunked

    if type is not None:
        options.type = type.sp_type

    if size is not None:
        options.size = size

    options.from_pandas = from_pandas
    options.ignore_timezone = os.environ.get('PYARROW_IGNORE_TIMEZONE', False)

    with nogil:
        chunked = GetResultValue(
            ConvertPySequence(sequence, mask, options, pool)
        )

    if chunked.get().num_chunks() == 1:
        return pyarrow_wrap_array(chunked.get().chunk(0))
    else:
        return pyarrow_wrap_chunked_array(chunked)


cdef inline _is_array_like(obj):
    if isinstance(obj, np.ndarray):
        return True
    return pandas_api._have_pandas_internal() and pandas_api.is_array_like(obj)


def _ndarray_to_arrow_type(object values, DataType type):
    return pyarrow_wrap_data_type(_ndarray_to_type(values, type))


cdef shared_ptr[CDataType] _ndarray_to_type(object values,
                                            DataType type) except *:
    cdef shared_ptr[CDataType] c_type

    dtype = values.dtype

    if type is None and dtype != object:
        with nogil:
            check_status(NumPyDtypeToArrow(dtype, &c_type))

    if type is not None:
        c_type = type.sp_type

    return c_type


cdef _ndarray_to_array(object values, object mask, DataType type,
                       c_bool from_pandas, c_bool safe, CMemoryPool* pool):
    cdef:
        shared_ptr[CChunkedArray] chunked_out
        shared_ptr[CDataType] c_type = _ndarray_to_type(values, type)
        CCastOptions cast_options = CCastOptions(safe)

    with nogil:
        check_status(NdarrayToArrow(pool, values, mask, from_pandas,
                                    c_type, cast_options, &chunked_out))

    if chunked_out.get().num_chunks() > 1:
        return pyarrow_wrap_chunked_array(chunked_out)
    else:
        return pyarrow_wrap_array(chunked_out.get().chunk(0))


cdef _codes_to_indices(object codes, object mask, DataType type,
                       MemoryPool memory_pool):
    """
    Convert the codes of a pandas Categorical to indices for a pyarrow
    DictionaryArray, taking into account missing values + mask
    """
    if mask is None:
        mask = codes == -1
    else:
        mask = mask | (codes == -1)
    return array(codes, mask=mask, type=type, memory_pool=memory_pool)


def _handle_arrow_array_protocol(obj, type, mask, size):
    if mask is not None or size is not None:
        raise ValueError(
            "Cannot specify a mask or a size when passing an object that is "
            "converted with the __arrow_array__ protocol.")
    res = obj.__arrow_array__(type=type)
    if not isinstance(res, (Array, ChunkedArray)):
        raise TypeError("The object's __arrow_array__ method does not "
                        "return a pyarrow Array or ChunkedArray.")
    return res


def array(object obj, type=None, mask=None, size=None, from_pandas=None,
          bint safe=True, MemoryPool memory_pool=None):
    """
    Create pyarrow.Array instance from a Python object.

    Parameters
    ----------
    obj : sequence, iterable, ndarray or pandas.Series
        If both type and size are specified may be a single use iterable. If
        not strongly-typed, Arrow type will be inferred for resulting array.
    type : pyarrow.DataType
        Explicit type to attempt to coerce to, otherwise will be inferred from
        the data.
    mask : array[bool], optional
        Indicate which values are null (True) or not null (False).
    size : int64, optional
        Size of the elements. If the input is larger than size bail at this
        length. For iterators, if size is larger than the input iterator this
        will be treated as a "max size", but will involve an initial allocation
        of size followed by a resize to the actual size (so if you know the
        exact size specifying it correctly will give you better performance).
    from_pandas : bool, default None
        Use pandas's semantics for inferring nulls from values in
        ndarray-like data. If passed, the mask tasks precedence, but
        if a value is unmasked (not-null), but still null according to
        pandas semantics, then it is null. Defaults to False if not
        passed explicitly by user, or True if a pandas object is
        passed in.
    safe : bool, default True
        Check for overflows or other unsafe conversions.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the currently-set default
        memory pool.

    Returns
    -------
    array : pyarrow.Array or pyarrow.ChunkedArray
        A ChunkedArray instead of an Array is returned if:

        - the object data overflowed binary storage.
        - the object's ``__arrow_array__`` protocol method returned a chunked
          array.

    Notes
    -----
    Timezone will be preserved in the returned array for timezone-aware data,
    else no timezone will be returned for naive timestamps.
    Internally, UTC values are stored for timezone-aware data with the
    timezone set in the data type.

    Pandas's DateOffsets and dateutil.relativedelta.relativedelta are by
    default converted as MonthDayNanoIntervalArray. relativedelta leapdays
    are ignored as are all absolute fields on both objects. datetime.timedelta
    can also be converted to MonthDayNanoIntervalArray but this requires
    passing MonthDayNanoIntervalType explicitly.

    Converting to dictionary array will promote to a wider integer type for
    indices if the number of distinct values cannot be represented, even if
    the index type was explicitly set. This means that if there are more than
    127 values the returned dictionary array's index type will be at least
    pa.int16() even if pa.int8() was passed to the function. Note that an
    explicit index type will not be demoted even if it is wider than required.

    Examples
    --------
    >>> import pandas as pd
    >>> import pyarrow as pa
    >>> pa.array(pd.Series([1, 2]))
    <pyarrow.lib.Int64Array object at ...>
    [
      1,
      2
    ]

    >>> pa.array(["a", "b", "a"], type=pa.dictionary(pa.int8(), pa.string()))
    <pyarrow.lib.DictionaryArray object at ...>
    ...
    -- dictionary:
      [
        "a",
        "b"
      ]
    -- indices:
      [
        0,
        1,
        0
      ]

    >>> import numpy as np
    >>> pa.array(pd.Series([1, 2]), mask=np.array([0, 1], dtype=bool))
    <pyarrow.lib.Int64Array object at ...>
    [
      1,
      null
    ]

    >>> arr = pa.array(range(1024), type=pa.dictionary(pa.int8(), pa.int64()))
    >>> arr.type.index_type
    DataType(int16)
    """
    cdef:
        CMemoryPool* pool = maybe_unbox_memory_pool(memory_pool)
        bint is_pandas_object = False
        bint c_from_pandas

    type = ensure_type(type, allow_none=True)

    extension_type = None
    if type is not None and type.id == _Type_EXTENSION:
        extension_type = type
        type = type.storage_type

    if from_pandas is None:
        c_from_pandas = False
    else:
        c_from_pandas = from_pandas

    if hasattr(obj, '__arrow_array__'):
        return _handle_arrow_array_protocol(obj, type, mask, size)
    elif _is_array_like(obj):
        if mask is not None:
            if _is_array_like(mask):
                mask = get_values(mask, &is_pandas_object)
            else:
                raise TypeError("Mask must be a numpy array "
                                "when converting numpy arrays")

        values = get_values(obj, &is_pandas_object)
        if is_pandas_object and from_pandas is None:
            c_from_pandas = True

        if isinstance(values, np.ma.MaskedArray):
            if mask is not None:
                raise ValueError("Cannot pass a numpy masked array and "
                                 "specify a mask at the same time")
            else:
                # don't use shrunken masks
                mask = None if values.mask is np.ma.nomask else values.mask
                values = values.data

        if mask is not None:
            if mask.dtype != np.bool_:
                raise TypeError("Mask must be boolean dtype")
            if mask.ndim != 1:
                raise ValueError("Mask must be 1D array")
            if len(values) != len(mask):
                raise ValueError(
                    "Mask is a different length from sequence being converted")

        if hasattr(values, '__arrow_array__'):
            return _handle_arrow_array_protocol(values, type, mask, size)
        elif (pandas_api.is_categorical(values) and
              type is not None and type.id != Type_DICTIONARY):
            result = _ndarray_to_array(
                np.asarray(values), mask, type, c_from_pandas, safe, pool
            )
        elif pandas_api.is_categorical(values):
            if type is not None:
                index_type = type.index_type
                value_type = type.value_type
                if values.ordered != type.ordered:
                    raise ValueError(
                        "The 'ordered' flag of the passed categorical values "
                        "does not match the 'ordered' of the specified type. ")
            else:
                index_type = None
                value_type = None

            indices = _codes_to_indices(
                values.codes, mask, index_type, memory_pool)
            try:
                dictionary = array(
                    values.categories.values, type=value_type,
                    memory_pool=memory_pool)
            except TypeError:
                # TODO when removing the deprecation warning, this whole
                # try/except can be removed (to bubble the TypeError of
                # the first array(..) call)
                if value_type is not None:
                    warnings.warn(
                        "The dtype of the 'categories' of the passed "
                        "categorical values ({0}) does not match the "
                        "specified type ({1}). For now ignoring the specified "
                        "type, but in the future this mismatch will raise a "
                        "TypeError".format(
                            values.categories.dtype, value_type),
                        FutureWarning, stacklevel=2)
                    dictionary = array(
                        values.categories.values, memory_pool=memory_pool)
                else:
                    raise

            return DictionaryArray.from_arrays(
                indices, dictionary, ordered=values.ordered, safe=safe)
        else:
            if pandas_api.have_pandas:
                values, type = pandas_api.compat.get_datetimetz_type(
                    values, obj.dtype, type)
            result = _ndarray_to_array(values, mask, type, c_from_pandas, safe,
                                       pool)
    else:
        # ConvertPySequence does strict conversion if type is explicitly passed
        result = _sequence_to_array(obj, mask, size, type, pool, c_from_pandas)

    if extension_type is not None:
        result = ExtensionArray.from_storage(extension_type, result)
    return result


def asarray(values, type=None):
    """
    Convert to pyarrow.Array, inferring type if not provided.

    Parameters
    ----------
    values : array-like
        This can be a sequence, numpy.ndarray, pyarrow.Array or
        pyarrow.ChunkedArray. If a ChunkedArray is passed, the output will be
        a ChunkedArray, otherwise the output will be a Array.
    type : string or DataType
        Explicitly construct the array with this type. Attempt to cast if
        indicated type is different.

    Returns
    -------
    arr : Array or ChunkedArray
    """
    if isinstance(values, (Array, ChunkedArray)):
        if type is not None and not values.type.equals(type):
            values = values.cast(type)
        return values
    else:
        return array(values, type=type)


def nulls(size, type=None, MemoryPool memory_pool=None):
    """
    Create a strongly-typed Array instance with all elements null.

    Parameters
    ----------
    size : int
        Array length.
    type : pyarrow.DataType, default None
        Explicit type for the array. By default use NullType.
    memory_pool : MemoryPool, default None
        Arrow MemoryPool to use for allocations. Uses the default memory
        pool is not passed.

    Returns
    -------
    arr : Array

    Examples
    --------
    >>> import pyarrow as pa
    >>> pa.nulls(10)
    <pyarrow.lib.NullArray object at ...>
    10 nulls

    >>> pa.nulls(3, pa.uint32())
    <pyarrow.lib.UInt32Array object at ...>
    [
      null,
      null,
      null
    ]
    """
    cdef:
        CMemoryPool* pool = maybe_unbox_memory_pool(memory_pool)
        int64_t length = size
        shared_ptr[CDataType] ty
        shared_ptr[CArray] arr

    type = ensure_type(type, allow_none=True)
    if type is None:
        type = null()

    ty = pyarrow_unwrap_data_type(type)
    with nogil:
        arr = GetResultValue(MakeArrayOfNull(ty, length, pool))

    return pyarrow_wrap_array(arr)


def repeat(value, size, MemoryPool memory_pool=None):
    """
    Create an Array instance whose slots are the given scalar.

    Parameters
    ----------
    value : Scalar-like object
        Either a pyarrow.Scalar or any python object coercible to a Scalar.
    size : int
        Number of times to repeat the scalar in the output Array.
    memory_pool : MemoryPool, default None
        Arrow MemoryPool to use for allocations. Uses the default memory
        pool is not passed.

    Returns
    -------
    arr : Array

    Examples
    --------
    >>> import pyarrow as pa
    >>> pa.repeat(10, 3)
    <pyarrow.lib.Int64Array object at ...>
    [
      10,
      10,
      10
    ]

    >>> pa.repeat([1, 2], 2)
    <pyarrow.lib.ListArray object at ...>
    [
      [
        1,
        2
      ],
      [
        1,
        2
      ]
    ]

    >>> pa.repeat("string", 3)
    <pyarrow.lib.StringArray object at ...>
    [
      "string",
      "string",
      "string"
    ]

    >>> pa.repeat(pa.scalar({'a': 1, 'b': [1, 2]}), 2)
    <pyarrow.lib.StructArray object at ...>
    -- is_valid: all not null
    -- child 0 type: int64
      [
        1,
        1
      ]
    -- child 1 type: list<item: int64>
      [
        [
          1,
          2
        ],
        [
          1,
          2
        ]
      ]
    """
    cdef:
        CMemoryPool* pool = maybe_unbox_memory_pool(memory_pool)
        int64_t length = size
        shared_ptr[CArray] c_array
        shared_ptr[CScalar] c_scalar

    if not isinstance(value, Scalar):
        value = scalar(value, memory_pool=memory_pool)

    c_scalar = (<Scalar> value).unwrap()
    with nogil:
        c_array = GetResultValue(
            MakeArrayFromScalar(deref(c_scalar), length, pool)
        )

    return pyarrow_wrap_array(c_array)


def infer_type(values, mask=None, from_pandas=False):
    """
    Attempt to infer Arrow data type that can hold the passed Python
    sequence type in an Array object

    Parameters
    ----------
    values : array-like
        Sequence to infer type from.
    mask : ndarray (bool type), optional
        Optional exclusion mask where True marks null, False non-null.
    from_pandas : bool, default False
        Use pandas's NA/null sentinel values for type inference.

    Returns
    -------
    type : DataType
    """
    cdef:
        shared_ptr[CDataType] out
        c_bool use_pandas_sentinels = from_pandas

    if mask is not None and not isinstance(mask, np.ndarray):
        mask = np.array(mask, dtype=bool)

    out = GetResultValue(InferArrowType(values, mask, use_pandas_sentinels))
    return pyarrow_wrap_data_type(out)


def _normalize_slice(object arrow_obj, slice key):
    """
    Slices with step not equal to 1 (or None) will produce a copy
    rather than a zero-copy view
    """
    cdef:
        Py_ssize_t start, stop, step
        Py_ssize_t n = len(arrow_obj)

    start = key.start or 0
    if start < 0:
        start += n
        if start < 0:
            start = 0
    elif start >= n:
        start = n

    stop = key.stop if key.stop is not None else n
    if stop < 0:
        stop += n
        if stop < 0:
            stop = 0
    elif stop >= n:
        stop = n

    step = key.step or 1
    if step != 1:
        if step < 0:
            # Negative steps require some special handling
            if key.start is None:
                start = n - 1

            if key.stop is None:
                stop = -1

        indices = np.arange(start, stop, step)
        return arrow_obj.take(indices)
    else:
        length = max(stop - start, 0)
        return arrow_obj.slice(start, length)


cdef Py_ssize_t _normalize_index(Py_ssize_t index,
                                 Py_ssize_t length) except -1:
    if index < 0:
        index += length
        if index < 0:
            raise IndexError("index out of bounds")
    elif index >= length:
        raise IndexError("index out of bounds")
    return index


cdef wrap_datum(const CDatum& datum):
    if datum.kind() == DatumType_ARRAY:
        return pyarrow_wrap_array(MakeArray(datum.array()))
    elif datum.kind() == DatumType_CHUNKED_ARRAY:
        return pyarrow_wrap_chunked_array(datum.chunked_array())
    elif datum.kind() == DatumType_RECORD_BATCH:
        return pyarrow_wrap_batch(datum.record_batch())
    elif datum.kind() == DatumType_TABLE:
        return pyarrow_wrap_table(datum.table())
    elif datum.kind() == DatumType_SCALAR:
        return pyarrow_wrap_scalar(datum.scalar())
    else:
        raise ValueError("Unable to wrap Datum in a Python object")


cdef _append_array_buffers(const CArrayData* ad, list res):
    """
    Recursively append Buffer wrappers from *ad* and its children.
    """
    cdef size_t i, n
    assert ad != NULL
    n = ad.buffers.size()
    for i in range(n):
        buf = ad.buffers[i]
        res.append(pyarrow_wrap_buffer(buf)
                   if buf.get() != NULL else None)
    n = ad.child_data.size()
    for i in range(n):
        _append_array_buffers(ad.child_data[i].get(), res)


cdef _reduce_array_data(const CArrayData* ad):
    """
    Recursively dissect ArrayData to (pickable) tuples.
    """
    cdef size_t i, n
    assert ad != NULL

    n = ad.buffers.size()
    buffers = []
    for i in range(n):
        buf = ad.buffers[i]
        buffers.append(pyarrow_wrap_buffer(buf)
                       if buf.get() != NULL else None)

    children = []
    n = ad.child_data.size()
    for i in range(n):
        children.append(_reduce_array_data(ad.child_data[i].get()))

    if ad.dictionary.get() != NULL:
        dictionary = _reduce_array_data(ad.dictionary.get())
    else:
        dictionary = None

    return pyarrow_wrap_data_type(ad.type), ad.length, ad.null_count, \
        ad.offset, buffers, children, dictionary


cdef shared_ptr[CArrayData] _reconstruct_array_data(data):
    """
    Reconstruct CArrayData objects from the tuple structure generated
    by _reduce_array_data.
    """
    cdef:
        int64_t length, null_count, offset, i
        DataType dtype
        Buffer buf
        vector[shared_ptr[CBuffer]] c_buffers
        vector[shared_ptr[CArrayData]] c_children
        shared_ptr[CArrayData] c_dictionary

    dtype, length, null_count, offset, buffers, children, dictionary = data

    for i in range(len(buffers)):
        buf = buffers[i]
        if buf is None:
            c_buffers.push_back(shared_ptr[CBuffer]())
        else:
            c_buffers.push_back(buf.buffer)

    for i in range(len(children)):
        c_children.push_back(_reconstruct_array_data(children[i]))

    if dictionary is not None:
        c_dictionary = _reconstruct_array_data(dictionary)

    return CArrayData.MakeWithChildrenAndDictionary(
        dtype.sp_type,
        length,
        c_buffers,
        c_children,
        c_dictionary,
        null_count,
        offset)


def _restore_array(data):
    """
    Reconstruct an Array from pickled ArrayData.
    """
    cdef shared_ptr[CArrayData] ad = _reconstruct_array_data(data)
    return pyarrow_wrap_array(MakeArray(ad))


cdef class _PandasConvertible(_Weakrefable):

    def to_pandas(
            self,
            memory_pool=None,
            categories=None,
            bint strings_to_categorical=False,
            bint zero_copy_only=False,
            bint integer_object_nulls=False,
            bint date_as_object=True,
            bint timestamp_as_object=False,
            bint use_threads=True,
            bint deduplicate_objects=True,
            bint ignore_metadata=False,
            bint safe=True,
            bint split_blocks=False,
            bint self_destruct=False,
            types_mapper=None
    ):
        """
        Convert to a pandas-compatible NumPy array or DataFrame, as appropriate

        Parameters
        ----------
        memory_pool : MemoryPool, default None
            Arrow MemoryPool to use for allocations. Uses the default memory
            pool is not passed.
        categories : list, default empty
            List of fields that should be returned as pandas.Categorical. Only
            applies to table-like data structures.
        strings_to_categorical : bool, default False
            Encode string (UTF8) and binary types to pandas.Categorical.
        zero_copy_only : bool, default False
            Raise an ArrowException if this function call would require copying
            the underlying data.
        integer_object_nulls : bool, default False
            Cast integers with nulls to objects
        date_as_object : bool, default True
            Cast dates to objects. If False, convert to datetime64[ns] dtype.
        timestamp_as_object : bool, default False
            Cast non-nanosecond timestamps (np.datetime64) to objects. This is
            useful if you have timestamps that don't fit in the normal date
            range of nanosecond timestamps (1678 CE-2262 CE).
            If False, all timestamps are converted to datetime64[ns] dtype.
        use_threads : bool, default True
            Whether to parallelize the conversion using multiple threads.
        deduplicate_objects : bool, default False
            Do not create multiple copies Python objects when created, to save
            on memory use. Conversion will be slower.
        ignore_metadata : bool, default False
            If True, do not use the 'pandas' metadata to reconstruct the
            DataFrame index, if present
        safe : bool, default True
            For certain data types, a cast is needed in order to store the
            data in a pandas DataFrame or Series (e.g. timestamps are always
            stored as nanoseconds in pandas). This option controls whether it
            is a safe cast or not.
        split_blocks : bool, default False
            If True, generate one internal "block" for each column when
            creating a pandas.DataFrame from a RecordBatch or Table. While this
            can temporarily reduce memory note that various pandas operations
            can trigger "consolidation" which may balloon memory use.
        self_destruct : bool, default False
            EXPERIMENTAL: If True, attempt to deallocate the originating Arrow
            memory while converting the Arrow object to pandas. If you use the
            object after calling to_pandas with this option it will crash your
            program.

            Note that you may not see always memory usage improvements. For
            example, if multiple columns share an underlying allocation,
            memory can't be freed until all columns are converted.
        types_mapper : function, default None
            A function mapping a pyarrow DataType to a pandas ExtensionDtype.
            This can be used to override the default pandas type for conversion
            of built-in pyarrow types or in absence of pandas_metadata in the
            Table schema. The function receives a pyarrow DataType and is
            expected to return a pandas ExtensionDtype or ``None`` if the
            default conversion should be used for that type. If you have
            a dictionary mapping, you can pass ``dict.get`` as function.

        Returns
        -------
        pandas.Series or pandas.DataFrame depending on type of object

        Examples
        --------
        >>> import pyarrow as pa
        >>> import pandas as pd

        Convert a Table to pandas DataFrame:

        >>> table = pa.table([
        ...    pa.array([2, 4, 5, 100]),
        ...    pa.array(["Flamingo", "Horse", "Brittle stars", "Centipede"])
        ...    ], names=['n_legs', 'animals'])
        >>> table.to_pandas()
           n_legs        animals
        0       2       Flamingo
        1       4          Horse
        2       5  Brittle stars
        3     100      Centipede
        >>> isinstance(table.to_pandas(), pd.DataFrame)
        True

        Convert a RecordBatch to pandas DataFrame:

        >>> import pyarrow as pa
        >>> n_legs = pa.array([2, 4, 5, 100])
        >>> animals = pa.array(["Flamingo", "Horse", "Brittle stars", "Centipede"])
        >>> batch = pa.record_batch([n_legs, animals],
        ...                         names=["n_legs", "animals"])
        >>> batch
        pyarrow.RecordBatch
        n_legs: int64
        animals: string
        >>> batch.to_pandas()
           n_legs        animals
        0       2       Flamingo
        1       4          Horse
        2       5  Brittle stars
        3     100      Centipede
        >>> isinstance(batch.to_pandas(), pd.DataFrame)
        True

        Convert a Chunked Array to pandas Series:

        >>> import pyarrow as pa
        >>> n_legs = pa.chunked_array([[2, 2, 4], [4, 5, 100]])
        >>> n_legs.to_pandas()
        0      2
        1      2
        2      4
        3      4
        4      5
        5    100
        dtype: int64
        >>> isinstance(n_legs.to_pandas(), pd.Series)
        True
        """
        options = dict(
            pool=memory_pool,
            strings_to_categorical=strings_to_categorical,
            zero_copy_only=zero_copy_only,
            integer_object_nulls=integer_object_nulls,
            date_as_object=date_as_object,
            timestamp_as_object=timestamp_as_object,
            use_threads=use_threads,
            deduplicate_objects=deduplicate_objects,
            safe=safe,
            split_blocks=split_blocks,
            self_destruct=self_destruct
        )
        return self._to_pandas(options, categories=categories,
                               ignore_metadata=ignore_metadata,
                               types_mapper=types_mapper)


cdef PandasOptions _convert_pandas_options(dict options):
    cdef PandasOptions result
    result.pool = maybe_unbox_memory_pool(options['pool'])
    result.strings_to_categorical = options['strings_to_categorical']
    result.zero_copy_only = options['zero_copy_only']
    result.integer_object_nulls = options['integer_object_nulls']
    result.date_as_object = options['date_as_object']
    result.timestamp_as_object = options['timestamp_as_object']
    result.use_threads = options['use_threads']
    result.deduplicate_objects = options['deduplicate_objects']
    result.safe_cast = options['safe']
    result.split_blocks = options['split_blocks']
    result.self_destruct = options['self_destruct']
    result.ignore_timezone = os.environ.get('PYARROW_IGNORE_TIMEZONE', False)
    return result


cdef class Array(_PandasConvertible):
    """
    The base class for all Arrow arrays.
    """

    def __init__(self):
        raise TypeError("Do not call {}'s constructor directly, use one of "
                        "the `pyarrow.Array.from_*` functions instead."
                        .format(self.__class__.__name__))

    cdef void init(self, const shared_ptr[CArray]& sp_array) except *:
        self.sp_array = sp_array
        self.ap = sp_array.get()
        self.type = pyarrow_wrap_data_type(self.sp_array.get().type())

    def _debug_print(self):
        with nogil:
            check_status(DebugPrint(deref(self.ap), 0))

    def diff(self, Array other):
        """
        Compare contents of this array against another one.

        Return a string containing the result of diffing this array
        (on the left side) against the other array (on the right side).

        Parameters
        ----------
        other : Array
            The other array to compare this array with.

        Returns
        -------
        diff : str
            A human-readable printout of the differences.

        Examples
        --------
        >>> import pyarrow as pa
        >>> left = pa.array(["one", "two", "three"])
        >>> right = pa.array(["two", None, "two-and-a-half", "three"])
        >>> print(left.diff(right)) # doctest: +SKIP

        @@ -0, +0 @@
        -"one"
        @@ -2, +1 @@
        +null
        +"two-and-a-half"

        """
        cdef c_string result
        with nogil:
            result = self.ap.Diff(deref(other.ap))
        return frombytes(result, safe=True)

    def cast(self, object target_type=None, safe=None, options=None):
        """
        Cast array values to another data type

        See :func:`pyarrow.compute.cast` for usage.

        Parameters
        ----------
        target_type : DataType, default None
            Type to cast array to.
        safe : boolean, default True
            Whether to check for conversion errors such as overflow.
        options : CastOptions, default None
            Additional checks pass by CastOptions

        Returns
        -------
        cast : Array
        """
        return _pc().cast(self, target_type, safe=safe, options=options)

    def view(self, object target_type):
        """
        Return zero-copy "view" of array as another data type.

        The data types must have compatible columnar buffer layouts

        Parameters
        ----------
        target_type : DataType
            Type to construct view as.

        Returns
        -------
        view : Array
        """
        cdef DataType type = ensure_type(target_type)
        cdef shared_ptr[CArray] result
        with nogil:
            result = GetResultValue(self.ap.View(type.sp_type))
        return pyarrow_wrap_array(result)

    def sum(self, **kwargs):
        """
        Sum the values in a numerical array.

        See :func:`pyarrow.compute.sum` for full usage.

        Parameters
        ----------
        **kwargs : dict, optional
            Options to pass to :func:`pyarrow.compute.sum`.

        Returns
        -------
        sum : Scalar
            A scalar containing the sum value.
        """
        options = _pc().ScalarAggregateOptions(**kwargs)
        return _pc().call_function('sum', [self], options)

    def unique(self):
        """
        Compute distinct elements in array.

        Returns
        -------
        unique : Array
            An array of the same data type, with deduplicated elements.
        """
        return _pc().call_function('unique', [self])

    def dictionary_encode(self, null_encoding='mask'):
        """
        Compute dictionary-encoded representation of array.

        See :func:`pyarrow.compute.dictionary_encode` for full usage.

        Parameters
        ----------
        null_encoding : str, default "mask"
            How to handle null entries.

        Returns
        -------
        encoded : DictionaryArray
            A dictionary-encoded version of this array.
        """
        options = _pc().DictionaryEncodeOptions(null_encoding)
        return _pc().call_function('dictionary_encode', [self], options)

    def value_counts(self):
        """
        Compute counts of unique elements in array.

        Returns
        -------
        StructArray
            An array of  <input type "Values", int64 "Counts"> structs
        """
        return _pc().call_function('value_counts', [self])

    @staticmethod
    def from_pandas(obj, mask=None, type=None, bint safe=True,
                    MemoryPool memory_pool=None):
        """
        Convert pandas.Series to an Arrow Array.

        This method uses Pandas semantics about what values indicate
        nulls. See pyarrow.array for more general conversion from arrays or
        sequences to Arrow arrays.

        Parameters
        ----------
        obj : ndarray, pandas.Series, array-like
        mask : array (boolean), optional
            Indicate which values are null (True) or not null (False).
        type : pyarrow.DataType
            Explicit type to attempt to coerce to, otherwise will be inferred
            from the data.
        safe : bool, default True
            Check for overflows or other unsafe conversions.
        memory_pool : pyarrow.MemoryPool, optional
            If not passed, will allocate memory from the currently-set default
            memory pool.

        Notes
        -----
        Localized timestamps will currently be returned as UTC (pandas's native
        representation). Timezone-naive data will be implicitly interpreted as
        UTC.

        Returns
        -------
        array : pyarrow.Array or pyarrow.ChunkedArray
            ChunkedArray is returned if object data overflows binary buffer.
        """
        return array(obj, mask=mask, type=type, safe=safe, from_pandas=True,
                     memory_pool=memory_pool)

    def __reduce__(self):
        return _restore_array, \
            (_reduce_array_data(self.sp_array.get().data().get()),)

    @staticmethod
    def from_buffers(DataType type, length, buffers, null_count=-1, offset=0,
                     children=None):
        """
        Construct an Array from a sequence of buffers.

        The concrete type returned depends on the datatype.

        Parameters
        ----------
        type : DataType
            The value type of the array.
        length : int
            The number of values in the array.
        buffers : List[Buffer]
            The buffers backing this array.
        null_count : int, default -1
            The number of null entries in the array. Negative value means that
            the null count is not known.
        offset : int, default 0
            The array's logical offset (in values, not in bytes) from the
            start of each buffer.
        children : List[Array], default None
            Nested type children with length matching type.num_fields.

        Returns
        -------
        array : Array
        """
        cdef:
            Buffer buf
            Array child
            vector[shared_ptr[CBuffer]] c_buffers
            vector[shared_ptr[CArrayData]] c_child_data
            shared_ptr[CArrayData] array_data

        children = children or []

        if type.num_fields != len(children):
            raise ValueError("Type's expected number of children "
                             "({0}) did not match the passed number "
                             "({1}).".format(type.num_fields, len(children)))

        if type.num_buffers != len(buffers):
            raise ValueError("Type's expected number of buffers "
                             "({0}) did not match the passed number "
                             "({1}).".format(type.num_buffers, len(buffers)))

        for buf in buffers:
            # None will produce a null buffer pointer
            c_buffers.push_back(pyarrow_unwrap_buffer(buf))

        for child in children:
            c_child_data.push_back(child.ap.data())

        array_data = CArrayData.MakeWithChildren(type.sp_type, length,
                                                 c_buffers, c_child_data,
                                                 null_count, offset)
        cdef Array result = pyarrow_wrap_array(MakeArray(array_data))
        result.validate()
        return result

    @property
    def null_count(self):
        return self.sp_array.get().null_count()

    @property
    def nbytes(self):
        """
        Total number of bytes consumed by the elements of the array.

        In other words, the sum of bytes from all buffer
        ranges referenced.

        Unlike `get_total_buffer_size` this method will account for array
        offsets.

        If buffers are shared between arrays then the shared
        portion will be counted multiple times.

        The dictionary of dictionary arrays will always be counted in their
        entirety even if the array only references a portion of the dictionary.
        """
        cdef:
            CResult[int64_t] c_size_res

        c_size_res = ReferencedBufferSize(deref(self.ap))
        size = GetResultValue(c_size_res)
        return size

    def get_total_buffer_size(self):
        """
        The sum of bytes in each buffer referenced by the array.

        An array may only reference a portion of a buffer.
        This method will overestimate in this case and return the
        byte size of the entire buffer.

        If a buffer is referenced multiple times then it will
        only be counted once.
        """
        cdef:
            int64_t total_buffer_size

        total_buffer_size = TotalBufferSize(deref(self.ap))
        return total_buffer_size

    def __sizeof__(self):
        return super(Array, self).__sizeof__() + self.nbytes

    def __iter__(self):
        for i in range(len(self)):
            yield self.getitem(i)

    def __repr__(self):
        type_format = object.__repr__(self)
        return '{0}\n{1}'.format(type_format, str(self))

    def to_string(self, *, int indent=2, int top_level_indent=0, int window=10,
                  int container_window=2, c_bool skip_new_lines=False):
        """
        Render a "pretty-printed" string representation of the Array.

        Parameters
        ----------
        indent : int, default 2
            How much to indent the internal items in the string to
            the right, by default ``2``.
        top_level_indent : int, default 0
            How much to indent right the entire content of the array,
            by default ``0``.
        window : int
            How many primitive items to preview at the begin and end
            of the array when the array is bigger than the window.
            The other items will be ellipsed.
        container_window : int
            How many container items (such as a list in a list array)
            to preview at the begin and end of the array when the array
            is bigger than the window.
        skip_new_lines : bool
            If the array should be rendered as a single line of text
            or if each element should be on its own line.
        """
        cdef:
            c_string result
            PrettyPrintOptions options

        with nogil:
            options = PrettyPrintOptions(top_level_indent, window)
            options.skip_new_lines = skip_new_lines
            options.indent_size = indent
            check_status(
                PrettyPrint(
                    deref(self.ap),
                    options,
                    &result
                )
            )

        return frombytes(result, safe=True)

    def format(self, **kwargs):
        import warnings
        warnings.warn('Array.format is deprecated, use Array.to_string')
        return self.to_string(**kwargs)

    def __str__(self):
        return self.to_string()

    def __eq__(self, other):
        try:
            return self.equals(other)
        except TypeError:
            # This also handles comparing with None
            # as Array.equals(None) raises a TypeError.
            return NotImplemented

    def equals(Array self, Array other not None):
        return self.ap.Equals(deref(other.ap))

    def __len__(self):
        return self.length()

    cdef int64_t length(self):
        if self.sp_array.get():
            return self.sp_array.get().length()
        else:
            return 0

    def is_null(self, *, nan_is_null=False):
        """
        Return BooleanArray indicating the null values.

        Parameters
        ----------
        nan_is_null : bool (optional, default False)
            Whether floating-point NaN values should also be considered null.

        Returns
        -------
        array : boolean Array
        """
        options = _pc().NullOptions(nan_is_null=nan_is_null)
        return _pc().call_function('is_null', [self], options)

    def is_valid(self):
        """
        Return BooleanArray indicating the non-null values.
        """
        return _pc().is_valid(self)

    def fill_null(self, fill_value):
        """
        See :func:`pyarrow.compute.fill_null` for usage.

        Parameters
        ----------
        fill_value : any
            The replacement value for null entries.

        Returns
        -------
        result : Array
            A new array with nulls replaced by the given value.
        """
        return _pc().fill_null(self, fill_value)

    def __getitem__(self, key):
        """
        Slice or return value at given index

        Parameters
        ----------
        key : integer or slice
            Slices with step not equal to 1 (or None) will produce a copy
            rather than a zero-copy view

        Returns
        -------
        value : Scalar (index) or Array (slice)
        """
        if isinstance(key, slice):
            return _normalize_slice(self, key)

        return self.getitem(_normalize_index(key, self.length()))

    cdef getitem(self, int64_t i):
        return Scalar.wrap(GetResultValue(self.ap.GetScalar(i)))

    def slice(self, offset=0, length=None):
        """
        Compute zero-copy slice of this array.

        Parameters
        ----------
        offset : int, default 0
            Offset from start of array to slice.
        length : int, default None
            Length of slice (default is until end of Array starting from
            offset).

        Returns
        -------
        sliced : RecordBatch
        """
        cdef:
            shared_ptr[CArray] result

        if offset < 0:
            raise IndexError('Offset must be non-negative')

        offset = min(len(self), offset)
        if length is None:
            result = self.ap.Slice(offset)
        else:
            if length < 0:
                raise ValueError('Length must be non-negative')
            result = self.ap.Slice(offset, length)

        return pyarrow_wrap_array(result)

    def take(self, object indices):
        """
        Select values from an array.

        See :func:`pyarrow.compute.take` for full usage.

        Parameters
        ----------
        indices : Array or array-like
            The indices in the array whose values will be returned.

        Returns
        -------
        taken : Array
            An array with the same datatype, containing the taken values.
        """
        return _pc().take(self, indices)

    def drop_null(self):
        """
        Remove missing values from an array.
        """
        return _pc().drop_null(self)

    def filter(self, Array mask, *, null_selection_behavior='drop'):
        """
        Select values from an array.

        See :func:`pyarrow.compute.filter` for full usage.

        Parameters
        ----------
        mask : Array or array-like
            The boolean mask to filter the array with.
        null_selection_behavior : str, default "drop"
            How nulls in the mask should be handled.

        Returns
        -------
        filtered : Array
            An array of the same type, with only the elements selected by
            the boolean mask.
        """
        return _pc().filter(self, mask,
                            null_selection_behavior=null_selection_behavior)

    def index(self, value, start=None, end=None, *, memory_pool=None):
        """
        Find the first index of a value.

        See :func:`pyarrow.compute.index` for full usage.

        Parameters
        ----------
        value : Scalar or object
            The value to look for in the array.
        start : int, optional
            The start index where to look for `value`.
        end : int, optional
            The end index where to look for `value`.
        memory_pool : MemoryPool, optional
            A memory pool for potential memory allocations.

        Returns
        -------
        index : Int64Scalar
            The index of the value in the array (-1 if not found).
        """
        return _pc().index(self, value, start, end, memory_pool=memory_pool)

    def sort(self, order="ascending", **kwargs):
        """
        Sort the Array

        Parameters
        ----------
        order : str, default "ascending"
            Which order to sort values in.
            Accepted values are "ascending", "descending".
        **kwargs : dict, optional
            Additional sorting options.
            As allowed by :class:`SortOptions`

        Returns
        -------
        result : Array
        """
        indices = _pc().sort_indices(
            self,
            options=_pc().SortOptions(sort_keys=[("", order)], **kwargs)
        )
        return self.take(indices)

    def _to_pandas(self, options, types_mapper=None, **kwargs):
        return _array_like_to_pandas(self, options, types_mapper=types_mapper)

    def __array__(self, dtype=None):
        values = self.to_numpy(zero_copy_only=False)
        if dtype is None:
            return values
        return values.astype(dtype)

    def to_numpy(self, zero_copy_only=True, writable=False):
        """
        Return a NumPy view or copy of this array (experimental).

        By default, tries to return a view of this array. This is only
        supported for primitive arrays with the same memory layout as NumPy
        (i.e. integers, floating point, ..) and without any nulls.

        Parameters
        ----------
        zero_copy_only : bool, default True
            If True, an exception will be raised if the conversion to a numpy
            array would require copying the underlying data (e.g. in presence
            of nulls, or for non-primitive types).
        writable : bool, default False
            For numpy arrays created with zero copy (view on the Arrow data),
            the resulting array is not writable (Arrow data is immutable).
            By setting this to True, a copy of the array is made to ensure
            it is writable.

        Returns
        -------
        array : numpy.ndarray
        """
        cdef:
            PyObject* out
            PandasOptions c_options
            object values

        if zero_copy_only and writable:
            raise ValueError(
                "Cannot return a writable array if asking for zero-copy")

        # If there are nulls and the array is a DictionaryArray
        # decoding the dictionary will make sure nulls are correctly handled.
        # Decoding a dictionary does imply a copy by the way,
        # so it can't be done if the user requested a zero_copy.
        c_options.decode_dictionaries = not zero_copy_only
        c_options.zero_copy_only = zero_copy_only

        with nogil:
            check_status(ConvertArrayToPandas(c_options, self.sp_array,
                                              self, &out))

        # wrap_array_output uses pandas to convert to Categorical, here
        # always convert to numpy array without pandas dependency
        array = PyObject_to_object(out)

        if isinstance(array, dict):
            array = np.take(array['dictionary'], array['indices'])

        if writable and not array.flags.writeable:
            # if the conversion already needed to a copy, writeable is True
            array = array.copy()
        return array

    def to_pylist(self):
        """
        Convert to a list of native Python objects.

        Returns
        -------
        lst : list
        """
        return [x.as_py() for x in self]

    def tolist(self):
        """
        Alias of to_pylist for compatibility with NumPy.
        """
        return self.to_pylist()

    def validate(self, *, full=False):
        """
        Perform validation checks.  An exception is raised if validation fails.

        By default only cheap validation checks are run.  Pass `full=True`
        for thorough validation checks (potentially O(n)).

        Parameters
        ----------
        full : bool, default False
            If True, run expensive checks, otherwise cheap checks only.

        Raises
        ------
        ArrowInvalid
        """
        if full:
            with nogil:
                check_status(self.ap.ValidateFull())
        else:
            with nogil:
                check_status(self.ap.Validate())

    @property
    def offset(self):
        """
        A relative position into another array's data.

        The purpose is to enable zero-copy slicing. This value defaults to zero
        but must be applied on all operations with the physical storage
        buffers.
        """
        return self.sp_array.get().offset()

    def buffers(self):
        """
        Return a list of Buffer objects pointing to this array's physical
        storage.

        To correctly interpret these buffers, you need to also apply the offset
        multiplied with the size of the stored data type.
        """
        res = []
        _append_array_buffers(self.sp_array.get().data().get(), res)
        return res

    def _export_to_c(self, out_ptr, out_schema_ptr=0):
        """
        Export to a C ArrowArray struct, given its pointer.

        If a C ArrowSchema struct pointer is also given, the array type
        is exported to it at the same time.

        Parameters
        ----------
        out_ptr: int
            The raw pointer to a C ArrowArray struct.
        out_schema_ptr: int (optional)
            The raw pointer to a C ArrowSchema struct.

        Be careful: if you don't pass the ArrowArray struct to a consumer,
        array memory will leak.  This is a low-level function intended for
        expert users.
        """
        cdef:
            void* c_ptr = _as_c_pointer(out_ptr)
            void* c_schema_ptr = _as_c_pointer(out_schema_ptr,
                                               allow_null=True)
        with nogil:
            check_status(ExportArray(deref(self.sp_array),
                                     <ArrowArray*> c_ptr,
                                     <ArrowSchema*> c_schema_ptr))

    @staticmethod
    def _import_from_c(in_ptr, type):
        """
        Import Array from a C ArrowArray struct, given its pointer
        and the imported array type.

        Parameters
        ----------
        in_ptr: int
            The raw pointer to a C ArrowArray struct.
        type: DataType or int
            Either a DataType object, or the raw pointer to a C ArrowSchema
            struct.

        This is a low-level function intended for expert users.
        """
        cdef:
            void* c_ptr = _as_c_pointer(in_ptr)
            void* c_type_ptr
            shared_ptr[CArray] c_array

        c_type = pyarrow_unwrap_data_type(type)
        if c_type == nullptr:
            # Not a DataType object, perhaps a raw ArrowSchema pointer
            c_type_ptr = _as_c_pointer(type)
            with nogil:
                c_array = GetResultValue(ImportArray(
                    <ArrowArray*> c_ptr, <ArrowSchema*> c_type_ptr))
        else:
            with nogil:
                c_array = GetResultValue(ImportArray(<ArrowArray*> c_ptr,
                                                     c_type))
        return pyarrow_wrap_array(c_array)


cdef _array_like_to_pandas(obj, options, types_mapper):
    cdef:
        PyObject* out
        PandasOptions c_options = _convert_pandas_options(options)

    original_type = obj.type
    name = obj._name

    # ARROW-3789(wesm): Convert date/timestamp types to datetime64[ns]
    c_options.coerce_temporal_nanoseconds = True

    if isinstance(obj, Array):
        with nogil:
            check_status(ConvertArrayToPandas(c_options,
                                              (<Array> obj).sp_array,
                                              obj, &out))
    elif isinstance(obj, ChunkedArray):
        with nogil:
            check_status(libarrow_python.ConvertChunkedArrayToPandas(
                c_options,
                (<ChunkedArray> obj).sp_chunked_array,
                obj, &out))

    arr = wrap_array_output(out)

    if (isinstance(original_type, TimestampType) and
            options["timestamp_as_object"]):
        # ARROW-5359 - need to specify object dtype to avoid pandas to
        # coerce back to ns resolution
        dtype = "object"
    elif types_mapper:
        dtype = types_mapper(original_type)
    else:
        dtype = None

    result = pandas_api.series(arr, dtype=dtype, name=name)

    if (isinstance(original_type, TimestampType) and
            original_type.tz is not None and
            # can be object dtype for non-ns and timestamp_as_object=True
            result.dtype.kind == "M"):
        from pyarrow.pandas_compat import make_tz_aware
        result = make_tz_aware(result, original_type.tz)

    return result


cdef wrap_array_output(PyObject* output):
    cdef object obj = PyObject_to_object(output)

    if isinstance(obj, dict):
        return pandas_api.categorical_type(obj['indices'],
                                           categories=obj['dictionary'],
                                           ordered=obj['ordered'],
                                           fastpath=True)
    else:
        return obj


cdef class NullArray(Array):
    """
    Concrete class for Arrow arrays of null data type.
    """


cdef class BooleanArray(Array):
    """
    Concrete class for Arrow arrays of boolean data type.
    """
    @property
    def false_count(self):
        return (<CBooleanArray*> self.ap).false_count()

    @property
    def true_count(self):
        return (<CBooleanArray*> self.ap).true_count()


cdef class NumericArray(Array):
    """
    A base class for Arrow numeric arrays.
    """


cdef class IntegerArray(NumericArray):
    """
    A base class for Arrow integer arrays.
    """


cdef class FloatingPointArray(NumericArray):
    """
    A base class for Arrow floating-point arrays.
    """


cdef class Int8Array(IntegerArray):
    """
    Concrete class for Arrow arrays of int8 data type.
    """


cdef class UInt8Array(IntegerArray):
    """
    Concrete class for Arrow arrays of uint8 data type.
    """


cdef class Int16Array(IntegerArray):
    """
    Concrete class for Arrow arrays of int16 data type.
    """


cdef class UInt16Array(IntegerArray):
    """
    Concrete class for Arrow arrays of uint16 data type.
    """


cdef class Int32Array(IntegerArray):
    """
    Concrete class for Arrow arrays of int32 data type.
    """


cdef class UInt32Array(IntegerArray):
    """
    Concrete class for Arrow arrays of uint32 data type.
    """


cdef class Int64Array(IntegerArray):
    """
    Concrete class for Arrow arrays of int64 data type.
    """


cdef class UInt64Array(IntegerArray):
    """
    Concrete class for Arrow arrays of uint64 data type.
    """


cdef class Date32Array(NumericArray):
    """
    Concrete class for Arrow arrays of date32 data type.
    """


cdef class Date64Array(NumericArray):
    """
    Concrete class for Arrow arrays of date64 data type.
    """


cdef class TimestampArray(NumericArray):
    """
    Concrete class for Arrow arrays of timestamp data type.
    """


cdef class Time32Array(NumericArray):
    """
    Concrete class for Arrow arrays of time32 data type.
    """


cdef class Time64Array(NumericArray):
    """
    Concrete class for Arrow arrays of time64 data type.
    """


cdef class DurationArray(NumericArray):
    """
    Concrete class for Arrow arrays of duration data type.
    """


cdef class MonthDayNanoIntervalArray(Array):
    """
    Concrete class for Arrow arrays of interval[MonthDayNano] type.
    """

    def to_pylist(self):
        """
        Convert to a list of native Python objects.

        pyarrow.MonthDayNano is used as the native representation.

        Returns
        -------
        lst : list
        """
        cdef:
            CResult[PyObject*] maybe_py_list
            PyObject* py_list
            CMonthDayNanoIntervalArray* array
        array = <CMonthDayNanoIntervalArray*>self.sp_array.get()
        maybe_py_list = MonthDayNanoIntervalArrayToPyList(deref(array))
        py_list = GetResultValue(maybe_py_list)
        return PyObject_to_object(py_list)


cdef class HalfFloatArray(FloatingPointArray):
    """
    Concrete class for Arrow arrays of float16 data type.
    """


cdef class FloatArray(FloatingPointArray):
    """
    Concrete class for Arrow arrays of float32 data type.
    """


cdef class DoubleArray(FloatingPointArray):
    """
    Concrete class for Arrow arrays of float64 data type.
    """


cdef class FixedSizeBinaryArray(Array):
    """
    Concrete class for Arrow arrays of a fixed-size binary data type.
    """


cdef class Decimal128Array(FixedSizeBinaryArray):
    """
    Concrete class for Arrow arrays of decimal128 data type.
    """


cdef class Decimal256Array(FixedSizeBinaryArray):
    """
    Concrete class for Arrow arrays of decimal256 data type.
    """

cdef class BaseListArray(Array):

    def flatten(self):
        """
        Unnest this ListArray/LargeListArray by one level.

        The returned Array is logically a concatenation of all the sub-lists
        in this Array.

        Note that this method is different from ``self.values()`` in that
        it takes care of the slicing offset as well as null elements backed
        by non-empty sub-lists.

        Returns
        -------
        result : Array
        """
        return _pc().list_flatten(self)

    def value_parent_indices(self):
        """
        Return array of same length as list child values array where each
        output value is the index of the parent list array slot containing each
        child value.

        Examples
        --------
        >>> import pyarrow as pa
        >>> arr = pa.array([[1, 2, 3], [], None, [4]],
        ...                type=pa.list_(pa.int32()))
        >>> arr.value_parent_indices()
        <pyarrow.lib.Int64Array object at ...>
        [
          0,
          0,
          0,
          3
        ]
        """
        return _pc().list_parent_indices(self)

    def value_lengths(self):
        """
        Return integers array with values equal to the respective length of
        each list element. Null list values are null in the output.

        Examples
        --------
        >>> import pyarrow as pa
        >>> arr = pa.array([[1, 2, 3], [], None, [4]],
        ...                type=pa.list_(pa.int32()))
        >>> arr.value_lengths()
        <pyarrow.lib.Int32Array object at ...>
        [
          3,
          0,
          null,
          1
        ]
        """
        return _pc().list_value_length(self)


cdef class ListArray(BaseListArray):
    """
    Concrete class for Arrow arrays of a list data type.
    """

    @staticmethod
    def from_arrays(offsets, values, DataType type=None, MemoryPool pool=None, mask=None):
        """
        Construct ListArray from arrays of int32 offsets and values.

        Parameters
        ----------
        offsets : Array (int32 type)
        values : Array (any type)
        type : DataType, optional
            If not specified, a default ListType with the values' type is
            used.
        pool : MemoryPool, optional
        mask : Array (boolean type), optional
            Indicate which values are null (True) or not null (False).

        Returns
        -------
        list_array : ListArray

        Examples
        --------
        >>> import pyarrow as pa
        >>> values = pa.array([1, 2, 3, 4])
        >>> offsets = pa.array([0, 2, 4])
        >>> pa.ListArray.from_arrays(offsets, values)
        <pyarrow.lib.ListArray object at ...>
        [
          [
            1,
            2
          ],
          [
            3,
            4
          ]
        ]
        >>> # nulls in the offsets array become null lists
        >>> offsets = pa.array([0, None, 2, 4])
        >>> pa.ListArray.from_arrays(offsets, values)
        <pyarrow.lib.ListArray object at ...>
        [
          [
            1,
            2
          ],
          null,
          [
            3,
            4
          ]
        ]
        """
        cdef:
            Array _offsets, _values
            shared_ptr[CArray] out
            shared_ptr[CBuffer] c_mask
        cdef CMemoryPool* cpool = maybe_unbox_memory_pool(pool)

        _offsets = asarray(offsets, type='int32')
        _values = asarray(values)

        c_mask = c_mask_inverted_from_obj(mask, pool)

        if type is not None:
            with nogil:
                out = GetResultValue(
                    CListArray.FromArraysAndType(
                        type.sp_type, _offsets.ap[0], _values.ap[0], cpool, c_mask))
        else:
            with nogil:
                out = GetResultValue(
                    CListArray.FromArrays(
                        _offsets.ap[0], _values.ap[0], cpool, c_mask))
        cdef Array result = pyarrow_wrap_array(out)
        result.validate()
        return result

    @property
    def values(self):
        cdef CListArray* arr = <CListArray*> self.ap
        return pyarrow_wrap_array(arr.values())

    @property
    def offsets(self):
        """
        Return the list offsets as an int32 array.

        The returned array will not have a validity bitmap, so you cannot
        expect to pass it to `ListArray.from_arrays` and get back the same
        list array if the original one has nulls.

        Returns
        -------
        offsets : Int32Array

        Examples
        --------
        >>> import pyarrow as pa
        >>> array = pa.array([[1, 2], None, [3, 4, 5]])
        >>> array.offsets
        <pyarrow.lib.Int32Array object at ...>
        [
          0,
          2,
          2,
          5
        ]
        """
        return pyarrow_wrap_array((<CListArray*> self.ap).offsets())


cdef class LargeListArray(BaseListArray):
    """
    Concrete class for Arrow arrays of a large list data type.

    Identical to ListArray, but 64-bit offsets.
    """

    @staticmethod
    def from_arrays(offsets, values, DataType type=None, MemoryPool pool=None, mask=None):
        """
        Construct LargeListArray from arrays of int64 offsets and values.

        Parameters
        ----------
        offsets : Array (int64 type)
        values : Array (any type)
        type : DataType, optional
            If not specified, a default ListType with the values' type is
            used.
        pool : MemoryPool, optional
        mask : Array (boolean type), optional
            Indicate which values are null (True) or not null (False).

        Returns
        -------
        list_array : LargeListArray
        """
        cdef:
            Array _offsets, _values
            shared_ptr[CArray] out
            shared_ptr[CBuffer] c_mask

        cdef CMemoryPool* cpool = maybe_unbox_memory_pool(pool)

        _offsets = asarray(offsets, type='int64')
        _values = asarray(values)

        c_mask = c_mask_inverted_from_obj(mask, pool)

        if type is not None:
            with nogil:
                out = GetResultValue(
                    CLargeListArray.FromArraysAndType(
                        type.sp_type, _offsets.ap[0], _values.ap[0], cpool, c_mask))
        else:
            with nogil:
                out = GetResultValue(
                    CLargeListArray.FromArrays(
                        _offsets.ap[0], _values.ap[0], cpool, c_mask))
        cdef Array result = pyarrow_wrap_array(out)
        result.validate()
        return result

    @property
    def values(self):
        cdef CLargeListArray* arr = <CLargeListArray*> self.ap
        return pyarrow_wrap_array(arr.values())

    @property
    def offsets(self):
        """
        Return the list offsets as an int64 array.

        The returned array will not have a validity bitmap, so you cannot
        expect to pass it to `LargeListArray.from_arrays` and get back the
        same list array if the original one has nulls.

        Returns
        -------
        offsets : Int64Array
        """
        return pyarrow_wrap_array((<CLargeListArray*> self.ap).offsets())


cdef class MapArray(ListArray):
    """
    Concrete class for Arrow arrays of a map data type.
    """

    @staticmethod
    def from_arrays(offsets, keys, items, MemoryPool pool=None):
        """
        Construct MapArray from arrays of int32 offsets and key, item arrays.

        Parameters
        ----------
        offsets : array-like or sequence (int32 type)
        keys : array-like or sequence (any type)
        items : array-like or sequence (any type)
        pool : MemoryPool

        Returns
        -------
        map_array : MapArray
        """
        cdef:
            Array _offsets, _keys, _items
            shared_ptr[CArray] out
        cdef CMemoryPool* cpool = maybe_unbox_memory_pool(pool)

        _offsets = asarray(offsets, type='int32')
        _keys = asarray(keys)
        _items = asarray(items)

        with nogil:
            out = GetResultValue(
                CMapArray.FromArrays(_offsets.sp_array,
                                     _keys.sp_array,
                                     _items.sp_array, cpool))
        cdef Array result = pyarrow_wrap_array(out)
        result.validate()
        return result

    @property
    def keys(self):
        """Flattened array of keys across all maps in array"""
        return pyarrow_wrap_array((<CMapArray*> self.ap).keys())

    @property
    def items(self):
        """Flattened array of items across all maps in array"""
        return pyarrow_wrap_array((<CMapArray*> self.ap).items())


cdef class FixedSizeListArray(BaseListArray):
    """
    Concrete class for Arrow arrays of a fixed size list data type.
    """

    @staticmethod
    def from_arrays(values, list_size=None, DataType type=None):
        """
        Construct FixedSizeListArray from array of values and a list length.

        Parameters
        ----------
        values : Array (any type)
        list_size : int
            The fixed length of the lists.
        type : DataType, optional
            If not specified, a default ListType with the values' type and
            `list_size` length is used.

        Returns
        -------
        FixedSizeListArray

        Examples
        --------

        Create from a values array and a list size:

        >>> import pyarrow as pa
        >>> values = pa.array([1, 2, 3, 4])
        >>> arr = pa.FixedSizeListArray.from_arrays(values, 2)
        >>> arr
        <pyarrow.lib.FixedSizeListArray object at ...>
        [
          [
            1,
            2
          ],
          [
            3,
            4
          ]
        ]

        Or create from a values array, list size and matching type:

        >>> typ = pa.list_(pa.field("values", pa.int64()), 2)
        >>> arr = pa.FixedSizeListArray.from_arrays(values,type=typ)
        >>> arr
        <pyarrow.lib.FixedSizeListArray object at ...>
        [
          [
            1,
            2
          ],
          [
            3,
            4
          ]
        ]
        """
        cdef:
            Array _values
            int32_t _list_size
            CResult[shared_ptr[CArray]] c_result

        _values = asarray(values)

        if type is not None:
            if list_size is not None:
                raise ValueError("Cannot specify both list_size and type")
            with nogil:
                c_result = CFixedSizeListArray.FromArraysAndType(
                    _values.sp_array, type.sp_type)
        else:
            if list_size is None:
                raise ValueError("Should specify one of list_size and type")
            _list_size = <int32_t>list_size
            with nogil:
                c_result = CFixedSizeListArray.FromArrays(
                    _values.sp_array, _list_size)
        cdef Array result = pyarrow_wrap_array(GetResultValue(c_result))
        result.validate()
        return result

    @property
    def values(self):
        cdef CFixedSizeListArray* arr = <CFixedSizeListArray*> self.ap
        return pyarrow_wrap_array(arr.values())


cdef class UnionArray(Array):
    """
    Concrete class for Arrow arrays of a Union data type.
    """

    def child(self, int pos):
        import warnings
        warnings.warn("child is deprecated, use field", FutureWarning)
        return self.field(pos)

    def field(self, int pos):
        """
        Return the given child field as an individual array.

        For sparse unions, the returned array has its offset, length,
        and null count adjusted.

        For dense unions, the returned array is unchanged.

        Parameters
        ----------
        pos : int
            The physical index of the union child field (not its type code).

        Returns
        -------
        field : Array
            The given child field.
        """
        cdef shared_ptr[CArray] result
        result = (<CUnionArray*> self.ap).field(pos)
        if result != NULL:
            return pyarrow_wrap_array(result)
        raise KeyError("UnionArray does not have child {}".format(pos))

    @property
    def type_codes(self):
        """Get the type codes array."""
        buf = pyarrow_wrap_buffer((<CUnionArray*> self.ap).type_codes())
        return Array.from_buffers(int8(), len(self), [None, buf])

    @property
    def offsets(self):
        """
        Get the value offsets array (dense arrays only).

        Does not account for any slice offset.
        """
        if self.type.mode != "dense":
            raise ArrowTypeError("Can only get value offsets for dense arrays")
        cdef CDenseUnionArray* dense = <CDenseUnionArray*> self.ap
        buf = pyarrow_wrap_buffer(dense.value_offsets())
        return Array.from_buffers(int32(), len(self), [None, buf])

    @staticmethod
    def from_dense(Array types, Array value_offsets, list children,
                   list field_names=None, list type_codes=None):
        """
        Construct dense UnionArray from arrays of int8 types, int32 offsets and
        children arrays

        Parameters
        ----------
        types : Array (int8 type)
        value_offsets : Array (int32 type)
        children : list
        field_names : list
        type_codes : list

        Returns
        -------
        union_array : UnionArray
        """
        cdef:
            shared_ptr[CArray] out
            vector[shared_ptr[CArray]] c
            Array child
            vector[c_string] c_field_names
            vector[int8_t] c_type_codes

        for child in children:
            c.push_back(child.sp_array)
        if field_names is not None:
            for x in field_names:
                c_field_names.push_back(tobytes(x))
        if type_codes is not None:
            for x in type_codes:
                c_type_codes.push_back(x)

        with nogil:
            out = GetResultValue(CDenseUnionArray.Make(
                deref(types.ap), deref(value_offsets.ap), c, c_field_names,
                c_type_codes))

        cdef Array result = pyarrow_wrap_array(out)
        result.validate()
        return result

    @staticmethod
    def from_sparse(Array types, list children, list field_names=None,
                    list type_codes=None):
        """
        Construct sparse UnionArray from arrays of int8 types and children
        arrays

        Parameters
        ----------
        types : Array (int8 type)
        children : list
        field_names : list
        type_codes : list

        Returns
        -------
        union_array : UnionArray
        """
        cdef:
            shared_ptr[CArray] out
            vector[shared_ptr[CArray]] c
            Array child
            vector[c_string] c_field_names
            vector[int8_t] c_type_codes

        for child in children:
            c.push_back(child.sp_array)
        if field_names is not None:
            for x in field_names:
                c_field_names.push_back(tobytes(x))
        if type_codes is not None:
            for x in type_codes:
                c_type_codes.push_back(x)

        with nogil:
            out = GetResultValue(CSparseUnionArray.Make(
                deref(types.ap), c, c_field_names, c_type_codes))

        cdef Array result = pyarrow_wrap_array(out)
        result.validate()
        return result


cdef class StringArray(Array):
    """
    Concrete class for Arrow arrays of string (or utf8) data type.
    """

    @staticmethod
    def from_buffers(int length, Buffer value_offsets, Buffer data,
                     Buffer null_bitmap=None, int null_count=-1,
                     int offset=0):
        """
        Construct a StringArray from value_offsets and data buffers.
        If there are nulls in the data, also a null_bitmap and the matching
        null_count must be passed.

        Parameters
        ----------
        length : int
        value_offsets : Buffer
        data : Buffer
        null_bitmap : Buffer, optional
        null_count : int, default 0
        offset : int, default 0

        Returns
        -------
        string_array : StringArray
        """
        return Array.from_buffers(utf8(), length,
                                  [null_bitmap, value_offsets, data],
                                  null_count, offset)


cdef class LargeStringArray(Array):
    """
    Concrete class for Arrow arrays of large string (or utf8) data type.
    """

    @staticmethod
    def from_buffers(int length, Buffer value_offsets, Buffer data,
                     Buffer null_bitmap=None, int null_count=-1,
                     int offset=0):
        """
        Construct a LargeStringArray from value_offsets and data buffers.
        If there are nulls in the data, also a null_bitmap and the matching
        null_count must be passed.

        Parameters
        ----------
        length : int
        value_offsets : Buffer
        data : Buffer
        null_bitmap : Buffer, optional
        null_count : int, default 0
        offset : int, default 0

        Returns
        -------
        string_array : StringArray
        """
        return Array.from_buffers(large_utf8(), length,
                                  [null_bitmap, value_offsets, data],
                                  null_count, offset)


cdef class BinaryArray(Array):
    """
    Concrete class for Arrow arrays of variable-sized binary data type.
    """
    @property
    def total_values_length(self):
        """
        The number of bytes from beginning to end of the data buffer addressed
        by the offsets of this BinaryArray.
        """
        return (<CBinaryArray*> self.ap).total_values_length()


cdef class LargeBinaryArray(Array):
    """
    Concrete class for Arrow arrays of large variable-sized binary data type.
    """
    @property
    def total_values_length(self):
        """
        The number of bytes from beginning to end of the data buffer addressed
        by the offsets of this LargeBinaryArray.
        """
        return (<CLargeBinaryArray*> self.ap).total_values_length()


cdef class DictionaryArray(Array):
    """
    Concrete class for dictionary-encoded Arrow arrays.
    """

    def dictionary_encode(self):
        return self

    def dictionary_decode(self):
        """
        Decodes the DictionaryArray to an Array.
        """
        return self.dictionary.take(self.indices)

    @property
    def dictionary(self):
        cdef CDictionaryArray* darr = <CDictionaryArray*>(self.ap)

        if self._dictionary is None:
            self._dictionary = pyarrow_wrap_array(darr.dictionary())

        return self._dictionary

    @property
    def indices(self):
        cdef CDictionaryArray* darr = <CDictionaryArray*>(self.ap)

        if self._indices is None:
            self._indices = pyarrow_wrap_array(darr.indices())

        return self._indices

    @staticmethod
    def from_buffers(DataType type, int64_t length, buffers, Array dictionary,
                     int64_t null_count=-1, int64_t offset=0):
        """
        Construct a DictionaryArray from buffers.

        Parameters
        ----------
        type : pyarrow.DataType
        length : int
            The number of values in the array.
        buffers : List[Buffer]
            The buffers backing the indices array.
        dictionary : pyarrow.Array, ndarray or pandas.Series
            The array of values referenced by the indices.
        null_count : int, default -1
            The number of null entries in the indices array. Negative value means that
            the null count is not known.
        offset : int, default 0
            The array's logical offset (in values, not in bytes) from the
            start of each buffer.

        Returns
        -------
        dict_array : DictionaryArray
        """
        cdef:
            vector[shared_ptr[CBuffer]] c_buffers
            shared_ptr[CDataType] c_type
            shared_ptr[CArrayData] c_data
            shared_ptr[CArray] c_result

        for buf in buffers:
            c_buffers.push_back(pyarrow_unwrap_buffer(buf))

        c_type = pyarrow_unwrap_data_type(type)

        with nogil:
            c_data = CArrayData.Make(
                c_type, length, c_buffers, null_count, offset)
            c_data.get().dictionary = dictionary.sp_array.get().data()
            c_result.reset(new CDictionaryArray(c_data))

        cdef Array result = pyarrow_wrap_array(c_result)
        result.validate()
        return result

    @staticmethod
    def from_arrays(indices, dictionary, mask=None, bint ordered=False,
                    bint from_pandas=False, bint safe=True,
                    MemoryPool memory_pool=None):
        """
        Construct a DictionaryArray from indices and values.

        Parameters
        ----------
        indices : pyarrow.Array, numpy.ndarray or pandas.Series, int type
            Non-negative integers referencing the dictionary values by zero
            based index.
        dictionary : pyarrow.Array, ndarray or pandas.Series
            The array of values referenced by the indices.
        mask : ndarray or pandas.Series, bool type
            True values indicate that indices are actually null.
        ordered : bool, default False
            Set to True if the category values are ordered.
        from_pandas : bool, default False
            If True, the indices should be treated as though they originated in
            a pandas.Categorical (null encoded as -1).
        safe : bool, default True
            If True, check that the dictionary indices are in range.
        memory_pool : MemoryPool, default None
            For memory allocations, if required, otherwise uses default pool.

        Returns
        -------
        dict_array : DictionaryArray
        """
        cdef:
            Array _indices, _dictionary
            shared_ptr[CDataType] c_type
            shared_ptr[CArray] c_result

        if isinstance(indices, Array):
            if mask is not None:
                raise NotImplementedError(
                    "mask not implemented with Arrow array inputs yet")
            _indices = indices
        else:
            if from_pandas:
                _indices = _codes_to_indices(indices, mask, None, memory_pool)
            else:
                _indices = array(indices, mask=mask, memory_pool=memory_pool)

        if isinstance(dictionary, Array):
            _dictionary = dictionary
        else:
            _dictionary = array(dictionary, memory_pool=memory_pool)

        if not isinstance(_indices, IntegerArray):
            raise ValueError('Indices must be integer type')

        cdef c_bool c_ordered = ordered

        c_type.reset(new CDictionaryType(_indices.type.sp_type,
                                         _dictionary.sp_array.get().type(),
                                         c_ordered))

        if safe:
            with nogil:
                c_result = GetResultValue(
                    CDictionaryArray.FromArrays(c_type, _indices.sp_array,
                                                _dictionary.sp_array))
        else:
            c_result.reset(new CDictionaryArray(c_type, _indices.sp_array,
                                                _dictionary.sp_array))

        cdef Array result = pyarrow_wrap_array(c_result)
        result.validate()
        return result


cdef class StructArray(Array):
    """
    Concrete class for Arrow arrays of a struct data type.
    """

    def field(self, index):
        """
        Retrieves the child array belonging to field.

        Parameters
        ----------
        index : Union[int, str]
            Index / position or name of the field.

        Returns
        -------
        result : Array
        """
        cdef:
            CStructArray* arr = <CStructArray*> self.ap
            shared_ptr[CArray] child

        if isinstance(index, (bytes, str)):
            child = arr.GetFieldByName(tobytes(index))
            if child == nullptr:
                raise KeyError(index)
        elif isinstance(index, int):
            child = arr.field(
                <int>_normalize_index(index, self.ap.num_fields()))
        else:
            raise TypeError('Expected integer or string index')

        return pyarrow_wrap_array(child)

    def _flattened_field(self, index, MemoryPool memory_pool=None):
        """
        Retrieves the child array belonging to field,
        accounting for the parent array null bitmap.

        Parameters
        ----------
        index : Union[int, str]
            Index / position or name of the field.
        memory_pool : MemoryPool, default None
            For memory allocations, if required, otherwise use default pool.

        Returns
        -------
        result : Array
        """
        cdef:
            CStructArray* arr = <CStructArray*> self.ap
            shared_ptr[CArray] child
            CMemoryPool* pool = maybe_unbox_memory_pool(memory_pool)

        if isinstance(index, (bytes, str)):
            int_index = self.type.get_field_index(index)
            if int_index < 0:
                raise KeyError(index)
        elif isinstance(index, int):
            int_index = _normalize_index(index, self.ap.num_fields())
        else:
            raise TypeError('Expected integer or string index')

        child = GetResultValue(arr.GetFlattenedField(int_index, pool))
        return pyarrow_wrap_array(child)

    def flatten(self, MemoryPool memory_pool=None):
        """
        Return one individual array for each field in the struct.

        Parameters
        ----------
        memory_pool : MemoryPool, default None
            For memory allocations, if required, otherwise use default pool.

        Returns
        -------
        result : List[Array]
        """
        cdef:
            vector[shared_ptr[CArray]] arrays
            CMemoryPool* pool = maybe_unbox_memory_pool(memory_pool)
            CStructArray* sarr = <CStructArray*> self.ap

        with nogil:
            arrays = GetResultValue(sarr.Flatten(pool))

        return [pyarrow_wrap_array(arr) for arr in arrays]

    @staticmethod
    def from_arrays(arrays, names=None, fields=None, mask=None,
                    memory_pool=None):
        """
        Construct StructArray from collection of arrays representing
        each field in the struct.

        Either field names or field instances must be passed.

        Parameters
        ----------
        arrays : sequence of Array
        names : List[str] (optional)
            Field names for each struct child.
        fields : List[Field] (optional)
            Field instances for each struct child.
        mask : pyarrow.Array[bool] (optional)
            Indicate which values are null (True) or not null (False).
        memory_pool : MemoryPool (optional)
            For memory allocations, if required, otherwise uses default pool.

        Returns
        -------
        result : StructArray
        """
        cdef:
            shared_ptr[CArray] c_array
            shared_ptr[CBuffer] c_mask
            vector[shared_ptr[CArray]] c_arrays
            vector[c_string] c_names
            vector[shared_ptr[CField]] c_fields
            CResult[shared_ptr[CArray]] c_result
            ssize_t num_arrays
            ssize_t length
            ssize_t i
            Field py_field
            DataType struct_type

        if names is None and fields is None:
            raise ValueError('Must pass either names or fields')
        if names is not None and fields is not None:
            raise ValueError('Must pass either names or fields, not both')

        c_mask = c_mask_inverted_from_obj(mask, memory_pool)

        arrays = [asarray(x) for x in arrays]
        for arr in arrays:
            c_array = pyarrow_unwrap_array(arr)
            if c_array == nullptr:
                raise TypeError(f"Expected Array, got {arr.__class__}")
            c_arrays.push_back(c_array)
        if names is not None:
            for name in names:
                c_names.push_back(tobytes(name))
        else:
            for item in fields:
                if isinstance(item, tuple):
                    py_field = field(*item)
                else:
                    py_field = item
                c_fields.push_back(py_field.sp_field)

        if (c_arrays.size() == 0 and c_names.size() == 0 and
                c_fields.size() == 0):
            # The C++ side doesn't allow this
            return array([], struct([]))

        if names is not None:
            # XXX Cannot pass "nullptr" for a shared_ptr<T> argument:
            # https://github.com/cython/cython/issues/3020
            c_result = CStructArray.MakeFromFieldNames(
                c_arrays, c_names, c_mask, -1, 0)
        else:
            c_result = CStructArray.MakeFromFields(
                c_arrays, c_fields, c_mask, -1, 0)
        cdef Array result = pyarrow_wrap_array(GetResultValue(c_result))
        result.validate()
        return result

    def sort(self, order="ascending", by=None, **kwargs):
        """
        Sort the StructArray

        Parameters
        ----------
        order : str, default "ascending"
            Which order to sort values in.
            Accepted values are "ascending", "descending".
        by : str or None, default None
            If to sort the array by one of its fields
            or by the whole array.
        **kwargs : dict, optional
            Additional sorting options.
            As allowed by :class:`SortOptions`

        Returns
        -------
        result : StructArray
        """
        if by is not None:
            tosort = self._flattened_field(by)
        else:
            tosort = self
        indices = _pc().sort_indices(
            tosort,
            options=_pc().SortOptions(sort_keys=[("", order)], **kwargs)
        )
        return self.take(indices)


cdef class ExtensionArray(Array):
    """
    Concrete class for Arrow extension arrays.
    """

    @property
    def storage(self):
        cdef:
            CExtensionArray* ext_array = <CExtensionArray*>(self.ap)

        return pyarrow_wrap_array(ext_array.storage())

    @staticmethod
    def from_storage(BaseExtensionType typ, Array storage):
        """
        Construct ExtensionArray from type and storage array.

        Parameters
        ----------
        typ : DataType
            The extension type for the result array.
        storage : Array
            The underlying storage for the result array.

        Returns
        -------
        ext_array : ExtensionArray
        """
        cdef:
            shared_ptr[CExtensionArray] ext_array

        if storage.type != typ.storage_type:
            raise TypeError("Incompatible storage type {0} "
                            "for extension type {1}".format(storage.type, typ))

        ext_array = make_shared[CExtensionArray](typ.sp_type, storage.sp_array)
        cdef Array result = pyarrow_wrap_array(<shared_ptr[CArray]> ext_array)
        result.validate()
        return result

    def _to_pandas(self, options, **kwargs):
        pandas_dtype = None
        try:
            pandas_dtype = self.type.to_pandas_dtype()
        except NotImplementedError:
            pass

        # pandas ExtensionDtype that implements conversion from pyarrow
        if hasattr(pandas_dtype, '__from_arrow__'):
            arr = pandas_dtype.__from_arrow__(self)
            return pandas_api.series(arr)

        # otherwise convert the storage array with the base implementation
        return Array._to_pandas(self.storage, options, **kwargs)

    def to_numpy(self, **kwargs):
        """
        Convert extension array to a numpy ndarray.

        This method simply delegates to the underlying storage array.

        Parameters
        ----------
        **kwargs : dict, optional
            See `Array.to_numpy` for parameter description.

        See Also
        --------
        Array.to_numpy
        """
        return self.storage.to_numpy(**kwargs)


cdef dict _array_classes = {
    _Type_NA: NullArray,
    _Type_BOOL: BooleanArray,
    _Type_UINT8: UInt8Array,
    _Type_UINT16: UInt16Array,
    _Type_UINT32: UInt32Array,
    _Type_UINT64: UInt64Array,
    _Type_INT8: Int8Array,
    _Type_INT16: Int16Array,
    _Type_INT32: Int32Array,
    _Type_INT64: Int64Array,
    _Type_DATE32: Date32Array,
    _Type_DATE64: Date64Array,
    _Type_TIMESTAMP: TimestampArray,
    _Type_TIME32: Time32Array,
    _Type_TIME64: Time64Array,
    _Type_DURATION: DurationArray,
    _Type_INTERVAL_MONTH_DAY_NANO: MonthDayNanoIntervalArray,
    _Type_HALF_FLOAT: HalfFloatArray,
    _Type_FLOAT: FloatArray,
    _Type_DOUBLE: DoubleArray,
    _Type_LIST: ListArray,
    _Type_LARGE_LIST: LargeListArray,
    _Type_MAP: MapArray,
    _Type_FIXED_SIZE_LIST: FixedSizeListArray,
    _Type_SPARSE_UNION: UnionArray,
    _Type_DENSE_UNION: UnionArray,
    _Type_BINARY: BinaryArray,
    _Type_STRING: StringArray,
    _Type_LARGE_BINARY: LargeBinaryArray,
    _Type_LARGE_STRING: LargeStringArray,
    _Type_DICTIONARY: DictionaryArray,
    _Type_FIXED_SIZE_BINARY: FixedSizeBinaryArray,
    _Type_DECIMAL128: Decimal128Array,
    _Type_DECIMAL256: Decimal256Array,
    _Type_STRUCT: StructArray,
    _Type_EXTENSION: ExtensionArray,
}


cdef inline shared_ptr[CBuffer] c_mask_inverted_from_obj(object mask, MemoryPool pool) except *:
    """
    Convert mask array obj to c_mask while also inverting to signify 1 for valid and 0 for null
    """
    cdef shared_ptr[CBuffer] c_mask
    if mask is None:
        c_mask = shared_ptr[CBuffer]()
    elif isinstance(mask, Array):
        if mask.type.id != Type_BOOL:
            raise TypeError('Mask must be a pyarrow.Array of type boolean')
        if mask.null_count != 0:
            raise ValueError('Mask must not contain nulls')
        inverted_mask = _pc().invert(mask, memory_pool=pool)
        c_mask = pyarrow_unwrap_buffer(inverted_mask.buffers()[1])
    else:
        raise TypeError('Mask must be a pyarrow.Array of type boolean')
    return c_mask


cdef object get_array_class_from_type(
        const shared_ptr[CDataType]& sp_data_type):
    cdef CDataType* data_type = sp_data_type.get()
    if data_type == NULL:
        raise ValueError('Array data type was NULL')

    if data_type.id() == _Type_EXTENSION:
        py_ext_data_type = pyarrow_wrap_data_type(sp_data_type)
        return py_ext_data_type.__arrow_ext_class__()
    else:
        return _array_classes[data_type.id()]


cdef object get_values(object obj, bint* is_series):
    if pandas_api.is_series(obj) or pandas_api.is_index(obj):
        result = pandas_api.get_values(obj)
        is_series[0] = True
    elif isinstance(obj, np.ndarray):
        result = obj
        is_series[0] = False
    else:
        result = pandas_api.series(obj).values
        is_series[0] = False

    return result


def concat_arrays(arrays, MemoryPool memory_pool=None):
    """
    Concatenate the given arrays.

    The contents of the input arrays are copied into the returned array.

    Raises
    ------
    ArrowInvalid
        If not all of the arrays have the same type.

    Parameters
    ----------
    arrays : iterable of pyarrow.Array
        Arrays to concatenate, must be identically typed.
    memory_pool : MemoryPool, default None
        For memory allocations. If None, the default pool is used.

    Examples
    --------
    >>> import pyarrow as pa
    >>> arr1 = pa.array([2, 4, 5, 100])
    >>> arr2 = pa.array([2, 4])
    >>> pa.concat_arrays([arr1, arr2])
    <pyarrow.lib.Int64Array object at ...>
    [
      2,
      4,
      5,
      100,
      2,
      4
    ]

    """
    cdef:
        vector[shared_ptr[CArray]] c_arrays
        shared_ptr[CArray] c_concatenated
        CMemoryPool* pool = maybe_unbox_memory_pool(memory_pool)

    for array in arrays:
        if not isinstance(array, Array):
            raise TypeError("Iterable should contain Array objects, "
                            "got {0} instead".format(type(array)))
        c_arrays.push_back(pyarrow_unwrap_array(array))

    with nogil:
        c_concatenated = GetResultValue(Concatenate(c_arrays, pool))

    return pyarrow_wrap_array(c_concatenated)


def _empty_array(DataType type):
    """
    Create empty array of the given type.
    """
    if type.id == Type_DICTIONARY:
        arr = DictionaryArray.from_arrays(
            _empty_array(type.index_type), _empty_array(type.value_type),
            ordered=type.ordered)
    else:
        arr = array([], type=type)
    return arr

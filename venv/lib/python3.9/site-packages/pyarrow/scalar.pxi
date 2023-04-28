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

import collections


cdef class Scalar(_Weakrefable):
    """
    The base class for scalars.
    """

    def __init__(self):
        raise TypeError("Do not call {}'s constructor directly, use "
                        "pa.scalar() instead.".format(self.__class__.__name__))

    cdef void init(self, const shared_ptr[CScalar]& wrapped):
        self.wrapped = wrapped

    @staticmethod
    cdef wrap(const shared_ptr[CScalar]& wrapped):
        cdef:
            Scalar self
            Type type_id = wrapped.get().type.get().id()
            shared_ptr[CDataType] sp_data_type = wrapped.get().type

        if type_id == _Type_NA:
            return _NULL

        if type_id not in _scalar_classes:
            raise NotImplementedError(
                "Wrapping scalar of type " + frombytes(sp_data_type.get().ToString()))

        typ = get_scalar_class_from_type(sp_data_type)
        self = typ.__new__(typ)
        self.init(wrapped)

        return self

    cdef inline shared_ptr[CScalar] unwrap(self) nogil:
        return self.wrapped

    @property
    def type(self):
        """
        Data type of the Scalar object.
        """
        return pyarrow_wrap_data_type(self.wrapped.get().type)

    @property
    def is_valid(self):
        """
        Holds a valid (non-null) value.
        """
        return self.wrapped.get().is_valid

    def cast(self, object target_type):
        """
        Attempt a safe cast to target data type.

        Parameters
        ----------
        target_type : DataType or string coercible to DataType
            The type to cast the scalar to.

        Returns
        -------
        scalar : A Scalar of the given target data type.
        """
        cdef:
            DataType type = ensure_type(target_type)
            shared_ptr[CScalar] result

        with nogil:
            result = GetResultValue(self.wrapped.get().CastTo(type.sp_type))

        return Scalar.wrap(result)

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
                check_status(self.wrapped.get().ValidateFull())
        else:
            with nogil:
                check_status(self.wrapped.get().Validate())

    def __repr__(self):
        return '<pyarrow.{}: {!r}>'.format(
            self.__class__.__name__, self.as_py()
        )

    def __str__(self):
        return str(self.as_py())

    def equals(self, Scalar other not None):
        return self.wrapped.get().Equals(other.unwrap().get()[0])

    def __eq__(self, other):
        try:
            return self.equals(other)
        except TypeError:
            return NotImplemented

    def __hash__(self):
        cdef CScalarHash hasher
        return hasher(self.wrapped)

    def __reduce__(self):
        return scalar, (self.as_py(), self.type)

    def as_py(self):
        raise NotImplementedError()


_NULL = NA = None


cdef class NullScalar(Scalar):
    """
    Concrete class for null scalars.
    """

    def __cinit__(self):
        global NA
        if NA is not None:
            raise RuntimeError('Cannot create multiple NullScalar instances')
        self.init(shared_ptr[CScalar](new CNullScalar()))

    def __init__(self):
        pass

    def as_py(self):
        """
        Return this value as a Python None.
        """
        return None


_NULL = NA = NullScalar()


cdef class BooleanScalar(Scalar):
    """
    Concrete class for boolean scalars.
    """

    def as_py(self):
        """
        Return this value as a Python bool.
        """
        cdef CBooleanScalar* sp = <CBooleanScalar*> self.wrapped.get()
        return sp.value if sp.is_valid else None


cdef class UInt8Scalar(Scalar):
    """
    Concrete class for uint8 scalars.
    """

    def as_py(self):
        """
        Return this value as a Python int.
        """
        cdef CUInt8Scalar* sp = <CUInt8Scalar*> self.wrapped.get()
        return sp.value if sp.is_valid else None


cdef class Int8Scalar(Scalar):
    """
    Concrete class for int8 scalars.
    """

    def as_py(self):
        """
        Return this value as a Python int.
        """
        cdef CInt8Scalar* sp = <CInt8Scalar*> self.wrapped.get()
        return sp.value if sp.is_valid else None


cdef class UInt16Scalar(Scalar):
    """
    Concrete class for uint16 scalars.
    """

    def as_py(self):
        """
        Return this value as a Python int.
        """
        cdef CUInt16Scalar* sp = <CUInt16Scalar*> self.wrapped.get()
        return sp.value if sp.is_valid else None


cdef class Int16Scalar(Scalar):
    """
    Concrete class for int16 scalars.
    """

    def as_py(self):
        """
        Return this value as a Python int.
        """
        cdef CInt16Scalar* sp = <CInt16Scalar*> self.wrapped.get()
        return sp.value if sp.is_valid else None


cdef class UInt32Scalar(Scalar):
    """
    Concrete class for uint32 scalars.
    """

    def as_py(self):
        """
        Return this value as a Python int.
        """
        cdef CUInt32Scalar* sp = <CUInt32Scalar*> self.wrapped.get()
        return sp.value if sp.is_valid else None


cdef class Int32Scalar(Scalar):
    """
    Concrete class for int32 scalars.
    """

    def as_py(self):
        """
        Return this value as a Python int.
        """
        cdef CInt32Scalar* sp = <CInt32Scalar*> self.wrapped.get()
        return sp.value if sp.is_valid else None


cdef class UInt64Scalar(Scalar):
    """
    Concrete class for uint64 scalars.
    """

    def as_py(self):
        """
        Return this value as a Python int.
        """
        cdef CUInt64Scalar* sp = <CUInt64Scalar*> self.wrapped.get()
        return sp.value if sp.is_valid else None


cdef class Int64Scalar(Scalar):
    """
    Concrete class for int64 scalars.
    """

    def as_py(self):
        """
        Return this value as a Python int.
        """
        cdef CInt64Scalar* sp = <CInt64Scalar*> self.wrapped.get()
        return sp.value if sp.is_valid else None


cdef class HalfFloatScalar(Scalar):
    """
    Concrete class for float scalars.
    """

    def as_py(self):
        """
        Return this value as a Python float.
        """
        cdef CHalfFloatScalar* sp = <CHalfFloatScalar*> self.wrapped.get()
        return PyHalf_FromHalf(sp.value) if sp.is_valid else None


cdef class FloatScalar(Scalar):
    """
    Concrete class for float scalars.
    """

    def as_py(self):
        """
        Return this value as a Python float.
        """
        cdef CFloatScalar* sp = <CFloatScalar*> self.wrapped.get()
        return sp.value if sp.is_valid else None


cdef class DoubleScalar(Scalar):
    """
    Concrete class for double scalars.
    """

    def as_py(self):
        """
        Return this value as a Python float.
        """
        cdef CDoubleScalar* sp = <CDoubleScalar*> self.wrapped.get()
        return sp.value if sp.is_valid else None


cdef class Decimal128Scalar(Scalar):
    """
    Concrete class for decimal128 scalars.
    """

    def as_py(self):
        """
        Return this value as a Python Decimal.
        """
        cdef:
            CDecimal128Scalar* sp = <CDecimal128Scalar*> self.wrapped.get()
            CDecimal128Type* dtype = <CDecimal128Type*> sp.type.get()
        if sp.is_valid:
            return _pydecimal.Decimal(
                frombytes(sp.value.ToString(dtype.scale()))
            )
        else:
            return None


cdef class Decimal256Scalar(Scalar):
    """
    Concrete class for decimal256 scalars.
    """

    def as_py(self):
        """
        Return this value as a Python Decimal.
        """
        cdef:
            CDecimal256Scalar* sp = <CDecimal256Scalar*> self.wrapped.get()
            CDecimal256Type* dtype = <CDecimal256Type*> sp.type.get()
        if sp.is_valid:
            return _pydecimal.Decimal(
                frombytes(sp.value.ToString(dtype.scale()))
            )
        else:
            return None


cdef class Date32Scalar(Scalar):
    """
    Concrete class for date32 scalars.
    """

    @property
    def value(self):
        cdef CDate32Scalar* sp = <CDate32Scalar*> self.wrapped.get()
        return sp.value if sp.is_valid else None

    def as_py(self):
        """
        Return this value as a Python datetime.datetime instance.
        """
        cdef CDate32Scalar* sp = <CDate32Scalar*> self.wrapped.get()

        if sp.is_valid:
            # shift to seconds since epoch
            return (
                datetime.date(1970, 1, 1) + datetime.timedelta(days=sp.value)
            )
        else:
            return None


cdef class Date64Scalar(Scalar):
    """
    Concrete class for date64 scalars.
    """

    @property
    def value(self):
        cdef CDate64Scalar* sp = <CDate64Scalar*> self.wrapped.get()
        return sp.value if sp.is_valid else None

    def as_py(self):
        """
        Return this value as a Python datetime.datetime instance.
        """
        cdef CDate64Scalar* sp = <CDate64Scalar*> self.wrapped.get()

        if sp.is_valid:
            return (
                datetime.date(1970, 1, 1) +
                datetime.timedelta(days=sp.value / 86400000)
            )
        else:
            return None


def _datetime_from_int(int64_t value, TimeUnit unit, tzinfo=None):
    if unit == TimeUnit_SECOND:
        delta = datetime.timedelta(seconds=value)
    elif unit == TimeUnit_MILLI:
        delta = datetime.timedelta(milliseconds=value)
    elif unit == TimeUnit_MICRO:
        delta = datetime.timedelta(microseconds=value)
    else:
        # TimeUnit_NANO: prefer pandas timestamps if available
        if _pandas_api.have_pandas:
            return _pandas_api.pd.Timestamp(value, tz=tzinfo, unit='ns')
        # otherwise safely truncate to microsecond resolution datetime
        if value % 1000 != 0:
            raise ValueError(
                "Nanosecond resolution temporal type {} is not safely "
                "convertible to microseconds to convert to datetime.datetime. "
                "Install pandas to return as Timestamp with nanosecond "
                "support or access the .value attribute.".format(value)
            )
        delta = datetime.timedelta(microseconds=value // 1000)

    dt = datetime.datetime(1970, 1, 1) + delta
    # adjust timezone if set to the datatype
    if tzinfo is not None:
        dt = dt.replace(tzinfo=datetime.timezone.utc).astimezone(tzinfo)

    return dt


cdef class Time32Scalar(Scalar):
    """
    Concrete class for time32 scalars.
    """

    @property
    def value(self):
        cdef CTime32Scalar* sp = <CTime32Scalar*> self.wrapped.get()
        return sp.value if sp.is_valid else None

    def as_py(self):
        """
        Return this value as a Python datetime.timedelta instance.
        """
        cdef:
            CTime32Scalar* sp = <CTime32Scalar*> self.wrapped.get()
            CTime32Type* dtype = <CTime32Type*> sp.type.get()

        if sp.is_valid:
            return _datetime_from_int(sp.value, unit=dtype.unit()).time()
        else:
            return None


cdef class Time64Scalar(Scalar):
    """
    Concrete class for time64 scalars.
    """

    @property
    def value(self):
        cdef CTime64Scalar* sp = <CTime64Scalar*> self.wrapped.get()
        return sp.value if sp.is_valid else None

    def as_py(self):
        """
        Return this value as a Python datetime.timedelta instance.
        """
        cdef:
            CTime64Scalar* sp = <CTime64Scalar*> self.wrapped.get()
            CTime64Type* dtype = <CTime64Type*> sp.type.get()

        if sp.is_valid:
            return _datetime_from_int(sp.value, unit=dtype.unit()).time()
        else:
            return None


cdef class TimestampScalar(Scalar):
    """
    Concrete class for timestamp scalars.
    """

    @property
    def value(self):
        cdef CTimestampScalar* sp = <CTimestampScalar*> self.wrapped.get()
        return sp.value if sp.is_valid else None

    def as_py(self):
        """
        Return this value as a Pandas Timestamp instance (if units are
        nanoseconds and pandas is available), otherwise as a Python
        datetime.datetime instance.
        """
        cdef:
            CTimestampScalar* sp = <CTimestampScalar*> self.wrapped.get()
            CTimestampType* dtype = <CTimestampType*> sp.type.get()

        if not sp.is_valid:
            return None

        if not dtype.timezone().empty():
            tzinfo = string_to_tzinfo(frombytes(dtype.timezone()))
        else:
            tzinfo = None

        return _datetime_from_int(sp.value, unit=dtype.unit(), tzinfo=tzinfo)


cdef class DurationScalar(Scalar):
    """
    Concrete class for duration scalars.
    """

    @property
    def value(self):
        cdef CDurationScalar* sp = <CDurationScalar*> self.wrapped.get()
        return sp.value if sp.is_valid else None

    def as_py(self):
        """
        Return this value as a Pandas Timedelta instance (if units are
        nanoseconds and pandas is available), otherwise as a Python
        datetime.timedelta instance.
        """
        cdef:
            CDurationScalar* sp = <CDurationScalar*> self.wrapped.get()
            CDurationType* dtype = <CDurationType*> sp.type.get()
            TimeUnit unit = dtype.unit()

        if not sp.is_valid:
            return None

        if unit == TimeUnit_SECOND:
            return datetime.timedelta(seconds=sp.value)
        elif unit == TimeUnit_MILLI:
            return datetime.timedelta(milliseconds=sp.value)
        elif unit == TimeUnit_MICRO:
            return datetime.timedelta(microseconds=sp.value)
        else:
            # TimeUnit_NANO: prefer pandas timestamps if available
            if _pandas_api.have_pandas:
                return _pandas_api.pd.Timedelta(sp.value, unit='ns')
            # otherwise safely truncate to microsecond resolution timedelta
            if sp.value % 1000 != 0:
                raise ValueError(
                    "Nanosecond duration {} is not safely convertible to "
                    "microseconds to convert to datetime.timedelta. Install "
                    "pandas to return as Timedelta with nanosecond support or "
                    "access the .value attribute.".format(sp.value)
                )
            return datetime.timedelta(microseconds=sp.value // 1000)


cdef class MonthDayNanoIntervalScalar(Scalar):
    """
    Concrete class for month, day, nanosecond interval scalars.
    """

    @property
    def value(self):
        """
        Same as self.as_py()
        """
        return self.as_py()

    def as_py(self):
        """
        Return this value as a pyarrow.MonthDayNano.
        """
        cdef:
            PyObject* val
            CMonthDayNanoIntervalScalar* scalar
        scalar = <CMonthDayNanoIntervalScalar*>self.wrapped.get()
        val = GetResultValue(MonthDayNanoIntervalScalarToPyObject(
            deref(scalar)))
        return PyObject_to_object(val)


cdef class BinaryScalar(Scalar):
    """
    Concrete class for binary-like scalars.
    """

    def as_buffer(self):
        """
        Return a view over this value as a Buffer object.
        """
        cdef CBaseBinaryScalar* sp = <CBaseBinaryScalar*> self.wrapped.get()
        return pyarrow_wrap_buffer(sp.value) if sp.is_valid else None

    def as_py(self):
        """
        Return this value as a Python bytes.
        """
        buffer = self.as_buffer()
        return None if buffer is None else buffer.to_pybytes()


cdef class LargeBinaryScalar(BinaryScalar):
    pass


cdef class FixedSizeBinaryScalar(BinaryScalar):
    pass


cdef class StringScalar(BinaryScalar):
    """
    Concrete class for string-like (utf8) scalars.
    """

    def as_py(self):
        """
        Return this value as a Python string.
        """
        buffer = self.as_buffer()
        return None if buffer is None else str(buffer, 'utf8')


cdef class LargeStringScalar(StringScalar):
    pass


cdef class ListScalar(Scalar):
    """
    Concrete class for list-like scalars.
    """

    @property
    def values(self):
        cdef CBaseListScalar* sp = <CBaseListScalar*> self.wrapped.get()
        if sp.is_valid:
            return pyarrow_wrap_array(sp.value)
        else:
            return None

    def __len__(self):
        """
        Return the number of values.
        """
        return len(self.values)

    def __getitem__(self, i):
        """
        Return the value at the given index.
        """
        return self.values[_normalize_index(i, len(self))]

    def __iter__(self):
        """
        Iterate over this element's values.
        """
        return iter(self.values)

    def as_py(self):
        """
        Return this value as a Python list.
        """
        arr = self.values
        return None if arr is None else arr.to_pylist()


cdef class FixedSizeListScalar(ListScalar):
    pass


cdef class LargeListScalar(ListScalar):
    pass


cdef class StructScalar(Scalar, collections.abc.Mapping):
    """
    Concrete class for struct scalars.
    """

    def __len__(self):
        cdef CStructScalar* sp = <CStructScalar*> self.wrapped.get()
        return sp.value.size()

    def __iter__(self):
        cdef:
            CStructScalar* sp = <CStructScalar*> self.wrapped.get()
            CStructType* dtype = <CStructType*> sp.type.get()
            vector[shared_ptr[CField]] fields = dtype.fields()

        for i in range(dtype.num_fields()):
            yield frombytes(fields[i].get().name())

    def items(self):
        return ((key, self[i]) for i, key in enumerate(self))

    def __contains__(self, key):
        return key in list(self)

    def __getitem__(self, key):
        """
        Return the child value for the given field.

        Parameters
        ----------
        index : Union[int, str]
            Index / position or name of the field.

        Returns
        -------
        result : Scalar
        """
        cdef:
            CFieldRef ref
            CStructScalar* sp = <CStructScalar*> self.wrapped.get()

        if isinstance(key, (bytes, str)):
            ref = CFieldRef(<c_string> tobytes(key))
        elif isinstance(key, int):
            ref = CFieldRef(<int> key)
        else:
            raise TypeError('Expected integer or string index')

        try:
            return Scalar.wrap(GetResultValue(sp.field(ref)))
        except ArrowInvalid as exc:
            if isinstance(key, int):
                raise IndexError(key) from exc
            else:
                raise KeyError(key) from exc

    def as_py(self):
        """
        Return this value as a Python dict.
        """
        if self.is_valid:
            try:
                return {k: self[k].as_py() for k in self.keys()}
            except KeyError:
                raise ValueError(
                    "Converting to Python dictionary is not supported when "
                    "duplicate field names are present")
        else:
            return None

    def _as_py_tuple(self):
        # a version that returns a tuple instead of dict to support repr/str
        # with the presence of duplicate field names
        if self.is_valid:
            return [(key, self[i].as_py()) for i, key in enumerate(self)]
        else:
            return None

    def __repr__(self):
        return '<pyarrow.{}: {!r}>'.format(
            self.__class__.__name__, self._as_py_tuple()
        )

    def __str__(self):
        return str(self._as_py_tuple())


cdef class MapScalar(ListScalar):
    """
    Concrete class for map scalars.
    """

    def __getitem__(self, i):
        """
        Return the value at the given index.
        """
        arr = self.values
        if arr is None:
            raise IndexError(i)
        dct = arr[_normalize_index(i, len(arr))]
        return (dct['key'], dct['value'])

    def __iter__(self):
        """
        Iterate over this element's values.
        """
        arr = self.values
        if array is None:
            raise StopIteration
        for k, v in zip(arr.field('key'), arr.field('value')):
            yield (k.as_py(), v.as_py())

    def as_py(self):
        """
        Return this value as a Python list.
        """
        cdef CStructScalar* sp = <CStructScalar*> self.wrapped.get()
        return list(self) if sp.is_valid else None


cdef class DictionaryScalar(Scalar):
    """
    Concrete class for dictionary-encoded scalars.
    """

    @classmethod
    def _reconstruct(cls, type, is_valid, index, dictionary):
        cdef:
            CDictionaryScalarIndexAndDictionary value
            shared_ptr[CDictionaryScalar] wrapped
            DataType type_
            Scalar index_
            Array dictionary_

        type_ = ensure_type(type, allow_none=False)
        if not isinstance(type_, DictionaryType):
            raise TypeError('Must pass a DictionaryType instance')

        if isinstance(index, Scalar):
            if not index.type.equals(type.index_type):
                raise TypeError("The Scalar value passed as index must have "
                                "identical type to the dictionary type's "
                                "index_type")
            index_ = index
        else:
            index_ = scalar(index, type=type_.index_type)

        if isinstance(dictionary, Array):
            if not dictionary.type.equals(type.value_type):
                raise TypeError("The Array passed as dictionary must have "
                                "identical type to the dictionary type's "
                                "value_type")
            dictionary_ = dictionary
        else:
            dictionary_ = array(dictionary, type=type_.value_type)

        value.index = pyarrow_unwrap_scalar(index_)
        value.dictionary = pyarrow_unwrap_array(dictionary_)

        wrapped = make_shared[CDictionaryScalar](
            value, pyarrow_unwrap_data_type(type_), <c_bool>(is_valid)
        )
        return Scalar.wrap(<shared_ptr[CScalar]> wrapped)

    def __reduce__(self):
        return DictionaryScalar._reconstruct, (
            self.type, self.is_valid, self.index, self.dictionary
        )

    @property
    def index(self):
        """
        Return this value's underlying index as a scalar.
        """
        cdef CDictionaryScalar* sp = <CDictionaryScalar*> self.wrapped.get()
        return Scalar.wrap(sp.value.index)

    @property
    def value(self):
        """
        Return the encoded value as a scalar.
        """
        cdef CDictionaryScalar* sp = <CDictionaryScalar*> self.wrapped.get()
        return Scalar.wrap(GetResultValue(sp.GetEncodedValue()))

    @property
    def dictionary(self):
        cdef CDictionaryScalar* sp = <CDictionaryScalar*> self.wrapped.get()
        return pyarrow_wrap_array(sp.value.dictionary)

    def as_py(self):
        """
        Return this encoded value as a Python object.
        """
        return self.value.as_py() if self.is_valid else None


cdef class UnionScalar(Scalar):
    """
    Concrete class for Union scalars.
    """

    @property
    def value(self):
        """
        Return underlying value as a scalar.
        """
        cdef CSparseUnionScalar* sp
        cdef CDenseUnionScalar* dp
        if self.type.id == _Type_SPARSE_UNION:
            sp = <CSparseUnionScalar*> self.wrapped.get()
            return Scalar.wrap(sp.value[sp.child_id]) if sp.is_valid else None
        else:
            dp = <CDenseUnionScalar*> self.wrapped.get()
            return Scalar.wrap(dp.value) if dp.is_valid else None

    def as_py(self):
        """
        Return underlying value as a Python object.
        """
        value = self.value
        return None if value is None else value.as_py()

    @property
    def type_code(self):
        """
        Return the union type code for this scalar.
        """
        cdef CUnionScalar* sp = <CUnionScalar*> self.wrapped.get()
        return sp.type_code


cdef class ExtensionScalar(Scalar):
    """
    Concrete class for Extension scalars.
    """

    @property
    def value(self):
        """
        Return storage value as a scalar.
        """
        cdef CExtensionScalar* sp = <CExtensionScalar*> self.wrapped.get()
        return Scalar.wrap(sp.value) if sp.is_valid else None

    def as_py(self):
        """
        Return this scalar as a Python object.
        """
        return None if self.value is None else self.value.as_py()

    @staticmethod
    def from_storage(BaseExtensionType typ, value):
        """
        Construct ExtensionScalar from type and storage value.

        Parameters
        ----------
        typ : DataType
            The extension type for the result scalar.
        value : object
            The storage value for the result scalar.

        Returns
        -------
        ext_scalar : ExtensionScalar
        """
        cdef:
            shared_ptr[CExtensionScalar] sp_scalar
            shared_ptr[CScalar] sp_storage
            CExtensionScalar* ext_scalar

        if value is None:
            storage = None
        elif isinstance(value, Scalar):
            if value.type != typ.storage_type:
                raise TypeError("Incompatible storage type {0} "
                                "for extension type {1}"
                                .format(value.type, typ))
            storage = value
        else:
            storage = scalar(value, typ.storage_type)

        cdef c_bool is_valid = storage is not None and storage.is_valid
        if is_valid:
            sp_storage = pyarrow_unwrap_scalar(storage)
        else:
            sp_storage = MakeNullScalar((<DataType> typ.storage_type).sp_type)
        sp_scalar = make_shared[CExtensionScalar](sp_storage, typ.sp_type,
                                                  is_valid)
        with nogil:
            check_status(sp_scalar.get().Validate())
        return pyarrow_wrap_scalar(<shared_ptr[CScalar]> sp_scalar)


cdef dict _scalar_classes = {
    _Type_BOOL: BooleanScalar,
    _Type_UINT8: UInt8Scalar,
    _Type_UINT16: UInt16Scalar,
    _Type_UINT32: UInt32Scalar,
    _Type_UINT64: UInt64Scalar,
    _Type_INT8: Int8Scalar,
    _Type_INT16: Int16Scalar,
    _Type_INT32: Int32Scalar,
    _Type_INT64: Int64Scalar,
    _Type_HALF_FLOAT: HalfFloatScalar,
    _Type_FLOAT: FloatScalar,
    _Type_DOUBLE: DoubleScalar,
    _Type_DECIMAL128: Decimal128Scalar,
    _Type_DECIMAL256: Decimal256Scalar,
    _Type_DATE32: Date32Scalar,
    _Type_DATE64: Date64Scalar,
    _Type_TIME32: Time32Scalar,
    _Type_TIME64: Time64Scalar,
    _Type_TIMESTAMP: TimestampScalar,
    _Type_DURATION: DurationScalar,
    _Type_BINARY: BinaryScalar,
    _Type_LARGE_BINARY: LargeBinaryScalar,
    _Type_FIXED_SIZE_BINARY: FixedSizeBinaryScalar,
    _Type_STRING: StringScalar,
    _Type_LARGE_STRING: LargeStringScalar,
    _Type_LIST: ListScalar,
    _Type_LARGE_LIST: LargeListScalar,
    _Type_FIXED_SIZE_LIST: FixedSizeListScalar,
    _Type_STRUCT: StructScalar,
    _Type_MAP: MapScalar,
    _Type_DICTIONARY: DictionaryScalar,
    _Type_SPARSE_UNION: UnionScalar,
    _Type_DENSE_UNION: UnionScalar,
    _Type_INTERVAL_MONTH_DAY_NANO: MonthDayNanoIntervalScalar,
    _Type_EXTENSION: ExtensionScalar,
}


cdef object get_scalar_class_from_type(
        const shared_ptr[CDataType]& sp_data_type):
    cdef CDataType* data_type = sp_data_type.get()
    if data_type == NULL:
        raise ValueError('Scalar data type was NULL')

    if data_type.id() == _Type_EXTENSION:
        py_ext_data_type = pyarrow_wrap_data_type(sp_data_type)
        return py_ext_data_type.__arrow_ext_scalar_class__()
    else:
        return _scalar_classes[data_type.id()]


def scalar(value, type=None, *, from_pandas=None, MemoryPool memory_pool=None):
    """
    Create a pyarrow.Scalar instance from a Python object.

    Parameters
    ----------
    value : Any
        Python object coercible to arrow's type system.
    type : pyarrow.DataType
        Explicit type to attempt to coerce to, otherwise will be inferred from
        the value.
    from_pandas : bool, default None
        Use pandas's semantics for inferring nulls from values in
        ndarray-like data. Defaults to False if not passed explicitly by user,
        or True if a pandas object is passed in.
    memory_pool : pyarrow.MemoryPool, optional
        If not passed, will allocate memory from the currently-set default
        memory pool.

    Returns
    -------
    scalar : pyarrow.Scalar

    Examples
    --------
    >>> import pyarrow as pa

    >>> pa.scalar(42)
    <pyarrow.Int64Scalar: 42>

    >>> pa.scalar("string")
    <pyarrow.StringScalar: 'string'>

    >>> pa.scalar([1, 2])
    <pyarrow.ListScalar: [1, 2]>

    >>> pa.scalar([1, 2], type=pa.list_(pa.int16()))
    <pyarrow.ListScalar: [1, 2]>
    """
    cdef:
        DataType ty
        PyConversionOptions options
        shared_ptr[CScalar] scalar
        shared_ptr[CArray] array
        shared_ptr[CChunkedArray] chunked
        bint is_pandas_object = False
        CMemoryPool* pool

    type = ensure_type(type, allow_none=True)
    pool = maybe_unbox_memory_pool(memory_pool)

    if _is_array_like(value):
        value = get_values(value, &is_pandas_object)

    options.size = 1

    if type is not None:
        ty = ensure_type(type)
        options.type = ty.sp_type

    if from_pandas is None:
        options.from_pandas = is_pandas_object
    else:
        options.from_pandas = from_pandas

    value = [value]
    with nogil:
        chunked = GetResultValue(ConvertPySequence(value, None, options, pool))

    # get the first chunk
    assert chunked.get().num_chunks() == 1
    array = chunked.get().chunk(0)

    # retrieve the scalar from the first position
    scalar = GetResultValue(array.get().GetScalar(0))
    return Scalar.wrap(scalar)

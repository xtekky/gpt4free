from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
)
import operator
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Sequence,
    TypeVar,
    Union,
    cast,
    final,
    overload,
)
import warnings

import numpy as np

from pandas._libs import (
    algos,
    lib,
)
from pandas._libs.arrays import NDArrayBacked
from pandas._libs.tslibs import (
    BaseOffset,
    IncompatibleFrequency,
    NaT,
    NaTType,
    Period,
    Resolution,
    Tick,
    Timestamp,
    delta_to_nanoseconds,
    get_unit_from_dtype,
    iNaT,
    ints_to_pydatetime,
    ints_to_pytimedelta,
    to_offset,
)
from pandas._libs.tslibs.fields import (
    RoundTo,
    round_nsint64,
)
from pandas._libs.tslibs.np_datetime import compare_mismatched_resolutions
from pandas._libs.tslibs.timestamps import integer_op_not_supported
from pandas._typing import (
    ArrayLike,
    DatetimeLikeScalar,
    Dtype,
    DtypeObj,
    NpDtype,
    PositionalIndexer2D,
    PositionalIndexerTuple,
    ScalarIndexer,
    SequenceIndexer,
    npt,
)
from pandas.compat.numpy import function as nv
from pandas.errors import (
    AbstractMethodError,
    NullFrequencyError,
    PerformanceWarning,
)
from pandas.util._decorators import (
    Appender,
    Substitution,
    cache_readonly,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    is_all_strings,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_datetime64_dtype,
    is_datetime64tz_dtype,
    is_datetime_or_timedelta_dtype,
    is_dtype_equal,
    is_float_dtype,
    is_integer_dtype,
    is_list_like,
    is_object_dtype,
    is_period_dtype,
    is_string_dtype,
    is_timedelta64_dtype,
    is_unsigned_integer_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    ExtensionDtype,
)
from pandas.core.dtypes.generic import (
    ABCCategorical,
    ABCMultiIndex,
)
from pandas.core.dtypes.missing import (
    is_valid_na_for_dtype,
    isna,
)

from pandas.core import (
    nanops,
    ops,
)
from pandas.core.algorithms import (
    checked_add_with_arr,
    isin,
    mode,
    unique1d,
)
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._mixins import (
    NDArrayBackedExtensionArray,
    ravel_compat,
)
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.integer import IntegerArray
import pandas.core.common as com
from pandas.core.construction import (
    array as pd_array,
    ensure_wrapped_if_datetimelike,
    extract_array,
)
from pandas.core.indexers import (
    check_array_indexer,
    check_setitem_lengths,
)
from pandas.core.ops.common import unpack_zerodim_and_defer
from pandas.core.ops.invalid import (
    invalid_comparison,
    make_invalid_op,
)

from pandas.tseries import frequencies

if TYPE_CHECKING:

    from pandas.core.arrays import (
        DatetimeArray,
        PeriodArray,
        TimedeltaArray,
    )

DTScalarOrNaT = Union[DatetimeLikeScalar, NaTType]
DatetimeLikeArrayT = TypeVar("DatetimeLikeArrayT", bound="DatetimeLikeArrayMixin")


class InvalidComparison(Exception):
    """
    Raised by _validate_comparison_value to indicate to caller it should
    return invalid_comparison.
    """

    pass


class DatetimeLikeArrayMixin(OpsMixin, NDArrayBackedExtensionArray):
    """
    Shared Base/Mixin class for DatetimeArray, TimedeltaArray, PeriodArray

    Assumes that __new__/__init__ defines:
        _data
        _freq

    and that the inheriting class has methods:
        _generate_range
    """

    # _infer_matches -> which infer_dtype strings are close enough to our own
    _infer_matches: tuple[str, ...]
    _is_recognized_dtype: Callable[[DtypeObj], bool]
    _recognized_scalars: tuple[type, ...]
    _ndarray: np.ndarray

    @cache_readonly
    def _can_hold_na(self) -> bool:
        return True

    def __init__(self, data, dtype: Dtype | None = None, freq=None, copy=False) -> None:
        raise AbstractMethodError(self)

    @property
    def _scalar_type(self) -> type[DatetimeLikeScalar]:
        """
        The scalar associated with this datelike

        * PeriodArray : Period
        * DatetimeArray : Timestamp
        * TimedeltaArray : Timedelta
        """
        raise AbstractMethodError(self)

    def _scalar_from_string(self, value: str) -> DTScalarOrNaT:
        """
        Construct a scalar type from a string.

        Parameters
        ----------
        value : str

        Returns
        -------
        Period, Timestamp, or Timedelta, or NaT
            Whatever the type of ``self._scalar_type`` is.

        Notes
        -----
        This should call ``self._check_compatible_with`` before
        unboxing the result.
        """
        raise AbstractMethodError(self)

    def _unbox_scalar(
        self, value: DTScalarOrNaT, setitem: bool = False
    ) -> np.int64 | np.datetime64 | np.timedelta64:
        """
        Unbox the integer value of a scalar `value`.

        Parameters
        ----------
        value : Period, Timestamp, Timedelta, or NaT
            Depending on subclass.
        setitem : bool, default False
            Whether to check compatibility with setitem strictness.

        Returns
        -------
        int

        Examples
        --------
        >>> self._unbox_scalar(Timedelta("10s"))  # doctest: +SKIP
        10000000000
        """
        raise AbstractMethodError(self)

    def _check_compatible_with(
        self, other: DTScalarOrNaT, setitem: bool = False
    ) -> None:
        """
        Verify that `self` and `other` are compatible.

        * DatetimeArray verifies that the timezones (if any) match
        * PeriodArray verifies that the freq matches
        * Timedelta has no verification

        In each case, NaT is considered compatible.

        Parameters
        ----------
        other
        setitem : bool, default False
            For __setitem__ we may have stricter compatibility restrictions than
            for comparisons.

        Raises
        ------
        Exception
        """
        raise AbstractMethodError(self)

    # ------------------------------------------------------------------
    # NDArrayBackedExtensionArray compat

    @cache_readonly
    def _data(self) -> np.ndarray:
        return self._ndarray

    # ------------------------------------------------------------------

    def _box_func(self, x):
        """
        box function to get object from internal representation
        """
        raise AbstractMethodError(self)

    def _box_values(self, values) -> np.ndarray:
        """
        apply box func to passed values
        """
        return lib.map_infer(values, self._box_func, convert=False)

    def __iter__(self):
        if self.ndim > 1:
            return (self[n] for n in range(len(self)))
        else:
            return (self._box_func(v) for v in self.asi8)

    @property
    def asi8(self) -> npt.NDArray[np.int64]:
        """
        Integer representation of the values.

        Returns
        -------
        ndarray
            An ndarray with int64 dtype.
        """
        # do not cache or you'll create a memory leak
        return self._ndarray.view("i8")

    # ----------------------------------------------------------------
    # Rendering Methods

    def _format_native_types(
        self, *, na_rep="NaT", date_format=None
    ) -> npt.NDArray[np.object_]:
        """
        Helper method for astype when converting to strings.

        Returns
        -------
        ndarray[str]
        """
        raise AbstractMethodError(self)

    def _formatter(self, boxed: bool = False):
        # TODO: Remove Datetime & DatetimeTZ formatters.
        return "'{}'".format

    # ----------------------------------------------------------------
    # Array-Like / EA-Interface Methods

    def __array__(self, dtype: NpDtype | None = None) -> np.ndarray:
        # used for Timedelta/DatetimeArray, overwritten by PeriodArray
        if is_object_dtype(dtype):
            return np.array(list(self), dtype=object)
        return self._ndarray

    @overload
    def __getitem__(self, item: ScalarIndexer) -> DTScalarOrNaT:
        ...

    @overload
    def __getitem__(
        self: DatetimeLikeArrayT,
        item: SequenceIndexer | PositionalIndexerTuple,
    ) -> DatetimeLikeArrayT:
        ...

    def __getitem__(
        self: DatetimeLikeArrayT, key: PositionalIndexer2D
    ) -> DatetimeLikeArrayT | DTScalarOrNaT:
        """
        This getitem defers to the underlying array, which by-definition can
        only handle list-likes, slices, and integer scalars
        """
        # Use cast as we know we will get back a DatetimeLikeArray or DTScalar,
        # but skip evaluating the Union at runtime for performance
        # (see https://github.com/pandas-dev/pandas/pull/44624)
        result = cast(
            "Union[DatetimeLikeArrayT, DTScalarOrNaT]", super().__getitem__(key)
        )
        if lib.is_scalar(result):
            return result
        else:
            # At this point we know the result is an array.
            result = cast(DatetimeLikeArrayT, result)
        result._freq = self._get_getitem_freq(key)
        return result

    def _get_getitem_freq(self, key) -> BaseOffset | None:
        """
        Find the `freq` attribute to assign to the result of a __getitem__ lookup.
        """
        is_period = is_period_dtype(self.dtype)
        if is_period:
            freq = self.freq
        elif self.ndim != 1:
            freq = None
        else:
            key = check_array_indexer(self, key)  # maybe ndarray[bool] -> slice
            freq = None
            if isinstance(key, slice):
                if self.freq is not None and key.step is not None:
                    freq = key.step * self.freq
                else:
                    freq = self.freq
            elif key is Ellipsis:
                # GH#21282 indexing with Ellipsis is similar to a full slice,
                #  should preserve `freq` attribute
                freq = self.freq
            elif com.is_bool_indexer(key):
                new_key = lib.maybe_booleans_to_slice(key.view(np.uint8))
                if isinstance(new_key, slice):
                    return self._get_getitem_freq(new_key)
        return freq

    # error: Argument 1 of "__setitem__" is incompatible with supertype
    # "ExtensionArray"; supertype defines the argument type as "Union[int,
    # ndarray]"
    def __setitem__(  # type: ignore[override]
        self,
        key: int | Sequence[int] | Sequence[bool] | slice,
        value: NaTType | Any | Sequence[Any],
    ) -> None:
        # I'm fudging the types a bit here. "Any" above really depends
        # on type(self). For PeriodArray, it's Period (or stuff coercible
        # to a period in from_sequence). For DatetimeArray, it's Timestamp...
        # I don't know if mypy can do that, possibly with Generics.
        # https://mypy.readthedocs.io/en/latest/generics.html

        no_op = check_setitem_lengths(key, value, self)

        # Calling super() before the no_op short-circuit means that we raise
        #  on invalid 'value' even if this is a no-op, e.g. wrong-dtype empty array.
        super().__setitem__(key, value)

        if no_op:
            return

        self._maybe_clear_freq()

    def _maybe_clear_freq(self):
        # inplace operations like __setitem__ may invalidate the freq of
        # DatetimeArray and TimedeltaArray
        pass

    def astype(self, dtype, copy: bool = True):
        # Some notes on cases we don't have to handle here in the base class:
        #   1. PeriodArray.astype handles period -> period
        #   2. DatetimeArray.astype handles conversion between tz.
        #   3. DatetimeArray.astype handles datetime -> period
        dtype = pandas_dtype(dtype)

        if is_object_dtype(dtype):
            if self.dtype.kind == "M":
                self = cast("DatetimeArray", self)
                # *much* faster than self._box_values
                #  for e.g. test_get_loc_tuple_monotonic_above_size_cutoff
                i8data = self.asi8
                converted = ints_to_pydatetime(
                    i8data,
                    tz=self.tz,
                    freq=self.freq,
                    box="timestamp",
                    reso=self._reso,
                )
                return converted

            elif self.dtype.kind == "m":
                return ints_to_pytimedelta(self._ndarray, box=True)

            return self._box_values(self.asi8.ravel()).reshape(self.shape)

        elif isinstance(dtype, ExtensionDtype):
            return super().astype(dtype, copy=copy)
        elif is_string_dtype(dtype):
            return self._format_native_types()
        elif is_integer_dtype(dtype):
            # we deliberately ignore int32 vs. int64 here.
            # See https://github.com/pandas-dev/pandas/issues/24381 for more.
            values = self.asi8

            if is_unsigned_integer_dtype(dtype):
                # Again, we ignore int32 vs. int64
                values = values.view("uint64")
                if dtype != np.uint64:
                    # GH#45034
                    warnings.warn(
                        f"The behavior of .astype from {self.dtype} to {dtype} is "
                        "deprecated. In a future version, this astype will return "
                        "exactly the specified dtype instead of uint64, and will "
                        "raise if that conversion overflows.",
                        FutureWarning,
                        stacklevel=find_stack_level(),
                    )
                elif (self.asi8 < 0).any():
                    # GH#45034
                    warnings.warn(
                        f"The behavior of .astype from {self.dtype} to {dtype} is "
                        "deprecated. In a future version, this astype will "
                        "raise if the conversion overflows, as it did in this "
                        "case with negative int64 values.",
                        FutureWarning,
                        stacklevel=find_stack_level(),
                    )
            elif dtype != np.int64:
                # GH#45034
                warnings.warn(
                    f"The behavior of .astype from {self.dtype} to {dtype} is "
                    "deprecated. In a future version, this astype will return "
                    "exactly the specified dtype instead of int64, and will "
                    "raise if that conversion overflows.",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )

            if copy:
                values = values.copy()
            return values
        elif (
            is_datetime_or_timedelta_dtype(dtype)
            and not is_dtype_equal(self.dtype, dtype)
        ) or is_float_dtype(dtype):
            # disallow conversion between datetime/timedelta,
            # and conversions for any datetimelike to float
            msg = f"Cannot cast {type(self).__name__} to dtype {dtype}"
            raise TypeError(msg)
        else:
            return np.asarray(self, dtype=dtype)

    @overload
    def view(self: DatetimeLikeArrayT) -> DatetimeLikeArrayT:
        ...

    @overload
    def view(self, dtype: Literal["M8[ns]"]) -> DatetimeArray:
        ...

    @overload
    def view(self, dtype: Literal["m8[ns]"]) -> TimedeltaArray:
        ...

    @overload
    def view(self, dtype: Dtype | None = ...) -> ArrayLike:
        ...

    def view(self, dtype: Dtype | None = None) -> ArrayLike:
        # we need to explicitly call super() method as long as the `@overload`s
        #  are present in this file.
        return super().view(dtype)

    # ------------------------------------------------------------------
    # ExtensionArray Interface

    @classmethod
    def _concat_same_type(
        cls: type[DatetimeLikeArrayT],
        to_concat: Sequence[DatetimeLikeArrayT],
        axis: int = 0,
    ) -> DatetimeLikeArrayT:
        new_obj = super()._concat_same_type(to_concat, axis)

        obj = to_concat[0]
        dtype = obj.dtype

        new_freq = None
        if is_period_dtype(dtype):
            new_freq = obj.freq
        elif axis == 0:
            # GH 3232: If the concat result is evenly spaced, we can retain the
            # original frequency
            to_concat = [x for x in to_concat if len(x)]

            if obj.freq is not None and all(x.freq == obj.freq for x in to_concat):
                pairs = zip(to_concat[:-1], to_concat[1:])
                if all(pair[0][-1] + obj.freq == pair[1][0] for pair in pairs):
                    new_freq = obj.freq

        new_obj._freq = new_freq
        return new_obj

    def copy(self: DatetimeLikeArrayT, order="C") -> DatetimeLikeArrayT:
        # error: Unexpected keyword argument "order" for "copy"
        new_obj = super().copy(order=order)  # type: ignore[call-arg]
        new_obj._freq = self.freq
        return new_obj

    # ------------------------------------------------------------------
    # Validation Methods
    # TODO: try to de-duplicate these, ensure identical behavior

    def _validate_comparison_value(self, other):
        if isinstance(other, str):
            try:
                # GH#18435 strings get a pass from tzawareness compat
                other = self._scalar_from_string(other)
            except (ValueError, IncompatibleFrequency):
                # failed to parse as Timestamp/Timedelta/Period
                raise InvalidComparison(other)

        if isinstance(other, self._recognized_scalars) or other is NaT:
            other = self._scalar_type(other)
            try:
                self._check_compatible_with(other)
            except (TypeError, IncompatibleFrequency) as err:
                # e.g. tzawareness mismatch
                raise InvalidComparison(other) from err

        elif not is_list_like(other):
            raise InvalidComparison(other)

        elif len(other) != len(self):
            raise ValueError("Lengths must match")

        else:
            try:
                other = self._validate_listlike(other, allow_object=True)
                self._check_compatible_with(other)
            except (TypeError, IncompatibleFrequency) as err:
                if is_object_dtype(getattr(other, "dtype", None)):
                    # We will have to operate element-wise
                    pass
                else:
                    raise InvalidComparison(other) from err

        return other

    def _validate_shift_value(self, fill_value):
        # TODO(2.0): once this deprecation is enforced, use _validate_scalar
        if is_valid_na_for_dtype(fill_value, self.dtype):
            fill_value = NaT
        elif isinstance(fill_value, self._recognized_scalars):
            fill_value = self._scalar_type(fill_value)
        else:
            new_fill: DatetimeLikeScalar

            # only warn if we're not going to raise
            if self._scalar_type is Period and lib.is_integer(fill_value):
                # kludge for #31971 since Period(integer) tries to cast to str
                new_fill = Period._from_ordinal(fill_value, freq=self.freq)
            else:
                new_fill = self._scalar_type(fill_value)

            # stacklevel here is chosen to be correct when called from
            #  DataFrame.shift or Series.shift
            warnings.warn(
                f"Passing {type(fill_value)} to shift is deprecated and "
                "will raise in a future version, pass "
                f"{self._scalar_type.__name__} instead.",
                FutureWarning,
                # There is no way to hard-code the level since this might be
                #  reached directly or called from the Index or Block method
                stacklevel=find_stack_level(),
            )
            fill_value = new_fill

        return self._unbox(fill_value, setitem=True)

    def _validate_scalar(
        self,
        value,
        *,
        allow_listlike: bool = False,
        setitem: bool = True,
        unbox: bool = True,
    ):
        """
        Validate that the input value can be cast to our scalar_type.

        Parameters
        ----------
        value : object
        allow_listlike: bool, default False
            When raising an exception, whether the message should say
            listlike inputs are allowed.
        setitem : bool, default True
            Whether to check compatibility with setitem strictness.
        unbox : bool, default True
            Whether to unbox the result before returning.  Note: unbox=False
            skips the setitem compatibility check.

        Returns
        -------
        self._scalar_type or NaT
        """
        if isinstance(value, self._scalar_type):
            pass

        elif isinstance(value, str):
            # NB: Careful about tzawareness
            try:
                value = self._scalar_from_string(value)
            except ValueError as err:
                msg = self._validation_error_message(value, allow_listlike)
                raise TypeError(msg) from err

        elif is_valid_na_for_dtype(value, self.dtype):
            # GH#18295
            value = NaT

        elif isna(value):
            # if we are dt64tz and value is dt64("NaT"), dont cast to NaT,
            #  or else we'll fail to raise in _unbox_scalar
            msg = self._validation_error_message(value, allow_listlike)
            raise TypeError(msg)

        elif isinstance(value, self._recognized_scalars):
            value = self._scalar_type(value)

        else:
            msg = self._validation_error_message(value, allow_listlike)
            raise TypeError(msg)

        if not unbox:
            # NB: In general NDArrayBackedExtensionArray will unbox here;
            #  this option exists to prevent a performance hit in
            #  TimedeltaIndex.get_loc
            return value
        return self._unbox_scalar(value, setitem=setitem)

    def _validation_error_message(self, value, allow_listlike: bool = False) -> str:
        """
        Construct an exception message on validation error.

        Some methods allow only scalar inputs, while others allow either scalar
        or listlike.

        Parameters
        ----------
        allow_listlike: bool, default False

        Returns
        -------
        str
        """
        if allow_listlike:
            msg = (
                f"value should be a '{self._scalar_type.__name__}', 'NaT', "
                f"or array of those. Got '{type(value).__name__}' instead."
            )
        else:
            msg = (
                f"value should be a '{self._scalar_type.__name__}' or 'NaT'. "
                f"Got '{type(value).__name__}' instead."
            )
        return msg

    def _validate_listlike(self, value, allow_object: bool = False):
        if isinstance(value, type(self)):
            return value

        if isinstance(value, list) and len(value) == 0:
            # We treat empty list as our own dtype.
            return type(self)._from_sequence([], dtype=self.dtype)

        if hasattr(value, "dtype") and value.dtype == object:
            # `array` below won't do inference if value is an Index or Series.
            #  so do so here.  in the Index case, inferred_type may be cached.
            if lib.infer_dtype(value) in self._infer_matches:
                try:
                    value = type(self)._from_sequence(value)
                except (ValueError, TypeError):
                    if allow_object:
                        return value
                    msg = self._validation_error_message(value, True)
                    raise TypeError(msg)

        # Do type inference if necessary up front (after unpacking PandasArray)
        # e.g. we passed PeriodIndex.values and got an ndarray of Periods
        value = extract_array(value, extract_numpy=True)
        value = pd_array(value)
        value = extract_array(value, extract_numpy=True)

        if is_all_strings(value):
            # We got a StringArray
            try:
                # TODO: Could use from_sequence_of_strings if implemented
                # Note: passing dtype is necessary for PeriodArray tests
                value = type(self)._from_sequence(value, dtype=self.dtype)
            except ValueError:
                pass

        if is_categorical_dtype(value.dtype):
            # e.g. we have a Categorical holding self.dtype
            if is_dtype_equal(value.categories.dtype, self.dtype):
                # TODO: do we need equal dtype or just comparable?
                value = value._internal_get_values()
                value = extract_array(value, extract_numpy=True)

        if allow_object and is_object_dtype(value.dtype):
            pass

        elif not type(self)._is_recognized_dtype(value.dtype):
            msg = self._validation_error_message(value, True)
            raise TypeError(msg)

        return value

    def _validate_searchsorted_value(self, value):
        if not is_list_like(value):
            return self._validate_scalar(value, allow_listlike=True, setitem=False)
        else:
            value = self._validate_listlike(value)

        return self._unbox(value)

    def _validate_setitem_value(self, value):
        if is_list_like(value):
            value = self._validate_listlike(value)
        else:
            return self._validate_scalar(value, allow_listlike=True)

        return self._unbox(value, setitem=True)

    def _unbox(
        self, other, setitem: bool = False
    ) -> np.int64 | np.datetime64 | np.timedelta64 | np.ndarray:
        """
        Unbox either a scalar with _unbox_scalar or an instance of our own type.
        """
        if lib.is_scalar(other):
            other = self._unbox_scalar(other, setitem=setitem)
        else:
            # same type as self
            self._check_compatible_with(other, setitem=setitem)
            other = other._ndarray
        return other

    # ------------------------------------------------------------------
    # Additional array methods
    #  These are not part of the EA API, but we implement them because
    #  pandas assumes they're there.

    @ravel_compat
    def map(self, mapper):
        # TODO(GH-23179): Add ExtensionArray.map
        # Need to figure out if we want ExtensionArray.map first.
        # If so, then we can refactor IndexOpsMixin._map_values to
        # a standalone function and call from here..
        # Else, just rewrite _map_infer_values to do the right thing.
        from pandas import Index

        return Index(self).map(mapper).array

    def isin(self, values) -> npt.NDArray[np.bool_]:
        """
        Compute boolean array of whether each value is found in the
        passed set of values.

        Parameters
        ----------
        values : set or sequence of values

        Returns
        -------
        ndarray[bool]
        """
        if not hasattr(values, "dtype"):
            values = np.asarray(values)

        if values.dtype.kind in ["f", "i", "u", "c"]:
            # TODO: de-duplicate with equals, validate_comparison_value
            return np.zeros(self.shape, dtype=bool)

        if not isinstance(values, type(self)):
            inferable = [
                "timedelta",
                "timedelta64",
                "datetime",
                "datetime64",
                "date",
                "period",
            ]
            if values.dtype == object:
                inferred = lib.infer_dtype(values, skipna=False)
                if inferred not in inferable:
                    if inferred == "string":
                        pass

                    elif "mixed" in inferred:
                        return isin(self.astype(object), values)
                    else:
                        return np.zeros(self.shape, dtype=bool)

            try:
                values = type(self)._from_sequence(values)
            except ValueError:
                return isin(self.astype(object), values)

        try:
            self._check_compatible_with(values)
        except (TypeError, ValueError):
            # Includes tzawareness mismatch and IncompatibleFrequencyError
            return np.zeros(self.shape, dtype=bool)

        return isin(self.asi8, values.asi8)

    # ------------------------------------------------------------------
    # Null Handling

    def isna(self) -> npt.NDArray[np.bool_]:
        return self._isnan

    @property  # NB: override with cache_readonly in immutable subclasses
    def _isnan(self) -> npt.NDArray[np.bool_]:
        """
        return if each value is nan
        """
        return self.asi8 == iNaT

    @property  # NB: override with cache_readonly in immutable subclasses
    def _hasna(self) -> bool:
        """
        return if I have any nans; enables various perf speedups
        """
        return bool(self._isnan.any())

    def _maybe_mask_results(
        self, result: np.ndarray, fill_value=iNaT, convert=None
    ) -> np.ndarray:
        """
        Parameters
        ----------
        result : np.ndarray
        fill_value : object, default iNaT
        convert : str, dtype or None

        Returns
        -------
        result : ndarray with values replace by the fill_value

        mask the result if needed, convert to the provided dtype if its not
        None

        This is an internal routine.
        """
        if self._hasna:
            if convert:
                result = result.astype(convert)
            if fill_value is None:
                fill_value = np.nan
            np.putmask(result, self._isnan, fill_value)
        return result

    # ------------------------------------------------------------------
    # Frequency Properties/Methods

    @property
    def freq(self):
        """
        Return the frequency object if it is set, otherwise None.
        """
        return self._freq

    @freq.setter
    def freq(self, value) -> None:
        if value is not None:
            value = to_offset(value)
            self._validate_frequency(self, value)

            if self.ndim > 1:
                raise ValueError("Cannot set freq with ndim > 1")

        self._freq = value

    @property
    def freqstr(self) -> str | None:
        """
        Return the frequency object as a string if its set, otherwise None.
        """
        if self.freq is None:
            return None
        return self.freq.freqstr

    @property  # NB: override with cache_readonly in immutable subclasses
    def inferred_freq(self) -> str | None:
        """
        Tries to return a string representing a frequency generated by infer_freq.

        Returns None if it can't autodetect the frequency.
        """
        if self.ndim != 1:
            return None
        try:
            return frequencies.infer_freq(self)
        except ValueError:
            return None

    @property  # NB: override with cache_readonly in immutable subclasses
    def _resolution_obj(self) -> Resolution | None:
        freqstr = self.freqstr
        if freqstr is None:
            return None
        try:
            return Resolution.get_reso_from_freqstr(freqstr)
        except KeyError:
            return None

    @property  # NB: override with cache_readonly in immutable subclasses
    def resolution(self) -> str:
        """
        Returns day, hour, minute, second, millisecond or microsecond
        """
        # error: Item "None" of "Optional[Any]" has no attribute "attrname"
        return self._resolution_obj.attrname  # type: ignore[union-attr]

    @classmethod
    def _validate_frequency(cls, index, freq, **kwargs):
        """
        Validate that a frequency is compatible with the values of a given
        Datetime Array/Index or Timedelta Array/Index

        Parameters
        ----------
        index : DatetimeIndex or TimedeltaIndex
            The index on which to determine if the given frequency is valid
        freq : DateOffset
            The frequency to validate
        """
        # TODO: this is not applicable to PeriodArray, move to correct Mixin
        inferred = index.inferred_freq
        if index.size == 0 or inferred == freq.freqstr:
            return None

        try:
            on_freq = cls._generate_range(
                start=index[0], end=None, periods=len(index), freq=freq, **kwargs
            )
            if not np.array_equal(index.asi8, on_freq.asi8):
                raise ValueError
        except ValueError as e:
            if "non-fixed" in str(e):
                # non-fixed frequencies are not meaningful for timedelta64;
                #  we retain that error message
                raise e
            # GH#11587 the main way this is reached is if the `np.array_equal`
            #  check above is False.  This can also be reached if index[0]
            #  is `NaT`, in which case the call to `cls._generate_range` will
            #  raise a ValueError, which we re-raise with a more targeted
            #  message.
            raise ValueError(
                f"Inferred frequency {inferred} from passed values "
                f"does not conform to passed frequency {freq.freqstr}"
            ) from e

    @classmethod
    def _generate_range(
        cls: type[DatetimeLikeArrayT], start, end, periods, freq, *args, **kwargs
    ) -> DatetimeLikeArrayT:
        raise AbstractMethodError(cls)

    # monotonicity/uniqueness properties are called via frequencies.infer_freq,
    #  see GH#23789

    @property
    def _is_monotonic_increasing(self) -> bool:
        return algos.is_monotonic(self.asi8, timelike=True)[0]

    @property
    def _is_monotonic_decreasing(self) -> bool:
        return algos.is_monotonic(self.asi8, timelike=True)[1]

    @property
    def _is_unique(self) -> bool:
        return len(unique1d(self.asi8.ravel("K"))) == self.size

    # ------------------------------------------------------------------
    # Arithmetic Methods

    def _cmp_method(self, other, op):
        if self.ndim > 1 and getattr(other, "shape", None) == self.shape:
            # TODO: handle 2D-like listlikes
            return op(self.ravel(), other.ravel()).reshape(self.shape)

        try:
            other = self._validate_comparison_value(other)
        except InvalidComparison:
            return invalid_comparison(self, other, op)

        dtype = getattr(other, "dtype", None)
        if is_object_dtype(dtype):
            # We have to use comp_method_OBJECT_ARRAY instead of numpy
            #  comparison otherwise it would fail to raise when
            #  comparing tz-aware and tz-naive
            with np.errstate(all="ignore"):
                result = ops.comp_method_OBJECT_ARRAY(
                    op, np.asarray(self.astype(object)), other
                )
            return result

        if other is NaT:
            if op is operator.ne:
                result = np.ones(self.shape, dtype=bool)
            else:
                result = np.zeros(self.shape, dtype=bool)
            return result

        if not is_period_dtype(self.dtype):
            self = cast(TimelikeOps, self)
            if self._reso != other._reso:
                if not isinstance(other, type(self)):
                    # i.e. Timedelta/Timestamp, cast to ndarray and let
                    #  compare_mismatched_resolutions handle broadcasting
                    other_arr = np.array(other.asm8)
                else:
                    other_arr = other._ndarray
                return compare_mismatched_resolutions(self._ndarray, other_arr, op)

        other_vals = self._unbox(other)
        # GH#37462 comparison on i8 values is almost 2x faster than M8/m8
        result = op(self._ndarray.view("i8"), other_vals.view("i8"))

        o_mask = isna(other)
        mask = self._isnan | o_mask
        if mask.any():
            nat_result = op is operator.ne
            np.putmask(result, mask, nat_result)

        return result

    # pow is invalid for all three subclasses; TimedeltaArray will override
    #  the multiplication and division ops
    __pow__ = make_invalid_op("__pow__")
    __rpow__ = make_invalid_op("__rpow__")
    __mul__ = make_invalid_op("__mul__")
    __rmul__ = make_invalid_op("__rmul__")
    __truediv__ = make_invalid_op("__truediv__")
    __rtruediv__ = make_invalid_op("__rtruediv__")
    __floordiv__ = make_invalid_op("__floordiv__")
    __rfloordiv__ = make_invalid_op("__rfloordiv__")
    __mod__ = make_invalid_op("__mod__")
    __rmod__ = make_invalid_op("__rmod__")
    __divmod__ = make_invalid_op("__divmod__")
    __rdivmod__ = make_invalid_op("__rdivmod__")

    @final
    def _add_datetimelike_scalar(self, other) -> DatetimeArray:
        if not is_timedelta64_dtype(self.dtype):
            raise TypeError(
                f"cannot add {type(self).__name__} and {type(other).__name__}"
            )

        self = cast("TimedeltaArray", self)

        from pandas.core.arrays import DatetimeArray
        from pandas.core.arrays.datetimes import tz_to_dtype

        assert other is not NaT
        other = Timestamp(other)
        if other is NaT:
            # In this case we specifically interpret NaT as a datetime, not
            # the timedelta interpretation we would get by returning self + NaT
            result = self._ndarray + NaT.to_datetime64().astype(f"M8[{self._unit}]")
            # Preserve our resolution
            return DatetimeArray._simple_new(result, dtype=result.dtype)

        if self._reso != other._reso:
            raise NotImplementedError(
                "Addition between TimedeltaArray and Timestamp with mis-matched "
                "resolutions is not yet supported."
            )

        i8 = self.asi8
        result = checked_add_with_arr(i8, other.value, arr_mask=self._isnan)
        dtype = tz_to_dtype(tz=other.tz, unit=self._unit)
        res_values = result.view(f"M8[{self._unit}]")
        return DatetimeArray._simple_new(res_values, dtype=dtype, freq=self.freq)

    @final
    def _add_datetime_arraylike(self, other) -> DatetimeArray:
        if not is_timedelta64_dtype(self.dtype):
            raise TypeError(
                f"cannot add {type(self).__name__} and {type(other).__name__}"
            )

        # At this point we have already checked that other.dtype is datetime64
        other = ensure_wrapped_if_datetimelike(other)
        # defer to DatetimeArray.__add__
        return other + self

    @final
    def _sub_datetimelike_scalar(self, other: datetime | np.datetime64):
        if self.dtype.kind != "M":
            raise TypeError(f"cannot subtract a datelike from a {type(self).__name__}")

        self = cast("DatetimeArray", self)
        # subtract a datetime from myself, yielding a ndarray[timedelta64[ns]]

        # error: Non-overlapping identity check (left operand type: "Union[datetime,
        # datetime64]", right operand type: "NaTType")  [comparison-overlap]
        assert other is not NaT  # type: ignore[comparison-overlap]
        other = Timestamp(other)
        # error: Non-overlapping identity check (left operand type: "Timestamp",
        # right operand type: "NaTType")
        if other is NaT:  # type: ignore[comparison-overlap]
            return self - NaT

        try:
            self._assert_tzawareness_compat(other)
        except TypeError as err:
            new_message = str(err).replace("compare", "subtract")
            raise type(err)(new_message) from err

        i8 = self.asi8
        result = checked_add_with_arr(i8, -other.value, arr_mask=self._isnan)
        return result.view("timedelta64[ns]")

    @final
    def _sub_datetime_arraylike(self, other):
        if self.dtype.kind != "M":
            raise TypeError(f"cannot subtract a datelike from a {type(self).__name__}")

        if len(self) != len(other):
            raise ValueError("cannot add indices of unequal length")

        self = cast("DatetimeArray", self)
        other = ensure_wrapped_if_datetimelike(other)

        try:
            self._assert_tzawareness_compat(other)
        except TypeError as err:
            new_message = str(err).replace("compare", "subtract")
            raise type(err)(new_message) from err

        self_i8 = self.asi8
        other_i8 = other.asi8
        new_values = checked_add_with_arr(
            self_i8, -other_i8, arr_mask=self._isnan, b_mask=other._isnan
        )
        return new_values.view("timedelta64[ns]")

    @final
    def _sub_period(self, other: Period) -> npt.NDArray[np.object_]:
        if not is_period_dtype(self.dtype):
            raise TypeError(f"cannot subtract Period from a {type(self).__name__}")

        # If the operation is well-defined, we return an object-dtype ndarray
        # of DateOffsets.  Null entries are filled with pd.NaT
        self._check_compatible_with(other)
        new_i8_data = checked_add_with_arr(
            self.asi8, -other.ordinal, arr_mask=self._isnan
        )
        new_data = np.array([self.freq.base * x for x in new_i8_data])

        if self._hasna:
            new_data[self._isnan] = NaT

        return new_data

    @final
    def _add_period(self, other: Period) -> PeriodArray:
        if not is_timedelta64_dtype(self.dtype):
            raise TypeError(f"cannot add Period to a {type(self).__name__}")

        # We will wrap in a PeriodArray and defer to the reversed operation
        from pandas.core.arrays.period import PeriodArray

        i8vals = np.broadcast_to(other.ordinal, self.shape)
        parr = PeriodArray(i8vals, freq=other.freq)
        return parr + self

    def _add_offset(self, offset):
        raise AbstractMethodError(self)

    def _add_timedeltalike_scalar(self, other):
        """
        Add a delta of a timedeltalike

        Returns
        -------
        Same type as self
        """
        if isna(other):
            # i.e np.timedelta64("NaT"), not recognized by delta_to_nanoseconds
            new_values = np.empty(self.shape, dtype="i8").view(self._ndarray.dtype)
            new_values.fill(iNaT)
            return type(self)._simple_new(new_values, dtype=self.dtype)

        # PeriodArray overrides, so we only get here with DTA/TDA
        # error: "DatetimeLikeArrayMixin" has no attribute "_reso"
        inc = delta_to_nanoseconds(other, reso=self._reso)  # type: ignore[attr-defined]

        new_values = checked_add_with_arr(self.asi8, inc, arr_mask=self._isnan)
        new_values = new_values.view(self._ndarray.dtype)

        new_freq = None
        if isinstance(self.freq, Tick) or is_period_dtype(self.dtype):
            # adding a scalar preserves freq
            new_freq = self.freq

        # error: Unexpected keyword argument "freq" for "_simple_new" of "NDArrayBacked"
        return type(self)._simple_new(  # type: ignore[call-arg]
            new_values, dtype=self.dtype, freq=new_freq
        )

    def _add_timedelta_arraylike(
        self, other: TimedeltaArray | npt.NDArray[np.timedelta64]
    ):
        """
        Add a delta of a TimedeltaIndex

        Returns
        -------
        Same type as self
        """
        # overridden by PeriodArray

        if len(self) != len(other):
            raise ValueError("cannot add indices of unequal length")

        other = ensure_wrapped_if_datetimelike(other)
        other = cast("TimedeltaArray", other)

        self_i8 = self.asi8
        other_i8 = other.asi8
        new_values = checked_add_with_arr(
            self_i8, other_i8, arr_mask=self._isnan, b_mask=other._isnan
        )
        return type(self)(new_values, dtype=self.dtype)

    @final
    def _add_nat(self):
        """
        Add pd.NaT to self
        """
        if is_period_dtype(self.dtype):
            raise TypeError(
                f"Cannot add {type(self).__name__} and {type(NaT).__name__}"
            )
        self = cast("TimedeltaArray | DatetimeArray", self)

        # GH#19124 pd.NaT is treated like a timedelta for both timedelta
        # and datetime dtypes
        result = np.empty(self.shape, dtype=np.int64)
        result.fill(iNaT)
        result = result.view(self._ndarray.dtype)  # preserve reso
        return type(self)._simple_new(result, dtype=self.dtype, freq=None)

    @final
    def _sub_nat(self):
        """
        Subtract pd.NaT from self
        """
        # GH#19124 Timedelta - datetime is not in general well-defined.
        # We make an exception for pd.NaT, which in this case quacks
        # like a timedelta.
        # For datetime64 dtypes by convention we treat NaT as a datetime, so
        # this subtraction returns a timedelta64 dtype.
        # For period dtype, timedelta64 is a close-enough return dtype.
        result = np.empty(self.shape, dtype=np.int64)
        result.fill(iNaT)
        return result.view("timedelta64[ns]")

    @final
    def _sub_period_array(self, other: PeriodArray) -> npt.NDArray[np.object_]:
        if not is_period_dtype(self.dtype):
            raise TypeError(
                f"cannot subtract {other.dtype}-dtype from {type(self).__name__}"
            )

        self = cast("PeriodArray", self)
        self._require_matching_freq(other)

        new_i8_values = checked_add_with_arr(
            self.asi8, -other.asi8, arr_mask=self._isnan, b_mask=other._isnan
        )

        new_values = np.array([self.freq.base * x for x in new_i8_values])
        if self._hasna or other._hasna:
            mask = self._isnan | other._isnan
            new_values[mask] = NaT
        return new_values

    @final
    def _addsub_object_array(self, other: np.ndarray, op):
        """
        Add or subtract array-like of DateOffset objects

        Parameters
        ----------
        other : np.ndarray[object]
        op : {operator.add, operator.sub}

        Returns
        -------
        result : same class as self
        """
        assert op in [operator.add, operator.sub]
        if len(other) == 1 and self.ndim == 1:
            # If both 1D then broadcasting is unambiguous
            return op(self, other[0])

        warnings.warn(
            "Adding/subtracting object-dtype array to "
            f"{type(self).__name__} not vectorized.",
            PerformanceWarning,
            stacklevel=find_stack_level(),
        )

        # Caller is responsible for broadcasting if necessary
        assert self.shape == other.shape, (self.shape, other.shape)

        with warnings.catch_warnings():
            # filter out warnings about Timestamp.freq
            warnings.filterwarnings("ignore", category=FutureWarning)
            res_values = op(self.astype("O"), np.asarray(other))

        result = pd_array(res_values.ravel())
        result = extract_array(result, extract_numpy=True).reshape(self.shape)
        return result

    def _time_shift(
        self: DatetimeLikeArrayT, periods: int, freq=None
    ) -> DatetimeLikeArrayT:
        """
        Shift each value by `periods`.

        Note this is different from ExtensionArray.shift, which
        shifts the *position* of each element, padding the end with
        missing values.

        Parameters
        ----------
        periods : int
            Number of periods to shift by.
        freq : pandas.DateOffset, pandas.Timedelta, or str
            Frequency increment to shift by.
        """
        if freq is not None and freq != self.freq:
            if isinstance(freq, str):
                freq = to_offset(freq)
            offset = periods * freq
            return self + offset

        if periods == 0 or len(self) == 0:
            # GH#14811 empty case
            return self.copy()

        if self.freq is None:
            raise NullFrequencyError("Cannot shift with no freq")

        start = self[0] + periods * self.freq
        end = self[-1] + periods * self.freq

        # Note: in the DatetimeTZ case, _generate_range will infer the
        #  appropriate timezone from `start` and `end`, so tz does not need
        #  to be passed explicitly.
        return self._generate_range(start=start, end=end, periods=None, freq=self.freq)

    @unpack_zerodim_and_defer("__add__")
    def __add__(self, other):
        other_dtype = getattr(other, "dtype", None)

        # scalar others
        if other is NaT:
            result = self._add_nat()
        elif isinstance(other, (Tick, timedelta, np.timedelta64)):
            result = self._add_timedeltalike_scalar(other)
        elif isinstance(other, BaseOffset):
            # specifically _not_ a Tick
            result = self._add_offset(other)
        elif isinstance(other, (datetime, np.datetime64)):
            result = self._add_datetimelike_scalar(other)
        elif isinstance(other, Period) and is_timedelta64_dtype(self.dtype):
            result = self._add_period(other)
        elif lib.is_integer(other):
            # This check must come after the check for np.timedelta64
            # as is_integer returns True for these
            if not is_period_dtype(self.dtype):
                raise integer_op_not_supported(self)
            result = cast("PeriodArray", self)._addsub_int_array_or_scalar(
                other * self.freq.n, operator.add
            )

        # array-like others
        elif is_timedelta64_dtype(other_dtype):
            # TimedeltaIndex, ndarray[timedelta64]
            result = self._add_timedelta_arraylike(other)
        elif is_object_dtype(other_dtype):
            # e.g. Array/Index of DateOffset objects
            result = self._addsub_object_array(other, operator.add)
        elif is_datetime64_dtype(other_dtype) or is_datetime64tz_dtype(other_dtype):
            # DatetimeIndex, ndarray[datetime64]
            return self._add_datetime_arraylike(other)
        elif is_integer_dtype(other_dtype):
            if not is_period_dtype(self.dtype):
                raise integer_op_not_supported(self)
            result = cast("PeriodArray", self)._addsub_int_array_or_scalar(
                other * self.freq.n, operator.add
            )
        else:
            # Includes Categorical, other ExtensionArrays
            # For PeriodDtype, if self is a TimedeltaArray and other is a
            #  PeriodArray with  a timedelta-like (i.e. Tick) freq, this
            #  operation is valid.  Defer to the PeriodArray implementation.
            #  In remaining cases, this will end up raising TypeError.
            return NotImplemented

        if isinstance(result, np.ndarray) and is_timedelta64_dtype(result.dtype):
            from pandas.core.arrays import TimedeltaArray

            return TimedeltaArray(result)
        return result

    def __radd__(self, other):
        # alias for __add__
        return self.__add__(other)

    @unpack_zerodim_and_defer("__sub__")
    def __sub__(self, other):

        other_dtype = getattr(other, "dtype", None)

        # scalar others
        if other is NaT:
            result = self._sub_nat()
        elif isinstance(other, (Tick, timedelta, np.timedelta64)):
            result = self._add_timedeltalike_scalar(-other)
        elif isinstance(other, BaseOffset):
            # specifically _not_ a Tick
            result = self._add_offset(-other)
        elif isinstance(other, (datetime, np.datetime64)):
            result = self._sub_datetimelike_scalar(other)
        elif lib.is_integer(other):
            # This check must come after the check for np.timedelta64
            # as is_integer returns True for these
            if not is_period_dtype(self.dtype):
                raise integer_op_not_supported(self)
            result = cast("PeriodArray", self)._addsub_int_array_or_scalar(
                other * self.freq.n, operator.sub
            )

        elif isinstance(other, Period):
            result = self._sub_period(other)

        # array-like others
        elif is_timedelta64_dtype(other_dtype):
            # TimedeltaIndex, ndarray[timedelta64]
            result = self._add_timedelta_arraylike(-other)
        elif is_object_dtype(other_dtype):
            # e.g. Array/Index of DateOffset objects
            result = self._addsub_object_array(other, operator.sub)
        elif is_datetime64_dtype(other_dtype) or is_datetime64tz_dtype(other_dtype):
            # DatetimeIndex, ndarray[datetime64]
            result = self._sub_datetime_arraylike(other)
        elif is_period_dtype(other_dtype):
            # PeriodIndex
            result = self._sub_period_array(other)
        elif is_integer_dtype(other_dtype):
            if not is_period_dtype(self.dtype):
                raise integer_op_not_supported(self)
            result = cast("PeriodArray", self)._addsub_int_array_or_scalar(
                other * self.freq.n, operator.sub
            )
        else:
            # Includes ExtensionArrays, float_dtype
            return NotImplemented

        if isinstance(result, np.ndarray) and is_timedelta64_dtype(result.dtype):
            from pandas.core.arrays import TimedeltaArray

            return TimedeltaArray(result)
        return result

    def __rsub__(self, other):
        other_dtype = getattr(other, "dtype", None)

        if is_datetime64_any_dtype(other_dtype) and is_timedelta64_dtype(self.dtype):
            # ndarray[datetime64] cannot be subtracted from self, so
            # we need to wrap in DatetimeArray/Index and flip the operation
            if lib.is_scalar(other):
                # i.e. np.datetime64 object
                return Timestamp(other) - self
            if not isinstance(other, DatetimeLikeArrayMixin):
                # Avoid down-casting DatetimeIndex
                from pandas.core.arrays import DatetimeArray

                other = DatetimeArray(other)
            return other - self
        elif (
            is_datetime64_any_dtype(self.dtype)
            and hasattr(other, "dtype")
            and not is_datetime64_any_dtype(other.dtype)
        ):
            # GH#19959 datetime - datetime is well-defined as timedelta,
            # but any other type - datetime is not well-defined.
            raise TypeError(
                f"cannot subtract {type(self).__name__} from {type(other).__name__}"
            )
        elif is_period_dtype(self.dtype) and is_timedelta64_dtype(other_dtype):
            # TODO: Can we simplify/generalize these cases at all?
            raise TypeError(f"cannot subtract {type(self).__name__} from {other.dtype}")
        elif is_timedelta64_dtype(self.dtype):
            self = cast("TimedeltaArray", self)
            return (-self) + other

        # We get here with e.g. datetime objects
        return -(self - other)

    def __iadd__(self: DatetimeLikeArrayT, other) -> DatetimeLikeArrayT:
        result = self + other
        self[:] = result[:]

        if not is_period_dtype(self.dtype):
            # restore freq, which is invalidated by setitem
            self._freq = result.freq
        return self

    def __isub__(self: DatetimeLikeArrayT, other) -> DatetimeLikeArrayT:
        result = self - other
        self[:] = result[:]

        if not is_period_dtype(self.dtype):
            # restore freq, which is invalidated by setitem
            self._freq = result.freq
        return self

    # --------------------------------------------------------------
    # Reductions

    def min(self, *, axis: int | None = None, skipna: bool = True, **kwargs):
        """
        Return the minimum value of the Array or minimum along
        an axis.

        See Also
        --------
        numpy.ndarray.min
        Index.min : Return the minimum value in an Index.
        Series.min : Return the minimum value in a Series.
        """
        nv.validate_min((), kwargs)
        nv.validate_minmax_axis(axis, self.ndim)

        if is_period_dtype(self.dtype):
            # pass datetime64 values to nanops to get correct NaT semantics
            result = nanops.nanmin(
                self._ndarray.view("M8[ns]"), axis=axis, skipna=skipna
            )
            if result is NaT:
                return NaT
            result = result.view("i8")
            if axis is None or self.ndim == 1:
                return self._box_func(result)
            return self._from_backing_data(result)

        result = nanops.nanmin(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def max(self, *, axis: int | None = None, skipna: bool = True, **kwargs):
        """
        Return the maximum value of the Array or maximum along
        an axis.

        See Also
        --------
        numpy.ndarray.max
        Index.max : Return the maximum value in an Index.
        Series.max : Return the maximum value in a Series.
        """
        nv.validate_max((), kwargs)
        nv.validate_minmax_axis(axis, self.ndim)

        if is_period_dtype(self.dtype):
            # pass datetime64 values to nanops to get correct NaT semantics
            result = nanops.nanmax(
                self._ndarray.view("M8[ns]"), axis=axis, skipna=skipna
            )
            if result is NaT:
                return result
            result = result.view("i8")
            if axis is None or self.ndim == 1:
                return self._box_func(result)
            return self._from_backing_data(result)

        result = nanops.nanmax(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def mean(self, *, skipna: bool = True, axis: int | None = 0):
        """
        Return the mean value of the Array.

        .. versionadded:: 0.25.0

        Parameters
        ----------
        skipna : bool, default True
            Whether to ignore any NaT elements.
        axis : int, optional, default 0

        Returns
        -------
        scalar
            Timestamp or Timedelta.

        See Also
        --------
        numpy.ndarray.mean : Returns the average of array elements along a given axis.
        Series.mean : Return the mean value in a Series.

        Notes
        -----
        mean is only defined for Datetime and Timedelta dtypes, not for Period.
        """
        if is_period_dtype(self.dtype):
            # See discussion in GH#24757
            raise TypeError(
                f"mean is not implemented for {type(self).__name__} since the "
                "meaning is ambiguous.  An alternative is "
                "obj.to_timestamp(how='start').mean()"
            )

        result = nanops.nanmean(
            self._ndarray, axis=axis, skipna=skipna, mask=self.isna()
        )
        return self._wrap_reduction_result(axis, result)

    def median(self, *, axis: int | None = None, skipna: bool = True, **kwargs):
        nv.validate_median((), kwargs)

        if axis is not None and abs(axis) >= self.ndim:
            raise ValueError("abs(axis) must be less than ndim")

        if is_period_dtype(self.dtype):
            # pass datetime64 values to nanops to get correct NaT semantics
            result = nanops.nanmedian(
                self._ndarray.view("M8[ns]"), axis=axis, skipna=skipna
            )
            result = result.view("i8")
            if axis is None or self.ndim == 1:
                return self._box_func(result)
            return self._from_backing_data(result)

        result = nanops.nanmedian(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def _mode(self, dropna: bool = True):
        mask = None
        if dropna:
            mask = self.isna()

        i8modes = mode(self.view("i8"), mask=mask)
        npmodes = i8modes.view(self._ndarray.dtype)
        npmodes = cast(np.ndarray, npmodes)
        return self._from_backing_data(npmodes)


class DatelikeOps(DatetimeLikeArrayMixin):
    """
    Common ops for DatetimeIndex/PeriodIndex, but not TimedeltaIndex.
    """

    @Substitution(
        URL="https://docs.python.org/3/library/datetime.html"
        "#strftime-and-strptime-behavior"
    )
    def strftime(self, date_format: str) -> npt.NDArray[np.object_]:
        """
        Convert to Index using specified date_format.

        Return an Index of formatted strings specified by date_format, which
        supports the same string format as the python standard library. Details
        of the string format can be found in `python string format
        doc <%(URL)s>`__.

        Formats supported by the C `strftime` API but not by the python string format
        doc (such as `"%%R"`, `"%%r"`) are not officially supported and should be
        preferably replaced with their supported equivalents (such as `"%%H:%%M"`,
        `"%%I:%%M:%%S %%p"`).

        Note that `PeriodIndex` support additional directives, detailed in
        `Period.strftime`.

        Parameters
        ----------
        date_format : str
            Date format string (e.g. "%%Y-%%m-%%d").

        Returns
        -------
        ndarray[object]
            NumPy ndarray of formatted strings.

        See Also
        --------
        to_datetime : Convert the given argument to datetime.
        DatetimeIndex.normalize : Return DatetimeIndex with times to midnight.
        DatetimeIndex.round : Round the DatetimeIndex to the specified freq.
        DatetimeIndex.floor : Floor the DatetimeIndex to the specified freq.
        Timestamp.strftime : Format a single Timestamp.
        Period.strftime : Format a single Period.

        Examples
        --------
        >>> rng = pd.date_range(pd.Timestamp("2018-03-10 09:00"),
        ...                     periods=3, freq='s')
        >>> rng.strftime('%%B %%d, %%Y, %%r')
        Index(['March 10, 2018, 09:00:00 AM', 'March 10, 2018, 09:00:01 AM',
               'March 10, 2018, 09:00:02 AM'],
              dtype='object')
        """
        result = self._format_native_types(date_format=date_format, na_rep=np.nan)
        return result.astype(object, copy=False)


_round_doc = """
    Perform {op} operation on the data to the specified `freq`.

    Parameters
    ----------
    freq : str or Offset
        The frequency level to {op} the index to. Must be a fixed
        frequency like 'S' (second) not 'ME' (month end). See
        :ref:`frequency aliases <timeseries.offset_aliases>` for
        a list of possible `freq` values.
    ambiguous : 'infer', bool-ndarray, 'NaT', default 'raise'
        Only relevant for DatetimeIndex:

        - 'infer' will attempt to infer fall dst-transition hours based on
          order
        - bool-ndarray where True signifies a DST time, False designates
          a non-DST time (note that this flag is only applicable for
          ambiguous times)
        - 'NaT' will return NaT where there are ambiguous times
        - 'raise' will raise an AmbiguousTimeError if there are ambiguous
          times.

    nonexistent : 'shift_forward', 'shift_backward', 'NaT', timedelta, default 'raise'
        A nonexistent time does not exist in a particular timezone
        where clocks moved forward due to DST.

        - 'shift_forward' will shift the nonexistent time forward to the
          closest existing time
        - 'shift_backward' will shift the nonexistent time backward to the
          closest existing time
        - 'NaT' will return NaT where there are nonexistent times
        - timedelta objects will shift nonexistent times by the timedelta
        - 'raise' will raise an NonExistentTimeError if there are
          nonexistent times.

    Returns
    -------
    DatetimeIndex, TimedeltaIndex, or Series
        Index of the same type for a DatetimeIndex or TimedeltaIndex,
        or a Series with the same index for a Series.

    Raises
    ------
    ValueError if the `freq` cannot be converted.

    Notes
    -----
    If the timestamps have a timezone, {op}ing will take place relative to the
    local ("wall") time and re-localized to the same timezone. When {op}ing
    near daylight savings time, use ``nonexistent`` and ``ambiguous`` to
    control the re-localization behavior.

    Examples
    --------
    **DatetimeIndex**

    >>> rng = pd.date_range('1/1/2018 11:59:00', periods=3, freq='min')
    >>> rng
    DatetimeIndex(['2018-01-01 11:59:00', '2018-01-01 12:00:00',
                   '2018-01-01 12:01:00'],
                  dtype='datetime64[ns]', freq='T')
    """

_round_example = """>>> rng.round('H')
    DatetimeIndex(['2018-01-01 12:00:00', '2018-01-01 12:00:00',
                   '2018-01-01 12:00:00'],
                  dtype='datetime64[ns]', freq=None)

    **Series**

    >>> pd.Series(rng).dt.round("H")
    0   2018-01-01 12:00:00
    1   2018-01-01 12:00:00
    2   2018-01-01 12:00:00
    dtype: datetime64[ns]

    When rounding near a daylight savings time transition, use ``ambiguous`` or
    ``nonexistent`` to control how the timestamp should be re-localized.

    >>> rng_tz = pd.DatetimeIndex(["2021-10-31 03:30:00"], tz="Europe/Amsterdam")

    >>> rng_tz.floor("2H", ambiguous=False)
    DatetimeIndex(['2021-10-31 02:00:00+01:00'],
                  dtype='datetime64[ns, Europe/Amsterdam]', freq=None)

    >>> rng_tz.floor("2H", ambiguous=True)
    DatetimeIndex(['2021-10-31 02:00:00+02:00'],
                  dtype='datetime64[ns, Europe/Amsterdam]', freq=None)
    """

_floor_example = """>>> rng.floor('H')
    DatetimeIndex(['2018-01-01 11:00:00', '2018-01-01 12:00:00',
                   '2018-01-01 12:00:00'],
                  dtype='datetime64[ns]', freq=None)

    **Series**

    >>> pd.Series(rng).dt.floor("H")
    0   2018-01-01 11:00:00
    1   2018-01-01 12:00:00
    2   2018-01-01 12:00:00
    dtype: datetime64[ns]

    When rounding near a daylight savings time transition, use ``ambiguous`` or
    ``nonexistent`` to control how the timestamp should be re-localized.

    >>> rng_tz = pd.DatetimeIndex(["2021-10-31 03:30:00"], tz="Europe/Amsterdam")

    >>> rng_tz.floor("2H", ambiguous=False)
    DatetimeIndex(['2021-10-31 02:00:00+01:00'],
                 dtype='datetime64[ns, Europe/Amsterdam]', freq=None)

    >>> rng_tz.floor("2H", ambiguous=True)
    DatetimeIndex(['2021-10-31 02:00:00+02:00'],
                  dtype='datetime64[ns, Europe/Amsterdam]', freq=None)
    """

_ceil_example = """>>> rng.ceil('H')
    DatetimeIndex(['2018-01-01 12:00:00', '2018-01-01 12:00:00',
                   '2018-01-01 13:00:00'],
                  dtype='datetime64[ns]', freq=None)

    **Series**

    >>> pd.Series(rng).dt.ceil("H")
    0   2018-01-01 12:00:00
    1   2018-01-01 12:00:00
    2   2018-01-01 13:00:00
    dtype: datetime64[ns]

    When rounding near a daylight savings time transition, use ``ambiguous`` or
    ``nonexistent`` to control how the timestamp should be re-localized.

    >>> rng_tz = pd.DatetimeIndex(["2021-10-31 01:30:00"], tz="Europe/Amsterdam")

    >>> rng_tz.ceil("H", ambiguous=False)
    DatetimeIndex(['2021-10-31 02:00:00+01:00'],
                  dtype='datetime64[ns, Europe/Amsterdam]', freq=None)

    >>> rng_tz.ceil("H", ambiguous=True)
    DatetimeIndex(['2021-10-31 02:00:00+02:00'],
                  dtype='datetime64[ns, Europe/Amsterdam]', freq=None)
    """


TimelikeOpsT = TypeVar("TimelikeOpsT", bound="TimelikeOps")


class TimelikeOps(DatetimeLikeArrayMixin):
    """
    Common ops for TimedeltaIndex/DatetimeIndex, but not PeriodIndex.
    """

    _default_dtype: np.dtype

    def __init__(self, values, dtype=None, freq=lib.no_default, copy: bool = False):
        values = extract_array(values, extract_numpy=True)
        if isinstance(values, IntegerArray):
            values = values.to_numpy("int64", na_value=iNaT)

        inferred_freq = getattr(values, "_freq", None)
        explicit_none = freq is None
        freq = freq if freq is not lib.no_default else None

        if isinstance(values, type(self)):
            if explicit_none:
                # don't inherit from values
                pass
            elif freq is None:
                freq = values.freq
            elif freq and values.freq:
                freq = to_offset(freq)
                freq, _ = validate_inferred_freq(freq, values.freq, False)

            if dtype is not None:
                dtype = pandas_dtype(dtype)
                if not is_dtype_equal(dtype, values.dtype):
                    # TODO: we only have tests for this for DTA, not TDA (2022-07-01)
                    raise TypeError(
                        f"dtype={dtype} does not match data dtype {values.dtype}"
                    )

            dtype = values.dtype
            values = values._ndarray

        elif dtype is None:
            dtype = self._default_dtype

        if not isinstance(values, np.ndarray):
            raise ValueError(
                f"Unexpected type '{type(values).__name__}'. 'values' must be a "
                f"{type(self).__name__}, ndarray, or Series or Index "
                "containing one of those."
            )
        if values.ndim not in [1, 2]:
            raise ValueError("Only 1-dimensional input arrays are supported.")

        if values.dtype == "i8":
            # for compat with datetime/timedelta/period shared methods,
            #  we can sometimes get here with int64 values.  These represent
            #  nanosecond UTC (or tz-naive) unix timestamps
            values = values.view(self._default_dtype)

        dtype = self._validate_dtype(values, dtype)

        if freq == "infer":
            raise ValueError(
                f"Frequency inference not allowed in {type(self).__name__}.__init__. "
                "Use 'pd.array()' instead."
            )

        if copy:
            values = values.copy()
        if freq:
            freq = to_offset(freq)

        NDArrayBacked.__init__(self, values=values, dtype=dtype)
        self._freq = freq

        if inferred_freq is None and freq is not None:
            type(self)._validate_frequency(self, freq)

    @classmethod
    def _validate_dtype(cls, values, dtype):
        raise AbstractMethodError(cls)

    # --------------------------------------------------------------

    @cache_readonly
    def _reso(self) -> int:
        return get_unit_from_dtype(self._ndarray.dtype)

    @cache_readonly
    def _unit(self) -> str:
        # e.g. "ns", "us", "ms"
        # error: Argument 1 to "dtype_to_unit" has incompatible type
        # "ExtensionDtype"; expected "Union[DatetimeTZDtype, dtype[Any]]"
        return dtype_to_unit(self.dtype)  # type: ignore[arg-type]

    # --------------------------------------------------------------

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        if (
            ufunc in [np.isnan, np.isinf, np.isfinite]
            and len(inputs) == 1
            and inputs[0] is self
        ):
            # numpy 1.18 changed isinf and isnan to not raise on dt64/td64
            return getattr(ufunc, method)(self._ndarray, **kwargs)

        return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)

    def _round(self, freq, mode, ambiguous, nonexistent):
        # round the local times
        if is_datetime64tz_dtype(self.dtype):
            # operate on naive timestamps, then convert back to aware
            self = cast("DatetimeArray", self)
            naive = self.tz_localize(None)
            result = naive._round(freq, mode, ambiguous, nonexistent)
            return result.tz_localize(
                self.tz, ambiguous=ambiguous, nonexistent=nonexistent
            )

        values = self.view("i8")
        values = cast(np.ndarray, values)
        nanos = to_offset(freq).nanos  # raises on non-fixed frequencies
        nanos = delta_to_nanoseconds(to_offset(freq), self._reso)
        result_i8 = round_nsint64(values, mode, nanos)
        result = self._maybe_mask_results(result_i8, fill_value=iNaT)
        result = result.view(self._ndarray.dtype)
        return self._simple_new(result, dtype=self.dtype)

    @Appender((_round_doc + _round_example).format(op="round"))
    def round(self, freq, ambiguous="raise", nonexistent="raise"):
        return self._round(freq, RoundTo.NEAREST_HALF_EVEN, ambiguous, nonexistent)

    @Appender((_round_doc + _floor_example).format(op="floor"))
    def floor(self, freq, ambiguous="raise", nonexistent="raise"):
        return self._round(freq, RoundTo.MINUS_INFTY, ambiguous, nonexistent)

    @Appender((_round_doc + _ceil_example).format(op="ceil"))
    def ceil(self, freq, ambiguous="raise", nonexistent="raise"):
        return self._round(freq, RoundTo.PLUS_INFTY, ambiguous, nonexistent)

    # --------------------------------------------------------------
    # Reductions

    def any(self, *, axis: int | None = None, skipna: bool = True) -> bool:
        # GH#34479 discussion of desired behavior long-term
        return nanops.nanany(self._ndarray, axis=axis, skipna=skipna, mask=self.isna())

    def all(self, *, axis: int | None = None, skipna: bool = True) -> bool:
        # GH#34479 discussion of desired behavior long-term
        return nanops.nanall(self._ndarray, axis=axis, skipna=skipna, mask=self.isna())

    # --------------------------------------------------------------
    # Frequency Methods

    def _maybe_clear_freq(self) -> None:
        self._freq = None

    def _with_freq(self, freq):
        """
        Helper to get a view on the same data, with a new freq.

        Parameters
        ----------
        freq : DateOffset, None, or "infer"

        Returns
        -------
        Same type as self
        """
        # GH#29843
        if freq is None:
            # Always valid
            pass
        elif len(self) == 0 and isinstance(freq, BaseOffset):
            # Always valid.  In the TimedeltaArray case, we assume this
            #  is a Tick offset.
            pass
        else:
            # As an internal method, we can ensure this assertion always holds
            assert freq == "infer"
            freq = to_offset(self.inferred_freq)

        arr = self.view()
        arr._freq = freq
        return arr

    # --------------------------------------------------------------

    # GH#46910 - Keep old signature to test we don't break things for EA library authors
    def factorize(  # type:ignore[override]
        self,
        na_sentinel: int = -1,
        sort: bool = False,
    ):
        if self.freq is not None:
            # We must be unique, so can short-circuit (and retain freq)
            codes = np.arange(len(self), dtype=np.intp)
            uniques = self.copy()  # TODO: copy or view?
            if sort and self.freq.n < 0:
                codes = codes[::-1]
                uniques = uniques[::-1]
            return codes, uniques
        # FIXME: shouldn't get here; we are ignoring sort
        return super().factorize(na_sentinel=na_sentinel)


# -------------------------------------------------------------------
# Shared Constructor Helpers


def ensure_arraylike_for_datetimelike(data, copy: bool, cls_name: str):
    if not hasattr(data, "dtype"):
        # e.g. list, tuple
        if np.ndim(data) == 0:
            # i.e. generator
            data = list(data)
        data = np.asarray(data)
        copy = False
    elif isinstance(data, ABCMultiIndex):
        raise TypeError(f"Cannot create a {cls_name} from a MultiIndex.")
    else:
        data = extract_array(data, extract_numpy=True)

    if isinstance(data, IntegerArray):
        data = data.to_numpy("int64", na_value=iNaT)
        copy = False
    elif not isinstance(data, (np.ndarray, ExtensionArray)):
        # GH#24539 e.g. xarray, dask object
        data = np.asarray(data)

    elif isinstance(data, ABCCategorical):
        # GH#18664 preserve tz in going DTI->Categorical->DTI
        # TODO: cases where we need to do another pass through maybe_convert_dtype,
        #  e.g. the categories are timedelta64s
        data = data.categories.take(data.codes, fill_value=NaT)._values
        copy = False

    return data, copy


@overload
def validate_periods(periods: None) -> None:
    ...


@overload
def validate_periods(periods: float) -> int:
    ...


def validate_periods(periods: float | None) -> int | None:
    """
    If a `periods` argument is passed to the Datetime/Timedelta Array/Index
    constructor, cast it to an integer.

    Parameters
    ----------
    periods : None, float, int

    Returns
    -------
    periods : None or int

    Raises
    ------
    TypeError
        if periods is None, float, or int
    """
    if periods is not None:
        if lib.is_float(periods):
            periods = int(periods)
        elif not lib.is_integer(periods):
            raise TypeError(f"periods must be a number, got {periods}")
    # error: Incompatible return value type (got "Optional[float]",
    # expected "Optional[int]")
    return periods  # type: ignore[return-value]


def validate_inferred_freq(
    freq, inferred_freq, freq_infer
) -> tuple[BaseOffset | None, bool]:
    """
    If the user passes a freq and another freq is inferred from passed data,
    require that they match.

    Parameters
    ----------
    freq : DateOffset or None
    inferred_freq : DateOffset or None
    freq_infer : bool

    Returns
    -------
    freq : DateOffset or None
    freq_infer : bool

    Notes
    -----
    We assume at this point that `maybe_infer_freq` has been called, so
    `freq` is either a DateOffset object or None.
    """
    if inferred_freq is not None:
        if freq is not None and freq != inferred_freq:
            raise ValueError(
                f"Inferred frequency {inferred_freq} from passed "
                "values does not conform to passed frequency "
                f"{freq.freqstr}"
            )
        elif freq is None:
            freq = inferred_freq
        freq_infer = False

    return freq, freq_infer


def maybe_infer_freq(freq):
    """
    Comparing a DateOffset to the string "infer" raises, so we need to
    be careful about comparisons.  Make a dummy variable `freq_infer` to
    signify the case where the given freq is "infer" and set freq to None
    to avoid comparison trouble later on.

    Parameters
    ----------
    freq : {DateOffset, None, str}

    Returns
    -------
    freq : {DateOffset, None}
    freq_infer : bool
        Whether we should inherit the freq of passed data.
    """
    freq_infer = False
    if not isinstance(freq, BaseOffset):
        # if a passed freq is None, don't infer automatically
        if freq != "infer":
            freq = to_offset(freq)
        else:
            freq_infer = True
            freq = None
    return freq, freq_infer


def dtype_to_unit(dtype: DatetimeTZDtype | np.dtype) -> str:
    """
    Return the unit str corresponding to the dtype's resolution.

    Parameters
    ----------
    dtype : DatetimeTZDtype or np.dtype
        If np.dtype, we assume it is a datetime64 dtype.

    Returns
    -------
    str
    """
    if isinstance(dtype, DatetimeTZDtype):
        return dtype.unit
    return np.datetime_data(dtype)[0]

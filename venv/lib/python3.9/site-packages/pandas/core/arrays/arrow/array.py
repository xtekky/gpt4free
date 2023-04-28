from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

import numpy as np

from pandas._libs import lib
from pandas._typing import (
    Dtype,
    PositionalIndexer,
    TakeIndexer,
    npt,
)
from pandas.compat import (
    pa_version_under1p01,
    pa_version_under2p0,
    pa_version_under3p0,
    pa_version_under4p0,
    pa_version_under5p0,
    pa_version_under6p0,
    pa_version_under7p0,
)
from pandas.util._decorators import (
    deprecate_nonkeyword_arguments,
    doc,
)

from pandas.core.dtypes.common import (
    is_array_like,
    is_bool_dtype,
    is_integer,
    is_integer_dtype,
    is_scalar,
)
from pandas.core.dtypes.missing import isna

from pandas.core.algorithms import resolve_na_sentinel
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays.base import ExtensionArray
from pandas.core.indexers import (
    check_array_indexer,
    unpack_tuple_and_ellipses,
    validate_indices,
)

if not pa_version_under1p01:
    import pyarrow as pa
    import pyarrow.compute as pc

    from pandas.core.arrays.arrow._arrow_utils import fallback_performancewarning
    from pandas.core.arrays.arrow.dtype import ArrowDtype

    ARROW_CMP_FUNCS = {
        "eq": pc.equal,
        "ne": pc.not_equal,
        "lt": pc.less,
        "gt": pc.greater,
        "le": pc.less_equal,
        "ge": pc.greater_equal,
    }

    ARROW_LOGICAL_FUNCS = {
        "and": NotImplemented if pa_version_under2p0 else pc.and_kleene,
        "rand": NotImplemented
        if pa_version_under2p0
        else lambda x, y: pc.and_kleene(y, x),
        "or": NotImplemented if pa_version_under2p0 else pc.or_kleene,
        "ror": NotImplemented
        if pa_version_under2p0
        else lambda x, y: pc.or_kleene(y, x),
        "xor": NotImplemented if pa_version_under2p0 else pc.xor,
        "rxor": NotImplemented if pa_version_under2p0 else lambda x, y: pc.xor(y, x),
    }

    def cast_for_truediv(
        arrow_array: pa.ChunkedArray, pa_object: pa.Array | pa.Scalar
    ) -> pa.ChunkedArray:
        # Ensure int / int -> float mirroring Python/Numpy behavior
        # as pc.divide_checked(int, int) -> int
        if pa.types.is_integer(arrow_array.type) and pa.types.is_integer(
            pa_object.type
        ):
            return arrow_array.cast(pa.float64())
        return arrow_array

    def floordiv_compat(
        left: pa.ChunkedArray | pa.Array | pa.Scalar,
        right: pa.ChunkedArray | pa.Array | pa.Scalar,
    ) -> pa.ChunkedArray:
        # Ensure int // int -> int mirroring Python/Numpy behavior
        # as pc.floor(pc.divide_checked(int, int)) -> float
        result = pc.floor(pc.divide_checked(left, right))
        if pa.types.is_integer(left.type) and pa.types.is_integer(right.type):
            result = result.cast(left.type)
        return result

    ARROW_ARITHMETIC_FUNCS = {
        "add": NotImplemented if pa_version_under2p0 else pc.add_checked,
        "radd": NotImplemented
        if pa_version_under2p0
        else lambda x, y: pc.add_checked(y, x),
        "sub": NotImplemented if pa_version_under2p0 else pc.subtract_checked,
        "rsub": NotImplemented
        if pa_version_under2p0
        else lambda x, y: pc.subtract_checked(y, x),
        "mul": NotImplemented if pa_version_under2p0 else pc.multiply_checked,
        "rmul": NotImplemented
        if pa_version_under2p0
        else lambda x, y: pc.multiply_checked(y, x),
        "truediv": NotImplemented
        if pa_version_under2p0
        else lambda x, y: pc.divide_checked(cast_for_truediv(x, y), y),
        "rtruediv": NotImplemented
        if pa_version_under2p0
        else lambda x, y: pc.divide_checked(y, cast_for_truediv(x, y)),
        "floordiv": NotImplemented
        if pa_version_under2p0
        else lambda x, y: floordiv_compat(x, y),
        "rfloordiv": NotImplemented
        if pa_version_under2p0
        else lambda x, y: floordiv_compat(y, x),
        "mod": NotImplemented,
        "rmod": NotImplemented,
        "divmod": NotImplemented,
        "rdivmod": NotImplemented,
        "pow": NotImplemented if pa_version_under4p0 else pc.power_checked,
        "rpow": NotImplemented
        if pa_version_under4p0
        else lambda x, y: pc.power_checked(y, x),
    }

if TYPE_CHECKING:
    from pandas import Series

ArrowExtensionArrayT = TypeVar("ArrowExtensionArrayT", bound="ArrowExtensionArray")


def to_pyarrow_type(
    dtype: ArrowDtype | pa.DataType | Dtype | None,
) -> pa.DataType | None:
    """
    Convert dtype to a pyarrow type instance.
    """
    if isinstance(dtype, ArrowDtype):
        pa_dtype = dtype.pyarrow_dtype
    elif isinstance(dtype, pa.DataType):
        pa_dtype = dtype
    elif dtype:
        # Accepts python types too
        pa_dtype = pa.from_numpy_dtype(dtype)
    else:
        pa_dtype = None
    return pa_dtype


class ArrowExtensionArray(OpsMixin, ExtensionArray):
    """
    Pandas ExtensionArray backed by a PyArrow ChunkedArray.

    .. warning::

       ArrowExtensionArray is considered experimental. The implementation and
       parts of the API may change without warning.

    Parameters
    ----------
    values : pyarrow.Array or pyarrow.ChunkedArray

    Attributes
    ----------
    None

    Methods
    -------
    None

    Returns
    -------
    ArrowExtensionArray

    Notes
    -----
    Most methods are implemented using `pyarrow compute functions. <https://arrow.apache.org/docs/python/api/compute.html>`__
    Some methods may either raise an exception or raise a ``PerformanceWarning`` if an
    associated compute function is not available based on the installed version of PyArrow.

    Please install the latest version of PyArrow to enable the best functionality and avoid
    potential bugs in prior versions of PyArrow.

    Examples
    --------
    Create an ArrowExtensionArray with :func:`pandas.array`:

    >>> pd.array([1, 1, None], dtype="int64[pyarrow]")
    <ArrowExtensionArray>
    [1, 1, <NA>]
    Length: 3, dtype: int64[pyarrow]
    """  # noqa: E501 (http link too long)

    _data: pa.ChunkedArray
    _dtype: ArrowDtype

    def __init__(self, values: pa.Array | pa.ChunkedArray) -> None:
        if pa_version_under1p01:
            msg = "pyarrow>=1.0.0 is required for PyArrow backed ArrowExtensionArray."
            raise ImportError(msg)
        if isinstance(values, pa.Array):
            self._data = pa.chunked_array([values])
        elif isinstance(values, pa.ChunkedArray):
            self._data = values
        else:
            raise ValueError(
                f"Unsupported type '{type(values)}' for ArrowExtensionArray"
            )
        self._dtype = ArrowDtype(self._data.type)

    @classmethod
    def _from_sequence(cls, scalars, *, dtype: Dtype | None = None, copy=False):
        """
        Construct a new ExtensionArray from a sequence of scalars.
        """
        pa_dtype = to_pyarrow_type(dtype)
        is_cls = isinstance(scalars, cls)
        if is_cls or isinstance(scalars, (pa.Array, pa.ChunkedArray)):
            if is_cls:
                scalars = scalars._data
            if pa_dtype:
                scalars = scalars.cast(pa_dtype)
            return cls(scalars)
        else:
            return cls(
                pa.chunked_array(pa.array(scalars, type=pa_dtype, from_pandas=True))
            )

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, *, dtype: Dtype | None = None, copy=False
    ):
        """
        Construct a new ExtensionArray from a sequence of strings.
        """
        pa_type = to_pyarrow_type(dtype)
        if pa_type is None:
            # Let pyarrow try to infer or raise
            scalars = strings
        elif pa.types.is_timestamp(pa_type):
            from pandas.core.tools.datetimes import to_datetime

            scalars = to_datetime(strings, errors="raise")
        elif pa.types.is_date(pa_type):
            from pandas.core.tools.datetimes import to_datetime

            scalars = to_datetime(strings, errors="raise").date
        elif pa.types.is_duration(pa_type):
            from pandas.core.tools.timedeltas import to_timedelta

            scalars = to_timedelta(strings, errors="raise")
        elif pa.types.is_time(pa_type):
            from pandas.core.tools.times import to_time

            # "coerce" to allow "null times" (None) to not raise
            scalars = to_time(strings, errors="coerce")
        elif pa.types.is_boolean(pa_type):
            from pandas.core.arrays import BooleanArray

            scalars = BooleanArray._from_sequence_of_strings(strings).to_numpy()
        elif (
            pa.types.is_integer(pa_type)
            or pa.types.is_floating(pa_type)
            or pa.types.is_decimal(pa_type)
        ):
            from pandas.core.tools.numeric import to_numeric

            scalars = to_numeric(strings, errors="raise")
        else:
            raise NotImplementedError(
                f"Converting strings to {pa_type} is not implemented."
            )
        return cls._from_sequence(scalars, dtype=pa_type, copy=copy)

    def __getitem__(self, item: PositionalIndexer):
        """Select a subset of self.

        Parameters
        ----------
        item : int, slice, or ndarray
            * int: The position in 'self' to get.
            * slice: A slice object, where 'start', 'stop', and 'step' are
              integers or None
            * ndarray: A 1-d boolean NumPy ndarray the same length as 'self'

        Returns
        -------
        item : scalar or ExtensionArray

        Notes
        -----
        For scalar ``item``, return a scalar value suitable for the array's
        type. This should be an instance of ``self.dtype.type``.
        For slice ``key``, return an instance of ``ExtensionArray``, even
        if the slice is length 0 or 1.
        For a boolean mask, return an instance of ``ExtensionArray``, filtered
        to the values where ``item`` is True.
        """
        item = check_array_indexer(self, item)

        if isinstance(item, np.ndarray):
            if not len(item):
                # Removable once we migrate StringDtype[pyarrow] to ArrowDtype[string]
                if self._dtype.name == "string" and self._dtype.storage == "pyarrow":
                    pa_dtype = pa.string()
                else:
                    pa_dtype = self._dtype.pyarrow_dtype
                return type(self)(pa.chunked_array([], type=pa_dtype))
            elif is_integer_dtype(item.dtype):
                return self.take(item)
            elif is_bool_dtype(item.dtype):
                return type(self)(self._data.filter(item))
            else:
                raise IndexError(
                    "Only integers, slices and integer or "
                    "boolean arrays are valid indices."
                )
        elif isinstance(item, tuple):
            item = unpack_tuple_and_ellipses(item)

        # error: Non-overlapping identity check (left operand type:
        # "Union[Union[int, integer[Any]], Union[slice, List[int],
        # ndarray[Any, Any]]]", right operand type: "ellipsis")
        if item is Ellipsis:  # type: ignore[comparison-overlap]
            # TODO: should be handled by pyarrow?
            item = slice(None)

        if is_scalar(item) and not is_integer(item):
            # e.g. "foo" or 2.5
            # exception message copied from numpy
            raise IndexError(
                r"only integers, slices (`:`), ellipsis (`...`), numpy.newaxis "
                r"(`None`) and integer or boolean arrays are valid indices"
            )
        # We are not an array indexer, so maybe e.g. a slice or integer
        # indexer. We dispatch to pyarrow.
        value = self._data[item]
        if isinstance(value, pa.ChunkedArray):
            return type(self)(value)
        else:
            scalar = value.as_py()
            if scalar is None:
                return self._dtype.na_value
            else:
                return scalar

    def __arrow_array__(self, type=None):
        """Convert myself to a pyarrow ChunkedArray."""
        return self._data

    def __invert__(self: ArrowExtensionArrayT) -> ArrowExtensionArrayT:
        if pa_version_under2p0:
            raise NotImplementedError("__invert__ not implement for pyarrow < 2.0")
        return type(self)(pc.invert(self._data))

    def __neg__(self: ArrowExtensionArrayT) -> ArrowExtensionArrayT:
        return type(self)(pc.negate_checked(self._data))

    def __pos__(self: ArrowExtensionArrayT) -> ArrowExtensionArrayT:
        return type(self)(self._data)

    def __abs__(self: ArrowExtensionArrayT) -> ArrowExtensionArrayT:
        return type(self)(pc.abs_checked(self._data))

    def _cmp_method(self, other, op):
        from pandas.arrays import BooleanArray

        pc_func = ARROW_CMP_FUNCS[op.__name__]
        if isinstance(other, ArrowExtensionArray):
            result = pc_func(self._data, other._data)
        elif isinstance(other, (np.ndarray, list)):
            result = pc_func(self._data, other)
        elif is_scalar(other):
            try:
                result = pc_func(self._data, pa.scalar(other))
            except (pa.lib.ArrowNotImplementedError, pa.lib.ArrowInvalid):
                mask = isna(self) | isna(other)
                valid = ~mask
                result = np.zeros(len(self), dtype="bool")
                result[valid] = op(np.array(self)[valid], other)
                return BooleanArray(result, mask)
        else:
            raise NotImplementedError(
                f"{op.__name__} not implemented for {type(other)}"
            )

        if pa_version_under2p0:
            result = result.to_pandas().values
        else:
            result = result.to_numpy()
        return BooleanArray._from_sequence(result)

    def _evaluate_op_method(self, other, op, arrow_funcs):
        pc_func = arrow_funcs[op.__name__]
        if pc_func is NotImplemented:
            raise NotImplementedError(f"{op.__name__} not implemented.")
        if isinstance(other, ArrowExtensionArray):
            result = pc_func(self._data, other._data)
        elif isinstance(other, (np.ndarray, list)):
            result = pc_func(self._data, pa.array(other, from_pandas=True))
        elif is_scalar(other):
            result = pc_func(self._data, pa.scalar(other))
        else:
            raise NotImplementedError(
                f"{op.__name__} not implemented for {type(other)}"
            )
        return type(self)(result)

    def _logical_method(self, other, op):
        return self._evaluate_op_method(other, op, ARROW_LOGICAL_FUNCS)

    def _arith_method(self, other, op):
        return self._evaluate_op_method(other, op, ARROW_ARITHMETIC_FUNCS)

    def equals(self, other) -> bool:
        if not isinstance(other, ArrowExtensionArray):
            return False
        # I'm told that pyarrow makes __eq__ behave like pandas' equals;
        #  TODO: is this documented somewhere?
        return self._data == other._data

    @property
    def dtype(self) -> ArrowDtype:
        """
        An instance of 'ExtensionDtype'.
        """
        return self._dtype

    @property
    def nbytes(self) -> int:
        """
        The number of bytes needed to store this object in memory.
        """
        return self._data.nbytes

    def __len__(self) -> int:
        """
        Length of this array.

        Returns
        -------
        length : int
        """
        return len(self._data)

    @property
    def _hasna(self) -> bool:
        return self._data.null_count > 0

    def isna(self) -> npt.NDArray[np.bool_]:
        """
        Boolean NumPy array indicating if each value is missing.

        This should return a 1-D array the same length as 'self'.
        """
        if pa_version_under2p0:
            return self._data.is_null().to_pandas().values
        else:
            return self._data.is_null().to_numpy()

    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self"])
    def argsort(
        self,
        ascending: bool = True,
        kind: str = "quicksort",
        na_position: str = "last",
        *args,
        **kwargs,
    ) -> np.ndarray:
        order = "ascending" if ascending else "descending"
        null_placement = {"last": "at_end", "first": "at_start"}.get(na_position, None)
        if null_placement is None or pa_version_under7p0:
            # Although pc.array_sort_indices exists in version 6
            # there's a bug that affects the pa.ChunkedArray backing
            # https://issues.apache.org/jira/browse/ARROW-12042
            fallback_performancewarning("7")
            return super().argsort(
                ascending=ascending, kind=kind, na_position=na_position
            )

        result = pc.array_sort_indices(
            self._data, order=order, null_placement=null_placement
        )
        if pa_version_under2p0:
            np_result = result.to_pandas().values
        else:
            np_result = result.to_numpy()
        return np_result.astype(np.intp, copy=False)

    def _argmin_max(self, skipna: bool, method: str) -> int:
        if self._data.length() in (0, self._data.null_count) or (
            self._hasna and not skipna
        ):
            # For empty or all null, pyarrow returns -1 but pandas expects TypeError
            # For skipna=False and data w/ null, pandas expects NotImplementedError
            # let ExtensionArray.arg{max|min} raise
            return getattr(super(), f"arg{method}")(skipna=skipna)

        if pa_version_under6p0:
            raise NotImplementedError(
                f"arg{method} only implemented for pyarrow version >= 6.0"
            )

        value = getattr(pc, method)(self._data, skip_nulls=skipna)
        return pc.index(self._data, value).as_py()

    def argmin(self, skipna: bool = True) -> int:
        return self._argmin_max(skipna, "min")

    def argmax(self, skipna: bool = True) -> int:
        return self._argmin_max(skipna, "max")

    def copy(self: ArrowExtensionArrayT) -> ArrowExtensionArrayT:
        """
        Return a shallow copy of the array.

        Underlying ChunkedArray is immutable, so a deep copy is unnecessary.

        Returns
        -------
        type(self)
        """
        return type(self)(self._data)

    def dropna(self: ArrowExtensionArrayT) -> ArrowExtensionArrayT:
        """
        Return ArrowExtensionArray without NA values.

        Returns
        -------
        ArrowExtensionArray
        """
        if pa_version_under6p0:
            fallback_performancewarning(version="6")
            return super().dropna()
        else:
            return type(self)(pc.drop_null(self._data))

    def isin(self, values) -> npt.NDArray[np.bool_]:
        if pa_version_under2p0:
            fallback_performancewarning(version="2")
            return super().isin(values)

        # for an empty value_set pyarrow 3.0.0 segfaults and pyarrow 2.0.0 returns True
        # for null values, so we short-circuit to return all False array.
        if not len(values):
            return np.zeros(len(self), dtype=bool)

        kwargs = {}
        if pa_version_under3p0:
            # in pyarrow 2.0.0 skip_null is ignored but is a required keyword and raises
            # with unexpected keyword argument in pyarrow 3.0.0+
            kwargs["skip_null"] = True

        result = pc.is_in(
            self._data, value_set=pa.array(values, from_pandas=True), **kwargs
        )
        # pyarrow 2.0.0 returned nulls, so we explicitly specify dtype to convert nulls
        # to False
        return np.array(result, dtype=np.bool_)

    def _values_for_factorize(self) -> tuple[np.ndarray, Any]:
        """
        Return an array and missing value suitable for factorization.

        Returns
        -------
        values : ndarray
        na_value : pd.NA

        Notes
        -----
        The values returned by this method are also used in
        :func:`pandas.util.hash_pandas_object`.
        """
        if pa_version_under2p0:
            values = self._data.to_pandas().values
        else:
            values = self._data.to_numpy()
        return values, self.dtype.na_value

    @doc(ExtensionArray.factorize)
    def factorize(
        self,
        na_sentinel: int | lib.NoDefault = lib.no_default,
        use_na_sentinel: bool | lib.NoDefault = lib.no_default,
    ) -> tuple[np.ndarray, ExtensionArray]:
        resolved_na_sentinel = resolve_na_sentinel(na_sentinel, use_na_sentinel)
        if pa_version_under4p0:
            encoded = self._data.dictionary_encode()
        else:
            null_encoding = "mask" if resolved_na_sentinel is not None else "encode"
            encoded = self._data.dictionary_encode(null_encoding=null_encoding)
        indices = pa.chunked_array(
            [c.indices for c in encoded.chunks], type=encoded.type.index_type
        ).to_pandas()
        if indices.dtype.kind == "f":
            indices[np.isnan(indices)] = (
                resolved_na_sentinel if resolved_na_sentinel is not None else -1
            )
        indices = indices.astype(np.int64, copy=False)

        if encoded.num_chunks:
            uniques = type(self)(encoded.chunk(0).dictionary)
            if resolved_na_sentinel is None and pa_version_under4p0:
                # TODO: share logic with BaseMaskedArray.factorize
                # Insert na with the proper code
                na_mask = indices.values == -1
                na_index = na_mask.argmax()
                if na_mask[na_index]:
                    na_code = 0 if na_index == 0 else indices[:na_index].max() + 1
                    uniques = uniques.insert(na_code, self.dtype.na_value)
                    indices[indices >= na_code] += 1
                    indices[indices == -1] = na_code
        else:
            uniques = type(self)(pa.array([], type=encoded.type.value_type))

        return indices.values, uniques

    def reshape(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self)} does not support reshape "
            f"as backed by a 1D pyarrow.ChunkedArray."
        )

    def take(
        self,
        indices: TakeIndexer,
        allow_fill: bool = False,
        fill_value: Any = None,
    ) -> ArrowExtensionArray:
        """
        Take elements from an array.

        Parameters
        ----------
        indices : sequence of int or one-dimensional np.ndarray of int
            Indices to be taken.
        allow_fill : bool, default False
            How to handle negative values in `indices`.

            * False: negative values in `indices` indicate positional indices
              from the right (the default). This is similar to
              :func:`numpy.take`.

            * True: negative values in `indices` indicate
              missing values. These values are set to `fill_value`. Any other
              other negative values raise a ``ValueError``.

        fill_value : any, optional
            Fill value to use for NA-indices when `allow_fill` is True.
            This may be ``None``, in which case the default NA value for
            the type, ``self.dtype.na_value``, is used.

            For many ExtensionArrays, there will be two representations of
            `fill_value`: a user-facing "boxed" scalar, and a low-level
            physical NA value. `fill_value` should be the user-facing version,
            and the implementation should handle translating that to the
            physical version for processing the take if necessary.

        Returns
        -------
        ExtensionArray

        Raises
        ------
        IndexError
            When the indices are out of bounds for the array.
        ValueError
            When `indices` contains negative values other than ``-1``
            and `allow_fill` is True.

        See Also
        --------
        numpy.take
        api.extensions.take

        Notes
        -----
        ExtensionArray.take is called by ``Series.__getitem__``, ``.loc``,
        ``iloc``, when `indices` is a sequence of values. Additionally,
        it's called by :meth:`Series.reindex`, or any other method
        that causes realignment, with a `fill_value`.
        """
        # TODO: Remove once we got rid of the (indices < 0) check
        if not is_array_like(indices):
            indices_array = np.asanyarray(indices)
        else:
            # error: Incompatible types in assignment (expression has type
            # "Sequence[int]", variable has type "ndarray")
            indices_array = indices  # type: ignore[assignment]

        if len(self._data) == 0 and (indices_array >= 0).any():
            raise IndexError("cannot do a non-empty take")
        if indices_array.size > 0 and indices_array.max() >= len(self._data):
            raise IndexError("out of bounds value in 'indices'.")

        if allow_fill:
            fill_mask = indices_array < 0
            if fill_mask.any():
                validate_indices(indices_array, len(self._data))
                # TODO(ARROW-9433): Treat negative indices as NULL
                indices_array = pa.array(indices_array, mask=fill_mask)
                result = self._data.take(indices_array)
                if isna(fill_value):
                    return type(self)(result)
                # TODO: ArrowNotImplementedError: Function fill_null has no
                # kernel matching input types (array[string], scalar[string])
                result = type(self)(result)
                result[fill_mask] = fill_value
                return result
                # return type(self)(pc.fill_null(result, pa.scalar(fill_value)))
            else:
                # Nothing to fill
                return type(self)(self._data.take(indices))
        else:  # allow_fill=False
            # TODO(ARROW-9432): Treat negative indices as indices from the right.
            if (indices_array < 0).any():
                # Don't modify in-place
                indices_array = np.copy(indices_array)
                indices_array[indices_array < 0] += len(self._data)
            return type(self)(self._data.take(indices_array))

    def unique(self: ArrowExtensionArrayT) -> ArrowExtensionArrayT:
        """
        Compute the ArrowExtensionArray of unique values.

        Returns
        -------
        ArrowExtensionArray
        """
        if pa_version_under2p0:
            fallback_performancewarning(version="2")
            return super().unique()
        else:
            return type(self)(pc.unique(self._data))

    def value_counts(self, dropna: bool = True) -> Series:
        """
        Return a Series containing counts of each unique value.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of missing values.

        Returns
        -------
        counts : Series

        See Also
        --------
        Series.value_counts
        """
        from pandas import (
            Index,
            Series,
        )

        vc = self._data.value_counts()

        values = vc.field(0)
        counts = vc.field(1)
        if dropna and self._data.null_count > 0:
            mask = values.is_valid()
            values = values.filter(mask)
            counts = counts.filter(mask)

        # No missing values so we can adhere to the interface and return a numpy array.
        counts = np.array(counts)

        index = Index(type(self)(values))

        return Series(counts, index=index).astype("Int64")

    @classmethod
    def _concat_same_type(
        cls: type[ArrowExtensionArrayT], to_concat
    ) -> ArrowExtensionArrayT:
        """
        Concatenate multiple ArrowExtensionArrays.

        Parameters
        ----------
        to_concat : sequence of ArrowExtensionArrays

        Returns
        -------
        ArrowExtensionArray
        """
        chunks = [array for ea in to_concat for array in ea._data.iterchunks()]
        arr = pa.chunked_array(chunks)
        return cls(arr)

    def _reduce(self, name: str, *, skipna: bool = True, **kwargs):
        """
        Return a scalar result of performing the reduction operation.

        Parameters
        ----------
        name : str
            Name of the function, supported values are:
            { any, all, min, max, sum, mean, median, prod,
            std, var, sem, kurt, skew }.
        skipna : bool, default True
            If True, skip NaN values.
        **kwargs
            Additional keyword arguments passed to the reduction function.
            Currently, `ddof` is the only supported kwarg.

        Returns
        -------
        scalar

        Raises
        ------
        TypeError : subclass does not define reductions
        """
        if name == "sem":

            def pyarrow_meth(data, skipna, **kwargs):
                numerator = pc.stddev(data, skip_nulls=skipna, **kwargs)
                denominator = pc.sqrt_checked(
                    pc.subtract_checked(
                        pc.count(self._data, skip_nulls=skipna), kwargs["ddof"]
                    )
                )
                return pc.divide_checked(numerator, denominator)

        else:
            pyarrow_name = {
                "median": "approximate_median",
                "prod": "product",
                "std": "stddev",
                "var": "variance",
            }.get(name, name)
            # error: Incompatible types in assignment
            # (expression has type "Optional[Any]", variable has type
            # "Callable[[Any, Any, KwArg(Any)], Any]")
            pyarrow_meth = getattr(pc, pyarrow_name, None)  # type: ignore[assignment]
            if pyarrow_meth is None:
                # Let ExtensionArray._reduce raise the TypeError
                return super()._reduce(name, skipna=skipna, **kwargs)
        try:
            result = pyarrow_meth(self._data, skip_nulls=skipna, **kwargs)
        except (AttributeError, NotImplementedError, TypeError) as err:
            msg = (
                f"'{type(self).__name__}' with dtype {self.dtype} "
                f"does not support reduction '{name}' with pyarrow "
                f"version {pa.__version__}. '{name}' may be supported by "
                f"upgrading pyarrow."
            )
            raise TypeError(msg) from err
        if pc.is_null(result).as_py():
            return self.dtype.na_value
        return result.as_py()

    def __setitem__(self, key: int | slice | np.ndarray, value: Any) -> None:
        """Set one or more values inplace.

        Parameters
        ----------
        key : int, ndarray, or slice
            When called from, e.g. ``Series.__setitem__``, ``key`` will be
            one of

            * scalar int
            * ndarray of integers.
            * boolean ndarray
            * slice object

        value : ExtensionDtype.type, Sequence[ExtensionDtype.type], or object
            value or values to be set of ``key``.

        Returns
        -------
        None
        """
        key = check_array_indexer(self, key)
        indices = self._indexing_key_to_indices(key)
        value = self._maybe_convert_setitem_value(value)

        argsort = np.argsort(indices)
        indices = indices[argsort]

        if is_scalar(value):
            value = np.broadcast_to(value, len(self))
        elif len(indices) != len(value):
            raise ValueError("Length of indexer and values mismatch")
        else:
            value = np.asarray(value)[argsort]

        self._data = self._set_via_chunk_iteration(indices=indices, value=value)

    def _indexing_key_to_indices(
        self, key: int | slice | np.ndarray
    ) -> npt.NDArray[np.intp]:
        """
        Convert indexing key for self into positional indices.

        Parameters
        ----------
        key : int | slice | np.ndarray

        Returns
        -------
        npt.NDArray[np.intp]
        """
        n = len(self)
        if isinstance(key, slice):
            indices = np.arange(n)[key]
        elif is_integer(key):
            # error: Invalid index type "List[Union[int, ndarray[Any, Any]]]"
            # for "ndarray[Any, dtype[signedinteger[Any]]]"; expected type
            # "Union[SupportsIndex, _SupportsArray[dtype[Union[bool_,
            # integer[Any]]]], _NestedSequence[_SupportsArray[dtype[Union
            # [bool_, integer[Any]]]]], _NestedSequence[Union[bool, int]]
            # , Tuple[Union[SupportsIndex, _SupportsArray[dtype[Union[bool_
            # , integer[Any]]]], _NestedSequence[_SupportsArray[dtype[Union
            # [bool_, integer[Any]]]]], _NestedSequence[Union[bool, int]]], ...]]"
            indices = np.arange(n)[[key]]  # type: ignore[index]
        elif is_bool_dtype(key):
            key = np.asarray(key)
            if len(key) != n:
                raise ValueError("Length of indexer and values mismatch")
            indices = key.nonzero()[0]
        else:
            key = np.asarray(key)
            indices = np.arange(n)[key]
        return indices

    # TODO: redefine _rank using pc.rank with pyarrow 9.0

    def _quantile(
        self: ArrowExtensionArrayT, qs: npt.NDArray[np.float64], interpolation: str
    ) -> ArrowExtensionArrayT:
        """
        Compute the quantiles of self for each quantile in `qs`.

        Parameters
        ----------
        qs : np.ndarray[float64]
        interpolation: str

        Returns
        -------
        same type as self
        """
        if pa_version_under4p0:
            raise NotImplementedError(
                "quantile only supported for pyarrow version >= 4.0"
            )
        result = pc.quantile(self._data, q=qs, interpolation=interpolation)
        return type(self)(result)

    def _mode(self: ArrowExtensionArrayT, dropna: bool = True) -> ArrowExtensionArrayT:
        """
        Returns the mode(s) of the ExtensionArray.

        Always returns `ExtensionArray` even if only one value.

        Parameters
        ----------
        dropna : bool, default True
            Don't consider counts of NA values.
            Not implemented by pyarrow.

        Returns
        -------
        same type as self
            Sorted, if possible.
        """
        if pa_version_under6p0:
            raise NotImplementedError("mode only supported for pyarrow version >= 6.0")
        modes = pc.mode(self._data, pc.count_distinct(self._data).as_py())
        values = modes.field(0)
        counts = modes.field(1)
        # counts sorted descending i.e counts[0] = max
        mask = pc.equal(counts, counts[0])
        most_common = values.filter(mask)
        return type(self)(most_common)

    def _maybe_convert_setitem_value(self, value):
        """Maybe convert value to be pyarrow compatible."""
        # TODO: Make more robust like ArrowStringArray._maybe_convert_setitem_value
        return value

    def _set_via_chunk_iteration(
        self, indices: npt.NDArray[np.intp], value: npt.NDArray[Any]
    ) -> pa.ChunkedArray:
        """
        Loop through the array chunks and set the new values while
        leaving the chunking layout unchanged.

        Parameters
        ----------
        indices : npt.NDArray[np.intp]
            Position indices for the underlying ChunkedArray.

        value : ExtensionDtype.type, Sequence[ExtensionDtype.type], or object
            value or values to be set of ``key``.

        Notes
        -----
        Assumes that indices is sorted. Caller is responsible for sorting.
        """
        new_data = []
        stop = 0
        for chunk in self._data.iterchunks():
            start, stop = stop, stop + len(chunk)
            if len(indices) == 0 or stop <= indices[0]:
                new_data.append(chunk)
            else:
                n = int(np.searchsorted(indices, stop, side="left"))
                c_ind = indices[:n] - start
                indices = indices[n:]
                n = len(c_ind)
                c_value, value = value[:n], value[n:]
                new_data.append(self._replace_with_indices(chunk, c_ind, c_value))
        return pa.chunked_array(new_data)

    @classmethod
    def _replace_with_indices(
        cls,
        chunk: pa.Array,
        indices: npt.NDArray[np.intp],
        value: npt.NDArray[Any],
    ) -> pa.Array:
        """
        Replace items selected with a set of positional indices.

        Analogous to pyarrow.compute.replace_with_mask, except that replacement
        positions are identified via indices rather than a mask.

        Parameters
        ----------
        chunk : pa.Array
        indices : npt.NDArray[np.intp]
        value : npt.NDArray[Any]
            Replacement value(s).

        Returns
        -------
        pa.Array
        """
        n = len(indices)

        if n == 0:
            return chunk

        start, stop = indices[[0, -1]]

        if (stop - start) == (n - 1):
            # fast path for a contiguous set of indices
            arrays = [
                chunk[:start],
                pa.array(value, type=chunk.type, from_pandas=True),
                chunk[stop + 1 :],
            ]
            arrays = [arr for arr in arrays if len(arr)]
            if len(arrays) == 1:
                return arrays[0]
            return pa.concat_arrays(arrays)

        mask = np.zeros(len(chunk), dtype=np.bool_)
        mask[indices] = True

        if pa_version_under5p0:
            arr = chunk.to_numpy(zero_copy_only=False)
            arr[mask] = value
            return pa.array(arr, type=chunk.type)

        if isna(value).all():
            return pc.if_else(mask, None, chunk)

        return pc.replace_with_mask(chunk, mask, value)

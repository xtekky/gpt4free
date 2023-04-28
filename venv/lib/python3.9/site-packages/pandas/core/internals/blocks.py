from __future__ import annotations

from functools import wraps
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Sequence,
    cast,
    final,
)
import warnings

import numpy as np

from pandas._libs import (
    Timestamp,
    internals as libinternals,
    lib,
    writers,
)
from pandas._libs.internals import BlockPlacement
from pandas._libs.tslibs import IncompatibleFrequency
from pandas._typing import (
    ArrayLike,
    DtypeObj,
    F,
    IgnoreRaise,
    Shape,
    npt,
)
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_bool_kwarg

from pandas.core.dtypes.astype import astype_array_safe
from pandas.core.dtypes.cast import (
    LossySetitemError,
    can_hold_element,
    find_result_type,
    maybe_downcast_to_dtype,
    np_can_hold_element,
    soft_convert_objects,
)
from pandas.core.dtypes.common import (
    ensure_platform_int,
    is_1d_only_ea_dtype,
    is_1d_only_ea_obj,
    is_dtype_equal,
    is_interval_dtype,
    is_list_like,
    is_sparse,
    is_string_dtype,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    ExtensionDtype,
    PandasDtype,
    PeriodDtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCIndex,
    ABCPandasArray,
    ABCSeries,
)
from pandas.core.dtypes.inference import is_inferred_bool_dtype
from pandas.core.dtypes.missing import (
    is_valid_na_for_dtype,
    isna,
    na_value_for_dtype,
)

import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import (
    extract_bool_array,
    putmask_inplace,
    putmask_without_repeat,
    setitem_datetimelike_compat,
    validate_putmask,
)
from pandas.core.array_algos.quantile import quantile_compat
from pandas.core.array_algos.replace import (
    compare_or_regex_search,
    replace_regex,
    should_use_regex,
)
from pandas.core.array_algos.transforms import shift
from pandas.core.arrays import (
    Categorical,
    DatetimeArray,
    ExtensionArray,
    IntervalArray,
    PandasArray,
    PeriodArray,
    TimedeltaArray,
)
from pandas.core.arrays.sparse import SparseDtype
from pandas.core.base import PandasObject
import pandas.core.common as com
import pandas.core.computation.expressions as expressions
from pandas.core.construction import (
    ensure_wrapped_if_datetimelike,
    extract_array,
)
from pandas.core.indexers import check_setitem_lengths
import pandas.core.missing as missing

if TYPE_CHECKING:
    from pandas import (
        Float64Index,
        Index,
    )
    from pandas.core.arrays._mixins import NDArrayBackedExtensionArray

# comparison is faster than is_object_dtype
_dtype_obj = np.dtype("object")


def maybe_split(meth: F) -> F:
    """
    If we have a multi-column block, split and operate block-wise.  Otherwise
    use the original method.
    """

    @wraps(meth)
    def newfunc(self, *args, **kwargs) -> list[Block]:

        if self.ndim == 1 or self.shape[0] == 1:
            return meth(self, *args, **kwargs)
        else:
            # Split and operate column-by-column
            return self.split_and_operate(meth, *args, **kwargs)

    return cast(F, newfunc)


class Block(PandasObject):
    """
    Canonical n-dimensional unit of homogeneous dtype contained in a pandas
    data structure

    Index-ignorant; let the container take care of that
    """

    values: np.ndarray | ExtensionArray
    ndim: int
    __init__: Callable

    __slots__ = ()
    is_numeric = False
    is_object = False
    is_extension = False
    _can_consolidate = True
    _validate_ndim = True

    @final
    @cache_readonly
    def _consolidate_key(self):
        return self._can_consolidate, self.dtype.name

    @final
    @cache_readonly
    def _can_hold_na(self) -> bool:
        """
        Can we store NA values in this Block?
        """
        dtype = self.dtype
        if isinstance(dtype, np.dtype):
            return dtype.kind not in ["b", "i", "u"]
        return dtype._can_hold_na

    @final
    @cache_readonly
    def is_categorical(self) -> bool:
        warnings.warn(
            "Block.is_categorical is deprecated and will be removed in a "
            "future version.  Use isinstance(block.values, Categorical) "
            "instead. See https://github.com/pandas-dev/pandas/issues/40226",
            DeprecationWarning,
            stacklevel=find_stack_level(),
        )
        return isinstance(self.values, Categorical)

    @final
    @property
    def is_bool(self) -> bool:
        """
        We can be bool if a) we are bool dtype or b) object dtype with bool objects.
        """
        return is_inferred_bool_dtype(self.values)

    @final
    def external_values(self):
        return external_values(self.values)

    @final
    @cache_readonly
    def fill_value(self):
        # Used in reindex_indexer
        return na_value_for_dtype(self.dtype, compat=False)

    @final
    def _standardize_fill_value(self, value):
        # if we are passed a scalar None, convert it here
        if self.dtype != _dtype_obj and is_valid_na_for_dtype(value, self.dtype):
            value = self.fill_value
        return value

    @property
    def mgr_locs(self) -> BlockPlacement:
        return self._mgr_locs

    @mgr_locs.setter
    def mgr_locs(self, new_mgr_locs: BlockPlacement) -> None:
        self._mgr_locs = new_mgr_locs

    @final
    def make_block(self, values, placement=None) -> Block:
        """
        Create a new block, with type inference propagate any values that are
        not specified
        """
        if placement is None:
            placement = self._mgr_locs
        if self.is_extension:
            values = ensure_block_shape(values, ndim=self.ndim)

        # TODO: perf by not going through new_block
        # We assume maybe_coerce_values has already been called
        return new_block(values, placement=placement, ndim=self.ndim)

    @final
    def make_block_same_class(
        self, values, placement: BlockPlacement | None = None
    ) -> Block:
        """Wrap given values in a block of same type as self."""
        if placement is None:
            placement = self._mgr_locs

        if values.dtype.kind in ["m", "M"]:

            new_values = ensure_wrapped_if_datetimelike(values)
            if new_values is not values:
                # TODO(2.0): remove once fastparquet has stopped relying on it
                warnings.warn(
                    "In a future version, Block.make_block_same_class will "
                    "assume that datetime64 and timedelta64 ndarrays have "
                    "already been cast to DatetimeArray and TimedeltaArray, "
                    "respectively.",
                    DeprecationWarning,
                    stacklevel=find_stack_level(),
                )
            values = new_values

        # We assume maybe_coerce_values has already been called
        return type(self)(values, placement=placement, ndim=self.ndim)

    @final
    def __repr__(self) -> str:
        # don't want to print out all of the items here
        name = type(self).__name__
        if self.ndim == 1:
            result = f"{name}: {len(self)} dtype: {self.dtype}"
        else:

            shape = " x ".join([str(s) for s in self.shape])
            result = f"{name}: {self.mgr_locs.indexer}, {shape}, dtype: {self.dtype}"

        return result

    @final
    def __len__(self) -> int:
        return len(self.values)

    @final
    def getitem_block(self, slicer: slice | npt.NDArray[np.intp]) -> Block:
        """
        Perform __getitem__-like, return result as block.

        Only supports slices that preserve dimensionality.
        """
        # Note: the only place where we are called with ndarray[intp]
        #  is from internals.concat, and we can verify that never happens
        #  with 1-column blocks, i.e. never for ExtensionBlock.

        # Invalid index type "Union[slice, ndarray[Any, dtype[signedinteger[Any]]]]"
        # for "BlockPlacement"; expected type "Union[slice, Sequence[int]]"
        new_mgr_locs = self._mgr_locs[slicer]  # type: ignore[index]

        new_values = self._slice(slicer)

        if new_values.ndim != self.values.ndim:
            raise ValueError("Only same dim slicing is allowed")

        return type(self)(new_values, new_mgr_locs, self.ndim)

    @final
    def getitem_block_columns(
        self, slicer: slice, new_mgr_locs: BlockPlacement
    ) -> Block:
        """
        Perform __getitem__-like, return result as block.

        Only supports slices that preserve dimensionality.
        """
        new_values = self._slice(slicer)

        if new_values.ndim != self.values.ndim:
            raise ValueError("Only same dim slicing is allowed")

        return type(self)(new_values, new_mgr_locs, self.ndim)

    @final
    def _can_hold_element(self, element: Any) -> bool:
        """require the same dtype as ourselves"""
        element = extract_array(element, extract_numpy=True)
        return can_hold_element(self.values, element)

    @final
    def should_store(self, value: ArrayLike) -> bool:
        """
        Should we set self.values[indexer] = value inplace or do we need to cast?

        Parameters
        ----------
        value : np.ndarray or ExtensionArray

        Returns
        -------
        bool
        """
        # faster equivalent to is_dtype_equal(value.dtype, self.dtype)
        try:
            return value.dtype == self.dtype
        except TypeError:
            return False

    # ---------------------------------------------------------------------
    # Apply/Reduce and Helpers

    @final
    def apply(self, func, **kwargs) -> list[Block]:
        """
        apply the function to my values; return a block if we are not
        one
        """
        result = func(self.values, **kwargs)

        return self._split_op_result(result)

    def reduce(self, func, ignore_failures: bool = False) -> list[Block]:
        # We will apply the function and reshape the result into a single-row
        #  Block with the same mgr_locs; squeezing will be done at a higher level
        assert self.ndim == 2

        try:
            result = func(self.values)
        except (TypeError, NotImplementedError):
            if ignore_failures:
                return []
            raise

        if self.values.ndim == 1:
            # TODO(EA2D): special case not needed with 2D EAs
            res_values = np.array([[result]])
        else:
            res_values = result.reshape(-1, 1)

        nb = self.make_block(res_values)
        return [nb]

    @final
    def _split_op_result(self, result: ArrayLike) -> list[Block]:
        # See also: split_and_operate
        if result.ndim > 1 and isinstance(result.dtype, ExtensionDtype):
            # TODO(EA2D): unnecessary with 2D EAs
            # if we get a 2D ExtensionArray, we need to split it into 1D pieces
            nbs = []
            for i, loc in enumerate(self._mgr_locs):
                if not is_1d_only_ea_obj(result):
                    vals = result[i : i + 1]
                else:
                    vals = result[i]

                block = self.make_block(values=vals, placement=loc)
                nbs.append(block)
            return nbs

        nb = self.make_block(result)

        return [nb]

    @final
    def _split(self) -> list[Block]:
        """
        Split a block into a list of single-column blocks.
        """
        assert self.ndim == 2

        new_blocks = []
        for i, ref_loc in enumerate(self._mgr_locs):
            vals = self.values[slice(i, i + 1)]

            bp = BlockPlacement(ref_loc)
            nb = type(self)(vals, placement=bp, ndim=2)
            new_blocks.append(nb)
        return new_blocks

    @final
    def split_and_operate(self, func, *args, **kwargs) -> list[Block]:
        """
        Split the block and apply func column-by-column.

        Parameters
        ----------
        func : Block method
        *args
        **kwargs

        Returns
        -------
        List[Block]
        """
        assert self.ndim == 2 and self.shape[0] != 1

        res_blocks = []
        for nb in self._split():
            rbs = func(nb, *args, **kwargs)
            res_blocks.extend(rbs)
        return res_blocks

    # ---------------------------------------------------------------------
    # Up/Down-casting

    @final
    def coerce_to_target_dtype(self, other) -> Block:
        """
        coerce the current block to a dtype compat for other
        we will return a block, possibly object, and not raise

        we can also safely try to coerce to the same dtype
        and will receive the same block
        """
        new_dtype = find_result_type(self.values, other)

        return self.astype(new_dtype, copy=False)

    @final
    def _maybe_downcast(self, blocks: list[Block], downcast=None) -> list[Block]:
        if downcast is False:
            return blocks

        if self.dtype == _dtype_obj:
            # GH#44241 We downcast regardless of the argument;
            #  respecting 'downcast=None' may be worthwhile at some point,
            #  but ATM it breaks too much existing code.
            # split and convert the blocks

            return extend_blocks(
                [blk.convert(datetime=True, numeric=False) for blk in blocks]
            )

        if downcast is None:
            return blocks

        return extend_blocks([b._downcast_2d(downcast) for b in blocks])

    @final
    @maybe_split
    def _downcast_2d(self, dtype) -> list[Block]:
        """
        downcast specialized to 2D case post-validation.

        Refactored to allow use of maybe_split.
        """
        new_values = maybe_downcast_to_dtype(self.values, dtype=dtype)
        return [self.make_block(new_values)]

    def convert(
        self,
        copy: bool = True,
        datetime: bool = True,
        numeric: bool = True,
        timedelta: bool = True,
    ) -> list[Block]:
        """
        attempt to coerce any object types to better types return a copy
        of the block (if copy = True) by definition we are not an ObjectBlock
        here!
        """
        return [self.copy()] if copy else [self]

    # ---------------------------------------------------------------------
    # Array-Like Methods

    @cache_readonly
    def dtype(self) -> DtypeObj:
        return self.values.dtype

    @final
    def astype(
        self, dtype: DtypeObj, copy: bool = False, errors: IgnoreRaise = "raise"
    ) -> Block:
        """
        Coerce to the new dtype.

        Parameters
        ----------
        dtype : np.dtype or ExtensionDtype
        copy : bool, default False
            copy if indicated
        errors : str, {'raise', 'ignore'}, default 'raise'
            - ``raise`` : allow exceptions to be raised
            - ``ignore`` : suppress exceptions. On error return original object

        Returns
        -------
        Block
        """
        values = self.values

        new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)

        new_values = maybe_coerce_values(new_values)
        newb = self.make_block(new_values)
        if newb.shape != self.shape:
            raise TypeError(
                f"cannot set astype for copy = [{copy}] for dtype "
                f"({self.dtype.name} [{self.shape}]) to different shape "
                f"({newb.dtype.name} [{newb.shape}])"
            )
        return newb

    @final
    def to_native_types(self, na_rep="nan", quoting=None, **kwargs) -> Block:
        """convert to our native types format"""
        result = to_native_types(self.values, na_rep=na_rep, quoting=quoting, **kwargs)
        return self.make_block(result)

    @final
    def copy(self, deep: bool = True) -> Block:
        """copy constructor"""
        values = self.values
        if deep:
            values = values.copy()
        return type(self)(values, placement=self._mgr_locs, ndim=self.ndim)

    # ---------------------------------------------------------------------
    # Replace

    @final
    def replace(
        self,
        to_replace,
        value,
        inplace: bool = False,
        # mask may be pre-computed if we're called from replace_list
        mask: npt.NDArray[np.bool_] | None = None,
    ) -> list[Block]:
        """
        replace the to_replace value with value, possible to create new
        blocks here this is just a call to putmask.
        """

        # Note: the checks we do in NDFrame.replace ensure we never get
        #  here with listlike to_replace or value, as those cases
        #  go through replace_list
        values = self.values

        if isinstance(values, Categorical):
            # TODO: avoid special-casing
            blk = self if inplace else self.copy()
            # error: Item "ExtensionArray" of "Union[ndarray[Any, Any],
            # ExtensionArray]" has no attribute "_replace"
            blk.values._replace(  # type: ignore[union-attr]
                to_replace=to_replace, value=value, inplace=True
            )
            return [blk]

        if not self._can_hold_element(to_replace):
            # We cannot hold `to_replace`, so we know immediately that
            #  replacing it is a no-op.
            # Note: If to_replace were a list, NDFrame.replace would call
            #  replace_list instead of replace.
            return [self] if inplace else [self.copy()]

        if mask is None:
            mask = missing.mask_missing(values, to_replace)
        if not mask.any():
            # Note: we get here with test_replace_extension_other incorrectly
            #  bc _can_hold_element is incorrect.
            return [self] if inplace else [self.copy()]

        elif self._can_hold_element(value):
            blk = self if inplace else self.copy()
            putmask_inplace(blk.values, mask, value)
            if not (self.is_object and value is None):
                # if the user *explicitly* gave None, we keep None, otherwise
                #  may downcast to NaN
                blocks = blk.convert(numeric=False, copy=False)
            else:
                blocks = [blk]
            return blocks

        elif self.ndim == 1 or self.shape[0] == 1:
            if value is None:
                blk = self.astype(np.dtype(object))
            else:
                blk = self.coerce_to_target_dtype(value)
            return blk.replace(
                to_replace=to_replace,
                value=value,
                inplace=True,
                mask=mask,
            )

        else:
            # split so that we only upcast where necessary
            blocks = []
            for i, nb in enumerate(self._split()):
                blocks.extend(
                    type(self).replace(
                        nb,
                        to_replace=to_replace,
                        value=value,
                        inplace=True,
                        mask=mask[i : i + 1],
                    )
                )
            return blocks

    @final
    def _replace_regex(
        self,
        to_replace,
        value,
        inplace: bool = False,
        convert: bool = True,
        mask=None,
    ) -> list[Block]:
        """
        Replace elements by the given value.

        Parameters
        ----------
        to_replace : object or pattern
            Scalar to replace or regular expression to match.
        value : object
            Replacement object.
        inplace : bool, default False
            Perform inplace modification.
        convert : bool, default True
            If true, try to coerce any object types to better types.
        mask : array-like of bool, optional
            True indicate corresponding element is ignored.

        Returns
        -------
        List[Block]
        """
        if not self._can_hold_element(to_replace):
            # i.e. only ObjectBlock, but could in principle include a
            #  String ExtensionBlock
            return [self] if inplace else [self.copy()]

        rx = re.compile(to_replace)

        new_values = self.values if inplace else self.values.copy()
        replace_regex(new_values, rx, value, mask)

        block = self.make_block(new_values)
        return block.convert(numeric=False, copy=False)

    @final
    def replace_list(
        self,
        src_list: Iterable[Any],
        dest_list: Sequence[Any],
        inplace: bool = False,
        regex: bool = False,
    ) -> list[Block]:
        """
        See BlockManager.replace_list docstring.
        """
        values = self.values

        # Exclude anything that we know we won't contain
        pairs = [
            (x, y) for x, y in zip(src_list, dest_list) if self._can_hold_element(x)
        ]
        if not len(pairs):
            # shortcut, nothing to replace
            return [self] if inplace else [self.copy()]

        src_len = len(pairs) - 1

        if is_string_dtype(values.dtype):
            # Calculate the mask once, prior to the call of comp
            # in order to avoid repeating the same computations
            mask = ~isna(values)
            masks = [
                compare_or_regex_search(values, s[0], regex=regex, mask=mask)
                for s in pairs
            ]
        else:
            # GH#38086 faster if we know we dont need to check for regex
            masks = [missing.mask_missing(values, s[0]) for s in pairs]

        # error: Argument 1 to "extract_bool_array" has incompatible type
        # "Union[ExtensionArray, ndarray, bool]"; expected "Union[ExtensionArray,
        # ndarray]"
        masks = [extract_bool_array(x) for x in masks]  # type: ignore[arg-type]

        rb = [self if inplace else self.copy()]
        for i, (src, dest) in enumerate(pairs):
            convert = i == src_len  # only convert once at the end
            new_rb: list[Block] = []

            # GH-39338: _replace_coerce can split a block into
            # single-column blocks, so track the index so we know
            # where to index into the mask
            for blk_num, blk in enumerate(rb):
                if len(rb) == 1:
                    m = masks[i]
                else:
                    mib = masks[i]
                    assert not isinstance(mib, bool)
                    m = mib[blk_num : blk_num + 1]

                # error: Argument "mask" to "_replace_coerce" of "Block" has
                # incompatible type "Union[ExtensionArray, ndarray[Any, Any], bool]";
                # expected "ndarray[Any, dtype[bool_]]"
                result = blk._replace_coerce(
                    to_replace=src,
                    value=dest,
                    mask=m,  # type: ignore[arg-type]
                    inplace=inplace,
                    regex=regex,
                )
                if convert and blk.is_object and not all(x is None for x in dest_list):
                    # GH#44498 avoid unwanted cast-back
                    result = extend_blocks(
                        [b.convert(numeric=False, copy=True) for b in result]
                    )
                new_rb.extend(result)
            rb = new_rb
        return rb

    @final
    def _replace_coerce(
        self,
        to_replace,
        value,
        mask: npt.NDArray[np.bool_],
        inplace: bool = True,
        regex: bool = False,
    ) -> list[Block]:
        """
        Replace value corresponding to the given boolean array with another
        value.

        Parameters
        ----------
        to_replace : object or pattern
            Scalar to replace or regular expression to match.
        value : object
            Replacement object.
        mask : np.ndarray[bool]
            True indicate corresponding element is ignored.
        inplace : bool, default True
            Perform inplace modification.
        regex : bool, default False
            If true, perform regular expression substitution.

        Returns
        -------
        List[Block]
        """
        if should_use_regex(regex, to_replace):
            return self._replace_regex(
                to_replace,
                value,
                inplace=inplace,
                convert=False,
                mask=mask,
            )
        else:
            if value is None:
                # gh-45601, gh-45836, gh-46634
                if mask.any():
                    nb = self.astype(np.dtype(object), copy=False)
                    if nb is self and not inplace:
                        nb = nb.copy()
                    putmask_inplace(nb.values, mask, value)
                    return [nb]
                return [self] if inplace else [self.copy()]
            return self.replace(
                to_replace=to_replace, value=value, inplace=inplace, mask=mask
            )

    # ---------------------------------------------------------------------
    # 2D Methods - Shared by NumpyBlock and NDArrayBackedExtensionBlock
    #  but not ExtensionBlock

    def _maybe_squeeze_arg(self, arg: np.ndarray) -> np.ndarray:
        """
        For compatibility with 1D-only ExtensionArrays.
        """
        return arg

    def _unwrap_setitem_indexer(self, indexer):
        """
        For compatibility with 1D-only ExtensionArrays.
        """
        return indexer

    # NB: this cannot be made cache_readonly because in mgr.set_values we pin
    #  new .values that can have different shape GH#42631
    @property
    def shape(self) -> Shape:
        return self.values.shape

    def iget(self, i: int | tuple[int, int] | tuple[slice, int]) -> np.ndarray:
        # In the case where we have a tuple[slice, int], the slice will always
        #  be slice(None)
        # Note: only reached with self.ndim == 2
        # Invalid index type "Union[int, Tuple[int, int], Tuple[slice, int]]"
        # for "Union[ndarray[Any, Any], ExtensionArray]"; expected type
        # "Union[int, integer[Any]]"
        return self.values[i]  # type: ignore[index]

    def _slice(
        self, slicer: slice | npt.NDArray[np.bool_] | npt.NDArray[np.intp]
    ) -> ArrayLike:
        """return a slice of my values"""

        return self.values[slicer]

    def set_inplace(self, locs, values: ArrayLike, copy: bool = False) -> None:
        """
        Modify block values in-place with new item value.

        If copy=True, first copy the underlying values in place before modifying
        (for Copy-on-Write).

        Notes
        -----
        `set_inplace` never creates a new array or new Block, whereas `setitem`
        _may_ create a new array and always creates a new Block.

        Caller is responsible for checking values.dtype == self.dtype.
        """
        if copy:
            self.values = self.values.copy()
        self.values[locs] = values

    def take_nd(
        self,
        indexer: npt.NDArray[np.intp],
        axis: int,
        new_mgr_locs: BlockPlacement | None = None,
        fill_value=lib.no_default,
    ) -> Block:
        """
        Take values according to indexer and return them as a block.
        """
        values = self.values

        if fill_value is lib.no_default:
            fill_value = self.fill_value
            allow_fill = False
        else:
            allow_fill = True

        # Note: algos.take_nd has upcast logic similar to coerce_to_target_dtype
        new_values = algos.take_nd(
            values, indexer, axis=axis, allow_fill=allow_fill, fill_value=fill_value
        )

        # Called from three places in managers, all of which satisfy
        #  this assertion
        assert not (axis == 0 and new_mgr_locs is None)
        if new_mgr_locs is None:
            new_mgr_locs = self._mgr_locs

        if not is_dtype_equal(new_values.dtype, self.dtype):
            return self.make_block(new_values, new_mgr_locs)
        else:
            return self.make_block_same_class(new_values, new_mgr_locs)

    def _unstack(
        self,
        unstacker,
        fill_value,
        new_placement: npt.NDArray[np.intp],
        needs_masking: npt.NDArray[np.bool_],
    ):
        """
        Return a list of unstacked blocks of self

        Parameters
        ----------
        unstacker : reshape._Unstacker
        fill_value : int
            Only used in ExtensionBlock._unstack
        new_placement : np.ndarray[np.intp]
        allow_fill : bool
        needs_masking : np.ndarray[bool]

        Returns
        -------
        blocks : list of Block
            New blocks of unstacked values.
        mask : array-like of bool
            The mask of columns of `blocks` we should keep.
        """
        new_values, mask = unstacker.get_new_values(
            self.values.T, fill_value=fill_value
        )

        mask = mask.any(0)
        # TODO: in all tests we have mask.all(); can we rely on that?

        # Note: these next two lines ensure that
        #  mask.sum() == sum(len(nb.mgr_locs) for nb in blocks)
        #  which the calling function needs in order to pass verify_integrity=False
        #  to the BlockManager constructor
        new_values = new_values.T[mask]
        new_placement = new_placement[mask]

        bp = BlockPlacement(new_placement)
        blocks = [new_block_2d(new_values, placement=bp)]
        return blocks, mask

    # ---------------------------------------------------------------------

    def setitem(self, indexer, value) -> Block:
        """
        Attempt self.values[indexer] = value, possibly creating a new array.

        Parameters
        ----------
        indexer : tuple, list-like, array-like, slice, int
            The subset of self.values to set
        value : object
            The value being set

        Returns
        -------
        Block

        Notes
        -----
        `indexer` is a direct slice/positional indexer. `value` must
        be a compatible shape.
        """

        value = self._standardize_fill_value(value)

        values = cast(np.ndarray, self.values)
        if self.ndim == 2:
            values = values.T

        # length checking
        check_setitem_lengths(indexer, value, values)

        value = extract_array(value, extract_numpy=True)
        try:
            casted = np_can_hold_element(values.dtype, value)
        except LossySetitemError:
            # current dtype cannot store value, coerce to common dtype
            nb = self.coerce_to_target_dtype(value)
            return nb.setitem(indexer, value)
        else:
            if self.dtype == _dtype_obj:
                # TODO: avoid having to construct values[indexer]
                vi = values[indexer]
                if lib.is_list_like(vi):
                    # checking lib.is_scalar here fails on
                    #  test_iloc_setitem_custom_object
                    casted = setitem_datetimelike_compat(values, len(vi), casted)
            values[indexer] = casted
        return self

    def putmask(self, mask, new) -> list[Block]:
        """
        putmask the data to the block; it is possible that we may create a
        new dtype of block

        Return the resulting block(s).

        Parameters
        ----------
        mask : np.ndarray[bool], SparseArray[bool], or BooleanArray
        new : a ndarray/object

        Returns
        -------
        List[Block]
        """
        orig_mask = mask
        values = cast(np.ndarray, self.values)
        mask, noop = validate_putmask(values.T, mask)
        assert not isinstance(new, (ABCIndex, ABCSeries, ABCDataFrame))

        if new is lib.no_default:
            new = self.fill_value

        new = self._standardize_fill_value(new)
        new = extract_array(new, extract_numpy=True)

        if noop:
            return [self]

        try:
            casted = np_can_hold_element(values.dtype, new)
            putmask_without_repeat(values.T, mask, casted)
            return [self]
        except LossySetitemError:

            if self.ndim == 1 or self.shape[0] == 1:
                # no need to split columns

                if not is_list_like(new):
                    # using just new[indexer] can't save us the need to cast
                    return self.coerce_to_target_dtype(new).putmask(mask, new)
                else:
                    indexer = mask.nonzero()[0]
                    nb = self.setitem(indexer, new[indexer])
                    return [nb]

            else:
                is_array = isinstance(new, np.ndarray)

                res_blocks = []
                nbs = self._split()
                for i, nb in enumerate(nbs):
                    n = new
                    if is_array:
                        # we have a different value per-column
                        n = new[:, i : i + 1]

                    submask = orig_mask[:, i : i + 1]
                    rbs = nb.putmask(submask, n)
                    res_blocks.extend(rbs)
                return res_blocks

    def where(self, other, cond, _downcast="infer") -> list[Block]:
        """
        evaluate the block; return result block(s) from the result

        Parameters
        ----------
        other : a ndarray/object
        cond : np.ndarray[bool], SparseArray[bool], or BooleanArray
        _downcast : str or None, default "infer"
            Private because we only specify it when calling from fillna.

        Returns
        -------
        List[Block]
        """
        assert cond.ndim == self.ndim
        assert not isinstance(other, (ABCIndex, ABCSeries, ABCDataFrame))

        transpose = self.ndim == 2

        cond = extract_bool_array(cond)

        # EABlocks override where
        values = cast(np.ndarray, self.values)
        orig_other = other
        if transpose:
            values = values.T

        icond, noop = validate_putmask(values, ~cond)
        if noop:
            # GH-39595: Always return a copy; short-circuit up/downcasting
            return [self.copy()]

        if other is lib.no_default:
            other = self.fill_value

        other = self._standardize_fill_value(other)

        try:
            # try/except here is equivalent to a self._can_hold_element check,
            #  but this gets us back 'casted' which we will re-use below;
            #  without using 'casted', expressions.where may do unwanted upcasts.
            casted = np_can_hold_element(values.dtype, other)
        except (ValueError, TypeError, LossySetitemError):
            # we cannot coerce, return a compat dtype

            if self.ndim == 1 or self.shape[0] == 1:
                # no need to split columns

                block = self.coerce_to_target_dtype(other)
                blocks = block.where(orig_other, cond)
                return self._maybe_downcast(blocks, downcast=_downcast)

            else:
                # since _maybe_downcast would split blocks anyway, we
                #  can avoid some potential upcast/downcast by splitting
                #  on the front end.
                is_array = isinstance(other, (np.ndarray, ExtensionArray))

                res_blocks = []
                nbs = self._split()
                for i, nb in enumerate(nbs):
                    oth = other
                    if is_array:
                        # we have a different value per-column
                        oth = other[:, i : i + 1]

                    submask = cond[:, i : i + 1]
                    rbs = nb.where(oth, submask, _downcast=_downcast)
                    res_blocks.extend(rbs)
                return res_blocks

        else:
            other = casted
            alt = setitem_datetimelike_compat(values, icond.sum(), other)
            if alt is not other:
                if is_list_like(other) and len(other) < len(values):
                    # call np.where with other to get the appropriate ValueError
                    np.where(~icond, values, other)
                    raise NotImplementedError(
                        "This should not be reached; call to np.where above is "
                        "expected to raise ValueError. Please report a bug at "
                        "github.com/pandas-dev/pandas"
                    )
                result = values.copy()
                np.putmask(result, icond, alt)
            else:
                # By the time we get here, we should have all Series/Index
                #  args extracted to ndarray
                if (
                    is_list_like(other)
                    and not isinstance(other, np.ndarray)
                    and len(other) == self.shape[-1]
                ):
                    # If we don't do this broadcasting here, then expressions.where
                    #  will broadcast a 1D other to be row-like instead of
                    #  column-like.
                    other = np.array(other).reshape(values.shape)
                    # If lengths don't match (or len(other)==1), we will raise
                    #  inside expressions.where, see test_series_where

                # Note: expressions.where may upcast.
                result = expressions.where(~icond, values, other)
                # The np_can_hold_element check _should_ ensure that we always
                #  have result.dtype == self.dtype here.

        if transpose:
            result = result.T

        return [self.make_block(result)]

    def fillna(
        self, value, limit: int | None = None, inplace: bool = False, downcast=None
    ) -> list[Block]:
        """
        fillna on the block with the value. If we fail, then convert to
        ObjectBlock and try again
        """
        # Caller is responsible for validating limit; if int it is strictly positive
        inplace = validate_bool_kwarg(inplace, "inplace")

        if not self._can_hold_na:
            # can short-circuit the isna call
            noop = True
        else:
            mask = isna(self.values)
            mask, noop = validate_putmask(self.values, mask)

        if noop:
            # we can't process the value, but nothing to do
            if inplace:
                # Arbitrarily imposing the convention that we ignore downcast
                #  on no-op when inplace=True
                return [self]
            else:
                # GH#45423 consistent downcasting on no-ops.
                nb = self.copy()
                nbs = nb._maybe_downcast([nb], downcast=downcast)
                return nbs

        if limit is not None:
            mask[mask.cumsum(self.ndim - 1) > limit] = False

        if inplace:
            nbs = self.putmask(mask.T, value)
        else:
            # without _downcast, we would break
            #  test_fillna_dtype_conversion_equiv_replace
            nbs = self.where(value, ~mask.T, _downcast=False)

        # Note: blk._maybe_downcast vs self._maybe_downcast(nbs)
        #  makes a difference bc blk may have object dtype, which has
        #  different behavior in _maybe_downcast.
        return extend_blocks(
            [blk._maybe_downcast([blk], downcast=downcast) for blk in nbs]
        )

    def interpolate(
        self,
        method: str = "pad",
        axis: int = 0,
        index: Index | None = None,
        inplace: bool = False,
        limit: int | None = None,
        limit_direction: str = "forward",
        limit_area: str | None = None,
        fill_value: Any | None = None,
        downcast: str | None = None,
        **kwargs,
    ) -> list[Block]:

        inplace = validate_bool_kwarg(inplace, "inplace")

        if not self._can_hold_na:
            # If there are no NAs, then interpolate is a no-op
            return [self] if inplace else [self.copy()]

        try:
            m = missing.clean_fill_method(method)
        except ValueError:
            m = None
        if m is None and self.dtype.kind != "f":
            # only deal with floats
            # bc we already checked that can_hold_na, we dont have int dtype here
            # test_interp_basic checks that we make a copy here
            return [self] if inplace else [self.copy()]

        if self.is_object and self.ndim == 2 and self.shape[0] != 1 and axis == 0:
            # split improves performance in ndarray.copy()
            return self.split_and_operate(
                type(self).interpolate,
                method,
                axis,
                index,
                inplace,
                limit,
                limit_direction,
                limit_area,
                fill_value,
                downcast,
                **kwargs,
            )

        data = self.values if inplace else self.values.copy()
        data = cast(np.ndarray, data)  # bc overridden by ExtensionBlock

        missing.interpolate_array_2d(
            data,
            method=method,
            axis=axis,
            index=index,
            limit=limit,
            limit_direction=limit_direction,
            limit_area=limit_area,
            fill_value=fill_value,
            **kwargs,
        )

        nb = self.make_block_same_class(data)
        return nb._maybe_downcast([nb], downcast)

    def diff(self, n: int, axis: int = 1) -> list[Block]:
        """return block for the diff of the values"""
        new_values = algos.diff(self.values, n, axis=axis)
        return [self.make_block(values=new_values)]

    def shift(self, periods: int, axis: int = 0, fill_value: Any = None) -> list[Block]:
        """shift the block by periods, possibly upcast"""
        # convert integer to float if necessary. need to do a lot more than
        # that, handle boolean etc also

        # Note: periods is never 0 here, as that is handled at the top of
        #  NDFrame.shift.  If that ever changes, we can do a check for periods=0
        #  and possibly avoid coercing.

        if not lib.is_scalar(fill_value) and self.dtype != _dtype_obj:
            # with object dtype there is nothing to promote, and the user can
            #  pass pretty much any weird fill_value they like
            # see test_shift_object_non_scalar_fill
            raise ValueError("fill_value must be a scalar")

        fill_value = self._standardize_fill_value(fill_value)

        try:
            # error: Argument 1 to "np_can_hold_element" has incompatible type
            # "Union[dtype[Any], ExtensionDtype]"; expected "dtype[Any]"
            casted = np_can_hold_element(
                self.dtype, fill_value  # type: ignore[arg-type]
            )
        except LossySetitemError:
            nb = self.coerce_to_target_dtype(fill_value)
            return nb.shift(periods, axis=axis, fill_value=fill_value)

        else:
            values = cast(np.ndarray, self.values)
            new_values = shift(values, periods, axis, casted)
            return [self.make_block(new_values)]

    @final
    def quantile(
        self, qs: Float64Index, interpolation="linear", axis: int = 0
    ) -> Block:
        """
        compute the quantiles of the

        Parameters
        ----------
        qs : Float64Index
            List of the quantiles to be computed.
        interpolation : str, default 'linear'
            Type of interpolation.
        axis : int, default 0
            Axis to compute.

        Returns
        -------
        Block
        """
        # We should always have ndim == 2 because Series dispatches to DataFrame
        assert self.ndim == 2
        assert axis == 1  # only ever called this way
        assert is_list_like(qs)  # caller is responsible for this

        result = quantile_compat(self.values, np.asarray(qs._values), interpolation)
        # ensure_block_shape needed for cases where we start with EA and result
        #  is ndarray, e.g. IntegerArray, SparseArray
        result = ensure_block_shape(result, ndim=2)
        return new_block_2d(result, placement=self._mgr_locs)

    # ---------------------------------------------------------------------
    # Abstract Methods Overridden By EABackedBlock and NumpyBlock

    def delete(self, loc) -> Block:
        """
        Return a new Block with the given loc(s) deleted.
        """
        raise AbstractMethodError(self)

    @property
    def is_view(self) -> bool:
        """return a boolean if I am possibly a view"""
        raise AbstractMethodError(self)

    @property
    def array_values(self) -> ExtensionArray:
        """
        The array that Series.array returns. Always an ExtensionArray.
        """
        raise AbstractMethodError(self)

    def get_values(self, dtype: DtypeObj | None = None) -> np.ndarray:
        """
        return an internal format, currently just the ndarray
        this is often overridden to handle to_dense like operations
        """
        raise AbstractMethodError(self)

    def values_for_json(self) -> np.ndarray:
        raise AbstractMethodError(self)


class EABackedBlock(Block):
    """
    Mixin for Block subclasses backed by ExtensionArray.
    """

    values: ExtensionArray

    def setitem(self, indexer, value):
        """
        Attempt self.values[indexer] = value, possibly creating a new array.

        This differs from Block.setitem by not allowing setitem to change
        the dtype of the Block.

        Parameters
        ----------
        indexer : tuple, list-like, array-like, slice, int
            The subset of self.values to set
        value : object
            The value being set

        Returns
        -------
        Block

        Notes
        -----
        `indexer` is a direct slice/positional indexer. `value` must
        be a compatible shape.
        """
        orig_indexer = indexer
        orig_value = value

        indexer = self._unwrap_setitem_indexer(indexer)
        value = self._maybe_squeeze_arg(value)

        values = self.values
        if values.ndim == 2:
            # TODO(GH#45419): string[pyarrow] tests break if we transpose
            #  unconditionally
            values = values.T
        check_setitem_lengths(indexer, value, values)

        try:
            values[indexer] = value
        except (ValueError, TypeError) as err:
            _catch_deprecated_value_error(err)

            if is_interval_dtype(self.dtype):
                # see TestSetitemFloatIntervalWithIntIntervalValues
                nb = self.coerce_to_target_dtype(orig_value)
                return nb.setitem(orig_indexer, orig_value)

            elif isinstance(self, NDArrayBackedExtensionBlock):
                nb = self.coerce_to_target_dtype(orig_value)
                return nb.setitem(orig_indexer, orig_value)

            else:
                raise

        else:
            return self

    def where(self, other, cond, _downcast="infer") -> list[Block]:
        # _downcast private bc we only specify it when calling from fillna
        arr = self.values.T

        cond = extract_bool_array(cond)

        orig_other = other
        orig_cond = cond
        other = self._maybe_squeeze_arg(other)
        cond = self._maybe_squeeze_arg(cond)

        if other is lib.no_default:
            other = self.fill_value

        icond, noop = validate_putmask(arr, ~cond)
        if noop:
            # GH#44181, GH#45135
            # Avoid a) raising for Interval/PeriodDtype and b) unnecessary object upcast
            return [self.copy()]

        try:
            res_values = arr._where(cond, other).T
        except (ValueError, TypeError) as err:
            _catch_deprecated_value_error(err)

            if self.ndim == 1 or self.shape[0] == 1:

                if is_interval_dtype(self.dtype):
                    # TestSetitemFloatIntervalWithIntIntervalValues
                    blk = self.coerce_to_target_dtype(orig_other)
                    nbs = blk.where(orig_other, orig_cond)
                    return self._maybe_downcast(nbs, downcast=_downcast)

                elif isinstance(self, NDArrayBackedExtensionBlock):
                    # NB: not (yet) the same as
                    #  isinstance(values, NDArrayBackedExtensionArray)
                    blk = self.coerce_to_target_dtype(orig_other)
                    nbs = blk.where(orig_other, orig_cond)
                    return self._maybe_downcast(nbs, downcast=_downcast)

                else:
                    raise

            else:
                # Same pattern we use in Block.putmask
                is_array = isinstance(orig_other, (np.ndarray, ExtensionArray))

                res_blocks = []
                nbs = self._split()
                for i, nb in enumerate(nbs):
                    n = orig_other
                    if is_array:
                        # we have a different value per-column
                        n = orig_other[:, i : i + 1]

                    submask = orig_cond[:, i : i + 1]
                    rbs = nb.where(n, submask)
                    res_blocks.extend(rbs)
                return res_blocks

        nb = self.make_block_same_class(res_values)
        return [nb]

    def putmask(self, mask, new) -> list[Block]:
        """
        See Block.putmask.__doc__
        """
        mask = extract_bool_array(mask)

        values = self.values
        if values.ndim == 2:
            values = values.T

        orig_new = new
        orig_mask = mask
        new = self._maybe_squeeze_arg(new)
        mask = self._maybe_squeeze_arg(mask)

        if not mask.any():
            return [self]

        try:
            # Caller is responsible for ensuring matching lengths
            values._putmask(mask, new)
        except (TypeError, ValueError) as err:
            _catch_deprecated_value_error(err)

            if self.ndim == 1 or self.shape[0] == 1:

                if is_interval_dtype(self.dtype):
                    # Discussion about what we want to support in the general
                    #  case GH#39584
                    blk = self.coerce_to_target_dtype(orig_new)
                    return blk.putmask(orig_mask, orig_new)

                elif isinstance(self, NDArrayBackedExtensionBlock):
                    # NB: not (yet) the same as
                    #  isinstance(values, NDArrayBackedExtensionArray)
                    blk = self.coerce_to_target_dtype(orig_new)
                    return blk.putmask(orig_mask, orig_new)

                else:
                    raise

            else:
                # Same pattern we use in Block.putmask
                is_array = isinstance(orig_new, (np.ndarray, ExtensionArray))

                res_blocks = []
                nbs = self._split()
                for i, nb in enumerate(nbs):
                    n = orig_new
                    if is_array:
                        # we have a different value per-column
                        n = orig_new[:, i : i + 1]

                    submask = orig_mask[:, i : i + 1]
                    rbs = nb.putmask(submask, n)
                    res_blocks.extend(rbs)
                return res_blocks

        return [self]

    def fillna(
        self, value, limit: int | None = None, inplace: bool = False, downcast=None
    ) -> list[Block]:
        # Caller is responsible for validating limit; if int it is strictly positive

        if self.dtype.kind == "m":
            try:
                res_values = self.values.fillna(value, limit=limit)
            except (ValueError, TypeError):
                # GH#45746
                warnings.warn(
                    "The behavior of fillna with timedelta64[ns] dtype and "
                    f"an incompatible value ({type(value)}) is deprecated. "
                    "In a future version, this will cast to a common dtype "
                    "(usually object) instead of raising, matching the "
                    "behavior of other dtypes.",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
                raise
            else:
                res_blk = self.make_block(res_values)
                return [res_blk]

        # TODO: since this now dispatches to super, which in turn dispatches
        #  to putmask, it may *actually* respect 'inplace=True'. If so, add
        #  tests for this.
        return super().fillna(value, limit=limit, inplace=inplace, downcast=downcast)

    def delete(self, loc) -> Block:
        # This will be unnecessary if/when __array_function__ is implemented
        values = self.values.delete(loc)
        mgr_locs = self._mgr_locs.delete(loc)
        return type(self)(values, placement=mgr_locs, ndim=self.ndim)

    @cache_readonly
    def array_values(self) -> ExtensionArray:
        return self.values

    def get_values(self, dtype: DtypeObj | None = None) -> np.ndarray:
        """
        return object dtype as boxed values, such as Timestamps/Timedelta
        """
        values: ArrayLike = self.values
        if dtype == _dtype_obj:
            values = values.astype(object)
        # TODO(EA2D): reshape not needed with 2D EAs
        return np.asarray(values).reshape(self.shape)

    def values_for_json(self) -> np.ndarray:
        return np.asarray(self.values)

    def interpolate(
        self, method="pad", axis=0, inplace=False, limit=None, fill_value=None, **kwargs
    ):
        values = self.values
        if values.ndim == 2 and axis == 0:
            # NDArrayBackedExtensionArray.fillna assumes axis=1
            new_values = values.T.fillna(value=fill_value, method=method, limit=limit).T
        else:
            new_values = values.fillna(value=fill_value, method=method, limit=limit)
        return self.make_block_same_class(new_values)


class ExtensionBlock(libinternals.Block, EABackedBlock):
    """
    Block for holding extension types.

    Notes
    -----
    This holds all 3rd-party extension array types. It's also the immediate
    parent class for our internal extension types' blocks, CategoricalBlock.

    ExtensionArrays are limited to 1-D.
    """

    _can_consolidate = False
    _validate_ndim = False
    is_extension = True

    values: ExtensionArray

    @cache_readonly
    def shape(self) -> Shape:
        # TODO(EA2D): override unnecessary with 2D EAs
        if self.ndim == 1:
            return (len(self.values),)
        return len(self._mgr_locs), len(self.values)

    def iget(self, i: int | tuple[int, int] | tuple[slice, int]):
        # In the case where we have a tuple[slice, int], the slice will always
        #  be slice(None)
        # We _could_ make the annotation more specific, but mypy would
        #  complain about override mismatch:
        #  Literal[0] | tuple[Literal[0], int] | tuple[slice, int]

        # Note: only reached with self.ndim == 2

        if isinstance(i, tuple):
            # TODO(EA2D): unnecessary with 2D EAs
            col, loc = i
            if not com.is_null_slice(col) and col != 0:
                raise IndexError(f"{self} only contains one item")
            elif isinstance(col, slice):
                # the is_null_slice check above assures that col is slice(None)
                #  so what we want is a view on all our columns and row loc
                if loc < 0:
                    loc += len(self.values)
                # Note: loc:loc+1 vs [[loc]] makes a difference when called
                #  from fast_xs because we want to get a view back.
                return self.values[loc : loc + 1]
            return self.values[loc]
        else:
            if i != 0:
                raise IndexError(f"{self} only contains one item")
            return self.values

    def set_inplace(self, locs, values: ArrayLike, copy: bool = False) -> None:
        # When an ndarray, we should have locs.tolist() == [0]
        # When a BlockPlacement we should have list(locs) == [0]
        if copy:
            self.values = self.values.copy()
        self.values[:] = values

    def _maybe_squeeze_arg(self, arg):
        """
        If necessary, squeeze a (N, 1) ndarray to (N,)
        """
        # e.g. if we are passed a 2D mask for putmask
        if (
            isinstance(arg, (np.ndarray, ExtensionArray))
            and arg.ndim == self.values.ndim + 1
        ):
            # TODO(EA2D): unnecessary with 2D EAs
            assert arg.shape[1] == 1
            # error: No overload variant of "__getitem__" of "ExtensionArray"
            # matches argument type "Tuple[slice, int]"
            arg = arg[:, 0]  # type: ignore[call-overload]
        elif isinstance(arg, ABCDataFrame):
            # 2022-01-06 only reached for setitem
            # TODO: should we avoid getting here with DataFrame?
            assert arg.shape[1] == 1
            arg = arg._ixs(0, axis=1)._values

        return arg

    def _unwrap_setitem_indexer(self, indexer):
        """
        Adapt a 2D-indexer to our 1D values.

        This is intended for 'setitem', not 'iget' or '_slice'.
        """
        # TODO: ATM this doesn't work for iget/_slice, can we change that?

        if isinstance(indexer, tuple):
            # TODO(EA2D): not needed with 2D EAs
            #  Should never have length > 2.  Caller is responsible for checking.
            #  Length 1 is reached vis setitem_single_block and setitem_single_column
            #  each of which pass indexer=(pi,)
            if len(indexer) == 2:

                if all(isinstance(x, np.ndarray) and x.ndim == 2 for x in indexer):
                    # GH#44703 went through indexing.maybe_convert_ix
                    first, second = indexer
                    if not (
                        second.size == 1 and (second == 0).all() and first.shape[1] == 1
                    ):
                        raise NotImplementedError(
                            "This should not be reached. Please report a bug at "
                            "github.com/pandas-dev/pandas/"
                        )
                    indexer = first[:, 0]

                elif lib.is_integer(indexer[1]) and indexer[1] == 0:
                    # reached via setitem_single_block passing the whole indexer
                    indexer = indexer[0]

                elif com.is_null_slice(indexer[1]):
                    indexer = indexer[0]

                elif is_list_like(indexer[1]) and indexer[1][0] == 0:
                    indexer = indexer[0]

                else:
                    raise NotImplementedError(
                        "This should not be reached. Please report a bug at "
                        "github.com/pandas-dev/pandas/"
                    )
        return indexer

    @property
    def is_view(self) -> bool:
        """Extension arrays are never treated as views."""
        return False

    @cache_readonly
    def is_numeric(self):
        return self.values.dtype._is_numeric

    def take_nd(
        self,
        indexer: npt.NDArray[np.intp],
        axis: int = 0,
        new_mgr_locs: BlockPlacement | None = None,
        fill_value=lib.no_default,
    ) -> Block:
        """
        Take values according to indexer and return them as a block.
        """
        if fill_value is lib.no_default:
            fill_value = None

        # TODO(EA2D): special case not needed with 2D EAs
        # axis doesn't matter; we are really a single-dim object
        # but are passed the axis depending on the calling routing
        # if its REALLY axis 0, then this will be a reindex and not a take
        new_values = self.values.take(indexer, fill_value=fill_value, allow_fill=True)

        # Called from three places in managers, all of which satisfy
        #  this assertion
        assert not (self.ndim == 1 and new_mgr_locs is None)
        if new_mgr_locs is None:
            new_mgr_locs = self._mgr_locs

        return self.make_block_same_class(new_values, new_mgr_locs)

    def _slice(
        self, slicer: slice | npt.NDArray[np.bool_] | npt.NDArray[np.intp]
    ) -> ExtensionArray:
        """
        Return a slice of my values.

        Parameters
        ----------
        slicer : slice, ndarray[int], or ndarray[bool]
            Valid (non-reducing) indexer for self.values.

        Returns
        -------
        ExtensionArray
        """
        # Notes: ndarray[bool] is only reachable when via getitem_mgr, which
        #  is only for Series, i.e. self.ndim == 1.

        # return same dims as we currently have
        if self.ndim == 2:
            # reached via getitem_block via _slice_take_blocks_ax0
            # TODO(EA2D): won't be necessary with 2D EAs

            if not isinstance(slicer, slice):
                raise AssertionError(
                    "invalid slicing for a 1-ndim ExtensionArray", slicer
                )
            # GH#32959 only full-slicers along fake-dim0 are valid
            # TODO(EA2D): won't be necessary with 2D EAs
            # range(1) instead of self._mgr_locs to avoid exception on [::-1]
            #  see test_iloc_getitem_slice_negative_step_ea_block
            new_locs = range(1)[slicer]
            if not len(new_locs):
                raise AssertionError(
                    "invalid slicing for a 1-ndim ExtensionArray", slicer
                )
            slicer = slice(None)

        return self.values[slicer]

    @final
    def getitem_block_index(self, slicer: slice) -> ExtensionBlock:
        """
        Perform __getitem__-like specialized to slicing along index.
        """
        # GH#42787 in principle this is equivalent to values[..., slicer], but we don't
        # require subclasses of ExtensionArray to support that form (for now).
        new_values = self.values[slicer]
        return type(self)(new_values, self._mgr_locs, ndim=self.ndim)

    def diff(self, n: int, axis: int = 1) -> list[Block]:
        if axis == 0 and n != 0:
            # n==0 case will be a no-op so let is fall through
            # Since we only have one column, the result will be all-NA.
            #  Create this result by shifting along axis=0 past the length of
            #  our values.
            return super().diff(len(self.values), axis=0)
        if axis == 1:
            # TODO(EA2D): unnecessary with 2D EAs
            # we are by definition 1D.
            axis = 0
        return super().diff(n, axis)

    def shift(self, periods: int, axis: int = 0, fill_value: Any = None) -> list[Block]:
        """
        Shift the block by `periods`.

        Dispatches to underlying ExtensionArray and re-boxes in an
        ExtensionBlock.
        """
        new_values = self.values.shift(periods=periods, fill_value=fill_value)
        return [self.make_block_same_class(new_values)]

    def _unstack(
        self,
        unstacker,
        fill_value,
        new_placement: npt.NDArray[np.intp],
        needs_masking: npt.NDArray[np.bool_],
    ):
        # ExtensionArray-safe unstack.
        # We override ObjectBlock._unstack, which unstacks directly on the
        # values of the array. For EA-backed blocks, this would require
        # converting to a 2-D ndarray of objects.
        # Instead, we unstack an ndarray of integer positions, followed by
        # a `take` on the actual values.

        # Caller is responsible for ensuring self.shape[-1] == len(unstacker.index)
        new_values, mask = unstacker.arange_result

        # Note: these next two lines ensure that
        #  mask.sum() == sum(len(nb.mgr_locs) for nb in blocks)
        #  which the calling function needs in order to pass verify_integrity=False
        #  to the BlockManager constructor
        new_values = new_values.T[mask]
        new_placement = new_placement[mask]

        # needs_masking[i] calculated once in BlockManager.unstack tells
        #  us if there are any -1s in the relevant indices.  When False,
        #  that allows us to go through a faster path in 'take', among
        #  other things avoiding e.g. Categorical._validate_scalar.
        blocks = [
            # TODO: could cast to object depending on fill_value?
            type(self)(
                self.values.take(
                    indices, allow_fill=needs_masking[i], fill_value=fill_value
                ),
                BlockPlacement(place),
                ndim=2,
            )
            for i, (indices, place) in enumerate(zip(new_values, new_placement))
        ]
        return blocks, mask


class NumpyBlock(libinternals.NumpyBlock, Block):
    values: np.ndarray

    @property
    def is_view(self) -> bool:
        """return a boolean if I am possibly a view"""
        return self.values.base is not None

    @property
    def array_values(self) -> ExtensionArray:
        return PandasArray(self.values)

    def get_values(self, dtype: DtypeObj | None = None) -> np.ndarray:
        if dtype == _dtype_obj:
            return self.values.astype(_dtype_obj)
        return self.values

    def values_for_json(self) -> np.ndarray:
        return self.values

    def delete(self, loc) -> Block:
        values = np.delete(self.values, loc, 0)
        mgr_locs = self._mgr_locs.delete(loc)
        return type(self)(values, placement=mgr_locs, ndim=self.ndim)


class NumericBlock(NumpyBlock):
    __slots__ = ()
    is_numeric = True


class NDArrayBackedExtensionBlock(libinternals.NDArrayBackedBlock, EABackedBlock):
    """
    Block backed by an NDArrayBackedExtensionArray
    """

    values: NDArrayBackedExtensionArray

    # error: Signature of "is_extension" incompatible with supertype "Block"
    @cache_readonly
    def is_extension(self) -> bool:  # type: ignore[override]
        # i.e. datetime64tz, PeriodDtype
        return not isinstance(self.dtype, np.dtype)

    @property
    def is_view(self) -> bool:
        """return a boolean if I am possibly a view"""
        # check the ndarray values of the DatetimeIndex values
        return self.values._ndarray.base is not None

    def diff(self, n: int, axis: int = 0) -> list[Block]:
        """
        1st discrete difference.

        Parameters
        ----------
        n : int
            Number of periods to diff.
        axis : int, default 0
            Axis to diff upon.

        Returns
        -------
        A list with a new Block.

        Notes
        -----
        The arguments here are mimicking shift so they are called correctly
        by apply.
        """
        values = self.values

        new_values = values - values.shift(n, axis=axis)
        return [self.make_block(new_values)]

    def shift(self, periods: int, axis: int = 0, fill_value: Any = None) -> list[Block]:
        values = self.values
        new_values = values.shift(periods, fill_value=fill_value, axis=axis)
        return [self.make_block_same_class(new_values)]


def _catch_deprecated_value_error(err: Exception) -> None:
    """
    We catch ValueError for now, but only a specific one raised by DatetimeArray
    which will no longer be raised in version.2.0.
    """
    if isinstance(err, ValueError):
        # TODO(2.0): once DTA._validate_setitem_value deprecation
        #  is enforced, stop catching ValueError here altogether
        if isinstance(err, IncompatibleFrequency):
            pass
        elif "'value.closed' is" in str(err):
            # IntervalDtype mismatched 'closed'
            pass
        elif "Timezones don't match" not in str(err):
            raise


class DatetimeLikeBlock(NDArrayBackedExtensionBlock):
    """Block for datetime64[ns], timedelta64[ns]."""

    __slots__ = ()
    is_numeric = False
    values: DatetimeArray | TimedeltaArray

    def values_for_json(self) -> np.ndarray:
        return self.values._ndarray


class DatetimeTZBlock(DatetimeLikeBlock):
    """implement a datetime64 block with a tz attribute"""

    values: DatetimeArray

    __slots__ = ()
    is_extension = True
    _validate_ndim = True
    _can_consolidate = False

    # Don't use values_for_json from DatetimeLikeBlock since it is
    # an invalid optimization here(drop the tz)
    values_for_json = NDArrayBackedExtensionBlock.values_for_json


class ObjectBlock(NumpyBlock):
    __slots__ = ()
    is_object = True

    @maybe_split
    def reduce(self, func, ignore_failures: bool = False) -> list[Block]:
        """
        For object-dtype, we operate column-wise.
        """
        assert self.ndim == 2

        try:
            res = func(self.values)
        except TypeError:
            if not ignore_failures:
                raise
            return []

        assert isinstance(res, np.ndarray)
        assert res.ndim == 1
        res = res.reshape(1, -1)
        return [self.make_block_same_class(res)]

    @maybe_split
    def convert(
        self,
        copy: bool = True,
        datetime: bool = True,
        numeric: bool = True,
        timedelta: bool = True,
    ) -> list[Block]:
        """
        attempt to cast any object types to better types return a copy of
        the block (if copy = True) by definition we ARE an ObjectBlock!!!!!
        """
        values = self.values
        if values.ndim == 2:
            # maybe_split ensures we only get here with values.shape[0] == 1,
            # avoid doing .ravel as that might make a copy
            values = values[0]

        res_values = soft_convert_objects(
            values,
            datetime=datetime,
            numeric=numeric,
            timedelta=timedelta,
            copy=copy,
        )
        res_values = ensure_block_shape(res_values, self.ndim)
        return [self.make_block(res_values)]


class CategoricalBlock(ExtensionBlock):
    # this Block type is kept for backwards-compatibility
    __slots__ = ()

    # GH#43232, GH#43334 self.values.dtype can be changed inplace until 2.0,
    #  so this cannot be cached
    @property
    def dtype(self) -> DtypeObj:
        return self.values.dtype


# -----------------------------------------------------------------
# Constructor Helpers


def maybe_coerce_values(values: ArrayLike) -> ArrayLike:
    """
    Input validation for values passed to __init__. Ensure that
    any datetime64/timedelta64 dtypes are in nanoseconds.  Ensure
    that we do not have string dtypes.

    Parameters
    ----------
    values : np.ndarray or ExtensionArray

    Returns
    -------
    values : np.ndarray or ExtensionArray
    """
    # Caller is responsible for ensuring PandasArray is already extracted.

    if isinstance(values, np.ndarray):
        values = ensure_wrapped_if_datetimelike(values)

        if issubclass(values.dtype.type, str):
            values = np.array(values, dtype=object)

    if isinstance(values, (DatetimeArray, TimedeltaArray)) and values.freq is not None:
        # freq is only stored in DatetimeIndex/TimedeltaIndex, not in Series/DataFrame
        values = values._with_freq(None)

    return values


def get_block_type(dtype: DtypeObj):
    """
    Find the appropriate Block subclass to use for the given values and dtype.

    Parameters
    ----------
    dtype : numpy or pandas dtype

    Returns
    -------
    cls : class, subclass of Block
    """
    # We use vtype and kind checks because they are much more performant
    #  than is_foo_dtype
    vtype = dtype.type
    kind = dtype.kind

    cls: type[Block]

    if isinstance(dtype, SparseDtype):
        # Need this first(ish) so that Sparse[datetime] is sparse
        cls = ExtensionBlock
    elif isinstance(dtype, CategoricalDtype):
        cls = CategoricalBlock
    elif vtype is Timestamp:
        cls = DatetimeTZBlock
    elif isinstance(dtype, PeriodDtype):
        cls = NDArrayBackedExtensionBlock
    elif isinstance(dtype, ExtensionDtype):
        # Note: need to be sure PandasArray is unwrapped before we get here
        cls = ExtensionBlock

    elif kind in ["M", "m"]:
        cls = DatetimeLikeBlock
    elif kind in ["f", "c", "i", "u", "b"]:
        cls = NumericBlock
    else:
        cls = ObjectBlock
    return cls


def new_block_2d(values: ArrayLike, placement: BlockPlacement):
    # new_block specialized to case with
    #  ndim=2
    #  isinstance(placement, BlockPlacement)
    #  check_ndim/ensure_block_shape already checked
    klass = get_block_type(values.dtype)

    values = maybe_coerce_values(values)
    return klass(values, ndim=2, placement=placement)


def new_block(values, placement, *, ndim: int) -> Block:
    # caller is responsible for ensuring values is NOT a PandasArray

    if not isinstance(placement, BlockPlacement):
        placement = BlockPlacement(placement)

    check_ndim(values, placement, ndim)

    klass = get_block_type(values.dtype)

    values = maybe_coerce_values(values)
    return klass(values, ndim=ndim, placement=placement)


def check_ndim(values, placement: BlockPlacement, ndim: int) -> None:
    """
    ndim inference and validation.

    Validates that values.ndim and ndim are consistent.
    Validates that len(values) and len(placement) are consistent.

    Parameters
    ----------
    values : array-like
    placement : BlockPlacement
    ndim : int

    Raises
    ------
    ValueError : the number of dimensions do not match
    """

    if values.ndim > ndim:
        # Check for both np.ndarray and ExtensionArray
        raise ValueError(
            "Wrong number of dimensions. "
            f"values.ndim > ndim [{values.ndim} > {ndim}]"
        )

    elif not is_1d_only_ea_dtype(values.dtype):
        # TODO(EA2D): special case not needed with 2D EAs
        if values.ndim != ndim:
            raise ValueError(
                "Wrong number of dimensions. "
                f"values.ndim != ndim [{values.ndim} != {ndim}]"
            )
        if len(placement) != len(values):
            raise ValueError(
                f"Wrong number of items passed {len(values)}, "
                f"placement implies {len(placement)}"
            )
    elif ndim == 2 and len(placement) != 1:
        # TODO(EA2D): special case unnecessary with 2D EAs
        raise ValueError("need to split")


def extract_pandas_array(
    values: np.ndarray | ExtensionArray, dtype: DtypeObj | None, ndim: int
) -> tuple[np.ndarray | ExtensionArray, DtypeObj | None]:
    """
    Ensure that we don't allow PandasArray / PandasDtype in internals.
    """
    # For now, blocks should be backed by ndarrays when possible.
    if isinstance(values, ABCPandasArray):
        values = values.to_numpy()
        if ndim and ndim > 1:
            # TODO(EA2D): special case not needed with 2D EAs
            values = np.atleast_2d(values)

    if isinstance(dtype, PandasDtype):
        dtype = dtype.numpy_dtype

    return values, dtype


# -----------------------------------------------------------------


def extend_blocks(result, blocks=None) -> list[Block]:
    """return a new extended blocks, given the result"""
    if blocks is None:
        blocks = []
    if isinstance(result, list):
        for r in result:
            if isinstance(r, list):
                blocks.extend(r)
            else:
                blocks.append(r)
    else:
        assert isinstance(result, Block), type(result)
        blocks.append(result)
    return blocks


def ensure_block_shape(values: ArrayLike, ndim: int = 1) -> ArrayLike:
    """
    Reshape if possible to have values.ndim == ndim.
    """

    if values.ndim < ndim:
        if not is_1d_only_ea_dtype(values.dtype):
            # TODO(EA2D): https://github.com/pandas-dev/pandas/issues/23023
            # block.shape is incorrect for "2D" ExtensionArrays
            # We can't, and don't need to, reshape.
            values = cast("np.ndarray | DatetimeArray | TimedeltaArray", values)
            values = values.reshape(1, -1)

    return values


def to_native_types(
    values: ArrayLike,
    *,
    na_rep="nan",
    quoting=None,
    float_format=None,
    decimal=".",
    **kwargs,
) -> np.ndarray:
    """convert to our native types format"""
    if isinstance(values, Categorical) and values.categories.dtype.kind in "Mm":
        # GH#40754 Convert categorical datetimes to datetime array
        values = algos.take_nd(
            values.categories._values,
            ensure_platform_int(values._codes),
            fill_value=na_rep,
        )

    values = ensure_wrapped_if_datetimelike(values)

    if isinstance(values, (DatetimeArray, TimedeltaArray)):
        if values.ndim == 1:
            result = values._format_native_types(na_rep=na_rep, **kwargs)
            result = result.astype(object, copy=False)
            return result

        # GH#21734 Process every column separately, they might have different formats
        results_converted = []
        for i in range(len(values)):
            result = values[i, :]._format_native_types(na_rep=na_rep, **kwargs)
            results_converted.append(result.astype(object, copy=False))
        return np.vstack(results_converted)

    elif values.dtype.kind == "f" and not is_sparse(values):
        # see GH#13418: no special formatting is desired at the
        # output (important for appropriate 'quoting' behaviour),
        # so do not pass it through the FloatArrayFormatter
        if float_format is None and decimal == ".":
            mask = isna(values)

            if not quoting:
                values = values.astype(str)
            else:
                values = np.array(values, dtype="object")

            values[mask] = na_rep
            values = values.astype(object, copy=False)
            return values

        from pandas.io.formats.format import FloatArrayFormatter

        formatter = FloatArrayFormatter(
            values,
            na_rep=na_rep,
            float_format=float_format,
            decimal=decimal,
            quoting=quoting,
            fixed_width=False,
        )
        res = formatter.get_result_as_array()
        res = res.astype(object, copy=False)
        return res

    elif isinstance(values, ExtensionArray):
        mask = isna(values)

        new_values = np.asarray(values.astype(object))
        new_values[mask] = na_rep
        return new_values

    else:

        mask = isna(values)
        itemsize = writers.word_len(na_rep)

        if values.dtype != _dtype_obj and not quoting and itemsize:
            values = values.astype(str)
            if values.dtype.itemsize / np.dtype("U1").itemsize < itemsize:
                # enlarge for the na_rep
                values = values.astype(f"<U{itemsize}")
        else:
            values = np.array(values, dtype="object")

        values[mask] = na_rep
        values = values.astype(object, copy=False)
        return values


def external_values(values: ArrayLike) -> ArrayLike:
    """
    The array that Series.values returns (public attribute).

    This has some historical constraints, and is overridden in block
    subclasses to return the correct array (e.g. period returns
    object ndarray and datetimetz a datetime64[ns] ndarray instead of
    proper extension array).
    """
    if isinstance(values, (PeriodArray, IntervalArray)):
        return values.astype(object)
    elif isinstance(values, (DatetimeArray, TimedeltaArray)):
        # NB: for datetime64tz this is different from np.asarray(values), since
        #  that returns an object-dtype ndarray of Timestamps.
        # Avoid FutureWarning in .astype in casting from dt64tz to dt64
        return values._data
    else:
        return values

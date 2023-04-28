from __future__ import annotations

import functools
from typing import (
    TYPE_CHECKING,
    Callable,
)

import numpy as np

from pandas._typing import Scalar
from pandas.compat._optional import import_optional_dependency


@functools.lru_cache(maxsize=None)
def generate_shared_aggregator(
    func: Callable[..., Scalar],
    nopython: bool,
    nogil: bool,
    parallel: bool,
):
    """
    Generate a Numba function that loops over the columns 2D object and applies
    a 1D numba kernel over each column.

    Parameters
    ----------
    func : function
        aggregation function to be applied to each column
    nopython : bool
        nopython to be passed into numba.jit
    nogil : bool
        nogil to be passed into numba.jit
    parallel : bool
        parallel to be passed into numba.jit

    Returns
    -------
    Numba function
    """
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency("numba")

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def column_looper(
        values: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
        min_periods: int,
        *args,
    ):
        result = np.empty((len(start), values.shape[1]), dtype=np.float64)
        for i in numba.prange(values.shape[1]):
            result[:, i] = func(values[:, i], start, end, min_periods, *args)
        return result

    return column_looper

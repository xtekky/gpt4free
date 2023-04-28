"""
Utility functions and objects for implementing the interchange API.
"""

from __future__ import annotations

import re
import typing

import numpy as np

from pandas._typing import DtypeObj

import pandas as pd
from pandas.api.types import is_datetime64_dtype


class ArrowCTypes:
    """
    Enum for Apache Arrow C type format strings.

    The Arrow C data interface:
    https://arrow.apache.org/docs/format/CDataInterface.html#data-type-description-format-strings
    """

    NULL = "n"
    BOOL = "b"
    INT8 = "c"
    UINT8 = "C"
    INT16 = "s"
    UINT16 = "S"
    INT32 = "i"
    UINT32 = "I"
    INT64 = "l"
    UINT64 = "L"
    FLOAT16 = "e"
    FLOAT32 = "f"
    FLOAT64 = "g"
    STRING = "u"  # utf-8
    DATE32 = "tdD"
    DATE64 = "tdm"
    # Resoulution:
    #   - seconds -> 's'
    #   - milliseconds -> 'm'
    #   - microseconds -> 'u'
    #   - nanoseconds -> 'n'
    TIMESTAMP = "ts{resolution}:{tz}"
    TIME = "tt{resolution}"


class Endianness:
    """Enum indicating the byte-order of a data-type."""

    LITTLE = "<"
    BIG = ">"
    NATIVE = "="
    NA = "|"


def dtype_to_arrow_c_fmt(dtype: DtypeObj) -> str:
    """
    Represent pandas `dtype` as a format string in Apache Arrow C notation.

    Parameters
    ----------
    dtype : np.dtype
        Datatype of pandas DataFrame to represent.

    Returns
    -------
    str
        Format string in Apache Arrow C notation of the given `dtype`.
    """
    if isinstance(dtype, pd.CategoricalDtype):
        return ArrowCTypes.INT64
    elif dtype == np.dtype("O"):
        return ArrowCTypes.STRING

    format_str = getattr(ArrowCTypes, dtype.name.upper(), None)
    if format_str is not None:
        return format_str

    if is_datetime64_dtype(dtype):
        # Selecting the first char of resolution string:
        # dtype.str -> '<M8[ns]'
        resolution = re.findall(r"\[(.*)\]", typing.cast(np.dtype, dtype).str)[0][:1]
        return ArrowCTypes.TIMESTAMP.format(resolution=resolution, tz="")

    raise NotImplementedError(
        f"Conversion of {dtype} to Arrow C format string is not implemented."
    )


class NoBufferPresent(Exception):
    """Exception to signal that there is no requested buffer."""

import numpy as np
import pytest

import pandas as pd
from pandas.core.interchange.utils import dtype_to_arrow_c_fmt

# TODO: use ArrowSchema to get reference C-string.
# At the time, there is no way to access ArrowSchema holding a type format string
# from python. The only way to access it is to export the structure to a C-pointer,
# see DataType._export_to_c() method defined in
# https://github.com/apache/arrow/blob/master/python/pyarrow/types.pxi


@pytest.mark.parametrize(
    "pandas_dtype, c_string",
    [
        (np.dtype("bool"), "b"),
        (np.dtype("int8"), "c"),
        (np.dtype("uint8"), "C"),
        (np.dtype("int16"), "s"),
        (np.dtype("uint16"), "S"),
        (np.dtype("int32"), "i"),
        (np.dtype("uint32"), "I"),
        (np.dtype("int64"), "l"),
        (np.dtype("uint64"), "L"),
        (np.dtype("float16"), "e"),
        (np.dtype("float32"), "f"),
        (np.dtype("float64"), "g"),
        (pd.Series(["a"]).dtype, "u"),
        (
            pd.Series([0]).astype("datetime64[ns]").dtype,
            "tsn:",
        ),
        (pd.CategoricalDtype(["a"]), "l"),
        (np.dtype("O"), "u"),
    ],
)
def test_dtype_to_arrow_c_fmt(pandas_dtype, c_string):  # PR01
    """Test ``dtype_to_arrow_c_fmt`` utility function."""
    assert dtype_to_arrow_c_fmt(pandas_dtype) == c_string

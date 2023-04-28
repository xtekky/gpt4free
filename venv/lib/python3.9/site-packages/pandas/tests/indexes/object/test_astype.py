import pytest

from pandas import (
    Index,
    NaT,
)
import pandas._testing as tm


def test_astype_str_from_bytes():
    # https://github.com/pandas-dev/pandas/issues/38607
    idx = Index(["あ", b"a"], dtype="object")
    result = idx.astype(str)
    expected = Index(["あ", "a"], dtype="object")
    tm.assert_index_equal(result, expected)


def test_astype_invalid_nas_to_tdt64_raises():
    # GH#45722 don't cast np.datetime64 NaTs to timedelta64 NaT
    idx = Index([NaT.asm8] * 2, dtype=object)

    msg = r"Cannot cast Index to dtype timedelta64\[ns\]"
    with pytest.raises(TypeError, match=msg):
        idx.astype("m8[ns]")

import pytest

from pandas import (
    DatetimeIndex,
    date_range,
)
import pandas._testing as tm


def astype_non_nano(dti_nano, unit):
    # TODO(2.0): remove once DTI/DTA.astype supports non-nano
    if unit == "ns":
        return dti_nano

    dta_nano = dti_nano._data
    arr_nano = dta_nano._ndarray

    arr = arr_nano.astype(f"M8[{unit}]")
    if dti_nano.tz is None:
        dtype = arr.dtype
    else:
        dtype = type(dti_nano.dtype)(tz=dti_nano.tz, unit=unit)
    dta = type(dta_nano)._simple_new(arr, dtype=dtype)
    dti = DatetimeIndex(dta, name=dti_nano.name)
    assert dti.dtype == dtype
    return dti


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("tz", [None, "Asia/Shanghai", "Europe/Berlin"])
@pytest.mark.parametrize("name", [None, "my_dti"])
@pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
def test_dti_snap(name, tz, unit):
    dti = DatetimeIndex(
        [
            "1/1/2002",
            "1/2/2002",
            "1/3/2002",
            "1/4/2002",
            "1/5/2002",
            "1/6/2002",
            "1/7/2002",
        ],
        name=name,
        tz=tz,
        freq="D",
    )
    dti = astype_non_nano(dti, unit)

    result = dti.snap(freq="W-MON")
    expected = date_range("12/31/2001", "1/7/2002", name=name, tz=tz, freq="w-mon")
    expected = expected.repeat([3, 4])
    expected = astype_non_nano(expected, unit)
    tm.assert_index_equal(result, expected)
    assert result.tz == expected.tz
    assert result.freq is None
    assert expected.freq is None

    result = dti.snap(freq="B")

    expected = date_range("1/1/2002", "1/7/2002", name=name, tz=tz, freq="b")
    expected = expected.repeat([1, 1, 1, 2, 2])
    expected = astype_non_nano(expected, unit)
    tm.assert_index_equal(result, expected)
    assert result.tz == expected.tz
    assert result.freq is None
    assert expected.freq is None

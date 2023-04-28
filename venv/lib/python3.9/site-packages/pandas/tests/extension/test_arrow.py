"""
This file contains a minimal set of tests for compliance with the extension
array interface test suite, and should contain no other tests.
The test suite for the full functionality of the array is located in
`pandas/tests/arrays/`.
The tests in this file are inherited from the BaseExtensionTests, and only
minimal tweaks should be applied to get the tests passing (by overwriting a
parent method).
Additional tests should either be added to one of the BaseExtensionTests
classes (if they are relevant for the extension interface for all dtypes), or
be added to the array-specific tests in `pandas/tests/arrays/`.
"""
from datetime import (
    date,
    datetime,
    time,
    timedelta,
)

import numpy as np
import pytest

from pandas.compat import (
    is_ci_environment,
    is_platform_windows,
    pa_version_under2p0,
    pa_version_under3p0,
    pa_version_under4p0,
    pa_version_under6p0,
    pa_version_under7p0,
    pa_version_under8p0,
    pa_version_under9p0,
)
from pandas.errors import PerformanceWarning

import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base

pa = pytest.importorskip("pyarrow", minversion="1.0.1")

from pandas.core.arrays.arrow.array import ArrowExtensionArray

from pandas.core.arrays.arrow.dtype import ArrowDtype  # isort:skip


@pytest.fixture(params=tm.ALL_PYARROW_DTYPES, ids=str)
def dtype(request):
    return ArrowDtype(pyarrow_dtype=request.param)


@pytest.fixture
def data(dtype):
    pa_dtype = dtype.pyarrow_dtype
    if pa.types.is_boolean(pa_dtype):
        data = [True, False] * 4 + [None] + [True, False] * 44 + [None] + [True, False]
    elif pa.types.is_floating(pa_dtype):
        data = [1.0, 0.0] * 4 + [None] + [-2.0, -1.0] * 44 + [None] + [0.5, 99.5]
    elif pa.types.is_signed_integer(pa_dtype):
        data = [1, 0] * 4 + [None] + [-2, -1] * 44 + [None] + [1, 99]
    elif pa.types.is_unsigned_integer(pa_dtype):
        data = [1, 0] * 4 + [None] + [2, 1] * 44 + [None] + [1, 99]
    elif pa.types.is_date(pa_dtype):
        data = (
            [date(2022, 1, 1), date(1999, 12, 31)] * 4
            + [None]
            + [date(2022, 1, 1), date(2022, 1, 1)] * 44
            + [None]
            + [date(1999, 12, 31), date(1999, 12, 31)]
        )
    elif pa.types.is_timestamp(pa_dtype):
        data = (
            [datetime(2020, 1, 1, 1, 1, 1, 1), datetime(1999, 1, 1, 1, 1, 1, 1)] * 4
            + [None]
            + [datetime(2020, 1, 1, 1), datetime(1999, 1, 1, 1)] * 44
            + [None]
            + [datetime(2020, 1, 1), datetime(1999, 1, 1)]
        )
    elif pa.types.is_duration(pa_dtype):
        data = (
            [timedelta(1), timedelta(1, 1)] * 4
            + [None]
            + [timedelta(-1), timedelta(0)] * 44
            + [None]
            + [timedelta(-10), timedelta(10)]
        )
    elif pa.types.is_time(pa_dtype):
        data = (
            [time(12, 0), time(0, 12)] * 4
            + [None]
            + [time(0, 0), time(1, 1)] * 44
            + [None]
            + [time(0, 5), time(5, 0)]
        )
    else:
        raise NotImplementedError
    return pd.array(data, dtype=dtype)


@pytest.fixture
def data_missing(data):
    """Length-2 array with [NA, Valid]"""
    return type(data)._from_sequence([None, data[0]])


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    """Parametrized fixture returning 'data' or 'data_missing' integer arrays.

    Used to test dtype conversion with and without missing values.
    """
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing


@pytest.fixture
def data_for_grouping(dtype):
    """
    Data for factorization, grouping, and unique tests.

    Expected to be like [B, B, NA, NA, A, A, B, C]

    Where A < B < C and NA is missing
    """
    pa_dtype = dtype.pyarrow_dtype
    if pa.types.is_boolean(pa_dtype):
        A = False
        B = True
        C = True
    elif pa.types.is_floating(pa_dtype):
        A = -1.1
        B = 0.0
        C = 1.1
    elif pa.types.is_signed_integer(pa_dtype):
        A = -1
        B = 0
        C = 1
    elif pa.types.is_unsigned_integer(pa_dtype):
        A = 0
        B = 1
        C = 10
    elif pa.types.is_date(pa_dtype):
        A = date(1999, 12, 31)
        B = date(2010, 1, 1)
        C = date(2022, 1, 1)
    elif pa.types.is_timestamp(pa_dtype):
        A = datetime(1999, 1, 1, 1, 1, 1, 1)
        B = datetime(2020, 1, 1)
        C = datetime(2020, 1, 1, 1)
    elif pa.types.is_duration(pa_dtype):
        A = timedelta(-1)
        B = timedelta(0)
        C = timedelta(1, 4)
    elif pa.types.is_time(pa_dtype):
        A = time(0, 0)
        B = time(0, 12)
        C = time(12, 12)
    else:
        raise NotImplementedError
    return pd.array([B, B, None, None, A, A, B, C], dtype=dtype)


@pytest.fixture
def data_for_sorting(data_for_grouping):
    """
    Length-3 array with a known sort order.

    This should be three items [B, C, A] with
    A < B < C
    """
    return type(data_for_grouping)._from_sequence(
        [data_for_grouping[0], data_for_grouping[7], data_for_grouping[4]]
    )


@pytest.fixture
def data_missing_for_sorting(data_for_grouping):
    """
    Length-3 array with a known sort order.

    This should be three items [B, NA, A] with
    A < B and NA missing.
    """
    return type(data_for_grouping)._from_sequence(
        [data_for_grouping[0], data_for_grouping[2], data_for_grouping[4]]
    )


@pytest.fixture
def data_for_twos(data):
    """Length-100 array in which all the elements are two."""
    pa_dtype = data.dtype.pyarrow_dtype
    if pa.types.is_integer(pa_dtype) or pa.types.is_floating(pa_dtype):
        return pd.array([2] * 100, dtype=data.dtype)
    # tests will be xfailed where 2 is not a valid scalar for pa_dtype
    return data


@pytest.fixture
def na_value():
    """The scalar missing value for this type. Default 'None'"""
    return pd.NA


class TestBaseCasting(base.BaseCastingTests):
    pass


class TestConstructors(base.BaseConstructorsTests):
    def test_from_dtype(self, data, request):
        pa_dtype = data.dtype.pyarrow_dtype
        if pa.types.is_timestamp(pa_dtype) and pa_dtype.tz:
            if pa_version_under2p0:
                request.node.add_marker(
                    pytest.mark.xfail(
                        reason=f"timestamp data with tz={pa_dtype.tz} "
                        "converted to integer when pyarrow < 2.0",
                    )
                )
            else:
                request.node.add_marker(
                    pytest.mark.xfail(
                        raises=NotImplementedError,
                        reason=f"pyarrow.type_for_alias cannot infer {pa_dtype}",
                    )
                )
        super().test_from_dtype(data)

    def test_from_sequence_pa_array(self, data, request):
        # https://github.com/pandas-dev/pandas/pull/47034#discussion_r955500784
        # data._data = pa.ChunkedArray
        if pa_version_under3p0:
            request.node.add_marker(
                pytest.mark.xfail(
                    reason="ChunkedArray has no attribute combine_chunks",
                )
            )
        result = type(data)._from_sequence(data._data)
        tm.assert_extension_array_equal(result, data)
        assert isinstance(result._data, pa.ChunkedArray)

        result = type(data)._from_sequence(data._data.combine_chunks())
        tm.assert_extension_array_equal(result, data)
        assert isinstance(result._data, pa.ChunkedArray)

    def test_from_sequence_pa_array_notimplemented(self, request):
        if pa_version_under6p0:
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=AttributeError,
                    reason="month_day_nano_interval not implemented by pyarrow.",
                )
            )
        with pytest.raises(NotImplementedError, match="Converting strings to"):
            ArrowExtensionArray._from_sequence_of_strings(
                ["12-1"], dtype=pa.month_day_nano_interval()
            )

    def test_from_sequence_of_strings_pa_array(self, data, request):
        pa_dtype = data.dtype.pyarrow_dtype
        if pa_version_under3p0:
            request.node.add_marker(
                pytest.mark.xfail(
                    reason="ChunkedArray has no attribute combine_chunks",
                )
            )
        elif pa.types.is_time64(pa_dtype) and pa_dtype.equals("time64[ns]"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason="Nanosecond time parsing not supported.",
                )
            )
        elif pa.types.is_duration(pa_dtype):
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=pa.ArrowNotImplementedError,
                    reason=f"pyarrow doesn't support parsing {pa_dtype}",
                )
            )
        elif pa.types.is_boolean(pa_dtype):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason="Iterating over ChunkedArray[bool] returns PyArrow scalars.",
                )
            )
        elif pa.types.is_timestamp(pa_dtype) and pa_dtype.tz is not None:
            if pa_version_under7p0:
                request.node.add_marker(
                    pytest.mark.xfail(
                        raises=pa.ArrowNotImplementedError,
                        reason=f"pyarrow doesn't support string cast from {pa_dtype}",
                    )
                )
            elif is_platform_windows() and is_ci_environment():
                request.node.add_marker(
                    pytest.mark.xfail(
                        raises=pa.ArrowInvalid,
                        reason=(
                            "TODO: Set ARROW_TIMEZONE_DATABASE environment variable "
                            "on CI to path to the tzdata for pyarrow."
                        ),
                    )
                )
        elif pa_version_under6p0 and pa.types.is_temporal(pa_dtype):
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=pa.ArrowNotImplementedError,
                    reason=f"pyarrow doesn't support string cast from {pa_dtype}",
                )
            )
        pa_array = data._data.cast(pa.string())
        result = type(data)._from_sequence_of_strings(pa_array, dtype=data.dtype)
        tm.assert_extension_array_equal(result, data)

        pa_array = pa_array.combine_chunks()
        result = type(data)._from_sequence_of_strings(pa_array, dtype=data.dtype)
        tm.assert_extension_array_equal(result, data)


class TestGetitemTests(base.BaseGetitemTests):
    @pytest.mark.xfail(
        reason=(
            "data.dtype.type return pyarrow.DataType "
            "but this (intentionally) returns "
            "Python scalars or pd.Na"
        )
    )
    def test_getitem_scalar(self, data):
        super().test_getitem_scalar(data)

    def test_take_series(self, request, data):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        unit = getattr(data.dtype.pyarrow_dtype, "unit", None)
        bad_units = ["ns"]
        if pa_version_under2p0:
            bad_units.extend(["s", "ms", "us"])
        if pa_version_under3p0 and tz not in (None, "UTC") and unit in bad_units:
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=(
                        f"Not supported by pyarrow < 3.0 "
                        f"with timestamp type {tz} and {unit}"
                    )
                )
            )
        super().test_take_series(data)

    def test_reindex(self, request, data, na_value):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        unit = getattr(data.dtype.pyarrow_dtype, "unit", None)
        bad_units = ["ns"]
        if pa_version_under2p0:
            bad_units.extend(["s", "ms", "us"])
        if pa_version_under3p0 and tz not in (None, "UTC") and unit in bad_units:
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=(
                        f"Not supported by pyarrow < 3.0 "
                        f"with timestamp type {tz} and {unit}"
                    )
                )
            )
        super().test_reindex(data, na_value)

    def test_loc_iloc_frame_single_dtype(self, request, using_array_manager, data):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        unit = getattr(data.dtype.pyarrow_dtype, "unit", None)
        bad_units = ["ns"]
        if pa_version_under2p0:
            bad_units.extend(["s", "ms", "us"])
        if (
            pa_version_under3p0
            and not using_array_manager
            and tz not in (None, "UTC")
            and unit in bad_units
        ):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=(
                        f"Not supported by pyarrow < 3.0 "
                        f"with timestamp type {tz} and {unit}"
                    )
                )
            )
        super().test_loc_iloc_frame_single_dtype(data)


class TestBaseNumericReduce(base.BaseNumericReduceTests):
    def check_reduce(self, ser, op_name, skipna):
        pa_dtype = ser.dtype.pyarrow_dtype
        result = getattr(ser, op_name)(skipna=skipna)
        if pa.types.is_boolean(pa_dtype):
            # Can't convert if ser contains NA
            pytest.skip(
                "pandas boolean data with NA does not fully support all reductions"
            )
        elif pa.types.is_integer(pa_dtype) or pa.types.is_floating(pa_dtype):
            ser = ser.astype("Float64")
        expected = getattr(ser, op_name)(skipna=skipna)
        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series(self, data, all_numeric_reductions, skipna, request):
        pa_dtype = data.dtype.pyarrow_dtype
        xfail_mark = pytest.mark.xfail(
            raises=TypeError,
            reason=(
                f"{all_numeric_reductions} is not implemented in "
                f"pyarrow={pa.__version__} for {pa_dtype}"
            ),
        )
        if all_numeric_reductions in {"skew", "kurt"}:
            request.node.add_marker(xfail_mark)
        elif (
            all_numeric_reductions in {"median", "var", "std", "prod", "max", "min"}
            and pa_version_under6p0
        ):
            request.node.add_marker(xfail_mark)
        elif all_numeric_reductions in {"sum", "mean"} and pa_version_under2p0:
            request.node.add_marker(xfail_mark)
        elif (
            all_numeric_reductions in {"sum", "mean"}
            and skipna is False
            and pa_version_under6p0
            and (pa.types.is_integer(pa_dtype) or pa.types.is_floating(pa_dtype))
        ):
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=AssertionError,
                    reason=(
                        f"{all_numeric_reductions} with skip_nulls={skipna} did not "
                        f"return NA for {pa_dtype} with pyarrow={pa.__version__}"
                    ),
                )
            )
        elif not (
            pa.types.is_integer(pa_dtype)
            or pa.types.is_floating(pa_dtype)
            or pa.types.is_boolean(pa_dtype)
        ) and not (
            all_numeric_reductions in {"min", "max"}
            and (pa.types.is_temporal(pa_dtype) and not pa.types.is_duration(pa_dtype))
        ):
            request.node.add_marker(xfail_mark)
        elif pa.types.is_boolean(pa_dtype) and all_numeric_reductions in {
            "std",
            "var",
            "median",
        }:
            request.node.add_marker(xfail_mark)
        super().test_reduce_series(data, all_numeric_reductions, skipna)


class TestBaseBooleanReduce(base.BaseBooleanReduceTests):
    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series(
        self, data, all_boolean_reductions, skipna, na_value, request
    ):
        pa_dtype = data.dtype.pyarrow_dtype
        xfail_mark = pytest.mark.xfail(
            raises=TypeError,
            reason=(
                f"{all_boolean_reductions} is not implemented in "
                f"pyarrow={pa.__version__} for {pa_dtype}"
            ),
        )
        if not pa.types.is_boolean(pa_dtype):
            request.node.add_marker(xfail_mark)
        elif pa_version_under3p0:
            request.node.add_marker(xfail_mark)
        op_name = all_boolean_reductions
        s = pd.Series(data)
        result = getattr(s, op_name)(skipna=skipna)
        assert result is (op_name == "any")


class TestBaseGroupby(base.BaseGroupbyTests):
    def test_groupby_agg_extension(self, data_for_grouping, request):
        tz = getattr(data_for_grouping.dtype.pyarrow_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=f"Not supported by pyarrow < 2.0 with timestamp type {tz}."
                )
            )
        super().test_groupby_agg_extension(data_for_grouping)

    def test_groupby_extension_no_sort(self, data_for_grouping, request):
        pa_dtype = data_for_grouping.dtype.pyarrow_dtype
        if pa.types.is_boolean(pa_dtype):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=f"{pa_dtype} only has 2 unique possible values",
                )
            )
        elif pa.types.is_duration(pa_dtype):
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=pa.ArrowNotImplementedError,
                    reason=f"pyarrow doesn't support factorizing {pa_dtype}",
                )
            )
        super().test_groupby_extension_no_sort(data_for_grouping)

    def test_groupby_extension_transform(self, data_for_grouping, request):
        pa_dtype = data_for_grouping.dtype.pyarrow_dtype
        if pa.types.is_boolean(pa_dtype):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=f"{pa_dtype} only has 2 unique possible values",
                )
            )
        elif pa.types.is_duration(pa_dtype):
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=pa.ArrowNotImplementedError,
                    reason=f"pyarrow doesn't support factorizing {pa_dtype}",
                )
            )
        super().test_groupby_extension_transform(data_for_grouping)

    def test_groupby_extension_apply(
        self, data_for_grouping, groupby_apply_op, request
    ):
        pa_dtype = data_for_grouping.dtype.pyarrow_dtype
        if pa.types.is_duration(pa_dtype):
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=pa.ArrowNotImplementedError,
                    reason=f"pyarrow doesn't support factorizing {pa_dtype}",
                )
            )
        with tm.maybe_produces_warning(
            PerformanceWarning, pa_version_under7p0, check_stacklevel=False
        ):
            super().test_groupby_extension_apply(data_for_grouping, groupby_apply_op)

    def test_in_numeric_groupby(self, data_for_grouping, request):
        pa_dtype = data_for_grouping.dtype.pyarrow_dtype
        if pa.types.is_integer(pa_dtype) or pa.types.is_floating(pa_dtype):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason="ArrowExtensionArray doesn't support .sum() yet.",
                )
            )
        super().test_in_numeric_groupby(data_for_grouping)

    @pytest.mark.parametrize("as_index", [True, False])
    def test_groupby_extension_agg(self, as_index, data_for_grouping, request):
        pa_dtype = data_for_grouping.dtype.pyarrow_dtype
        if pa.types.is_boolean(pa_dtype):
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=ValueError,
                    reason=f"{pa_dtype} only has 2 unique possible values",
                )
            )
        elif pa.types.is_duration(pa_dtype):
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=pa.ArrowNotImplementedError,
                    reason=f"pyarrow doesn't support factorizing {pa_dtype}",
                )
            )
        with tm.maybe_produces_warning(
            PerformanceWarning, pa_version_under7p0, check_stacklevel=False
        ):
            super().test_groupby_extension_agg(as_index, data_for_grouping)


class TestBaseDtype(base.BaseDtypeTests):
    def test_construct_from_string_own_name(self, dtype, request):
        pa_dtype = dtype.pyarrow_dtype
        if pa.types.is_timestamp(pa_dtype) and pa_dtype.tz is not None:
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=NotImplementedError,
                    reason=f"pyarrow.type_for_alias cannot infer {pa_dtype}",
                )
            )
        super().test_construct_from_string_own_name(dtype)

    def test_is_dtype_from_name(self, dtype, request):
        pa_dtype = dtype.pyarrow_dtype
        if pa.types.is_timestamp(pa_dtype) and pa_dtype.tz is not None:
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=NotImplementedError,
                    reason=f"pyarrow.type_for_alias cannot infer {pa_dtype}",
                )
            )
        super().test_is_dtype_from_name(dtype)

    def test_construct_from_string(self, dtype, request):
        pa_dtype = dtype.pyarrow_dtype
        if pa.types.is_timestamp(pa_dtype) and pa_dtype.tz is not None:
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=NotImplementedError,
                    reason=f"pyarrow.type_for_alias cannot infer {pa_dtype}",
                )
            )
        super().test_construct_from_string(dtype)

    def test_construct_from_string_another_type_raises(self, dtype):
        msg = r"'another_type' must end with '\[pyarrow\]'"
        with pytest.raises(TypeError, match=msg):
            type(dtype).construct_from_string("another_type")

    def test_get_common_dtype(self, dtype, request):
        pa_dtype = dtype.pyarrow_dtype
        if (
            pa.types.is_date(pa_dtype)
            or pa.types.is_time(pa_dtype)
            or (
                pa.types.is_timestamp(pa_dtype)
                and (pa_dtype.unit != "ns" or pa_dtype.tz is not None)
            )
            or (pa.types.is_duration(pa_dtype) and pa_dtype.unit != "ns")
        ):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=(
                        f"{pa_dtype} does not have associated numpy "
                        f"dtype findable by find_common_type"
                    )
                )
            )
        super().test_get_common_dtype(dtype)


class TestBaseIndex(base.BaseIndexTests):
    pass


class TestBaseInterface(base.BaseInterfaceTests):
    def test_contains(self, data, data_missing, request):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        unit = getattr(data.dtype.pyarrow_dtype, "unit", None)
        if pa_version_under2p0 and tz not in (None, "UTC") and unit == "us":
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=(
                        f"Not supported by pyarrow < 2.0 "
                        f"with timestamp type {tz} and {unit}"
                    )
                )
            )
        super().test_contains(data, data_missing)

    @pytest.mark.xfail(reason="pyarrow.ChunkedArray does not support views.")
    def test_view(self, data):
        super().test_view(data)


class TestBaseMissing(base.BaseMissingTests):
    @pytest.mark.filterwarnings("ignore:Falling back:pandas.errors.PerformanceWarning")
    def test_dropna_array(self, data_missing):
        super().test_dropna_array(data_missing)


class TestBasePrinting(base.BasePrintingTests):
    pass


class TestBaseReshaping(base.BaseReshapingTests):
    def test_concat_columns(self, data, na_value, request):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=f"Not supported by pyarrow < 2.0 with timestamp type {tz}"
                )
            )
        super().test_concat_columns(data, na_value)

    def test_concat_extension_arrays_copy_false(self, data, na_value, request):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=f"Not supported by pyarrow < 2.0 with timestamp type {tz}"
                )
            )
        super().test_concat_extension_arrays_copy_false(data, na_value)

    def test_align(self, data, na_value, request):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=f"Not supported by pyarrow < 2.0 with timestamp type {tz}"
                )
            )
        super().test_align(data, na_value)

    def test_align_frame(self, data, na_value, request):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=f"Not supported by pyarrow < 2.0 with timestamp type {tz}"
                )
            )
        super().test_align_frame(data, na_value)

    def test_align_series_frame(self, data, na_value, request):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=f"Not supported by pyarrow < 2.0 with timestamp type {tz}"
                )
            )
        super().test_align_series_frame(data, na_value)

    def test_merge(self, data, na_value, request):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=f"Not supported by pyarrow < 2.0 with timestamp type {tz}"
                )
            )
        super().test_merge(data, na_value)

    def test_ravel(self, data, request):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=f"Not supported by pyarrow < 2.0 with timestamp type {tz}"
                )
            )
        super().test_ravel(data)

    @pytest.mark.xfail(reason="GH 45419: pyarrow.ChunkedArray does not support views")
    def test_transpose(self, data):
        super().test_transpose(data)

    def test_transpose_frame(self, data, request):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=f"Not supported by pyarrow < 2.0 with timestamp type {tz}"
                )
            )
        super().test_transpose_frame(data)


class TestBaseSetitem(base.BaseSetitemTests):
    def test_setitem_scalar_series(self, data, box_in_series, request):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=f"Not supported by pyarrow < 2.0 with timestamp type {tz}"
                )
            )
        super().test_setitem_scalar_series(data, box_in_series)

    def test_setitem_sequence(self, data, box_in_series, request):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=(f"Not supported by pyarrow < 2.0 with timestamp type {tz}")
                )
            )
        super().test_setitem_sequence(data, box_in_series)

    def test_setitem_sequence_broadcasts(self, data, box_in_series, request):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=(f"Not supported by pyarrow < 2.0 with timestamp type {tz}")
                )
            )
        super().test_setitem_sequence_broadcasts(data, box_in_series)

    @pytest.mark.parametrize("setter", ["loc", "iloc"])
    def test_setitem_scalar(self, data, setter, request):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=(f"Not supported by pyarrow < 2.0 with timestamp type {tz}")
                )
            )
        super().test_setitem_scalar(data, setter)

    def test_setitem_loc_scalar_mixed(self, data, request):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=(f"Not supported by pyarrow < 2.0 with timestamp type {tz}")
                )
            )
        super().test_setitem_loc_scalar_mixed(data)

    def test_setitem_loc_scalar_single(self, data, request):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=f"Not supported by pyarrow < 2.0 with timestamp type {tz}"
                )
            )
        super().test_setitem_loc_scalar_single(data)

    def test_setitem_loc_scalar_multiple_homogoneous(self, data, request):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=(f"Not supported by pyarrow < 2.0 with timestamp type {tz}")
                )
            )
        super().test_setitem_loc_scalar_multiple_homogoneous(data)

    def test_setitem_iloc_scalar_mixed(self, data, request):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=(f"Not supported by pyarrow < 2.0 with timestamp type {tz}")
                )
            )
        super().test_setitem_iloc_scalar_mixed(data)

    def test_setitem_iloc_scalar_single(self, data, request):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=(f"Not supported by pyarrow < 2.0 with timestamp type {tz}")
                )
            )
        super().test_setitem_iloc_scalar_single(data)

    def test_setitem_iloc_scalar_multiple_homogoneous(self, data, request):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=(f"Not supported by pyarrow < 2.0 with timestamp type {tz}")
                )
            )
        super().test_setitem_iloc_scalar_multiple_homogoneous(data)

    @pytest.mark.parametrize(
        "mask",
        [
            np.array([True, True, True, False, False]),
            pd.array([True, True, True, False, False], dtype="boolean"),
            pd.array([True, True, True, pd.NA, pd.NA], dtype="boolean"),
        ],
        ids=["numpy-array", "boolean-array", "boolean-array-na"],
    )
    def test_setitem_mask(self, data, mask, box_in_series, request):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=(f"Not supported by pyarrow < 2.0 with timestamp type {tz}")
                )
            )
        super().test_setitem_mask(data, mask, box_in_series)

    def test_setitem_mask_boolean_array_with_na(self, data, box_in_series, request):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        unit = getattr(data.dtype.pyarrow_dtype, "unit", None)
        if pa_version_under2p0 and tz not in (None, "UTC") and unit == "us":
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=(f"Not supported by pyarrow < 2.0 with timestamp type {tz}")
                )
            )
        super().test_setitem_mask_boolean_array_with_na(data, box_in_series)

    @pytest.mark.parametrize(
        "idx",
        [[0, 1, 2], pd.array([0, 1, 2], dtype="Int64"), np.array([0, 1, 2])],
        ids=["list", "integer-array", "numpy-array"],
    )
    def test_setitem_integer_array(self, data, idx, box_in_series, request):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=(f"Not supported by pyarrow < 2.0 with timestamp type {tz}")
                )
            )
        super().test_setitem_integer_array(data, idx, box_in_series)

    @pytest.mark.parametrize("as_callable", [True, False])
    @pytest.mark.parametrize("setter", ["loc", None])
    def test_setitem_mask_aligned(self, data, as_callable, setter, request):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=(f"Not supported by pyarrow < 2.0 with timestamp type {tz}")
                )
            )
        super().test_setitem_mask_aligned(data, as_callable, setter)

    @pytest.mark.parametrize("setter", ["loc", None])
    def test_setitem_mask_broadcast(self, data, setter, request):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=(f"Not supported by pyarrow < 2.0 with timestamp type {tz}")
                )
            )
        super().test_setitem_mask_broadcast(data, setter)

    def test_setitem_tuple_index(self, data, request):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=(f"Not supported by pyarrow < 2.0 with timestamp type {tz}")
                )
            )
        super().test_setitem_tuple_index(data)

    def test_setitem_slice(self, data, box_in_series, request):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=(f"Not supported by pyarrow < 2.0 with timestamp type {tz}")
                )
            )
        super().test_setitem_slice(data, box_in_series)

    def test_setitem_loc_iloc_slice(self, data, request):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=f"Not supported by pyarrow < 2.0 with timestamp type {tz}"
                )
            )
        super().test_setitem_loc_iloc_slice(data)

    def test_setitem_slice_array(self, data, request):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=f"Not supported by pyarrow < 2.0 with timestamp type {tz}"
                )
            )
        super().test_setitem_slice_array(data)

    def test_setitem_with_expansion_dataframe_column(self, data, full_indexer, request):
        # Is there a better way to get the full_indexer id "null_slice"?
        is_null_slice = "null_slice" in request.node.nodeid
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC") and not is_null_slice:
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=f"Not supported by pyarrow < 2.0 with timestamp type {tz}"
                )
            )
        super().test_setitem_with_expansion_dataframe_column(data, full_indexer)

    def test_setitem_with_expansion_row(self, data, na_value, request):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=(f"Not supported by pyarrow < 2.0 with timestamp type {tz}")
                )
            )
        super().test_setitem_with_expansion_row(data, na_value)

    def test_setitem_frame_2d_values(self, data, request):
        tz = getattr(data.dtype.pyarrow_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=f"Not supported by pyarrow < 2.0 with timestamp type {tz}"
                )
            )
        super().test_setitem_frame_2d_values(data)

    @pytest.mark.xfail(reason="GH 45419: pyarrow.ChunkedArray does not support views")
    def test_setitem_preserves_views(self, data):
        super().test_setitem_preserves_views(data)


class TestBaseParsing(base.BaseParsingTests):
    @pytest.mark.parametrize("engine", ["c", "python"])
    def test_EA_types(self, engine, data, request):
        pa_dtype = data.dtype.pyarrow_dtype
        if pa.types.is_boolean(pa_dtype):
            request.node.add_marker(
                pytest.mark.xfail(raises=TypeError, reason="GH 47534")
            )
        elif pa.types.is_timestamp(pa_dtype) and pa_dtype.tz is not None:
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=NotImplementedError,
                    reason=f"Parameterized types with tz={pa_dtype.tz} not supported.",
                )
            )
        super().test_EA_types(engine, data)


class TestBaseUnaryOps(base.BaseUnaryOpsTests):
    @pytest.mark.xfail(
        pa_version_under2p0,
        raises=NotImplementedError,
        reason="pyarrow.compute.invert not supported in pyarrow<2.0",
    )
    def test_invert(self, data, request):
        pa_dtype = data.dtype.pyarrow_dtype
        if not pa.types.is_boolean(pa_dtype):
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=pa.ArrowNotImplementedError,
                    reason=f"pyarrow.compute.invert does support {pa_dtype}",
                )
            )
        super().test_invert(data)


class TestBaseMethods(base.BaseMethodsTests):
    @pytest.mark.parametrize("periods", [1, -2])
    def test_diff(self, data, periods, request):
        pa_dtype = data.dtype.pyarrow_dtype
        if pa.types.is_unsigned_integer(pa_dtype) and periods == 1:
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=pa.ArrowInvalid,
                    reason=(
                        f"diff with {pa_dtype} and periods={periods} will overflow"
                    ),
                )
            )
        super().test_diff(data, periods)

    @pytest.mark.parametrize("dropna", [True, False])
    def test_value_counts(self, all_data, dropna, request):
        pa_dtype = all_data.dtype.pyarrow_dtype
        if pa.types.is_duration(pa_dtype):
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=pa.ArrowNotImplementedError,
                    reason=f"value_count has no kernel for {pa_dtype}",
                )
            )
        super().test_value_counts(all_data, dropna)

    def test_value_counts_with_normalize(self, data, request):
        pa_dtype = data.dtype.pyarrow_dtype
        if pa.types.is_duration(pa_dtype):
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=pa.ArrowNotImplementedError,
                    reason=f"value_count has no pyarrow kernel for {pa_dtype}",
                )
            )
        super().test_value_counts_with_normalize(data)

    @pytest.mark.xfail(
        pa_version_under6p0,
        raises=NotImplementedError,
        reason="argmin/max only implemented for pyarrow version >= 6.0",
    )
    def test_argmin_argmax(
        self, data_for_sorting, data_missing_for_sorting, na_value, request
    ):
        pa_dtype = data_for_sorting.dtype.pyarrow_dtype
        if pa.types.is_boolean(pa_dtype):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=f"{pa_dtype} only has 2 unique possible values",
                )
            )
        elif pa.types.is_duration(pa_dtype):
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=pa.ArrowNotImplementedError,
                    reason=f"min_max not supported in pyarrow for {pa_dtype}",
                )
            )
        super().test_argmin_argmax(data_for_sorting, data_missing_for_sorting, na_value)

    @pytest.mark.parametrize(
        "op_name, skipna, expected",
        [
            ("idxmax", True, 0),
            ("idxmin", True, 2),
            ("argmax", True, 0),
            ("argmin", True, 2),
            ("idxmax", False, np.nan),
            ("idxmin", False, np.nan),
            ("argmax", False, -1),
            ("argmin", False, -1),
        ],
    )
    def test_argreduce_series(
        self, data_missing_for_sorting, op_name, skipna, expected, request
    ):
        pa_dtype = data_missing_for_sorting.dtype.pyarrow_dtype
        if pa_version_under6p0 and skipna:
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=NotImplementedError,
                    reason="min_max not supported in pyarrow",
                )
            )
        elif not pa_version_under6p0 and pa.types.is_duration(pa_dtype) and skipna:
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=pa.ArrowNotImplementedError,
                    reason=f"min_max not supported in pyarrow for {pa_dtype}",
                )
            )
        super().test_argreduce_series(
            data_missing_for_sorting, op_name, skipna, expected
        )

    @pytest.mark.parametrize("ascending", [True, False])
    def test_sort_values(self, data_for_sorting, ascending, sort_by_key, request):
        pa_dtype = data_for_sorting.dtype.pyarrow_dtype
        if pa.types.is_duration(pa_dtype) and not ascending and not pa_version_under2p0:
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=pa.ArrowNotImplementedError,
                    reason=(
                        f"unique has no pyarrow kernel "
                        f"for {pa_dtype} when ascending={ascending}"
                    ),
                )
            )
        super().test_sort_values(data_for_sorting, ascending, sort_by_key)

    @pytest.mark.parametrize("ascending", [True, False])
    def test_sort_values_frame(self, data_for_sorting, ascending, request):
        pa_dtype = data_for_sorting.dtype.pyarrow_dtype
        if pa.types.is_duration(pa_dtype):
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=pa.ArrowNotImplementedError,
                    reason=(
                        f"dictionary_encode has no pyarrow kernel "
                        f"for {pa_dtype} when ascending={ascending}"
                    ),
                )
            )
        super().test_sort_values_frame(data_for_sorting, ascending)

    @pytest.mark.parametrize("box", [pd.Series, lambda x: x])
    @pytest.mark.parametrize("method", [lambda x: x.unique(), pd.unique])
    def test_unique(self, data, box, method, request):
        pa_dtype = data.dtype.pyarrow_dtype
        if pa.types.is_duration(pa_dtype) and not pa_version_under2p0:
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=pa.ArrowNotImplementedError,
                    reason=f"unique has no pyarrow kernel for {pa_dtype}.",
                )
            )
        super().test_unique(data, box, method)

    @pytest.mark.parametrize("na_sentinel", [-1, -2])
    def test_factorize(self, data_for_grouping, na_sentinel, request):
        pa_dtype = data_for_grouping.dtype.pyarrow_dtype
        if pa.types.is_duration(pa_dtype):
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=pa.ArrowNotImplementedError,
                    reason=f"dictionary_encode has no pyarrow kernel for {pa_dtype}",
                )
            )
        elif pa.types.is_boolean(pa_dtype):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=f"{pa_dtype} only has 2 unique possible values",
                )
            )
        super().test_factorize(data_for_grouping, na_sentinel)

    @pytest.mark.parametrize("na_sentinel", [-1, -2])
    def test_factorize_equivalence(self, data_for_grouping, na_sentinel, request):
        pa_dtype = data_for_grouping.dtype.pyarrow_dtype
        if pa.types.is_duration(pa_dtype):
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=pa.ArrowNotImplementedError,
                    reason=f"dictionary_encode has no pyarrow kernel for {pa_dtype}",
                )
            )
        super().test_factorize_equivalence(data_for_grouping, na_sentinel)

    def test_factorize_empty(self, data, request):
        pa_dtype = data.dtype.pyarrow_dtype
        if pa.types.is_duration(pa_dtype):
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=pa.ArrowNotImplementedError,
                    reason=f"dictionary_encode has no pyarrow kernel for {pa_dtype}",
                )
            )
        super().test_factorize_empty(data)

    def test_shift_fill_value(self, data, request):
        pa_dtype = data.dtype.pyarrow_dtype
        tz = getattr(pa_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=f"Not supported by pyarrow < 2.0 with timestamp type {tz}"
                )
            )
        super().test_shift_fill_value(data)

    @pytest.mark.parametrize("repeats", [0, 1, 2, [1, 2, 3]])
    def test_repeat(self, data, repeats, as_series, use_numpy, request):
        pa_dtype = data.dtype.pyarrow_dtype
        tz = getattr(pa_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC") and repeats != 0:
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=(
                        f"Not supported by pyarrow < 2.0 with "
                        f"timestamp type {tz} when repeats={repeats}"
                    )
                )
            )
        super().test_repeat(data, repeats, as_series, use_numpy)

    def test_insert(self, data, request):
        pa_dtype = data.dtype.pyarrow_dtype
        tz = getattr(pa_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=f"Not supported by pyarrow < 2.0 with timestamp type {tz}"
                )
            )
        super().test_insert(data)

    def test_combine_first(self, data, request):
        pa_dtype = data.dtype.pyarrow_dtype
        tz = getattr(pa_dtype, "tz", None)
        if pa_version_under2p0 and tz not in (None, "UTC"):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=f"Not supported by pyarrow < 2.0 with timestamp type {tz}"
                )
            )
        super().test_combine_first(data)

    @pytest.mark.xfail(
        reason="result dtype pyarrow[bool] better than expected dtype object"
    )
    def test_combine_le(self, data_repeated):
        super().test_combine_le(data_repeated)

    def test_combine_add(self, data_repeated, request):
        pa_dtype = next(data_repeated(1)).dtype.pyarrow_dtype
        if pa.types.is_temporal(pa_dtype):
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=TypeError,
                    reason=f"{pa_dtype} cannot be added to {pa_dtype}",
                )
            )
        super().test_combine_add(data_repeated)

    def test_searchsorted(self, data_for_sorting, as_series, request):
        pa_dtype = data_for_sorting.dtype.pyarrow_dtype
        if pa.types.is_boolean(pa_dtype):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=f"{pa_dtype} only has 2 unique possible values",
                )
            )
        super().test_searchsorted(data_for_sorting, as_series)

    def test_where_series(self, data, na_value, as_frame, request):
        pa_dtype = data.dtype.pyarrow_dtype
        if pa.types.is_temporal(pa_dtype):
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=pa.ArrowNotImplementedError,
                    reason=f"Unsupported cast from double to {pa_dtype}",
                )
            )
        super().test_where_series(data, na_value, as_frame)


class TestBaseArithmeticOps(base.BaseArithmeticOpsTests):

    divmod_exc = NotImplementedError

    def _patch_combine(self, obj, other, op):
        # BaseOpsUtil._combine can upcast expected dtype
        # (because it generates expected on python scalars)
        # while ArrowExtensionArray maintains original type
        expected = base.BaseArithmeticOpsTests._combine(self, obj, other, op)
        was_frame = False
        if isinstance(expected, pd.DataFrame):
            was_frame = True
            expected_data = expected.iloc[:, 0]
            original_dtype = obj.iloc[:, 0].dtype
        else:
            expected_data = expected
            original_dtype = obj.dtype
        pa_array = pa.array(expected_data._values).cast(original_dtype.pyarrow_dtype)
        pd_array = type(expected_data._values)(pa_array)
        if was_frame:
            expected = pd.DataFrame(
                pd_array, index=expected.index, columns=expected.columns
            )
        else:
            expected = pd.Series(pd_array)
        return expected

    def test_arith_series_with_scalar(
        self, data, all_arithmetic_operators, request, monkeypatch
    ):
        pa_dtype = data.dtype.pyarrow_dtype

        arrow_temporal_supported = not pa_version_under8p0 and (
            all_arithmetic_operators in ("__add__", "__radd__")
            and pa.types.is_duration(pa_dtype)
            or all_arithmetic_operators in ("__sub__", "__rsub__")
            and pa.types.is_temporal(pa_dtype)
        )
        if (
            all_arithmetic_operators
            in {
                "__mod__",
                "__rmod__",
            }
            or pa_version_under2p0
        ):
            self.series_scalar_exc = NotImplementedError
        elif arrow_temporal_supported:
            self.series_scalar_exc = None
        elif not (
            pa.types.is_floating(pa_dtype)
            or pa.types.is_integer(pa_dtype)
            or arrow_temporal_supported
        ):
            self.series_scalar_exc = pa.ArrowNotImplementedError
        else:
            self.series_scalar_exc = None
        if (
            all_arithmetic_operators == "__rpow__"
            and (pa.types.is_floating(pa_dtype) or pa.types.is_integer(pa_dtype))
            and not pa_version_under2p0
        ):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=(
                        f"GH 29997: 1**pandas.NA == 1 while 1**pyarrow.NA == NULL "
                        f"for {pa_dtype}"
                    )
                )
            )
        elif arrow_temporal_supported:
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=TypeError,
                    reason=(
                        f"{all_arithmetic_operators} not supported between"
                        f"pd.NA and {pa_dtype} Python scalar"
                    ),
                )
            )
        elif (
            all_arithmetic_operators in {"__rtruediv__", "__rfloordiv__"}
            and (pa.types.is_floating(pa_dtype) or pa.types.is_integer(pa_dtype))
            and not pa_version_under2p0
        ):
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=pa.ArrowInvalid,
                    reason="divide by 0",
                )
            )
        if all_arithmetic_operators == "__floordiv__" and pa.types.is_integer(pa_dtype):
            # BaseOpsUtil._combine always returns int64, while ArrowExtensionArray does
            # not upcast
            monkeypatch.setattr(TestBaseArithmeticOps, "_combine", self._patch_combine)
        super().test_arith_series_with_scalar(data, all_arithmetic_operators)

    def test_arith_frame_with_scalar(
        self, data, all_arithmetic_operators, request, monkeypatch
    ):
        pa_dtype = data.dtype.pyarrow_dtype

        arrow_temporal_supported = not pa_version_under8p0 and (
            all_arithmetic_operators in ("__add__", "__radd__")
            and pa.types.is_duration(pa_dtype)
            or all_arithmetic_operators in ("__sub__", "__rsub__")
            and pa.types.is_temporal(pa_dtype)
        )
        if (
            all_arithmetic_operators
            in {
                "__mod__",
                "__rmod__",
            }
            or pa_version_under2p0
        ):
            self.frame_scalar_exc = NotImplementedError
        elif arrow_temporal_supported:
            self.frame_scalar_exc = None
        elif not (pa.types.is_floating(pa_dtype) or pa.types.is_integer(pa_dtype)):
            self.frame_scalar_exc = pa.ArrowNotImplementedError
        else:
            self.frame_scalar_exc = None
        if (
            all_arithmetic_operators == "__rpow__"
            and (pa.types.is_floating(pa_dtype) or pa.types.is_integer(pa_dtype))
            and not pa_version_under2p0
        ):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=(
                        f"GH 29997: 1**pandas.NA == 1 while 1**pyarrow.NA == NULL "
                        f"for {pa_dtype}"
                    )
                )
            )
        elif arrow_temporal_supported:
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=TypeError,
                    reason=(
                        f"{all_arithmetic_operators} not supported between"
                        f"pd.NA and {pa_dtype} Python scalar"
                    ),
                )
            )
        elif (
            all_arithmetic_operators in {"__rtruediv__", "__rfloordiv__"}
            and (pa.types.is_floating(pa_dtype) or pa.types.is_integer(pa_dtype))
            and not pa_version_under2p0
        ):
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=pa.ArrowInvalid,
                    reason="divide by 0",
                )
            )
        if all_arithmetic_operators == "__floordiv__" and pa.types.is_integer(pa_dtype):
            # BaseOpsUtil._combine always returns int64, while ArrowExtensionArray does
            # not upcast
            monkeypatch.setattr(TestBaseArithmeticOps, "_combine", self._patch_combine)
        super().test_arith_frame_with_scalar(data, all_arithmetic_operators)

    def test_arith_series_with_array(
        self, data, all_arithmetic_operators, request, monkeypatch
    ):
        pa_dtype = data.dtype.pyarrow_dtype

        arrow_temporal_supported = not pa_version_under8p0 and (
            all_arithmetic_operators in ("__add__", "__radd__")
            and pa.types.is_duration(pa_dtype)
            or all_arithmetic_operators in ("__sub__", "__rsub__")
            and pa.types.is_temporal(pa_dtype)
        )
        if (
            all_arithmetic_operators
            in {
                "__mod__",
                "__rmod__",
            }
            or pa_version_under2p0
        ):
            self.series_array_exc = NotImplementedError
        elif arrow_temporal_supported:
            self.series_array_exc = None
        elif not (pa.types.is_floating(pa_dtype) or pa.types.is_integer(pa_dtype)):
            self.series_array_exc = pa.ArrowNotImplementedError
        else:
            self.series_array_exc = None
        if (
            all_arithmetic_operators == "__rpow__"
            and (pa.types.is_floating(pa_dtype) or pa.types.is_integer(pa_dtype))
            and not pa_version_under2p0
        ):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=(
                        f"GH 29997: 1**pandas.NA == 1 while 1**pyarrow.NA == NULL "
                        f"for {pa_dtype}"
                    )
                )
            )
        elif (
            all_arithmetic_operators
            in (
                "__sub__",
                "__rsub__",
            )
            and pa.types.is_unsigned_integer(pa_dtype)
            and not pa_version_under2p0
        ):
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=pa.ArrowInvalid,
                    reason=(
                        f"Implemented pyarrow.compute.subtract_checked "
                        f"which raises on overflow for {pa_dtype}"
                    ),
                )
            )
        elif arrow_temporal_supported:
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=TypeError,
                    reason=(
                        f"{all_arithmetic_operators} not supported between"
                        f"pd.NA and {pa_dtype} Python scalar"
                    ),
                )
            )
        elif (
            all_arithmetic_operators in {"__rtruediv__", "__rfloordiv__"}
            and (pa.types.is_floating(pa_dtype) or pa.types.is_integer(pa_dtype))
            and not pa_version_under2p0
        ):
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=pa.ArrowInvalid,
                    reason="divide by 0",
                )
            )
        op_name = all_arithmetic_operators
        ser = pd.Series(data)
        # pd.Series([ser.iloc[0]] * len(ser)) may not return ArrowExtensionArray
        # since ser.iloc[0] is a python scalar
        other = pd.Series(pd.array([ser.iloc[0]] * len(ser), dtype=data.dtype))
        if pa.types.is_floating(pa_dtype) or (
            pa.types.is_integer(pa_dtype) and all_arithmetic_operators != "__truediv__"
        ):
            monkeypatch.setattr(TestBaseArithmeticOps, "_combine", self._patch_combine)
        self.check_opname(ser, op_name, other, exc=self.series_array_exc)

    def test_add_series_with_extension_array(self, data, request):
        pa_dtype = data.dtype.pyarrow_dtype
        if (
            not (
                pa.types.is_integer(pa_dtype)
                or pa.types.is_floating(pa_dtype)
                or (not pa_version_under8p0 and pa.types.is_duration(pa_dtype))
            )
            or pa_version_under2p0
        ):
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=NotImplementedError,
                    reason=f"add_checked not implemented for {pa_dtype}",
                )
            )
        elif pa_dtype.equals("int8"):
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=pa.ArrowInvalid,
                    reason=f"raises on overflow for {pa_dtype}",
                )
            )
        super().test_add_series_with_extension_array(data)


class TestBaseComparisonOps(base.BaseComparisonOpsTests):
    def assert_series_equal(self, left, right, *args, **kwargs):
        # Series.combine for "expected" retains bool[pyarrow] dtype
        # While "result" return "boolean" dtype
        right = pd.Series(right._values.to_numpy(), dtype="boolean")
        super().assert_series_equal(left, right, *args, **kwargs)

    def test_compare_array(self, data, comparison_op, na_value, request):
        pa_dtype = data.dtype.pyarrow_dtype
        ser = pd.Series(data)
        # pd.Series([ser.iloc[0]] * len(ser)) may not return ArrowExtensionArray
        # since ser.iloc[0] is a python scalar
        other = pd.Series(pd.array([ser.iloc[0]] * len(ser), dtype=data.dtype))
        if comparison_op.__name__ in ["eq", "ne"]:
            # comparison should match point-wise comparisons
            result = comparison_op(ser, other)
            # Series.combine does not calculate the NA mask correctly
            # when comparing over an array
            assert result[8] is na_value
            assert result[97] is na_value
            expected = ser.combine(other, comparison_op)
            expected[8] = na_value
            expected[97] = na_value
            self.assert_series_equal(result, expected)

        else:
            exc = None
            try:
                result = comparison_op(ser, other)
            except Exception as err:
                exc = err

            if exc is None:
                # Didn't error, then should match point-wise behavior
                if pa.types.is_temporal(pa_dtype):
                    # point-wise comparison with pd.NA raises TypeError
                    assert result[8] is na_value
                    assert result[97] is na_value
                    result = result.drop([8, 97]).reset_index(drop=True)
                    ser = ser.drop([8, 97])
                    other = other.drop([8, 97])
                expected = ser.combine(other, comparison_op)
                self.assert_series_equal(result, expected)
            else:
                with pytest.raises(type(exc)):
                    ser.combine(other, comparison_op)

    def test_invalid_other_comp(self, data, comparison_op):
        # GH 48833
        with pytest.raises(
            NotImplementedError, match=".* not implemented for <class 'object'>"
        ):
            comparison_op(data, object())


def test_arrowdtype_construct_from_string_type_with_unsupported_parameters():
    with pytest.raises(NotImplementedError, match="Passing pyarrow type"):
        ArrowDtype.construct_from_string("timestamp[s, tz=UTC][pyarrow]")


@pytest.mark.xfail(
    pa_version_under4p0,
    raises=NotImplementedError,
    reason="quantile only supported for pyarrow version >= 4.0",
)
@pytest.mark.parametrize(
    "interpolation", ["linear", "lower", "higher", "nearest", "midpoint"]
)
@pytest.mark.parametrize("quantile", [0.5, [0.5, 0.5]])
def test_quantile(data, interpolation, quantile, request):
    pa_dtype = data.dtype.pyarrow_dtype
    if not (pa.types.is_integer(pa_dtype) or pa.types.is_floating(pa_dtype)):
        request.node.add_marker(
            pytest.mark.xfail(
                raises=pa.ArrowNotImplementedError,
                reason=f"quantile not supported by pyarrow for {pa_dtype}",
            )
        )
    data = data.take([0, 0, 0])
    ser = pd.Series(data)
    result = ser.quantile(q=quantile, interpolation=interpolation)
    if quantile == 0.5:
        assert result == data[0]
    else:
        # Just check the values
        result = result.astype("float64[pyarrow]")
        expected = pd.Series(
            data.take([0, 0]).astype("float64[pyarrow]"), index=[0.5, 0.5]
        )
        tm.assert_series_equal(result, expected)


@pytest.mark.xfail(
    pa_version_under6p0,
    raises=NotImplementedError,
    reason="mode only supported for pyarrow version >= 6.0",
)
@pytest.mark.parametrize("dropna", [True, False])
@pytest.mark.parametrize(
    "take_idx, exp_idx",
    [[[0, 0, 2, 2, 4, 4], [4, 0]], [[0, 0, 0, 2, 4, 4], [0]]],
    ids=["multi_mode", "single_mode"],
)
def test_mode(data_for_grouping, dropna, take_idx, exp_idx, request):
    pa_dtype = data_for_grouping.dtype.pyarrow_dtype
    if pa.types.is_temporal(pa_dtype):
        request.node.add_marker(
            pytest.mark.xfail(
                raises=pa.ArrowNotImplementedError,
                reason=f"mode not supported by pyarrow for {pa_dtype}",
            )
        )
    elif (
        pa.types.is_boolean(pa_dtype)
        and "multi_mode" in request.node.nodeid
        and pa_version_under9p0
    ):
        request.node.add_marker(
            pytest.mark.xfail(
                reason="https://issues.apache.org/jira/browse/ARROW-17096",
            )
        )
    data = data_for_grouping.take(take_idx)
    ser = pd.Series(data)
    result = ser.mode(dropna=dropna)
    expected = pd.Series(data_for_grouping.take(exp_idx))
    tm.assert_series_equal(result, expected)

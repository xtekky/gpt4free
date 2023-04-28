""" test the scalar Timestamp """

import calendar
from datetime import (
    datetime,
    timedelta,
)
import locale
import pickle
import unicodedata

from dateutil.tz import tzutc
import numpy as np
import pytest
import pytz
from pytz import (
    timezone,
    utc,
)

from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.timezones import (
    dateutil_gettz as gettz,
    get_timezone,
    maybe_get_tz,
    tz_compare,
)
from pandas.errors import OutOfBoundsDatetime
import pandas.util._test_decorators as td

from pandas import (
    NaT,
    Timedelta,
    Timestamp,
)
import pandas._testing as tm

from pandas.tseries import offsets


class TestTimestampProperties:
    def test_freq_deprecation(self):
        # GH#41586
        msg = "The 'freq' argument in Timestamp is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # warning issued at construction
            ts = Timestamp("2021-06-01", freq="D")
            ts2 = Timestamp("2021-06-01", freq="B")

        msg = "Timestamp.freq is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # warning issued at attribute lookup
            ts.freq

        for per in ["month", "quarter", "year"]:
            for side in ["start", "end"]:
                attr = f"is_{per}_{side}"

                with tm.assert_produces_warning(FutureWarning, match=msg):
                    getattr(ts2, attr)

                # is_(month|quarter|year)_(start|end) does _not_ issue a warning
                #  with freq="D" bc the result will be unaffected by the deprecation
                with tm.assert_produces_warning(None):
                    getattr(ts, attr)

    @pytest.mark.filterwarnings("ignore:The 'freq' argument:FutureWarning")
    @pytest.mark.filterwarnings("ignore:Timestamp.freq is deprecated:FutureWarning")
    def test_properties_business(self):
        ts = Timestamp("2017-10-01", freq="B")
        control = Timestamp("2017-10-01")
        assert ts.dayofweek == 6
        assert ts.day_of_week == 6
        assert not ts.is_month_start  # not a weekday
        assert not ts.freq.is_month_start(ts)
        assert ts.freq.is_month_start(ts + Timedelta(days=1))
        assert not ts.is_quarter_start  # not a weekday
        assert not ts.freq.is_quarter_start(ts)
        assert ts.freq.is_quarter_start(ts + Timedelta(days=1))
        # Control case: non-business is month/qtr start
        assert control.is_month_start
        assert control.is_quarter_start

        ts = Timestamp("2017-09-30", freq="B")
        control = Timestamp("2017-09-30")
        assert ts.dayofweek == 5
        assert ts.day_of_week == 5
        assert not ts.is_month_end  # not a weekday
        assert not ts.freq.is_month_end(ts)
        assert ts.freq.is_month_end(ts - Timedelta(days=1))
        assert not ts.is_quarter_end  # not a weekday
        assert not ts.freq.is_quarter_end(ts)
        assert ts.freq.is_quarter_end(ts - Timedelta(days=1))
        # Control case: non-business is month/qtr start
        assert control.is_month_end
        assert control.is_quarter_end

    @pytest.mark.parametrize(
        "attr, expected",
        [
            ["year", 2014],
            ["month", 12],
            ["day", 31],
            ["hour", 23],
            ["minute", 59],
            ["second", 0],
            ["microsecond", 0],
            ["nanosecond", 0],
            ["dayofweek", 2],
            ["day_of_week", 2],
            ["quarter", 4],
            ["dayofyear", 365],
            ["day_of_year", 365],
            ["week", 1],
            ["daysinmonth", 31],
        ],
    )
    @pytest.mark.parametrize("tz", [None, "US/Eastern"])
    def test_fields(self, attr, expected, tz):
        # GH 10050
        # GH 13303
        ts = Timestamp("2014-12-31 23:59:00", tz=tz)
        result = getattr(ts, attr)
        # that we are int like
        assert isinstance(result, int)
        assert result == expected

    @pytest.mark.parametrize("tz", [None, "US/Eastern"])
    def test_millisecond_raises(self, tz):
        ts = Timestamp("2014-12-31 23:59:00", tz=tz)
        msg = "'Timestamp' object has no attribute 'millisecond'"
        with pytest.raises(AttributeError, match=msg):
            ts.millisecond

    @pytest.mark.parametrize(
        "start", ["is_month_start", "is_quarter_start", "is_year_start"]
    )
    @pytest.mark.parametrize("tz", [None, "US/Eastern"])
    def test_is_start(self, start, tz):
        ts = Timestamp("2014-01-01 00:00:00", tz=tz)
        assert getattr(ts, start)

    @pytest.mark.parametrize("end", ["is_month_end", "is_year_end", "is_quarter_end"])
    @pytest.mark.parametrize("tz", [None, "US/Eastern"])
    def test_is_end(self, end, tz):
        ts = Timestamp("2014-12-31 23:59:59", tz=tz)
        assert getattr(ts, end)

    # GH 12806
    @pytest.mark.parametrize(
        "data",
        [Timestamp("2017-08-28 23:00:00"), Timestamp("2017-08-28 23:00:00", tz="EST")],
    )
    # error: Unsupported operand types for + ("List[None]" and "List[str]")
    @pytest.mark.parametrize(
        "time_locale", [None] + (tm.get_locales() or [])  # type: ignore[operator]
    )
    def test_names(self, data, time_locale):
        # GH 17354
        # Test .day_name(), .month_name
        if time_locale is None:
            expected_day = "Monday"
            expected_month = "August"
        else:
            with tm.set_locale(time_locale, locale.LC_TIME):
                expected_day = calendar.day_name[0].capitalize()
                expected_month = calendar.month_name[8].capitalize()

        result_day = data.day_name(time_locale)
        result_month = data.month_name(time_locale)

        # Work around https://github.com/pandas-dev/pandas/issues/22342
        # different normalizations
        expected_day = unicodedata.normalize("NFD", expected_day)
        expected_month = unicodedata.normalize("NFD", expected_month)

        result_day = unicodedata.normalize("NFD", result_day)
        result_month = unicodedata.normalize("NFD", result_month)

        assert result_day == expected_day
        assert result_month == expected_month

        # Test NaT
        nan_ts = Timestamp(NaT)
        assert np.isnan(nan_ts.day_name(time_locale))
        assert np.isnan(nan_ts.month_name(time_locale))

    def test_is_leap_year(self, tz_naive_fixture):
        tz = tz_naive_fixture
        # GH 13727
        dt = Timestamp("2000-01-01 00:00:00", tz=tz)
        assert dt.is_leap_year
        assert isinstance(dt.is_leap_year, bool)

        dt = Timestamp("1999-01-01 00:00:00", tz=tz)
        assert not dt.is_leap_year

        dt = Timestamp("2004-01-01 00:00:00", tz=tz)
        assert dt.is_leap_year

        dt = Timestamp("2100-01-01 00:00:00", tz=tz)
        assert not dt.is_leap_year

    def test_woy_boundary(self):
        # make sure weeks at year boundaries are correct
        d = datetime(2013, 12, 31)
        result = Timestamp(d).week
        expected = 1  # ISO standard
        assert result == expected

        d = datetime(2008, 12, 28)
        result = Timestamp(d).week
        expected = 52  # ISO standard
        assert result == expected

        d = datetime(2009, 12, 31)
        result = Timestamp(d).week
        expected = 53  # ISO standard
        assert result == expected

        d = datetime(2010, 1, 1)
        result = Timestamp(d).week
        expected = 53  # ISO standard
        assert result == expected

        d = datetime(2010, 1, 3)
        result = Timestamp(d).week
        expected = 53  # ISO standard
        assert result == expected

        result = np.array(
            [
                Timestamp(datetime(*args)).week
                for args in [(2000, 1, 1), (2000, 1, 2), (2005, 1, 1), (2005, 1, 2)]
            ]
        )
        assert (result == [52, 52, 53, 53]).all()

    def test_resolution(self):
        # GH#21336, GH#21365
        dt = Timestamp("2100-01-01 00:00:00")
        assert dt.resolution == Timedelta(nanoseconds=1)

        # Check that the attribute is available on the class, mirroring
        #  the stdlib datetime behavior
        assert Timestamp.resolution == Timedelta(nanoseconds=1)


class TestTimestamp:
    def test_tz(self):
        tstr = "2014-02-01 09:00"
        ts = Timestamp(tstr)
        local = ts.tz_localize("Asia/Tokyo")
        assert local.hour == 9
        assert local == Timestamp(tstr, tz="Asia/Tokyo")
        conv = local.tz_convert("US/Eastern")
        assert conv == Timestamp("2014-01-31 19:00", tz="US/Eastern")
        assert conv.hour == 19

        # preserves nanosecond
        ts = Timestamp(tstr) + offsets.Nano(5)
        local = ts.tz_localize("Asia/Tokyo")
        assert local.hour == 9
        assert local.nanosecond == 5
        conv = local.tz_convert("US/Eastern")
        assert conv.nanosecond == 5
        assert conv.hour == 19

    def test_utc_z_designator(self):
        assert get_timezone(Timestamp("2014-11-02 01:00Z").tzinfo) is utc

    def test_asm8(self):
        np.random.seed(7_960_929)
        ns = [Timestamp.min.value, Timestamp.max.value, 1000]

        for n in ns:
            assert (
                Timestamp(n).asm8.view("i8") == np.datetime64(n, "ns").view("i8") == n
            )

        assert Timestamp("nat").asm8.view("i8") == np.datetime64("nat", "ns").view("i8")

    def test_class_ops_pytz(self):
        def compare(x, y):
            assert int((Timestamp(x).value - Timestamp(y).value) / 1e9) == 0

        compare(Timestamp.now(), datetime.now())
        compare(Timestamp.now("UTC"), datetime.now(timezone("UTC")))
        compare(Timestamp.utcnow(), datetime.utcnow())
        compare(Timestamp.today(), datetime.today())
        current_time = calendar.timegm(datetime.now().utctimetuple())
        msg = "timezone-aware Timestamp with UTC"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # GH#22451
            ts_utc = Timestamp.utcfromtimestamp(current_time)
        compare(
            ts_utc,
            datetime.utcfromtimestamp(current_time),
        )
        compare(
            Timestamp.fromtimestamp(current_time), datetime.fromtimestamp(current_time)
        )
        compare(
            # Support tz kwarg in Timestamp.fromtimestamp
            Timestamp.fromtimestamp(current_time, "UTC"),
            datetime.fromtimestamp(current_time, utc),
        )
        compare(
            # Support tz kwarg in Timestamp.fromtimestamp
            Timestamp.fromtimestamp(current_time, tz="UTC"),
            datetime.fromtimestamp(current_time, utc),
        )

        date_component = datetime.utcnow()
        time_component = (date_component + timedelta(minutes=10)).time()
        compare(
            Timestamp.combine(date_component, time_component),
            datetime.combine(date_component, time_component),
        )

    def test_class_ops_dateutil(self):
        def compare(x, y):
            assert (
                int(
                    np.round(Timestamp(x).value / 1e9)
                    - np.round(Timestamp(y).value / 1e9)
                )
                == 0
            )

        compare(Timestamp.now(), datetime.now())
        compare(Timestamp.now("UTC"), datetime.now(tzutc()))
        compare(Timestamp.utcnow(), datetime.utcnow())
        compare(Timestamp.today(), datetime.today())
        current_time = calendar.timegm(datetime.now().utctimetuple())

        msg = "timezone-aware Timestamp with UTC"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # GH#22451
            ts_utc = Timestamp.utcfromtimestamp(current_time)

        compare(
            ts_utc,
            datetime.utcfromtimestamp(current_time),
        )
        compare(
            Timestamp.fromtimestamp(current_time), datetime.fromtimestamp(current_time)
        )

        date_component = datetime.utcnow()
        time_component = (date_component + timedelta(minutes=10)).time()
        compare(
            Timestamp.combine(date_component, time_component),
            datetime.combine(date_component, time_component),
        )

    def test_basics_nanos(self):
        val = np.int64(946_684_800_000_000_000).view("M8[ns]")
        stamp = Timestamp(val.view("i8") + 500)
        assert stamp.year == 2000
        assert stamp.month == 1
        assert stamp.microsecond == 0
        assert stamp.nanosecond == 500

        # GH 14415
        val = np.iinfo(np.int64).min + 80_000_000_000_000
        stamp = Timestamp(val)
        assert stamp.year == 1677
        assert stamp.month == 9
        assert stamp.day == 21
        assert stamp.microsecond == 145224
        assert stamp.nanosecond == 192

    @pytest.mark.parametrize(
        "value, check_kwargs",
        [
            [946688461000000000, {}],
            [946688461000000000 / 1000, {"unit": "us"}],
            [946688461000000000 / 1_000_000, {"unit": "ms"}],
            [946688461000000000 / 1_000_000_000, {"unit": "s"}],
            [10957, {"unit": "D", "h": 0}],
            [
                (946688461000000000 + 500000) / 1000000000,
                {"unit": "s", "us": 499, "ns": 964},
            ],
            [
                (946688461000000000 + 500000000) / 1000000000,
                {"unit": "s", "us": 500000},
            ],
            [(946688461000000000 + 500000) / 1000000, {"unit": "ms", "us": 500}],
            [(946688461000000000 + 500000) / 1000, {"unit": "us", "us": 500}],
            [(946688461000000000 + 500000000) / 1000000, {"unit": "ms", "us": 500000}],
            [946688461000000000 / 1000.0 + 5, {"unit": "us", "us": 5}],
            [946688461000000000 / 1000.0 + 5000, {"unit": "us", "us": 5000}],
            [946688461000000000 / 1000000.0 + 0.5, {"unit": "ms", "us": 500}],
            [946688461000000000 / 1000000.0 + 0.005, {"unit": "ms", "us": 5, "ns": 5}],
            [946688461000000000 / 1000000000.0 + 0.5, {"unit": "s", "us": 500000}],
            [10957 + 0.5, {"unit": "D", "h": 12}],
        ],
    )
    def test_unit(self, value, check_kwargs):
        def check(value, unit=None, h=1, s=1, us=0, ns=0):
            stamp = Timestamp(value, unit=unit)
            assert stamp.year == 2000
            assert stamp.month == 1
            assert stamp.day == 1
            assert stamp.hour == h
            if unit != "D":
                assert stamp.minute == 1
                assert stamp.second == s
                assert stamp.microsecond == us
            else:
                assert stamp.minute == 0
                assert stamp.second == 0
                assert stamp.microsecond == 0
            assert stamp.nanosecond == ns

        check(value, **check_kwargs)

    def test_roundtrip(self):

        # test value to string and back conversions
        # further test accessors
        base = Timestamp("20140101 00:00:00")

        result = Timestamp(base.value + Timedelta("5ms").value)
        assert result == Timestamp(f"{base}.005000")
        assert result.microsecond == 5000

        result = Timestamp(base.value + Timedelta("5us").value)
        assert result == Timestamp(f"{base}.000005")
        assert result.microsecond == 5

        result = Timestamp(base.value + Timedelta("5ns").value)
        assert result == Timestamp(f"{base}.000000005")
        assert result.nanosecond == 5
        assert result.microsecond == 0

        result = Timestamp(base.value + Timedelta("6ms 5us").value)
        assert result == Timestamp(f"{base}.006005")
        assert result.microsecond == 5 + 6 * 1000

        result = Timestamp(base.value + Timedelta("200ms 5us").value)
        assert result == Timestamp(f"{base}.200005")
        assert result.microsecond == 5 + 200 * 1000

    def test_hash_equivalent(self):
        d = {datetime(2011, 1, 1): 5}
        stamp = Timestamp(datetime(2011, 1, 1))
        assert d[stamp] == 5

    @pytest.mark.parametrize(
        "timezone, year, month, day, hour",
        [["America/Chicago", 2013, 11, 3, 1], ["America/Santiago", 2021, 4, 3, 23]],
    )
    def test_hash_timestamp_with_fold(self, timezone, year, month, day, hour):
        # see gh-33931
        test_timezone = gettz(timezone)
        transition_1 = Timestamp(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=0,
            fold=0,
            tzinfo=test_timezone,
        )
        transition_2 = Timestamp(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=0,
            fold=1,
            tzinfo=test_timezone,
        )
        assert hash(transition_1) == hash(transition_2)

    def test_tz_conversion_freq(self, tz_naive_fixture):
        # GH25241
        with tm.assert_produces_warning(FutureWarning, match="freq"):
            t1 = Timestamp("2019-01-01 10:00", freq="H")
            assert t1.tz_localize(tz=tz_naive_fixture).freq == t1.freq
        with tm.assert_produces_warning(FutureWarning, match="freq"):
            t2 = Timestamp("2019-01-02 12:00", tz="UTC", freq="T")
            assert t2.tz_convert(tz="UTC").freq == t2.freq

    def test_pickle_freq_no_warning(self):
        # GH#41949 we don't want a warning on unpickling
        with tm.assert_produces_warning(FutureWarning, match="freq"):
            ts = Timestamp("2019-01-01 10:00", freq="H")

        out = pickle.dumps(ts)
        with tm.assert_produces_warning(None):
            res = pickle.loads(out)

        assert res._freq == ts._freq


class TestTimestampNsOperations:
    def test_nanosecond_string_parsing(self):
        ts = Timestamp("2013-05-01 07:15:45.123456789")
        # GH 7878
        expected_repr = "2013-05-01 07:15:45.123456789"
        expected_value = 1_367_392_545_123_456_789
        assert ts.value == expected_value
        assert expected_repr in repr(ts)

        ts = Timestamp("2013-05-01 07:15:45.123456789+09:00", tz="Asia/Tokyo")
        assert ts.value == expected_value - 9 * 3600 * 1_000_000_000
        assert expected_repr in repr(ts)

        ts = Timestamp("2013-05-01 07:15:45.123456789", tz="UTC")
        assert ts.value == expected_value
        assert expected_repr in repr(ts)

        ts = Timestamp("2013-05-01 07:15:45.123456789", tz="US/Eastern")
        assert ts.value == expected_value + 4 * 3600 * 1_000_000_000
        assert expected_repr in repr(ts)

        # GH 10041
        ts = Timestamp("20130501T071545.123456789")
        assert ts.value == expected_value
        assert expected_repr in repr(ts)

    def test_nanosecond_timestamp(self):
        # GH 7610
        expected = 1_293_840_000_000_000_005
        t = Timestamp("2011-01-01") + offsets.Nano(5)
        assert repr(t) == "Timestamp('2011-01-01 00:00:00.000000005')"
        assert t.value == expected
        assert t.nanosecond == 5

        t = Timestamp(t)
        assert repr(t) == "Timestamp('2011-01-01 00:00:00.000000005')"
        assert t.value == expected
        assert t.nanosecond == 5

        t = Timestamp("2011-01-01 00:00:00.000000005")
        assert repr(t) == "Timestamp('2011-01-01 00:00:00.000000005')"
        assert t.value == expected
        assert t.nanosecond == 5

        expected = 1_293_840_000_000_000_010
        t = t + offsets.Nano(5)
        assert repr(t) == "Timestamp('2011-01-01 00:00:00.000000010')"
        assert t.value == expected
        assert t.nanosecond == 10

        t = Timestamp(t)
        assert repr(t) == "Timestamp('2011-01-01 00:00:00.000000010')"
        assert t.value == expected
        assert t.nanosecond == 10

        t = Timestamp("2011-01-01 00:00:00.000000010")
        assert repr(t) == "Timestamp('2011-01-01 00:00:00.000000010')"
        assert t.value == expected
        assert t.nanosecond == 10


class TestTimestampToJulianDate:
    def test_compare_1700(self):
        r = Timestamp("1700-06-23").to_julian_date()
        assert r == 2_342_145.5

    def test_compare_2000(self):
        r = Timestamp("2000-04-12").to_julian_date()
        assert r == 2_451_646.5

    def test_compare_2100(self):
        r = Timestamp("2100-08-12").to_julian_date()
        assert r == 2_488_292.5

    def test_compare_hour01(self):
        r = Timestamp("2000-08-12T01:00:00").to_julian_date()
        assert r == 2_451_768.5416666666666666

    def test_compare_hour13(self):
        r = Timestamp("2000-08-12T13:00:00").to_julian_date()
        assert r == 2_451_769.0416666666666666


class TestTimestampConversion:
    def test_conversion(self):
        # GH#9255
        ts = Timestamp("2000-01-01")

        result = ts.to_pydatetime()
        expected = datetime(2000, 1, 1)
        assert result == expected
        assert type(result) == type(expected)

        result = ts.to_datetime64()
        expected = np.datetime64(ts.value, "ns")
        assert result == expected
        assert type(result) == type(expected)
        assert result.dtype == expected.dtype

    def test_to_pydatetime_fold(self):
        # GH#45087
        tzstr = "dateutil/usr/share/zoneinfo/America/Chicago"
        ts = Timestamp(year=2013, month=11, day=3, hour=1, minute=0, fold=1, tz=tzstr)
        dt = ts.to_pydatetime()
        assert dt.fold == 1

    def test_to_pydatetime_nonzero_nano(self):
        ts = Timestamp("2011-01-01 9:00:00.123456789")

        # Warn the user of data loss (nanoseconds).
        with tm.assert_produces_warning(UserWarning):
            expected = datetime(2011, 1, 1, 9, 0, 0, 123456)
            result = ts.to_pydatetime()
            assert result == expected

    def test_timestamp_to_datetime(self):
        stamp = Timestamp("20090415", tz="US/Eastern")
        dtval = stamp.to_pydatetime()
        assert stamp == dtval
        assert stamp.tzinfo == dtval.tzinfo

    def test_timestamp_to_datetime_dateutil(self):
        stamp = Timestamp("20090415", tz="dateutil/US/Eastern")
        dtval = stamp.to_pydatetime()
        assert stamp == dtval
        assert stamp.tzinfo == dtval.tzinfo

    def test_timestamp_to_datetime_explicit_pytz(self):
        stamp = Timestamp("20090415", tz=pytz.timezone("US/Eastern"))
        dtval = stamp.to_pydatetime()
        assert stamp == dtval
        assert stamp.tzinfo == dtval.tzinfo

    @td.skip_if_windows
    def test_timestamp_to_datetime_explicit_dateutil(self):
        stamp = Timestamp("20090415", tz=gettz("US/Eastern"))
        dtval = stamp.to_pydatetime()
        assert stamp == dtval
        assert stamp.tzinfo == dtval.tzinfo

    def test_to_datetime_bijective(self):
        # Ensure that converting to datetime and back only loses precision
        # by going from nanoseconds to microseconds.
        exp_warning = None if Timestamp.max.nanosecond == 0 else UserWarning
        with tm.assert_produces_warning(exp_warning):
            pydt_max = Timestamp.max.to_pydatetime()

        assert Timestamp(pydt_max).value / 1000 == Timestamp.max.value / 1000

        exp_warning = None if Timestamp.min.nanosecond == 0 else UserWarning
        with tm.assert_produces_warning(exp_warning):
            pydt_min = Timestamp.min.to_pydatetime()

        # The next assertion can be enabled once GH#39221 is merged
        #  assert pydt_min < Timestamp.min  # this is bc nanos are dropped
        tdus = timedelta(microseconds=1)
        assert pydt_min + tdus > Timestamp.min

        assert Timestamp(pydt_min + tdus).value / 1000 == Timestamp.min.value / 1000

    def test_to_period_tz_warning(self):
        # GH#21333 make sure a warning is issued when timezone
        # info is lost
        ts = Timestamp("2009-04-15 16:17:18", tz="US/Eastern")
        with tm.assert_produces_warning(UserWarning):
            # warning that timezone info will be lost
            ts.to_period("D")

    def test_to_numpy_alias(self):
        # GH 24653: alias .to_numpy() for scalars
        ts = Timestamp(datetime.now())
        assert ts.to_datetime64() == ts.to_numpy()

        # GH#44460
        msg = "dtype and copy arguments are ignored"
        with pytest.raises(ValueError, match=msg):
            ts.to_numpy("M8[s]")
        with pytest.raises(ValueError, match=msg):
            ts.to_numpy(copy=True)


class SubDatetime(datetime):
    pass


@pytest.mark.parametrize(
    "lh,rh",
    [
        (SubDatetime(2000, 1, 1), Timedelta(hours=1)),
        (Timedelta(hours=1), SubDatetime(2000, 1, 1)),
    ],
)
def test_dt_subclass_add_timedelta(lh, rh):
    # GH#25851
    # ensure that subclassed datetime works for
    # Timedelta operations
    result = lh + rh
    expected = SubDatetime(2000, 1, 1, 1)
    assert result == expected


class TestNonNano:
    @pytest.fixture(params=["s", "ms", "us"])
    def reso(self, request):
        return request.param

    @pytest.fixture
    def dt64(self, reso):
        # cases that are in-bounds for nanosecond, so we can compare against
        #  the existing implementation.
        return np.datetime64("2016-01-01", reso)

    @pytest.fixture
    def ts(self, dt64):
        return Timestamp._from_dt64(dt64)

    @pytest.fixture
    def ts_tz(self, ts, tz_aware_fixture):
        tz = maybe_get_tz(tz_aware_fixture)
        return Timestamp._from_value_and_reso(ts.value, ts._reso, tz)

    def test_non_nano_construction(self, dt64, ts, reso):
        assert ts.value == dt64.view("i8")

        if reso == "s":
            assert ts._reso == NpyDatetimeUnit.NPY_FR_s.value
        elif reso == "ms":
            assert ts._reso == NpyDatetimeUnit.NPY_FR_ms.value
        elif reso == "us":
            assert ts._reso == NpyDatetimeUnit.NPY_FR_us.value

    def test_non_nano_fields(self, dt64, ts):
        alt = Timestamp(dt64)

        assert ts.year == alt.year
        assert ts.month == alt.month
        assert ts.day == alt.day
        assert ts.hour == ts.minute == ts.second == ts.microsecond == 0
        assert ts.nanosecond == 0

        assert ts.to_julian_date() == alt.to_julian_date()
        assert ts.weekday() == alt.weekday()
        assert ts.isoweekday() == alt.isoweekday()

    def test_start_end_fields(self, ts):
        assert ts.is_year_start
        assert ts.is_quarter_start
        assert ts.is_month_start
        assert not ts.is_year_end
        assert not ts.is_month_end
        assert not ts.is_month_end

        freq = offsets.BDay()
        ts._set_freq(freq)

        # 2016-01-01 is a Friday, so is year/quarter/month start with this freq
        msg = "Timestamp.freq is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert ts.is_year_start
            assert ts.is_quarter_start
            assert ts.is_month_start
            assert not ts.is_year_end
            assert not ts.is_month_end
            assert not ts.is_month_end

    def test_day_name(self, dt64, ts):
        alt = Timestamp(dt64)
        assert ts.day_name() == alt.day_name()

    def test_month_name(self, dt64, ts):
        alt = Timestamp(dt64)
        assert ts.month_name() == alt.month_name()

    def test_tz_convert(self, ts):
        ts = Timestamp._from_value_and_reso(ts.value, ts._reso, utc)

        tz = pytz.timezone("US/Pacific")
        result = ts.tz_convert(tz)

        assert isinstance(result, Timestamp)
        assert result._reso == ts._reso
        assert tz_compare(result.tz, tz)

    def test_repr(self, dt64, ts):
        alt = Timestamp(dt64)

        assert str(ts) == str(alt)
        assert repr(ts) == repr(alt)

    def test_comparison(self, dt64, ts):
        alt = Timestamp(dt64)

        assert ts == dt64
        assert dt64 == ts
        assert ts == alt
        assert alt == ts

        assert not ts != dt64
        assert not dt64 != ts
        assert not ts != alt
        assert not alt != ts

        assert not ts < dt64
        assert not dt64 < ts
        assert not ts < alt
        assert not alt < ts

        assert not ts > dt64
        assert not dt64 > ts
        assert not ts > alt
        assert not alt > ts

        assert ts >= dt64
        assert dt64 >= ts
        assert ts >= alt
        assert alt >= ts

        assert ts <= dt64
        assert dt64 <= ts
        assert ts <= alt
        assert alt <= ts

    def test_cmp_cross_reso(self):
        # numpy gets this wrong because of silent overflow
        dt64 = np.datetime64(9223372800, "s")  # won't fit in M8[ns]
        ts = Timestamp._from_dt64(dt64)

        # subtracting 3600*24 gives a datetime64 that _can_ fit inside the
        #  nanosecond implementation bounds.
        other = Timestamp(dt64 - 3600 * 24)
        assert other < ts
        assert other.asm8 > ts.asm8  # <- numpy gets this wrong
        assert ts > other
        assert ts.asm8 < other.asm8  # <- numpy gets this wrong
        assert not other == ts
        assert ts != other

    @pytest.mark.xfail(reason="Dispatches to np.datetime64 which is wrong")
    def test_cmp_cross_reso_reversed_dt64(self):
        dt64 = np.datetime64(106752, "D")  # won't fit in M8[ns]
        ts = Timestamp._from_dt64(dt64)
        other = Timestamp(dt64 - 1)

        assert other.asm8 < ts

    def test_pickle(self, ts, tz_aware_fixture):
        tz = tz_aware_fixture
        tz = maybe_get_tz(tz)
        ts = Timestamp._from_value_and_reso(ts.value, ts._reso, tz)
        rt = tm.round_trip_pickle(ts)
        assert rt._reso == ts._reso
        assert rt == ts

    def test_normalize(self, dt64, ts):
        alt = Timestamp(dt64)
        result = ts.normalize()
        assert result._reso == ts._reso
        assert result == alt.normalize()

    def test_asm8(self, dt64, ts):
        rt = ts.asm8
        assert rt == dt64
        assert rt.dtype == dt64.dtype

    def test_to_numpy(self, dt64, ts):
        res = ts.to_numpy()
        assert res == dt64
        assert res.dtype == dt64.dtype

    def test_to_datetime64(self, dt64, ts):
        res = ts.to_datetime64()
        assert res == dt64
        assert res.dtype == dt64.dtype

    def test_timestamp(self, dt64, ts):
        alt = Timestamp(dt64)
        assert ts.timestamp() == alt.timestamp()

    def test_to_period(self, dt64, ts):
        alt = Timestamp(dt64)
        assert ts.to_period("D") == alt.to_period("D")

    @pytest.mark.parametrize(
        "td", [timedelta(days=4), Timedelta(days=4), np.timedelta64(4, "D")]
    )
    def test_addsub_timedeltalike_non_nano(self, dt64, ts, td):

        result = ts - td
        expected = Timestamp(dt64) - td
        assert isinstance(result, Timestamp)
        assert result._reso == ts._reso
        assert result == expected

        result = ts + td
        expected = Timestamp(dt64) + td
        assert isinstance(result, Timestamp)
        assert result._reso == ts._reso
        assert result == expected

        result = td + ts
        expected = td + Timestamp(dt64)
        assert isinstance(result, Timestamp)
        assert result._reso == ts._reso
        assert result == expected

    @pytest.mark.xfail(reason="tz_localize not yet implemented for non-nano")
    def test_addsub_offset(self, ts_tz):
        # specifically non-Tick offset
        off = offsets.YearBegin(1)
        result = ts_tz + off

        assert isinstance(result, Timestamp)
        assert result._reso == ts_tz._reso
        # If ts_tz is ever on the last day of the year, the year would be
        #  incremented by one
        assert result.year == ts_tz.year
        assert result.day == 31
        assert result.month == 12
        assert tz_compare(result.tz, ts_tz.tz)

        result = ts_tz - off

        assert isinstance(result, Timestamp)
        assert result._reso == ts_tz._reso
        assert result.year == ts_tz.year - 1
        assert result.day == 31
        assert result.month == 12
        assert tz_compare(result.tz, ts_tz.tz)

    def test_sub_datetimelike_mismatched_reso(self, ts_tz):
        # case with non-lossy rounding
        ts = ts_tz

        # choose a unit for `other` that doesn't match ts_tz's;
        #  this construction ensures we get cases with other._reso < ts._reso
        #  and cases with other._reso > ts._reso
        unit = {
            NpyDatetimeUnit.NPY_FR_us.value: "ms",
            NpyDatetimeUnit.NPY_FR_ms.value: "s",
            NpyDatetimeUnit.NPY_FR_s.value: "us",
        }[ts._reso]
        other = ts._as_unit(unit)
        assert other._reso != ts._reso

        result = ts - other
        assert isinstance(result, Timedelta)
        assert result.value == 0
        assert result._reso == min(ts._reso, other._reso)

        result = other - ts
        assert isinstance(result, Timedelta)
        assert result.value == 0
        assert result._reso == min(ts._reso, other._reso)

        msg = "Timestamp subtraction with mismatched resolutions"
        if ts._reso < other._reso:
            # Case where rounding is lossy
            other2 = other + Timedelta._from_value_and_reso(1, other._reso)
            with pytest.raises(ValueError, match=msg):
                ts - other2
            with pytest.raises(ValueError, match=msg):
                other2 - ts
        else:
            ts2 = ts + Timedelta._from_value_and_reso(1, ts._reso)
            with pytest.raises(ValueError, match=msg):
                ts2 - other
            with pytest.raises(ValueError, match=msg):
                other - ts2

    def test_sub_timedeltalike_mismatched_reso(self, ts_tz):
        # case with non-lossy rounding
        ts = ts_tz

        # choose a unit for `other` that doesn't match ts_tz's;
        #  this construction ensures we get cases with other._reso < ts._reso
        #  and cases with other._reso > ts._reso
        unit = {
            NpyDatetimeUnit.NPY_FR_us.value: "ms",
            NpyDatetimeUnit.NPY_FR_ms.value: "s",
            NpyDatetimeUnit.NPY_FR_s.value: "us",
        }[ts._reso]
        other = Timedelta(0)._as_unit(unit)
        assert other._reso != ts._reso

        result = ts + other
        assert isinstance(result, Timestamp)
        assert result == ts
        assert result._reso == min(ts._reso, other._reso)

        result = other + ts
        assert isinstance(result, Timestamp)
        assert result == ts
        assert result._reso == min(ts._reso, other._reso)

        msg = "Timestamp addition with mismatched resolutions"
        if ts._reso < other._reso:
            # Case where rounding is lossy
            other2 = other + Timedelta._from_value_and_reso(1, other._reso)
            with pytest.raises(ValueError, match=msg):
                ts + other2
            with pytest.raises(ValueError, match=msg):
                other2 + ts
        else:
            ts2 = ts + Timedelta._from_value_and_reso(1, ts._reso)
            with pytest.raises(ValueError, match=msg):
                ts2 + other
            with pytest.raises(ValueError, match=msg):
                other + ts2

        msg = "Addition between Timestamp and Timedelta with mismatched resolutions"
        with pytest.raises(ValueError, match=msg):
            # With a mismatched td64 as opposed to Timedelta
            ts + np.timedelta64(1, "ns")

    def test_min(self, ts):
        assert ts.min <= ts
        assert ts.min._reso == ts._reso
        assert ts.min.value == NaT.value + 1

    def test_max(self, ts):
        assert ts.max >= ts
        assert ts.max._reso == ts._reso
        assert ts.max.value == np.iinfo(np.int64).max

    def test_resolution(self, ts):
        expected = Timedelta._from_value_and_reso(1, ts._reso)
        result = ts.resolution
        assert result == expected
        assert result._reso == expected._reso


def test_timestamp_class_min_max_resolution():
    # when accessed on the class (as opposed to an instance), we default
    #  to nanoseconds
    assert Timestamp.min == Timestamp(NaT.value + 1)
    assert Timestamp.min._reso == NpyDatetimeUnit.NPY_FR_ns.value

    assert Timestamp.max == Timestamp(np.iinfo(np.int64).max)
    assert Timestamp.max._reso == NpyDatetimeUnit.NPY_FR_ns.value

    assert Timestamp.resolution == Timedelta(1)
    assert Timestamp.resolution._reso == NpyDatetimeUnit.NPY_FR_ns.value


class TestAsUnit:
    def test_as_unit(self):
        ts = Timestamp("1970-01-01")

        assert ts._as_unit("ns") is ts

        res = ts._as_unit("us")
        assert res.value == ts.value // 1000
        assert res._reso == NpyDatetimeUnit.NPY_FR_us.value

        rt = res._as_unit("ns")
        assert rt.value == ts.value
        assert rt._reso == ts._reso

        res = ts._as_unit("ms")
        assert res.value == ts.value // 1_000_000
        assert res._reso == NpyDatetimeUnit.NPY_FR_ms.value

        rt = res._as_unit("ns")
        assert rt.value == ts.value
        assert rt._reso == ts._reso

        res = ts._as_unit("s")
        assert res.value == ts.value // 1_000_000_000
        assert res._reso == NpyDatetimeUnit.NPY_FR_s.value

        rt = res._as_unit("ns")
        assert rt.value == ts.value
        assert rt._reso == ts._reso

    def test_as_unit_overflows(self):
        # microsecond that would be just out of bounds for nano
        us = 9223372800000000
        ts = Timestamp._from_value_and_reso(us, NpyDatetimeUnit.NPY_FR_us.value, None)

        msg = "Cannot cast 2262-04-12 00:00:00 to unit='ns' without overflow"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            ts._as_unit("ns")

        res = ts._as_unit("ms")
        assert res.value == us // 1000
        assert res._reso == NpyDatetimeUnit.NPY_FR_ms.value

    def test_as_unit_rounding(self):
        ts = Timestamp(1_500_000)  # i.e. 1500 microseconds
        res = ts._as_unit("ms")

        expected = Timestamp(1_000_000)  # i.e. 1 millisecond
        assert res == expected

        assert res._reso == NpyDatetimeUnit.NPY_FR_ms.value
        assert res.value == 1

        with pytest.raises(ValueError, match="Cannot losslessly convert units"):
            ts._as_unit("ms", round_ok=False)

    def test_as_unit_non_nano(self):
        # case where we are going neither to nor from nano
        ts = Timestamp("1970-01-02")._as_unit("ms")
        assert ts.year == 1970
        assert ts.month == 1
        assert ts.day == 2
        assert ts.hour == ts.minute == ts.second == ts.microsecond == ts.nanosecond == 0

        res = ts._as_unit("s")
        assert res.value == 24 * 3600
        assert res.year == 1970
        assert res.month == 1
        assert res.day == 2
        assert (
            res.hour
            == res.minute
            == res.second
            == res.microsecond
            == res.nanosecond
            == 0
        )

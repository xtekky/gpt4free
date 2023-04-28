cimport cython

import warnings

import numpy as np

from pandas.util._exceptions import find_stack_level

cimport numpy as cnp
from cpython.object cimport PyObject
from numpy cimport (
    int32_t,
    int64_t,
    intp_t,
    ndarray,
)

cnp.import_array()

import pytz

# stdlib datetime imports

from cpython.datetime cimport (
    PyDate_Check,
    PyDateTime_Check,
    datetime,
    import_datetime,
    time,
    tzinfo,
)

import_datetime()

from pandas._libs.tslibs.base cimport ABCTimestamp
from pandas._libs.tslibs.dtypes cimport (
    abbrev_to_npy_unit,
    periods_per_second,
)
from pandas._libs.tslibs.np_datetime cimport (
    NPY_DATETIMEUNIT,
    NPY_FR_ns,
    astype_overflowsafe,
    check_dts_bounds,
    dtstruct_to_dt64,
    get_datetime64_unit,
    get_datetime64_value,
    get_implementation_bounds,
    get_unit_from_dtype,
    npy_datetime,
    npy_datetimestruct,
    npy_datetimestruct_to_datetime,
    pandas_datetime_to_datetimestruct,
    pydatetime_to_dt64,
    pydatetime_to_dtstruct,
    string_to_dts,
)

from pandas._libs.tslibs.np_datetime import (
    OutOfBoundsDatetime,
    OutOfBoundsTimedelta,
)

from pandas._libs.tslibs.timezones cimport (
    get_utcoffset,
    is_utc,
    maybe_get_tz,
    tz_compare,
    utc_pytz as UTC,
)
from pandas._libs.tslibs.util cimport (
    is_datetime64_object,
    is_float_object,
    is_integer_object,
)

from pandas._libs.tslibs.parsing import parse_datetime_string

from pandas._libs.tslibs.nattype cimport (
    NPY_NAT,
    c_NaT as NaT,
    c_nat_strings as nat_strings,
    checknull_with_nat,
)
from pandas._libs.tslibs.tzconversion cimport (
    Localizer,
    tz_localize_to_utc_single,
)

# ----------------------------------------------------------------------
# Constants

DT64NS_DTYPE = np.dtype('M8[ns]')
TD64NS_DTYPE = np.dtype('m8[ns]')


# ----------------------------------------------------------------------
# Unit Conversion Helpers

cdef inline int64_t cast_from_unit(object ts, str unit) except? -1:
    """
    Return a casting of the unit represented to nanoseconds
    round the fractional part of a float to our precision, p.

    Parameters
    ----------
    ts : int, float, or None
    unit : str

    Returns
    -------
    int64_t
    """
    cdef:
        int64_t m
        int p

    m, p = precision_from_unit(unit)

    # just give me the unit back
    if ts is None:
        return m

    # cast the unit, multiply base/frace separately
    # to avoid precision issues from float -> int
    base = <int64_t>ts
    frac = ts - base
    if p:
        frac = round(frac, p)
    return <int64_t>(base * m) + <int64_t>(frac * m)


cpdef inline (int64_t, int) precision_from_unit(str unit):
    """
    Return a casting of the unit represented to nanoseconds + the precision
    to round the fractional part.

    Notes
    -----
    The caller is responsible for ensuring that the default value of "ns"
    takes the place of None.
    """
    cdef:
        int64_t m
        int p
        NPY_DATETIMEUNIT reso = abbrev_to_npy_unit(unit)

    if reso == NPY_DATETIMEUNIT.NPY_FR_Y:
        # each 400 years we have 97 leap years, for an average of 97/400=.2425
        #  extra days each year. We get 31556952 by writing
        #  3600*24*365.2425=31556952
        m = 1_000_000_000 * 31556952
        p = 9
    elif reso == NPY_DATETIMEUNIT.NPY_FR_M:
        # 2629746 comes from dividing the "Y" case by 12.
        m = 1_000_000_000 * 2629746
        p = 9
    elif reso == NPY_DATETIMEUNIT.NPY_FR_W:
        m = 1_000_000_000 * 3600 * 24 * 7
        p = 9
    elif reso == NPY_DATETIMEUNIT.NPY_FR_D:
        m = 1_000_000_000 * 3600 * 24
        p = 9
    elif reso == NPY_DATETIMEUNIT.NPY_FR_h:
        m = 1_000_000_000 * 3600
        p = 9
    elif reso == NPY_DATETIMEUNIT.NPY_FR_m:
        m = 1_000_000_000 * 60
        p = 9
    elif reso == NPY_DATETIMEUNIT.NPY_FR_s:
        m = 1_000_000_000
        p = 9
    elif reso == NPY_DATETIMEUNIT.NPY_FR_ms:
        m = 1_000_000
        p = 6
    elif reso == NPY_DATETIMEUNIT.NPY_FR_us:
        m = 1000
        p = 3
    elif reso == NPY_DATETIMEUNIT.NPY_FR_ns or reso == NPY_DATETIMEUNIT.NPY_FR_GENERIC:
        m = 1
        p = 0
    else:
        raise ValueError(f"cannot cast unit {unit}")
    return m, p


cdef inline int64_t get_datetime64_nanos(object val) except? -1:
    """
    Extract the value and unit from a np.datetime64 object, then convert the
    value to nanoseconds if necessary.
    """
    cdef:
        npy_datetimestruct dts
        NPY_DATETIMEUNIT unit
        npy_datetime ival

    ival = get_datetime64_value(val)
    if ival == NPY_NAT:
        return NPY_NAT

    unit = get_datetime64_unit(val)

    if unit != NPY_FR_ns:
        pandas_datetime_to_datetimestruct(ival, unit, &dts)
        check_dts_bounds(&dts)
        ival = dtstruct_to_dt64(&dts)

    return ival


# ----------------------------------------------------------------------
# _TSObject Conversion

# lightweight C object to hold datetime & int64 pair
cdef class _TSObject:
    # cdef:
    #    npy_datetimestruct dts      # npy_datetimestruct
    #    int64_t value               # numpy dt64
    #    tzinfo tzinfo
    #    bint fold

    def __cinit__(self):
        # GH 25057. As per PEP 495, set fold to 0 by default
        self.fold = 0


cdef _TSObject convert_to_tsobject(object ts, tzinfo tz, str unit,
                                   bint dayfirst, bint yearfirst, int32_t nanos=0):
    """
    Extract datetime and int64 from any of:
        - np.int64 (with unit providing a possible modifier)
        - np.datetime64
        - a float (with unit providing a possible modifier)
        - python int or long object (with unit providing a possible modifier)
        - iso8601 string object
        - python datetime object
        - another timestamp object

    Raises
    ------
    OutOfBoundsDatetime : ts cannot be converted within implementation bounds
    """
    cdef:
        _TSObject obj

    obj = _TSObject()

    if isinstance(ts, str):
        return _convert_str_to_tsobject(ts, tz, unit, dayfirst, yearfirst)

    if ts is None or ts is NaT:
        obj.value = NPY_NAT
    elif is_datetime64_object(ts):
        obj.value = get_datetime64_nanos(ts)
        if obj.value != NPY_NAT:
            pandas_datetime_to_datetimestruct(obj.value, NPY_FR_ns, &obj.dts)
    elif is_integer_object(ts):
        try:
            ts = <int64_t>ts
        except OverflowError:
            # GH#26651 re-raise as OutOfBoundsDatetime
            raise OutOfBoundsDatetime(f"Out of bounds nanosecond timestamp {ts}")
        if ts == NPY_NAT:
            obj.value = NPY_NAT
        else:
            if unit in ["Y", "M"]:
                # GH#47266 cast_from_unit leads to weird results e.g. with "Y"
                #  and 150 we'd get 2120-01-01 09:00:00
                ts = np.datetime64(ts, unit)
                return convert_to_tsobject(ts, tz, None, False, False)

            ts = ts * cast_from_unit(None, unit)
            obj.value = ts
            pandas_datetime_to_datetimestruct(ts, NPY_FR_ns, &obj.dts)
    elif is_float_object(ts):
        if ts != ts or ts == NPY_NAT:
            obj.value = NPY_NAT
        else:
            if unit in ["Y", "M"]:
                if ts == int(ts):
                    # GH#47266 Avoid cast_from_unit, which would give weird results
                    #  e.g. with "Y" and 150.0 we'd get 2120-01-01 09:00:00
                    return convert_to_tsobject(int(ts), tz, unit, False, False)
                else:
                    # GH#47267 it is clear that 2 "M" corresponds to 1970-02-01,
                    #  but not clear what 2.5 "M" corresponds to, so we will
                    #  disallow that case.
                    warnings.warn(
                        "Conversion of non-round float with unit={unit} is ambiguous "
                        "and will raise in a future version.",
                        FutureWarning,
                        stacklevel=find_stack_level(),
                    )

            ts = cast_from_unit(ts, unit)
            obj.value = ts
            pandas_datetime_to_datetimestruct(ts, NPY_FR_ns, &obj.dts)
    elif PyDateTime_Check(ts):
        return convert_datetime_to_tsobject(ts, tz, nanos)
    elif PyDate_Check(ts):
        # Keep the converter same as PyDateTime's
        ts = datetime.combine(ts, time())
        return convert_datetime_to_tsobject(ts, tz)
    else:
        from .period import Period
        if isinstance(ts, Period):
            raise ValueError("Cannot convert Period to Timestamp "
                             "unambiguously. Use to_timestamp")
        raise TypeError(f'Cannot convert input [{ts}] of type {type(ts)} to '
                        f'Timestamp')

    maybe_localize_tso(obj, tz, NPY_FR_ns)
    return obj


cdef maybe_localize_tso(_TSObject obj, tzinfo tz, NPY_DATETIMEUNIT reso):
    if tz is not None:
        _localize_tso(obj, tz, reso)

    if obj.value != NPY_NAT:
        # check_overflows needs to run after _localize_tso
        check_dts_bounds(&obj.dts, reso)
        check_overflows(obj, reso)


cdef _TSObject convert_datetime_to_tsobject(
    datetime ts,
    tzinfo tz,
    int32_t nanos=0,
    NPY_DATETIMEUNIT reso=NPY_FR_ns,
):
    """
    Convert a datetime (or Timestamp) input `ts`, along with optional timezone
    object `tz` to a _TSObject.

    The optional argument `nanos` allows for cases where datetime input
    needs to be supplemented with higher-precision information.

    Parameters
    ----------
    ts : datetime or Timestamp
        Value to be converted to _TSObject
    tz : tzinfo or None
        timezone for the timezone-aware output
    nanos : int32_t, default is 0
        nanoseconds supplement the precision of the datetime input ts
    reso : NPY_DATETIMEUNIT, default NPY_FR_ns

    Returns
    -------
    obj : _TSObject
    """
    cdef:
        _TSObject obj = _TSObject()
        int64_t pps

    obj.fold = ts.fold
    if tz is not None:
        tz = maybe_get_tz(tz)

        if ts.tzinfo is not None:
            # Convert the current timezone to the passed timezone
            ts = ts.astimezone(tz)
            pydatetime_to_dtstruct(ts, &obj.dts)
            obj.tzinfo = ts.tzinfo
        elif not is_utc(tz):
            ts = _localize_pydatetime(ts, tz)
            pydatetime_to_dtstruct(ts, &obj.dts)
            obj.tzinfo = ts.tzinfo
        else:
            # UTC
            pydatetime_to_dtstruct(ts, &obj.dts)
            obj.tzinfo = tz
    else:
        pydatetime_to_dtstruct(ts, &obj.dts)
        obj.tzinfo = ts.tzinfo

    if isinstance(ts, ABCTimestamp):
        obj.dts.ps = ts.nanosecond * 1000

    if nanos:
        obj.dts.ps = nanos * 1000

    obj.value = npy_datetimestruct_to_datetime(reso, &obj.dts)

    if obj.tzinfo is not None and not is_utc(obj.tzinfo):
        offset = get_utcoffset(obj.tzinfo, ts)
        pps = periods_per_second(reso)
        obj.value -= int(offset.total_seconds() * pps)

    check_dts_bounds(&obj.dts, reso)
    check_overflows(obj, reso)
    return obj


cdef _TSObject _create_tsobject_tz_using_offset(npy_datetimestruct dts,
                                                int tzoffset, tzinfo tz=None):
    """
    Convert a datetimestruct `dts`, along with initial timezone offset
    `tzoffset` to a _TSObject (with timezone object `tz` - optional).

    Parameters
    ----------
    dts: npy_datetimestruct
    tzoffset: int
    tz : tzinfo or None
        timezone for the timezone-aware output.

    Returns
    -------
    obj : _TSObject
    """
    cdef:
        _TSObject obj = _TSObject()
        int64_t value  # numpy dt64
        datetime dt
        Py_ssize_t pos

    value = dtstruct_to_dt64(&dts)
    obj.dts = dts
    obj.tzinfo = pytz.FixedOffset(tzoffset)
    obj.value = tz_localize_to_utc_single(value, obj.tzinfo)
    if tz is None:
        check_overflows(obj, NPY_FR_ns)
        return obj

    cdef:
        Localizer info = Localizer(tz, NPY_FR_ns)

    # Infer fold from offset-adjusted obj.value
    # see PEP 495 https://www.python.org/dev/peps/pep-0495/#the-fold-attribute
    if info.use_utc:
        pass
    elif info.use_tzlocal:
        info.utc_val_to_local_val(obj.value, &pos, &obj.fold)
    elif info.use_dst and not info.use_pytz:
        # i.e. dateutil
        info.utc_val_to_local_val(obj.value, &pos, &obj.fold)

    # Keep the converter same as PyDateTime's
    dt = datetime(obj.dts.year, obj.dts.month, obj.dts.day,
                  obj.dts.hour, obj.dts.min, obj.dts.sec,
                  obj.dts.us, obj.tzinfo, fold=obj.fold)
    obj = convert_datetime_to_tsobject(
        dt, tz, nanos=obj.dts.ps // 1000)
    return obj


cdef _TSObject _convert_str_to_tsobject(object ts, tzinfo tz, str unit,
                                        bint dayfirst=False,
                                        bint yearfirst=False):
    """
    Convert a string input `ts`, along with optional timezone object`tz`
    to a _TSObject.

    The optional arguments `dayfirst` and `yearfirst` are passed to the
    dateutil parser.

    Parameters
    ----------
    ts : str
        Value to be converted to _TSObject
    tz : tzinfo or None
        timezone for the timezone-aware output
    unit : str or None
    dayfirst : bool, default False
        When parsing an ambiguous date string, interpret e.g. "3/4/1975" as
        April 3, as opposed to the standard US interpretation March 4.
    yearfirst : bool, default False
        When parsing an ambiguous date string, interpret e.g. "01/05/09"
        as "May 9, 2001", as opposed to the default "Jan 5, 2009"

    Returns
    -------
    obj : _TSObject
    """
    cdef:
        npy_datetimestruct dts
        int out_local = 0, out_tzoffset = 0, string_to_dts_failed
        datetime dt
        int64_t ival
        NPY_DATETIMEUNIT out_bestunit

    if len(ts) == 0 or ts in nat_strings:
        ts = NaT
        obj = _TSObject()
        obj.value = NPY_NAT
        obj.tzinfo = tz
        return obj
    elif ts == 'now':
        # Issue 9000, we short-circuit rather than going
        # into np_datetime_strings which returns utc
        dt = datetime.now(tz)
    elif ts == 'today':
        # Issue 9000, we short-circuit rather than going
        # into np_datetime_strings which returns a normalized datetime
        dt = datetime.now(tz)
        # equiv: datetime.today().replace(tzinfo=tz)
    else:
        string_to_dts_failed = string_to_dts(
            ts, &dts, &out_bestunit, &out_local,
            &out_tzoffset, False
        )
        if not string_to_dts_failed:
            try:
                check_dts_bounds(&dts)
                if out_local == 1:
                    return _create_tsobject_tz_using_offset(dts,
                                                            out_tzoffset, tz)
                else:
                    ival = dtstruct_to_dt64(&dts)
                    if tz is not None:
                        # shift for _localize_tso
                        ival = tz_localize_to_utc_single(ival, tz,
                                                         ambiguous="raise")

                    return convert_to_tsobject(ival, tz, None, False, False)

            except OutOfBoundsDatetime:
                # GH#19382 for just-barely-OutOfBounds falling back to dateutil
                # parser will return incorrect result because it will ignore
                # nanoseconds
                raise

            except ValueError:
                # Fall through to parse_datetime_string
                pass

        try:
            dt = parse_datetime_string(ts, dayfirst=dayfirst,
                                       yearfirst=yearfirst)
        except (ValueError, OverflowError):
            raise ValueError("could not convert string to Timestamp")

    return convert_datetime_to_tsobject(dt, tz)


cdef inline check_overflows(_TSObject obj, NPY_DATETIMEUNIT reso=NPY_FR_ns):
    """
    Check that we haven't silently overflowed in timezone conversion

    Parameters
    ----------
    obj : _TSObject
    reso : NPY_DATETIMEUNIT, default NPY_FR_ns

    Returns
    -------
    None

    Raises
    ------
    OutOfBoundsDatetime
    """
    # GH#12677
    cdef:
        npy_datetimestruct lb, ub

    get_implementation_bounds(reso, &lb, &ub)

    if obj.dts.year == lb.year:
        if not (obj.value < 0):
            from pandas._libs.tslibs.timestamps import Timestamp
            fmt = (f"{obj.dts.year}-{obj.dts.month:02d}-{obj.dts.day:02d} "
                   f"{obj.dts.hour:02d}:{obj.dts.min:02d}:{obj.dts.sec:02d}")
            raise OutOfBoundsDatetime(
                f"Converting {fmt} underflows past {Timestamp.min}"
            )
    elif obj.dts.year == ub.year:
        if not (obj.value > 0):
            from pandas._libs.tslibs.timestamps import Timestamp
            fmt = (f"{obj.dts.year}-{obj.dts.month:02d}-{obj.dts.day:02d} "
                   f"{obj.dts.hour:02d}:{obj.dts.min:02d}:{obj.dts.sec:02d}")
            raise OutOfBoundsDatetime(
                f"Converting {fmt} overflows past {Timestamp.max}"
            )

# ----------------------------------------------------------------------
# Localization

cdef inline void _localize_tso(_TSObject obj, tzinfo tz, NPY_DATETIMEUNIT reso):
    """
    Given the UTC nanosecond timestamp in obj.value, find the wall-clock
    representation of that timestamp in the given timezone.

    Parameters
    ----------
    obj : _TSObject
    tz : tzinfo
    reso : NPY_DATETIMEUNIT

    Returns
    -------
    None

    Notes
    -----
    Sets obj.tzinfo inplace, alters obj.dts inplace.
    """
    cdef:
        int64_t local_val
        Py_ssize_t outpos = -1
        Localizer info = Localizer(tz, reso)

    assert obj.tzinfo is None

    if info.use_utc:
        pass
    elif obj.value == NPY_NAT:
        pass
    else:
        local_val = info.utc_val_to_local_val(obj.value, &outpos, &obj.fold)

        if info.use_pytz:
            # infer we went through a pytz path, will have outpos!=-1
            tz = tz._tzinfos[tz._transition_info[outpos]]

        pandas_datetime_to_datetimestruct(local_val, reso, &obj.dts)

    obj.tzinfo = tz


cdef inline datetime _localize_pydatetime(datetime dt, tzinfo tz):
    """
    Take a datetime/Timestamp in UTC and localizes to timezone tz.

    NB: Unlike the public version, this treats datetime and Timestamp objects
        identically, i.e. discards nanos from Timestamps.
        It also assumes that the `tz` input is not None.
    """
    try:
        # datetime.replace with pytz may be incorrect result
        return tz.localize(dt)
    except AttributeError:
        return dt.replace(tzinfo=tz)


cpdef inline datetime localize_pydatetime(datetime dt, tzinfo tz):
    """
    Take a datetime/Timestamp in UTC and localizes to timezone tz.

    Parameters
    ----------
    dt : datetime or Timestamp
    tz : tzinfo or None

    Returns
    -------
    localized : datetime or Timestamp
    """
    if tz is None:
        return dt
    elif isinstance(dt, ABCTimestamp):
        return dt.tz_localize(tz)
    return _localize_pydatetime(dt, tz)

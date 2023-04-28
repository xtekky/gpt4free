from datetime import datetime
import pytz_deprecation_shim as pds

try:
    import _winreg as winreg
except ImportError:
    import winreg

from tzlocal.windows_tz import win_tz
from tzlocal import utils

_cache_tz = None
_cache_tz_name = None


def valuestodict(key):
    """Convert a registry key's values to a dictionary."""
    result = {}
    size = winreg.QueryInfoKey(key)[1]
    for i in range(size):
        data = winreg.EnumValue(key, i)
        result[data[0]] = data[1]
    return result


def _get_dst_info(tz):
    # Find the offset for when it doesn't have DST:
    dst_offset = std_offset = None
    has_dst = False
    year = datetime.now().year
    for dt in (datetime(year, 1, 1), datetime(year, 6, 1)):
        if tz.dst(dt).total_seconds() == 0.0:
            # OK, no DST during winter, get this offset
            std_offset = tz.utcoffset(dt).total_seconds()
        else:
            has_dst = True

    return has_dst, std_offset, dst_offset


def _get_localzone_name():
    # Windows is special. It has unique time zone names (in several
    # meanings of the word) available, but unfortunately, they can be
    # translated to the language of the operating system, so we need to
    # do a backwards lookup, by going through all time zones and see which
    # one matches.
    tzenv = utils._tz_name_from_env()
    if tzenv:
        return tzenv

    handle = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)

    TZLOCALKEYNAME = r"SYSTEM\CurrentControlSet\Control\TimeZoneInformation"
    localtz = winreg.OpenKey(handle, TZLOCALKEYNAME)
    keyvalues = valuestodict(localtz)
    localtz.Close()

    if "TimeZoneKeyName" in keyvalues:
        # Windows 7 and later

        # For some reason this returns a string with loads of NUL bytes at
        # least on some systems. I don't know if this is a bug somewhere, I
        # just work around it.
        tzkeyname = keyvalues["TimeZoneKeyName"].split("\x00", 1)[0]
    else:
        # Don't support XP any longer
        raise LookupError("Can not find Windows timezone configuration")

    timezone = win_tz.get(tzkeyname)
    if timezone is None:
        # Nope, that didn't work. Try adding "Standard Time",
        # it seems to work a lot of times:
        timezone = win_tz.get(tzkeyname + " Standard Time")

    # Return what we have.
    if timezone is None:
        raise utils.ZoneInfoNotFoundError(tzkeyname)

    if keyvalues.get("DynamicDaylightTimeDisabled", 0) == 1:
        # DST is disabled, so don't return the timezone name,
        # instead return Etc/GMT+offset

        tz = pds.timezone(timezone)
        has_dst, std_offset, dst_offset = _get_dst_info(tz)
        if not has_dst:
            # The DST is turned off in the windows configuration,
            # but this timezone doesn't have DST so it doesn't matter
            return timezone

        if std_offset is None:
            raise utils.ZoneInfoNotFoundError(
                f"{tzkeyname} claims to not have a non-DST time!?"
            )

        if std_offset % 3600:
            # I can't convert this to an hourly offset
            raise utils.ZoneInfoNotFoundError(
                f"tzlocal can't support disabling DST in the {timezone} zone."
            )

        # This has whole hours as offset, return it as Etc/GMT
        return f"Etc/GMT{-std_offset//3600:+.0f}"

    return timezone


def get_localzone_name():
    """Get the zoneinfo timezone name that matches the Windows-configured timezone."""
    global _cache_tz_name
    if _cache_tz_name is None:
        _cache_tz_name = _get_localzone_name()

    return _cache_tz_name


def get_localzone():
    """Returns the zoneinfo-based tzinfo object that matches the Windows-configured timezone."""

    global _cache_tz
    if _cache_tz is None:
        _cache_tz = pds.timezone(get_localzone_name())

    if not utils._tz_name_from_env():
        # If the timezone does NOT come from a TZ environment variable,
        # verify that it's correct. If it's from the environment,
        # we accept it, this is so you can run tests with different timezones.
        utils.assert_tz_offset(_cache_tz)

    return _cache_tz


def reload_localzone():
    """Reload the cached localzone. You need to call this if the timezone has changed."""
    global _cache_tz
    global _cache_tz_name
    _cache_tz_name = _get_localzone_name()
    _cache_tz = pds.timezone(_cache_tz_name)
    utils.assert_tz_offset(_cache_tz)
    return _cache_tz

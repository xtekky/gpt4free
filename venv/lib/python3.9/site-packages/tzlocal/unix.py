import os
import re
import sys
import warnings
from datetime import timezone
import pytz_deprecation_shim as pds

from tzlocal import utils

if sys.version_info >= (3, 9):
    from zoneinfo import ZoneInfo  # pragma: no cover
else:
    from backports.zoneinfo import ZoneInfo  # pragma: no cover

_cache_tz = None
_cache_tz_name = None


def _get_localzone_name(_root="/"):
    """Tries to find the local timezone configuration.

    This method finds the timezone name, if it can, or it returns None.

    The parameter _root makes the function look for files like /etc/localtime
    beneath the _root directory. This is primarily used by the tests.
    In normal usage you call the function without parameters."""

    # First try the ENV setting.
    tzenv = utils._tz_name_from_env()
    if tzenv:
        return tzenv

    # Are we under Termux on Android?
    if os.path.exists(os.path.join(_root, "system/bin/getprop")):
        import subprocess

        try:
            androidtz = (
                subprocess.check_output(["getprop", "persist.sys.timezone"])
                .strip()
                .decode()
            )
            return androidtz
        except (OSError, subprocess.CalledProcessError):
            # proot environment or failed to getprop
            pass

    # Now look for distribution specific configuration files
    # that contain the timezone name.

    # Stick all of them in a dict, to compare later.
    found_configs = {}

    for configfile in ("etc/timezone", "var/db/zoneinfo"):
        tzpath = os.path.join(_root, configfile)
        try:
            with open(tzpath) as tzfile:
                data = tzfile.read()

                etctz = data.strip("/ \t\r\n")
                if not etctz:
                    # Empty file, skip
                    continue
                for etctz in etctz.splitlines():
                    # Get rid of host definitions and comments:
                    if " " in etctz:
                        etctz, dummy = etctz.split(" ", 1)
                    if "#" in etctz:
                        etctz, dummy = etctz.split("#", 1)
                    if not etctz:
                        continue

                    found_configs[tzpath] = etctz.replace(" ", "_")

        except (OSError, UnicodeDecodeError):
            # File doesn't exist or is a directory, or it's a binary file.
            continue

    # CentOS has a ZONE setting in /etc/sysconfig/clock,
    # OpenSUSE has a TIMEZONE setting in /etc/sysconfig/clock and
    # Gentoo has a TIMEZONE setting in /etc/conf.d/clock
    # We look through these files for a timezone:

    zone_re = re.compile(r"\s*ZONE\s*=\s*\"")
    timezone_re = re.compile(r"\s*TIMEZONE\s*=\s*\"")
    end_re = re.compile('"')

    for filename in ("etc/sysconfig/clock", "etc/conf.d/clock"):
        tzpath = os.path.join(_root, filename)
        try:
            with open(tzpath, "rt") as tzfile:
                data = tzfile.readlines()

            for line in data:
                # Look for the ZONE= setting.
                match = zone_re.match(line)
                if match is None:
                    # No ZONE= setting. Look for the TIMEZONE= setting.
                    match = timezone_re.match(line)
                if match is not None:
                    # Some setting existed
                    line = line[match.end() :]
                    etctz = line[: end_re.search(line).start()]

                    # We found a timezone
                    found_configs[tzpath] = etctz.replace(" ", "_")

        except (OSError, UnicodeDecodeError):
            # UnicodeDecode handles when clock is symlink to /etc/localtime
            continue

    # systemd distributions use symlinks that include the zone name,
    # see manpage of localtime(5) and timedatectl(1)
    tzpath = os.path.join(_root, "etc/localtime")
    if os.path.exists(tzpath) and os.path.islink(tzpath):
        etctz = os.path.realpath(tzpath)
        start = etctz.find("/") + 1
        while start != 0:
            etctz = etctz[start:]
            try:
                pds.timezone(etctz)
                tzinfo = f"{tzpath} is a symlink to"
                found_configs[tzinfo] = etctz.replace(" ", "_")
                # Only need first valid relative path in simlink.
                break
            except pds.UnknownTimeZoneError:
                pass
            start = etctz.find("/") + 1

    if len(found_configs) > 0:
        # We found some explicit config of some sort!
        if len(found_configs) > 1:
            # Uh-oh, multiple configs. See if they match:
            unique_tzs = set()
            zoneinfo = os.path.join(_root, "usr", "share", "zoneinfo")
            directory_depth = len(zoneinfo.split(os.path.sep))

            for tzname in found_configs.values():
                # Look them up in /usr/share/zoneinfo, and find what they
                # really point to:
                path = os.path.realpath(os.path.join(zoneinfo, *tzname.split("/")))
                real_zone_name = "/".join(path.split(os.path.sep)[directory_depth:])
                unique_tzs.add(real_zone_name)

            if len(unique_tzs) != 1:
                message = "Multiple conflicting time zone configurations found:\n"
                for key, value in found_configs.items():
                    message += f"{key}: {value}\n"
                message += "Fix the configuration, or set the time zone in a TZ environment variable.\n"
                raise utils.ZoneInfoNotFoundError(message)

        # We found exactly one config! Use it.
        return list(found_configs.values())[0]


def _get_localzone(_root="/"):
    """Creates a timezone object from the timezone name.

    If there is no timezone config, it will try to create a file from the
    localtime timezone, and if there isn't one, it will default to UTC.

    The parameter _root makes the function look for files like /etc/localtime
    beneath the _root directory. This is primarily used by the tests.
    In normal usage you call the function without parameters."""

    # First try the ENV setting.
    tzenv = utils._tz_from_env()
    if tzenv:
        return tzenv

    tzname = _get_localzone_name(_root)
    if tzname is None:
        # No explicit setting existed. Use localtime
        for filename in ("etc/localtime", "usr/local/etc/localtime"):
            tzpath = os.path.join(_root, filename)

            if not os.path.exists(tzpath):
                continue
            with open(tzpath, "rb") as tzfile:
                tz = pds.wrap_zone(ZoneInfo.from_file(tzfile, key="local"))
                break
        else:
            warnings.warn("Can not find any timezone configuration, defaulting to UTC.")
            tz = timezone.utc
    else:
        tz = pds.timezone(tzname)

    if _root == "/":
        # We are using a file in etc to name the timezone.
        # Verify that the timezone specified there is actually used:
        utils.assert_tz_offset(tz)
    return tz


def get_localzone_name():
    """Get the computers configured local timezone name, if any."""
    global _cache_tz_name
    if _cache_tz_name is None:
        _cache_tz_name = _get_localzone_name()

    return _cache_tz_name


def get_localzone():
    """Get the computers configured local timezone, if any."""

    global _cache_tz
    if _cache_tz is None:
        _cache_tz = _get_localzone()

    return _cache_tz


def reload_localzone():
    """Reload the cached localzone. You need to call this if the timezone has changed."""
    global _cache_tz_name
    global _cache_tz
    _cache_tz_name = _get_localzone_name()
    _cache_tz = _get_localzone()

    return _cache_tz

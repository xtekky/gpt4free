# -*- coding: utf-8 -*-
import warnings
from datetime import tzinfo

from . import _compat
from ._exceptions import (
    AmbiguousTimeError,
    NonExistentTimeError,
    PytzUsageWarning,
    UnknownTimeZoneError,
    get_exception,
)

IS_DST_SENTINEL = object()
KEY_SENTINEL = object()


def timezone(key, _cache={}):
    """Builds an IANA database time zone shim.

    This is the equivalent of ``pytz.timezone``.

    :param key:
        A valid key from the IANA time zone database.

    :raises UnknownTimeZoneError:
        If an unknown value is passed, this will raise an exception that can be
        caught by :exc:`pytz_deprecation_shim.UnknownTimeZoneError` or
        ``pytz.UnknownTimeZoneError``. Like
        :exc:`zoneinfo.ZoneInfoNotFoundError`, both of those are subclasses of
        :exc:`KeyError`.
    """
    instance = _cache.get(key, None)
    if instance is None:
        if len(key) == 3 and key.lower() == "utc":
            instance = _cache.setdefault(key, UTC)
        else:
            try:
                zone = _compat.get_timezone(key)
            except KeyError:
                raise get_exception(UnknownTimeZoneError, key)
            instance = _cache.setdefault(key, wrap_zone(zone, key=key))

    return instance


def fixed_offset_timezone(offset, _cache={}):
    """Builds a fixed offset time zone shim.

    This is the equivalent of ``pytz.FixedOffset``. An alias is available as
    ``pytz_deprecation_shim.FixedOffset`` as well.

    :param offset:
        A fixed offset from UTC, in minutes. This must be in the range ``-1439
        <= offset <= 1439``.

    :raises ValueError:
        For offsets whose absolute value is greater than or equal to 24 hours.

    :return:
        A shim time zone.
    """
    if not (-1440 < offset < 1440):
        raise ValueError("absolute offset is too large", offset)

    instance = _cache.get(offset, None)
    if instance is None:
        if offset == 0:
            instance = _cache.setdefault(offset, UTC)
        else:
            zone = _compat.get_fixed_offset_zone(offset)
            instance = _cache.setdefault(offset, wrap_zone(zone, key=None))

    return instance


def build_tzinfo(zone, fp):
    """Builds a shim object from a TZif file.

    This is a shim for ``pytz.build_tzinfo``. Given a value to use as the zone
    IANA key and a file-like object containing a valid TZif file (i.e.
    conforming to :rfc:`8536`), this builds a time zone object and wraps it in
    a shim class.

    The argument names are chosen to match those in ``pytz.build_tzinfo``.

    :param zone:
        A string to be used as the time zone object's IANA key.

    :param fp:
        A readable file-like object emitting bytes, pointing to a valid TZif
        file.

    :return:
        A shim time zone.
    """
    zone_file = _compat.get_timezone_file(fp)

    return wrap_zone(zone_file, key=zone)


def wrap_zone(tz, key=KEY_SENTINEL, _cache={}):
    """Wrap an existing time zone object in a shim class.

    This is likely to be useful if you would like to work internally with
    non-``pytz`` zones, but you expose an interface to callers relying on
    ``pytz``'s interface. It may also be useful for passing non-``pytz`` zones
    to libraries expecting to use ``pytz``'s interface.

    :param tz:
        A :pep:`495`-compatible time zone, such as those provided by
        :mod:`dateutil.tz` or :mod:`zoneinfo`.

    :param key:
        The value for the IANA time zone key. This is optional for ``zoneinfo``
        zones, but required for ``dateutil.tz`` zones.

    :return:
        A shim time zone.
    """
    if key is KEY_SENTINEL:
        key = getattr(tz, "key", KEY_SENTINEL)

    if key is KEY_SENTINEL:
        raise TypeError(
            "The `key` argument is required when wrapping zones that do not "
            + "have a `key` attribute."
        )

    instance = _cache.get((id(tz), key), None)
    if instance is None:
        instance = _cache.setdefault((id(tz), key), _PytzShimTimezone(tz, key))

    return instance


class _PytzShimTimezone(tzinfo):
    # Add instance variables for _zone and _key because this will make error
    # reporting with partially-initialized _BasePytzShimTimezone objects
    # work better.
    _zone = None
    _key = None

    def __init__(self, zone, key):
        self._key = key
        self._zone = zone

    def utcoffset(self, dt):
        return self._zone.utcoffset(dt)

    def dst(self, dt):
        return self._zone.dst(dt)

    def tzname(self, dt):
        return self._zone.tzname(dt)

    def fromutc(self, dt):
        # The default fromutc implementation only works if tzinfo is "self"
        dt_base = dt.replace(tzinfo=self._zone)
        dt_out = self._zone.fromutc(dt_base)

        return dt_out.replace(tzinfo=self)

    def __str__(self):
        if self._key is not None:
            return str(self._key)
        else:
            return repr(self)

    def __repr__(self):
        return "%s(%s, %s)" % (
            self.__class__.__name__,
            repr(self._zone),
            repr(self._key),
        )

    def unwrap_shim(self):
        """Returns the underlying class that the shim is a wrapper for.

        This is a shim-specific method equivalent to
        :func:`pytz_deprecation_shim.helpers.upgrade_tzinfo`. It is provided as
        a method to allow end-users to upgrade shim timezones without requiring
        an explicit dependency on ``pytz_deprecation_shim``, e.g.:

        .. code-block:: python

            if getattr(tz, "unwrap_shim", None) is None:
                tz = tz.unwrap_shim()
        """
        return self._zone

    @property
    def zone(self):
        warnings.warn(
            "The zone attribute is specific to pytz's interface; "
            + "please migrate to a new time zone provider. "
            + "For more details on how to do so, see %s"
            % PYTZ_MIGRATION_GUIDE_URL,
            PytzUsageWarning,
            stacklevel=2,
        )

        return self._key

    def localize(self, dt, is_dst=IS_DST_SENTINEL):
        warnings.warn(
            "The localize method is no longer necessary, as this "
            + "time zone supports the fold attribute (PEP 495). "
            + "For more details on migrating to a PEP 495-compliant "
            + "implementation, see %s" % PYTZ_MIGRATION_GUIDE_URL,
            PytzUsageWarning,
            stacklevel=2,
        )

        if dt.tzinfo is not None:
            raise ValueError("Not naive datetime (tzinfo is already set)")

        dt_out = dt.replace(tzinfo=self)

        if is_dst is IS_DST_SENTINEL:
            return dt_out

        dt_ambiguous = _compat.is_ambiguous(dt_out)
        dt_imaginary = (
            _compat.is_imaginary(dt_out) if not dt_ambiguous else False
        )

        if is_dst is None:
            if dt_imaginary:
                raise get_exception(
                    NonExistentTimeError, dt.replace(tzinfo=None)
                )

            if dt_ambiguous:
                raise get_exception(AmbiguousTimeError, dt.replace(tzinfo=None))

        elif dt_ambiguous or dt_imaginary:
            # Start by normalizing the folds; dt_out may have fold=0 or fold=1,
            # but we need to know the DST offset on both sides anyway, so we
            # will get one datetime representing each side of the fold, then
            # decide which one we're going to return.
            if _compat.get_fold(dt_out):
                dt_enfolded = dt_out
                dt_out = _compat.enfold(dt_out, fold=0)
            else:
                dt_enfolded = _compat.enfold(dt_out, fold=1)

            # Now we want to decide whether the fold=0 or fold=1 represents
            # what pytz would return for `is_dst=True`
            enfolded_dst = bool(dt_enfolded.dst())
            if bool(dt_out.dst()) == enfolded_dst:
                # If this is not a transition between standard time and
                # daylight saving time, pytz will consider the larger offset
                # the DST offset.
                enfolded_dst = dt_enfolded.utcoffset() > dt_out.utcoffset()

            # The default we've established is that dt_out is fold=0; swap it
            # for the fold=1 datetime if is_dst == True and the enfolded side
            # is DST or if is_dst == False and the enfolded side is *not* DST.
            if is_dst == enfolded_dst:
                dt_out = dt_enfolded

        return dt_out

    def normalize(self, dt):
        warnings.warn(
            "The normalize method is no longer necessary, as this "
            + "time zone supports the fold attribute (PEP 495). "
            + "For more details on migrating to a PEP 495-compliant "
            + "implementation, see %s" % PYTZ_MIGRATION_GUIDE_URL,
            PytzUsageWarning,
            stacklevel=2,
        )

        if dt.tzinfo is None:
            raise ValueError("Naive time - no tzinfo set")

        if dt.tzinfo is self:
            return dt

        return dt.astimezone(self)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo=None):
        return self

    def __reduce__(self):
        return wrap_zone, (self._zone, self._key)


UTC = wrap_zone(_compat.UTC, "UTC")
PYTZ_MIGRATION_GUIDE_URL = (
    "https://pytz-deprecation-shim.readthedocs.io/en/latest/migration.html"
)

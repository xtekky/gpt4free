"""
This module contains helper functions to ease the transition from ``pytz`` to
another :pep:`495`-compatible library.
"""
from . import _common, _compat
from ._impl import _PytzShimTimezone

_PYTZ_BASE_CLASSES = None


def is_pytz_zone(tz):
    """Check if a time zone is a ``pytz`` time zone.

    This will only import ``pytz`` if it has already been imported, and does
    not rely on the existence of the ``localize`` or ``normalize`` methods
    (since the shim classes also have these methods, but are not ``pytz``
    zones).
    """

    # If pytz is not in sys.modules, then we will assume the time zone is not a
    # pytz zone. It is possible that someone has manipulated sys.modules to
    # remove pytz, but that's the kind of thing that causes all kinds of other
    # problems anyway, so we'll call that an unsupported configuration.
    if not _common.pytz_imported():
        return False

    if _PYTZ_BASE_CLASSES is None:
        _populate_pytz_base_classes()

    return isinstance(tz, _PYTZ_BASE_CLASSES)


def upgrade_tzinfo(tz):
    """Convert a ``pytz`` or shim timezone into its modern equivalent.

    The shim classes are thin wrappers around :mod:`zoneinfo` or
    :mod:`dateutil.tz` implementations of the :class:`datetime.tzinfo` base
    class. This function removes the shim and returns the underlying "upgraded"
    time zone.

    When passed a ``pytz`` zone (not a shim), this returns the non-``pytz``
    equivalent. This may fail if ``pytz`` is using a data source incompatible
    with the upgraded provider's data source, or if the ``pytz`` zone was built
    from a file rather than an IANA key.

    When passed an object that is not a shim or a ``pytz`` zone, this returns
    the original object.

    :param tz:
        A :class:`datetime.tzinfo` object.

    :raises KeyError:
        If a ``pytz`` zone is passed to the function with no equivalent in the
        :pep:`495`-compatible library's version of the Olson database.

    :return:
        A :pep:`495`-compatible equivalent of any ``pytz`` or shim
        class, or the original object.
    """
    if isinstance(tz, _PytzShimTimezone):
        return tz._zone

    if is_pytz_zone(tz):
        if tz.zone is None:
            # This is a fixed offset zone
            offset = tz.utcoffset(None)
            offset_minutes = offset.total_seconds() / 60

            return _compat.get_fixed_offset_zone(offset_minutes)

        if tz.zone == "UTC":
            return _compat.UTC

        return _compat.get_timezone(tz.zone)

    return tz


def _populate_pytz_base_classes():
    import pytz
    from pytz.tzinfo import BaseTzInfo

    base_classes = (BaseTzInfo, pytz._FixedOffset)

    # In releases prior to 2018.4, pytz.UTC was not a subclass of BaseTzInfo
    if not isinstance(pytz.UTC, BaseTzInfo):  # pragma: nocover
        base_classes = base_classes + (type(pytz.UTC),)

    global _PYTZ_BASE_CLASSES
    _PYTZ_BASE_CLASSES = base_classes

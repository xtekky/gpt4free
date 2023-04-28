from ._common import pytz_imported


class PytzUsageWarning(RuntimeWarning):
    """Warning raised when accessing features specific to ``pytz``'s interface.

    This warning is used to direct users of ``pytz``-specific features like the
    ``localize`` and ``normalize`` methods towards using the standard
    ``tzinfo`` interface, so that these shims can be replaced with one of the
    underlying libraries they are wrapping.
    """


class UnknownTimeZoneError(KeyError):
    """Raised when no time zone is found for a specified key."""


class InvalidTimeError(Exception):
    """The base class for exceptions related to folds and gaps."""


class AmbiguousTimeError(InvalidTimeError):
    """Exception raised when ``is_dst=None`` for an ambiguous time (fold)."""


class NonExistentTimeError(InvalidTimeError):
    """Exception raised when ``is_dst=None`` for a non-existent time (gap)."""


PYTZ_BASE_ERROR_MAPPING = {}


def _make_pytz_derived_errors(
    InvalidTimeError_=InvalidTimeError,
    AmbiguousTimeError_=AmbiguousTimeError,
    NonExistentTimeError_=NonExistentTimeError,
    UnknownTimeZoneError_=UnknownTimeZoneError,
):
    if PYTZ_BASE_ERROR_MAPPING or not pytz_imported():
        return

    import pytz

    class InvalidTimeError(InvalidTimeError_, pytz.InvalidTimeError):
        pass

    class AmbiguousTimeError(AmbiguousTimeError_, pytz.AmbiguousTimeError):
        pass

    class NonExistentTimeError(
        NonExistentTimeError_, pytz.NonExistentTimeError
    ):
        pass

    class UnknownTimeZoneError(
        UnknownTimeZoneError_, pytz.UnknownTimeZoneError
    ):
        pass

    PYTZ_BASE_ERROR_MAPPING.update(
        {
            InvalidTimeError_: InvalidTimeError,
            AmbiguousTimeError_: AmbiguousTimeError,
            NonExistentTimeError_: NonExistentTimeError,
            UnknownTimeZoneError_: UnknownTimeZoneError,
        }
    )


def get_exception(exc_type, msg):
    _make_pytz_derived_errors()

    out_exc_type = PYTZ_BASE_ERROR_MAPPING.get(exc_type, exc_type)

    return out_exc_type(msg)

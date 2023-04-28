import sys

_PYTZ_IMPORTED = False


def pytz_imported():
    """Detects whether or not pytz has been imported without importing pytz."""
    global _PYTZ_IMPORTED

    if not _PYTZ_IMPORTED and "pytz" in sys.modules:
        _PYTZ_IMPORTED = True

    return _PYTZ_IMPORTED

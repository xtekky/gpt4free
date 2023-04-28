import sys

if sys.platform == "win32":
    from tzlocal.win32 import (
        get_localzone,
        get_localzone_name,
        reload_localzone,
    )  # pragma: no cover
else:
    from tzlocal.unix import get_localzone, get_localzone_name, reload_localzone


__all__ = ["get_localzone", "get_localzone_name", "reload_localzone"]

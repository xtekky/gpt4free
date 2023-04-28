import os


try:
    from test.support import import_helper  # type: ignore
except ImportError:
    # Python 3.9 and earlier
    class import_helper:  # type: ignore
        from test.support import (
            modules_setup,
            modules_cleanup,
            DirsOnSysPath,
            CleanImport,
        )


try:
    from test.support import os_helper  # type: ignore
except ImportError:
    # Python 3.9 compat
    class os_helper:  # type:ignore
        from test.support import temp_dir


try:
    # Python 3.10
    from test.support.os_helper import unlink
except ImportError:
    from test.support import unlink as _unlink

    def unlink(target):
        return _unlink(os.fspath(target))

# being a bit too dynamic
from __future__ import annotations

import operator

from pandas.util.version import Version


def _mpl_version(version, op):
    def inner():
        try:
            import matplotlib as mpl
        except ImportError:
            return False
        return op(Version(mpl.__version__), Version(version))

    return inner


mpl_ge_3_4_0 = _mpl_version("3.4.0", operator.ge)
mpl_ge_3_5_0 = _mpl_version("3.5.0", operator.ge)
mpl_ge_3_6_0 = _mpl_version("3.6.0", operator.ge)

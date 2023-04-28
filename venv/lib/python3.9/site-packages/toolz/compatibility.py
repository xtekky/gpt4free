import warnings
warnings.warn("The toolz.compatibility module is no longer "
              "needed in Python 3 and has been deprecated. Please "
              "import these utilities directly from the standard library. "
              "This module will be removed in a future release.",
              category=DeprecationWarning, stacklevel=2)

import operator
import sys

PY3 = sys.version_info[0] > 2
PY34 = sys.version_info[0] == 3 and sys.version_info[1] == 4
PYPY = hasattr(sys, 'pypy_version_info') and PY3

__all__ = ('map', 'filter', 'range', 'zip', 'reduce', 'zip_longest',
           'iteritems', 'iterkeys', 'itervalues', 'filterfalse',
           'PY3', 'PY34', 'PYPY')


map = map
filter = filter
range = range
zip = zip
from functools import reduce
from itertools import zip_longest
from itertools import filterfalse
iteritems = operator.methodcaller('items')
iterkeys = operator.methodcaller('keys')
itervalues = operator.methodcaller('values')
from collections.abc import Sequence

from .itertoolz import *

from .functoolz import *

from .dicttoolz import *

from .recipes import *

from functools import partial, reduce

sorted = sorted

map = map

filter = filter

# Aliases
comp = compose

from . import curried, sandbox

functoolz._sigs.create_signature_registry()

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

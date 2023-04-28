from __future__ import absolute_import

import operator

from toolz.functoolz import curry


# Tests will catch if/when this needs updated
IGNORE = {
    "__abs__", "__index__", "__inv__", "__invert__", "__neg__", "__not__",
    "__pos__", "_abs", "abs", "attrgetter", "index", "inv", "invert",
    "itemgetter", "neg", "not_", "pos", "truth"
}
locals().update(
    {name: f if name in IGNORE else curry(f)
     for name, f in vars(operator).items() if callable(f)}
)

# Clean up the namespace.
del IGNORE
del curry
del operator

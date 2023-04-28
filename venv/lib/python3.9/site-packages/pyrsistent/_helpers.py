from functools import wraps
from pyrsistent._pmap import PMap, pmap
from pyrsistent._pset import PSet, pset
from pyrsistent._pvector import PVector, pvector

def freeze(o, strict=True):
    """
    Recursively convert simple Python containers into pyrsistent versions
    of those containers.

    - list is converted to pvector, recursively
    - dict is converted to pmap, recursively on values (but not keys)
    - set is converted to pset, but not recursively
    - tuple is converted to tuple, recursively.

    If strict == True (default):

    - freeze is called on elements of pvectors
    - freeze is called on values of pmaps

    Sets and dict keys are not recursively frozen because they do not contain
    mutable data by convention. The main exception to this rule is that
    dict keys and set elements are often instances of mutable objects that
    support hash-by-id, which this function can't convert anyway.

    >>> freeze(set([1, 2]))
    pset([1, 2])
    >>> freeze([1, {'a': 3}])
    pvector([1, pmap({'a': 3})])
    >>> freeze((1, []))
    (1, pvector([]))
    """
    typ = type(o)
    if typ is dict or (strict and isinstance(o, PMap)):
        return pmap({k: freeze(v, strict) for k, v in o.items()})
    if typ is list or (strict and isinstance(o, PVector)):
        curried_freeze = lambda x: freeze(x, strict)
        return pvector(map(curried_freeze, o))
    if typ is tuple:
        curried_freeze = lambda x: freeze(x, strict)
        return tuple(map(curried_freeze, o))
    if typ is set:
        # impossible to have anything that needs freezing inside a set or pset
        return pset(o)
    return o


def thaw(o, strict=True):
    """
    Recursively convert pyrsistent containers into simple Python containers.

    - pvector is converted to list, recursively
    - pmap is converted to dict, recursively on values (but not keys)
    - pset is converted to set, but not recursively
    - tuple is converted to tuple, recursively.

    If strict == True (the default):

    - thaw is called on elements of lists
    - thaw is called on values in dicts

    >>> from pyrsistent import s, m, v
    >>> thaw(s(1, 2))
    {1, 2}
    >>> thaw(v(1, m(a=3)))
    [1, {'a': 3}]
    >>> thaw((1, v()))
    (1, [])
    """
    typ = type(o)
    if isinstance(o, PVector) or (strict and typ is list):
        curried_thaw = lambda x: thaw(x, strict)
        return list(map(curried_thaw, o))
    if isinstance(o, PMap) or (strict and typ is dict):
        return {k: thaw(v, strict) for k, v in o.items()}
    if typ is tuple:
        curried_thaw = lambda x: thaw(x, strict)
        return tuple(map(curried_thaw, o))
    if isinstance(o, PSet):
        # impossible to thaw inside psets or sets
        return set(o)
    return o


def mutant(fn):
    """
    Convenience decorator to isolate mutation to within the decorated function (with respect
    to the input arguments).

    All arguments to the decorated function will be frozen so that they are guaranteed not to change.
    The return value is also frozen.
    """
    @wraps(fn)
    def inner_f(*args, **kwargs):
        return freeze(fn(*[freeze(e) for e in args], **dict(freeze(item) for item in kwargs.items())))

    return inner_f

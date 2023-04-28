import re
try:
    from inspect import Parameter, signature
except ImportError:
    signature = None
    from inspect import getfullargspec


_EMPTY_SENTINEL = object()


def inc(x):
    """ Add one to the current value """
    return x + 1


def dec(x):
    """ Subtract one from the current value """
    return x - 1


def discard(evolver, key):
    """ Discard the element and returns a structure without the discarded elements """
    try:
        del evolver[key]
    except KeyError:
        pass


# Matchers
def rex(expr):
    """ Regular expression matcher to use together with transform functions """
    r = re.compile(expr)
    return lambda key: isinstance(key, str) and r.match(key)


def ny(_):
    """ Matcher that matches any value """
    return True


# Support functions
def _chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def transform(structure, transformations):
    r = structure
    for path, command in _chunks(transformations, 2):
        r = _do_to_path(r, path, command)
    return r


def _do_to_path(structure, path, command):
    if not path:
        return command(structure) if callable(command) else command

    kvs = _get_keys_and_values(structure, path[0])
    return _update_structure(structure, kvs, path[1:], command)


def _items(structure):
    try:
        return structure.items()
    except AttributeError:
        # Support wider range of structures by adding a transform_items() or similar?
        return list(enumerate(structure))


def _get(structure, key, default):
    try:
        if hasattr(structure, '__getitem__'):
            return structure[key]

        return getattr(structure, key)

    except (IndexError, KeyError):
        return default


def _get_keys_and_values(structure, key_spec):
    if callable(key_spec):
        # Support predicates as callable objects in the path
        arity = _get_arity(key_spec)
        if arity == 1:
            # Unary predicates are called with the "key" of the path
            # - eg a key in a mapping, an index in a sequence.
            return [(k, v) for k, v in _items(structure) if key_spec(k)]
        elif arity == 2:
            # Binary predicates are called with the key and the corresponding
            # value.
            return [(k, v) for k, v in _items(structure) if key_spec(k, v)]
        else:
            # Other arities are an error.
            raise ValueError(
                "callable in transform path must take 1 or 2 arguments"
            )

    # Non-callables are used as-is as a key.
    return [(key_spec, _get(structure, key_spec, _EMPTY_SENTINEL))]


if signature is None:
    def _get_arity(f):
        argspec = getfullargspec(f)
        return len(argspec.args) - len(argspec.defaults or ())
else:
    def _get_arity(f):
        return sum(
            1
            for p
            in signature(f).parameters.values()
            if p.default is Parameter.empty
            and p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
        )


def _update_structure(structure, kvs, path, command):
    from pyrsistent._pmap import pmap
    e = structure.evolver()
    if not path and command is discard:
        # Do this in reverse to avoid index problems with vectors. See #92.
        for k, v in reversed(kvs):
            discard(e, k)
    else:
        for k, v in kvs:
            is_empty = False
            if v is _EMPTY_SENTINEL:
                # Allow expansion of structure but make sure to cover the case
                # when an empty pmap is added as leaf node. See #154.
                is_empty = True
                v = pmap()

            result = _do_to_path(v, path, command)
            if result is not v or is_empty:
                e[k] = result

    return e.persistent()

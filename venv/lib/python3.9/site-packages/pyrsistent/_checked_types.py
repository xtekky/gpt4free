from enum import Enum

from abc import abstractmethod, ABCMeta
from collections.abc import Iterable

from pyrsistent._pmap import PMap, pmap
from pyrsistent._pset import PSet, pset
from pyrsistent._pvector import PythonPVector, python_pvector


class CheckedType(object):
    """
    Marker class to enable creation and serialization of checked object graphs.
    """
    __slots__ = ()

    @classmethod
    @abstractmethod
    def create(cls, source_data, _factory_fields=None):
        raise NotImplementedError()

    @abstractmethod
    def serialize(self, format=None):
        raise NotImplementedError()


def _restore_pickle(cls, data):
    return cls.create(data, _factory_fields=set())


class InvariantException(Exception):
    """
    Exception raised from a :py:class:`CheckedType` when invariant tests fail or when a mandatory
    field is missing.

    Contains two fields of interest:
    invariant_errors, a tuple of error data for the failing invariants
    missing_fields, a tuple of strings specifying the missing names
    """

    def __init__(self, error_codes=(), missing_fields=(), *args, **kwargs):
        self.invariant_errors = tuple(e() if callable(e) else e for e in error_codes)
        self.missing_fields = missing_fields
        super(InvariantException, self).__init__(*args, **kwargs)

    def __str__(self):
        return super(InvariantException, self).__str__() + \
            ", invariant_errors=[{invariant_errors}], missing_fields=[{missing_fields}]".format(
            invariant_errors=', '.join(str(e) for e in self.invariant_errors),
            missing_fields=', '.join(self.missing_fields))


_preserved_iterable_types = (
    Enum,
)
"""Some types are themselves iterable, but we want to use the type itself and
not its members for the type specification. This defines a set of such types
that we explicitly preserve.

Note that strings are not such types because the string inputs we pass in are
values, not types.
"""


def maybe_parse_user_type(t):
    """Try to coerce a user-supplied type directive into a list of types.

    This function should be used in all places where a user specifies a type,
    for consistency.

    The policy for what defines valid user input should be clear from the implementation.
    """
    is_type = isinstance(t, type)
    is_preserved = isinstance(t, type) and issubclass(t, _preserved_iterable_types)
    is_string = isinstance(t, str)
    is_iterable = isinstance(t, Iterable)

    if is_preserved:
        return [t]
    elif is_string:
        return [t]
    elif is_type and not is_iterable:
        return [t]
    elif is_iterable:
        # Recur to validate contained types as well.
        ts = t
        return tuple(e for t in ts for e in maybe_parse_user_type(t))
    else:
        # If this raises because `t` cannot be formatted, so be it.
        raise TypeError(
            'Type specifications must be types or strings. Input: {}'.format(t)
        )


def maybe_parse_many_user_types(ts):
    # Just a different name to communicate that you're parsing multiple user
    # inputs. `maybe_parse_user_type` handles the iterable case anyway.
    return maybe_parse_user_type(ts)


def _store_types(dct, bases, destination_name, source_name):
    maybe_types = maybe_parse_many_user_types([
        d[source_name]
        for d in ([dct] + [b.__dict__ for b in bases]) if source_name in d
    ])

    dct[destination_name] = maybe_types


def _merge_invariant_results(result):
    verdict = True
    data = []
    for verd, dat in result:
        if not verd:
            verdict = False
            data.append(dat)

    return verdict, tuple(data)


def wrap_invariant(invariant):
    # Invariant functions may return the outcome of several tests
    # In those cases the results have to be merged before being passed
    # back to the client.
    def f(*args, **kwargs):
        result = invariant(*args, **kwargs)
        if isinstance(result[0], bool):
            return result

        return _merge_invariant_results(result)

    return f


def _all_dicts(bases, seen=None):
    """
    Yield each class in ``bases`` and each of their base classes.
    """
    if seen is None:
        seen = set()
    for cls in bases:
        if cls in seen:
            continue
        seen.add(cls)
        yield cls.__dict__
        for b in _all_dicts(cls.__bases__, seen):
            yield b


def store_invariants(dct, bases, destination_name, source_name):
    # Invariants are inherited
    invariants = []
    for ns in [dct] + list(_all_dicts(bases)):
        try:
            invariant = ns[source_name]
        except KeyError:
            continue
        invariants.append(invariant)

    if not all(callable(invariant) for invariant in invariants):
        raise TypeError('Invariants must be callable')
    dct[destination_name] = tuple(wrap_invariant(inv) for inv in invariants)


class _CheckedTypeMeta(ABCMeta):
    def __new__(mcs, name, bases, dct):
        _store_types(dct, bases, '_checked_types', '__type__')
        store_invariants(dct, bases, '_checked_invariants', '__invariant__')

        def default_serializer(self, _, value):
            if isinstance(value, CheckedType):
                return value.serialize()
            return value

        dct.setdefault('__serializer__', default_serializer)

        dct['__slots__'] = ()

        return super(_CheckedTypeMeta, mcs).__new__(mcs, name, bases, dct)


class CheckedTypeError(TypeError):
    def __init__(self, source_class, expected_types, actual_type, actual_value, *args, **kwargs):
        super(CheckedTypeError, self).__init__(*args, **kwargs)
        self.source_class = source_class
        self.expected_types = expected_types
        self.actual_type = actual_type
        self.actual_value = actual_value


class CheckedKeyTypeError(CheckedTypeError):
    """
    Raised when trying to set a value using a key with a type that doesn't match the declared type.

    Attributes:
    source_class -- The class of the collection
    expected_types  -- Allowed types
    actual_type -- The non matching type
    actual_value -- Value of the variable with the non matching type
    """
    pass


class CheckedValueTypeError(CheckedTypeError):
    """
    Raised when trying to set a value using a key with a type that doesn't match the declared type.

    Attributes:
    source_class -- The class of the collection
    expected_types  -- Allowed types
    actual_type -- The non matching type
    actual_value -- Value of the variable with the non matching type
    """
    pass


def _get_class(type_name):
    module_name, class_name = type_name.rsplit('.', 1)
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def get_type(typ):
    if isinstance(typ, type):
        return typ

    return _get_class(typ)


def get_types(typs):
    return [get_type(typ) for typ in typs]


def _check_types(it, expected_types, source_class, exception_type=CheckedValueTypeError):
    if expected_types:
        for e in it:
            if not any(isinstance(e, get_type(t)) for t in expected_types):
                actual_type = type(e)
                msg = "Type {source_class} can only be used with {expected_types}, not {actual_type}".format(
                    source_class=source_class.__name__,
                    expected_types=tuple(get_type(et).__name__ for et in expected_types),
                    actual_type=actual_type.__name__)
                raise exception_type(source_class, expected_types, actual_type, e, msg)


def _invariant_errors(elem, invariants):
    return [data for valid, data in (invariant(elem) for invariant in invariants) if not valid]


def _invariant_errors_iterable(it, invariants):
    return sum([_invariant_errors(elem, invariants) for elem in it], [])


def optional(*typs):
    """ Convenience function to specify that a value may be of any of the types in type 'typs' or None """
    return tuple(typs) + (type(None),)


def _checked_type_create(cls, source_data, _factory_fields=None, ignore_extra=False):
    if isinstance(source_data, cls):
        return source_data

    # Recursively apply create methods of checked types if the types of the supplied data
    # does not match any of the valid types.
    types = get_types(cls._checked_types)
    checked_type = next((t for t in types if issubclass(t, CheckedType)), None)
    if checked_type:
        return cls([checked_type.create(data, ignore_extra=ignore_extra)
                    if not any(isinstance(data, t) for t in types) else data
                    for data in source_data])

    return cls(source_data)

class CheckedPVector(PythonPVector, CheckedType, metaclass=_CheckedTypeMeta):
    """
    A CheckedPVector is a PVector which allows specifying type and invariant checks.

    >>> class Positives(CheckedPVector):
    ...     __type__ = (int, float)
    ...     __invariant__ = lambda n: (n >= 0, 'Negative')
    ...
    >>> Positives([1, 2, 3])
    Positives([1, 2, 3])
    """

    __slots__ = ()

    def __new__(cls, initial=()):
        if type(initial) == PythonPVector:
            return super(CheckedPVector, cls).__new__(cls, initial._count, initial._shift, initial._root, initial._tail)

        return CheckedPVector.Evolver(cls, python_pvector()).extend(initial).persistent()

    def set(self, key, value):
        return self.evolver().set(key, value).persistent()

    def append(self, val):
        return self.evolver().append(val).persistent()

    def extend(self, it):
        return self.evolver().extend(it).persistent()

    create = classmethod(_checked_type_create)

    def serialize(self, format=None):
        serializer = self.__serializer__
        return list(serializer(format, v) for v in self)

    def __reduce__(self):
        # Pickling support
        return _restore_pickle, (self.__class__, list(self),)

    class Evolver(PythonPVector.Evolver):
        __slots__ = ('_destination_class', '_invariant_errors')

        def __init__(self, destination_class, vector):
            super(CheckedPVector.Evolver, self).__init__(vector)
            self._destination_class = destination_class
            self._invariant_errors = []

        def _check(self, it):
            _check_types(it, self._destination_class._checked_types, self._destination_class)
            error_data = _invariant_errors_iterable(it, self._destination_class._checked_invariants)
            self._invariant_errors.extend(error_data)

        def __setitem__(self, key, value):
            self._check([value])
            return super(CheckedPVector.Evolver, self).__setitem__(key, value)

        def append(self, elem):
            self._check([elem])
            return super(CheckedPVector.Evolver, self).append(elem)

        def extend(self, it):
            it = list(it)
            self._check(it)
            return super(CheckedPVector.Evolver, self).extend(it)

        def persistent(self):
            if self._invariant_errors:
                raise InvariantException(error_codes=self._invariant_errors)

            result = self._orig_pvector
            if self.is_dirty() or (self._destination_class != type(self._orig_pvector)):
                pv = super(CheckedPVector.Evolver, self).persistent().extend(self._extra_tail)
                result = self._destination_class(pv)
                self._reset(result)

            return result

    def __repr__(self):
        return self.__class__.__name__ + "({0})".format(self.tolist())

    __str__ = __repr__

    def evolver(self):
        return CheckedPVector.Evolver(self.__class__, self)


class CheckedPSet(PSet, CheckedType, metaclass=_CheckedTypeMeta):
    """
    A CheckedPSet is a PSet which allows specifying type and invariant checks.

    >>> class Positives(CheckedPSet):
    ...     __type__ = (int, float)
    ...     __invariant__ = lambda n: (n >= 0, 'Negative')
    ...
    >>> Positives([1, 2, 3])
    Positives([1, 2, 3])
    """

    __slots__ = ()

    def __new__(cls, initial=()):
        if type(initial) is PMap:
            return super(CheckedPSet, cls).__new__(cls, initial)

        evolver = CheckedPSet.Evolver(cls, pset())
        for e in initial:
            evolver.add(e)

        return evolver.persistent()

    def __repr__(self):
        return self.__class__.__name__ + super(CheckedPSet, self).__repr__()[4:]

    def __str__(self):
        return self.__repr__()

    def serialize(self, format=None):
        serializer = self.__serializer__
        return set(serializer(format, v) for v in self)

    create = classmethod(_checked_type_create)

    def __reduce__(self):
        # Pickling support
        return _restore_pickle, (self.__class__, list(self),)

    def evolver(self):
        return CheckedPSet.Evolver(self.__class__, self)

    class Evolver(PSet._Evolver):
        __slots__ = ('_destination_class', '_invariant_errors')

        def __init__(self, destination_class, original_set):
            super(CheckedPSet.Evolver, self).__init__(original_set)
            self._destination_class = destination_class
            self._invariant_errors = []

        def _check(self, it):
            _check_types(it, self._destination_class._checked_types, self._destination_class)
            error_data = _invariant_errors_iterable(it, self._destination_class._checked_invariants)
            self._invariant_errors.extend(error_data)

        def add(self, element):
            self._check([element])
            self._pmap_evolver[element] = True
            return self

        def persistent(self):
            if self._invariant_errors:
                raise InvariantException(error_codes=self._invariant_errors)

            if self.is_dirty() or self._destination_class != type(self._original_pset):
                return self._destination_class(self._pmap_evolver.persistent())

            return self._original_pset


class _CheckedMapTypeMeta(type):
    def __new__(mcs, name, bases, dct):
        _store_types(dct, bases, '_checked_key_types', '__key_type__')
        _store_types(dct, bases, '_checked_value_types', '__value_type__')
        store_invariants(dct, bases, '_checked_invariants', '__invariant__')

        def default_serializer(self, _, key, value):
            sk = key
            if isinstance(key, CheckedType):
                sk = key.serialize()

            sv = value
            if isinstance(value, CheckedType):
                sv = value.serialize()

            return sk, sv

        dct.setdefault('__serializer__', default_serializer)

        dct['__slots__'] = ()

        return super(_CheckedMapTypeMeta, mcs).__new__(mcs, name, bases, dct)

# Marker object
_UNDEFINED_CHECKED_PMAP_SIZE = object()


class CheckedPMap(PMap, CheckedType, metaclass=_CheckedMapTypeMeta):
    """
    A CheckedPMap is a PMap which allows specifying type and invariant checks.

    >>> class IntToFloatMap(CheckedPMap):
    ...     __key_type__ = int
    ...     __value_type__ = float
    ...     __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    ...
    >>> IntToFloatMap({1: 1.5, 2: 2.25})
    IntToFloatMap({1: 1.5, 2: 2.25})
    """

    __slots__ = ()

    def __new__(cls, initial={}, size=_UNDEFINED_CHECKED_PMAP_SIZE):
        if size is not _UNDEFINED_CHECKED_PMAP_SIZE:
            return super(CheckedPMap, cls).__new__(cls, size, initial)

        evolver = CheckedPMap.Evolver(cls, pmap())
        for k, v in initial.items():
            evolver.set(k, v)

        return evolver.persistent()

    def evolver(self):
        return CheckedPMap.Evolver(self.__class__, self)

    def __repr__(self):
        return self.__class__.__name__ + "({0})".format(str(dict(self)))

    __str__ = __repr__

    def serialize(self, format=None):
        serializer = self.__serializer__
        return dict(serializer(format, k, v) for k, v in self.items())

    @classmethod
    def create(cls, source_data, _factory_fields=None):
        if isinstance(source_data, cls):
            return source_data

        # Recursively apply create methods of checked types if the types of the supplied data
        # does not match any of the valid types.
        key_types = get_types(cls._checked_key_types)
        checked_key_type = next((t for t in key_types if issubclass(t, CheckedType)), None)
        value_types = get_types(cls._checked_value_types)
        checked_value_type = next((t for t in value_types if issubclass(t, CheckedType)), None)

        if checked_key_type or checked_value_type:
            return cls(dict((checked_key_type.create(key) if checked_key_type and not any(isinstance(key, t) for t in key_types) else key,
                             checked_value_type.create(value) if checked_value_type and not any(isinstance(value, t) for t in value_types) else value)
                            for key, value in source_data.items()))

        return cls(source_data)

    def __reduce__(self):
        # Pickling support
        return _restore_pickle, (self.__class__, dict(self),)

    class Evolver(PMap._Evolver):
        __slots__ = ('_destination_class', '_invariant_errors')

        def __init__(self, destination_class, original_map):
            super(CheckedPMap.Evolver, self).__init__(original_map)
            self._destination_class = destination_class
            self._invariant_errors = []

        def set(self, key, value):
            _check_types([key], self._destination_class._checked_key_types, self._destination_class, CheckedKeyTypeError)
            _check_types([value], self._destination_class._checked_value_types, self._destination_class)
            self._invariant_errors.extend(data for valid, data in (invariant(key, value)
                                                                   for invariant in self._destination_class._checked_invariants)
                                          if not valid)

            return super(CheckedPMap.Evolver, self).set(key, value)

        def persistent(self):
            if self._invariant_errors:
                raise InvariantException(error_codes=self._invariant_errors)

            if self.is_dirty() or type(self._original_pmap) != self._destination_class:
                return self._destination_class(self._buckets_evolver.persistent(), self._size)

            return self._original_pmap

from collections.abc import Set, Hashable
import sys
from pyrsistent._pmap import pmap


class PSet(object):
    """
    Persistent set implementation. Built on top of the persistent map. The set supports all operations
    in the Set protocol and is Hashable.

    Do not instantiate directly, instead use the factory functions :py:func:`s` or :py:func:`pset`
    to create an instance.

    Random access and insert is log32(n) where n is the size of the set.

    Some examples:

    >>> s = pset([1, 2, 3, 1])
    >>> s2 = s.add(4)
    >>> s3 = s2.remove(2)
    >>> s
    pset([1, 2, 3])
    >>> s2
    pset([1, 2, 3, 4])
    >>> s3
    pset([1, 3, 4])
    """
    __slots__ = ('_map', '__weakref__')

    def __new__(cls, m):
        self = super(PSet, cls).__new__(cls)
        self._map = m
        return self

    def __contains__(self, element):
        return element in self._map

    def __iter__(self):
        return iter(self._map)

    def __len__(self):
        return len(self._map)

    def __repr__(self):
        if not self:
            return 'p' + str(set(self))

        return 'pset([{0}])'.format(str(set(self))[1:-1])

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(self._map)

    def __reduce__(self):
        # Pickling support
        return pset, (list(self),)

    @classmethod
    def _from_iterable(cls, it, pre_size=8):
        return PSet(pmap(dict((k, True) for k in it), pre_size=pre_size))

    def add(self, element):
        """
        Return a new PSet with element added

        >>> s1 = s(1, 2)
        >>> s1.add(3)
        pset([1, 2, 3])
        """
        return self.evolver().add(element).persistent()

    def update(self, iterable):
        """
        Return a new PSet with elements in iterable added

        >>> s1 = s(1, 2)
        >>> s1.update([3, 4, 4])
        pset([1, 2, 3, 4])
        """
        e = self.evolver()
        for element in iterable:
            e.add(element)

        return e.persistent()

    def remove(self, element):
        """
        Return a new PSet with element removed. Raises KeyError if element is not present.

        >>> s1 = s(1, 2)
        >>> s1.remove(2)
        pset([1])
        """
        if element in self._map:
            return self.evolver().remove(element).persistent()

        raise KeyError("Element '%s' not present in PSet" % repr(element))

    def discard(self, element):
        """
        Return a new PSet with element removed. Returns itself if element is not present.
        """
        if element in self._map:
            return self.evolver().remove(element).persistent()

        return self

    class _Evolver(object):
        __slots__ = ('_original_pset', '_pmap_evolver')

        def __init__(self, original_pset):
            self._original_pset = original_pset
            self._pmap_evolver = original_pset._map.evolver()

        def add(self, element):
            self._pmap_evolver[element] = True
            return self

        def remove(self, element):
            del self._pmap_evolver[element]
            return self

        def is_dirty(self):
            return self._pmap_evolver.is_dirty()

        def persistent(self):
            if not self.is_dirty():
                return  self._original_pset

            return PSet(self._pmap_evolver.persistent())

        def __len__(self):
            return len(self._pmap_evolver)

    def copy(self):
        return self

    def evolver(self):
        """
        Create a new evolver for this pset. For a discussion on evolvers in general see the
        documentation for the pvector evolver.

        Create the evolver and perform various mutating updates to it:

        >>> s1 = s(1, 2, 3)
        >>> e = s1.evolver()
        >>> _ = e.add(4)
        >>> len(e)
        4
        >>> _ = e.remove(1)

        The underlying pset remains the same:

        >>> s1
        pset([1, 2, 3])

        The changes are kept in the evolver. An updated pmap can be created using the
        persistent() function on the evolver.

        >>> s2 = e.persistent()
        >>> s2
        pset([2, 3, 4])

        The new pset will share data with the original pset in the same way that would have
        been done if only using operations on the pset.
        """
        return PSet._Evolver(self)

    # All the operations and comparisons you would expect on a set.
    #
    # This is not very beautiful. If we avoid inheriting from PSet we can use the
    # __slots__ concepts (which requires a new style class) and hopefully save some memory.
    __le__ = Set.__le__
    __lt__ = Set.__lt__
    __gt__ = Set.__gt__
    __ge__ = Set.__ge__
    __eq__ = Set.__eq__
    __ne__ = Set.__ne__

    __and__ = Set.__and__
    __or__ = Set.__or__
    __sub__ = Set.__sub__
    __xor__ = Set.__xor__

    issubset = __le__
    issuperset = __ge__
    union = __or__
    intersection = __and__
    difference = __sub__
    symmetric_difference = __xor__

    isdisjoint = Set.isdisjoint

Set.register(PSet)
Hashable.register(PSet)

_EMPTY_PSET = PSet(pmap())


def pset(iterable=(), pre_size=8):
    """
    Creates a persistent set from iterable. Optionally takes a sizing parameter equivalent to that
    used for :py:func:`pmap`.

    >>> s1 = pset([1, 2, 3, 2])
    >>> s1
    pset([1, 2, 3])
    """
    if not iterable:
        return _EMPTY_PSET

    return PSet._from_iterable(iterable, pre_size=pre_size)


def s(*elements):
    """
    Create a persistent set.

    Takes an arbitrary number of arguments to insert into the new set.

    >>> s1 = s(1, 2, 3, 2)
    >>> s1
    pset([1, 2, 3])
    """
    return pset(elements)

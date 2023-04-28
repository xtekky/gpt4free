from collections.abc import Sequence, Hashable
from numbers import Integral
from functools import reduce


class _PListBuilder(object):
    """
    Helper class to allow construction of a list without
    having to reverse it in the end.
    """
    __slots__ = ('_head', '_tail')

    def __init__(self):
        self._head = _EMPTY_PLIST
        self._tail = _EMPTY_PLIST

    def _append(self, elem, constructor):
        if not self._tail:
            self._head = constructor(elem)
            self._tail = self._head
        else:
            self._tail.rest = constructor(elem)
            self._tail = self._tail.rest

        return self._head

    def append_elem(self, elem):
        return self._append(elem, lambda e: PList(e, _EMPTY_PLIST))

    def append_plist(self, pl):
        return self._append(pl, lambda l: l)

    def build(self):
        return self._head


class _PListBase(object):
    __slots__ = ('__weakref__',)

    # Selected implementations can be taken straight from the Sequence
    # class, other are less suitable. Especially those that work with
    # index lookups.
    count = Sequence.count
    index = Sequence.index

    def __reduce__(self):
        # Pickling support
        return plist, (list(self),)

    def __len__(self):
        """
        Return the length of the list, computed by traversing it.

        This is obviously O(n) but with the current implementation
        where a list is also a node the overhead of storing the length
        in every node would be quite significant.
        """
        return sum(1 for _ in self)

    def __repr__(self):
        return "plist({0})".format(list(self))
    __str__ = __repr__

    def cons(self, elem):
        """
        Return a new list with elem inserted as new head.

        >>> plist([1, 2]).cons(3)
        plist([3, 1, 2])
        """
        return PList(elem, self)

    def mcons(self, iterable):
        """
        Return a new list with all elements of iterable repeatedly cons:ed to the current list.
        NB! The elements will be inserted in the reverse order of the iterable.
        Runs in O(len(iterable)).

        >>> plist([1, 2]).mcons([3, 4])
        plist([4, 3, 1, 2])
        """
        head = self
        for elem in iterable:
            head = head.cons(elem)

        return head

    def reverse(self):
        """
        Return a reversed version of list. Runs in O(n) where n is the length of the list.

        >>> plist([1, 2, 3]).reverse()
        plist([3, 2, 1])

        Also supports the standard reversed function.

        >>> reversed(plist([1, 2, 3]))
        plist([3, 2, 1])
        """
        result = plist()
        head = self
        while head:
            result = result.cons(head.first)
            head = head.rest

        return result
    __reversed__ = reverse

    def split(self, index):
        """
        Spilt the list at position specified by index. Returns a tuple containing the
        list up until index and the list after the index. Runs in O(index).

        >>> plist([1, 2, 3, 4]).split(2)
        (plist([1, 2]), plist([3, 4]))
        """
        lb = _PListBuilder()
        right_list = self
        i = 0
        while right_list and i < index:
            lb.append_elem(right_list.first)
            right_list = right_list.rest
            i += 1

        if not right_list:
            # Just a small optimization in the cases where no split occurred
            return self, _EMPTY_PLIST

        return lb.build(), right_list

    def __iter__(self):
        li = self
        while li:
            yield li.first
            li = li.rest

    def __lt__(self, other):
        if not isinstance(other, _PListBase):
            return NotImplemented

        return tuple(self) < tuple(other)

    def __eq__(self, other):
        """
        Traverses the lists, checking equality of elements.

        This is an O(n) operation, but preserves the standard semantics of list equality.
        """
        if not isinstance(other, _PListBase):
            return NotImplemented

        self_head = self
        other_head = other
        while self_head and other_head:
            if not self_head.first == other_head.first:
                return False
            self_head = self_head.rest
            other_head = other_head.rest

        return not self_head and not other_head

    def __getitem__(self, index):
        # Don't use this this data structure if you plan to do a lot of indexing, it is
        # very inefficient! Use a PVector instead!

        if isinstance(index, slice):
            if index.start is not None and index.stop is None and (index.step is None or index.step == 1):
                return self._drop(index.start)

            # Take the easy way out for all other slicing cases, not much structural reuse possible anyway
            return plist(tuple(self)[index])

        if not isinstance(index, Integral):
            raise TypeError("'%s' object cannot be interpreted as an index" % type(index).__name__)

        if index < 0:
            # NB: O(n)!
            index += len(self)

        try:
            return self._drop(index).first
        except AttributeError as e:
            raise IndexError("PList index out of range") from e

    def _drop(self, count):
        if count < 0:
            raise IndexError("PList index out of range")

        head = self
        while count > 0:
            head = head.rest
            count -= 1

        return head

    def __hash__(self):
        return hash(tuple(self))

    def remove(self, elem):
        """
        Return new list with first element equal to elem removed. O(k) where k is the position
        of the element that is removed.

        Raises ValueError if no matching element is found.

        >>> plist([1, 2, 1]).remove(1)
        plist([2, 1])
        """

        builder = _PListBuilder()
        head = self
        while head:
            if head.first == elem:
                return builder.append_plist(head.rest)

            builder.append_elem(head.first)
            head = head.rest

        raise ValueError('{0} not found in PList'.format(elem))


class PList(_PListBase):
    """
    Classical Lisp style singly linked list. Adding elements to the head using cons is O(1).
    Element access is O(k) where k is the position of the element in the list. Taking the
    length of the list is O(n).

    Fully supports the Sequence and Hashable protocols including indexing and slicing but
    if you need fast random access go for the PVector instead.

    Do not instantiate directly, instead use the factory functions :py:func:`l` or :py:func:`plist` to
    create an instance.

    Some examples:

    >>> x = plist([1, 2])
    >>> y = x.cons(3)
    >>> x
    plist([1, 2])
    >>> y
    plist([3, 1, 2])
    >>> y.first
    3
    >>> y.rest == x
    True
    >>> y[:2]
    plist([3, 1])
    """
    __slots__ = ('first', 'rest')

    def __new__(cls, first, rest):
        instance = super(PList, cls).__new__(cls)
        instance.first = first
        instance.rest = rest
        return instance

    def __bool__(self):
        return True
    __nonzero__ = __bool__


Sequence.register(PList)
Hashable.register(PList)


class _EmptyPList(_PListBase):
    __slots__ = ()

    def __bool__(self):
        return False
    __nonzero__ = __bool__

    @property
    def first(self):
        raise AttributeError("Empty PList has no first")

    @property
    def rest(self):
        return self


Sequence.register(_EmptyPList)
Hashable.register(_EmptyPList)

_EMPTY_PLIST = _EmptyPList()


def plist(iterable=(), reverse=False):
    """
    Creates a new persistent list containing all elements of iterable.
    Optional parameter reverse specifies if the elements should be inserted in
    reverse order or not.

    >>> plist([1, 2, 3])
    plist([1, 2, 3])
    >>> plist([1, 2, 3], reverse=True)
    plist([3, 2, 1])
    """
    if not reverse:
        iterable = list(iterable)
        iterable.reverse()

    return reduce(lambda pl, elem: pl.cons(elem), iterable, _EMPTY_PLIST)


def l(*elements):
    """
    Creates a new persistent list containing all arguments.

    >>> l(1, 2, 3)
    plist([1, 2, 3])
    """
    return plist(elements)

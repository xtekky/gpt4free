from collections.abc import Sequence, Hashable
from itertools import islice, chain
from numbers import Integral
from pyrsistent._plist import plist


class PDeque(object):
    """
    Persistent double ended queue (deque). Allows quick appends and pops in both ends. Implemented
    using two persistent lists.

    A maximum length can be specified to create a bounded queue.

    Fully supports the Sequence and Hashable protocols including indexing and slicing but
    if you need fast random access go for the PVector instead.

    Do not instantiate directly, instead use the factory functions :py:func:`dq` or :py:func:`pdeque` to
    create an instance.

    Some examples:

    >>> x = pdeque([1, 2, 3])
    >>> x.left
    1
    >>> x.right
    3
    >>> x[0] == x.left
    True
    >>> x[-1] == x.right
    True
    >>> x.pop()
    pdeque([1, 2])
    >>> x.pop() == x[:-1]
    True
    >>> x.popleft()
    pdeque([2, 3])
    >>> x.append(4)
    pdeque([1, 2, 3, 4])
    >>> x.appendleft(4)
    pdeque([4, 1, 2, 3])

    >>> y = pdeque([1, 2, 3], maxlen=3)
    >>> y.append(4)
    pdeque([2, 3, 4], maxlen=3)
    >>> y.appendleft(4)
    pdeque([4, 1, 2], maxlen=3)
    """
    __slots__ = ('_left_list', '_right_list', '_length', '_maxlen', '__weakref__')

    def __new__(cls, left_list, right_list, length, maxlen=None):
        instance = super(PDeque, cls).__new__(cls)
        instance._left_list = left_list
        instance._right_list = right_list
        instance._length = length

        if maxlen is not None:
            if not isinstance(maxlen, Integral):
                raise TypeError('An integer is required as maxlen')

            if maxlen < 0:
                raise ValueError("maxlen must be non-negative")

        instance._maxlen = maxlen
        return instance

    @property
    def right(self):
        """
        Rightmost element in dqueue.
        """
        return PDeque._tip_from_lists(self._right_list, self._left_list)

    @property
    def left(self):
        """
        Leftmost element in dqueue.
        """
        return PDeque._tip_from_lists(self._left_list, self._right_list)

    @staticmethod
    def _tip_from_lists(primary_list, secondary_list):
        if primary_list:
            return primary_list.first

        if secondary_list:
            return secondary_list[-1]

        raise IndexError('No elements in empty deque')

    def __iter__(self):
        return chain(self._left_list, self._right_list.reverse())

    def __repr__(self):
        return "pdeque({0}{1})".format(list(self),
                                       ', maxlen={0}'.format(self._maxlen) if self._maxlen is not None else '')
    __str__ = __repr__

    @property
    def maxlen(self):
        """
        Maximum length of the queue.
        """
        return self._maxlen

    def pop(self, count=1):
        """
        Return new deque with rightmost element removed. Popping the empty queue
        will return the empty queue. A optional count can be given to indicate the
        number of elements to pop. Popping with a negative index is the same as
        popleft. Executes in amortized O(k) where k is the number of elements to pop.

        >>> pdeque([1, 2]).pop()
        pdeque([1])
        >>> pdeque([1, 2]).pop(2)
        pdeque([])
        >>> pdeque([1, 2]).pop(-1)
        pdeque([2])
        """
        if count < 0:
            return self.popleft(-count)

        new_right_list, new_left_list = PDeque._pop_lists(self._right_list, self._left_list, count)
        return PDeque(new_left_list, new_right_list, max(self._length - count, 0), self._maxlen)

    def popleft(self, count=1):
        """
        Return new deque with leftmost element removed. Otherwise functionally
        equivalent to pop().

        >>> pdeque([1, 2]).popleft()
        pdeque([2])
        """
        if count < 0:
            return self.pop(-count)

        new_left_list, new_right_list = PDeque._pop_lists(self._left_list, self._right_list, count)
        return PDeque(new_left_list, new_right_list, max(self._length - count, 0), self._maxlen)

    @staticmethod
    def _pop_lists(primary_list, secondary_list, count):
        new_primary_list = primary_list
        new_secondary_list = secondary_list

        while count > 0 and (new_primary_list or new_secondary_list):
            count -= 1
            if new_primary_list.rest:
                new_primary_list = new_primary_list.rest
            elif new_primary_list:
                new_primary_list = new_secondary_list.reverse()
                new_secondary_list = plist()
            else:
                new_primary_list = new_secondary_list.reverse().rest
                new_secondary_list = plist()

        return new_primary_list, new_secondary_list

    def _is_empty(self):
        return not self._left_list and not self._right_list

    def __lt__(self, other):
        if not isinstance(other, PDeque):
            return NotImplemented

        return tuple(self) < tuple(other)

    def __eq__(self, other):
        if not isinstance(other, PDeque):
            return NotImplemented

        if tuple(self) == tuple(other):
            # Sanity check of the length value since it is redundant (there for performance)
            assert len(self) == len(other)
            return True

        return False

    def __hash__(self):
        return  hash(tuple(self))

    def __len__(self):
        return self._length

    def append(self, elem):
        """
        Return new deque with elem as the rightmost element.

        >>> pdeque([1, 2]).append(3)
        pdeque([1, 2, 3])
        """
        new_left_list, new_right_list, new_length = self._append(self._left_list, self._right_list, elem)
        return PDeque(new_left_list, new_right_list, new_length, self._maxlen)

    def appendleft(self, elem):
        """
        Return new deque with elem as the leftmost element.

        >>> pdeque([1, 2]).appendleft(3)
        pdeque([3, 1, 2])
        """
        new_right_list, new_left_list, new_length = self._append(self._right_list, self._left_list, elem)
        return PDeque(new_left_list, new_right_list, new_length, self._maxlen)

    def _append(self, primary_list, secondary_list, elem):
        if self._maxlen is not None and self._length == self._maxlen:
            if self._maxlen == 0:
                return primary_list, secondary_list, 0
            new_primary_list, new_secondary_list = PDeque._pop_lists(primary_list, secondary_list, 1)
            return new_primary_list, new_secondary_list.cons(elem), self._length

        return primary_list, secondary_list.cons(elem), self._length + 1

    @staticmethod
    def _extend_list(the_list, iterable):
        count = 0
        for elem in iterable:
            the_list = the_list.cons(elem)
            count += 1

        return the_list, count

    def _extend(self, primary_list, secondary_list, iterable):
        new_primary_list, extend_count = PDeque._extend_list(primary_list, iterable)
        new_secondary_list = secondary_list
        current_len = self._length + extend_count
        if self._maxlen is not None and current_len > self._maxlen:
            pop_len = current_len - self._maxlen
            new_secondary_list, new_primary_list = PDeque._pop_lists(new_secondary_list, new_primary_list, pop_len)
            extend_count -= pop_len

        return new_primary_list, new_secondary_list, extend_count

    def extend(self, iterable):
        """
        Return new deque with all elements of iterable appended to the right.

        >>> pdeque([1, 2]).extend([3, 4])
        pdeque([1, 2, 3, 4])
        """
        new_right_list, new_left_list, extend_count = self._extend(self._right_list, self._left_list, iterable)
        return PDeque(new_left_list, new_right_list, self._length + extend_count, self._maxlen)

    def extendleft(self, iterable):
        """
        Return new deque with all elements of iterable appended to the left.

        NB! The elements will be inserted in reverse order compared to the order in the iterable.

        >>> pdeque([1, 2]).extendleft([3, 4])
        pdeque([4, 3, 1, 2])
        """
        new_left_list, new_right_list, extend_count = self._extend(self._left_list, self._right_list, iterable)
        return PDeque(new_left_list, new_right_list, self._length + extend_count, self._maxlen)

    def count(self, elem):
        """
        Return the number of elements equal to elem present in the queue

        >>> pdeque([1, 2, 1]).count(1)
        2
        """
        return self._left_list.count(elem) + self._right_list.count(elem)

    def remove(self, elem):
        """
        Return new deque with first element from left equal to elem removed. If no such element is found
        a ValueError is raised.

        >>> pdeque([2, 1, 2]).remove(2)
        pdeque([1, 2])
        """
        try:
            return PDeque(self._left_list.remove(elem), self._right_list, self._length - 1)
        except ValueError:
            # Value not found in left list, try the right list
            try:
                # This is severely inefficient with a double reverse, should perhaps implement a remove_last()?
                return PDeque(self._left_list,
                               self._right_list.reverse().remove(elem).reverse(), self._length - 1)
            except ValueError as e:
                raise ValueError('{0} not found in PDeque'.format(elem)) from e

    def reverse(self):
        """
        Return reversed deque.

        >>> pdeque([1, 2, 3]).reverse()
        pdeque([3, 2, 1])

        Also supports the standard python reverse function.

        >>> reversed(pdeque([1, 2, 3]))
        pdeque([3, 2, 1])
        """
        return PDeque(self._right_list, self._left_list, self._length)
    __reversed__ = reverse

    def rotate(self, steps):
        """
        Return deque with elements rotated steps steps.

        >>> x = pdeque([1, 2, 3])
        >>> x.rotate(1)
        pdeque([3, 1, 2])
        >>> x.rotate(-2)
        pdeque([3, 1, 2])
        """
        popped_deque = self.pop(steps)
        if steps >= 0:
            return popped_deque.extendleft(islice(self.reverse(), steps))

        return popped_deque.extend(islice(self, -steps))

    def __reduce__(self):
        # Pickling support
        return pdeque, (list(self), self._maxlen)

    def __getitem__(self, index):
        if isinstance(index, slice):
            if index.step is not None and index.step != 1:
                # Too difficult, no structural sharing possible
                return pdeque(tuple(self)[index], maxlen=self._maxlen)

            result = self
            if index.start is not None:
                result = result.popleft(index.start % self._length)
            if index.stop is not None:
                result = result.pop(self._length - (index.stop % self._length))

            return result

        if not isinstance(index, Integral):
            raise TypeError("'%s' object cannot be interpreted as an index" % type(index).__name__)

        if index >= 0:
            return self.popleft(index).left

        shifted = len(self) + index
        if shifted < 0:
            raise IndexError(
                "pdeque index {0} out of range {1}".format(index, len(self)),
            )
        return self.popleft(shifted).left

    index = Sequence.index

Sequence.register(PDeque)
Hashable.register(PDeque)


def pdeque(iterable=(), maxlen=None):
    """
    Return deque containing the elements of iterable. If maxlen is specified then
    len(iterable) - maxlen elements are discarded from the left to if len(iterable) > maxlen.

    >>> pdeque([1, 2, 3])
    pdeque([1, 2, 3])
    >>> pdeque([1, 2, 3, 4], maxlen=2)
    pdeque([3, 4], maxlen=2)
    """
    t = tuple(iterable)
    if maxlen is not None:
        t = t[-maxlen:]
    length = len(t)
    pivot = int(length / 2)
    left = plist(t[:pivot])
    right = plist(t[pivot:], reverse=True)
    return PDeque(left, right, length, maxlen)

def dq(*elements):
    """
    Return deque containing all arguments.

    >>> dq(1, 2, 3)
    pdeque([1, 2, 3])
    """
    return pdeque(elements)

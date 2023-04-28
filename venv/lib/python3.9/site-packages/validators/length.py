from .between import between
from .utils import validator


@validator
def length(value, min=None, max=None):
    """
    Return whether or not the length of given string is within a specified
    range.

    Examples::

        >>> length('something', min=2)
        True

        >>> length('something', min=9, max=9)
        True

        >>> length('something', max=5)
        ValidationFailure(func=length, ...)

    :param value:
        The string to validate.
    :param min:
        The minimum required length of the string. If not provided, minimum
        length will not be checked.
    :param max:
        The maximum length of the string. If not provided, maximum length
        will not be checked.

    .. versionadded:: 0.2
    """
    if (min is not None and min < 0) or (max is not None and max < 0):
        raise AssertionError(
            '`min` and `max` need to be greater than zero.'
        )
    return between(len(value), min=min, max=max)

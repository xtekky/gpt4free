"""
String utility functions.
"""

from typing import Any, Optional, Union


def safe_repr(obj: Any, clip: Optional[int] = None) -> str:
    """
    Convert object to string representation, yielding the same result a `repr`
    but catches all exceptions and returns 'N/A' instead of raising the
    exception. Strings may be truncated by providing `clip`.

    >>> safe_repr(42)
    '42'
    >>> safe_repr('Clipped text', clip=8)
    'Clip..xt'
    >>> safe_repr([1,2,3,4], clip=8)
    '[1,2..4]'
    """
    try:
        s = repr(obj)
        if not clip or len(s) <= clip:
            return s
        else:
            return s[:clip - 4] + '..' + s[-2:]
    except:
        return 'N/A'


def trunc(obj: str, max: int, left: bool = False) -> str:
    """
    Convert `obj` to string, eliminate newlines and truncate the string to
    `max` characters. If there are more characters in the string add ``...`` to
    the string. With `left=True`, the string can be truncated at the beginning.

    @note: Does not catch exceptions when converting `obj` to string with
        `str`.

    >>> trunc('This is a long text.', 8)
    This ...
    >>> trunc('This is a long text.', 8, left=True)
    ...text.
    """
    s = str(obj)
    s = s.replace('\n', '|')
    if len(s) > max:
        if left:
            return '...' + s[len(s) - max + 3:]
        else:
            return s[:(max - 3)] + '...'
    else:
        return s


def pp(i: Union[int, float], base: int = 1024) -> str:
    """
    Pretty-print the integer `i` as a human-readable size representation.
    """
    degree = 0
    pattern = "%4d     %s"
    while i > base:
        pattern = "%7.2f %s"
        i = i / float(base)
        degree += 1
    scales = ['B', 'KB', 'MB', 'GB', 'TB', 'EB']
    return pattern % (i, scales[degree])


def pp_timestamp(t: Optional[float]) -> str:
    """
    Get a friendly timestamp represented as a string.
    """
    if t is None:
        return ''
    h, m, s = int(t / 3600), int(t / 60 % 60), t % 60
    return "%02d:%02d:%05.2f" % (h, m, s)

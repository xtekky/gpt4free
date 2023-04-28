import re

from .utils import validator

pattern = re.compile(
    r'^(?:[a-zA-Z0-9]'  # First character of the domain
    r'(?:[a-zA-Z0-9-_]{0,61}[A-Za-z0-9])?\.)'  # Sub domain + hostname
    r'+[A-Za-z0-9][A-Za-z0-9-_]{0,61}'  # First 61 characters of the gTLD
    r'[A-Za-z]$'  # Last character of the gTLD
)


def to_unicode(obj, charset='utf-8', errors='strict'):
    if obj is None:
        return None
    if not isinstance(obj, bytes):
        return str(obj)
    return obj.decode(charset, errors)


@validator
def domain(value):
    """
    Return whether or not given value is a valid domain.

    If the value is valid domain name this function returns ``True``, otherwise
    :class:`~validators.utils.ValidationFailure`.

    Examples::

        >>> domain('example.com')
        True

        >>> domain('example.com/')
        ValidationFailure(func=domain, ...)


    Supports IDN domains as well::

        >>> domain('xn----gtbspbbmkef.xn--p1ai')
        True

    .. versionadded:: 0.9

    .. versionchanged:: 0.10

        Added support for internationalized domain name (IDN) validation.

    :param value: domain string to validate
    """
    try:
        return pattern.match(to_unicode(value).encode('idna').decode('ascii'))
    except (UnicodeError, AttributeError):
        return False

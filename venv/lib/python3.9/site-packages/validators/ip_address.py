from .utils import validator


@validator
def ipv4(value):
    """
    Return whether a given value is a valid IP version 4 address.

    This validator is based on `WTForms IPAddress validator`_

    .. _WTForms IPAddress validator:
       https://github.com/wtforms/wtforms/blob/master/wtforms/validators.py

    Examples::

        >>> ipv4('123.0.0.7')
        True

        >>> ipv4('900.80.70.11')
        ValidationFailure(func=ipv4, args={'value': '900.80.70.11'})

    .. versionadded:: 0.2

    :param value: IP address string to validate
    """
    groups = value.split(".")
    if (
        len(groups) != 4
        or any(not x.isdigit() for x in groups)
        or any(len(x) > 3 for x in groups)
    ):
        return False
    return all(0 <= int(part) < 256 for part in groups)


@validator
def ipv4_cidr(value):
    """
    Return whether a given value is a valid CIDR-notated IP version 4
    address range.

    This validator is based on RFC4632 3.1.

    Examples::

        >>> ipv4_cidr('1.1.1.1/8')
        True

        >>> ipv4_cidr('1.1.1.1')
        ValidationFailure(func=ipv4_cidr, args={'value': '1.1.1.1'})
    """
    try:
        prefix, suffix = value.split('/', 2)
    except ValueError:
        return False
    if not ipv4(prefix) or not suffix.isdigit():
        return False
    return 0 <= int(suffix) <= 32


@validator
def ipv6(value):
    """
    Return whether a given value is a valid IP version 6 address
    (including IPv4-mapped IPv6 addresses).

    This validator is based on `WTForms IPAddress validator`_.

    .. _WTForms IPAddress validator:
       https://github.com/wtforms/wtforms/blob/master/wtforms/validators.py

    Examples::

        >>> ipv6('abcd:ef::42:1')
        True

        >>> ipv6('::ffff:192.0.2.128')
        True

        >>> ipv6('::192.0.2.128')
        True

        >>> ipv6('abc.0.0.1')
        ValidationFailure(func=ipv6, args={'value': 'abc.0.0.1'})

    .. versionadded:: 0.2

    :param value: IP address string to validate
    """
    ipv6_groups = value.split(':')
    if len(ipv6_groups) == 1:
        return False
    ipv4_groups = ipv6_groups[-1].split('.')

    if len(ipv4_groups) > 1:
        if not ipv4(ipv6_groups[-1]):
            return False
        ipv6_groups = ipv6_groups[:-1]
    else:
        ipv4_groups = []

    count_blank = 0
    for part in ipv6_groups:
        if not part:
            count_blank += 1
            continue
        try:
            num = int(part, 16)
        except ValueError:
            return False
        else:
            if not 0 <= num <= 65536 or len(part) > 4:
                return False

    max_groups = 6 if ipv4_groups else 8
    part_count = len(ipv6_groups) - count_blank
    if count_blank == 0 and part_count == max_groups:
        # no :: -> must have size of max_groups
        return True
    elif count_blank == 1 and ipv6_groups[-1] and ipv6_groups[0] and part_count < max_groups:
        # one :: inside the address or prefix or suffix : -> filter least two cases
        return True
    elif count_blank == 2 and part_count < max_groups and (
            ((ipv6_groups[0] and not ipv6_groups[-1]) or (not ipv6_groups[0] and ipv6_groups[-1])) or ipv4_groups):
        # leading or trailing :: or : at end and begin -> filter last case
        # Check if it has ipv4 groups because they get removed from the ipv6_groups
        return True
    elif count_blank == 3 and part_count == 0:
        # :: is the address -> filter everything else
        return True
    return False


@validator
def ipv6_cidr(value):
    """
    Returns whether a given value is a valid CIDR-notated IP version 6
    address range.

    This validator is based on RFC4632 3.1.

    Examples::

        >>> ipv6_cidr('::1/128')
        True

        >>> ipv6_cidr('::1')
        ValidationFailure(func=ipv6_cidr, args={'value': '::1'})
    """
    try:
        prefix, suffix = value.split('/', 2)
    except ValueError:
        return False
    if not ipv6(prefix) or not suffix.isdigit():
        return False
    return 0 <= int(suffix) <= 128

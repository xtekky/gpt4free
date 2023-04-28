import re

from validators.utils import validator

business_id_pattern = re.compile(r'^[0-9]{7}-[0-9]$')
ssn_checkmarks = '0123456789ABCDEFHJKLMNPRSTUVWXY'
ssn_pattern = re.compile(
    r"""^
    (?P<date>(0[1-9]|[1-2]\d|3[01])
    (0[1-9]|1[012])
    (\d{{2}}))
    [A+-]
    (?P<serial>(\d{{3}}))
    (?P<checksum>[{checkmarks}])$""".format(checkmarks=ssn_checkmarks),
    re.VERBOSE
)


@validator
def fi_business_id(business_id):
    """
    Validate a Finnish Business ID.

    Each company in Finland has a distinct business id. For more
    information see `Finnish Trade Register`_

    .. _Finnish Trade Register:
        http://en.wikipedia.org/wiki/Finnish_Trade_Register

    Examples::

        >>> fi_business_id('0112038-9')  # Fast Monkeys Ltd
        True

        >>> fi_business_id('1234567-8')  # Bogus ID
        ValidationFailure(func=fi_business_id, ...)

    .. versionadded:: 0.4
    .. versionchanged:: 0.5
        Method renamed from ``finnish_business_id`` to ``fi_business_id``

    :param business_id: business_id to validate
    """
    if not business_id or not re.match(business_id_pattern, business_id):
        return False
    factors = [7, 9, 10, 5, 8, 4, 2]
    numbers = map(int, business_id[:7])
    checksum = int(business_id[8])
    sum_ = sum(f * n for f, n in zip(factors, numbers))
    modulo = sum_ % 11
    return (11 - modulo == checksum) or (modulo == 0 and checksum == 0)


@validator
def fi_ssn(ssn, allow_temporal_ssn=True):
    """
    Validate a Finnish Social Security Number.

    This validator is based on `django-localflavor-fi`_.

    .. _django-localflavor-fi:
        https://github.com/django/django-localflavor-fi/

    Examples::

        >>> fi_ssn('010101-0101')
        True

        >>> fi_ssn('101010-0102')
        ValidationFailure(func=fi_ssn, args=...)

    .. versionadded:: 0.5

    :param ssn: Social Security Number to validate
    :param allow_temporal_ssn:
        Whether to accept temporal SSN numbers. Temporal SSN numbers are the
        ones where the serial is in the range [900-999]. By default temporal
        SSN numbers are valid.

    """
    if not ssn:
        return False

    result = re.match(ssn_pattern, ssn)
    if not result:
        return False
    gd = result.groupdict()
    checksum = int(gd['date'] + gd['serial'])
    return (
        int(gd['serial']) >= 2 and
        (allow_temporal_ssn or int(gd['serial']) <= 899) and
        ssn_checkmarks[checksum % len(ssn_checkmarks)] ==
        gd['checksum']
    )

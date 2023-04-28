import re

from .utils import validator

regex = (
    r'^[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}$'
)
pattern = re.compile(regex)


def char_value(char):
    """A=10, B=11, ..., Z=35
    """
    if char.isdigit():
        return int(char)
    else:
        return 10 + ord(char) - ord('A')


def modcheck(value):
    """Check if the value string passes the mod97-test.
    """
    # move country code and check numbers to end
    rearranged = value[4:] + value[:4]
    # convert letters to numbers
    converted = [char_value(char) for char in rearranged]
    # interpret as integer
    integerized = int(''.join([str(i) for i in converted]))
    return (integerized % 97 == 1)


@validator
def iban(value):
    """
    Return whether or not given value is a valid IBAN code.

    If the value is a valid IBAN this function returns ``True``, otherwise
    :class:`~validators.utils.ValidationFailure`.

    Examples::

        >>> iban('DE29100500001061045672')
        True

        >>> iban('123456')
        ValidationFailure(func=iban, ...)

    .. versionadded:: 0.8

    :param value: IBAN string to validate
    """
    return pattern.match(value) and modcheck(value)

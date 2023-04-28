import re

from .utils import validator


@validator
def card_number(value):
    """
    Return whether or not given value is a valid card number.

    This validator is based on Luhn algorithm.

    .. luhn:
       https://github.com/mmcloughlin/luhn

    Examples::

        >>> card_number('4242424242424242')
        True

        >>> card_number('4242424242424241')
        ValidationFailure(func=card_number, args={'value': '4242424242424241'})

    .. versionadded:: 0.15.0

    :param value: card number string to validate
    """
    try:
        digits = list(map(int, value))
        odd_sum = sum(digits[-1::-2])
        even_sum = sum([sum(divmod(2 * d, 10)) for d in digits[-2::-2]])
        return (odd_sum + even_sum) % 10 == 0
    except ValueError:
        return False


@validator
def visa(value):
    """
    Return whether or not given value is a valid Visa card number.

    Examples::

        >>> visa('4242424242424242')
        True

        >>> visa('2223003122003222')
        ValidationFailure(func=visa, args={'value': '2223003122003222'})

    .. versionadded:: 0.15.0

    :param value: Visa card number string to validate
    """
    pattern = re.compile(r'^4')
    return card_number(value) and len(value) == 16 and pattern.match(value)


@validator
def mastercard(value):
    """
    Return whether or not given value is a valid Mastercard card number.

    Examples::

        >>> mastercard('5555555555554444')
        True

        >>> mastercard('4242424242424242')
        ValidationFailure(func=mastercard, args={'value': '4242424242424242'})

    .. versionadded:: 0.15.0

    :param value: Mastercard card number string to validate
    """
    pattern = re.compile(r'^(51|52|53|54|55|22|23|24|25|26|27)')
    return card_number(value) and len(value) == 16 and pattern.match(value)


@validator
def amex(value):
    """
    Return whether or not given value is a valid American Express card number.

    Examples::

        >>> amex('378282246310005')
        True

        >>> amex('4242424242424242')
        ValidationFailure(func=amex, args={'value': '4242424242424242'})

    .. versionadded:: 0.15.0

    :param value: American Express card number string to validate
    """
    pattern = re.compile(r'^(34|37)')
    return card_number(value) and len(value) == 15 and pattern.match(value)


@validator
def unionpay(value):
    """
    Return whether or not given value is a valid UnionPay card number.

    Examples::

        >>> unionpay('6200000000000005')
        True

        >>> unionpay('4242424242424242')
        ValidationFailure(func=unionpay, args={'value': '4242424242424242'})

    .. versionadded:: 0.15.0

    :param value: UnionPay card number string to validate
    """
    pattern = re.compile(r'^62')
    return card_number(value) and len(value) == 16 and pattern.match(value)


@validator
def diners(value):
    """
    Return whether or not given value is a valid Diners Club card number.

    Examples::

        >>> diners('3056930009020004')
        True

        >>> diners('4242424242424242')
        ValidationFailure(func=diners, args={'value': '4242424242424242'})

    .. versionadded:: 0.15.0

    :param value: Diners Club card number string to validate
    """
    pattern = re.compile(r'^(30|36|38|39)')
    return (
        card_number(value) and len(value) in [14, 16] and pattern.match(value)
    )


@validator
def jcb(value):
    """
    Return whether or not given value is a valid JCB card number.

    Examples::

        >>> jcb('3566002020360505')
        True

        >>> jcb('4242424242424242')
        ValidationFailure(func=jcb, args={'value': '4242424242424242'})

    .. versionadded:: 0.15.0

    :param value: JCB card number string to validate
    """
    pattern = re.compile(r'^35')
    return card_number(value) and len(value) == 16 and pattern.match(value)


@validator
def discover(value):
    """
    Return whether or not given value is a valid Discover card number.

    Examples::

        >>> discover('6011111111111117')
        True

        >>> discover('4242424242424242')
        ValidationFailure(func=discover, args={'value': '4242424242424242'})

    .. versionadded:: 0.15.0

    :param value: Discover card number string to validate
    """
    pattern = re.compile(r'^(60|64|65)')
    return card_number(value) and len(value) == 16 and pattern.match(value)

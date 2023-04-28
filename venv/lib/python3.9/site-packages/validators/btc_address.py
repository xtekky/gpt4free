import re
from hashlib import sha256

from .utils import validator

segwit_pattern = re.compile(
    r'^(bc|tc)[0-3][02-9ac-hj-np-z]{14,74}$')


def validate_segwit_address(addr):
    return segwit_pattern.match(addr)


def decode_base58(addr):
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    return sum([
        (58 ** e) * alphabet.index(i)
        for e, i in enumerate(addr[::-1])
    ])


def validate_old_btc_address(addr):
    "Validate P2PKH and P2SH type address"
    if not len(addr) in range(25, 35):
        return False
    decoded_bytes = decode_base58(addr).to_bytes(25, "big")
    header = decoded_bytes[:-4]
    checksum = decoded_bytes[-4:]
    return checksum == sha256(sha256(header).digest()).digest()[:4]


@validator
def btc_address(value):
    """
    Return whether or not given value is a valid bitcoin address.

    If the value is valid bitcoin address this function returns ``True``,
    otherwise :class:`~validators.utils.ValidationFailure`.

    Full validation is implemented for P2PKH and P2SH addresses.
    For segwit addresses a regexp is used to provide a reasonable estimate
    on whether the address is valid.

    Examples::

        >>> btc_address('3Cwgr2g7vsi1bXDUkpEnVoRLA9w4FZfC69')
        True

    :param value: Bitcoin address string to validate
    """
    if not value or not isinstance(value, str):
        return False
    if value[:2] in ("bc", "tb"):
        return validate_segwit_address(value)
    return validate_old_btc_address(value)

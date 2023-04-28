import re

from .utils import validator

md5_regex = re.compile(
    r"^[0-9a-f]{32}$",
    re.IGNORECASE
)
sha1_regex = re.compile(
    r"^[0-9a-f]{40}$",
    re.IGNORECASE
)
sha224_regex = re.compile(
    r"^[0-9a-f]{56}$",
    re.IGNORECASE
)
sha256_regex = re.compile(
    r"^[0-9a-f]{64}$",
    re.IGNORECASE
)
sha512_regex = re.compile(
    r"^[0-9a-f]{128}$",
    re.IGNORECASE
)


@validator
def md5(value):
    """
    Return whether or not given value is a valid MD5 hash.

    Examples::

        >>> md5('d41d8cd98f00b204e9800998ecf8427e')
        True

        >>> md5('900zz11')
        ValidationFailure(func=md5, args={'value': '900zz11'})

    :param value: MD5 string to validate
    """
    return md5_regex.match(value)


@validator
def sha1(value):
    """
    Return whether or not given value is a valid SHA1 hash.

    Examples::

        >>> sha1('da39a3ee5e6b4b0d3255bfef95601890afd80709')
        True

        >>> sha1('900zz11')
        ValidationFailure(func=sha1, args={'value': '900zz11'})

    :param value: SHA1 string to validate
    """
    return sha1_regex.match(value)


@validator
def sha224(value):
    """
    Return whether or not given value is a valid SHA224 hash.

    Examples::

        >>> sha224('d14a028c2a3a2bc9476102bb288234c415a2b01f828ea62ac5b3e42f')
        True

        >>> sha224('900zz11')
        ValidationFailure(func=sha224, args={'value': '900zz11'})

    :param value: SHA224 string to validate
    """
    return sha224_regex.match(value)


@validator
def sha256(value):
    """
    Return whether or not given value is a valid SHA256 hash.

    Examples::

        >>> sha256(
        ...     'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b'
        ...     '855'
        ... )
        True

        >>> sha256('900zz11')
        ValidationFailure(func=sha256, args={'value': '900zz11'})

    :param value: SHA256 string to validate
    """
    return sha256_regex.match(value)


@validator
def sha512(value):
    """
    Return whether or not given value is a valid SHA512 hash.

    Examples::

        >>> sha512(
        ...     'cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce'
        ...     '9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af9'
        ...     '27da3e'
        ... )
        True

        >>> sha512('900zz11')
        ValidationFailure(func=sha512, args={'value': '900zz11'})

    :param value: SHA512 string to validate
    """
    return sha512_regex.match(value)

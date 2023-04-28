from __future__ import absolute_import

import re
from uuid import UUID

from .utils import validator

pattern = re.compile(r'^[0-9a-fA-F]{8}-([0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}$')


@validator
def uuid(value):
    """
    Return whether or not given value is a valid UUID.

    If the value is valid UUID this function returns ``True``, otherwise
    :class:`~validators.utils.ValidationFailure`.

    This validator is based on `WTForms UUID validator`_.

    .. _WTForms UUID validator:
       https://github.com/wtforms/wtforms/blob/master/wtforms/validators.py

    Examples::

        >>> uuid('2bc1c94f-0deb-43e9-92a1-4775189ec9f8')
        True

        >>> uuid('2bc1c94f 0deb-43e9-92a1-4775189ec9f8')
        ValidationFailure(func=uuid, ...)

    .. versionadded:: 0.2

    :param value: UUID value to validate
    """
    if isinstance(value, UUID):
        return True
    try:
        return pattern.match(value)
    except TypeError:
        return False

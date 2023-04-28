from .utils import validator


@validator
def truthy(value):
    """
    Validate that given value is not a falsey value.

    This validator is based on `WTForms DataRequired validator`_.

    .. _WTForms DataRequired validator:
       https://github.com/wtforms/wtforms/blob/master/wtforms/validators.py

    Examples::

        >>> truthy(1)
        True

        >>> truthy('someone')
        True

        >>> truthy(0)
        ValidationFailure(func=truthy, args={'value': 0})

        >>> truthy('    ')
        ValidationFailure(func=truthy, args={'value': '    '})

        >>> truthy(False)
        ValidationFailure(func=truthy, args={'value': False})

        >>> truthy(None)
        ValidationFailure(func=truthy, args={'value': None})

    .. versionadded:: 0.2
    """
    return (
        value and
        (not isinstance(value, str) or value.strip())
    )

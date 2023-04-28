import warnings
import functools


class AltairDeprecationWarning(UserWarning):
    pass


def deprecated(message=None):
    """Decorator to deprecate a function or class.

    Parameters
    ----------
    message : string (optional)
        The deprecation message
    """

    def wrapper(obj):
        return _deprecate(obj, message=message)

    return wrapper


def _deprecate(obj, name=None, message=None):
    """Return a version of a class or function that raises a deprecation warning.

    Parameters
    ----------
    obj : class or function
        The object to create a deprecated version of.
    name : string (optional)
        The name of the deprecated object
    message : string (optional)
        The deprecation message

    Returns
    -------
    deprecated_obj :
        The deprecated version of obj

    Examples
    --------
    >>> class Foo(object): pass
    >>> OldFoo = _deprecate(Foo, "OldFoo")
    >>> f = OldFoo()  # doctest: +SKIP
    AltairDeprecationWarning: alt.OldFoo is deprecated. Use alt.Foo instead.
    """
    if message is None:
        message = "alt.{} is deprecated. Use alt.{} instead." "".format(
            name, obj.__name__
        )
    if isinstance(obj, type):
        return type(
            name,
            (obj,),
            {
                "__doc__": obj.__doc__,
                "__init__": _deprecate(obj.__init__, "__init__", message),
            },
        )
    elif callable(obj):

        @functools.wraps(obj)
        def new_obj(*args, **kwargs):
            warnings.warn(message, AltairDeprecationWarning)
            return obj(*args, **kwargs)

        return new_obj
    else:
        raise ValueError("Cannot deprecate object of type {}".format(type(obj)))

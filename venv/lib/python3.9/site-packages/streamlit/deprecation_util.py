# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from typing import Any, Callable, List, TypeVar, cast

import streamlit
from streamlit import config
from streamlit.logger import get_logger

_LOGGER = get_logger(__name__)

TFunc = TypeVar("TFunc", bound=Callable[..., Any])
TObj = TypeVar("TObj", bound=object)


def _should_show_deprecation_warning_in_browser() -> bool:
    """True if we should print deprecation warnings to the browser."""
    return bool(config.get_option("client.showErrorDetails"))


def show_deprecation_warning(message: str) -> None:
    """Show a deprecation warning message."""
    if _should_show_deprecation_warning_in_browser():
        streamlit.warning(message)

    # We always log deprecation warnings
    _LOGGER.warning(message)


def make_deprecated_name_warning(
    old_name: str, new_name: str, removal_date: str
) -> str:
    return (
        f"Please replace `st.{old_name}` with `st.{new_name}`.\n\n"
        f"`st.{old_name}` will be removed after {removal_date}."
    )


def deprecate_func_name(func: TFunc, old_name: str, removal_date: str) -> TFunc:
    """Wrap an `st` function whose name has changed.

    Wrapped functions will run as normal, but will also show an st.warning
    saying that the old name will be removed after removal_date.

    (We generally set `removal_date` to 3 months from the deprecation date.)

    Parameters
    ----------
    func
        The `st.` function whose name has changed.

    old_name
        The function's deprecated name within __init__.py.

    removal_date
        A date like "2020-01-01", indicating the last day we'll guarantee
        support for the deprecated name.
    """

    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        result = func(*args, **kwargs)
        show_deprecation_warning(
            make_deprecated_name_warning(old_name, func.__name__, removal_date)
        )
        return result

    # Update the wrapped func's name & docstring so st.help does the right thing
    wrapped_func.__name__ = old_name
    wrapped_func.__doc__ = func.__doc__
    return cast(TFunc, wrapped_func)


def deprecate_obj_name(
    obj: TObj, old_name: str, new_name: str, removal_date: str
) -> TObj:
    """Wrap an `st` object whose name has changed.

    Wrapped objects will behave as normal, but will also show an st.warning
    saying that the old name will be removed after `removal_date`.

    (We generally set `removal_date` to 3 months from the deprecation date.)

    Parameters
    ----------
    obj
        The `st.` object whose name has changed.

    old_name
        The object's deprecated name within __init__.py.

    new_name
        The object's new name within __init__.py.

    removal_date
        A date like "2020-01-01", indicating the last day we'll guarantee
        support for the deprecated name.
    """

    return _create_deprecated_obj_wrapper(
        obj,
        lambda: show_deprecation_warning(
            make_deprecated_name_warning(old_name, new_name, removal_date)
        ),
    )


def _create_deprecated_obj_wrapper(obj: TObj, show_warning: Callable[[], Any]) -> TObj:
    """Create a wrapper for an object that has been deprecated. The first
    time one of the object's properties or functions is accessed, the
    given `show_warning` callback will be called.
    """
    has_shown_warning = False

    def maybe_show_warning() -> None:
        # Call `show_warning` if it hasn't already been called once.
        nonlocal has_shown_warning
        if not has_shown_warning:
            has_shown_warning = True
            show_warning()

    class Wrapper:
        def __init__(self):
            # Override all the Wrapped object's magic functions
            for name in Wrapper._get_magic_functions(obj.__class__):
                setattr(
                    self.__class__,
                    name,
                    property(self._make_magic_function_proxy(name)),
                )

        def __getattr__(self, attr):
            # We handle __getattr__ separately from our other magic
            # functions. The wrapped class may not actually implement it,
            # but we still need to implement it to call all its normal
            # functions.
            if attr in self.__dict__:
                return getattr(self, attr)

            maybe_show_warning()
            return getattr(obj, attr)

        @staticmethod
        def _get_magic_functions(cls) -> List[str]:
            # ignore the handful of magic functions we cannot override without
            # breaking the Wrapper.
            ignore = ("__class__", "__dict__", "__getattribute__", "__getattr__")
            return [
                name
                for name in dir(cls)
                if name not in ignore and name.startswith("__")
            ]

        @staticmethod
        def _make_magic_function_proxy(name):
            def proxy(self, *args):
                maybe_show_warning()
                return getattr(obj, name)

            return proxy

    return cast(TObj, Wrapper())

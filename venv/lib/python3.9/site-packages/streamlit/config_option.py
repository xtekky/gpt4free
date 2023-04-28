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

"""Class to store a key-value pair for the config system."""

import datetime
import re
import textwrap
from typing import Any, Callable, Optional

from streamlit import util
from streamlit.errors import DeprecationError


class ConfigOption:
    '''Stores a Streamlit configuration option.

    A configuration option, like 'browser.serverPort', which indicates which port
    to use when connecting to the proxy. There are two ways to create a
    ConfigOption:

    Simple ConfigOptions are created as follows:

        ConfigOption('browser.serverPort',
            description = 'Connect to the proxy at this port.',
            default_val = 8501)

    More complex config options resolve their values at runtime as follows:

        @ConfigOption('browser.serverPort')
        def _proxy_port():
            """Connect to the proxy at this port.

            Defaults to 8501.
            """
            return 8501

    NOTE: For complex config options, the function is called each time the
    option.value is evaluated!

    Attributes
    ----------
    key : str
        The fully qualified section.name
    value : any
        The value for this option. If this is a a complex config option then
        the callback is called EACH TIME value is evaluated.
    section : str
        The section of this option. Example: 'global'.
    name : str
        See __init__.
    description : str
        See __init__.
    where_defined : str
        Indicates which file set this config option.
        ConfigOption.DEFAULT_DEFINITION means this file.
    is_default: bool
        True if the config value is equal to its default value.
    visibility : {"visible", "hidden"}
        See __init__.
    scriptable : bool
        See __init__.
    deprecated: bool
        See __init__.
    deprecation_text : str or None
        See __init__.
    expiration_date : str or None
        See __init__.
    replaced_by : str or None
        See __init__.
    '''

    # This is a special value for ConfigOption.where_defined which indicates
    # that the option default was not overridden.
    DEFAULT_DEFINITION = "<default>"

    # This is a special value for ConfigOption.where_defined which indicates
    # that the options was defined by Streamlit's own code.
    STREAMLIT_DEFINITION = "<streamlit>"

    def __init__(
        self,
        key: str,
        description: Optional[str] = None,
        default_val: Optional[Any] = None,
        visibility: str = "visible",
        scriptable: bool = False,
        deprecated: bool = False,
        deprecation_text: Optional[str] = None,
        expiration_date: Optional[str] = None,
        replaced_by: Optional[str] = None,
        type_: type = str,
    ):
        """Create a ConfigOption with the given name.

        Parameters
        ----------
        key : str
            Should be of the form "section.optionName"
            Examples: server.name, deprecation.v1_0_featureName
        description : str
            Like a comment for the config option.
        default_val : any
            The value for this config option.
        visibility : {"visible", "hidden"}
            Whether this option should be shown to users.
        scriptable : bool
            Whether this config option can be set within a user script.
        deprecated: bool
            Whether this config option is deprecated.
        deprecation_text : str or None
            Required if deprecated == True. Set this to a string explaining
            what to use instead.
        expiration_date : str or None
            Required if deprecated == True. set this to the date at which it
            will no longer be accepted. Format: 'YYYY-MM-DD'.
        replaced_by : str or None
            If this is option has been deprecated in favor or another option,
            set this to the path to the new option. Example:
            'server.runOnSave'. If this is set, the 'deprecated' option
            will automatically be set to True, and deprecation_text will have a
            meaningful default (unless you override it).
        type_ : one of str, int, float or bool
            Useful to cast the config params sent by cmd option parameter.
        """
        # Parse out the section and name.
        self.key = key
        key_format = (
            # Capture a group called "section"
            r"(?P<section>"
            # Matching text comprised of letters and numbers that begins
            # with a lowercase letter with an optional "_" preceding it.
            # Examples: "_section", "section1"
            r"\_?[a-z][a-zA-Z0-9]*"
            r")"
            # Separator between groups
            r"\."
            # Capture a group called "name"
            r"(?P<name>"
            # Match text comprised of letters and numbers beginning with a
            # lowercase letter.
            # Examples: "name", "nameOfConfig", "config1"
            r"[a-z][a-zA-Z0-9]*"
            r")$"
        )
        match = re.match(key_format, self.key)
        assert match, f'Key "{self.key}" has invalid format.'
        self.section, self.name = match.group("section"), match.group("name")

        self.description = description

        self.visibility = visibility
        self.scriptable = scriptable
        self.default_val = default_val
        self.deprecated = deprecated
        self.replaced_by = replaced_by
        self.is_default = True
        self._get_val_func: Optional[Callable[[], Any]] = None
        self.where_defined = ConfigOption.DEFAULT_DEFINITION
        self.type = type_

        if self.replaced_by:
            self.deprecated = True
            if deprecation_text is None:
                deprecation_text = "Replaced by %s." % self.replaced_by

        if self.deprecated:
            assert expiration_date, "expiration_date is required for deprecated items"
            assert deprecation_text, "deprecation_text is required for deprecated items"
            self.expiration_date = expiration_date
            self.deprecation_text = textwrap.dedent(deprecation_text)

        self.set_value(default_val)

    def __repr__(self) -> str:
        return util.repr_(self)

    def __call__(self, get_val_func: Callable[[], Any]) -> "ConfigOption":
        """Assign a function to compute the value for this option.

        This method is called when ConfigOption is used as a decorator.

        Parameters
        ----------
        get_val_func : function
            A function which will be called to get the value of this parameter.
            We will use its docString as the description.

        Returns
        -------
        ConfigOption
            Returns self, which makes testing easier. See config_test.py.

        """
        assert (
            get_val_func.__doc__
        ), "Complex config options require doc strings for their description."
        self.description = get_val_func.__doc__
        self._get_val_func = get_val_func
        return self

    @property
    def value(self) -> Any:
        """Get the value of this config option."""
        if self._get_val_func is None:
            return None
        return self._get_val_func()

    def set_value(self, value: Any, where_defined: Optional[str] = None) -> None:
        """Set the value of this option.

        Parameters
        ----------
        value
            The new value for this parameter.
        where_defined : str
            New value to remember where this parameter was set.

        """
        self._get_val_func = lambda: value

        if where_defined is None:
            self.where_defined = ConfigOption.DEFAULT_DEFINITION
        else:
            self.where_defined = where_defined

        self.is_default = value == self.default_val

        if self.deprecated and self.where_defined != ConfigOption.DEFAULT_DEFINITION:

            details = {
                "key": self.key,
                "file": self.where_defined,
                "explanation": self.deprecation_text,
                "date": self.expiration_date,
            }

            if self.is_expired():
                raise DeprecationError(
                    textwrap.dedent(
                        """
                    ════════════════════════════════════════════════
                    %(key)s IS NO LONGER SUPPORTED.

                    %(explanation)s

                    Please update %(file)s.
                    ════════════════════════════════════════════════
                    """
                    )
                    % details
                )
            else:
                # Import here to avoid circular imports
                from streamlit.logger import get_logger

                LOGGER = get_logger(__name__)
                LOGGER.warning(
                    textwrap.dedent(
                        """
                    ════════════════════════════════════════════════
                    %(key)s IS DEPRECATED.
                    %(explanation)s

                    This option will be removed on or after %(date)s.

                    Please update %(file)s.
                    ════════════════════════════════════════════════
                    """
                    )
                    % details
                )

    def is_expired(self) -> bool:
        """Returns true if expiration_date is in the past."""
        if not self.deprecated:
            return False

        expiration_date = _parse_yyyymmdd_str(self.expiration_date)
        now = datetime.datetime.now()
        return now > expiration_date


def _parse_yyyymmdd_str(date_str: str) -> datetime.datetime:
    year, month, day = [int(token) for token in date_str.split("-", 2)]
    return datetime.datetime(year, month, day)

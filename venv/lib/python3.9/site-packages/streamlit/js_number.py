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

import numbers
from typing import Optional, Union


class JSNumberBoundsException(Exception):
    pass


class JSNumber(object):
    """Utility class for exposing JavaScript Number constants."""

    # The largest int that can be represented with perfect precision
    # in JavaScript.
    MAX_SAFE_INTEGER = (1 << 53) - 1

    # The smallest int that can be represented with perfect precision
    # in JavaScript.
    MIN_SAFE_INTEGER = -((1 << 53) - 1)

    # The largest float that can be represented in JavaScript.
    MAX_VALUE = 1.7976931348623157e308

    # The closest number to zero that can be represented in JavaScript.
    MIN_VALUE = 5e-324

    # The largest negative float that can be represented in JavaScript.
    MIN_NEGATIVE_VALUE = -MAX_VALUE

    @classmethod
    def validate_int_bounds(cls, value: int, value_name: Optional[str] = None) -> None:
        """Validate that an int value can be represented with perfect precision
        by a JavaScript Number.

        Parameters
        ----------
        value : int
        value_name : str or None
            The name of the value parameter. If specified, this will be used
            in any exception that is thrown.

        Raises
        ------
        JSNumberBoundsException
            Raised with a human-readable explanation if the value falls outside
            JavaScript int bounds.

        """
        if value_name is None:
            value_name = "value"

        if value < cls.MIN_SAFE_INTEGER:
            raise JSNumberBoundsException(
                "%s (%s) must be >= -((1 << 53) - 1)" % (value_name, value)
            )
        elif value > cls.MAX_SAFE_INTEGER:
            raise JSNumberBoundsException(
                "%s (%s) must be <= (1 << 53) - 1" % (value_name, value)
            )

    @classmethod
    def validate_float_bounds(
        cls, value: Union[int, float], value_name: Optional[str]
    ) -> None:
        """Validate that a float value can be represented by a JavaScript Number.

        Parameters
        ----------
        value : float
        value_name : str or None
            The name of the value parameter. If specified, this will be used
            in any exception that is thrown.

        Raises
        ------
        JSNumberBoundsException
            Raised with a human-readable explanation if the value falls outside
            JavaScript float bounds.

        """
        if value_name is None:
            value_name = "value"

        if not isinstance(value, (numbers.Integral, float)):
            raise JSNumberBoundsException(
                "%s (%s) is not a float" % (value_name, value)
            )
        elif value < cls.MIN_NEGATIVE_VALUE:
            raise JSNumberBoundsException(
                "%s (%s) must be >= -1.797e+308" % (value_name, value)
            )
        elif value > cls.MAX_VALUE:
            raise JSNumberBoundsException(
                "%s (%s) must be <= 1.797e+308" % (value_name, value)
            )

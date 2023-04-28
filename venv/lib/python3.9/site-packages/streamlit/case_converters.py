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

import re


def to_upper_camel_case(snake_case_str):
    """Converts snake_case to UpperCamelCase.

    Example
    -------
        foo_bar -> FooBar

    """
    return "".join(map(str.title, snake_case_str.split("_")))


def to_lower_camel_case(snake_case_str):
    """Converts snake_case to lowerCamelCase.

    Example
    -------
        foo_bar -> fooBar
        fooBar -> foobar

    """
    words = snake_case_str.split("_")
    if len(words) > 1:
        capitalized = [w.title() for w in words]
        capitalized[0] = words[0]
        return "".join(capitalized)
    else:
        return snake_case_str


def to_snake_case(camel_case_str):
    """Converts UpperCamelCase and lowerCamelCase to snake_case.

    Examples
    --------
        fooBar -> foo_bar
        BazBang -> baz_bang

    """
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_case_str)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def convert_dict_keys(func, in_dict):
    """Apply a conversion function to all keys in a dict.

    Parameters
    ----------
    func : callable
        The function to apply. Takes a str and returns a str.
    in_dict : dict
        The dictionary to convert. If some value in this dict is itself a dict,
        it also gets recursively converted.

    Returns
    -------
    dict
        A new dict with all the contents of `in_dict`, but with the keys
        converted by `func`.

    """
    out_dict = dict()
    for k, v in in_dict.items():
        converted_key = func(k)

        if type(v) is dict:
            out_dict[converted_key] = convert_dict_keys(func, v)
        else:
            out_dict[converted_key] = v
    return out_dict

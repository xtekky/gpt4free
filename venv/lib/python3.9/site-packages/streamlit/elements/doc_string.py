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

"""Allows us to create and absorb changes (aka Deltas) to elements."""

import ast
import contextlib
import inspect
import re
import sys
import types
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import Final

import streamlit
from streamlit.logger import get_logger
from streamlit.proto.DocString_pb2 import DocString as DocStringProto
from streamlit.proto.DocString_pb2 import Member as MemberProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner.script_runner import (
    __file__ as SCRIPTRUNNER_FILENAME,
)
from streamlit.runtime.secrets import Secrets
from streamlit.string_util import is_mem_address_str

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator


LOGGER: Final = get_logger(__name__)


CONFUSING_STREAMLIT_SIG_PREFIXES: Final = ("(element, ",)


class HelpMixin:
    @gather_metrics("help")
    def help(self, obj: Any = streamlit) -> "DeltaGenerator":
        """Display help and other information for a given object.

        Depending on the type of object that is passed in, this displays the
        object's name, type, value, signature, docstring, and member variables,
        methods — as well as the values/docstring of members and methods.

        Parameters
        ----------
        obj : any
            The object whose information should be displayed. If left
            unspecified, this call will display help for Streamlit itself.

        Example
        -------

        Don't remember how to initialize a dataframe? Try this:

        >>> import streamlit as st
        >>> import pandas
        >>>
        >>> st.help(pandas.DataFrame)

        .. output::
            https://doc-string.streamlit.app/
            height: 700px

        Want to quickly check what data type is output by a certain function?
        Try:

        >>> import streamlit as st
        >>>
        >>> x = my_poorly_documented_function()
        >>> st.help(x)

        Want to quickly inspect an object? No sweat:

        >>> class Dog:
        >>>   '''A typical dog.'''
        >>>
        >>>   def __init__(self, breed, color):
        >>>     self.breed = breed
        >>>     self.color = color
        >>>
        >>>   def bark(self):
        >>>     return 'Woof!'
        >>>
        >>>
        >>> fido = Dog('poodle', 'white')
        >>>
        >>> st.help(fido)

        .. output::
            https://doc-string1.streamlit.app/
            height: 300px

        And if you're using Magic, you can get help for functions, classes,
        and modules without even typing ``st.help``:

        >>> import streamlit as st
        >>> import pandas
        >>>
        >>> # Get help for Pandas read_csv:
        >>> pandas.read_csv
        >>>
        >>> # Get help for Streamlit itself:
        >>> st

        .. output::
            https://doc-string2.streamlit.app/
            height: 700px
        """
        doc_string_proto = DocStringProto()
        _marshall(doc_string_proto, obj)
        return self.dg._enqueue("doc_string", doc_string_proto)

    @property
    def dg(self) -> "DeltaGenerator":
        """Get our DeltaGenerator."""
        return cast("DeltaGenerator", self)


def _marshall(doc_string_proto: DocStringProto, obj: Any) -> None:
    """Construct a DocString object.

    See DeltaGenerator.help for docs.
    """
    var_name = _get_variable_name()
    if var_name is not None:
        doc_string_proto.name = var_name

    obj_type = _get_type_as_str(obj)
    doc_string_proto.type = obj_type

    obj_docs = _get_docstring(obj)
    if obj_docs is not None:
        doc_string_proto.doc_string = obj_docs

    obj_value = _get_value(obj, var_name)
    if obj_value is not None:
        doc_string_proto.value = obj_value

    doc_string_proto.members.extend(_get_members(obj))


def _get_name(obj):
    # Try to get the fully-qualified name of the object.
    # For example:
    #   st.help(bar.Baz(123))
    #
    #   The name is bar.Baz
    name = getattr(obj, "__qualname__", None)
    if name:
        return name

    # Try to get the name of the object.
    # For example:
    #   st.help(bar.Baz(123))
    #
    #   The name is Baz
    return getattr(obj, "__name__", None)


def _get_module(obj):
    return getattr(obj, "__module__", None)


def _get_signature(obj):
    if not inspect.isclass(obj) and not callable(obj):
        return None

    sig = ""

    # TODO: Can we replace below with this?
    # with contextlib.suppress(ValueError):
    #     sig = str(inspect.signature(obj))

    try:
        sig = str(inspect.signature(obj))
    except ValueError:
        sig = "(...)"
    except TypeError:
        return None

    is_delta_gen = False
    with contextlib.suppress(AttributeError):
        is_delta_gen = obj.__module__ == "streamlit.delta_generator"
        # Functions such as numpy.minimum don't have a __module__ attribute,
        # since we're only using it to check if its a DeltaGenerator, its ok
        # to continue

    if is_delta_gen:
        for prefix in CONFUSING_STREAMLIT_SIG_PREFIXES:
            if sig.startswith(prefix):
                sig = sig.replace(prefix, "(")
                break

    return sig


def _get_docstring(obj):
    doc_string = inspect.getdoc(obj)

    # Sometimes an object has no docstring, but the object's type does.
    # If that's the case here, use the type's docstring.
    # For objects where type is "type" we do not print the docs (e.g. int).
    # We also do not print the docs for functions and methods if the docstring is empty.
    if doc_string is None:
        obj_type = type(obj)

        if (
            obj_type is not type
            and obj_type is not types.ModuleType
            and not inspect.isfunction(obj)
            and not inspect.ismethod(obj)
        ):
            doc_string = inspect.getdoc(obj_type)

    if doc_string:
        return doc_string.strip()

    return None


def _get_variable_name():
    """Try to get the name of the variable in the current line, as set by the user.

    For example:
    foo = bar.Baz(123)
    st.help(foo)

    The name is "foo"
    """
    code = _get_current_line_of_code_as_str()

    if code is None:
        return None

    return _get_variable_name_from_code_str(code)


def _get_variable_name_from_code_str(code):
    tree = ast.parse(code)

    # Example:
    #
    # tree = Module(
    #   body=[
    #     Expr(
    #       value=Call(
    #         args=[
    #           Name(id='the variable name')
    #         ],
    #         keywords=[
    #           ???
    #         ],
    #       )
    #     )
    #   ]
    # )

    # Check if this is an magic call (i.e. it's not st.help or st.write).
    # If that's the case, just clean it up and return it.
    if not _is_stcommand(tree, command_name="help") and not _is_stcommand(
        tree, command_name="write"
    ):
        # A common pattern is to add "," at the end of a magic command to make it print.
        # This removes that final ",", so it looks nicer.
        if code.endswith(","):
            code = code[:-1]

        return code

    arg_node = _get_stcommand_arg(tree)

    # If st.help() is called without an argument, return no variable name.
    if not arg_node:
        return None

    # If walrus, get name.
    # E.g. st.help(foo := 123) should give you "foo".
    # (The hasattr is there to support Python 3.7)
    elif hasattr(ast, "NamedExpr") and type(arg_node) is ast.NamedExpr:
        # This next "if" will always be true, but need to add this for the type-checking test to
        # pass.
        if type(arg_node.target) is ast.Name:
            return arg_node.target.id

    # If constant, there's no variable name.
    # E.g. st.help("foo") or st.help(123) should give you None.
    elif type(arg_node) in (
        ast.Constant,
        # Python 3.7 support:
        ast.Num,
        ast.Str,
        ast.Bytes,
        ast.NameConstant,
        ast.Ellipsis,
    ):
        return None

    # Otherwise, return whatever is inside st.help(<-- here -->)

    # But, if multiline, only return the first line.
    code_lines = code.split("\n")
    is_multiline = len(code_lines) > 1

    start_offset = arg_node.col_offset

    if is_multiline:
        first_lineno = arg_node.lineno - 1  # Lines are 1-indexed!
        first_line = code_lines[first_lineno]
        end_offset = None

    else:
        first_line = code_lines[0]
        end_offset = getattr(arg_node, "end_col_offset", -1)

    # Python 3.7 and below have a bug where offset in some cases is off by one.
    # See https://github.com/python/cpython/commit/b619b097923155a7034c05c4018bf06af9f994d0
    # By the way, Python 3.7 also displays this bug when arg_node is a generator
    # expression, but in that case there are further complications, so we're leaving it out
    # of here. See the unit test for this for more details.
    if sys.version_info < (3, 8) and type(arg_node) is ast.ListComp:
        start_offset -= 1

    return first_line[start_offset:end_offset]


_NEWLINES = re.compile(r"[\n\r]+")


def _get_current_line_of_code_as_str():
    scriptrunner_frame = _get_scriptrunner_frame()

    if scriptrunner_frame is None:
        # If there's no ScriptRunner frame, something weird is going on. This
        # can happen when the script is executed with `python myscript.py`.
        # Either way, let's bail out nicely just in case there's some valid
        # edge case where this is OK.
        return None

    code_context = scriptrunner_frame.code_context

    if not code_context:
        # Sometimes a frame has no code_context. This can happen inside certain exec() calls, for
        # example. If this happens, we can't determine the variable name. Just return.
        # For the background on why exec() doesn't produce code_context, see
        # https://stackoverflow.com/a/12072941
        return None

    code_as_string = "".join(code_context)
    return re.sub(_NEWLINES, "", code_as_string.strip())


def _get_scriptrunner_frame():
    prev_frame = None
    scriptrunner_frame = None

    # Look back in call stack to get the variable name passed into st.help().
    # The frame *before* the ScriptRunner frame is the correct one.
    # IMPORTANT: This will change if we refactor the code. But hopefully our tests will catch the
    # issue and we'll fix it before it lands upstream!
    for frame in inspect.stack():
        # Check if this is running inside a funny "exec()" block that won't provide the info we
        # need. If so, just quit.
        if frame.code_context is None:
            return None

        if frame.filename == SCRIPTRUNNER_FILENAME:
            scriptrunner_frame = prev_frame
            break

        prev_frame = frame

    return scriptrunner_frame


def _is_stcommand(tree, command_name):
    """Checks whether the AST in tree is a call for command_name."""
    root_node = tree.body[0].value

    if not type(root_node) is ast.Call:
        return False

    return (
        # st call called without module. E.g. "help()"
        getattr(root_node.func, "id", None) == command_name
        or
        # st call called with module. E.g. "foo.help()" (where usually "foo" is "st")
        getattr(root_node.func, "attr", None) == command_name
    )


def _get_stcommand_arg(tree):
    """Gets the argument node for the st command in tree (AST)."""

    root_node = tree.body[0].value

    if root_node.args:
        return root_node.args[0]

    return None


def _get_type_as_str(obj):
    if inspect.isclass(obj):
        return "class"

    return str(type(obj).__name__)


def _get_first_line(text):
    if not text:
        return ""

    left, _, _ = text.partition("\n")
    return left


def _get_weight(value):
    if inspect.ismodule(value):
        return 3
    if inspect.isclass(value):
        return 2
    if callable(value):
        return 1
    return 0


def _get_value(obj, var_name):
    obj_value = _get_human_readable_value(obj)

    if obj_value is not None:
        return obj_value

    # If there's no human-readable value, it's some complex object.
    # So let's provide other info about it.
    name = _get_name(obj)

    if name:
        name_obj = obj
    else:
        # If the object itself doesn't have a name, then it's probably an instance
        # of some class Foo. So let's show info about Foo in the value slot.
        name_obj = type(obj)
        name = _get_name(name_obj)

    module = _get_module(name_obj)
    sig = _get_signature(name_obj) or ""

    if name:
        if module:
            obj_value = f"{module}.{name}{sig}"
        else:
            obj_value = f"{name}{sig}"

    if obj_value == var_name:
        # No need to repeat the same info.
        # For example: st.help(re) shouldn't show "re module re", just "re module".
        obj_value = None

    return obj_value


def _get_human_readable_value(value):
    if isinstance(value, Secrets):
        # Don't want to read secrets.toml because that will show a warning if there's no
        # secrets.toml file.
        return None

    if inspect.isclass(value) or inspect.ismodule(value) or callable(value):
        return None

    value_str = repr(value)

    if isinstance(value, str):
        # Special-case strings as human-readable because they're allowed to look like
        # "<foo blarg at 0x15ee6f9a0>".
        return _shorten(value_str)

    if is_mem_address_str(value_str):
        # If value_str looks like "<foo blarg at 0x15ee6f9a0>" it's not human readable.
        return None

    return _shorten(value_str)


def _shorten(s, length=300):
    s = s.strip()
    return s[:length] + "..." if len(s) > length else s


def _is_computed_property(obj, attr_name):
    obj_class = getattr(obj, "__class__", None)

    if not obj_class:
        return False

    # Go through superclasses in order of inheritance (mro) to see if any of them have an
    # attribute called attr_name. If so, check if it's a @property.
    for parent_class in inspect.getmro(obj_class):
        class_attr = getattr(parent_class, attr_name, None)

        if class_attr is None:
            continue

        # If is property, return it.
        if isinstance(class_attr, property) or inspect.isgetsetdescriptor(class_attr):
            return True

    return False


def _get_members(obj):
    members_for_sorting = []

    for attr_name in dir(obj):
        if attr_name.startswith("_"):
            continue

        is_computed_value = _is_computed_property(obj, attr_name)

        if is_computed_value:
            parent_attr = getattr(obj.__class__, attr_name)

            member_type = "property"

            weight = 0
            member_docs = _get_docstring(parent_attr)
            member_value = None
        else:
            attr_value = getattr(obj, attr_name)
            weight = _get_weight(attr_value)

            human_readable_value = _get_human_readable_value(attr_value)

            member_type = _get_type_as_str(attr_value)

            if human_readable_value is None:
                member_docs = _get_docstring(attr_value)
                member_value = None
            else:
                member_docs = None
                member_value = human_readable_value

        if member_type == "module":
            # Don't pollute the output with all imported modules.
            continue

        member = MemberProto()
        member.name = attr_name
        member.type = member_type

        if member_docs is not None:
            member.doc_string = _get_first_line(member_docs)

        if member_value is not None:
            member.value = member_value

        members_for_sorting.append((weight, member))

    if members_for_sorting:
        sorted_members = sorted(members_for_sorting, key=lambda x: (x[0], x[1].name))
        return [m for _, m in sorted_members]

    return []

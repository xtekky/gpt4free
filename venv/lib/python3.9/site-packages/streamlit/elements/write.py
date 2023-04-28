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

import inspect
import json as json
import types
from typing import TYPE_CHECKING, Any, List, Tuple, Type, cast

import numpy as np
from typing_extensions import Final

from streamlit import type_util
from streamlit.errors import StreamlitAPIException
from streamlit.logger import get_logger
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.state import SessionStateProxy
from streamlit.string_util import is_mem_address_str
from streamlit.user_info import UserInfoProxy

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator


# Special methods:
HELP_TYPES: Final[Tuple[Type[Any], ...]] = (
    types.BuiltinFunctionType,
    types.BuiltinMethodType,
    types.FunctionType,
    types.MethodType,
    types.ModuleType,
)

_LOGGER = get_logger(__name__)


class WriteMixin:
    @gather_metrics("write")
    def write(self, *args: Any, unsafe_allow_html: bool = False, **kwargs) -> None:
        """Write arguments to the app.

        This is the Swiss Army knife of Streamlit commands: it does different
        things depending on what you throw at it. Unlike other Streamlit commands,
        write() has some unique properties:

        1. You can pass in multiple arguments, all of which will be written.
        2. Its behavior depends on the input types as follows.
        3. It returns None, so its "slot" in the App cannot be reused.

        Parameters
        ----------
        *args : any
            One or many objects to print to the App.

            Arguments are handled as follows:

            - write(string)     : Prints the formatted Markdown string, with
                support for LaTeX expression, emoji shortcodes, and colored text.
                See docs for st.markdown for more.
            - write(data_frame) : Displays the DataFrame as a table.
            - write(error)      : Prints an exception specially.
            - write(func)       : Displays information about a function.
            - write(module)     : Displays information about the module.
            - write(class)      : Displays information about a class.
            - write(dict)       : Displays dict in an interactive widget.
            - write(mpl_fig)    : Displays a Matplotlib figure.
            - write(altair)     : Displays an Altair chart.
            - write(keras)      : Displays a Keras model.
            - write(graphviz)   : Displays a Graphviz graph.
            - write(plotly_fig) : Displays a Plotly figure.
            - write(bokeh_fig)  : Displays a Bokeh figure.
            - write(sympy_expr) : Prints SymPy expression using LaTeX.
            - write(htmlable)   : Prints _repr_html_() for the object if available.
            - write(obj)        : Prints str(obj) if otherwise unknown.

        unsafe_allow_html : bool
            This is a keyword-only argument that defaults to False.

            By default, any HTML tags found in strings will be escaped and
            therefore treated as pure text. This behavior may be turned off by
            setting this argument to True.

            That said, *we strongly advise against it*. It is hard to write secure
            HTML, so by using this argument you may be compromising your users'
            security. For more information, see:

            https://github.com/streamlit/streamlit/issues/152

        Example
        -------

        Its basic use case is to draw Markdown-formatted text, whenever the
        input is a string:

        >>> import streamlit as st
        >>>
        >>> st.write('Hello, *World!* :sunglasses:')

        ..  output::
            https://doc-write1.streamlitapp.com/
            height: 150px

        As mentioned earlier, ``st.write()`` also accepts other data formats, such as
        numbers, data frames, styled data frames, and assorted objects:

        >>> import streamlit as st
        >>> import pandas as pd
        >>>
        >>> st.write(1234)
        >>> st.write(pd.DataFrame({
        ...     'first column': [1, 2, 3, 4],
        ...     'second column': [10, 20, 30, 40],
        ... }))

        ..  output::
            https://doc-write2.streamlitapp.com/
            height: 350px

        Finally, you can pass in multiple arguments to do things like:

        >>> import streamlit as st
        >>>
        >>> st.write('1 + 1 = ', 2)
        >>> st.write('Below is a DataFrame:', data_frame, 'Above is a dataframe.')

        ..  output::
            https://doc-write3.streamlitapp.com/
            height: 410px

        Oh, one more thing: ``st.write`` accepts chart objects too! For example:

        >>> import streamlit as st
        >>> import pandas as pd
        >>> import numpy as np
        >>> import altair as alt
        >>>
        >>> df = pd.DataFrame(
        ...     np.random.randn(200, 3),
        ...     columns=['a', 'b', 'c'])
        ...
        >>> c = alt.Chart(df).mark_circle().encode(
        ...     x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])
        >>>
        >>> st.write(c)

        ..  output::
            https://doc-vega-lite-chart.streamlitapp.com/
            height: 300px

        """
        if kwargs:
            _LOGGER.warning(
                'Invalid arguments were passed to "st.write" function. Support for '
                "passing such unknown keywords arguments will be dropped in future. "
                "Invalid arguments were: %s",
                kwargs,
            )

        string_buffer: List[str] = []

        # This bans some valid cases like: e = st.empty(); e.write("a", "b").
        # BUT: 1) such cases are rare, 2) this rule is easy to understand,
        # and 3) this rule should be removed once we have st.container()
        if not self.dg._is_top_level and len(args) > 1:
            raise StreamlitAPIException(
                "Cannot replace a single element with multiple elements.\n\n"
                "The `write()` method only supports multiple elements when "
                "inserting elements rather than replacing. That is, only "
                "when called as `st.write()` or `st.sidebar.write()`."
            )

        def flush_buffer():
            if string_buffer:
                self.dg.markdown(
                    " ".join(string_buffer),
                    unsafe_allow_html=unsafe_allow_html,
                )
                string_buffer[:] = []

        for arg in args:
            # Order matters!
            if isinstance(arg, str):
                string_buffer.append(arg)
            elif type_util.is_snowpark_or_pyspark_data_object(arg):
                flush_buffer()
                self.dg.dataframe(arg)
            elif type_util.is_dataframe_like(arg):
                flush_buffer()
                if len(np.shape(arg)) > 2:
                    self.dg.text(arg)
                else:
                    self.dg.dataframe(arg)
            elif isinstance(arg, Exception):
                flush_buffer()
                self.dg.exception(arg)
            elif isinstance(arg, HELP_TYPES):
                flush_buffer()
                self.dg.help(arg)
            elif type_util.is_altair_chart(arg):
                flush_buffer()
                self.dg.altair_chart(arg)
            elif type_util.is_type(arg, "matplotlib.figure.Figure"):
                flush_buffer()
                self.dg.pyplot(arg)
            elif type_util.is_plotly_chart(arg):
                flush_buffer()
                self.dg.plotly_chart(arg)
            elif type_util.is_type(arg, "bokeh.plotting.figure.Figure"):
                flush_buffer()
                self.dg.bokeh_chart(arg)
            elif type_util.is_graphviz_chart(arg):
                flush_buffer()
                self.dg.graphviz_chart(arg)
            elif type_util.is_sympy_expession(arg):
                flush_buffer()
                self.dg.latex(arg)
            elif type_util.is_keras_model(arg):
                from tensorflow.python.keras.utils import vis_utils

                flush_buffer()
                dot = vis_utils.model_to_dot(arg)
                self.dg.graphviz_chart(dot.to_string())
            elif isinstance(arg, (dict, list, SessionStateProxy, UserInfoProxy)):
                flush_buffer()
                self.dg.json(arg)
            elif type_util.is_namedtuple(arg):
                flush_buffer()
                self.dg.json(json.dumps(arg._asdict()))
            elif type_util.is_pydeck(arg):
                flush_buffer()
                self.dg.pydeck_chart(arg)
            elif inspect.isclass(arg):
                flush_buffer()
                # We cast arg to type here to appease mypy, due to bug in mypy:
                # https://github.com/python/mypy/issues/12933
                self.dg.help(cast(type, arg))
            elif hasattr(arg, "_repr_html_"):
                self.dg.markdown(
                    arg._repr_html_(),
                    unsafe_allow_html=True,
                )
            else:
                stringified_arg = str(arg)

                if is_mem_address_str(stringified_arg):
                    flush_buffer()
                    self.dg.help(arg)

                else:
                    string_buffer.append("`%s`" % stringified_arg.replace("`", "\\`"))

        flush_buffer()

    @property
    def dg(self) -> "DeltaGenerator":
        """Get our DeltaGenerator."""
        return cast("DeltaGenerator", self)

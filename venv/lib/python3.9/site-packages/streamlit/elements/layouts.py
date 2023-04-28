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

from typing import TYPE_CHECKING, List, Optional, Sequence, Union, cast

from streamlit.deprecation_util import deprecate_func_name
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Block_pb2 import Block as BlockProto
from streamlit.runtime.metrics_util import gather_metrics

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

SpecType = Union[int, Sequence[Union[int, float]]]


class LayoutsMixin:
    @gather_metrics("container")
    def container(self) -> "DeltaGenerator":
        """Insert a multi-element container.

        Inserts an invisible container into your app that can be used to hold
        multiple elements. This allows you to, for example, insert multiple
        elements into your app out of order.

        To add elements to the returned container, you can use "with" notation
        (preferred) or just call methods directly on the returned object. See
        examples below.

        Examples
        --------
        Inserting elements using "with" notation:

        >>> import streamlit as st
        >>>
        >>> with st.container():
        ...    st.write("This is inside the container")
        ...
        ...    # You can call any Streamlit command, including custom components:
        ...    st.bar_chart(np.random.randn(50, 3))
        ...
        >>> st.write("This is outside the container")

        .. output ::
            https://doc-container1.streamlitapp.com/
            height: 520px

        Inserting elements out of order:

        >>> import streamlit as st
        >>>
        >>> container = st.container()
        >>> container.write("This is inside the container")
        >>> st.write("This is outside the container")
        >>>
        >>> # Now insert some more in the container
        >>> container.write("This is inside too")

        .. output ::
            https://doc-container2.streamlitapp.com/
            height: 480px
        """
        return self.dg._block()

    # TODO: Enforce that columns are not nested or in Sidebar
    @gather_metrics("columns")
    def columns(
        self, spec: SpecType, *, gap: Optional[str] = "small"
    ) -> List["DeltaGenerator"]:
        """Insert containers laid out as side-by-side columns.

        Inserts a number of multi-element containers laid out side-by-side and
        returns a list of container objects.

        To add elements to the returned containers, you can use "with" notation
        (preferred) or just call methods directly on the returned object. See
        examples below.

        Columns can only be placed inside other columns up to one level of nesting.

        .. warning::
            Columns cannot be placed inside other columns in the sidebar. This is only possible in the main area of the app.

        Parameters
        ----------
        spec : int or list of numbers
            If an int
                Specifies the number of columns to insert, and all columns
                have equal width.

            If a list of numbers
                Creates a column for each number, and each
                column's width is proportional to the number provided. Numbers can
                be ints or floats, but they must be positive.

                For example, `st.columns([3, 1, 2])` creates 3 columns where
                the first column is 3 times the width of the second, and the last
                column is 2 times that width.
        gap : string ("small", "medium", or "large")
            An optional string, which indicates the size of the gap between each column.
            The default is a small gap between columns. This argument can only be supplied by
            keyword.

        Returns
        -------
        list of containers
            A list of container objects.

        Examples
        --------
        You can use `with` notation to insert any element into a column:

        >>> import streamlit as st
        >>>
        >>> col1, col2, col3 = st.columns(3)
        >>>
        >>> with col1:
        ...    st.header("A cat")
        ...    st.image("https://static.streamlit.io/examples/cat.jpg")
        ...
        >>> with col2:
        ...    st.header("A dog")
        ...    st.image("https://static.streamlit.io/examples/dog.jpg")
        ...
        >>> with col3:
        ...    st.header("An owl")
        ...    st.image("https://static.streamlit.io/examples/owl.jpg")

        .. output ::
            https://doc-columns1.streamlitapp.com/
            height: 620px

        Or you can just call methods directly in the returned objects:

        >>> import streamlit as st
        >>> import numpy as np
        >>>
        >>> col1, col2 = st.columns([3, 1])
        >>> data = np.random.randn(10, 1)
        >>>
        >>> col1.subheader("A wide column with a chart")
        >>> col1.line_chart(data)
        >>>
        >>> col2.subheader("A narrow column with the data")
        >>> col2.write(data)

        .. output ::
            https://doc-columns2.streamlitapp.com/
            height: 550px

        """
        weights = spec
        weights_exception = StreamlitAPIException(
            "The input argument to st.columns must be either a "
            + "positive integer or a list of positive numeric weights. "
            + "See [documentation](https://docs.streamlit.io/library/api-reference/layout/st.columns) "
            + "for more information."
        )

        if isinstance(weights, int):
            # If the user provided a single number, expand into equal weights.
            # E.g. (1,) * 3 => (1, 1, 1)
            # NOTE: A negative/zero spec will expand into an empty tuple.
            weights = (1,) * weights

        if len(weights) == 0 or any(weight <= 0 for weight in weights):
            raise weights_exception

        def column_gap(gap):
            if type(gap) == str:
                gap_size = gap.lower()
                valid_sizes = ["small", "medium", "large"]

                if gap_size in valid_sizes:
                    return gap_size

            raise StreamlitAPIException(
                'The gap argument to st.columns must be "small", "medium", or "large". \n'
                f"The argument passed was {gap}."
            )

        gap_size = column_gap(gap)

        def column_proto(normalized_weight: float) -> BlockProto:
            col_proto = BlockProto()
            col_proto.column.weight = normalized_weight
            col_proto.column.gap = gap_size
            col_proto.allow_empty = True
            return col_proto

        block_proto = BlockProto()
        block_proto.horizontal.gap = gap_size
        row = self.dg._block(block_proto)
        total_weight = sum(weights)
        return [row._block(column_proto(w / total_weight)) for w in weights]

    @gather_metrics("tabs")
    def tabs(self, tabs: Sequence[str]) -> Sequence["DeltaGenerator"]:
        r"""Insert containers separated into tabs.

        Inserts a number of multi-element containers as tabs.
        Tabs are a navigational element that allows users to easily
        move between groups of related content.

        To add elements to the returned containers, you can use "with" notation
        (preferred) or just call methods directly on the returned object. See
        examples below.

        .. warning::
            All the content of every tab is always sent to and rendered on the frontend.
            Conditional rendering is currently not supported.

        Parameters
        ----------
        tabs : list of strings
            Creates a tab for each string in the list. The first tab is selected by default.
            The string is used as the name of the tab and can optionally contain Markdown,
            supporting the following elements: Bold, Italics, Strikethroughs, Inline Code,
            Emojis, and Links.

            This also supports:

            * Emoji shortcodes, such as ``:+1:``  and ``:sunglasses:``.
              For a list of all supported codes,
              see https://share.streamlit.io/streamlit/emoji-shortcodes.

            * LaTeX expressions, by wrapping them in "$" or "$$" (the "$$"
              must be on their own lines). Supported LaTeX functions are listed
              at https://katex.org/docs/supported.html.

            * Colored text, using the syntax ``:color[text to be colored]``,
              where ``color`` needs to be replaced with any of the following
              supported colors: blue, green, orange, red, violet.

            Unsupported elements are unwrapped so only their children (text contents) render.
            Display unsupported elements as literal characters by
            backslash-escaping them. E.g. ``1\. Not an ordered list``.

        Returns
        -------
        list of containers
            A list of container objects.

        Examples
        --------
        You can use `with` notation to insert any element into a tab:

        >>> import streamlit as st
        >>>
        >>> tab1, tab2, tab3 = st.tabs(["Cat", "Dog", "Owl"])
        >>>
        >>> with tab1:
        ...    st.header("A cat")
        ...    st.image("https://static.streamlit.io/examples/cat.jpg", width=200)
        ...
        >>> with tab2:
        ...    st.header("A dog")
        ...    st.image("https://static.streamlit.io/examples/dog.jpg", width=200)
        ...
        >>> with tab3:
        ...    st.header("An owl")
        ...    st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

        .. output ::
            https://doc-tabs1.streamlitapp.com/
            height: 620px

        Or you can just call methods directly in the returned objects:

        >>> import streamlit as st
        >>> import numpy as np
        >>>
        >>> tab1, tab2 = st.tabs(["📈 Chart", "🗃 Data"])
        >>> data = np.random.randn(10, 1)
        >>>
        >>> tab1.subheader("A tab with a chart")
        >>> tab1.line_chart(data)
        >>>
        >>> tab2.subheader("A tab with the data")
        >>> tab2.write(data)


        .. output ::
            https://doc-tabs2.streamlitapp.com/
            height: 700px

        """
        if not tabs:
            raise StreamlitAPIException(
                "The input argument to st.tabs must contain at least one tab label."
            )

        if any(isinstance(tab, str) == False for tab in tabs):
            raise StreamlitAPIException(
                "The tabs input list to st.tabs is only allowed to contain strings."
            )

        def tab_proto(label: str) -> BlockProto:
            tab_proto = BlockProto()
            tab_proto.tab.label = label
            tab_proto.allow_empty = True
            return tab_proto

        block_proto = BlockProto()
        block_proto.tab_container.SetInParent()
        tab_container = self.dg._block(block_proto)
        return tuple(tab_container._block(tab_proto(tab_label)) for tab_label in tabs)

    @gather_metrics("expander")
    def expander(self, label: str, expanded: bool = False) -> "DeltaGenerator":
        r"""Insert a multi-element container that can be expanded/collapsed.

        Inserts a container into your app that can be used to hold multiple elements
        and can be expanded or collapsed by the user. When collapsed, all that is
        visible is the provided label.

        To add elements to the returned container, you can use "with" notation
        (preferred) or just call methods directly on the returned object. See
        examples below.

        .. warning::
            Currently, you may not put expanders inside another expander.

        Parameters
        ----------
        label : str
            A string to use as the header for the expander. The label can optionally
            contain Markdown and supports the following elements: Bold, Italics,
            Strikethroughs, Inline Code, Emojis, and Links.

            This also supports:

            * Emoji shortcodes, such as ``:+1:``  and ``:sunglasses:``.
              For a list of all supported codes,
              see https://share.streamlit.io/streamlit/emoji-shortcodes.

            * LaTeX expressions, by wrapping them in "$" or "$$" (the "$$"
              must be on their own lines). Supported LaTeX functions are listed
              at https://katex.org/docs/supported.html.

            * Colored text, using the syntax ``:color[text to be colored]``,
              where ``color`` needs to be replaced with any of the following
              supported colors: blue, green, orange, red, violet.

            Unsupported elements are unwrapped so only their children (text contents) render.
            Display unsupported elements as literal characters by
            backslash-escaping them. E.g. ``1\. Not an ordered list``.
        expanded : bool
            If True, initializes the expander in "expanded" state. Defaults to
            False (collapsed).

        Examples
        --------
        You can use `with` notation to insert any element into an expander

        >>> import streamlit as st
        >>>
        >>> st.bar_chart({"data": [1, 5, 2, 6, 2, 1]})
        >>>
        >>> with st.expander("See explanation"):
        ...     st.write(\"\"\"
        ...         The chart above shows some numbers I picked for you.
        ...         I rolled actual dice for these, so they're *guaranteed* to
        ...         be random.
        ...     \"\"\")
        ...     st.image("https://static.streamlit.io/examples/dice.jpg")

        .. output ::
            https://doc-expander.streamlitapp.com/
            height: 750px

        Or you can just call methods directly in the returned objects:

        >>> import streamlit as st
        >>>
        >>> st.bar_chart({"data": [1, 5, 2, 6, 2, 1]})
        >>>
        >>> expander = st.expander("See explanation")
        >>> expander.write(\"\"\"
        ...     The chart above shows some numbers I picked for you.
        ...     I rolled actual dice for these, so they're *guaranteed* to
        ...     be random.
        ... \"\"\")
        >>> expander.image("https://static.streamlit.io/examples/dice.jpg")

        .. output ::
            https://doc-expander.streamlitapp.com/
            height: 750px

        """
        if label is None:
            raise StreamlitAPIException("A label is required for an expander")

        expandable_proto = BlockProto.Expandable()
        expandable_proto.expanded = expanded
        expandable_proto.label = label

        block_proto = BlockProto()
        block_proto.allow_empty = True
        block_proto.expandable.CopyFrom(expandable_proto)

        return self.dg._block(block_proto=block_proto)

    @property
    def dg(self) -> "DeltaGenerator":
        """Get our DeltaGenerator."""
        return cast("DeltaGenerator", self)

    # Deprecated beta_ functions
    beta_container = deprecate_func_name(container, "beta_container", "2021-11-02")
    beta_expander = deprecate_func_name(expander, "beta_expander", "2021-11-02")
    beta_columns = deprecate_func_name(columns, "beta_columns", "2021-11-02")

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

"""Streamlit support for Matplotlib PyPlot charts."""

import io
from typing import TYPE_CHECKING, Any, Optional, cast

from typing_extensions import Final

import streamlit.elements.image as image_utils
from streamlit import config
from streamlit.errors import StreamlitDeprecationWarning
from streamlit.logger import get_logger
from streamlit.proto.Image_pb2 import ImageList as ImageListProto
from streamlit.runtime.metrics_util import gather_metrics

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from streamlit.delta_generator import DeltaGenerator

LOGGER: Final = get_logger(__name__)


class PyplotMixin:
    @gather_metrics("pyplot")
    def pyplot(
        self,
        fig: Optional["Figure"] = None,
        clear_figure: Optional[bool] = None,
        use_container_width: bool = True,
        **kwargs: Any,
    ) -> "DeltaGenerator":
        """Display a matplotlib.pyplot figure.

        Parameters
        ----------
        fig : Matplotlib Figure
            The figure to plot. When this argument isn't specified, this
            function will render the global figure (but this is deprecated,
            as described below)

        clear_figure : bool
            If True, the figure will be cleared after being rendered.
            If False, the figure will not be cleared after being rendered.
            If left unspecified, we pick a default based on the value of `fig`.

            * If `fig` is set, defaults to `False`.

            * If `fig` is not set, defaults to `True`. This simulates Jupyter's
              approach to matplotlib rendering.

        use_container_width : bool
            If True, set the chart width to the column width. Defaults to `True`.

        **kwargs : any
            Arguments to pass to Matplotlib's savefig function.

        Example
        -------
        >>> import streamlit as st
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>>
        >>> arr = np.random.normal(1, 1, size=100)
        >>> fig, ax = plt.subplots()
        >>> ax.hist(arr, bins=20)
        >>>
        >>> st.pyplot(fig)

        .. output::
           https://doc-pyplot.streamlitapp.com/
           height: 630px

        Notes
        -----
        .. note::
           Deprecation warning. After December 1st, 2020, we will remove the ability
           to specify no arguments in `st.pyplot()`, as that requires the use of
           Matplotlib's global figure object, which is not thread-safe. So
           please always pass a figure object as shown in the example section
           above.

        Matplotlib supports several types of "backends". If you're getting an
        error using Matplotlib with Streamlit, try setting your backend to "TkAgg"::

            echo "backend: TkAgg" >> ~/.matplotlib/matplotlibrc

        For more information, see https://matplotlib.org/faq/usage_faq.html.

        """

        if not fig and config.get_option("deprecation.showPyplotGlobalUse"):
            self.dg.exception(PyplotGlobalUseWarning())

        image_list_proto = ImageListProto()
        marshall(
            self.dg._get_delta_path_str(),
            image_list_proto,
            fig,
            clear_figure,
            use_container_width,
            **kwargs,
        )
        return self.dg._enqueue("imgs", image_list_proto)

    @property
    def dg(self) -> "DeltaGenerator":
        """Get our DeltaGenerator."""
        return cast("DeltaGenerator", self)


def marshall(
    coordinates: str,
    image_list_proto: ImageListProto,
    fig: Optional["Figure"] = None,
    clear_figure: Optional[bool] = True,
    use_container_width: bool = True,
    **kwargs: Any,
) -> None:
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        plt.ioff()
    except ImportError:
        raise ImportError("pyplot() command requires matplotlib")

    # You can call .savefig() on a Figure object or directly on the pyplot
    # module, in which case you're doing it to the latest Figure.
    if not fig:
        if clear_figure is None:
            clear_figure = True

        fig = plt

    # Normally, dpi is set to 'figure', and the figure's dpi is set to 100.
    # So here we pick double of that to make things look good in a high
    # DPI display.
    options = {"bbox_inches": "tight", "dpi": 200, "format": "png"}

    # If some options are passed in from kwargs then replace the values in
    # options with the ones from kwargs
    options = {a: kwargs.get(a, b) for a, b in options.items()}
    # Merge options back into kwargs.
    kwargs.update(options)

    image = io.BytesIO()
    fig.savefig(image, **kwargs)
    image_width = (
        image_utils.WidthBehaviour.COLUMN
        if use_container_width
        else image_utils.WidthBehaviour.ORIGINAL
    )
    image_utils.marshall_images(
        coordinates=coordinates,
        image=image,
        caption=None,
        width=image_width,
        proto_imgs=image_list_proto,
        clamp=False,
        channels="RGB",
        output_format="PNG",
    )

    # Clear the figure after rendering it. This means that subsequent
    # plt calls will be starting fresh.
    if clear_figure:
        fig.clf()


class PyplotGlobalUseWarning(StreamlitDeprecationWarning):
    def __init__(self) -> None:
        super(PyplotGlobalUseWarning, self).__init__(
            msg=self._get_message(), config_option="deprecation.showPyplotGlobalUse"
        )

    def _get_message(self) -> str:
        return """
You are calling `st.pyplot()` without any arguments. After December 1st, 2020,
we will remove the ability to do this as it requires the use of Matplotlib's global
figure object, which is not thread-safe.

To future-proof this code, you should pass in a figure as shown below:

```python
>>> fig, ax = plt.subplots()
>>> ax.scatter([1, 2, 3], [1, 2, 3])
>>>    ... other plotting actions ...
>>> st.pyplot(fig)
```
"""

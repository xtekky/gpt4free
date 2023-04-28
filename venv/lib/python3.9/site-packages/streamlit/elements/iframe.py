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

from typing import TYPE_CHECKING, Optional, cast

from streamlit.proto.IFrame_pb2 import IFrame as IFrameProto
from streamlit.runtime.metrics_util import gather_metrics

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator


class IframeMixin:
    @gather_metrics("_iframe")
    def _iframe(
        self,
        src: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        scrolling: bool = False,
    ) -> "DeltaGenerator":
        """Load a remote URL in an iframe.

        Parameters
        ----------
        src : str
            The URL of the page to embed.
        width : int
            The width of the frame in CSS pixels. Defaults to the app's
            default element width.
        height : int
            The height of the frame in CSS pixels. Defaults to 150.
        scrolling : bool
            If True, show a scrollbar when the content is larger than the iframe.
            Otherwise, do not show a scrollbar. Defaults to False.

        """
        iframe_proto = IFrameProto()
        marshall(
            iframe_proto,
            src=src,
            width=width,
            height=height,
            scrolling=scrolling,
        )
        return self.dg._enqueue("iframe", iframe_proto)

    @gather_metrics("_html")
    def _html(
        self,
        html: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        scrolling: bool = False,
    ) -> "DeltaGenerator":
        """Display an HTML string in an iframe.

        Parameters
        ----------
        html : str
            The HTML string to embed in the iframe.
        width : int
            The width of the frame in CSS pixels. Defaults to the app's
            default element width.
        height : int
            The height of the frame in CSS pixels. Defaults to 150.
        scrolling : bool
            If True, show a scrollbar when the content is larger than the iframe.
            Otherwise, do not show a scrollbar. Defaults to False.

        """
        iframe_proto = IFrameProto()
        marshall(
            iframe_proto,
            srcdoc=html,
            width=width,
            height=height,
            scrolling=scrolling,
        )
        return self.dg._enqueue("iframe", iframe_proto)

    @property
    def dg(self) -> "DeltaGenerator":
        """Get our DeltaGenerator."""
        return cast("DeltaGenerator", self)


def marshall(
    proto: IFrameProto,
    src: Optional[str] = None,
    srcdoc: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    scrolling: bool = False,
) -> None:
    """Marshalls data into an IFrame proto.

    These parameters correspond directly to <iframe> attributes, which are
    described in more detail at
    https://developer.mozilla.org/en-US/docs/Web/HTML/Element/iframe.

    Parameters
    ----------
    proto : IFrame protobuf
        The protobuf object to marshall data into.
    src : str
        The URL of the page to embed.
    srcdoc : str
        Inline HTML to embed. Overrides src.
    width : int
        The width of the frame in CSS pixels. Defaults to the app's
        default element width.
    height : int
        The height of the frame in CSS pixels. Defaults to 150.
    scrolling : bool
        If true, show a scrollbar when the content is larger than the iframe.
        Otherwise, never show a scrollbar.

    """
    if src is not None:
        proto.src = src

    if srcdoc is not None:
        proto.srcdoc = srcdoc

    if width is not None:
        proto.width = width
        proto.has_width = True

    if height is not None:
        proto.height = height
    else:
        proto.height = 150

    proto.scrolling = scrolling

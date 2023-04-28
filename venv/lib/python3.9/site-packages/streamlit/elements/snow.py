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

from typing import TYPE_CHECKING, cast

from streamlit.proto.Snow_pb2 import Snow as SnowProto
from streamlit.runtime.metrics_util import gather_metrics

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator


class SnowMixin:
    @gather_metrics("snow")
    def snow(self) -> "DeltaGenerator":
        """Draw celebratory snowfall.

        Example
        -------
        >>> import streamlit as st
        >>>
        >>> st.snow()

        ...then watch your app and get ready for a cool celebration!

        """
        snow_proto = SnowProto()
        snow_proto.show = True
        return self.dg._enqueue("snow", snow_proto)

    @property
    def dg(self) -> "DeltaGenerator":
        """Get our DeltaGenerator."""
        return cast("DeltaGenerator", self)

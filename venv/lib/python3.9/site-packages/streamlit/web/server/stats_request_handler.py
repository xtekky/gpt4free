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

from typing import List

import tornado.web

from streamlit.proto.openmetrics_data_model_pb2 import GAUGE
from streamlit.proto.openmetrics_data_model_pb2 import MetricSet as MetricSetProto
from streamlit.runtime.stats import CacheStat, StatsManager
from streamlit.web.server.server_util import emit_endpoint_deprecation_notice


class StatsRequestHandler(tornado.web.RequestHandler):
    def initialize(self, stats_manager: StatsManager) -> None:
        self._manager = stats_manager

    def set_default_headers(self):
        # Avoid a circular import
        from streamlit.web.server import allow_cross_origin_requests

        if allow_cross_origin_requests():
            self.set_header("Access-Control-Allow-Origin", "*")

    def options(self):
        """/OPTIONS handler for preflight CORS checks."""
        self.set_status(204)
        self.finish()

    def get(self) -> None:
        if self.request.uri and "_stcore/" not in self.request.uri:
            emit_endpoint_deprecation_notice(self, new_path="/_stcore/metrics")

        stats = self._manager.get_stats()

        # If the request asked for protobuf output, we return a serialized
        # protobuf. Else we return text.
        if "application/x-protobuf" in self.request.headers.get_list("Accept"):
            self.write(self._stats_to_proto(stats).SerializeToString())
            self.set_header("Content-Type", "application/x-protobuf")
            self.set_status(200)
        else:
            self.write(self._stats_to_text(self._manager.get_stats()))
            self.set_header("Content-Type", "application/openmetrics-text")
            self.set_status(200)

    @staticmethod
    def _stats_to_text(stats: List[CacheStat]) -> str:
        metric_type = "# TYPE cache_memory_bytes gauge"
        metric_unit = "# UNIT cache_memory_bytes bytes"
        metric_help = "# HELP Total memory consumed by a cache."
        openmetrics_eof = "# EOF\n"

        # Format: header, stats, EOF
        result = [metric_type, metric_unit, metric_help]
        result.extend(stat.to_metric_str() for stat in stats)
        result.append(openmetrics_eof)

        return "\n".join(result)

    @staticmethod
    def _stats_to_proto(stats: List[CacheStat]) -> MetricSetProto:
        metric_set = MetricSetProto()

        metric_family = metric_set.metric_families.add()
        metric_family.name = "cache_memory_bytes"
        metric_family.type = GAUGE
        metric_family.unit = "bytes"
        metric_family.help = "Total memory consumed by a cache."

        for stat in stats:
            metric_proto = metric_family.metrics.add()
            stat.marshall_metric_proto(metric_proto)

        metric_set = MetricSetProto()
        metric_set.metric_families.append(metric_family)
        return metric_set

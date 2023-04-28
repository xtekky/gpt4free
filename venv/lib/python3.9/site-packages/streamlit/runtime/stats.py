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

from abc import abstractmethod
from typing import List, NamedTuple

from typing_extensions import Protocol, runtime_checkable

from streamlit.proto.openmetrics_data_model_pb2 import Metric as MetricProto


class CacheStat(NamedTuple):
    """Describes a single cache entry.

    Properties
    ----------
    category_name : str
        A human-readable name for the cache "category" that the entry belongs
        to - e.g. "st.memo", "session_state", etc.
    cache_name : str
        A human-readable name for cache instance that the entry belongs to.
        For "st.memo" and other function decorator caches, this might be the
        name of the cached function. If the cache category doesn't have
        multiple separate cache instances, this can just be the empty string.
    byte_length : int
        The entry's memory footprint in bytes.
    """

    category_name: str
    cache_name: str
    byte_length: int

    def to_metric_str(self) -> str:
        return 'cache_memory_bytes{cache_type="%s",cache="%s"} %s' % (
            self.category_name,
            self.cache_name,
            self.byte_length,
        )

    def marshall_metric_proto(self, metric: MetricProto) -> None:
        """Fill an OpenMetrics `Metric` protobuf object."""
        label = metric.labels.add()
        label.name = "cache_type"
        label.value = self.category_name

        label = metric.labels.add()
        label.name = "cache"
        label.value = self.cache_name

        metric_point = metric.metric_points.add()
        metric_point.gauge_value.int_value = self.byte_length


@runtime_checkable
class CacheStatsProvider(Protocol):
    @abstractmethod
    def get_stats(self) -> List[CacheStat]:
        raise NotImplementedError


class StatsManager:
    def __init__(self):
        self._cache_stats_providers: List[CacheStatsProvider] = []

    def register_provider(self, provider: CacheStatsProvider) -> None:
        """Register a CacheStatsProvider with the manager.
        This function is not thread-safe. Call it immediately after
        creation.
        """
        self._cache_stats_providers.append(provider)

    def get_stats(self) -> List[CacheStat]:
        """Return a list containing all stats from each registered provider."""
        all_stats: List[CacheStat] = []
        for provider in self._cache_stats_providers:
            all_stats.extend(provider.get_stats())
        return all_stats

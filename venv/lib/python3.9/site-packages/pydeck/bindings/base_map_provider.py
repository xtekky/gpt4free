from enum import Enum


class BaseMapProvider(Enum):
    """Basemap provider available in pydeck"""

    MAPBOX = "mapbox"
    GOOGLE_MAPS = "google_maps"
    CARTO = "carto"

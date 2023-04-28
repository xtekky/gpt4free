import warnings


DARK = "dark"
LIGHT = "light"
SATELLITE = "satellite"
ROAD = "road"
DARK_NO_LABELS = "dark_no_labels"
LIGHT_NO_LABELS = "light_no_labels"

MAPBOX_LIGHT = "mapbox://styles/mapbox/light-v9"
MAPBOX_DARK = "mapbox://styles/mapbox/dark-v9"
MAPBOX_ROAD = "mapbox://styles/mapbox/streets-v9"
MAPBOX_SATELLITE = "mapbox://styles/mapbox/satellite-v9"

CARTO_DARK = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
CARTO_DARK_NO_LABELS = "https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json"
CARTO_LIGHT = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
CARTO_LIGHT_NO_LABELS = "https://basemaps.cartocdn.com/gl/positron-nolabels-gl-style/style.json"
CARTO_ROAD = "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json"

GOOGLE_SATELLITE = "satellite"
GOOGLE_ROAD = "roadmap"

styles = {
    DARK: {"mapbox": MAPBOX_DARK, "carto": CARTO_DARK},
    DARK_NO_LABELS: {"carto": CARTO_DARK_NO_LABELS},
    LIGHT: {"mapbox": MAPBOX_LIGHT, "carto": CARTO_LIGHT},
    LIGHT_NO_LABELS: {"carto": CARTO_LIGHT_NO_LABELS},
    ROAD: {"carto": CARTO_ROAD, "google_maps": GOOGLE_ROAD, "mapbox": MAPBOX_ROAD},
    SATELLITE: {"mapbox": MAPBOX_SATELLITE, "google_maps": GOOGLE_SATELLITE},
}


def get_from_map_identifier(map_identifier: str, provider: str) -> str:
    """Attempt to get a style URI by map provider, otherwise pass the map identifier
    to the API service

    Provide reasonable cross-provider default map styles

    Parameters
    ----------
    map_identifier : str
        Either a specific map provider style or a token indicating a map style. Currently
        tokens are "dark", "light", "satellite", "road", "dark_no_labels", or "light_no_labels".
        Not all map styles are available for all providers.
    provider : str
        One of "carto", "mapbox", or "google_maps", indicating the associated base map tile provider.

    Returns
    -------
    str
        Base map URI

    """
    try:
        return styles[map_identifier][provider]
    except KeyError:
        return map_identifier

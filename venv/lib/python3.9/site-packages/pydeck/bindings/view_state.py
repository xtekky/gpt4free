from .json_tools import JSONMixin


class ViewState(JSONMixin):
    """An object that represents where the state of a viewport, essentially where the screen is focused.

    If you have two dimensional data and you don't want to set this manually,
    see :func:`pydeck.data_utils.viewport_helpers.compute_view`.


    Parameters
    ---------
    longitude : float, default None
        x-coordinate of focus
    latitude : float, default None
        y-coordinate of focus
    zoom : float, default None
        Magnification level of the map, usually between 0 (representing the whole world)
        and 24 (close to individual buildings)
    min_zoom : float, default None
        Least mangified zoom level the user can navigate to
    max_zoom : float, default None
        Most magnified zoom level the user can navigate to
    pitch : float, default None
        Up/down angle relative to the map's plane, with 0 being looking directly at the map
    bearing : float, default None
        Left/right angle relative to the map's true north, with 0 being aligned to true north
    """

    def __init__(
        self, longitude=None, latitude=None, zoom=None, min_zoom=None, max_zoom=None, pitch=None, bearing=None, **kwargs
    ):
        self.longitude = longitude
        self.latitude = latitude
        self.zoom = zoom
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.pitch = pitch
        self.bearing = bearing

        if kwargs:
            self.__dict__.update(kwargs)

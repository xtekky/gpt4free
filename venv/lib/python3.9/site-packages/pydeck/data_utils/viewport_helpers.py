"""
Functions that make it easier to provide a default centering
for a view state
"""
import math
from ..bindings.view_state import ViewState
from .type_checking import is_pandas_df


def _squared_diff(x, x0):
    return (x0 - x) * (x0 - x)


def euclidean(y, y1):
    """Euclidean distance in n-dimensions

    Parameters
    ----------
    y : tuple of float
        A point in n-dimensions
    y1 : tuple of float
        A point in n-dimensions

    Examples
    --------
    >>> EPSILON = 0.001
    >>> euclidean((3, 6, 5), (7, -5, 1)) - 12.369 < EPSILON
    True
    """
    if not len(y) == len(y1):
        raise Exception("Input coordinates must be of the same length")
    return math.sqrt(sum([_squared_diff(x, x0) for x, x0 in zip(y, y1)]))


def geometric_mean(points):
    """Gets centroid in a series of points

    Parameters
    ----------
    points : list of list of float
        List of (x, y) coordinates

    Returns
    -------
    tuple
        The centroid of a list of points
    """
    avg_x = sum([float(p[0]) for p in points]) / len(points)
    avg_y = sum([float(p[1]) for p in points]) / len(points)
    return (avg_x, avg_y)


def get_bbox(points):
    """Get the bounding box around the data,

    Parameters
    ----------
    points : list of list of float
        List of (x, y) coordinates

    Returns
    -------
    dict
        Dictionary containing the top left and bottom right points of a bounding box
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    max_x = max(xs)
    max_y = max(ys)
    min_x = min(xs)
    min_y = min(ys)
    return ((min_x, max_y), (max_x, min_y))


def k_nearest_neighbors(points, center, k):
    """Gets the k furthest points from the center

    Parameters
    ----------
    points : list of list of float
        List of (x, y) coordinates
    center : list of list of float
        Center point
    k : int
        Number of points

    Returns
    -------
    list
        Index of the k furthest points

    Todo
    ---
    Currently implemently naively, needs to be more efficient
    """
    pts_with_distance = [(pt, euclidean(pt, center)) for pt in points]
    sorted_pts = sorted(pts_with_distance, key=lambda x: x[1])
    return [x[0] for x in sorted_pts][: int(k)]


def get_n_pct(points, proportion=1):
    """Computes the bounding box of the maximum zoom for the specified list of points

    Parameters
    ----------
    points : list of list of float
        List of (x, y) coordinates
    proportion : float, default 1
        Value between 0 and 1 representing the minimum proportion of data to be captured

    Returns
    -------
    list
        k nearest data points
    """
    if proportion == 1:
        return points
    # Compute the medioid of the data
    centroid = geometric_mean(points)
    # Retain the closest n*proportion points
    n_to_keep = math.floor(proportion * len(points))
    return k_nearest_neighbors(points, centroid, n_to_keep)


def bbox_to_zoom_level(bbox):
    """Computes the zoom level of a lat/lng bounding box

    Parameters
    ----------
    bbox : list of list of float
        Northwest and southeast corners of a bounding box, given as two points in a list

    Returns
    -------
    int
        Zoom level of map in a WGS84 Mercator projection (e.g., like that of Google Maps)
    """
    lat_diff = max(bbox[0][0], bbox[1][0]) - min(bbox[0][0], bbox[1][0])
    lng_diff = max(bbox[0][1], bbox[1][1]) - min(bbox[0][1], bbox[1][1])

    max_diff = max(lng_diff, lat_diff)
    zoom_level = None
    if max_diff < (360.0 / math.pow(2, 20)):
        zoom_level = 21
    else:
        zoom_level = int(-1 * ((math.log(max_diff) / math.log(2.0)) - (math.log(360.0) / math.log(2))))
        if zoom_level < 1:
            zoom_level = 1
    return zoom_level


def compute_view(points, view_proportion=1, view_type=ViewState):
    """Automatically computes a zoom level for the points passed in.

    Parameters
    ----------
    points : list of list of float or pandas.DataFrame
        A list of points
    view_propotion : float, default 1
        Proportion of the data that is meaningful to plot
    view_type : class constructor for pydeck.ViewState, default :class:`pydeck.bindings.view_state.ViewState`
        Class constructor for a viewport. In the current version of pydeck,
        users most likely do not have to modify this attribute.

    Returns
    -------
    pydeck.Viewport
        Viewport fitted to the data
    """
    if is_pandas_df(points):
        points = points.to_records(index=False)
    bbox = get_bbox(get_n_pct(points, view_proportion))
    zoom = bbox_to_zoom_level(bbox)
    center = geometric_mean(points)
    instance = view_type(latitude=center[1], longitude=center[0], zoom=zoom)
    return instance

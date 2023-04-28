from .json_tools import JSONMixin

TYPE_IDENTIFIER = "@@type"


class View(JSONMixin):
    """
    Represents a "hard configuration" of a camera location

    Parameters
    ---------
    type : str, default None
        deck.gl view to display, e.g., MapView
    controller : bool, default None
        If enabled, camera becomes interactive.
    **kwargs
        Any of the parameters passable to a deck.gl View
    """

    def __init__(self, type=None, controller=None, width=None, height=None, **kwargs):
        self.type = type
        self.controller = controller
        self.width = width
        self.height = height
        self.__dict__.update(kwargs)

    @property
    def type(self):
        return getattr(self, TYPE_IDENTIFIER)

    @type.setter
    def type(self, type_name):
        self.__setattr__(TYPE_IDENTIFIER, type_name)

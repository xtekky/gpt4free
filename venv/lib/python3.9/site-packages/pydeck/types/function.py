from functools import total_ordering

from .base import PydeckType


@total_ordering
class Function(PydeckType):
    """Indicate a function type with arguments and set already in pydeck

    Parameters
    ----------

    name : str
        Function name
    **kwargs
        arguments and value of each argument to be storing the function information
    """

    __KEY = "@@function"

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.arguments = kwargs

    def __lt__(self, other):
        return str(self) < str(other)

    def __eq__(self, other):
        return str(self) == str(other)

    def serialize(self):
        repr = {self.__KEY: self.name}
        repr.update(self.arguments)
        return repr

"""
Compatibility layer to allow Pympler being used from different Python versions.
"""

from typing import Any, Iterable

try:
    import tkinter
except ImportError:
    tkinter = None  # type: ignore


# Helper functions

def object_in_list(obj: Any, l: Iterable) -> bool:
    """Returns True if object o is in list.

    Required compatibility function to handle WeakSet objects.
    """
    for o in l:
        if o is obj:
            return True
    return False

"""Utilities for registering and working with themes"""

from .plugin_registry import PluginRegistry
from typing import Callable

ThemeType = Callable[..., dict]


class ThemeRegistry(PluginRegistry[ThemeType]):
    pass

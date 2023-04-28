from typing import Any, Dict, List, Optional, Generic, TypeVar, cast
from types import TracebackType

import entrypoints
from toolz import curry


PluginType = TypeVar("PluginType")


class PluginEnabler(object):
    """Context manager for enabling plugins

    This object lets you use enable() as a context manager to
    temporarily enable a given plugin::

        with plugins.enable('name'):
            do_something()  # 'name' plugin temporarily enabled
        # plugins back to original state
    """

    def __init__(self, registry: "PluginRegistry", name: str, **options):
        self.registry = registry  # type: PluginRegistry
        self.name = name  # type: str
        self.options = options  # type: Dict[str, Any]
        self.original_state = registry._get_state()  # type: Dict[str, Any]
        self.registry._enable(name, **options)

    def __enter__(self) -> "PluginEnabler":
        return self

    def __exit__(self, typ: type, value: Exception, traceback: TracebackType) -> None:
        self.registry._set_state(self.original_state)

    def __repr__(self) -> str:
        return "{}.enable({!r})".format(self.registry.__class__.__name__, self.name)


class PluginRegistry(Generic[PluginType]):
    """A registry for plugins.

    This is a plugin registry that allows plugins to be loaded/registered
    in two ways:

    1. Through an explicit call to ``.register(name, value)``.
    2. By looking for other Python packages that are installed and provide
       a setuptools entry point group.

    When you create an instance of this class, provide the name of the
    entry point group to use::

        reg = PluginRegister('my_entrypoint_group')

    """

    # this is a mapping of name to error message to allow custom error messages
    # in case an entrypoint is not found
    entrypoint_err_messages = {}  # type: Dict[str, str]

    # global settings is a key-value mapping of settings that are stored globally
    # in the registry rather than passed to the plugins
    _global_settings = {}  # type: Dict[str, Any]

    def __init__(self, entry_point_group: str = "", plugin_type: type = object):
        """Create a PluginRegistry for a named entry point group.

        Parameters
        ==========
        entry_point_group: str
            The name of the entry point group.
        plugin_type: object
            A type that will optionally be used for runtime type checking of
            loaded plugins using isinstance.
        """
        self.entry_point_group = entry_point_group  # type: str
        self.plugin_type = plugin_type  # type: Optional[type]
        self._active = None  # type: Optional[PluginType]
        self._active_name = ""  # type: str
        self._plugins = {}  # type: Dict[str, PluginType]
        self._options = {}  # type: Dict[str, Any]
        self._global_settings = self.__class__._global_settings.copy()  # type: dict

    def register(self, name: str, value: Optional[PluginType]) -> Optional[PluginType]:
        """Register a plugin by name and value.

        This method is used for explicit registration of a plugin and shouldn't be
        used to manage entry point managed plugins, which are auto-loaded.

        Parameters
        ==========
        name: str
            The name of the plugin.
        value: PluginType or None
            The actual plugin object to register or None to unregister that plugin.

        Returns
        =======
        plugin: PluginType or None
            The plugin that was registered or unregistered.
        """
        if value is None:
            return self._plugins.pop(name, None)
        else:
            assert isinstance(value, self.plugin_type)
            self._plugins[name] = value
            return value

    def names(self) -> List[str]:
        """List the names of the registered and entry points plugins."""
        exts = list(self._plugins.keys())
        more_exts = [
            ep.name for ep in entrypoints.get_group_all(self.entry_point_group)
        ]
        exts.extend(more_exts)
        return sorted(set(exts))

    def _get_state(self) -> Dict[str, Any]:
        """Return a dictionary representing the current state of the registry"""
        return {
            "_active": self._active,
            "_active_name": self._active_name,
            "_plugins": self._plugins.copy(),
            "_options": self._options.copy(),
            "_global_settings": self._global_settings.copy(),
        }

    def _set_state(self, state: Dict[str, Any]) -> None:
        """Reset the state of the registry"""
        assert set(state.keys()) == {
            "_active",
            "_active_name",
            "_plugins",
            "_options",
            "_global_settings",
        }
        for key, val in state.items():
            setattr(self, key, val)

    def _enable(self, name: str, **options) -> None:
        if name not in self._plugins:
            try:
                ep = entrypoints.get_single(self.entry_point_group, name)
            except entrypoints.NoSuchEntryPoint:
                if name in self.entrypoint_err_messages:
                    raise ValueError(self.entrypoint_err_messages[name])
                else:
                    raise
            value = cast(PluginType, ep.load())
            self.register(name, value)
        self._active_name = name
        self._active = self._plugins[name]
        for key in set(options.keys()) & set(self._global_settings.keys()):
            self._global_settings[key] = options.pop(key)
        self._options = options

    def enable(self, name: Optional[str] = None, **options) -> PluginEnabler:
        """Enable a plugin by name.

        This can be either called directly, or used as a context manager.

        Parameters
        ----------
        name : string (optional)
            The name of the plugin to enable. If not specified, then use the
            current active name.
        **options :
            Any additional parameters will be passed to the plugin as keyword
            arguments

        Returns
        -------
        PluginEnabler:
            An object that allows enable() to be used as a context manager
        """
        if name is None:
            name = self.active
        return PluginEnabler(self, name, **options)

    @property
    def active(self) -> str:
        """Return the name of the currently active plugin"""
        return self._active_name

    @property
    def options(self) -> Dict[str, Any]:
        """Return the current options dictionary"""
        return self._options

    def get(self) -> Optional[PluginType]:
        """Return the currently active plugin."""
        if self._options:
            return curry(self._active, **self._options)
        else:
            return self._active

    def __repr__(self) -> str:
        return "{}(active={!r}, registered={!r})" "".format(
            self.__class__.__name__, self._active_name, list(self.names())
        )

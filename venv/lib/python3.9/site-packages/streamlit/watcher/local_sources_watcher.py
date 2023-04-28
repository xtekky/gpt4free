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

import collections
import os
import sys
import types
from typing import Callable, Dict, List, Optional, Set

from streamlit import config, file_util
from streamlit.folder_black_list import FolderBlackList
from streamlit.logger import get_logger
from streamlit.source_util import get_pages
from streamlit.watcher.path_watcher import (
    NoOpPathWatcher,
    get_default_path_watcher_class,
)

LOGGER = get_logger(__name__)

WatchedModule = collections.namedtuple("WatchedModule", ["watcher", "module_name"])

# This needs to be initialized lazily to avoid calling config.get_option() and
# thus initializing config options when this file is first imported.
PathWatcher = None


class LocalSourcesWatcher:
    def __init__(self, main_script_path: str):
        self._main_script_path = os.path.abspath(main_script_path)
        self._script_folder = os.path.dirname(self._main_script_path)
        self._on_file_changed: List[Callable[[str], None]] = []
        self._is_closed = False
        self._cached_sys_modules: Set[str] = set()

        # Blacklist for folders that should not be watched
        self._folder_black_list = FolderBlackList(
            config.get_option("server.folderWatchBlacklist")
        )

        self._watched_modules: Dict[str, WatchedModule] = {}

        for page_info in get_pages(self._main_script_path).values():
            self._register_watcher(
                page_info["script_path"],
                module_name=None,  # Only root scripts have their modules set to None
            )

    def register_file_change_callback(self, cb: Callable[[str], None]) -> None:
        self._on_file_changed.append(cb)

    def on_file_changed(self, filepath):
        if filepath not in self._watched_modules:
            LOGGER.error("Received event for non-watched file: %s", filepath)
            return

        # Workaround:
        # Delete all watched modules so we can guarantee changes to the
        # updated module are reflected on reload.
        #
        # In principle, for reloading a given module, we only need to unload
        # the module itself and all of the modules which import it (directly
        # or indirectly) such that when we exec the application code, the
        # changes are reloaded and reflected in the running application.
        #
        # However, determining all import paths for a given loaded module is
        # non-trivial, and so as a workaround we simply unload all watched
        # modules.
        for wm in self._watched_modules.values():
            if wm.module_name is not None and wm.module_name in sys.modules:
                del sys.modules[wm.module_name]

        for cb in self._on_file_changed:
            cb(filepath)

    def close(self):
        for wm in self._watched_modules.values():
            wm.watcher.close()
        self._watched_modules = {}
        self._is_closed = True

    def _register_watcher(self, filepath, module_name):
        global PathWatcher
        if PathWatcher is None:
            PathWatcher = get_default_path_watcher_class()

        if PathWatcher is NoOpPathWatcher:
            return

        try:
            wm = WatchedModule(
                watcher=PathWatcher(filepath, self.on_file_changed),
                module_name=module_name,
            )
        except PermissionError:
            # If you don't have permission to read this file, don't even add it
            # to watchers.
            return

        self._watched_modules[filepath] = wm

    def _deregister_watcher(self, filepath):
        if filepath not in self._watched_modules:
            return

        if filepath == self._main_script_path:
            return

        wm = self._watched_modules[filepath]
        wm.watcher.close()
        del self._watched_modules[filepath]

    def _file_is_new(self, filepath):
        return filepath not in self._watched_modules

    def _file_should_be_watched(self, filepath):
        # Using short circuiting for performance.
        return self._file_is_new(filepath) and (
            file_util.file_is_in_folder_glob(filepath, self._script_folder)
            or file_util.file_in_pythonpath(filepath)
        )

    def update_watched_modules(self):
        if self._is_closed:
            return

        if set(sys.modules) != self._cached_sys_modules:
            modules_paths = {
                name: self._exclude_blacklisted_paths(get_module_paths(module))
                for name, module in dict(sys.modules).items()
            }
            self._cached_sys_modules = set(sys.modules)
            self._register_necessary_watchers(modules_paths)

    def _register_necessary_watchers(self, module_paths: Dict[str, Set[str]]) -> None:
        for name, paths in module_paths.items():
            for path in paths:
                if self._file_should_be_watched(path):
                    self._register_watcher(path, name)

    def _exclude_blacklisted_paths(self, paths: Set[str]) -> Set[str]:
        return {p for p in paths if not self._folder_black_list.is_blacklisted(p)}


def get_module_paths(module: types.ModuleType) -> Set[str]:
    paths_extractors = [
        # https://docs.python.org/3/reference/datamodel.html
        # __file__ is the pathname of the file from which the module was loaded
        # if it was loaded from a file.
        # The __file__ attribute may be missing for certain types of modules
        lambda m: [m.__file__],
        # https://docs.python.org/3/reference/import.html#__spec__
        # The __spec__ attribute is set to the module spec that was used
        # when importing the module. one exception is __main__,
        # where __spec__ is set to None in some cases.
        # https://www.python.org/dev/peps/pep-0451/#id16
        # "origin" in an import context means the system
        # (or resource within a system) from which a module originates
        # ... It is up to the loader to decide on how to interpret
        # and use a module's origin, if at all.
        lambda m: [m.__spec__.origin],
        # https://www.python.org/dev/peps/pep-0420/
        # Handling of "namespace packages" in which the __path__ attribute
        # is a _NamespacePath object with a _path attribute containing
        # the various paths of the package.
        lambda m: [p for p in m.__path__._path],
    ]

    all_paths = set()
    for extract_paths in paths_extractors:
        potential_paths = []
        try:
            potential_paths = extract_paths(module)
        except AttributeError:
            # Some modules might not have __file__ or __spec__ attributes.
            pass
        except Exception as e:
            LOGGER.warning(f"Examining the path of {module.__name__} raised: {e}")

        all_paths.update(
            [os.path.abspath(str(p)) for p in potential_paths if _is_valid_path(p)]
        )
    return all_paths


def _is_valid_path(path: Optional[str]) -> bool:
    return isinstance(path, str) and (os.path.isfile(path) or os.path.isdir(path))

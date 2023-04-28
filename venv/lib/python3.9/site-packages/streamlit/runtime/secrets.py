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

import os
import threading
from copy import deepcopy
from typing import (
    Any,
    Dict,
    ItemsView,
    Iterator,
    KeysView,
    List,
    Mapping,
    NoReturn,
    Optional,
    ValuesView,
)

import toml
from blinker import Signal
from typing_extensions import Final

import streamlit as st
import streamlit.watcher.path_watcher
from streamlit import file_util, runtime
from streamlit.logger import get_logger

_LOGGER = get_logger(__name__)
SECRETS_FILE_LOCS: Final[List[str]] = [
    file_util.get_streamlit_file_path("secrets.toml"),
    # NOTE: The order here is important! Project-level secrets should overwrite global
    # secrets.
    file_util.get_project_streamlit_file_path("secrets.toml"),
]


def _missing_attr_error_message(attr_name: str) -> str:
    return (
        f'st.secrets has no attribute "{attr_name}". '
        f"Did you forget to add it to secrets.toml or the app settings on Streamlit Cloud? "
        f"More info: https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app/connect-to-data-sources/secrets-management"
    )


def _missing_key_error_message(key: str) -> str:
    return (
        f'st.secrets has no key "{key}". '
        f"Did you forget to add it to secrets.toml or the app settings on Streamlit Cloud? "
        f"More info: https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app/connect-to-data-sources/secrets-management"
    )


class AttrDict(Mapping[str, Any]):
    """
    We use AttrDict to wrap up dictionary values from secrets
    to provide dot access to nested secrets
    """

    def __init__(self, value):
        self.__dict__["__nested_secrets__"] = dict(value)

    @staticmethod
    def _maybe_wrap_in_attr_dict(value) -> Any:
        if not isinstance(value, Mapping):
            return value
        else:
            return AttrDict(value)

    def __len__(self) -> int:
        return len(self.__nested_secrets__)

    def __iter__(self) -> Iterator[str]:
        return iter(self.__nested_secrets__)

    def __getitem__(self, key: str) -> Any:
        try:
            value = self.__nested_secrets__[key]
            return self._maybe_wrap_in_attr_dict(value)
        except KeyError:
            raise KeyError(_missing_key_error_message(key))

    def __getattr__(self, attr_name: str) -> Any:
        try:
            value = self.__nested_secrets__[attr_name]
            return self._maybe_wrap_in_attr_dict(value)
        except KeyError:
            raise AttributeError(_missing_attr_error_message(attr_name))

    def __repr__(self):
        return repr(self.__nested_secrets__)

    def __setitem__(self, key, value) -> NoReturn:
        raise TypeError("Secrets does not support item assignment.")

    def __setattr__(self, key, value) -> NoReturn:
        raise TypeError("Secrets does not support attribute assignment.")

    def to_dict(self) -> Dict[str, Any]:
        return deepcopy(self.__nested_secrets__)


class Secrets(Mapping[str, Any]):
    """A dict-like class that stores secrets.
    Parses secrets.toml on-demand. Cannot be externally mutated.

    Safe to use from multiple threads.
    """

    def __init__(self, file_paths: List[str]):
        # Our secrets dict.
        self._secrets: Optional[Mapping[str, Any]] = None
        self._lock = threading.RLock()
        self._file_watchers_installed = False
        self._file_paths = file_paths

        self.file_change_listener = Signal(
            doc="Emitted when a `secrets.toml` file has been changed."
        )

    def load_if_toml_exists(self) -> bool:
        """Load secrets.toml files from disk if they exists. If none exist,
        no exception will be raised. (If a file exists but is malformed,
        an exception *will* be raised.)

        Returns True if a secrets.toml file was successfully parsed, False otherwise.

        Thread-safe.
        """
        try:
            self._parse(print_exceptions=False)
            return True
        except FileNotFoundError:
            # No secrets.toml files exist. That's fine.
            return False

    def _reset(self) -> None:
        """Clear the secrets dictionary and remove any secrets that were
        added to os.environ.

        Thread-safe.
        """
        with self._lock:
            if self._secrets is None:
                return

            for k, v in self._secrets.items():
                self._maybe_delete_environment_variable(k, v)
            self._secrets = None

    def _parse(self, print_exceptions: bool) -> Mapping[str, Any]:
        """Parse our secrets.toml files if they're not already parsed.
        This function is safe to call from multiple threads.

        Parameters
        ----------
        print_exceptions : bool
            If True, then exceptions will be printed with `st.error` before
            being re-raised.

        Raises
        ------
        FileNotFoundError
            Raised if secrets.toml doesn't exist.

        """
        # Avoid taking a lock for the common case where secrets are already
        # loaded.
        secrets = self._secrets
        if secrets is not None:
            return secrets

        with self._lock:
            if self._secrets is not None:
                return self._secrets

            # It's fine for a user to only have one secrets.toml file defined, so
            # we ignore individual FileNotFoundErrors when attempting to read files
            # below and only raise an exception if we weren't able read *any* secrets
            # file.
            found_secrets_file = False
            secrets = {}

            for path in self._file_paths:
                try:
                    with open(path, encoding="utf-8") as f:
                        secrets_file_str = f.read()
                    found_secrets_file = True
                except FileNotFoundError:
                    continue

                try:
                    secrets.update(toml.loads(secrets_file_str))
                except:
                    if print_exceptions:
                        st.error(f"Error parsing secrets file at {path}")
                    raise

            if not found_secrets_file:
                err_msg = f"No secrets files found. Valid paths for a secrets.toml file are: {', '.join(self._file_paths)}"
                if print_exceptions:
                    st.error(err_msg)
                raise FileNotFoundError(err_msg)

            if len([p for p in self._file_paths if os.path.exists(p)]) > 1:
                _LOGGER.info(
                    f"Secrets found in multiple locations: {', '.join(self._file_paths)}. "
                    "When multiple secret.toml files exist, local secrets will take precedence over global secrets."
                )

            for k, v in secrets.items():
                self._maybe_set_environment_variable(k, v)

            self._secrets = secrets
            self._maybe_install_file_watchers()

            return self._secrets

    @staticmethod
    def _maybe_set_environment_variable(k: Any, v: Any) -> None:
        """Add the given key/value pair to os.environ if the value
        is a string, int, or float.
        """
        value_type = type(v)
        if value_type in (str, int, float):
            os.environ[k] = str(v)

    @staticmethod
    def _maybe_delete_environment_variable(k: Any, v: Any) -> None:
        """Remove the given key/value pair from os.environ if the value
        is a string, int, or float.
        """
        value_type = type(v)
        if value_type in (str, int, float) and os.environ.get(k) == v:
            del os.environ[k]

    def _maybe_install_file_watchers(self) -> None:
        with self._lock:
            if self._file_watchers_installed:
                return

            for path in self._file_paths:
                try:
                    streamlit.watcher.path_watcher.watch_file(
                        path,
                        self._on_secrets_file_changed,
                        watcher_type="poll",
                    )
                except FileNotFoundError:
                    # A user may only have one secrets.toml file defined, so we'd expect
                    # FileNotFoundErrors to be raised when attempting to install a
                    # watcher on the nonexistent ones.
                    pass

            # We set file_watchers_installed to True even if the installation attempt
            # failed to avoid repeatedly trying to install it.
            self._file_watchers_installed = True

    def _on_secrets_file_changed(self, changed_file_path) -> None:
        with self._lock:
            _LOGGER.debug("Secrets file %s changed, reloading", changed_file_path)
            self._reset()
            self._parse(print_exceptions=True)

        # Emit a signal to notify receivers that the `secrets.toml` file
        # has been changed.
        self.file_change_listener.send()

    def __getattr__(self, key: str) -> Any:
        """Return the value with the given key. If no such key
        exists, raise an AttributeError.

        Thread-safe.
        """
        try:
            value = self._parse(True)[key]
            if not isinstance(value, Mapping):
                return value
            else:
                return AttrDict(value)
        # We add FileNotFoundError since __getattr__ is expected to only raise
        # AttributeError. Without handling FileNotFoundError, unittests.mocks
        # fails during mock creation on Python3.9
        except (KeyError, FileNotFoundError):
            raise AttributeError(_missing_attr_error_message(key))

    def __getitem__(self, key: str) -> Any:
        """Return the value with the given key. If no such key
        exists, raise a KeyError.

        Thread-safe.
        """
        try:
            value = self._parse(True)[key]
            if not isinstance(value, Mapping):
                return value
            else:
                return AttrDict(value)
        except KeyError:
            raise KeyError(_missing_key_error_message(key))

    def __repr__(self) -> str:
        # If the runtime is NOT initialized, it is a method call outside
        # the streamlit app, so we avoid reading the secrets file as it may not exist.
        # If the runtime is initialized, display the contents of the file and
        # the file must already exist.
        """A string representation of the contents of the dict. Thread-safe."""
        if not runtime.exists():
            return f"{self.__class__.__name__}(file_paths={self._file_paths!r})"
        return repr(self._parse(True))

    def __len__(self) -> int:
        """The number of entries in the dict. Thread-safe."""
        return len(self._parse(True))

    def has_key(self, k: str) -> bool:
        """True if the given key is in the dict. Thread-safe."""
        return k in self._parse(True)

    def keys(self) -> KeysView[str]:
        """A view of the keys in the dict. Thread-safe."""
        return self._parse(True).keys()

    def values(self) -> ValuesView[Any]:
        """A view of the values in the dict. Thread-safe."""
        return self._parse(True).values()

    def items(self) -> ItemsView[str, Any]:
        """A view of the key-value items in the dict. Thread-safe."""
        return self._parse(True).items()

    def __contains__(self, key: Any) -> bool:
        """True if the given key is in the dict. Thread-safe."""
        return key in self._parse(True)

    def __iter__(self) -> Iterator[str]:
        """An iterator over the keys in the dict. Thread-safe."""
        return iter(self._parse(True))


secrets_singleton: Final = Secrets(SECRETS_FILE_LOCS)

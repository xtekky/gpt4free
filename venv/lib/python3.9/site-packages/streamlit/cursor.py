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

from typing import Any, List, Optional, Tuple

from streamlit import util
from streamlit.runtime.scriptrunner import get_script_run_ctx


def make_delta_path(
    root_container: int, parent_path: Tuple[int, ...], index: int
) -> List[int]:
    delta_path = [root_container]
    delta_path.extend(parent_path)
    delta_path.append(index)
    return delta_path


def get_container_cursor(
    root_container: Optional[int],
) -> Optional["RunningCursor"]:
    """Return the top-level RunningCursor for the given container.
    This is the cursor that is used when user code calls something like
    `st.foo` (which uses the main container) or `st.sidebar.foo` (which uses
    the sidebar container).
    """
    if root_container is None:
        return None

    ctx = get_script_run_ctx()

    if ctx is None:
        return None

    if root_container in ctx.cursors:
        return ctx.cursors[root_container]

    cursor = RunningCursor(root_container=root_container)
    ctx.cursors[root_container] = cursor
    return cursor


class Cursor:
    """A pointer to a delta location in the app.

    When adding an element to the app, you should always call
    get_locked_cursor() on that element's respective Cursor.
    """

    def __repr__(self) -> str:
        return util.repr_(self)

    @property
    def root_container(self) -> int:
        """The top-level container this cursor lives within - either
        RootContainer.MAIN or RootContainer.SIDEBAR.
        """
        raise NotImplementedError()

    @property
    def parent_path(self) -> Tuple[int, ...]:
        """The cursor's parent's path within its container."""
        raise NotImplementedError()

    @property
    def index(self) -> int:
        """The index of the Delta within its parent block."""
        raise NotImplementedError()

    @property
    def delta_path(self) -> List[int]:
        """The complete path of the delta pointed to by this cursor - its
        container, parent path, and index.
        """
        return make_delta_path(self.root_container, self.parent_path, self.index)

    @property
    def is_locked(self) -> bool:
        raise NotImplementedError()

    def get_locked_cursor(self, **props) -> "LockedCursor":
        raise NotImplementedError()

    @property
    def props(self) -> Any:
        """Other data in this cursor. This is a temporary measure that will go
        away when we implement improved return values for elements.

        This is only implemented in LockedCursor.
        """
        raise NotImplementedError()


class RunningCursor(Cursor):
    def __init__(self, root_container: int, parent_path: Tuple[int, ...] = ()):
        """A moving pointer to a delta location in the app.

        RunningCursors auto-increment to the next available location when you
        call get_locked_cursor() on them.

        Parameters
        ----------
        root_container: int
            The root container this cursor lives in.
        parent_path: tuple of ints
          The full path of this cursor, consisting of the IDs of all ancestors.
          The 0th item is the topmost ancestor.

        """
        self._root_container = root_container
        self._parent_path = parent_path
        self._index = 0

    @property
    def root_container(self) -> int:
        return self._root_container

    @property
    def parent_path(self) -> Tuple[int, ...]:
        return self._parent_path

    @property
    def index(self) -> int:
        return self._index

    @property
    def is_locked(self) -> bool:
        return False

    def get_locked_cursor(self, **props) -> "LockedCursor":
        locked_cursor = LockedCursor(
            root_container=self._root_container,
            parent_path=self._parent_path,
            index=self._index,
            **props,
        )

        self._index += 1

        return locked_cursor


class LockedCursor(Cursor):
    def __init__(
        self,
        root_container: int,
        parent_path: Tuple[int, ...] = (),
        index: int = 0,
        **props,
    ):
        """A locked pointer to a location in the app.

        LockedCursors always point to the same location, even when you call
        get_locked_cursor() on them.

        Parameters
        ----------
        root_container: int
            The root container this cursor lives in.
        parent_path: tuple of ints
          The full path of this cursor, consisting of the IDs of all ancestors. The
          0th item is the topmost ancestor.
        index: int
        **props: any
          Anything else you want to store in this cursor. This is a temporary
          measure that will go away when we implement improved return values
          for elements.

        """
        self._root_container = root_container
        self._index = index
        self._parent_path = parent_path
        self._props = props

    @property
    def root_container(self) -> int:
        return self._root_container

    @property
    def parent_path(self) -> Tuple[int, ...]:
        return self._parent_path

    @property
    def index(self) -> int:
        return self._index

    @property
    def is_locked(self) -> bool:
        return True

    def get_locked_cursor(self, **props) -> "LockedCursor":
        self._props = props
        return self

    @property
    def props(self) -> Any:
        return self._props

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

import io
import threading
from typing import Dict, List, NamedTuple, Tuple

from blinker import Signal

from streamlit import util
from streamlit.logger import get_logger
from streamlit.runtime.stats import CacheStat, CacheStatsProvider

LOGGER = get_logger(__name__)


class UploadedFileRec(NamedTuple):
    """Metadata and raw bytes for an uploaded file. Immutable."""

    id: int
    name: str
    type: str
    data: bytes


class UploadedFile(io.BytesIO):
    """A mutable uploaded file.

    This class extends BytesIO, which has copy-on-write semantics when
    initialized with `bytes`.
    """

    def __init__(self, record: UploadedFileRec):
        # BytesIO's copy-on-write semantics doesn't seem to be mentioned in
        # the Python docs - possibly because it's a CPython-only optimization
        # and not guaranteed to be in other Python runtimes. But it's detailed
        # here: https://hg.python.org/cpython/rev/79a5fbe2c78f
        super(UploadedFile, self).__init__(record.data)
        self.id = record.id
        self.name = record.name
        self.type = record.type
        self.size = len(record.data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UploadedFile):
            return NotImplemented
        return self.id == other.id

    def __repr__(self) -> str:
        return util.repr_(self)


class UploadedFileManager(CacheStatsProvider):
    """Holds files uploaded by users of the running Streamlit app,
    and emits an event signal when a file is added.

    This class can be used safely from multiple threads simultaneously.
    """

    def __init__(self):
        # List of files for a given widget in a given session.
        self._files_by_id: Dict[Tuple[str, str], List[UploadedFileRec]] = {}

        # A counter that generates unique file IDs. Each file ID is greater
        # than the previous ID, which means we can use IDs to compare files
        # by age.
        self._file_id_counter = 1
        self._file_id_lock = threading.Lock()

        # Prevents concurrent access to the _files_by_id dict.
        # In remove_session_files(), we iterate over the dict's keys. It's
        # an error to mutate a dict while iterating; this lock prevents that.
        self._files_lock = threading.Lock()
        self.on_files_updated = Signal(
            doc="""Emitted when a file list is added to the manager or updated.

            Parameters
            ----------
            session_id : str
                The session_id for the session whose files were updated.
            """
        )

    def __repr__(self) -> str:
        return util.repr_(self)

    def add_file(
        self,
        session_id: str,
        widget_id: str,
        file: UploadedFileRec,
    ) -> UploadedFileRec:
        """Add a file to the FileManager, and return a new UploadedFileRec
        with its ID assigned.

        The "on_files_updated" Signal will be emitted.

        Safe to call from any thread.

        Parameters
        ----------
        session_id
            The ID of the session that owns the file.
        widget_id
            The widget ID of the FileUploader that created the file.
        file
            The file to add.

        Returns
        -------
        UploadedFileRec
            The added file, which has its unique ID assigned.
        """
        files_by_widget = session_id, widget_id

        # Assign the file a unique ID
        file_id = self._get_next_file_id()
        file = UploadedFileRec(
            id=file_id, name=file.name, type=file.type, data=file.data
        )

        with self._files_lock:
            file_list = self._files_by_id.get(files_by_widget, None)
            if file_list is not None:
                file_list.append(file)
            else:
                self._files_by_id[files_by_widget] = [file]

        self.on_files_updated.send(session_id)
        return file

    def get_all_files(self, session_id: str, widget_id: str) -> List[UploadedFileRec]:
        """Return all the files stored for the given widget.

        Safe to call from any thread.

        Parameters
        ----------
        session_id
            The ID of the session that owns the files.
        widget_id
            The widget ID of the FileUploader that created the files.
        """
        file_list_id = (session_id, widget_id)
        with self._files_lock:
            return self._files_by_id.get(file_list_id, []).copy()

    def get_files(
        self, session_id: str, widget_id: str, file_ids: List[int]
    ) -> List[UploadedFileRec]:
        """Return the files with the given widget_id and file_ids.

        Safe to call from any thread.

        Parameters
        ----------
        session_id
            The ID of the session that owns the files.
        widget_id
            The widget ID of the FileUploader that created the files.
        file_ids
            List of file IDs. Only files whose IDs are in this list will be
            returned.
        """
        return [
            f for f in self.get_all_files(session_id, widget_id) if f.id in file_ids
        ]

    def remove_orphaned_files(
        self,
        session_id: str,
        widget_id: str,
        newest_file_id: int,
        active_file_ids: List[int],
    ) -> None:
        """Remove 'orphaned' files: files that have been uploaded and
        subsequently deleted, but haven't yet been removed from memory.

        Because FileUploader can live inside forms, file deletion is made a
        bit tricky: a file deletion should only happen after the form is
        submitted.

        FileUploader's widget value is an array of numbers that has two parts:
        - The first number is always 'this.state.newestServerFileId'.
        - The remaining 0 or more numbers are the file IDs of all the
          uploader's uploaded files.

        When the server receives the widget value, it deletes "orphaned"
        uploaded files. An orphaned file is any file associated with a given
        FileUploader whose file ID is not in the active_file_ids, and whose
        ID is <= `newestServerFileId`.

        This logic ensures that a FileUploader within a form doesn't have any
        of its "unsubmitted" uploads prematurely deleted when the script is
        re-run.

        Safe to call from any thread.
        """
        file_list_id = (session_id, widget_id)
        with self._files_lock:
            file_list = self._files_by_id.get(file_list_id)
            if file_list is None:
                return

            # Remove orphaned files from the list:
            # - `f.id in active_file_ids`:
            #   File is currently tracked by the widget. DON'T remove.
            # - `f.id > newest_file_id`:
            #   file was uploaded *after* the widget  was most recently
            #   updated. (It's probably in a form.) DON'T remove.
            # - `f.id < newest_file_id and f.id not in active_file_ids`:
            #   File is not currently tracked by the widget, and was uploaded
            #   *before* this most recent update. This means it's been deleted
            #   by the user on the frontend, and is now "orphaned". Remove!
            new_list = [
                f for f in file_list if f.id > newest_file_id or f.id in active_file_ids
            ]
            self._files_by_id[file_list_id] = new_list
            num_removed = len(file_list) - len(new_list)

        if num_removed > 0:
            LOGGER.debug("Removed %s orphaned files" % num_removed)

    def remove_file(self, session_id: str, widget_id: str, file_id: int) -> bool:
        """Remove the file list with the given ID, if it exists.

        The "on_files_updated" Signal will be emitted.

        Safe to call from any thread.

        Returns
        -------
        bool
            True if the file was removed, or False if no such file exists.
        """
        file_list_id = (session_id, widget_id)
        with self._files_lock:
            file_list = self._files_by_id.get(file_list_id, None)
            if file_list is None:
                return False

            # Remove the file from its list.
            new_file_list = [file for file in file_list if file.id != file_id]
            self._files_by_id[file_list_id] = new_file_list

        self.on_files_updated.send(session_id)
        return True

    def _remove_files(self, session_id: str, widget_id: str) -> None:
        """Remove the file list for the provided widget in the
        provided session, if it exists.

        Does not emit any signals.

        Safe to call from any thread.
        """
        files_by_widget = session_id, widget_id
        with self._files_lock:
            self._files_by_id.pop(files_by_widget, None)

    def remove_files(self, session_id: str, widget_id: str) -> None:
        """Remove the file list for the provided widget in the
        provided session, if it exists.

        The "on_files_updated" Signal will be emitted.

        Safe to call from any thread.

        Parameters
        ----------
        session_id : str
            The ID of the session that owns the files.
        widget_id : str
            The widget ID of the FileUploader that created the files.
        """
        self._remove_files(session_id, widget_id)
        self.on_files_updated.send(session_id)

    def remove_session_files(self, session_id: str) -> None:
        """Remove all files that belong to the given session.

        Safe to call from any thread.

        Parameters
        ----------
        session_id : str
            The ID of the session whose files we're removing.

        """
        # Copy the keys into a list, because we'll be mutating the dictionary.
        with self._files_lock:
            all_ids = list(self._files_by_id.keys())

        for files_id in all_ids:
            if files_id[0] == session_id:
                self.remove_files(*files_id)

    def _get_next_file_id(self) -> int:
        """Return the next file ID and increment our ID counter."""
        with self._file_id_lock:
            file_id = self._file_id_counter
            self._file_id_counter += 1
            return file_id

    def get_stats(self) -> List[CacheStat]:
        """Return the manager's CacheStats.

        Safe to call from any thread.
        """
        with self._files_lock:
            # Flatten all files into a single list
            all_files: List[UploadedFileRec] = []
            for file_list in self._files_by_id.values():
                all_files.extend(file_list)

        return [
            CacheStat(
                category_name="UploadedFileManager",
                cache_name="",
                byte_length=len(file.data),
            )
            for file in all_files
        ]

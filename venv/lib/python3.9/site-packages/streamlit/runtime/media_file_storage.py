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

from abc import abstractmethod
from enum import Enum
from typing import Optional, Union

from typing_extensions import Protocol


class MediaFileKind(Enum):
    # st.image, st.video, st.audio files
    MEDIA = "media"

    # st.download_button files
    DOWNLOADABLE = "downloadable"


class MediaFileStorageError(Exception):
    """Exception class for errors raised by MediaFileStorage.

    When running in "development mode", the full text of these errors
    is displayed in the frontend, so errors should be human-readable
    (and actionable).

    When running in "release mode", errors are redacted on the
    frontend; we instead show a generic "Something went wrong!" message.
    """


class MediaFileStorage(Protocol):
    @abstractmethod
    def load_and_get_id(
        self,
        path_or_data: Union[str, bytes],
        mimetype: str,
        kind: MediaFileKind,
        filename: Optional[str] = None,
    ) -> str:
        """Load the given file path or bytes into the manager and return
        an ID that uniquely identifies it.

        It’s an error to pass a URL to this function. (Media stored at
        external URLs can be served directly to the Streamlit frontend;
        there’s no need to store this data in MediaFileStorage.)

        Parameters
        ----------
        path_or_data
            A path to a file, or the file's raw data as bytes.

        mimetype
            The media’s mimetype. Used to set the Content-Type header when
            serving the media over HTTP.

        kind
            The kind of file this is: either MEDIA, or DOWNLOADABLE.

        filename : str or None
            Optional filename. Used to set the filename in the response header.

        Returns
        -------
        str
            The unique ID of the media file.

        Raises
        ------
        MediaFileStorageError
            Raised if the media can't be loaded (for example, if a file
            path is invalid).

        """
        raise NotImplementedError

    @abstractmethod
    def get_url(self, file_id: str) -> str:
        """Return a URL for a file in the manager.

        Parameters
        ----------
        file_id
            The file's ID, returned from load_media_and_get_id().

        Returns
        -------
        str
            A URL that the frontend can load the file from. Because this
            URL may expire, it should not be cached!

        Raises
        ------
        MediaFileStorageError
            Raised if the manager doesn't contain an object with the given ID.

        """
        raise NotImplementedError

    @abstractmethod
    def delete_file(self, file_id: str) -> None:
        """Delete a file from the manager.

        This should be called when a given file is no longer referenced
        by any connected client, so that the MediaFileStorage can free its
        resources.

        Calling `delete_file` on a file_id that doesn't exist is allowed,
        and is a no-op. (This means that multiple `delete_file` calls with
        the same file_id is not an error.)

        Note: implementations can choose to ignore `delete_file` calls -
        this function is a *suggestion*, not a *command*. Callers should
        not rely on file deletion happening immediately (or at all).

        Parameters
        ----------
        file_id
            The file's ID, returned from load_media_and_get_id().

        Returns
        -------
        None

        Raises
        ------
        MediaFileStorageError
            Raised if file deletion fails for any reason. Note that these
            failures will generally not be shown on the frontend (file
            deletion usually occurs on session disconnect).

        """
        raise NotImplementedError

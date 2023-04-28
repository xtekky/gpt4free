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

"""A bunch of useful utilities for the watcher.

These are functions that only make sense within the watcher. In particular,
functions that use streamlit.config can go here to avoid a dependency cycle.
"""

import hashlib
import os
import time
from pathlib import Path
from typing import Optional

# How many times to try to grab the MD5 hash.
_MAX_RETRIES = 5

# How long to wait between retries.
_RETRY_WAIT_SECS = 0.1


def calc_md5_with_blocking_retries(
    path: str,
    *,  # keyword-only arguments:
    glob_pattern: Optional[str] = None,
    allow_nonexistent: bool = False,
) -> str:
    """Calculate the MD5 checksum of a given path.

    For a file, this means calculating the md5 of the file's contents. For a
    directory, we concatenate the directory's path with the names of all the
    files in it and calculate the md5 of that.

    IMPORTANT: This method calls time.sleep(), which blocks execution. So you
    should only use this outside the main thread.
    """

    if allow_nonexistent and not os.path.exists(path):
        content = path.encode("UTF-8")
    elif os.path.isdir(path):
        glob_pattern = glob_pattern or "*"
        content = _stable_dir_identifier(path, glob_pattern).encode("UTF-8")
    else:
        content = _get_file_content_with_blocking_retries(path)

    md5 = hashlib.md5()
    md5.update(content)

    # Use hexdigest() instead of digest(), so it's easier to debug.
    return md5.hexdigest()


def path_modification_time(path: str, allow_nonexistent: bool = False) -> float:
    """Return the modification time of a path (file or directory).

    If allow_nonexistent is True and the path does not exist, we return 0.0 to
    guarantee that any file/dir later created at the path has a later
    modification time than the last time returned by this function for that
    path.

    If allow_nonexistent is False and no file/dir exists at the path, a
    FileNotFoundError is raised (by os.stat).

    For any path that does correspond to an existing file/dir, we return its
    modification time.
    """
    if allow_nonexistent and not os.path.exists(path):
        return 0.0
    return os.stat(path).st_mtime


def _get_file_content_with_blocking_retries(file_path: str) -> bytes:
    content = b""
    # There's a race condition where sometimes file_path no longer exists when
    # we try to read it (since the file is in the process of being written).
    # So here we retry a few times using this loop. See issue #186.
    for i in range(_MAX_RETRIES):
        try:
            with open(file_path, "rb") as f:
                content = f.read()
                break
        except FileNotFoundError as e:
            if i >= _MAX_RETRIES - 1:
                raise e
            time.sleep(_RETRY_WAIT_SECS)
    return content


def _dirfiles(dir_path: str, glob_pattern: str) -> str:
    p = Path(dir_path)
    filenames = sorted(
        [f.name for f in p.glob(glob_pattern) if not f.name.startswith(".")]
    )
    return "+".join(filenames)


def _stable_dir_identifier(dir_path: str, glob_pattern: str) -> str:
    """Wait for the files in a directory to look stable-ish before returning an id.

    We do this to deal with problems that would otherwise arise from many tools
    (e.g. git) and editors (e.g. vim) "editing" files (from the user's
    perspective) by doing some combination of deleting, creating, and moving
    various files under the hood.

    Because of this, we're unable to rely on FileSystemEvents that we receive
    from watchdog to determine when a file has been added to or removed from a
    directory.

    This is a bit of an unfortunate situation, but the approach we take here is
    most likely fine as:
      * The worst thing that can happen taking this approach is a false
        positive page added/removed notification, which isn't too disastrous
        and can just be ignored.
      * It is impossible (that is, I'm fairly certain that the problem is
        undecidable) to know whether a file created/deleted/moved event
        corresponds to a legitimate file creation/deletion/move or is part of
        some sequence of events that results in what the user sees as a file
        "edit".
    """
    dirfiles = _dirfiles(dir_path, glob_pattern)

    for _ in range(_MAX_RETRIES):
        time.sleep(_RETRY_WAIT_SECS)

        new_dirfiles = _dirfiles(dir_path, glob_pattern)
        if dirfiles == new_dirfiles:
            break

        dirfiles = new_dirfiles

    return f"{dir_path}+{dirfiles}"

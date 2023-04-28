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

import re
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, cast

from blinker import Signal

from streamlit.logger import get_logger
from streamlit.string_util import extract_leading_emoji
from streamlit.util import calc_md5

LOGGER = get_logger(__name__)


def open_python_file(filename):
    """Open a read-only Python file taking proper care of its encoding.

    In Python 3, we would like all files to be opened with utf-8 encoding.
    However, some author like to specify PEP263 headers in their source files
    with their own encodings. In that case, we should respect the author's
    encoding.
    """
    import tokenize

    if hasattr(tokenize, "open"):  # Added in Python 3.2
        # Open file respecting PEP263 encoding. If no encoding header is
        # found, opens as utf-8.
        return tokenize.open(filename)
    else:
        return open(filename, "r", encoding="utf-8")


PAGE_FILENAME_REGEX = re.compile(r"([0-9]*)[_ -]*(.*)\.py")


def page_sort_key(script_path: Path) -> Tuple[float, str]:
    matches = re.findall(PAGE_FILENAME_REGEX, script_path.name)

    # Failing this assert should only be possible if script_path isn't a Python
    # file, which should never happen.
    assert len(matches) > 0, f"{script_path} is not a Python file"

    [(number, label)] = matches
    label = label.lower()

    if number == "":
        return (float("inf"), label)

    return (float(number), label)


def page_icon_and_name(script_path: Path) -> Tuple[str, str]:
    """Compute the icon and name of a page from its script path.

    This is *almost* the page name displayed in the nav UI, but it has
    underscores instead of spaces. The reason we do this is because having
    spaces in URLs both looks bad and is hard to deal with due to the need to
    URL-encode them. To solve this, we only swap the underscores for spaces
    right before we render page names.
    """
    extraction = re.search(PAGE_FILENAME_REGEX, script_path.name)
    if extraction is None:
        return "", ""

    # This cast to Any+type annotation weirdness is done because
    # cast(re.Match[str], ...) explodes at runtime since Python interprets it
    # as an attempt to index into re.Match instead of as a type annotation.
    extraction: re.Match[str] = cast(Any, extraction)

    icon_and_name = re.sub(
        r"[_ ]+", "_", extraction.group(2)
    ).strip() or extraction.group(1)

    return extract_leading_emoji(icon_and_name)


_pages_cache_lock = threading.RLock()
_cached_pages: Optional[Dict[str, Dict[str, str]]] = None
_on_pages_changed = Signal(doc="Emitted when the pages directory is changed")


def invalidate_pages_cache():
    global _cached_pages

    LOGGER.debug("Pages directory changed")
    with _pages_cache_lock:
        _cached_pages = None

    _on_pages_changed.send()


def get_pages(main_script_path_str: str) -> Dict[str, Dict[str, str]]:
    global _cached_pages

    # Avoid taking the lock if the pages cache hasn't been invalidated.
    pages = _cached_pages
    if pages is not None:
        return pages

    with _pages_cache_lock:
        # The cache may have been repopulated while we were waiting to grab
        # the lock.
        if _cached_pages is not None:
            return _cached_pages

        main_script_path = Path(main_script_path_str)
        main_page_icon, main_page_name = page_icon_and_name(main_script_path)
        main_page_script_hash = calc_md5(main_script_path_str)

        # NOTE: We include the page_script_hash in the dict even though it is
        #       already used as the key because that occasionally makes things
        #       easier for us when we need to iterate over pages.
        pages = {
            main_page_script_hash: {
                "page_script_hash": main_page_script_hash,
                "page_name": main_page_name,
                "icon": main_page_icon,
                "script_path": str(main_script_path.resolve()),
            }
        }

        pages_dir = main_script_path.parent / "pages"
        page_scripts = sorted(
            [
                f
                for f in pages_dir.glob("*.py")
                if not f.name.startswith(".") and not f.name == "__init__.py"
            ],
            key=page_sort_key,
        )

        for script_path in page_scripts:
            script_path_str = str(script_path.resolve())
            pi, pn = page_icon_and_name(script_path)
            psh = calc_md5(script_path_str)

            pages[psh] = {
                "page_script_hash": psh,
                "page_name": pn,
                "icon": pi,
                "script_path": script_path_str,
            }

        _cached_pages = pages

        return pages


def register_pages_changed_callback(
    callback: Callable[[str], None],
):
    def disconnect():
        _on_pages_changed.disconnect(callback)

    # weak=False so that we have control of when the pages changed
    # callback is deregistered.
    _on_pages_changed.connect(callback, weak=False)

    return disconnect

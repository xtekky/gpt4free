#!/usr/bin/env python3
"""Upload files to AnonDrop from the command line.

AnonDrop is an anonymous file drop service.  Its public API is documented at
<https://anondrop.io/info#api>.  This module implements the parts that are
useful from a shell: direct and chunked uploads, remote URL fetching, listing
and deleting files, and account registration.

Typical use::

    bin/anondrop file.zip
    bin/anondrop drop file.zip --chunked
    bin/anondrop remote https://example.com/file.zip
    bin/anondrop list
    bin/anondrop delete https://anondrop.io/abcdef

Only the Python standard library is used.  The account key is read from
``--key``, ``$ANONDROP_KEY`` or ``$XDG_CONFIG_HOME/anondrop/key`` (by default
``~/.config/anondrop/key``).  Uploads work without an account; chunked uploads,
``remote``, ``list`` and ``delete`` need one.
"""

from __future__ import annotations

import argparse
import http.client
import json
import mimetypes
import os
import re
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

DEFAULT_BASE_URL = (
    os.environ.get("ANONDROP_URL") or "https://anondrop.io"
).rstrip("/")
CHUNK_SIZE = 9 * 1024 * 1024
KEY_ENV = "ANONDROP_KEY"
KEY_FILE_ENV = "ANONDROP_KEY_FILE"
USER_AGENT = "gpt4free-anondrop/1.0 (+https://g4f.dev)"

_LINK_RE = re.compile(r"https?://[^\s'\"<>]+/[A-Za-z0-9]+")
_FILE_ID_RE = re.compile(r"^[A-Za-z0-9]+$")
_REGISTER_RE = re.compile(r"userkey['\"]\s*,\s*['\"]([A-Za-z0-9]+)")
_RESERVED_IDS = {
    "register",
    "files",
    "upload",
    "uploadchunk",
    "initiateupload",
    "endupload",
    "remoteuploadurl",
    "delete",
    "edit",
    "editform",
    "dashboard",
    "info",
}


class AnonDropError(RuntimeError):
    """Raised when AnonDrop rejects a request or returns an unusable answer."""


@dataclass
class UploadResult:
    """The outcome of a single upload."""

    name: str
    size: int
    url: str
    chunks: int = 1
    account_key: Optional[str] = None
    direct_url: Optional[str] = None


ProgressCallback = Callable[[int, int], None]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def default_key_path() -> Path:
    """Return the default location of the stored account key."""

    override = os.environ.get(KEY_FILE_ENV)
    if override:
        return Path(override).expanduser()
    config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    return base / "anondrop" / "key"


def load_key(path: Optional[Path] = None) -> Optional[str]:
    """Read an account key from disk, or return ``None`` when absent."""

    key_path = Path(path).expanduser() if path else default_key_path()
    try:
        key = key_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return key or None


def save_key(key: str, path: Optional[Path] = None) -> Path:
    """Persist an account key with owner-only permissions."""

    key = key.strip()
    if not key:
        raise AnonDropError("Refusing to save an empty account key.")
    key_path = Path(path).expanduser() if path else default_key_path()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        key_path,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        stat.S_IRUSR | stat.S_IWUSR,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(key + "\n")
    try:
        key_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    return key_path


def resolve_key(explicit: Optional[str] = None, path: Optional[Path] = None) -> Optional[str]:
    """Pick the account key: explicit value, environment, then config file."""

    if explicit and explicit.strip():
        return explicit.strip()
    from_env = os.environ.get(KEY_ENV)
    if from_env and from_env.strip():
        return from_env.strip()
    return load_key(path)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _normalise_base(base_url: str) -> str:
    return (base_url or DEFAULT_BASE_URL).rstrip("/") or DEFAULT_BASE_URL


class _MultipartBody:
    """A multipart/form-data body that streams a file instead of buffering it."""

    def __init__(
        self,
        prefix: bytes,
        fileobj: BinaryIO,
        suffix: bytes,
        file_size: int,
    ) -> None:
        self._prefix = prefix
        self._file = fileobj
        self._suffix = suffix
        self._size = len(prefix) + file_size + len(suffix)
        self._mode = "prefix"

    def __len__(self) -> int:
        return self._size

    def read(self, size: int = -1) -> bytes:
        if self._mode == "prefix":
            self._mode = "file"
            return self._prefix
        if self._mode == "file":
            if size is None or size < 0:
                chunk = self._file.read()
            else:
                chunk = self._file.read(size)
            if chunk:
                return chunk
            self._mode = "suffix"
        if self._mode == "suffix":
            self._mode = "done"
            return self._suffix
        return b""


def _multipart_file_body(path: Path) -> Tuple[_MultipartBody, str, BinaryIO]:
    boundary = "----g4f-anondrop-" + uuid.uuid4().hex
    filename = path.name.replace("\\", "_").replace('"', "_")
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    prefix = (
        "--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: {content_type}\r\n\r\n"
    ).format(
        boundary=boundary,
        filename=filename,
        content_type=content_type,
    ).encode("utf-8")
    suffix = "\r\n--{boundary}--\r\n".format(boundary=boundary).encode("ascii")
    fileobj = path.open("rb")
    return _MultipartBody(prefix, fileobj, suffix, path.stat().st_size), (
        "multipart/form-data; boundary=" + boundary
    ), fileobj


def _http(
    method: str,
    url: str,
    data: object = None,
    headers: Optional[Mapping[str, str]] = None,
    timeout: float = 300.0,
) -> Tuple[bytes, Mapping[str, str]]:
    """Perform an HTTP request and return ``(body, headers)``."""

    all_headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    if headers:
        all_headers.update(headers)
    request = urllib.request.Request(
        url,
        data=data,
        headers=all_headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            return body, dict(response.headers.items())
    except urllib.error.HTTPError as exc:  # pragma: no cover - network
        detail = exc.read().decode("utf-8", "replace").strip()
        raise AnonDropError(
            "HTTP {code} from {url}: {detail}".format(
                code=exc.code,
                url=url,
                detail=detail[:300] or exc.reason,
            )
        ) from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network
        raise AnonDropError("Could not reach {url}: {reason}".format(
            url=url, reason=exc.reason
        )) from exc
    except OSError as exc:  # pragma: no cover - network
        raise AnonDropError("Request to {url} failed: {exc}".format(
            url=url, exc=exc
        )) from exc


def extract_link(text: str) -> str:
    """Extract the share link from an AnonDrop upload response."""

    for match in reversed(_LINK_RE.findall(text or "")):
        identifier = match.rstrip("/").rsplit("/", 1)[-1]
        if _FILE_ID_RE.match(identifier) and identifier.lower() not in _RESERVED_IDS:
            return match
    snippet = " ".join((text or "").split())[:200]
    raise AnonDropError(
        "AnonDrop returned no file link"
        + (": " + snippet if snippet else ".")
    )


def extract_file_id(value: str) -> str:
    """Accept a file id, a share link or a direct link and return the id."""

    return _link_parts(value)[0]


def _link_parts(value: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Split an id, share link or direct link into ``(id, filename, origin)``."""

    candidate = (value or "").strip()
    parsed = urllib.parse.urlparse(candidate)
    filename: Optional[str] = None
    origin: Optional[str] = None
    if parsed.scheme in ("http", "https"):
        segments = [segment for segment in parsed.path.split("/") if segment]
        if not segments:
            raise AnonDropError("Not an AnonDrop file id or link: {0}".format(value))
        identifier = urllib.parse.unquote(segments[0])
        if len(segments) >= 2:
            filename = urllib.parse.unquote(segments[1])
        origin = "{0}://{1}".format(parsed.scheme, parsed.netloc)
    elif "/" in candidate or "?" in candidate or "#" in candidate:
        raise AnonDropError("Not an AnonDrop file id or link: {0}".format(value))
    else:
        identifier = candidate
    if not _FILE_ID_RE.match(identifier) or identifier.lower() in _RESERVED_IDS:
        raise AnonDropError("Not an AnonDrop file id or link: {0}".format(value))
    return identifier, filename, origin


def _clean_filename(name: str) -> str:
    """Reduce a name to a single URL path segment."""

    candidate = (name or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    return candidate


def _download_url(url: str) -> str:
    separator = "&" if "?" in url else "?"
    return url + separator + "download=true"


def build_direct_url(
    file_id: str,
    filename: str,
    base_url: Optional[str] = None,
    download: bool = False,
) -> str:
    """Build the direct file URL ``/FILE_ID/FILENAME`` for a share link."""

    identifier = extract_file_id(file_id)
    name = _clean_filename(filename)
    if not name:
        raise AnonDropError("A filename is required to build a direct link.")
    url = "{0}/{1}/{2}".format(
        _normalise_base(base_url),
        identifier,
        urllib.parse.quote(name, safe=""),
    )
    return _download_url(url) if download else url


def _fetch_filename(
    link_or_id: str,
    identifier: str,
    base_url: Optional[str],
    timeout: float,
) -> str:
    """Read the stored filename from an existing share page."""

    origin = _link_parts(link_or_id)[2] or _normalise_base(base_url)
    page_url = "{0}/{1}".format(origin, identifier)
    body, _ = _http("GET", page_url, timeout=timeout)
    text = body.decode("utf-8", "replace")
    pattern = re.compile(r"/" + re.escape(identifier) + r"/([^\"'<>?#]+)")
    match = pattern.search(text)
    if not match:
        raise AnonDropError(
            "Could not determine the file name for {0}; pass --filename.".format(
                identifier
            )
        )
    return urllib.parse.unquote(match.group(1).strip())


def resolve_direct_link(
    link_or_id: str,
    filename: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 60.0,
    download: bool = False,
) -> str:
    """Return the direct download URL for a share link or file id."""

    identifier, link_filename, origin = _link_parts(link_or_id)
    if filename is None:
        filename = link_filename
    if filename is None:
        filename = _fetch_filename(link_or_id, identifier, base_url, timeout)
    return build_direct_url(
        identifier,
        filename,
        base_url=base_url or origin,
        download=download,
    )


def _account_key(headers: Mapping[str, str]) -> Optional[str]:
    for name, value in headers.items():
        if name.lower() == "x-account-key" and value:
            return value.strip()
    return None


# ---------------------------------------------------------------------------
# API operations
# ---------------------------------------------------------------------------


def direct_upload(
    path: Path,
    key: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 300.0,
) -> UploadResult:
    """Upload a file in a single multipart POST to ``/upload``."""

    path = Path(path)
    if not path.is_file():
        raise AnonDropError("No such file: {0}".format(path))
    base = _normalise_base(base_url)
    url = base + "/upload"
    if key:
        url += "?key=" + urllib.parse.quote(key, safe="")
    body, content_type, fileobj = _multipart_file_body(path)
    try:
        response, headers = _http(
            "POST",
            url,
            data=body,
            headers={
                "Content-Type": content_type,
                "Content-Length": str(len(body)),
            },
            timeout=timeout,
        )
    finally:
        fileobj.close()
    link = extract_link(response.decode("utf-8", "replace"))
    return UploadResult(
        name=path.name,
        size=path.stat().st_size,
        url=link,
        chunks=1,
        account_key=_account_key(headers),
        direct_url=build_direct_url(link, path.name, base_url=base),
    )


def _initiate_upload(
    filename: str,
    key: str,
    base_url: str,
    timeout: float,
) -> str:
    url = (
        _normalise_base(base_url)
        + "/initiateupload?filename="
        + urllib.parse.quote(filename, safe="")
        + "&key="
        + urllib.parse.quote(key, safe="")
    )
    body, _ = _http("GET", url, timeout=timeout)
    session = body.decode("utf-8", "replace").strip()
    if not session or session.lower().startswith("invalid"):
        raise AnonDropError("AnonDrop did not start the upload: " + (session[:200] or "empty response"))
    return session


def _upload_chunk(
    session: str,
    index: int,
    data: bytes,
    base_url: str,
    timeout: float,
    retries: int = 5,
) -> None:
    url = (
        _normalise_base(base_url)
        + "/uploadchunk?session_hash="
        + urllib.parse.quote(session, safe="")
        + "&chunk_index={0}&size={1}".format(index, len(data))
    )
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            _http(
                "POST",
                url,
                data=data,
                headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(len(data)),
                },
                timeout=timeout,
            )
            return
        except AnonDropError as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(0.25 * attempt)
    raise AnonDropError(
        "Chunk {0} failed after {1} attempts: {2}".format(index, retries, last_error)
    )


def chunked_upload(
    path: Path,
    key: str,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 300.0,
    chunk_size: int = CHUNK_SIZE,
    progress: Optional[ProgressCallback] = None,
) -> UploadResult:
    """Upload a file in resumable chunks (``initiateupload``/``uploadchunk``/``endupload``)."""

    path = Path(path)
    if not path.is_file():
        raise AnonDropError("No such file: {0}".format(path))
    if not key:
        raise AnonDropError(
            "Chunked uploads need an account key. Run `bin/anondrop register` "
            "or set ${0}.".format(KEY_ENV)
        )
    if chunk_size <= 0:
        raise AnonDropError("Chunk size must be positive.")
    size = path.stat().st_size
    total_chunks = max(1, (size + chunk_size - 1) // chunk_size)
    base = _normalise_base(base_url)
    session = _initiate_upload(path.name, key, base, timeout)
    sent = 0
    with path.open("rb") as fileobj:
        for index in range(total_chunks):
            data = fileobj.read(chunk_size)
            _upload_chunk(session, index, data, base, timeout)
            sent += len(data)
            if progress:
                progress(sent, size)
    end_url = (
        base
        + "/endupload?session_hash="
        + urllib.parse.quote(session, safe="")
        + "&size={0}".format(size)
    )
    body, headers = _http("GET", end_url, timeout=timeout)
    link = extract_link(body.decode("utf-8", "replace"))
    return UploadResult(
        name=path.name,
        size=size,
        url=link,
        chunks=total_chunks,
        account_key=_account_key(headers),
        direct_url=build_direct_url(link, path.name, base_url=base),
    )


def upload(
    path: Path,
    key: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 300.0,
    chunked: bool = False,
    chunk_size: int = CHUNK_SIZE,
    progress: Optional[ProgressCallback] = None,
) -> UploadResult:
    """Upload a local file, using the chunked API when requested."""

    if chunked:
        if not key:
            raise AnonDropError(
                "Chunked uploads need an account key. Run `bin/anondrop register` "
                "or set ${0}.".format(KEY_ENV)
            )
        return chunked_upload(
            path,
            key,
            base_url=base_url,
            timeout=timeout,
            chunk_size=chunk_size,
            progress=progress,
        )
    return direct_upload(path, key=key, base_url=base_url, timeout=timeout)


def remote_upload(
    url: str,
    key: str,
    base_url: str = DEFAULT_BASE_URL,
    filename: Optional[str] = None,
    timeout: float = 300.0,
) -> UploadResult:
    """Ask AnonDrop to fetch a URL and store the result."""

    if not key:
        raise AnonDropError(
            "Remote uploads need an account key. Run `bin/anondrop register` "
            "or set ${0}.".format(KEY_ENV)
        )
    query = "url=" + urllib.parse.quote(url, safe="")
    if filename:
        query += "&filename=" + urllib.parse.quote(filename, safe="")
    request_url = (
        _normalise_base(base_url)
        + "/remoteuploadurl?key="
        + urllib.parse.quote(key, safe="")
        + "&"
        + query
    )
    request = urllib.request.Request(
        request_url,
        headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
    )
    chunks: List[bytes] = []
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            while True:
                try:
                    chunk = response.read(65536)
                except http.client.IncompleteRead as exc:
                    # AnonDrop streams progress lines and closes the chunked
                    # response without a terminating chunk.
                    chunks.append(exc.partial)
                    break
                if not chunk:
                    break
                chunks.append(chunk)
    except urllib.error.HTTPError as exc:  # pragma: no cover - network
        detail = exc.read().decode("utf-8", "replace").strip()
        raise AnonDropError("Remote upload failed (HTTP {0}): {1}".format(
            exc.code, detail[:300] or exc.reason
        )) from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network
        raise AnonDropError("Could not reach {0}: {1}".format(url, exc.reason)) from exc
    except (OSError, http.client.HTTPException) as exc:  # pragma: no cover - network
        if not chunks:
            raise AnonDropError("Remote upload to {0} failed: {1}".format(url, exc)) from exc
    link = extract_link(b"".join(chunks).decode("utf-8", "replace"))
    identifier = extract_file_id(link)
    resolved_name = _clean_filename(filename) if filename else None
    direct_url = None
    if resolved_name:
        direct_url = build_direct_url(identifier, resolved_name, base_url=base_url)
    else:
        try:
            resolved_name = _fetch_filename(link, identifier, base_url, timeout)
            direct_url = build_direct_url(identifier, resolved_name, base_url=base_url)
        except AnonDropError:
            resolved_name = None
    return UploadResult(
        name=resolved_name or identifier,
        size=0,
        url=link,
        direct_url=direct_url,
    )


def list_files(
    key: str,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 60.0,
) -> Dict[str, object]:
    """Return the account's file listing as decoded JSON."""

    if not key:
        raise AnonDropError(
            "Listing files needs an account key. Run `bin/anondrop register` "
            "or set ${0}.".format(KEY_ENV)
        )
    url = _normalise_base(base_url) + "/files?key=" + urllib.parse.quote(key, safe="")
    body, _ = _http("GET", url, timeout=timeout)
    try:
        payload = json.loads(body.decode("utf-8", "replace"))
    except ValueError as exc:
        raise AnonDropError("AnonDrop returned invalid JSON for /files") from exc
    if not isinstance(payload, dict):
        raise AnonDropError("Unexpected /files response from AnonDrop")
    return payload


def delete_file(
    file_id: str,
    key: str,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 60.0,
) -> Tuple[str, bool]:
    """Delete one file and return ``(server message, success)``."""

    if not key:
        raise AnonDropError(
            "Deleting files needs an account key. Run `bin/anondrop register` "
            "or set ${0}.".format(KEY_ENV)
        )
    identifier = extract_file_id(file_id)
    url = (
        _normalise_base(base_url)
        + "/delete/"
        + urllib.parse.quote(identifier, safe="")
        + "?key="
        + urllib.parse.quote(key, safe="")
    )
    body, _ = _http("POST", url, data=b"", timeout=timeout)
    message = body.decode("utf-8", "replace").strip()
    return message, message.lower().startswith("deleted")


def register(base_url: str = DEFAULT_BASE_URL, timeout: float = 60.0) -> str:
    """Create a new anonymous account and return its key."""

    url = _normalise_base(base_url) + "/register"
    body, _ = _http("GET", url, timeout=timeout)
    text = body.decode("utf-8", "replace")
    match = _REGISTER_RE.search(text)
    if not match:
        raise AnonDropError("AnonDrop did not return a new account key")
    return match.group(1)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


COMMANDS = (
    "drop",
    "upload",
    "remote",
    "list",
    "ls",
    "direct",
    "download",
    "delete",
    "register",
)


def _add_common(parser: argparse.ArgumentParser, suppress: bool = False) -> None:
    default = argparse.SUPPRESS if suppress else None
    parser.add_argument(
        "--base-url",
        default=default,
        metavar="URL",
        help="AnonDrop base URL (default: %s)" % DEFAULT_BASE_URL,
    )
    parser.add_argument(
        "--key",
        default=default,
        metavar="KEY",
        help="account key (default: $%s or %s)" % (KEY_ENV, default_key_path()),
    )
    parser.add_argument(
        "--json",
        default=argparse.SUPPRESS if suppress else False,
        action="store_true",
        help="print machine-readable JSON",
    )
    parser.add_argument(
        "--timeout",
        default=argparse.SUPPRESS if suppress else 300.0,
        type=float,
        metavar="SECONDS",
        help="network timeout per request (default: 300)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anondrop",
        description="Upload files to AnonDrop (https://anondrop.io) from the shell.",
    )
    _add_common(parser)
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    drop = sub.add_parser(
        "drop",
        aliases=["upload"],
        help="upload one or more local files",
        parents=[_common_parent()],
    )
    drop.add_argument("files", nargs="+", type=Path, help="files to upload")
    drop.add_argument(
        "--chunked",
        action="store_true",
        help="use resumable chunked upload (requires an account key)",
    )
    drop.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        metavar="BYTES",
        help="chunk size for --chunked (default: 9 MiB)",
    )
    drop.add_argument(
        "--save-key",
        action="store_true",
        help="save the account key returned by AnonDrop",
    )
    drop.add_argument(
        "--no-progress",
        action="store_true",
        help="do not show upload progress",
    )
    _add_link_output(drop)

    remote = sub.add_parser(
        "remote",
        help="ask AnonDrop to fetch a URL (requires an account key)",
        parents=[_common_parent()],
    )
    remote.add_argument("url", help="URL to fetch")
    remote.add_argument("--filename", help="override the stored file name")
    _add_link_output(remote)

    listing = sub.add_parser(
        "list",
        aliases=["ls"],
        help="list files of the configured account",
        parents=[_common_parent()],
    )
    _add_link_output(listing)

    direct = sub.add_parser(
        "direct",
        aliases=["download"],
        help="turn a share link or file id into a direct download link",
        parents=[_common_parent()],
    )
    direct.add_argument("link", help="share link or file id")
    direct.add_argument(
        "--filename",
        help="stored file name (read from the share page when omitted)",
    )
    direct.add_argument(
        "--download",
        action="store_true",
        help="append ?download=true so browsers download the file",
    )

    delete = sub.add_parser(
        "delete",
        help="delete files or share links from the configured account",
        parents=[_common_parent()],
    )
    delete.add_argument("files", nargs="+", help="file ids or share links")

    reg = sub.add_parser(
        "register",
        help="create a new anonymous account and store its key",
        parents=[_common_parent()],
    )
    reg.add_argument(
        "--no-save",
        action="store_true",
        help="print the key without writing it to the config file",
    )
    reg.add_argument(
        "--key-file",
        type=Path,
        help="write the key to this file instead of the default location",
    )
    return parser


def _common_parent() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    _add_common(parent, suppress=True)
    return parent


def _add_link_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--direct",
        action="store_true",
        help="print direct file links instead of share pages",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="append ?download=true to direct links (implies --direct)",
    )


def _print_progress(name: str, sent: int, total: int) -> None:
    if not sys.stderr.isatty():
        return
    percent = 100.0 if total <= 0 else min(100.0, sent * 100.0 / total)
    sys.stderr.write(
        "\r{name}: {sent}/{total} bytes ({percent:.0f}%)".format(
            name=name,
            sent=sent,
            total=total,
            percent=percent,
        )
    )
    if sent >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def _result_dict(result: UploadResult) -> Dict[str, object]:
    direct = result.direct_url
    return {
        "name": result.name,
        "size": result.size,
        "url": result.url,
        "chunks": result.chunks,
        "direct_url": direct,
        "download_url": _download_url(direct) if direct else None,
    }


def _output_link(result: UploadResult, args: argparse.Namespace) -> str:
    """Choose the share page or direct link to print for one result."""

    if not (getattr(args, "direct", False) or getattr(args, "download", False)):
        return result.url
    link = result.direct_url
    if link is None:
        return resolve_direct_link(
            result.url,
            base_url=args.base_url,
            timeout=args.timeout,
            download=getattr(args, "download", False),
        )
    if getattr(args, "download", False):
        link = _download_url(link)
    return link


def _run_drop(args: argparse.Namespace, key: Optional[str]) -> int:
    results: List[UploadResult] = []
    failures = 0
    for path in args.files:
        progress = None
        if args.chunked and not args.no_progress:
            progress = lambda sent, total, name=path.name: _print_progress(name, sent, total)
        try:
            result = upload(
                path,
                key=key,
                base_url=args.base_url,
                timeout=args.timeout,
                chunked=args.chunked,
                chunk_size=args.chunk_size,
                progress=progress,
            )
        except (AnonDropError, OSError) as exc:
            failures += 1
            sys.stderr.write("anondrop: {0}: {1}\n".format(path, exc))
            continue
        results.append(result)
        if result.account_key and not key and args.save_key:
            stored = save_key(result.account_key)
            sys.stderr.write("anondrop: saved account key to {0}\n".format(stored))
        elif result.account_key and not key:
            sys.stderr.write(
                "anondrop: AnonDrop issued account key {0}; use --save-key to keep it\n".format(
                    result.account_key
                )
            )
    if args.json:
        print(json.dumps([_result_dict(result) for result in results], indent=2))
    else:
        for result in results:
            try:
                print(_output_link(result, args))
            except AnonDropError as exc:
                failures += 1
                sys.stderr.write("anondrop: {0}: {1}\n".format(result.name, exc))
    return 1 if failures else 0


def _run_remote(args: argparse.Namespace, key: Optional[str]) -> int:
    result = remote_upload(
        args.url,
        key or "",
        base_url=args.base_url,
        filename=args.filename,
        timeout=args.timeout,
    )
    if args.json:
        print(json.dumps(_result_dict(result), indent=2))
    else:
        print(_output_link(result, args))
    return 0


def _run_list(args: argparse.Namespace, key: Optional[str]) -> int:
    payload = list_files(key or "", base_url=args.base_url, timeout=args.timeout)
    files = payload.get("files") or []
    pages = payload.get("pages") or []
    items = [item for item in list(files) + list(pages) if isinstance(item, dict)]
    if args.direct or args.download:
        rows = []
        for item in items:
            identifier = item.get("id")
            name = item.get("name")
            if not identifier or not name:
                continue
            try:
                url = build_direct_url(
                    str(identifier),
                    str(name),
                    base_url=args.base_url,
                    download=args.download,
                )
            except AnonDropError:
                continue
            rows.append({"id": str(identifier), "name": str(name), "direct_url": url})
        if args.json:
            print(json.dumps(rows, indent=2))
        elif not rows:
            print("No files.")
        else:
            for row in rows:
                print(row["direct_url"])
        return 0
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    if not items:
        print("No files.")
        return 0
    for item in items:
        print("{0}\t{1}".format(item.get("id", "?"), item.get("name", "")))
    return 0


def _run_direct(args: argparse.Namespace) -> int:
    link = resolve_direct_link(
        args.link,
        filename=args.filename,
        base_url=args.base_url,
        timeout=args.timeout,
        download=args.download,
    )
    if args.json:
        identifier = extract_file_id(args.link)
        parsed = urllib.parse.urlparse(link)
        name = urllib.parse.unquote(parsed.path.rstrip("/").rsplit("/", 1)[-1])
        print(json.dumps(
            {"id": identifier, "filename": name, "direct_url": link},
            indent=2,
        ))
    else:
        print(link)
    return 0


def _run_delete(args: argparse.Namespace, key: Optional[str]) -> int:
    failures = 0
    outcomes: List[Dict[str, object]] = []
    for value in args.files:
        try:
            message, deleted = delete_file(
                value,
                key or "",
                base_url=args.base_url,
                timeout=args.timeout,
            )
        except AnonDropError as exc:
            failures += 1
            sys.stderr.write("anondrop: {0}: {1}\n".format(value, exc))
            continue
        outcomes.append({"file": value, "message": message, "deleted": deleted})
        if not deleted:
            failures += 1
            sys.stderr.write("anondrop: {0}: {1}\n".format(value, message or "not deleted"))
    if args.json:
        print(json.dumps(outcomes, indent=2))
    else:
        for outcome in outcomes:
            print("{0}: {1}".format(outcome["file"], outcome["message"] or outcome["deleted"]))
    return 1 if failures else 0


def _run_register(args: argparse.Namespace) -> int:
    key = register(base_url=args.base_url, timeout=args.timeout)
    saved: Optional[Path] = None
    if not args.no_save:
        saved = save_key(key, path=args.key_file)
    if args.json:
        print(json.dumps({"key": key, "saved": str(saved) if saved else None}, indent=2))
    else:
        print(key)
        if saved:
            sys.stderr.write("anondrop: saved account key to {0}\n".format(saved))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and not raw[0].startswith("-") and raw[0] not in COMMANDS:
        raw.insert(0, "drop")
    parser = build_parser()
    args = parser.parse_args(raw)
    if not args.command:
        parser.print_help()
        return 2

    key = resolve_key(getattr(args, "key", None))
    try:
        if args.command in ("drop", "upload"):
            return _run_drop(args, key)
        if args.command == "remote":
            return _run_remote(args, key)
        if args.command in ("list", "ls"):
            return _run_list(args, key)
        if args.command in ("direct", "download"):
            return _run_direct(args)
        if args.command == "delete":
            return _run_delete(args, key)
        if args.command == "register":
            return _run_register(args)
    except AnonDropError as exc:
        sys.stderr.write("anondrop: {0}\n".format(exc))
        return 1
    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

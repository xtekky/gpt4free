"""Helpers for reading HAR (HTTP Archive) files without binary payloads.

HAR captures embed request and response bodies, often as base64 encoded
images, fonts or media. Those blobs are useless for text based consumers
(LLM context, logs, cookie extraction) and can be several megabytes each,
so they are replaced by a short placeholder whenever they can be detected.

The same consumers never look at browser bookkeeping such as ``_initiator``,
``_priority``, timings or cache metadata, so those keys are dropped as well
to keep the captures small.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import shutil
from typing import Any, Optional

# MIME types whose bodies are never useful as text.
BINARY_MIME_PREFIXES = (
    "image/",
    "audio/",
    "video/",
    "font/",
    "application/octet-stream",
    "application/pdf",
    "application/zip",
    "application/gzip",
    "application/wasm",
    "application/x-font",
    "application/x-protobuf",
    "application/vnd.ms-fontobject",
)

# A base64 body is a single long token without whitespace.
_BASE64_RE = re.compile(r"[A-Za-z0-9+/]+={0,2}")
_BASE64_MIN_LENGTH = 512
# Real base64 payloads use a wide alphabet; repetitive text does not.
_BASE64_MIN_DISTINCT_CHARS = 8

# Browser bookkeeping that no consumer of a HAR file needs. The ``_`` prefixed
# keys are Chrome DevTools extensions (initiator stacks, priorities, resource
# types, transfer sizes) and are often the largest part of an entry.
_HAR_ENTRY_KEYS = frozenset({
    "cache",
    "comment",
    "connection",
    "pageref",
    "serverIPAddress",
    "timings",
    "_blockedQueue",
    "_connect",
    "_dns",
    "_edge",
    "_error",
    "_fetchType",
    "_fromCache",
    "_initiator",
    "_priority",
    "_receive",
    "_resourceType",
    "_securityDetails",
    "_send",
    "_ssl",
    "_transferSize",
    "_wait",
    "_wasBlocked",
    "_webSocketMessages",
})

# ``queryString`` duplicates the URL query and the size fields are either -1
# or trivially recomputable from the body.
_HAR_REQUEST_KEYS = frozenset({
    "bodySize",
    "comment",
    "headersSize",
    "httpVersion",
    "queryString",
})

_HAR_RESPONSE_KEYS = frozenset({
    "bodySize",
    "comment",
    "headersSize",
    "httpVersion",
    "redirectURL",
    "statusText",
})

_HAR_LOG_KEYS = frozenset({"comment", "pages"})


def is_binary_mime_type(mime_type: Optional[str]) -> bool:
    """Return True when *mime_type* denotes a non-textual payload."""
    if not mime_type:
        return False
    mime_type = mime_type.split(";", 1)[0].strip().lower()
    if mime_type.startswith("text/"):
        return False
    if mime_type.endswith("+json") or mime_type.endswith("+xml"):
        return False
    return mime_type.startswith(BINARY_MIME_PREFIXES)


def looks_like_base64(text: str) -> bool:
    """Heuristically detect a base64 encoded body."""
    if len(text) < _BASE64_MIN_LENGTH:
        return False
    if len(set(text)) < _BASE64_MIN_DISTINCT_CHARS:
        return False
    return _BASE64_RE.fullmatch(text) is not None

def decode_base64_text(text: str) -> Optional[str]:
    """Return the decoded text when *text* is base64 of UTF-8 text, else None."""
    try:
        decoded = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError):
        return None
    try:
        return decoded.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _placeholder(content: dict, size: int) -> str:
    mime_type = content.get("mimeType") or "unknown"
    return f"<binary data removed: {mime_type}, {size} bytes>"


def _strip_content(content: Any) -> bool:
    """Sanitize a binary ``content`` body in place. Returns True when changed."""
    if not isinstance(content, dict):
        return False
    text = content.get("text")
    if not isinstance(text, str) or not text:
        return False
    binary_mime = is_binary_mime_type(content.get("mimeType"))
    if content.get("encoding") == "base64" and not binary_mime:
        decoded = decode_base64_text(text)
        if decoded is not None:
            content["text"] = decoded
            content.pop("encoding", None)
            return True
    if not binary_mime and content.get("encoding") != "base64" and not looks_like_base64(text):
        return False
    size = content.get("size")
    if not isinstance(size, int) or size <= 0:
        size = len(text)
    content["text"] = _placeholder(content, size)
    content.pop("encoding", None)
    return True


def _prune_keys(mapping: Any, keys: frozenset) -> int:
    """Drop *keys* from *mapping* in place, returning the number removed."""
    if not isinstance(mapping, dict):
        return 0
    removed = 0
    for key in keys:
        if key in mapping:
            del mapping[key]
            removed += 1
    return removed

def _prune_har(har: Any) -> int:
    """Drop bookkeeping fields from a parsed HAR document in place."""
    if not isinstance(har, dict):
        return 0
    log = har.get("log")
    if not isinstance(log, dict):
        return 0
    changed = _prune_keys(log, _HAR_LOG_KEYS)
    entries = log.get("entries")
    if not isinstance(entries, list):
        return changed
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        changed += _prune_keys(entry, _HAR_ENTRY_KEYS)
        changed += _prune_keys(entry.get("request"), _HAR_REQUEST_KEYS)
        changed += _prune_keys(entry.get("response"), _HAR_RESPONSE_KEYS)
    return changed

def _strip_har_binary(har: Any) -> int:
    """Sanitize a parsed HAR document in place, returning the number of changes."""
    if not isinstance(har, dict):
        return 0
    entries = har.get("log", {}).get("entries")
    if not isinstance(entries, list):
        return 0
    changed = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        response = entry.get("response")
        if isinstance(response, dict):
            changed += _strip_content(response.get("content"))
        request = entry.get("request")
        if isinstance(request, dict):
            changed += _strip_content(request.get("postData"))
    return changed

def strip_har_binary(har: Any) -> Any:
    """Remove binary bodies and bookkeeping fields from a parsed HAR document.

    The document is modified in place. Only bodies that are clearly binary
    are replaced: base64 encoded content, binary MIME types and long base64
    looking bodies. Text bodies, headers, cookies and request parameters are
    left untouched, while unused metadata (timings, cache, ``_initiator``,
    ``_priority``, sizes, ...) is dropped.
    """
    _strip_har_binary(har)
    _prune_har(har)
    return har


def sanitize_har_text(text: str) -> str:
    """Strip binary bodies and bookkeeping fields from HAR *text*.

    Returns the original text when it is not a HAR document.
    """
    try:
        har = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return text
    if not isinstance(har, dict):
        return text
    strip_har_binary(har)
    try:
        return json.dumps(har, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return text


def _backup_path(path) -> str:
    """Return the one-time backup path for a HAR file (``x.har`` -> ``x.har.orig``).

    The ``.orig`` suffix is deliberately not ``.har`` so the backup is not
    picked up again by the ``*.har`` scans in the cookie directories.
    """
    return f"{path}.orig"

def _write_har(path, har: dict) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(har, file, ensure_ascii=False, indent=2)

def load_har(path, save: bool = True) -> dict:
    """Read a HAR file and strip binary bodies and bookkeeping fields from it.

    When anything was removed and *save* is true, the sanitized document is
    written back to *path* and the untouched original is kept once as
    ``<name>.har.orig``. The modification time is preserved so mtime based
    ordering of the cookie directory does not change. Failures to write are
    ignored, so reading a read-only capture still works.
    """
    with open(path, "rb") as file:
        har = json.load(file)
    changed = _strip_har_binary(har) + _prune_har(har)
    if not changed or not save:
        return har
    try:
        stat = os.stat(path)
        backup = _backup_path(path)
        if not os.path.exists(backup):
            shutil.copy2(path, backup)
        _write_har(path, har)
        os.utime(path, (stat.st_atime, stat.st_mtime))
    except OSError:
        pass
    return har
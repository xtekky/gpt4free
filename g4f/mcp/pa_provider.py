"""PA Provider - .pa.py file parser and executor for custom providers

This module provides:

1. Safe Python code execution with whitelisted modules and restricted built-ins.
2. ``.pa.py`` file loading — parse and execute provider-adapter files that define
   custom gpt4free providers.
3. Workspace management at ``~/.g4f/workspace``.

A ``.pa.py`` file is a plain Python file that is executed inside a sandbox.
Inside that sandbox the code may only import from the *whitelisted* module set
and may only access the file-system through a workspace-scoped ``open()``.

Security model
--------------
The sandbox mitigates the following vectors:

* **Arbitrary module imports** — only modules in :data:`SAFE_MODULES` may be
  imported.  The built-in ``__import__`` is replaced with a wrapper that raises
  ``ImportError`` for any top-level name not in the allowlist.  Relative imports
  are unconditionally blocked.
* **Filesystem escape** — ``open()`` is replaced with a workspace-scoped version
  that resolves symlinks and checks that the canonical path starts with the
  workspace root.  Direct ``os``/``pathlib`` access is blocked because those
  modules are not in the allowlist.
* **Code injection** — ``exec``, ``eval``, ``compile``, and ``input`` are removed
  from the sandbox built-ins so code in the sandbox cannot spawn secondary
  execution contexts.
* **Execution timeout** — code runs in a dedicated thread; if it does not
  complete within :data:`MAX_EXEC_TIMEOUT` seconds the result is returned with
  an error and the thread is abandoned.
* **Runaway recursion** — ``sys.setrecursionlimit`` is reduced to
  :data:`MAX_RECURSION_DEPTH` for the duration of the sandboxed call.
* **Output flooding** — stdout and stderr are each capped at
  :data:`MAX_OUTPUT_BYTES`; excess output is silently truncated.

Typical layout of a ``.pa.py`` file::

    from aiohttp import ClientSession
    from g4f.Provider.base_provider import AsyncGeneratorProvider, ProviderModelMixin
    from g4f.Provider.helper import format_prompt
    from g4f.typing import AsyncResult, Messages

    class Provider(AsyncGeneratorProvider, ProviderModelMixin):
        label = "MyCustomProvider"
        url   = "https://example.com"
        working = True
        default_model = "gpt-4"
        models = ["gpt-4", "gpt-3.5-turbo"]

        @classmethod
        async def create_async_generator(cls, model, messages, **kwargs):
            ...
            yield chunk
"""

from __future__ import annotations

import io
import os as _os
import sys
import json
import re
import hashlib
import threading
import time as _time_module
import traceback
import types
import builtins as _builtins
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Type
from .. import debug
from ..files import secure_filename
from ..config import CONFIG_DIR

# ---------------------------------------------------------------------------
# Workspace directory
# ---------------------------------------------------------------------------

#: Private directory for secret data (outside the shared workspace).
SECRET_DIR: Path = CONFIG_DIR / "secret"


def get_workspace_dir() -> Path:
    """Return the workspace directory ``~/.g4f/workspace``, creating it if needed."""
    workspace = Path.home() / ".g4f" / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def _migrate_secret_dir(old_secret_ws: Path, new_secret_ws: Path) -> None:
    """Move any existing secret files from the workspace into the private dir."""
    if not old_secret_ws.exists():
        return
    new_secret_ws.mkdir(parents=True, exist_ok=True)
    for item in old_secret_ws.iterdir():
        target = new_secret_ws / item.name
        if target.exists():
            continue
        item.rename(target)
    # Remove the old empty directory tree
    try:
        old_secret_ws.rmdir()
        # Also remove the now-empty "secret" parent if it's empty
        old_secret_parent = get_workspace_dir() / "secret"
        if old_secret_parent.exists() and not any(old_secret_parent.iterdir()):
            old_secret_parent.rmdir()
    except OSError:
        pass  # Directory not empty — leave it


def get_user_workspace_dir(user_id: str) -> Path:
    """Return a per-user workspace subdirectory ``~/.g4f/workspace/users/<user_id>``.

    The directory is created on demand.  ``user_id`` is sanitised so that
    only alphanumeric characters, ``-`` and ``_`` are kept, preventing path
    traversal.
    """
    if not user_id:
        return get_workspace_dir()
    safe_id = re.sub(r"[^a-zA-Z0-9_\-]+", "_", user_id).strip("_") or "anonymous"
    user_workspace = get_workspace_dir() / "users" / safe_id
    user_workspace.mkdir(parents=True, exist_ok=True)
    return user_workspace


def get_secret_workspace_dir(user_id: str) -> Path:
    """Return a per-user *secret* directory ``~/.g4f/secret/<user_id>``.

    This is used when a workspace secret is provided.  Files are saved here
    when the user is logged in; reads fall back to the root workspace when
    the file does not exist in the secret workspace.

    Secrets are stored **outside** the shared workspace (under
    ``~/.g4f/secret/``) so they are never synced or exposed alongside
    workspace files — similar to how cookies live under ``~/.g4f/cookies/``.
    """
    if not user_id:
        return get_workspace_dir()
    safe_id = re.sub(r"[^a-zA-Z0-9_\-]+", "_", user_id).strip("_") or "anonymous"
    secret_workspace = SECRET_DIR / safe_id
    # Migrate any existing data from the old workspace-relative location
    old_location = get_workspace_dir() / "secret" / safe_id
    _migrate_secret_dir(old_location, secret_workspace)
    secret_workspace.mkdir(parents=True, exist_ok=True)
    return secret_workspace


def resolve_workspace_path(
    rel_path: str,
    user_id: str = None,
    workspace_secret: str = None,
    for_write: bool = False,
) -> Tuple[Path, Path]:
    """Resolve a relative path to an actual filesystem path with fallback.

    When *workspace_secret* and *user_id* are provided, writes go to the
    user's secret workspace and reads first check the secret workspace, then
    fall back to the root workspace.

    Returns a tuple ``(target, workspace_root)`` where *target* is the
    resolved path to use and *workspace_root* is the workspace root that
    contains it (used for containment checks).
    """
    root = get_workspace_dir().resolve()
    if workspace_secret and user_id:
        user_ws = get_secret_workspace_dir(user_id).resolve()
        target = (user_ws / rel_path).resolve()
        if for_write:
            return target, user_ws
        # Read: check secret workspace first, fall back to root
        if target.exists():
            return target, user_ws
        # Fall back to root workspace
        target = (root / rel_path).resolve()
        return target, root
    target = (root / rel_path).resolve()
    return target, root


def is_hidden_file(path: str) -> bool:
    """Return True if *path* is a hidden file (starts with a dot)."""
    return any(part.startswith(".") or part.startswith("__") for part in str(path).replace("\\", "/").split("/"))


# ---------------------------------------------------------------------------
# Secret conversation storage
# ---------------------------------------------------------------------------

def _derive_key(workspace_secret: str) -> bytes:
    """Derive a 32-byte AES key from the workspace secret via SHA-256."""
    return hashlib.sha256(workspace_secret.encode("utf-8")).digest()


def _encrypt_data(data: bytes, workspace_secret: str) -> bytes:
    """Encrypt *data* with AES-256-GCM using *workspace_secret*.

    Returns a binary blob: ``nonce (12) || ciphertext || tag (16)``.
    The nonce is randomly generated for each encryption.
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    import os

    key = _derive_key(workspace_secret)
    nonce = os.urandom(12)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, data, None)
    return nonce + ciphertext


def _decrypt_data(blob: bytes, workspace_secret: str) -> Optional[bytes]:
    """Decrypt a blob produced by :func:`_encrypt_data`.

    Returns ``None`` if decryption fails (wrong key / corrupted data).
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if len(blob) < 28:  # 12-byte nonce + 16-byte GCM tag minimum
        return None
    key = _derive_key(workspace_secret)
    nonce = blob[:12]
    ciphertext = blob[12:]
    aesgcm = AESGCM(key)
    try:
        return aesgcm.decrypt(nonce, ciphertext, None)
    except Exception:
        return None


def _encrypt_json(obj: dict, workspace_secret: str) -> bytes:
    """Serialise *obj* to JSON and encrypt with AES-256-GCM."""
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return _encrypt_data(raw, workspace_secret)


def _decrypt_json(blob: bytes, workspace_secret: str) -> Optional[dict]:
    """Decrypt a blob and parse the plaintext as JSON."""
    raw = _decrypt_data(blob, workspace_secret)
    if raw is None:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def _is_encrypted_blob(data: bytes) -> bool:
    """Heuristic: check if *data* looks like an encrypted binary blob.

    Encrypted files start with the magic prefix ``G4FENC`` followed by
    a version byte.  Plaintext JSON files start with ``{``.
    """
    return data[:6] == b"G4FENC"


def _encrypt_json_file(obj: dict, workspace_secret: str) -> bytes:
    """Encrypt *obj* as JSON with a magic header for easy identification.

    Format: ``G4FENC (6) || version (1) || nonce (12) || ciphertext || tag``
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    import os

    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    key = _derive_key(workspace_secret)
    nonce = os.urandom(12)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, raw, None)
    return b"G4FENC" + b"\x01" + nonce + ciphertext


def _decrypt_json_file(data: bytes, workspace_secret: str) -> Optional[dict]:
    """Decrypt and parse a file produced by :func:`_encrypt_json_file`.

    Falls back to plaintext JSON parsing if the data is not encrypted
    (for backward compatibility with previously stored plaintext files).
    """
    if not _is_encrypted_blob(data):
        # Plaintext fallback — old files stored before encryption
        try:
            return json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    version = data[6]
    if version != 1:
        return None
    nonce = data[7:19]
    ciphertext = data[19:]
    key = _derive_key(workspace_secret)
    aesgcm = AESGCM(key)
    try:
        raw = aesgcm.decrypt(nonce, ciphertext, None)
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def get_secret_conversation_dir(user_id: str) -> Path:
    """Return the directory used to store secret conversations for *user_id*.

    Conversations are saved as individual JSON files under
    ``~/.g4f/secret/<user_id>/conversations/``.  An index file
    ``index.json`` lists all stored conversation IDs.
    """
    secret_ws = get_secret_workspace_dir(user_id)
    conv_dir = secret_ws / "conversations"
    conv_dir.mkdir(parents=True, exist_ok=True)
    return conv_dir


def save_secret_conversation(user_id: str, conversation: dict, workspace_secret: str = None) -> dict:
    """Save a single conversation to the user's secret workspace.

    *conversation* must contain an ``id`` field.  The conversation is
    written as ``<id>.json`` (encrypted with AES-256-GCM when
    *workspace_secret* is provided) and the index file is updated.
    """
    conv_id = conversation.get("id")
    if not conv_id:
        return {"error": "Conversation must have an 'id' field"}
    conv_dir = get_secret_conversation_dir(user_id)
    safe_id = secure_filename(str(conv_id))
    conv_file = conv_dir / f"{safe_id}.json"
    if workspace_secret:
        blob = _encrypt_json_file(conversation, workspace_secret)
        conv_file.write_bytes(blob)
    else:
        conv_file.write_text(json.dumps(conversation, ensure_ascii=False, indent=2), encoding="utf-8")
    _update_secret_conversation_index(user_id, conversation)
    return {"saved": True, "id": conv_id, "path": str(conv_file.name), "encrypted": bool(workspace_secret)}


def _update_secret_conversation_index(user_id: str, conversation: dict) -> None:
    """Update the index file with a summary of *conversation*."""
    conv_dir = get_secret_conversation_dir(user_id)
    index_file = conv_dir / "index.json"
    index: list = []
    if index_file.exists():
        try:
            index = json.loads(index_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            index = []
    conv_id = conversation.get("id")
    # Remove existing entry for this conversation
    index = [e for e in index if e.get("id") != conv_id]
    # Add fresh entry
    entry = {
        "id": conv_id,
        "title": conversation.get("title") or conversation.get("new_title") or "",
        "updated": conversation.get("updated"),
        "added": conversation.get("added"),
        "items_count": len(conversation.get("items", [])),
    }
    index.append(entry)
    # Sort by updated descending
    index.sort(key=lambda e: e.get("updated") or e.get("added") or 0, reverse=True)
    index_file.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def list_secret_conversations(user_id: str) -> list:
    """Return the index of all secret conversations for *user_id*."""
    conv_dir = get_secret_conversation_dir(user_id)
    index_file = conv_dir / "index.json"
    if index_file.exists():
        try:
            return json.loads(index_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []


def get_secret_conversation(user_id: str, conv_id: str, workspace_secret: str = None) -> Optional[dict]:
    """Retrieve a single secret conversation by ID.

    If *workspace_secret* is provided, encrypted files are decrypted.
    Plaintext files (stored before encryption was enabled) are read as-is.
    """
    conv_dir = get_secret_conversation_dir(user_id)
    safe_id = secure_filename(str(conv_id))
    conv_file = conv_dir / f"{safe_id}.json"
    if not conv_file.exists():
        return None
    try:
        raw = conv_file.read_bytes()
    except OSError:
        return None
    return _decrypt_json_file(raw, workspace_secret or "")


def delete_secret_conversation(user_id: str, conv_id: str) -> bool:
    """Delete a secret conversation and update the index."""
    conv_dir = get_secret_conversation_dir(user_id)
    safe_id = secure_filename(str(conv_id))
    conv_file = conv_dir / f"{safe_id}.json"
    deleted = False
    if conv_file.exists():
        conv_file.unlink()
        deleted = True
    # Update index
    index_file = conv_dir / "index.json"
    if index_file.exists():
        try:
            index = json.loads(index_file.read_text(encoding="utf-8"))
            index = [e for e in index if e.get("id") != conv_id]
            index_file.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        except (json.JSONDecodeError, OSError):
            pass
    return deleted

# ---------------------------------------------------------------------------
# Cross-device workspace secret sharing
# ---------------------------------------------------------------------------

def _get_secret_requests_dir(user_id: str) -> Path:
    """Return the directory for pending secret-sharing requests.

    Stored under ``~/.g4f/workspace/secret/<user_id>/secret_requests/``.
    """
    secret_ws = get_secret_workspace_dir(user_id)
    req_dir = secret_ws / "secret_requests"
    req_dir.mkdir(parents=True, exist_ok=True)
    return req_dir


def create_secret_request(user_id: str, device_name: str = "") -> dict:
    """Create a pending secret-sharing request from a new device.

    Returns a dict with ``request_id`` and ``status``.  The online device
    polls ``list_secret_requests`` to discover it.
    """
    import uuid

    request_id = uuid.uuid4().hex[:12]
    req_dir = _get_secret_requests_dir(user_id)
    req_file = req_dir / f"{request_id}.json"
    now = _time_module.time()
    request_data = {
        "id": request_id,
        "user_id": user_id,
        "device_name": device_name or "unknown",
        "status": "pending",  # pending -> confirmed -> completed
        "created": now,
        "expires": now + 300,  # 5-minute expiry
        "secret": None,  # filled in by the confirming device
    }
    req_file.write_text(json.dumps(request_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"request_id": request_id, "status": "pending"}


def list_secret_requests(user_id: str) -> list:
    """List all pending secret-sharing requests for *user_id*.

    Expired requests (older than 5 minutes) are automatically removed.
    """
    req_dir = _get_secret_requests_dir(user_id)
    now = _time_module.time()
    requests = []
    for req_file in req_dir.glob("*.json"):
        try:
            data = json.loads(req_file.read_text(encoding="utf-8"))
            if data.get("expires", 0) < now and data.get("status") != "completed":
                req_file.unlink(missing_ok=True)
                continue
            # Don't expose the secret in the list
            safe = {k: v for k, v in data.items() if k != "secret"}
            requests.append(safe)
        except (json.JSONDecodeError, OSError):
            continue
    requests.sort(key=lambda r: r.get("created", 0), reverse=True)
    return requests


def confirm_secret_request(user_id: str, request_id: str, workspace_secret: str) -> dict:
    """Confirm a pending secret request by providing the workspace secret.

    Called by the online device that already has the secret.
    """
    req_dir = _get_secret_requests_dir(user_id)
    safe_id = secure_filename(str(request_id))
    req_file = req_dir / f"{safe_id}.json"
    if not req_file.exists():
        return {"error": "Request not found"}
    try:
        data = json.loads(req_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"error": "Invalid request file"}
    if data.get("status") != "pending":
        return {"error": f"Request is not pending (status={data.get('status')})"}
    data["status"] = "confirmed"
    data["secret"] = workspace_secret
    data["confirmed_at"] = _time_module.time()
    req_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"confirmed": True, "request_id": request_id}


def poll_secret_request(user_id: str, request_id: str) -> dict:
    """Poll a secret request to check if it has been confirmed.

    Returns the request data including the secret if confirmed.
    If the secret has been retrieved, the request is marked as completed
    and cleaned up.
    """
    req_dir = _get_secret_requests_dir(user_id)
    safe_id = secure_filename(str(request_id))
    req_file = req_dir / f"{safe_id}.json"
    if not req_file.exists():
        return {"status": "not_found"}
    try:
        data = json.loads(req_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"status": "error", "error": "Invalid request file"}
    now = _time_module.time()
    if data.get("expires", 0) < now and data.get("status") != "completed":
        req_file.unlink(missing_ok=True)
        return {"status": "expired"}
    if data.get("status") == "confirmed":
        # Mark as completed and clean up
        data["status"] = "completed"
        req_file.unlink(missing_ok=True)
        return {
            "status": "confirmed",
            "secret": data.get("secret"),
            "request_id": request_id,
        }
    return {"status": data.get("status", "pending"), "request_id": request_id}


def delete_secret_request(user_id: str, request_id: str) -> bool:
    """Delete (cancel) a secret-sharing request."""
    req_dir = _get_secret_requests_dir(user_id)
    safe_id = secure_filename(str(request_id))
    req_file = req_dir / f"{safe_id}.json"
    if req_file.exists():
        req_file.unlink()
        return True
    return False


# ---------------------------------------------------------------------------
# Whitelisted modules
# ---------------------------------------------------------------------------

#: Modules that are allowed inside the safe execution sandbox.
SAFE_MODULES: FrozenSet[str] = frozenset(
    {
        "__future__",
        "concurrent",
        "warnings",
        "urllib3",
        "urllib3.exceptions",
        "uuid",
        "secrets",
        # Math / numeric
        "math",
        "cmath",
        "decimal",
        "fractions",
        "statistics",
        "random",
        "numbers",
        # String / text
        "string",
        "re",
        "textwrap",
        "unicodedata",
        "difflib",
        "fnmatch",
        # Data structures
        "json",
        "csv",
        "collections",
        "heapq",
        "bisect",
        "array",
        "queue",
        # Functional
        "itertools",
        "functools",
        "operator",
        # Type system
        "typing",
        "types",
        "abc",
        "dataclasses",
        "enum",
        # Time / date
        "datetime",
        "time",
        "calendar",
        # I/O
        "io",
        # Async
        "asyncio",
        # Encoding / hashing
        "base64",
        "hashlib",
        "hmac",
        "binascii",
        "codecs",
        "struct",
        # URL / HTTP
        "urllib",
        "urllib.parse",
        "http",
        "http.client",
        # Compression
        "gzip",
        "zlib",
        # Misc safe stdlib
        "copy",
        "pprint",
        "reprlib",
        "warnings",
        "contextlib",
        # Third-party HTTP (used by providers)
        "aiohttp",
        "requests",
        # gpt4free itself
        "g4f",
        # wasmtime
        "wasmtime",
        # Restricted os shim (only urandom and safe read-only attrs exposed)
        "os",
    }
)


# ---------------------------------------------------------------------------
# Security limits
# ---------------------------------------------------------------------------

#: Wall-clock seconds allowed for a single :func:`execute_safe_code` call.
MAX_EXEC_TIMEOUT: float = 30.0

#: Maximum Python call-stack depth inside the sandbox (passed to
#: ``sys.setrecursionlimit``).  The default CPython limit is 1 000; using a
#: lower value catches infinite-recursion attacks early.
MAX_RECURSION_DEPTH: int = 500

#: Maximum number of UTF-8 bytes captured from *each* of stdout and stderr.
#: Writes beyond this limit are silently dropped and a truncation notice is
#: appended to stderr.
MAX_OUTPUT_BYTES: int = 65_536  # 64 KiB


# ---------------------------------------------------------------------------
# Sandbox helpers
# ---------------------------------------------------------------------------


class _LimitedStringIO(io.StringIO):
    """StringIO that stops accepting writes once *max_bytes* of UTF-8 content
    have been accumulated.  Additional writes are silently discarded and
    ``truncated`` is set to ``True``."""

    def __init__(self, max_bytes: int = MAX_OUTPUT_BYTES) -> None:
        super().__init__()
        self._max_bytes = max_bytes
        self._bytes_written = 0
        self.truncated = False

    def write(self, s: str) -> int:
        if self._bytes_written >= self._max_bytes:
            self.truncated = True
            return 0
        encoded = s.encode("utf-8", errors="replace")
        remaining = self._max_bytes - self._bytes_written
        if len(encoded) > remaining:
            s = encoded[:remaining].decode("utf-8", errors="replace")
            self.truncated = True
        n = super().write(s)
        self._bytes_written += len(s.encode("utf-8", errors="replace"))
        return n


def _exec_in_thread(
    compiled: Any,
    safe_globals: Dict[str, Any],
    local_vars: Dict[str, Any],
    max_depth: int,
    exc_box: List,
) -> None:
    """Run *compiled* code with a bounded recursion depth.

    ``sys.setrecursionlimit`` is set to *max_depth* for the lifetime of this
    call and restored afterwards.  stdout / stderr capture is handled by
    the custom ``print`` injected into the sandbox builtins — no global
    ``sys.stdout`` redirection is performed so an abandoned timeout thread
    cannot corrupt the caller's output streams.

    Any exception is stored in *exc_box* (a one-element list) so the caller
    can inspect it without needing to join the thread.

    This function is designed to run in a *daemon* thread so that it is
    automatically discarded when the process exits, even if the sandboxed
    code is stuck in an infinite loop.
    """
    prev = sys.getrecursionlimit()
    sys.setrecursionlimit(max_depth)
    try:
        # Use safe_globals as both globals and locals so that top-level
        # imports (e.g. ``from aiohttp import ClientSession``) are stored in
        # the same namespace that class method ``__globals__`` points to.
        # Otherwise imports land only in the locals dict and are invisible
        # to class methods (NameError at call time).
        exec(compiled, safe_globals, safe_globals)  # noqa: S102
    except Exception:  # noqa: BLE001
        exc_box.append(traceback.format_exc())
    finally:
        sys.setrecursionlimit(prev)


def _load_workspace_module(
    name: str,
    workspace: Path,
    globals_dict: Optional[Dict[str, Any]],
    fromlist: tuple = (),
    level: int = 0,
) -> Optional[types.ModuleType]:
    """Try to load *name* as a ``.py`` file from the workspace.

    Searches recursively for ``<name>.py`` (or ``<name>/__init__.py`` for
    packages) anywhere under *workspace*.  If found, executes it inside the
    sandbox and returns the resulting module object.  Returns ``None`` if no
    matching file exists.

    The directory of ``__file__`` in *globals_dict* (if set) is searched
    first so sibling modules are found quickly, then the entire workspace is
    searched recursively as a fallback.

    The loaded module is cached in :data:`sys.modules` so subsequent imports
    return the same object.

    Args:
        name: Top-level module name (e.g. ``"freegpt_wasm_signer"``).
        workspace: Workspace root directory to search.
        globals_dict: Globals dict of the calling frame (used to find
            ``__file__`` for sibling-first lookup).
        fromlist: ``fromlist`` argument from the import statement.
        level: Relative import level (always 0 for absolute imports).
    """
    # Only handle simple top-level names (no dots).
    if "." in name:
        return None

    # Build search directories: __file__ dir first, then workspace root
    search_dirs: List[Path] = []
    if globals_dict:
        cur_file = globals_dict.get("__file__")
        if cur_file:
            search_dirs.append(Path(cur_file).parent)
    search_dirs.append(workspace)

    source_path: Optional[Path] = None
    for d in search_dirs:
        py_file = d / f"{name}.py"
        pkg_init = d / name / "__init__.py"
        if py_file.is_file():
            source_path = py_file
            break
        elif pkg_init.is_file():
            source_path = pkg_init
            break
    else:
        # Recursive fallback: search entire workspace
        for candidate in workspace.rglob(f"{name}.py"):
            # Skip .pa.py files — those are providers, not importable modules
            if not candidate.name.endswith(".pa.py"):
                source_path = candidate
                pkg_init = candidate.parent / "__init__.py"
                break
        if source_path is None:
            for candidate in workspace.rglob(f"{name}/__init__.py"):
                source_path = candidate
                pkg_init = candidate
                break

    if source_path is None:
        return None

    # Return cached module if already loaded
    if name in sys.modules:
        return sys.modules[name]

    # Read and execute the module source in a sandbox
    code = source_path.read_text(encoding="utf-8")
    module = types.ModuleType(name)
    module.__file__ = str(source_path.resolve())
    module.__name__ = name
    if pkg_init.is_file():
        module.__path__ = [str((workspace / name).resolve())]
        module.__package__ = name
    else:
        module.__package__ = ""

    # Build sandbox globals for the module
    module_globals = _make_safe_globals(SAFE_MODULES)
    module_globals["__file__"] = str(source_path.resolve())
    module_globals["__name__"] = name
    module_globals["__package__"] = module.__package__
    module.__dict__.update(module_globals)

    try:
        compiled = compile(code, str(source_path), "exec")
    except SyntaxError:
        raise ImportError(
            f"Syntax error in workspace module '{name}' "
            f"({source_path}):\n{traceback.format_exc()}"
        )

    # Execute in the current thread (no timeout — module loading is expected
    # to be fast and we need the module object synchronously).
    prev_depth = sys.getrecursionlimit()
    sys.setrecursionlimit(MAX_RECURSION_DEPTH)
    try:
        exec(compiled, module.__dict__, module.__dict__)  # noqa: S102
    except Exception:
        raise ImportError(
            f"Failed to load workspace module '{name}' "
            f"({source_path}):\n{traceback.format_exc()}"
        )
    finally:
        sys.setrecursionlimit(prev_depth)

    sys.modules[name] = module
    return module


# ---------------------------------------------------------------------------
# Restricted os shim
# ---------------------------------------------------------------------------


def _make_restricted_os() -> types.ModuleType:
    """Return a restricted ``os`` module that only exposes safe, read-only
    attributes (``urandom``, ``name``, ``sep``, ``linesep``, ``altsep``,
    ``pathsep``).  All filesystem, process, and environment operations are
    absent.
    """
    _SAFE_OS_ATTRS = frozenset(
        {
            "urandom",
            "name",
            "sep",
            "linesep",
            "altsep",
            "pathsep",
        }
    )
    shim = types.ModuleType("os")
    for attr in _SAFE_OS_ATTRS:
        if hasattr(_os, attr):
            setattr(shim, attr, getattr(_os, attr))
    shim.__name__ = "os"
    return shim


def _make_restricted_import(allowed: FrozenSet[str]):
    """Return a ``__import__`` replacement that only allows *allowed* modules."""
    original = _builtins.__import__

    # Sensitive g4f submodules that must never be accessible inside a sandbox,
    # regardless of the top-level module being in *allowed*.
    _BLOCKED_SUBMODULES: FrozenSet[str] = frozenset(
        {
            # API-key / credential management
            "g4f.tools.auth",
            "g4f.tools.run_tools",
            # App config (holds g4f_api_key, disable_custom_api_key, …)
            "g4f.config",
            # Cookie / session storage
            "g4f.cookies",
            # Internals that expose auth helpers transitively
            "g4f.providers.retry_provider",
            "g4f.providers.config_provider",
            # Block the entire Provider package; only specific safe submodules are
            # explicitly permitted via _ALLOWED_G4F_SUBPATHS below.
            "g4f.Provider",
            "g4f.config",
        }
    )

    # Explicit allowlist for g4f sub-paths that would otherwise be blocked.
    # Checked *before* the blocklist so these entries take priority.
    _ALLOWED_G4F_SUBPATHS: FrozenSet[str] = frozenset(
        {
            "g4f.Provider.helper",
            "g4f.Provider.base_provider",
            "g4f.Provider.template",
            "g4f.typing",
        }
    )

    def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level > 0:
            raise ImportError(
                "Relative imports are not allowed inside a .pa.py sandbox."
            )
        base = name.split(".")[0]
        # Return the restricted os shim instead of the real os module.
        if base == "os":
            _os_shim = _make_restricted_os()
            if name == "os":
                return _os_shim
            # Handle "os.submodule" — try to resolve from the shim
            obj = _os_shim
            for part in name.split(".")[1:]:
                obj = getattr(obj, part, None)
                if obj is None:
                    raise ImportError(
                        f"'{name}' is not available in the restricted os shim."
                    )
            return obj
        if base not in allowed:
            # Before rejecting, check if it's a workspace module (sibling .py file).
            workspace = get_workspace_dir()
            ws_module = _load_workspace_module(
                base, workspace, globals, fromlist, level
            )
            if ws_module is not None:
                # Handle submodule imports (e.g. "pkg.sub")
                if name != base:
                    # Try to resolve the full dotted path from the loaded module
                    obj = ws_module
                    for part in name.split(".")[1:]:
                        obj = getattr(obj, part, None)
                        if obj is None:
                            raise ImportError(
                                f"Cannot find submodule '{name}' in workspace module '{base}'."
                            )
                    return obj
                return ws_module
            raise ImportError(
                f"Import of '{name}' is not allowed in safe execution mode.\n"
                f"Allowed top-level modules: {', '.join(sorted(allowed))}"
            )
        # Explicit allowlist takes priority over the blocklist below.
        # This permits e.g. "g4f.Provider.helper" even though "g4f.Provider"
        # is blocked.
        for allowed_sub in _ALLOWED_G4F_SUBPATHS:
            if name == allowed_sub or name.startswith(allowed_sub + "."):
                return original(name, globals, locals, fromlist, level)
        # Block sensitive g4f submodules even though g4f itself is allowed.
        if name in _BLOCKED_SUBMODULES:
            raise ImportError(
                f"Import of '{name}' is not allowed inside a .pa.py sandbox "
                f"for security reasons."
            )
        # Also block when a blocked submodule is the parent of a deeper import
        # (e.g. "g4f.tools.auth.something", "g4f.Provider.OpenAI").
        for blocked in _BLOCKED_SUBMODULES:
            if name.startswith(blocked + "."):
                raise ImportError(
                    f"Import of '{name}' is not allowed inside a .pa.py sandbox "
                    f"for security reasons."
                )
        return original(name, globals, locals, fromlist, level)

    return _restricted_import


def _make_safe_globals(
    allowed: FrozenSet[str] = SAFE_MODULES,
    stdout_buf: Optional[io.StringIO] = None,
    stderr_buf: Optional[io.StringIO] = None,
    file_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Return a ``globals`` dict suitable for sandboxed ``exec``."""
    workspace = get_workspace_dir()

    # Build a reduced copy of the real built-ins
    _blocked = frozenset(
        {"exec", "eval", "compile", "input", "breakpoint", "__import__"}
    )
    safe_builtins: Dict[str, Any] = {
        k: getattr(_builtins, k) for k in dir(_builtins) if k not in _blocked
    }

    # Provide a workspace-scoped open()
    def _safe_open(file, mode="r", *args, **kwargs):
        """open() restricted to the workspace directory."""
        path = Path(file)
        if not path.is_absolute():
            if file_path is None:
                path = workspace / path
            else:
                path = file_path.parent / path.name
        try:
            resolved = path.resolve()
            ws_resolved = workspace.resolve()
            if not str(resolved).startswith(str(ws_resolved)):
                raise PermissionError(
                    f"File access outside workspace is denied: '{file}'. "
                    f"Workspace: {workspace}"
                )
        except (ValueError, OSError) as exc:
            raise PermissionError(f"Invalid file path: '{file}'") from exc
        return open(resolved, mode, *args, **kwargs)

    safe_builtins["open"] = _safe_open
    safe_builtins["__import__"] = _make_restricted_import(allowed)

    # Override print / input so stdout/stderr stay local to this sandbox
    # execution and are never written to the real sys.stdout/stderr.  This
    # avoids the global-state side-effect that contextlib.redirect_stdout
    # would cause when the thread is abandoned after a timeout.
    if stdout_buf is not None:
        _real_print = _builtins.print

        def _safe_print(*args, **kwargs):
            kwargs.setdefault("file", stdout_buf)
            _real_print(*args, **kwargs)

        safe_builtins["print"] = _safe_print

    return {
        "__builtins__": safe_builtins,
        "__name__": "__pa_provider__",
    }


# ---------------------------------------------------------------------------
# Execution result
# ---------------------------------------------------------------------------


class SafeExecutionResult:
    """Holds the outcome of a sandboxed code execution."""

    def __init__(
        self,
        success: bool,
        stdout: str = "",
        stderr: str = "",
        result: Any = None,
        error: Optional[str] = None,
        locals: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.success = success
        self.stdout = stdout
        self.stderr = stderr
        self.result = result
        self.error = error
        self.locals: Dict[str, Any] = locals or {}

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }
        if self.error:
            data["error"] = self.error
        if self.result is not None:
            try:
                json.dumps(self.result)
                data["result"] = self.result
            except (TypeError, ValueError):
                data["result"] = repr(self.result)
        return data


# ---------------------------------------------------------------------------
# Safe executor
# ---------------------------------------------------------------------------


def execute_safe_code(
    code: str,
    extra_globals: Optional[Dict[str, Any]] = None,
    allowed_modules: FrozenSet[str] = SAFE_MODULES,
    timeout: Optional[float] = MAX_EXEC_TIMEOUT,
    max_depth: int = MAX_RECURSION_DEPTH,
    file_path: "Optional[str | Path]" = None,
) -> SafeExecutionResult:
    """Execute *code* inside a safe sandbox with whitelisted module imports.

    The execution runs in a dedicated thread so that a wall-clock *timeout*
    can be enforced without blocking the caller's event loop.  A custom
    ``sys.setrecursionlimit`` guards against stack-overflow attacks.  Both
    stdout and stderr are capped at :data:`MAX_OUTPUT_BYTES`.

    Args:
        code: Python source code to execute.
        extra_globals: Additional names injected into the execution globals.
        allowed_modules: Frozenset of top-level module names that may be imported.
        timeout: Wall-clock seconds before the execution is abandoned.  Pass
            ``None`` to disable.  Defaults to :data:`MAX_EXEC_TIMEOUT`.
        max_depth: Maximum recursion depth inside the sandbox.  Defaults to
            :data:`MAX_RECURSION_DEPTH`.
        file_path: When provided, sets ``__file__`` in the sandbox globals so
            the executed code can reference its own location (e.g. to load
            sibling files relative to the ``.pa.py`` file).

    Returns:
        :class:`SafeExecutionResult` containing captured stdout/stderr, any
        ``result`` variable assigned in the code, or error information.
    """
    stdout_buf = _LimitedStringIO(MAX_OUTPUT_BYTES)
    stderr_buf = _LimitedStringIO(MAX_OUTPUT_BYTES)

    safe_globals = _make_safe_globals(
        allowed_modules, stdout_buf=stdout_buf, stderr_buf=stderr_buf, file_path=None if file_path is None else Path(file_path)
    )
    if file_path is not None:
        safe_globals["__file__"] = str(Path(file_path).resolve())
    if extra_globals:
        safe_globals.update(extra_globals)

    # Compile outside the thread so SyntaxErrors surface immediately.
    try:
        compiled = compile(code, "<pa_provider>", "exec")
    except SyntaxError:
        return SafeExecutionResult(
            success=False,
            stdout="",
            stderr="",
            error=traceback.format_exc(),
        )

    # Run in a daemon thread with timeout and recursion-depth enforcement.
    # We use a raw daemon Thread (not ThreadPoolExecutor) so that if the
    # sandboxed code runs forever the thread is discarded when the process
    # exits rather than blocking interpreter shutdown.
    exc_box: List = []
    thread = threading.Thread(
        target=_exec_in_thread,
        args=(compiled, safe_globals, safe_globals, max_depth, exc_box),
        daemon=True,
        name="g4f-sandbox",
    )
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        # The thread is still running — timeout was hit.  We cannot kill it
        # but as a daemon thread it will be reaped when the process exits.
        stdout = stdout_buf.getvalue()
        stderr = stderr_buf.getvalue()
        if stdout_buf.truncated or stderr_buf.truncated:
            stderr += "\n[Output truncated: size limit reached]"
        return SafeExecutionResult(
            success=False,
            stdout=stdout,
            stderr=stderr,
            error=(
                f"Execution timed out after {timeout:.1f} s. "
                "The thread has been abandoned."
            ),
        )

    if exc_box:
        return SafeExecutionResult(
            success=False,
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            error=exc_box[0],
        )

    stdout = stdout_buf.getvalue()
    stderr = stderr_buf.getvalue()
    if stdout_buf.truncated or stderr_buf.truncated:
        stderr += "\n[Output truncated: size limit reached]"

    return SafeExecutionResult(
        success=True,
        stdout=stdout,
        stderr=stderr,
        result=safe_globals.get("result"),
        locals=safe_globals,
    )


# ---------------------------------------------------------------------------
# .pa.py provider loader
# ---------------------------------------------------------------------------


def load_pa_provider(file_path: "str | Path") -> Optional[Type]:
    """Load a ``.pa.py`` file and return the provider class it defines.

    The file is executed inside the safe sandbox.  The module is expected to
    define a class named ``Provider``; if that name is absent the first class
    with a ``create_completion`` or ``create_async_generator`` attribute is
    returned instead.

    Args:
        file_path: Path to the ``.pa.py`` file.

    Returns:
        The provider class, or ``None`` if none could be found.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        ValueError: If *file_path* does not end with ``.pa.py``.
        RuntimeError: If the file fails to execute.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PA provider file not found: {file_path}")
    if not file_path.name.endswith(".pa.py"):
        raise ValueError(f"File must have .pa.py extension: {file_path}")

    code = file_path.read_text(encoding="utf-8")
    result = execute_safe_code(code, file_path=file_path, timeout=0.1, max_depth=100)

    if not result.success:
        raise RuntimeError(
            f"Failed to load PA provider from {file_path}:\n{result.error}"
        )

    # Prefer an explicit 'Provider' name
    provider_class = result.locals.get("Provider")
    if provider_class is not None:
        return provider_class

    # Fall back to any class that looks like a provider
    for obj in result.locals.values():
        if isinstance(obj, type) and (
            hasattr(obj, "create_completion") or hasattr(obj, "create_async_generator")
        ):
            return obj

    return None


def list_pa_providers(directory: "Optional[str | Path]" = None) -> Tuple[Path, List[Path]]:
    """Return all ``.pa.py`` files found (recursively) in *directory*.

    Args:
        directory: Directory to search.  Defaults to the workspace.

    Returns:
        Sorted list of :class:`pathlib.Path` objects.
    """
    if directory is None:
        directory = get_workspace_dir()
    directory = Path(directory)
    if not directory.exists():
        return directory, []
    return directory, sorted(directory.rglob("*.pa.py"))


# ---------------------------------------------------------------------------
# PA Provider Registry
# ---------------------------------------------------------------------------


class PaProviderRegistry:
    """Singleton registry for PA providers loaded from the workspace.

    Each provider is assigned a **stable opaque ID** derived from the SHA-256
    hash of its canonical file path (truncated to 8 hex chars).  The filename
    is never exposed in any public-facing method.

    The registry is automatically refreshed when the cache is older than
    :attr:`TTL` seconds so hot-reloaded PA files are picked up without a
    restart.
    """

    #: How long (in seconds) the cached entries remain valid.
    #: A short TTL (5 s) is intentional: PA provider files are typically edited
    #: interactively during development, so near-instant pick-up of changes is
    #: more important than avoiding the cheap filesystem scan.  Production
    #: deployments that want to reduce I/O can increase this value.
    TTL: float = 5.0

    def __init__(self) -> None:
        # Each entry: (id, label, models, working, url, cls)
        self._entries: List[tuple] = []
        # Force a refresh on the first access.
        self._loaded_at: float = -self.TTL

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_id(path: Path) -> str:
        """Return a stable 8-char hex ID for *path* (no path info exposed)."""
        return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:8]

    def _ensure_fresh(self) -> None:
        if _time_module.monotonic() - self._loaded_at >= self.TTL:
            self.refresh()

    def _ensure_index(self) -> None:
        if not self._entries:
            self.index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(self) -> None:
        """Re-scan the workspace and reload all ``.pa.py`` providers."""
        entries: List[tuple] = []
        directory, pa_paths = list_pa_providers()
        for pa_path in pa_paths:
            try:
                relative_path = pa_path.relative_to(directory).as_posix()
                provider_id = self._make_id(pa_path)
                entries.append(
                    (
                        provider_id,
                        None,
                        None,
                        True,
                        None,
                        None,
                        relative_path,
                    )
                )
            except Exception as e:
                debug.error(f"Failed to load PA provider from {pa_path}:", e)
                pass
        self._entries = entries


    def refresh(self) -> None:
        """Re-scan the workspace and reload all ``.pa.py`` providers."""
        entries: List[tuple] = []
        directory, pa_paths = list_pa_providers()
        for pa_path in pa_paths:
            try:
                cls = load_pa_provider(pa_path)
                if cls is None:
                    continue
                provider_id = self._make_id(pa_path)
                cls.__name__ = f"pa:{provider_id}"
                models_list: List[str] = []
                try:
                    if hasattr(cls, "get_models"):
                        raw = cls.get_models()
                        models_list = list(raw) if raw else []
                    elif hasattr(cls, "models"):
                        models_list = list(getattr(cls, "models") or [])
                except Exception:
                    pass
                relative_path = pa_path.relative_to(directory).as_posix()
                debug.log(f"Loaded PA provider: {provider_id} ({relative_path})")
                if is_hidden_file(relative_path):
                    relative_path = None
                entries.append(
                    (
                        provider_id,
                        getattr(cls, "label", cls.__name__),
                        models_list,
                        bool(getattr(cls, "working", True)),
                        getattr(cls, "url", None),
                        cls,
                        relative_path,
                    )
                )
            except Exception as e:
                debug.error(f"Failed to load PA provider from {pa_path}:", e)
                pass
        self._entries = entries
        self._loaded_at = _time_module.monotonic()

    def list_providers(self) -> List[Dict[str, Any]]:
        """Return a list of provider info dicts (no filesystem paths)."""
        self._ensure_fresh()
        return [
            {
                "id": e[0],
                "object": "pa_provider",
                "label": e[1],
                "models": e[2],
                "working": e[3],
                "url": e[4],
                "path": e[6],
            }
            for e in self._entries
        ]

    def get_provider_class(self, provider_id: str) -> Optional[Type]:
        """Return the provider class for *provider_id*, or ``None``."""
        self._ensure_index()
        for e in self._entries:
            if e[0] == provider_id:
                if e[5] is None:
                    return load_pa_provider(get_workspace_dir() / e[6])
                return e[5]
        return None

    def get_provider_info(self, provider_id: str) -> Optional[Dict[str, Any]]:
        """Return the info dict for *provider_id*, or ``None``."""
        self._ensure_fresh()
        for e in self._entries:
            if e[0] == provider_id:
                return {
                    "id": e[0],
                    "object": "pa_provider",
                    "label": e[1],
                    "models": e[2],
                    "working": e[3],
                    "url": e[4],
                    "path": e[6],
                }
        return None


#: Module-level singleton.
_pa_registry: Optional[PaProviderRegistry] = None


def get_pa_registry() -> PaProviderRegistry:
    """Return the singleton :class:`PaProviderRegistry`, creating it if needed."""
    global _pa_registry
    if _pa_registry is None:
        _pa_registry = PaProviderRegistry()
    return _pa_registry

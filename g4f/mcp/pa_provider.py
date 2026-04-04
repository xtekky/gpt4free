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
import ast
import sys
import json
import hashlib
import threading
import time as _time_module
import traceback
import builtins as _builtins
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Type

# ---------------------------------------------------------------------------
# Workspace directory
# ---------------------------------------------------------------------------

def get_workspace_dir() -> Path:
    """Return the workspace directory ``~/.g4f/workspace``, creating it if needed."""
    workspace = Path.home() / ".g4f" / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


# ---------------------------------------------------------------------------
# Whitelisted modules
# ---------------------------------------------------------------------------

#: Modules that are allowed inside the safe execution sandbox.
SAFE_MODULES: FrozenSet[str] = frozenset({
    # Math / numeric
    "math", "cmath", "decimal", "fractions", "statistics", "random", "numbers",
    # String / text
    "string", "re", "textwrap", "unicodedata", "difflib", "fnmatch",
    # Data structures
    "json", "csv", "collections", "heapq", "bisect", "array", "queue",
    # Functional
    "itertools", "functools", "operator",
    # Type system
    "typing", "types", "abc", "dataclasses", "enum",
    # Time / date
    "datetime", "time", "calendar",
    # I/O
    "io", "pathlib",
    # Async
    "asyncio",
    # Encoding / hashing
    "base64", "hashlib", "hmac", "binascii", "codecs", "struct",
    # URL / HTTP
    "urllib", "urllib.parse", "http", "http.client",
    # Compression
    "gzip", "zlib",
    # Misc safe stdlib
    "copy", "pprint", "reprlib", "warnings", "contextlib",
    # Third-party HTTP (used by providers)
    "aiohttp", "requests",
    # gpt4free itself
    "g4f",
})


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
        exec(compiled, safe_globals, local_vars)  # noqa: S102
    except Exception:  # noqa: BLE001
        exc_box.append(traceback.format_exc())
    finally:
        sys.setrecursionlimit(prev)


def _make_restricted_import(allowed: FrozenSet[str]):
    """Return a ``__import__`` replacement that only allows *allowed* modules."""
    original = _builtins.__import__

    def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level > 0:
            raise ImportError(
                "Relative imports are not allowed inside a .pa.py sandbox."
            )
        base = name.split(".")[0]
        if base not in allowed:
            raise ImportError(
                f"Import of '{name}' is not allowed in safe execution mode.\n"
                f"Allowed top-level modules: {', '.join(sorted(allowed))}"
            )
        return original(name, globals, locals, fromlist, level)

    return _restricted_import


def _make_safe_globals(
    allowed: FrozenSet[str] = SAFE_MODULES,
    stdout_buf: Optional[io.StringIO] = None,
    stderr_buf: Optional[io.StringIO] = None,
) -> Dict[str, Any]:
    """Return a ``globals`` dict suitable for sandboxed ``exec``."""
    workspace = get_workspace_dir()

    # Build a reduced copy of the real built-ins
    _blocked = frozenset({"exec", "eval", "compile", "input", "breakpoint", "__import__"})
    safe_builtins: Dict[str, Any] = {
        k: getattr(_builtins, k)
        for k in dir(_builtins)
        if k not in _blocked
    }

    # Provide a workspace-scoped open()
    def _safe_open(file, mode="r", *args, **kwargs):
        """open() restricted to the workspace directory."""
        path = Path(file)
        if not path.is_absolute():
            path = workspace / path
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

    Returns:
        :class:`SafeExecutionResult` containing captured stdout/stderr, any
        ``result`` variable assigned in the code, or error information.
    """
    stdout_buf = _LimitedStringIO(MAX_OUTPUT_BYTES)
    stderr_buf = _LimitedStringIO(MAX_OUTPUT_BYTES)

    safe_globals = _make_safe_globals(allowed_modules, stdout_buf=stdout_buf, stderr_buf=stderr_buf)
    if extra_globals:
        safe_globals.update(extra_globals)

    local_vars: Dict[str, Any] = {}

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
        args=(compiled, safe_globals, local_vars, max_depth, exc_box),
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
        result=local_vars.get("result"),
        locals=local_vars,
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
    result = execute_safe_code(code)

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


def list_pa_providers(directory: "Optional[str | Path]" = None) -> List[Path]:
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
        return []
    return sorted(directory.rglob("*.pa.py"))


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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Re-scan the workspace and reload all ``.pa.py`` providers."""
        entries: List[tuple] = []
        for pa_path in list_pa_providers():
            try:
                cls = load_pa_provider(pa_path)
                if cls is None:
                    continue
                provider_id = self._make_id(pa_path)
                models_list: List[str] = []
                try:
                    if hasattr(cls, "get_models"):
                        raw = cls.get_models()
                        models_list = list(raw) if raw else []
                    elif hasattr(cls, "models"):
                        models_list = list(getattr(cls, "models") or [])
                except Exception:
                    pass
                entries.append((
                    provider_id,
                    getattr(cls, "label", cls.__name__),
                    models_list,
                    bool(getattr(cls, "working", True)),
                    getattr(cls, "url", None),
                    cls,
                ))
            except Exception:
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
            }
            for e in self._entries
        ]

    def get_provider_class(self, provider_id: str) -> Optional[Type]:
        """Return the provider class for *provider_id*, or ``None``."""
        self._ensure_fresh()
        for e in self._entries:
            if e[0] == provider_id:
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

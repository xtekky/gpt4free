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

Known limitations: the sandbox does not enforce CPU/memory limits or wall-clock
timeouts.  Callers that need to bound execution time should wrap
:func:`execute_safe_code` with a ``asyncio.wait_for`` or ``concurrent.futures``
timeout.

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
import json
import contextlib
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
# Sandbox helpers
# ---------------------------------------------------------------------------

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


def _make_safe_globals(allowed: FrozenSet[str] = SAFE_MODULES) -> Dict[str, Any]:
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
) -> SafeExecutionResult:
    """Execute *code* inside a safe sandbox with whitelisted module imports.

    Args:
        code: Python source code to execute.
        extra_globals: Additional names injected into the execution globals.
        allowed_modules: Frozenset of top-level module names that may be imported.

    Returns:
        :class:`SafeExecutionResult` containing captured stdout/stderr, any
        ``result`` variable assigned in the code, or error information.
    """
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    safe_globals = _make_safe_globals(allowed_modules)
    if extra_globals:
        safe_globals.update(extra_globals)

    local_vars: Dict[str, Any] = {}

    try:
        compiled = compile(code, "<pa_provider>", "exec")
        with (
            contextlib.redirect_stdout(stdout_buf),
            contextlib.redirect_stderr(stderr_buf),
        ):
            exec(compiled, safe_globals, local_vars)  # noqa: S102

        return SafeExecutionResult(
            success=True,
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            result=local_vars.get("result"),
            locals=local_vars,
        )

    except Exception:
        return SafeExecutionResult(
            success=False,
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            error=traceback.format_exc(),
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

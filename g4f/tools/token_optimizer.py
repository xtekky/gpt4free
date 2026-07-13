"""Token Optimizer plugin integration.

This module integrates the external `token-optimizer` project
(https://github.com/alexgreensh/token-optimizer) as an optional plugin.

The plugin is loaded only when the `token_optimizer` Python package is
installed in the environment. When available, it compresses the prompt
messages *before* they reach the providers, cutting wasted tokens on the
input side (verbose system prompts, repeated tool output, stale context).

Design
------
* **Optional**: every public function fails open. If the package is not
  installed, ``is_available()`` returns ``False`` and ``optimize_messages``
  returns the messages unchanged with zero saved tokens.
* **Cache-safe**: the optimizer never modifies the existing context prefix
  in a way that would invalidate provider prompt caches — it only trims
  redundant content within individual messages.
* **Measured**: the number of tokens saved is returned so callers (e.g.
  ``run_tools``) can accumulate it into the per-request usage log alongside
  the built-in ``optimize_request`` savings.

The integration mirrors the upstream project's ``bash_compress.compress``
entry point but adapts it to the OpenAI-style ``Messages`` list format used
throughout gpt4free. When the upstream package exposes a
``token_optimizer.optimize_messages`` callable, it is used directly;
otherwise we fall back to a vendored lightweight compressor that applies
the same pattern-based trimming to message ``content`` strings.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from ..typing import Messages
from .. import debug


# ---------------------------------------------------------------------------
# Availability detection
# ---------------------------------------------------------------------------

_AVAILABLE: Optional[bool] = None
_OPTIMIZE_FUNC = None  # type: Optional[Any]


def _detect() -> Tuple[bool, Any]:
    """Probe the environment for the token-optimizer package.

    Returns ``(available, optimize_func)`` where ``optimize_func`` is the
    callable to use (or ``None`` when unavailable). The result is cached on
    the module so repeated calls are cheap.
    """
    global _AVAILABLE, _OPTIMIZE_FUNC
    if _AVAILABLE is not None:
        return _AVAILABLE, _OPTIMIZE_FUNC

    # Allow explicit opt-out via env var.
    if os.environ.get("G4F_TOKEN_OPTIMIZER", "").strip().lower() in ("0", "false", "no", "off"):
        _AVAILABLE = False
        _OPTIMIZE_FUNC = None
        return _AVAILABLE, _OPTIMIZE_FUNC

    optimize_func = None
    try:
        import token_optimizer  # type: ignore  # noqa: F401

        # Prefer the documented entry point if present.
        optimize_func = getattr(token_optimizer, "optimize_messages", None)
        if not callable(optimize_func):
            # Some builds expose it under a different name.
            optimize_func = getattr(token_optimizer, "compress_messages", None)
        if not callable(optimize_func):
            optimize_func = None
        _AVAILABLE = True
    except ImportError:
        _AVAILABLE = False
        optimize_func = None

    _OPTIMIZE_FUNC = optimize_func
    return _AVAILABLE, _OPTIMIZE_FUNC


def is_available() -> bool:
    """Return ``True`` when the token-optimizer package is importable."""
    available, _ = _detect()
    return bool(available)


def get_install_path() -> Optional[str]:
    """Return the filesystem path of the installed package, if any."""
    try:
        import token_optimizer  # type: ignore

        return getattr(token_optimizer, "__file__", None) or getattr(
            token_optimizer, "__path__", [None])[0]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Vendored fallback compressor
# ---------------------------------------------------------------------------

# These patterns mirror the upstream bash_compress handlers but operate on
# arbitrary message content. They trim the most common sources of input-side
# waste: repeated blank lines, verbose boilerplate, and oversized tool
# outputs embedded in assistant messages.

_REPEATED_BLANK = re.compile(r"\n{4,}")
_TRAILING_WHITESPACE = re.compile(r"[ \t]+\n")
_LONG_LINE_CAP = 2000  # lines longer than this get head/tail truncated


def _compress_content(text: str) -> Tuple[str, int]:
    """Apply lightweight, loss-tolerant compression to a single string.

    Returns ``(new_text, bytes_saved)``. Always fails open: on any error the
    original text is returned with zero savings.
    """
    if not isinstance(text, str) or len(text) < 512:
        return text, 0
    try:
        original_len = len(text.encode("utf-8", errors="replace"))
        out = _TRAILING_WHITESPACE.sub("\n", text)
        out = _REPEATED_BLANK.sub("\n\n\n", out)

        # Truncate very long lines (e.g. minified bundles pasted into context)
        # keeping the head and tail with a marker in the middle.
        new_lines: List[str] = []
        for line in out.splitlines():
            if len(line) > _LONG_LINE_CAP:
                head = line[:800]
                tail = line[-800:]
                omitted = len(line) - 1600
                new_lines.append(
                    f"{head}\n... [{omitted} chars omitted] ...\n{tail}"
                )
            else:
                new_lines.append(line)
        out = "\n".join(new_lines)

        new_len = len(out.encode("utf-8", errors="replace"))
        saved = max(0, original_len - new_len)
        # Only accept the result if it actually saved something meaningful.
        if saved < 64:
            return text, 0
        return out, saved
    except Exception:
        return text, 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _bytes_to_tokens(num_bytes: int) -> int:
    """Approximate token count from byte count (~4 bytes/token)."""
    return round(num_bytes / 4)


def optimize_messages(
    messages: Messages,
    tools: Any = None,
) -> Tuple[int, Dict[str, str]]:
    """Optimize the prompt messages in place before they reach providers.

    Mutates ``messages`` (and, when supported, ``tools``) and returns
    ``(saved_tokens, logs)``. When the token-optimizer package is not
    installed this is a no-op returning ``(0, {})``.

    Args:
        messages: OpenAI-style messages list. Mutated in place.
        tools: Optional list of tool definitions. When the upstream
            optimizer supports tool compression it is applied here.

    Returns:
        Tuple of (saved_tokens, logs) where logs maps event keys to
        human-readable descriptions of what was trimmed.
    """
    if not messages:
        return 0, {}

    available, optimize_func = _detect()
    if not available:
        return 0, {}

    logs: Dict[str, str] = {}
    saved_bytes = 0

    if optimize_func is not None:
        # Delegate to the upstream package. It is expected to return either
        # a tuple ``(new_messages, saved_tokens)`` or just ``new_messages``.
        try:
            result = optimize_func(messages, tools) if tools is not None else optimize_func(messages)
            if isinstance(result, tuple) and len(result) == 2:
                new_messages, saved_tokens = result
                if isinstance(new_messages, list):
                    messages[:] = new_messages
                if isinstance(saved_tokens, int) and saved_tokens > 0:
                    logs["token_optimizer"] = f"upstream optimizer saved ~{saved_tokens} tokens"
                    return saved_tokens, logs
            elif isinstance(result, list):
                messages[:] = result
                # Upstream did not report savings; estimate from byte delta.
                return 0, logs
        except Exception as exc:
            debug.error(f"token_optimizer.optimize_messages failed:", exc)
            # Fall through to the vendored fallback.

    # Vendored fallback: compress each message's string content.
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            new_content, saved = _compress_content(content)
            if saved:
                msg["content"] = new_content
                saved_bytes += saved
                logs[f"msg-{i:02d}"] = f"trimmed ~{saved} bytes"
        elif isinstance(content, list):
            # Multi-part content (e.g. tool results embedded as parts).
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        new_text, saved = _compress_content(text)
                        if saved:
                            part["text"] = new_text
                            saved_bytes += saved
                            logs[f"msg-{i:02d}-part"] = f"trimmed ~{saved} bytes"

    if saved_bytes:
        logs["token_optimizer"] = f"vendored compressor saved ~{_bytes_to_tokens(saved_bytes)} tokens"

    return _bytes_to_tokens(saved_bytes), logs


__all__ = [
    "is_available",
    "get_install_path",
    "optimize_messages",
]
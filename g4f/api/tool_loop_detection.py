"""Detect and block repetitive tool-call loops in incoming messages.

Some clients (notably AI coding agents) can get stuck in infinite loops where
the assistant repeatedly issues the *same* tool call (e.g. ``file_search`` or
``grep_search`` with identical arguments) and the corresponding ``tool`` role
responses come back empty ("(empty)", "No files found", ...).  When this
pattern is detected in an incoming request, the API server should refuse to
forward the request and instead return a descriptive error so the caller can
break out of the loop.

The detection rules (see ``/memories/repo/tool-loop-prevention.md``):

* After **2** failed attempts with the *same* query, STOP.
* Never repeat the same ``file_search`` / ``grep_search`` query more than twice.
* If a tool returns the same empty result **3+** times, switch strategy.

This module exposes :func:`detect_tool_loop`, which inspects a ``messages``
array and raises :class:`ToolLoopError` when a loop is detected.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..typing import Messages


# Thresholds (tunable via env vars if needed in the future).
MAX_REPEATS_PER_QUERY = 2          # same call allowed at most twice
EMPTY_RESULT_LOOP_THRESHOLD = 3    # 3+ empty results for same call => loop
GLOBAL_TOOL_CALL_LIMIT = 100       # hard cap on total tool calls in one request

# Substrings that indicate an empty / failed tool result.
EMPTY_RESULT_MARKERS = (
    "(empty)",
    "no files found",
    "no matches found",
    "no results found",
    "nothing found",
    "0 results",
    "0 matches",
)


class ToolLoopError(Exception):
    """Raised when a repetitive tool-call loop is detected in ``messages``."""

    def __init__(self, message: str, *, function_name: str = None,
                 arguments: Any = None, repeats: int = 0):
        super().__init__(message)
        self.function_name = function_name
        self.arguments = arguments
        self.repeats = repeats


def _normalize_arguments(arguments: Any) -> str:
    """Return a stable string key for tool-call arguments.

    ``arguments`` may arrive as a JSON string, a dict, or ``None``.  We
    canonicalise to a sorted JSON string so that semantically identical
    arguments compare equal regardless of key ordering.
    """
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            return arguments.strip()
    if isinstance(arguments, dict):
        return json.dumps(arguments, sort_keys=True, ensure_ascii=False)
    return str(arguments)


def _tool_call_key(tool_call: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """Build a ``(function_name, arguments_key)`` tuple from a tool call dict.

    Returns ``None`` if the dict is not a recognised tool call.
    """
    if not isinstance(tool_call, dict):
        return None
    if tool_call.get("type") not in (None, "function"):
        return None
    function = tool_call.get("function") or {}
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    if not name:
        return None
    args_key = _normalize_arguments(function.get("arguments"))
    return (name, args_key)


def _is_empty_result(content: Any) -> bool:
    """Return True when a ``tool`` message content looks like an empty result."""
    if content is None:
        return True
    if isinstance(content, list):
        # Content parts: join any text parts.
        text = " ".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
        return _is_empty_result(text)
    if not isinstance(content, str):
        content = str(content)
    stripped = content.strip().lower()
    if not stripped:
        return True
    return any(marker in stripped for marker in EMPTY_RESULT_MARKERS)


def detect_tool_loop(messages: Messages) -> None:
    """Inspect ``messages`` and raise :class:`ToolLoopError` on a detected loop.

    The function walks the conversation in order, pairing each assistant
    ``tool_calls`` entry with the subsequent ``tool`` role responses, and
    counts how many times each unique ``(name, arguments)`` combination is
    invoked and how many of those invocations yielded an empty result.

    Detection triggers when:

    * The same ``(name, arguments)`` combination is invoked more than
      ``MAX_REPEATS_PER_QUERY`` times, **or**
    * The same combination yields ``EMPTY_RESULT_LOOP_THRESHOLD`` or more
      empty results, **or**
    * The total number of tool calls in the request exceeds
      ``GLOBAL_TOOL_CALL_LIMIT``.
    """
    if not messages:
        return

    # Counters keyed by (function_name, arguments_key).
    call_counts: Dict[Tuple[str, str], int] = {}
    empty_counts: Dict[Tuple[str, str], int] = {}
    total_tool_calls = 0

    # Map tool_call_id -> (name, args_key) so we can attribute tool responses.
    pending_calls: Dict[str, Tuple[str, str]] = {}

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message.get("role")

        # Assistant message may carry tool_calls.
        if role == "assistant":
            tool_calls = message.get("tool_calls") or []
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    key = _tool_call_key(tc)
                    if key is None:
                        continue
                    total_tool_calls += 1
                    call_counts[key] = call_counts.get(key, 0) + 1
                    tc_id = tc.get("id") if isinstance(tc, dict) else None
                    if tc_id:
                        pending_calls[tc_id] = key

                    # Hard global cap.
                    if total_tool_calls > GLOBAL_TOOL_CALL_LIMIT:
                        raise ToolLoopError(
                            f"Tool loop detected: request contains {total_tool_calls} "
                            f"tool calls (limit {GLOBAL_TOOL_CALL_LIMIT}). The assistant "
                            f"is stuck calling tools repeatedly. Stop calling tools and "
                            f"answer directly, or switch strategy (e.g. use read_file "
                            f"with an absolute path instead of search).",
                            function_name=key[0],
                            arguments=key[1],
                            repeats=total_tool_calls,
                        )

                    # Per-query repeat cap.
                    if call_counts[key] > MAX_REPEATS_PER_QUERY:
                        raise ToolLoopError(
                            f"Tool loop detected: function '{key[0]}' was called "
                            f"{call_counts[key]} times with identical arguments "
                            f"({call_counts[key] - 1} retries, limit is "
                            f"{MAX_REPEATS_PER_QUERY - 1}). Stop repeating this call. "
                            f"Arguments: {key[1]}",
                            function_name=key[0],
                            arguments=key[1],
                            repeats=call_counts[key],
                        )

        # Tool result message: attribute to the originating call.
        elif role == "tool":
            tool_call_id = message.get("tool_call_id")
            content = message.get("content")
            key = pending_calls.get(tool_call_id) if tool_call_id else None
            # If we can't attribute via id, fall back to the most recent call.
            if key is None and pending_calls:
                # Heuristic: attribute to the last registered pending call.
                key = next(reversed(pending_calls), None)
            if key is not None and _is_empty_result(content):
                empty_counts[key] = empty_counts.get(key, 0) + 1
                if empty_counts[key] >= EMPTY_RESULT_LOOP_THRESHOLD:
                    raise ToolLoopError(
                        f"Tool loop detected: function '{key[0]}' returned empty "
                        f"results {empty_counts[key]} times with the same arguments. "
                        f"The target is likely not indexed by this tool (e.g. a git "
                        f"submodule). Switch strategy: use read_file with an absolute "
                        f"path, or list_dir to confirm the file exists. Arguments: "
                        f"{key[1]}",
                        function_name=key[0],
                        arguments=key[1],
                        repeats=empty_counts[key],
                    )


def has_tool_loop(messages: Messages) -> Optional[ToolLoopError]:
    """Convenience wrapper: return the error object if a loop is detected, else None."""
    try:
        detect_tool_loop(messages)
    except ToolLoopError as e:
        return e
    return None
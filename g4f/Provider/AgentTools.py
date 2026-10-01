"""AgentTools provider — runs a g4f MCP tools agent behind a simple
``/v1/chat/completions`` endpoint (model: ``agent-tools``).

The agent loop:
1. Builds the tool list from the local MCP server (``g4f/mcp/server.py``),
   including browser tools (CDP session) and notebook tools (markdown files).
2. Calls the underlying model with **native OpenAI-style tool calls**
   (``tools`` / ``tool_choice`` are forwarded to the provider — not the
   prompt-injection emulation used by ``ToolSupportProvider``). A JSON
   tool-call prompt is only used as fallback for providers without
   native tool support.
3. Executes returned tool calls through the MCP server and feeds results
   back (``role: "tool"`` messages) until the model answers in plain text.
4. Streams chunks (content, ``Usage``) and always terminates with a
   ``FinishReason`` — also when the total time budget (default 30s) is
   exhausted. On timeout the agent does not stop: it keeps running in a
   background task and the same stream keeps consuming it live (tool-call
   fences, reasoning, final answer). If the agent goes quiet, the stream
   ends with a session token (``JsonConversation``) and a later request
   whose messages match the session resumes it and receives the result.
5. Always ends with a session token (``JsonConversation``) so clients can
   continue/resume the task later — even after a normal completion (the
   finished run is cached and replayed on resume). Inner provider/model
   selections are passed through as ``ProviderInfo`` chunks.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import time

from typing import Optional, Union

from ..typing import AsyncResult, Messages, MediaListType
from ..providers.base_provider import AsyncGeneratorProvider, get_async_provider_method, wait_for
from ..providers.response import (
    ToolCalls,
    FinishReason,
    Usage,
    Reasoning,
    ProviderInfo,
    JsonConversation,
)
from ..providers.types import ProviderType
from ..providers.retry_provider import IterListProvider, RotatedProvider
from ..tools.tool_support import normalize_tool_defs, normalize_tool_calls, parse_tool_calls_from_text

# Total time budget for one agent request. After this many seconds the agent
# stops passing responses and yields a final FinishReason("stop").
AGENT_TIMEOUT = float(os.environ.get("G4F_AGENT_TIMEOUT", "30") or 30)

# Maximum number of model <-> tool round trips per request.
AGENT_MAX_STEPS = int(os.environ.get("G4F_AGENT_MAX_STEPS", "8") or 8)

# Continue the agent in the background when the time budget expires instead of
# stopping it. The running stream ends with a session token (JsonConversation)
# and a later request with matching messages resumes the session.
AGENT_BACKGROUND = os.environ.get("G4F_AGENT_BACKGROUND", "1").strip().lower() not in (
    "0", "false", "no", "off",
)

# Extra time budget for the background run after the request timed out.
# ``0`` or a negative value means: no limit — the background agent runs until
# it produces a final answer (or hits the step limit / fails).
AGENT_BACKGROUND_TIMEOUT = float(os.environ.get("G4F_AGENT_BACKGROUND_TIMEOUT", "0") or 0)

# Maximum model <-> tool round trips for a background run. Background
# sessions have no time budget by default, so this is the runaway guard
# (the foreground request uses AGENT_MAX_STEPS).
AGENT_BACKGROUND_MAX_STEPS = int(os.environ.get("G4F_AGENT_BACKGROUND_MAX_STEPS", "32") or 32)

# Maximum size of a single tool result kept in the loop history (chars).
# Oversized results (file reads, browser scrapes, base64 screenshots) are
# truncated to a head+tail window so one tool call cannot blow up context.
_TOOL_RESULT_MAX_CHARS = int(os.environ.get("G4F_AGENT_TOOL_RESULT_CHARS", "20000") or 20000)

# Soft context budget for the messages sent to the model per step (chars,
# ~4 chars per token). Oldest tool-call steps are dropped and old oversized
# texts truncated when the history exceeds it, keeping the request within
# the endpoint's context length.
_AGENT_CONTEXT_BUDGET = int(os.environ.get("G4F_AGENT_CONTEXT_BUDGET", "600000") or 600000)

# Keep consuming a background session while it showed activity within this
# window (seconds) — long model calls / tool executions emit no events until
# they finish, but the agent is still working.
_AGENT_ACTIVITY_WINDOW = float(os.environ.get("G4F_AGENT_ACTIVITY_WINDOW", "300") or 300)

# ---- Background agent sessions ----------------------------------------------
# Sessions are keyed by a hash of the conversation prefix (all messages up to
# and including the last user message). When a request times out, the agent
# loop keeps running in a background task; a later request whose messages
# produce the same key resumes the session and streams the result.
_agent_sessions: dict = {}
_SESSION_TTL = 3600.0
_SESSION_MAX = 32

# Dedicated event loop for background agent sessions. It runs in a daemon
# thread so background tasks survive the request's event loop being closed
# (sync / Flask request flows tear down their loop after the stream ends,
# which would cancel a plain ``asyncio.create_task`` background agent).
_background_loop: Optional[asyncio.AbstractEventLoop] = None
_background_loop_lock = threading.Lock()

def _get_background_loop() -> asyncio.AbstractEventLoop:
    """Return the persistent event loop used for background agent sessions."""
    global _background_loop
    with _background_loop_lock:
        if _background_loop is None or _background_loop.is_closed():
            _background_loop = asyncio.new_event_loop()
            threading.Thread(
                target=_background_loop.run_forever,
                name="g4f-agent-background",
                daemon=True,
            ).start()
        return _background_loop

def _session_key(messages: Messages, model: str) -> Optional[str]:
    """Build the resume key for a request (``None`` without a user message)."""
    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        return None
    parts = [model or ""]
    for msg in messages[:last_user_idx + 1]:
        try:
            parts.append(json.dumps(msg, sort_keys=True, ensure_ascii=True, default=str))
        except Exception:
            return None
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()

def _get_session(key: Optional[str], running_only: bool = False) -> Optional[dict]:
    """Return a live background session for *key* or ``None``.

    With ``running_only`` the session must still be in progress (not done),
    used for token-based resume after the message history moved on.
    """
    if not key:
        return None
    session = _agent_sessions.get(key)
    if session is None:
        return None
    if time.time() - session["created"] > _SESSION_TTL:
        _agent_sessions.pop(key, None)
        return None
    if running_only and session.get("done"):
        return None
    return session

def _put_session(key: Optional[str], session: dict) -> None:
    """Store a background session, evicting expired / oldest entries."""
    if not key:
        return
    now = time.time()
    expired = [k for k, s in _agent_sessions.items() if now - s["created"] > _SESSION_TTL]
    for k in expired:
        _agent_sessions.pop(k, None)
    while len(_agent_sessions) >= _SESSION_MAX:
        oldest = min(_agent_sessions, key=lambda k: _agent_sessions[k]["created"])
        _agent_sessions.pop(oldest, None)
    _agent_sessions[key] = session

def _extract_session_token(conversation) -> Optional[str]:
    """Extract the ``agent_session`` token from a conversation object or dict.

    Handles flat tokens, plain dicts and the nested ``{provider: {...}}``
    shape used by the GUI and retry providers.
    """
    if conversation is None:
        return None
    candidates = [conversation]
    if callable(getattr(conversation, "get_dict", None)):
        try:
            candidates.append(conversation.get_dict())
        except Exception:
            pass
    for candidate in candidates:
        token = getattr(candidate, "agent_session", None)
        if isinstance(token, str) and token:
            return token
        if not isinstance(candidate, dict):
            continue
        token = candidate.get("agent_session")
        if isinstance(token, str) and token:
            return token
        for nested in candidate.values():
            token = getattr(nested, "agent_session", None)
            if not token and isinstance(nested, dict):
                token = nested.get("agent_session")
            if isinstance(token, str) and token:
                return token
    return None

def _content_hash(content) -> Optional[str]:
    """Hash a message content (string or structured) for resume matching."""
    if isinstance(content, str):
        if not content.strip():
            return None
        return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()
    try:
        return hashlib.sha256(
            json.dumps(content, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
        ).hexdigest()
    except Exception:
        return None

def _user_message_hashes(messages: Messages) -> set:
    """Hashes of all user messages in a request (any-of fallback match)."""
    hashes = set()
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            h = _content_hash(msg.get("content"))
            if h:
                hashes.add(h)
    return hashes

def _session_user_hashes(session: dict) -> set:
    """Hashes of all user messages known to a background session."""
    hashes = session.get("user_hashes")
    if hashes is None:
        hashes = _user_message_hashes(session.get("messages") or [])
        session["user_hashes"] = hashes
    return hashes

def _find_resumable_session(incoming_hashes: set) -> Optional[dict]:
    """Find a background session sharing a user message with the request.

    Fallback for clients that re-send a request with a changed message
    prefix (extra history, a new "continue" message, injected system
    prompts): attach to the existing background job instead of starting a
    duplicate run that times out again. Running sessions are preferred over
    finished ones, newer ones over older ones.
    """
    if not incoming_hashes:
        return None
    now = time.time()
    best = None
    for session in _agent_sessions.values():
        if now - session["created"] > _SESSION_TTL:
            continue
        if not (_session_user_hashes(session) & incoming_hashes):
            continue
        if best is None:
            best = session
        elif not session.get("done") and best.get("done"):
            best = session  # prefer a running job
        elif (not session.get("done")) == (not best.get("done")) and session["created"] > best["created"]:
            best = session
    return best

def _merge_tool_call_fragments(accumulated: list, fragments: list) -> list:
    """Merge streamed tool-call delta fragments into complete tool calls.

    OpenAI-compatible APIs stream tool calls as fragments sharing an ``index``
    (and ``id``); argument strings arrive in pieces and must be concatenated.
    """
    for frag in fragments:
        if not isinstance(frag, dict):
            continue
        matched = None
        if matched is None and frag.get("id"):
            for call in accumulated:
                if call.get("id") == frag.get("id"):
                    matched = call
                    break
        index = frag.get("index")
        if index is not None:
            for call in accumulated:
                if call.get("index") == index:
                    matched = call
                    break
        if matched is None:
            accumulated.append(dict(frag))
            continue
        if frag.get("id"):
            matched["id"] = frag["id"]
        fn = frag.get("function") or {}
        if isinstance(fn, dict):
            matched_fn = matched.setdefault("function", {})
            if fn.get("name"):
                matched_fn["name"] = fn.get("name")
            if "arguments" in fn:
                if fn["arguments"] == "":
                    matched_fn["arguments"] = fn["arguments"]
                else:
                    matched_fn["arguments"] += fn["arguments"]
    return accumulated

def _select_native_provider(inner_provider):
    """Return a provider (or filtered wrapper) that supports native tool calls.

    Retry/rotation wrappers (``IterListProvider`` / ``RotatedProvider``) are
    filtered down to their native-tool providers, mirroring the routing in
    ``DefaultProvider``. Returns None when no native provider is available.
    """
    providers = getattr(inner_provider, "providers", None)
    if providers:
        from ..Provider import ProviderLoader

        native = []
        for p in providers:
            if isinstance(p, str):
                try:
                    p = ProviderLoader.from_name(p)
                except ImportError:
                    continue
            if getattr(p, "supports_native_tools", False):
                native.append(p)
        if not native:
            return None
        if isinstance(inner_provider, IterListProvider):
            return IterListProvider(native, shuffle=getattr(inner_provider, "shuffle", True))
        return RotatedProvider(native)
    if getattr(inner_provider, "supports_native_tools", False):
        return inner_provider
    return None


def _build_tool_prompt(tool_defs: list, tool_choice: Optional[Union[str, dict]]) -> str:
    """Build the tool-support system prompt for the underlying model."""
    tool_names = [t["function"]["name"] for t in tool_defs]
    lines = [
        "You are a helpful agent with access to tools. When you decide a tool is needed, "
        "respond with ONLY a valid JSON object (no markdown, no explanation) in this format:",
        '{"tool_calls": [{"name": "TOOL_NAME", "arguments": {}}]}',
        "You may include multiple tool calls in the array. The `arguments` value MUST be "
        "a JSON object matching the tool's parameter schema.",
        "After you receive tool results you can call more tools or answer the user in plain text.",
        "If no tool is needed, respond normally with plain text.",
        f"Available tools: {', '.join(tool_names)}",
    ]
    for t in tool_defs:
        fn = t["function"]
        desc = fn.get("description", "")
        lines.append(f"- Tool `{fn['name']}`" + (f": {desc}" if desc else ""))
        if fn.get("parameters"):
            lines.append(f"  Parameter Schema: {json.dumps(fn['parameters'], ensure_ascii=True)}")
    if tool_choice is not None:
        if tool_choice == "required":
            lines.append("You MUST call at least one tool. Respond with the JSON tool-call object only.")
        elif tool_choice == "none":
            lines.append("Do not call any tools. Respond with plain text only.")
        elif isinstance(tool_choice, dict):
            fn = tool_choice.get("function") if tool_choice.get("type") == "function" else None
            if isinstance(fn, dict) and fn.get("name"):
                lines.append(f"You must call the tool `{fn['name']}`.")
    return "\n".join(lines)


async def _execute_tool_call(server, call: dict, kwargs: dict) -> tuple:
    """Execute a single MCP tool call."""
    from ..mcp.server import MCPRequest

    fn = call.get("function", {})
    name = fn.get("name")
    try:
        arguments = json.loads(fn.get("arguments") or "{}")
        if not isinstance(arguments, dict):
            arguments = {}
    except Exception:
        arguments = {}
    try:
        call_response = await server.handle_request(
            MCPRequest(
                method="tools/call",
                params={"name": name, "arguments": arguments},
                origin=kwargs.get("origin"),
                user_id=kwargs.get("user_id"),
                workspace_secret=kwargs.get("workspace_secret"),
            )
        )
        if call_response.error:
            result = {"error": call_response.error.get("message", "tool failed")}
        else:
            result = call_response.result
    except Exception as e:
        result = {"error": str(e)}
    return call, name, result

def _make_session(key, server, inner_provider, inner_model, loop_messages, kwargs, media, api_key,
                  tool_defs, tool_choice, use_native, tool_names, completion_tokens, usage,
                  pending=None, partial="") -> dict:
    """Create a background session that continues the agent loop."""
    return {
        "key": key,
        "server": server,
        "inner_provider": inner_provider,
        "inner_model": inner_model,
        "messages": loop_messages,
        "kwargs": kwargs,
        "media": media,
        "api_key": api_key,
        "tool_defs": tool_defs,
        "tool_choice": tool_choice,
        "use_native": use_native,
        "tool_names": tool_names,
        # Background runs get their own (larger) step budget — the foreground
        # request counts its steps separately against AGENT_MAX_STEPS.
        "max_steps": AGENT_BACKGROUND_MAX_STEPS,
        "steps": 0,
        # ``None`` deadline: the background run has no time budget.
        "deadline": (time.time() + AGENT_BACKGROUND_TIMEOUT) if AGENT_BACKGROUND_TIMEOUT > 0 else None,
        "completion_tokens": completion_tokens,
        "usage": usage,
        "provider_info": None,
        "events": [],
        "replayed": 0,
        "pending": pending or [],
        "partial": partial or "",
        "result": None,
        "finish": None,
        "status": "running",
        "done": False,
        "error": None,
        "created": time.time(),
        "origin": os.environ.get("G4F_AGENT_ORIGIN"),
        # Heartbeat: updated on every background step, so consumers can keep
        # waiting while the agent is actively working (long tool executions
        # emit no events until they finish).
        "last_activity": time.time(),
    }

def _format_tool_calls_fence(calls: list) -> str:
    """Render tool calls as a highlighted markdown code fence for the client."""
    try:
        body = json.dumps({"tool_calls": calls}, indent=2, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        body = str(calls)
    return Reasoning(f"\n```json\n{body}\n```\n")

def _append_step_messages(session: dict, calls: list, tool_results: list, content: str) -> None:
    """Append the assistant tool-call message and the tool results."""
    session["messages"].append({
        "role": "assistant",
        "content": content if content and not content.lstrip().startswith("{") else None,
        "tool_calls": calls,
    })
    for call, name, result in tool_results:
        session["messages"].append({
            "role": "tool",
            "tool_call_id": call.get("id", ""),
            "name": name,
            "content": _truncate_text(json.dumps(result, ensure_ascii=True, default=str)),
        })

# Placeholder left behind when an old tool result is stripped from the
# history to keep the prompt small.
_TOOL_RESULT_PLACEHOLDER = "[previous tool result omitted]"

def _truncate_text(text: str, max_chars: int = _TOOL_RESULT_MAX_CHARS) -> str:
    """Truncate *text* to a head+tail window of ``max_chars`` characters."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max_chars - head
    omitted = len(text) - max_chars
    return f"{text[:head]}\n...[{omitted} characters truncated]...\n{text[-tail:]}"

def _message_size(message) -> int:
    """Approximate the size of a message in characters."""
    try:
        return len(json.dumps(message, ensure_ascii=True, default=str))
    except Exception:
        return len(str(message))

def _enforce_context_budget(messages: Messages, budget: int) -> Messages:
    """Shrink the history so the model request stays within *budget* chars.

    Old tool-call steps (assistant + their tool replies) are dropped from the
    front first; if still over budget, oversized text contents of older
    messages are truncated. System messages and the newest exchange stay.
    """
    total = sum(_message_size(m) for m in messages)
    if budget <= 0 or total <= budget:
        return messages
    result = list(messages)
    # Index of the newest tool-call step: never dropped, so the model keeps
    # the current task context.
    last_step = max(
        (i for i, m in enumerate(result)
         if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")),
        default=-1,
    )
    # Drop complete old tool-call steps (assistant + tool replies) from the
    # front, after the leading system messages, until the history fits.
    i = 0
    while total > budget and i < len(result):
        m = result[i]
        if i < last_step and isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls"):
            j = i + 1
            while j < len(result) and isinstance(result[j], dict) and result[j].get("role") == "tool":
                total -= _message_size(result[j])
                j += 1
            total -= _message_size(m)
            del result[i:j]
            continue
        i += 1
    # Still over budget: truncate long text contents (including the newest
    # tool results — the model can work with a truncated result).
    if total > budget:
        for idx in range(len(result)):
            m = result[idx]
            if not isinstance(m, dict) or not isinstance(m.get("content"), str):
                continue
            if idx == len(result) - 1 and m.get("role") != "tool":
                # Keep the newest non-tool message (current answer) intact.
                continue
            content = m["content"]
            if len(content) > _TOOL_RESULT_MAX_CHARS:
                total -= len(content) - _TOOL_RESULT_MAX_CHARS
                result[idx] = {**m, "content": _truncate_text(content)}
            if total <= budget:
                break
    return result

def _trim_loop_messages(messages: Messages, keep_last: int = 1, budget: int = _AGENT_CONTEXT_BUDGET) -> Messages:
    """Strip old reasoning and tool results from the agent message history.

    Everything before the last ``keep_last`` assistant tool-call steps is
    collapsed: tool result contents are replaced with a short placeholder and
    old assistant step text (reasoning / tool-call preamble) is dropped. The
    original conversation (user/system messages) and the most recent exchange
    stay intact, so the model keeps its current task context while the prompt
    no longer grows with every executed tool (large file reads, browser
    output, ...). Message structure is preserved: every ``tool_calls`` entry
    keeps its matching ``tool`` replies.

    When the result still exceeds ``budget`` characters (huge tool results,
    many steps, a large client conversation), old tool-call steps are dropped
    and old oversized texts truncated so the request fits the endpoint's
    context length.
    """
    step_idx = [
        i for i, m in enumerate(messages)
        if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")
    ]
    if len(step_idx) <= keep_last:
        return _enforce_context_budget(messages, budget)
    cutoff = step_idx[-keep_last]
    trimmed = []
    for i, m in enumerate(messages):
        if i >= cutoff:
            trimmed.append(m)
            continue
        if isinstance(m, dict) and m.get("role") == "tool":
            trimmed.append({**m, "content": _TOOL_RESULT_PLACEHOLDER})
        elif isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls"):
            # Keep the tool_calls structure, drop the old step text.
            trimmed.append({**m, "content": None})
        else:
            trimmed.append(m)
    return _enforce_context_budget(trimmed, budget)

def _store_done_session(session_key: Optional[str], server, inner_provider, inner_model,
                        loop_messages, kwargs, media, api_key, tool_defs, tool_choice, use_native,
                        tool_names, completion_tokens, usage, result: Optional[str],
                        finish_reason: str, provider_info: Optional[ProviderInfo] = None) -> None:
    """Cache a finished agent run so a later matching request can resume it.

    The stored session replays via ``_resume_session``: a client that re-sends
    the same conversation prefix receives the cached result instead of
    re-running the agent.
    """
    if not session_key:
        return
    session = _make_session(
        session_key, server, inner_provider, inner_model, loop_messages,
        kwargs, media, api_key, tool_defs, tool_choice, use_native, tool_names,
        completion_tokens, usage,
    )
    session["result"] = result
    session["finish"] = finish_reason
    session["status"] = "done"
    session["done"] = True
    if provider_info is not None:
        session["provider_info"] = provider_info.get_dict()
    _put_session(session_key, session)

async def _run_background_session(session: dict) -> None:
    """Continue the agent loop in the background until a final answer is found.

    Runs as a detached asyncio task after the request's time budget expired.
    Executed tool calls are recorded in ``session["events"]`` so a resumed
    request can replay them; the final answer is stored in ``session["result"]``.
    """
    from ..providers.tool_support import _preprocess_tool_messages, _merge_messages_to_single_user

    try:
        pending = session.pop("pending", None) or []
        if pending:
            # Complete tool calls received before the timeout: run them first.
            content = session.get("partial") or ""
            tool_results = [
                await _execute_tool_call(session["server"], call, session["kwargs"])
                for call in pending
            ]
            # All pending calls merged into a single fence event.
            session["events"].append({"text": _format_tool_calls_fence(pending)})
            _append_step_messages(session, pending, tool_results, content)
            session["last_activity"] = time.time()
        while True:
            session["last_activity"] = time.time()
            session["steps"] += 1
            remaining = None if session["deadline"] is None else session["deadline"] - time.time()
            if (remaining is not None and remaining <= 0) or session["steps"] > session["max_steps"]:
                session["status"] = "timeout"
                break
            method = get_async_provider_method(session["inner_provider"])
            if session["use_native"]:
                inner_kwargs = dict(session["kwargs"])
                inner_kwargs["tools"] = session["tool_defs"]
                if session["tool_choice"] is not None:
                    inner_kwargs["tool_choice"] = session["tool_choice"]
                inner_messages = _trim_loop_messages(session["messages"])
            else:
                inner_kwargs = session["kwargs"]
                inner_messages = _merge_messages_to_single_user(
                    _preprocess_tool_messages(_trim_loop_messages(session["messages"]))
                )
            response = method(
                model=session["inner_model"],
                messages=inner_messages,
                stream=True,
                media=session["media"],
                api_key=session["api_key"],
                **inner_kwargs,
            )
            # ``remaining`` is ``None`` without a background time budget:
            # wait for the model response without a timeout.
            response = wait_for(response, timeout=remaining)
            content_chunks: list[str] = []
            native_calls: list = []
            finish = None
            try:
                async for chunk in response:
                    if isinstance(chunk, str):
                        content_chunks.append(chunk)
                        session["completion_tokens"] += round(len(chunk.encode("utf-8")) / 4)
                    elif isinstance(chunk, ToolCalls):
                        native_calls = _merge_tool_call_fragments(native_calls, chunk.get_list())
                    elif isinstance(chunk, FinishReason):
                        finish = chunk
                    elif isinstance(chunk, Usage):
                        session["usage"] = chunk
                    elif isinstance(chunk, ProviderInfo):
                        # Remember the inner provider/model selection for resume.
                        session["provider_info"] = chunk.get_dict()
                    elif isinstance(chunk, JsonConversation):
                        continue
                    elif isinstance(chunk, Reasoning):
                        # Forward reasoning to the outer request on resume.
                        if chunk.token:
                            session["events"].append({"reasoning": chunk.token})
                    else:
                        yield chunk
            except TimeoutError:
                # Model call stalled: retry it with the remaining budget.
                continue
            content = "".join(content_chunks)
            session["partial"] = content
            parsed_calls = native_calls or None
            if parsed_calls is None and not session["use_native"] and content and session["tool_names"]:
                parsed_calls = parse_tool_calls_from_text(content, session["tool_names"])
            openai_calls = normalize_tool_calls(parsed_calls) if parsed_calls else []
            openai_calls = [
                call for call in openai_calls
                if call.get("function", {}).get("name") in session["tool_names"]
            ]
            if not openai_calls:
                session["result"] = content
                session["finish"] = finish.reason if finish is not None else "stop"
                session["status"] = "done"
                break
            tool_results = [
                await _execute_tool_call(session["server"], call, session["kwargs"])
                for call in openai_calls
            ]
            # All calls of this step merged into a single fence event.
            session["events"].append({"text": _format_tool_calls_fence(openai_calls)})
            _append_step_messages(session, openai_calls, tool_results, content)
            session["last_activity"] = time.time()
    except Exception as e:
        session["status"] = "error"
        session["error"] = str(e)
    finally:
        session["done"] = True


class AgentTools(AsyncGeneratorProvider):
    """Agent provider that executes g4f MCP tools (browser, notebooks, files, ...)
    in a loop and exposes everything through the standard chat completions flow.

    Use model ``agent-tools`` (optionally ``agent-tools:<inner-model>`` to select
    the underlying model). Tool calls are passed to the inner model natively
    via ``tools`` / ``tool_choice`` (prompt emulation is only a fallback for
    providers without native tool support). The request is capped by
    ``G4F_AGENT_TIMEOUT`` seconds (default 30); on expiry the agent keeps
    running in the background — without an extra time budget by default —
    and the same stream keeps consuming it live (tool-call fences, reasoning,
    final answer). If the agent goes quiet, the stream ends with a session
    token (``JsonConversation``); sending the same messages again reattaches
    to the session and streams its progress. Every stream ends with a session
    token so the task can always be continued/resumed, and inner
    provider/model selections are passed through as ``ProviderInfo`` chunks.
    """

    working = True
    supports_native_tools = True
    use_stream_timeout = False
    models = [os.getenv("G4F_AGENT_MODEL", "auto")]

    @classmethod
    async def _start_background(
        cls, session: dict, messages: Messages, completion_tokens: int,
        timeout: float = AGENT_TIMEOUT, note: Optional[str] = None,
    ) -> AsyncResult:
        """Spawn the background task and keep consuming it live."""
        if not session.get("task"):
            # Run on the dedicated background loop, so the agent survives the
            # request's event loop being closed (sync / Flask request flows).
            # ``_run_background_session`` is an async generator, which
            # ``run_coroutine_threadsafe`` rejects ("A coroutine object is
            # required") — drain it inside a plain coroutine instead.
            async def _run_background_task() -> None:
                async for _chunk in _run_background_session(session):
                    pass
            session["task"] = asyncio.run_coroutine_threadsafe(
                _run_background_task(), _get_background_loop()
            )
        _put_session(session["key"], session)
        # Keep the stream open and consume the background job live: new steps
        # (tool-call fences, reasoning) and the final answer are streamed as
        # they happen. If the agent goes quiet, ``_consume_session`` ends the
        # stream with the session token so the client can reattach later.
        if note is not None:
            yield note
        else:
            yield "\n\n⏳ *Time budget reached — the agent keeps running in the background. Live progress follows; if this stream ends, send the same messages again to reattach.*\n"
        async for chunk in cls._consume_session(session, timeout):
            yield chunk

    @classmethod
    async def _consume_session(cls, session: dict, timeout: float, initial_note: str = None) -> AsyncResult:
        """Consume a background session and stream its progress live.

        Replays tool calls executed in the background, then keeps consuming
        the running job: new steps (tool-call fences, reasoning) are streamed
        as they happen and the wait is extended while the agent makes
        progress. When the agent goes quiet, the final answer is yielded —
        or an explicit "still running" note plus a fresh session token.
        """
        def pending_events() -> list:
            events = session["events"][session["replayed"]:]
            session["replayed"] = len(session["events"])
            chunks = []
            for event in events:
                if "text" in event:
                    # Highlighted tool-call code fence.
                    chunks.append(event["text"])
                elif "reasoning" in event:
                    chunks.append(Reasoning(token=event["reasoning"]))
            return chunks

        for chunk in pending_events():
            yield chunk
        info = session.get("provider_info")
        if info:
            # Pass the inner provider/model selection through on resume too.
            yield ProviderInfo(**info)
        if initial_note:
            # e.g. the "time budget reached" note on the background handoff.
            yield initial_note
        deadline = time.time() + max(timeout, 1.0)
        while not session["done"]:
            await asyncio.sleep(0.25)
            events = pending_events()
            if events:
                # Live steps: stream fences/reasoning as they happen and
                # extend the wait while the agent keeps making progress.
                for chunk in events:
                    yield chunk
                deadline = time.time() + max(timeout, 1.0)
            elif time.time() >= deadline:
                # Keep waiting while the agent shows signs of life: long
                # model calls / tool executions emit no events until done.
                last = session.get("last_activity") or session["created"]
                if time.time() - last < _AGENT_ACTIVITY_WINDOW:
                    continue
                break
        # Flush events recorded between the last poll and completion.
        for chunk in pending_events():
            yield chunk
        if not session["done"]:
            # Still running: inform the client and end this stream with the
            # session token so it can resume again later.
            yield '\n\n⏳ *The background agent is still running — send the same messages again (or just "continue") to keep streaming its progress.*\n'
            yield JsonConversation(provider=cls.__name__, agent_session=session["key"], status="running")
            yield FinishReason("stop")
            return
        if session["status"] == "error":
            yield f"The background agent failed: {session['error']}"
        elif session["status"] == "done" and session.get("result"):
            yield session["result"]
        elif session.get("partial"):
            yield session["partial"]
        else:
            # Step limit reached without a final answer: report the progress
            # made so far instead of a bare dead-end message.
            used_tools = []
            for msg in session["messages"]:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    for call in msg.get("tool_calls") or []:
                        name = (call.get("function") or {}).get("name")
                        if name and name not in used_tools:
                            used_tools.append(name)
            summary = f"The agent stopped after {session.get('steps', 0)} step(s) without a final answer"
            if used_tools:
                summary += " (tools used: " + ", ".join(used_tools[:8]) + ")"
            summary += ". Send the same messages again to continue this task."
            yield summary
        # Always end with the session token so the task can be continued /
        # resumed again later (finished runs are cached and replayed).
        yield JsonConversation(provider=cls.__name__, agent_session=session["key"], status=session["status"])
        usage = session.get("usage")
        if usage is not None:
            yield usage
        else:
            completion_tokens = session.get("completion_tokens", 0)
            yield Usage(
                promptTokens=round(len(json.dumps(session["messages"], default=str).encode("utf-8")) / 4),
                completionTokens=completion_tokens,
                totalTokens=completion_tokens,
            )
        yield FinishReason(session.get("finish") or "stop")

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = True,
        media: MediaListType = None,
        tools: list = None,
        tool_choice: Optional[Union[str, dict]] = None,
        api_key: Optional[str] = None,
        provider: Optional[Union[ProviderType, str]] = None,
        **kwargs,
    ) -> AsyncResult:
        from ..client.service import get_model_and_provider
        from ..providers.tool_support import _preprocess_tool_messages, _merge_messages_to_single_user
        from ..mcp.server import MCPServer

        api_key = api_key or os.getenv("G4F_AGENT_API_KEY")

        # Resolve the inner model / provider ("agent-tools:<model>" supported).
        model = model or cls.default_model
        model, inner_provider = get_model_and_provider(model, provider, stream, logging=False)

        # The agent manages tool calling itself; only strip unrelated client
        # tool kwargs. ``tools`` / ``tool_choice`` are forwarded natively.
        for key in ("parallel_tool_calls", "tool_emulation"):
            kwargs.pop(key, None)

        # Total time budget: ``agent_timeout`` (request field) > ``timeout`` > default.
        timeout = float(kwargs.pop("agent_timeout", None) or kwargs.pop("timeout", None) or AGENT_TIMEOUT)
        max_steps = int(kwargs.pop("max_steps", None) or AGENT_MAX_STEPS)
        deadline = time.time() + timeout

        # Background continuation: on timeout the agent keeps running in a
        # background task and the stream ends with a session token. A later
        # request with matching messages (same conversation prefix) resumes it.
        background = kwargs.pop("agent_background", None)
        if background is None:
            background = AGENT_BACKGROUND
        background = bool(background)
        # The session key is always computed so a conversation token can be
        # yielded (and the task resumed) even after a normal completion.
        session_key = _session_key(messages, model or "agent-tools")
        session = None
        if session_key:
            # Resume by conversation prefix (exact match) or, as a fallback, by
            # the session token from the conversation while the agent is still
            # running (the client may send an updated message history).
            session = _get_session(session_key)
            if session is None:
                token = _extract_session_token(kwargs.get("conversation"))
                if token and token != session_key:
                    session = _get_session(token, running_only=True)
            if session is None:
                # Fallback: attach to a background session that shares any
                # user message with this request (the client may have re-sent
                # with a changed prefix or a new "continue" message).
                session = _find_resumable_session(_user_message_hashes(messages))
            # The agent manages its own session state; never forward the
            # conversation token to the inner provider.
            kwargs.pop("conversation", None)
            if session is not None:
                yield ProviderInfo(**cls.get_dict(), model=model or "agent-tools")
                unseen = _user_message_hashes(messages) - _session_user_hashes(session)
                if session.get("done") and (unseen or (session.get("status") == "timeout" and not session.get("result"))):
                    # Finished run + new user message, or stalled run (step
                    # limit, no final answer): continue the task with the
                    # accumulated history instead of replaying the cached /
                    # dead-end message forever. New user messages (e.g.
                    # "continue") are appended so the agent acts on them.
                    for msg in messages:
                        if not isinstance(msg, dict) or msg.get("role") != "user":
                            continue
                        h = _content_hash(msg.get("content"))
                        if h and h in unseen:
                            unseen.discard(h)
                            session["messages"].append(msg)
                    session.update(
                        done=False, status="running", steps=0, task=None,
                        result=None, finish=None, partial="",
                        last_activity=time.time(),
                    )
                    _put_session(session["key"], session)
                    async for chunk in cls._start_background(
                        session, messages, 0, timeout,
                        note="\n\n🔄 *Continuing the previous agent run…*\n",
                    ):
                        yield chunk
                else:
                    if unseen:
                        # Running session: append new user messages (e.g. a
                        # "continue" nudge) so the agent sees them next step.
                        seen = _session_user_hashes(session)
                        for msg in messages:
                            if not isinstance(msg, dict) or msg.get("role") != "user":
                                continue
                            h = _content_hash(msg.get("content"))
                            if h and h in unseen:
                                unseen.discard(h)
                                seen.add(h)
                                session["messages"].append(msg)
                    async for chunk in cls._consume_session(session, timeout):
                        yield chunk
                return

        # Build the tool definitions from the local MCP server.
        server = MCPServer()
        mcp_tool_defs = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["inputSchema"],
                },
            }
            for t in server.get_tool_list()
        ]
        client_tool_defs = normalize_tool_defs(tools) if tools else []
        tool_defs = mcp_tool_defs + [
            t for t in client_tool_defs
            if t.get("function", {}).get("name") not in {d["function"]["name"] for d in mcp_tool_defs}
        ]
        tool_names = [t["function"]["name"] for t in tool_defs]

        # Prefer native tool calls: forward ``tools`` / ``tool_choice`` to the
        # inner provider instead of injecting a JSON tool-call prompt.
        native_provider = _select_native_provider(inner_provider)
        use_native = native_provider is not None
        if use_native:
            inner_provider = native_provider

        yield ProviderInfo(**cls.get_dict(), model=model or "agent-tools")

        loop_messages: Messages = list(messages)
        if not use_native:
            # Fallback for providers without native tool support: inject the
            # JSON tool-call prompt (ToolSupportProvider-style emulation).
            loop_messages = [
                {"role": "system", "content": _build_tool_prompt(tool_defs, tool_choice)}
            ] + loop_messages

        method = get_async_provider_method(inner_provider)
        completion_tokens = 0
        usage = None
        inner_info: Optional[ProviderInfo] = None
        steps = 0

        while True:
            steps += 1
            remaining = deadline - time.time()
            if remaining <= 0 or steps > max_steps:
                break

            if use_native:
                inner_kwargs = dict(kwargs)
                inner_kwargs["tools"] = tool_defs
                if tool_choice is not None:
                    inner_kwargs["tool_choice"] = tool_choice
                # Native tool APIs expect proper assistant/tool message roles.
                # Old reasoning and tool results are stripped to save tokens.
                inner_messages = _trim_loop_messages(loop_messages)
            else:
                inner_kwargs = kwargs
                inner_messages = _merge_messages_to_single_user(_preprocess_tool_messages(_trim_loop_messages(loop_messages)))
            response = method(
                model=model,
                messages=inner_messages,
                stream=stream,
                media=media,
                api_key=api_key,
                **inner_kwargs,
            )
            # response = wait_for(response, timeout=max(remaining, 0.1))

            content_chunks: list[str] = []
            native_calls: list = []
            finish = None
            timed_out = False

            try:
                async for chunk in response:
                    if isinstance(chunk, str):
                        content_chunks.append(chunk)
                        completion_tokens += round(len(chunk.encode("utf-8")) / 4)
                        yield chunk
                    elif isinstance(chunk, Reasoning):
                        yield chunk
                        if chunk.token:
                            completion_tokens += round(len(chunk.token.encode("utf-8")) / 4)
                    elif isinstance(chunk, ToolCalls):
                        # Streamed deltas arrive as fragments sharing an index.
                        native_calls = _merge_tool_call_fragments(native_calls, chunk.get_list())
                    elif isinstance(chunk, FinishReason):
                        finish = chunk
                    elif isinstance(chunk, Usage):
                        usage = chunk
                    elif isinstance(chunk, ProviderInfo):
                        # Pass the inner provider/model selection through.
                        inner_info = chunk
                        yield chunk
                    elif isinstance(chunk, JsonConversation):
                        continue
                    else:
                        yield chunk
            except TimeoutError:
                timed_out = True

            content = "".join(content_chunks)

            # Determine the tool calls for this step.
            parsed_calls = native_calls or None
            if parsed_calls is None and not use_native and content and tool_names:
                # Emulation fallback: parse the JSON tool-call text format.
                parsed_calls = parse_tool_calls_from_text(content, tool_names)

            openai_calls = normalize_tool_calls(parsed_calls) if parsed_calls else []
            openai_calls = [call for call in openai_calls if call.get("function", {}).get("name") in tool_names]

            if not openai_calls and not timed_out:
                # Final answer: terminate with a finish reason. Always yield a
                # conversation token so the client can resume the task later.
                if session_key:
                    _store_done_session(
                        session_key, server, inner_provider, model, loop_messages,
                        kwargs, media, api_key, tool_defs, tool_choice, use_native, tool_names,
                        completion_tokens, usage, result=content or None,
                        finish_reason=finish.reason if finish is not None else "stop",
                        provider_info=inner_info,
                    )
                    yield JsonConversation(provider=cls.__name__, agent_session=session_key, status="done")
                if usage is not None:
                    yield usage
                else:
                    yield Usage(
                        promptTokens=round(len(json.dumps(messages, default=str).encode("utf-8")) / 4),
                        completionTokens=completion_tokens,
                        totalTokens=completion_tokens,
                    )
                if finish is not None:
                    yield finish
                else:
                    yield FinishReason("stop")
                return

            if timed_out and not (background and session_key):
                # Background continuation disabled (or impossible): end with
                # the partial answer so clients never see a hanging response.
                if session_key:
                    _store_done_session(
                        session_key, server, inner_provider, model, loop_messages,
                        kwargs, media, api_key, tool_defs, tool_choice, use_native, tool_names,
                        completion_tokens, usage, result=content or None, finish_reason="stop",
                        provider_info=inner_info,
                    )
                    yield JsonConversation(provider=cls.__name__, agent_session=session_key, status="done")
                if usage is not None:
                    yield usage
                else:
                    yield Usage(
                        promptTokens=round(len(json.dumps(messages, default=str).encode("utf-8")) / 4),
                        completionTokens=completion_tokens,
                        totalTokens=completion_tokens,
                    )
                yield FinishReason("stop")
                return

            if timed_out:
                # Time budget exhausted mid-step: keep the agent running in
                # the background and keep consuming it live in this stream.
                # Complete tool calls are handed over so no work is lost.
                agent_session = _make_session(
                    session_key, server, inner_provider, model, loop_messages,
                    kwargs, media, api_key, tool_defs, tool_choice, use_native, tool_names,
                    completion_tokens, usage, pending=openai_calls, partial=content,
                )
                async for chunk in cls._start_background(
                    agent_session, messages, completion_tokens, timeout
                ):
                    yield chunk
                return

            # Surface the tool calls to the client as a single highlighted
            # code fence (visible in any markdown UI). No structured
            # ``ToolCalls`` chunks are passed to the frontend.
            if openai_calls:
                yield _format_tool_calls_fence(openai_calls)
            tool_results = [
                await _execute_tool_call(server, call, kwargs) for call in openai_calls
            ]

            # Feed the tool results back into the conversation.
            loop_messages.append({
                "role": "assistant",
                "content": content if content and not content.lstrip().startswith("{") else None,
                "tool_calls": openai_calls,
            })
            for call, name, result in tool_results:
                loop_messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "name": name,
                    "content": _truncate_text(json.dumps(result, ensure_ascii=True, default=str)),
                })

            remaining = deadline - time.time()
            if remaining <= 0:
                if background and session_key:
                    # Budget exhausted after tool execution: continue the loop
                    # in the background (tool results are already part of the
                    # messages) and keep consuming it live in this stream.
                    agent_session = _make_session(
                        session_key, server, inner_provider, model, loop_messages,
                        kwargs, media, api_key, tool_defs, tool_choice, use_native, tool_names,
                        completion_tokens, usage,
                    )
                    async for chunk in cls._start_background(
                        agent_session, messages, completion_tokens, timeout
                    ):
                        yield chunk
                    return
                if session_key:
                    _store_done_session(
                        session_key, server, inner_provider, model, loop_messages,
                        kwargs, media, tool_defs, tool_choice, use_native, tool_names,
                        completion_tokens, usage, result=content or None, finish_reason="stop",
                        provider_info=inner_info,
                    )
                    yield JsonConversation(provider=cls.__name__, agent_session=session_key, status="done")
                if usage is not None:
                    yield usage
                else:
                    yield Usage(
                        promptTokens=round(len(json.dumps(messages, default=str).encode("utf-8")) / 4),
                        completionTokens=completion_tokens,
                        totalTokens=completion_tokens,
                    )
                yield FinishReason("stop")
                return

        # Time budget or step limit exhausted before a final answer. When the
        # time budget expired, keep the agent running in the background and
        # keep consuming it live in this stream.
        if session_key and time.time() >= deadline:
            agent_session = _make_session(
                session_key, server, inner_provider, model, loop_messages,
                kwargs, media, api_key, tool_defs, tool_choice, use_native, tool_names,
                completion_tokens, usage,
            )
            async for chunk in cls._start_background(
                agent_session, messages, completion_tokens, timeout
            ):
                yield chunk
            return
        # Step limit exhausted with time remaining (or no resumable session):
        # cache the partial run and end with a conversation token + finish reason.
        if session_key:
            _store_done_session(
                session_key, server, inner_provider, model, loop_messages,
                kwargs, media, tool_defs, tool_choice, use_native, tool_names,
                completion_tokens, usage, result=content or None, finish_reason="stop",
                provider_info=inner_info,
            )
            yield JsonConversation(provider=cls.__name__, agent_session=session_key, status="done")
        if usage is not None:
            yield usage
        else:
            yield Usage(
                promptTokens=round(len(json.dumps(messages, default=str).encode("utf-8")) / 4),
                completionTokens=completion_tokens,
                totalTokens=completion_tokens,
            )
        yield FinishReason("stop")

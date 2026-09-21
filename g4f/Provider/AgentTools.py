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
4. Streams chunks (content, ``ToolCalls``, ``Usage``) and always terminates
   with a ``FinishReason`` — also when the total time budget (default 30s)
   is exhausted, so clients never see a hanging response. On timeout the
   agent does not stop: it keeps running in a background task and the stream
   ends with a session token (``JsonConversation``). A later request whose
   messages match the session resumes it and receives the result.
5. Always ends with a session token (``JsonConversation``) so clients can
   continue/resume the task later — even after a normal completion (the
   finished run is cached and replayed on resume). Inner provider/model
   selections are passed through as ``ProviderInfo`` chunks.

Tool calls carry an ``extra_content`` dict with file paths and change
summaries (created / replaced / deleted / size) so GUIs and API clients can
highlight changed files directly in the tool calls.
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

# Tools that modify files, used to build the extra_content change summary.
_FILE_RESULT_KEYS = ("filePath", "path", "notebook", "dirPath", "screenshot")
_CHANGE_KEYS = ("created", "overwritten", "replaced", "deleted", "appended", "size")

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
        from ..Provider import __getattr__ as get_provider

        native = []
        for p in providers:
            if isinstance(p, str):
                try:
                    p = get_provider(p)
                except AttributeError:
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


def _build_extra_content(name: str, arguments: dict, result) -> Optional[dict]:
    """Summarize a tool execution for UI code-highlight / file-change rendering."""
    extra: dict = {"tool": name}
    if not isinstance(result, dict):
        return extra
    for key in _FILE_RESULT_KEYS:
        if result.get(key):
            extra["file"] = result[key]
            break
    changes = {key: result[key] for key in _CHANGE_KEYS if key in result}
    if changes:
        extra["changes"] = changes
    if isinstance(arguments.get("oldString"), str):
        extra["oldString"] = arguments["oldString"][:2000]
    if isinstance(arguments.get("newString"), str):
        extra["newString"] = arguments["newString"][:2000]
    if isinstance(result.get("error"), str):
        extra["error"] = result["error"]
    return extra

async def _execute_tool_call(server, call: dict, kwargs: dict) -> tuple:
    """Execute a single MCP tool call and attach ``extra_content`` metadata."""
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
    call["extra_content"] = _build_extra_content(name, arguments, result)
    return call, name, result

def _make_session(key, server, inner_provider, inner_model, loop_messages, kwargs, media,
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
        "tool_defs": tool_defs,
        "tool_choice": tool_choice,
        "use_native": use_native,
        "tool_names": tool_names,
        "max_steps": AGENT_MAX_STEPS,
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
    }

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
            "content": json.dumps(result, ensure_ascii=True, default=str),
        })

def _store_done_session(session_key: Optional[str], server, inner_provider, inner_model,
                        loop_messages, kwargs, media, tool_defs, tool_choice, use_native,
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
        kwargs, media, tool_defs, tool_choice, use_native, tool_names,
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
            session["events"].append({"tool_calls": pending})
            _append_step_messages(session, pending, tool_results, content)
        while True:
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
                inner_messages = session["messages"]
            else:
                inner_kwargs = session["kwargs"]
                inner_messages = _merge_messages_to_single_user(
                    _preprocess_tool_messages(session["messages"])
                )
            response = method(
                model=session["inner_model"],
                messages=inner_messages,
                stream=True,
                media=session["media"],
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
                    elif isinstance(chunk, (JsonConversation, Reasoning)):
                        continue
                    elif isinstance(chunk, Exception):
                        raise chunk
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
            session["events"].append({"tool_calls": openai_calls})
            _append_step_messages(session, openai_calls, tool_results, content)
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
    and the stream ends with a finish reason plus a session token
    (``JsonConversation``). Sending the same messages again
    resumes the session and streams the final result. Every stream ends with a
    session token so the task can always be continued/resumed, and inner
    provider/model selections are passed through as ``ProviderInfo`` chunks.
    """

    working = True
    supports_native_tools = True
    use_stream_timeout = False
    models = [os.getenv("G4F_AGENT_MODEL", "auto")]

    @classmethod
    async def _start_background(
        cls, session: dict, messages: Messages, completion_tokens: int
    ) -> AsyncResult:
        """Spawn the background task and end the stream with a session token."""
        if not session.get("task"):
            # Run on the dedicated background loop, so the agent survives the
            # request's event loop being closed (sync / Flask request flows).
            session["task"] = asyncio.run_coroutine_threadsafe(
                _run_background_session(session), _get_background_loop()
            )
        _put_session(session["key"], session)
        yield JsonConversation(provider=cls.__name__, agent_session=session["key"], status="running")
        usage = session.get("usage")
        if usage is None:
            usage = Usage(
                promptTokens=round(len(json.dumps(messages, default=str).encode("utf-8")) / 4),
                completionTokens=completion_tokens,
                totalTokens=completion_tokens,
            )
        yield usage
        yield FinishReason("stop")

    @classmethod
    async def _resume_session(cls, session: dict, timeout: float) -> AsyncResult:
        """Wait for a background session and stream its result.

        Replays tool calls executed in the background, waits up to ``timeout``
        seconds for completion and yields the final answer — or a fresh session
        token when the agent is still running.
        """
        def pending_events() -> list:
            events = session["events"][session["replayed"]:]
            session["replayed"] = len(session["events"])
            return [ToolCalls(event["tool_calls"]) for event in events]

        for chunk in pending_events():
            yield chunk
        info = session.get("provider_info")
        if info:
            # Pass the inner provider/model selection through on resume too.
            yield ProviderInfo(**info)
        deadline = time.time() + max(timeout, 1.0)
        while not session["done"] and time.time() < deadline:
            await asyncio.sleep(0.25)
            for chunk in pending_events():
                yield chunk
        if not session["done"]:
            # Still running: end this stream with the session token so the
            # client can resume again later.
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
            # Without a result or partial output the run was stopped by the
            # step limit (background sessions have no time budget by default).
            yield "The background agent ended without a result."
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
            # The agent manages its own session state; never forward the
            # conversation token to the inner provider.
            kwargs.pop("conversation", None)
            if session is not None:
                yield ProviderInfo(**cls.get_dict(), model=model or "agent-tools")
                async for chunk in cls._resume_session(session, timeout):
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
                inner_messages = loop_messages
            else:
                inner_kwargs = kwargs
                inner_messages = _merge_messages_to_single_user(_preprocess_tool_messages(loop_messages))
            response = method(
                model=model,
                messages=inner_messages,
                stream=stream,
                media=media,
                api_key=api_key,
                **inner_kwargs,
            )
            response = wait_for(response, timeout=max(remaining, 0.1))

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
                        kwargs, media, tool_defs, tool_choice, use_native, tool_names,
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

            if timed_out:
                # Time budget exhausted mid-step: keep the agent running in the
                # background and end this stream with a session token. Complete
                # tool calls are handed over so no work is lost.
                if openai_calls:
                    yield ToolCalls(openai_calls)
                agent_session = _make_session(
                    session_key, server, inner_provider, model, loop_messages,
                    kwargs, media, tool_defs, tool_choice, use_native, tool_names,
                    completion_tokens, usage, pending=openai_calls, partial=content,
                )
                async for chunk in cls._start_background(agent_session, messages, completion_tokens):
                    yield chunk
                return

            # Surface the tool calls (with file/change metadata) to the client.
            tool_results = [
                await _execute_tool_call(server, call, kwargs) for call in openai_calls
            ]

            yield ToolCalls(openai_calls)

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
                    "content": json.dumps(result, ensure_ascii=True, default=str),
                })

            remaining = deadline - time.time()
            if remaining <= 0:
                if background and session_key:
                    # Budget exhausted after tool execution: continue the loop
                    # in the background (tool results are already part of the
                    # messages) and end this stream with a session token.
                    agent_session = _make_session(
                        session_key, server, inner_provider, model, loop_messages,
                        kwargs, media, tool_defs, tool_choice, use_native, tool_names,
                        completion_tokens, usage,
                    )
                    async for chunk in cls._start_background(agent_session, messages, completion_tokens):
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
        # time budget expired, keep the agent running in the background and end
        # the stream with a session token so the client can resume it later.
        if session_key and time.time() >= deadline:
            agent_session = _make_session(
                session_key, server, inner_provider, model, loop_messages,
                kwargs, media, tool_defs, tool_choice, use_native, tool_names,
                completion_tokens, usage,
            )
            async for chunk in cls._start_background(agent_session, messages, completion_tokens):
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

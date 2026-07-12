from __future__ import annotations

import json
import re
from typing import Optional, Union

from ..typing import AsyncResult, Messages, MediaListType
from ..client.service import get_model_and_provider
from ..client.helper import filter_json
from .types import ProviderType
from .base_provider import AsyncGeneratorProvider, get_async_provider_method
from .response import ToolCalls, FinishReason, Usage, Reasoning, JsonConversation


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ``` or ``` ... ```) wrapping a payload."""
    if not text:
        return text
    text = text.strip()
    m = re.match(r"^```(?:json|JSON)?\s*\n?([\s\S]*?)\n?```\s*$", text)
    if m:
        return m.group(1).strip()
    m = re.match(r"^```(?:json|JSON)?\s*\n?([\s\S]*)$", text)
    if m:
        return m.group(1).strip()
    return text


def _parse_json_maybe(s: str):
    """Best-effort extraction of a JSON object/array from a model response."""
    if not s:
        return None
    s = _strip_code_fences(s)
    if "</tool_response>" in s:
        s = s.split("</tool_response>", 1)[-1]
    s = s.strip()
    if s.startswith("{") and not s.endswith("}"):
        s += "}"
    try:
        return json.loads(s)
    except Exception:
        pass
    m = None
    if "{" in s and "}" in s:
        m = re.search(r"\{[\s\S]*\}", s)
    if m is None and "[" in s and "]" in s:
        m = re.search(r"\[[\s\S]*\]", s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        try:
            fixed = re.sub(r",\s*([}\]])", r"\1", m.group(0))
            return json.loads(fixed)
        except Exception:
            return None


def _stringify_tool_calls(tool_calls: list) -> str:
    """Render an assistant ``tool_calls`` list as a human-readable text block."""
    parts = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if tc.get("type") == "function" else tc
        if not isinstance(fn, dict):
            continue
        name = fn.get("name") or tc.get("name") or "unknown"
        args = fn.get("arguments")
        if isinstance(args, str):
            args_str = args
        else:
            try:
                args_str = json.dumps(args if isinstance(args, dict) else {}, ensure_ascii=True)
            except Exception:
                args_str = "{}"
        call_id = tc.get("id", "")
        header = f"[Tool call: {name}]"
        if call_id:
            header += f" (id={call_id})"
        parts.append(f"{header}\nArguments: {args_str}")
    return "\n\n".join(parts)


# Matches the text format produced by ``_stringify_tool_calls``:
#   [Tool call: NAME] (id=CALL_ID)
#   Arguments: { ... JSON ... }
# The ``(id=...)`` part is optional. The JSON arguments may span multiple
# lines and contain nested braces, so we balance them manually.
_TOOL_CALL_HEADER_RE = re.compile(
    r"\[\s*Tool\s*call\s*:\s*([^\]]+?)\s*\](?:\s*\(id=([^\)]*)\))?\s*\n\s*Arguments\s*:\s*",
    re.IGNORECASE,
)


def _extract_balanced_json(s: str, start: int) -> tuple[str, int]:
    """Return the balanced JSON object/array starting at ``s[start]`` and the
    index just past it. If ``s[start]`` is not ``{`` or ``[``, returns the
    remainder of the line as a best-effort argument string."""
    if start >= len(s):
        return "", start
    open_ch = s[start]
    if open_ch not in "{[":
        # Take until end of line as the argument text.
        end = s.find("\n", start)
        if end == -1:
            end = len(s)
        return s[start:end].strip(), end
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_str = False
    escape = False
    i = start
    while i < len(s):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return s[start:i + 1], i + 1
        i += 1
    # Unbalanced — return everything from start as a best effort.
    return s[start:].strip(), len(s)


def _parse_stringified_tool_calls(text: str) -> list[dict]:
    """Parse the human-readable ``[Tool call: NAME] (id=ID)\\nArguments: {JSON}``
    format back into a list of tool-call dicts.

    This handles the case where a web-API model echoes back the stringified
    tool-call format (the same format produced by ``_stringify_tool_calls``)
    instead of emitting a raw JSON tool-call object. Returns an empty list when
    no stringified tool calls are found.
    """
    calls: list[dict] = []
    if not text:
        return calls
    for m in _TOOL_CALL_HEADER_RE.finditer(text):
        name = (m.group(1) or "").strip()
        call_id = (m.group(2) or "").strip()
        args_start = m.end()
        args_str, _ = _extract_balanced_json(text, args_start)
        if not name:
            continue
        # ``arguments`` may be a JSON object, a JSON string, or plain text.
        arguments: object
        if args_str:
            try:
                arguments = json.loads(args_str)
            except Exception:
                arguments = args_str
        else:
            arguments = {}
        calls.append({
            "name": name,
            "id": call_id or "",
            "arguments": arguments,
        })
    return calls


def _stringify_tool_response(message: dict) -> str:
    """Render a ``role: tool`` message as a human-readable text block."""
    content = message.get("content")
    if isinstance(content, list):
        # Concatenate text parts from a multipart content list.
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif isinstance(part, str):
                text_parts.append(part)
        content = "\n".join(text_parts)
    if content is None:
        content = ""
    tool_call_id = message.get("tool_call_id", "")
    name = message.get("name", "")
    header = "[Tool response"
    if name:
        header += f": {name}"
    if tool_call_id:
        header += f" (id={tool_call_id})"
    header += "]"
    return f"{header}\n{content}"


def _extract_text(content) -> str:
    """Extract plain text from a message ``content`` field (str or multipart list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return ""


def _merge_messages_to_single_user(messages: Messages) -> Messages:
    """Merge all assistant and user messages into a single ``role: user`` message.

    Web-API providers (e.g. Qwen, Copilot) that rely on ``get_last_user_message``
    only forward the last user message to the server when no ``conversation``
    handle is available, dropping all prior context.  Folding the whole history
    into one user message ensures the model still sees the full conversation.
    """
    if not messages:
        return messages
    parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _extract_text(msg.get("content"))
        if not text:
            continue
        header = "User" if role == "user" else "Assistant"
        parts.append(f"[{header}]\n{text}")
    if not parts:
        return messages
    return [{"role": "user", "content": "\n\n".join(parts)}]


def _preprocess_tool_messages(messages: Messages) -> Messages:
    """Convert ``tool_calls`` on assistant messages and ``role: tool`` messages
    into readable text so web-only providers can follow the conversation history.

    Assistant messages keep their textual ``content`` (if any) and get the
    stringified tool calls appended. Tool response messages are rewritten as
    ``role: user`` messages with the stringified response as content.

    ``role: system`` messages are folded into the next ``role: user`` message
    (prefixed with a ``[System]`` header) so providers that only accept
    ``user``/``assistant`` roles still receive the system instructions. If no
    user message follows, the system content is emitted as a user message.
    """
    processed: Messages = []
    pending_system: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            # Flush any pending system text before non-dict entries.
            if pending_system:
                processed.append({"role": "user", "content": "[System]\n" + "\n".join(pending_system)})
                pending_system = []
            processed.append(msg)
            continue
        role = msg.get("role")
        tool_calls = msg.get("tool_calls")
        if role == "system":
            text = _extract_text(msg.get("content"))
            if text:
                pending_system.append(text)
            continue
        if role == "assistant" and isinstance(tool_calls, list) and tool_calls:
            new_msg = {k: v for k, v in msg.items() if k != "tool_calls"}
            text_parts = []
            existing_content = new_msg.get("content")
            if isinstance(existing_content, str) and existing_content.strip():
                text_parts.append(existing_content)
            elif isinstance(existing_content, list):
                for part in existing_content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        t = part.get("text", "")
                        if t:
                            text_parts.append(t)
                    elif isinstance(part, str) and part:
                        text_parts.append(part)
            rendered = _stringify_tool_calls(tool_calls)
            if rendered:
                text_parts.append(rendered)
            new_msg["content"] = "\n\n".join(text_parts) if text_parts else rendered
            # Flush any pending system text before this assistant message.
            if pending_system:
                processed.append({"role": "user", "content": "[System]\n" + "\n".join(pending_system)})
                pending_system = []
            processed.append(new_msg)
        elif role == "tool":
            rendered = _stringify_tool_response(msg)
            # Fold any pending system text into this rewritten user message.
            if pending_system:
                rendered = "[System]\n" + "\n".join(pending_system) + "\n\n" + rendered
                pending_system = []
            processed.append({"role": "user", "content": rendered})
        elif role == "user":
            new_msg = dict(msg)
            if pending_system:
                existing = _extract_text(new_msg.get("content"))
                new_msg["content"] = "[System]\n" + "\n".join(pending_system) + (
                    ("\n\n" + existing) if existing else ""
                )
                pending_system = []
            processed.append(new_msg)
        else:
            # Flush any pending system text before unhandled roles.
            if pending_system:
                processed.append({"role": "user", "content": "[System]\n" + "\n".join(pending_system)})
                pending_system = []
            processed.append(msg)
    # Flush any trailing system text as a user message.
    if pending_system:
        processed.append({"role": "user", "content": "[System]\n" + "\n".join(pending_system)})
    return processed


class ToolSupportProvider(AsyncGeneratorProvider):
    """Emulates OpenAI-style tool calling for providers that only expose web APIs.

    Injects a system prompt instructing the model to emit a JSON tool-call plan,
    delegates to the real provider, parses the JSON response and converts it into
    ``ToolCalls`` + ``FinishReason("tool_calls")`` chunks. ``Reasoning`` / thinking
    chunks emitted by the underlying provider are forwarded to the caller so the
    client still sees the model's reasoning.
    """

    working = True

    @staticmethod
    async def create_async_generator(
        model: str,
        messages: Messages,
        stream: bool = True,
        media: MediaListType = None,
        tools: list = None,
        tool_choice: Optional[Union[str, dict]] = None,
        response_format: dict = None,
        provider: Optional[Union[ProviderType, str]] = None,
        **kwargs,
    ) -> AsyncResult:
        if provider is None and ":" in model:
            provider, model = model.split(":", 1)
        model, provider = get_model_and_provider(
            model, provider, stream, logging=False, has_images=media is not None
        )
        tool_names: list[str] = []
        tool_schemas: dict[str, dict] = {}
        if tools:
            tool_defs = tools if isinstance(tools, list) else []
            for t in tool_defs:
                if not isinstance(t, dict) or t.get("type") != "function":
                    continue
                fn = t.get("function")
                if not isinstance(fn, dict):
                    continue
                name = fn.get("name")
                if not isinstance(name, str) or not name:
                    continue
                tool_names.append(name)
                params = fn.get("parameters")
                if isinstance(params, dict):
                    tool_schemas[name] = params

            if tool_names:
                # Only force JSON output when the caller hasn't requested a
                # specific format. Some web providers reject/ignore this field.
                if response_format is None:
                    response_format = {"type": "json"}

                lines = [
                    *getattr(provider, "tool_support_prompts", []),
                    "You have access to the following tools. When you decide a tool is needed, "
                    "respond with ONLY a valid JSON object (no markdown, no explanation) in this format:",
                    '{"tool_calls": [{"name": "TOOL_NAME", "arguments": {}}]}',
                    "You may include multiple tool calls in the array. The `arguments` value MUST be "
                    "a JSON object matching the tool's parameter schema.",
                    "If no tool is needed, respond normally with plain text. Don't try to call a tool, simply respond only with the JSON object.",
                    f"Available tools: {', '.join(tool_names)}",
                ]
                if tool_schemas:
                    lines.append(
                        "Tool parameter schemas (JSON Schema): "
                        f"{json.dumps(tool_schemas, ensure_ascii=True)}"
                    )
                if tool_choice is not None:
                    if tool_choice == "required":
                        lines.append(
                            "You MUST call at least one tool. Respond with the JSON tool-call object only."
                        )
                    elif tool_choice == "none":
                        lines.append("Do not call any tools. Respond with plain text only.")
                    elif isinstance(tool_choice, dict):
                        fn = tool_choice.get("function") if tool_choice.get("type") == "function" else None
                        if isinstance(fn, dict) and fn.get("name"):
                            lines.append(f"You must call the tool `{fn['name']}`.")
                    else:
                        lines.append(f"Tool choice: {tool_choice}")
                messages = [{"role": "system", "content": "\n".join(lines)}] + messages

        # Rewrite any prior assistant tool_calls and tool-role response messages
        # into readable text so the underlying web provider can follow the
        # conversation history.
        messages = _preprocess_tool_messages(messages)

        # When no conversation handle is provided, web-API providers (e.g. Qwen,
        # Copilot) only forward the last user message to the server via
        # ``get_last_user_message``, dropping all prior context.  Merge the
        # whole history into a single user message so the model still sees the
        # full conversation.
        if kwargs.get("conversation") is None:
            messages = _merge_messages_to_single_user(messages)

        finish = None
        content_chunks: list[str] = []
        has_usage = False

        method = get_async_provider_method(provider)
        async for chunk in method(
            model=model,
            messages=messages,
            stream=stream,
            media=media,
            response_format=response_format,
            **kwargs,
        ):
            if isinstance(chunk, str):
                content_chunks.append(chunk)
            elif isinstance(chunk, Reasoning):
                # Forward thinking output to the client; do not mix it into
                # the content that will be parsed as a tool-call JSON.
                yield chunk
            elif isinstance(chunk, Usage):
                yield chunk
                has_usage = True
            elif isinstance(chunk, FinishReason):
                # Store the finish reason but keep consuming chunks so we don't
                # miss trailing ``Usage`` / ``JsonConversation`` chunks that
                # some providers (e.g. Qwen image-gen) emit *after* the finish.
                finish = chunk
            else:
                yield chunk

        if not has_usage:
            yield Usage(
                completion_tokens=len(content_chunks),
                total_tokens=len(content_chunks),
            )

        content = "".join(content_chunks)

        if tool_names:
            payload = filter_json(content)
            obj = _parse_json_maybe(payload)
            calls = None
            if isinstance(obj, dict) and isinstance(obj.get("tool_calls"), list):
                calls = obj.get("tool_calls")
            elif isinstance(obj, dict) and ("name" in obj or "tool" in obj):
                calls = [obj]
            elif isinstance(obj, list):
                calls = obj

            openai_calls = []
            if isinstance(calls, list):
                idx = -1
                for c in calls:
                    if not isinstance(c, dict):
                        continue
                    name = c.get("name") or c.get("tool")
                    if not isinstance(name, str) or not name or name not in tool_names:
                        continue
                    args = c.get("arguments")
                    if isinstance(args, str):
                        arguments_str = args
                    else:
                        try:
                            arguments_str = json.dumps(
                                args if isinstance(args, dict) else {},
                                ensure_ascii=True,
                            )
                        except Exception:
                            arguments_str = "{}"
                    idx += 1
                    openai_calls.append(
                        {
                            "index": idx,
                            "id": f"call_{idx}",
                            "type": "function",
                            "function": {"name": name, "arguments": arguments_str},
                        }
                    )

            # Fallback: the model may have echoed back the stringified
            # ``[Tool call: NAME] (id=ID)\nArguments: {JSON}`` format that
            # ``_stringify_tool_calls`` produces instead of emitting a raw
            # JSON tool-call object. Parse those too so the conversation can
            # continue with real tool execution.
            if not openai_calls:
                stringified = _parse_stringified_tool_calls(content)
                if stringified:
                    idx = -1
                    for c in stringified:
                        name = c.get("name")
                        if not isinstance(name, str) or not name or name not in tool_names:
                            continue
                        args = c.get("arguments")
                        if isinstance(args, str):
                            arguments_str = args
                        else:
                            try:
                                arguments_str = json.dumps(
                                    args if isinstance(args, dict) else {},
                                    ensure_ascii=True,
                                )
                            except Exception:
                                arguments_str = "{}"
                        idx += 1
                        openai_calls.append(
                            {
                                "index": idx,
                                "id": f"call_{idx}",
                                "type": "function",
                                "function": {"name": name, "arguments": arguments_str},
                            }
                        )

            if openai_calls:
                yield ToolCalls(openai_calls)
                yield FinishReason("tool_calls")
                return

        if content:
            yield content
        if finish is not None:
            yield finish

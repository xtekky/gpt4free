from __future__ import annotations

import json
import re
import inspect
from typing import Optional, Union, Any

from ..typing import AsyncResult, Messages, MediaListType
from ..client.service import get_model_and_provider
from ..client.helper import filter_json
from .types import ProviderType
from .base_provider import AsyncGeneratorProvider, get_async_provider_method
from .response import ToolCalls, FinishReason, Usage, Reasoning, JsonConversation
from ..tools.tool_support import (
    normalize_tool_defs,
    normalize_tool_calls,
    parse_tool_calls_from_text,
)


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
    normalized = normalize_tool_calls(tool_calls)
    parts = []
    for tc in normalized:
        fn = tc.get("function", {})
        name = fn.get("name") or "unknown"
        args_str = fn.get("arguments", "{}")
        call_id = tc.get("id", "")
        header = f"[Tool call: {name}]"
        if call_id:
            header += f" (id={call_id})"
        parts.append(f"{header}\nArguments: {args_str}")
    return "\n\n".join(parts)


# Matches the text format produced by ``_stringify_tool_calls``:
#   [Tool call: NAME] (id=CALL_ID)
#   Arguments: { ... JSON ... }
_TOOL_CALL_HEADER_RE = re.compile(
    r"\[\s*Tool\s*call\s*:\s*([^\]]+?)\s*\](?:\s*\(id=([^\)]*)\))?\s*\n\s*Arguments\s*:\s*",
    re.IGNORECASE,
)


def _extract_balanced_json(s: str, start: int) -> tuple[str, int]:
    """Return the balanced JSON object/array starting at ``s[start]``."""
    if start >= len(s):
        return "", start
    open_ch = s[start]
    if open_ch not in "{[":
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
                    return s[start : i + 1], i + 1
        i += 1
    return s[start:].strip(), len(s)


def _parse_stringified_tool_calls(text: str) -> list[dict]:
    """Parse the human-readable ``[Tool call: NAME] (id=ID)\\nArguments: {JSON}`` format."""
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
        if args_str:
            try:
                arguments = json.loads(args_str)
            except Exception:
                arguments = args_str
        else:
            arguments = {}
        calls.append(
            {
                "name": name,
                "id": call_id or "",
                "arguments": arguments,
            }
        )
    return calls


def _stringify_tool_response(message: dict) -> str:
    """Render a ``role: tool`` or ``role: function`` message as a human-readable text block."""
    content = message.get("content")
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif isinstance(part, str):
                text_parts.append(part)
        content = "\n".join(text_parts)
    if content is None:
        content = ""
    tool_call_id = message.get("tool_call_id") or message.get("id") or ""
    name = message.get("name") or ""
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
    """Merge all assistant and user messages into a single ``role: user`` message."""
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
    """Convert ``tool_calls`` on assistant messages and ``role: tool`` messages into readable text."""
    processed: Messages = []
    pending_system: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            if pending_system:
                processed.append(
                    {
                        "role": "user",
                        "content": "[System]\n" + "\n".join(pending_system),
                    }
                )
                pending_system = []
            processed.append(msg)
            continue

        role = msg.get("role")
        tool_calls = msg.get("tool_calls") or msg.get("function_call")

        content = msg.get("content")
        anthropic_tool_uses = []
        anthropic_tool_results = []
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "tool_use":
                        anthropic_tool_uses.append(part)
                    elif part.get("type") == "tool_result":
                        anthropic_tool_results.append(part)

        if role == "system":
            text = _extract_text(msg.get("content"))
            if text:
                pending_system.append(text)
            continue

        if role == "assistant" and (tool_calls or anthropic_tool_uses):
            calls_to_stringify = tool_calls or anthropic_tool_uses
            new_msg = {k: v for k, v in msg.items() if k not in ("tool_calls", "function_call")}
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
            rendered = _stringify_tool_calls(
                calls_to_stringify if isinstance(calls_to_stringify, list) else [calls_to_stringify]
            )
            if rendered:
                text_parts.append(rendered)
            new_msg["content"] = "\n\n".join(text_parts) if text_parts else rendered
            if pending_system:
                processed.append(
                    {
                        "role": "user",
                        "content": "[System]\n" + "\n".join(pending_system),
                    }
                )
                pending_system = []
            processed.append(new_msg)
        elif role in ("tool", "function") or anthropic_tool_results:
            if anthropic_tool_results:
                rendered_parts = []
                for res in anthropic_tool_results:
                    res_msg = {
                        "name": res.get("name", ""),
                        "tool_call_id": res.get("tool_use_id", ""),
                        "content": res.get("content", ""),
                    }
                    rendered_parts.append(_stringify_tool_response(res_msg))
                rendered = "\n\n".join(rendered_parts)
            else:
                rendered = _stringify_tool_response(msg)

            if pending_system:
                rendered = "[System]\n" + "\n".join(pending_system) + "\n\n" + rendered
                pending_system = []
            processed.append({"role": "user", "content": rendered})
        elif role == "user":
            new_msg = dict(msg)
            if pending_system:
                existing = _extract_text(new_msg.get("content"))
                new_msg["content"] = (
                    "[System]\n"
                    + "\n".join(pending_system)
                    + (("\n\n" + existing) if existing else "")
                )
                pending_system = []
            processed.append(new_msg)
        else:
            if pending_system:
                processed.append(
                    {
                        "role": "user",
                        "content": "[System]\n" + "\n".join(pending_system),
                    }
                )
                pending_system = []
            processed.append(msg)

    if pending_system:
        processed.append(
            {"role": "user", "content": "[System]\n" + "\n".join(pending_system)}
        )
    return processed


class ToolSupportProvider(AsyncGeneratorProvider):
    """Emulates OpenAI-style tool calling for providers that only expose web APIs.

    Injects a system prompt instructing the model to emit a JSON tool-call plan,
    delegates to the real provider, parses the JSON response and converts it into
    ``ToolCalls`` + ``FinishReason("tool_calls")`` chunks.
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

        normalized_tools = normalize_tool_defs(tools) if tools else []
        tool_names: list[str] = [
            t["function"]["name"]
            for t in normalized_tools
            if isinstance(t, dict) and t.get("function", {}).get("name")
        ]

        if tool_names:
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
            for t in normalized_tools:
                fn = t["function"]
                desc = fn.get("description", "")
                tool_str = f"- Tool `{fn['name']}`" + (f": {desc}" if desc else "")
                lines.append(tool_str)
                if fn.get("parameters"):
                    lines.append(
                        f"  Parameter Schema: {json.dumps(fn['parameters'], ensure_ascii=True)}"
                    )

            if tool_choice is not None:
                if tool_choice == "required":
                    lines.append(
                        "You MUST call at least one tool. Respond with the JSON tool-call object only."
                    )
                elif tool_choice == "none":
                    lines.append("Do not call any tools. Respond with plain text only.")
                elif isinstance(tool_choice, dict):
                    fn = (
                        tool_choice.get("function")
                        if tool_choice.get("type") == "function"
                        else None
                    )
                    if isinstance(fn, dict) and fn.get("name"):
                        lines.append(f"You must call the tool `{fn['name']}`.")
                else:
                    lines.append(f"Tool choice: {tool_choice}")
            messages = [{"role": "system", "content": "\n".join(lines)}] + messages

        messages = _preprocess_tool_messages(messages)

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
                yield chunk
            elif isinstance(chunk, Usage):
                yield chunk
                has_usage = True
            elif isinstance(chunk, FinishReason):
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
            parsed_calls = parse_tool_calls_from_text(content, tool_names)
            openai_calls = normalize_tool_calls(parsed_calls)

            if openai_calls:
                yield ToolCalls(openai_calls)
                yield FinishReason("tool_calls")
                return

        if content:
            yield content
        if finish is not None:
            yield finish

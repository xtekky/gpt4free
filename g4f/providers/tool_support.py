from __future__ import annotations

import json
import re
from typing import Optional, Union

from ..typing import AsyncResult, Messages, MediaListType
from ..client.service import get_model_and_provider
from ..client.helper import filter_json
from ..providers.types import ProviderType
from .base_provider import AsyncGeneratorProvider
from .response import ToolCalls, FinishReason, Usage


class ToolSupportProvider(AsyncGeneratorProvider):
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
            # Tool emulation: ask for a tool call plan in strict JSON.
            if response_format is None:
                response_format = {"type": "json"}

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
                lines = [
                    "If you need to use tools, respond with ONLY valid JSON (no markdown).",
                    "Format:",
                    '{"tool_calls": [{"name": "TOOL_NAME", "arguments": {}}]}',
                    "You may include multiple tool calls in the array.",
                    "If no tool is needed, respond normally with plain text.",
                    f"Available tools: {', '.join(tool_names)}",
                ]
                if tool_choice is not None:
                    lines.append(f"Tool choice: {tool_choice}")
                if tool_schemas:
                    lines.append(
                        f"Tool schemas: {json.dumps(tool_schemas, ensure_ascii=True)}"
                    )
                messages = [{"role": "system", "content": "\n".join(lines)}] + messages

        finish = None
        chunks = []
        has_usage = False
        async for chunk in provider.async_create_function(
            model,
            messages,
            stream=stream,
            media=media,
            response_format=response_format,
            **kwargs,
        ):
            if isinstance(chunk, str):
                chunks.append(chunk)
            elif isinstance(chunk, Usage):
                yield chunk
                has_usage = True
            elif isinstance(chunk, FinishReason):
                finish = chunk
                break
            else:
                yield chunk

        if not has_usage:
            yield Usage(completion_tokens=len(chunks), total_tokens=len(chunks))

        chunks = "".join(chunks)

        if tool_names:
            payload = filter_json(chunks)

            def parse_json_maybe(s: str):
                if not s:
                    return None
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
                    return None

            obj = parse_json_maybe(payload)
            calls = None
            if isinstance(obj, dict) and isinstance(obj.get("tool_calls"), list):
                calls = obj.get("tool_calls")
            elif isinstance(obj, dict) and ("name" in obj or "tool" in obj):
                calls = [obj]
            elif isinstance(obj, list):
                calls = obj

            openai_calls = []
            if isinstance(calls, list):
                idx = 0
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
                            "id": f"call_{idx}",
                            "type": "function",
                            "function": {"name": name, "arguments": arguments_str},
                        }
                    )

            if openai_calls:
                yield ToolCalls(openai_calls)
                yield FinishReason("tool_calls")
                return

        if chunks:
            yield chunks
        if finish is not None:
            yield finish

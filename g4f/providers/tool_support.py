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
    """
    Tool emulation provider for models that don't support native tool calling.

    This provider adds system prompts to instruct models to return tool calls
    in JSON format, then parses the response and converts it to OpenAI-compatible
    tool call format.

    Works with ALL providers including free ones (Copilot, OIVSCodeSer2, etc.)
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
            # Tool emulation: ask for a tool call plan in strict JSON.
            if response_format is None:
                response_format = {"type": "json_object"}

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
                # Enhanced system prompt for better tool emulation
                lines = [
                    "You are a tool-calling assistant with access to these tools:",
                    "",
                    "IMPORTANT RULES:",
                    "1. When you need to use a tool, respond with ONLY valid JSON (no markdown, no code blocks).",
                    "2. Use this exact format:",
                    '   {"tool_calls": [{"name": "TOOL_NAME", "arguments": {"param": "value"}}]}',
                    "3. You may include multiple tool calls in the array.",
                    "4. If no tool is needed, respond normally with plain text.",
                    "5. Do not include any explanations when using tools - just the JSON.",
                    "",
                    f"Available tools: {', '.join(tool_names)}",
                ]

                # Add specific instructions for filesystem tools
                fs_tools = {
                    "read_file": "Read file contents. Requires: path",
                    "write_file": "Write content to file. Requires: path, content",
                    "list_directory": "List directory contents. Requires: path",
                    "create_directory": "Create a directory. Requires: path",
                    "delete_file": "Delete a file. Requires: path",
                    "delete_directory": "Delete a directory. Requires: path",
                    "move_file": "Move/rename a file. Requires: source, destination",
                    "copy_file": "Copy a file. Requires: source, destination",
                    "search_files": "Search for files by pattern. Requires: path, pattern",
                    "get_file_info": "Get file metadata. Requires: path",
                    "file_exists": "Check if file exists. Requires: path",
                    "search_in_files": "Search text in files. Requires: path, query",
                    "search_tool": "Search the web. Requires: query",
                    "get_weather": "Get weather. Requires: location",
                }

                # Add tool descriptions
                has_fs_tools = any(name in fs_tools for name in tool_names)
                if has_fs_tools:
                    lines.append("")
                    lines.append("TOOL DESCRIPTIONS:")
                    for name in tool_names:
                        if name in fs_tools:
                            lines.append(f"- {name}: {fs_tools[name]}")

                # Add tool choice instructions
                if tool_choice is not None:
                    if isinstance(tool_choice, dict):
                        tool_name = tool_choice.get("function", {}).get("name")
                        if tool_name:
                            lines.append(f"")
                            lines.append(f"REQUIRED: You MUST use the tool: {tool_name}")
                    elif tool_choice == "required":
                        lines.append("")
                        lines.append("REQUIRED: You MUST use at least one tool.")
                    elif tool_choice == "none":
                        lines.append("")
                        lines.append("CONSTRAINT: Do NOT use any tools.")

                # Add tool schemas for better guidance
                if tool_schemas:
                    lines.append("")
                    lines.append("TOOL SCHEMAS:")
                    for name, schema in list(tool_schemas.items())[:5]:  # Limit to 5 schemas
                        lines.append(f"- {name}: {json.dumps(schema, ensure_ascii=True)}")

                # Add examples for filesystem tools
                if has_fs_tools:
                    lines.append("")
                    lines.append("EXAMPLES:")
                    lines.append('{"tool_calls": [{"name": "list_directory", "arguments": {"path": "/tmp"}}]}')
                    lines.append('{"tool_calls": [{"name": "read_file", "arguments": {"path": "/tmp/test.txt"}}]}')
                    lines.append('{"tool_calls": [{"name": "write_file", "arguments": {"path": "/tmp/test.txt", "content": "Hello"}}]}')

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
                """Try to parse JSON from string, handling various formats"""
                if not s:
                    return None
                try:
                    return json.loads(s)
                except Exception:
                    pass

                # Try to extract JSON from text
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

            def extract_tool_calls(obj):
                """Extract tool calls from various response formats"""
                if not isinstance(obj, (dict, list)):
                    return None

                # Format 1: {"tool_calls": [...]}
                if isinstance(obj, dict) and isinstance(obj.get("tool_calls"), list):
                    return obj.get("tool_calls")

                # Format 2: {"tools": [...]}
                if isinstance(obj, dict) and isinstance(obj.get("tools"), list):
                    return obj.get("tools")

                # Format 3: {"name": "...", "arguments": {...}} (single tool)
                if isinstance(obj, dict) and ("name" in obj or "function" in obj):
                    return [obj]

                # Format 4: [{"name": "...", ...}] (list of tools)
                if isinstance(obj, list):
                    return obj

                # Format 5: {"function": {"name": "...", "arguments": {...}}}
                if isinstance(obj, dict) and "function" in obj:
                    func = obj.get("function", {})
                    if isinstance(func, dict) and "name" in func:
                        return [{"name": func["name"], "arguments": func.get("arguments", {})}]

                return None

            obj = parse_json_maybe(payload)
            calls = extract_tool_calls(obj)

            openai_calls = []
            if isinstance(calls, list):
                idx = 0
                for c in calls:
                    if not isinstance(c, dict):
                        continue

                    # Handle different tool call formats
                    name = c.get("name") or c.get("tool") or c.get("function", {}).get("name")
                    if not isinstance(name, str) or not name or name not in tool_names:
                        continue

                    # Get arguments from various formats
                    args = c.get("arguments")
                    if args is None and "function" in c:
                        args = c.get("function", {}).get("arguments")
                    if args is None and "parameters" in c:
                        args = c.get("parameters")
                    if args is None and "args" in c:
                        args = c.get("args")

                    # Convert arguments to JSON string
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

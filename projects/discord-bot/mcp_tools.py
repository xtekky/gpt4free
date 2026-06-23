"""
MCP tools integration for the Discord bot.

Wraps g4f's MCPServer to:
- Build OpenAI-compatible tool definitions from the registered MCP tools
- Execute tool calls and return results in the OpenAI "tool" message format
- Provide a configurable allowlist so bot owners can enable only safe tools
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from g4f.mcp.server import MCPServer, MCPRequest

log = logging.getLogger("g4f-discord.mcp")

# Tools that are safe to expose to Discord users by default.
# File/python/patch tools are excluded by default since they operate on the
# bot's local ~/.g4f/workspace directory.
SAFE_DEFAULT_TOOLS = {
    "web_search",
    "web_scrape",
    "mark_it_down",
    "text_to_audio",
    "image_generation",
}

# All tools that *can* be enabled (everything the MCPServer registers).
ALL_AVAILABLE_TOOLS = {
    "web_search",
    "web_scrape",
    "image_generation",
    "text_to_audio",
    "mark_it_down",
    "python_execute",
    "apply_patch",
    "file_read",
    "file_read_lines",
    "file_search",
    "file_write",
    "file_list",
    "file_delete",
}


class MCPToolManager:
    """Manage MCP tool definitions, execution, and the tool-call loop."""

    def __init__(self, enabled_tools: Optional[set[str]] = None, safe_mode: bool = True):
        """
        Args:
            enabled_tools: Set of tool names to expose. Defaults to SAFE_DEFAULT_TOOLS.
            safe_mode: Passed to MCPServer. When True, python_execute / file_list
                run in restricted mode.
        """
        self.server = MCPServer(safe_mode=safe_mode)
        self.enabled_tools: set[str] = set(enabled_tools) if enabled_tools else set(SAFE_DEFAULT_TOOLS)
        self._definitions: List[dict] = []
        self._rebuild_definitions()

    # ------------------------------------------------------------------
    # Tool definitions (OpenAI function-calling format)
    # ------------------------------------------------------------------
    def _rebuild_definitions(self) -> None:
        """Build the OpenAI-format tool list from the MCPServer's registered tools."""
        tool_list = self.server.get_tool_list()
        defs: List[dict] = []
        for tool in tool_list:
            name = tool["name"]
            if name not in self.enabled_tools:
                continue
            defs.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool["description"],
                        "parameters": tool["inputSchema"],
                    },
                }
            )
        self._definitions = defs
        log.info("MCP tools enabled: %s", [d["function"]["name"] for d in defs])

    @property
    def definitions(self) -> List[dict]:
        """OpenAI-compatible tool definitions for chat.completions.create(tools=...)."""
        return self._definitions

    @property
    def enabled_names(self) -> List[str]:
        return sorted(self.enabled_tools)

    def enable(self, name: str) -> bool:
        """Enable a tool by name. Returns True if it exists and was enabled."""
        if name in ALL_AVAILABLE_TOOLS and name in self.server.tools:
            self.enabled_tools.add(name)
            self._rebuild_definitions()
            return True
        return False

    def disable(self, name: str) -> bool:
        """Disable a tool by name. Returns True if it was disabled."""
        if name in self.enabled_tools:
            self.enabled_tools.discard(name)
            self._rebuild_definitions()
            return True
        return False

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------
    async def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single MCP tool and return its result dict."""
        if name not in self.server.tools:
            return {"error": f"Tool not found: {name}"}
        if name not in self.enabled_tools:
            return {"error": f"Tool not enabled: {name}"}
        request = MCPRequest(
            jsonrpc="2.0",
            id=0,
            method="tools/call",
            params={"name": name, "arguments": arguments},
        )
        response = await self.server.handle_request(request)
        if response.error:
            return {"error": response.error.get("message", "Unknown MCP error")}
        return response.result or {}

    async def execute_tool_calls(self, tool_calls: List[Any]) -> List[dict]:
        """
        Execute a list of tool calls (from ChatCompletionMessage.tool_calls)
        and return OpenAI-format tool-result messages.

        Each returned dict has:
            role="tool", tool_call_id=..., name=..., content=<json string>
        """
        results: List[dict] = []
        for call in tool_calls:
            fn = call.function
            name = fn.name
            try:
                args = json.loads(fn.arguments) if fn.arguments else {}
            except (json.JSONDecodeError, TypeError):
                args = {}

            log.info("Executing MCP tool: %s(%s)", name, args)
            try:
                result = await self.execute_tool(name, args)
            except Exception as e:
                log.exception("Tool execution failed: %s", name)
                result = {"error": str(e)}

            results.append(
                {
                    "role": "tool",
                    "tool_call_id": getattr(call, "id", "call_0"),
                    "name": name,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Helpers for Discord display
    # ------------------------------------------------------------------
    @staticmethod
    def format_tool_results_for_discord(tool_results: List[dict]) -> str:
        """Render tool results as a concise Discord markdown block."""
        lines: List[str] = []
        for r in tool_results:
            name = r.get("name", "tool")
            content = r.get("content", "")
            try:
                parsed = json.loads(content)
                pretty = json.dumps(parsed, indent=2, ensure_ascii=False, default=str)
            except (json.JSONDecodeError, TypeError):
                pretty = content
            # Truncate to keep Discord messages manageable
            if len(pretty) > 800:
                pretty = pretty[:800] + "\n…(truncated)"
            lines.append(f"🔧 **{name}**\n```json\n{pretty}\n```")
        return "\n".join(lines)

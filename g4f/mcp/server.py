"""MCP Server implementation with stdio and HTTP transports

This module implements a Model Context Protocol (MCP) server that communicates
over standard input/output using JSON-RPC 2.0, or via HTTP POST endpoints.
The server exposes tools for:
- Web search
- Web scraping
- Image generation
"""

from __future__ import annotations

import os
import re
import sys
import json
import asyncio
import hashlib
from email.utils import formatdate
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from urllib.parse import unquote_plus

from ..debug import enable_logging
from ..cookies import read_cookie_files
from ..image import EXTENSIONS_MAP
from ..image.copy_images import get_media_dir, copy_media, get_source_url

from .tools import (
    MarkItDownTool,
    TextToAudioTool,
    WebSearchTool,
    ImageGenerationTool,
    PythonExecuteTool,
    FileReadTool,
    FileListTool,
    FileDeleteTool,
    ApplyPatchTool,
    CreateFileTool,
    FileWriteTool,
    ReplaceStringInFileTool,
    FetchWebpageTool,
    FileSearchGlobTool,
    GrepSearchTool,
    GithubRepoTool,
    GithubTextSearchTool,
)


@dataclass
class MCPRequest:
    """MCP request following JSON-RPC 2.0 format"""

    jsonrpc: str = "2.0"
    id: Optional[Union[int, str]] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    origin: Optional[str] = None
    user_id: Optional[str] = None
    workspace_secret: Optional[str] = None


@dataclass
class MCPResponse:
    """MCP response following JSON-RPC 2.0 format"""

    jsonrpc: str = "2.0"
    id: Optional[Union[int, str]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


class MCPServer:
    """Model Context Protocol server for gpt4free

    This server exposes gpt4free capabilities through the MCP standard,
    allowing AI assistants to utilize web search, scraping, and image generation.
    """

    def __init__(self, safe_mode: bool = False, github_token: Optional[str] = None):
        """Initialize MCP server with available tools

        Args:
            safe_mode: When ``True`` the server starts in safe mode, where
                callers cannot expand the Python sandbox module allowlist and
                listing the workspace root directory is blocked.
        """
        self.safe_mode = safe_mode
        self.github_token = github_token
        self.tools = {
            "web_search": WebSearchTool(),
            "image_generation": ImageGenerationTool(),
            "text_to_audio": TextToAudioTool(),
            "mark_it_down": MarkItDownTool(),
            "python_execute": PythonExecuteTool(safe_mode=safe_mode),
            "apply_patch": ApplyPatchTool(),
            "file_read": FileReadTool(),
            "file_list": FileListTool(safe_mode=safe_mode),
            "file_delete": FileDeleteTool(),
            "create_file": CreateFileTool(),
            "file_write": FileWriteTool(),
            "replace_string_in_file": ReplaceStringInFileTool(),
            "fetch_webpage": FetchWebpageTool(),
            "file_search_glob": FileSearchGlobTool(),
            "grep_search": GrepSearchTool(),
            "github_repo": GithubRepoTool(False, self.github_token),
            "github_text_search": GithubTextSearchTool(False, self.github_token),
        }
        self.server_info = {
            "name": "gpt4free-mcp-server",
            "version": "1.0.0",
            "description": (
                "MCP server providing web search, scraping, image generation, "
                "safe Python execution, and workspace file management capabilities"
            ),
        }

    def get_tool_list(self) -> List[Dict[str, Any]]:
        """Get list of available tools with their schemas"""
        tool_list = []
        for name, tool in self.tools.items():
            tool_list.append(
                {
                    "name": name,
                    "description": tool.description,
                    "inputSchema": tool.input_schema,
                }
            )
        return tool_list

    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """Handle incoming MCP request"""
        try:
            method = request.method
            params = request.params or {}

            # Handle MCP protocol methods
            if method == "initialize":
                tool_list = self.get_tool_list()
                result = {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": self.server_info,
                    "capabilities": {
                        "tools": {tool["name"]: tool for tool in tool_list}
                    },
                }
                return MCPResponse(jsonrpc="2.0", id=request.id, result=result)

            elif method == "tools/list":
                result = {"tools": self.get_tool_list()}
                return MCPResponse(jsonrpc="2.0", id=request.id, result=result)

            elif method == "tools/call":
                tool_name = params.get("name")
                tool_arguments = params.get("arguments", {})
                tool_arguments.setdefault("origin", request.origin)
                # Pass through user identity and workspace secret so file tools
                # can resolve per-user workspace paths.
                if request.user_id:
                    tool_arguments.setdefault("user_id", request.user_id)
                if request.workspace_secret:
                    tool_arguments.setdefault("workspace_secret", request.workspace_secret)

                if tool_name not in self.tools:
                    return MCPResponse(
                        jsonrpc="2.0",
                        id=request.id,
                        error={
                            "code": -32601,
                            "message": f"Tool not found: {tool_name}",
                        },
                    )

                tool = self.tools[tool_name]
                result = await tool.execute(tool_arguments)

                return MCPResponse(jsonrpc="2.0", id=request.id, result=result)

            elif method == "ping":
                return MCPResponse(jsonrpc="2.0", id=request.id, result={})

            else:
                return MCPResponse(
                    jsonrpc="2.0",
                    id=request.id,
                    error={"code": -32601, "message": f"Method not found: {method}"},
                )

        except Exception as e:
            return MCPResponse(
                jsonrpc="2.0",
                id=request.id,
                error={"code": -32603, "message": f"Internal error: {str(e)}"},
            )

    async def run(self):
        """Run the MCP server with stdio transport"""
        real_stdout = sys.stdout
        sys.stdout = sys.stderr

        try:
            # Write server info to stderr for debugging
            sys.stderr.write(
                f"Starting {self.server_info['name']} v{self.server_info['version']}\n"
            )
            sys.stderr.flush()

            while True:
                try:
                    # Read line from stdin
                    line = await asyncio.get_event_loop().run_in_executor(
                        None, sys.stdin.readline
                    )

                    if not line:
                        break

                    # Parse JSON-RPC request
                    request_data = json.loads(line)
                    request = MCPRequest(
                        jsonrpc=request_data.get("jsonrpc", "2.0"),
                        id=request_data.get("id"),
                        method=request_data.get("method"),
                        params=request_data.get("params"),
                    )

                    # Handle request
                    response = await self.handle_request(request)

                    # Write response to protocol stdout
                    response_dict = {"jsonrpc": response.jsonrpc, "id": response.id}
                    if response.result is not None:
                        response_dict["result"] = response.result
                    if response.error is not None:
                        response_dict["error"] = response.error

                    real_stdout.write(json.dumps(response_dict) + "\n")
                    real_stdout.flush()

                except json.JSONDecodeError as e:
                    sys.stderr.write(f"JSON decode error: {e}\n")
                    sys.stderr.flush()
                except Exception as e:
                    sys.stderr.write(f"Error: {e}\n")
                    sys.stderr.flush()
        finally:
            sys.stdout = real_stdout

    async def run_http(
        self, host: str = "0.0.0.0", port: int = 8765, origin: Optional[str] = None
    ):
        """Run the MCP server with HTTP transport

        Args:
            host: Host to bind the HTTP server to
            port: Port to bind the HTTP server to
        """
        try:
            from aiohttp import web
        except ImportError:
            sys.stderr.write("Error: aiohttp is required for HTTP transport\n")
            sys.stderr.write("Install it with: pip install aiohttp\n")
            sys.exit(1)

        enable_logging()
        read_cookie_files()

        async def handle_mcp_request(request: web.Request) -> web.Response:
            nonlocal origin
            """Handle MCP JSON-RPC request over HTTP POST"""
            try:
                # Parse JSON-RPC request from POST body
                request_data = await request.json()
                if origin is None:
                    origin = request.headers.get("origin")

                mcp_request = MCPRequest(
                    jsonrpc=request_data.get("jsonrpc", "2.0"),
                    id=request_data.get("id"),
                    method=request_data.get("method"),
                    params=request_data.get("params"),
                    origin=origin,
                    user_id=request.headers.get("x-user-id"),
                    workspace_secret=request.headers.get("x-workspace-secret"),
                )

                # Handle request
                response = await self.handle_request(mcp_request)

                # Build response dict
                response_dict = {"jsonrpc": response.jsonrpc, "id": response.id}
                if response.result is not None:
                    response_dict["result"] = response.result
                if response.error is not None:
                    response_dict["error"] = response.error

                return web.json_response(
                    response_dict, headers={"access-control-allow-origin": "*"}
                )

            except json.JSONDecodeError as e:
                return web.json_response(
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": f"Parse error: {str(e)}"},
                    },
                    status=400,
                )
            except Exception as e:
                return web.json_response(
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32603,
                            "message": f"Internal error: {str(e)}",
                        },
                    },
                    status=500,
                )

        async def handle_health(request: web.Request) -> web.Response:
            """Health check endpoint"""
            return web.json_response({"status": "ok", "server": self.server_info})

        async def handle_media(request: web.Request) -> web.Response:
            """Serve media files from generated_media directory"""
            filename = request.match_info.get("filename", "")
            if not filename:
                return web.Response(status=404, text="File not found")

            def get_timestamp(s):
                m = re.match("^[0-9]+", s)
                return int(m.group(0)) if m else 0

            media_dir = os.path.realpath(get_media_dir())
            clean_name = secure_filename(os.path.basename(filename))
            if not clean_name:
                return web.Response(status=404, text="File not found")

            target = os.path.realpath(os.path.join(media_dir, clean_name))
            if not target.startswith(media_dir + os.sep):
                return web.Response(status=403, text="Access denied")

            # Try URL-decoded filename if not found
            if not os.path.isfile(target):
                decoded_name = secure_filename(os.path.basename(unquote_plus(filename)))
                if decoded_name:
                    candidate = os.path.realpath(os.path.join(media_dir, decoded_name))
                    if candidate.startswith(media_dir + os.sep) and os.path.isfile(candidate):
                        target = candidate

            # Get file extension and mime type
            ext = os.path.splitext(filename)[1][1:].lower()
            mime_type = EXTENSIONS_MAP.get(ext, "application/octet-stream")

            # Try to fetch from backend if file doesn't exist
            if not os.path.isfile(target) and mime_type != "application/octet-stream":
                source_url = get_source_url(str(request.query_string))
                ssl = None
                if source_url is not None:
                    try:
                        await copy_media([source_url], target=target, ssl=ssl)
                        sys.stderr.write(f"File copied from {source_url}\n")
                    except Exception as e:
                        sys.stderr.write(f"Download failed: {source_url} - {e}\n")
                        return web.Response(status=404, text="File not found")

            if not os.path.isfile(target):
                return web.Response(status=404, text="File not found")

            # Build response headers
            stat_result = os.stat(target)
            headers = {
                "cache-control": "public, max-age=31536000",
                "last-modified": formatdate(get_timestamp(filename), usegmt=True),
                "etag": f'"{hashlib.md5(filename.encode()).hexdigest()}"',
                "content-length": str(stat_result.st_size),
                "content-type": mime_type,
                "access-control-allow-origin": "*",
            }

            # Check for conditional request
            if_none_match = request.headers.get("if-none-match")
            if if_none_match:
                etag = headers["etag"]
                if etag in [tag.strip(" W/") for tag in if_none_match.split(",")]:
                    return web.Response(status=304, headers=headers)

            # Serve the file
            return web.FileResponse(target, headers=headers)

        async def handle_synthesize(request: web.Request) -> web.Response:
            """Handle synthesize requests for text-to-speech"""
            provider_name = request.match_info.get("provider", "")
            if not provider_name:
                return web.Response(status=400, text="Provider not specified")

            try:
                from ..Provider import ProviderUtils

                provider_handler = ProviderUtils.convert.get(provider_name)
                if provider_handler is None:
                    return web.Response(
                        status=404, text=f"Provider not found: {provider_name}"
                    )
            except Exception as e:
                return web.Response(
                    status=404, text=f"Provider not found: {provider_name}"
                )

            if not hasattr(provider_handler, "synthesize"):
                return web.Response(
                    status=500,
                    text=f"Provider doesn't support synthesize: {provider_name}",
                )

            # Get query parameters
            params = dict(request.query)

            try:
                # Call the synthesize method
                response_data = provider_handler.synthesize(params)

                # Handle async generator
                async def generate():
                    async for chunk in response_data:
                        yield chunk

                content_type = getattr(
                    provider_handler,
                    "synthesize_content_type",
                    "application/octet-stream",
                )
                return web.Response(
                    body=b"".join([chunk async for chunk in generate()]),
                    content_type=content_type,
                    headers={
                        "cache-control": "max-age=604800",
                        "access-control-allow-origin": "*",
                    },
                )
            except Exception as e:
                sys.stderr.write(f"Synthesize error: {e}\n")
                return web.Response(status=500, text="Synthesize error: An internal error occurred")

        _WORKSPACE_SAFE_TYPES: Dict[str, str] = {
            "html": "text/html; charset=utf-8",
            "htm": "text/html; charset=utf-8",
            "css": "text/css; charset=utf-8",
            "js": "application/javascript; charset=utf-8",
            "mjs": "application/javascript; charset=utf-8",
            "json": "application/json; charset=utf-8",
            "txt": "text/plain; charset=utf-8",
            "md": "text/markdown; charset=utf-8",
            "svg": "image/svg+xml",
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "gif": "image/gif",
            "webp": "image/webp",
            "ico": "image/x-icon",
            "woff": "font/woff",
            "woff2": "font/woff2",
            "ttf": "font/ttf",
            "otf": "font/otf",
        }

        async def handle_pa_providers(request: web.Request) -> web.Response:
            """List all PA providers from workspace."""
            from .pa_provider import get_pa_registry

            providers = get_pa_registry().list_providers()
            return web.json_response(
                providers, headers={"access-control-allow-origin": "*"}
            )

        async def handle_pa_file(request: web.Request) -> web.Response:
            """Securely serve a workspace file for browser rendering.

            Only files within ``~/.g4f/workspace`` are served.  Path traversal
            is blocked.  Only the MIME types in ``_WORKSPACE_SAFE_TYPES`` are
            allowed — ``.py``, ``.env``, and other sensitive types return 403.
            HTML files are served with a ``Content-Security-Policy: sandbox``
            header so they run in an isolated null origin.

            When ``x-user-id`` and ``x-workspace-secret`` headers are present,
            files are resolved from the user's secret workspace first, falling
            back to the root workspace.
            """
            from .pa_provider import resolve_workspace_path

            file_path = request.match_info.get("file_path", "")
            user_id = request.headers.get("x-user-id", "")
            workspace_secret = request.headers.get("x-workspace-secret", "")
            try:
                resolved, workspace = resolve_workspace_path(
                    file_path, user_id=user_id, workspace_secret=workspace_secret, for_write=False
                )
                # Security: ensure the resolved path is still inside the workspace directory
                resolved.relative_to(workspace)
            except (ValueError, Exception):
                return web.Response(status=403, text="Path traversal is not allowed")

            if not resolved.exists() or not resolved.is_file():
                return web.Response(status=404, text=f"File not found: {file_path}")

            ext = resolved.suffix.lstrip(".").lower()
            mime = _WORKSPACE_SAFE_TYPES.get(ext)
            if mime is None:
                return web.Response(status=403, text=f"File type not allowed: .{ext}")

            content = resolved.read_bytes()
            headers: Dict[str, str] = {"access-control-allow-origin": "*"}
            if ext in ("html", "htm"):
                req_origin = f"{request.scheme}://{request.host}"
                headers["content-security-policy"] = (
                    f"sandbox allow-scripts allow-forms allow-popups allow-same-origin; "
                    f"default-src {req_origin} https://g4f.space; "
                    f"img-src {req_origin} data: blob: https:; "
                    f"media-src {req_origin} blob: https:; "
                    f"font-src {req_origin} https:; "
                    f"style-src {req_origin} 'unsafe-inline' https:; "
                    f"script-src {req_origin} 'unsafe-inline' https:; "
                    f"connect-src {req_origin} https: wss:; "
                    f"frame-src {req_origin}"
                )
            return web.Response(
                body=content, content_type=mime.split(";")[0].strip(), headers=headers
            )

        async def handle_secret_conversations_list(request: web.Request) -> web.Response:
            """List all secret conversations for the authenticated user."""
            from .pa_provider import list_secret_conversations

            user_id = request.headers.get("x-user-id", "")
            if not user_id:
                return web.json_response(
                    {"error": "User ID is required"}, status=401
                )
            return web.json_response(
                {"conversations": list_secret_conversations(user_id)},
                headers={"access-control-allow-origin": "*"},
            )

        async def handle_secret_conversations_save(request: web.Request) -> web.Response:
            """Save a conversation to the user's secret workspace."""
            from .pa_provider import save_secret_conversation

            user_id = request.headers.get("x-user-id", "")
            if not user_id:
                return web.json_response(
                    {"error": "User ID is required"}, status=401
                )
            workspace_secret = request.headers.get("x-workspace-secret", "")
            try:
                body = await request.json()
            except Exception:
                return web.json_response({"error": "Invalid JSON body"}, status=400)
            result = save_secret_conversation(user_id, body, workspace_secret or None)
            return web.json_response(
                result, headers={"access-control-allow-origin": "*"}
            )

        async def handle_secret_conversations_sync(request: web.Request) -> web.Response:
            """Sync (upload) multiple conversations to the user's secret workspace."""
            from .pa_provider import save_secret_conversation

            user_id = request.headers.get("x-user-id", "")
            if not user_id:
                return web.json_response(
                    {"error": "User ID is required"}, status=401
                )
            workspace_secret = request.headers.get("x-workspace-secret", "")
            try:
                body = await request.json()
            except Exception:
                return web.json_response({"error": "Invalid JSON body"}, status=400)
            conversations = body.get("conversations", []) if isinstance(body, dict) else body
            saved = 0
            errors = []
            for conv in conversations:
                res = save_secret_conversation(user_id, conv, workspace_secret or None)
                if res.get("saved"):
                    saved += 1
                else:
                    errors.append(res.get("error", "Unknown error"))
            return web.json_response(
                {"saved": saved, "errors": errors},
                headers={"access-control-allow-origin": "*"},
            )

        async def handle_secret_conversation_get(request: web.Request) -> web.Response:
            """Retrieve a single secret conversation by ID."""
            from .pa_provider import get_secret_conversation

            user_id = request.headers.get("x-user-id", "")
            if not user_id:
                return web.json_response(
                    {"error": "User ID is required"}, status=401
                )
            workspace_secret = request.headers.get("x-workspace-secret", "")
            conv_id = request.match_info.get("conversation_id", "")
            conv = get_secret_conversation(user_id, conv_id, workspace_secret or None)
            if conv is None:
                return web.json_response(
                    {"error": f"Conversation '{conv_id}' not found"}, status=404
                )
            return web.json_response(conv, headers={"access-control-allow-origin": "*"})

        async def handle_secret_conversation_delete(request: web.Request) -> web.Response:
            """Delete a secret conversation by ID."""
            from .pa_provider import delete_secret_conversation

            user_id = request.headers.get("x-user-id", "")
            if not user_id:
                return web.json_response(
                    {"error": "User ID is required"}, status=401
                )
            conv_id = request.match_info.get("conversation_id", "")
            deleted = delete_secret_conversation(user_id, conv_id)
            if not deleted:
                return web.json_response(
                    {"error": f"Conversation '{conv_id}' not found"}, status=404
                )
            return web.json_response(
                {"deleted": True, "id": conv_id},
                headers={"access-control-allow-origin": "*"},
            )

        # ── Cross-device workspace secret sharing ───────────────────────────

        async def handle_secret_request_create(request: web.Request) -> web.Response:
            """Create a pending secret-sharing request from a new device."""
            from .pa_provider import create_secret_request

            user_id = request.headers.get("x-user-id", "")
            if not user_id:
                return web.json_response(
                    {"error": "User ID is required"}, status=401
                )
            device_name = ""
            try:
                body = await request.json()
                device_name = body.get("device_name", "") if isinstance(body, dict) else ""
            except Exception:
                pass
            result = create_secret_request(user_id, device_name)
            return web.json_response(
                result, headers={"access-control-allow-origin": "*"}
            )

        async def handle_secret_request_list(request: web.Request) -> web.Response:
            """List pending secret-sharing requests (for the online device to confirm)."""
            from .pa_provider import list_secret_requests

            user_id = request.headers.get("x-user-id", "")
            if not user_id:
                return web.json_response(
                    {"error": "User ID is required"}, status=401
                )
            requests_list = list_secret_requests(user_id)
            return web.json_response(
                {"requests": requests_list},
                headers={"access-control-allow-origin": "*"},
            )

        async def handle_secret_request_confirm(request: web.Request) -> web.Response:
            """Confirm a secret request by sending the workspace secret to the new device."""
            from .pa_provider import confirm_secret_request

            user_id = request.headers.get("x-user-id", "")
            if not user_id:
                return web.json_response(
                    {"error": "User ID is required"}, status=401
                )
            try:
                body = await request.json()
            except Exception:
                return web.json_response({"error": "Invalid JSON body"}, status=400)
            request_id = body.get("request_id", "")
            workspace_secret = body.get("workspace_secret", "")
            if not request_id or not workspace_secret:
                return web.json_response(
                    {"error": "request_id and workspace_secret are required"}, status=400
                )
            result = confirm_secret_request(user_id, request_id, workspace_secret)
            if "error" in result:
                return web.json_response(result, status=404)
            return web.json_response(
                result, headers={"access-control-allow-origin": "*"}
            )

        async def handle_secret_request_poll(request: web.Request) -> web.Response:
            """Poll a secret request to check if it has been confirmed."""
            from .pa_provider import poll_secret_request

            user_id = request.headers.get("x-user-id", "")
            if not user_id:
                return web.json_response(
                    {"error": "User ID is required"}, status=401
                )
            request_id = request.match_info.get("request_id", "")
            result = poll_secret_request(user_id, request_id)
            return web.json_response(
                result, headers={"access-control-allow-origin": "*"}
            )

        async def handle_secret_request_delete(request: web.Request) -> web.Response:
            """Cancel / delete a secret-sharing request."""
            from .pa_provider import delete_secret_request

            user_id = request.headers.get("x-user-id", "")
            if not user_id:
                return web.json_response(
                    {"error": "User ID is required"}, status=401
                )
            request_id = request.match_info.get("request_id", "")
            deleted = delete_secret_request(user_id, request_id)
            if not deleted:
                return web.json_response(
                    {"error": "Request not found"}, status=404
                )
            return web.json_response(
                {"deleted": True, "request_id": request_id},
                headers={"access-control-allow-origin": "*"},
            )

        # Create aiohttp application
        app = web.Application()
        app.router.add_options(
            "/mcp",
            lambda request: web.Response(
                headers={
                    "access-control-allow-origin": "*",
                    "access-control-allow-methods": "POST, OPTIONS",
                    "access-control-allow-headers": "Content-Type",
                }
            ),
        )
        app.router.add_post("/mcp", handle_mcp_request)
        app.router.add_get("/health", handle_health)
        app.router.add_get("/media/{filename:.*}", handle_media)
        app.router.add_get("/backend-api/v2/synthesize/{provider}", handle_synthesize)
        app.router.add_get("/pa/providers", handle_pa_providers)
        app.router.add_get("/pa/files/{file_path:.*}", handle_pa_file)
        app.router.add_get("/v1/secret/conversations", handle_secret_conversations_list)
        app.router.add_post("/v1/secret/conversations", handle_secret_conversations_save)
        app.router.add_post("/v1/secret/conversations/sync", handle_secret_conversations_sync)
        app.router.add_get("/v1/secret/conversations/{conversation_id}", handle_secret_conversation_get)
        app.router.add_delete("/v1/secret/conversations/{conversation_id}", handle_secret_conversation_delete)
        app.router.add_post("/v1/secret/request", handle_secret_request_create)
        app.router.add_get("/v1/secret/requests", handle_secret_request_list)
        app.router.add_post("/v1/secret/request/confirm", handle_secret_request_confirm)
        app.router.add_get("/v1/secret/request/{request_id}", handle_secret_request_poll)
        app.router.add_delete("/v1/secret/request/{request_id}", handle_secret_request_delete)

        # Start server
        sys.stderr.write(
            f"Starting {self.server_info['name']} v{self.server_info['version']} (HTTP mode)\n"
        )
        sys.stderr.write(f"Listening on http://{host}:{port}\n")
        sys.stderr.write(f"MCP endpoint: http://{host}:{port}/mcp\n")
        sys.stderr.write(f"Health check: http://{host}:{port}/health\n")
        sys.stderr.write(f"Media files: http://{host}:{port}/media/{{filename}}\n")
        sys.stderr.write(f"PA providers: http://{host}:{port}/pa/providers\n")
        sys.stderr.write(f"PA files: http://{host}:{port}/pa/files/{{path}}\n")
        sys.stderr.flush()

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()

        # Keep server running
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            sys.stderr.write("\nShutting down HTTP server...\n")
            sys.stderr.flush()
        finally:
            await runner.cleanup()


def main(
    http: bool = False,
    host: str = "0.0.0.0",
    port: int = 8765,
    origin: Optional[str] = None,
    safe: bool = False,
):
    """Main entry point for MCP server

    Args:
        http: If True, use HTTP transport instead of stdio
        host: Host to bind HTTP server to (only used when http=True)
        port: Port to bind HTTP server to (only used when http=True)
        safe: If True, start in safe mode — callers cannot override the module
            allowlist for Python execution and workspace root listing is blocked.
    """
    server = MCPServer(safe_mode=safe, github_token=os.environ.get("GITHUB_TOKEN"))
    if http:
        asyncio.run(server.run_http(host, port, origin))
    else:
        asyncio.run(server.run())


if __name__ == "__main__":
    main()

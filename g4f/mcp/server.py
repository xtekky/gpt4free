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

from .tools import MarkItDownTool, TextToAudioTool, WebSearchTool, WebScrapeTool, ImageGenerationTool
from .tools import WebSearchTool, WebScrapeTool, ImageGenerationTool


@dataclass
class MCPRequest:
    """MCP request following JSON-RPC 2.0 format"""
    jsonrpc: str = "2.0"
    id: Optional[Union[int, str]] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    origin: Optional[str] = None


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
    
    def __init__(self):
        """Initialize MCP server with available tools"""
        self.tools = {
            'web_search': WebSearchTool(),
            'web_scrape': WebScrapeTool(),
            'image_generation': ImageGenerationTool(),
            'text_to_audio': TextToAudioTool(),
            'mark_it_down': MarkItDownTool()
        }
        self.server_info = {
            "name": "gpt4free-mcp-server",
            "version": "1.0.0",
            "description": "MCP server providing web search, scraping, and image generation capabilities"
        }
        
    def get_tool_list(self) -> List[Dict[str, Any]]:
        """Get list of available tools with their schemas"""
        tool_list = []
        for name, tool in self.tools.items():
            tool_list.append({
                "name": name,
                "description": tool.description,
                "inputSchema": tool.input_schema
            })
        return tool_list
    
    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """Handle incoming MCP request"""
        try:
            method = request.method
            params = request.params or {}
            
            # Handle MCP protocol methods
            if method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": self.server_info,
                    "capabilities": {
                        "tools": {}
                    }
                }
                return MCPResponse(jsonrpc="2.0", id=request.id, result=result)
            
            elif method == "tools/list":
                result = {
                    "tools": self.get_tool_list()
                }
                return MCPResponse(jsonrpc="2.0", id=request.id, result=result)
            
            elif method == "tools/call":
                tool_name = params.get("name")
                tool_arguments = params.get("arguments", {})
                tool_arguments.setdefault("origin", request.origin)
                
                if tool_name not in self.tools:
                    return MCPResponse(
                        jsonrpc="2.0",
                        id=request.id,
                        error={
                            "code": -32601,
                            "message": f"Tool not found: {tool_name}"
                        }
                    )
                
                tool = self.tools[tool_name]
                result = await tool.execute(tool_arguments)
                
                return MCPResponse(
                    jsonrpc="2.0",
                    id=request.id,
                    result={
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, indent=2)
                            }
                        ]
                    }
                )
            
            elif method == "ping":
                return MCPResponse(jsonrpc="2.0", id=request.id, result={})
            
            else:
                return MCPResponse(
                    jsonrpc="2.0",
                    id=request.id,
                    error={
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                )
        
        except Exception as e:
            return MCPResponse(
                jsonrpc="2.0",
                id=request.id,
                error={
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            )
    
    async def run(self):
        """Run the MCP server with stdio transport"""
        # Write server info to stderr for debugging
        sys.stderr.write(f"Starting {self.server_info['name']} v{self.server_info['version']}\n")
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
                
                # Write response to stdout
                response_dict = {
                    "jsonrpc": response.jsonrpc,
                    "id": response.id
                }
                if response.result is not None:
                    response_dict["result"] = response.result
                if response.error is not None:
                    response_dict["error"] = response.error
                    
                sys.stdout.write(json.dumps(response_dict) + "\n")
                sys.stdout.flush()
                
            except json.JSONDecodeError as e:
                sys.stderr.write(f"JSON decode error: {e}\n")
                sys.stderr.flush()
            except Exception as e:
                sys.stderr.write(f"Error: {e}\n")
                sys.stderr.flush()
    
    async def run_http(self, host: str = "0.0.0.0", port: int = 8765, origin: Optional[str] = None):
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
                    origin=origin
                )
                
                # Handle request
                response = await self.handle_request(mcp_request)
                
                # Build response dict
                response_dict = {
                    "jsonrpc": response.jsonrpc,
                    "id": response.id
                }
                if response.result is not None:
                    response_dict["result"] = response.result
                if response.error is not None:
                    response_dict["error"] = response.error
                
                return web.json_response(response_dict, headers={"access-control-allow-origin": "*"})
                
            except json.JSONDecodeError as e:
                return web.json_response({
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": f"Parse error: {str(e)}"
                    }
                }, status=400)
            except Exception as e:
                return web.json_response({
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32603,
                        "message": f"Internal error: {str(e)}"
                    }
                }, status=500)
        
        async def handle_health(request: web.Request) -> web.Response:
            """Health check endpoint"""
            return web.json_response({
                "status": "ok",
                "server": self.server_info
            })
        
        async def handle_media(request: web.Request) -> web.Response:
            """Serve media files from generated_media directory"""
            filename = request.match_info.get('filename', '')
            if not filename:
                return web.Response(status=404, text="File not found")
            
            def get_timestamp(s):
                m = re.match("^[0-9]+", s)
                return int(m.group(0)) if m else 0
            
            target = os.path.join(get_media_dir(), os.path.basename(filename))
            
            # Try URL-decoded filename if not found
            if not os.path.isfile(target):
                other_name = os.path.join(get_media_dir(), os.path.basename(unquote_plus(filename)))
                if os.path.isfile(other_name):
                    target = other_name
            
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
                        raise web.HTTPFound(location=source_url)
            
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
            provider_name = request.match_info.get('provider', '')
            if not provider_name:
                return web.Response(status=400, text="Provider not specified")
            
            try:
                from ..Provider import ProviderUtils
                provider_handler = ProviderUtils.convert.get(provider_name)
                if provider_handler is None:
                    return web.Response(status=404, text=f"Provider not found: {provider_name}")
            except Exception as e:
                return web.Response(status=404, text=f"Provider not found: {provider_name}")
            
            if not hasattr(provider_handler, "synthesize"):
                return web.Response(status=500, text=f"Provider doesn't support synthesize: {provider_name}")
            
            # Get query parameters
            params = dict(request.query)
            
            try:
                # Call the synthesize method
                response_data = provider_handler.synthesize(params)
                
                # Handle async generator
                async def generate():
                    async for chunk in response_data:
                        yield chunk
                
                content_type = getattr(provider_handler, "synthesize_content_type", "application/octet-stream")
                return web.Response(
                    body=b"".join([chunk async for chunk in generate()]),
                    content_type=content_type,
                    headers={
                        "cache-control": "max-age=604800",
                        "access-control-allow-origin": "*",
                    }
                )
            except Exception as e:
                sys.stderr.write(f"Synthesize error: {e}\n")
                return web.Response(status=500, text=f"Synthesize error: {str(e)}")
        
        # Create aiohttp application
        app = web.Application()
        app.router.add_options('/mcp', lambda request: web.Response(headers={"access-control-allow-origin": "*", "access-control-allow-methods": "POST, OPTIONS", "access-control-allow-headers": "Content-Type"}))
        app.router.add_post('/mcp', handle_mcp_request)
        app.router.add_get('/health', handle_health)
        app.router.add_get('/media/{filename:.*}', handle_media)
        app.router.add_get('/backend-api/v2/synthesize/{provider}', handle_synthesize)
        
        # Start server
        sys.stderr.write(f"Starting {self.server_info['name']} v{self.server_info['version']} (HTTP mode)\n")
        sys.stderr.write(f"Listening on http://{host}:{port}\n")
        sys.stderr.write(f"MCP endpoint: http://{host}:{port}/mcp\n")
        sys.stderr.write(f"Health check: http://{host}:{port}/health\n")
        sys.stderr.write(f"Media files: http://{host}:{port}/media/{{filename}}\n")
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


def main(http: bool = False, host: str = "0.0.0.0", port: int = 8765, origin: Optional[str] = None):
    """Main entry point for MCP server
    
    Args:
        http: If True, use HTTP transport instead of stdio
        host: Host to bind HTTP server to (only used when http=True)
        port: Port to bind HTTP server to (only used when http=True)
    """
    server = MCPServer()
    if http:
        asyncio.run(server.run_http(host, port, origin))
    else:
        asyncio.run(server.run())


if __name__ == "__main__":
    main()

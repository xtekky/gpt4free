"""MCP Server implementation using stdio transport

This module implements a Model Context Protocol (MCP) server that communicates
over standard input/output using JSON-RPC 2.0. The server exposes tools for:
- Web search
- Web scraping
- Image generation
"""

from __future__ import annotations

import sys
import json
import asyncio
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict

from .tools import WebSearchTool, WebScrapeTool, ImageGenerationTool


@dataclass
class MCPRequest:
    """MCP request following JSON-RPC 2.0 format"""
    jsonrpc: str = "2.0"
    id: Optional[Union[int, str]] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


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
                    params=request_data.get("params")
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


def main():
    """Main entry point for MCP server"""
    server = MCPServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()

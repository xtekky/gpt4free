from __future__ import annotations

import json
import unittest

from g4f.mcp.server import MCPServer, MCPRequest
from g4f.mcp.tools import WebSearchTool, WebScrapeTool, ImageGenerationTool

try:
    from ddgs import DDGS, DDGSError
    from bs4 import BeautifulSoup
    has_requirements = True
except ImportError:
    has_requirements = False


class TestMCPServer(unittest.IsolatedAsyncioTestCase):
    """Test cases for MCP server"""
    
    async def test_server_initialization(self):
        """Test that server initializes correctly"""
        server = MCPServer()
        self.assertIsNotNone(server)
        self.assertEqual(server.server_info["name"], "gpt4free-mcp-server")
        self.assertEqual(len(server.tools), 5)
        self.assertIn('web_search', server.tools)
        self.assertIn('web_scrape', server.tools)
        self.assertIn('image_generation', server.tools)
    
    async def test_initialize_request(self):
        """Test initialize method"""
        server = MCPServer()
        request = MCPRequest(
            jsonrpc="2.0",
            id=1,
            method="initialize",
            params={}
        )
        response = await server.handle_request(request)
        self.assertEqual(response.jsonrpc, "2.0")
        self.assertEqual(response.id, 1)
        self.assertIsNotNone(response.result)
        self.assertEqual(response.result["protocolVersion"], "2024-11-05")
        self.assertIn("serverInfo", response.result)
    
    async def test_tools_list(self):
        """Test tools/list method"""
        server = MCPServer()
        request = MCPRequest(
            jsonrpc="2.0",
            id=2,
            method="tools/list",
            params={}
        )
        response = await server.handle_request(request)
        self.assertEqual(response.jsonrpc, "2.0")
        self.assertEqual(response.id, 2)
        self.assertIsNotNone(response.result)
        self.assertIn("tools", response.result)
        self.assertEqual(len(response.result["tools"]), 5)
        
        # Check tool structure
        tool_names = [tool["name"] for tool in response.result["tools"]]
        self.assertIn("web_search", tool_names)
        self.assertIn("web_scrape", tool_names)
        self.assertIn("image_generation", tool_names)
    
    async def test_ping(self):
        """Test ping method"""
        server = MCPServer()
        request = MCPRequest(
            jsonrpc="2.0",
            id=3,
            method="ping",
            params={}
        )
        response = await server.handle_request(request)
        self.assertEqual(response.jsonrpc, "2.0")
        self.assertEqual(response.id, 3)
        self.assertIsNotNone(response.result)
    
    async def test_invalid_method(self):
        """Test invalid method returns error"""
        server = MCPServer()
        request = MCPRequest(
            jsonrpc="2.0",
            id=4,
            method="invalid_method",
            params={}
        )
        response = await server.handle_request(request)
        self.assertEqual(response.jsonrpc, "2.0")
        self.assertEqual(response.id, 4)
        self.assertIsNotNone(response.error)
        self.assertEqual(response.error["code"], -32601)
    
    async def test_tool_call_invalid_tool(self):
        """Test calling non-existent tool"""
        server = MCPServer()
        request = MCPRequest(
            jsonrpc="2.0",
            id=5,
            method="tools/call",
            params={
                "name": "nonexistent_tool",
                "arguments": {}
            }
        )
        response = await server.handle_request(request)
        self.assertEqual(response.jsonrpc, "2.0")
        self.assertEqual(response.id, 5)
        self.assertIsNotNone(response.error)
        self.assertEqual(response.error["code"], -32601)


class TestMCPTools(unittest.IsolatedAsyncioTestCase):
    """Test cases for MCP tools"""
    
    def setUp(self) -> None:
        if not has_requirements:
            self.skipTest('MCP tools requirements not installed')
    
    async def test_web_search_tool_schema(self):
        """Test WebSearchTool schema"""
        tool = WebSearchTool()
        self.assertIsNotNone(tool.description)
        self.assertIsNotNone(tool.input_schema)
        self.assertEqual(tool.input_schema["type"], "object")
        self.assertIn("query", tool.input_schema["properties"])
        self.assertIn("query", tool.input_schema["required"])
    
    async def test_web_scrape_tool_schema(self):
        """Test WebScrapeTool schema"""
        tool = WebScrapeTool()
        self.assertIsNotNone(tool.description)
        self.assertIsNotNone(tool.input_schema)
        self.assertEqual(tool.input_schema["type"], "object")
        self.assertIn("url", tool.input_schema["properties"])
        self.assertIn("url", tool.input_schema["required"])
    
    async def test_image_generation_tool_schema(self):
        """Test ImageGenerationTool schema"""
        tool = ImageGenerationTool()
        self.assertIsNotNone(tool.description)
        self.assertIsNotNone(tool.input_schema)
        self.assertEqual(tool.input_schema["type"], "object")
        self.assertIn("prompt", tool.input_schema["properties"])
        self.assertIn("prompt", tool.input_schema["required"])
    
    async def test_web_search_missing_query(self):
        """Test web search with missing query parameter"""
        tool = WebSearchTool()
        result = await tool.execute({})
        self.assertIn("error", result)
    
    async def test_web_scrape_missing_url(self):
        """Test web scrape with missing url parameter"""
        tool = WebScrapeTool()
        result = await tool.execute({})
        self.assertIn("error", result)
    
    async def test_image_generation_missing_prompt(self):
        """Test image generation with missing prompt parameter"""
        tool = ImageGenerationTool()
        result = await tool.execute({})
        self.assertIn("error", result)

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from g4f.mcp.server import MCPServer, MCPRequest
from g4f.mcp.tools import (
    WebSearchTool, WebScrapeTool, ImageGenerationTool,
    PythonExecuteTool, FileReadTool, FileWriteTool, FileListTool, FileDeleteTool,
)
from g4f.mcp.pa_provider import execute_safe_code, get_workspace_dir, SAFE_MODULES

try:
    from ddgs import DDGS, DDGSError
    from bs4 import BeautifulSoup
    has_requirements = True
except ImportError:
    has_requirements = False

# Total number of tools registered in MCPServer
_TOOL_COUNT = 10


class TestMCPServer(unittest.IsolatedAsyncioTestCase):
    """Test cases for MCP server"""

    async def test_server_initialization(self):
        """Test that server initializes correctly"""
        server = MCPServer()
        self.assertIsNotNone(server)
        self.assertEqual(server.server_info["name"], "gpt4free-mcp-server")
        self.assertEqual(len(server.tools), _TOOL_COUNT)
        self.assertIn('web_search', server.tools)
        self.assertIn('web_scrape', server.tools)
        self.assertIn('image_generation', server.tools)
        self.assertIn('python_execute', server.tools)
        self.assertIn('file_read', server.tools)
        self.assertIn('file_write', server.tools)
        self.assertIn('file_list', server.tools)
        self.assertIn('file_delete', server.tools)

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
        self.assertEqual(len(response.result["tools"]), _TOOL_COUNT)

        tool_names = [tool["name"] for tool in response.result["tools"]]
        self.assertIn("web_search", tool_names)
        self.assertIn("web_scrape", tool_names)
        self.assertIn("image_generation", tool_names)
        self.assertIn("python_execute", tool_names)
        self.assertIn("file_read", tool_names)
        self.assertIn("file_write", tool_names)
        self.assertIn("file_list", tool_names)
        self.assertIn("file_delete", tool_names)

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
    """Test cases for existing MCP tools"""

    def setUp(self) -> None:
        if not has_requirements:
            self.skipTest('MCP tools requirements not installed')

    async def test_web_search_tool_schema(self):
        tool = WebSearchTool()
        self.assertIsNotNone(tool.description)
        self.assertIsNotNone(tool.input_schema)
        self.assertEqual(tool.input_schema["type"], "object")
        self.assertIn("query", tool.input_schema["properties"])
        self.assertIn("query", tool.input_schema["required"])

    async def test_web_scrape_tool_schema(self):
        tool = WebScrapeTool()
        self.assertIsNotNone(tool.description)
        self.assertIsNotNone(tool.input_schema)
        self.assertEqual(tool.input_schema["type"], "object")
        self.assertIn("url", tool.input_schema["properties"])
        self.assertIn("url", tool.input_schema["required"])

    async def test_image_generation_tool_schema(self):
        tool = ImageGenerationTool()
        self.assertIsNotNone(tool.description)
        self.assertIsNotNone(tool.input_schema)
        self.assertEqual(tool.input_schema["type"], "object")
        self.assertIn("prompt", tool.input_schema["properties"])
        self.assertIn("prompt", tool.input_schema["required"])

    async def test_web_search_missing_query(self):
        tool = WebSearchTool()
        result = await tool.execute({})
        self.assertIn("error", result)

    async def test_web_scrape_missing_url(self):
        tool = WebScrapeTool()
        result = await tool.execute({})
        self.assertIn("error", result)

    async def test_image_generation_missing_prompt(self):
        tool = ImageGenerationTool()
        result = await tool.execute({})
        self.assertIn("error", result)


class TestPythonExecuteTool(unittest.IsolatedAsyncioTestCase):
    """Tests for the PythonExecuteTool (no network required)."""

    async def test_schema(self):
        tool = PythonExecuteTool()
        self.assertIsNotNone(tool.description)
        schema = tool.input_schema
        self.assertEqual(schema["type"], "object")
        self.assertIn("code", schema["properties"])
        self.assertIn("code", schema["required"])

    async def test_missing_code(self):
        tool = PythonExecuteTool()
        result = await tool.execute({})
        self.assertIn("error", result)

    async def test_simple_execution(self):
        tool = PythonExecuteTool()
        result = await tool.execute({"code": "result = 1 + 2"})
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("result"), 3)

    async def test_stdout_captured(self):
        tool = PythonExecuteTool()
        result = await tool.execute({"code": "print('hello')"})
        self.assertTrue(result.get("success"))
        self.assertIn("hello", result.get("stdout", ""))

    async def test_syntax_error(self):
        tool = PythonExecuteTool()
        result = await tool.execute({"code": "def foo(:"})
        self.assertFalse(result.get("success"))
        self.assertIn("error", result)

    async def test_blocked_import(self):
        tool = PythonExecuteTool()
        result = await tool.execute({"code": "import subprocess"})
        self.assertFalse(result.get("success"))
        self.assertIn("error", result)

    async def test_blocked_builtin_exec(self):
        """exec() is removed from safe builtins."""
        tool = PythonExecuteTool()
        result = await tool.execute({"code": "exec('x=1')"})
        self.assertFalse(result.get("success"))
        self.assertIn("error", result)

    async def test_allowed_module(self):
        tool = PythonExecuteTool()
        result = await tool.execute({"code": "import math\nresult = math.sqrt(4)"})
        self.assertTrue(result.get("success"))
        self.assertAlmostEqual(result.get("result"), 2.0)

    async def test_allowed_json_module(self):
        tool = PythonExecuteTool()
        result = await tool.execute({
            "code": "import json\nresult = json.dumps({'a': 1})"
        })
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("result"), '{"a": 1}')


class TestFilesTools(unittest.IsolatedAsyncioTestCase):
    """Tests for the file manipulation tools (workspace)."""

    def setUp(self):
        self.workspace = get_workspace_dir()
        self.test_file = "unittest_temp_test.txt"
        # Clean up leftover test file
        target = self.workspace / self.test_file
        if target.exists():
            target.unlink()

    def tearDown(self):
        target = self.workspace / self.test_file
        if target.exists():
            target.unlink()

    async def test_file_list_schema(self):
        tool = FileListTool()
        self.assertIsNotNone(tool.description)
        schema = tool.input_schema
        self.assertEqual(schema["type"], "object")

    async def test_file_write_schema(self):
        tool = FileWriteTool()
        schema = tool.input_schema
        self.assertIn("path", schema["properties"])
        self.assertIn("content", schema["properties"])

    async def test_file_read_schema(self):
        tool = FileReadTool()
        schema = tool.input_schema
        self.assertIn("path", schema["properties"])
        self.assertIn("path", schema["required"])

    async def test_file_delete_schema(self):
        tool = FileDeleteTool()
        schema = tool.input_schema
        self.assertIn("path", schema["properties"])
        self.assertIn("path", schema["required"])

    async def test_write_and_read(self):
        write_tool = FileWriteTool()
        read_tool = FileReadTool()

        write_result = await write_tool.execute({
            "path": self.test_file,
            "content": "hello workspace",
        })
        self.assertNotIn("error", write_result)
        self.assertEqual(write_result["path"], self.test_file)

        read_result = await read_tool.execute({"path": self.test_file})
        self.assertNotIn("error", read_result)
        self.assertEqual(read_result["content"], "hello workspace")

    async def test_read_missing_file(self):
        tool = FileReadTool()
        result = await tool.execute({"path": "definitely_does_not_exist.txt"})
        self.assertIn("error", result)

    async def test_read_missing_path_param(self):
        tool = FileReadTool()
        result = await tool.execute({})
        self.assertIn("error", result)

    async def test_write_and_delete(self):
        write_tool = FileWriteTool()
        delete_tool = FileDeleteTool()

        await write_tool.execute({"path": self.test_file, "content": "to delete"})
        self.assertTrue((self.workspace / self.test_file).exists())

        delete_result = await delete_tool.execute({"path": self.test_file})
        self.assertNotIn("error", delete_result)
        self.assertTrue(delete_result.get("deleted"))
        self.assertFalse((self.workspace / self.test_file).exists())

    async def test_delete_missing_file(self):
        tool = FileDeleteTool()
        result = await tool.execute({"path": "no_such_file.txt"})
        self.assertIn("error", result)

    async def test_list_workspace(self):
        # Write a temp file so workspace is non-empty
        write_tool = FileWriteTool()
        await write_tool.execute({"path": self.test_file, "content": "x"})

        list_tool = FileListTool()
        result = await list_tool.execute({})
        self.assertNotIn("error", result)
        self.assertIn("entries", result)
        paths = [e["path"] for e in result["entries"]]
        self.assertIn(self.test_file, paths)

    async def test_path_traversal_blocked(self):
        """Ensure path traversal outside workspace is rejected."""
        read_tool = FileReadTool()
        result = await read_tool.execute({"path": "../../etc/passwd"})
        self.assertIn("error", result)

    async def test_file_append(self):
        write_tool = FileWriteTool()
        read_tool = FileReadTool()

        await write_tool.execute({"path": self.test_file, "content": "line1\n"})
        await write_tool.execute({"path": self.test_file, "content": "line2\n", "append": True})

        read_result = await read_tool.execute({"path": self.test_file})
        self.assertEqual(read_result["content"], "line1\nline2\n")


class TestSafeCodeExecution(unittest.TestCase):
    """Unit tests for the execute_safe_code() function directly."""

    def test_basic_result(self):
        r = execute_safe_code("result = 42")
        self.assertTrue(r.success)
        self.assertEqual(r.result, 42)

    def test_stdout(self):
        r = execute_safe_code("print('hi')")
        self.assertTrue(r.success)
        self.assertIn("hi", r.stdout)

    def test_runtime_error(self):
        r = execute_safe_code("1/0")
        self.assertFalse(r.success)
        self.assertIn("ZeroDivisionError", r.error)

    def test_blocked_os_import(self):
        r = execute_safe_code("import os")
        self.assertFalse(r.success)

    def test_blocked_sys_import(self):
        r = execute_safe_code("import sys")
        self.assertFalse(r.success)

    def test_blocked_subprocess(self):
        r = execute_safe_code("import subprocess")
        self.assertFalse(r.success)

    def test_allowed_math(self):
        r = execute_safe_code("import math\nresult = math.pi")
        self.assertTrue(r.success)
        self.assertAlmostEqual(r.result, 3.14159, places=4)

    def test_to_dict_success(self):
        r = execute_safe_code("result = [1, 2, 3]")
        d = r.to_dict()
        self.assertTrue(d["success"])
        self.assertEqual(d["result"], [1, 2, 3])

    def test_to_dict_failure(self):
        r = execute_safe_code("raise ValueError('boom')")
        d = r.to_dict()
        self.assertFalse(d["success"])
        self.assertIn("error", d)

    def test_safe_modules_frozenset(self):
        self.assertIsInstance(SAFE_MODULES, frozenset)
        self.assertIn("math", SAFE_MODULES)
        self.assertIn("json", SAFE_MODULES)
        self.assertIn("asyncio", SAFE_MODULES)
        self.assertNotIn("os", SAFE_MODULES)
        self.assertNotIn("subprocess", SAFE_MODULES)

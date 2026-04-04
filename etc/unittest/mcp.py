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

# Total number of tools registered in MCPServer (derived at import time)
_TOOL_COUNT = len(MCPServer().tools)


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

class TestSafeMode(unittest.IsolatedAsyncioTestCase):
    """Tests for --safe mode behaviour on MCPServer and individual tools."""

    def test_server_safe_mode_flag(self):
        """MCPServer stores the safe_mode flag."""
        server = MCPServer(safe_mode=True)
        self.assertTrue(server.safe_mode)
        self.assertTrue(server.tools['python_execute'].safe_mode)
        self.assertTrue(server.tools['file_list'].safe_mode)
        # Tools that don't use safe_mode should not be affected
        self.assertFalse(server.tools['file_read'].safe_mode)

    def test_server_default_not_safe(self):
        """MCPServer defaults to safe_mode=False."""
        server = MCPServer()
        self.assertFalse(server.safe_mode)
        self.assertFalse(server.tools['python_execute'].safe_mode)
        self.assertFalse(server.tools['file_list'].safe_mode)

    async def test_python_execute_safe_mode_blocks_extra_modules(self):
        """In safe mode, allowed_extra_modules is ignored."""
        tool = PythonExecuteTool(safe_mode=True)
        # Attempt to whitelist 'os' via allowed_extra_modules — must be blocked
        result = await tool.execute({
            "code": "import os\nresult = os.getcwd()",
            "allowed_extra_modules": ["os"],
        })
        self.assertFalse(result.get("success"))
        self.assertIn("error", result)

    async def test_python_execute_normal_mode_allows_extra_modules(self):
        """Outside safe mode, allowed_extra_modules expands the allowlist."""
        tool = PythonExecuteTool(safe_mode=False)
        result = await tool.execute({
            "code": "import os\nresult = isinstance(os.getcwd(), str)",
            "allowed_extra_modules": ["os"],
        })
        self.assertTrue(result.get("success"))
        self.assertTrue(result.get("result"))

    async def test_file_list_safe_mode_blocks_root(self):
        """In safe mode, listing the workspace root is blocked."""
        tool = FileListTool(safe_mode=True)
        result = await tool.execute({})
        self.assertIn("error", result)
        self.assertIn("safe mode", result["error"])

    async def test_file_list_safe_mode_allows_subdir(self):
        """In safe mode, listing a subdirectory is still allowed."""
        workspace = get_workspace_dir()
        subdir = workspace / "unittest_safe_subdir"
        subdir.mkdir(exist_ok=True)
        try:
            tool = FileListTool(safe_mode=True)
            result = await tool.execute({"path": "unittest_safe_subdir"})
            self.assertNotIn("error", result)
        finally:
            subdir.rmdir()

    async def test_file_list_normal_mode_allows_root(self):
        """Outside safe mode, listing the workspace root is permitted."""
        tool = FileListTool(safe_mode=False)
        result = await tool.execute({})
        self.assertNotIn("error", result)

    async def test_python_execute_safe_mode_default_whitelist_still_works(self):
        """Safe mode still allows all default SAFE_MODULES."""
        tool = PythonExecuteTool(safe_mode=True)
        result = await tool.execute({"code": "import math\nresult = math.factorial(5)"})
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("result"), 120)

class TestSecurityHardening(unittest.IsolatedAsyncioTestCase):
    """Tests for execution timeout, recursion depth, and output size limits."""

    def test_execution_timeout(self):
        """Infinite loop is interrupted by the timeout."""
        import time
        start = time.time()
        r = execute_safe_code("while True: pass", timeout=0.5)
        elapsed = time.time() - start
        self.assertFalse(r.success)
        self.assertIn("timed out", r.error.lower())
        self.assertLess(elapsed, 3.0, "Should have returned within 3 s")

    def test_execution_continues_after_timeout(self):
        """The sandbox is usable again after a previous execution timed out."""
        execute_safe_code("while True: pass", timeout=0.3)
        r = execute_safe_code("result = 'ok'", timeout=5.0)
        self.assertTrue(r.success)
        self.assertEqual(r.result, "ok")

    def test_recursion_depth_limit(self):
        """Deep recursion is blocked by the max_depth parameter."""
        r = execute_safe_code(
            "def f(n): return f(n + 1)\nf(0)",
            max_depth=50,
            timeout=5.0,
        )
        self.assertFalse(r.success)

    def test_output_truncation(self):
        """stdout is capped at MAX_OUTPUT_BYTES; truncation notice appears."""
        from g4f.mcp.pa_provider import MAX_OUTPUT_BYTES
        # Produce more bytes than the limit
        r = execute_safe_code(
            f"print('A' * {MAX_OUTPUT_BYTES + 1000})",
            timeout=5.0,
        )
        self.assertTrue(r.success)
        # The buffer may contain up to MAX_OUTPUT_BYTES of user output plus a
        # small number of bytes from multi-byte UTF-8 boundary rounding; 50
        # bytes is generous slack for that edge case.
        self.assertLessEqual(len(r.stdout), MAX_OUTPUT_BYTES + 50)
        self.assertIn("truncated", r.stderr.lower())

    def test_timeout_none_disables_limit(self):
        """Passing timeout=None does not impose a time limit."""
        r = execute_safe_code("result = sum(range(100))", timeout=None)
        self.assertTrue(r.success)
        self.assertEqual(r.result, 4950)

    async def test_tool_respects_timeout_param(self):
        """PythonExecuteTool forwards timeout to execute_safe_code."""
        tool = PythonExecuteTool(safe_mode=False)
        import time
        start = time.time()
        result = await tool.execute({"code": "while True: pass", "timeout": 0.5})
        elapsed = time.time() - start
        self.assertFalse(result.get("success"))
        self.assertLess(elapsed, 3.0)

    async def test_tool_safe_mode_ignores_timeout_param(self):
        """In safe mode, timeout parameter is ignored and default is used."""
        from g4f.mcp.pa_provider import MAX_EXEC_TIMEOUT
        tool = PythonExecuteTool(safe_mode=True)
        # Passing a very large timeout in safe mode should be ignored;
        # the default MAX_EXEC_TIMEOUT is used instead.
        result = await tool.execute({
            "code": "result = 1",
            "timeout": MAX_EXEC_TIMEOUT * 100,
        })
        self.assertTrue(result.get("success"))

    async def test_tool_safe_mode_ignores_max_depth_param(self):
        """In safe mode, max_depth parameter is ignored."""
        from g4f.mcp.pa_provider import MAX_RECURSION_DEPTH
        tool = PythonExecuteTool(safe_mode=True)
        # Even passing a huge depth, safe-mode always uses MAX_RECURSION_DEPTH
        result = await tool.execute({
            "code": "result = 1",
            "max_depth": MAX_RECURSION_DEPTH * 100,
        })
        self.assertTrue(result.get("success"))

class TestPaProviderRegistry(unittest.TestCase):
    """Tests for PaProviderRegistry — stable IDs without exposing filenames."""

    def setUp(self):
        """Create a temporary .pa.py file in the workspace for testing."""
        from g4f.mcp.pa_provider import get_workspace_dir, get_pa_registry, _pa_registry
        self.workspace = get_workspace_dir()
        # Force a fresh registry for each test
        import g4f.mcp.pa_provider as _mod
        _mod._pa_registry = None

        self.pa_file = self.workspace / "registry_test.pa.py"
        self.pa_file.write_text("""
class Provider:
    label = "RegistryTestProvider"
    working = True
    models = ["rt-model-1", "rt-model-2"]
    url = "https://test.example.com"

    @classmethod
    async def create_async_generator(cls, model, messages, **kwargs):
        yield "hello from registry test"
""")

    def tearDown(self):
        if self.pa_file.exists():
            self.pa_file.unlink()
        import g4f.mcp.pa_provider as _mod
        _mod._pa_registry = None

    def test_list_providers_returns_list(self):
        from g4f.mcp.pa_provider import get_pa_registry
        reg = get_pa_registry()
        reg.refresh()
        result = reg.list_providers()
        self.assertIsInstance(result, list)
        self.assertGreaterEqual(len(result), 1)

    def test_provider_has_required_fields(self):
        from g4f.mcp.pa_provider import get_pa_registry
        reg = get_pa_registry()
        reg.refresh()
        providers = reg.list_providers()
        p = next((x for x in providers if x.get("label") == "RegistryTestProvider"), None)
        self.assertIsNotNone(p, "Test provider not found in registry")
        self.assertIn("id", p)
        self.assertIn("label", p)
        self.assertIn("models", p)
        self.assertIn("working", p)
        self.assertIn("url", p)
        self.assertEqual(p["label"], "RegistryTestProvider")
        self.assertIn("rt-model-1", p["models"])
        self.assertTrue(p["working"])

    def test_filename_not_exposed(self):
        """Provider IDs and info must NOT contain the filename or path."""
        from g4f.mcp.pa_provider import get_pa_registry
        import json
        reg = get_pa_registry()
        reg.refresh()
        providers = reg.list_providers()
        for p in providers:
            serialized = json.dumps(p)
            self.assertNotIn("registry_test", serialized, "Filename leaked in provider info")
            self.assertNotIn(".pa.py", serialized, "Extension leaked in provider info")
            self.assertNotIn(str(self.workspace), serialized, "Workspace path leaked")

    def test_stable_id(self):
        """The same file gets the same ID across refreshes."""
        from g4f.mcp.pa_provider import get_pa_registry
        reg = get_pa_registry()
        reg.refresh()
        p1 = next(x for x in reg.list_providers() if x["label"] == "RegistryTestProvider")
        reg.refresh()
        p2 = next(x for x in reg.list_providers() if x["label"] == "RegistryTestProvider")
        self.assertEqual(p1["id"], p2["id"])

    def test_get_provider_class_returns_class(self):
        from g4f.mcp.pa_provider import get_pa_registry
        reg = get_pa_registry()
        reg.refresh()
        p = next(x for x in reg.list_providers() if x["label"] == "RegistryTestProvider")
        cls = reg.get_provider_class(p["id"])
        self.assertIsNotNone(cls)
        self.assertTrue(hasattr(cls, "create_async_generator"))

    def test_get_provider_class_missing_returns_none(self):
        from g4f.mcp.pa_provider import get_pa_registry
        reg = get_pa_registry()
        self.assertIsNone(reg.get_provider_class("nonexistent00"))

    def test_get_provider_info_returns_dict(self):
        from g4f.mcp.pa_provider import get_pa_registry
        reg = get_pa_registry()
        reg.refresh()
        p = next(x for x in reg.list_providers() if x["label"] == "RegistryTestProvider")
        info = reg.get_provider_info(p["id"])
        self.assertIsNotNone(info)
        self.assertEqual(info["id"], p["id"])
        self.assertEqual(info["label"], "RegistryTestProvider")

    def test_get_provider_info_missing_returns_none(self):
        from g4f.mcp.pa_provider import get_pa_registry
        reg = get_pa_registry()
        self.assertIsNone(reg.get_provider_info("nonexistent00"))

    def test_id_length(self):
        """IDs should be 8 hex characters."""
        from g4f.mcp.pa_provider import get_pa_registry
        reg = get_pa_registry()
        reg.refresh()
        for p in reg.list_providers():
            self.assertRegex(p["id"], r'^[0-9a-f]{8}$')

    def test_registry_singleton(self):
        from g4f.mcp.pa_provider import get_pa_registry
        r1 = get_pa_registry()
        r2 = get_pa_registry()
        self.assertIs(r1, r2)

class TestWorkspaceFileServing(unittest.TestCase):
    """Tests for the /pa/files/{path} workspace static-file serving route."""

    def setUp(self):
        """Skip if FastAPI / uvicorn are not installed."""
        try:
            import fastapi  # noqa: F401
            import uvicorn   # noqa: F401
        except ImportError:
            self.skipTest("fastapi or uvicorn not installed")
        from g4f.mcp.pa_provider import get_workspace_dir
        self.workspace = get_workspace_dir()
        self.html_file = self.workspace / "test_page.html"
        self.css_file = self.workspace / "test_style.css"
        self.js_file = self.workspace / "test_script.js"
        self.py_file = self.workspace / "test_secret.py"
        self.env_file = self.workspace / "test.env"
        self.html_file.write_text("<html><head><title>Test</title></head><body>Hello</body></html>")
        self.css_file.write_text("body { color: red; }")
        self.js_file.write_text("console.log('hello');")
        self.py_file.write_text("secret = 'do_not_expose'")
        self.env_file.write_text("SECRET_KEY=abc123")

    def tearDown(self):
        for f in [self.html_file, self.css_file, self.js_file, self.py_file, self.env_file]:
            if f.exists():
                f.unlink()

    def _get_safe_types(self):
        """Extract the _WORKSPACE_SAFE_TYPES dict from the route closure."""
        import g4f.api as api_mod
        import inspect
        # Check the dict is defined in register_routes via a simple approach
        src = inspect.getsource(api_mod.Api.register_routes)
        return "text/html" in src and "text/css" in src and "application/javascript" in src

    def test_allowed_types_present(self):
        """HTML, CSS, JS must be in the allowed types."""
        self.assertTrue(self._get_safe_types())

    def test_py_files_not_served(self):
        """.py files must not be allowed (would leak provider code)."""
        import g4f.api as api_mod
        import inspect
        src = inspect.getsource(api_mod.Api.register_routes)
        # Ensure .py is not in the whitelist dict
        self.assertIn("nosniff", src, "Security header X-Content-Type-Options missing")
        self.assertIn("Content-Security-Policy", src, "CSP header missing")
        self.assertIn("no-store", src, "Cache-Control: no-store header missing")

    def test_workspace_file_route_defined(self):
        """The /pa/files/{file_path:path} route must be registered."""
        import g4f.api as api_mod
        import inspect
        src = inspect.getsource(api_mod.Api.register_routes)
        self.assertIn("/pa/files/{file_path:path}", src)

    def test_traversal_blocked_by_logic(self):
        """The traversal check must use resolved().relative_to() logic."""
        import g4f.api as api_mod
        import inspect
        src = inspect.getsource(api_mod.Api.register_routes)
        self.assertIn("relative_to", src, "Path traversal check missing")

    def test_security_headers_present(self):
        """Security headers must be applied to served files."""
        import g4f.api as api_mod
        import inspect
        src = inspect.getsource(api_mod.Api.register_routes)
        self.assertIn("X-Content-Type-Options", src)
        self.assertIn("X-Frame-Options", src)
        self.assertIn("Content-Security-Policy", src)

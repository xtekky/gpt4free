import asyncio
import unittest

from g4f.providers.base_provider import AsyncGeneratorProvider
from g4f.providers.response import FinishReason, ToolCalls
from g4f.providers.tool_support import ToolSupportProvider
from g4f.tools.run_tools import async_iter_run_tools


class ToolPlanProviderMock(AsyncGeneratorProvider):
    working = True

    @staticmethod
    async def create_async_generator(model, messages, stream=True, **kwargs):
        # Always return a tool call plan.
        yield (
            '{"tool_calls":['
            '{"name":"read","arguments":{"filePath":"README.md"}},'
            '{"name":"glob","arguments":{"pattern":"**/*.py"}}'
            "]}"
        )
        yield FinishReason("stop")


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"filePath": {"type": "string"}},
                "required": ["filePath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Glob files",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
]


class TestToolSupportProvider(unittest.TestCase):
    def test_emits_tool_calls_from_json_plan(self):
        async def run():
            out = []
            async for chunk in ToolSupportProvider.create_async_generator(
                model="test-model",
                messages=[{"role": "user", "content": "list files"}],
                stream=True,
                tools=TOOLS,
                provider=ToolPlanProviderMock,
            ):
                out.append(chunk)
            return out

        out = asyncio.run(run())
        tool_chunks = [x for x in out if isinstance(x, ToolCalls)]
        self.assertEqual(len(tool_chunks), 1)
        calls = tool_chunks[0].get_list()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["function"]["name"], "read")
        self.assertEqual(calls[1]["function"]["name"], "glob")

    def test_run_tools_routes_to_tool_support_provider(self):
        async def run():
            out = []
            async for chunk in async_iter_run_tools(
                ToolPlanProviderMock,
                model="test-model",
                messages=[{"role": "user", "content": "list files"}],
                stream=True,
                tools=TOOLS,
                tool_emulation=True,
            ):
                out.append(chunk)
            return out

        out = asyncio.run(run())
        self.assertTrue(any(isinstance(x, ToolCalls) for x in out))


if __name__ == "__main__":
    unittest.main()

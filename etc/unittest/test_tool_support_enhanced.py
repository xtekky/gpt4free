import asyncio
import unittest
from typing import List, Optional

from g4f.providers.base_provider import AsyncGeneratorProvider
from g4f.providers.response import FinishReason, ToolCalls
from g4f.providers.tool_support import ToolSupportProvider, _preprocess_tool_messages
from g4f.tools.tool_support import (
    function_to_tool_def,
    normalize_tool_defs,
    normalize_tool_calls,
    parse_tool_calls_from_text,
)


def sample_get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a location."""
    return f"Weather in {location}: 20 degrees {unit}"


class DummyToolObj:
    name = "dummy_func"
    description = "Dummy tool description"
    parameters = {
        "type": "object",
        "properties": {"val": {"type": "integer"}},
        "required": ["val"],
    }


class XMLToolProviderMock(AsyncGeneratorProvider):
    working = True

    @staticmethod
    async def create_async_generator(model, messages, stream=True, **kwargs):
        yield "<tool_call>{\"name\":\"sample_get_weather\",\"arguments\":{\"location\":\"Tokyo\"}}</tool_call>"
        yield FinishReason("stop")


class TestToolSupportEnhanced(unittest.TestCase):
    def test_function_to_tool_def(self):
        tool = function_to_tool_def(sample_get_weather)
        self.assertEqual(tool["type"], "function")
        fn = tool["function"]
        self.assertEqual(fn["name"], "sample_get_weather")
        self.assertIn("Get the current weather", fn["description"])
        props = fn["parameters"]["properties"]
        self.assertIn("location", props)
        self.assertIn("unit", props)
        self.assertEqual(props["location"]["type"], "string")
        self.assertEqual(fn["parameters"]["required"], ["location"])

    def test_normalize_tool_defs_diverse_formats(self):
        # 1. Python Callable
        # 2. Anthropic input_schema
        # 3. Flat dict
        # 4. Object attributes
        anthropic_tool = {
            "name": "search_db",
            "description": "Search database",
            "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
        }
        flat_tool = {
            "name": "calc",
            "description": "Calculate expression",
            "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}},
        }
        obj_tool = DummyToolObj()

        defs = normalize_tool_defs([sample_get_weather, anthropic_tool, flat_tool, obj_tool])
        self.assertEqual(len(defs), 4)
        names = [d["function"]["name"] for d in defs]
        self.assertEqual(names, ["sample_get_weather", "search_db", "calc", "dummy_func"])

    def test_normalize_tool_calls_diverse_formats(self):
        # 1. OpenAI standard
        openai_call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "calc", "arguments": '{"expr": "1+1"}'},
        }
        # 2. Anthropic tool_use
        anthropic_call = {
            "type": "tool_use",
            "id": "toolu_99",
            "name": "search_db",
            "input": {"q": "query"},
        }
        # 3. Gemini functionCall
        gemini_call = {
            "functionCall": {"name": "sample_get_weather", "args": {"location": "London"}}
        }
        # 4. Flat call
        flat_call = {"name": "calc", "arguments": {"expr": "2+2"}}

        normalized = normalize_tool_calls([openai_call, anthropic_call, gemini_call, flat_call])
        self.assertEqual(len(normalized), 4)

        self.assertEqual(normalized[0]["function"]["name"], "calc")
        self.assertEqual(normalized[1]["function"]["name"], "search_db")
        self.assertEqual(normalized[1]["id"], "toolu_99")
        self.assertEqual(normalized[2]["function"]["name"], "sample_get_weather")
        self.assertEqual(normalized[3]["function"]["name"], "calc")

    def test_parse_tool_calls_from_text_formats(self):
        # XML tag
        xml_text = "<tool_call>{\"name\": \"calc\", \"arguments\": {\"expr\": \"3*3\"}}</tool_call>"
        res_xml = parse_tool_calls_from_text(xml_text)
        self.assertEqual(len(res_xml), 1)
        self.assertEqual(res_xml[0]["name"], "calc")

        # ReAct style
        react_text = "Action: sample_get_weather\nAction Input: {\"location\": \"Berlin\"}"
        res_react = parse_tool_calls_from_text(react_text)
        self.assertEqual(len(res_react), 1)
        self.assertEqual(res_react[0]["name"], "sample_get_weather")

        # Stringified format
        stringified = "[Tool call: search_db] (id=call_abc)\nArguments: {\"q\": \"test\"}"
        res_str = parse_tool_calls_from_text(stringified)
        self.assertEqual(len(res_str), 1)
        self.assertEqual(res_str[0]["name"], "search_db")

        # JSON block in code fences
        code_fence = "```json\n{\"tool_calls\": [{\"name\": \"calc\", \"arguments\": {\"expr\": \"5+5\"}}]}\n```"
        res_code = parse_tool_calls_from_text(code_fence)
        self.assertEqual(len(res_code), 1)
        self.assertEqual(res_code[0]["name"], "calc")

    def test_preprocess_tool_messages_anthropic_and_legacy(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Using tool..."},
                    {"type": "tool_use", "id": "t1", "name": "calc", "input": {"expr": "1+1"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "Result: 2"}
                ],
            },
        ]
        processed = _preprocess_tool_messages(messages)
        self.assertEqual(len(processed), 3)
        self.assertEqual(processed[0]["role"], "user")
        self.assertIn("[System]", processed[0]["content"])
        self.assertEqual(processed[1]["role"], "assistant")
        self.assertIn("[Tool call: calc]", processed[1]["content"])
        self.assertEqual(processed[2]["role"], "user")
        self.assertIn("[Tool response (id=t1)]", processed[2]["content"])


    def test_tool_support_provider_with_xml_tool_call(self):
        async def run():
            out = []
            async for chunk in ToolSupportProvider.create_async_generator(
                model="test-model",
                messages=[{"role": "user", "content": "Weather in Tokyo"}],
                stream=True,
                tools=[sample_get_weather],
                provider=XMLToolProviderMock,
            ):
                out.append(chunk)
            return out

        out = asyncio.run(run())
        tool_chunks = [x for x in out if isinstance(x, ToolCalls)]
        self.assertEqual(len(tool_chunks), 1)
        calls = tool_chunks[0].get_list()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "sample_get_weather")


if __name__ == "__main__":
    unittest.main()

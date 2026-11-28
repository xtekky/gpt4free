from __future__ import annotations

import unittest

from g4f.tools.run_tools import resolve_tool_message_names


class TestResolveToolMessageNames(unittest.TestCase):
    def test_resolves_name_from_assistant_tool_calls(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ]

        resolve_tool_message_names(messages)

        self.assertEqual(messages[1]["name"], "lookup")

    def test_replaces_tool_call_id_used_as_name(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "call_1",
                "content": "result",
            },
        ]

        resolve_tool_message_names(messages)

        self.assertEqual(messages[1]["name"], "lookup")

    def test_keeps_an_existing_valid_name(self):
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "custom_name",
                "content": "result",
            }
        ]

        resolve_tool_message_names(messages)

        self.assertEqual(messages[0]["name"], "custom_name")

    def test_splits_name_index_style_ids(self):
        messages = [
            {"role": "tool", "tool_call_id": "lookup:0", "content": "result"}
        ]

        resolve_tool_message_names(messages)

        self.assertEqual(messages[0]["name"], "lookup")

    def test_leaves_unresolvable_ids_alone(self):
        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": "result"}
        ]

        resolve_tool_message_names(messages)

        self.assertNotIn("name", messages[0])

    def test_ignores_messages_without_tool_call_id(self):
        messages = [{"role": "tool", "content": "result"}]

        resolve_tool_message_names(messages)

        self.assertNotIn("name", messages[0])


if __name__ == "__main__":
    unittest.main()

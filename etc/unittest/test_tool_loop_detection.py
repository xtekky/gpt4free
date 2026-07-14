"""Tests for g4f.api.tool_loop_detection."""

import unittest

from g4f.api.tool_loop_detection import detect_tool_loop, ToolLoopError


def _make_loop(name: str, args: str, result: str, count: int):
    """Build a messages list with `count` identical tool calls + results."""
    messages = [{"role": "user", "content": "find the file"}]
    for i in range(count):
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": name, "arguments": args},
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": f"call_{i}",
            "content": result,
        })
    return messages


class TestToolLoopDetection(unittest.TestCase):

    def test_normal_conversation_passes(self):
        """A conversation with no tool calls should never trigger."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        detect_tool_loop(messages)  # should not raise

    def test_two_calls_allowed(self):
        """Two identical calls are within the MAX_REPEATS_PER_QUERY=2 limit."""
        messages = _make_loop("file_search", '{"query": "**/x.js"}', "No files found", 2)
        detect_tool_loop(messages)  # should not raise

    def test_three_identical_calls_blocked(self):
        """Three identical calls exceed MAX_REPEATS_PER_QUERY=2."""
        messages = _make_loop("file_search", '{"query": "**/members-worker.js"}', "No files found", 3)
        with self.assertRaises(ToolLoopError) as ctx:
            detect_tool_loop(messages)
        self.assertEqual(ctx.exception.function_name, "file_search")
        self.assertIn("3 times", str(ctx.exception))

    def test_three_empty_results_blocked(self):
        """Three identical calls with empty results are blocked (per-query rule fires first)."""
        messages = _make_loop("grep_search", '{"query": "foo"}', "(empty)", 3)
        with self.assertRaises(ToolLoopError) as ctx:
            detect_tool_loop(messages)
        # The per-query rule (max 2 repeats) fires before the empty-result
        # threshold (3) because the 3rd assistant tool_call is seen before
        # the 3rd tool response.
        self.assertEqual(ctx.exception.function_name, "grep_search")

    def test_different_arguments_not_blocked(self):
        """Calls with different arguments are independent and should pass."""
        messages = [{"role": "user", "content": "search"}]
        for i in range(5):
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": "file_search", "arguments": f'{{"query": "q{i}"}}'},
                }],
            })
            messages.append({"role": "tool", "tool_call_id": f"call_{i}", "content": "(empty)"})
        # Each query is unique, so no single (name, args) exceeds thresholds.
        detect_tool_loop(messages)

    def test_dict_arguments_normalized(self):
        """Arguments as dict (not JSON string) should be normalized and detected."""
        messages = [{"role": "user", "content": "find"}]
        for i in range(3):
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": "file_search", "arguments": {"query": "**/x.js"}},
                }],
            })
            messages.append({"role": "tool", "tool_call_id": f"call_{i}", "content": "No files found"})
        with self.assertRaises(ToolLoopError):
            detect_tool_loop(messages)

    def test_global_tool_call_limit(self):
        """Exceeding GLOBAL_TOOL_CALL_LIMIT triggers the hard cap."""
        # Use unique args so per-query rules don't fire first.
        messages = [{"role": "user", "content": "find"}]
        for i in range(60):
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": "file_search", "arguments": f'{{"query": "q{i}"}}'},
                }],
            })
            messages.append({"role": "tool", "tool_call_id": f"call_{i}", "content": "ok"})
        with self.assertRaises(ToolLoopError) as ctx:
            detect_tool_loop(messages)
        self.assertIn("limit", str(ctx.exception).lower())

    def test_empty_messages_passes(self):
        detect_tool_loop([])  # should not raise

    def test_non_empty_result_not_blocked(self):
        """Successful tool results should not trigger the empty-result rule."""
        messages = _make_loop("file_search", '{"query": "x"}', "Found 3 files: a, b, c", 5)
        # 5 identical calls with non-empty results: per-query rule fires at call 3.
        with self.assertRaises(ToolLoopError):
            detect_tool_loop(messages)


if __name__ == "__main__":
    unittest.main()
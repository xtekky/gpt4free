import unittest
import time

from g4f.tools.run_tools import ThinkingProcessor, Reasoning

class TestThinkingProcessor(unittest.TestCase):
    def test_non_thinking_chunk(self):
        chunk = "This is a regular text."
        expected_time, expected_result = 0, [chunk]
        actual_time, actual_result = ThinkingProcessor.process_thinking_chunk(chunk)
        self.assertEqual(actual_time, expected_time)
        self.assertEqual(actual_result, expected_result)

    def test_thinking_start(self):
        chunk = "Hello <think>World"
        expected_time = time.time()
        expected_result = ["Hello ", Reasoning(status="🤔 Is thinking...", is_thinking="<think>"), Reasoning("World")]
        actual_time, actual_result = ThinkingProcessor.process_thinking_chunk(chunk)
        self.assertAlmostEqual(actual_time, expected_time, delta=1)
        self.assertEqual(actual_result[0], expected_result[0])
        self.assertEqual(actual_result[1], expected_result[1])
        self.assertEqual(actual_result[2], expected_result[2])

    def test_thinking_end(self):
        start_time = time.time()
        chunk = "token</think> content after"
        expected_result = [Reasoning("token"), Reasoning(status="Finished", is_thinking="</think>"), " content after"]
        actual_time, actual_result = ThinkingProcessor.process_thinking_chunk(chunk, start_time)
        self.assertEqual(actual_time, 0)
        self.assertEqual(actual_result[0], expected_result[0])
        self.assertEqual(actual_result[1], expected_result[1])
        self.assertEqual(actual_result[2], expected_result[2])

    def test_thinking_start_and_end(self):
        start_time = time.time()
        chunk = "<think>token</think> content after"
        expected_result = [Reasoning(status="🤔 Is thinking...", is_thinking="<think>"), Reasoning("token"), Reasoning(status="Finished", is_thinking="</think>"), " content after"]
        actual_time, actual_result = ThinkingProcessor.process_thinking_chunk(chunk, start_time)
        self.assertEqual(actual_time, 0)
        self.assertEqual(actual_result[0], expected_result[0])
        self.assertEqual(actual_result[1], expected_result[1])
        self.assertEqual(actual_result[2], expected_result[2])
        self.assertEqual(actual_result[3], expected_result[3])

    def test_ongoing_thinking(self):
        start_time = time.time()
        chunk = "Still thinking..."
        expected_result = [Reasoning("Still thinking...")]
        actual_time, actual_result = ThinkingProcessor.process_thinking_chunk(chunk, start_time)
        self.assertEqual(actual_time, start_time)
        self.assertEqual(actual_result, expected_result)

    def test_chunk_with_text_after_think(self):
        chunk = "Start <think>Middle</think>End"
        expected_time = 0
        expected_result = ["Start ", Reasoning(status="🤔 Is thinking...", is_thinking="<think>"), Reasoning("Middle"), Reasoning(status="Finished", is_thinking="</think>"), "End"]
        actual_time, actual_result = ThinkingProcessor.process_thinking_chunk(chunk)
        self.assertEqual(actual_time, expected_time)
        self.assertEqual(actual_result, expected_result)
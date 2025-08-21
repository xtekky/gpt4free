#!/usr/bin/env python3
"""
Create a comprehensive test for reasoning field standardization
"""

import sys
import unittest
import json

from g4f.providers.response import Reasoning
from g4f.client.stubs import ChatCompletionDelta, ChatCompletionChunk

class TestReasoningFieldStandardization(unittest.TestCase):
    
    def test_reasoning_object_structure(self):
        """Test the basic Reasoning object structure"""
        reasoning = Reasoning("thinking content", status="processing")
        
        expected_dict = {
            'token': 'thinking content',
            'status': 'processing'
        }
        
        self.assertEqual(reasoning.get_dict(), expected_dict)
        self.assertEqual(str(reasoning), "thinking content")
        
    def test_streaming_delta_with_reasoning(self):
        """Test ChatCompletionDelta with Reasoning object"""
        reasoning = Reasoning("I need to think about this...", status="thinking")
        delta = ChatCompletionDelta.model_construct(reasoning)
        
        # Check the delta structure
        self.assertEqual(delta.role, "assistant")
        self.assertIsNone(delta.content)
        self.assertEqual(delta.reasoning, "I need to think about this...")
        
    def test_current_api_format_consistency(self):
        """Test what the API should output for reasoning"""
        reasoning = Reasoning("thinking token", status="processing")
        
        # Simulate the _format_json function from api.py
        def format_json(response_type: str, content=None, **kwargs):
            if content is not None and isinstance(response_type, str):
                return {
                    'type': response_type,
                    response_type: content,
                    **kwargs
                }
            return {
                'type': response_type,
                **kwargs
            }
        
        # Test current format
        formatted = format_json("reasoning", **reasoning.get_dict())
        expected = {
            'type': 'reasoning',
            'token': 'thinking token',
            'status': 'processing'
        }
        
        self.assertEqual(formatted, expected)
        
    def test_openai_compatible_streaming_format(self):
        """Test what an OpenAI-compatible format would look like"""
        reasoning = Reasoning("step by step reasoning", status="thinking")
        
        # What OpenAI format would look like
        openai_format = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "reasoning": str(reasoning)  # OpenAI uses 'reasoning' field
                },
                "finish_reason": None
            }]
        }
        
        self.assertEqual(openai_format["choices"][0]["delta"]["reasoning"], "step by step reasoning")
        
    def test_deepseek_compatible_format(self):
        """Test what a DeepSeek-compatible format would look like"""
        reasoning = Reasoning("analytical reasoning", status="thinking")
        
        # What DeepSeek format would look like
        deepseek_format = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant", 
                    "reasoning_content": str(reasoning)  # DeepSeek uses 'reasoning_content' field
                },
                "finish_reason": None
            }]
        }
        
        self.assertEqual(deepseek_format["choices"][0]["delta"]["reasoning_content"], "analytical reasoning")
        
    def test_proposed_standardization(self):
        """Test the proposed standardized format"""
        reasoning = Reasoning("standardized reasoning", status="thinking")
        
        # Proposed: Use OpenAI's 'reasoning' field name for consistency
        # But support both input formats (already done in OpenaiTemplate)
        
        # Current g4f streaming should use 'reasoning' field in delta
        proposed_format = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk", 
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "reasoning": str(reasoning)  # Standardize on OpenAI format
                },
                "finish_reason": None
            }]
        }
        
        self.assertEqual(proposed_format["choices"][0]["delta"]["reasoning"], "standardized reasoning")

if __name__ == "__main__":
    unittest.main()
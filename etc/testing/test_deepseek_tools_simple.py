#!/usr/bin/env python3
"""
Simple test for DeepSeek API tool calls (function calling).
This uses the official DeepSeek API directly.
"""

import os
import json
import requests

# DeepSeek API configuration
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

def test_deepseek_tool_calls():
    """Test DeepSeek API with tool calls"""
    print("=" * 60)
    print("Testing DeepSeek API Tool Calls")
    print("=" * 60)
    
    if not DEEPSEEK_API_KEY:
        print("\n❌ Error: DEEPSEEK_API_KEY environment variable not set.")
        print("Please set it with: export DEEPSEEK_API_KEY='your-key-here'")
        return
    
    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit of temperature"
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform a mathematical calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["add", "subtract", "multiply", "divide", "power"],
                            "description": "The operation to perform"
                        },
                        "a": {"type": "number"},
                        "b": {"type": "number"}
                    },
                    "required": ["operation", "a", "b"]
                }
            }
        }
    ]
    
    # Test 1: Weather query
    print("\n[Test 1] Weather query:")
    messages = [
        {"role": "user", "content": "What's the weather like in Paris?"}
    ]
    
    response = requests.post(
        DEEPSEEK_API_URL,
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "deepseek-chat",
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "stream": False
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        message = data["choices"][0]["message"]
        print(f"  Message content: {message.get('content', '')}")
        print(f"  Tool calls: {message.get('tool_calls', [])}")
        
        # Handle tool calls if present
        if message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                func_name = tool_call["function"]["name"]
                func_args = json.loads(tool_call["function"]["arguments"])
                print(f"  Executing: {func_name}({func_args})")
                
                # Simulate tool execution
                if func_name == "get_current_weather":
                    result = {
                        "location": func_args.get("location"),
                        "temperature": 22 if func_args.get("unit") == "celsius" else 72,
                        "unit": func_args.get("unit", "celsius"),
                        "condition": "sunny"
                    }
                elif func_name == "calculate":
                    ops = {
                        "add": lambda x, y: x + y,
                        "subtract": lambda x, y: x - y,
                        "multiply": lambda x, y: x * y,
                        "divide": lambda x, y: x / y if y != 0 else float('inf'),
                        "power": lambda x, y: x ** y
                    }
                    result = ops[func_args["operation"]](func_args["a"], func_args["b"])
                else:
                    result = {"error": f"Unknown function: {func_name}"}
                
                print(f"  Result: {result}")
    else:
        print(f"  Error: {response.status_code} - {response.text}")
    
    # Test 2: Math calculation
    print("\n[Test 2] Math calculation:")
    messages2 = [
        {"role": "user", "content": "Calculate 15 * 37"}
    ]
    
    response2 = requests.post(
        DEEPSEEK_API_URL,
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "deepseek-chat",
            "messages": messages2,
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": "calculate"}},
            "stream": False
        }
    )
    
    if response2.status_code == 200:
        data = response2.json()
        message = data["choices"][0]["message"]
        print(f"  Message content: {message.get('content', '')}")
        print(f"  Tool calls: {message.get('tool_calls', [])}")
        
        # Process tool call
        if message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                func_name = tool_call["function"]["name"]
                func_args = json.loads(tool_call["function"]["arguments"])
                print(f"  Executing: {func_name}({func_args})")
                
                if func_name == "calculate":
                    ops = {
                        "add": lambda x, y: x + y,
                        "subtract": lambda x, y: x - y,
                        "multiply": lambda x, y: x * y,
                        "divide": lambda x, y: x / y if y != 0 else float('inf'),
                        "power": lambda x, y: x ** y
                    }
                    result = ops[func_args["operation"]](func_args["a"], func_args["b"])
                    print(f"  Result: {result}")
    else:
        print(f"  Error: {response2.status_code} - {response2.text}")

if __name__ == "__main__":
    test_deepseek_tool_calls()

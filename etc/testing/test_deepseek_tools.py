#!/usr/bin/env python3
"""
Test script for DeepSeek API tool calls (function calling).
Uses the g4f client with DeepSeek provider.
"""

import json
from g4f.client import Client
from g4f.Provider.needs_auth import DeepSeek

# Define test tools/functions
weather_tool = {
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
}

calculator_tool = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Perform a mathematical calculation",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate"
                },
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

# Mock function implementations
def get_current_weather(location: str, unit: str = "celsius") -> dict:
    """Mock weather function"""
    return {
        "location": location,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "condition": "sunny",
        "wind_speed": 15
    }

def calculate(operation: str, a: float, b: float, expression: str = None) -> dict:
    """Mock calculation function"""
    ops = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else float('inf'),
        "power": lambda x, y: x ** y
    }
    result = ops.get(operation, lambda x, y: 0)(a, b)
    return {
        "operation": operation,
        "a": a,
        "b": b,
        "result": result,
        "expression": expression or f"{a} {operation} {b}"
    }

def test_deepseek_tool_calls():
    """Test DeepSeek API with tool calls"""
    print("=" * 60)
    print("Testing DeepSeek API Tool Calls")
    print("=" * 60)
    
    # Initialize client with DeepSeek provider
    client = Client(
        provider=DeepSeek,
        # api_key="your-api-key-here"  # Uncomment if needed
    )
    
    # Test 1: Weather query
    print("\n[Test 1] Weather query:")
    messages = [
        {"role": "user", "content": "What's the weather like in Paris?"}
    ]
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=[weather_tool, calculator_tool],
        tool_choice="auto"
    )
    
    print(f"  Response: {response.choices[0].message.content}")
    print(f"  Tool calls: {response.choices[0].message.tool_calls}")
    
    # Simulate tool execution if tool calls were made
    if response.choices[0].message.tool_calls:
        for tool_call in response.choices[0].message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            print(f"  Executing: {func_name}({func_args})")
            
            if func_name == "get_current_weather":
                result = get_current_weather(**func_args)
            elif func_name == "calculate":
                result = calculate(**func_args)
            else:
                result = {"error": f"Unknown function: {func_name}"}
            
            print(f"  Result: {result}")
            
            # Continue conversation with tool result
            messages.append(response.choices[0].message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })
    
    # Test 2: Math calculation
    print("\n[Test 2] Math calculation:")
    messages2 = [
        {"role": "user", "content": "Calculate 15 * 37"}
    ]
    
    response2 = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages2,
        tools=[calculator_tool],
        tool_choice={"type": "function", "function": {"name": "calculate"}}
    )
    
    print(f"  Response: {response2.choices[0].message.content}")
    print(f"  Tool calls: {response2.choices[0].message.tool_calls}")

if __name__ == "__main__":
    try:
        test_deepseek_tool_calls()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

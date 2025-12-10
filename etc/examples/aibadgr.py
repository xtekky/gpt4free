"""
Example usage of AI Badgr provider.
AI Badgr is an OpenAI-compatible API provider.
Get your API key at: https://aibadgr.com/api-keys

Usage:
    export AIBADGR_API_KEY="your-api-key-here"
    python aibadgr.py
"""

from g4f.client import Client
from g4f.Provider import AIBadgr

# Using AI Badgr with the g4f client
client = Client(
    provider=AIBadgr,
    api_key="your-api-key-here"  # Or set AIBADGR_API_KEY environment variable
)

# Example 1: Simple chat completion
print("Example 1: Simple chat completion")
response = client.chat.completions.create(
    model="gpt-4o-mini",  # AI Badgr supports OpenAI-compatible models
    messages=[{"role": "user", "content": "Hello! What can you help me with?"}]
)
print(response.choices[0].message.content)
print()

# Example 2: Streaming response
print("Example 2: Streaming response")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Count from 1 to 5"}],
    stream=True
)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print("\n")

# Example 3: With system message
print("Example 3: With system message")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that speaks like a pirate."},
        {"role": "user", "content": "Tell me about the weather"}
    ]
)
print(response.choices[0].message.content)

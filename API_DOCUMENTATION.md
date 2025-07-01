# G4F (GPT4Free) API Documentation

## Overview

G4F (GPT4Free) is a comprehensive Python library that provides free access to various AI models through multiple providers. It supports text generation, image generation, and provides both synchronous and asynchronous interfaces.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Client API](#client-api)
4. [Legacy API](#legacy-api)
5. [Models](#models)
6. [Providers](#providers)
7. [REST API](#rest-api)
8. [CLI Interface](#cli-interface)
9. [GUI Interface](#gui-interface)
10. [Error Handling](#error-handling)
11. [Configuration](#configuration)
12. [Examples](#examples)

## Installation

### Basic Installation
```bash
pip install g4f
```

### Full Installation with All Features
```bash
pip install g4f[all]
```

### Docker Installation
```bash
docker pull hlohaus789/g4f
docker run -p 8080:8080 hlohaus789/g4f
```

## Quick Start

### Simple Text Generation
```python
from g4f.client import Client

client = Client()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello, world!"}]
)
print(response.choices[0].message.content)
```

### Image Generation
```python
from g4f.client import Client

client = Client()
response = client.images.generate(
    model="flux",
    prompt="A beautiful sunset over mountains"
)
print(f"Generated image URL: {response.data[0].url}")
```

## Client API

The Client API provides a modern, OpenAI-compatible interface for interacting with AI models.

### Client Class

#### `Client(**kwargs)`

Main client class for interacting with AI models.

**Parameters:**
- `provider` (Optional[ProviderType]): Default provider to use
- `media_provider` (Optional[ProviderType]): Provider for image/media generation
- `proxy` (Optional[str]): Proxy server URL
- `api_key` (Optional[str]): API key for authenticated providers

**Example:**
```python
from g4f.client import Client
from g4f.Provider import OpenaiChat

client = Client(
    provider=OpenaiChat,
    proxy="http://proxy.example.com:8080"
)
```

### Chat Completions

#### `client.chat.completions.create(**kwargs)`

Creates a chat completion.

**Parameters:**
- `messages` (Messages): List of message dictionaries
- `model` (str): Model name to use
- `provider` (Optional[ProviderType]): Provider override
- `stream` (Optional[bool]): Enable streaming response
- `proxy` (Optional[str]): Proxy override
- `image` (Optional[ImageType]): Image for vision models
- `response_format` (Optional[dict]): Response format specification
- `max_tokens` (Optional[int]): Maximum tokens to generate
- `stop` (Optional[Union[list[str], str]]): Stop sequences
- `api_key` (Optional[str]): API key override

**Returns:**
- `ChatCompletion`: Completion response object

**Example:**
```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing"}
    ],
    max_tokens=500,
    temperature=0.7
)

print(response.choices[0].message.content)
print(f"Usage: {response.usage.total_tokens} tokens")
```

#### Streaming Example
```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Image Generation

#### `client.images.generate(**kwargs)`

Generates images from text prompts.

**Parameters:**
- `prompt` (str): Text description of the image
- `model` (Optional[str]): Image model to use
- `provider` (Optional[ProviderType]): Provider override
- `response_format` (Optional[str]): "url" or "b64_json"
- `proxy` (Optional[str]): Proxy override

**Returns:**
- `ImagesResponse`: Response containing generated images

**Example:**
```python
response = client.images.generate(
    model="dall-e-3",
    prompt="A futuristic city with flying cars",
    response_format="url"
)

for image in response.data:
    print(f"Image URL: {image.url}")
```

#### `client.images.create_variation(**kwargs)`

Creates variations of an existing image.

**Parameters:**
- `image` (ImageType): Source image (path, URL, or bytes)
- `model` (Optional[str]): Model to use
- `provider` (Optional[ProviderType]): Provider override
- `response_format` (Optional[str]): Response format

**Example:**
```python
response = client.images.create_variation(
    image="path/to/image.jpg",
    model="dall-e-3"
)
```

### Async Client

#### `AsyncClient(**kwargs)`

Asynchronous version of the Client class.

**Example:**
```python
import asyncio
from g4f.client import AsyncClient

async def main():
    client = AsyncClient()
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    print(response.choices[0].message.content)

asyncio.run(main())
```

#### Async Streaming Example
```python
async def stream_example():
    client = AsyncClient()
    
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Tell me a joke"}],
        stream=True
    )
    
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")

asyncio.run(stream_example())
```

## Legacy API

The legacy API provides direct access to the core functionality.

### ChatCompletion

#### `g4f.ChatCompletion.create(**kwargs)`

Creates a chat completion using the legacy interface.

**Parameters:**
- `model` (Union[Model, str]): Model to use
- `messages` (Messages): Message list
- `provider` (Union[ProviderType, str, None]): Provider
- `stream` (bool): Enable streaming
- `image` (ImageType): Image for vision models
- `ignore_working` (bool): Ignore provider working status
- `ignore_stream` (bool): Ignore streaming support

**Example:**
```python
import g4f

response = g4f.ChatCompletion.create(
    model=g4f.models.gpt_4o,
    messages=[{"role": "user", "content": "Hello!"}],
    provider=g4f.Provider.Copilot
)

print(response)
```

#### `g4f.ChatCompletion.create_async(**kwargs)`

Asynchronous version of create.

**Example:**
```python
import asyncio
import g4f

async def main():
    response = await g4f.ChatCompletion.create_async(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response)

asyncio.run(main())
```

## Models

### Available Models

#### Text Models
- **GPT-4 Family**: `gpt-4`, `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`
- **GPT-3.5**: `gpt-3.5-turbo`
- **Claude**: `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`
- **Llama**: `llama-3-70b`, `llama-3-8b`, `llama-2-70b`
- **Gemini**: `gemini-pro`, `gemini-1.5-pro`
- **Others**: `mistral-7b`, `mixtral-8x7b`, `phi-4`

#### Image Models
- **DALL-E**: `dall-e-3`
- **Flux**: `flux`, `flux-dev`, `flux-schnell`
- **Stable Diffusion**: `stable-diffusion-xl`

#### Vision Models
- **GPT-4 Vision**: `gpt-4o`, `gpt-4-vision-preview`
- **Gemini Vision**: `gemini-pro-vision`
- **Claude Vision**: `claude-3-opus`, `claude-3-sonnet`

### Model Usage

```python
from g4f import models

# Use predefined model
response = client.chat.completions.create(
    model=models.gpt_4o,
    messages=[{"role": "user", "content": "Hello!"}]
)

# Or use string name
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Model Information

```python
from g4f.models import ModelUtils

# Get all available models
all_models = ModelUtils.convert

# Get model by name
model = ModelUtils.get_model("gpt-4o")
if model:
    print(f"Provider: {model.base_provider}")
    print(f"Best provider: {model.best_provider}")
```

## Providers

### Provider Types

#### Working Providers
- **Blackbox**: Free GPT-4 access
- **Copilot**: Microsoft Copilot integration
- **PollinationsAI**: Multi-model support
- **DeepInfraChat**: Various open-source models
- **Free2GPT**: Free GPT access
- **OpenaiChat**: Official OpenAI API

#### Authentication Required
- **OpenaiAccount**: Official OpenAI with account
- **Gemini**: Google Gemini API
- **MetaAI**: Meta's AI models
- **HuggingChat**: Hugging Face chat

### Provider Usage

```python
from g4f import Provider

# Use specific provider
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    provider=Provider.Copilot
)

# Get provider information
print(Provider.Copilot.params)
print(Provider.Copilot.working)
```

### Custom Provider Selection

```python
from g4f.providers.retry_provider import IterListProvider
from g4f import Provider

# Create custom provider list with retry logic
custom_provider = IterListProvider([
    Provider.Copilot,
    Provider.Blackbox,
    Provider.PollinationsAI
])

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    provider=custom_provider
)
```

## REST API

G4F provides a FastAPI-based REST API compatible with OpenAI's API.

### Starting the API Server

```bash
# Start with default settings
python -m g4f.cli api

# Start with custom port and debug
python -m g4f.cli api --port 8080 --debug

# Start with GUI
python -m g4f.cli api --gui --port 8080
```

### API Endpoints

#### Chat Completions
```
POST /v1/chat/completions
```

**Request Body:**
```json
{
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "user", "content": "Hello!"}
    ],
    "stream": false,
    "max_tokens": 500
}
```

**Response:**
```json
{
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-4o-mini",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "Hello! How can I help you?"
        },
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 5,
        "completion_tokens": 7,
        "total_tokens": 12
    }
}
```

#### Image Generation
```
POST /v1/images/generations
```

**Request Body:**
```json
{
    "prompt": "A beautiful landscape",
    "model": "dall-e-3",
    "response_format": "url"
}
```

#### Models List
```
GET /v1/models
```

**Response:**
```json
{
    "object": "list",
    "data": [
        {
            "id": "gpt-4o",
            "object": "model",
            "created": 0,
            "owned_by": "OpenAI"
        }
    ]
}
```

### Client Usage with API

```python
import openai

# Configure client to use G4F API
client = openai.OpenAI(
    api_key="your-g4f-api-key",  # Optional
    base_url="http://localhost:1337/v1"
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## CLI Interface

The CLI provides command-line access to G4F functionality.

### Available Commands

#### Start API Server
```bash
g4f api --port 8080 --debug
```

#### Start GUI
```bash
g4f gui --port 8080
```

#### Chat Client
```bash
g4f client --model gpt-4o --provider Copilot
```

### CLI Options

#### API Command
- `--port, -p`: Server port (default: 1337)
- `--bind`: Bind address (default: 0.0.0.0:1337)
- `--debug, -d`: Enable debug mode
- `--gui, -g`: Start with GUI
- `--model`: Default model
- `--provider`: Default provider
- `--proxy`: Proxy server URL
- `--g4f-api-key`: API authentication key

#### GUI Command
- `--port, -p`: Server port
- `--debug, -d`: Enable debug mode
- `--demo`: Enable demo mode

#### Client Command
- `--model`: Model to use
- `--provider`: Provider to use
- `--stream`: Enable streaming
- `--proxy`: Proxy server URL

### Examples

```bash
# Start API with authentication
g4f api --port 8080 --g4f-api-key "your-secret-key"

# Start GUI in demo mode
g4f gui --port 8080 --demo

# Interactive chat session
g4f client --model gpt-4o --provider Copilot --stream
```

## GUI Interface

G4F provides a web-based GUI for easy interaction.

### Starting the GUI

```python
from g4f.gui import run_gui

# Start GUI programmatically
run_gui(port=8080, debug=True)
```

Or using CLI:
```bash
g4f gui --port 8080
```

### Features

- **Chat Interface**: Interactive chat with AI models
- **Provider Selection**: Choose from available providers
- **Model Selection**: Select different AI models
- **Image Generation**: Generate images from text prompts
- **Settings**: Configure proxy, API keys, and other options
- **Conversation History**: Save and load conversations

### Accessing the GUI

Once started, access the GUI at: `http://localhost:8080/chat/`

## Error Handling

G4F provides comprehensive error handling with specific exception types.

### Exception Types

```python
from g4f.errors import (
    ProviderNotFoundError,
    ProviderNotWorkingError,
    ModelNotFoundError,
    MissingAuthError,
    PaymentRequiredError,
    RateLimitError,
    TimeoutError,
    NoMediaResponseError
)
```

### Error Handling Examples

```python
from g4f.client import Client
from g4f.errors import ProviderNotWorkingError, ModelNotFoundError

client = Client()

try:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}]
    )
except ProviderNotWorkingError as e:
    print(f"Provider error: {e}")
except ModelNotFoundError as e:
    print(f"Model error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Retry Logic

```python
from g4f.providers.retry_provider import RetryProvider
from g4f import Provider

# Automatic retry with multiple providers
retry_provider = RetryProvider([
    Provider.Copilot,
    Provider.Blackbox,
    Provider.PollinationsAI
], max_retries=3)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    provider=retry_provider
)
```

## Configuration

### Environment Variables

```bash
# Set default proxy
export G4F_PROXY="http://proxy.example.com:8080"

# Set debug mode
export G4F_DEBUG="true"
```

### Configuration in Code

```python
import g4f

# Enable debug logging
g4f.debug.logging = True

# Set global proxy
import os
os.environ["G4F_PROXY"] = "http://proxy.example.com:8080"
```

### Cookie Management

```python
from g4f.cookies import get_cookies, set_cookies

# Get cookies for a domain
cookies = get_cookies("chat.openai.com")

# Set cookies
set_cookies("chat.openai.com", {"session": "value"})
```

## Examples

### Advanced Chat with Vision

```python
from g4f.client import Client
import base64

client = Client()

# Read and encode image
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
```

### Batch Processing

```python
import asyncio
from g4f.client import AsyncClient

async def process_multiple_requests():
    client = AsyncClient()
    
    prompts = [
        "Explain machine learning",
        "What is quantum computing?",
        "How does photosynthesis work?"
    ]
    
    tasks = [
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        for prompt in prompts
    ]
    
    responses = await asyncio.gather(*tasks)
    
    for i, response in enumerate(responses):
        print(f"Response {i+1}: {response.choices[0].message.content}")

asyncio.run(process_multiple_requests())
```

### Custom Provider Implementation

```python
from g4f.providers.base_provider import AsyncGeneratorProvider
from g4f.typing import AsyncResult, Messages

class CustomProvider(AsyncGeneratorProvider):
    url = "https://api.example.com"
    working = True
    supports_stream = True
    
    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        **kwargs
    ) -> AsyncResult:
        # Implement your custom provider logic
        yield "Custom response from your provider"

# Use custom provider
from g4f.client import Client

client = Client(provider=CustomProvider)
response = client.chat.completions.create(
    model="custom-model",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Function Calling / Tools

```python
from g4f.client import Client

client = Client()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools
)

# Handle tool calls
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        print(f"Tool: {tool_call.function.name}")
        print(f"Arguments: {tool_call.function.arguments}")
```

This documentation covers all the major public APIs, functions, and components of the G4F library. For the most up-to-date information, always refer to the official repository and documentation.
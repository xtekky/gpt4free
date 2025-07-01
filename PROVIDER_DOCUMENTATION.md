# G4F Provider Documentation

## Overview

The provider system in G4F is the core mechanism that enables access to different AI models through various endpoints and services. Each provider implements a standardized interface while handling the specifics of different AI services.

## Provider Architecture

### Base Provider Classes

#### `BaseProvider`
The abstract base class that all providers inherit from.

```python
from g4f.providers.types import BaseProvider

class BaseProvider(ABC):
    url: str = None
    working: bool = False
    supports_stream: bool = False
    supports_system_message: bool = True
    supports_message_history: bool = True
```

#### `AbstractProvider`
Provides synchronous completion functionality.

```python
from g4f.providers.base_provider import AbstractProvider

class MyProvider(AbstractProvider):
    @classmethod
    def create_completion(cls, model: str, messages: Messages, stream: bool, **kwargs) -> CreateResult:
        # Implementation here
        pass
```

#### `AsyncProvider` 
For asynchronous single-response providers.

```python
from g4f.providers.base_provider import AsyncProvider

class MyAsyncProvider(AsyncProvider):
    @staticmethod
    async def create_async(model: str, messages: Messages, **kwargs) -> str:
        # Implementation here
        pass
```

#### `AsyncGeneratorProvider`
For asynchronous streaming providers (most common).

```python
from g4f.providers.base_provider import AsyncGeneratorProvider

class MyStreamingProvider(AsyncGeneratorProvider):
    @staticmethod
    async def create_async_generator(model: str, messages: Messages, stream: bool = True, **kwargs) -> AsyncResult:
        # Implementation here
        yield "Response chunk"
```

### Provider Mixins

#### `ProviderModelMixin`
Adds model management capabilities.

```python
from g4f.providers.base_provider import ProviderModelMixin

class MyProvider(AsyncGeneratorProvider, ProviderModelMixin):
    default_model = "gpt-4"
    models = ["gpt-4", "gpt-3.5-turbo"]
    model_aliases = {"gpt-4": "gpt-4-0613"}
    
    @classmethod
    def get_model(cls, model: str, **kwargs) -> str:
        return super().get_model(model, **kwargs)
```

#### `AuthFileMixin`
For providers requiring authentication with file-based credential storage.

```python
from g4f.providers.base_provider import AuthFileMixin

class AuthProvider(AsyncGeneratorProvider, AuthFileMixin):
    @classmethod
    def get_cache_file(cls) -> Path:
        return super().get_cache_file()
```

## Working Providers

### Free Providers (No Authentication Required)

#### Blackbox
- **URL**: `https://www.blackbox.ai`
- **Models**: GPT-4, GPT-3.5, Claude models
- **Features**: Code generation, general chat
- **Streaming**: Yes

```python
from g4f.Provider import Blackbox

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}],
    provider=Blackbox
)
```

#### Copilot
- **URL**: `https://copilot.microsoft.com`
- **Models**: GPT-4, GPT-4 Vision
- **Features**: Search integration, image analysis
- **Streaming**: Yes

```python
from g4f.Provider import Copilot

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Search for latest AI news"}],
    provider=Copilot,
    web_search=True
)
```

#### PollinationsAI
- **URL**: `https://pollinations.ai`
- **Models**: Multiple models including image generation
- **Features**: Text and image generation
- **Streaming**: Yes

```python
from g4f.Provider import PollinationsAI

# Text generation
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}],
    provider=PollinationsAI
)

# Image generation
image_response = client.images.generate(
    prompt="A beautiful landscape",
    provider=PollinationsAI
)
```

#### DeepInfraChat
- **URL**: `https://deepinfra.com`
- **Models**: Llama, Mistral, and other open-source models
- **Features**: Open-source model access
- **Streaming**: Yes

```python
from g4f.Provider import DeepInfraChat

response = client.chat.completions.create(
    model="llama-3-70b",
    messages=[{"role": "user", "content": "Hello!"}],
    provider=DeepInfraChat
)
```

#### Free2GPT
- **URL**: Various endpoints
- **Models**: GPT-3.5, GPT-4
- **Features**: Free GPT access
- **Streaming**: No

```python
from g4f.Provider import Free2GPT

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}],
    provider=Free2GPT
)
```

#### LambdaChat
- **URL**: Multiple lambda endpoints
- **Models**: Various models
- **Features**: Serverless model access
- **Streaming**: Yes

#### Together
- **URL**: `https://together.ai`
- **Models**: Llama, Mistral, CodeLlama models
- **Features**: Open-source model hosting
- **Streaming**: Yes

### Authentication Required Providers

#### OpenaiAccount
- **URL**: `https://chat.openai.com`
- **Models**: All OpenAI models
- **Features**: Full OpenAI functionality
- **Authentication**: Session cookies or HAR files

```python
from g4f.Provider import OpenaiAccount

# Requires authentication setup
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}],
    provider=OpenaiAccount
)
```

#### Gemini
- **URL**: `https://gemini.google.com`
- **Models**: Gemini Pro, Gemini Vision
- **Features**: Google's AI models
- **Authentication**: Google account session

#### MetaAI
- **URL**: `https://meta.ai`
- **Models**: Llama models
- **Features**: Meta's AI assistant
- **Authentication**: Meta account session

#### HuggingChat
- **URL**: `https://huggingface.co/chat`
- **Models**: Multiple open-source models
- **Features**: Hugging Face model hub
- **Authentication**: Hugging Face account

## Provider Selection and Retry Logic

### IterListProvider
Iterates through multiple providers until one succeeds.

```python
from g4f.providers.retry_provider import IterListProvider
from g4f import Provider

# Create provider list with automatic fallback
provider_list = IterListProvider([
    Provider.Copilot,
    Provider.Blackbox,
    Provider.PollinationsAI,
    Provider.DeepInfraChat
])

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}],
    provider=provider_list
)
```

### RetryProvider
Extends IterListProvider with configurable retry logic.

```python
from g4f.providers.retry_provider import RetryProvider
from g4f import Provider

retry_provider = RetryProvider([
    Provider.Copilot,
    Provider.Blackbox
], max_retries=3, retry_delay=1.0)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}],
    provider=retry_provider
)
```

### AnyProvider
Automatically selects the best available provider for a model.

```python
from g4f.providers.any_provider import AnyProvider

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}],
    provider=AnyProvider  # Automatically selects best provider
)
```

## Creating Custom Providers

### Basic Custom Provider

```python
from g4f.providers.base_provider import AsyncGeneratorProvider, ProviderModelMixin
from g4f.typing import AsyncResult, Messages
import aiohttp
import json

class CustomProvider(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://api.example.com"
    working = True
    supports_stream = True
    default_model = "custom-model"
    models = ["custom-model", "another-model"]
    
    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = True,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Custom G4F Provider"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(f"{cls.url}/chat/completions", json=data) as response:
                if stream:
                    async for line in response.content:
                        if line:
                            yield line.decode().strip()
                else:
                    result = await response.json()
                    yield result["choices"][0]["message"]["content"]
```

### Provider with Authentication

```python
from g4f.providers.base_provider import AsyncGeneratorProvider, AuthFileMixin
from g4f.errors import MissingAuthError

class AuthenticatedProvider(AsyncGeneratorProvider, AuthFileMixin):
    url = "https://api.secure-example.com"
    working = True
    
    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        api_key: str = None,
        **kwargs
    ) -> AsyncResult:
        if not api_key:
            raise MissingAuthError(f"API key required for {cls.__name__}")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Implementation here
        yield "Authenticated response"
```

### Provider with Image Support

```python
from g4f.providers.base_provider import AsyncGeneratorProvider
from g4f.providers.create_images import CreateImagesProvider

class ImageProvider(AsyncGeneratorProvider, CreateImagesProvider):
    url = "https://api.image-example.com"
    working = True
    image_models = ["image-model-1", "image-model-2"]
    
    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        **kwargs
    ) -> AsyncResult:
        # Handle both text and image generation
        if model in cls.image_models:
            # Image generation logic
            yield cls.create_image_response(messages[-1]["content"])
        else:
            # Text generation logic
            yield "Text response"
```

## Provider Parameters

### Common Parameters

All providers support these standard parameters:

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    provider=SomeProvider,
    
    # Common parameters
    stream=True,                    # Enable streaming
    proxy="http://proxy:8080",      # Proxy server
    timeout=30,                     # Request timeout
    max_tokens=1000,               # Maximum tokens
    temperature=0.7,               # Response randomness
    top_p=0.9,                     # Nucleus sampling
    stop=["stop", "end"],          # Stop sequences
    
    # Provider-specific parameters
    api_key="your-api-key",        # For authenticated providers
    custom_param="value"           # Provider-specific options
)
```

### Getting Provider Parameters

```python
from g4f.Provider import Copilot

# Get supported parameters
params = Copilot.get_parameters()
print(params)

# Get parameters as JSON with examples
json_params = Copilot.get_parameters(as_json=True)
print(json_params)

# Get parameter information string
print(Copilot.params)
```

## Provider Status and Health

### Checking Provider Status

```python
from g4f import Provider

# Check if provider is working
if Provider.Copilot.working:
    print("Copilot is available")

# Check streaming support
if Provider.Copilot.supports_stream:
    print("Copilot supports streaming")

# Check system message support
if Provider.Copilot.supports_system_message:
    print("Copilot supports system messages")
```

### Provider Information

```python
from g4f.Provider import ProviderUtils

# Get all providers
all_providers = ProviderUtils.convert

# Get working providers
working_providers = {
    name: provider for name, provider in all_providers.items() 
    if provider.working
}

# Get providers supporting specific features
streaming_providers = {
    name: provider for name, provider in all_providers.items()
    if provider.supports_stream
}
```

## Provider Error Handling

### Common Provider Errors

```python
from g4f.errors import (
    ProviderNotFoundError,
    ProviderNotWorkingError,
    MissingAuthError,
    RateLimitError,
    PaymentRequiredError
)

try:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}],
        provider=SomeProvider
    )
except ProviderNotWorkingError:
    print("Provider is currently not working")
except MissingAuthError:
    print("Authentication required for this provider")
except RateLimitError:
    print("Rate limit exceeded")
except PaymentRequiredError:
    print("Payment or subscription required")
```

### Provider-Specific Error Handling

```python
from g4f.providers.base_provider import RaiseErrorMixin

class SafeProvider(AsyncGeneratorProvider, RaiseErrorMixin):
    @classmethod
    async def create_async_generator(cls, model, messages, **kwargs):
        try:
            # Provider implementation
            yield "response"
        except Exception as e:
            # Use built-in error handling
            cls.raise_error({"error": str(e)})
```

## Provider Testing

### Testing Custom Providers

```python
import asyncio
from g4f.client import AsyncClient

async def test_provider():
    client = AsyncClient()
    
    try:
        response = await client.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "Test message"}],
            provider=CustomProvider
        )
        print(f"Success: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(test_provider())
```

### Provider Performance Testing

```python
import time
import asyncio

async def benchmark_provider(provider, model, message, iterations=10):
    client = AsyncClient()
    times = []
    
    for i in range(iterations):
        start_time = time.time()
        
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": message}],
                provider=provider
            )
            end_time = time.time()
            times.append(end_time - start_time)
            print(f"Iteration {i+1}: {end_time - start_time:.2f}s")
        except Exception as e:
            print(f"Iteration {i+1}: Error - {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"Average response time: {avg_time:.2f}s")
        print(f"Success rate: {len(times)}/{iterations}")

# Example usage
asyncio.run(benchmark_provider(
    Provider.Copilot, 
    "gpt-4", 
    "Hello, how are you?", 
    5
))
```

## Best Practices

### 1. Provider Selection Strategy

```python
from g4f.providers.retry_provider import IterListProvider
from g4f import Provider

# Prioritize reliable providers
reliable_providers = IterListProvider([
    Provider.Copilot,        # High reliability, good features
    Provider.Blackbox,       # Good fallback
    Provider.PollinationsAI, # Good for diverse models
    Provider.DeepInfraChat   # Open source models
])
```

### 2. Error Recovery

```python
async def robust_chat_completion(client, model, messages, max_retries=3):
    providers = [Provider.Copilot, Provider.Blackbox, Provider.PollinationsAI]
    
    for attempt in range(max_retries):
        for provider in providers:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    provider=provider,
                    timeout=30
                )
                return response
            except Exception as e:
                print(f"Attempt {attempt+1} with {provider.__name__} failed: {e}")
                continue
    
    raise Exception("All providers failed")
```

### 3. Provider Health Monitoring

```python
async def check_provider_health():
    test_message = [{"role": "user", "content": "Hello"}]
    client = AsyncClient()
    
    providers_to_test = [
        Provider.Copilot,
        Provider.Blackbox,
        Provider.PollinationsAI
    ]
    
    health_status = {}
    
    for provider in providers_to_test:
        try:
            start_time = time.time()
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=test_message,
                provider=provider,
                timeout=10
            )
            response_time = time.time() - start_time
            
            health_status[provider.__name__] = {
                "status": "healthy",
                "response_time": response_time,
                "response_length": len(response.choices[0].message.content)
            }
        except Exception as e:
            health_status[provider.__name__] = {
                "status": "unhealthy",
                "error": str(e)
            }
    
    return health_status
```

This documentation provides a comprehensive guide to understanding and working with the G4F provider system. For the latest provider status and capabilities, always check the official repository.
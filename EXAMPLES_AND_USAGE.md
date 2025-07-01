# G4F Examples and Advanced Usage Guide

## Table of Contents

1. [Basic Usage Examples](#basic-usage-examples)
2. [Advanced Features](#advanced-features)
3. [Provider-Specific Examples](#provider-specific-examples)
4. [Integration Examples](#integration-examples)
5. [Error Handling Patterns](#error-handling-patterns)
6. [Performance Optimization](#performance-optimization)
7. [Production Use Cases](#production-use-cases)

## Basic Usage Examples

### Simple Chat Completion

```python
from g4f.client import Client

client = Client()

# Basic chat
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello, world!"}]
)

print(response.choices[0].message.content)
```

### Streaming Response

```python
from g4f.client import Client

client = Client()

stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Image Generation

```python
from g4f.client import Client

client = Client()

response = client.images.generate(
    model="dall-e-3",
    prompt="A beautiful sunset over mountains",
    response_format="url"
)

print(f"Generated image: {response.data[0].url}")
```

## Advanced Features

### Vision Models with Images

```python
import base64
from g4f.client import Client

client = Client()

# Read and encode image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

base64_image = encode_image("path/to/your/image.jpg")

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
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
```

### Function Calling

```python
from g4f.client import Client
import json

client = Client()

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
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    }
]

def get_current_weather(location, unit="fahrenheit"):
    """Mock function to get weather"""
    return f"The weather in {location} is 72°{unit[0].upper()}"

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather like in Boston?"}],
    tools=tools
)

# Handle tool calls
message = response.choices[0].message
if message.tool_calls:
    for tool_call in message.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        if function_name == "get_current_weather":
            weather_result = get_current_weather(**function_args)
            print(f"Weather: {weather_result}")
```

### JSON Response Format

```python
from g4f.client import Client

client = Client()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user", 
            "content": "Generate a JSON object with information about Paris, France including population, landmarks, and cuisine."
        }
    ],
    response_format={"type": "json_object"}
)

import json
data = json.loads(response.choices[0].message.content)
print(json.dumps(data, indent=2))
```

## Provider-Specific Examples

### Using Different Providers

```python
from g4f.client import Client
from g4f import Provider

client = Client()

# Use specific provider
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}],
    provider=Provider.Copilot
)

# Provider with custom configuration
response = client.chat.completions.create(
    model="llama-3-70b",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    provider=Provider.DeepInfraChat,
    temperature=0.7,
    max_tokens=1000
)
```

### Provider Fallback Strategy

```python
from g4f.providers.retry_provider import IterListProvider
from g4f import Provider
from g4f.client import Client

# Create fallback provider list
fallback_providers = IterListProvider([
    Provider.Copilot,
    Provider.Blackbox,
    Provider.PollinationsAI,
    Provider.DeepInfraChat
])

client = Client(provider=fallback_providers)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Authenticated Providers

```python
from g4f.client import Client
from g4f.Provider import OpenaiAccount

# Using OpenAI account (requires authentication setup)
client = Client(provider=OpenaiAccount)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}],
    api_key="your-openai-api-key"  # If needed
)
```

## Integration Examples

### Async Client Usage

```python
import asyncio
from g4f.client import AsyncClient

async def async_chat_example():
    client = AsyncClient()
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    return response.choices[0].message.content

# Run async function
result = asyncio.run(async_chat_example())
print(result)
```

### Batch Processing

```python
import asyncio
from g4f.client import AsyncClient

async def process_batch_requests():
    client = AsyncClient()
    
    prompts = [
        "Explain machine learning",
        "What is quantum computing?", 
        "How does blockchain work?",
        "What is artificial intelligence?"
    ]
    
    # Create tasks for concurrent processing
    tasks = [
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        for prompt in prompts
    ]
    
    # Execute all tasks concurrently
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            print(f"Error for prompt {i+1}: {response}")
        else:
            print(f"Response {i+1}: {response.choices[0].message.content[:100]}...")

asyncio.run(process_batch_requests())
```

### Web Framework Integration (FastAPI)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from g4f.client import AsyncClient
import asyncio

app = FastAPI()
client = AsyncClient()

class ChatRequest(BaseModel):
    message: str
    model: str = "gpt-4o-mini"

class ChatResponse(BaseModel):
    response: str
    model: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response = await client.chat.completions.create(
            model=request.model,
            messages=[{"role": "user", "content": request.message}]
        )
        
        return ChatResponse(
            response=response.choices[0].message.content,
            model=response.model
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn main:app --reload
```

### LangChain Integration

```python
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from g4f.client import Client
from typing import List

class G4FChatModel(BaseChatModel):
    client: Client = Client()
    model_name: str = "gpt-4o-mini"
    
    def _generate(self, messages: List[BaseMessage], **kwargs):
        # Convert LangChain messages to G4F format
        g4f_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                g4f_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                g4f_messages.append({"role": "assistant", "content": msg.content})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=g4f_messages,
            **kwargs
        )
        
        return AIMessage(content=response.choices[0].message.content)

# Usage
llm = G4FChatModel()
response = llm([HumanMessage(content="Hello!")])
print(response.content)
```

## Error Handling Patterns

### Comprehensive Error Handling

```python
from g4f.client import Client
from g4f.errors import (
    ProviderNotWorkingError,
    ModelNotFoundError,
    MissingAuthError,
    RateLimitError,
    TimeoutError
)
import time

def robust_chat_completion(message, max_retries=3, retry_delay=1):
    client = Client()
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": message}],
                timeout=30
            )
            return response.choices[0].message.content
            
        except ProviderNotWorkingError:
            print(f"Provider not working, attempt {attempt + 1}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            raise
            
        except ModelNotFoundError as e:
            print(f"Model not found: {e}")
            # Try with different model
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": message}]
                )
                return response.choices[0].message.content
            except:
                raise e
                
        except RateLimitError:
            print(f"Rate limited, waiting before retry {attempt + 1}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * 2)  # Longer wait for rate limits
                continue
            raise
            
        except TimeoutError:
            print(f"Timeout, attempt {attempt + 1}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            raise
            
        except Exception as e:
            print(f"Unexpected error: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            raise

# Usage
try:
    result = robust_chat_completion("Hello, how are you?")
    print(result)
except Exception as e:
    print(f"All retry attempts failed: {e}")
```

### Provider Health Monitoring

```python
import asyncio
import time
from g4f.client import AsyncClient
from g4f import Provider

async def check_provider_health():
    client = AsyncClient()
    test_message = [{"role": "user", "content": "Hello"}]
    
    providers = [
        Provider.Copilot,
        Provider.Blackbox,
        Provider.PollinationsAI,
        Provider.DeepInfraChat
    ]
    
    health_status = {}
    
    for provider in providers:
        try:
            start_time = time.time()
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=test_message,
                provider=provider,
                timeout=10
            )
            end_time = time.time()
            
            health_status[provider.__name__] = {
                "status": "healthy",
                "response_time": round(end_time - start_time, 2),
                "response_preview": response.choices[0].message.content[:50]
            }
        except Exception as e:
            health_status[provider.__name__] = {
                "status": "unhealthy",
                "error": str(e)
            }
    
    return health_status

# Check provider health
health = asyncio.run(check_provider_health())
for provider, status in health.items():
    print(f"{provider}: {status}")
```

## Performance Optimization

### Connection Pooling and Reuse

```python
from g4f.client import AsyncClient
import asyncio

class G4FManager:
    def __init__(self):
        self.client = AsyncClient()
        self.session_pool = {}
    
    async def chat_completion(self, message, model="gpt-4o-mini"):
        response = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message}]
        )
        return response.choices[0].message.content
    
    async def batch_completions(self, messages, model="gpt-4o-mini", max_concurrent=5):
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_message(message):
            async with semaphore:
                return await self.chat_completion(message, model)
        
        tasks = [process_message(msg) for msg in messages]
        return await asyncio.gather(*tasks, return_exceptions=True)

# Usage
manager = G4FManager()

# Single completion
result = asyncio.run(manager.chat_completion("Hello!"))
print(result)

# Batch processing with concurrency control
messages = ["Hello!", "How are you?", "What's AI?", "Explain ML"]
results = asyncio.run(manager.batch_completions(messages, max_concurrent=3))
```

### Caching Responses

```python
import hashlib
import json
import time
from functools import wraps
from g4f.client import Client

class ResponseCache:
    def __init__(self, ttl=3600):  # 1 hour TTL
        self.cache = {}
        self.ttl = ttl
    
    def get_cache_key(self, model, messages, **kwargs):
        # Create deterministic hash of request
        cache_data = {
            "model": model,
            "messages": messages,
            **{k: v for k, v in kwargs.items() if k not in ['stream']}
        }
        return hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()
    
    def get(self, key):
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key, value):
        self.cache[key] = (value, time.time())

def cached_completion(cache_instance):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract model and messages for cache key
            model = kwargs.get('model', 'gpt-4o-mini')
            messages = kwargs.get('messages', [])
            
            cache_key = cache_instance.get_cache_key(model, messages, **kwargs)
            
            # Check cache first
            cached_result = cache_instance.get(cache_key)
            if cached_result:
                print("Cache hit!")
                return cached_result
            
            # If not in cache, make actual request
            result = func(*args, **kwargs)
            
            # Cache the result
            cache_instance.set(cache_key, result)
            return result
        
        return wrapper
    return decorator

# Usage
cache = ResponseCache(ttl=1800)  # 30 minutes
client = Client()

@cached_completion(cache)
def get_completion(**kwargs):
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content

# This will hit the API
result1 = get_completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is AI?"}]
)

# This will use cache
result2 = get_completion(
    model="gpt-4o-mini", 
    messages=[{"role": "user", "content": "What is AI?"}]
)
```

## Production Use Cases

### Chatbot Implementation

```python
import asyncio
from datetime import datetime
from g4f.client import AsyncClient
from g4f import Provider

class Chatbot:
    def __init__(self, name="Assistant", model="gpt-4o-mini"):
        self.name = name
        self.model = model
        self.client = AsyncClient()
        self.conversation_history = []
        self.system_prompt = f"You are {name}, a helpful AI assistant."
        
    async def chat(self, user_message, maintain_history=True):
        # Prepare messages
        messages = [{"role": "system", "content": self.system_prompt}]
        
        if maintain_history:
            messages.extend(self.conversation_history)
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                provider=Provider.Copilot
            )
            
            assistant_response = response.choices[0].message.content
            
            # Update conversation history
            if maintain_history:
                self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append({"role": "assistant", "content": assistant_response})
                
                # Keep only last 10 exchanges to manage context length
                if len(self.conversation_history) > 20:
                    self.conversation_history = self.conversation_history[-20:]
            
            return assistant_response
            
        except Exception as e:
            return f"I'm sorry, I encountered an error: {str(e)}"
    
    def clear_history(self):
        self.conversation_history = []
    
    def get_conversation_summary(self):
        return {
            "total_exchanges": len(self.conversation_history) // 2,
            "last_interaction": datetime.now().isoformat()
        }

# Usage
async def main():
    bot = Chatbot("Alex", "gpt-4o-mini")
    
    print("Chatbot started! Type 'quit' to exit.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
        
        response = await bot.chat(user_input)
        print(f"{bot.name}: {response}")

# Run the chatbot
if __name__ == "__main__":
    asyncio.run(main())
```

### Content Generation Pipeline

```python
import asyncio
from g4f.client import AsyncClient
from g4f import Provider

class ContentGenerator:
    def __init__(self):
        self.client = AsyncClient()
    
    async def generate_blog_post(self, topic, target_length=1000):
        """Generate a complete blog post with title, outline, and content"""
        
        # Generate title
        title_prompt = f"Generate a compelling blog post title about: {topic}"
        title_response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": title_prompt}],
            provider=Provider.Copilot
        )
        title = title_response.choices[0].message.content.strip()
        
        # Generate outline
        outline_prompt = f"Create a detailed outline for a blog post titled '{title}' about {topic}. Include 4-6 main sections."
        outline_response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": outline_prompt}]
        )
        outline = outline_response.choices[0].message.content
        
        # Generate content
        content_prompt = f"""
        Write a {target_length}-word blog post with the following details:
        Title: {title}
        Topic: {topic}
        Outline: {outline}
        
        Make it engaging, informative, and well-structured with proper headings.
        """
        
        content_response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content_prompt}],
            max_tokens=target_length * 2  # Allow extra tokens for formatting
        )
        content = content_response.choices[0].message.content
        
        return {
            "title": title,
            "outline": outline,
            "content": content,
            "word_count": len(content.split())
        }
    
    async def generate_social_media_content(self, main_content, platforms):
        """Generate social media adaptations of main content"""
        
        platform_configs = {
            "twitter": {"limit": 280, "style": "concise and engaging with hashtags"},
            "linkedin": {"limit": 3000, "style": "professional and insightful"},
            "instagram": {"limit": 2200, "style": "visual and inspiring with emojis"},
            "facebook": {"limit": 63206, "style": "conversational and community-focused"}
        }
        
        social_content = {}
        
        for platform in platforms:
            if platform in platform_configs:
                config = platform_configs[platform]
                
                prompt = f"""
                Adapt the following content for {platform}:
                
                Original content: {main_content[:500]}...
                
                Requirements:
                - Maximum {config['limit']} characters
                - Style: {config['style']}
                - Platform: {platform}
                
                Create engaging {platform} post:
                """
                
                response = await self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                
                social_content[platform] = response.choices[0].message.content
        
        return social_content

# Usage example
async def content_pipeline_example():
    generator = ContentGenerator()
    
    # Generate blog post
    blog_post = await generator.generate_blog_post(
        "The Future of Artificial Intelligence in Healthcare",
        target_length=1200
    )
    
    print(f"Title: {blog_post['title']}")
    print(f"Word count: {blog_post['word_count']}")
    print(f"Content preview: {blog_post['content'][:200]}...")
    
    # Generate social media adaptations
    social_content = await generator.generate_social_media_content(
        blog_post['content'],
        ['twitter', 'linkedin', 'instagram']
    )
    
    for platform, content in social_content.items():
        print(f"\n{platform.upper()}:")
        print(content)

asyncio.run(content_pipeline_example())
```

This comprehensive examples guide demonstrates practical usage patterns for G4F across different scenarios, from basic chat completions to complex production workflows. The examples show how to handle errors gracefully, optimize performance, and integrate G4F into larger applications.
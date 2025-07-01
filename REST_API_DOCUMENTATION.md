# G4F REST API Documentation

## Overview

G4F provides a FastAPI-based REST API that is fully compatible with OpenAI's API specifications. This allows you to use existing OpenAI-compatible tools and libraries with G4F's free AI providers.

## Getting Started

### Starting the API Server

#### Command Line
```bash
# Basic startup
g4f api

# Custom port and debug mode
g4f api --port 8080 --debug

# With GUI interface
g4f api --gui --port 8080

# With authentication
g4f api --g4f-api-key "your-secret-key"

# With custom provider and model defaults
g4f api --provider Copilot --model gpt-4o

# Full configuration example
g4f api \
  --port 8080 \
  --debug \
  --gui \
  --g4f-api-key "secret-key" \
  --provider Copilot \
  --model gpt-4o-mini \
  --proxy "http://proxy.example.com:8080" \
  --timeout 300
```

#### Programmatic Startup
```python
from g4f.api import run_api, AppConfig

# Configure the application
AppConfig.set_config(
    g4f_api_key="your-secret-key",
    provider="Copilot",
    model="gpt-4o-mini",
    gui=True,
    timeout=300
)

# Start the server
run_api(host="0.0.0.0", port=8080, debug=True)
```

### Base URL

Once started, the API is available at:
- **Default**: `http://localhost:1337`
- **Custom port**: `http://localhost:<PORT>`

## Authentication

G4F API supports optional authentication via API keys.

### Setting Up Authentication
```bash
# Start server with authentication
g4f api --g4f-api-key "your-secret-key"
```

### Using Authentication
```python
import openai

client = openai.OpenAI(
    api_key="your-secret-key",
    base_url="http://localhost:1337/v1"
)
```

### HTTP Headers
```http
Authorization: Bearer your-secret-key
# OR
g4f-api-key: your-secret-key
```

## API Endpoints

### Chat Completions

#### `POST /v1/chat/completions`

Creates a chat completion response.

**Request Body:**
```json
{
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    "stream": false,
    "max_tokens": 1000,
    "temperature": 0.7,
    "top_p": 0.9,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "stop": ["Human:", "AI:"],
    "provider": "Copilot",
    "proxy": "http://proxy.example.com:8080",
    "response_format": {"type": "json_object"},
    "tools": [
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
}
```

**Parameters:**
- `model` (string, required): Model to use for completion
- `messages` (array, required): List of message objects
- `stream` (boolean): Enable streaming responses
- `max_tokens` (integer): Maximum tokens to generate
- `temperature` (number): Sampling temperature (0-2)
- `top_p` (number): Nucleus sampling parameter
- `frequency_penalty` (number): Frequency penalty (-2 to 2)
- `presence_penalty` (number): Presence penalty (-2 to 2)
- `stop` (string|array): Stop sequences
- `provider` (string): Specific provider to use
- `proxy` (string): Proxy server URL
- `response_format` (object): Response format specification
- `tools` (array): Available tools/functions

**Response (Non-streaming):**
```json
{
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-4o-mini",
    "provider": "Copilot",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 8,
        "total_tokens": 18
    }
}
```

**Response (Streaming):**
```http
Content-Type: text/event-stream

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":8,"total_tokens":18}}

data: [DONE]
```

#### Example Usage

**cURL:**
```bash
curl -X POST "http://localhost:1337/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "g4f-api-key: your-secret-key" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

**Python:**
```python
import openai

client = openai.OpenAI(
    api_key="your-secret-key",
    base_url="http://localhost:1337/v1"
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

**JavaScript:**
```javascript
const OpenAI = require('openai');

const client = new OpenAI({
    apiKey: 'your-secret-key',
    baseURL: 'http://localhost:1337/v1'
});

async function main() {
    const response = await client.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: 'Hello!' }]
    });
    
    console.log(response.choices[0].message.content);
}

main();
```

### Image Generation

#### `POST /v1/images/generations`

Generates images from text prompts.

**Request Body:**
```json
{
    "prompt": "A beautiful sunset over mountains",
    "model": "dall-e-3",
    "n": 1,
    "size": "1024x1024",
    "response_format": "url",
    "provider": "PollinationsAI"
}
```

**Parameters:**
- `prompt` (string, required): Text description of desired image
- `model` (string): Image model to use
- `n` (integer): Number of images to generate (1-4)
- `size` (string): Image dimensions
- `response_format` (string): "url" or "b64_json"
- `provider` (string): Specific provider to use

**Response:**
```json
{
    "created": 1677652288,
    "data": [
        {
            "url": "https://example.com/generated-image.jpg"
        }
    ]
}
```

#### Example Usage

**cURL:**
```bash
curl -X POST "http://localhost:1337/v1/images/generations" \
  -H "Content-Type: application/json" \
  -H "g4f-api-key: your-secret-key" \
  -d '{
    "prompt": "A beautiful sunset",
    "model": "dall-e-3",
    "response_format": "url"
  }'
```

**Python:**
```python
response = client.images.generate(
    prompt="A beautiful sunset over mountains",
    model="dall-e-3",
    size="1024x1024",
    response_format="url"
)

print(response.data[0].url)
```

### Models

#### `GET /v1/models`

Lists available models.

**Response:**
```json
{
    "object": "list",
    "data": [
        {
            "id": "gpt-4o",
            "object": "model",
            "created": 0,
            "owned_by": "OpenAI",
            "image": false,
            "provider": false
        },
        {
            "id": "gpt-4o-mini",
            "object": "model",
            "created": 0,
            "owned_by": "OpenAI",
            "image": false,
            "provider": false
        },
        {
            "id": "Copilot",
            "object": "model",
            "created": 0,
            "owned_by": "Microsoft",
            "image": false,
            "provider": true
        }
    ]
}
```

#### `GET /v1/models/{model_name}`

Get information about a specific model.

**Response:**
```json
{
    "id": "gpt-4o",
    "object": "model",
    "created": 0,
    "owned_by": "OpenAI"
}
```

### Provider-Specific Endpoints

#### `POST /api/{provider}/chat/completions`

Use a specific provider for chat completions.

**Example:**
```bash
curl -X POST "http://localhost:1337/api/Copilot/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

#### `GET /api/{provider}/models`

Get models available for a specific provider.

**Response:**
```json
{
    "object": "list",
    "data": [
        {
            "id": "gpt-4",
            "object": "model",
            "created": 0,
            "owned_by": "Microsoft",
            "image": false,
            "vision": true
        }
    ]
}
```

### Providers

#### `GET /v1/providers`

Lists all available providers.

**Response:**
```json
{
    "object": "list",
    "data": [
        {
            "provider": "Copilot",
            "models": ["gpt-4", "gpt-4-vision"],
            "image_models": [],
            "vision_models": ["gpt-4-vision"],
            "url": "https://copilot.microsoft.com",
            "working": true,
            "auth": false
        }
    ]
}
```

#### `GET /v1/providers/{provider}`

Get detailed information about a specific provider.

**Response:**
```json
{
    "provider": "Copilot",
    "models": ["gpt-4", "gpt-4-vision"],
    "image_models": [],
    "vision_models": ["gpt-4-vision"],
    "url": "https://copilot.microsoft.com",
    "working": true,
    "auth": false,
    "stream": true,
    "description": "Microsoft Copilot AI assistant"
}
```

### Audio

#### `POST /v1/audio/transcriptions`

Transcribe audio to text.

**Request:**
```bash
curl -X POST "http://localhost:1337/v1/audio/transcriptions" \
  -H "g4f-api-key: your-secret-key" \
  -F "file=@audio.mp3" \
  -F "model=whisper-1"
```

#### `POST /v1/audio/speech`

Generate speech from text.

**Request Body:**
```json
{
    "model": "tts-1",
    "input": "Hello, this is a test.",
    "voice": "alloy"
}
```

### File Upload and Media

#### `POST /v1/upload_cookies`

Upload cookie files for authentication.

**Request:**
```bash
curl -X POST "http://localhost:1337/v1/upload_cookies" \
  -H "g4f-api-key: your-secret-key" \
  -F "files=@cookies.json"
```

#### `GET /media/{filename}`

Access generated media files.

**Example:**
```
GET /media/generated-image-abc123.jpg
```

## Advanced Features

### Conversation Management

#### Conversation ID
Use conversation IDs to maintain context across requests:

```json
{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}],
    "conversation_id": "conv-abc123"
}
```

#### Provider-Specific Conversations
```bash
curl -X POST "http://localhost:1337/api/Copilot/conv-123/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Continue our conversation"}]
  }'
```

### Vision Models

Send images with text for vision-capable models:

```json
{
    "model": "gpt-4o",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
                    }
                }
            ]
        }
    ]
}
```

### Tool/Function Calling

Define and use tools in your requests:

```json
{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            }
        }
    ]
}
```

### Custom Response Formats

#### JSON Mode
```json
{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Generate a JSON object with user info"}],
    "response_format": {"type": "json_object"}
}
```

## Error Handling

### Error Response Format
```json
{
    "error": {
        "message": "Model not found",
        "type": "model_not_found",
        "code": "model_not_found"
    }
}
```

### Common HTTP Status Codes

- **200**: Success
- **400**: Bad Request (invalid parameters)
- **401**: Unauthorized (missing or invalid API key)
- **403**: Forbidden (insufficient permissions)
- **404**: Not Found (model or provider not found)
- **422**: Unprocessable Entity (validation error)
- **500**: Internal Server Error

### Error Types

#### Authentication Errors
```json
{
    "error": {
        "message": "Invalid API key",
        "type": "authentication_error",
        "code": "invalid_api_key"
    }
}
```

#### Model Errors
```json
{
    "error": {
        "message": "Model 'invalid-model' not found",
        "type": "model_not_found",
        "code": "model_not_found"
    }
}
```

#### Provider Errors
```json
{
    "error": {
        "message": "Provider not working",
        "type": "provider_error",
        "code": "provider_not_working"
    }
}
```

## Configuration

### Environment Variables

```bash
# Set API configuration via environment
export G4F_PROXY="http://proxy.example.com:8080"
export G4F_API_KEY="your-secret-key"
export G4F_DEBUG="true"
```

### Runtime Configuration

```python
from g4f.api import AppConfig

# Configure at runtime
AppConfig.set_config(
    g4f_api_key="secret-key",
    provider="Copilot",
    model="gpt-4o",
    proxy="http://proxy.example.com:8080",
    timeout=300,
    ignored_providers=["SomeProvider"],
    gui=True,
    demo=False
)
```

## Integration Examples

### OpenAI Python Client

```python
import openai

client = openai.OpenAI(
    api_key="g4f-api-key",
    base_url="http://localhost:1337/v1"
)

# Standard usage
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Streaming
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### LangChain Integration

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

llm = ChatOpenAI(
    openai_api_base="http://localhost:1337/v1",
    openai_api_key="g4f-api-key",
    model_name="gpt-4o-mini"
)

response = llm([HumanMessage(content="Hello!")])
print(response.content)
```

### Node.js Integration

```javascript
const { Configuration, OpenAIApi } = require("openai");

const configuration = new Configuration({
    apiKey: "g4f-api-key",
    basePath: "http://localhost:1337/v1"
});

const openai = new OpenAIApi(configuration);

async function main() {
    const response = await openai.createChatCompletion({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "Hello!" }]
    });
    
    console.log(response.data.choices[0].message.content);
}
```

## Performance and Scaling

### Rate Limiting

G4F API doesn't implement built-in rate limiting, but you can add it using:

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request

limiter = Limiter(key_func=get_remote_address)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Custom rate limiting logic
    pass
```

### Caching

Implement response caching for improved performance:

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def get_cached_response(request_hash):
    # Cache implementation
    pass
```

### Load Balancing

Use multiple G4F instances behind a load balancer:

```yaml
# docker-compose.yml
version: '3.8'
services:
  g4f-1:
    image: hlohaus789/g4f
    ports:
      - "1337:1337"
  g4f-2:
    image: hlohaus789/g4f
    ports:
      - "1338:1337"
  nginx:
    image: nginx
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

## Security Considerations

### API Key Management

```python
import secrets

# Generate secure API key
api_key = secrets.token_urlsafe(32)

# Validate API key format
def is_valid_api_key(key):
    return len(key) >= 32 and key.isalnum()
```

### Input Validation

The API automatically validates:
- Message format and structure
- Model name validity
- Parameter ranges and types
- File upload security

### CORS Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## Monitoring and Logging

### Enable Debug Logging

```bash
g4f api --debug
```

### Custom Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("g4f.api")
```

### Health Checks

```bash
# Check API health
curl http://localhost:1337/v1/models
```

## Deployment

### Docker Deployment

```dockerfile
FROM hlohaus789/g4f:latest

# Set environment variables
ENV G4F_API_KEY=your-secret-key
ENV G4F_DEBUG=false

# Expose port
EXPOSE 1337

# Start API
CMD ["python", "-m", "g4f.cli", "api", "--host", "0.0.0.0", "--port", "1337"]
```

### Production Deployment

```bash
# Install production dependencies
pip install gunicorn uvloop

# Run with Gunicorn
gunicorn g4f.api:create_app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:1337 \
  --access-logfile - \
  --error-logfile -
```

This comprehensive REST API documentation covers all aspects of using G4F's API endpoints. The API is designed to be fully compatible with OpenAI's API, making it easy to integrate with existing tools and workflows.
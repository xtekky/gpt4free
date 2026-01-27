# GPT4Free Development Setup Guide

## Prerequisites
- Docker & Docker Compose V2
- Python 3.10+
- Git

## Quick Start (4 Steps)

```bash
# 1. Copy environment template
cp .env.example .env

# 2. (Optional) Add provider tokens - edit .env
nano .env

# 3. Start development environment
docker compose -f docker-compose.dev.yml up --build

# 4. Access the API
# Swagger UI: http://localhost:8080/docs
# OpenAI-compatible: http://localhost:1337/v1
```

## Local Development (Without Docker)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run API server with hot reload
python -m g4f --port 8080 --debug --reload

# Or run GUI only
python -m g4f.cli gui --port 8080 --debug
```

## Environment Variables

### Provider Authentication
| Variable | Provider | How to Get |
|----------|----------|------------|
| `YUPP_API_KEY` | Yupp.ai | Browser cookie `__Secure-yupp.session-token` |
| `OPENAI_API_KEY` | OpenAI | platform.openai.com |
| `ANTHROPIC_API_KEY` | Anthropic | console.anthropic.com |
| `GEMINI_API_KEY` | Google Gemini | aistudio.google.com |
| `HF_TOKEN` | HuggingFace | huggingface.co/settings/tokens |
| `AZURE_API_KEY` | Azure OpenAI | Azure Portal |

### Server Settings
| Variable | Description | Default |
|----------|-------------|---------|
| `G4F_DEBUG` | Enable debug logging | false |
| `G4F_RELOAD` | Auto-reload on code changes | false |
| `DISABLED_PROVIDERS` | Comma-separated list | - |
| `DEFAULT_PROVIDER` | Fallback provider name | Auto |

### Proxy Settings (Optional)
```bash
HTTP_PROXY=http://127.0.0.1:7890
HTTPS_PROXY=http://127.0.0.1:7890
SOCKS_PROXY=socks5://127.0.0.1:7890
```

## Docker Image Options

| Image | Command | Best For |
|-------|---------|----------|
| Dev | `docker compose -f docker-compose.dev.yml up` | Active development with hot reload |
| Slim | `docker compose -f docker-compose-slim.yml up` | Lightweight deployment |
| Full | `docker compose up` | Browser-based providers |

**Note:** Dev/Slim images don't include Chrome. For browser automation (some providers), use the Full image.

## Testing

### Test API Endpoints
```bash
# List available models
curl http://localhost:8080/v1/models

# Chat completion
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Test with Python Client
```python
from g4f.client import Client

client = Client()
response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[{'role': 'user', 'content': 'Hello!'}]
)
print(response.choices[0].message.content)
```

### Run Test Suite
```bash
pytest              # All tests
pytest -k "yupp"    # Specific provider tests
pytest --cov=g4f    # With coverage
```

## Common Issues

| Issue | Solution |
|-------|----------|
| Port in use | Change port in docker-compose.dev.yml |
| Permission denied | `chown -R 1000:1000 har_and_cookies generated_media models` |
| Module not found | `pip install -r requirements.txt` |
| Browser provider fails | Use full Docker image (`docker compose up`) |

## Useful Commands

```bash
# Docker
docker compose -f docker-compose.dev.yml up --build
docker logs g4f-dev
docker exec -it g4f-dev bash

# Local
python -m g4f --port 8080 --debug --reload

# Code quality
ruff check g4f/
ruff format g4f/
mypy g4f/
```

## Project Structure

```
gpt4free/
├── g4f/
│   ├── Provider/        # 35+ provider implementations
│   ├── api/             # FastAPI REST endpoints
│   ├── gui/             # Web interface
│   ├── client.py        # Sync Python client
│   └── async_client.py  # Async Python client
├── docker/              # Docker configurations
├── har_and_cookies/     # Cookie storage (gitignored)
├── generated_media/     # Generated content (gitignored)
└── models/              # Model data (gitignored)
```

## Adding a New Provider

1. Create `g4f/Provider/YourProvider.py`
2. Extend appropriate base class
3. Implement required methods
4. Add tests in `tests/`

```python
from g4f.providers.base_provider import AsyncGeneratorProvider

class YourProvider(AsyncGeneratorProvider):
    async def create_async_generator(self, model, messages):
        # Implementation here
        yield "response"
```

## Resources
- Full Docs: https://g4f.dev/docs
- API Reference: http://localhost:8080/docs (when running)
- GitHub: https://github.com/xtekky/gpt4free

# UberLLM

UberLLM needs an API key for chat. Create one at [uberllm.dev/signup](https://uberllm.dev/signup),
then pass it to the client or set `UBERLLM_API_KEY` (the g4f provider-name convention).
The model list at `https://api.uberllm.dev/v1/models` is public, so `get_models()`
works without a key.

```python
import os

from g4f.client import Client
from g4f.Provider import UberLLM

print(list(UberLLM.get_models(timeout=20)))

client = Client(provider=UberLLM, api_key=os.environ["UBERLLM_API_KEY"])
response = client.chat.completions.create(
    model="deepseek/deepseek-v4.1-flash",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
```

Use the full model id from the list, such as `deepseek/deepseek-v4.1-flash` or
`anthropic/claude-sonnet-4.6`. Every model uses the same OpenAI-compatible Chat
Completions endpoint, and streaming uses `stream=True` with the existing client
interface. The public list holds chat models only; this provider does not add
image generation.

Requests and prompts are sent to `https://api.uberllm.dev/v1`. Prices per model are
listed at [uberllm.dev/models](https://uberllm.dev/models). The account is prepaid;
failed requests are not billed.

## Homepage check

```bash
python -m etc.testing.test_provider_urls UberLLM
```

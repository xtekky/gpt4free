# MartLLM

MartLLM needs an API key for chat. Create one at [martllm.com/signup](https://martllm.com/signup),
then pass it to the client or set `MARTLLM_API_KEY` (the g4f provider-name convention).
The model list at `https://api.martllm.com/v1/models` is public, so `get_models()`
works without a key.

```python
import os

from g4f.client import Client
from g4f.Provider import MartLLM

print(list(MartLLM.get_models(timeout=20)))

client = Client(provider=MartLLM, api_key=os.environ["MARTLLM_API_KEY"])
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

Requests and prompts are sent to `https://api.martllm.com/v1`. Prices per model are
listed at [martllm.com/models](https://martllm.com/models). The account is prepaid;
failed requests are not billed.

## Homepage check

```bash
python -m etc.testing.test_provider_urls MartLLM
```

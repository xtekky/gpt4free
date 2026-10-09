# API Route

API Route requires an API key. Create one at [API keys](https://www.api-route.com/api-keys),
then pass it to the client or set `APIROUTE_API_KEY` (the g4f provider-name convention).

```python
import os

from g4f.client import Client
from g4f.Provider import APIRoute

key = os.environ["APIROUTE_API_KEY"]
models = APIRoute.get_models(api_key=key, timeout=20)
print(list(models))

client = Client(provider=APIRoute, api_key=key)
response = client.chat.completions.create(
    model="gpt-6.1-sol",  # Example: choose an available chat model from your catalog.
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
```

Use the complete gateway model ID, such as `gpt-6.1-sol` or `claude-fable-5-1`.
Both use the same OpenAI-compatible Chat Completions endpoint. Streaming uses
`stream=True` with the existing client interface.

The authenticated model catalog depends on the key's group and account permissions.
It can include media models, so select a chat model for `chat.completions`.
The default is an example, not a guarantee of access for every key. This provider
does not advertise image generation or infer vision support from model names.

Requests and prompts are sent to `https://global.api-route.com/v1`.
See the [API documentation](https://github.com/DennyHo0917/api-route/blob/main/API.md)
for endpoint, authentication, and model-discovery details.

## Homepage checks

Run the live homepage test without an API key:

```bash
python -m etc.testing.test_provider_urls APIRoute
# Check every extra provider's homepage:
python -m etc.testing.test_provider_urls
```

The test follows redirects, requires a final HTTP 2xx response, and reports HTTP
errors and connection failures. It runs separately from the offline unit suite.
Bot protection or local network restrictions can also cause failures; review the
reported result before concluding that a provider's homepage has disappeared.

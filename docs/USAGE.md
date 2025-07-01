# g4f Usage Guide

This guide provides practical, copy-paste ready examples demonstrating the most common ways to use **g4f** in your own projects.

---

## 1. Installation

```
pip install g4f  # or install from source
```

> **Tip** – If you are in a PEP-668 managed environment (e.g. Debian/Ubuntu 24.04) add the `--break-system-packages` flag:
>
> ```bash
> pip install --break-system-packages g4f
> ```

---

## 2. Quick start – one-liner

```python
import g4f

response = g4f.ChatCompletion.create(
    model="gpt-4o",                     # or any other supported model name
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response)  # → "Hello! How can I help you today?"
```

---

## 3. Chat completions in detail

### Synchronous API

```python
from g4f import ChatCompletion

messages = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "Summarise the plot of Dune in one sentence."},
]

result = ChatCompletion.create(
    model="gpt-4o-mini",   # model alias or full name
    messages=messages,
    # provider="DeepInfraChat",  # optional – override automatic routing
    stream=False               # default: return single string
)
print(result.content)
```

### Streaming responses

```python
for chunk in ChatCompletion.create(
    model="gpt-4o-mini", messages=messages, stream=True
):
    print(chunk, end="", flush=True)  # each chunk is a ChatCompletionChunk
```

### Asynchronous API

```python
import asyncio
from g4f import ChatCompletion

async def main():
    async for chunk in ChatCompletion.create_async(
        model="gpt-4", messages=messages, stream=True
    ):
        print(chunk)

asyncio.run(main())
```

---

## 4. High–level clients

The `Client` and `AsyncClient` classes wrap chat, image and (soon) voice endpoints in a single object.

```python
from g4f.client import Client, AsyncClient

client = Client(proxy="http://127.0.0.1:7890")

# Chat
answer = client.chat.completions.create(
    messages="Why is the sky blue?"
)
print(answer.content)

# Images (sync)
image_resp = client.images.generate("A cyber-punk cityscape at night")
image_resp.save("cyberpunk.png")

# Asynchronous variant
async def run():
    async_client = AsyncClient()
    answer = await async_client.chat.completions.create(
        messages="List the first 5 prime numbers"
    )
    print(answer.content)

import asyncio; asyncio.run(run())
```

---

## 5. Image generation

```python
img = client.images.generate(
    prompt="A photo-realistic cat wearing sunglasses",
    model="dall-e-3"  # or leave blank for automatic provider selection
)

# Access the image as Pillow object
img_pil = img.images[0]
img_pil.show()

# Or save to disk
img.save_all("output/")
```

---

## 6. Model registry utilities

```python
from g4f.models import ModelRegistry

print("All models:", ModelRegistry.all_models().keys())
print("Aliases for Llama providers:", ModelRegistry.list_models_by_provider("Together"))
```

---

## 7. Environment variables

• `G4F_PROXY` – default HTTP(S) proxy used when `proxy` is not supplied.

• `G4F_PROVIDER_TIMEOUT` – override default request timeout (in seconds).

---

## 8. Error handling basics

```python
from g4f.errors import StreamNotSupportedError

try:
    ChatCompletion.create(model="some-model", messages=[], stream=True)
except StreamNotSupportedError:
    print("Selected provider does not support streaming")
```

---

## 9. CLI usage

The project ships with an experimental CLI:

```bash
g4f "Translate 'Good morning' to Spanish" --model gpt-4o
```

Run `g4f --help` to see the full list of flags.

---

## 10. Next steps

* Dive into the [API reference](./API_REFERENCE.md) for every public class and function.
* Read the [Contributing guide](../CONTRIBUTING.md) if you want to add a new provider or model.
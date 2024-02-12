### Client API
##### from g4f (beta)

#### Start
This new client could:

```python
from g4f.client import Client
```
replaces this:

```python
from openai import OpenAI
```
in your Python Code.

New client have the same API as OpenAI.

#### Client

Create the client with custom providers:

```python
from g4f.client import Client
from g4f.Provider import BingCreateImages, OpenaiChat, Gemini

client = Client(
    provider=OpenaiChat,
    image_provider=Gemini,
    proxies=None
)
```

#### Examples

Use the ChatCompletions:

```python
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Say this is a test"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

Or use it for creating a image:
```python
response = client.images.generate(
  model="dall-e-3",
  prompt="a white siamese cat",
  ...
)

image_url = response.data[0].url
```

Also this works with the client:
```python
response = client.images.create_variation(
  image=open('cat.jpg')
  model="bing",
  ...
)

image_url = response.data[0].url
```

Orginal:
[![Image with cat](/docs/cat.jpeg)](/docs/client.md)

Variant:
[![Image with cat](/docs/cat.webp)](/docs/client.md)

[to Home](/docs/client.md)

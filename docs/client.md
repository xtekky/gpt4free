### G4F - Client API (Beta Version)

#### Introduction

Welcome to the G4F Client API, a cutting-edge tool for seamlessly integrating advanced AI capabilities into your Python applications. This guide is designed to facilitate your transition from using the OpenAI client to the G4F Client, offering enhanced features while maintaining compatibility with the existing OpenAI API.

#### Getting Started

**Switching to G4F Client:**

To begin using the G4F Client, simply update your import statement in your Python code:

Old Import:
```python
from openai import OpenAI
```

New Import:
```python
from g4f.client import Client as OpenAI
```

The G4F Client preserves the same familiar API interface as OpenAI, ensuring a smooth transition process.

### Initializing the Client

To utilize the G4F Client, create an new instance. Below is an example showcasing custom providers:

```python
from g4f.client import Client
from g4f.Provider import BingCreateImages, OpenaiChat, Gemini

client = Client(
    provider=OpenaiChat,
    image_provider=Gemini,
    ...
)
```

You also have the option to define a proxy in the client for all outgoing requests:

```python
from g4f.client import Client

client = Client(
    proxies="http://user:pass@host",
    ...
)
```

#### Usage Examples

**Text Completions:**

You can use the `ChatCompletions` endpoint to generate text completions as follows:

```python
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Say this is a test"}],
    ...
)
print(response.choices[0].message.content)
```

Also streaming are supported:

```python
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Say this is a test"}],
    stream=True,
    ...
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

**Image Generation:**

Generate images using a specified prompt:

```python
response = client.images.generate(
    model="dall-e-3",
    prompt="a white siamese cat",
    ...
)

image_url = response.data[0].url
```

**Creating Image Variations:**

Create variations of an existing image:

```python
response = client.images.create_variation(
    image=open("cat.jpg", "rb"),
    model="bing",
    ...
)

image_url = response.data[0].url
```

#### Visual Examples

Original / Variant:

[![Original Image](/docs/cat.jpeg)](/docs/client.md)
[![Variant Image](/docs/cat.webp)](/docs/client.md)

[Return to Home](/)
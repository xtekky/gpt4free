# How to Use the G4F AsyncClient API

The AsyncClient API is the asynchronous counterpart to the standard G4F Client API. It offers the same functionality as the synchronous API, but with the added benefit of improved performance due to its asynchronous nature.

Designed to maintain compatibility with the existing OpenAI API, the G4F AsyncClient API ensures a seamless transition for users already familiar with the OpenAI client.

## Key Features

The G4F AsyncClient API offers several key features:

- **Custom Providers:** The G4F Client API allows you to use custom providers. This feature enhances the flexibility of the API, enabling it to cater to a wide range of use cases.
- **ChatCompletion Interface:** The G4F package provides an interface for interacting with chat models through the ChatCompletion class. This class provides methods for creating both streaming and non-streaming responses.
- **Streaming Responses:** The ChatCompletion.create method can return a response iteratively as and when they are received if the stream parameter is set to True.
- **Non-Streaming Responses:** The ChatCompletion.create method can also generate non-streaming responses.
- **Image Generation and Vision Models:** The G4F Client API also supports image generation and vision models, expanding its utility beyond text-based interactions.

## Initializing the Client

To utilize the G4F `AsyncClient`, you need to create a new instance. Below is an example showcasing how to initialize the client with custom providers:

```python
from g4f.client import AsyncClient
from g4f.Provider import BingCreateImages, OpenaiChat, Gemini

client = AsyncClient(
    provider=OpenaiChat,
    image_provider=Gemini,
    ...
)
```

In this example:
- `provider` specifies the primary provider for generating text completions.
- `image_provider` specifies the provider for image-related functionalities.

## Configuration

You can configure the `AsyncClient` with additional settings, such as an API key for your provider and a proxy for all outgoing requests:

```python
from g4f.client import AsyncClient

client = AsyncClient(
    api_key="your_api_key_here",
    proxies="http://user:pass@host",
    ...
)
```

- `api_key`: Your API key for the provider.
- `proxies`: The proxy configuration for routing requests.

## Using AsyncClient

### Text Completions

You can use the `ChatCompletions` endpoint to generate text completions. Here’s how you can do it:

```python
response = await client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Say this is a test"}],
    ...
)
print(response.choices[0].message.content)
```

### Streaming Completions

The `AsyncClient` also supports streaming completions. This allows you to process the response incrementally as it is generated:

```python
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Say this is a test"}],
    stream=True,
    ...
)
async for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content or "", end="")
```

In this example:
- `stream=True` enables streaming of the response.

### Example: Using a Vision Model

The following code snippet demonstrates how to use a vision model to analyze an image and generate a description based on the content of the image. This example shows how to fetch an image, send it to the model, and then process the response.

```python
import requests
from g4f.client import Client
from g4f.Provider import Bing

client = AsyncClient(
    provider=Bing
)

image = requests.get("https://my_website/image.jpg", stream=True).raw
# Or: image = open("local_path/image.jpg", "rb")

response = client.chat.completions.create(
    "",
    messages=[{"role": "user", "content": "what is in this picture?"}],
    image=image
)
print(response.choices[0].message.content)
```

### Image Generation:

You can generate images using a specified prompt:

```python
response = await client.images.generate(
    model="dall-e-3",
    prompt="a white siamese cat",
    ...
)

image_url = response.data[0].url
```

#### Base64 as the response format

```python
response = await client.images.generate(
    prompt="a cool cat",
    response_format="b64_json"
)

base64_text = response.data[0].b64_json
```

### Example usage with asyncio.gather

Start two tasks at the same time:

```python
import asyncio

from g4f.client import AsyncClient
from g4f.Provider import BingCreateImages, OpenaiChat, Gemini

async def main():
    client = AsyncClient(
        provider=OpenaiChat,
        image_provider=Gemini,
        # other parameters...
    )

    task1 = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Say this is a test"}],
    )
    task2 = client.images.generate(
        model="dall-e-3",
        prompt="a white siamese cat",
    )
    responses = await asyncio.gather(task1, task2)

    print(responses)

asyncio.run(main())
```
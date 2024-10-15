
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
    # Add any other necessary parameters
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
    # Add any other necessary parameters
)
```

- `api_key`: Your API key for the provider.
- `proxies`: The proxy configuration for routing requests.

## Using AsyncClient

### Text Completions

You can use the `ChatCompletions` endpoint to generate text completions. Here’s how you can do it:

```python
import asyncio

from g4f.client import Client

async def main():
    client = Client()
    response = await client.chat.completions.async_create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "say this is a test"}],
        # Add any other necessary parameters
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

### Streaming Completions

The `AsyncClient` also supports streaming completions. This allows you to process the response incrementally as it is generated:

```python
import asyncio

from g4f.client import Client

async def main():
    client = Client()
    stream = await client.chat.completions.async_create(
        model="gpt-4",
        messages=[{"role": "user", "content": "say this is a test"}],
        stream=True,
        # Add any other necessary parameters
    )
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content or "", end="")

asyncio.run(main())
```

In this example:
- `stream=True` enables streaming of the response.

### Example: Using a Vision Model

The following code snippet demonstrates how to use a vision model to analyze an image and generate a description based on the content of the image. This example shows how to fetch an image, send it to the model, and then process the response.

```python
import g4f
import requests
import asyncio

from g4f.client import Client

image = requests.get("https://raw.githubusercontent.com/xtekky/gpt4free/refs/heads/main/docs/cat.jpeg", stream=True).raw
# Or: image = open("docs/cat.jpeg", "rb")


async def main():
    client = Client()
    response = await client.chat.completions.async_create(
        model=g4f.models.default,
        provider=g4f.Provider.Bing,
        messages=[{"role": "user", "content": "What are on this image?"}],
        image=image
        # Add any other necessary parameters
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

### Image Generation:

You can generate images using a specified prompt:

```python
import asyncio
from g4f.client import Client

async def main():
    client = Client()
    response = await client.images.async_generate(
        prompt="a white siamese cat",
        model="dall-e-3",
        # Add any other necessary parameters
    )
    image_url = response.data[0].url
    print(f"Generated image URL: {image_url}")

asyncio.run(main())
```

#### Base64 as the response format

```python
import asyncio
from g4f.client import Client

async def main():
    client = Client()
    response = await client.images.async_generate(
        prompt="a white siamese cat",
        model="dall-e-3",
        response_format="b64_json"
        # Add any other necessary parameters
    )
    base64_text = response.data[0].b64_json
    print(base64_text)

asyncio.run(main())
```

### Example usage with asyncio.gather

Start two tasks at the same time:

```python
import asyncio

from g4f.client import Client

async def main():
    client = Client()

    task1 = client.chat.completions.async_create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Say this is a test"}],
    )
    task2 = client.images.async_generate(
        model="dall-e-3",
        prompt="a white siamese cat",
    )

    responses = await asyncio.gather(task1, task2)
    
    chat_response, image_response = responses

    print("Chat Response:")
    print(chat_response.choices[0].message.content)

    print("\nImage Response:")
    image_url = image_response.data[0].url
    print(image_url)

asyncio.run(main())
```

[Return to Home](/)

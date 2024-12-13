## Vision Support in Chat Completion

This documentation provides an overview of how to integrate vision support into chat completions using an API and a client. It includes examples to guide you through the process.

### Example with the API

To use vision support in chat completion with the API, follow the example below:

```python
import requests
import json
from g4f.image import to_data_uri
from g4f.requests.raise_for_status import raise_for_status

url = "http://localhost:8080/v1/chat/completions"
body = {
    "model": "",
    "provider": "Copilot",
    "messages": [
        {"role": "user", "content": "what are on this image?"}
    ],
    "images": [
        ["data:image/jpeg;base64,...", "cat.jpeg"]
    ]
}
response = requests.post(url, json=body, headers={"g4f-api-key": "secret"})
raise_for_status(response)
print(response.json())
```

In this example:
- `url` is the endpoint for the chat completion API.
- `body` contains the model, provider, messages, and images.
- `messages` is a list of message objects with roles and content.
- `images` is a list of image data in Data URI format and optional filenames.
- `response` stores the API response.

### Example with the Client

To use vision support in chat completion with the client, follow the example below:

```python
import g4f
import g4f.Provider

def chat_completion(prompt):
    client = g4f.Client(provider=g4f.Provider.Blackbox)
    images = [
        [open("docs/images/waterfall.jpeg", "rb"), "waterfall.jpeg"],
        [open("docs/images/cat.webp", "rb"), "cat.webp"]
    ]
    response = client.chat.completions.create([{"content": prompt, "role": "user"}], "", images=images)
    print(response.choices[0].message.content)

prompt = "what are on this images?"
chat_completion(prompt)
```

```
**Image 1**

* A waterfall with a rainbow
* Lush greenery surrounding the waterfall
* A stream flowing from the waterfall

**Image 2**

* A white cat with blue eyes
* A bird perched on a window sill
* Sunlight streaming through the window
```

In this example:
- `client` initializes a new client with the specified provider.
- `images` is a list of image data and optional filenames.
- `response` stores the response from the client.
- The `chat_completion` function prints the chat completion output.

### Notes

- Multiple images can be sent. Each image has two data parts: image data (in Data URI format for the API) and an optional filename.
- The client supports bytes, IO objects, and PIL images as input.
- Ensure you use a provider that supports vision and multiple images.
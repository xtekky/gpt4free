
# G4F Client API Guide
 

## Table of Contents
   - [Introduction](#introduction)
   - [Getting Started](#getting-started)
   - [Switching to G4F Client](#switching-to-g4f-client)
   - [Initializing the Client](#initializing-the-client)
   - [Creating Chat Completions](#creating-chat-completions)
   - [Configuration](#configuration)
   - [Usage Examples](#usage-examples)
   - [Text Completions](#text-completions)
   - [Streaming Completions](#streaming-completions)
   - [Image Generation](#image-generation)
   - [Creating Image Variations](#creating-image-variations)
   - [Advanced Usage](#advanced-usage)
   - [Using a List of Providers with RetryProvider](#using-a-list-of-providers-with-retryprovider)
   - [Using GeminiProVision](#using-geminiprovision)
   - [Using a Vision Model](#using-a-vision-model)
   - [Command-line Chat Program](#command-line-chat-program)

  

## Introduction
Welcome to the G4F Client API, a cutting-edge tool for seamlessly integrating advanced AI capabilities into your Python applications. This guide is designed to facilitate your transition from using the OpenAI client to the G4F Client, offering enhanced features while maintaining compatibility with the existing OpenAI API.

## Getting Started
### Switching to G4F Client
**To begin using the G4F Client, simply update your import statement in your Python code:**

**Old Import:**
```python
from openai import OpenAI
```

  

**New Import:**
```python
from g4f.client import Client as OpenAI
```

  

The G4F Client preserves the same familiar API interface as OpenAI, ensuring a smooth transition process.

## Initializing the Client
To utilize the G4F Client, create a new instance. **Below is an example showcasing custom providers:**
```python
from g4f.client import Client
from g4f.Provider import BingCreateImages, OpenaiChat, Gemini

client = Client(
    provider=OpenaiChat,
    image_provider=Gemini,
    # Add any other necessary parameters
)
```

## Creating Chat Completions
**Here’s an improved example of creating chat completions:**
```python
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": "Say this is a test"
        }
    ]
    # Add any other necessary parameters
)
```

**This example:**
   - Asks a specific question `Say this is a test`
   - Configures various parameters like temperature and max_tokens for more control over the output
   - Disables streaming for a complete response

You can adjust these parameters based on your specific needs.
 

## Configuration
**You can set an `api_key` for your provider in the client and define a proxy for all outgoing requests:**
```python
from g4f.client import Client

client = Client(
    api_key="your_api_key_here",
    proxies="http://user:pass@host",
    # Add any other necessary parameters
)
```

  

## Usage Examples
### Text Completions
**Generate text completions using the `ChatCompletions` endpoint:** 
```python
from g4f.client import Client

client = Client()

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": "Say this is a test"
        }
    ]
    # Add any other necessary parameters
)

print(response.choices[0].message.content)
```

  

### Streaming Completions
**Process responses incrementally as they are generated:**
```python
from g4f.client import Client

client = Client()

stream = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {
            "role": "user",
            "content": "Say this is a test"
        }
    ],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content or "", end="")
```

  

### Image Generation
**Generate images using a specified prompt:**
```python
from g4f.client import Client

client = Client()

response = client.images.generate(
    model="flux",
    prompt="a white siamese cat"
    # Add any other necessary parameters
)

image_url = response.data[0].url

print(f"Generated image URL: {image_url}")
```


#### Base64 Response Format
```python
from g4f.client import Client

client = Client()

response = client.images.generate(
    model="flux",
    prompt="a white siamese cat",
    response_format="b64_json"
)

base64_text = response.data[0].b64_json
print(base64_text)
```

  

### Creating Image Variations
**Create variations of an existing image:**
```python
from g4f.client import Client

client = Client()

response = client.images.create_variation(
    image=open("cat.jpg", "rb"),
    model="bing"
    # Add any other necessary parameters
)

image_url = response.data[0].url

print(f"Generated image URL: {image_url}")
```

  

## Advanced Usage

### Using a List of Providers with RetryProvider
```python
from g4f.client import Client
from g4f.Provider import RetryProvider, Phind, FreeChatgpt, Liaobots
import g4f.debug

g4f.debug.logging = True
g4f.debug.version_check = False

client = Client(
    provider=RetryProvider([Phind, FreeChatgpt, Liaobots], shuffle=False)
)

response = client.chat.completions.create(
    model="",
    messages=[
        {
            "role": "user",
            "content": "Hello"
        }
    ]
)

print(response.choices[0].message.content)
```

  
### Using GeminiProVision
```python
from g4f.client import Client
from g4f.Provider.GeminiPro import GeminiPro

client = Client(
    api_key="your_api_key_here",
    provider=GeminiPro
)

response = client.chat.completions.create(
    model="gemini-pro-vision",
    messages=[
        {
            "role": "user",
            "content": "What are on this image?"
        }
    ],
    image=open("docs/waterfall.jpeg", "rb")
)

print(response.choices[0].message.content)
```

  
### Using a Vision Model
**Analyze an image and generate a description:**
```python
import g4f
import requests
from g4f.client import Client

image = requests.get("https://raw.githubusercontent.com/xtekky/gpt4free/refs/heads/main/docs/cat.jpeg", stream=True).raw
# Or: image = open("docs/cat.jpeg", "rb")

client = Client()

response = client.chat.completions.create(
    model=g4f.models.default,
    messages=[
        {
            "role": "user",
            "content": "What are on this image?"
        }
    ],
    provider=g4f.Provider.Bing,
    image=image
    # Add any other necessary parameters
)

print(response.choices[0].message.content)
```

  
## Command-line Chat Program
**Here's an example of a simple command-line chat program using the G4F Client:**
```python
import g4f
from g4f.client import Client

# Initialize the GPT client with the desired provider
client = Client()

# Initialize an empty conversation history
messages = []

while True:
    # Get user input
    user_input = input("You: ")

    # Check if the user wants to exit the chat
    if user_input.lower() == "exit":
        print("Exiting chat...")
        break  # Exit the loop to end the conversation

    # Update the conversation history with the user's message
    messages.append({"role": "user", "content": user_input})

    try:
        # Get GPT's response
        response = client.chat.completions.create(
            messages=messages,
            model=g4f.models.default,
        )

        # Extract the GPT response and print it
        gpt_response = response.choices[0].message.content
        print(f"Bot: {gpt_response}")

        # Update the conversation history with GPT's response
        messages.append({"role": "assistant", "content": gpt_response})

    except Exception as e:
        print(f"An error occurred: {e}")
```
 
This guide provides a comprehensive overview of the G4F Client API, demonstrating its versatility in handling various AI tasks, from text generation to image analysis and creation. By leveraging these features, you can build powerful and responsive applications that harness the capabilities of advanced AI models.


---  
[Return to Home](/)

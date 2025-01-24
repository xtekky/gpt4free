


# G4F - AsyncClient API Guide
The G4F AsyncClient API is a powerful asynchronous interface for interacting with various AI models. This guide provides comprehensive information on how to use the API effectively, including setup, usage examples, best practices, and important considerations for optimal performance.

## Compatibility Note
The G4F AsyncClient API is designed to be compatible with the OpenAI API, making it easy for developers familiar with OpenAI's interface to transition to G4F.

## Table of Contents
   - [Introduction](#introduction)
   - [Key Features](#key-features)
   - [Getting Started](#getting-started)
   - [Initializing the Client](#initializing-the-client)
   - [Creating Chat Completions](#creating-chat-completions)
   - [Configuration](#configuration)
   - [Explanation of Parameters](#explanation-of-parameters)
   - [Usage Examples](#usage-examples)
   - [Text Completions](#text-completions)
   - [Streaming Completions](#streaming-completions)
   - [Using a Vision Model](#using-a-vision-model)
   - [Image Generation](#image-generation)
   - [Advanced Usage](#advanced-usage)
   - [Conversation Memory](#conversation-memory)
   - [Search Tool Support](#search-tool-support)
   - [Concurrent Tasks](#concurrent-tasks-with-asynciogather)
   - [Available Models and Providers](#available-models-and-providers)
   - [Error Handling and Best Practices](#error-handling-and-best-practices)
   - [Rate Limiting and API Usage](#rate-limiting-and-api-usage)
   - [Conclusion](#conclusion)

## Introduction
The G4F AsyncClient API is an asynchronous version of the standard G4F Client API. It offers the same functionality as the synchronous API but with improved performance due to its asynchronous nature. This guide will walk you through the key features and usage of the G4F AsyncClient API.
  

## Key Features
   - **Custom Providers**: Use custom providers for enhanced flexibility.
   - **ChatCompletion Interface**: Interact with chat models through the ChatCompletion class.
   - **Streaming Responses**: Get responses iteratively as they are received.
   - **Non-Streaming Responses**: Generate complete responses in a single call.
   - **Image Generation and Vision Models**: Support for image-related tasks.

## Getting Started
### Initializing the AsyncClient
**To use the G4F `AsyncClient`, create a new instance:**
```python
from g4f.client import AsyncClient
from g4f.Provider import OpenaiChat, Gemini

client = AsyncClient(
    provider=OpenaiChat,
    image_provider=Gemini,
    # Add other parameters as needed
)
```

## Creating Chat Completions
**Here’s an improved example of creating chat completions:**
```python
response = await client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Say this is a test"
        }
    ]
     # Add other parameters as needed
)
```

**This example:**
   - Asks a specific question `Say this is a test`
   - Configures various parameters like temperature and max_tokens for more control over the output
   - Disables streaming for a complete response

You can adjust these parameters based on your specific needs.

### Configuration
**Configure the `AsyncClient` with additional settings:**
```python
client = AsyncClient(
    api_key="your_api_key_here",
    proxies="http://user:pass@host",
    # Add other parameters as needed
)
```

## Explanation of Parameters
**When using the G4F to create chat completions or perform related tasks, you can configure the following parameters:**
- **`model`**:  
  Specifies the AI model to be used for the task. Examples include `"gpt-4o"` for GPT-4 Optimized or `"gpt-4o-mini"` for a lightweight version. The choice of model determines the quality and speed of the response. Always ensure the selected model is supported by the provider.

- **`messages`**:  
  **A list of dictionaries representing the conversation context. Each dictionary contains two keys:**
      - `role`: Defines the role of the message sender, such as `"user"` (input from the user) or `"system"` (instructions to the AI).  
      - `content`: The actual text of the message.  
  **Example:**
  ```python
  [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What day is it today?"}
  ]
  ```


- **`web_search`**:
*(Optional)* A Boolean flag indicating whether to enable internet-based search capabilities. If set to **True**, the system performs a web search using the provider’s native method to retrieve up-to-date information. This is especially useful for obtaining real-time or specific details not included in the model’s training data.
  -  **Providers Supporting** `web_search`:
  - ChatGPT  
  - HuggingChat  
  - Blackbox  
  - RubiksAI

- **`provider`**:
*(Optional)* Specifies the backend provider for the API. Examples include `g4f.Provider.Blackbox` or `g4f.Provider.OpenaiChat`. Each provider may support a different subset of models and features, so select one that matches your requirements.

## Usage Examples
### Text Completions
**Generate text completions using the ChatCompletions endpoint:**
```python
import asyncio
from g4f.client import AsyncClient

async def main():
    client = AsyncClient()
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Say this is a test"
            }
        ],
        web_search = False
    )
    
    print(response.choices[0].message.content)

asyncio.run(main())
```

### Streaming Completions
**Process responses incrementally as they are generated:**
```python
import asyncio
from g4f.client import AsyncClient

async def main():
    client = AsyncClient()

    stream = client.chat.completions.stream(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": "Say this is a test"
            }
        ],
        web_search = False
    )

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")

asyncio.run(main())
```

---

### Using a Vision Model
**Analyze an image and generate a description:**
```python
import g4f
import requests
import asyncio
from g4f.client import AsyncClient
from g4f.Provider.CopilotAccount import CopilotAccount

async def main():
    client = AsyncClient(
        provider=CopilotAccount
    )

    image = requests.get("https://raw.githubusercontent.com/xtekky/gpt4free/refs/heads/main/docs/images/cat.jpeg", stream=True).raw
    # Or: image = open("docs/images/cat.jpeg", "rb")

    response = await client.chat.completions.create(
        model=g4f.models.default,
        messages=[
            {
                "role": "user",
                "content": "What's in this image?"
            }
        ],
        image=image
    )

    print(response.choices[0].message.content)

asyncio.run(main())
```

### Image Generation
**The `response_format` parameter is optional and can have the following values:**
- **If not specified (default):** The image will be saved locally, and a local path will be returned (e.g., "/images/1733331238_cf9d6aa9-f606-4fea-ba4b-f06576cba309.jpg").
- **"url":** Returns a URL to the generated image.
- **"b64_json":** Returns the image as a base64-encoded JSON string.

**Generate images using a specified prompt:**
```python
import asyncio
from g4f.client import AsyncClient

async def main():
    client = AsyncClient()
    
    response = await client.images.generate(
        prompt="a white siamese cat",
        model="flux",
        response_format="url"
        # Add any other necessary parameters
    )
    
    image_url = response.data[0].url
    print(f"Generated image URL: {image_url}")

asyncio.run(main())
```

#### Base64 Response Format
```python
import asyncio
from g4f.client import AsyncClient

async def main():
    client = AsyncClient()
    
    response = await client.images.generate(
        prompt="a white siamese cat",
        model="flux",
        response_format="b64_json"
        # Add any other necessary parameters
    )
    
    base64_text = response.data[0].b64_json
    print(base64_text)

asyncio.run(main())
```

---

### Creating Image Variations
**Create variations of an existing image:**
```python
import asyncio
from g4f.client import AsyncClient
from g4f.Provider import OpenaiChat

async def main():
    client = AsyncClient(image_provider=OpenaiChat)
    
    response = await client.images.create_variation(
        prompt="a white siamese cat",
        image=open("docs/images/cat.jpg", "rb"),
        model="dall-e-3",
        # Add any other necessary parameters
    )
    
    image_url = response.data[0].url
    print(f"Generated image URL: {image_url}")

asyncio.run(main())
```

---


## Advanced Usage

### Conversation Memory
To maintain a coherent conversation, it's important to store the context or history of the dialogue. This can be achieved by appending both the user's inputs and the bot's responses to a messages list. This allows the model to reference past exchanges when generating responses.

**The following example demonstrates how to implement conversation memory with the G4F:**
```python
import asyncio
from g4f.client import AsyncClient

class Conversation:
    def __init__(self):
        self.client = AsyncClient()
        self.history = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            }
        ]
    
    def add_message(self, role, content):
        self.history.append({
            "role": role,
            "content": content
        })
    
    async def get_response(self, user_message):
        # Add user message to history
        self.add_message("user", user_message)
        
        # Get response from AI
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.history,
            web_search=False
        )
        
        # Add AI response to history
        assistant_response = response.choices[0].message.content
        self.add_message("assistant", assistant_response)
        
        return assistant_response

async def main():
    conversation = Conversation()
    
    print("=" * 50)
    print("G4F Chat started (type 'exit' to end)".center(50))
    print("=" * 50)
    print("\nAI: Hello! How can I assist you today?")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'exit':
            print("\nGoodbye!")
            break
            
        response = await conversation.get_response(user_input)
        print("\nAI:", response)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Search Tool Support

The **Search Tool Support** feature enables triggering a web search during chat completions. This is useful for retrieving real-time or specific data, offering a more flexible solution than `web_search`.

**Example Usage:**
```python
import asyncio
from g4f.client import AsyncClient

async def main():
    client = AsyncClient()

    tool_calls = [
        {
            "function": {
                "arguments": {
                    "query": "Latest advancements in AI",
                    "max_results": 5,
                    "max_words": 2500,
                    "backend": "auto",
                    "add_text": True,
                    "timeout": 5
                },
                "name": "search_tool"
            },
            "type": "function"
        }
    ]

    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": "Tell me about recent advancements in AI."
            }
        ],
        tool_calls=tool_calls
    )

    print(response.choices[0].message.content)

if __name__ == "__main__":
    asyncio.run(main())
```

**Parameters for `search_tool`:**
- **`query`**: The search query string.
- **`max_results`**: Number of search results to retrieve.
- **`max_words`**: Maximum number of words in the response.
- **`backend`**: The backend used for search (e.g., `"api"`).
- **`add_text`**: Whether to include text snippets in the response.
- **`timeout`**: Maximum time (in seconds) for the search operation.

**Advantages of Search Tool Support:**
- Works with any provider, irrespective of `web_search` support.
- Offers more customization and control over the search process.
- Bypasses provider-specific limitations.

---

### Using a List of Providers with RetryProvider
```python
import asyncio
from g4f.client import AsyncClient

import g4f.debug
g4f.debug.logging = True
g4f.debug.version_check = False

from g4f.Provider import RetryProvider, Phind, FreeChatgpt, Liaobots

async def main():
    client = AsyncClient(provider=RetryProvider([Phind, FreeChatgpt, Liaobots], shuffle=False)
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Hello"
            }
        ],
        web_search = False
    )
    
    print(response.choices[0].message.content)

asyncio.run(main())
```

---

### Concurrent Tasks with asyncio.gather
**Execute multiple tasks concurrently:**
```python
import asyncio
from g4f.client import AsyncClient

async def main():
    client = AsyncClient()
    
    task1 = client.chat.completions.create(
        model=None,
        messages=[
            {
                "role": "user",
                "content": "Say this is a test"
            }
        ]
    )
    
    task2 = client.images.generate(
        model="flux",
        prompt="a white siamese cat",
        response_format="url"
    )
    
    try:
        chat_response, image_response = await asyncio.gather(task1, task2)
        
        print("Chat Response:")
        print(chat_response.choices[0].message.content)
        
        print("\nImage Response:")
        print(image_response.data[0].url)
    except Exception as e:
        print(f"An error occurred: {e}")

asyncio.run(main())
```

## Available Models and Providers
The G4F AsyncClient supports a wide range of AI models and providers, allowing you to choose the best option for your specific use case. 

**Here's a brief overview of the available models and providers:**
**Models**
   - GPT-3.5-Turbo
   - GPT-4o-Mini
   - GPT-4
   - DALL-E 3
   - Gemini
   - Claude (Anthropic)
   - And more...

**Providers**
   - OpenAI
   - Google (for Gemini)
   - Anthropic
   - Microsoft Copilot 
   - Custom providers

**To use a specific model or provider, specify it when creating the client or in the API call:**
```python
client = AsyncClient(provider=g4f.Provider.OpenaiChat)

# or

response = await client.chat.completions.create(
    model="gpt-4",
    provider=g4f.Provider.CopilotAccount,
    messages=[
        {
            "role": "user",
            "content": "Hello, world!"
        }
    ]
)
```

## Error Handling and Best Practices
Implementing proper error handling and following best practices is crucial when working with the G4F AsyncClient API. This ensures your application remains robust and can gracefully handle various scenarios. 

**Here are some key practices to follow:**

1. **Use try-except blocks to catch and handle exceptions:**
```python
try:
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Hello, world!"
            }
        ]
    )
except Exception as e:
    print(f"An error occurred: {e}")
```

2. **Check the response status and handle different scenarios:**
```python
if response.choices:
    print(response.choices[0].message.content)
else:
    print("No response generated")
```

3. **Implement retries for transient errors:**
```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def make_api_call():
    # Your API call here
    pass
```

## Rate Limiting and API Usage
When working with the G4F AsyncClient API, it's important to implement rate limiting and monitor your API usage. This helps ensure fair usage, prevents overloading the service, and optimizes your application's performance. Here are some key strategies to consider:
  

1. **Implement rate limiting in your application:**
```python
import asyncio
from aiolimiter import AsyncLimiter

rate_limit = AsyncLimiter(max_rate=10, time_period=1)  # 10 requests per second

async def make_api_call():
    async with rate_limit:
        # Your API call here
        pass
```

2. **Monitor your API usage and implement logging:**
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def make_api_call():
    try:
        response = await client.chat.completions.create(...)
        logger.info(f"API call successful. Tokens used: {response.usage.total_tokens}")
    except Exception as e:
        logger.error(f"API call failed: {e}")
```

3. **Use caching to reduce API calls for repeated queries:**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_response(query):
    # Your API call here
    pass
```

## Conclusion
The G4F AsyncClient API provides a powerful and flexible way to interact with various AI models asynchronously. By leveraging its features and following best practices, you can build efficient and responsive applications that harness the power of AI for text generation, image analysis, and image creation.

Remember to handle errors gracefully, implement rate limiting, and monitor your API usage to ensure optimal performance and reliability in your applications.

---

[Return to Home](/)

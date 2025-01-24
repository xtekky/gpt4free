

# G4F Client API Guide

## Table of Contents
   - [Introduction](#introduction)
   - [Getting Started](#getting-started)
   - [Switching to G4F Client](#switching-to-g4f-client)
   - [Initializing the Client](#initializing-the-client)
   - [Creating Chat Completions](#creating-chat-completions)
   - [Configuration](#configuration)
   - [Explanation of Parameters](#explanation-of-parameters)
   - [Usage Examples](#usage-examples)
   - [Text Completions](#text-completions)
   - [Streaming Completions](#streaming-completions)
   - [Using a Vision Model](#using-a-vision-model)
   - [Image Generation](#image-generation)
   - [Creating Image Variations](#creating-image-variations)
   - [Advanced Usage](#advanced-usage)
   - [Conversation Memory](#conversation-memory)
   - [Search Tool Support](#search-tool-support)
   - [Using a List of Providers with RetryProvider](#using-a-list-of-providers-with-retryprovider)
   - [Using a Vision Model](#using-a-vision-model)
   - [Command-line Chat Program](#command-line-chat-program)

## Introduction
Welcome to the G4F Client API, a cutting-edge tool for seamlessly integrating advanced AI capabilities into your Python applications. This guide is designed to facilitate your transition from using the OpenAI client to the G4F Client, offering enhanced features while maintaining compatibility with the existing OpenAI API.

---

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

---

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

---

## Creating Chat Completions
**Here’s an improved example of creating chat completions:**
```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
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

---

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

- **`provider`**:
*(Optional)* Specifies the backend provider for the API. Examples include `g4f.Provider.Blackbox` or `g4f.Provider.OpenaiChat`. Each provider may support a different subset of models and features, so select one that matches your requirements.

- **`web_search`** (Optional):  
  Boolean flag indicating whether to enable internet-based search capabilities. This is useful for obtaining real-time or specific details not included in the model’s training data.

#### Providers Limitation
The `web_search` argument is **limited to specific providers**, including:
  - ChatGPT
  - HuggingChat
  - Blackbox
  - RubiksAI

If your chosen provider does not support `web_search`, it will not function as expected.  

**Alternative Solution:**  
Instead of relying on the `web_search` argument, you can use the more versatile **Search Tool Support**, which allows for highly customizable web search operations. The search tool enables you to define parameters such as query, number of results, word limit, and timeout, offering greater control over search capabilities.

---

## Usage Examples

### Text Completions
**Generate text completions using the `ChatCompletions` endpoint:** 
```python
from g4f.client import Client

client = Client()

response = client.chat.completions.create(
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
    web_search = False
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content or "", end="")
```
---

### Using a Vision Model
**Analyze an image and generate a description:**
```python
import g4f
import requests

from g4f.client import Client
from g4f.Provider.GeminiPro import GeminiPro

# Initialize the GPT client with the desired provider and api key
client = Client(
    api_key="your_api_key_here",
    provider=GeminiPro
)

image = requests.get("https://raw.githubusercontent.com/xtekky/gpt4free/refs/heads/main/docs/images/cat.jpeg", stream=True).raw
# Or: image = open("docs/images/cat.jpeg", "rb")

response = client.chat.completions.create(
    model=g4f.models.default,
    messages=[
        {
            "role": "user",
            "content": "What's in this image?"
        }
    ],
    image=image
    # Add any other necessary parameters
)

print(response.choices[0].message.content)
```

---

### Image Generation
**The `response_format` parameter is optional and can have the following values:**
- **If not specified (default):** The image will be saved locally, and a local path will be returned (e.g., "/images/1733331238_cf9d6aa9-f606-4fea-ba4b-f06576cba309.jpg").
- **"url":** Returns a URL to the generated image.
- **"b64_json":** Returns the image as a base64-encoded JSON string.

**Generate images using a specified prompt:**
```python
from g4f.client import Client

client = Client()

response = client.images.generate(
    model="flux",
    prompt="a white siamese cat",
    response_format="url"
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
    # Add any other necessary parameters
)

base64_text = response.data[0].b64_json
print(base64_text)
```

### Creating Image Variations
**Create variations of an existing image:**
```python
from g4f.client import Client
from g4f.Provider import OpenaiChat

client = Client(
    image_provider=OpenaiChat
)

response = client.images.create_variation(
    image=open("docs/images/cat.jpg", "rb"),
    model="dall-e-3",
    # Add any other necessary parameters
)

image_url = response.data[0].url

print(f"Generated image URL: {image_url}")
```

---

## Advanced Usage

### Conversation Memory
To maintain a coherent conversation, it's important to store the context or history of the dialogue. This can be achieved by appending both the user's inputs and the bot's responses to a messages list. This allows the model to reference past exchanges when generating responses.

**The conversation history consists of messages with different roles:**
- `system`: Initial instructions that define the AI's behavior
- `user`: Messages from the user
- `assistant`: Responses from the AI

**The following example demonstrates how to implement conversation memory with the G4F:**
```python
from g4f.client import Client

class Conversation:
    def __init__(self):
        self.client = Client()
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
    
    def get_response(self, user_message):
        # Add user message to history
        self.add_message("user", user_message)
        
        # Get response from AI
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.history,
            web_search=False
        )
        
        # Add AI response to history
        assistant_response = response.choices[0].message.content
        self.add_message("assistant", assistant_response)
        
        return assistant_response

def main():
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
            
        response = conversation.get_response(user_input)
        print("\nAI:", response)

if __name__ == "__main__":
    main()
```

**Key Features:**
- Maintains conversation context through a message history
- Includes system instructions for AI behavior
- Automatically stores both user inputs and AI responses
- Simple and clean implementation using a class-based approach

**Usage Example:**
```python
conversation = Conversation()
response = conversation.get_response("Hello, how are you?")
print(response)
```

**Note:**
The conversation history grows with each interaction. For long conversations, you might want to implement a method to limit the history size or clear old messages to manage token usage.

---

## Search Tool Support

The **Search Tool Support** feature enables triggering a web search during chat completions. This is useful for retrieving real-time or specific data, offering a more flexible solution than `web_search`.

**Example Usage**:
```python
from g4f.client import Client

client = Client()

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

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Tell me about recent advancements in AI."}
    ],
    tool_calls=tool_calls
)

print(response.choices[0].message.content)
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

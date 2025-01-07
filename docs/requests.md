# G4F Requests API Guide

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Installing Dependencies](#installing-dependencies)
- [Making API Requests](#making-api-requests)
- [Text Generation](#text-generation)
  - [Using the Chat Completions Endpoint](#using-the-chat-completions-endpoint)
  - [Streaming Text Generation](#streaming-text-generation)
- [Model Retrieval](#model-retrieval)
  - [Fetching Available Models](#fetching-available-models)
- [Image Generation](#image-generation)
  - [Creating Images with AI](#creating-images-with-ai)
- [Advanced Usage](#advanced-usage)

## Introduction

Welcome to the G4F Requests API Guide, a powerful tool for leveraging AI capabilities directly from your Python applications using HTTP requests. This guide will take you through the steps of setting up requests to interact with AI models for a variety of tasks, from text generation to image creation.

## Getting Started

### Installing Dependencies

Ensure you have the `requests` library installed in your environment. You can install it via `pip` if needed:

```bash
pip install requests
```

This guide provides examples on how to make API requests using Python's `requests` library, focusing on tasks such as text and image generation, as well as retrieving available models.

## Making API Requests

Before diving into specific functionalities, it's essential to understand how to structure your API requests. All endpoints assume that your server is running locally at `http://localhost`. If your server is running on a different port, adjust the URLs accordingly (e.g., `http://localhost:8000`).

## Text Generation

### Using the Chat Completions Endpoint

To generate text responses using the chat completions endpoint, follow this example:

```python
import requests

# Define the payload
payload = {
    "model": "gpt-4o",
    "temperature": 0.9,
    "messages": [{"role": "system", "content": "Hello, how are you?"}]
}

# Send the POST request to the chat completions endpoint
response = requests.post("http://localhost/v1/chat/completions", json=payload)

# Check if the request was successful
if response.status_code == 200:
    # Print the response text
    print(response.text)
else:
    print(f"Request failed with status code {response.status_code}")
    print("Response:", response.text)
```

**Explanation:**
- This request sends a conversation context to the model, which in turn generates and returns a response.
- The `temperature` parameter controls the randomness of the output.

### Streaming Text Generation

For scenarios where you want to receive partial responses or stream data as it's generated, you can utilize the streaming capabilities of the API. Here's how you can implement streaming text generation using Python's `requests` library:

```python
import requests
import json

def fetch_response(url, model, messages):
    """
    Sends a POST request to the streaming chat completions endpoint.

    Args:
        url (str): The API endpoint URL.
        model (str): The model to use for text generation.
        messages (list): A list of message dictionaries.

    Returns:
        requests.Response: The streamed response object.
    """
    payload = {"model": model, "messages": messages, "stream": True}
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    response = requests.post(url, headers=headers, json=payload, stream=True)
    if response.status_code != 200:
        raise Exception(
            f"Failed to send message: {response.status_code} {response.text}"
        )
    return response

def process_stream(response):
    """
    Processes the streamed response and extracts messages.

    Args:
        response (requests.Response): The streamed response object.
        output_queue (Queue): A queue to store the extracted messages.
    """
    for line in response.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line == "data: [DONE]":
                print("\n\nConversation completed.")
                break
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    message = data.get("choices", [{}])[0].get("delta", {}).get("content")
                    if message:
                        print(message, end="", flush=True)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    continue

# Define the API endpoint
chat_url = "http://localhost:8080/v1/chat/completions"

# Define the payload
model = ""
messages = [{"role": "user", "content": "Hello, how are you?"}]

try:
    # Fetch the streamed response
    response = fetch_response(chat_url, model, messages)
    
    # Process the streamed response
    process_stream(response)

except Exception as e:
    print(f"An error occurred: {e}")
```

**Explanation:**
- **`fetch_response` Function:**
  - Sends a POST request to the streaming chat completions endpoint with the specified model and messages.
  - Sets `stream` parameter to `true` to enable streaming.
  - Raises an exception if the request fails.

- **`process_stream` Function:**
  - Iterates over each line in the streamed response.
  - Decodes the line and checks for the termination signal `"data: [DONE]"`.
  - Parses lines that start with `"data: "` to extract the message content.

- **Main Execution:**
  - Defines the API endpoint, model, and messages.
  - Fetches and processes the streamed response.
  - Retrieves and prints messages.

**Usage Tips:**
- Ensure your local server supports streaming.
- Adjust the `chat_url` if your local server runs on a different port or path.
- Use threading or asynchronous programming for handling streams in real-time applications.

## Model Retrieval

### Fetching Available Models

To retrieve a list of available models, you can use the following function:

```python
import requests

def fetch_models():
    """
    Retrieves the list of available models from the API.

    Returns:
        dict: A dictionary containing available models or an error message.
    """
    url = "http://localhost/v1/models/"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for HTTP issues
        return response.json()  # Parse and return the JSON response
    except Exception as e:
        return {"error": str(e)}  # Return an error message if something goes wrong

models = fetch_models()

print(models)
```

**Explanation:**
- The `fetch_models` function makes a GET request to the models endpoint.
- It handles HTTP errors and returns a parsed JSON response containing available models or an error message.

## Image Generation

### Creating Images with AI

The following function demonstrates how to generate images using a specified model:

```python
import requests

def generate_image(prompt: str, model: str = "flux-4o"):
    """
    Generates an image based on the provided text prompt.

    Args:
        prompt (str): The text prompt for image generation.
        model (str, optional): The model to use for image generation. Defaults to "flux-4o".

    Returns:
        tuple: A tuple containing the image URL, caption, and the full response.
    """
    payload = {
        "model": model,
        "temperature": 0.9,
        "prompt": prompt.replace(" ", "+"),
    }

    try:
        response = requests.post("http://localhost/v1/images/generate", json=payload)
        response.raise_for_status()
        res = response.json()

        data = res.get("data")
        if not data or not isinstance(data, list):
            raise ValueError("Invalid 'data' in response")

        image_url = data[0].get("url")
        if not image_url:
            raise ValueError("No 'url' found in response data")

        timestamp = res.get("created")
        caption = f"Prompt: {prompt}\nCreated: {timestamp}\nModel: {model}"
        return image_url, caption, res

    except Exception as e:
        return None, f"Error: {e}", None

prompt = "A tiger in a forest"

image_url, caption, res = generate_image(prompt)

print("API Response:", res)
print("Image URL:", image_url)
print("Caption:", caption)
```

**Explanation:**
- The `generate_image` function constructs a request to create an image based on a text prompt.
- It handles responses and possible errors, ensuring a URL and caption are returned if successful.

## Advanced Usage

This guide has demonstrated basic usage scenarios for the G4F Requests API. The API provides robust capabilities for integrating advanced AI into your applications. You can expand upon these examples to fit more complex workflows and tasks, ensuring your applications are built with cutting-edge AI features.

### Handling Concurrency and Asynchronous Requests

For applications requiring high performance and non-blocking operations, consider using asynchronous programming libraries such as `aiohttp` or `httpx`. Here's an example using `aiohttp`:

```python
import aiohttp
import asyncio
import json
from queue import Queue

async def fetch_response_async(url, model, messages, output_queue):
    """
    Asynchronously sends a POST request to the streaming chat completions endpoint and processes the stream.

    Args:
        url (str): The API endpoint URL.
        model (str): The model to use for text generation.
        messages (list): A list of message dictionaries.
        output_queue (Queue): A queue to store the extracted messages.
    """
    payload = {"model": model, "messages": messages, "stream": True}
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Failed to send message: {resp.status} {text}")
            
            async for line in resp.content:
                decoded_line = line.decode('utf-8').strip()
                if decoded_line == "data: [DONE]":
                    break
                if decoded_line.startswith("data: "):
                    try:
                        data = json.loads(decoded_line[6:])
                        message = data.get("choices", [{}])[0].get("delta", {}).get("content")
                        if message:
                            output_queue.put(message)
                    except json.JSONDecodeError:
                        continue

async def main():
    chat_url = "http://localhost/v1/chat/completions"
    model = "gpt-4o"
    messages = [{"role": "system", "content": "Hello, how are you?"}]
    output_queue = Queue()

    try:
        await fetch_response_async(chat_url, model, messages, output_queue)
        
        while not output_queue.empty():
            msg = output_queue.get()
            print(msg)

    except Exception as e:
        print(f"An error occurred: {e}")

# Run the asynchronous main function
asyncio.run(main())
```

**Explanation:**
- **`aiohttp` Library:** Facilitates asynchronous HTTP requests, allowing your application to handle multiple requests concurrently without blocking.
- **`fetch_response_async` Function:**
  - Sends an asynchronous POST request to the streaming chat completions endpoint.
  - Processes the streamed response line by line.
  - Extracts messages and enqueues them into `output_queue`.
- **`main` Function:**
  - Defines the API endpoint, model, and messages.
  - Initializes a `Queue` to store incoming messages.
  - Invokes the asynchronous fetch function and processes the messages.

**Benefits:**
- **Performance:** Handles multiple requests efficiently, reducing latency in high-throughput applications.
- **Scalability:** Easily scales with increasing demand, making it suitable for production environments.

**Note:** Ensure you have `aiohttp` installed:

```bash
pip install aiohttp
```

## Conclusion

By following this guide, you can effectively integrate the G4F Requests API into your Python applications, enabling powerful AI-driven functionalities such as text and image generation, model retrieval, and handling streaming data. Whether you're building simple scripts or complex, high-performance applications, the examples provided offer a solid foundation to harness the full potential of AI in your projects.

Feel free to customize and expand upon these examples to suit your specific needs. If you encounter any issues or have further questions, don't hesitate to seek assistance or refer to additional resources.

---

# Additional Notes

1. **Adjusting the Base URL:**
   - The guide assumes your API server is accessible at `http://localhost`. If your server runs on a different port (e.g., `8000`), update the URLs accordingly:
     ```python
     # Example for port 8000
     chat_url = "http://localhost:8000/v1/chat/completions"
     ```
   
2. **Environment Variables (Optional):**
   - For better flexibility and security, consider using environment variables to store your base URL and other sensitive information.
     ```python
     import os

     BASE_URL = os.getenv("API_BASE_URL", "http://localhost")
     chat_url = f"{BASE_URL}/v1/chat/completions"
     ```

3. **Error Handling:**
   - Always implement robust error handling to gracefully manage unexpected scenarios, such as network failures or invalid responses.

4. **Security Considerations:**
   - Ensure that your local API server is secured, especially if accessible over a network. Implement authentication mechanisms if necessary.

5. **Testing:**
   - Utilize tools like [Postman](https://www.postman.com/) or [Insomnia](https://insomnia.rest/) for testing your API endpoints before integrating them into your code.

6. **Logging:**
   - Implement logging to monitor the behavior of your applications, which is crucial for debugging and maintaining your systems.

---

[Return to Home](/)

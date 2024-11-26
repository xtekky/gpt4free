# G4F Requests API Guide

## Table of Contents
   - [Introduction](#introduction)
   - [Getting Started](#getting-started)
   - [Making API Requests](#making-api-requests)
   - [Text Generation](#text-generation)
   - [Model Retrieval](#model-retrieval)
   - [Image Generation](#image-generation)
   - [Advanced Usage](#advanced-usage)

## Introduction
Welcome to the G4F Requests API Guide, a powerful tool for leveraging AI capabilities directly from your Python applications using HTTP requests. This guide will take you through the steps of setting up requests to interact with AI models for a variety of tasks, from text generation to image creation.

## Getting Started
### Making API Requests
To start using the G4F Requests API, ensure you have the `requests` library installed in your environment. You can install it via pip if needed:

```bash
pip install requests
```

This guide provides examples on how to make API requests using Python's `requests` library, focusing on tasks such as text and image generation, as well as retrieving available models.

## Text Generation
### Using the Chat Completions Endpoint
To generate text responses using the chat completions endpoint, follow this example:

```python
import requests

payload = {
    "model": "gpt-4o",
    "temperature": 0.9,
    "messages": [{"role": "system", "content": "Hello, how are you?"}]
}

# Use the JSON parameter to send the payload
response = requests.post("https://g4f-api.g4f-api/v1/chat/completions", json=payload)

# Print the response text
print(response.text)
```

**Explanation:**
- This request sends a conversation context to the model, which in turn generates and returns a response.
- The `temperature` parameter controls the randomness of the output.

## Model Retrieval
### Fetching Available Models
To retrieve a list of available models, you can use the following function:

```python
import requests

def fetch_models():
    url = "https://g4f-api.g4f-api/v1/models/"
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
    payload = {
        "model": model,
        "temperature": 0.9,
        "prompt": prompt.replace(" ", "+"),
    }

    try:
        response = requests.post("https://g4f-api.g4f-api/v1/images/generate", json=payload)
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
        return None, f"Error: {e}"

prompt = "A tiger"

image_url, caption, res = generate_image(prompt)

print("API Response:", res)
```

**Explanation:**
- The `generate_image` function constructs a request to create an image based on a text prompt.
- It handles responses and possible errors, ensuring a URL and caption are returned if successful.

## Advanced Usage
This guide has demonstrated basic usage scenarios for the G4F Requests API. The API provides robust capabilities for integrating advanced AI into your applications. You can expand upon these examples to fit more complex workflows and tasks, ensuring your applications are built with cutting-edge AI features.

This guide offers a structured approach for leveraging the G4F Requests API effectively, providing your applications with advanced functionalities boosted by powerful AI models.

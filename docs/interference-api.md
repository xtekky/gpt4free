# G4F - Interference API Usage Guide
  

## Table of Contents
   - [Introduction](#introduction)
   - [Running the Interference API](#running-the-interference-api)
   - [From PyPI Package](#from-pypi-package)
   - [From Repository](#from-repository)
   - [Using the Interference API](#using-the-interference-api)
   - [Basic Usage](#basic-usage)
   - [With OpenAI Library](#with-openai-library)
   - [With Requests Library](#with-requests-library)
   - [Selecting a Provider](#selecting-a-provider)
   - [Key Points](#key-points)
   - [Conclusion](#conclusion)

## Introduction
The G4F Interference API is a powerful tool that allows you to serve other OpenAI integrations using G4F (Gpt4free). It acts as a proxy, translating requests intended for the OpenAI API into requests compatible with G4F providers. This guide will walk you through the process of setting up, running, and using the Interference API effectively.
  

## Running the Interference API
**You can run the Interference API in two ways:** using the PyPI package or from the repository.
  

### From PyPI Package
**To run the Interference API directly from the G4F PyPI package, use the following Python code:**

```python
from g4f.api import run_api

run_api()
```

  
### From Repository
**If you prefer to run the Interference API from the cloned repository, you have two options:**

1. **Using the command line:**
```bash
g4f api
```

2. **Using Python:**
```bash
python -m g4f.api.run
```

**Once running, the API will be accessible at:** `http://localhost:1337/v1`

**(Advanced) Bind to custom port:**
```bash
python -m g4f.cli api --bind "0.0.0.0:2400" 
```

## Using the Interference API

### Basic Usage
**You can interact with the Interference API using curl commands for both text and image generation:**

**For text generation:**
```bash
curl -X POST "http://localhost:1337/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
           "messages": [
             {
               "role": "user",
               "content": "Hello"
             }
           ],
           "model": "gpt-4o-mini"
         }'
```

**For image generation:**
1. **url:**
```bash
curl -X POST "http://localhost:1337/v1/images/generate" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "a white siamese cat",
           "model": "flux",
           "response_format": "url"
         }'
```

2. **b64_json**
```bash
curl -X POST "http://localhost:1337/v1/images/generate" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "a white siamese cat",
           "model": "flux",
           "response_format": "b64_json"
         }'
```


### With OpenAI Library

**You can use the Interference API with the OpenAI Python library by changing the `base_url`:**
```python
from openai import OpenAI

client = OpenAI(
    api_key="",
    base_url="http://localhost:1337/v1"
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a poem about a tree"}],
    stream=True,
)

if isinstance(response, dict):
    # Not streaming
    print(response.choices[0].message.content)
else:
    # Streaming
    for token in response:
        content = token.choices[0].delta.content
        if content is not None:
            print(content, end="", flush=True)

```
  

### With Requests Library

**You can also send requests directly to the Interference API using the `requests` library:**
```python
import requests

url = "http://localhost:1337/v1/chat/completions"

body = {
    "model": "gpt-4o-mini",
    "stream": False,
    "messages": [
        {"role": "assistant", "content": "What can you do?"}
    ]
}

json_response = requests.post(url, json=body).json().get('choices', [])

for choice in json_response:
    print(choice.get('message', {}).get('content', ''))

```

## Selecting a Provider

**Provider Selection**: [How to Specify a Provider?](docs/selecting_a_provider.md)

Selecting the right provider is a key step in configuring the G4F Interference API to suit your needs. Refer to the guide linked above for detailed instructions on choosing and specifying a provider.

## Key Points
   - The Interference API translates OpenAI API requests into G4F provider requests.
   - It can be run from either the PyPI package or the cloned repository.
   - The API supports usage with the OpenAI Python library by changing the `base_url`.
   - Direct requests can be sent to the API endpoints using libraries like `requests`.
   - Both text and image generation are supported.

  
## Conclusion
The G4F Interference API provides a seamless way to integrate G4F with existing OpenAI-based applications and tools. By following this guide, you should now be able to set up, run, and use the Interference API effectively. Whether you're using it for text generation, image creation, or as a drop-in replacement for OpenAI in your projects, the Interference API offers flexibility and power for your AI-driven applications.
 

---

[Return to Home](/)

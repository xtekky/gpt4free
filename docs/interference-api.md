
# G4F - Interference API Usage Guide


## Table of Contents
   - [Introduction](#introduction)
   - [Running the Interference API](#running-the-interference-api)
   - [From PyPI Package](#from-pypi-package)
   - [From Repository](#from-repository)
   - [Usage with OpenAI Library](#usage-with-openai-library)
   - [Usage with Requests Library](#usage-with-requests-library)
   - [Key Points](#key-points)

## Introduction
The Interference API allows you to serve other OpenAI integrations with G4F. It acts as a proxy, translating requests to the OpenAI API into requests to the G4F providers.

## Running the Interference API

### From PyPI Package
**You can run the Interference API directly from the G4F PyPI package:**
```python
from g4f.api import run_api

run_api()
```

  

### From Repository
Alternatively, you can run the Interference API from the cloned repository.  

**Run the server with:**
```bash
g4f api
```
or
```bash
python -m g4f.api.run
```

  

## Usage with OpenAI Library

  

```python
from openai import OpenAI

client = OpenAI(
    api_key="",
    # Change the API base URL to the local interference API
    base_url="http://localhost:1337/v1"  
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "write a poem about a tree"}],
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

  

## Usage with Requests Library
You can also send requests directly to the Interference API using the requests library. 

**Send a POST request to `/v1/chat/completions` with the request body containing the model and other parameters:**
```python
import requests

url = "http://localhost:1337/v1/chat/completions"
body = {
    "model": "gpt-3.5-turbo", 
    "stream": False,
    "messages": [
        {"role": "assistant", "content": "What can you do?"}
    ]
}

json_response = requests.post(url, json=body).json().get('choices', [])

for choice in json_response:
    print(choice.get('message', {}).get('content', ''))
```

  

## Key Points
- The Interference API translates OpenAI API requests into G4F provider requests
- You can run it from the PyPI package or the cloned repository
- It supports usage with the OpenAI Python library by changing the `base_url`
- Direct requests can be sent to the API endpoints using libraries like `requests`

  
**_The Interference API allows easy integration of G4F with existing OpenAI-based applications and tools._**

---

[Return to Home](/)


### Selecting a Provider

**The Interference API also allows you to specify which provider(s) to use for processing requests. This is done using the `provider` parameter, which can be included alongside the `model` parameter in your API requests. Providers can be specified as a space-separated string of provider IDs.**

#### How to Specify a Provider

To select one or more providers, include the `provider` parameter in your request body. This parameter accepts a string of space-separated provider IDs. Each ID represents a specific provider available in the system.

#### Example: Getting a List of Available Providers

Use the following Python code to fetch the list of available providers:

```python
import requests

url = "http://localhost:1337/v1/providers"

response = requests.get(url, headers={"accept": "application/json"})
providers = response.json()

for provider in providers:
    print(f"ID: {provider['id']}, URL: {provider['url']}")
```

#### Example: Getting Detailed Information About a Specific Provider

Retrieve details about a specific provider, including supported models and parameters:

```python
provider_id = "HuggingChat"
url = f"http://localhost:1337/v1/providers/{provider_id}"

response = requests.get(url, headers={"accept": "application/json"})
provider_details = response.json()

print(f"Provider ID: {provider_details['id']}")
print(f"Supported Models: {provider_details['models']}")
print(f"Parameters: {provider_details['params']}")
```

#### Example: Using a Single Provider in Text Generation

Specify a single provider (`HuggingChat`) in the request body:

```python
import requests

url = "http://localhost:1337/v1/chat/completions"

payload = {
    "model": "gpt-4o-mini",
    "provider": "HuggingChat",
    "messages": [
        {"role": "user", "content": "Write a short story about a robot"}
    ]
}

response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
data = response.json()

if "choices" in data:
    for choice in data["choices"]:
        print(choice["message"]["content"])
else:
    print("No response received")
```

#### Example: Using Multiple Providers in Text Generation

Specify multiple providers by separating their IDs with a space:

```python
import requests

url = "http://localhost:1337/v1/chat/completions"

payload = {
    "model": "gpt-4o-mini",
    "provider": "HuggingChat AnotherProvider",
    "messages": [
        {"role": "user", "content": "What are the benefits of AI in education?"}
    ]
}

response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
data = response.json()

if "choices" in data:
    for choice in data["choices"]:
        print(choice["message"]["content"])
else:
    print("No response received")
```

#### Example: Using a Provider for Image Generation

You can also use the `provider` parameter for image generation:

```python
import requests

url = "http://localhost:1337/v1/images/generate"

payload = {
    "prompt": "a futuristic cityscape at sunset",
    "model": "flux",
    "provider": "HuggingSpace",
    "response_format": "url"
}

response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
data = response.json()

if "data" in data:
    for item in data["data"]:
        print(f"Image URL: {item['url']}")
else:
    print("No response received")
```

### Key Points About Providers
- **Flexibility:** Use the `provider` parameter to select one or more providers for your requests.
- **Discoverability:** Fetch available providers using the `/providers` endpoint.
- **Compatibility:** Check provider details to ensure support for the desired models and parameters.

By specifying providers in a space-separated string, you can efficiently target specific providers or combine multiple providers in a single request. This approach gives you fine-grained control over how your requests are processed.


---

[Go to Interference API Docs](docs/interference-api.md)
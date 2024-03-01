### Interference openai-proxy API

#### Run interference API from PyPi package

```python
from g4f.api import run_api

run_api()
```

#### Run interference API from repo

Run server:

```sh
g4f api
```

or

```sh
python -m g4f.api.run
```

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

####  API usage (POST)
Send the POST request to /v1/chat/completions with body containing the `model` method. This example uses python with requests library:
```python
import requests
url = "http://localhost:1337/v1/chat/completions"
body = {
    "model": "gpt-3.5-turbo-16k",
    "stream": False,
    "messages": [
        {"role": "assistant", "content": "What can you do?"}
    ]
}
json_response = requests.post(url, json=body).json().get('choices', [])

for choice in json_response:
    print(choice.get('message', {}).get('content', ''))
```

[Return to Home](/)
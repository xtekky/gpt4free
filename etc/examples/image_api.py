import requests
url = "http://localhost:1337/v1/images/generations"
body = {
    "model": "dall-e",
    "prompt": "hello world user",
    #"response_format": "b64_json",
}
data = requests.post(url, json=body, stream=True).json()
print(data)
import requests
url = "http://localhost:1337/v1/images/generations"
body = {
    "prompt": "heaven for dogs",
    "provider": "OpenaiAccount",
    "response_format": "b64_json",
}
data = requests.post(url, json=body, stream=True).json()
print(data)
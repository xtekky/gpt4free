import requests
import json
url = "http://localhost:1337/v1/chat/completions"
body = {
    "model": "",
    "provider": "",
    "stream": True,
    "messages": [
        {"role": "user", "content": "What can you do? Who are you?"}
    ]
}
response = requests.post(url, json=body, stream=True)
response.raise_for_status()
for line in response.iter_lines():
    if line.startswith(b"data: "):
        try:
            json_data = json.loads(line[6:])
            if json_data.get("error"):
                print(json_data)
                break
            print(json_data.get("choices", [{"delta": {}}])[0]["delta"].get("content", ""), end="")
        except json.JSONDecodeError:
            pass
print()
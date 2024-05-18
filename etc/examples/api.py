import requests
import json
url = "http://localhost:1337/v1/chat/completions"
body = {
    "model": "",
    "provider": "",
    "stream": True,
    "messages": [
        {"role": "assistant", "content": "What can you do? Who are you?"}
    ]
}
lines = requests.post(url, json=body, stream=True).iter_lines()
for line in lines:
    if line.startswith(b"data: "):
        try:
            print(json.loads(line[6:]).get("choices", [{"delta": {}}])[0]["delta"].get("content", ""), end="")
        except json.JSONDecodeError:
            pass
print()
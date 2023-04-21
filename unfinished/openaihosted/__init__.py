import requests
import json
import re

headers = {
    'authority': 'openai.a2hosted.com',
    'accept': 'text/event-stream',
    'accept-language': 'en-US,en;q=0.9,id;q=0.8,ja;q=0.7',
    'cache-control': 'no-cache',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'cross-site',
    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36 Edg/113.0.0.0',
}

def create_query_param(conversation):
    encoded_conversation = json.dumps(conversation)
    return encoded_conversation.replace(" ", "%20").replace('"', '%22').replace("'", "%27")

user_input = input("Enter your message: ")

data = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "Hello! How can I assist you today?"},
    {"role": "user", "content": user_input},
]

query_param = create_query_param(data)
url = f'https://openai.a2hosted.com/chat?q={query_param}'

response = requests.get(url, headers=headers, stream=True)

for message in response.iter_content(chunk_size=1024):
    message = message.decode('utf-8')
    msg_match, num_match = re.search(r'"msg":"(.*?)"', message), re.search(r'\[DONE\] (\d+)', message)
    if msg_match: print(msg_match.group(1))
    if num_match: print(num_match.group(1))

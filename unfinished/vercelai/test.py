import requests
from base64    import b64decode, b64encode
from json      import loads
from json      import dumps

headers = {
    'Accept': '*/*',
    'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8',
    'Connection': 'keep-alive',
    'Referer': 'https://play.vercel.ai/',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36',
    'sec-ch-ua': '"Chromium";v="110", "Google Chrome";v="110", "Not:A-Brand";v="99"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"',
}

response = requests.get('https://play.vercel.ai/openai.jpeg', headers=headers)

token_data = loads(b64decode(response.text))
print(token_data)

raw_token = {
    'a': token_data['a'] * .1 * .2,
    't': token_data['t']
}

print(raw_token)

new_token = b64encode(dumps(raw_token, separators=(',', ':')).encode()).decode()
print(new_token)

import requests

headers = {
    'authority': 'play.vercel.ai',
    'accept': '*/*',
    'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
    'content-type': 'application/json',
    'custom-encoding': new_token,
    'origin': 'https://play.vercel.ai',
    'referer': 'https://play.vercel.ai/',
    'sec-ch-ua': '"Chromium";v="112", "Google Chrome";v="112", "Not:A-Brand";v="99"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
}

json_data = {
    'prompt': 'hello\n',
    'model': 'openai:gpt-3.5-turbo',
    'temperature': 0.7,
    'maxTokens': 200,
    'topK': 1,
    'topP': 1,
    'frequencyPenalty': 1,
    'presencePenalty': 1,
    'stopSequences': [],
}

response = requests.post('https://play.vercel.ai/api/generate', headers=headers, json=json_data)
print(response.text)
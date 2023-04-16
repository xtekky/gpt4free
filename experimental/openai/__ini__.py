# experimental, needs chat.openai.com to be loaded with cf_clearance on browser ( can be closed after )

from tls_client import Session
from uuid       import uuid4

from browser_cookie3 import chrome

def session_auth(client):
    headers = {
        'authority': 'chat.openai.com',
        'accept': '*/*',
        'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
        'cache-control': 'no-cache',
        'pragma': 'no-cache',
        'referer': 'https://chat.openai.com/chat',
        'sec-ch-ua': '"Chromium";v="112", "Google Chrome";v="112", "Not:A-Brand";v="99"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
    }

    return client.get('https://chat.openai.com/api/auth/session', headers=headers).json()

client  = Session(client_identifier='chrome110')

for cookie in chrome(domain_name='chat.openai.com'):
    client.cookies[cookie.name] = cookie.value

client.headers = {
    'authority': 'chat.openai.com',
    'accept': 'text/event-stream',
    'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
    'authorization': 'Bearer ' + session_auth(client)['accessToken'],
    'cache-control': 'no-cache',
    'content-type': 'application/json',
    'origin': 'https://chat.openai.com',
    'pragma': 'no-cache',
    'referer': 'https://chat.openai.com/chat',
    'sec-ch-ua': '"Chromium";v="112", "Google Chrome";v="112", "Not:A-Brand";v="99"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
}

response = client.post('https://chat.openai.com/backend-api/conversation', json = {
    'action': 'next',
    'messages': [
        {
            'id': str(uuid4()),
            'author': {
                'role': 'user',
            },
            'content': {
                'content_type': 'text',
                'parts': [
                    'hello world',
                ],
            },
        },
    ],
    'parent_message_id': '9b4682f7-977c-4c8a-b5e6-9713e73dfe01',
    'model': 'text-davinci-002-render-sha',
    'timezone_offset_min': -120,
})

print(response.text)
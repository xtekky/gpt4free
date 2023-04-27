# Import required libraries
from uuid import uuid4

from browser_cookie3 import chrome
from tls_client import Session


class OpenAIChat:
    def __init__(self):
        self.client = Session(client_identifier='chrome110')
        self._load_cookies()
        self._set_headers()

    def _load_cookies(self):
        # Load cookies for the specified domain
        for cookie in chrome(domain_name='chat.openai.com'):
            self.client.cookies[cookie.name] = cookie.value

    def _set_headers(self):
        # Set headers for the client
        self.client.headers = {
            'authority': 'chat.openai.com',
            'accept': 'text/event-stream',
            'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
            'authorization': 'Bearer ' + self.session_auth()['accessToken'],
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

    def session_auth(self):
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

        return self.client.get('https://chat.openai.com/api/auth/session', headers=headers).json()

    def send_message(self, message):
        response = self.client.post('https://chat.openai.com/backend-api/conversation', json={
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
                            message,
                        ],
                    },
                },
            ],
            'parent_message_id': '9b4682f7-977c-4c8a-b5e6-9713e73dfe01',
            'model': 'text-davinci-002-render-sha',
            'timezone_offset_min': -120,
        })

        return response.text


if __name__ == "__main__":
    chat = OpenAIChat()
    response = chat.send_message("hello world")
    print(response)

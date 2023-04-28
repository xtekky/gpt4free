from json import loads
from re import match
from time import time, sleep
from uuid import uuid4

from requests import post
from tls_client import Session

from forefront.mail import Mail
from forefront.typing import ForeFrontResponse


class Account:
    @staticmethod
    def create(proxy=None, logging=False):

        proxies = {
            'http': 'http://' + proxy,
            'https': 'http://' + proxy} if proxy else False

        start = time()

        mail = Mail(proxies)
        mail_token = None
        mail_adress = mail.get_mail()

        # print(mail_adress)

        client = Session(client_identifier='chrome110')
        client.proxies = proxies
        client.headers = {
            "origin": "https://accounts.forefront.ai",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36",
        }

        response = client.post('https://clerk.forefront.ai/v1/client/sign_ups?_clerk_js_version=4.32.6',
                               data={
                                   "email_address": mail_adress
                               }
                               )
        
        try:
            trace_token = response.json()['response']['id']
            if logging: print(trace_token)
        except KeyError:
            return 'Failed to create account!'

        response = client.post(
            f"https://clerk.forefront.ai/v1/client/sign_ups/{trace_token}/prepare_verification?_clerk_js_version=4.32.6",
            data={
                "strategy": "email_code",
            }
            )

        if logging: print(response.text)

        if not 'sign_up_attempt' in response.text:
            return 'Failed to create account!'

        while True:
            sleep(1)
            for _ in mail.fetch_inbox():
                print(mail.get_message_content(_["id"]))
                mail_token = match(r"(\d){5,6}", mail.get_message_content(_["id"])).group(0)

            if mail_token:
                break

        if logging: print(mail_token)

        response = client.post(
            f'https://clerk.forefront.ai/v1/client/sign_ups/{trace_token}/attempt_verification?_clerk_js_version=4.38.4',
            data={
                'code': mail_token,
                'strategy': 'email_code'
            })

        if logging: print(response.json())

        token = response.json()['client']['sessions'][0]['last_active_token']['jwt']

        with open('accounts.txt', 'a') as f:
            f.write(f'{mail_adress}:{token}\n')

        if logging: print(time() - start)

        return token


class StreamingCompletion:
    @staticmethod
    def create(
            token=None,
            chatId=None,
            prompt='',
            actionType='new',
            defaultPersona='607e41fe-95be-497e-8e97-010a59b2e2c0',  # default
            model='gpt-4') -> ForeFrontResponse:

        if not token: raise Exception('Token is required!')
        if not chatId: chatId = str(uuid4())

        headers = {
            'authority': 'chat-server.tenant-forefront-default.knative.chi.coreweave.com',
            'accept': '*/*',
            'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
            'authorization': 'Bearer ' + token,
            'cache-control': 'no-cache',
            'content-type': 'application/json',
            'origin': 'https://chat.forefront.ai',
            'pragma': 'no-cache',
            'referer': 'https://chat.forefront.ai/',
            'sec-ch-ua': '"Chromium";v="112", "Google Chrome";v="112", "Not:A-Brand";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'cross-site',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
        }

        json_data = {
            'text': prompt,
            'action': actionType,
            'parentId': chatId,
            'workspaceId': chatId,
            'messagePersona': defaultPersona,
            'model': model
        }

        for chunk in post('https://chat-server.tenant-forefront-default.knative.chi.coreweave.com/chat',
                          headers=headers, json=json_data, stream=True).iter_lines():

            if b'finish_reason":null' in chunk:
                data = loads(chunk.decode('utf-8').split('data: ')[1])
                token = data['choices'][0]['delta'].get('content')

                if token != None:
                    yield ForeFrontResponse({
                        'id': chatId,
                        'object': 'text_completion',
                        'created': int(time()),
                        'model': model,
                        'choices': [{
                            'text': token,
                            'index': 0,
                            'logprobs': None,
                            'finish_reason': 'stop'
                        }],
                        'usage': {
                            'prompt_tokens': len(prompt),
                            'completion_tokens': len(token),
                            'total_tokens': len(prompt) + len(token)
                        }
                    })

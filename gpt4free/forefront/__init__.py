import os
import pickle
from json import loads
from re import findall
from time import time, sleep
from typing import Generator, Optional
from uuid import uuid4

from fake_useragent import UserAgent
from pymailtm import MailTm, Message
from requests import post
from tls_client import Session

from .typing import ForeFrontResponse


class Account:
    COOKIES_FILE_NAME = 'cookies.pickle'

    @staticmethod
    def login(proxy: Optional[str] = None, logging: bool = False) -> str:
        if not os.path.isfile(Account.COOKIES_FILE_NAME):
            return Account.create(proxy, logging)

        with open(Account.COOKIES_FILE_NAME, 'rb') as f:
            cookies = pickle.load(f)
        proxies = {'http': 'http://' + proxy, 'https': 'http://' + proxy} if proxy else False

        client = Session(client_identifier='chrome110')
        client.proxies = proxies
        client.cookies.update(cookies)

        if Account.is_cookie_enabled(client):
            response = client.get('https://clerk.forefront.ai/v1/client?_clerk_js_version=4.38.4')
            return response.json()['response']['sessions'][0]['last_active_token']['jwt']
        else:
            return Account.create(proxy, logging)

    @staticmethod
    def create(proxy: Optional[str] = None, logging: bool = False, save_cookies: bool = False) -> str:
        proxies = {'http': 'http://' + proxy, 'https': 'http://' + proxy} if proxy else False

        start = time()

        mail_client = MailTm().get_account()
        mail_address = mail_client.address

        client = Session(client_identifier='chrome110')
        client.proxies = proxies
        client.headers = {
            'origin': 'https://accounts.forefront.ai',
            'user-agent': UserAgent().random,
        }

        response = client.post(
            'https://clerk.forefront.ai/v1/client/sign_ups?_clerk_js_version=4.38.4',
            data={'email_address': mail_address},
        )

        try:
            trace_token = response.json()['response']['id']
            if logging:
                print(trace_token)
        except KeyError:
            return 'Failed to create account!'

        response = client.post(
            f'https://clerk.forefront.ai/v1/client/sign_ups/{trace_token}/prepare_verification?_clerk_js_version=4.38.4',
            data={'strategy': 'email_link', 'redirect_url': 'https://accounts.forefront.ai/sign-up/verify'},
        )

        if logging:
            print(response.text)

        if 'sign_up_attempt' not in response.text:
            return 'Failed to create account!'

        while True:
            sleep(1)
            new_message: Message = mail_client.wait_for_message()
            if logging:
                print(new_message.data['id'])

            verification_url = findall(r'https:\/\/clerk\.forefront\.ai\/v1\/verify\?token=\w.+', new_message.text)[0]

            if verification_url:
                break

        if logging:
            print(verification_url)

        response = client.get(verification_url)

        response = client.get('https://clerk.forefront.ai/v1/client?_clerk_js_version=4.38.4')

        token = response.json()['response']['sessions'][0]['last_active_token']['jwt']

        if save_cookies:
            with open(Account.COOKIES_FILE_NAME, 'wb') as f:
                pickle.dump(client.cookies, f)

        with open('accounts.txt', 'a') as f:
            f.write(f'{mail_address}:{token}\n')

        if logging:
            print(time() - start)

        return token

    @staticmethod
    def is_cookie_enabled(client: Session) -> bool:
        response = client.get('https://chat.forefront.ai/')
        return 'window.startClerk' in response.text


class StreamingCompletion:
    @staticmethod
    def create(
        token=None,
        chat_id=None,
        prompt='',
        action_type='new',
        default_persona='607e41fe-95be-497e-8e97-010a59b2e2c0',  # default
        model='gpt-4',
        proxy=None,
    ) -> Generator[ForeFrontResponse, None, None]:
        if not token:
            raise Exception('Token is required!')
        if not chat_id:
            chat_id = str(uuid4())

        proxies = {'http': 'http://' + proxy, 'https': 'http://' + proxy} if proxy else None

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
            'user-agent': UserAgent().random,
        }

        json_data = {
            'text': prompt,
            'action': action_type,
            'parentId': chat_id,
            'workspaceId': chat_id,
            'messagePersona': default_persona,
            'model': model,
        }

        for chunk in post(
            'https://chat-server.tenant-forefront-default.knative.chi.coreweave.com/chat',
            headers=headers,
            proxies=proxies,
            json=json_data,
            stream=True,
        ).iter_lines():
            if b'finish_reason":null' in chunk:
                data = loads(chunk.decode('utf-8').split('data: ')[1])
                token = data['choices'][0]['delta'].get('content')

                if token is not None:
                    yield ForeFrontResponse(
                        **{
                            'id': chat_id,
                            'object': 'text_completion',
                            'created': int(time()),
                            'text': token,
                            'model': model,
                            'choices': [{'text': token, 'index': 0, 'logprobs': None, 'finish_reason': 'stop'}],
                            'usage': {
                                'prompt_tokens': len(prompt),
                                'completion_tokens': len(token),
                                'total_tokens': len(prompt) + len(token),
                            },
                        }
                    )


class Completion:
    @staticmethod
    def create(
        token=None,
        chat_id=None,
        prompt='',
        action_type='new',
        default_persona='607e41fe-95be-497e-8e97-010a59b2e2c0',  # default
        model='gpt-4',
        proxy=None,
    ) -> ForeFrontResponse:
        text = ''
        final_response = None
        for response in StreamingCompletion.create(
            token=token,
            chat_id=chat_id,
            prompt=prompt,
            action_type=action_type,
            default_persona=default_persona,
            model=model,
            proxy=proxy,
        ):
            if response:
                final_response = response
                text += response.text

        if final_response:
            final_response.text = text
        else:
            raise Exception('Unable to get the response, Please try again')

        return final_response

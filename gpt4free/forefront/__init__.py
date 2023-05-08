import hashlib
import os
from base64 import b64encode
from json import loads
from re import findall
from time import time, sleep
from typing import Generator, Optional
from uuid import uuid4

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from fake_useragent import UserAgent
from mailgw_temporary_email import Email
from requests import post
from tls_client import Session

from .typing import ForeFrontResponse, AccountData


class Account:
    COOKIES_FILE_NAME = 'cookie.txt'

    @staticmethod
    def create(proxy: Optional[str] = None, logging: bool = False, use_cookie: bool = False) -> AccountData:
        start = time()

        http_client = Account.__create_http_client(proxy=proxy)

        if use_cookie:
            is_cookie_loaded = Account.__load_cookie(http_client)
            if not is_cookie_loaded:
                Account.__create_forefront_account(http_client, logging)
            Account.__save_cookie(http_client)
        else:
            Account.__create_forefront_account(http_client, logging)

        account_data = Account.__get_account_data(http_client)

        if logging:
            print(time() - start)

        return account_data

    @staticmethod
    def __is_cookie_enabled(http_client: Session) -> bool:
        response = http_client.get('https://clerk.forefront.ai/v1/client?_clerk_js_version=4.38.4')
        response = response.json()
        return response['response'] is not None

    @staticmethod
    def __create_http_client(proxy: Optional[str] = None) -> Session:
        proxies = {'http': 'http://' + proxy, 'https': 'http://' + proxy} if proxy else False

        http_client = Session(client_identifier='chrome110')
        http_client.proxies = proxies
        http_client.headers = {
            'origin': 'https://accounts.forefront.ai',
            'user-agent': UserAgent().random,
        }

        return http_client

    @staticmethod
    def __create_forefront_account(http_client: Session, logging: bool):
        email = Email()
        email.register()
        mail_address = email.address

        trace_token = Account.__signup_forefront_account(http_client, mail_address)
        if logging:
            print(trace_token)

        Account.__verify_forefront_account(http_client, email, trace_token, logging)

    @staticmethod
    def __signup_forefront_account(http_client: Session, mail_address: str) -> str:
        response = http_client.post(
            'https://clerk.forefront.ai/v1/client/sign_ups?_clerk_js_version=4.38.4',
            data={'email_address': mail_address},
        )
        try:
            return response.json()['response']['id']
        except KeyError:
            raise RuntimeError('Failed to create account!')

    @staticmethod
    def __verify_forefront_account(http_client: Session, email: Email, trace_token: str, logging: bool):
        response = http_client.post(
            f'https://clerk.forefront.ai/v1/client/sign_ups/{trace_token}/prepare_verification?_clerk_js_version=4.38.4',
            data={'strategy': 'email_link', 'redirect_url': 'https://accounts.forefront.ai/sign-up/verify'},
        )
        if logging:
            print(response.text)

        if 'sign_up_attempt' not in response.text:
            raise RuntimeError('Failed to create account!')

        while True:
            sleep(5)
            message_id = email.message_list()[0]['id']
            message = email.message(message_id)
            verification_url = findall(r'https:\/\/clerk\.forefront\.ai\/v1\/verify\?token=\w.+', message["text"])[0]
            if verification_url:
                break

        if verification_url is None or not verification_url:
            raise RuntimeError('Error while obtaining verification URL!')

        if logging:
            print(verification_url)

        http_client.get(verification_url)

    @staticmethod
    def __get_account_data(http_client: Session) -> AccountData:
        response = http_client.get('https://clerk.forefront.ai/v1/client?_clerk_js_version=4.38.4').json()
        session_data = response['response']['sessions'][0]

        return AccountData(
            user_id=session_data['user']['id'],
            session_id=session_data['id'],
            token=session_data['last_active_token']['jwt'],
        )

    @staticmethod
    def __save_cookie(http_client: Session):
        with open(Account.COOKIES_FILE_NAME, 'w') as f:
            f.write(http_client.cookies.get('__client'))

    @staticmethod
    def __load_cookie(http_client: Session) -> bool:
        if not os.path.isfile(Account.COOKIES_FILE_NAME):
            return False

        with open(Account.COOKIES_FILE_NAME, 'r') as f:
            http_client.cookies.set('__client', f.read())

        return Account.__is_cookie_enabled(http_client)


class StreamingCompletion:
    @staticmethod
    def create(
        prompt: str,
        account_data: AccountData,
        chat_id=None,
        action_type='new',
        default_persona='607e41fe-95be-497e-8e97-010a59b2e2c0',  # default
        model='gpt-4',
        proxy=None
    ) -> Generator[ForeFrontResponse, None, None]:
        token = account_data.token
        if not chat_id:
            chat_id = str(uuid4())

        proxies = {'http': 'http://' + proxy, 'https': 'http://' + proxy} if proxy else None
        base64_data = b64encode((account_data.user_id + default_persona + chat_id).encode()).decode()
        encrypted_signature = StreamingCompletion.__encrypt(base64_data, account_data.session_id)

        headers = {
            'authority': 'chat-server.tenant-forefront-default.knative.chi.coreweave.com',
            'accept': '*/*',
            'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
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
            'authorization': f"Bearer {token}",
            'X-Signature': encrypted_signature,
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
            'https://streaming.tenant-forefront-default.knative.chi.coreweave.com/chat',
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

    @staticmethod
    def __encrypt(data: str, key: str) -> str:
        hash_key = hashlib.sha256(key.encode()).digest()
        iv = get_random_bytes(16)
        cipher = AES.new(hash_key, AES.MODE_CBC, iv)
        encrypted_data = cipher.encrypt(StreamingCompletion.__pad_data(data.encode()))
        return iv.hex() + encrypted_data.hex()

    @staticmethod
    def __pad_data(data: bytes) -> bytes:
        block_size = AES.block_size
        padding_size = block_size - len(data) % block_size
        padding = bytes([padding_size] * padding_size)
        return data + padding


class Completion:
    @staticmethod
    def create(
        prompt: str,
        account_data: AccountData,
        chat_id=None,
        action_type='new',
        default_persona='607e41fe-95be-497e-8e97-010a59b2e2c0',  # default
        model='gpt-4',
        proxy=None
    ) -> ForeFrontResponse:
        text = ''
        final_response = None
        for response in StreamingCompletion.create(
            account_data=account_data,
            chat_id=chat_id,
            prompt=prompt,
            action_type=action_type,
            default_persona=default_persona,
            model=model,
            proxy=proxy
        ):
            if response:
                final_response = response
                text += response.text

        if final_response:
            final_response.text = text
        else:
            raise RuntimeError('Unable to get the response, Please try again')

        return final_response

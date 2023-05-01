import re
import time
from typing import Generator, Optional
from uuid import uuid4

import fake_useragent
import pymailtm
import requests

from .typing import ForeFrontResponse


def speed_logging(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        print(time() - start)
        return res
    return wrapper


class Account:
    @speed_logging
    @staticmethod
    def create_forefront_account(proxy: Optional[str] = None) -> Optional[str]:
        """Create a ForeFront account.

        Args:
            proxy: The proxy to use for the request.

        Returns:
            The ForeFront token if successful, else None.
        """
        proxies = {'http': f'http://{proxy}', 'https': f'http://{proxy}'} if proxy else None

        mail_client = pymailtm.MailTm().get_account()
        mail_address = mail_client.address

        session = requests.Session()
        session.proxies = proxies
        session.headers = {
            'origin': 'https://accounts.forefront.ai',
            'user-agent': fake_useragent.UserAgent().random,
        }

        response = session.post(
            'https://clerk.forefront.ai/v1/client/sign_ups?_clerk_js_version=4.38.4',
            data={'email_address': mail_address},
        )

        try:
            trace_token = response.json()['response']['id']
        except KeyError:
            return None

        response = session.post(
            f'https://clerk.forefront.ai/v1/client/sign_ups/{trace_token}/prepare_verification?_clerk_js_version=4.38.4',
            data={
                'strategy': 'email_link',
                'redirect_url': 'https://accounts.forefront.ai/sign-up/verify'
            },
        )

        if 'sign_up_attempt' not in response.text:
            return None

        while True:
            time.sleep(1)
            new_message = mail_client.wait_for_message()

            verification_url = re.findall(r'https:\/\/clerk\.forefront\.ai\/v1\/verify\?token=\w.+', new_message.text)
            if verification_url:
                break

        response = session.get(verification_url[0])

        response = session.get('https://clerk.forefront.ai/v1/client?_clerk_js_version=4.38.4')

        token = response.json()['response']['sessions'][0]['last_active_token']['jwt']

        with open('accounts.txt', 'a') as f:
            f.write(f'{mail_address}:{token}\n')

        return token


class StreamingCompletion:
    @staticmethod
    def create(
        token=None,
        chat_id=None,
        prompt='',
        action_type='new',
        default_persona='607e41fe-95be-497e-8e97-010a59b2e2c0',  # default
        model='gpt-4',
        proxy=None
    ) -> Generator[ForeFrontResponse, None, None]:
        if not token:
            raise Exception('Token is required!')
        if not chat_id:
            chat_id = str(uuid4())

        proxies = { 'http': f'http://{proxy}', 'https': f'http://{proxy}' } if proxy else None

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
        proxy=None
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
            proxy=proxy
        ):
            if response:
                final_response = response
                text += response.text

        if final_response:
            final_response.text = text
        else:
            raise Exception('Unable to get the response, Please try again')

        return final_response




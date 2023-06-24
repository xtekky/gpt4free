from hashlib import md5
from json import dumps
from re import findall
from typing import Optional

from tls_client import Session as TLS
from twocaptcha import TwoCaptcha

from gpt4free.quora import extract_formkey
from gpt4free.quora.mail import Emailnator

solver = TwoCaptcha('')


class Account:
    @staticmethod
    def create(proxy: Optional[str] = None, logging: bool = False, enable_bot_creation: bool = False):
        client = TLS(client_identifier='chrome110')
        client.proxies = {'http': f'http://{proxy}', 'https': f'http://{proxy}'} if proxy else None

        mail_client = Emailnator()
        mail_address = mail_client.get_mail()

        if logging:
            print('email', mail_address)

        client.headers = {
            'authority': 'poe.com',
            'accept': '*/*',
            'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
            'content-type': 'application/json',
            'origin': 'https://poe.com',
            'poe-formkey': 'null',
            'poe-tag-id': 'null',
            'poe-tchannel': 'null',
            'referer': 'https://poe.com/login',
            'sec-ch-ua': '"Chromium";v="112", "Google Chrome";v="112", "Not:A-Brand";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
        }

        client.headers["poe-formkey"] = extract_formkey(client.get('https://poe.com/login').text)
        client.headers["poe-tchannel"] = client.get('https://poe.com/api/settings').json()['tchannelData']['channel']

        # token = reCaptchaV3('https://www.recaptcha.net/recaptcha/enterprise/anchor?ar=1&k=6LflhEElAAAAAI_ewVwRWI9hsyV4mbZnYAslSvlG&co=aHR0cHM6Ly9wb2UuY29tOjQ0Mw..&hl=en&v=4PnKmGB9wRHh1i04o7YUICeI&size=invisible&cb=bi6ivxoskyal')
        token = solver.recaptcha(
            sitekey='6LflhEElAAAAAI_ewVwRWI9hsyV4mbZnYAslSvlG',
            url='https://poe.com/login?redirect_url=%2F',
            version='v3',
            enterprise=1,
            invisible=1,
            action='login',
        )['code']

        payload = dumps(
            separators=(',', ':'),
            obj={
                'queryName': 'MainSignupLoginSection_sendVerificationCodeMutation_Mutation',
                'variables': {'emailAddress': mail_address, 'phoneNumber': None, 'recaptchaToken': token},
                'query': 'mutation MainSignupLoginSection_sendVerificationCodeMutation_Mutation(\n  $emailAddress: String\n  $phoneNumber: String\n  $recaptchaToken: String\n) {\n  sendVerificationCode(verificationReason: login, emailAddress: $emailAddress, phoneNumber: $phoneNumber, recaptchaToken: $recaptchaToken) {\n    status\n    errorMessage\n  }\n}\n',
            },
        )

        base_string = payload + client.headers["poe-formkey"] + 'WpuLMiXEKKE98j56k'
        client.headers["poe-tag-id"] = md5(base_string.encode()).hexdigest()

        print(dumps(client.headers, indent=4))

        response = client.post('https://poe.com/api/gql_POST', data=payload)

        if 'automated_request_detected' in response.text:
            print('please try using a proxy / wait for fix')

        if 'Bad Request' in response.text:
            if logging:
                print('bad request, retrying...', response.json())
            quit()

        if logging:
            print('send_code', response.json())

        mail_content = mail_client.get_message()
        mail_token = findall(r';">(\d{6,7})</div>', mail_content)[0]

        if logging:
            print('code', mail_token)

        payload = dumps(
            separators=(',', ':'),
            obj={
                "queryName": "SignupOrLoginWithCodeSection_signupWithVerificationCodeMutation_Mutation",
                "variables": {"verificationCode": str(mail_token), "emailAddress": mail_address, "phoneNumber": None},
                "query": "mutation SignupOrLoginWithCodeSection_signupWithVerificationCodeMutation_Mutation(\n  $verificationCode: String!\n  $emailAddress: String\n  $phoneNumber: String\n) {\n  signupWithVerificationCode(verificationCode: $verificationCode, emailAddress: $emailAddress, phoneNumber: $phoneNumber) {\n    status\n    errorMessage\n  }\n}\n",
            },
        )

        base_string = payload + client.headers["poe-formkey"] + 'WpuLMiXEKKE98j56k'
        client.headers["poe-tag-id"] = md5(base_string.encode()).hexdigest()

        response = client.post('https://poe.com/api/gql_POST', data=payload)
        if logging:
            print('verify_code', response.json())


Account.create(proxy='', logging=True)

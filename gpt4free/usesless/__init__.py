import time
import re
import json
import requests
import fake_useragent
import names

from mailgw_temporary_email import Email
from password_generator import PasswordGenerator


class Account:
    @staticmethod
    def create(logging: bool = False):
        mail_client = Email()
        mail_client.register()
        mail_address = mail_client.address

        pwo = PasswordGenerator()
        pwo.minlen = 8
        password = pwo.generate()

        session = requests.Session()

        if logging:
            print(f"email: {mail_address}")

        register_url = "https://ai.usesless.com/api/cms/auth/local/register"
        register_json = {
            "username": names.get_first_name(),
            "password": password,
            "email": mail_address,
        }
        headers = {
            "authority": "ai.usesless.com",
            "accept": "application/json, text/plain, */*",
            "accept-language": "en-US,en;q=0.5",
            "cache-control": "no-cache",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": fake_useragent.UserAgent().random,
        }
        register = session.post(register_url, json=register_json, headers=headers)
        if logging:
            if register.status_code == 200:
                print("register success")
            else:
                print("there's a problem with account creation, try again")

        if register.status_code != 200:
            quit()

        while True:
            time.sleep(5)
            message_id = mail_client.message_list()[0]["id"]
            message = mail_client.message(message_id)
            verification_url = re.findall(
                r"http:\/\/ai\.usesless\.com\/api\/cms\/auth\/email-confirmation\?confirmation=\w.+\w\w",
                message["text"],
            )[0]
            if verification_url:
                break

        session.get(verification_url)
        login_json = {"identifier": mail_address, "password": password}
        login_request = session.post(
            url="https://ai.usesless.com/api/cms/auth/local", json=login_json
        )
        token = login_request.json()["jwt"]
        if logging:
            print(f"token: {token}")

        with open("accounts.txt", "w") as f:
            f.write(f"{mail_address}\n")
            f.write(f"{token}")

        return token


class Completion:
    @staticmethod
    def create(
        token: str,
        systemMessage: str = "You are a helpful assistant",
        prompt: str = "",
        parentMessageId: str = "",
        presence_penalty: float = 1,
        temperature: float = 1,
        model: str = "gpt-3.5-turbo",
    ):
        headers = {
            "authority": "ai.usesless.com",
            "accept": "application/json, text/plain, */*",
            "accept-language": "en-US,en;q=0.5",
            "cache-control": "no-cache",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": fake_useragent.UserAgent().random,
            "Authorization": f"Bearer {token}",
        }

        json_data = {
            "openaiKey": "",
            "prompt": prompt,
            "options": {
                "parentMessageId": parentMessageId,
                "systemMessage": systemMessage,
                "completionParams": {
                    "presence_penalty": presence_penalty,
                    "temperature": temperature,
                    "model": model,
                },
            },
        }

        url = "https://ai.usesless.com/api/chat-process"
        request = requests.post(url, headers=headers, json=json_data)
        request.encoding = request.apparent_encoding
        content = request.content

        response = Completion.__response_to_json(content)
        return response

    @classmethod
    def __response_to_json(cls, text) -> str:
        text = str(text.decode("utf-8"))

        split_text = text.rsplit("\n", 1)[1]
        to_json = json.loads(split_text)
        return to_json

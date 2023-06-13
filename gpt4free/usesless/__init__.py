import string
import time
import re
import json
import requests
import fake_useragent
import random
from password_generator import PasswordGenerator

from .utils import create_email, check_email


class Account:
    @staticmethod
    def create(logging: bool = False):
        is_custom_domain = input(
            "Do you want to use your custom domain name for temporary email? [Y/n]: "
        ).upper()

        if is_custom_domain == "Y":
            mail_address = create_email(custom_domain=True, logging=logging)
        elif is_custom_domain == "N":
            mail_address = create_email(custom_domain=False, logging=logging)
        else:
            print("Please, enter either Y or N")
            return

        name = string.ascii_lowercase + string.digits
        username = "".join(random.choice(name) for i in range(20))

        pwo = PasswordGenerator()
        pwo.minlen = 8
        password = pwo.generate()

        session = requests.Session()

        register_url = "https://ai.usesless.com/api/cms/auth/local/register"
        register_json = {
            "username": username,
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
                print("Registered successfully")
            else:
                print(register.status_code)
                print(register.json())
                print("There was a problem with account registration, try again")

        if register.status_code != 200:
            quit()

        while True:
            time.sleep(5)
            messages = check_email(mail=mail_address, logging=logging)

            # Check if method `message_list()` didn't return None or empty list.
            if not messages or len(messages) == 0:
                # If it returned None or empty list sleep for 5 seconds to wait for new message.
                continue

            message_text = messages[0]["content"]
            verification_url = re.findall(
                r"http:\/\/ai\.usesless\.com\/api\/cms\/auth\/email-confirmation\?confirmation=\w.+\w\w",
                message_text,
            )[0]
            if verification_url:
                break

        session.get(verification_url)
        login_json = {"identifier": mail_address, "password": password}
        login_request = session.post(
            url="https://ai.usesless.com/api/cms/auth/local", json=login_json
        )

        token = login_request.json()["jwt"]
        if logging and token:
            print(f"Token: {token}")

        with open("account.json", "w") as file:
            json.dump({"email": mail_address, "token": token}, file)
            if logging:
                print(
                    "\nNew account credentials has been successfully saved in 'account.json' file"
                )

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

        split_text = text.rsplit("\n", 1)
        if len(split_text) > 1:
            to_json = json.loads(split_text[1])
            return to_json
        else:
            return None


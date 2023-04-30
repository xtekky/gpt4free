import json
import re
import time

from fake_useragent import UserAgent
from requests import Session


class Emailnator:
    def __init__(self) -> None:
        self.session = requests.Session()
        self._initialize_session()

        self.email_address = None

    def _initialize_session(self):
        self.session.get("https://www.emailnator.com/", timeout=6)
        self.session.headers = {
            "authority": "www.emailnator.com",
            "origin": "https://www.emailnator.com",
            "referer": "https://www.emailnator.com/",
            "user-agent": UserAgent().random,
            "x-xsrf-token": self.session.cookies.get("XSRF-TOKEN")[:-3] + "=",
        }

    def generate_email_address(self):
        response = self.session.post(
            "https://www.emailnator.com/generate-email",
            json={"email": ["domain", "plusGmail", "dotGmail"]},
        )
        self.email_address = json.loads(response.text)["email"][0]
        return self.email_address

    def _wait_for_message(self):
        print("Waiting for message...")

        while True:
            time.sleep(2)
            mail_token_response = self.session.post(
                "https://www.emailnator.com/message-list",
                json={"email": self.email_address},
            )

            mail_token = json.loads(mail_token_response.text)["messageData"]

            if len(mail_token) == 2:
                print("Message received!")
                print(mail_token[1]["messageID"])
                break

        return mail_token[1]["messageID"]

    def get_message_content(self):
        message_id = self._wait_for_message()

        mail_context_response = self.session.post(
            "https://www.emailnator.com/message-list",
            json={"email": self.email_address, "messageID": message_id},
        )

        return mail_context_response.text

    def get_verification_code(self):
        message_content = self.get_message_content()
        verification_code = re.findall(
            r';">(\d{6,7})</div>', message_content)[0]
        print(f"Verification code: {verification_code}")
        return verification_code

    def clear_inbox(self):
        print("Clearing inbox...")
        self.session.post(
            "https://www.emailnator.com/delete-all",
            json={"email": self.email_address},
        )
        print("Inbox cleared!")

    def __del__(self):
        if self.email_address:
            self.clear_inbox()

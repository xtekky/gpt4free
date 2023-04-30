import json
from requests import Session
from string import ascii_letters
from random import choices


class Mail:
    BASE_URL = "https://api.mail.tm"

    def __init__(self, proxies: dict = None) -> None:
        self.client = self._create_session(proxies)

    @staticmethod
    def _create_session(proxies: dict = None) -> Session:
        session = Session()
        session.proxies = proxies
        session.headers = {
            "host": "api.mail.tm",
            "connection": "keep-alive",
            "sec-ch-ua": "\"Google Chrome\";v=\"111\", \"Not(A:Brand\";v=\"8\", \"Chromium\";v=\"111\"",
            "accept": "application/json, text/plain, */*",
            "content-type": "application/json",
            "sec-ch-ua-mobile": "?0",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36",
            "sec-ch-ua-platform": "\"macOS\"",
            "origin": "https://mail.tm",
            "sec-fetch-site": "same-site",
            "sec-fetch-mode": "cors",
            "sec-fetch-dest": "empty",
            "referer": "https://mail.tm/",
            "accept-encoding": "gzip, deflate, br",
            "accept-language": "en-GB,en-US;q=0.9,en;q=0.8"
        }
        return session

    @staticmethod
    def _generate_token(length: int = 14) -> str:
        return ''.join(choices(ascii_letters, k=length)).lower()

    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        response = self.client.request(
            method, f"{self.BASE_URL}/{endpoint}", **kwargs)
        response.raise_for_status()
        return response.json()

    def get_mail(self) -> str:
        token = self._generate_token()
        init = self._request("post", "accounts", json={
            "address": f"{token}@bugfoo.com",
            "password": token
        })

        resp = self._request("post", "token", json={
            **init,
            "password": token
        })

        self.client.headers['authorization'] = f"Bearer {resp['token']}"
        return f"{token}@bugfoo.com"

    def fetch_inbox(self):
        return self._request("get", "messages")["hydra:member"]

    def get_message(self, message_id: str):
        return self._request("get", f"messages/{message_id}")

    def get_message_content(self, message_id: str):
        return self.get_message(message_id)["text"]

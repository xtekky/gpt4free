from random import choices
from string import ascii_letters

from requests import Session


class Mail:
    def __init__(self, proxies: dict = None) -> None:
        self.client = Session()
        self.client.proxies = proxies
        self.client.headers = {
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

    def get_mail(self) -> str:
        token = ''.join(choices(ascii_letters, k=14)).lower()
        init = self.client.post("https://api.mail.tm/accounts", json={
            "address": f"{token}@bugfoo.com",
            "password": token
        })

        if init.status_code == 201:
            resp = self.client.post("https://api.mail.tm/token", json={
                **init.json(),
                "password": token
            })

            self.client.headers['authorization'] = 'Bearer ' + resp.json()['token']

            return f"{token}@bugfoo.com"

        else:
            raise Exception("Failed to create email")

    def fetch_inbox(self):
        return self.client.get(f"https://api.mail.tm/messages").json()["hydra:member"]

    def get_message(self, message_id: str):
        return self.client.get(f"https://api.mail.tm/messages/{message_id}").json()

    def get_message_content(self, message_id: str):
        return self.get_message(message_id)["text"]

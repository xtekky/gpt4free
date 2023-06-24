from json import loads
from re import findall
from time import sleep

from requests import Session


class Mail:
    def __init__(self) -> None:
        self.client = Session()
        self.client.post("https://etempmail.com/")
        self.cookies = {'acceptcookie': 'true'}
        self.cookies["ci_session"] = self.client.cookies.get_dict()["ci_session"]
        self.email = None

    def get_mail(self):
        respone = self.client.post("https://etempmail.com/getEmailAddress")
        # cookies
        self.cookies["lisansimo"] = eval(respone.text)["recover_key"]
        self.email = eval(respone.text)["address"]
        return self.email

    def get_message(self):
        print("Waiting for message...")
        while True:
            sleep(5)
            respone = self.client.post("https://etempmail.com/getInbox")
            mail_token = loads(respone.text)
            print(self.client.cookies.get_dict())
            if len(mail_token) == 1:
                break

        params = {
            'id': '1',
        }
        self.mail_context = self.client.post("https://etempmail.com/getInbox", params=params)
        self.mail_context = eval(self.mail_context.text)[0]["body"]
        return self.mail_context

    # ,cookies=self.cookies
    def get_verification_code(self):
        message = self.mail_context
        code = findall(r';">(\d{6,7})</div>', message)[0]
        print(f"Verification code: {code}")
        return code

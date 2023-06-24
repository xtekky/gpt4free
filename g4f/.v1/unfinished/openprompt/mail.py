import email

import requests


class MailClient:

    def __init__(self):
        self.username = None
        self.token = None
        self.raw = None
        self.mailids = None
        self.mails = None
        self.mail = None

    def create(self, force=False):
        headers = {
            'accept': 'application/json',
        }

        if self.username:
            pass
        else:
            self.response = requests.put(
                'https://www.developermail.com/api/v1/mailbox', headers=headers)
            self.response = self.response.json()
            self.username = self.response['result']['name']
            self.token = self.response['result']['token']

        return {'username': self.username, 'token': self.token}

    def destroy(self):
        headers = {
            'accept': 'application/json',
            'X-MailboxToken': self.token,
        }
        self.response = requests.delete(
            f'https://www.developermail.com/api/v1/mailbox/{self.username}', headers=headers)
        self.response = self.response.json()
        self.username = None
        self.token = None
        return self.response

    def newtoken(self):
        headers = {
            'accept': 'application/json',
            'X-MailboxToken': self.token,
        }
        self.response = requests.put(
            f'https://www.developermail.com/api/v1/mailbox/{self.username}/token', headers=headers)
        self.response = self.response.json()
        self.token = self.response['result']['token']
        return {'username': self.username, 'token': self.token}

    def getmailids(self):
        headers = {
            'accept': 'application/json',
            'X-MailboxToken': self.token,
        }

        self.response = requests.get(
            f'https://www.developermail.com/api/v1/mailbox/{self.username}', headers=headers)
        self.response = self.response.json()
        self.mailids = self.response['result']
        return self.mailids

    def getmails(self, mailids: list = None):
        headers = {
            'accept': 'application/json',
            'X-MailboxToken': self.token,
            'Content-Type': 'application/json',
        }

        if mailids is None:
            mailids = self.mailids

        data = str(mailids)

        self.response = requests.post(
            f'https://www.developermail.com/api/v1/mailbox/{self.username}/messages', headers=headers, data=data)
        self.response = self.response.json()
        self.mails = self.response['result']
        return self.mails

    def getmail(self, mailid: str, raw=False):
        headers = {
            'accept': 'application/json',
            'X-MailboxToken': self.token,
        }
        self.response = requests.get(
            f'https://www.developermail.com/api/v1/mailbox/{self.username}/messages/{mailid}', headers=headers)
        self.response = self.response.json()
        self.mail = self.response['result']
        if raw is False:
            self.mail = email.message_from_string(self.mail)
        return self.mail

    def delmail(self, mailid: str):
        headers = {
            'accept': 'application/json',
            'X-MailboxToken': self.token,
        }
        self.response = requests.delete(
            f'https://www.developermail.com/api/v1/mailbox/{self.username}/messages/{mailid}', headers=headers)
        self.response = self.response.json()
        return self.response


client = MailClient()
client.newtoken()
print(client.getmails())

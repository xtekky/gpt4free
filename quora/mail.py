from requests import Session
from string import ascii_letters
from random import choices

class Mail:
    def __init__(self, proxies: dict = None) -> None:
        self.client = Session()
        self.client.proxies = None #proxies
        self.client.headers = {
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        self.domain = "guerrillamail.com"
        
        self.sid_token = self.client.get("https://api.guerrillamail.com/ajax.php?f=get_email_address").json()['sid_token']

    def get_mail(self) -> str:
        token = ''.join(choices(ascii_letters, k=10)).lower()

        email_id = f"{token}@{self.domain}"
        self.client.get(f"https://api.guerrillamail.com/ajax.php?f=set_email_user&email_user={token}&sid_token={self.sid_token}")
       
        return email_id

    def fetch_inbox(self):
        return self.client.get(f"https://api.guerrillamail.com/ajax.php?f=get_emails&sid_token={self.sid_token}").json()

    def get_message(self, message_id: str):
        return self.client.get(f"https://api.guerrillamail.com/ajax.php?f=fetch_email&email_id={message_id}&sid_token={self.sid_token}").json()

    def get_message_content(self, message_id: str):
        return self.get_message(message_id)["mail_body"]

if __name__ == "__main__":
    client = Mail()
    client.get_mail()

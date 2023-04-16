import html
import json
from   tls_client import Session

class Mail:
    def __init__(self, proxies: str = None, timeout: int = 15, bearer_token: str or None = None) -> None:
        self.session  = Session(client_identifier='chrome110')
        self.base_url = 'https://web2.temp-mail.org'
        self.proxies  = proxies
        self.timeout  = timeout
        
        self.session.headers['authorization'] = f'Bearer {bearer_token}' if bearer_token else None

    def get_mail(self) -> str:
        status: html = self.session.get(self.base_url).status_code
        
        try:
            if status == 200:
                data = self.session.post(f'{self.base_url}/mailbox').json()

                self.session.headers['authorization'] = f'Bearer {data["token"]}'
                return data["token"], data["mailbox"]
        
        except Exception as e:
            print(e)
            return f'Email creation error. {e} | use proxies', False
    
    def fetch_inbox(self) -> json:
        return self.session.get(f'{self.base_url}/messages').json()
    
    def get_message_content(self, message_id: str):
        return self.session.get(f'{self.base_url}/messages/{message_id}').json()["bodyHtml"]

# if __name__ == '__main__':

#     email_client = TempMail()
#     token, email = email_client.get_mail()
#     print(email)
#     print(token)
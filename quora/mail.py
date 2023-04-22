from requests import Session
from time     import sleep
from re       import search, findall
from json     import loads

class Emailnator:
    def __init__(self) -> None:
        self.client = Session()
        self.client.get('https://www.emailnator.com/', timeout=6)
        self.cookies = self.client.cookies.get_dict()

        self.client.headers = {
            'authority'      : 'www.emailnator.com',
            'origin'         : 'https://www.emailnator.com',
            'referer'        : 'https://www.emailnator.com/',
            'user-agent'     : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.0.0 Safari/537.36 Edg/101.0.1722.39',
            'x-xsrf-token'   : self.client.cookies.get("XSRF-TOKEN")[:-3]+"=",
        }
        
        self.email = None
        
    def get_mail(self):
        response = self.client.post('https://www.emailnator.com/generate-email',json = {
            'email': [
                'domain',
                'plusGmail',
                'dotGmail',
            ]
        })
        
        self.email = loads(response.text)["email"][0]
        return self.email
    
    def get_message(self):
        print("waiting for code...")
        
        while True:
            sleep(2)
            mail_token = self.client.post('https://www.emailnator.com/message-list', 
                json   = {'email': self.email})
            
            mail_token = loads(mail_token.text)["messageData"]
            
            if len(mail_token) == 2:
                print(mail_token[1]["messageID"])
                break
        
        mail_context =  self.client.post('https://www.emailnator.com/message-list', json = {
            'email'    : self.email,
            'messageID': mail_token[1]["messageID"],
        })
        
        return mail_context.text

# mail_client  = Emailnator()
# mail_adress  = mail_client.get_mail()

# print(mail_adress)

# mail_content = mail_client.get_message()

# print(mail_content)

# code = findall(r';">(\d{6,7})</div>', mail_content)[0]
# print(code)


from quora.api    import Client as PoeClient
from quora.mail   import Mail
from requests     import Session
from re           import search, findall
from json         import loads
from time         import sleep, time
from pathlib      import Path
from random       import choice
from urllib       import parse
 
class PoeResponse:
    
    class Completion:
        
        class Choices:
            def __init__(self, choice: dict) -> None:
                self.text           = choice['text']
                self.content        = self.text.encode()
                self.index          = choice['index']
                self.logprobs       = choice['logprobs']
                self.finish_reason  = choice['finish_reason']
                
            def __repr__(self) -> str:
                return f'''<__main__.APIResponse.Completion.Choices(\n    text           = {self.text.encode()},\n    index          = {self.index},\n    logprobs       = {self.logprobs},\n    finish_reason  = {self.finish_reason})object at 0x1337>'''

        def __init__(self, choices: dict) -> None:
            self.choices = [self.Choices(choice) for choice in choices]

    class Usage:
        def __init__(self, usage_dict: dict) -> None:
            self.prompt_tokens      = usage_dict['prompt_tokens']
            self.completion_tokens  = usage_dict['completion_tokens']
            self.total_tokens       = usage_dict['total_tokens']

        def __repr__(self):
            return f'''<__main__.APIResponse.Usage(\n    prompt_tokens      = {self.prompt_tokens},\n    completion_tokens  = {self.completion_tokens},\n    total_tokens       = {self.total_tokens})object at 0x1337>'''
    
    def __init__(self, response_dict: dict) -> None:
        
        self.response_dict  = response_dict
        self.id             = response_dict['id']
        self.object         = response_dict['object']
        self.created        = response_dict['created']
        self.model          = response_dict['model']
        self.completion     = self.Completion(response_dict['choices'])
        self.usage          = self.Usage(response_dict['usage'])

    def json(self) -> dict:
        return self.response_dict

class Account:
    def create(proxy: None or str = None, logging: bool = False):
        
        client       = Session()
        client.proxies = {
            'http': f'http://{proxy}',
            'https': f'http://{proxy}'} if proxy else None
        
        mail         = Mail(client.proxies)
        mail_token   = None
        mail_address = mail.get_mail()

        if logging: print('email', mail_address)

        client.headers = {
            "host"              : "poe.com",
            "connection"        : "keep-alive",
            "cache-control"     : "max-age=0",
            "sec-ch-ua"         : "\"Microsoft Edge\";v=\"111\", \"Not(A:Brand\";v=\"8\", \"Chromium\";v=\"111\"",
            "sec-ch-ua-mobile"  : "?0",
            "sec-ch-ua-platform": "\"macOS\"",
            "user-agent"        : "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36 Edg/111.0.1661.54",
            "accept"            : "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "sec-fetch-site"    : "same-origin",
            "sec-fetch-mode"    : "navigate",
            "sec-fetch-user"    : "?1",
            "sec-fetch-dest"    : "document",
            "accept-encoding"   : "gzip, deflate, br",
            "accept-language"   : "en-GB,en;q=0.9,en-US;q=0.8",
            "upgrade-insecure-requests": "1",
        }

        init      = client.get('https://poe.com/login')
        next_data = loads(search(r'json">(.+?)</script>', init.text).group(1))

        client.headers["poe-formkey"]  =  next_data['props']['formkey']
        client.headers["poe-tchannel"] =  client.get('https://poe.com/api/settings').json()['tchannelData']['channel']

        payload = {
            "queryName": "MainSignupLoginSection_sendVerificationCodeMutation_Mutation",
            "variables": {
                "emailAddress": mail_address,
                "phoneNumber" : None
            },
            "query": "mutation MainSignupLoginSection_sendVerificationCodeMutation_Mutation(\n  $emailAddress: String\n  $phoneNumber: String\n) {\n  sendVerificationCode(verificationReason: login, emailAddress: $emailAddress, phoneNumber: $phoneNumber) {\n    status\n    errorMessage\n  }\n}\n"
        }

        response = client.post('https://poe.com/api/gql_POST', json=payload)
        if 'Bad Request' in response.text:
            if logging: print('bad request, retrying...' , response.json())
            Account.create(proxy = proxy, logging = logging)

        if logging: print('send_code' ,response.json())

        while True:
            sleep(1)
            inbox = mail.fetch_inbox()

            for _ in inbox:
                content    = mail.get_message(_["id"])
                mail_token = findall(r';">(\d{6,7})</div>', content['html'][0])[0]

            if mail_token:
                break

        if logging: print('code', mail_token)

        payload = {
            "queryName": "SignupOrLoginWithCodeSection_signupWithVerificationCodeMutation_Mutation",
            "variables": {
                "verificationCode"  : mail_token,
                "emailAddress"      : mail_address,
                "phoneNumber"       : None
            },
            "query": "mutation SignupOrLoginWithCodeSection_signupWithVerificationCodeMutation_Mutation(\n  $verificationCode: String!\n  $emailAddress: String\n  $phoneNumber: String\n) {\n  signupWithVerificationCode(verificationCode: $verificationCode, emailAddress: $emailAddress, phoneNumber: $phoneNumber) {\n    status\n    errorMessage\n  }\n}\n"
        }

        response = client.post('https://poe.com/api/gql_POST', json = payload)
        if logging: print('verify_code', response.json())

        token = parse.unquote(client.cookies.get_dict()['p-b'])
        
        with open(Path(__file__).resolve().parent / 'cookies.txt', 'a') as f:
            f.write(f'{token}\n')
            
        return token
    
    def get():
        cookies = open(Path(__file__).resolve().parent / 'cookies.txt', 'r').read().splitlines()
        return choice(cookies)

class StreamingCompletion:
    def create(
        model : str = 'gpt-4',
        prompt: str = 'hello world',
        token : str = ''):

        models = {
            'sage'   : 'capybara',
            'gpt-4'  : 'beaver',
            'claude+': 'a2_2',
            'claude' : 'a2',
            'gpt-3.5': 'chinchilla'
        }
        
        client = PoeClient(token)
        
        for chunk in client.send_message(models[model], prompt):
            
            yield PoeResponse({
                'id'     : chunk["messageId"], 
                'object' : 'text_completion', 
                'created': chunk['creationTime'], 
                'model'  : models[model], 
                'choices': [{
                        'text'          : chunk["text_new"], 
                        'index'         : 0, 
                        'logprobs'      : None, 
                        'finish_reason' : 'stop'
                }],
                'usage': {
                    'prompt_tokens'     : len(prompt), 
                    'completion_tokens' : len(chunk["text_new"]), 
                    'total_tokens'      : len(prompt) + len(chunk["text_new"])
                }
            })

class Completion:
    def create(
        model : str = 'gpt-4',
        prompt: str = 'hello world',
        token : str = ''):

        models = {
            'sage'   : 'capybara',
            'gpt-4'  : 'beaver',
            'claude+': 'a2_2',
            'claude' : 'a2',
            'gpt-3.5': 'chinchilla'
        }
        
        client = PoeClient(token)
        
        for chunk in client.send_message(models[model], prompt):
            pass
        
        return PoeResponse({
                'id'     : chunk["messageId"], 
                'object' : 'text_completion', 
                'created': chunk['creationTime'], 
                'model'  : models[model], 
                'choices': [{
                        'text'          : chunk["text"], 
                        'index'         : 0, 
                        'logprobs'      : None, 
                        'finish_reason' : 'stop'
                }],
                'usage': {
                    'prompt_tokens'     : len(prompt), 
                    'completion_tokens' : len(chunk["text"]), 
                    'total_tokens'      : len(prompt) + len(chunk["text"])
                }
            })
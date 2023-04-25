from requests   import Session
from names      import get_first_name, get_last_name
from random     import choice
from requests   import post
from time       import time
from colorama   import Fore, init; init()

class logger:
    @staticmethod
    def info(string) -> print:
        import datetime
        now = datetime.datetime.now()
        return print(f"{Fore.CYAN}{now.strftime('%Y-%m-%d %H:%M:%S')} {Fore.BLUE}INFO {Fore.MAGENTA}__main__ -> {Fore.RESET}{string}")

class SonicResponse:
    
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
            self.prompt_tokens      = usage_dict['prompt_chars']
            self.completion_tokens  = usage_dict['completion_chars']
            self.total_tokens       = usage_dict['total_chars']

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
    session = Session()
    session.headers = {
        "connection"        : "keep-alive",
        "sec-ch-ua"         : "\"Not_A Brand\";v=\"99\", \"Google Chrome\";v=\"109\", \"Chromium\";v=\"109\"",
        "accept"            : "application/json, text/plain, */*",
        "content-type"      : "application/json",
        "sec-ch-ua-mobile"  : "?0",
        "user-agent"        : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-site"    : "same-origin",
        "sec-fetch-mode"    : "cors",
        "sec-fetch-dest"    : "empty",
        # "accept-encoding"   : "gzip, deflate, br",
        "accept-language"   : "en-GB,en-US;q=0.9,en;q=0.8",
        "cookie"            : ""
    }
    
    @staticmethod
    def get_user():
        password = f'0opsYouGoTme@1234'
        f_name   = get_first_name()
        l_name   = get_last_name()
        hosts    = ['gmail.com', 'protonmail.com', 'proton.me', 'outlook.com']
        
        return {
            "email"             : f"{f_name.lower()}.{l_name.lower()}@{choice(hosts)}",
            "password"          : password,
            "confirm_password"  : password,
            "full_name"         : f'{f_name} {l_name}'
        }

    @staticmethod
    def create(logging: bool = False):
        while True:
            try:
                user     = Account.get_user()
                start    = time()
                response = Account.session.post("https://app.writesonic.com/api/session-login", json = user | {
                    "utmParams"         : "{}",
                    "visitorId"         : "0",
                    "locale"            : "en",
                    "userAgent"         : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
                    "signInWith"        : "password",
                    "request_type"      : "signup",
                })
                
                if logging:
                    logger.info(f"\x1b[31mregister success\x1b[0m : '{response.text[:30]}...' ({int(time() - start)}s)")
                    logger.info(f"\x1b[31mid\x1b[0m : '{response.json()['id']}'")
                    logger.info(f"\x1b[31mtoken\x1b[0m : '{response.json()['token'][:30]}...'")
                
                start = time()
                response = Account.session.post("https://api.writesonic.com/v1/business/set-business-active", headers={"authorization": "Bearer " + response.json()['token']})
                key = response.json()["business"]["api_key"]
                if logging: logger.info(f"\x1b[31mgot key\x1b[0m : '{key}' ({int(time() - start)}s)")

                return Account.AccountResponse(user['email'], user['password'], key)
            
            except Exception as e:
                if logging: logger.info(f"\x1b[31merror\x1b[0m : '{e}'")
                continue
            
    class AccountResponse:
        def __init__(self, email, password, key):
            self.email    = email
            self.password = password
            self.key      = key
            

class Completion:
    def create(
        api_key: str,
        prompt: str,
        enable_memory: bool = False,
        enable_google_results: bool = False,
        history_data: list = []) -> SonicResponse:
        
        response = post('https://api.writesonic.com/v2/business/content/chatsonic?engine=premium', headers = {"X-API-KEY": api_key},
            json = {
                    "enable_memory"         : enable_memory,
                    "enable_google_results" : enable_google_results,
                    "input_text"            : prompt,
                    "history_data"          : history_data}).json()

        return SonicResponse({
                'id'     : f'cmpl-premium-{int(time())}', 
                'object' : 'text_completion', 
                'created': int(time()), 
                'model'  : 'premium', 
                
                'choices': [{
                        'text'          : response['message'], 
                        'index'         : 0, 
                        'logprobs'      : None, 
                        'finish_reason' : 'stop'
                }],
                
                'usage': {
                    'prompt_chars'     : len(prompt), 
                    'completion_chars' : len(response['message']), 
                    'total_chars'      : len(prompt) + len(response['message'])
                }
            })
from requests import post
from time     import time

headers = {
    'authority'      : 'www.sqlchat.ai',
    'accept'         : '*/*',
    'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
    'content-type'   : 'text/plain;charset=UTF-8',
    'origin'         : 'https://www.sqlchat.ai',
    'referer'        : 'https://www.sqlchat.ai/',
    'sec-fetch-dest' : 'empty',
    'sec-fetch-mode' : 'cors',
    'sec-fetch-site' : 'same-origin',
    'user-agent'     : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
}

class SqlchatResponse:
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

class Completion:
    def create(
        prompt: str    = 'hello world',
        messages: list = []) -> SqlchatResponse:
        
        response = post('https://www.sqlchat.ai/api/chat', headers=headers, stream=True,
            json = {
                'messages': messages,
                'openAIApiConfig':{'key':'','endpoint':''}})

        return SqlchatResponse({
            'id'     : f'cmpl-1337-{int(time())}', 
            'object' : 'text_completion', 
            'created': int(time()), 
            'model'  : 'gpt-3.5-turbo', 
            'choices': [{
                    'text'          : response.text, 
                    'index'         : 0, 
                    'logprobs'      : None, 
                    'finish_reason' : 'stop'
            }], 
            'usage': {
                'prompt_chars'     : len(prompt), 
                'completion_chars' : len(response.text), 
                'total_chars'      : len(prompt) + len(response.text)
            }
        })

class StreamCompletion:
    def create(
        prompt  : str    = 'hello world',
        messages: list = [])  -> SqlchatResponse:
        
        messages.append({
            'role':'user',
            'content':prompt
        })

        response = post('https://www.sqlchat.ai/api/chat', headers=headers, stream=True,
            json = {
                'messages': messages,
                'openAIApiConfig':{'key':'','endpoint':''}})
        
        for chunk in response.iter_content(chunk_size = 2046):
            yield SqlchatResponse({
                'id'     : f'cmpl-1337-{int(time())}', 
                'object' : 'text_completion', 
                'created': int(time()), 
                'model'  : 'gpt-3.5-turbo', 
                
                'choices': [{
                        'text'          : chunk.decode(), 
                        'index'         : 0, 
                        'logprobs'      : None, 
                        'finish_reason' : 'stop'
                }],
                
                'usage': {
                    'prompt_chars'     : len(prompt), 
                    'completion_chars' : len(chunk.decode()), 
                    'total_chars'      : len(prompt) + len(chunk.decode())
                }
            })
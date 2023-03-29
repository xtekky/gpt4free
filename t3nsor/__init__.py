from requests import post
from time     import time

class T3nsorResponse:
    
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

class Completion:
    model = {
        'model': {
                'id'   : 'gpt-3.5-turbo', 
                'name' : 'Default (GPT-3.5)'
        }
    }

    def create(
        prompt: str    = 'hello world',
        messages: list = []) -> T3nsorResponse:

        response = post('https://www.t3nsor.tech/api/chat', json = Completion.model | {
            'messages'  : messages,
            'key'       : '',
            'prompt'    : prompt
        })

        return T3nsorResponse({
            'id'     : f'cmpl-1337-{int(time())}', 
            'object' : 'text_completion', 
            'created': int(time()), 
            'model'  : Completion.model, 
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
    model = {
        'model': {
            'id'   : 'gpt-3.5-turbo', 
            'name' : 'Default (GPT-3.5)'
        }
    }

    def create(
        prompt: str    = 'hello world',
        messages: list = [])  -> T3nsorResponse:

        response = post('https://www.t3nsor.tech/api/chat', stream = True, json = Completion.model | {
            'messages'  : messages,
            'key'       : '',
            'prompt'    : prompt
        })
        
        for resp in response.iter_lines():
            if resp:
                yield T3nsorResponse({
                    'id'     : f'cmpl-1337-{int(time())}', 
                    'object' : 'text_completion', 
                    'created': int(time()), 
                    'model'  : Completion.model, 
                    
                    'choices': [{
                            'text'          : resp.decode(), 
                            'index'         : 0, 
                            'logprobs'      : None, 
                            'finish_reason' : 'stop'
                    }],
                    
                    'usage': {
                        'prompt_chars'     : len(prompt), 
                        'completion_chars' : len(resp.decode()), 
                        'total_chars'      : len(prompt) + len(resp.decode())
                    }
                })
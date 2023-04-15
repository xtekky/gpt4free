from urllib.parse import quote
from tls_client   import Session
from time         import time
from datetime     import datetime

client         = Session(client_identifier='chrome110')
client.headers = {
    'authority': 'www.phind.com',
    'accept': '*/*',
    'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
    'content-type': 'application/json',
    'origin': 'https://www.phind.com',
    'referer': 'https://www.phind.com/search',
    'sec-ch-ua': '"Chromium";v="110", "Google Chrome";v="110", "Not:A-Brand";v="99"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36',
}

class PhindResponse:
    
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


class Search:
    def create(prompt: str, actualSearch: bool = True, language: str = 'en') -> dict: # None = no search
        if not actualSearch:
            return {
                '_type': 'SearchResponse',
                'queryContext': {
                    'originalQuery': prompt
                },
                'webPages': {
                    'webSearchUrl': f'https://www.bing.com/search?q={quote(prompt)}',
                    'totalEstimatedMatches': 0,
                    'value': []
                },
                'rankingResponse': {
                    'mainline': {
                        'items': []
                    }
                }
            }
        
        return client.post('https://www.phind.com/api/bing/search', json = { 
            'q': prompt,
            'userRankList': {},
            'browserLanguage': language}).json()['rawBingResults']

class Completion:
    def create(
        model = 'gpt-4', 
        prompt: str = '', 
        results: dict = None, 
        creative: bool = False, 
        detailed: bool = False, 
        codeContext: str = '',
        language: str = 'en') -> PhindResponse:
        
        if results is None:
            results = Search.create(prompt, actualSearch = True)
        
        if len(codeContext) > 2999:
            raise ValueError('codeContext must be less than 3000 characters')
        
        models = {
            'gpt-4' : 'expert',
            'gpt-3.5-turbo' : 'intermediate',
            'gpt-3.5': 'intermediate',
        }
        
        json_data = {
            'question'    : prompt,
            'bingResults' : results, #response.json()['rawBingResults'],
            'codeContext' : codeContext,
            'options': {
                'skill'   : models[model],
                'date'    : datetime.now().strftime("%d/%m/%Y"),
                'language': language,
                'detailed': detailed,
                'creative': creative
            }
        }
        
        completion = ''
        response   = client.post('https://www.phind.com/api/infer/answer', json=json_data, timeout_seconds=200)
        for line in response.text.split('\r\n\r\n'):
            completion += (line.replace('data: ', ''))
            
        return  PhindResponse({
            'id'     : f'cmpl-1337-{int(time())}', 
            'object' : 'text_completion', 
            'created': int(time()), 
            'model'  : models[model], 
            'choices': [{
                    'text'          : completion, 
                    'index'         : 0, 
                    'logprobs'      : None, 
                    'finish_reason' : 'stop'
            }], 
            'usage': {
                'prompt_tokens'     : len(prompt), 
                'completion_tokens' : len(completion), 
                'total_tokens'      : len(prompt) + len(completion)
            }
        })
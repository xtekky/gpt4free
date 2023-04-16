from ora.model  import CompletionModel
from ora.typing import OraResponse
from requests   import post
from time       import time
from random     import randint

class Completion:
    def create(
        model : CompletionModel,
        prompt: str,
        includeHistory: bool = True,
        conversationId: str or None = None) -> OraResponse:
        
        extra = {
            'conversationId': conversationId} if conversationId else {}
        
        response = post('https://ora.sh/api/conversation', 
            headers = {
                "host"          : "ora.sh",
                "authorization" : f"Bearer AY0{randint(1111, 9999)}",
                "user-agent"    : "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
                "origin"        : "https://ora.sh",
                "referer"       : "https://ora.sh/chat/",
            },
            json = extra | {
                'chatbotId': model.id,
                'input'    : prompt,
                'userId'   : model.createdBy,
                'model'    : model.modelName,
                'provider' : 'OPEN_AI',
                'includeHistory': includeHistory}).json()

        return OraResponse({
            'id'     : response['conversationId'], 
            'object' : 'text_completion', 
            'created': int(time()),
            'model'  : model.slug, 
            'choices': [{
                    'text'          : response['response'], 
                    'index'         : 0, 
                    'logprobs'      : None, 
                    'finish_reason' : 'stop'
            }],
            'usage': {
                'prompt_tokens'     : len(prompt), 
                'completion_tokens' : len(response['response']), 
                'total_tokens'      : len(prompt) + len(response['response'])
            }
        })
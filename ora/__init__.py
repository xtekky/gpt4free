from ora.model  import CompletionModel
from ora.typing import OraResponse
from requests   import post
from time       import time

class Completion:
    def create(
        model : CompletionModel,
        prompt: str,
        conversationId: str or None = None) -> OraResponse:
        
        extra = {
            'conversationId': conversationId} if conversationId else {}
            
        response = post('https://ora.sh/api/conversation', json = extra | {
            'chatbotId': model.id,
            'input'    : prompt,
            'userId'   : model.createdBy}).json()
        
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
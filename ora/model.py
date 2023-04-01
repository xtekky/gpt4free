from uuid     import uuid4
from requests import post

class CompletionModel:
    system_prompt = None
    description   = None
    createdBy     = None
    createdAt     = None
    slug          = None
    id            = None
    
    def create(
        system_prompt: str = 'You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible', 
        description  : str = 'ChatGPT Openai Language Model',
        name         : str = 'gpt-3.5'):

        CompletionModel.system_prompt = system_prompt
        CompletionModel.description   = description
        CompletionModel.slug          = name
        
        response = post('https://ora.sh/api/assistant', json = {
            'prompt'     : system_prompt,
            'userId'     : f'auto:{uuid4()}',
            'name'       : name,
            'description': description})
        
        CompletionModel.id        = response.json()['id']
        CompletionModel.createdBy = response.json()['createdBy']
        CompletionModel.createdAt = response.json()['createdAt']
        
        return CompletionModel
    
from uuid     import uuid4
from requests import post

class CompletionModel:
    system_prompt = None
    description   = None
    createdBy     = None
    createdAt     = None
    slug          = None
    id            = None
    model         = 'gpt-3.5-turbo'
    
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
    
    def load(chatbotId: str, modelName: str = 'gpt-3.5-turbo', userId: str = None):
        if userId is None: userId = f'{uuid4()}'

        CompletionModel.system_prompt = None
        CompletionModel.description   = None
        CompletionModel.slug          = None
        CompletionModel.id        = chatbotId
        CompletionModel.createdBy = userId
        CompletionModel.createdAt = None
        CompletionModel.modelName = modelName
        
        return CompletionModel
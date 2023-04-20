from uuid     import uuid4
from requests import post

class CompletionModel:
    system_prompt = None
    description   = None
    createdBy     = None
    createdAt     = None
    slug          = None
    id            = None
    modelName     = None
    model         = 'gpt-3.5-turbo'
    
    def create(
        system_prompt: str = 'You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible', 
        description  : str = 'ChatGPT Openai Language Model',
        name         : str = 'gpt-3.5'):

        CompletionModel.system_prompt = system_prompt
        CompletionModel.description   = description
        CompletionModel.slug          = name
        
        headers = {
            'Origin'    : 'https://ora.sh',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Safari/605.1.15',
            'Referer'   : 'https://ora.sh/',
            'Host'      : 'ora.sh',
        }
        
        response = post('https://ora.sh/api/assistant', headers = headers, json = {
            'prompt'     : system_prompt,
            'userId'     : f'auto:{uuid4()}',
            'name'       : name,
            'description': description})
        
        print(response.json())
        
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
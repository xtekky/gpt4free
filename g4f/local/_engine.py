import os

from gpt4all  import GPT4All
from ._models import models

class LocalProvider:
    @staticmethod
    def create_completion(model, messages, stream, **kwargs):
        if model not in models:
            raise ValueError(f"Model '{model}' not found / not yet implemented")
        
        model           = models[model]
        model_dir       = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/')
        full_model_path = os.path.join(model_dir, model['path'])
        
        if not os.path.isfile(full_model_path):
            print(f"Model file '{full_model_path}' not found.")
            download = input(f'Do you want to download {model["path"]} ? [y/n]')
            
            if download in ['y', 'Y']:
                GPT4All.download_model(model['path'], model_dir)
            else:
                raise ValueError(f"Model '{model['path']}' not found.")
        
        model = GPT4All(model_name=model['path'],
                               #n_threads=8,
                               verbose=False,
                               allow_download=False,
                               model_path=model_dir)
        
        system_template = next((message['content'] for message in messages if message['role'] == 'system'), 
                               'A chat between a curious user and an artificial intelligence assistant.')
        
        prompt_template = 'USER: {0}\nASSISTANT: '
        conversation    = '\n'.join(f"{msg['role'].upper()}: {msg['content']}" for msg in messages) + "\nASSISTANT: "
        
        with model.chat_session(system_template, prompt_template):
            if stream:
                for token in model.generate(conversation, streaming=True):
                    yield token
            else:
                yield model.generate(conversation)
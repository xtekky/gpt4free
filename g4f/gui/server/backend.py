import g4f

from flask      import request
from .internet  import search
from .config    import special_instructions
from .provider  import get_provider

g4f.logging = True

class Backend_Api:
    def __init__(self, app) -> None:
        self.app = app
        self.routes = {
            '/backend-api/v2/models': {
                'function': self.models,
                'methods' : ['GET']
            },
            '/backend-api/v2/conversation': {
                'function': self._conversation,
                'methods': ['POST']
            },
            '/backend-api/v2/gen.set.summarize:title': {
                'function': self._gen_title,
                'methods': ['POST']
            },
            '/backend-api/v2/error': {
                'function': self.error,
                'methods': ['POST']
            }
        }
    
    def error(self):
        print(request.json)
        
        return 'ok', 200
    
    def models(self):
        return g4f._all_models
    
    def _gen_title(self):
        return {
            'title': ''
        }
    
    def _conversation(self):
        try:
            #jailbreak       = request.json['jailbreak']
            #internet_access = request.json['meta']['content']['internet_access']
            #conversation    = request.json['meta']['content']['conversation']
            prompt          = request.json['meta']['content']['parts']
            model           = request.json['model']
            provider        = request.json.get('provider').split('g4f.Provider.')[1]

            messages = prompt
            print(messages)

            def stream():
                yield from g4f.ChatCompletion.create(
                    model=model,
                    provider=get_provider(provider),
                    messages=messages,
                    stream=True,
                ) if provider else g4f.ChatCompletion.create(
                    model=model, messages=messages, stream=True
                )

            return self.app.response_class(stream(), mimetype='text/event-stream')

        except Exception as e:    
            return {
                'code'   : 'G4F_ERROR',
                '_action': '_ask',
                'success': False,
                'error'  : f'an error occurred {str(e)}'}, 400

import g4f
from g4f.Provider import __providers__

import json
from flask      import request, Flask
from .internet  import get_search_message
from g4f import debug

debug.logging = True

class Backend_Api:
    def __init__(self, app: Flask) -> None:
        self.app: Flask = app
        self.routes = {
            '/backend-api/v2/models': {
                'function': self.models,
                'methods' : ['GET']
            },
            '/backend-api/v2/providers': {
                'function': self.providers,
                'methods' : ['GET']
            },
            '/backend-api/v2/version': {
                'function': self.version,
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
    
    def providers(self):
        return [
            provider.__name__ for provider in __providers__ if provider.working
        ]
        
    def version(self):
        return {
            "version": debug.get_version(),
            "lastet_version": debug.get_latest_version(),
        }
    
    def _gen_title(self):
        return {
            'title': ''
        }
    
    def _conversation(self):
        #jailbreak = request.json['jailbreak']
        messages = request.json['meta']['content']['parts']
        if request.json['internet_access']:
            messages[-1]["content"] = get_search_message(messages[-1]["content"])
        model = request.json.get('model')
        model = model if model else g4f.models.default
        provider = request.json.get('provider', '').replace('g4f.Provider.', '')
        provider = provider if provider and provider != "Auto" else None
            
        def try_response():
            try:
                yield from g4f.ChatCompletion.create(
                    model=model,
                    provider=provider,
                    messages=messages,
                    stream=True,
                    ignore_stream_and_auth=True
                )
            except Exception as e:
                print(e)
                yield json.dumps({
                    'code'   : 'G4F_ERROR',
                    '_action': '_ask',
                    'success': False,
                    'error'  : f'{e.__class__.__name__}: {e}'
                })

        return self.app.response_class(try_response(), mimetype='text/event-stream')
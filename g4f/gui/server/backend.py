import logging
import json
from flask import request, Flask
from g4f import debug, version, models
from g4f import _all_models, get_last_provider, ChatCompletion
from g4f.image import is_allowed_extension, to_image
from g4f.Provider import __providers__
from g4f.Provider.bing.create_images import patch_provider
from .internet import get_search_message

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
        return _all_models
    
    def providers(self):
        return [
            provider.__name__ for provider in __providers__ if provider.working
        ]
        
    def version(self):
        return {
            "version": version.utils.current_version,
            "lastet_version": version.get_latest_version(),
        }
    
    def _gen_title(self):
        return {
            'title': ''
        }
    
    def _conversation(self):
        kwargs = {}
        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '' and is_allowed_extension(file.filename):
                kwargs['image'] = to_image(file.stream)
        if 'json' in request.form:
            json_data = json.loads(request.form['json'])
        else:
            json_data = request.json
            
        provider = json_data.get('provider', '').replace('g4f.Provider.', '')
        provider = provider if provider and provider != "Auto" else None
        if provider == 'OpenaiChat':
            kwargs['auto_continue'] = True
        messages = json_data['messages']
        if json_data.get('web_search'):
            if provider == "Bing":
                kwargs['web_search'] = True
            else:
                messages[-1]["content"] = get_search_message(messages[-1]["content"])
        model = json_data.get('model')
        model = model if model else models.default
        provider = json_data.get('provider', '').replace('g4f.Provider.', '')
        provider = provider if provider and provider != "Auto" else None
        patch = patch_provider if json_data.get('patch_provider') else None

        def try_response():
            try:
                first = True
                for chunk in ChatCompletion.create(
                    model=model,
                    provider=provider,
                    messages=messages,
                    stream=True,
                    ignore_stream_and_auth=True,
                    patch_provider=patch,
                    **kwargs
                ):
                    if first:
                        first = False
                        yield json.dumps({
                            'type'    : 'provider',
                            'provider': get_last_provider(True)
                        }) + "\n"
                    if isinstance(chunk, Exception):
                         yield json.dumps({
                            'type'   : 'message',
                            'message': get_error_message(chunk),
                        }) + "\n"
                    else:
                        yield json.dumps({
                            'type'   : 'content',
                            'content': str(chunk),
                        }) + "\n"
            except Exception as e:
                logging.exception(e)
                yield json.dumps({
                    'type' : 'error',
                    'error': get_error_message(e)
                })

        return self.app.response_class(try_response(), mimetype='text/event-stream')
    
def get_error_message(exception: Exception) -> str:
    return f"{get_last_provider().__name__}: {type(exception).__name__}: {exception}"
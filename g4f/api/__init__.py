import typing
from .. import BaseProvider
import g4f; g4f.debug.logging = True
import time
import json
import random
import string
import logging

from typing                        import Union
from loguru                        import logger
from waitress                      import serve
from ._logging                     import hook_logging
from ._tokenizer                   import tokenize
from flask_cors                    import CORS
from werkzeug.serving              import WSGIRequestHandler
from werkzeug.exceptions           import default_exceptions
from werkzeug.middleware.proxy_fix import ProxyFix

from flask                         import (
    Flask, 
    jsonify, 
    make_response, 
    request,
)

class Api:
    __default_ip   = '127.0.0.1'
    __default_port = 1337
    
    def __init__(self, engine: g4f, debug: bool = True, sentry: bool = False,
                 list_ignored_providers:typing.List[typing.Union[str, BaseProvider]]=None) -> None:
        self.engine    = engine
        self.debug     = debug
        self.sentry    = sentry
        self.list_ignored_providers     = list_ignored_providers
        self.log_level = logging.DEBUG if debug else logging.WARN
        
        hook_logging(level=self.log_level, format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
        self.logger = logging.getLogger('waitress')
    
        self.app = Flask(__name__)
        self.app.wsgi_app = ProxyFix(self.app.wsgi_app, x_port=1)
        self.app.after_request(self.__after_request)
        
    def run(self, bind_str, threads=8):
        host, port = self.__parse_bind(bind_str)

        CORS(self.app, resources={r'/v1/*': {'supports_credentials': True, 'expose_headers': [
            'Content-Type',
            'Authorization',
            'X-Requested-With',
            'Accept',
            'Origin',
            'Access-Control-Request-Method',
            'Access-Control-Request-Headers',
            'Content-Disposition'], 'max_age': 600}})

        self.app.route('/v1/models',           methods=['GET'])(self.models)
        self.app.route('/v1/models/<model_id>', methods=['GET'])(self.model_info)

        self.app.route('/v1/chat/completions', methods=['POST'])(self.chat_completions)
        self.app.route('/v1/completions',      methods=['POST'])(self.completions)

        for ex in default_exceptions:
            self.app.register_error_handler(ex, self.__handle_error)

        if not self.debug:
            self.logger.warning(f'Serving on http://{host}:{port}')

        WSGIRequestHandler.protocol_version = 'HTTP/1.1'
        serve(self.app, host=host, port=port, ident=None, threads=threads)
    
    def __handle_error(self, e: Exception):
        self.logger.error(e)

        return make_response(jsonify({
            'code': e.code,
            'message': str(e.original_exception if self.debug and hasattr(e, 'original_exception') else e.name)}), 500)
    
    @staticmethod
    def __after_request(resp):
        resp.headers['X-Server'] = f'g4f/{g4f.version}'

        return resp
    
    def __parse_bind(self, bind_str):
        sections = bind_str.split(':', 2)
        if len(sections) < 2:
            try:
                port = int(sections[0])
                return self.__default_ip, port
            except ValueError:
                return sections[0], self.__default_port

        return sections[0], int(sections[1])
    
    async def home(self):
        return 'Hello world | https://127.0.0.1:1337/v1'
    
    async def chat_completions(self):
        model    = request.json.get('model', 'gpt-3.5-turbo')
        stream   = request.json.get('stream', False)
        messages = request.json.get('messages')
        
        logger.info(f'model: {model}, stream: {stream}, request: {messages[-1]["content"]}')

        response = self.engine.ChatCompletion.create(model=model, 
                                                     stream=stream, messages=messages,
                                                     ignored=self.list_ignored_providers)

        completion_id        = ''.join(random.choices(string.ascii_letters + string.digits, k=28))
        completion_timestamp = int(time.time())

        if not stream:
            prompt_tokens, _ = tokenize(''.join([message['content'] for message in messages]))
            completion_tokens, _ = tokenize(response)
            
            return {
                'id': f'chatcmpl-{completion_id}',
                'object': 'chat.completion',
                'created': completion_timestamp,
                'model': model,
                'choices': [
                    {
                        'index': 0,
                        'message': {
                            'role': 'assistant',
                            'content': response,
                        },
                        'finish_reason': 'stop',
                    }
                ],
                'usage': {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': prompt_tokens + completion_tokens,
                },
            }

        def streaming():
            try:
                for chunk in response:
                    completion_data = {
                        'id': f'chatcmpl-{completion_id}',
                        'object': 'chat.completion.chunk',
                        'created': completion_timestamp,
                        'model': model,
                        'choices': [
                            {
                                'index': 0,
                                'delta': {
                                    'content': chunk,
                                },
                                'finish_reason': None,
                            }
                        ],
                    }

                    content = json.dumps(completion_data, separators=(',', ':'))
                    yield f'data: {content}\n\n'
                    time.sleep(0.03)

                end_completion_data = {
                    'id': f'chatcmpl-{completion_id}',
                    'object': 'chat.completion.chunk',
                    'created': completion_timestamp,
                    'model': model,
                    'choices': [
                        {
                            'index': 0,
                            'delta': {},
                            'finish_reason': 'stop',
                        }
                    ],
                }
                
                content = json.dumps(end_completion_data, separators=(',', ':'))
                yield f'data: {content}\n\n'
                
                logger.success(f'model: {model}, stream: {stream}')
            
            except GeneratorExit:
                pass

        return self.app.response_class(streaming(), mimetype='text/event-stream')
    
    async def completions(self):
        return 'not working yet', 500
    
    async def model_info(self, model_name):
        model_info = (g4f.ModelUtils.convert[model_name])
    
        return jsonify({
            'id'       : model_name,
            'object'   : 'model',
            'created'  : 0,
            'owned_by' : model_info.base_provider
        })
    
    async def models(self):
        model_list = [{
            'id'        : model,
            'object'    : 'model',
            'created'   : 0,
            'owned_by'  : 'g4f'} for model in g4f.Model.__all__()]
            
        return jsonify({
            'object': 'list',
            'data': model_list})
    
from fastapi            import FastAPI, Response, Request
from fastapi.responses  import StreamingResponse
from typing             import List, Union, Any, Dict, AnyStr
from ._tokenizer        import tokenize
from ..                 import BaseProvider
from rich               import print

import time
import json
import random
import string
import uvicorn
import nest_asyncio
import g4f

class Api:
    def __init__(self, engine: g4f, debug: bool = True, sentry: bool = False,
                 list_ignored_providers: List[Union[str, BaseProvider]] = None, key: str = None) -> None:
        self.engine = engine
        self.debug = debug
        self.sentry = sentry
        self.list_ignored_providers = list_ignored_providers

        self.key = key

        self.app = FastAPI()
        nest_asyncio.apply()

        JSONObject = Dict[AnyStr, Any]
        JSONArray = List[Any]
        JSONStructure = Union[JSONArray, JSONObject]

        @self.app.route("/", methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
        async def root(request: Request):
            if self.debug:
                print(f"[!] A request on the root endpoint was received from {request.client.host}.")
            return Response(content=json.dumps({"detail": "g4f API"}, indent=4), media_type="application/json")

        @self.app.route("/v1", methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
        async def v1_root(request: Request):
            if self.debug:
                print(f"[!] A request on the V1 root endpoint was received from {request.client.host}.")
            return Response(content=json.dumps({"detail": "Go to /v1/chat/completions or /v1/models."}, indent=4), media_type="application/json")

        @self.app.route("/v1/models", methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
        async def v1_models(request: Request):
            if self.debug:
                print(f"[!] A request on the model endpoint was received from {request.client.host}.")
            model_list = [{
                'id': model,
                'object': 'model',
                'created': 0,
                'owned_by': 'g4f'} for model in g4f.Model.__all__()]

            return Response(content=json.dumps({
                'object': 'list',
                'data': model_list}, indent=4), media_type="application/json")

        @self.app.route("/v1/models/{model_name}", methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
        async def v1_model_info(request: Request):
            model_name = request.url.path.split('/')[3]

            if self.debug:
                print(f"[!] A request on the model info endpoint was received from {request.client.host}.")
            try:
                model_info = (g4f.ModelUtils.convert[model_name])

                return Response(content=json.dumps({
                    'id': model_name,
                    'object': 'model',
                    'created': 0,
                    'owned_by': model_info.base_provider
                }, indent=4), media_type="application/json")
            except:
                return Response(content=json.dumps({"error": "The model does not exist."}, indent=4), media_type="application/json")


        @self.app.post("/v1/chat/completions")
        async def v1_chat_completions(request: Request, item: JSONStructure = None):
            if self.debug:
                print(f"[!] A request on the chat completion endpoint was received from {request.client.host}.")

            if self.key != None:
                try:
                    auth_header = request.headers.get('Authorization').split(' ')
                    if auth_header[1] != self.key:
                        if self.debug:
                            print(f"[!] The request from {request.client.host} failed because of authentication.")
                        return Response(content=json.dumps({"error": "An invalid API key was provided."}, indent=4), media_type="application/json")
                    
                except:
                    if self.debug:
                        print(f"[!] The request from {request.client.host} failed because of authentication.")
                    return Response(content=json.dumps({"error": "An exception occurred while validating your API key."}, indent=4), media_type="application/json")

            item_data = {
                'model': 'gpt-3.5-turbo',
                'stream': False,
            }

            item_data.update(item or {})
            model = item_data.get('model')
            stream = item_data.get('stream')
            messages = item_data.get('messages')

            try:
                response = g4f.ChatCompletion.create(model=model, stream=stream, messages=messages)
            except:
                if self.debug:
                    print(f"[!] The request from {request.client.host} failed because of a provider-side error.")
                return Response(content=json.dumps({"error": "An error occurred while generating the response."}, indent=4), media_type="application/json")

            completion_id = ''.join(random.choices(string.ascii_letters + string.digits, k=28))
            completion_timestamp = int(time.time())

            if not stream:
                prompt_tokens, _ = tokenize(''.join([message['content'] for message in messages]))
                completion_tokens, _ = tokenize(response)

                json_data = {
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

                if self.debug:
                    print(f"[!] The request from {request.client.host} succeeded.")

                return Response(content=json.dumps(json_data, indent=4), media_type="application/json")

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

                except GeneratorExit:
                    pass

            if self.debug:
                print(f"[!] The request from {request.client.host} succeeded.")

            return StreamingResponse(streaming(), media_type="text/event-stream")

        @self.app.route("/v1/completions", methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
        async def completions(request: Request):
            if self.debug:
                print(f"[!] A request on the completions endpoint was received from {request.client.host}.")
            return Response(content=json.dumps({'detail': 'Not working yet.'}, indent=4), media_type="application/json")

    def run(self, ip):
        split_ip = ip.split(":")
        print(f"[!] The g4f API is now running on http://{ip}/")
        uvicorn.run(app=self.app, host=split_ip[0], port=int(split_ip[1]), use_colors=False, log_level="error")

import ast
import logging
import time
import json
import random
import string
import uvicorn
import nest_asyncio

from fastapi           import FastAPI, Response, Request
from fastapi.responses import StreamingResponse
from typing            import List, Union, Any, Dict, AnyStr
#from ._tokenizer       import tokenize

import g4f
from .. import debug

debug.logging = True

class Api:
    def __init__(self, engine: g4f, debug: bool = True, sentry: bool = False,
                 list_ignored_providers: List[str] = None) -> None:
        self.engine = engine
        self.debug = debug
        self.sentry = sentry
        self.list_ignored_providers = list_ignored_providers

        self.app = FastAPI()
        nest_asyncio.apply()

        JSONObject = Dict[AnyStr, Any]
        JSONArray = List[Any]
        JSONStructure = Union[JSONArray, JSONObject]

        @self.app.get("/")
        async def read_root():
            return Response(content=json.dumps({"info": "g4f API"}, indent=4), media_type="application/json")

        @self.app.get("/v1")
        async def read_root_v1():
            return Response(content=json.dumps({"info": "Go to /v1/chat/completions or /v1/models."}, indent=4), media_type="application/json")

        @self.app.get("/v1/models")
        async def models():
            model_list = []
            for model in g4f.Model.__all__():
                model_info = (g4f.ModelUtils.convert[model])
                model_list.append({
                'id': model,
                'object': 'model',
                'created': 0,
                'owned_by': model_info.base_provider}
                )
            return Response(content=json.dumps({
                'object': 'list',
                'data': model_list}, indent=4), media_type="application/json")

        @self.app.get("/v1/models/{model_name}")
        async def model_info(model_name: str):
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
        async def chat_completions(request: Request, item: JSONStructure = None):
            item_data = {
                'model': 'gpt-3.5-turbo',
                'stream': False,
            }

            # item contains byte keys, and dict.get suppresses error
            item_data.update({
                key.decode('utf-8') if isinstance(key, bytes) else key: str(value)
                for key, value in (item or {}).items()
            })
            # messages is str, need dict
            if isinstance(item_data.get('messages'), str):
                item_data['messages'] = ast.literal_eval(item_data.get('messages'))

            model = item_data.get('model')
            stream = True if item_data.get("stream") == "True" else False
            messages = item_data.get('messages')
            provider = item_data.get('provider', '').replace('g4f.Provider.', '')
            provider = provider if provider and provider != "Auto" else None

            try:
                response = g4f.ChatCompletion.create(
                    model=model,
                    stream=stream,
                    messages=messages,
                    provider = provider,
                    ignored=self.list_ignored_providers
                )
            except Exception as e:
                logging.exception(e)
                content = json.dumps({
                    "error": {"message": f"An error occurred while generating the response:\n{e}"},
                    "model": model,
                    "provider": g4f.get_last_provider(True)
                })
                return Response(content=content, status_code=500, media_type="application/json")
            completion_id = ''.join(random.choices(string.ascii_letters + string.digits, k=28))
            completion_timestamp = int(time.time())

            if not stream:
                #prompt_tokens, _ = tokenize(''.join([message['content'] for message in messages]))
                #completion_tokens, _ = tokenize(response)

                json_data = {
                    'id': f'chatcmpl-{completion_id}',
                    'object': 'chat.completion',
                    'created': completion_timestamp,
                    'model': model,
                    'provider': g4f.get_last_provider(True),
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
                        'prompt_tokens': 0, #prompt_tokens,
                        'completion_tokens': 0, #completion_tokens,
                        'total_tokens': 0, #prompt_tokens + completion_tokens,
                    },
                }

                return Response(content=json.dumps(json_data, indent=4), media_type="application/json")

            def streaming():
                try:
                    for chunk in response:
                        completion_data = {
                            'id': f'chatcmpl-{completion_id}',
                            'object': 'chat.completion.chunk',
                            'created': completion_timestamp,
                            'model': model,
                            'provider': g4f.get_last_provider(True),
                            'choices': [
                                {
                                    'index': 0,
                                    'delta': {
                                        'role': 'assistant',
                                        'content': chunk,
                                    },
                                    'finish_reason': None,
                                }
                            ],
                        }
                        yield f'data: {json.dumps(completion_data)}\n\n'
                        time.sleep(0.03)
                    end_completion_data = {
                        'id': f'chatcmpl-{completion_id}',
                        'object': 'chat.completion.chunk',
                        'created': completion_timestamp,
                        'model': model,
                        'provider': g4f.get_last_provider(True),
                        'choices': [
                            {
                                'index': 0,
                                'delta': {},
                                'finish_reason': 'stop',
                            }
                        ],
                    }
                    yield f'data: {json.dumps(end_completion_data)}\n\n'
                except GeneratorExit:
                    pass
                except Exception as e:
                    logging.exception(e)
                    content = json.dumps({
                        "error": {"message": f"An error occurred while generating the response:\n{e}"},
                        "model": model,
                        "provider": g4f.get_last_provider(True),
                    })
                    yield f'data: {content}'

            return StreamingResponse(streaming(), media_type="text/event-stream")

        @self.app.post("/v1/completions")
        async def completions():
            return Response(content=json.dumps({'info': 'Not working yet.'}, indent=4), media_type="application/json")

    def run(self, ip):
        split_ip = ip.split(":")
        uvicorn.run(app=self.app, host=split_ip[0], port=int(split_ip[1]), use_colors=False)

import g4f
import ast
import json
import logging
import random
import string
import time

from flask import request, Response
from typing import List, Union
from ... import BaseProvider

g4f.logging = True


class Api:
    def __init__(self,
                 app,
                 env,
                 list_ignored_providers: List[Union[str, BaseProvider]] = None
                 ) -> None:
        self.app = app
        self.env = env
        self.list_ignored_providers = list_ignored_providers
        self.routes = {
            '/v1': {
                'function': self.v1,
                'methods': ['GET', 'POST']
            },
            '/v1/models': {
                'function': self.v1_models,
                'methods': ['GET', 'POST']
            },
            '/v1/models/<model_name>': {
                'function': self.v1_model_info,
                'methods': ['GET', 'POST']
            },
            '/v1/chat/completions': {
                'function': self.v1_chat_completions,
                'methods': ['GET', 'POST']
            },
            '/v1/completions': {
                'function': self.v1_completions,
                'methods': ['GET', 'POST']
            },
        }

    async def v1(self):
        return self.responseJson({
            "error": {
                "message": "Go to /v1/chat/completions or /v1/models.",
                "type": "invalid_request_error",
                "param": "null",
                "code": "null",
            }
        }, status=404)

    async def v1_models(self):
        model_list = []
        for model in g4f.Model.__all__():
            model_info = (g4f.ModelUtils.convert[model])
            model_list.append({
                'id': model,
                'object': 'model',
                'created': 0,
                'owned_by': model_info.base_provider}
            )
        return self.responseJson({
            'object': 'list',
            'data': model_list})

    async def v1_model_info(self, model_name: str):
        try:
            model_info = (g4f.ModelUtils.convert[model_name])
            return self.responseJson({
                'id': model_name,
                'object': 'model',
                'created': 0,
                'owned_by': model_info.base_provider
            })
        except Exception:
            return self.responseJson({
                "error": {
                    "message": "The model does not exist.",
                    "type": "invalid_request_error",
                    "param": "null",
                    "code": "null",
                }
            }, status=500)

    async def v1_chat_completions(self):
        if request.method == 'GET':
            return self.responseJson({
                "error": {
                    "message":
                    "You must use POST to access this endpoint.",
                    "type": "invalid_request_error",
                    "param": "null",
                    "code": "null",
                }
            }, status=501)
        item = request.get_json()
        item_data = {
            'model': 'gpt-3.5-turbo',
            'stream': False,
        }
        item_data.update({
            key.decode('utf-8')
            if isinstance(key, bytes) else key: str(value)
            for key, value in (item or {}).items()
        })
        if isinstance(item_data.get('messages'), str):
            item_data['messages'] = ast.literal_eval(
                item_data.get('messages'))

        model = item_data.get('model')
        stream = True if item_data.get("stream") == "True" else False
        messages = item_data.get('messages')
        provider = item_data.get('provider', '').replace('g4f.Provider.', '')
        provider = provider if provider and provider != "Auto" else None
        # conversation = item_data.get('conversation') if item_data.get(
        #     'conversation') is not None else None

        try:
            response = g4f.ChatCompletion.create(
                model=model,
                stream=stream,
                messages=messages,
                provider=provider,
                proxy=self.env.get('proxy', None),
                socks5=self.env.get('socks5', None),
                time=self.env.get('timeout', 120),
                ignored=self.list_ignored_providers)
        except Exception as e:
            logging.exception(e)
            return self.responseJson({
                "error": {
                    "message":
                    "An error occurred while generating the response.",
                    "type": "invalid_request_error",
                    "param": "null",
                    "code": "null",
                }
            }, status=500)

        completion_id = ''.join(random.choices(
            string.ascii_letters + string.digits, k=28))
        completion_timestamp = int(time.time())

        if not stream:
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
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0,
                },
            }

            return self.responseJson(
                json_data,
                content_type="application/json"
            )

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

                    content = json.dumps(
                        completion_data, separators=(',', ':'))
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

                content = json.dumps(end_completion_data, separators=(
                    ',', ':'))
                yield f'data: {content}\n\n'

            except GeneratorExit:
                pass

        return Response(
            streaming(),
            content_type='text/event-stream'
        )

    async def v1_completions(self):
        if request.method == 'GET':
            return self.responseJson({
                "error": {
                    "message":
                    "You must use POST to access this endpoint.",
                    "type": "invalid_request_error",
                    "param": "null",
                    "code": "null",
                }
            })
        item = request.get_json()
        item_data = {
            'model': 'text-davinci-003'
        }
        item_data.update({
            key.decode('utf-8')
            if isinstance(key, bytes) else key: str(value)
            for key, value in (item or {}).items()
        })
        model = item_data.get('model')
        prompt = item_data.get('prompt')
        try:
            response = g4f.Completion.create(
                model=model,
                prompt=prompt,
                proxy=self.env.get('proxy', None),
                socks5=self.env.get('socks5', None),
                time=self.env.get('timeout', 120))
            return self.responseJson(response)
        except Exception as e:
            logging.exception(e)
            return self.responseJson({
                "error": {
                    "message":
                    "An error occurred while generating the response.",
                    "type": "invalid_request_error",
                    "param": "null",
                    "code": "null",
                }
            }, status=500)

    def responseJson(self, response,
                     content_type="application/json", status=200):
        return Response(
            json.dumps(response, indent=4),
            content_type=content_type,
            status=status)

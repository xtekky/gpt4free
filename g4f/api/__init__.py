import logging
import json
import uvicorn
import nest_asyncio

from fastapi           import FastAPI, Response, Request
from fastapi.responses import StreamingResponse, RedirectResponse, HTMLResponse, JSONResponse
from pydantic          import BaseModel
from typing            import List, Union

import g4f
import g4f.debug
from g4f.client import Client
from g4f.typing import Messages

class ChatCompletionsConfig(BaseModel):
    messages: Messages
    model: str
    provider: Union[str, None] = None
    stream: bool = False
    temperature: Union[float, None] = None
    max_tokens: Union[int, None] = None
    stop: Union[list[str], str, None] = None
    api_key: Union[str, None] = None

class Api:
    def __init__(self, engine: g4f, debug: bool = True, sentry: bool = False,
                 list_ignored_providers: List[str] = None) -> None:
        self.engine = engine
        self.debug = debug
        self.sentry = sentry
        self.list_ignored_providers = list_ignored_providers

        if debug:
            g4f.debug.logging = True
        self.client = Client()

        nest_asyncio.apply()
        self.app = FastAPI()

        self.routes()

    def routes(self):
        @self.app.get("/")
        async def read_root():
            return RedirectResponse("/v1", 302)

        @self.app.get("/v1")
        async def read_root_v1():
            return HTMLResponse('g4f API: Go to '
                                '<a href="/v1/chat/completions">chat/completions</a> '
                                'or <a href="/v1/models">models</a>.')

        @self.app.get("/v1/models")
        async def models():
            model_list = dict(
                (model, g4f.ModelUtils.convert[model])
                for model in g4f.Model.__all__()
            )
            model_list = [{
                'id': model_id,
                'object': 'model',
                'created': 0,
                'owned_by': model.base_provider
            } for model_id, model in model_list.items()]
            return JSONResponse(model_list)

        @self.app.get("/v1/models/{model_name}")
        async def model_info(model_name: str):
            try:
                model_info = g4f.ModelUtils.convert[model_name]
                return JSONResponse({
                    'id': model_name,
                    'object': 'model',
                    'created': 0,
                    'owned_by': model_info.base_provider
                })
            except:
                return JSONResponse({"error": "The model does not exist."})

        @self.app.post("/v1/chat/completions")
        async def chat_completions(config: ChatCompletionsConfig = None, request: Request = None, provider: str = None):
            try:
                config.provider = provider if config.provider is None else config.provider
                if config.api_key is None and request is not None:
                    auth_header = request.headers.get("Authorization")
                    if auth_header is not None:
                        auth_header = auth_header.split(None, 1)[-1]
                        if auth_header and auth_header != "Bearer":
                            config.api_key = auth_header
                response = self.client.chat.completions.create(
                    **config.dict(exclude_none=True),
                    ignored=self.list_ignored_providers
                )
            except Exception as e:
                logging.exception(e)
                return Response(content=format_exception(e, config), status_code=500, media_type="application/json")

            if not config.stream:
                return JSONResponse(response.to_json())

            def streaming():
                try:
                    for chunk in response:
                        yield f"data: {json.dumps(chunk.to_json())}\n\n"
                except GeneratorExit:
                    pass
                except Exception as e:
                    logging.exception(e)
                    yield f'data: {format_exception(e, config)}'

            return StreamingResponse(streaming(), media_type="text/event-stream")

        @self.app.post("/v1/completions")
        async def completions():
            return Response(content=json.dumps({'info': 'Not working yet.'}, indent=4), media_type="application/json")

    def run(self, ip, use_colors : bool = False):
        split_ip = ip.split(":")
        uvicorn.run(app=self.app, host=split_ip[0], port=int(split_ip[1]), use_colors=use_colors)

def format_exception(e: Exception, config: ChatCompletionsConfig) -> str:
    last_provider = g4f.get_last_provider(True)
    return json.dumps({
        "error": {"message": f"{e.__class__.__name__}: {e}"},
        "model": last_provider.get("model") if last_provider else config.model,
        "provider": last_provider.get("name") if last_provider else config.provider
    })

def run_api(host: str = '0.0.0.0', port: int = 1337, debug: bool = False, use_colors=True) -> None:
    print(f'Starting server... [g4f v-{g4f.version.utils.current_version}]')
    app = Api(engine=g4f, debug=debug)
    app.run(f"{host}:{port}", use_colors=use_colors)
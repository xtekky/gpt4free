import logging
import json
import uvicorn

from fastapi import FastAPI, Response, Request
from fastapi.responses import StreamingResponse, RedirectResponse, HTMLResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import List, Union, Optional

import g4f
import g4f.debug
from g4f.client import AsyncClient
from g4f.typing import Messages

app = FastAPI()

class ChatCompletionsConfig(BaseModel):
    messages: Messages
    model: str
    provider: Optional[str] = None
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Union[list[str], str, None] = None
    api_key: Optional[str] = None
    web_search: Optional[bool] = None
    proxy: Optional[str] = None

class Api:
    def __init__(self, list_ignored_providers: List[str] = None) -> None:
        self.list_ignored_providers = list_ignored_providers
        self.client = AsyncClient()
    
    def set_list_ignored_providers(self, list: list):
        self.list_ignored_providers = list

    def register_validation_exception_handler(self):
        @app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):
            details = exc.errors()
            modified_details = []
            for error in details:
                modified_details.append({
                    "loc": error["loc"],
                    "message": error["msg"],
                    "type": error["type"],
                })
            return JSONResponse(
                status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                content=jsonable_encoder({"detail": modified_details}),
            )

    def register_routes(self):
        @app.get("/")
        async def read_root():
            return RedirectResponse("/v1", 302)

        @app.get("/v1")
        async def read_root_v1():
            return HTMLResponse('g4f API: Go to '
                                '<a href="/v1/chat/completions">chat/completions</a> '
                                'or <a href="/v1/models">models</a>.')

        @app.get("/v1/models")
        async def models():
            model_list = dict(
                (model, g4f.models.ModelUtils.convert[model])
                for model in g4f.Model.__all__()
            )
            model_list = [{
                'id': model_id,
                'object': 'model',
                'created': 0,
                'owned_by': model.base_provider
            } for model_id, model in model_list.items()]
            return JSONResponse(model_list)

        @app.get("/v1/models/{model_name}")
        async def model_info(model_name: str):
            try:
                model_info = g4f.models.ModelUtils.convert[model_name]
                return JSONResponse({
                    'id': model_name,
                    'object': 'model',
                    'created': 0,
                    'owned_by': model_info.base_provider
                })
            except:
                return JSONResponse({"error": "The model does not exist."})

        @app.post("/v1/chat/completions")
        async def chat_completions(config: ChatCompletionsConfig, request: Request = None, provider: str = None):
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
                return JSONResponse((await response).to_json())

            async def streaming():
                try:
                    async for chunk in response:
                        yield f"data: {json.dumps(chunk.to_json())}\n\n"
                except GeneratorExit:
                    pass
                except Exception as e:
                    logging.exception(e)
                    yield f'data: {format_exception(e, config)}\n\n'
                yield "data: [DONE]\n\n"

            return StreamingResponse(streaming(), media_type="text/event-stream")

        @app.post("/v1/completions")
        async def completions():
            return Response(content=json.dumps({'info': 'Not working yet.'}, indent=4), media_type="application/json")

api = Api()
api.register_routes()
api.register_validation_exception_handler()

def format_exception(e: Exception, config: ChatCompletionsConfig) -> str:
    last_provider = g4f.get_last_provider(True)
    return json.dumps({
        "error": {"message": f"{e.__class__.__name__}: {e}"},
        "model": last_provider.get("model") if last_provider else config.model,
        "provider": last_provider.get("name") if last_provider else config.provider
    })

def run_api(
    host: str = '0.0.0.0',
    port: int = 1337,
    bind: str = None,
    debug: bool = False,
    workers: int = None,
    use_colors: bool = None
) -> None:
    print(f'Starting server... [g4f v-{g4f.version.utils.current_version}]' + (" (debug)" if debug else ""))
    if use_colors is None:
        use_colors = debug
    if bind is not None:
        host, port = bind.split(":")
    if debug:
        g4f.debug.logging = True
    uvicorn.run("g4f.api:app", host=host, port=int(port), workers=workers, use_colors=use_colors)#
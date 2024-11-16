from __future__ import annotations

import logging
import json
import uvicorn
import secrets

from fastapi import FastAPI, Response, Request
from fastapi.responses import StreamingResponse, RedirectResponse, HTMLResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.security import APIKeyHeader
from starlette.exceptions import HTTPException
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY, HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Union, Optional

import g4f
import g4f.debug
from g4f.client import AsyncClient, ChatCompletion
from g4f.client.helper import filter_none
from g4f.typing import Messages
from g4f.cookies import read_cookie_files

logger = logging.getLogger(__name__)

def create_app(g4f_api_key: str = None):
    app = FastAPI()

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=".*",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api = Api(app, g4f_api_key=g4f_api_key)
    api.register_routes()
    api.register_authorization()
    api.register_validation_exception_handler()

    # Read cookie files if not ignored
    if not AppConfig.ignore_cookie_files:
        read_cookie_files()

    return app

def create_app_debug(g4f_api_key: str = None):
    g4f.debug.logging = True
    return create_app(g4f_api_key)

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

class ImageGenerationConfig(BaseModel):
    prompt: str
    model: Optional[str] = None
    provider: Optional[str] = None
    response_format: str = "url"
    api_key: Optional[str] = None
    proxy: Optional[str] = None

class AppConfig:
    ignored_providers: Optional[list[str]] = None
    g4f_api_key: Optional[str] = None
    ignore_cookie_files: bool = False
    model: str = None,
    provider: str = None
    image_provider: str = None
    proxy: str = None

    @classmethod
    def set_config(cls, **data):
        for key, value in data.items():
            setattr(cls, key, value)

list_ignored_providers: list[str] = None

def set_list_ignored_providers(ignored: list[str]):
    global list_ignored_providers
    list_ignored_providers = ignored

class Api:
    def __init__(self, app: FastAPI, g4f_api_key=None) -> None:
        self.app = app
        self.client = AsyncClient()
        self.g4f_api_key = g4f_api_key
        self.get_g4f_api_key = APIKeyHeader(name="g4f-api-key")

    def register_authorization(self):
        @self.app.middleware("http")
        async def authorization(request: Request, call_next):
            if self.g4f_api_key and request.url.path in ["/v1/chat/completions", "/v1/completions", "/v1/images/generate"]:
                try:
                    user_g4f_api_key = await self.get_g4f_api_key(request)
                except HTTPException as e:
                    if e.status_code == 403:
                        return JSONResponse(
                            status_code=HTTP_401_UNAUTHORIZED,
                            content=jsonable_encoder({"detail": "G4F API key required"}),
                        )
                if not secrets.compare_digest(self.g4f_api_key, user_g4f_api_key):
                    return JSONResponse(
                        status_code=HTTP_403_FORBIDDEN,
                        content=jsonable_encoder({"detail": "Invalid G4F API key"}),
                    )

            response = await call_next(request)
            return response

    def register_validation_exception_handler(self):
        @self.app.exception_handler(RequestValidationError)
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
        @self.app.get("/")
        async def read_root():
            return RedirectResponse("/v1", 302)

        @self.app.get("/v1")
        async def read_root_v1():
            return HTMLResponse('g4f API: Go to '
                                '<a href="/v1/models">models</a>, '
                                '<a href="/v1/chat/completions">chat/completions</a>, or '
                                '<a href="/v1/images/generate">images/generate</a>.')

        @self.app.get("/v1/models")
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

        @self.app.get("/v1/models/{model_name}")
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

        @self.app.post("/v1/chat/completions")
        async def chat_completions(config: ChatCompletionsConfig, request: Request = None, provider: str = None):
            try:
                config.provider = provider if config.provider is None else config.provider
                if config.api_key is None and request is not None:
                    auth_header = request.headers.get("Authorization")
                    if auth_header is not None:
                        auth_header = auth_header.split(None, 1)[-1]
                        if auth_header and auth_header != "Bearer":
                            config.api_key = auth_header

                # Create the completion response
                response = self.client.chat.completions.create(
                    **filter_none(
                        **{
                            "model": AppConfig.model,
                            "provider": AppConfig.provider,
                            "proxy": AppConfig.proxy,
                            **config.dict(exclude_none=True),
                        },
                        ignored=AppConfig.ignored_providers
                    ),
                )

                if not config.stream:
                    response: ChatCompletion = await response
                    return JSONResponse(response.to_json())

                async def streaming():
                    try:
                        async for chunk in response:
                            yield f"data: {json.dumps(chunk.to_json())}\n\n"
                    except GeneratorExit:
                        pass
                    except Exception as e:
                        logger.exception(e)
                        yield f'data: {format_exception(e, config)}\n\n'
                    yield "data: [DONE]\n\n"

                return StreamingResponse(streaming(), media_type="text/event-stream")

            except Exception as e:
                logger.exception(e)
                return Response(content=format_exception(e, config), status_code=500, media_type="application/json")

        @self.app.post("/v1/images/generate")
        @self.app.post("/v1/images/generations")
        async def generate_image(config: ImageGenerationConfig):
            try:
                response = await self.client.images.generate(
                    prompt=config.prompt,
                    model=config.model,
                    provider=AppConfig.image_provider if config.provider is None else config.provider,
                    **filter_none(
                        response_format = config.response_format,
                        api_key = config.api_key,
                        proxy = config.proxy
                    )
                )
                return JSONResponse(response.to_json())
            except Exception as e:
                logger.exception(e)
                return Response(content=format_exception(e, config, True), status_code=500, media_type="application/json")

        @self.app.post("/v1/completions")
        async def completions():
            return Response(content=json.dumps({'info': 'Not working yet.'}, indent=4), media_type="application/json")

def format_exception(e: Exception, config: Union[ChatCompletionsConfig, ImageGenerationConfig], image: bool = False) -> str:
    last_provider = {} if not image else g4f.get_last_provider(True)
    provider = (AppConfig.image_provider if image else AppConfig.provider) if config.provider is None else config.provider
    model = AppConfig.model if config.model is None else config.model 
    return json.dumps({
        "error": {"message": f"{e.__class__.__name__}: {e}"},
        "model":  last_provider.get("model") if model is None else model,
        **filter_none(
            provider=last_provider.get("name") if provider is None else provider
        )
    })

def run_api(
    host: str = '0.0.0.0',
    port: int = 1337,
    bind: str = None,
    debug: bool = False,
    workers: int = None,
    use_colors: bool = None,
    reload: bool = False
) -> None:
    print(f'Starting server... [g4f v-{g4f.version.utils.current_version}]' + (" (debug)" if debug else ""))
    if use_colors is None:
        use_colors = debug
    if bind is not None:
        host, port = bind.split(":")
    uvicorn.run(
        f"g4f.api:create_app{'_debug' if debug else ''}", 
        host=host, 
        port=int(port), 
        workers=workers, 
        use_colors=use_colors, 
        factory=True, 
        reload=reload
    )
from __future__ import annotations

import logging
import json
import uvicorn
import secrets
import os
import shutil
from email.utils import formatdate
import os.path
import hashlib
import asyncio
from fastapi import FastAPI, Response, Request, UploadFile, Depends
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse, HTMLResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.security import APIKeyHeader
from starlette.exceptions import HTTPException
from starlette.status import (
    HTTP_200_OK,
    HTTP_422_UNPROCESSABLE_ENTITY, 
    HTTP_404_NOT_FOUND,
    HTTP_401_UNAUTHORIZED,
    HTTP_403_FORBIDDEN,
    HTTP_500_INTERNAL_SERVER_ERROR,
)
from starlette.staticfiles import NotModifiedResponse
from fastapi.encoders import jsonable_encoder
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, HTTPBasic
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from types import SimpleNamespace
from typing import Union, Optional, List

import g4f
import g4f.Provider
import g4f.debug
from g4f.client import AsyncClient, ChatCompletion, ImagesResponse, convert_to_provider
from g4f.providers.response import BaseConversation, JsonConversation
from g4f.client.helper import filter_none
from g4f.image import is_data_uri_an_image, images_dir, copy_images
from g4f.errors import ProviderNotFoundError, ModelNotFoundError, MissingAuthError, NoValidHarFileError
from g4f.cookies import read_cookie_files, get_cookies_dir
from g4f.Provider import ProviderType, ProviderUtils, __providers__
from g4f.gui import get_gui_app
from g4f.tools.files import supports_filename, get_async_streaming
from .stubs import (
    ChatCompletionsConfig, ImageGenerationConfig,
    ProviderResponseModel, ModelResponseModel,
    ErrorResponseModel, ProviderResponseDetailModel,
    FileResponseModel, UploadResponseModel, Annotated
)
from g4f import debug

logger = logging.getLogger(__name__)

DEFAULT_PORT = 1337

def create_app():
    app = FastAPI()

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=".*",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api = Api(app)

    api.register_routes()
    api.register_authorization()
    api.register_validation_exception_handler()

    if AppConfig.gui:
        gui_app = WSGIMiddleware(get_gui_app(AppConfig.demo))
        app.mount("/", gui_app)

    # Read cookie files if not ignored
    if not AppConfig.ignore_cookie_files:
        read_cookie_files()

    if AppConfig.ignored_providers:
        for provider in AppConfig.ignored_providers:
            if provider in ProviderUtils.convert:
                ProviderUtils.convert[provider].working = False

    return app

def create_app_debug():
    g4f.debug.logging = True
    return create_app()

def create_app_with_gui_and_debug():
    g4f.debug.logging = True
    AppConfig.gui = True
    return create_app()

def create_app_with_demo_and_debug():
    g4f.debug.logging = True
    AppConfig.gui = True
    AppConfig.demo = True
    return create_app()

class ErrorResponse(Response):
    media_type = "application/json"

    @classmethod
    def from_exception(cls, exception: Exception,
                       config: Union[ChatCompletionsConfig, ImageGenerationConfig] = None,
                       status_code: int = HTTP_500_INTERNAL_SERVER_ERROR):
        return cls(format_exception(exception, config), status_code)

    @classmethod
    def from_message(cls, message: str, status_code: int = HTTP_500_INTERNAL_SERVER_ERROR, headers: dict = None):
        return cls(format_exception(message), status_code, headers=headers)

    def render(self, content) -> bytes:
        return str(content).encode(errors="ignore")

class AppConfig:
    ignored_providers: Optional[list[str]] = None
    g4f_api_key: Optional[str] = None
    ignore_cookie_files: bool = False
    model: str = None
    provider: str = None
    image_provider: str = None
    proxy: str = None
    gui: bool = False
    demo: bool = False

    @classmethod
    def set_config(cls, **data):
        for key, value in data.items():
            setattr(cls, key, value)

class Api:
    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self.client = AsyncClient()
        self.get_g4f_api_key = APIKeyHeader(name="g4f-api-key")
        self.conversations: dict[str, dict[str, BaseConversation]] = {}

    security = HTTPBearer(auto_error=False)
    basic_security = HTTPBasic()

    async def get_username(self, request: Request) -> str:
        credentials = await self.basic_security(request)
        current_password_bytes = credentials.password.encode()
        is_correct_password = secrets.compare_digest(
            current_password_bytes, AppConfig.g4f_api_key.encode()
        )
        if not is_correct_password:
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Basic"},
            )
        return credentials.username

    def register_authorization(self):
        if AppConfig.g4f_api_key:
            print(f"Register authentication key: {''.join(['*' for _ in range(len(AppConfig.g4f_api_key))])}")
        @self.app.middleware("http")
        async def authorization(request: Request, call_next):
            if AppConfig.g4f_api_key is not None or AppConfig.demo:
                try:
                    user_g4f_api_key = await self.get_g4f_api_key(request)
                except HTTPException:
                    user_g4f_api_key = None
                path = request.url.path
                if path.startswith("/v1") or path.startswith("/api/") or (AppConfig.demo and path == '/backend-api/v2/upload_cookies'):
                    if user_g4f_api_key is None:
                        return ErrorResponse.from_message("G4F API key required", HTTP_401_UNAUTHORIZED)
                    if not secrets.compare_digest(AppConfig.g4f_api_key, user_g4f_api_key):
                        return ErrorResponse.from_message("Invalid G4F API key", HTTP_403_FORBIDDEN)
                elif not AppConfig.demo:
                    if user_g4f_api_key is not None and path.startswith("/images/"):
                        if not secrets.compare_digest(AppConfig.g4f_api_key, user_g4f_api_key):
                            return ErrorResponse.from_message("Invalid G4F API key", HTTP_403_FORBIDDEN)
                    elif path.startswith("/backend-api/") or path.startswith("/images/") or path.startswith("/chat/") and path != "/chat/":
                        try:
                            username = await self.get_username(request)
                        except HTTPException as e:
                            return ErrorResponse.from_message(e.detail, e.status_code, e.headers)
                        response = await call_next(request)
                        response.headers["X-Username"] = username
                        return response
            return await call_next(request)

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
        if not AppConfig.gui:
            @self.app.get("/")
            async def read_root():
                return RedirectResponse("/v1", 302)

        @self.app.get("/v1")
        async def read_root_v1():
            return HTMLResponse('g4f API: Go to '
                                '<a href="/v1/models">models</a>, '
                                '<a href="/v1/chat/completions">chat/completions</a>, or '
                                '<a href="/v1/images/generate">images/generate</a> <br><br>'
                                'Open Swagger UI at: '
                                '<a href="/docs">/docs</a>')

        @self.app.get("/v1/models", responses={
            HTTP_200_OK: {"model": List[ModelResponseModel]},
        })
        async def models():
            return {
                "object": "list",
                "data": [{
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": model.base_provider,
                    "image": isinstance(model, g4f.models.ImageModel),
                    "provider": False,
                } for model_id, model in g4f.models.ModelUtils.convert.items()] +
                [{
                    "id": provider_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": getattr(provider, "label", None),
                    "image": bool(getattr(provider, "image_models", False)),
                    "provider": True,
                } for provider_name, provider in g4f.Provider.ProviderUtils.convert.items()
                    if provider.working and provider_name != "Custom"
                ]
            }

        @self.app.get("/api/{provider}/models", responses={
            HTTP_200_OK: {"model": List[ModelResponseModel]},
        })
        async def models(provider: str, credentials: Annotated[HTTPAuthorizationCredentials, Depends(Api.security)] = None):
            if provider not in ProviderUtils.convert:
                return ErrorResponse.from_message("The provider does not exist.", 404)
            provider: ProviderType = ProviderUtils.convert[provider]
            if not hasattr(provider, "get_models"):
                models = []
            elif credentials is not None:
                models = provider.get_models(api_key=credentials.credentials)
            else:
                models = provider.get_models()
            return {
                "object": "list",
                "data": [{
                    "id": model,
                    "object": "model",
                    "created": 0,
                    "owned_by": getattr(provider, "label", provider.__name__),
                    "image": model in getattr(provider, "image_models", []),
                    "vision": model in getattr(provider, "vision_models", []),
                } for model in models]
            }

        @self.app.get("/v1/models/{model_name}", responses={
            HTTP_200_OK: {"model": ModelResponseModel},
            HTTP_404_NOT_FOUND: {"model": ErrorResponseModel},
        })
        async def model_info(model_name: str) -> ModelResponseModel:
            if model_name in g4f.models.ModelUtils.convert:
                model_info = g4f.models.ModelUtils.convert[model_name]
                return JSONResponse({
                    'id': model_name,
                    'object': 'model',
                    'created': 0,
                    'owned_by': model_info.base_provider
                })
            return ErrorResponse.from_message("The model does not exist.", HTTP_404_NOT_FOUND)

        @self.app.post("/v1/chat/completions", responses={
            HTTP_200_OK: {"model": ChatCompletion},
            HTTP_401_UNAUTHORIZED: {"model": ErrorResponseModel},
            HTTP_404_NOT_FOUND: {"model": ErrorResponseModel},
            HTTP_422_UNPROCESSABLE_ENTITY: {"model": ErrorResponseModel},
            HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponseModel},
        })
        async def chat_completions(
            config: ChatCompletionsConfig,
            credentials: Annotated[HTTPAuthorizationCredentials, Depends(Api.security)] = None,
            provider: str = None
        ):
            try:
                config.provider = provider if config.provider is None else config.provider
                if config.provider is None:
                    config.provider = AppConfig.provider
                if credentials is not None:
                    config.api_key = credentials.credentials

                conversation = return_conversation = None
                if conversation is not None:
                    conversation = JsonConversation(**conversation)
                    return_conversation = True
                elif config.conversation_id is not None and config.provider is not None:
                    return_conversation = True
                    if config.conversation_id in self.conversations:
                        if config.provider in self.conversations[config.conversation_id]:
                            conversation = self.conversations[config.conversation_id][config.provider]

                if config.image is not None:
                    try:
                        is_data_uri_an_image(config.image)
                    except ValueError as e:
                        return ErrorResponse.from_message(f"The image you send must be a data URI. Example: data:image/jpeg;base64,...", status_code=HTTP_422_UNPROCESSABLE_ENTITY)
                if config.images is not None:
                    for image in config.images:
                        try:
                            is_data_uri_an_image(image[0])
                        except ValueError as e:
                            example = json.dumps({"images": [["data:image/jpeg;base64,...", "filename"]]})
                            return ErrorResponse.from_message(f'The image you send must be a data URI. Example: {example}', status_code=HTTP_422_UNPROCESSABLE_ENTITY)

                # Create the completion response
                response = self.client.chat.completions.create(
                    **filter_none(
                        **{
                            "model": AppConfig.model,
                            "provider": AppConfig.provider,
                            "proxy": AppConfig.proxy,
                            **config.dict(exclude_none=True),
                            **{
                                "conversation_id": None,
                                "return_conversation": return_conversation,
                                "conversation": conversation
                            }
                        },
                        ignored=AppConfig.ignored_providers
                    ),
                )

                if not config.stream:
                    return await response

                async def streaming():
                    try:
                        async for chunk in response:
                            if isinstance(chunk, BaseConversation):
                                if config.conversation_id is not None and config.provider is not None:
                                    if config.conversation_id not in self.conversations:
                                        self.conversations[config.conversation_id] = {}
                                    self.conversations[config.conversation_id][config.provider] = chunk
                            else:
                                yield f"data: {chunk.json()}\n\n"
                    except GeneratorExit:
                        pass
                    except Exception as e:
                        logger.exception(e)
                        yield f'data: {format_exception(e, config)}\n\n'
                    yield "data: [DONE]\n\n"

                return StreamingResponse(streaming(), media_type="text/event-stream")

            except (ModelNotFoundError, ProviderNotFoundError) as e:
                logger.exception(e)
                return ErrorResponse.from_exception(e, config, HTTP_404_NOT_FOUND)
            except (MissingAuthError, NoValidHarFileError) as e:
                logger.exception(e)
                return ErrorResponse.from_exception(e, config, HTTP_401_UNAUTHORIZED)
            except Exception as e:
                logger.exception(e)
                return ErrorResponse.from_exception(e, config, HTTP_500_INTERNAL_SERVER_ERROR)

        @self.app.post("/api/{provider}/chat/completions", responses={
            HTTP_200_OK: {"model": ChatCompletion},
            HTTP_401_UNAUTHORIZED: {"model": ErrorResponseModel},
            HTTP_404_NOT_FOUND: {"model": ErrorResponseModel},
            HTTP_422_UNPROCESSABLE_ENTITY: {"model": ErrorResponseModel},
            HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponseModel},
        })
        async def provider_chat_completions(
            provider: str,
            config: ChatCompletionsConfig,
            credentials: Annotated[HTTPAuthorizationCredentials, Depends(Api.security)] = None,
        ):
            return await chat_completions(config, credentials, provider)

        responses = {
            HTTP_200_OK: {"model": ImagesResponse},
            HTTP_401_UNAUTHORIZED: {"model": ErrorResponseModel},
            HTTP_404_NOT_FOUND: {"model": ErrorResponseModel},
            HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponseModel},
        }
        @self.app.post("/v1/images/generate", responses=responses)
        @self.app.post("/v1/images/generations", responses=responses)
        async def generate_image(
            request: Request,
            config: ImageGenerationConfig,
            credentials: Annotated[HTTPAuthorizationCredentials, Depends(Api.security)] = None
        ):
            if credentials is not None:
                config.api_key = credentials.credentials
            try:
                response = await self.client.images.generate(
                    prompt=config.prompt,
                    model=config.model,
                    provider=AppConfig.image_provider if config.provider is None else config.provider,
                    **filter_none(
                        response_format=config.response_format,
                        api_key=config.api_key,
                        proxy=config.proxy
                    )
                )
                for image in response.data:
                    if hasattr(image, "url") and image.url.startswith("/"):
                        image.url = f"{request.base_url}{image.url.lstrip('/')}"
                return response
            except (ModelNotFoundError, ProviderNotFoundError) as e:
                logger.exception(e)
                return ErrorResponse.from_exception(e, config, HTTP_404_NOT_FOUND)
            except MissingAuthError as e:
                logger.exception(e)
                return ErrorResponse.from_exception(e, config, HTTP_401_UNAUTHORIZED)
            except Exception as e:
                logger.exception(e)
                return ErrorResponse.from_exception(e, config, HTTP_500_INTERNAL_SERVER_ERROR)

        @self.app.get("/v1/providers", responses={
            HTTP_200_OK: {"model": List[ProviderResponseModel]},
        })
        async def providers():
            return [{
                'id': provider.__name__,
                'object': 'provider',
                'created': 0,
                'url': provider.url,
                'label': getattr(provider, "label", None),
            } for provider in __providers__ if provider.working]

        @self.app.get("/v1/providers/{provider}", responses={
            HTTP_200_OK: {"model": ProviderResponseDetailModel},
            HTTP_404_NOT_FOUND: {"model": ErrorResponseModel},
        })
        async def providers_info(provider: str):
            if provider not in ProviderUtils.convert:
                return ErrorResponse.from_message("The provider does not exist.", 404)
            provider: ProviderType = ProviderUtils.convert[provider]
            def safe_get_models(provider: ProviderType) -> list[str]:
                try:
                    return provider.get_models() if hasattr(provider, "get_models") else []
                except:
                    return []
            return {
                'id': provider.__name__,
                'object': 'provider',
                'created': 0,
                'url': provider.url,
                'label': getattr(provider, "label", None),
                'models': safe_get_models(provider),
                'image_models': getattr(provider, "image_models", []) or [],
                'vision_models': [model for model in [getattr(provider, "default_vision_model", None)] if model],
                'params': [*provider.get_parameters()] if hasattr(provider, "get_parameters") else []
            }

        @self.app.post("/v1/upload_cookies", responses={
            HTTP_200_OK: {"model": List[FileResponseModel]},
        })
        def upload_cookies(files: List[UploadFile]):
            response_data = []
            if not AppConfig.ignore_cookie_files:
                for file in files:
                    try:
                        if file and file.filename.endswith(".json") or file.filename.endswith(".har"):
                            filename = os.path.basename(file.filename)
                            with open(os.path.join(get_cookies_dir(), filename), 'wb') as f:
                                shutil.copyfileobj(file.file, f)
                            response_data.append({"filename": filename})
                    finally:
                        file.file.close()
                read_cookie_files()
            return response_data

        @self.app.get("/v1/files/{bucket_id}", responses={
            HTTP_200_OK: {"content": {
                "text/event-stream": {"schema": {"type": "string"}},
                "text/plain": {"schema": {"type": "string"}},
            }},
            HTTP_404_NOT_FOUND: {"model": ErrorResponseModel},
        })
        def read_files(request: Request, bucket_id: str, delete_files: bool = True, refine_chunks_with_spacy: bool = False):
            bucket_dir = os.path.join(get_cookies_dir(), "buckets", bucket_id)
            event_stream = "text/event-stream" in request.headers.get("accept", "")
            if not os.path.isdir(bucket_dir):
                return ErrorResponse.from_message("Bucket dir not found", 404)
            return StreamingResponse(get_async_streaming(bucket_dir, delete_files, refine_chunks_with_spacy, event_stream),
                                     media_type="text/event-stream" if event_stream else "text/plain")

        @self.app.post("/v1/files/{bucket_id}", responses={
            HTTP_200_OK: {"model": UploadResponseModel}
        })
        def upload_files(bucket_id: str, files: List[UploadFile]):
            bucket_dir = os.path.join(get_cookies_dir(), "buckets", bucket_id)
            os.makedirs(bucket_dir, exist_ok=True)
            filenames = []
            for file in files:
                try:
                    filename = os.path.basename(file.filename)
                    if file and supports_filename(filename):
                        with open(os.path.join(bucket_dir, filename), 'wb') as f:
                            shutil.copyfileobj(file.file, f)
                        filenames.append(filename)
                finally:
                    file.file.close()
            with open(os.path.join(bucket_dir, "files.txt"), 'w') as f:
                [f.write(f"{filename}\n") for filename in filenames]
            return {"bucket_id": bucket_id, "url": f"/v1/files/{bucket_id}", "files": filenames}

        @self.app.get("/v1/synthesize/{provider}", responses={
            HTTP_200_OK: {"content": {"audio/*": {}}},
            HTTP_404_NOT_FOUND: {"model": ErrorResponseModel},
            HTTP_422_UNPROCESSABLE_ENTITY: {"model": ErrorResponseModel},
        })
        async def synthesize(request: Request, provider: str):
            try:
                provider_handler = convert_to_provider(provider)
            except ProviderNotFoundError as e:
                return ErrorResponse.from_exception(e, status_code=HTTP_404_NOT_FOUND)
            if not hasattr(provider_handler, "synthesize"):
                return ErrorResponse.from_message("Provider doesn't support synthesize", HTTP_404_NOT_FOUND)
            if len(request.query_params) == 0:
                return ErrorResponse.from_message("Missing query params", HTTP_422_UNPROCESSABLE_ENTITY)
            response_data = provider_handler.synthesize({**request.query_params})
            content_type = getattr(provider_handler, "synthesize_content_type", "application/octet-stream")
            return StreamingResponse(response_data, media_type=content_type)

        @self.app.post("/json/{filename}")
        async def get_json(filename, request: Request):
            await asyncio.sleep(30)
            return ""

        @self.app.get("/images/{filename}", responses={
            HTTP_200_OK: {"content": {"image/*": {}}},
            HTTP_404_NOT_FOUND: {}
        })
        async def get_image(filename, request: Request):
            target = os.path.join(images_dir, filename)
            ext = os.path.splitext(filename)[1]
            stat_result = SimpleNamespace()
            stat_result.st_size = 0
            if os.path.isfile(target):
                stat_result.st_size = os.stat(target).st_size
            stat_result.st_mtime = int(f"{filename.split('_')[0]}") if filename.startswith("1") else 0
            headers = {
                "cache-control": "public, max-age=31536000",
                "content-type": f"image/{ext.replace('jpg', 'jpeg')[1:] or 'jpeg'}",
                "content-length": str(stat_result.st_size),
                "last-modified": formatdate(stat_result.st_mtime, usegmt=True),
                "etag": f'"{hashlib.md5(filename.encode()).hexdigest()}"',
            }
            response = FileResponse(
                target,
                headers=headers,
                filename=filename,
            )
            try:
                if_none_match = request.headers["if-none-match"]
                etag = response.headers["etag"]
                if etag in [tag.strip(" W/") for tag in if_none_match.split(",")]:
                    return NotModifiedResponse(response.headers)
            except KeyError:
                pass
            if not os.path.isfile(target):
                source_url = str(request.query_params).split("url=", 1)
                if len(source_url) > 1:
                    source_url = source_url[1]
                    source_url = source_url.replace("%2F", "/").replace("%3A", ":").replace("%3F", "?")
                    if source_url.startswith("https://"):
                        await copy_images(
                            [source_url],
                            target=target)
                        debug.log(f"Image copied from {source_url}")
            if not os.path.isfile(target):
                return ErrorResponse.from_message("File not found", HTTP_404_NOT_FOUND)
            async def stream():
                with open(target, "rb") as file:
                    while True:
                        chunk = file.read(65536)
                        if not chunk:
                            break
                        yield chunk
            return StreamingResponse(stream(), headers=headers)

def format_exception(e: Union[Exception, str], config: Union[ChatCompletionsConfig, ImageGenerationConfig] = None, image: bool = False) -> str:
    last_provider = {} if not image else g4f.get_last_provider(True)
    provider = (AppConfig.image_provider if image else AppConfig.provider)
    model = AppConfig.model
    if config is not None:
        if config.provider is not None:
            provider = config.provider
        if config.model is not None:
            model = config.model
    if isinstance(e, str):
        message = e
    else:
        message = f"{e.__class__.__name__}: {e}"
    return json.dumps({
        "error": {"message": message},
        **filter_none(
            model=last_provider.get("model") if model is None else model,
            provider=last_provider.get("name") if provider is None else provider
        )
    })

def run_api(
    host: str = '0.0.0.0',
    port: int = None,
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
    if port is None:
        port = DEFAULT_PORT
    if AppConfig.demo and debug:
        method = "create_app_with_demo_and_debug"
    elif AppConfig.gui and debug:
        method = "create_app_with_gui_and_debug"
    else:
        method = "create_app_debug" if debug else "create_app"
    uvicorn.run(
        f"g4f.api:{method}", 
        host=host, 
        port=int(port), 
        workers=workers, 
        use_colors=use_colors, 
        factory=True, 
        reload=reload
    )
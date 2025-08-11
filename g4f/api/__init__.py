from __future__ import annotations

import logging
import json
import uvicorn
import secrets
import os
import re
import shutil
import time
from email.utils import formatdate
import os.path
import hashlib
import asyncio
import base64
from contextlib import asynccontextmanager
from urllib.parse import quote_plus
from fastapi import FastAPI, Response, Request, UploadFile, Form, Depends, Header
from fastapi.responses import StreamingResponse, RedirectResponse, HTMLResponse, JSONResponse, FileResponse
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
from starlette.background import BackgroundTask
try:
    from a2wsgi import WSGIMiddleware
    has_a2wsgi = True
except ImportError:
    has_a2wsgi = False
try:
    from PIL import Image 
    has_pillow = True
except ImportError:
    has_pillow = False
from types import SimpleNamespace
from typing import Union, Optional, List

try:
    from typing import Annotated
except ImportError:
    class Annotated:
        pass
try:
    from nodriver import util
    has_nodriver = True
except ImportError:
    has_nodriver = False

import g4f
import g4f.debug
from g4f.client import AsyncClient, ChatCompletion, ImagesResponse, ClientResponse
from g4f.providers.response import BaseConversation, JsonConversation
from g4f.client.helper import filter_none
from g4f.image import EXTENSIONS_MAP, is_data_an_media, process_image
from g4f.image.copy_images import get_media_dir, copy_media, get_source_url
from g4f.errors import ProviderNotFoundError, ModelNotFoundError, MissingAuthError, NoValidHarFileError, MissingRequirementsError
from g4f.cookies import read_cookie_files, get_cookies_dir
from g4f.providers.types import ProviderType
from g4f.providers.response import AudioResponse
from g4f.providers.any_provider import AnyProvider
from g4f import Provider
from g4f.gui import get_gui_app
from .stubs import (
    ChatCompletionsConfig, ImageGenerationConfig,
    ProviderResponseModel, ModelResponseModel,
    ErrorResponseModel, ProviderResponseDetailModel,
    FileResponseModel,
    TranscriptionResponseModel, AudioSpeechConfig,
    ResponsesConfig
)
from g4f import debug

try:
    from g4f.gui.server.crypto import create_or_read_keys, decrypt_data, get_session_key
    has_crypto = True
except ImportError:
    has_crypto = False

logger = logging.getLogger(__name__)

DEFAULT_PORT = 1337
DEFAULT_TIMEOUT = 600

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Read cookie files if not ignored
    if not AppConfig.ignore_cookie_files:
        read_cookie_files()
    AppConfig.g4f_api_key = os.environ.get("G4F_API_KEY", AppConfig.g4f_api_key)
    yield
    if has_nodriver:
        for browser in util.get_registered_instances():
            if browser.connection:
                browser.stop()
        lock_file = os.path.join(get_cookies_dir(), ".nodriver_is_open")
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except Exception as e:
                debug.error(f"Failed to remove lock file {lock_file}:" ,e)

def create_app():
    app = FastAPI(lifespan=lifespan)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    api = Api(app)

    api.register_routes()
    api.register_authorization()
    api.register_validation_exception_handler()

    if AppConfig.gui:
        if not has_a2wsgi:
            raise MissingRequirementsError("a2wsgi is required for GUI. Install it with: pip install a2wsgi")
        gui_app = WSGIMiddleware(get_gui_app(AppConfig.demo, AppConfig.timeout))
        app.mount("/", gui_app)

    if AppConfig.ignored_providers:
        for provider in AppConfig.ignored_providers:
            if provider in Provider.__map__:
                Provider.__map__[provider].working = False

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
    media_provider: str = None
    proxy: str = None
    gui: bool = False
    demo: bool = False
    timeout: int = DEFAULT_TIMEOUT

    @classmethod
    def set_config(cls, **data):
        for key, value in data.items():
            if value is not None:
                setattr(cls, key, value)

def update_headers(request: Request, user: str) -> Request:
    new_headers = request.headers.mutablecopy()
    del new_headers["Authorization"]
    if user:
        new_headers["x-user"] = user
    request.scope["headers"] = new_headers.raw
    delattr(request, "_headers")
    return request

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
        if has_crypto:
            private_key, _ = create_or_read_keys()
            session_key = get_session_key()
        @self.app.middleware("http")
        async def authorization(request: Request, call_next):
            user = None
            if request.method != "OPTIONS" and AppConfig.g4f_api_key is not None or AppConfig.demo:
                try:
                    user_g4f_api_key = await self.get_g4f_api_key(request)
                except HTTPException:
                    user_g4f_api_key = await self.security(request)
                    if hasattr(user_g4f_api_key, "credentials"):
                        user_g4f_api_key = user_g4f_api_key.credentials
                if AppConfig.g4f_api_key is None or not user_g4f_api_key or not secrets.compare_digest(AppConfig.g4f_api_key, user_g4f_api_key):
                    if has_crypto and user_g4f_api_key:
                        try:
                            expires, user = decrypt_data(private_key, user_g4f_api_key).split(":", 1)
                        except:
                            try:
                                data = json.loads(decrypt_data(session_key, user_g4f_api_key))
                                expires = int(decrypt_data(private_key, data["data"])) + 86400
                                user = data.get("user", user)
                            except:
                                return ErrorResponse.from_message(f"Invalid G4F API key", HTTP_401_UNAUTHORIZED)
                        expires = int(expires) - int(time.time())
                        debug.log(f"User: '{user}' G4F API key expires in {expires} seconds")
                        if expires < 0:
                            return ErrorResponse.from_message("G4F API key expired", HTTP_401_UNAUTHORIZED)
                else:
                    user = "admin"
                path = request.url.path
                if path.startswith("/v1") or path.startswith("/api/") or (AppConfig.demo and path == '/backend-api/v2/upload_cookies'):
                    if request.method != "OPTIONS":
                        if user_g4f_api_key is None:
                            return ErrorResponse.from_message("G4F API key required", HTTP_401_UNAUTHORIZED)
                        if AppConfig.g4f_api_key is None and user is None:
                            return ErrorResponse.from_message("Invalid G4F API key", HTTP_403_FORBIDDEN)
                elif not AppConfig.demo and not path.startswith("/images/") and not path.startswith("/media/"):
                    if user_g4f_api_key is not None:
                        if user is None:
                            return ErrorResponse.from_message("Invalid G4F API key", HTTP_403_FORBIDDEN)
                    elif path.startswith("/backend-api/") or path.startswith("/chat/") and path != "/chat/":
                        try:
                            user = await self.get_username(request)
                        except HTTPException as e:
                            return ErrorResponse.from_message(e.detail, e.status_code, e.headers)
                if user is None:
                    ip = request.headers.get("X-Forwarded-For", "")[:4].strip(":.")
                    country = request.headers.get("Cf-Ipcountry", "")
                    user = request.headers.get("x-user", ip)
                    user = f"{country}:{user}" if country else user
                request = update_headers(request, user)
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
        if not AppConfig.gui:
            @self.app.get("/")
            async def read_root():
                return RedirectResponse("/v1", 302)

        @self.app.get("/v1")
        async def read_root_v1():
            return HTMLResponse('g4f API: Go to '
                                '<a href="/v1/models">models</a>, '
                                '<a href="/v1/chat/completions">chat/completions</a>, or '
                                '<a href="/v1/media/generate">media/generate</a> <br><br>'
                                'Open Swagger UI at: '
                                '<a href="/docs">/docs</a>')

        @self.app.get("/v1/models", responses={
            HTTP_200_OK: {"model": List[ModelResponseModel]},
        })
        async def models():
            return {
                "object": "list",
                "data": [{
                    "id": model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "",
                    "image": isinstance(model, g4f.models.ImageModel),
                    "vision": isinstance(model, g4f.models.VisionModel),
                    "provider": False,
                } for model in AnyProvider.get_models()] +
                [{
                    "id": provider_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": getattr(provider, "label", ""),
                    "image": bool(getattr(provider, "image_models", False)),
                    "vision": bool(getattr(provider, "vision_models", False)),
                    "provider": True,
                } for provider_name, provider in Provider.ProviderUtils.convert.items()
                    if provider.working and provider_name not in ("Custom")
                ]
            }

        @self.app.get("/api/{provider}/models", responses={
            HTTP_200_OK: {"model": List[ModelResponseModel]},
        })
        async def models(provider: str, credentials: Annotated[HTTPAuthorizationCredentials, Depends(Api.security)] = None):
            if provider not in Provider.__map__:
                return ErrorResponse.from_message("The provider does not exist.", 404)
            provider: ProviderType = Provider.__map__[provider]
            if not hasattr(provider, "get_models"):
                models = []
            elif credentials is not None and credentials.credentials != "secret":
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
                    "audio": model in getattr(provider, "audio_models", []),
                    "video": model in getattr(provider, "video_models", []),
                    "type": "image" if model in getattr(provider, "image_models", []) else "text",
                } for model in models]
            }

        @self.app.get("/v1/models/{model_name}", responses={
            HTTP_200_OK: {"model": ModelResponseModel},
            HTTP_404_NOT_FOUND: {"model": ErrorResponseModel},
        })
        @self.app.post("/v1/models/{model_name}", responses={
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

        responses = {
            HTTP_200_OK: {"model": ChatCompletion},
            HTTP_401_UNAUTHORIZED: {"model": ErrorResponseModel},
            HTTP_404_NOT_FOUND: {"model": ErrorResponseModel},
            HTTP_422_UNPROCESSABLE_ENTITY: {"model": ErrorResponseModel},
            HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponseModel},
        }
        @self.app.post("/v1/chat/completions", responses=responses)
        @self.app.post("/api/{provider}/chat/completions", responses=responses)
        @self.app.post("/api/{provider}/{conversation_id}/chat/completions", responses=responses)
        async def chat_completions(
            config: ChatCompletionsConfig,
            credentials: Annotated[HTTPAuthorizationCredentials, Depends(Api.security)] = None,
            provider: str = None,
            conversation_id: str = None,
            x_user: Annotated[str | None, Header()] = None
        ):
            try:
                if config.provider is None:
                    config.provider = AppConfig.provider if provider is None else provider
                if config.conversation_id is None:
                    config.conversation_id = conversation_id
                if config.timeout is None:
                    config.timeout = AppConfig.timeout
                if credentials is not None and credentials.credentials != "secret":
                    config.api_key = credentials.credentials

                conversation = config.conversation
                if conversation:
                    conversation = JsonConversation(**conversation)
                elif config.conversation_id is not None and config.provider is not None:
                    if config.conversation_id in self.conversations:
                        if config.provider in self.conversations[config.conversation_id]:
                            conversation = self.conversations[config.conversation_id][config.provider]

                if config.image is not None:
                    try:
                        is_data_an_media(config.image)
                    except ValueError as e:
                        return ErrorResponse.from_message(f"The image you send must be a data URI. Example: data:image/jpeg;base64,...", status_code=HTTP_422_UNPROCESSABLE_ENTITY)
                if config.media is None:
                    config.media = config.images
                if config.media is not None:
                    for image in config.media:
                        try:
                            is_data_an_media(image[0], image[1])
                        except ValueError as e:
                            example = json.dumps({"media": [["data:image/jpeg;base64,...", "filename.jpg"]]})
                            return ErrorResponse.from_message(f'The media you send must be a data URIs. Example: {example}', status_code=HTTP_422_UNPROCESSABLE_ENTITY)

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
                                "conversation": conversation,
                                "user": x_user,
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

        responses = {
            HTTP_200_OK: {"model": ClientResponse},
            HTTP_401_UNAUTHORIZED: {"model": ErrorResponseModel},
            HTTP_404_NOT_FOUND: {"model": ErrorResponseModel},
            HTTP_422_UNPROCESSABLE_ENTITY: {"model": ErrorResponseModel},
            HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponseModel},
        }
        @self.app.post("/v1/responses", responses=responses)
        async def v1_responses(
            config: ResponsesConfig,
            credentials: Annotated[HTTPAuthorizationCredentials, Depends(Api.security)] = None,
            provider: str = None
        ):
            try:
                if config.provider is None:
                    config.provider = AppConfig.provider if provider is None else provider
                if config.api_key is None and credentials is not None and credentials.credentials != "secret":
                    config.api_key = credentials.credentials

                conversation = None
                if config.conversation is not None:
                    conversation = JsonConversation(**config.conversation)

                return await self.client.responses.create(
                    **filter_none(
                        **{
                            "model": AppConfig.model,
                            "proxy": AppConfig.proxy,
                            **config.dict(exclude_none=True),
                            "conversation": conversation
                        },
                        ignored=AppConfig.ignored_providers
                    ),
                )
            except (ModelNotFoundError, ProviderNotFoundError) as e:
                logger.exception(e)
                return ErrorResponse.from_exception(e, config, HTTP_404_NOT_FOUND)
            except (MissingAuthError, NoValidHarFileError) as e:
                logger.exception(e)
                return ErrorResponse.from_exception(e, config, HTTP_401_UNAUTHORIZED)
            except Exception as e:
                logger.exception(e)
                return ErrorResponse.from_exception(e, config, HTTP_500_INTERNAL_SERVER_ERROR)

        @self.app.post("/api/{provider}/responses", responses=responses)
        async def provider_responses(
            provider: str,
            config: ChatCompletionsConfig,
            credentials: Annotated[HTTPAuthorizationCredentials, Depends(Api.security)] = None,
        ):
            return await v1_responses(config, credentials, provider)

        responses = {
            HTTP_200_OK: {"model": ImagesResponse},
            HTTP_401_UNAUTHORIZED: {"model": ErrorResponseModel},
            HTTP_404_NOT_FOUND: {"model": ErrorResponseModel},
            HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponseModel},
        }
        @self.app.post("/v1/media/generate", responses=responses)
        @self.app.post("/v1/images/generate", responses=responses)
        @self.app.post("/v1/images/generations", responses=responses)
        @self.app.post("/api/{provider}/images/generations", responses=responses)
        async def generate_image(
            request: Request,
            config: ImageGenerationConfig,
            provider: str = None,
            credentials: Annotated[HTTPAuthorizationCredentials, Depends(Api.security)] = None
        ):
            if config.provider is None:
                config.provider = provider
            if config.provider is None:
                config.provider = AppConfig.media_provider
            if config.api_key is None and credentials is not None and credentials.credentials != "secret":
                config.api_key = credentials.credentials
            try:
                response = await self.client.images.generate(
                    **config.dict(exclude_none=True),
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
            } for provider in Provider.__providers__ if provider.working]

        @self.app.get("/v1/providers/{provider}", responses={
            HTTP_200_OK: {"model": ProviderResponseDetailModel},
            HTTP_404_NOT_FOUND: {"model": ErrorResponseModel},
        })
        async def providers_info(provider: str):
            if provider not in Provider.ProviderUtils.convert:
                return ErrorResponse.from_message("The provider does not exist.", 404)
            provider: ProviderType = Provider.ProviderUtils.convert[provider]
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

        responses = {
            HTTP_200_OK: {"model": TranscriptionResponseModel},
            HTTP_401_UNAUTHORIZED: {"model": ErrorResponseModel},
            HTTP_404_NOT_FOUND: {"model": ErrorResponseModel},
            HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponseModel},
        }
        @self.app.post("/v1/audio/transcriptions", responses=responses)
        @self.app.post("/api/{path_provider}/audio/transcriptions", responses=responses)
        async def convert(
            file: UploadFile,
            path_provider: str = None,
            model: Annotated[Optional[str], Form()] = None,
            provider: Annotated[Optional[str], Form()] = None,
            prompt: Annotated[Optional[str], Form()] = "Transcribe this audio"
        ):
            provider = provider if path_provider is None else path_provider
            kwargs = {"modalities": ["text"]}
            if provider == "MarkItDown":
                kwargs = {
                    "llm_client": self.client,
                }
            try:
                response = await self.client.chat.completions.create(
                    messages=prompt,
                    model=model,
                    provider=provider,
                    media=[[file.file, file.filename]],
                    **kwargs
                )
                return {"text": response.choices[0].message.content, "model": response.model, "provider": response.provider}
            except (ModelNotFoundError, ProviderNotFoundError) as e:
                logger.exception(e)
                return ErrorResponse.from_exception(e, None, HTTP_404_NOT_FOUND)
            except MissingAuthError as e:
                logger.exception(e)
                return ErrorResponse.from_exception(e, None, HTTP_401_UNAUTHORIZED)
            except Exception as e:
                logger.exception(e)
                return ErrorResponse.from_exception(e, None, HTTP_500_INTERNAL_SERVER_ERROR)

        @self.app.post("/api/markitdown", responses=responses)
        async def markitdown(
            file: UploadFile
        ):
            return await convert(file, "MarkItDown")

        responses = {
            HTTP_200_OK: {"content": {"audio/*": {}}},
            HTTP_401_UNAUTHORIZED: {"model": ErrorResponseModel},
            HTTP_404_NOT_FOUND: {"model": ErrorResponseModel},
            HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponseModel},
        }
        @self.app.post("/v1/audio/speech", responses=responses)
        @self.app.post("/api/{provider}/audio/speech", responses=responses)
        async def generate_speech(
            config: AudioSpeechConfig,
            provider: str = AppConfig.media_provider,
            credentials: Annotated[HTTPAuthorizationCredentials, Depends(Api.security)] = None
        ):
            api_key = None
            if credentials is not None and credentials.credentials != "secret":
                api_key = credentials.credentials
            try:
                audio = filter_none(voice=config.voice, format=config.response_format, language=config.language)
                response = await self.client.chat.completions.create(
                    messages=[
                        {"role": "user", "content": f"{config.instrcutions} Text: {config.input}"}
                    ],
                    model=config.model,
                    provider=config.provider if provider is None else provider,
                    prompt=config.input,
                    api_key=api_key,
                    download_media=config.download_media,
                    **filter_none(
                        audio=audio if audio else None,
                    )
                )
                if response.choices[0].message.audio is not None:
                    response = base64.b64decode(response.choices[0].message.audio.data)
                    return Response(response, media_type=f"audio/{config.response_format.replace('mp3', 'mpeg')}")
                elif isinstance(response.choices[0].message.content, AudioResponse):
                    response = response.choices[0].message.content.data
                    response = response.replace("/media", get_media_dir())
                    def delete_file():
                        try:
                            os.remove(response)
                        except Exception as e:
                            logger.exception(e)
                    return FileResponse(response, background=BackgroundTask(delete_file))
            except (ModelNotFoundError, ProviderNotFoundError) as e:
                logger.exception(e)
                return ErrorResponse.from_exception(e, None, HTTP_404_NOT_FOUND)
            except MissingAuthError as e:
                logger.exception(e)
                return ErrorResponse.from_exception(e, None, HTTP_401_UNAUTHORIZED)
            except Exception as e:
                logger.exception(e)
                return ErrorResponse.from_exception(e, None, HTTP_500_INTERNAL_SERVER_ERROR)

        @self.app.post("/v1/upload_cookies", responses={
            HTTP_200_OK: {"model": List[FileResponseModel]},
        })
        def upload_cookies(
            files: List[UploadFile],
            credentials: Annotated[HTTPAuthorizationCredentials, Depends(Api.security)] = None
        ):
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

        @self.app.post("/json/{filename}")
        async def get_json(filename, request: Request):
            await asyncio.sleep(30)
            return ""

        @self.app.get("/images/{filename}", responses={
            HTTP_200_OK: {"content": {"image/*": {}}},
            HTTP_404_NOT_FOUND: {}
        })
        @self.app.get("/media/{filename}", responses={
            HTTP_200_OK: {"content": {"image/*": {}, "audio/*": {}}, "video/*": {}},
            HTTP_404_NOT_FOUND: {}
        })
        async def get_media(filename, request: Request, thumbnail: bool = False):
            def get_timestamp(str):
                m=re.match("^[0-9]+", str)
                if m:
                    return int(m.group(0))
                else:
                    return 0
            target = os.path.join(get_media_dir(), os.path.basename(filename))
            if thumbnail and has_pillow:
                thumbnail_dir = os.path.join(get_media_dir(), "thumbnails")
                thumbnail = os.path.join(thumbnail_dir, filename)
            if not os.path.isfile(target):
                other_name = os.path.join(get_media_dir(), os.path.basename(quote_plus(filename)))
                if os.path.isfile(other_name):
                    target = other_name
            ext = os.path.splitext(filename)[1][1:]
            mime_type = EXTENSIONS_MAP.get(ext)
            stat_result = SimpleNamespace()
            stat_result.st_size = 0
            stat_result.st_mtime = get_timestamp(filename)
            if thumbnail and has_pillow and os.path.isfile(thumbnail):
                stat_result.st_size = os.stat(thumbnail).st_size
            elif not thumbnail and os.path.isfile(target):
                stat_result.st_size = os.stat(target).st_size
            headers = {
                "cache-control": "public, max-age=31536000",
                "last-modified": formatdate(stat_result.st_mtime, usegmt=True),
                "etag": f'"{hashlib.md5(filename.encode()).hexdigest()}"',
                **({
                    "content-length": str(stat_result.st_size),
                } if stat_result.st_size else {}),
                **({} if thumbnail or mime_type is None else {
                    "content-type": mime_type,
                })
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
            if not os.path.isfile(target) and mime_type is not None:
                source_url = get_source_url(str(request.query_params))
                ssl = None
                if source_url is None:
                    backend_url = os.environ.get("G4F_BACKEND_URL")
                    if backend_url:
                        source_url = f"{backend_url}/media/{filename}"
                        ssl = False
                if source_url is not None:
                    try:
                        await copy_media(
                            [source_url],
                            target=target, ssl=ssl)
                        debug.log(f"File copied from {source_url}")
                    except Exception as e:
                        debug.error(f"Download failed:  {source_url}")
                        debug.error(e)
                        return RedirectResponse(url=source_url)
            if thumbnail and has_pillow:
                try:
                    if not os.path.isfile(thumbnail):
                        image = Image.open(target)
                        os.makedirs(thumbnail_dir, exist_ok=True)
                        process_image(image, save=thumbnail)
                        debug.log(f"Thumbnail created: {thumbnail}")
                except Exception as e:
                    logger.exception(e)
            if thumbnail and os.path.isfile(thumbnail):
                result = thumbnail
            else:
                result = target
            if not os.path.isfile(result):
                return ErrorResponse.from_message("File not found", HTTP_404_NOT_FOUND)
            async def stream():
                with open(result, "rb") as file:
                    while True:
                        chunk = file.read(65536)
                        if not chunk:
                            break
                        yield chunk
            return StreamingResponse(stream(), headers=headers)

        @self.app.get("/thumbnail/{filename}", responses={
            HTTP_200_OK: {"content": {"image/*": {}, "audio/*": {}}, "video/*": {}},
            HTTP_404_NOT_FOUND: {}
        })
        async def get_media_thumbnail(filename: str, request: Request):
            return await get_media(filename, request, True)

def format_exception(e: Union[Exception, str], config: Union[ChatCompletionsConfig, ImageGenerationConfig] = None, image: bool = False) -> str:
    last_provider = {} if not image else g4f.get_last_provider(True)
    provider = (AppConfig.media_provider if image else AppConfig.provider)
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
    use_colors: bool = None,
    **kwargs
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
        factory=True,
        use_colors=use_colors,
        **filter_none(**kwargs)
    )

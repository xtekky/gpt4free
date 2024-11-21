import json
import asyncio
import flask
import os
from flask import request, Flask
from typing import AsyncGenerator, Generator
from werkzeug.utils import secure_filename

from g4f.image import is_allowed_extension, to_image
from g4f.client.service import convert_to_provider
from g4f.errors import ProviderNotFoundError
from g4f.cookies import get_cookies_dir
from .api import Api

def safe_iter_generator(generator: Generator) -> Generator:
    start = next(generator)
    def iter_generator():
        yield start
        yield from generator
    return iter_generator()

def to_sync_generator(gen: AsyncGenerator) -> Generator:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    gen = gen.__aiter__()
    async def get_next():
        try:
            obj = await gen.__anext__()
            return False, obj
        except StopAsyncIteration: return True, None
    while True:
        done, obj = loop.run_until_complete(get_next())
        if done:
            break
        yield obj

class Backend_Api(Api):    
    """
    Handles various endpoints in a Flask application for backend operations.

    This class provides methods to interact with models, providers, and to handle
    various functionalities like conversations, error handling, and version management.

    Attributes:
        app (Flask): A Flask application instance.
        routes (dict): A dictionary mapping API endpoints to their respective handlers.
    """
    def __init__(self, app: Flask) -> None:
        """
        Initialize the backend API with the given Flask application.

        Args:
            app (Flask): Flask application instance to attach routes to.
        """
        self.app: Flask = app
        self.routes = {
            '/backend-api/v2/models': {
                'function': self.get_models,
                'methods': ['GET']
            },
            '/backend-api/v2/models/<provider>': {
                'function': self.get_provider_models,
                'methods': ['GET']
            },
            '/backend-api/v2/image_models': {
                'function': self.get_image_models,
                'methods': ['GET']
            },
            '/backend-api/v2/providers': {
                'function': self.get_providers,
                'methods': ['GET']
            },
            '/backend-api/v2/version': {
                'function': self.get_version,
                'methods': ['GET']
            },
            '/backend-api/v2/conversation': {
                'function': self.handle_conversation,
                'methods': ['POST']
            },
            '/backend-api/v2/synthesize/<provider>': {
                'function': self.handle_synthesize,
                'methods': ['GET']
            },
            '/backend-api/v2/upload_cookies': {
                'function': self.upload_cookies,
                'methods': ['POST']
            },
            '/images/<path:name>': {
                'function': self.serve_images,
                'methods': ['GET']
            }
        }

    def upload_cookies(self):
        file = None
        if "file" in request.files:
            file = request.files['file']
            if file.filename == '':
                return 'No selected file', 400
        if file and file.filename.endswith(".json") or file.filename.endswith(".har"):
            filename = secure_filename(file.filename)
            file.save(os.path.join(get_cookies_dir(), filename))
            return "File saved", 200
        return 'Not supported file', 400

    def handle_conversation(self):
        """
        Handles conversation requests and streams responses back.

        Returns:
            Response: A Flask response object for streaming.
        """
        
        kwargs = {}
        if "file" in request.files:
            file = request.files['file']
            if file.filename != '' and is_allowed_extension(file.filename):
                kwargs['image'] = to_image(file.stream, file.filename.endswith('.svg'))
                kwargs['image_name'] = file.filename
        if "json" in request.form:
            json_data = json.loads(request.form['json'])
        else:
            json_data = request.json

        kwargs = self._prepare_conversation_kwargs(json_data, kwargs)

        return self.app.response_class(
            self._create_response_stream(
                kwargs,
                json_data.get("conversation_id"),
                json_data.get("provider"),
                json_data.get("download_images", True),
            ),
            mimetype='text/event-stream'
        )

    def handle_synthesize(self, provider: str):
        try:
            provider_handler = convert_to_provider(provider)
        except ProviderNotFoundError:
            return "Provider not found", 404
        if not hasattr(provider_handler, "synthesize"):
            return "Provider doesn't support synthesize", 500
        try:
            response_generator = provider_handler.synthesize({**request.args})
            if hasattr(response_generator, "__aiter__"):
                response_generator = to_sync_generator(response_generator)
            response = flask.Response(safe_iter_generator(response_generator), content_type="audio/mpeg")
            response.headers['Cache-Control'] = "max-age=604800"
            return response
        except Exception as e:
            return f"{e.__class__.__name__}: {e}", 500

    def get_provider_models(self, provider: str):
        api_key = None if request.authorization is None else request.authorization.token
        models = super().get_provider_models(provider, api_key)
        if models is None:
            return "Provider not found", 404
        return models

    def _format_json(self, response_type: str, content) -> str:
        """
        Formats and returns a JSON response.

        Args:
            response_type (str): The type of the response.
            content: The content to be included in the response.

        Returns:
            str: A JSON formatted string.
        """
        return json.dumps(super()._format_json(response_type, content)) + "\n"
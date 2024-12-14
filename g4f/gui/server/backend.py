import json
import flask
import os
import logging
import asyncio
from flask import Flask, request, jsonify
from typing import Generator
from werkzeug.utils import secure_filename

from g4f.image import is_allowed_extension, to_image
from g4f.client.service import convert_to_provider
from g4f.providers.asyncio import to_sync_generator
from g4f.errors import ProviderNotFoundError
from g4f.cookies import get_cookies_dir
from .api import Api

logger = logging.getLogger(__name__)

def safe_iter_generator(generator: Generator) -> Generator:
    start = next(generator)
    def iter_generator():
        yield start
        yield from generator
    return iter_generator()

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

        def jsonify_models(**kwargs):
            response = self.get_models(**kwargs)
            if isinstance(response, list):
                return jsonify(response)
            return response

        def jsonify_provider_models(**kwargs):
            response = self.get_provider_models(**kwargs)
            if isinstance(response, list):
                return jsonify(response)
            return response

        def jsonify_providers(**kwargs):
            response = self.get_providers(**kwargs)
            if isinstance(response, list):
                return jsonify(response)
            return response

        self.routes = {
            '/backend-api/v2/models': {
                'function': jsonify_models,
                'methods': ['GET']
            },
            '/backend-api/v2/models/<provider>': {
                'function': jsonify_provider_models,
                'methods': ['GET']
            },
            '/backend-api/v2/providers': {
                'function': jsonify_providers,
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
        if "files[]" in request.files:
            images = []
            for file in request.files.getlist('files[]'):
                if file.filename != '' and is_allowed_extension(file.filename):
                    images.append((to_image(file.stream, file.filename.endswith('.svg')), file.filename))
            kwargs['images'] = images
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
        response_data = provider_handler.synthesize({**request.args})
        if asyncio.iscoroutinefunction(provider_handler.synthesize):
            response_data = asyncio.run(response_data)
        else:
            if hasattr(response_data, "__aiter__"):
                response_data = to_sync_generator(response_data)
            response_data = safe_iter_generator(response_data)
        content_type = getattr(provider_handler, "synthesize_content_type", "application/octet-stream")
        response = flask.Response(response_data, content_type=content_type)
        response.headers['Cache-Control'] = "max-age=604800"
        return response

    def get_provider_models(self, provider: str):
        api_key = request.headers.get("x_api_key")
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
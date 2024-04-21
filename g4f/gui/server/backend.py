import json
from flask import request, Flask
from g4f.image import is_allowed_extension, to_image
from .api import Api

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
            '/backend-api/v2/gen.set.summarize:title': {
                'function': self.generate_title,
                'methods': ['POST']
            },
            '/backend-api/v2/error': {
                'function': self.handle_error,
                'methods': ['POST']
            }
        }

    def handle_error(self):
        """
        Initialize the backend API with the given Flask application.

        Args:
            app (Flask): Flask application instance to attach routes to.
        """
        print(request.json)
        return 'ok', 200

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
            self._create_response_stream(kwargs, json_data.get("conversation_id"), json_data.get("provider")),
            mimetype='text/event-stream'
        )

    def get_provider_models(self, provider: str):
        models = super().get_provider_models(provider)
        if models is None:
            return 404, "Provider not found"
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
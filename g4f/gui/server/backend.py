import logging
import json
from flask import request, Flask
from typing import Generator
from g4f import version, models
from g4f import get_last_provider, ChatCompletion
from g4f.image import is_allowed_extension, to_image
from g4f.errors import VersionNotFoundError
from g4f.Provider import __providers__
from g4f.Provider.bing.create_images import patch_provider
from .internet import get_search_message


class Backend_Api:
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
    
    def get_models(self):
        """
        Return a list of all models.

        Fetches and returns a list of all available models in the system.

        Returns:
            List[str]: A list of model names.
        """
        return models._all_models
    
    def get_providers(self):
        """
        Return a list of all working providers.
        """
        return [provider.__name__ for provider in __providers__ if provider.working]
        
    def get_version(self):
        """
        Returns the current and latest version of the application.

        Returns:
            dict: A dictionary containing the current and latest version.
        """
        try:
            current_version = version.utils.current_version
        except VersionNotFoundError:
            current_version = None
        return {
            "version": current_version,
            "latest_version": version.get_latest_version(),
        }
    
    def generate_title(self):
        """
        Generates and returns a title based on the request data.

        Returns:
            dict: A dictionary with the generated title.
        """
        return {'title': ''}
    
    def handle_conversation(self):
        """
        Handles conversation requests and streams responses back.

        Returns:
            Response: A Flask response object for streaming.
        """
        kwargs = self._prepare_conversation_kwargs()

        return self.app.response_class(
            self._create_response_stream(kwargs),
            mimetype='text/event-stream'
        )
    
    def _prepare_conversation_kwargs(self):
        """
        Prepares arguments for chat completion based on the request data.

        Reads the request and prepares the necessary arguments for handling 
        a chat completion request.

        Returns:
            dict: Arguments prepared for chat completion.
        """
        kwargs = {}
        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '' and is_allowed_extension(file.filename):
                kwargs['image'] = to_image(file.stream, file.filename.endswith('.svg'))
        if 'json' in request.form:
            json_data = json.loads(request.form['json'])
        else:
            json_data = request.json
            
        provider = json_data.get('provider', '').replace('g4f.Provider.', '')
        provider = provider if provider and provider != "Auto" else None
        if provider == 'OpenaiChat':
            kwargs['auto_continue'] = True
        messages = json_data['messages']
        if json_data.get('web_search'):
            if provider == "Bing":
                kwargs['web_search'] = True
            else:
                messages[-1]["content"] = get_search_message(messages[-1]["content"])
        model = json_data.get('model')
        model = model if model else models.default
        patch = patch_provider if json_data.get('patch_provider') else None

        return {
            "model": model,
            "provider": provider,
            "messages": messages,
            "stream": True,
            "ignore_stream_and_auth": True,
            "patch_provider": patch,
            **kwargs
        }

    def _create_response_stream(self, kwargs) -> Generator[str, None, None]:
        """
        Creates and returns a streaming response for the conversation.

        Args:
            kwargs (dict): Arguments for creating the chat completion.

        Yields:
            str: JSON formatted response chunks for the stream.

        Raises:
            Exception: If an error occurs during the streaming process.
        """
        try:
            first = True
            for chunk in ChatCompletion.create(**kwargs):
                if first:
                    first = False
                    yield self._format_json('provider', get_last_provider(True))
                if isinstance(chunk, Exception):
                    logging.exception(chunk)
                    yield self._format_json('message', get_error_message(chunk))
                else:
                    yield self._format_json('content', str(chunk))
        except Exception as e:
            logging.exception(e)
            yield self._format_json('error', get_error_message(e))
            
    def _format_json(self, response_type: str, content) -> str:
        """
        Formats and returns a JSON response.

        Args:
            response_type (str): The type of the response.
            content: The content to be included in the response.

        Returns:
            str: A JSON formatted string.
        """
        return json.dumps({
            'type': response_type,
            response_type: content
        }) + "\n"
    
def get_error_message(exception: Exception) -> str:
    """
    Generates a formatted error message from an exception.

    Args:
        exception (Exception): The exception to format.

    Returns:
        str: A formatted error message string.
    """
    return f"{get_last_provider().__name__}: {type(exception).__name__}: {exception}"
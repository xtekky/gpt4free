from __future__ import annotations

import logging
import json
from typing import Iterator

from g4f import version, models
from g4f import get_last_provider, ChatCompletion
from g4f.errors import VersionNotFoundError
from g4f.Provider import ProviderType, __providers__, __map__
from g4f.providers.base_provider import ProviderModelMixin
from g4f.providers.conversation import BaseConversation

conversations: dict[dict[str, BaseConversation]] = {}

class Api():

    def get_models(self) -> list[str]:
        """
        Return a list of all models.

        Fetches and returns a list of all available models in the system.

        Returns:
            List[str]: A list of model names.
        """
        return models._all_models

    def get_provider_models(self, provider: str) -> list[dict]:
        if provider in __map__:
            provider: ProviderType = __map__[provider]
            if issubclass(provider, ProviderModelMixin):
                return [{"model": model, "default": model == provider.default_model} for model in provider.get_models()]
            elif provider.supports_gpt_35_turbo or provider.supports_gpt_4:
                return [
                    *([{"model": "gpt-4", "default": not provider.supports_gpt_4}] if provider.supports_gpt_4 else []),
                    *([{"model": "gpt-3.5-turbo", "default": not provider.supports_gpt_4}] if provider.supports_gpt_35_turbo else [])
                ]
            else:
                return [];

    def get_providers(self) -> list[str]:
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
            "latest_version": version.utils.latest_version,
        }

    def generate_title(self):
        """
        Generates and returns a title based on the request data.

        Returns:
            dict: A dictionary with the generated title.
        """
        return {'title': ''}

    def _prepare_conversation_kwargs(self, json_data: dict, kwargs: dict):
        """
        Prepares arguments for chat completion based on the request data.

        Reads the request and prepares the necessary arguments for handling 
        a chat completion request.

        Returns:
            dict: Arguments prepared for chat completion.
        """ 
        provider = json_data.get('provider', None)
        if "image" in kwargs and provider is None:
            provider = "Bing"
        if provider == 'OpenaiChat':
            kwargs['auto_continue'] = True

        messages = json_data['messages']
        if json_data.get('web_search'):
            if provider == "Bing":
                kwargs['web_search'] = True
            else:
                from .internet import get_search_message
                messages[-1]["content"] = get_search_message(messages[-1]["content"])

        conversation_id = json_data.get("conversation_id")
        if conversation_id and provider in conversations and conversation_id in conversations[provider]:
            kwargs["conversation"] = conversations[provider][conversation_id]

        model = json_data.get('model') or models.default

        return {
            "model": model,
            "provider": provider,
            "messages": messages,
            "stream": True,
            "ignore_stream": True,
            "return_conversation": True,
            **kwargs
        }

    def _create_response_stream(self, kwargs: dict, conversation_id: str, provider: str) -> Iterator:
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
                    yield self._format_json("provider", get_last_provider(True))
                if isinstance(chunk, BaseConversation):
                    if provider not in conversations:
                        conversations[provider] = {}
                    conversations[provider][conversation_id] = chunk
                    yield self._format_json("conversation", conversation_id)
                elif isinstance(chunk, Exception):
                    logging.exception(chunk)
                    yield self._format_json("message", get_error_message(chunk))
                else:
                    yield self._format_json("content", str(chunk))
        except Exception as e:
            logging.exception(e)
            yield self._format_json('error', get_error_message(e))

    def _format_json(self, response_type: str, content):
        """
        Formats and returns a JSON response.

        Args:
            response_type (str): The type of the response.
            content: The content to be included in the response.

        Returns:
            str: A JSON formatted string.
        """
        return {
            'type': response_type,
            response_type: content
        }

def get_error_message(exception: Exception) -> str:
    """
    Generates a formatted error message from an exception.

    Args:
        exception (Exception): The exception to format.

    Returns:
        str: A formatted error message string.
    """
    message = f"{type(exception).__name__}: {exception}"
    provider = get_last_provider()
    if provider is None:
        return message
    return f"{provider.__name__}: {message}"
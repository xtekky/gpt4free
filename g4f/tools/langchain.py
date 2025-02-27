from __future__ import annotations

from typing import Any, Dict
from langchain_community.chat_models import openai
from langchain_community.chat_models.openai import convert_message_to_dict
from pydantic import Field
from g4f.client import AsyncClient, Client
from g4f.client.stubs import ChatCompletionMessage

def new_convert_message_to_dict(message: openai.BaseMessage) -> dict:
    message_dict: Dict[str, Any]
    if isinstance(message, ChatCompletionMessage):
        message_dict = {"role": message.role, "content": message.content}
        if message.tool_calls is not None:
            message_dict["tool_calls"] = [{
                "id": tool_call.id,
                "type": tool_call.type,
                "function": tool_call.function
            } for tool_call in message.tool_calls]
            if message_dict["content"] == "":
                message_dict["content"] = None
    else:
        message_dict = convert_message_to_dict(message)
    return message_dict

openai.convert_message_to_dict = new_convert_message_to_dict

class ChatAI(openai.ChatOpenAI):
    model_name: str = Field(default="gpt-4o", alias="model")

    @classmethod
    def validate_environment(cls, values: dict) -> dict:
        client_params = {
            "api_key": values["g4f_api_key"] if "g4f_api_key" in values else None,
            "provider": values["provider"] if "provider" in values else None,
        }
        values["client"] = Client(**client_params).chat.completions
        values["async_client"] = AsyncClient(
            **client_params
        ).chat.completions
        return values
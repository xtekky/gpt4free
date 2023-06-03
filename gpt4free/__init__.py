from enum import Enum

from gpt4free import forefront
from gpt4free import quora
from gpt4free import theb
from gpt4free import usesless
from gpt4free import you
from gpt4free import aicolors
from gpt4free import deepai


class Provider(Enum):
    """An enum representing  different providers."""

    You = "you"
    Poe = "poe"
    ForeFront = "fore_front"
    Theb = "theb"
    UseLess = "useless"
    AiColors = "ai_colors"
    DeepAI = "deepai"


class Completion:
    """This class will be used for invoking the given provider"""

    @staticmethod
    def create(provider: Provider, prompt: str, **kwargs) -> str:
        """
        Invokes the given provider with given prompt and addition arguments and returns the string response

        :param provider: an enum representing the provider to use while invoking
        :param prompt: input provided by the user
        :param kwargs:  Additional keyword arguments to pass to the provider while invoking
        :return: A string representing the response from the provider
        """
        if provider == Provider.Poe:
            return Completion.__poe_service(prompt, **kwargs)
        elif provider == Provider.You:
            return Completion.__you_service(prompt, **kwargs)
        elif provider == Provider.ForeFront:
            return Completion.__fore_front_service(prompt, **kwargs)
        elif provider == Provider.Theb:
            return Completion.__theb_service(prompt, **kwargs)
        elif provider == Provider.UseLess:
            return Completion.__useless_service(prompt, **kwargs)
        elif provider == Provider.AiColors:
            return Completion.__ai_colors_service(prompt, **kwargs)
        elif provider == Provider.DeepAI:
            return Completion.__deepai_service(prompt, **kwargs)
        else:
            raise Exception("Provider not exist, Please try again")

    @staticmethod
    def __ai_colors_service(prompt: str):
        return aicolors.Completion.create(prompt=prompt)

    @staticmethod
    def __useless_service(prompt: str, **kwargs) -> str:
        return usesless.Completion.create(prompt=prompt, **kwargs)

    @staticmethod
    def __you_service(prompt: str, **kwargs) -> str:
        return you.Completion.create(prompt, **kwargs).text

    @staticmethod
    def __poe_service(prompt: str, **kwargs) -> str:
        return quora.Completion.create(prompt=prompt, **kwargs).text

    @staticmethod
    def __fore_front_service(prompt: str, **kwargs) -> str:
        return forefront.Completion.create(prompt=prompt, **kwargs).text

    @staticmethod
    def __theb_service(prompt: str, **kwargs):
        return "".join(theb.Completion.create(prompt=prompt))

    @staticmethod
    def __deepai_service(prompt: str, **kwargs):
        return "".join(deepai.Completion.create(prompt=prompt))


class ChatCompletion:
    """This class is used to execute a chat completion for a specified provider"""

    @staticmethod
    def create(provider: Provider, messages: list, **kwargs) -> str:
        """
        Invokes the given provider with given chat messages and addition arguments and returns the string response

        :param provider: an enum representing the provider to use while invoking
        :param messages: a list of chat messages, see the OpenAI docs for how to format this (https://platform.openai.com/docs/guides/chat/introduction)
        :param kwargs:  Additional keyword arguments to pass to the provider while invoking
        :return: A string representing the response from the provider
        """
        if provider == Provider.DeepAI:
            return ChatCompletion.__deepai_service(messages, **kwargs)
        else:
            raise Exception("Provider not exist, Please try again")

    @staticmethod
    def __deepai_service(messages: list, **kwargs):
        return "".join(deepai.ChatCompletion.create(messages=messages))

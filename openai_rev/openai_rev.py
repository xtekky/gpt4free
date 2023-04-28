from enum import Enum

from openai_rev import forefront
from openai_rev import openaihosted
from openai_rev import quora
from openai_rev import theb
from openai_rev import you


class Provider(Enum):
    You = 'you'
    Poe = 'poe'
    ForeFront = 'fore_front'
    Theb = 'theb'
    OpenAiHosted = 'openai_hosted'


class Completion:
    @staticmethod
    def create(provider: Provider, prompt: str, **kwargs):
        if provider == Provider.Poe:
            return Completion.__poe_service(prompt, **kwargs)
        elif provider == Provider.You:
            return Completion.__you_service(prompt, **kwargs)
        elif provider == Provider.ForeFront:
            return Completion.__fore_front_service(prompt, **kwargs)
        elif provider == Provider.Theb:
            return Completion.__theb_service(prompt, **kwargs)
        elif provider == Provider.OpenAiHosted:
            return Completion.__openai_hosted(prompt, **kwargs)
        else:
            raise Exception('Provider not exist, Please try again')

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
        return ''.join(theb.Completion.create(prompt=prompt))

    @staticmethod
    def __openai_hosted(prompt: str, **kwargs):
        return openaihosted.Completion.create(systemprompt='', text=prompt, assistantprompt='').text

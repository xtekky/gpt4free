from enum import Enum

from openai_rev import forefront
from openai_rev import quora
from openai_rev import theb
from openai_rev import you


class Provider(Enum):
    You = 'you'
    Poe = 'poe'
    ForeFront = 'fore_front'
    Theb = 'theb'


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

    @classmethod
    def __you_service(cls, prompt: str, **kwargs) -> str:
        return you.Completion.create(prompt, **kwargs).text

    @classmethod
    def __poe_service(cls, prompt: str, **kwargs) -> str:
        return quora.Completion.create(prompt=prompt, **kwargs).text

    @classmethod
    def __fore_front_service(cls, prompt: str, **kwargs) -> str:
        return forefront.Completion.create(prompt=prompt, **kwargs).text

    @classmethod
    def __theb_service(cls, prompt: str, **kwargs):
        return ''.join(theb.Completion.create(prompt=prompt))
